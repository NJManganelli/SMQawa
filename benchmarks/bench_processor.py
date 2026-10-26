import argparse
import copy
import csv
import gzip
import itertools
import json
import os
import pickle
import resource
import subprocess
import sys
import threading
import time

import numpy as np

DESCRIPTION = """Equality and timing harness for wzinclusive_processor.

Subcommands:
  run       run the processor on one entry range of one file and dump the output
  compare   compare two dumps under the comparison rule of the optimisation plan
  selftest  check that compare reaches the expected verdict on mutated dumps
  scale     at-scale run of one variant (the reference tree, the worktree's C1 engine or a benchmark
            engine of fill_engines.py) in a fresh subprocess, with machine state and memory sampling
  report    G1 verdict (required) and the R2-R4 information rows over the scale results in a directory

The tree under test is chosen with PYTHONPATH (<tree>/src); `run` asserts that
qawa is imported from below the first PYTHONPATH entry.
"""

WPREFS = ("wWZ", "wWpZ", "wWmZ", "wZW", "wZWp", "wZWm")
POLS = ("long", "left", "right")
SUFFIXES = tuple(
    ["_eft"]
    + [f"_{w}_{p}" for w in WPREFS for p in POLS]
    + [f"_{w}_{p}_eft" for w in WPREFS for p in POLS]
)
SAME_SELECTION = {"dilep_ll_dphi": "dilep_deta", "dilep_ll_dphi_eft": "dilep_deta_eft"}
VARIANCE_RTOL = 1e-9
ORDER_TOLERANT_RTOL = 1e-9
CANCELLATION_RTOL = 1e-9
SUMW_RUNS_TOLERANCE = 1e-6
MAX_LISTED_CELLS = 50
RANDOM_SEED = 1


def maxrss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def under_tree(path, tree):
    return os.path.realpath(path).startswith(os.path.realpath(tree) + os.sep)


def import_qawa_checked(tree=None):
    import qawa
    if tree is None:
        entries = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        if not entries:
            raise SystemExit("PYTHONPATH must name the tree under test (<tree>/src)")
        tree = entries[0]
    print("qawa.__file__ =", qawa.__file__, flush=True)
    if not under_tree(qawa.__file__, tree):
        raise SystemExit(f"qawa imported from {qawa.__file__}, not from below {tree}")
    return qawa.__file__


# ---------------------------------------------------------------- synthetic EFT

class SyntheticParameterisation:
    def __init__(self, terms):
        self.terms = tuple(terms)

    def decompose(self, x):
        return x

    def validate(self, theta):
        return []

    def active_terms(self, theta=None):
        return list(self.terms)

    def __repr__(self):
        return f"SyntheticParameterisation(terms={len(self.terms)})"


def synthetic_terms(n_operators):
    ops = ["cW"] + [f"op{i}" for i in range(1, n_operators)]
    return tuple(
        ["sm"]
        + [f"lin_{o}" for o in ops]
        + [f"quad_{o}" for o in ops]
        + [f"cross_{a}_{b}" for a, b in itertools.combinations(ops, 2)]
    )


def synthetic_theta(event, terms, nan_modulus):
    import awkward as ak
    evt = ak.to_numpy(event.event).astype(np.int64)
    gw = ak.to_numpy(event.genWeight).astype(np.float64)
    theta = np.empty((len(evt), len(terms)))
    for t in range(len(terms)):
        theta[:, t] = gw * (0.5 + ((evt * (2 * t + 3)) % 101) / 100.0)
    theta[evt % nan_modulus == 0, terms.index("lin_cW")] = np.nan
    return theta


def install_synthetic_eft(module, n_operators, nan_modulus):
    terms = synthetic_terms(n_operators)
    module.points_from_events = lambda event: (SyntheticParameterisation(terms), terms)
    module.stack_named_weights = lambda event, points: synthetic_theta(event, tuple(points), nan_modulus)
    return terms


# ---------------------------------------------------------------- instrumentation

def histogram_bytes(output, suffix=None):
    total = 0
    for hists in output.values():
        for key, h in hists.items():
            if suffix is None or key.endswith(suffix):
                total += h.view(flow=True).nbytes
    return total


class Instrument:
    def __init__(self, pass_rss=False):
        self.passes = []
        self.expected_shift = set()
        self.expected_weight = set()
        self.current = None
        self.chunks = []
        self.pass_rss = pass_rss

    def install(self, wzmod):
        from coffea.analysis_tools import Weights
        inst = self
        weight = Weights.weight

        def recorded_weight(self, modifier=None):
            result = weight(self, modifier)
            if modifier is not None:
                inst.expected_weight.add(modifier)
            return result
        Weights.weight = recorded_weight

        cls = wzmod.wzinclusive_processor
        process_shift = cls.process_shift

        def timed_process_shift(self, event, shift_name=''):
            if shift_name not in (None, ''):
                inst.expected_shift.add(shift_name)
            record = {"shift": shift_name, "n_events": len(event), "raw_bytes": 0, "block_cache_bytes": 0,
                      "blocks": 0, "max_rows": 0, "multicell_buffer_bytes": 0}
            inst.current = record
            t0 = time.perf_counter()
            result = process_shift(self, event, shift_name)
            record["wall"] = time.perf_counter() - t0
            record["maxrss_mb"] = maxrss_mb()
            if inst.pass_rss:
                import psutil
                record["rss_mb"] = psutil.Process().memory_info().rss / 2**20
            record["output_bytes"] = histogram_bytes(result)
            record["eft_output_bytes"] = histogram_bytes(result, "_eft")
            inst.passes.append(record)
            inst.current = None
            return result
        cls.process_shift = timed_process_shift

        process = cls.process

        def chunk_process(self, event):
            start = len(inst.passes)
            t0 = time.perf_counter()
            result = process(self, event)
            wall = time.perf_counter() - t0
            passes = inst.passes[start:]
            inst.chunks.append({"entrystart": event.metadata.get("entrystart"),
                                "entrystop": event.metadata.get("entrystop"), "n_events": len(event),
                                "wall": wall, "passes": len(passes), "merge": wall - sum(p["wall"] for p in passes)})
            return result
        cls.process = chunk_process

        if hasattr(wzmod, "fill_histogram"):
            self.install_engine(wzmod)

    def install_engine(self, wzmod):
        inst = self
        fill_histogram = wzmod.fill_histogram
        channel_block = wzmod.channel_block
        standard_weights = wzmod.standard_weights

        def recorded_fill_histogram(histos, key, values, labels, weight, exclude, pass_state):
            result = fill_histogram(histos, key, values, labels, weight, exclude, pass_state)
            rec = inst.current
            if rec is not None:
                cache = pass_state["block_cache"]
                rec["blocks"] = max(rec["blocks"], len(cache))
                rec["block_cache_bytes"] = max(
                    rec["block_cache_bytes"], sum(rows.nbytes for block in cache.values() for _, rows in block)
                )
                buffers = [v.nbytes for k, v in pass_state.items() if k.endswith("_buffer")]
                rec["multicell_buffer_bytes"] = max([rec["multicell_buffer_bytes"]] + buffers)
            return result
        wzmod.fill_histogram = recorded_fill_histogram

        def recorded_channel_block(exclude, pass_state):
            result = channel_block(exclude, pass_state)
            rec = inst.current
            if rec is not None:
                rec["max_rows"] = max(rec["max_rows"], sum(len(rows) for _, rows in result))
            return result
        wzmod.channel_block = recorded_channel_block

        def recorded_standard_weights(raw):
            rec = inst.current
            if rec is not None:
                rec["raw_bytes"] = max(rec["raw_bytes"], raw.nbytes)
            return standard_weights(raw)
        wzmod.standard_weights = recorded_standard_weights

    def expected(self):
        return {"nominal": ["nominal"], "shift": sorted(self.expected_shift), "weight": sorted(self.expected_weight)}


def processor_kwargs(args):
    return dict(era=args.era, version=args.version, run_period=args.run_period,
                split_by_charge=args.split_by_charge, split_FV=args.split_fv, pol=args.pol, dd_file=args.dd_file)


def run_flags(args):
    return {
        "file": args.file, "era": args.era, "version": args.version, "run_period": args.run_period,
        "dataset": args.dataset, "is_data": args.is_data, "is_signal": args.is_signal, "pol": args.pol,
        "split_by_charge": args.split_by_charge, "split_fv": args.split_fv,
        "eft_synthetic": args.eft_synthetic, "eft_operators": args.eft_operators,
        "eft_nan_modulus": args.eft_nan_modulus, "no_instrument": args.no_instrument, "random_seed": RANDOM_SEED,
        "dd_file": args.dd_file,
    }


def cmd_run(args):
    qawa_file = import_qawa_checked()
    import qawa.process.wztau2lnu_inclusive as wzmod
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

    if args.eft_synthetic:
        install_synthetic_eft(wzmod, args.eft_operators, args.eft_nan_modulus)
    inst = None
    if not args.no_instrument:
        inst = Instrument()
        inst.install(wzmod)

    t_start = time.perf_counter()
    processor_instance = wzmod.wzinclusive_processor(**processor_kwargs(args))
    t_built = time.perf_counter()
    metadata = {"dataset": args.dataset, "is_data": args.is_data, "is_signal": args.is_signal, "is_flashsim": False}
    events = NanoEventsFactory.from_root(
        {args.file: "Events"}, schemaclass=NanoAODSchema, mode="virtual",
        entry_start=args.start, entry_stop=args.stop, metadata=metadata,
    ).events()
    n_events = len(events)
    print(f"range = ({args.start}, {args.stop}), len(events) = {n_events}", flush=True)

    raised = None
    np.random.seed(RANDOM_SEED)
    t0 = time.perf_counter()
    try:
        output = processor_instance.process(events)
    except Exception as exc:
        if args.expect_raise and type(exc).__name__ == args.expect_raise:
            raised = f"{type(exc).__name__}: {exc}"
            output = None
        else:
            raise
    wall = time.perf_counter() - t0
    if args.expect_raise and raised is None:
        print(f"FAIL: expected {args.expect_raise}, nothing was raised")
        return 1

    timings = {
        "qawa_file": qawa_file, "construct": t_built - t_start, "process": wall, "maxrss_mb": maxrss_mb(),
        "passes": inst.passes if inst else [], "raised": raised,
    }
    if inst and inst.passes:
        timings["pre_merge_maxrss_mb"] = inst.passes[-1]["maxrss_mb"]
    print(f"process wall = {wall:.2f} s, final maxrss = {timings['maxrss_mb']:.1f} MB", flush=True)
    if inst:
        for p in inst.passes:
            print(f"  pass {p['shift']!s:<28} {p['wall']:8.3f} s  maxrss {p['maxrss_mb']:8.1f} MB", flush=True)
    if raised:
        print("raised as expected:", raised)
    dump = {
        "output": output, "expected": inst.expected() if inst else None, "range": (args.start, args.stop),
        "n_events": n_events, "timings": timings, "flags": run_flags(args),
    }
    if args.dump:
        save_dump(dump, args.dump)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(timings, f, indent=1, default=str)
    return 0


# ---------------------------------------------------------------- comparison

class UsageError(Exception):
    pass


def load_map(path):
    hist_map, axis_map = {}, {}
    if path is None:
        return hist_map, axis_map
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            kinds = {k.strip() for k in row["kinds"].split(",")}
            if "hist" in kinds:
                hist_map[row["old"]] = row["new"]
            if "axis" in kinds:
                axis_map[row["old"]] = row["new"]
    return hist_map, axis_map


def map_key(key, hist_map):
    if key in hist_map:
        return hist_map[key]
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if key.endswith(s) and key[:-len(s)] in hist_map:
            return hist_map[key[:-len(s)]] + s
    return key


def coverage(changed, removed, restored):
    sets = []
    for name in changed:
        sets.append(("changed", name, {name} | {name + s for s in SUFFIXES}))
    for name in removed:
        sets.append(("removed", name, {name} | {name + s for s in SUFFIXES}))
    for name in restored:
        sets.append(("restored", name, {name, name + "_eft"}))
    for (fa, na, ca), (fb, nb, cb) in itertools.combinations(sets, 2):
        overlap = ca & cb
        if overlap:
            raise UsageError(f"--{fa} {na} and --{fb} {nb} both cover {sorted(overlap)}")
    return sets


def is_category(ax):
    import hist
    return isinstance(ax, (hist.axis.StrCategory, hist.axis.IntCategory))


def category_labels(ax):
    return list(ax)


def axis_meta(ax):
    return {k: v for k, v in ax.__dict__.items() if k != "name"}


class Result:
    def __init__(self):
        self.failures = []
        self.notes = []
        self.stats = []

    def fail(self, rule, key, reason):
        self.failures.append((str(rule), key, reason))

    def note(self, text):
        self.notes.append(text)

    @property
    def passed(self):
        return not self.failures


def structure_diffs(rh, oh, axis_map, check_labels=True, check_category_sets=True):
    diffs = []
    if rh.storage_type is not oh.storage_type:
        diffs.append(f"storage {rh.storage_type} vs {oh.storage_type}")
    if rh.__dict__ != oh.__dict__:
        diffs.append(f"hist metadata {rh.__dict__} vs {oh.__dict__}")
    if len(rh.axes) != len(oh.axes):
        diffs.append(f"axis count {len(rh.axes)} vs {len(oh.axes)}")
        return diffs
    for i, (ra, oa) in enumerate(zip(rh.axes, oh.axes)):
        if type(ra) is not type(oa):
            diffs.append(f"axis {i} type {type(ra).__name__} vs {type(oa).__name__}")
            continue
        if axis_map.get(ra.name, ra.name) != oa.name:
            diffs.append(f"axis {i} name {ra.name} (mapped {axis_map.get(ra.name, ra.name)}) vs {oa.name}")
        if check_labels and axis_meta(ra) != axis_meta(oa):
            diffs.append(f"axis {i} metadata {axis_meta(ra)} vs {axis_meta(oa)}")
        if ra.traits != oa.traits:
            diffs.append(f"axis {i} traits {ra.traits} vs {oa.traits}")
        if is_category(ra):
            if check_category_sets and set(category_labels(ra)) != set(category_labels(oa)):
                diffs.append(f"axis {i} ({ra.name}) label sets differ: R-only {sorted(set(ra) - set(oa))} "
                             f"O-only {sorted(set(oa) - set(ra))}")
        elif not np.array_equal(ra.edges, oa.edges):
            diffs.append(f"axis {i} ({ra.name}) dense edges differ")
    return diffs


def aligned_views(rh, oh):
    rv = rh.view(flow=True)
    ov = oh.view(flow=True)
    for i, (ra, oa) in enumerate(zip(rh.axes, oh.axes)):
        if not is_category(ra):
            continue
        olabels = category_labels(oa)
        idx = [olabels.index(lab) for lab in category_labels(ra)]
        if oa.traits.overflow:
            idx.append(len(olabels))
        ov = np.take(ov, idx, axis=i)
    return rv, ov


def ulp_distance(a, b):
    ia = a.view(np.int64)
    ib = b.view(np.int64)
    ia = np.where(ia < 0, np.int64(-2**63) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2**63) - ib, ib)
    return np.abs(ia - ib)


def tolerant_cells(r, o, rtol):
    both_nan = np.isnan(r) & np.isnan(o)
    finite = np.isfinite(r) & np.isfinite(o)
    with np.errstate(invalid="ignore", over="ignore"):
        close = finite & (np.abs(r - o) <= rtol * np.maximum(np.abs(r), np.abs(o)))
    return both_nan | (r == o) | close


def order_tolerant_values(rval, oval, rvar, key, res):
    strict_bad = ~((rval == oval) | (np.isnan(rval) & np.isnan(oval)))
    if strict_bad.any():
        res.note(f"{key}: {int(strict_bad.sum())} strict value mismatches (information)")
    failing = ~tolerant_cells(rval, oval, ORDER_TOLERANT_RTOL)
    if not failing.any():
        return True
    cells = np.argwhere(failing)
    r, o, var = rval[failing], oval[failing], rvar[failing]
    with np.errstate(invalid="ignore", over="ignore"):
        diff = np.abs(r - o)
        cancellation = np.isfinite(diff) & (diff <= CANCELLATION_RTOL * np.sqrt(np.abs(var)))
    res.fail("2t", key, f"{len(cells)} value cells outside rule 2t ({int(cancellation.sum())} cancellation, "
                        f"{int((~cancellation).sum())} real)")
    for i in range(min(len(cells), MAX_LISTED_CELLS)):
        cell = tuple(int(c) for c in cells[i])
        res.note(f"{key}{cell}: |R-O| {diff[i]:.6g}, |R| {abs(r[i]):.6g}, |O| {abs(o[i]):.6g}, "
                 f"sqrt(var_R) {np.sqrt(abs(var[i])):.6g}, class {'cancellation' if cancellation[i] else 'real'}")
    return False


def compare_content(rh, oh, key, res, identical=False, order_tolerant=False):
    rv, ov = aligned_views(rh, oh)
    rval, oval = np.ascontiguousarray(rv["value"]), np.ascontiguousarray(ov["value"])
    rvar, ovar = np.ascontiguousarray(rv["variance"]), np.ascontiguousarray(ov["variance"])
    ok = True
    if order_tolerant:
        ok = order_tolerant_values(rval, oval, rvar, key, res)
    elif not np.array_equal(rval, oval, equal_nan=True):
        bad = ~((rval == oval) | (np.isnan(rval) & np.isnan(oval)))
        res.fail(2, key, f"{int(bad.sum())} value cells differ")
        ok = False
    bitwise = int((rval.view(np.uint64) != oval.view(np.uint64)).sum())
    if bitwise and ok:
        res.note(f"{key}: {bitwise} bitwise-only value mismatches")
        if identical:
            res.fail(2, key, f"{bitwise} bitwise value mismatches (--identical)")
            ok = False
    both_nan = np.isnan(rvar) & np.isnan(ovar)
    equal = rvar == ovar
    finite = np.isfinite(rvar) & np.isfinite(ovar)
    passes = tolerant_cells(rvar, ovar, VARIANCE_RTOL)
    differing = ~(both_nan | equal)
    ndiff = int(differing.sum())
    max_rel, max_ulp = 0.0, 0
    if ndiff:
        sel = differing & finite
        if sel.any():
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(rvar[sel] - ovar[sel]) / np.maximum(np.abs(rvar[sel]), np.abs(ovar[sel]))
            max_rel = float(np.nanmax(rel))
            max_ulp = int(ulp_distance(rvar[sel], ovar[sel]).max())
    res.stats.append((key, ndiff, max_rel, max_ulp))
    if not passes.all():
        res.fail(3, key, f"{int((~passes).sum())} variance cells outside tolerance (max rel {max_rel:.3g})")
        ok = False
    elif identical and ndiff:
        res.fail(3, key, f"{ndiff} variance cells differ (--identical)")
        ok = False
    return ok


def compare_hist(rh, oh, key, axis_map, res, identical=False, order_tolerant=False):
    diffs = structure_diffs(rh, oh, axis_map)
    if diffs:
        for d in diffs:
            res.fail(1, key, d)
        return
    compare_content(rh, oh, key, res, identical, order_tolerant)


def n_category_labels(h):
    return sum(len(ax) for ax in h.axes if is_category(ax))


def finite_abs_sum(h):
    v = h.view(flow=True)["value"]
    return float(np.abs(v[np.isfinite(v)]).sum())


def check_changed(name, cov, rmapped, ohists, axis_map, res, order_tolerant=False):
    for key in sorted(cov):
        rin, oin = key in rmapped, key in ohists
        if not rin and not oin:
            if key == name:
                res.note(f"--changed {name}: not booked in this configuration")
            continue
        if rin != oin:
            res.fail("changed", key, f"present in one tree only ({'R' if rin else 'O'})")
            continue
        rh, oh = rmapped[key][1], ohists[key]
        for d in structure_diffs(rh, oh, axis_map, check_category_sets=False):
            res.fail("changed", key, d)
        for i, (ra, oa) in enumerate(zip(rh.axes, oh.axes)):
            if is_category(ra) and is_category(oa):
                extra = set(oa) - set(ra)
                if extra:
                    res.fail("changed", key, f"subset: labels in O but not in R on axis {ra.name}: {sorted(extra)}")
        if key in SAME_SELECTION:
            partner = ohists.get(SAME_SELECTION[key])
            if partner is not None:
                for ax, pax in zip(oh.axes, partner.axes):
                    if is_category(ax) and set(ax) != set(pax):
                        res.fail("changed", key, f"label set of axis {ax.name} differs from {SAME_SELECTION[key]}")
        rch, och = rh.axes[0], oh.axes[0]
        if is_category(rch) and "Pol_reg" in rch and "Pol_reg" in och:
            import hist
            sub = Result()
            compare_content(rh[{rch.name: hist.loc("Pol_reg")}], oh[{och.name: hist.loc("Pol_reg")}],
                            key + "[Pol_reg]", sub, order_tolerant=order_tolerant)
            for f in sub.failures:
                res.fail("changed", f[1], f[2])


def check_removed(name, rmapped, ohists, res):
    if name in ohists:
        res.fail("removed", name, "removal not applied")
        return
    if name in rmapped:
        if n_category_labels(rmapped[name][1]) > 0:
            res.fail("removed", name, "would drop content")
            return
    else:
        res.note(f"--removed {name}: not booked in this configuration")
    for s in SUFFIXES:
        if name + s in rmapped or name + s in ohists:
            res.fail("removed", name + s, "derived key present")


def check_restored(name, rmapped, ohists, axis_map, res):
    eft = name + "_eft"
    rin, oin = name in rmapped, name in ohists
    if not rin and not oin:
        res.note(f"--restored {name}: not booked in this configuration")
    elif rin != oin:
        res.fail("restored", name, "present in one tree")
    else:
        rh, oh = rmapped[name][1], ohists[name]
        diffs = structure_diffs(rh, oh, axis_map, check_category_sets=False)
        if diffs:
            res.fail("restored", name, "structure differs: " + "; ".join(diffs))
        if n_category_labels(rh) > 0:
            res.fail("restored", name, "R not empty")
        if n_category_labels(oh) == 0 or finite_abs_sum(oh) <= 0:
            res.fail("restored", name, "O empty")
    if eft in rmapped:
        res.fail("restored", eft, "_eft in R")
    other_eft = any(k.endswith("_eft") and k != eft for k in ohists)
    if other_eft and oin and eft not in ohists:
        res.fail("restored", eft, "_eft missing in an EFT run")
    if eft in ohists:
        if not other_eft:
            res.fail("restored", eft, "_eft present in a run without other _eft keys")
        elif oin:
            eh, base = ohists[eft], ohists[name]
            names = [ax.name for ax in eh.axes]
            if (names[:3] != ["channel", "systematic", "eft"] or len(eh.axes) != len(base.axes) + 1
                    or any(type(a) is not type(b) or a.name != b.name or not np.array_equal(a.edges, b.edges)
                           for a, b in zip(list(eh.axes)[3:], list(base.axes)[2:]))):
                res.fail("restored", eft, "structure differs")
            ref = ohists.get("dilep_deta_eft")
            if ref is not None and set(eh.axes[2]) != set(ref.axes[2]):
                res.fail("restored", eft, "eft labels differ")


def label_slices(h, label):
    for i, ax in enumerate(h.axes):
        if ax.name == "systematic":
            labels = category_labels(ax)
            if label not in labels:
                return None
            v = h.view(flow=True)["value"]
            return np.take(v, labels.index(label), axis=i)
    return None


def check_vacuity(rdump, odump, res, report=True):
    rexp, oexp = rdump.get("expected"), odump.get("expected")
    if rexp is None or oexp is None:
        res.fail(4, "-", "expected labels not recorded")
        return
    for cls in ("nominal", "shift", "weight"):
        if sorted(rexp.get(cls, [])) != sorted(oexp.get(cls, [])):
            res.fail(4, "-", f"expected sets differ ({cls})")
    expected = sorted(set(itertools.chain.from_iterable(rexp.values())))
    present = set()
    for hists in rdump["output"].values():
        for h in hists.values():
            for ax in h.axes:
                if ax.name == "systematic":
                    present.update(ax)
    lines = []
    identical_to_nominal = []
    for label in expected:
        if label not in present:
            res.fail(4, label, "expected label absent in R")
            continue
        total, nhists, has_nan, same = 0.0, 0, False, True
        for hists in rdump["output"].values():
            for h in hists.values():
                sl = label_slices(h, label)
                if sl is None:
                    continue
                nhists += 1
                total += float(np.abs(sl[np.isfinite(sl)]).sum())
                has_nan |= bool(np.isnan(sl).any())
                if label != "nominal":
                    nom = label_slices(h, "nominal")
                    if nom is None or not np.array_equal(sl, nom, equal_nan=True):
                        same = False
        if total <= 0:
            res.fail(4, label, "NaN-only" if has_nan else "zero finite content")
        if label != "nominal" and same:
            identical_to_nominal.append(label)
        lines.append(f"  {label:<40} hists {nhists:4d}  finite sum|value| {total:.6g}")
    if report:
        res.note("per-label coverage (R):\n" + "\n".join(lines))
        if identical_to_nominal:
            res.note("labels identical to nominal: " + ", ".join(identical_to_nominal))


def compare_dumps(rdump, odump, hist_map=None, axis_map=None, changed=(), removed=(), restored=(),
                  identical=False, report=True, order_tolerant=False):
    hist_map = hist_map or {}
    axis_map = axis_map or {}
    sets = coverage(changed, removed, restored)
    covered = set().union(*[c for _, _, c in sets]) if sets else set()
    res = Result()
    rout, oout = rdump["output"], odump["output"]
    if set(rout) != set(oout):
        res.fail(1, "-", f"dataset keys differ: {sorted(rout)} vs {sorted(oout)}")
    for ds in sorted(set(rout) & set(oout)):
        rmapped = {}
        for k, h in rout[ds].items():
            mk = map_key(k, hist_map)
            if mk in rmapped:
                res.fail(1, mk, f"two R keys map to {mk}: {rmapped[mk][0]}, {k}")
            rmapped[mk] = (k, h)
        ohists = oout[ds]
        rkeys = set(rmapped) - covered
        okeys = set(ohists) - covered
        for k in sorted(rkeys - okeys):
            res.fail(1, k, "key missing in O")
        for k in sorted(okeys - rkeys):
            res.fail(1, k, "extra key in O")
        for k in sorted(rkeys & okeys):
            compare_hist(rmapped[k][1], ohists[k], k, axis_map, res, identical, order_tolerant)
        for flag, name, cov in sets:
            if flag == "changed":
                check_changed(name, cov, rmapped, ohists, axis_map, res, order_tolerant)
            elif flag == "removed":
                check_removed(name, rmapped, ohists, res)
            else:
                check_restored(name, rmapped, ohists, axis_map, res)
    check_vacuity(rdump, odump, res, report)
    return res


SUMW_MODES = ("strict", "order_tolerant", "events_vs_runs")


def compare_sumw(rsumw, osumw, mode):
    if mode not in SUMW_MODES:
        raise UsageError(f"unknown sumw mode {mode!r}")
    res = Result()
    if set(rsumw) != set(osumw):
        res.fail("sumw", "-", f"key sets differ: R-only {sorted(set(rsumw) - set(osumw))} "
                              f"O-only {sorted(set(osumw) - set(rsumw))}")
    for ds in sorted(set(rsumw) & set(osumw)):
        r, o = float(rsumw[ds]), float(osumw[ds])
        if not (np.isfinite(r) and np.isfinite(o)):
            res.fail("sumw", ds, f"non-finite sumw: R {r!r}, O {o!r}")
        elif (r == 0.0) != (o == 0.0):
            res.fail("sumw", ds, f"zero must match exactly: R {r!r}, O {o!r}")
        elif (r < 0.0 or o < 0.0) and r != o:
            res.fail("sumw", ds, f"data exact: R {r!r}, O {o!r}")
        elif mode == "strict" and r != o:
            res.fail("sumw", ds, f"strict: R {r!r}, O {o!r}")
        elif mode == "order_tolerant" and abs(r - o) > ORDER_TOLERANT_RTOL * max(abs(r), abs(o)):
            res.fail("sumw", ds, f"order_tolerant: R {r!r}, O {o!r}")
        elif mode == "events_vs_runs" and abs(r - o) > SUMW_RUNS_TOLERANCE * max(abs(r), abs(o)):
            res.fail("sumw", ds, f"events_vs_runs: R {r!r}, O {o!r}")
    return res


def save_dump(dump, path):
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dump(path):
    with open(path, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rb") if gzipped else open(path, "rb")) as f:
        return pickle.load(f)


def split_names(values):
    out = []
    for v in values or []:
        out.extend(n for n in v.split(",") if n)
    return out


def print_result(res):
    for n in res.notes:
        print("NOTE", n)
    if res.stats:
        ndiff = sum(s[1] for s in res.stats)
        max_rel = max(s[2] for s in res.stats)
        max_ulp = max(s[3] for s in res.stats)
        print(f"variance differences: {ndiff} cells over {len(res.stats)} hists, max rel {max_rel:.3g}, "
              f"max ulp {max_ulp}")
    for rule, key, reason in res.failures:
        print(f"FAIL rule {rule}: {key}: {reason}")
    print("PASS" if res.passed else f"FAIL ({len(res.failures)} failures)")


def cmd_compare(args):
    try:
        coverage(split_names(args.changed), split_names(args.removed), split_names(args.restored))
    except UsageError as exc:
        print("usage error:", exc)
        return 2
    hist_map, axis_map = load_map(args.rename_map)
    res = compare_dumps(load_dump(args.ref), load_dump(args.opt), hist_map, axis_map,
                        split_names(args.changed), split_names(args.removed), split_names(args.restored),
                        identical=args.identical, order_tolerant=args.order_tolerant)
    print_result(res)
    return 0 if res.passed else 1


# ---------------------------------------------------------------- selftest

def tree_copy(dump):
    out = dict(dump)
    out["output"] = {ds: dict(h) for ds, h in dump["output"].items()}
    out["expected"] = {k: list(v) for k, v in dump["expected"].items()}
    return out


def only_dataset(dump):
    (ds,) = dump["output"].keys()
    return dump["output"][ds]


def rebuild(h, axis_index=None, labels=None, dense=None, rename_axis=None):
    import hist
    axes = []
    for i, ax in enumerate(h.axes):
        if i == axis_index and labels is not None:
            new = hist.axis.StrCategory(labels, name=ax.name, label=ax.label, growth=ax.traits.growth)
            new.__dict__.update(ax.__dict__)
            axes.append(new)
        elif i == axis_index and dense is not None:
            axes.append(dense)
        else:
            axes.append(copy.deepcopy(ax))
        if rename_axis and axes[-1].name in rename_axis:
            new = copy.deepcopy(axes[-1])
            new.__dict__["name"] = rename_axis[ax.name]
            axes[-1] = new
    out = hist.Hist(*axes, storage=hist.storage.Weight())
    out.__dict__.update(h.__dict__)
    src = h.view(flow=True)
    dst = out.view(flow=True)
    if labels is not None:
        old = category_labels(h.axes[axis_index])
        for j, lab in enumerate(labels):
            if lab in old:
                sl_dst = [slice(None)] * dst.ndim
                sl_dst[axis_index] = j
                sl_src = [slice(None)] * src.ndim
                sl_src[axis_index] = old.index(lab)
                dst[tuple(sl_dst)] = src[tuple(sl_src)]
    else:
        dst[...] = src
    return out


def renamed(h, new_axis_names):
    return rebuild(h, rename_axis=new_axis_names)


def first_cell(h, predicate):
    v = h.view(flow=True)
    idx = np.argwhere(predicate(v))
    return tuple(idx[0]) if len(idx) else None


def mutate_cell(hists, field, predicate, change):
    for key in sorted(hists):
        h = hists[key]
        if n_category_labels(h) == 0:
            continue
        cell = first_cell(h, predicate)
        if cell is not None:
            h2 = copy.deepcopy(h)
            v = h2.view(flow=True)
            v[field][cell] = change(v[field][cell])
            hists[key] = h2
            return key
    raise RuntimeError("no suitable cell")


def finite_nonzero_value(v):
    return np.isfinite(v["value"]) & (v["value"] != 0) & np.isfinite(v["variance"]) & (v["variance"] != 0)


def modify_values_by(hists, key, factor):
    h = copy.deepcopy(hists[key])
    v = h.view(flow=True)
    cell = first_cell(h, finite_nonzero_value)
    if cell is None:
        raise RuntimeError(f"{key}: no finite non-zero cell")
    v["value"][cell] = v["value"][cell] * factor
    hists[key] = h
    return key


def modify_values(hists, key):
    return modify_values_by(hists, key, 1.5)


def drop_label_everywhere(dump, label):
    for hists in dump["output"].values():
        for key, h in list(hists.items()):
            for i, ax in enumerate(h.axes):
                if ax.name == "systematic" and label in ax:
                    hists[key] = rebuild(h, i, [lab for lab in ax if lab != label])


def zero_label_everywhere(dump, label, keep_nan=False):
    for hists in dump["output"].values():
        for key, h in list(hists.items()):
            for i, ax in enumerate(h.axes):
                if ax.name == "systematic" and label in ax:
                    h2 = copy.deepcopy(h)
                    v = h2.view(flow=True)["value"]
                    sl = [slice(None)] * v.ndim
                    sl[i] = category_labels(ax).index(label)
                    part = v[tuple(sl)]
                    if keep_nan:
                        part[np.isfinite(part)] = 0.0
                    else:
                        part[...] = 0.0
                    v[tuple(sl)] = part
                    hists[key] = h2


def write_map(path, rows):
    with open(path, "w") as f:
        f.write("old\tnew\tkinds\ttau_wp\n")
        for old, new, kinds in rows:
            f.write(f"{old}\t{new}\t{kinds}\t-\n")
    return path


def cmd_selftest(args):
    import hist
    dumps = {}
    for name in ("m1", "m6", "m7", "m1r", "m7r"):
        path = getattr(args, name)
        if path is None:
            raise SystemExit(f"selftest needs --{name}")
        dumps[name] = load_dump(path)
    for a, b in (("m1", "m1r"), ("m7", "m7r")):
        for field in ("range", "n_events", "flags"):
            if dumps[a][field] != dumps[b][field]:
                print(f"SELFTEST FAILURE: splice precondition: {field} differs between {a} and {b}")
                return 1
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    m7_nominal_nan = any(
        np.isnan(label_slices(h, "nominal")).any()
        for h in only_dataset(dumps["m7"]).values() if label_slices(h, "nominal") is not None
    )
    print(f"precondition: nominal has >= 1 NaN value cell in M7: {m7_nominal_nan}")
    if not m7_nominal_nan:
        print("SELFTEST FAILURE: M7 nominal has no NaN value cell; re-run M7 with --eft-nan-modulus 1")
        return 1

    rows = []

    def row(desc, dump, verdict, build, changed=(), removed=(), restored=(), map_rows=None, order_tolerant=False):
        rows.append((desc, dump, verdict, build, changed, removed, restored, map_rows, order_tolerant))

    def standard(dump_name, mut=None, mut_r=None):
        def build():
            r = tree_copy(dumps[dump_name])
            o = tree_copy(dumps[dump_name])
            if mut_r:
                mut_r(r)
            if mut:
                mut(o)
            return r, o
        return build

    def oh(d):
        return only_dataset(d)

    def nextafter_value(o):
        mutate_cell(oh(o), "value", finite_nonzero_value, lambda x: np.nextafter(x, np.inf))

    def scale_variance(factor):
        return lambda o: mutate_cell(oh(o), "variance", finite_nonzero_value, lambda x: x * (1 + factor))

    def zero_variance(o):
        mutate_cell(oh(o), "variance", lambda v: v["variance"] == 0, lambda x: 1e-300)

    def drop_one_label(o):
        hs = oh(o)
        for key in sorted(hs):
            ax = hs[key].axes[1]
            if len(ax) > 1:
                hs[key] = rebuild(hs[key], 1, list(ax)[1:])
                return
        raise RuntimeError("no hist with >1 systematic label")

    def some_weight_label(d):
        return sorted(d["expected"]["weight"] or d["expected"]["shift"])[0]

    def rename_key(o, key, new):
        hs = oh(o)
        hs[new] = renamed(hs.pop(key), {key: new})

    def rename_eft_key(o, key, new):
        hs = oh(o)
        hs[new + "_eft"] = renamed(hs.pop(key + "_eft"), {key: new})

    def permute(o):
        hs = oh(o)
        for key in sorted(hs):
            ax = hs[key].axes[1]
            if len(ax) > 1:
                hs[key] = rebuild(hs[key], 1, list(reversed(list(ax))))
                hs[key] = rebuild(hs[key], 0, list(reversed(list(hs[key].axes[0]))))
                return
        raise RuntimeError("no permutable hist")

    def add_label(key):
        def mut(o):
            hs = oh(o)
            hs[key] = rebuild(hs[key], 1, list(hs[key].axes[1]) + ["zz_added"])
        return mut

    def remove_label(key):
        def mut(o):
            hs = oh(o)
            hs[key] = rebuild(hs[key], 1, list(hs[key].axes[1])[1:])
        return mut

    def delete(key):
        return lambda d: oh(d).pop(key)

    def take(src, *keys):
        def mut(d):
            for k in keys:
                oh(d)[k] = only_dataset(dumps[src])[k]
        return mut

    def chain(*fs):
        def mut(d):
            for f in fs:
                f(d)
        return mut

    def modified(key):
        return lambda o: modify_values(oh(o), key)

    def rebuilt_dense(key, axis):
        def mut(d):
            hs = oh(d)
            h = hs[key]
            new = rebuild(h, len(h.axes) - 1, dense=axis)
            hs[key] = new
        return mut

    def drop_eft_label(key):
        def mut(d):
            hs = oh(d)
            labels = [lab for lab in hs[key].axes[2]]
            drop = [lab for lab in labels if lab != "lin_cW"][0]
            hs[key] = rebuild(hs[key], 2, [lab for lab in labels if lab != drop])
        return mut

    def add_expected(o):
        o["expected"]["weight"] = list(o["expected"]["weight"]) + ["zz_not_a_label"]

    m1_label = some_weight_label(dumps["m1"])
    row("one value cell +1 ulp", "m1", ("FAIL", "2"), standard("m1", nextafter_value))
    row("one variance x(1+2e-9)", "m1", ("FAIL", "3"), standard("m1", scale_variance(2e-9)))
    row("one variance x(1+5e-10)", "m1", ("PASS",), standard("m1", scale_variance(5e-10)))
    row("one zero variance -> 1e-300", "m1", ("FAIL", "3"), standard("m1", zero_variance))
    row("one systematic label dropped", "m1", ("FAIL", "1"), standard("m1", drop_one_label))
    row(f"expected label {m1_label} zeroed in R and O", "m1", ("FAIL", "4"),
        standard("m1", lambda o: zero_label_everywhere(o, m1_label), lambda r: zero_label_everywhere(r, m1_label)))
    row("nominal finite cells zeroed (NaN kept) in R and O", "m7", ("FAIL", "4", "NaN-only"),
        standard("m7", lambda o: zero_label_everywhere(o, "nominal", True),
                 lambda r: zero_label_everywhere(r, "nominal", True)))
    row(f"expected label {m1_label} removed from hists in R and O", "m1", ("FAIL", "4", "absent"),
        standard("m1", lambda o: drop_label_everywhere(o, m1_label), lambda r: drop_label_everywhere(r, m1_label)))
    row("label added to O's expected set only", "m1", ("FAIL", "4", "expected sets differ"),
        standard("m1", add_expected))
    row("key renamed without map", "m1", ("FAIL", "1"), standard("m1", lambda o: rename_key(o, "dilep_deta", "dilep_deta_r")))
    row("key renamed with a one-entry map", "m1", ("PASS",),
        standard("m1", lambda o: rename_key(o, "dilep_deta", "dilep_deta_r")),
        map_rows=[("dilep_deta", "dilep_deta_r", "hist,axis")])
    row("categorical axis order permuted", "m1", ("PASS",), standard("m1", permute))
    row("--changed hist with a label added in O", "m1", ("FAIL", "changed", "subset"),
        standard("m1", add_label("dilep_deta")), changed=["dilep_deta"])
    row("--changed hist with a label removed in O", "m1", ("PASS",),
        standard("m1", remove_label("dilep_deta")), changed=["dilep_deta"])
    row("--changed Y in neither tree", "m1", ("PASS", "not booked"), standard("m1"), changed=["zz_not_booked"])
    row("--changed X deleted from O only", "m1", ("FAIL", "changed", "present in one tree"),
        standard("m1", delete("dilep_deta")), changed=["dilep_deta"])
    row("selection-only map row met_pt -> high_met_pt", "m1", ("PASS",), standard("m1"),
        map_rows=[("met_pt", "high_met_pt", "selection")])
    row("same row with kinds hist,axis", "m1", ("FAIL", "1", "high_met_pt"), standard("m1"),
        map_rows=[("met_pt", "high_met_pt", "hist,axis")])
    row("--changed dilep_deta, only dilep_deta_eft modified", "m7", ("PASS",),
        standard("m7", modified("dilep_deta_eft")), changed=["dilep_deta"])
    row("dilep_deta_eft modified, no --changed", "m7", ("FAIL", "2"), standard("m7", modified("dilep_deta_eft")))
    row("suffix path _eft, both renamed", "m7", ("PASS",),
        standard("m7", chain(lambda o: rename_key(o, "dilep_deta", "dilep_deta_r"),
                             lambda o: rename_eft_key(o, "dilep_deta", "dilep_deta_r"))),
        map_rows=[("dilep_deta", "dilep_deta_r", "hist,axis")])
    row("suffix path _eft, _eft left under old key", "m7", ("FAIL", "1"),
        standard("m7", lambda o: rename_key(o, "dilep_deta", "dilep_deta_r")),
        map_rows=[("dilep_deta", "dilep_deta_r", "hist,axis")])
    row("--changed mT with mT_W_eft modified", "m7", ("FAIL", "2"), standard("m7", modified("mT_W_eft")), changed=["mT"])
    row("--changed dilep_dphi with dilep_dphi_tau_eft modified", "m7", ("FAIL", "2"),
        standard("m7", modified("dilep_dphi_tau_eft")), changed=["dilep_dphi"])
    pol_keys = [f"W_mass_wZW_{p}" for p in POLS]

    def rename_pol(o, skip=None):
        hs = oh(o)
        hs["W_mass_r"] = renamed(hs.pop("W_mass"), {"W_mass": "W_mass_r"})
        for k in pol_keys:
            if k != skip:
                hs[k.replace("W_mass", "W_mass_r")] = renamed(hs.pop(k), {"W_mass": "W_mass_r"})
    row("suffix path pol clones, all renamed", "m6", ("PASS",), standard("m6", rename_pol),
        map_rows=[("W_mass", "W_mass_r", "hist,axis,ntuple")])
    row("suffix path pol clones, one left", "m6", ("FAIL", "1"),
        standard("m6", lambda o: rename_pol(o, "W_mass_wZW_long")),
        map_rows=[("W_mass", "W_mass_r", "hist,axis,ntuple")])
    row("--changed W_mass, only W_mass_wZW_left modified", "m6", ("PASS",),
        standard("m6", modified("W_mass_wZW_left")), changed=["W_mass"])
    row("W_mass_wZW_left modified, no --changed", "m6", ("FAIL", "2"), standard("m6", modified("W_mass_wZW_left")))
    row("removed: dilep_tau_pt deleted, --removed", "m1", ("PASS",), standard("m1", delete("dilep_tau_pt")),
        removed=["dilep_tau_pt"])
    row("removed: same deletion, no flag", "m1", ("FAIL", "1"), standard("m1", delete("dilep_tau_pt")))
    row("--removed dilep_tau_pt, O unchanged", "m1", ("FAIL", "removed", "removal not applied"), standard("m1"),
        removed=["dilep_tau_pt"])
    row("dilep_deta deleted, --removed dilep_deta", "m1", ("FAIL", "removed", "would drop content"),
        standard("m1", delete("dilep_deta")), removed=["dilep_deta"])
    row("restored: mT_WZ from M1', --restored", "m1+m1r", ("PASS",), standard("m1", take("m1r", "mT_WZ")),
        restored=["mT_WZ"])
    row("restored: same O, no flag", "m1+m1r", ("FAIL", "1", "label sets"), standard("m1", take("m1r", "mT_WZ")))
    row("--restored mT_WZ, O unchanged", "m1", ("FAIL", "restored", "O empty"), standard("m1"), restored=["mT_WZ"])
    row("--restored dilep_deta, O unchanged", "m1", ("FAIL", "restored", "R not empty"), standard("m1"),
        restored=["dilep_deta"])
    row("restored: mT_WZ + mT_WZ_eft from M7'", "m7+m7r", ("PASS",),
        standard("m7", take("m7r", "mT_WZ", "mT_WZ_eft")), restored=["mT_WZ"])
    row("restored: only mT_WZ from M7'", "m7+m7r", ("FAIL", "restored", "_eft missing in an EFT run"),
        standard("m7", take("m7r", "mT_WZ")), restored=["mT_WZ"])
    row("restored: mT_WZ_eft in R", "m7+m7r", ("FAIL", "restored", "_eft in R"),
        standard("m7", take("m7r", "mT_WZ", "mT_WZ_eft"), take("m7r", "mT_WZ_eft")), restored=["mT_WZ"])
    row("restored: mT_WZ on Regular(150, 0, 1510)", "m1+m1r", ("FAIL", "restored", "structure differs"),
        standard("m1", chain(take("m1r", "mT_WZ"),
                             rebuilt_dense("mT_WZ", hist.axis.Regular(150, 0, 1510, name="mT_WZ",
                                                                      label=r"$m_{T}^{WZ}$ (GeV)")))),
        restored=["mT_WZ"])
    row("dilep_deta on Regular(20, 0, 8.4), no flag", "m1", ("FAIL", "1", "edges"),
        standard("m1", rebuilt_dense("dilep_deta", hist.axis.Regular(20, 0, 8.4, name="dilep_deta",
                                                                     label=r"$Delta\eta_{\ell\ell}$"))))
    row("restored: one eft label dropped from mT_WZ_eft", "m7+m7r", ("FAIL", "restored", "eft labels differ"),
        standard("m7", chain(take("m7r", "mT_WZ", "mT_WZ_eft"), drop_eft_label("mT_WZ_eft"))), restored=["mT_WZ"])
    row("mT_WZ deleted from O, --restored", "m1", ("FAIL", "restored", "present in one tree"),
        standard("m1", delete("mT_WZ")), restored=["mT_WZ"])
    row("R without mT_WZ, O with mT_WZ from M1'", "m1+m1r", ("FAIL", "restored", "present in one tree"),
        standard("m1", take("m1r", "mT_WZ"), delete("mT_WZ")), restored=["mT_WZ"])

    def removed_with_derived(o):
        hs = oh(o)
        hs["dilep_tau_pt_eft"] = hs.pop("dilep_tau_pt")
    row("removed with a derived key present", "m1", ("FAIL", "removed", "derived key present"),
        standard("m1", removed_with_derived), removed=["dilep_tau_pt"])
    row("mT/mT_WZ boundary on real _eft key", "m7r", ("FAIL", "2"), standard("m7r", modified("mT_WZ_eft")),
        changed=["mT"])
    row("--changed mT_W with mT_WZ modified", "m1r", ("FAIL", "2"), standard("m1r", modified("mT_WZ")),
        changed=["mT_W"])
    row("usage: --changed dilep_deta --restored dilep_deta", "m1", ("USAGE",), standard("m1"),
        changed=["dilep_deta"], restored=["dilep_deta"])
    row("usage: --changed dilep_deta --removed dilep_deta_eft", "m7", ("USAGE",), standard("m7"),
        changed=["dilep_deta"], removed=["dilep_deta_eft"])
    row("usage: --restored mT_WZ --changed mT_WZ_eft", "m7r", ("USAGE",), standard("m7r"),
        changed=["mT_WZ_eft"], restored=["mT_WZ"])

    def scale_value(factor):
        return lambda o: mutate_cell(oh(o), "value", finite_nonzero_value, lambda x: x * (1 + factor))

    def zero_value(o):
        mutate_cell(oh(o), "value", lambda v: v["value"] == 0, lambda x: 1e-300)

    def flip_variance_known(o):
        hs = oh(o)
        h = copy.deepcopy(hs["dilep_deta"])
        h.__dict__["_variance_known"] = not h.__dict__["_variance_known"]
        hs["dilep_deta"] = h

    def nan_to_one(o):
        mutate_cell(oh(o), "value", lambda v: np.isnan(v["value"]), lambda x: 1.0)

    def finite_to_nan(o):
        mutate_cell(oh(o), "value", finite_nonzero_value, lambda x: np.nan)

    def near_cancellation(shift):
        def set_small(d, offset):
            hs = oh(d)
            for key in sorted(hs):
                h = hs[key]
                if n_category_labels(h) == 0:
                    continue
                cell = first_cell(h, finite_nonzero_value)
                if cell is not None:
                    h2 = copy.deepcopy(h)
                    v = h2.view(flow=True)
                    v["value"][cell] = 1e-12 + offset * np.sqrt(v["variance"][cell])
                    hs[key] = h2
                    return
            raise RuntimeError("no suitable cell")
        return standard("m1", lambda o: set_small(o, shift), lambda r: set_small(r, 0.0))

    row("order-tolerant: one value cell x(1+5e-10)", "m1", ("PASS", "strict value mismatches"),
        standard("m1", scale_value(5e-10)), order_tolerant=True)
    row("order-tolerant: one value cell x(1+2e-9)", "m1", ("FAIL", "2t"), standard("m1", scale_value(2e-9)),
        order_tolerant=True)
    row("order-tolerant: one value cell +1 ulp", "m1", ("PASS", "bitwise"), standard("m1", nextafter_value),
        order_tolerant=True)
    row("order-tolerant: one zero value cell -> 1e-300", "m1", ("FAIL", "2t"), standard("m1", zero_value),
        order_tolerant=True)
    row("order-tolerant: NaN cells kept, one finite cell x(1+5e-10)", "m7", ("PASS",),
        standard("m7", lambda o: modify_values_by(oh(o), "dilep_deta_eft", 1 + 5e-10)), order_tolerant=True)
    row("order-tolerant: dilep_deta _variance_known flipped", "m1", ("FAIL", "1"), standard("m1", flip_variance_known),
        order_tolerant=True)
    row("order-tolerant: one variance x(1+2e-9)", "m1", ("FAIL", "3"), standard("m1", scale_variance(2e-9)),
        order_tolerant=True)
    row("order-tolerant: a NaN value cell of R is 1.0 in O", "m7", ("FAIL", "2t"), standard("m7", nan_to_one),
        order_tolerant=True)
    row("order-tolerant: a finite non-zero value cell is NaN in O", "m7", ("FAIL", "2t"),
        standard("m7", finite_to_nan), order_tolerant=True)
    row("order-tolerant: near-cancellation, O + 0.5e-9 sqrt(var)", "m1", ("FAIL", "2t", "class cancellation"),
        near_cancellation(0.5e-9), order_tolerant=True)
    row("order-tolerant: near-cancellation, O + 2e-9 sqrt(var)", "m1", ("FAIL", "2t", "class real"),
        near_cancellation(2e-9), order_tolerant=True)

    failures = 0
    for i, (desc, dump, verdict, build, changed, removed, restored, map_rows, order_tolerant) in enumerate(rows):
        hist_map, axis_map = {}, {}
        if map_rows:
            hist_map, axis_map = load_map(write_map(os.path.join(workdir, f"map_{i}.tsv"), map_rows))
        try:
            coverage(changed, removed, restored)
        except UsageError as exc:
            got = ("USAGE",)
            ok = verdict == got
            print(f"[{'ok' if ok else 'MISMATCH'}] {desc}: usage error ({exc})")
            failures += not ok
            continue
        r, o = build()
        res = compare_dumps(r, o, hist_map, axis_map, changed, removed, restored, report=False,
                            order_tolerant=order_tolerant)
        rules = sorted({f[0] for f in res.failures})
        text = " | ".join(f"{f[0]}:{f[1]}:{f[2]}" for f in res.failures) + " | " + " | ".join(res.notes)
        if verdict[0] == "PASS":
            ok = res.passed and (len(verdict) == 1 or verdict[1] in text)
        elif verdict[0] == "FAIL":
            ok = (not res.passed) and rules == [verdict[1]] and (len(verdict) < 3 or verdict[2] in text)
        else:
            ok = False
        failures += not ok
        got = "PASS" if res.passed else f"FAIL rules {rules}"
        print(f"[{'ok' if ok else 'MISMATCH'}] {desc} ({dump}): expected {verdict}, got {got}")
        if not ok:
            for f in res.failures[:10]:
                print("      ", f)

    x = 1234.5678901234567
    sumw_rows = [
        ("sumw x(1+5e-10), order_tolerant", {"ds": x}, {"ds": x * (1 + 5e-10)}, "order_tolerant", ("PASS",)),
        ("sumw x(1+2e-9), order_tolerant", {"ds": x}, {"ds": x * (1 + 2e-9)}, "order_tolerant", ("FAIL", "sumw")),
        ("sumw +1 ulp, strict", {"ds": x}, {"ds": np.nextafter(x, np.inf)}, "strict", ("FAIL", "sumw")),
        ("sumw dataset missing in O", {"ds": x, "ds2": x}, {"ds": x}, "strict", ("FAIL", "sumw", "key sets")),
        ("sumw R 0.0 vs O 1e-300, strict", {"ds": 0.0}, {"ds": 1e-300}, "strict", ("FAIL", "sumw", "zero")),
        ("sumw R 0.0 vs O 1e-300, order_tolerant", {"ds": 0.0}, {"ds": 1e-300}, "order_tolerant",
         ("FAIL", "sumw", "zero")),
        ("sumw O NaN, strict", {"ds": x}, {"ds": np.nan}, "strict", ("FAIL", "sumw", "non-finite")),
        ("sumw O NaN, order_tolerant", {"ds": x}, {"ds": np.nan}, "order_tolerant", ("FAIL", "sumw", "non-finite")),
        ("sumw x(1+5e-7), events_vs_runs", {"ds": x}, {"ds": x * (1 + 5e-7)}, "events_vs_runs", ("PASS",)),
        ("sumw x(1+2e-6), events_vs_runs", {"ds": x}, {"ds": x * (1 + 2e-6)}, "events_vs_runs", ("FAIL", "sumw")),
        ("sumw data -2.0 vs -1.0, events_vs_runs", {"ds": -1.0}, {"ds": -2.0}, "events_vs_runs",
         ("FAIL", "sumw", "data exact")),
    ]
    for desc, rsumw, osumw, mode, verdict in sumw_rows:
        res = compare_sumw(rsumw, osumw, mode)
        rules = sorted({f[0] for f in res.failures})
        text = " | ".join(f"{f[0]}:{f[1]}:{f[2]}" for f in res.failures)
        if verdict[0] == "PASS":
            ok = res.passed
        else:
            ok = (not res.passed) and rules == [verdict[1]] and (len(verdict) < 3 or verdict[2] in text)
        failures += not ok
        print(f"[{'ok' if ok else 'MISMATCH'}] {desc} (synthetic): expected {verdict}, "
              f"got {'PASS' if res.passed else 'FAIL ' + text}")
    print(f"selftest: {len(rows) + len(sumw_rows)} rows, {failures} mismatches")
    return 1 if failures else 0


# ---------------------------------------------------------------- scale

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP_ENV = {k: "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
ENGINE_CHOICES = ("reference", "worktree", "multicell", "multicell_pr20", "multicell_string", "chunked",
                  "integer_axis")
SCALE_DESCRIPTIONS = {
    "reference": "unmodified reference tree: per-(channel, systematic, variable) _histogram_filler closures",
    "worktree": "C1: hist.fill per (channel, slot) into the booked Weight hist, on the shared prep",
}
R3_REFERENCE_MB = 256.0
R4_REFERENCE_RATIO = 1.25
PROTOTYPE_FILL_SECONDS = {"A1": 15.9, "A2": 13.47, "A3": 1.56, "A4": 10.11, "A5": 2.13, "E1": 82.8}
PHASE_A_REFERENCE_PEAK_MB = {"A1": 2813, "A2": 2707, "A3": 1185, "A4": 5301, "A5": 1517, "E1": 3869}
PHASE_A_SPREAD_CONFIGS = ("A2", "A4", "E1")
SAMPLING_BAND_MB = (192.0, 320.0)
OTHER_PROCESS_RISE_MB = 1024.0
OTHERS_INTERVAL = 5.0
QUIET_SWAPOUT_WINDOW = 60.0
QUIET_POLL = 15.0
QUIET_AVAILABLE_MARGIN_MB = 2048.0


def rusage_info(pid):
    if sys.platform != "darwin":
        return None
    import ctypes
    import ctypes.util

    class RusageInfoV4(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
            "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time",
            "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins",
            "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default",
            "cpu_time_qos_maintenance", "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy",
            "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive", "billed_system_time",
            "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles",
            "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time")]

    libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
    info = RusageInfoV4()
    if libproc.proc_pid_rusage(int(pid), 4, ctypes.byref(info)) != 0:
        return None
    return {name: getattr(info, name) for name in ("phys_footprint", "lifetime_max_phys_footprint",
                                                    "diskio_bytesread", "diskio_byteswritten")}


def footprint_mb(pid):
    info = rusage_info(pid)
    if info is None:
        return None
    return info["phys_footprint"] / 2**20, info["lifetime_max_phys_footprint"] / 2**20


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error: {exc}"


def machine_state():
    import re
    import psutil
    pages = {}
    for line in sh("vm_stat").splitlines():
        m = re.match(r'"?([^:"]+)"?:\s+(\d+)', line.strip())
        if m:
            pages[m.group(1)] = int(m.group(2))
    page = 16384
    free = re.search(r"free percentage:\s*(\d+)", sh("memory_pressure -Q"))
    level = sh("sysctl -n kern.memorystatus_vm_pressure_level").strip()
    batt = sh("pmset -g batt")
    therm = sh("pmset -g therm")
    speed = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "loadavg": os.getloadavg(),
        "pressure_level": int(level) if level.isdigit() else None,
        "free_percent": int(free.group(1)) if free else None,
        "available_mb": psutil.virtual_memory().available / 2**20,
        "compressor_stored_gb": pages.get("Pages stored in compressor", 0) * page / 2**30,
        "swapouts": pages.get("Swapouts"), "swapins": pages.get("Swapins"),
        "swapusage": sh("sysctl -n vm.swapusage").strip(),
        "power_source": "AC" if "AC Power" in batt else ("battery" if "Battery Power" in batt else batt[:60]),
        "cpu_speed_limit": int(speed.group(1)) if speed else None,
    }


def not_quiet_reasons(name, state, older):
    reasons = []
    if state["pressure_level"] != 1:
        reasons.append(f"pressure level {state['pressure_level']}")
    if state["loadavg"][0] >= 2:
        reasons.append(f"1-min load {state['loadavg'][0]:.2f}")
    if older is None:
        reasons.append(f"swapouts not yet observed over {QUIET_SWAPOUT_WINDOW:.0f} s")
    elif state["swapouts"] != older["swapouts"]:
        reasons.append(f"swapouts rose by {state['swapouts'] - older['swapouts']} pages")
    need = PHASE_A_REFERENCE_PEAK_MB.get(name, 0.0) + QUIET_AVAILABLE_MARGIN_MB
    if state["available_mb"] < need:
        reasons.append(f"available memory {state['available_mb']:.0f} MB < {need:.0f} MB")
    return reasons


def wait_until_quiet(name, max_wait):
    start = time.monotonic()
    history = [(start, machine_state())]
    while True:
        now = time.monotonic()
        older = [st for t, st in history if now - t >= QUIET_SWAPOUT_WINDOW]
        reasons = not_quiet_reasons(name, history[-1][1], older[-1] if older else None)
        if not reasons or now - start >= max_wait:
            return history[-1][1], reasons, now - start
        time.sleep(QUIET_POLL)
        history.append((time.monotonic(), machine_state()))


class OthersSampler(threading.Thread):
    def __init__(self, interval=OTHERS_INTERVAL):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.start_mb = {}
        self.max_mb = {}
        self.names = {}
        self.series = []
        self.unreadable = 0

    def sample(self):
        import psutil
        own, uid = os.getpid(), os.getuid()
        total, unreadable = 0.0, 0
        for p in psutil.process_iter(["pid", "uids", "name"]):
            try:
                if p.info["pid"] == own or p.info["uids"].real != uid:
                    continue
            except (psutil.Error, AttributeError):
                continue
            fp = footprint_mb(p.info["pid"])
            if fp is None:
                unreadable += 1
                continue
            pid = p.info["pid"]
            self.names[pid] = p.info["name"]
            self.start_mb.setdefault(pid, 0.0 if self.series else fp[0])
            self.max_mb[pid] = max(self.max_mb.get(pid, 0.0), fp[0])
            total += fp[0]
        self.unreadable = max(self.unreadable, unreadable)
        self.series.append((time.time(), total))

    def run(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def summary(self):
        self.stop_event.set()
        self.join(timeout=60)
        rises = sorted(((self.max_mb[pid] - self.start_mb.get(pid, 0.0), pid, self.names.get(pid))
                        for pid in self.max_mb), reverse=True)
        sums = [v for _, v in self.series]
        return {
            "available": sys.platform == "darwin", "samples": len(self.series), "unreadable": self.unreadable,
            "sum_start_mb": sums[0] if sums else None, "sum_max_rise_mb": (max(sums) - sums[0]) if sums else None,
            "max_single_rise_mb": rises[0][0] if rises else None,
            "top_rises": [{"rise_mb": r, "pid": pid, "name": name} for r, pid, name in rises[:5]],
            "series": self.series,
        }


def eft_assertions(output, terms):
    problems = []
    for hists in output.values():
        for key, h in hists.items():
            if key.endswith("_eft") and len(h.axes[0]) and set(h.axes[2]) != set(terms):
                problems.append(f"(a) {key}: eft labels {sorted(h.axes[2])}")
        h = hists.get("dilep_deta_eft")
        if h is None:
            problems.append("(b) dilep_deta_eft missing")
            continue
        v = h.view(flow=True)["value"]
        labels = list(h.axes[2])
        nan_terms = {labels[i[2]] for i in np.argwhere(np.isnan(v))}
        if not nan_terms:
            problems.append("(b) dilep_deta_eft has no NaN value cell")
        elif nan_terms != {"lin_cW"}:
            problems.append(f"(b) NaN value cells in terms {sorted(nan_terms)}")
    return problems


def scale_paths(args):
    label = args.label or args.engine
    if args.sample > 1:
        label += f"_s{args.sample}"
    base = os.path.join(args.out, args.name)
    os.makedirs(base, exist_ok=True)
    return label, os.path.join(base, label + ".json"), os.path.join(base, label + ".pkl")


def cmd_scale_worker(args):
    tree = args.reference_tree if args.engine == "reference" else WORKTREE
    import_qawa_checked(os.path.join(tree, "src"))
    import qawa.process.wztau2lnu_inclusive as wzmod
    from coffea import processor
    from coffea.nanoevents import NanoAODSchema

    sampler = OthersSampler()
    sampler.start()
    if args.eft_synthetic:
        install_synthetic_eft(wzmod, args.eft_operators, args.eft_nan_modulus)
    description = SCALE_DESCRIPTIONS.get(args.engine)
    fill_seconds = [0.0]
    if args.engine != "reference":
        if args.engine != "worktree":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import fill_engines
            wzmod.fill_histogram = fill_engines.ENGINES[args.engine]
            description = fill_engines.ENGINE_DESCRIPTIONS[args.engine]
        fill = wzmod.fill_histogram

        def timed_fill(histos, key, values, labels, weight, exclude, pass_state):
            t0 = time.perf_counter()
            result = fill(histos, key, values, labels, weight, exclude, pass_state)
            fill_seconds[0] += time.perf_counter() - t0
            return result
        wzmod.fill_histogram = timed_fill
    inst = Instrument(pass_rss=True)
    inst.install(wzmod)

    proc = wzmod.wzinclusive_processor(**processor_kwargs(args))
    runner = processor.Runner(executor=processor.IterativeExecutor(), schema=NanoAODSchema,
                              chunksize=args.chunksize, maxchunks=args.maxchunks)
    fileset = {args.dataset: {"files": {args.file: "Events"},
                              "metadata": {"is_data": args.is_data, "is_signal": args.is_signal, "is_flashsim": False}}}
    np.random.seed(RANDOM_SEED)
    t0 = time.perf_counter()
    output = runner(fileset, proc, treename="Events")
    total = time.perf_counter() - t0
    nominal = [p["wall"] for p in inst.passes if p["shift"] is None]
    shifts = [p["wall"] for p in inst.passes if p["shift"] is not None]
    result = {
        "engine": args.engine, "description": description, "total": total, "peak_rss_mb": maxrss_mb(),
        "pass_rss_peak_mb": max((p["rss_mb"] for p in inst.passes), default=None),
        "chunks": inst.chunks, "nominal_pass": nominal,
        "shift_pass_mean": float(np.mean(shifts)) if shifts else None,
        "shift_pass_max": float(np.max(shifts)) if shifts else None,
        "fill_seconds": None if args.engine == "reference" else fill_seconds[0],
        "merge_in_process": sum(c["merge"] for c in inst.chunks),
        "outside_process": total - sum(c["wall"] for c in inst.chunks),
        "output_bytes": histogram_bytes(output),
        "max_raw_bytes": max((p["raw_bytes"] for p in inst.passes), default=0),
        "max_block_cache_bytes": max((p["block_cache_bytes"] for p in inst.passes), default=0),
        "max_blocks": max((p["blocks"] for p in inst.passes), default=0),
        "multicell_buffer_bytes": max((p["multicell_buffer_bytes"] for p in inst.passes), default=0),
        "random_seed": RANDOM_SEED, "passes": inst.passes,
    }
    t1 = time.perf_counter()
    result["gzip_pickle_bytes"] = len(gzip.compress(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL), 1))
    result["size_seconds"] = time.perf_counter() - t1
    label, json_path, dump_path = scale_paths(args)
    save_dump({"output": output, "expected": inst.expected(), "range": [(c["entrystart"], c["entrystop"])
                                                                      for c in inst.chunks],
               "n_events": sum(c["n_events"] for c in inst.chunks), "timings": result,
               "flags": {k: v for k, v in vars(args).items() if k != "func"}},
              dump_path)
    result["others"] = sampler.summary()
    footprint = footprint_mb(os.getpid())
    result["lifetime_max_footprint_mb"] = None if footprint is None else footprint[1]
    with open(json_path + ".worker", "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"{label}: total {total:.1f} s, lifetime max footprint {result['lifetime_max_footprint_mb']} MB, "
          f"ru_maxrss {result['peak_rss_mb']:.1f} MB, fill {result['fill_seconds']}", flush=True)
    return 0


def native_chunked_size(output):
    from hist.chunked import ChunkedHist
    natives = {ds: {k: ChunkedHist.from_hist(h) for k, h in hists.items()} for ds, hists in output.items()}
    return len(gzip.compress(pickle.dumps(natives, protocol=pickle.HIGHEST_PROTOCOL), 1))


def reliability(result):
    reasons = []
    if result.get("not_quiet_reasons"):
        reasons.append("not quiet at start (" + "; ".join(result["not_quiet_reasons"]) + ")")
    rise = (result.get("others") or {}).get("max_single_rise_mb")
    if rise is not None and rise > OTHER_PROCESS_RISE_MB:
        reasons.append(f"another process grew by {rise:.0f} MB")
    return "reliable" if not reasons else "unreliable: " + "; ".join(reasons)


def cmd_scale(args):
    if args.worker:
        return cmd_scale_worker(args)
    if args.engine == "reference" and not args.reference_tree:
        raise SystemExit("--engine reference requires --reference-tree")
    if args.engine != "reference" and args.reference_tree:
        raise SystemExit("--reference-tree is only valid with --engine reference")
    label, json_path, dump_path = scale_paths(args)
    state_before, not_quiet, waited = wait_until_quiet(args.name, args.quiet_wait)
    tree = args.reference_tree if args.engine == "reference" else WORKTREE
    env = dict(os.environ)
    env.update(CAP_ENV)
    env["PYTHONPATH"] = os.path.join(tree, "src")
    ref_json = os.path.join(args.out, args.name, "reference.json")
    timeout = None
    if args.engine != "reference" and os.path.exists(ref_json):
        with open(ref_json) as f:
            timeout = 3 * json.load(f)["total"]
    cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + ["--worker"]
    t0 = time.perf_counter()
    try:
        rc = subprocess.call(cmd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = None
    wall = time.perf_counter() - t0
    state_after = machine_state()
    if rc is None:
        result = {"engine": args.engine, "stopped": True, "total_lower_bound": wall,
                  "reason": f"exceeded 3x the reference wall time ({timeout:.0f} s)"}
    elif rc != 0:
        result = {"engine": args.engine, "failed": True, "rc": rc}
    else:
        with open(json_path + ".worker") as f:
            result = json.load(f)
        os.remove(json_path + ".worker")
        dump = load_dump(dump_path)
        result["native_chunkedhist_gzip_pickle_bytes"] = native_chunked_size(dump["output"])
        if args.engine == "reference" and args.eft_synthetic:
            result["eft_assertions"] = eft_assertions(dump["output"], synthetic_terms(args.eft_operators))
        ref_dump = os.path.join(args.out, args.name, "reference.pkl")
        if args.engine != "reference" and os.path.exists(ref_dump):
            hist_map, axis_map = load_map(args.rename_map)
            res = compare_dumps(load_dump(ref_dump), dump, hist_map, axis_map, split_names(args.changed),
                                split_names(args.removed), split_names(args.restored), report=False)
            result["agreement"] = {"passed": res.passed, "failures": res.failures[:50], "notes": res.notes[:20],
                                   "variance_cells": sum(x[1] for x in res.stats),
                                   "max_rel": max((x[2] for x in res.stats), default=0.0),
                                   "max_ulp": max((x[3] for x in res.stats), default=0)}
            print("agreement:", "PASS" if res.passed else f"FAIL {res.failures[:5]}", flush=True)
        del dump
        if args.engine != "reference" and not args.keep_dump:
            os.remove(dump_path)
    result["label"] = label
    result["variant"] = args.label or args.engine
    result.setdefault("description", SCALE_DESCRIPTIONS.get(args.engine, args.engine))
    result["state_before"] = state_before
    result["state_after"] = state_after
    result["not_quiet_reasons"] = not_quiet
    result["quiet_wait_seconds"] = waited
    result["reliability"] = reliability(result)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1, default=str)
    return 0 if rc == 0 else 1


# ---------------------------------------------------------------- report

def median_of(results, field):
    values = [r[field] for r in results if r.get(field) is not None]
    return float(np.median(values)) if values else None


def memory_metric(r):
    if r.get("lifetime_max_footprint_mb") is not None:
        return r["lifetime_max_footprint_mb"]
    return r.get("peak_rss_mb")


def fmt(x, spec=".1f"):
    return "n/a" if x is None else format(x, spec)


def sample_row(label, r):
    if r.get("stopped"):
        return (f"| {label} | > {r['total_lower_bound']:.0f} (stopped) | | | | | | | | | | | "
                f"{r.get('reliability', '')} |")
    if r.get("failed"):
        return f"| {label} | FAILED rc={r['rc']} | | | | | | | | | | | {r.get('reliability', '')} |"
    agree = r.get("agreement")
    if r["engine"] == "reference":
        agree_text = "reference"
    elif agree is None:
        agree_text = "missing"
    else:
        agree_text = ("PASS" if agree["passed"] else "FAIL") + f" (var cells {agree['variance_cells']})"
    fill = "n/a (closures)" if r.get("fill_seconds") is None else f"{r['fill_seconds']:.2f}"
    before, after = r.get("state_before") or {}, r.get("state_after") or {}
    swapouts = None
    if before.get("swapouts") is not None and after.get("swapouts") is not None:
        swapouts = after["swapouts"] - before["swapouts"]
    return (f"| {label} | {r['total']:.1f} | {sum(r['nominal_pass']):.1f} | "
            f"{fmt(r['shift_pass_mean'], '.2f')}/{fmt(r['shift_pass_max'], '.2f')} | {fill} | "
            f"{r['merge_in_process']:.1f} | {fmt(r.get('lifetime_max_footprint_mb'), '.0f')} | "
            f"{r['peak_rss_mb']:.0f} | {fmt(r.get('pass_rss_peak_mb'), '.0f')} | "
            f"{before.get('pressure_level')}/{after.get('pressure_level')} | {fmt(swapouts, 'd')} | "
            f"{r['gzip_pickle_bytes'] / 2**20:.1f} | {agree_text} | {r.get('reliability', '')} |")


def cmd_report(args):
    configs = sorted(d for d in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, d)))
    if args.names:
        configs = [c for c in configs if c in split_names(args.names)]
    lines = []
    g1_failures = []
    for name in configs:
        results = {}
        for fn in sorted(os.listdir(os.path.join(args.out, name))):
            if fn.endswith(".json"):
                with open(os.path.join(args.out, name, fn)) as f:
                    r = json.load(f)
                results.setdefault(r["variant"], []).append(r)
        lines.append(f"\n### {name}\n")
        lines.append("| sample | total (s) | nominal pass (s) | shift pass mean/max (s) | fill (s) | merge (s) | "
                     "lifetime max footprint (MB) | ru_maxrss (MB) | psutil pass peak (MB) | pressure before/after | "
                     "swapouts | output gzip (MB) | agreement | reliability |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for variant, rs in results.items():
            lines.append(f"| **{variant}**: {rs[0].get('description', '')} | | | | | | | | | | | | | |")
            for r in rs:
                lines.append(sample_row(r["label"], r))
        ref = [r for r in results.get("reference", []) if not r.get("failed") and not r.get("stopped")]
        c1 = [r for r in results.get("worktree", []) if not r.get("failed") and not r.get("stopped")]
        g1 = [r for r in results.get("worktree", [])
              if r.get("failed") or r.get("stopped") or not (r.get("agreement") or {}).get("passed")]
        if not results.get("worktree") or g1:
            g1_failures.append(name)
            lines.append(f"\n**G1 (required): FAIL** - C1 samples without a passing agreement: "
                         f"{[r['label'] for r in g1] or 'no C1 sample'}")
        else:
            lines.append(f"\n**G1 (required): PASS** ({len(c1)} C1 sample(s) agree with the reference)")
        if not ref or not c1:
            lines.append("R2-R4: not evaluated (reference or C1 missing)")
            continue
        ref_total, c1_total = median_of(ref, "total"), median_of(c1, "total")
        lines.append(f"- R2 time: reference {ref_total:.1f} s, C1 {c1_total:.1f} s (medians), "
                     f"C1/reference {c1_total / ref_total:.3f}. Recommendation: "
                     + ("C1 is the default." if c1_total <= ref_total else
                        "C1 is slower than the reference here; the hand-back names the slower stage "
                        "(nominal pass, shift passes or merge columns above)."))
        ref_mem = float(np.median([memory_metric(r) for r in ref]))
        c1_mems = [memory_metric(r) for r in c1]
        delta = float(np.median(c1_mems)) - ref_mem
        metric = "lifetime max footprint" if c1[0].get("lifetime_max_footprint_mb") is not None else "ru_maxrss"
        lines.append(f"- R3 memory ({metric}): reference {ref_mem:.0f} MB, C1 {np.median(c1_mems):.0f} MB, "
                     f"C1 - reference {delta:+.0f} MB (reference line +{R3_REFERENCE_MB:.0f} MB). "
                     f"Recommendation: HTCondor request_memory >= {1.2 * max(c1_mems):.0f} MB "
                     f"(measured C1 peak {max(c1_mems):.0f} MB + 20 %).")
        if (SAMPLING_BAND_MB[0] <= delta <= SAMPLING_BAND_MB[1] or name in PHASE_A_SPREAD_CONFIGS) \
                and (len(ref) < 3 or len(c1) < 3):
            lines.append("  sampling rule: two more samples of each variant")
        proto = PROTOTYPE_FILL_SECONDS.get(name)
        c1_fill = median_of(c1, "fill_seconds")
        if proto is not None and c1_fill is not None:
            ratio = c1_fill / proto
            lines.append(f"- R4 fill against the phase-A prototype: C1 {c1_fill:.2f} s, prototype {proto:.2f} s, "
                         f"ratio {ratio:.3f} (reference line {R4_REFERENCE_RATIO})"
                         + ("; above the line: investigate and explain in impl-notes." if ratio > R4_REFERENCE_RATIO
                            else "."))
        for variant, rs in results.items():
            if variant not in ("reference", "worktree"):
                fills = median_of(rs, "fill_seconds")
                lines.append(f"- information, {variant}: fill {fmt(fills, '.2f')} s, "
                             f"total {fmt(median_of(rs, 'total'))} s")
        unreliable = [r for rs in results.values() for r in rs if r.get("reliability", "").startswith("unreliable")]
        for r in unreliable:
            top = ((r.get("others") or {}).get("top_rises") or [])[:5]
            lines.append(f"  {r['label']}: {r['reliability']}; largest other processes: "
                         + ", ".join(f"{t['name']} ({t['pid']}) {t['rise_mb']:+.0f} MB" for t in top))
    print("\n".join(lines))
    print()
    if g1_failures:
        print("G1 FAILED: " + ", ".join(g1_failures))
        return 1
    print("G1 passed in every configuration (R2-R4 are information)")
    return 0


# ---------------------------------------------------------------- argparse

def add_run_args(p):
    p.add_argument("--file", required=True)
    p.add_argument("--era", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--run-period", default="")
    p.add_argument("--dataset", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--is-data", dest="is_data", action="store_true")
    g.add_argument("--is-mc", dest="is_data", action="store_false")
    p.add_argument("--is-signal", action="store_true")
    p.add_argument("--pol", action="store_true")
    p.add_argument("--split-by-charge", action="store_true")
    p.add_argument("--split-fv", action="store_true")
    p.add_argument("--eft-synthetic", action="store_true")
    p.add_argument("--eft-operators", type=int, default=1)
    p.add_argument("--eft-nan-modulus", type=int, default=50)
    p.add_argument("--dd-file", help="data-driven estimate (correctionlib JSON) replacing the era default (data)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run the processor on one entry range and dump the output")
    add_run_args(p)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--no-instrument", action="store_true", help="no wrappers; only total time and RSS")
    p.add_argument("--dump")
    p.add_argument("--json")
    p.add_argument("--expect-raise", help="exception type name that must be raised")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="compare two dumps (exit 1 on failure, 2 on usage error)")
    p.add_argument("ref")
    p.add_argument("opt")
    p.add_argument("--rename-map")
    p.add_argument("--changed", action="append")
    p.add_argument("--removed", action="append")
    p.add_argument("--restored", action="append")
    p.add_argument("--identical", action="store_true", help="also fail on bitwise or variance differences")
    p.add_argument("--order-tolerant", action="store_true",
                   help="rule 2t: values within 1e-9 relative (for different summation orders); strict mismatches "
                        "are printed as information and failing cells are classed cancellation or real")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("selftest", help="compare against mutated copies of real dumps")
    for name in ("m1", "m6", "m7", "m1r", "m7r"):
        p.add_argument(f"--{name}")
    p.add_argument("--workdir", default=os.path.join(os.getcwd(), "selftest_maps"))
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("scale", help="one variant of the at-scale measurement, run in a fresh subprocess")
    add_run_args(p)
    p.add_argument("--engine", required=True, choices=ENGINE_CHOICES)
    p.add_argument("--name", required=True, help="configuration name (A1 ... E1)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--chunksize", type=int, default=100000)
    p.add_argument("--maxchunks", type=int, required=True)
    p.add_argument("--reference-tree")
    p.add_argument("--rename-map")
    p.add_argument("--changed", action="append")
    p.add_argument("--removed", action="append")
    p.add_argument("--restored", action="append")
    p.add_argument("--label", help="output label (default: the engine name)")
    p.add_argument("--sample", type=int, default=1, help="sample number for the sampling rule")
    p.add_argument("--quiet-wait", type=float, default=1800.0,
                   help="seconds to wait for a quiet machine before running anyway (labelled not quiet)")
    p.add_argument("--keep-dump", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_scale)

    p = sub.add_parser("report", help="G1 verdict and R2-R4 information rows over the scale results in a directory")
    p.add_argument("--out", required=True)
    p.add_argument("--names", action="append")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
