import argparse
import copy
import csv
import gzip
import itertools
import json
import os
import pickle
import resource
import sys
import time

import numpy as np

DESCRIPTION = """Equality and timing harness for wzinclusive_processor.

Subcommands:
  run       run the processor on one entry range of one file and dump the output
  compare   compare two dumps under the comparison rule of the optimisation plan
  selftest  check that compare reaches the expected verdict on mutated dumps

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
    def __init__(self):
        self.passes = []
        self.expected_shift = set()
        self.expected_weight = set()

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
            record = {"shift": shift_name, "n_events": len(event)}
            t0 = time.perf_counter()
            result = process_shift(self, event, shift_name)
            record["wall"] = time.perf_counter() - t0
            record["maxrss_mb"] = maxrss_mb()
            record["output_bytes"] = histogram_bytes(result)
            record["eft_output_bytes"] = histogram_bytes(result, "_eft")
            inst.passes.append(record)
            return result
        cls.process_shift = timed_process_shift

    def expected(self):
        return {"nominal": ["nominal"], "shift": sorted(self.expected_shift), "weight": sorted(self.expected_weight)}


def processor_kwargs(args):
    return dict(era=args.era, version=args.version, run_period=args.run_period,
                split_by_charge=args.split_by_charge, split_FV=args.split_fv, pol=args.pol)


def run_flags(args):
    return {
        "file": args.file, "era": args.era, "version": args.version, "run_period": args.run_period,
        "dataset": args.dataset, "is_data": args.is_data, "is_signal": args.is_signal, "pol": args.pol,
        "split_by_charge": args.split_by_charge, "split_fv": args.split_fv,
        "eft_synthetic": args.eft_synthetic, "eft_operators": args.eft_operators,
        "eft_nan_modulus": args.eft_nan_modulus, "no_instrument": args.no_instrument, "random_seed": RANDOM_SEED,
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


def compare_content(rh, oh, key, res, identical=False):
    rv, ov = aligned_views(rh, oh)
    rval, oval = np.ascontiguousarray(rv["value"]), np.ascontiguousarray(ov["value"])
    rvar, ovar = np.ascontiguousarray(rv["variance"]), np.ascontiguousarray(ov["variance"])
    ok = True
    if not np.array_equal(rval, oval, equal_nan=True):
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
    with np.errstate(invalid="ignore", over="ignore"):
        close = finite & (np.abs(rvar - ovar) <= VARIANCE_RTOL * np.maximum(np.abs(rvar), np.abs(ovar)))
    passes = both_nan | equal | close
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


def compare_hist(rh, oh, key, axis_map, res, identical=False):
    diffs = structure_diffs(rh, oh, axis_map)
    if diffs:
        for d in diffs:
            res.fail(1, key, d)
        return
    compare_content(rh, oh, key, res, identical)


def n_category_labels(h):
    return sum(len(ax) for ax in h.axes if is_category(ax))


def finite_abs_sum(h):
    v = h.view(flow=True)["value"]
    return float(np.abs(v[np.isfinite(v)]).sum())


def check_changed(name, cov, rmapped, ohists, axis_map, res):
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
                            key + "[Pol_reg]", sub)
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
                  identical=False, report=True):
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
            compare_hist(rmapped[k][1], ohists[k], k, axis_map, res, identical)
        for flag, name, cov in sets:
            if flag == "changed":
                check_changed(name, cov, rmapped, ohists, axis_map, res)
            elif flag == "removed":
                check_removed(name, rmapped, ohists, res)
            else:
                check_restored(name, rmapped, ohists, axis_map, res)
    check_vacuity(rdump, odump, res, report)
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
                        identical=args.identical)
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


def modify_values(hists, key):
    h = copy.deepcopy(hists[key])
    v = h.view(flow=True)
    cell = first_cell(h, finite_nonzero_value)
    if cell is None:
        raise RuntimeError(f"{key}: no finite non-zero cell")
    v["value"][cell] = v["value"][cell] * 1.5
    hists[key] = h
    return key


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

    def row(desc, dump, verdict, build, changed=(), removed=(), restored=(), map_rows=None):
        rows.append((desc, dump, verdict, build, changed, removed, restored, map_rows))

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

    failures = 0
    for i, (desc, dump, verdict, build, changed, removed, restored, map_rows) in enumerate(rows):
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
        res = compare_dumps(r, o, hist_map, axis_map, changed, removed, restored, report=False)
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
    print(f"selftest: {len(rows)} rows, {failures} mismatches")
    return 1 if failures else 0


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
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("selftest", help="compare against mutated copies of real dumps")
    for name in ("m1", "m6", "m7", "m1r", "m7r"):
        p.add_argument(f"--{name}")
    p.add_argument("--workdir", default=os.path.join(os.getcwd(), "selftest_maps"))
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
