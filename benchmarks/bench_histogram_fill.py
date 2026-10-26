import argparse
import gzip
import json
import os
import pickle
import subprocess
import sys
import time

import boost_histogram as bh
import hist
import numpy as np
from coffea.analysis_tools import PackedSelection
from hist.chunked import ChunkedHist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_processor as bp  # noqa: E402
import fill_engines  # noqa: E402
import qawa.process.wztau2lnu_inclusive as wzmod  # noqa: E402

DESCRIPTION = """Synthetic, file-free micro-benchmark of the histogram fill strategies of wzinclusive_processor.

Every strategy fills the same rows with the same weights as C1 (wztau2lnu_inclusive.fill_histogram),
through the shared prep of the processor module (channel_block, standard_weights, eft_term_weights).

Shapes (channels x slots x variables):
  nominal  27 x 35 x 55, one pass per chunk
  shift    27 x 1 x 55, 38 passes per chunk
  data     27 x 85 x 55, one pass per chunk
  eft      27 x (35 x 21) x 5, one pass per chunk

Dense axes come from wzinclusive_processor(era="2024", version="v15").build_histos(). The N-1
structure is a frozen copy of the 27 standard channel requirement lists of
wzinclusive_processor.process_shift (src/qawa/process/wztau2lnu_inclusive.py, branch
wz_fill_optimization at f40514c), plus one channel that selects no event. Values include NaN and
under/overflow entries; weights include NaN and +-inf.

Strategies: c1, every fill_engines.ENGINES entry, chunked_persistent (one ChunkedHist per target,
kept across chunks and passes, one to_hist() at the end) and multicell_growth (a MultiCell target
with a growing channel axis, which shows the boost-histogram growth-resize bug).

Per strategy: best-of-repeat fill time, accumulation time, deterministic bytes, peak RSS and
lifetime-max footprint (fresh subprocess, --only X --rss), pickle+gzip sizes of the native object and
of hist.Hist, and exactness against C1 under the comparison rule of bench_processor.py. A mismatch
is reported as a failure of that strategy. The accumulation row compares Hist +, ChunkedHist += and
merge_histogram_sets on the C1 outputs.
"""

COMMON = ["triggers", "lumimask", "metfilter", "require-ossf", "require-2lep", "dilep_m", "dilep_dphi_met", "dilep_pt"]
TAU_VTIGHT = ["1nhtaus_vtight", "~2plusnhtaus_tight"]
TAU_LOOSE = ["1nhtaus_loose", "~1nhtaus_tight", "~1nhtaus_vtight"]
TAU_TIGHT = ["1nhtaus_tight", "~2plusnhtaus_loose", "~1nhtaus_vtight"]
CHANNEL_TAILS = {
    "inc-SR0": ["0njets"] + TAU_VTIGHT + ["high_met_pt"],
    "inc-SR1": ["1njets_only", "1nhtaus_vtight", "high_met_pt"],
    "inc-SR01": ["1njets"] + TAU_VTIGHT + ["high_met_pt"],
    "inc-VR0": ["0njets"] + TAU_VTIGHT + ["medium_met_pt"],
    "inc-VR1": ["1njets_only", "1nhtaus_vtight", "medium_met_pt"],
    "inc-VR01": ["1njets", "1nhtaus_vtight", "medium_met_pt"],
    "inc-VB0": ["0njets"] + TAU_LOOSE + ["medium_met_pt"],
    "inc-VB1": ["1njets_only"] + TAU_LOOSE + ["medium_met_pt"],
    "inc-VB01": ["1njets"] + TAU_LOOSE + ["medium_met_pt"],
    "inc-B0": ["0njets"] + TAU_LOOSE + ["high_met_pt"],
    "inc-B1": ["1njets_only"] + TAU_LOOSE + ["high_met_pt"],
    "inc-B01": ["1njets"] + TAU_LOOSE + ["high_met_pt"],
    "inc-C0": ["low_met_pt", "1nhtaus_loose", "0njets", "~1nhtaus_tight", "~1nhtaus_vtight"],
    "inc-C01": ["low_met_pt", "1nhtaus_loose", "1njets", "~1nhtaus_tight", "~1nhtaus_vtight"],
    "inc-C1": ["low_met_pt", "1nhtaus_loose", "1njets_only", "~1nhtaus_tight", "~1nhtaus_vtight"],
    "inc-D0": ["0njets"] + TAU_VTIGHT + ["low_met_pt"],
    "inc-D01": ["1njets"] + TAU_VTIGHT + ["low_met_pt"],
    "inc-D1": ["1njets_only"] + TAU_VTIGHT + ["low_met_pt"],
    "inc-IR0L": ["0njets"] + TAU_TIGHT + ["low_met_pt"],
    "inc-IR01L": ["1njets"] + TAU_TIGHT + ["low_met_pt"],
    "inc-IR1L": ["1njets_only"] + TAU_TIGHT + ["low_met_pt"],
    "inc-IR0M": ["0njets"] + TAU_TIGHT + ["medium_met_pt"],
    "inc-IR1M": ["1njets_only"] + TAU_TIGHT + ["medium_met_pt"],
    "inc-IR01M": ["1njets"] + TAU_TIGHT + ["medium_met_pt"],
    "inc-IR0H": ["0njets"] + TAU_TIGHT + ["high_met_pt"],
    "inc-IR01H": ["1njets"] + TAU_TIGHT + ["high_met_pt"],
    "inc-IR1H": ["1njets_only"] + TAU_TIGHT + ["high_met_pt"],
}
CHANNELS = {name: COMMON + tail for name, tail in CHANNEL_TAILS.items()}
CHANNELS["empty"] = COMMON + ["never"]
EFT_VARIABLES = ("tau_loosetovtight_mT_WZ", "tau_loosetovtight_dilep_met_hadron_mt", "dilep_pt", "met_pt",
                 "tau_loosetovtight_inv_m_WZ")
SHAPES = {
    "nominal": dict(slots=35, terms=0, passes=1, variables=wzmod.STANDARD_FILL_VARIABLES),
    "shift": dict(slots=1, terms=0, passes=38, variables=wzmod.STANDARD_FILL_VARIABLES),
    "data": dict(slots=85, terms=0, passes=1, variables=wzmod.STANDARD_FILL_VARIABLES),
    "eft": dict(slots=35, terms=21, passes=1, variables=EFT_VARIABLES),
}
DATASET = "synthetic"


def booked_axes():
    histos = wzmod.wzinclusive_processor(era="2024", version="v15").build_histos()
    return {var: list(histos[var].axes)[2:] for var in wzmod.STANDARD_FILL_VARIABLES}


def make_selection(rng, n):
    selection = PackedSelection(dtype="uint64")
    met = rng.choice(3, n, p=[0.4, 0.35, 0.25])
    for k, name in enumerate(("low_met_pt", "medium_met_pt", "high_met_pt")):
        selection.add(name, met == k)
    njets = rng.choice(3, n, p=[0.5, 0.3, 0.2])
    selection.add("0njets", njets == 0)
    selection.add("1njets_only", njets == 1)
    selection.add("1njets", njets >= 1)
    names = {r.lstrip("~") for reqs in CHANNELS.values() for r in reqs} - set(selection.names) - {"never"}
    for name in sorted(names):
        selection.add(name, rng.random(n) < (0.95 if name in ("triggers", "lumimask", "metfilter") else 0.7))
    selection.add("never", np.zeros(n, dtype=bool))
    return selection


def make_values(rng, n, variables, axes):
    values = {}
    for var in variables:
        edges = axes[var][0].edges
        span = edges[-1] - edges[0]
        v = rng.uniform(edges[0] - 0.1 * span, edges[-1] + 0.1 * span, n)
        v[rng.random(n) < 0.01] = np.nan
        values[var] = v
    return values


def make_raw(rng, k, n):
    raw = rng.uniform(0.2, 1.8, (k, n))
    for bad in (np.nan, np.inf, -np.inf):
        raw[rng.random((k, n)) < 1e-3] = bad
    return raw


def pass_inputs(shape, axes, n, chunk, p):
    rng = np.random.default_rng([chunk, p])
    spec = SHAPES[shape]
    selection = make_selection(rng, n)
    values = make_values(rng, n, spec["variables"], axes)
    raw = make_raw(rng, spec["slots"], n)
    if spec["passes"] > 1:
        labels = [(f"shift{p}",)]
    elif spec["slots"] == 1:
        labels = [("nominal",)]
    else:
        labels = [("nominal",)] + [(f"syst{k}",) for k in range(1, spec["slots"])]
    if spec["terms"]:
        columns = [rng.uniform(-1.0, 1.0, n) for _ in range(spec["terms"])]
        terms = [f"t{t}" for t in range(spec["terms"])]
        labels = [(label, term) for (label,) in labels for term in terms]
        weight = wzmod.eft_term_weights(raw, columns)
    else:
        weight = wzmod.standard_weights(raw)
    state = {"selection": selection, "channels": CHANNELS, "block_cache": {}}
    return values, labels, weight, state


def key_of(shape, var):
    return var + "_eft" if SHAPES[shape]["terms"] else var


def book(shape, axes, var):
    if SHAPES[shape]["terms"]:
        return wzmod._wh_eft(*axes[var])
    return wzmod._wh(*axes[var])


def as_target(target, dense):
    out_axes = []
    for ax, dax in zip(target.axes, dense.axes):
        if isinstance(ax, hist.axis.StrCategory):
            new = hist.axis.StrCategory(list(dax), name=ax.name, label=ax.label,
                                        growth=ax.traits.growth, overflow=ax.traits.overflow)
            new.__dict__.update(ax.__dict__)
            out_axes.append(new)
        else:
            out_axes.append(ax)
    output = hist.Hist(*out_axes, storage=hist.storage.Weight())
    output.__dict__.update(target.__dict__)
    output.view(flow=True)[...] = dense.view(flow=True)
    return output


def multicell_growth(histos, key, values, labels, weight, exclude, pass_state):
    target = histos[key]
    dense_axes = [ax for ax in target.axes if not isinstance(ax, hist.axis.StrCategory)]
    k = len(labels)
    grown = hist.Hist(hist.axis.StrCategory([], name="channel", growth=True), *dense_axes,
                      storage=hist.storage.MultiCell(2 * k))
    channel_list = []
    for channel, rows in wzmod.channel_block(exclude, pass_state):
        cells = np.empty((len(rows), 2 * k))
        for s in range(k):
            cells[:, s] = weight(s, rows)
        np.multiply(cells[:, :k], cells[:, :k], out=cells[:, k:])
        grown.fill(channel=channel, **{ax.name: values[ax.name][rows] for ax in dense_axes}, weight=cells)
        channel_list.append(channel)
    if not channel_list:
        return
    output, slot_index, _ = fill_engines.output_like(target, channel_list, labels)
    view = output.view(flow=True)
    cell_view = grown.view(flow=True)
    index = (slice(0, len(channel_list)),) + tuple(slot_index)
    view.value[index] = np.swapaxes(cell_view[:k], 0, 1)
    view.variance[index] = np.swapaxes(cell_view[k:], 0, 1)
    histos[key] = output


STRATEGIES = dict({"c1": wzmod.fill_histogram}, **fill_engines.ENGINES, chunked_persistent=None,
                  multicell_growth=multicell_growth)


def run_strategy(name, shape, axes, events, chunks):
    spec = SHAPES[shape]
    fill_seconds = 0.0
    outputs = []
    sinks = {}
    for chunk in range(chunks):
        for p in range(spec["passes"]):
            values, labels, weight, state = pass_inputs(shape, axes, events, chunk, p)
            histos = {key_of(shape, var): book(shape, axes, var) for var in spec["variables"]}
            t0 = time.perf_counter()
            for var in spec["variables"]:
                key = key_of(shape, var)
                if name == "chunked_persistent":
                    target = histos[key]
                    slot_names = [ax.name for ax in target.axes[1:] if isinstance(ax, hist.axis.StrCategory)]
                    sink = sinks.setdefault(key, ChunkedHist(*target.axes, storage=bh.storage.Weight()))
                    for channel, rows in wzmod.channel_block((var,), state):
                        gathered = {var: values[var][rows]}
                        for k, label in enumerate(labels):
                            sink.fill(channel=channel, **dict(zip(slot_names, label)), **gathered,
                                      weight=weight(k, rows))
                else:
                    STRATEGIES[name](histos, key, {var: values[var]}, labels, weight, (var,), state)
            fill_seconds += time.perf_counter() - t0
            if name != "chunked_persistent":
                outputs.append({DATASET: histos})
    t0 = time.perf_counter()
    if name == "chunked_persistent":
        targets = {key_of(shape, var): book(shape, axes, var) for var in spec["variables"]}
        final = {key: as_target(targets[key], sink.to_hist()) for key, sink in sinks.items()}
        for h in final.values():
            h._variance_known = chunks * spec["passes"] == 1
        native = sinks
    else:
        final = wzmod.merge_histogram_sets(outputs)[DATASET]
        native = final
    accumulate_seconds = time.perf_counter() - t0
    return dict(fill=fill_seconds, accumulate=accumulate_seconds, final=final, native=native, outputs=outputs)


def gzip_size(obj):
    return len(gzip.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 1))


def exactness(reference, final, order_tolerant):
    res = bp.Result()
    for key in sorted(set(reference) | set(final)):
        if key not in final or key not in reference:
            res.fail(1, key, "missing on one side")
            continue
        bp.compare_hist(reference[key], final[key], key, {}, res, order_tolerant=order_tolerant)
    return res


def accumulation_costs(outputs):
    keys = list(outputs[0][DATASET])
    costs = {}
    t0 = time.perf_counter()
    try:
        for key in keys:
            total = outputs[0][DATASET][key]
            for out in outputs[1:]:
                total = total + out[DATASET][key]
        costs["hist_add"] = time.perf_counter() - t0
    except ValueError as err:
        costs["hist_add"] = f"n/a ({err})"
    natives = [{key: ChunkedHist.from_hist(h) for key, h in out[DATASET].items()} for out in outputs]
    t0 = time.perf_counter()
    for key in keys:
        total = natives[0][key]
        for out in natives[1:]:
            total += out[key]
    costs["chunkedhist_iadd"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    wzmod.merge_histogram_sets(outputs)
    costs["merge_histogram_sets"] = time.perf_counter() - t0
    return costs


def rss_of(name, shape, args):
    cmd = [sys.executable, os.path.abspath(__file__), "--shapes", shape, "--events", str(args.events),
           "--chunks", str(args.chunks), "--only", name, "--rss"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def measure_rss(args, axes):
    run_strategy(args.only, args.shapes[0], axes, args.events, args.chunks)
    footprint = bp.footprint_mb(os.getpid())
    print(json.dumps({"maxrss_mb": bp.maxrss_mb(), "footprint_mb": None if footprint is None else footprint[0],
                      "lifetime_max_footprint_mb": None if footprint is None else footprint[1]}))


def bench_shape(shape, args, axes):
    names = [args.only] if args.only else list(STRATEGIES)
    results = {}
    reference = None
    for name in names:
        best = None
        for _ in range(args.repeat):
            run = run_strategy(name, shape, axes, args.events, args.chunks)
            if best is None or run["fill"] < best["fill"]:
                best = run
        if name == "c1":
            reference = best
        row = {
            "fill_seconds": best["fill"],
            "accumulate_seconds": best["accumulate"],
            "bytes": bp.histogram_bytes({DATASET: best["final"]}),
            "gzip_native": gzip_size(best["native"]),
            "gzip_hist": gzip_size(best["final"]),
        }
        if reference is not None:
            res = exactness(reference["final"], best["final"], name == "chunked_persistent")
            row["exact"] = res.passed
            row["failures"] = [" ".join(f) for f in res.failures[:5]]
        if args.memory:
            row.update(rss_of(name, shape, args))
        results[name] = row
    if reference is not None:
        results["accumulation"] = accumulation_costs(reference["outputs"])
    if "multicell" in results and "chunked_persistent" in results:
        mc = run_strategy("multicell", shape, axes, args.events, args.chunks)["final"]
        ch = run_strategy("chunked_persistent", shape, axes, args.events, args.chunks)["final"]
        results["multicell_vs_chunkedhist"] = exactness(mc, ch, True).passed
    return results


def print_shape(shape, results):
    print(f"\n== {shape} ==")
    print(f"{'strategy':20s} {'fill[s]':>9s} {'accum[s]':>9s} {'bytes':>12s} {'gz native':>10s} {'gz hist':>10s} "
          f"{'maxrss[MB]':>10s} {'life fp[MB]':>11s}  exact")
    for name, row in results.items():
        if not isinstance(row, dict) or "fill_seconds" not in row:
            continue
        print(f"{name:20s} {row['fill_seconds']:9.3f} {row['accumulate_seconds']:9.3f} {row['bytes']:12d} "
              f"{row['gzip_native']:10d} {row['gzip_hist']:10d} {bp.fmt(row.get('maxrss_mb')):>10s} "
              f"{bp.fmt(row.get('lifetime_max_footprint_mb')):>11s}  {row.get('exact', '-')}")
        for failure in row.get("failures", []):
            print(f"    FAIL {failure}")
    if "accumulation" in results:
        print("accumulation of the C1 outputs:", results["accumulation"])
    if "multicell_vs_chunkedhist" in results:
        print("MultiCell against ChunkedHist:", "PASS" if results["multicell_vs_chunkedhist"] else "FAIL")


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default="nominal,shift,data,eft")
    parser.add_argument("--events", type=int, default=100000)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--only", choices=sorted(STRATEGIES))
    parser.add_argument("--rss", action="store_true", help="run --only once and print its memory as JSON")
    parser.add_argument("--no-memory", dest="memory", action="store_false",
                        help="skip the fresh-subprocess memory measurement")
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args(argv)
    args.shapes = args.shapes.split(",")
    unknown = set(args.shapes) - set(SHAPES)
    if unknown:
        parser.error(f"unknown shapes {sorted(unknown)}")
    axes = booked_axes()
    if args.rss:
        if not args.only or len(args.shapes) != 1:
            parser.error("--rss needs --only and exactly one shape")
        measure_rss(args, axes)
        return 0
    results = {}
    for shape in args.shapes:
        results[shape] = bench_shape(shape, args, axes)
        print_shape(shape, results[shape])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1, default=str)
    failed = [f"{shape}/{name}" for shape, rows in results.items() for name, row in rows.items()
              if isinstance(row, dict) and row.get("exact") is False and name != "multicell_growth"]
    if failed:
        print("exactness failures:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
