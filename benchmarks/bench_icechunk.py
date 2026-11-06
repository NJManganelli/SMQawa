import argparse
import copy
import gzip
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import boost_histogram as bh
import hist
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_processor as bp  # noqa: E402

DESCRIPTION = """Local icechunk storage benchmark for the WZ histogram outputs (information only).

icechunk, icechunkedhist and zarr are imported lazily; they are not dependencies of qawa. The
concurrency caps are set before any store is opened, and the script exits unless
TOKIO_WORKER_THREADS is set to at most 4.

Input: --input <pkl.gz> (a {dataset: {"hist": {key: Hist}}} or {dataset: {key: Hist}} output,
repeated under new prefixes to fill a tier) or a synthetic production-like set (27 channels x 35 systematics on the booked dense axes of
wzinclusive_processor(era="2024", version="v15")).

Every store is a local one, in a fresh directory under --workdir (a temporary directory by default,
removed unless --keep). Per tier and layout (native: ChunkedHist.from_hist, one zarr array per
category combination; packed: category axes as Integer index axes, one zarr array per histogram):
conversion, one-transaction save against 10 individual commits, load one and load all with the
conversion back to hist.Hist, an exact round trip under the comparison rule of bench_processor.py
(required), a partial update with a load of the pre-update commit, on-disk bytes and file count,
the codecs, and the pickle+gzip size for comparison. A thread sampler records the CPU and the
thread count, and the larger tiers are skipped when a tier's mean CPU exceeds 420 %.
"""

MAX_TOKIO_THREADS = 4
CPU_ABORT_PERCENT = 420.0
INDIVIDUAL_COMMITS = 10
CHANNELS = 27
SYSTEMATICS = 35


def require_icechunk():
    try:
        import icechunk
        import icechunkedhist
        import zarr
    except ImportError as err:
        raise SystemExit(f"bench_icechunk.py needs icechunk, icechunkedhist and zarr, which are not dependencies "
                         f"of qawa and are not installed in {sys.prefix} ({err}). Install them there to run it.")
    tokio = os.environ.get("TOKIO_WORKER_THREADS")
    if tokio is None or not tokio.isdigit() or int(tokio) > MAX_TOKIO_THREADS:
        raise SystemExit(f"set TOKIO_WORKER_THREADS to at most {MAX_TOKIO_THREADS} (found {tokio!r})")
    zarr.config.set({"async.concurrency": 4, "threading.max_workers": 4, "codec_pipeline.max_workers": 4})
    return icechunk, icechunkedhist, zarr


class Sampler:
    def __init__(self, interval=0.5):
        import psutil
        self.process = psutil.Process()
        self.interval = interval
        self.cpu = []
        self.threads = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        self.process.cpu_percent(None)
        while not self.stop.wait(self.interval):
            self.cpu.append(self.process.cpu_percent(None))
            self.threads.append(self.process.num_threads())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        return {"mean_cpu_percent": float(np.mean(self.cpu)) if self.cpu else 0.0,
                "max_cpu_percent": float(np.max(self.cpu)) if self.cpu else 0.0,
                "max_threads": max(self.threads) if self.threads else self.process.num_threads()}


def ps_threads():
    out = subprocess.run(["ps", "-M", str(os.getpid())], capture_output=True, text=True).stdout
    return max(len(out.splitlines()) - 1, 0)


def open_repository(icechunk, icechunkedhist, path):
    config = icechunk.RepositoryConfig(
        max_concurrent_requests=4, max_concurrent_decodes=4, get_partial_values_concurrency=4,
        storage=icechunk.StorageSettings(
            concurrency=icechunk.StorageConcurrencySettings(max_concurrent_requests_for_object=4)),
    )
    repository = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(path), config=config)
    return repository, icechunkedhist.Store(repository)


def input_hists(path):
    with gzip.open(path, "rb") as f:
        output = pickle.load(f)
    return [(f"{ds}.{key}", h) for ds, content in output.items() if isinstance(content, dict)
            for key, h in content.get("hist", content).items() if isinstance(h, hist.Hist)]


def synthetic_hists(n, seed=1):
    import qawa.process.wztau2lnu_inclusive as wzmod
    histos = wzmod.wzinclusive_processor(era="2024", version="v15").build_histos()
    variables = wzmod.STANDARD_FILL_VARIABLES
    rng = np.random.default_rng(seed)
    hists = []
    for i in range(n):
        var = variables[i % len(variables)]
        template = histos[var]
        h = hist.Hist(
            hist.axis.StrCategory([f"ch{c}" for c in range(CHANNELS)], name="channel", growth=True),
            hist.axis.StrCategory(["nominal"] + [f"syst{s}" for s in range(1, SYSTEMATICS)], name="systematic",
                                  growth=True),
            *list(template.axes)[2:], storage=hist.storage.Weight(),
        )
        view = h.view(flow=True)
        view.value = rng.uniform(0.0, 10.0, view.shape)
        view.variance = rng.uniform(0.0, 1.0, view.shape)
        hists.append((f"synthetic{i // len(variables)}.{var}", h))
    return hists


def tier_hists(source, n):
    if len(source) >= n:
        return source[:n]
    return [(f"copy{i // len(source)}.{source[i % len(source)][0]}", source[i % len(source)][1]) for i in range(n)]


def packed(h):
    from hist.chunked import ChunkedHist
    axes = []
    for ax in h.axes:
        if isinstance(ax, hist.axis.StrCategory):
            axes.append(hist.axis.Integer(0, max(len(ax), 1), name=ax.name, underflow=False, overflow=False,
                                          label=json.dumps(list(ax))))
        else:
            axes.append(ax)
    out = hist.Hist(*axes, storage=hist.storage.Weight())
    view = np.asarray(out.view(flow=True))
    source = np.asarray(h.view(flow=True))
    view[tuple(slice(0, n) for n in source.shape)] = source
    return ChunkedHist.from_hist(out)


def unpacked(chunked, template):
    labels = [json.loads(ax.label) for ax in chunked.axes if isinstance(ax, bh.axis.Integer)
              and ax.label.startswith("[")]
    axes = []
    for ax in template.axes:
        if isinstance(ax, hist.axis.StrCategory):
            new = hist.axis.StrCategory(labels.pop(0), name=ax.name, label=ax.label, growth=ax.traits.growth,
                                        overflow=ax.traits.overflow)
            new.__dict__.update(ax.__dict__)
            axes.append(new)
        else:
            axes.append(ax)
    out = hist.Hist(*axes, storage=hist.storage.Weight())
    out.__dict__.update(template.__dict__)
    view = np.asarray(out.view(flow=True))
    if chunked.chunks and view.size:
        (_, dense), = list(chunked.items())
        view[...] = dense[tuple(slice(0, n) for n in view.shape)]
    return out


def native(h):
    from hist.chunked import ChunkedHist
    return ChunkedHist.from_hist(h)


def unnative(chunked, template):
    if not chunked.chunks:
        return copy.deepcopy(template)
    dense = chunked.to_hist()
    axes = []
    for ax, dax in zip(template.axes, dense.axes):
        if isinstance(ax, hist.axis.StrCategory):
            new = hist.axis.StrCategory(list(dax), name=ax.name, label=ax.label, growth=ax.traits.growth,
                                        overflow=ax.traits.overflow)
            new.__dict__.update(ax.__dict__)
            axes.append(new)
        else:
            axes.append(ax)
    out = hist.Hist(*axes, storage=hist.storage.Weight())
    out.__dict__.update(template.__dict__)
    out.view(flow=True)[...] = dense.view(flow=True)
    return out


LAYOUTS = {"native": (native, unnative), "packed": (packed, unpacked)}


def disk_usage(path):
    size = files = 0
    for root, _, names in os.walk(path):
        for name in names:
            size += os.path.getsize(os.path.join(root, name))
            files += 1
    return size, files


def codecs_of(zarr, repository):
    group = zarr.open_group(repository.readonly_session(branch="main").store, mode="r")
    for _, member in group.members(max_depth=None):
        if isinstance(member, zarr.Array):
            return [type(c).__name__ for c in member.metadata.codecs]
    return []


def round_trip(hists, restored):
    res = bp.Result()
    for name, h in hists:
        bp.compare_hist(h, restored[name], name, {}, res, identical=True)
    return res


def bench_layout(modules, hists, layout, workdir):
    icechunk, icechunkedhist, zarr = modules
    to_stored, from_stored = LAYOUTS[layout]
    row = {}
    t0 = time.perf_counter()
    stored = [(name, to_stored(h)) for name, h in hists]
    row["from_hist_seconds"] = time.perf_counter() - t0
    row["arrays"] = sum(max(len(c.chunks), 1) for _, c in stored)

    path = os.path.join(workdir, f"{layout}_tx")
    repository, store = open_repository(icechunk, icechunkedhist, path)
    t0 = time.perf_counter()
    with store.transaction(message="all") as tx:
        for name, c in stored:
            tx.save(c, name=name)
    row["one_transaction_seconds"] = time.perf_counter() - t0
    first_commit = tx.result.commit_id

    individual = stored[:INDIVIDUAL_COMMITS]
    _, single_store = open_repository(icechunk, icechunkedhist, os.path.join(workdir, f"{layout}_single"))
    t0 = time.perf_counter()
    for name, c in individual:
        single_store.save(c, name=name, message=name)
    row["individual_commits"] = len(individual)
    row["individual_commits_seconds"] = time.perf_counter() - t0
    del stored

    templates = dict(hists)
    name0 = hists[0][0]
    t0 = time.perf_counter()
    from_stored(store.load(name0), templates[name0])
    row["load_one_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    restored = {name: from_stored(store.load(name), templates[name]) for name, _ in hists}
    row["load_all_seconds"] = time.perf_counter() - t0
    res = round_trip(hists, restored)
    row["round_trip_exact"] = res.passed
    row["round_trip_failures"] = [" ".join(f) for f in res.failures[:5]]
    del restored

    updated = hists[:max(1, len(hists) // 10)]
    changed = {}
    for name, h in updated:
        new = copy.deepcopy(h)
        view = new.view(flow=True)
        view.value = view.value * 2.0
        changed[name] = new
    t0 = time.perf_counter()
    with store.transaction(message="update") as tx:
        for name, h in changed.items():
            tx.save(to_stored(h), name=name)
    row["partial_update_hists"] = len(changed)
    row["partial_update_seconds"] = time.perf_counter() - t0
    head = {name: from_stored(store.load(name), templates[name]) for name in changed}
    before = {name: from_stored(store.load(name, commit=first_commit), templates[name]) for name in changed}
    row["update_head_exact"] = round_trip(list(changed.items()), head).passed
    row["update_previous_commit_exact"] = round_trip(updated, before).passed

    row["disk_bytes"], row["disk_files"] = disk_usage(path)
    row["codecs"] = codecs_of(zarr, repository)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="pkl.gz output ({dataset: {'hist': {key: Hist}, ...}} or {dataset: {key: Hist}}); "
                                        "default a synthetic production-like set")
    parser.add_argument("--tiers", default="10,75,375")
    parser.add_argument("--budget-seconds", type=float, default=1800.0,
                        help="skip the remaining tiers once this much time is spent")
    parser.add_argument("--layouts", default="native,packed")
    parser.add_argument("--workdir", help="parent of the fresh store directory (default: a temporary directory)")
    parser.add_argument("--keep", action="store_true", help="keep the store directory")
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args(argv)
    modules = require_icechunk()
    tiers = sorted(int(t) for t in args.tiers.split(","))
    layouts = args.layouts.split(",")
    if set(layouts) - set(LAYOUTS):
        parser.error(f"unknown layouts {sorted(set(layouts) - set(LAYOUTS))}")
    source = input_hists(args.input) if args.input else synthetic_hists(tiers[-1])
    if not source:
        raise SystemExit(f"no histograms in {args.input}")
    workdir = tempfile.mkdtemp(prefix="bench_icechunk_", dir=args.workdir)
    print("store directory:", workdir, flush=True)
    results = {"input": args.input or "synthetic", "ps_threads_at_start": ps_threads(), "tiers": {}}
    started = time.perf_counter()
    try:
        for n in tiers:
            if time.perf_counter() - started > args.budget_seconds:
                results["tiers"][n] = "skipped: budget"
                continue
            hists = tier_hists(source, n)
            tier_dir = os.path.join(workdir, f"tier{n}")
            os.makedirs(tier_dir)
            rows = {"pickle_gzip_bytes": len(gzip.compress(pickle.dumps(dict(hists), protocol=pickle.HIGHEST_PROTOCOL),
                                                           1))}
            with Sampler() as sampler:
                for layout in layouts:
                    rows[layout] = bench_layout(modules, hists, layout, tier_dir)
            rows.update(sampler.summary())
            results["tiers"][n] = rows
            print(f"\n== tier {n} hists ({rows['mean_cpu_percent']:.0f} % mean CPU, "
                  f"{rows['max_threads']} threads, pickle+gzip {rows['pickle_gzip_bytes']} B) ==")
            for layout in layouts:
                print(layout, json.dumps(rows[layout]))
            if rows["mean_cpu_percent"] > CPU_ABORT_PERCENT:
                print(f"mean CPU above {CPU_ABORT_PERCENT:.0f} %: the larger tiers are skipped")
                for m in tiers[tiers.index(n) + 1:]:
                    results["tiers"][m] = "skipped: cpu"
                break
        results["ps_threads_at_end"] = ps_threads()
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1, default=str)
    failed = [f"{n}/{layout}" for n, rows in results["tiers"].items() if isinstance(rows, dict)
              for layout in layouts if not (rows[layout]["round_trip_exact"] and rows[layout]["update_head_exact"]
                                            and rows[layout]["update_previous_commit_exact"])]
    if failed:
        print("round-trip failures:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
