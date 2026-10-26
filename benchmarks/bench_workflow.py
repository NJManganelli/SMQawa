import argparse
import glob
import gzip
import importlib.util
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

import bench_processor

DESCRIPTION = """End-to-end workflow driver and benchmark for wzinclusive_processor productions.

Subcommands:
  run      one workflow (reference, c1 or c4) of one configuration: setup, jobs on --slots single-core
           job processes, then the merge (hist-merger.py --legacy) or the C4 collect, with process
           sampling; with --scenario, one of the C4 dedup, completeness, poisoning and truncation
           scenarios, whose expected outcomes and exit codes are checked
  job      (internal) one harness job: the brewer's processor and coffea Runner on a WorkItem list
  compare  compare two outputs in the pickle layout ({dataset: {"hist", "sumw", ...}}), with the
           comparison rule of bench_processor.py and its sumw modes
  report   report.md and plots over the run directories of `run`

The tree under test is --tree (its src/ goes first on the jobs' PYTHONPATH); every job asserts that
qawa is imported from below it. Jobs are started with one thread each; the driver and the merge use
four. The C4 server is benchmarks/loopback_histserv.py on 127.0.0.1, stopped at the end of the run.
On macOS run the driver under `caffeinate -i -s -m --` for the final benchmark. The footprint and
per-process disk columns, the pressure level and the pmset probes are macOS only.
"""

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.abspath(__file__)
STORE = "/Volumes/WDMac/SMQawaTestFiles/store"
ERA = "2024"
VERSION = "v15"
CAPS = {name: "4" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS")}
JOB_CAPS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
SAMPLES = {
    "wz": {"file": STORE + "/mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/"
                           "150X_mcRun3_2024_realistic_v2-v2/100000/test.root",
           "das": "/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/"
                  "NANOAODSIM",
           "kind": "signal", "events": 1049760, "run_period": ""},
    "dy": {"file": STORE + "/mc/RunIII2024Summer24NanoAODv15/DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-"
                           "pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v1/90000/test.root",
           "das": "/DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-"
                  "150X_mcRun3_2024_realistic_v2-v1/NANOAODSIM",
           "kind": "mc", "events": 1292013, "run_period": ""},
    "data": {"file": STORE + "/data/Run2024C/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root",
             "das": "/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD", "kind": "data", "events": 1345797, "run_period": "C"},
}
CONFIGS = {
    "full": {"jobs": {"wz": [11, 10], "dy": [9, 9, 8], "data": [14, 13]}},
    "s-m1": {"jobs": {"wz": [1, 1, 1, 1]}},
    "s-e1": {"jobs": {"wz": [1, 1]}, "eft_operators": 5},
    "s-d1": {"jobs": {"data": [1, 1, 1]}},
    "s-m6": {"jobs": {"wz": [1, 1]}, "pol": True},
    "s-d4": {"jobs": {"data": [1, 1]}, "pol": True},
    "s-m4": {"jobs": {"wz": [1, 1]}, "split_by_charge": True},
}
SCENARIOS = ("clean", "duplicate-job", "kill-rerun", "kill-midpass-rerun", "kill-norerun", "never-started",
             "stale-pickle", "poison", "unavailable", "lost-response", "poison-unrecorded", "truncated")
KILL_EXIT_CODE = 137
KILL_PASSES = 10
MIDPASS_SENDS = 3
FAIL_SENDS = 5
SAMPLE_INTERVAL = 0.5
SYSTEM_INTERVAL = 5.0
POWER_INTERVAL = 60.0
ABORT_PRESSURE_LEVEL = 4
PRESSURED_LEVEL = 2
PRESSURED_SWAPOUT_MB = 512.0
PAGE_BYTES = 16384


def sample_of(name):
    sample = dict(SAMPLES[name])
    sample["dataset"] = sample["das"].split("/")[1]
    sample["is_data"] = sample["kind"] == "data"
    sample["is_signal"] = "WZto3LNu" in sample["dataset"] or "WZTo3LNu" in sample["dataset"]
    return sample


def sample_name_of(das):
    path = os.path.join(WORKTREE, "brewer-htcondor-inclusive.py")
    spec = importlib.util.spec_from_file_location("brewer_htcondor_inclusive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_name_of(das)[1]


def fileset_of(sample):
    metadata = {"era": ERA, "is_data": sample["is_data"], "is_signal": sample["is_signal"],
                "is_flashsim": False}
    return {sample["dataset"]: {"files": {sample["file"]: "Events"}, "metadata": metadata}}


def preprocess(sample, chunksize, maxchunks=None):
    from coffea import processor
    from coffea.nanoevents import NanoAODSchema
    runner = processor.Runner(executor=processor.IterativeExecutor(), schema=NanoAODSchema, chunksize=chunksize,
                              maxchunks=maxchunks, format="root")
    return list(runner.preprocess(fileset_of(sample), treename="Events"))


def genweight_sum(items):
    import uproot
    total = 0.0
    for item in items:
        with uproot.open(item.filename) as f:
            weights = f["Events"]["genWeight"].array(entry_start=item.entrystart, entry_stop=item.entrystop,
                                                     library="np")
        total += float(np.sum(weights, dtype=np.float64))
    return total


def read_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path, content):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def write_json(path, content):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(content, f, indent=1, default=str)
    os.replace(tmp, path)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def directory_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------- job

class JobCounter:
    def __init__(self, path, fail_after_passes=None, fail_after_sends=None):
        self.path = path
        self.fail_after_passes = fail_after_passes
        self.fail_after_sends = fail_after_sends
        self.passes = 0
        self.target_sent = 0
        self.pass_counts = None
        self.totals = {"target": {}, "accounting": {}}
        self.calls = {"target": 0, "ledger": 0, "sumw": 0, "lumi": 0}
        self.targets = set()

    def line(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def result(self, role, result):
        for counts in (self.totals[role], self.pass_counts[role] if self.pass_counts else None):
            if counts is not None:
                counts[result] = counts.get(result, 0) + 1
        if role == "target" and result == "sent":
            self.target_sent += 1
            if self.fail_after_sends is not None and self.target_sent >= self.fail_after_sends:
                self.line({"event": "killed", "after_sends": self.target_sent, "totals": self.totals})
                os._exit(KILL_EXIT_CODE)

    def pass_start(self, chunk, pass_name):
        self.pass_counts = {"target": {}, "accounting": {}}
        self.line({"event": "pass_start", "chunk": chunk, "pass": pass_name})

    def pass_end(self, chunk, pass_name):
        self.passes += 1
        self.line({"event": "pass_end", "chunk": chunk, "pass": pass_name, "counts": self.pass_counts})
        self.pass_counts = None
        if self.fail_after_passes is not None and self.passes >= self.fail_after_passes:
            self.line({"event": "killed", "after_passes": self.passes, "totals": self.totals})
            os._exit(KILL_EXIT_CODE)


def install_c4_hooks(ho, histserv, counter, registry, key, args):
    import grpc

    class InjectedRpcError(grpc.RpcError):
        def __init__(self, code, text):
            super().__init__(text)
            self.status = code
            self.text = text

        def code(self):
            return self.status

        def details(self):
            return self.text
    entry = registry["samples"][key]
    counter.targets = set(entry["hists"].values())
    send = ho.send

    def counted_send(registry, key, hist_id, fills, unique_id, where):
        result = send(registry, key, hist_id, fills, unique_id, where)
        counter.result("target" if hist_id in counter.targets else "accounting", result)
        return result
    ho.send = counted_send

    pass_context = ho.pass_context

    def counted_pass_context(registry, key, metadata, shift_name, eft):
        ctx = pass_context(registry, key, metadata, shift_name, eft)
        counter.pass_start(ctx["chunk"], ctx["pass"])
        return ctx
    ho.pass_context = counted_pass_context

    end_pass = ho.end_pass

    def counted_end_pass(pass_state):
        result = end_pass(pass_state)
        counter.pass_end(pass_state["histserv"]["chunk"], pass_state["histserv"]["pass"])
        return result
    ho.end_pass = counted_end_pass

    fill_many = histserv.RemoteHist.fill_many
    state = {"failed": False}
    code = getattr(grpc.StatusCode, args.fail_code)

    def injected_fill_many(self, fills, **kwargs):
        role = "target" if self.hist_id in counter.targets else (
            "ledger" if self.hist_id == entry["ledger"] else "lumi" if self.hist_id == entry.get("lumi") else "sumw")
        counter.calls[role] += 1
        if role == "ledger" and state["failed"] and args.fail_ledger_after_poison:
            raise InjectedRpcError(grpc.StatusCode.UNAVAILABLE, "injected ledger UNAVAILABLE after the poison")
        if role == "target" and args.fail_rpc_after_sends is not None and not state["failed"] \
                and counter.calls["target"] == args.fail_rpc_after_sends + 1:
            state["failed"] = True
            fills = list(fills)
            if args.fail_partial:
                fill_many(self, fills[:max(1, len(fills) // 2)], timeout=kwargs.get("timeout"))
            elif args.fail_after_call:
                fill_many(self, fills, **kwargs)
            raise InjectedRpcError(code, f"injected {args.fail_code} at target send {counter.calls['target']}")
        return fill_many(self, fills, **kwargs)
    histserv.RemoteHist.fill_many = injected_fill_many


def install_timers(wzmod, workflow, ho=None):
    from coffea.analysis_tools import Weights
    seconds = {"fill": 0.0, "weight": 0.0}
    weight = Weights.weight

    def timed_weight(self, modifier=None):
        t0 = time.perf_counter()
        result = weight(self, modifier)
        seconds["weight"] += time.perf_counter() - t0
        return result
    Weights.weight = timed_weight

    if workflow == "c4" or hasattr(wzmod, "fill_histogram"):
        owner, name = (ho, "send_histogram") if workflow == "c4" else (wzmod, "fill_histogram")
        fill = getattr(owner, name)

        def timed_fill(*fill_args):
            t0 = time.perf_counter()
            result = fill(*fill_args)
            seconds["fill"] += time.perf_counter() - t0
            return result
        setattr(owner, name, timed_fill)
    return seconds


def cmd_job(args):
    t_start = time.perf_counter()
    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)
    qawa_file = bench_processor.import_qawa_checked()
    import qawa.process.wztau2lnu_inclusive as wzmod
    from coffea import processor
    from coffea.nanoevents import NanoAODSchema, BaseSchema
    cfg = CONFIGS[args.config]
    sample = sample_of(args.sample)
    with open(args.items, "rb") as f:
        items = pickle.load(f)
    with open(args.all_items, "rb") as f:
        all_items = pickle.load(f)
    record = {"job_num": args.job_num, "workflow": args.workflow, "config": args.config, "sample": args.sample,
              "qawa_file": qawa_file, "items": [(i.entrystart, i.entrystop) for i in items],
              "events": sum(i.entrystop - i.entrystart for i in items), "flags": sys.argv[2:]}
    ho = histserv = registry = key = counter = None
    if args.workflow == "c4":
        import histserv
        from qawa import histserv_output as ho
        registry = ho.read_registry(args.registry)
        registry_chunks = registry["max_chunks"]
        wanted = args.max_chunks if args.max_chunks is not None else -1
        max_chunks_ok = wanted <= 0 if registry_chunks is None else wanted == registry_chunks
        if not max_chunks_ok:
            print(f"--max-chunks {args.max_chunks} does not match the registry's max_chunks {registry_chunks}")
            return 2
        os.environ[ho.POISON_FILE_ENV] = os.path.abspath(ho.POISON_FILE_NAME)
        reason = ho.poison_reason(None)
        if reason:
            print("C4 poison marker from an earlier run in this directory: " + reason)
            return ho.POISON_EXIT_CODE
        key = ho.resolve_key(registry, args.das_name, pol=bool(cfg.get("pol")),
                             split_by_charge=bool(cfg.get("split_by_charge")), split_FV=False, dd=None)
        counter = JobCounter(os.path.abspath(f"job_{args.job_num}.jsonl"), args.fail_after_passes, args.fail_after_sends)
        install_c4_hooks(ho, histserv, counter, registry, key, args)
    elif any(v is not None for v in (args.fail_after_passes, args.fail_after_sends, args.fail_rpc_after_sends)):
        print("failure injection needs --workflow c4")
        return 2
    if args.eft_operators:
        bench_processor.install_synthetic_eft(wzmod, args.eft_operators, args.eft_nan_modulus)
    inst = None
    seconds = {"fill": 0.0, "weight": 0.0}
    if not args.no_instrument:
        inst = bench_processor.Instrument()
        inst.install(wzmod)
        seconds = install_timers(wzmod, args.workflow, ho)
    kwargs = dict(era=ERA, version=VERSION, run_period=sample["run_period"], pol=bool(cfg.get("pol")),
                  split_by_charge=bool(cfg.get("split_by_charge")), split_FV=False, ewk_process_name=None)
    if args.workflow == "c4":
        kwargs.update(histserv_registry=registry, histserv_sample=key)
    proc = wzmod.wzinclusive_processor(**kwargs)
    record["construct"] = time.perf_counter() - t_start
    runner = processor.Runner(executor=processor.IterativeExecutor(), schema=NanoAODSchema, chunksize=args.chunksize,
                              format="root", savemetrics=True)
    sumw = 0.0
    t0 = time.perf_counter()
    if args.workflow == "c4":
        try:
            ho.check_poison(registry, key)
            file_of = {sample["file"]: sample["file"]}
            runs = None
            if not sample["is_data"]:
                import uproot
                with uproot.open(sample["file"]) as f:
                    sums = f["Runs"].arrays(["genEventSumw", "genEventCount"], library="np")
                runs = {sample["file"]: (float(np.sum(sums["genEventSumw"])), int(np.sum(sums["genEventCount"])))}
            truncated = None
            if registry["max_chunks"] is not None:
                untruncated = preprocess(sample, args.chunksize)
                files = ho.file_records(untruncated, file_of, runs)
                truncated = ho.truncation(untruncated, all_items, file_of)
            else:
                files = ho.file_records(all_items, file_of, runs)
            expected = {**{f: [] for f in files}, **ho.expected_chunks(all_items, file_of)}
            ho.begin_job(registry, key, expected, files, truncated)
            record["sumw_time"] = time.perf_counter() - t0
            np.random.seed(bench_processor.RANDOM_SEED)
            t1 = time.perf_counter()
            out, metrics = runner(items, processor_instance=proc)
            record["runner"] = time.perf_counter() - t1
        except Exception as err:
            reason = ho.poison_reason(err)
            if reason is None:
                raise
            print("C4 poison: " + reason)
            record.update(exit=ho.POISON_EXIT_CODE, poison=reason, counts=counter.totals, calls=counter.calls)
            write_json(f"job_{args.job_num}.json", record)
            return ho.POISON_EXIT_CODE
    else:
        if any(item.entrystart == 0 for item in items):
            from qawa.process.coffea_sumw import coffea_sumw
            fileset = fileset_of(sample)
            fileset[sample["dataset"]]["files"] = {sample["file"]: "Runs"}
            sumw_runner = processor.Runner(executor=processor.IterativeExecutor(), schema=BaseSchema, format="root",
                                           savemetrics=True)
            sumw = sumw_runner(fileset, processor_instance=coffea_sumw())[0][sample["dataset"]]
        record["sumw_time"] = time.perf_counter() - t0
        np.random.seed(bench_processor.RANDOM_SEED)
        t1 = time.perf_counter()
        out, metrics = runner(items, processor_instance=proc)
        record["runner"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    if args.workflow == "c4":
        ho.write_done_file(args.job_num, registry, key, items)
        record.update(counts=counter.totals, calls=counter.calls, returned=sorted(out),
                      returned_empty=all(not v for v in out.values()))
    else:
        record_of = getattr(wzmod, "dataset_record", lambda ds, content, sumw: {"hist": content, "sumw": sumw})
        write_pickle(f"histogram_{args.job_num}.pkl.gz", {ds: record_of(ds, content, float(sumw)) for ds, content in out.items()})
        record["output_bytes"] = os.path.getsize(f"histogram_{args.job_num}.pkl.gz")
    record["write"] = time.perf_counter() - t2
    if inst is not None:
        record.update(chunks=inst.chunks, process=sum(c["wall"] for c in inst.chunks),
                      passes_wall=sum(p["wall"] for p in inst.passes), passes=len(inst.passes),
                      merge=sum(c["merge"] for c in inst.chunks), expected=inst.expected())
        record["runner_overhead"] = record["runner"] - record["process"]
    fill_seconds = getattr(wzmod, "FILL_SECONDS", None)
    record["fill"] = fill_seconds[0] if fill_seconds is not None else seconds["fill"]
    record["weight"] = seconds["weight"]
    record["sumw"] = float(sumw)
    record["maxrss_mb"] = bench_processor.maxrss_mb()
    info = bench_processor.rusage_info(os.getpid())
    record["lifetime_max_footprint_mb"] = info["lifetime_max_phys_footprint"] / 2**20 if info else None
    record["wall"] = time.perf_counter() - t_start
    record["exit"] = 0
    write_json(f"job_{args.job_num}.json", record)
    return 0


# ---------------------------------------------------------------- sampling

class Sampler(threading.Thread):
    def __init__(self, path, root, on_abort):
        super().__init__(daemon=True)
        self.path = path
        self.root = root
        self.on_abort = on_abort
        self.roles = {os.getpid(): "driver"}
        self.stop_event = threading.Event()
        self.stages = []
        self.aborted = False
        self.max_pressure = None
        self.swapouts = []
        self.cpu_seconds = 0.0

    def stage(self, name):
        self.stages.append((name, time.time()))
        self.write({"t": time.time(), "type": "stage", "stage": name})

    def write(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def role_of(self, proc):
        try:
            chain = [proc] + proc.parents()
        except Exception:
            return None
        for depth, p in enumerate(chain):
            if p.pid in self.roles:
                return self.roles[p.pid] if depth == 0 else self.roles[p.pid] + "-child"
        return None

    def processes(self):
        import psutil
        me = psutil.Process()
        rows = []
        for p in [me] + me.children(recursive=True):
            try:
                cpu = p.cpu_times()
                row = {"pid": p.pid, "role": self.role_of(p), "rss": p.memory_info().rss, "cpu": cpu.user + cpu.system}
            except psutil.Error:
                continue
            info = bench_processor.rusage_info(p.pid)
            if info is not None:
                row.update(footprint=info["phys_footprint"], lifetime_max=info["lifetime_max_phys_footprint"],
                           read=info["diskio_bytesread"], written=info["diskio_byteswritten"])
            rows.append(row)
        return rows

    def system(self):
        import psutil
        record = {"t": time.time(), "type": "system", "percpu": psutil.cpu_percent(percpu=True),
                  "load": os.getloadavg(), "dir": directory_bytes(self.root),
                  "jobs_dir": directory_bytes(os.path.join(self.root, "jobs")),
                  "final_dir": directory_bytes(os.path.join(self.root, "final"))}
        if sys.platform == "darwin":
            level = bench_processor.sh("sysctl -n kern.memorystatus_vm_pressure_level").strip()
            record["pressure_level"] = int(level) if level.isdigit() else None
            for line in bench_processor.sh("vm_stat").splitlines():
                if line.startswith("Swapouts:"):
                    record["swapouts"] = int(line.split(":")[1].strip().rstrip("."))
            if record["pressure_level"] is not None:
                self.max_pressure = max(self.max_pressure or 0, record["pressure_level"])
            if record.get("swapouts") is not None:
                self.swapouts.append(record["swapouts"])
        return record

    def run(self):
        import psutil
        me = psutil.Process()
        next_system = next_power = 0.0
        while not self.stop_event.is_set():
            now = time.time()
            cpu0 = time.process_time()
            self.write({"t": now, "type": "proc", "procs": self.processes()})
            if now >= next_system:
                record = self.system()
                self.write(record)
                next_system = now + SYSTEM_INTERVAL
                if record.get("pressure_level") is not None and record["pressure_level"] >= ABORT_PRESSURE_LEVEL:
                    self.aborted = True
                    self.write({"t": time.time(), "type": "abort", "pressure_level": record["pressure_level"]})
                    self.on_abort()
            if now >= next_power and sys.platform == "darwin":
                self.write({"t": now, "type": "power", "batt": bench_processor.sh("pmset -g batt"),
                            "therm": bench_processor.sh("pmset -g therm")})
                next_power = now + POWER_INTERVAL
            self.cpu_seconds += time.process_time() - cpu0
            self.stop_event.wait(SAMPLE_INTERVAL)
        cpu = me.cpu_times()
        self.write({"t": time.time(), "type": "driver", "sampler_cpu": self.cpu_seconds, "driver_cpu": cpu.user + cpu.system})

    def finish(self):
        self.stop_event.set()
        self.join(timeout=60)
        swapout_mb = (self.swapouts[-1] - self.swapouts[0]) * PAGE_BYTES / 2**20 if len(self.swapouts) > 1 else 0.0
        return {"aborted": self.aborted, "max_pressure": self.max_pressure, "swapout_rise_mb": swapout_mb,
                "pressured": (self.max_pressure or 0) >= PRESSURED_LEVEL or swapout_mb > PRESSURED_SWAPOUT_MB,
                "sampler_cpu": self.cpu_seconds}


# ---------------------------------------------------------------- driver

class Workflow:
    def __init__(self, args):
        self.args = args
        self.cfg = CONFIGS[args.config]
        self.tree = os.path.realpath(args.tree)
        self.out = os.path.abspath(args.out)
        self.tag = args.tag
        self.max_chunks = args.max_chunks
        self.eft_operators = args.eft_operators if args.eft_synthetic else self.cfg.get("eft_operators", 0)
        self.checks = []
        self.records = []
        self.children = {}
        self.server = None
        self.address = None
        self.registries = {}
        self.sampler = None

    def env(self, caps):
        return {**os.environ, **caps, "PYTHONPATH": os.path.join(self.tree, "src") + os.pathsep + os.path.dirname(BENCH)}

    def check(self, name, ok, detail=""):
        self.checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else " :: " + str(detail)), flush=True)

    def kind_of(self, name):
        return "eft" if self.eft_operators else SAMPLES[name]["kind"]

    def setup(self):
        os.makedirs(os.path.join(self.out, "items"))
        os.makedirs(os.path.join(self.out, "jobs"))
        os.makedirs(os.path.join(self.out, "work"))
        os.makedirs(os.path.join(self.out, "manifests"))
        self.jobs = []
        self.all_items = {}
        self.untruncated = {}
        for name, split in self.cfg["jobs"].items():
            sample = sample_of(name)
            if self.args.config != "full" and self.max_chunks is None:
                self.max_chunks = self.args.items if self.args.items is not None else sum(split)
            items = preprocess(sample, self.args.chunksize, self.max_chunks)
            if self.args.config == "full":
                if len(items) != sum(split) or items[-1].entrystop != sample["events"]:
                    raise SystemExit(f"{sample['file']}: {len(items)} WorkItems over {items[-1].entrystop} entries, "
                                     f"expected {sum(split)} over {sample['events']}")
            else:
                n = self.args.items if self.args.items is not None else sum(split)
                items = items[:n]
                split = [1] * len(items)
            self.all_items[name] = os.path.join(self.out, "items", f"all_{name}.pkl")
            with open(self.all_items[name], "wb") as f:
                pickle.dump(items, f)
            start = 0
            for count in split:
                job_items = items[start:start + count]
                start += count
                path = os.path.join(self.out, "items", f"job_{len(self.jobs)}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(job_items, f)
                self.jobs.append({"num": len(self.jobs), "sample": name, "items": path, "workitems": job_items,
                                  "events": sum(i.entrystop - i.entrystart for i in job_items)})
            with open(os.path.join(self.out, "manifests", f"{name}.txt"), "w") as f:
                f.write(sample["file"] + "\n")
        if self.args.workflow == "c4":
            self.start_server()

    def warm_inputs(self):
        timings = {}
        for name in self.cfg["jobs"]:
            t0 = time.perf_counter()
            subprocess.run(["cat", SAMPLES[name]["file"]], stdout=subprocess.DEVNULL, check=True)
            timings[name] = time.perf_counter() - t0
        return timings

    def start_server(self):
        ready = os.path.join(self.out, "server_ready.json")
        log = open(os.path.join(self.out, "server.log"), "w")
        self.server = subprocess.Popen([sys.executable, os.path.join(WORKTREE, "benchmarks", "loopback_histserv.py"),
                                        "--host", "127.0.0.1", "--port", "0", "--executor-threads", "1",
                                        "--ready-file", ready], env=self.env(JOB_CAPS), stdout=log, stderr=log)
        self.sampler.roles[self.server.pid] = "server"
        deadline = time.monotonic() + 60
        while not os.path.exists(ready):
            if time.monotonic() > deadline or self.server.poll() is not None:
                raise SystemExit("loopback histserv server did not start")
            time.sleep(0.1)
        with open(ready) as f:
            self.address = json.load(f)["address"]

    def stop_server(self):
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            self.server.wait(60)

    def register(self, name):
        path = os.path.join(self.out, f"registry_{name}.json")
        tag = f"{self.tag}-{name}"
        cmd = [sys.executable, "-m", "qawa.histserv_output", "register", "--address", self.address, "--tag", tag,
               "--era", ERA, "--version", VERSION, "--output", path]
        for s in self.cfg["jobs"]:
            cmd += ["--sample", f"{SAMPLES[s]['das']}={self.kind_of(s)}"]
        if self.cfg.get("pol"):
            cmd.append("--pol")
        if self.cfg.get("split_by_charge"):
            cmd.append("--split-by-charge")
        if self.max_chunks is not None:
            cmd += ["--max-chunks", str(self.max_chunks)]
        proc = subprocess.run(cmd, env=self.env(CAPS), capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"register failed ({proc.returncode}): {proc.stdout}{proc.stderr}")
        with open(path) as f:
            self.registries[name] = json.load(f)
        return path

    def job_command(self, job, workdir, workflow, registry=None, max_chunks=None, flags=(), job_num=None):
        cmd = [sys.executable, BENCH, "job", "--workflow", workflow, "--config", self.args.config,
               "--sample", job["sample"], "--items", job["items"], "--all-items", self.all_items[job["sample"]],
               "--out", workdir, "--job-num", str(job["num"] if job_num is None else job_num),
               "--chunksize", str(self.args.chunksize), "--eft-operators", str(self.eft_operators)]
        if registry is not None:
            cmd += ["--registry", registry, "--das-name", SAMPLES[job["sample"]]["das"]]
        if max_chunks is not None:
            cmd += ["--max-chunks", str(max_chunks)]
        if self.args.no_instrument:
            cmd.append("--no-instrument")
        return cmd + list(flags)

    def run_jobs(self, runs):
        pending = sorted(runs, key=lambda r: -r["job"]["events"])
        running = []
        results = {}
        while pending or running:
            while pending and len(running) < self.args.slots and not self.sampler.aborted:
                run = pending.pop(0)
                os.makedirs(run["workdir"], exist_ok=True)
                log = open(os.path.join(run["workdir"], "job.log"), "w")
                proc = subprocess.Popen(run["cmd"], env=self.env(JOB_CAPS), stdout=log, stderr=subprocess.STDOUT,
                                        cwd=run["workdir"])
                self.children[proc.pid] = proc
                self.sampler.roles[proc.pid] = f"job_{run['job']['num']}"
                running.append((proc, run, log, time.time()))
            if self.sampler.aborted:
                pending = []
            for entry in list(running):
                proc, run, log, started = entry
                if proc.poll() is not None:
                    log.close()
                    running.remove(entry)
                    self.children.pop(proc.pid, None)
                    record_path = glob.glob(os.path.join(run["workdir"], "job_*.json"))
                    record = None
                    if record_path:
                        with open(record_path[0]) as f:
                            record = json.load(f)
                    lines = read_jsonl(os.path.join(run["workdir"], f"job_{run.get('job_num', run['job']['num'])}.jsonl"))
                    results[run["name"]] = {"rc": proc.returncode, "record": record, "lines": lines,
                                            "wall": time.time() - started, "workdir": run["workdir"]}
                    self.records.append({"name": run["name"], "rc": proc.returncode, "wall": time.time() - started,
                                         "record": record})
                    print(f"job {run['name']}: exit {proc.returncode} after {time.time() - started:.1f} s", flush=True)
            time.sleep(0.2)
        return results

    def job_run(self, job, name, workflow=None, registry=None, max_chunks="registry", flags=(), job_num=None):
        workdir = os.path.join(self.out, "work", name)
        if max_chunks == "registry":
            max_chunks = self.max_chunks if registry is not None else None
        return {"name": name, "job": job, "workdir": workdir, "job_num": job["num"] if job_num is None else job_num,
                "cmd": self.job_command(job, workdir, workflow or self.args.workflow, registry, max_chunks, flags,
                                        job_num)}

    def place_pickles(self, results):
        for result in results.values():
            for path in glob.glob(os.path.join(result["workdir"], "histogram_*.pkl.gz")):
                job = result["record"]["sample"]
                target = os.path.join(self.out, "jobs", f"jobs_{self.tag}_{ERA}_{sample_name_of(SAMPLES[job]['das'])}")
                os.makedirs(target, exist_ok=True)
                shutil.move(path, os.path.join(target, os.path.basename(path)))

    def merge_tool(self):
        if self.args.merge_tool:
            return self.args.merge_tool
        own = os.path.join(self.tree, "hist-merger.py")
        return own if os.path.exists(own) else os.path.join(WORKTREE, "hist-merger.py")

    def merge(self):
        final = os.path.join(self.out, "final")
        os.makedirs(final, exist_ok=True)
        cmd = [sys.executable, self.merge_tool(), "--legacy", "-t", self.tag, "--era", ERA, "--dir", "./",
               "--outdir", final]
        proc = subprocess.Popen(cmd, cwd=os.path.join(self.out, "jobs"), env=self.env(CAPS),
                                stdout=open(os.path.join(self.out, "merge.log"), "w"), stderr=subprocess.STDOUT)
        self.sampler.roles[proc.pid] = "merge"
        rc = proc.wait()
        return rc, os.path.join(final, f"merged-histogram-{self.tag}-{ERA}-Inc_WZ_latest_rawdeeptau.pkl.gz")

    def collect(self, name, outdir, extra=()):
        registry = os.path.join(self.out, f"registry_{name}.json")
        cmd = [sys.executable, "-m", "qawa.histserv_output", "collect", "--registry", registry, "--outdir", outdir]
        reg = self.registries[name]
        for key, entry in reg["samples"].items():
            sample = [s for s in self.cfg["jobs"] if SAMPLES[s]["das"] == entry["das"]][0]
            cmd += ["--manifest", f"{key}={os.path.join(self.out, 'manifests', sample + '.txt')}"]
        proc = subprocess.Popen(cmd + list(extra), env=self.env(CAPS), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        self.sampler.roles[proc.pid] = "collect"
        text = proc.communicate()[0]
        with open(outdir.rstrip("/") + ".log", "w") as f:
            f.write(text)
        return proc.returncode, text, sorted(glob.glob(os.path.join(outdir, "*.pkl.gz")))

    def ledger(self, name):
        sys.path.insert(0, os.path.join(self.tree, "src"))
        from qawa import histserv_output as ho
        reg = self.registries[name]
        key = next(iter(reg["samples"]))
        return ho.entry_state(reg, key)["ledger"], key

    def kill_children(self):
        for proc in list(self.children.values()):
            if proc.poll() is None:
                proc.kill()


def counts_of(result, role="target"):
    record = result.get("record") or {}
    totals = record.get("counts")
    if totals is None:
        killed = [line for line in result["lines"] if line.get("event") == "killed"]
        totals = killed[-1]["totals"] if killed else {"target": {}, "accounting": {}}
    return totals.get(role, {})


def sent_in_first_passes(result, n):
    ends = [line for line in result["lines"] if line.get("event") == "pass_end"][:n]
    return sum(line["counts"]["target"].get("sent", 0) for line in ends)


def compare_outputs(ref_paths, opt_paths, identical=False, order_tolerant=False, sumw_mode="strict", rename_map=None,
                    changed=(), removed=(), restored=(), report=False, expected=None, lumi_record=False,
                    require_lumi_record=False):
    ref, opt = load_side(ref_paths), load_side(opt_paths)
    res = bench_processor.Result()
    for side, (records, problems) in (("R", ref), ("O", opt)):
        for problem in problems:
            res.fail(1, "-", f"{side}: {problem}")
    ref, opt = ref[0], opt[0]
    hist_map, axis_map = bench_processor.load_map(rename_map)
    rdump = {"output": {ds: rec["hist"] for ds, rec in ref.items()}, "expected": expected}
    odump = {"output": {ds: rec["hist"] for ds, rec in opt.items()}, "expected": expected}
    hist_res = bench_processor.compare_dumps(rdump, odump, hist_map, axis_map, changed, removed, restored,
                                             identical=identical, report=report, order_tolerant=order_tolerant)
    if expected is None:
        hist_res.failures = [f for f in hist_res.failures if f[0] != "4"]
        res.note("rule 4 (vacuity) not checked: no --expected job records given")
    res.failures += hist_res.failures
    res.notes += hist_res.notes
    res.stats += hist_res.stats
    if sumw_mode != "skip":
        sumw_res = bench_processor.compare_sumw({ds: rec["sumw"] for ds, rec in ref.items()},
                                                {ds: rec["sumw"] for ds, rec in opt.items()}, sumw_mode)
        res.failures += sumw_res.failures
    for ds in sorted(opt):
        bench_processor.compare_lumi(ds, ref.get(ds, {}).get(bench_processor.LUMI_KEY),
                                     opt[ds].get(bench_processor.LUMI_KEY), bench_processor.record_is_data(opt[ds]),
                                     res, lumi_record or require_lumi_record, require_lumi_record, records=True)
    for ds in sorted(set(ref) & set(opt)):
        bench_processor.compare_kind(ds, ref[ds].get(bench_processor.KIND_KEY), opt[ds].get(bench_processor.KIND_KEY), res)
    return res


def is_hist(value):
    return hasattr(value, "axes") and hasattr(value, "view")


def load_side(paths):
    records = {}
    problems = []
    for path in paths:
        for ds, rec in read_pickle(path).items():
            if ds in records:
                problems.append(f"dataset {ds} in several files")
            records[ds] = dict(rec, hist={k: v for k, v in rec["hist"].items() if is_hist(v)})
    return records, problems


def expected_of(record_paths):
    expected = {"nominal": set(), "shift": set(), "weight": set()}
    for path in record_paths:
        with open(path) as f:
            record = json.load(f)
        for cls, labels in (record.get("expected") or {}).items():
            expected[cls].update(labels)
    return {cls: sorted(labels) for cls, labels in expected.items()}


def hist_equal_2t(wf, name, ref_paths, opt_paths, sumw_mode="skip"):
    res = compare_outputs(ref_paths, opt_paths, order_tolerant=True, sumw_mode=sumw_mode)
    wf.check(name, res.passed, "; ".join(f"{k}: {r}" for _, k, r in res.failures[:5]))


def collected_sumw(paths):
    return {ds: rec["sumw"] for path in paths for ds, rec in read_pickle(path).items()}


def clean_production(wf, name):
    registry = wf.register(name)
    results = wf.run_jobs([wf.job_run(job, f"{name}_job{job['num']}", registry=registry) for job in wf.jobs])
    for run, result in results.items():
        wf.check(f"{run} exits 0", result["rc"] == 0, result["rc"])
    return registry, results


def scenario_clean(wf):
    if wf.args.workflow != "c4":
        results = wf.run_jobs([wf.job_run(job, f"job{job['num']}") for job in wf.jobs])
        for run, result in results.items():
            wf.check(f"{run} exits 0", result["rc"] == 0, result["rc"])
        wf.place_pickles(results)
        wf.sampler.stage("merge")
        rc, final = wf.merge()
        wf.check("merge exits 0", rc == 0 and os.path.exists(final), rc)
        return [final]
    registry, results = clean_production(wf, "main")
    for run, result in results.items():
        record = result["record"] or {}
        wf.check(f"{run} returns {{dataset: {{}}}} and writes its done-file", record.get("returned_empty")
                 and os.path.exists(os.path.join(result["workdir"], f"histserv_done_{record.get('job_num')}.json")))
    wf.sampler.stage("collect")
    rc, text, finals = wf.collect("main", os.path.join(wf.out, "final"))
    wf.check("collect exits 0", rc == 0, text[-2000:])
    check_genweight(wf, finals, [job["workitems"] for job in wf.jobs])
    return finals


def check_genweight(wf, finals, item_lists):
    if not finals:
        return
    sumw = collected_sumw(finals)
    for name in wf.cfg["jobs"]:
        sample = sample_of(name)
        if sample["is_data"]:
            continue
        items = [i for items in item_lists for i in items if i.filename == sample["file"]]
        harness = genweight_sum(items)
        got = sumw.get(sample["dataset"])
        ok = got is not None and abs(got - harness) <= bench_processor.ORDER_TOLERANT_RTOL * max(abs(got), abs(harness))
        wf.check(f"sumw of {sample['dataset']} equals the harness genWeight sum (order_tolerant)", ok,
                 f"collected {got!r}, harness {harness!r}")


def scenario_duplicate_job(wf):
    registry, results = clean_production(wf, "main")
    rc, text, first = wf.collect("main", os.path.join(wf.out, "final_clean"))
    wf.check("clean collect exits 0", rc == 0, text[-1000:])
    job1 = wf.jobs[1]
    dup = wf.run_jobs([wf.job_run(job1, "main_job1_dup", registry=registry, job_num=100 + job1["num"])])["main_job1_dup"]
    wf.check("duplicate job exits 0", dup["rc"] == 0, dup["rc"])
    target, accounting = counts_of(dup), counts_of(dup, "accounting")
    original = counts_of(results[f"main_job{job1['num']}"])
    wf.check("every target RPC of the duplicate is DUPLICATE, as many as the job's SENT",
             target.get("sent", 0) == 0 and target.get("duplicate", 0) == original.get("sent"), f"{target} vs {original}")
    wf.check("every accounting RPC of the duplicate is DUPLICATE", set(accounting) == {"duplicate"}, accounting)
    rc, text, second = wf.collect("main", os.path.join(wf.out, "final"))
    wf.check("collect after the duplicate exits 0", rc == 0, text[-1000:])
    res = compare_outputs(first, second, identical=True, sumw_mode="strict")
    wf.check("collect is identical to the clean collect, sumw included", res.passed, res.failures[:3])
    return second


def scenario_kill(wf, rerun, midpass=False):
    base_registry, base = clean_production(wf, "base")
    rc, text, baseline = wf.collect("base", os.path.join(wf.out, "final_base"))
    wf.check("baseline collect exits 0", rc == 0, text[-1000:])
    registry = wf.register("main")
    job0 = wf.jobs[0]
    if midpass:
        n = sent_in_first_passes(base[f"base_job{job0['num']}"], KILL_PASSES) + MIDPASS_SENDS
        flags = ["--fail-after-sends", str(n)]
    else:
        flags = ["--fail-after-passes", str(KILL_PASSES)]
    runs = [wf.job_run(job0, "main_job0_killed", registry=registry, flags=flags)]
    runs += [wf.job_run(job, f"main_job{job['num']}", registry=registry) for job in wf.jobs[1:]]
    results = wf.run_jobs(runs)
    killed = results["main_job0_killed"]
    wf.check("job 0 is killed (exit 137)", killed["rc"] == KILL_EXIT_CODE, killed["rc"])
    if midpass:
        starts = [line for line in killed["lines"] if line.get("event") == "pass_start"]
        ends = [line for line in killed["lines"] if line.get("event") == "pass_end"]
        wf.check(f"pass {KILL_PASSES + 1} started and did not end", len(starts) == KILL_PASSES + 1
                 and len(ends) == KILL_PASSES, f"{len(starts)} started, {len(ends)} ended")
    chunk = job_chunk(wf, job0)
    rc, text, _ = wf.collect("main", os.path.join(wf.out, "final_after_kill"))
    wf.check("collect after the kill exits 1", rc == 1, rc)
    if not rerun:
        wf.check("it names job 0's chunk under 'started, not done'", "started, not done" in text and chunk in text,
                 text[-1500:])
        wf.check("it writes nothing", not glob.glob(os.path.join(wf.out, "final_after_kill", "*.pkl.gz")))
        rc, text, finals = wf.collect("main", os.path.join(wf.out, "final"), ["--allow-incomplete"])
        wf.check("collect --allow-incomplete exits 3 with .INCOMPLETE files", rc == 3 and finals
                 and all(".INCOMPLETE" in f for f in finals), f"{rc} {finals}")
        return finals
    again = wf.run_jobs([wf.job_run(job0, "main_job0_rerun", registry=registry)])["main_job0_rerun"]
    wf.check("the re-run exits 0", again["rc"] == 0, again["rc"])
    if midpass:
        wanted = n
    else:
        wanted = counts_of(killed).get("sent", 0)
    wf.check("the re-run's DUPLICATE target count equals the killed attempt's SENT count",
             counts_of(again).get("duplicate", 0) == wanted, f"{counts_of(again)} vs {wanted}")
    rc, text, finals = wf.collect("main", os.path.join(wf.out, "final"))
    wf.check("collect after the re-run exits 0", rc == 0, text[-1000:])
    hist_equal_2t(wf, "collect equals the clean run (rule 2t)", baseline, finals, sumw_mode="order_tolerant")
    return finals


def job_chunk(wf, job):
    import uuid
    item = job["workitems"][0]
    return f"{uuid.UUID(bytes=item.fileuuid)}:{item.entrystart}:{item.entrystop}"


def scenario_never_started(wf):
    registry = wf.register("main")
    job0 = wf.jobs[0]
    result = wf.run_jobs([wf.job_run(job0, "main_job0", registry=registry)])["main_job0"]
    wf.check("job 0 exits 0", result["rc"] == 0, result["rc"])
    rc, text, _ = wf.collect("main", os.path.join(wf.out, "final_refused"))
    wf.check("collect exits 1 naming job 1's chunk as 'expected, not started'",
             rc == 1 and "expected, not started" in text and job_chunk(wf, wf.jobs[1]) in text, text[-1500:])
    rc, text, finals = wf.collect("main", os.path.join(wf.out, "final"), ["--allow-incomplete"])
    wf.check("collect --allow-incomplete exits 3 with .PARTIAL files",
             rc == 3 and finals and all(".PARTIAL" in f for f in finals), f"{rc} {finals}")
    c1 = wf.run_jobs([wf.job_run(job0, "c1_job0", workflow="c1")])["c1_job0"]
    wf.check("the C1 job over job 0's chunk exits 0", c1["rc"] == 0, c1["rc"])
    c1_pickle = os.path.join(c1["workdir"], f"histogram_{job0['num']}.pkl.gz")
    hist_equal_2t(wf, ".PARTIAL hists equal the C1 run over job 0's chunk (rule 2t)", [c1_pickle], finals)
    check_genweight(wf, finals, [job0["workitems"]])
    return finals


def scenario_stale_pickle(wf):
    registry, results = clean_production(wf, "main")
    rc, text, first = wf.collect("main", os.path.join(wf.out, "final_clean"))
    wf.check("clean collect exits 0", rc == 0, text[-1000:])
    sample = sample_of(wf.jobs[0]["sample"])
    record = read_pickle(first[0])[sample["dataset"]]
    name, h = next((k, v) for k, v in record["hist"].items() if is_hist(v) and v.sum(flow=True).value)
    jobs_dir = os.path.join(wf.out, "jobs", f"jobs_{wf.tag}_{ERA}_{sample_name_of(sample['das'])}")
    os.makedirs(jobs_dir, exist_ok=True)
    write_pickle(os.path.join(jobs_dir, "histogram_99.pkl.gz"), {sample["dataset"]: {"hist": {name: h}, "sumw": 1e9}})
    rc, text, second = wf.collect("main", os.path.join(wf.out, "final"))
    wf.check("collect with the stale pickle exits 0", rc == 0, text[-1000:])
    res = compare_outputs(first, second, identical=True, sumw_mode="strict")
    wf.check("collect is identical to the clean one: it reads no pickle", res.passed, res.failures[:3])
    return second


def poison_entries(wf, name):
    ledger, key = wf.ledger(name)
    return [k for k in ledger if k[2] == "poisoned"]


def marker_records(result):
    return read_jsonl(os.path.join(result["workdir"], "histserv_poisoned.jsonl"))


def scenario_poison(wf, recorded=True):
    registry = wf.register("main")
    job0 = wf.jobs[0]
    flags = ["--fail-rpc-after-sends", str(FAIL_SENDS), "--fail-code", "INTERNAL", "--fail-partial"]
    if not recorded:
        flags.append("--fail-ledger-after-poison")
    runs = [wf.job_run(job0, "main_job0_poisoned", registry=registry, flags=flags)]
    runs += [wf.job_run(job, f"main_job{job['num']}", registry=registry) for job in wf.jobs[1:]] if not recorded else []
    results = wf.run_jobs(runs)
    poisoned = results["main_job0_poisoned"]
    log = open(os.path.join(poisoned["workdir"], "job.log")).read()
    wf.check("job 0 exits with POISON_EXIT_CODE", poisoned["rc"] == 87, poisoned["rc"])
    records = marker_records(poisoned)
    states = [r["state"] for r in records]
    entries = poison_entries(wf, "main")
    chunk = job_chunk(wf, job0)
    if recorded:
        wf.check("the ledger holds one poisoned entry for that chunk", len(entries) == 1 and entries[0][1] == chunk,
                 entries)
        wf.check("the marker holds one pending and one recorded record with the injected details",
                 states == ["pending", "recorded"] and all("injected INTERNAL" in r["details"] for r in records), records)
        again = wf.run_jobs([wf.job_run(job0, "main_job0_rerun", registry=registry)])["main_job0_rerun"]
        rerun_log = open(os.path.join(again["workdir"], "job.log")).read()
        wf.check("the re-run exits with POISON_EXIT_CODE at check_poison, with zero target RPCs",
                 again["rc"] == 87 and "is poisoned at" in rerun_log
                 and (again["record"] or {}).get("calls", {}).get("target") == 0, f"{again['rc']} {again['record']}")
        rc, text, _ = wf.collect("main", os.path.join(wf.out, "final"))
        wf.check("collect exits 1 and names the poisoned chunk and pass", rc == 1 and "poisoned" in text and chunk in text
                 and entries and entries[0][3] in text, text[-1500:])
    else:
        wf.check("its message says 'poison NOT recorded'", "poison NOT recorded" in log)
        wf.check("the marker holds one pending and one 'not recorded' record with all attempts",
                 states == ["pending", "not recorded"] and records[-1].get("attempts") == 4, records)
        wf.check("the ledger has no poisoned entry", not entries, entries)
        rc, text, _ = wf.collect("main", os.path.join(wf.out, "final"))
        wf.check("collect exits 1 naming the chunk under 'started, not done'",
                 rc == 1 and "started, not done" in text and chunk in text, text[-1500:])
    return []


def scenario_no_poison(wf, lost_response):
    base_registry, base = clean_production(wf, "base")
    rc, text, baseline = wf.collect("base", os.path.join(wf.out, "final_base"))
    wf.check("baseline collect exits 0", rc == 0, text[-1000:])
    registry = wf.register("main")
    job0 = wf.jobs[0]
    flags = ["--fail-rpc-after-sends", str(FAIL_SENDS), "--fail-code",
             "DEADLINE_EXCEEDED" if lost_response else "UNAVAILABLE"]
    if lost_response:
        flags.append("--fail-after-call")
    runs = [wf.job_run(job0, "main_job0_failed", registry=registry, flags=flags)]
    runs += [wf.job_run(job, f"main_job{job['num']}", registry=registry) for job in wf.jobs[1:]]
    results = wf.run_jobs(runs)
    failed = results["main_job0_failed"]
    wanted = FAIL_SENDS + 1 if lost_response else FAIL_SENDS
    got = counts_of(failed)
    calls = (failed["record"] or {}).get("calls", {})
    wf.check("job 0 exits 0 after coffea's in-process chunk retry, not POISON_EXIT_CODE", failed["rc"] == 0, failed["rc"])
    wf.check(f"the retry shows exactly {wanted} DUPLICATE target results, the rest SENT, and one failed call",
             got.get("duplicate", 0) == wanted and set(got) <= {"sent", "duplicate", "empty"}
             and calls.get("target") == got.get("sent", 0) + got.get("duplicate", 0) + 1, f"{got} {calls}")
    wf.check("the ledger has no poisoned entry", not poison_entries(wf, "main"))
    rc, text, finals = wf.collect("main", os.path.join(wf.out, "final"))
    wf.check("collect exits 0", rc == 0, text[-1000:])
    hist_equal_2t(wf, "collect equals the clean run (rule 2t)", baseline, finals, sumw_mode="order_tolerant")
    return finals


def scenario_truncated(wf):
    registry, results = clean_production(wf, "main")
    rc, text, finals = wf.collect("main", os.path.join(wf.out, "final"))
    wf.check("collect exits 0 with only .TRUNCATED names", rc == 0 and finals and all(".TRUNCATED" in f for f in finals),
             f"{rc} {finals} {text[-1000:]}")
    with open(os.path.join(wf.out, "final", "collect.json")) as f:
        summary = json.load(f)
    wf.check("collect.json has truncated true and max_chunks 1", summary.get("truncated") is True
             and summary.get("max_chunks") == 1, {k: summary.get(k) for k in ("truncated", "max_chunks")})
    total = len(preprocess(sample_of(wf.jobs[0]["sample"]), wf.args.chunksize))
    production = json.dumps([rec.get("production") for path in finals for rec in read_pickle(path).values()])
    wf.check(f"the production record says kept 1 / total {total}", f"kept 1 / total {total}" in production
             or ("\"kept\": 1" in production and f"\"total\": {total}" in production), production[:500])
    job0 = wf.jobs[0]
    c1 = wf.run_jobs([wf.job_run(job0, "c1_job0", workflow="c1")])["c1_job0"]
    hist_equal_2t(wf, "hists equal a C1 run over that chunk (rule 2t)",
                  [os.path.join(c1["workdir"], f"histogram_{job0['num']}.pkl.gz")], finals)
    check_genweight(wf, finals, [job0["workitems"]])
    for name, max_chunks in (("max2", 2), ("nomax", None)):
        result = wf.run_jobs([wf.job_run(job0, f"main_job0_{name}", registry=registry, max_chunks=max_chunks)])
        result = result[f"main_job0_{name}"]
        log = open(os.path.join(result["workdir"], "job.log")).read()
        wf.check(f"a job with --max-chunks {max_chunks} exits non-zero before any RPC",
                 result["rc"] != 0 and "does not match the registry's max_chunks" in log
                 and not os.path.exists(os.path.join(result["workdir"], f"job_{job0['num']}.jsonl")), result["rc"])
    return finals


def cmd_run(args):
    if args.scenario != "clean" and args.workflow != "c4":
        raise SystemExit("scenarios need --workflow c4")
    if args.scenario == "truncated":
        args.max_chunks = 1
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} exists and is not empty")
    os.makedirs(args.out, exist_ok=True)
    wf = Workflow(args)
    wf.sampler = Sampler(os.path.join(wf.out, "samples.jsonl"), wf.out, wf.kill_children)
    machine_before = bench_processor.machine_state() if sys.platform == "darwin" else None
    quiet = None
    if args.quiet_wait > 0 and sys.platform == "darwin":
        state, reasons, waited = bench_processor.wait_until_quiet("", args.quiet_wait)
        quiet = {"reasons": reasons, "waited": waited}
    wf.sampler.start()
    t_start = time.time()
    wf.sampler.stage("setup")
    finals = []
    try:
        wf.setup()
        warm = wf.warm_inputs() if args.warm_inputs else None
        wf.sampler.stage("jobs")
        scenario = {
            "clean": scenario_clean, "duplicate-job": scenario_duplicate_job,
            "kill-rerun": lambda w: scenario_kill(w, rerun=True),
            "kill-midpass-rerun": lambda w: scenario_kill(w, rerun=True, midpass=True),
            "kill-norerun": lambda w: scenario_kill(w, rerun=False), "never-started": scenario_never_started,
            "stale-pickle": scenario_stale_pickle, "poison": scenario_poison,
            "unavailable": lambda w: scenario_no_poison(w, lost_response=False),
            "lost-response": lambda w: scenario_no_poison(w, lost_response=True),
            "poison-unrecorded": lambda w: scenario_poison(w, recorded=False), "truncated": scenario_truncated,
        }[args.scenario]
        finals = scenario(wf)
        t_done = time.time()
        wf.sampler.stage("teardown")
    finally:
        wf.stop_server()
        wf.kill_children()
    t_end = time.time()
    sampling = wf.sampler.finish()
    summary = {
        "workflow": args.workflow, "config": args.config, "scenario": args.scenario, "tree": wf.tree,
        "tag": wf.tag, "slots": args.slots, "chunksize": args.chunksize, "items": args.items,
        "max_chunks": wf.max_chunks, "eft_operators": wf.eft_operators, "merge_tool": wf.merge_tool(),
        "stages": wf.sampler.stages, "total_time": t_done - t_start, "teardown_end": t_end - t_start,
        "jobs": wf.records, "checks": wf.checks, "finals": finals, "warm_inputs": warm, "sampling": sampling,
        "machine_before": machine_before, "machine_after": bench_processor.machine_state() if sys.platform == "darwin"
        else None, "quiet": quiet, "labels": [label for label, on in (("pressured", sampling["pressured"]),
                                                                      ("aborted", sampling["aborted"]),
                                                                      ("not quiet", quiet and quiet["reasons"])) if on],
    }
    write_json(os.path.join(wf.out, "run.json"), summary)
    failed = [c for c in wf.checks if not c["ok"]]
    print(f"{args.scenario}: {len(wf.checks) - len(failed)} / {len(wf.checks)} checks PASS; "
          f"total {summary['total_time']:.1f} s; finals {finals}")
    return 1 if failed or sampling["aborted"] else 0


# ---------------------------------------------------------------- compare

def cmd_compare(args):
    res = compare_outputs([args.ref] + (args.ref_also or []), [args.opt] + (args.opt_also or []),
                          identical=args.identical, order_tolerant=args.order_tolerant, sumw_mode=args.sumw_mode,
                          rename_map=args.rename_map, changed=bench_processor.split_names(args.changed),
                          removed=bench_processor.split_names(args.removed),
                          restored=bench_processor.split_names(args.restored), report=True,
                          expected=expected_of(args.expected) if args.expected else None,
                          lumi_record=args.lumi_record, require_lumi_record=args.require_lumi_record)
    bench_processor.print_result(res)
    return 0 if res.passed else 1


# ---------------------------------------------------------------- report

def load_run(path):
    with open(os.path.join(path, "run.json")) as f:
        run = json.load(f)
    procs, system = [], []
    for record in read_jsonl(os.path.join(path, "samples.jsonl")):
        if record["type"] == "proc":
            procs.append(record)
        elif record["type"] == "system":
            system.append(record)
    return run, procs, system


def role_group(role):
    if role is None:
        return "other"
    return "job" if role.startswith("job_") else role.split("-")[0]


def run_rows(run, procs):
    stages = dict((name, t) for name, t in run["stages"])
    t0 = run["stages"][0][1]
    peak_total = max((sum(p.get("footprint", p["rss"]) for p in r["procs"]) for r in procs), default=0)
    peak_rss = max((sum(p["rss"] for p in r["procs"]) for r in procs), default=0)
    peaks = {}
    cpu = {}
    disk = {}
    for r in procs:
        for p in r["procs"]:
            group = role_group(p["role"])
            peaks[group] = max(peaks.get(group, 0), p.get("footprint", p["rss"]))
            cpu[p["pid"]] = (group, p["cpu"])
            disk[p["pid"]] = (group, p.get("read", 0), p.get("written", 0))
    cpu_by_role = {}
    for group, seconds in cpu.values():
        cpu_by_role[group] = cpu_by_role.get(group, 0.0) + seconds
    disk_by_role = {}
    for group, read, written in disk.values():
        old = disk_by_role.get(group, (0, 0))
        disk_by_role[group] = (old[0] + read, old[1] + written)
    jobs = [j["record"] for j in run["jobs"] if j.get("record")]
    sums = {name: sum(j.get(name) or 0.0 for j in jobs) for name in
            ("process", "fill", "weight", "merge", "runner_overhead", "wall", "sumw_time", "write")}
    stage_times = {}
    names = [name for name, _ in run["stages"]]
    for a, b in zip(names, names[1:] + [None]):
        stage_times[a] = (stages[b] if b else t0 + run["teardown_end"]) - stages[a]
    return {"peak_total_mb": peak_total / 2**20, "peak_rss_mb": peak_rss / 2**20,
            "peaks_mb": {k: v / 2**20 for k, v in peaks.items()}, "cpu_by_role": cpu_by_role,
            "disk_by_role": disk_by_role, "sums": sums, "stage_times": stage_times,
            "job_peaks_mb": [j.get("lifetime_max_footprint_mb") for j in jobs]}


def plot_run(name, run, procs, system, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = run["stages"][0][1]
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    series = {}
    for r in procs:
        for p in r["procs"]:
            series.setdefault(p["role"] or "other", []).append((r["t"] - t0, p.get("footprint", 0) / 2**30,
                                                                p["rss"] / 2**30, p["cpu"]))
    for role, points in sorted(series.items()):
        t = [x[0] for x in points]
        axes[0].plot(t, [x[1] for x in points], label=role, linewidth=0.8)
        axes[1].plot(t, [x[2] for x in points], label=role, linewidth=0.8)
        cpu = [100.0 * (b[3] - a[3]) / max(b[0] - a[0], 1e-9) for a, b in zip(points, points[1:])]
        axes[2].plot(t[1:], cpu, label=role, linewidth=0.8)
    axes[0].set_ylabel("footprint [GiB]")
    axes[1].set_ylabel("RSS [GiB]")
    axes[2].set_ylabel("CPU [%]")
    axes[3].plot([s["t"] - t0 for s in system], [s["dir"] / 2**30 for s in system], label="workflow directory")
    axes[3].set_ylabel("directory [GiB]")
    axes[3].set_xlabel("time since setup [s]")
    for ax in axes:
        for stage, t in run["stages"]:
            ax.axvline(t - t0, color="grey", linewidth=0.5)
    axes[0].legend(fontsize=6, ncol=4)
    fig.suptitle(name)
    fig.tight_layout()
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out, exist_ok=True)
    runs = []
    for path in args.runs:
        run, procs, system = load_run(path)
        name = f"{run['workflow']}-{run['config']}-{os.path.basename(os.path.normpath(path))}"
        runs.append((name, run, procs, system, run_rows(run, procs), plot_run(name, run, procs, system, args.out)))
    base = runs[0][1]["total_time"] if runs else None
    lines = ["# Workflow benchmark", "", "## Total time to completion", "",
             "| run | labels | total [s] | ratio to first |", "|---|---|---|---|"]
    for name, run, _, _, rows, _ in runs:
        lines.append(f"| {name} | {', '.join(run['labels']) or '-'} | {run['total_time']:.1f} | "
                     f"{run['total_time'] / base:.2f} |")
    lines += ["", "## Stages [s]", "", "| run | merge tool | " + " | ".join(("setup", "jobs", "merge", "collect",
                                                                            "teardown")) + " |",
              "|---|---|---|---|---|---|---|"]
    for name, run, _, _, rows, _ in runs:
        st = rows["stage_times"]
        lines.append(f"| {name} | {run['merge_tool'] if run['workflow'] != 'c4' else 'collect'} | "
                     + " | ".join(f"{st[s]:.1f}" if s in st else "-" for s in ("setup", "jobs", "merge", "collect",
                                                                              "teardown")) + " |")
    lines += ["", "## Job phase CPU-seconds", "",
              "| run | process | fill (c4: fill+send) | weight prep | fill + weight prep | in-process merge | "
              "non-fill | Runner overhead | sumw | write |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, run, _, _, rows, _ in runs:
        s = rows["sums"]
        nonfill = s["process"] - s["fill"] - s["merge"]
        lines.append(f"| {name} | {s['process']:.1f} | {s['fill']:.1f} | {s['weight']:.1f} | "
                     f"{s['fill'] + s['weight']:.1f} | {s['merge']:.1f} | {nonfill:.1f} | {s['runner_overhead']:.1f} | "
                     f"{s['sumw_time']:.1f} | {s['write']:.1f} |")
    lines += ["", "## Memory [MB]", "", "| run | peak total footprint | peak total RSS | peak per role | job peaks |",
              "|---|---|---|---|---|"]
    for name, run, _, _, rows, _ in runs:
        peaks = ", ".join(f"{k} {v:.0f}" for k, v in sorted(rows["peaks_mb"].items()))
        jobs = ", ".join(f"{v:.0f}" for v in rows["job_peaks_mb"] if v is not None)
        lines.append(f"| {name} | {rows['peak_total_mb']:.0f} | {rows['peak_rss_mb']:.0f} | {peaks} | {jobs} |")
    lines += ["", "## CPU-seconds and disk by role", "", "| run | role | CPU [s] | read [MB] | written [MB] |",
              "|---|---|---|---|---|"]
    for name, run, _, _, rows, _ in runs:
        for role in sorted(rows["cpu_by_role"]):
            read, written = rows["disk_by_role"].get(role, (0, 0))
            lines.append(f"| {name} | {role} | {rows['cpu_by_role'][role]:.1f} | {read / 2**20:.0f} | "
                         f"{written / 2**20:.0f} |")
    lines += ["", "## Checks and machine state", ""]
    for name, run, _, _, rows, _ in runs:
        failed = [c for c in run["checks"] if not c["ok"]]
        lines.append(f"- {name}: {len(run['checks']) - len(failed)} / {len(run['checks'])} checks PASS; "
                     f"sampling {run['sampling']}; machine before {run['machine_before']}; after {run['machine_after']}")
    if sys.platform != "darwin":
        lines += ["", "Footprint, per-process disk, pressure and pmset columns: n/a (not macOS)."]
    lines += ["", "## Plots", ""] + [f"![{name}]({os.path.basename(png)})" for name, _, _, _, _, png in runs]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, run, procs, _, _, _ in runs:
        t0 = run["stages"][0][1]
        t = [r["t"] - t0 for r in procs]
        axes[0].plot(t, [sum(p.get("footprint", p["rss"]) for p in r["procs"]) / 2**30 for r in procs], label=name)
        cpu = [sum(p["cpu"] for p in r["procs"]) for r in procs]
        axes[1].plot(t[1:], [100.0 * (b - a) / max(tb - ta, 1e-9) for a, b, ta, tb in zip(cpu, cpu[1:], t, t[1:])],
                     label=name)
    axes[0].set_ylabel("total footprint [GiB]")
    axes[1].set_ylabel("total CPU [%]")
    axes[1].set_xlabel("time since setup [s]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "comparison.png"), dpi=100)
    plt.close(fig)
    lines.append("![comparison](comparison.png)")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("report written to", os.path.join(args.out, "report.md"))
    return 0


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="one workflow of one configuration, or one C4 scenario")
    p.add_argument("--workflow", required=True, choices=("reference", "c1", "c4"))
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--tree", required=True, help="tree under test; its src/ goes first on the jobs' PYTHONPATH")
    p.add_argument("--out", required=True, help="run directory (must be new or empty)")
    p.add_argument("--slots", type=int, required=True, help="concurrent single-core job processes")
    p.add_argument("--chunksize", type=int, default=50000)
    p.add_argument("--items", type=int, help="small configurations: the first N WorkItems, one per job")
    p.add_argument("--max-chunks", type=int, help="truncated production: coffea maxchunks and the registry max_chunks")
    p.add_argument("--eft-synthetic", action="store_true")
    p.add_argument("--eft-operators", type=int, default=0)
    p.add_argument("--scenario", default="clean", choices=SCENARIOS)
    p.add_argument("--merge-tool", help="hist-merger.py for reference and c1 (default: --tree's, else this one's)")
    p.add_argument("--no-instrument", action="store_true", help="jobs install no wrapper at all")
    p.add_argument("--tag", default="bench", help="production tag")
    p.add_argument("--quiet-wait", type=float, default=0.0, help="seconds to wait for a quiet machine first (macOS)")
    p.add_argument("--warm-inputs", action="store_true", help="read the input files once before the jobs")
    p = sub.add_parser("job", help="(internal) one harness job")
    p.add_argument("--workflow", required=True, choices=("reference", "c1", "c4"))
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--sample", required=True, choices=sorted(SAMPLES))
    p.add_argument("--items", required=True, help="pickle of this job's WorkItems")
    p.add_argument("--all-items", required=True, help="pickle of the configuration's WorkItems of this file")
    p.add_argument("--out", required=True, help="job directory: pickle or done-file, records, poison marker")
    p.add_argument("--job-num", type=int, required=True)
    p.add_argument("--chunksize", type=int, default=50000)
    p.add_argument("--registry")
    p.add_argument("--das-name")
    p.add_argument("--max-chunks", type=int)
    p.add_argument("--eft-operators", type=int, default=0)
    p.add_argument("--eft-nan-modulus", type=int, default=50)
    p.add_argument("--no-instrument", action="store_true")
    p.add_argument("--fail-after-passes", type=int, help="os._exit(137) after P passes")
    p.add_argument("--fail-after-sends", type=int, help="os._exit(137) after N SENT target RPCs")
    p.add_argument("--fail-rpc-after-sends", type=int, help="the (N+1)-th target fill_many fails")
    p.add_argument("--fail-code", default="INTERNAL", help="grpc.StatusCode name of the injected failure")
    p.add_argument("--fail-partial", action="store_true", help="send the first half with no unique id, then fail")
    p.add_argument("--fail-after-call", action="store_true", help="fail after the real call completed")
    p.add_argument("--fail-ledger-after-poison", action="store_true",
                   help="every later ledger fill_many raises an injected UNAVAILABLE")
    p = sub.add_parser("compare", help="compare two outputs in the pickle layout (exit 1 on failure)")
    p.add_argument("ref")
    p.add_argument("opt")
    p.add_argument("--ref-also", action="append", help="further R files with disjoint datasets")
    p.add_argument("--opt-also", action="append", help="further O files with disjoint datasets")
    p.add_argument("--rename-map")
    p.add_argument("--changed", action="append")
    p.add_argument("--removed", action="append")
    p.add_argument("--restored", action="append")
    p.add_argument("--identical", action="store_true")
    p.add_argument("--order-tolerant", action="store_true")
    p.add_argument("--sumw-mode", default="strict", choices=bench_processor.SUMW_MODES + ("skip",))
    p.add_argument("--expected", nargs="+", help="job records (job_<n>.json) whose expected labels rule 4 checks")
    p.add_argument("--lumi-record", action="store_true", help="print data datasets without a lumi record")
    p.add_argument("--require-lumi-record", action="store_true",
                   help="every data dataset must carry a lumi record on both sides, and the two must be equal")
    p = sub.add_parser("report", help="report.md and plots over run directories")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "job":
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    return {"run": cmd_run, "job": cmd_job, "compare": cmd_compare, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
