import gzip
import importlib.util
import json
import os
import pickle
import subprocess
import sys

import hist
import numpy as np
import pytest

import qawa.datadriven as datadriven
from qawa import lumi_record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "NUMEXPR_MAX_THREADS": "4"}
ENV = {**os.environ, **CAPS, "PYTHONPATH": os.path.join(ROOT, "src")}
LOCAL_2018 = "/Users/nmangane/vz-main/SMQawa/DoubleMuon_UL2018_NanoAODv9.root"
STORE = "/Volumes/WDMac/SMQawaTestFiles/store/data/"
D1_FILE = STORE + "Run2024C/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root"
D2_FILE = STORE + "Run2024I/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root"
RUN = 379000
CSV_HEADER = ("#Data tag : 24v1 , Norm tag: None\n"
              "#run:fill,ls,time,beamstatus,E(GeV),delivered(/pb),recorded(/pb),avgpu,source\n")


def needs(*paths):
    return pytest.mark.skipif(not all(os.path.exists(p) for p in paths), reason="local test files not available")


def write_pickle(path, content):
    with gzip.open(path, "wb") as f:
        pickle.dump(content, f)


def data_hist():
    h = hist.Hist(hist.axis.StrCategory(["chA"], name="channel"), hist.axis.StrCategory(["nominal"], name="systematic"),
                  hist.axis.Regular(4, 0, 4, name="x"), hist.storage.Weight())
    h.fill(channel="chA", systematic="nominal", x=[0.5, 1.5, 1.5])
    return h


def test_file_free_functions():
    runs = np.array([5, 5, 5, 6, 6, 7])
    lumis = np.array([1, 1, 2, 3, 3, 9])
    mask = np.array([True, True, True, True, False, False])
    assert lumi_record.lumi_pairs(runs, lumis, mask) == {(5, 1), (5, 2), (6, 3)}
    array = lumi_record.as_array({(6, 3), (5, 2), (5, 1)})
    assert array.dtype == np.uint32 and array.tolist() == [[5, 1], [5, 2], [6, 3]]
    empty = lumi_record.as_array(set())
    assert empty.shape == (0, 2) and empty.dtype == np.uint32
    merged = lumi_record.union(array, lumi_record.as_array({(5, 2), (8, 1)}), empty)
    assert merged.tolist() == [[5, 1], [5, 2], [6, 3], [8, 1]] and merged.dtype == np.uint32
    assert lumi_record.union(empty, empty).shape == (0, 2)
    pairs = lumi_record.as_array({(319450, lumi) for lumi in (1, 2, 3, 5)})
    assert lumi_record.to_cms_json(pairs) == {"319450": [[1, 3], [5, 5]]}
    from coffea.processor import accumulate
    out = accumulate([{"d": {lumi_record.LUMI_KEY: {(1, 1), (1, 2)}}}, {"d": {lumi_record.LUMI_KEY: {(1, 2), (2, 1)}}}])
    assert out["d"][lumi_record.LUMI_KEY] == {(1, 1), (1, 2), (2, 1)}


def independent_pairs(path, start, stop, golden, masked=True):
    import uproot
    from coffea.lumi_tools import LumiMask
    with uproot.open(path) as f:
        arrays = f["Events"].arrays(["run", "luminosityBlock"], entry_start=start, entry_stop=stop, library="np")
    mask = LumiMask(golden)(arrays["run"], arrays["luminosityBlock"]) if masked else np.ones(len(arrays["run"]), bool)
    return {(int(r), int(lumi)) for r, lumi in zip(arrays["run"][mask], arrays["luminosityBlock"][mask])}


def data_events(path, dataset, start=None, stop=None):
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    return NanoEventsFactory.from_root({path: "Events"}, schemaclass=NanoAODSchema, mode="virtual", entry_start=start,
                                       entry_stop=stop, metadata={"dataset": dataset, "is_data": True,
                                                                  "is_signal": False, "is_flashsim": False}).events()


@pytest.fixture(scope="module")
def local_output():
    from qawa.process.wztau2lnu_inclusive import wzinclusive_processor
    proc = wzinclusive_processor(era="2018", run_period="C", version="v9")
    return proc, proc.process(data_events(LOCAL_2018, "DoubleMuon"))["DoubleMuon"]


@needs(LOCAL_2018)
def test_processor_record_local_2018(local_output):
    proc, out = local_output
    record = out[lumi_record.LUMI_KEY]
    assert isinstance(record, set) and record
    assert record == independent_pairs(LOCAL_2018, None, None, proc._lumimask_path)


def nominal_only(monkeypatch):
    import qawa.process.wztau2lnu_inclusive as wzmod
    monkeypatch.setattr(wzmod.wzinclusive_processor, "process_shift",
                        lambda self, event, shift_name='': {event.metadata["dataset"]: {}})
    return wzmod.wzinclusive_processor


@needs(D1_FILE, D2_FILE)
def test_processor_record_2024(monkeypatch):
    from coffea import processor
    from coffea.nanoevents import NanoAODSchema
    proc = nominal_only(monkeypatch)(era="2024", run_period="C", version="v15")
    golden = proc._lumimask_path
    record = proc.process(data_events(D1_FILE, "Muon0", 0, 100000))["Muon0"][lumi_record.LUMI_KEY]
    assert record == independent_pairs(D1_FILE, 0, 100000, golden)
    proc_i = nominal_only(monkeypatch)(era="2024", run_period="I", version="v15")
    record = proc_i.process(data_events(D2_FILE, "Muon0", 280000, 300000))["Muon0"][lumi_record.LUMI_KEY]
    assert record == independent_pairs(D2_FILE, 280000, 300000, golden)
    assert record != independent_pairs(D2_FILE, 280000, 300000, golden, masked=False)
    fileset = {"Muon0": {"files": {D1_FILE: "Events"}, "metadata": {"is_data": True, "is_signal": False,
                                                                   "is_flashsim": False}}}
    runner = processor.Runner(processor.IterativeExecutor(), chunksize=50000, maxchunks=3, schema=NanoAODSchema,
                              format="root")
    items = list(runner.preprocess(fileset))
    assert len(items) == 3
    chunks = [independent_pairs(D1_FILE, item.entrystart, item.entrystop, golden) for item in items]
    assert chunks[0] & chunks[1] or chunks[1] & chunks[2]
    stop = items[-1].entrystop
    unioned = runner(items, processor_instance=proc)["Muon0"][lumi_record.LUMI_KEY]
    single = proc.process(data_events(D1_FILE, "Muon0", 0, stop))["Muon0"][lumi_record.LUMI_KEY]
    assert unioned == single == independent_pairs(D1_FILE, 0, stop, golden)


def brilcalc_csv(path, values):
    rows = [f"{run}:9000,{lumi}:{lumi},06/01/24 00:00:00,STABLE BEAMS,6800,{value * 1.1},{value},50.0,HFET\n"
            for (run, lumi), value in sorted(values.items())]
    with open(path, "w") as f:
        f.write(CSV_HEADER + "".join(rows) + "#Summary:\n")


def test_report_with_csv(tmp_path):
    values = {(RUN, lumi): 0.001 * lumi for lumi in range(1, 21)}
    csv = str(tmp_path / "lumi.csv")
    brilcalc_csv(csv, values)
    a = [(RUN, lumi) for lumi in range(1, 13)]
    b = [(RUN, lumi) for lumi in range(9, 21)]
    a_twice = np.array(a + a[:3], dtype=np.uint32)
    path = str(tmp_path / "data.pkl.gz")
    write_pickle(path, {"A": {"hist": {}, "sumw": -1.0, "lumi": a_twice},
                        "B": {"hist": {}, "sumw": -1.0, "lumi": lumi_record.as_array(b)},
                        "MC": {"hist": {}, "sumw": 2.0}})
    result = lumi_record.report([path], lumi_csv=csv)[path]

    def total(pairs):
        return sum(values[p] for p in pairs)
    assert result["datasets"]["A"]["lumi_sections"] == 12
    assert result["datasets"]["A"]["lumi"] == pytest.approx(total(a), rel=1e-12)
    assert result["datasets"]["B"]["lumi"] == pytest.approx(total(b), rel=1e-12)
    assert result["datasets"]["A"]["first_run"] == result["datasets"]["A"]["last_run"] == RUN
    assert result["no_lumi_record"] == ["MC"]
    (pair,) = result["pairs"]
    assert (pair["a"], pair["b"], pair["only_a"], pair["only_b"], pair["both"]) == ("A", "B", 8, 8, 4)
    assert pair["lumi"]["only_a"] == pytest.approx(total(set(a) - set(b)), rel=1e-12)
    assert pair["lumi"]["only_b"] == pytest.approx(total(set(b) - set(a)), rel=1e-12)
    assert pair["lumi"]["both"] == pytest.approx(total(set(a) & set(b)), rel=1e-12)
    out = tmp_path / "exported"
    run = subprocess.run([sys.executable, "-m", "qawa.lumi_record", "report", path, "--lumi-csv", csv, "--json",
                          str(tmp_path / "r.json"), "--export-json", str(out)], capture_output=True, text=True, env=ENV)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads((tmp_path / "r.json").read_text())[path]["pairs"][0]["both"] == 4
    from coffea.lumi_tools import LumiMask
    mask = LumiMask(str(out / "A_lumis.json"))
    probe = np.array([(RUN, lumi) for lumi in range(0, 25)], dtype=np.uint32)
    assert mask(probe[:, 0], probe[:, 1]).tolist() == [(RUN, lumi) in set(a) for lumi in range(0, 25)]
    assert sorted(os.listdir(out)) == ["A_lumis.json", "B_lumis.json"]


@pytest.mark.skipif(importlib.util.find_spec("dctools") is None, reason="DCTools is not importable")
def test_dctools_tolerates_the_record(tmp_path):
    import dctools
    path = tmp_path / "data.pkl.gz"
    write_pickle(path, {"DoubleMuon": {"hist": {"x": data_hist()}, "sumw": -1.0,
                                       "lumi": lumi_record.as_array({(RUN, 1), (RUN, 2)})}})
    config = tmp_path / "config.yaml"
    config.write_text(f"boosthist:\n  - {path}\n")
    loaded = dctools.read_config(str(config))
    assert set(loaded.boosthist) == {"DoubleMuon"}
    group = dctools.datagroup(loaded.boosthist, observable="x", name="data", channel="chA", ptype="data")
    assert group.stacked.sum().value == 3


def test_optional_record_and_csv(tmp_path):
    bare = str(tmp_path / "bare.pkl.gz")
    write_pickle(bare, {"DoubleMuon": {"hist": {}, "sumw": -1.0}})
    run = subprocess.run([sys.executable, "-m", "qawa.lumi_record", "report", bare], capture_output=True, text=True,
                         env=ENV)
    assert run.returncode == 0 and "DoubleMuon: no lumi record" in run.stdout, run.stdout + run.stderr
    recorded = str(tmp_path / "recorded.pkl.gz")
    write_pickle(recorded, {"DoubleMuon": {"hist": {}, "sumw": -1.0, "lumi": lumi_record.as_array({(RUN, 4)})}})
    run = subprocess.run([sys.executable, "-m", "qawa.lumi_record", "report", recorded], capture_output=True, text=True,
                         env=ENV)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "lumi sections 1, lumi: None" in run.stdout and "--lumi-csv" in run.stdout
    assert lumi_record.report([recorded])[recorded]["datasets"]["DoubleMuon"]["lumi"] is None
    first = {"DoubleMuon": {"hist": {"x": data_hist()}, "sumw": -1.0}}
    second = {"DoubleMuon": dict(first["DoubleMuon"], lumi=lumi_record.as_array({(RUN, 4)}))}
    assert datadriven.check_reprocessed(first, second) == []
    assert datadriven.check_reprocessed(second, first) == []


@needs(LOCAL_2018)
def test_kind_record(local_output):
    import qawa.process.wztau2lnu_inclusive as wzmod
    from coffea.processor import accumulate
    _, out = local_output
    assert out[wzmod.KIND_KEY] == {"data"}
    record = wzmod.dataset_record("DoubleMuon", dict(out), -1.0)
    assert record[wzmod.KIND_KEY] == "data" and set(record["hist"]).isdisjoint({wzmod.KIND_KEY, lumi_record.LUMI_KEY})
    assert record[lumi_record.LUMI_KEY].dtype == np.uint32
    chunks = accumulate([{"D": {wzmod.KIND_KEY: {"mc"}}}, {"D": {wzmod.KIND_KEY: {"eft"}}}])
    with pytest.raises(RuntimeError, match=r"dataset D: its chunks report several kinds \['eft', 'mc'\]"):
        wzmod.dataset_record("D", chunks["D"], 1.0)
    assert "kind" not in wzmod.dataset_record("D", {}, 1.0)
