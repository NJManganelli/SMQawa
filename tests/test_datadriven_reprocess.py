import gzip
import importlib.util
import json
import os
import pickle
import subprocess
import sys

import pytest

import hist
import numpy as np

import qawa.datadriven as datadriven

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = "/Users/nmangane/vz-main/SMQawa/DoubleMuon_UL2018_NanoAODv9.root"
DD_RUN2 = os.path.join(ROOT, "src", "qawa", "data", "dd", "Run2", "WZ_inclusive_data_driven_Run2.json")
DD_ESTIMATE = "LNTTau_HighMET_DY_to_Data_Estimate"
CAPS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "NUMEXPR_MAX_THREADS": "4"}
ENV = {**os.environ, **CAPS, "PYTHONPATH": os.path.join(ROOT, "src")}
MESSAGE = "must be an uncompressed correctionlib JSON (.json)"


def doubled(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("value", "content", "default") and isinstance(value, (int, float)) and not isinstance(value, bool):
                node[key] = 2.0 * value
            elif key == "content" and isinstance(value, list):
                node[key] = [2.0 * x if isinstance(x, (int, float)) and not isinstance(x, bool) else (doubled(x) or x)
                             for x in value]
            else:
                doubled(value)
    elif isinstance(node, list):
        for value in node:
            doubled(value)


def doubled_file(path):
    with open(DD_RUN2) as f:
        content = json.load(f)
    for correction in content["corrections"]:
        if correction["name"] == DD_ESTIMATE:
            doubled(correction["data"])
    with open(path, "w") as f:
        json.dump(content, f)
    return str(path)


def gzip_copies(tmp_path):
    with open(DD_RUN2, "rb") as f:
        raw = gzip.compress(f.read())
    gz = tmp_path / "dd.json.gz"
    gz.write_bytes(raw)
    fake = tmp_path / "gzip_bytes.json"
    fake.write_bytes(raw)
    return str(gz), str(fake)


def run_data(dd_file=None):
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    from qawa.process.wztau2lnu_inclusive import wzinclusive_processor
    proc = wzinclusive_processor(era="2018", run_period="C", version="v9", dd_file=dd_file)
    events = NanoEventsFactory.from_root({DATA_FILE: "Events"}, schemaclass=NanoAODSchema, mode="virtual",
                                         metadata={"dataset": "DoubleMuon", "is_data": True, "is_signal": False,
                                                   "is_flashsim": False}).events()
    return proc, proc.process(events)["DoubleMuon"]


def label_view(h, label):
    return h[{"systematic": label}].view(flow=True)


@pytest.mark.skipif(not os.path.exists(DATA_FILE), reason="local 2018 data file missing")
def test_override_doubles_the_estimate(tmp_path):
    proc, default = run_data()
    _, override = run_data(doubled_file(tmp_path / "dd_doubled.json"))
    directions = ("Up", "Down")
    estimate = {f"datadriven_DDDYNominal{d}" for d in directions}
    estimate |= {f"datadriven_{s}{d}" for s in proc._dd.stat_systematics for d in directions}
    estimate |= {f"{s}{d}" for s in proc._dd.mc_systematics for d in directions}
    seen = {"estimate": False, "IRH": False, "other": False}
    assert set(default) == set(override)
    for key, h in default.items():
        if not isinstance(h, hist.Hist) or "systematic" not in h.axes.name:
            continue
        assert set(h.axes["systematic"]) == set(override[key].axes["systematic"])
        for label in h.axes["systematic"]:
            a, b = label_view(h, label), label_view(override[key], label)
            if label in estimate or label.endswith(("_IRHUp", "_IRHDown")):
                assert np.allclose(b["value"], 2 * a["value"], rtol=1e-12, atol=0), (key, label)
                assert np.allclose(b["variance"], 4 * a["variance"], rtol=1e-12, atol=0), (key, label)
                seen["estimate" if label in estimate else "IRH"] |= bool(np.any(a["value"]))
            else:
                assert np.array_equal(a["value"], b["value"], equal_nan=True), (key, label)
                assert np.array_equal(a["variance"], b["variance"], equal_nan=True), (key, label)
                seen["other"] |= label != "nominal" and bool(np.any(a["value"]))
    assert all(seen.values()), seen


def data_hist(scale=1.0, up=1.0, channels=("chA", "chB")):
    h = hist.Hist(hist.axis.StrCategory([], name="channel", growth=True),
                  hist.axis.StrCategory([], name="systematic", growth=True),
                  hist.axis.Regular(10, 0, 100, name="x"), hist.storage.Weight())
    rng = np.random.default_rng(7)
    x = rng.uniform(-5, 105, 200)
    w = rng.uniform(0.5, 1.5, 200)
    for channel in channels:
        h.fill(channel=channel, systematic="nominal", x=x, weight=w * scale)
        h.fill(channel=channel, systematic="datadriven_DDDYNominalUp", x=x, weight=w * up)
    return h


def content(sumw=-1.0, kind=None, **hist_args):
    record = {"hist": {"x": data_hist(**hist_args), "y": data_hist(**hist_args)}, "sumw": sumw}
    if kind is not None:
        record["kind"] = kind
    return {"DoubleMuon": record}


def test_check_reprocessed():
    first = content()
    assert datadriven.check_reprocessed(first, content()) == []
    assert datadriven.check_reprocessed(first, content(up=2.0)) == []
    for factor, n_problems in ((1 + 2e-9, 1), (1 + 5e-10, 0)):
        one = content()
        h = one["DoubleMuon"]["hist"]["x"]
        h.view(flow=True)["value"][0, list(h.axes["systematic"]).index("nominal"), 3] *= factor
        problems = datadriven.check_reprocessed(first, one)
        assert len(problems) == n_problems and all("DoubleMuon x: nominal differs in 1 cells" in p for p in problems)
    assert datadriven.check_reprocessed(first, {}) == ["DoubleMuon: not re-processed"]
    extra = dict(content(), MuonEG=content()["DoubleMuon"])
    assert datadriven.check_reprocessed(first, extra) == ["MuonEG: not in the first pass"]
    for mc in (content(kind="mc"), content(sumw=3.0)):
        assert "DoubleMuon (re-processed file): not a data dataset" in datadriven.check_reprocessed(first, mc)
        assert "DoubleMuon (first file): not a data dataset" in datadriven.check_reprocessed(mc, first)
    annotated = content(kind="data")
    annotated["DoubleMuon"]["lumi"] = np.zeros((0, 2), dtype=np.uint32)
    annotated["DoubleMuon"]["production"] = {"truncated": False}
    assert datadriven.check_reprocessed(first, annotated) == []
    assert "channel labels differ" in datadriven.check_reprocessed(first, content(channels=("chA",)))[0]
    assert "sumw differs" in datadriven.check_reprocessed(first, content(sumw=-2.0))[0]
    fewer = content()
    del fewer["DoubleMuon"]["hist"]["y"]
    assert "hist keys differ" in datadriven.check_reprocessed(first, fewer)[0]


@pytest.mark.skipif(importlib.util.find_spec("dctools") is None, reason="DCTools is not importable")
def test_check_reprocessed_cli(tmp_path):
    paths = {}
    for name, value in (("first", content()), ("same", content(up=3.0)), ("changed", content(scale=1 + 1e-6))):
        paths[name] = str(tmp_path / f"{name}.pkl.gz")
        with gzip.open(paths[name], "wb") as f:
            pickle.dump(value, f)
    for name, code in (("same", 0), ("changed", 1)):
        run = subprocess.run([sys.executable, "-m", "qawa.datadriven", "--check-reprocessed", paths["first"], paths[name]],
                             capture_output=True, text=True, env=ENV)
        assert run.returncode == code, run.stdout + run.stderr


def test_check_dd_file(tmp_path):
    assert datadriven.check_dd_file(DD_RUN2) is None
    bad = list(gzip_copies(tmp_path))
    for name, text in (("text.json", "not json"), ("empty.json", "{}"), ("list.json", "[]"),
                       ("v1.json", json.dumps({"schema_version": 1, "corrections": []}))):
        (tmp_path / name).write_text(text)
        bad.append(str(tmp_path / name))
    for path in bad:
        with pytest.raises(ValueError, match=r"must be an uncompressed correctionlib JSON \(\.json\)"):
            datadriven.check_dd_file(path)


def test_brewer_refusals(tmp_path):
    gz, fake = gzip_copies(tmp_path)
    infile = "/store/data/Run2018C/DoubleMuon/NANOAOD/UL2018_MiniAODv2_NanoAODv9-v1/x.root"
    mc_infile = "/store/mc/RunIISummer20UL18NanoAODv9/WZTo3LNu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X-v2/x.root"
    base = [sys.executable, os.path.join(ROOT, "brewer-remote-inclusive.py"), "--jobNum", "0", "--era", "2018"]
    cases = [(["--infile", mc_infile, "--isMC", "1", "--analysis", "inc-WZ", "--dd-file", DD_RUN2],
              "re-processes inc-WZ data only"),
             (["--infile", infile, "--isMC", "0", "--analysis", "inc-ZZ", "--dd-file", DD_RUN2],
              "re-processes inc-WZ data only"),
             (["--infile", infile, "--isMC", "0", "--analysis", "inc-WZ", "--dd-file", gz], MESSAGE),
             (["--infile", infile, "--isMC", "0", "--analysis", "inc-WZ", "--dd-file", fake], MESSAGE)]
    for args, message in cases:
        run = subprocess.run(base + args, capture_output=True, text=True, env=ENV, cwd=tmp_path)
        assert run.returncode != 0 and message in run.stderr and "Opening" not in run.stdout + run.stderr, args
    with open(tmp_path / "input.txt", "w") as f:
        f.write("/DoubleMuon/Run2018C-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD\n")
    for path in (gz, fake):
        run = subprocess.run([sys.executable, os.path.join(ROOT, "brewer-htcondor-inclusive.py"), "-a", "inc-WZ",
                              "-i", "input.txt", "-t", "t", "-e", "2018", "--dryrun", "--dd-file", path],
                             capture_output=True, text=True, env=ENV, cwd=tmp_path)
        assert run.returncode != 0 and MESSAGE in run.stderr, run.stderr
        assert not [d for d in os.listdir(tmp_path) if d.startswith("jobs_")]
    from qawa.process.wztau2lnu_inclusive import wzinclusive_processor
    for path in (gz, fake):
        with pytest.raises(ValueError, match=r"must be an uncompressed correctionlib JSON"):
            wzinclusive_processor(era="2018", run_period="C", version="v9", dd_file=path)
