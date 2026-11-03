import gzip
import importlib.util
import os
import pickle
import subprocess
import sys

import pytest

import hist
import numpy as np

import qawa.datadriven as datadriven

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DD_RUN2 = os.path.join(ROOT, "src", "qawa", "data", "dd", "Run2", "WZ_inclusive_data_driven_Run2.json")
CAPS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "NUMEXPR_MAX_THREADS": "4"}
ENV = {**os.environ, **CAPS, "PYTHONPATH": os.path.join(ROOT, "src")}


def test_discover_systematics_reads_gzip(tmp_path):
    gz = tmp_path / "dd.json.gz"
    with open(DD_RUN2, "rb") as f:
        gz.write_bytes(gzip.compress(f.read()))
    assert datadriven.discover_systematics(str(gz)) == datadriven.discover_systematics(DD_RUN2)


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
