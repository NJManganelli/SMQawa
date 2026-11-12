"""The inclusive WZ processor fills the trigger and b-tag measurement histograms.

Data assertions are on key names and booleans only: no data yield is printed or compared.
"""
import hist
import numpy as np
import pytest

from conftest import STORE, input_path
import qawa.process.wztau2lnu_inclusive as wzmod

M1_FILE, needs_m1 = input_path(STORE, "mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/"
                                       "NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/test.root")
D1_FILE, needs_d1 = input_path(STORE, "data/Run2024C/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root")
MEASUREMENTS = ("trigger_measurement", "btag_measurement")
TT_DATASET = "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8"


def events(path, n, dataset, is_data):
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    return NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="virtual", entry_start=0, entry_stop=n,
        metadata={"dataset": dataset, "is_data": is_data, "is_signal": False, "is_flashsim": False},
    ).events()


def histogram_keys(out):
    return {k for k, v in out.items() if isinstance(v, hist.Hist)} - set(MEASUREMENTS)


def filled(h):
    return len(h.axes["channel"]) > 0 and bool(np.any(h.values(flow=True) != 0))


@pytest.fixture
def nominal_only(monkeypatch):
    process_shift = wzmod.wzinclusive_processor.process_shift

    def nominal(self, event, shift_name=''):
        return process_shift(self, event, shift_name) if shift_name is None else {}
    monkeypatch.setattr(wzmod.wzinclusive_processor, "process_shift", nominal)


@pytest.fixture
def recorders(monkeypatch):
    recorded = {}

    class RecordedWeights(wzmod.Weights):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorded["weights"] = self

    class RecordedSelection(wzmod.PackedSelection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorded["selection"] = self
    monkeypatch.setattr(wzmod, "Weights", RecordedWeights)
    monkeypatch.setattr(wzmod, "PackedSelection", RecordedSelection)
    return recorded


@pytest.fixture(scope="module")
def data_outputs():
    proc = wzmod.wzinclusive_processor(era="2024", version="v15", run_period="C")
    return {stream: proc.process(events(D1_FILE, 20000, f"{stream}_Run2024C", True))[f"{stream}_Run2024C"]
            for stream in ("JetMET0", "Muon0")}


@pytest.fixture(scope="module")
def mc_processor():
    return wzmod.wzinclusive_processor(era="2024", version="v15")


@needs_d1
def test_jetmet_fills_only_measurement(data_outputs):
    out = data_outputs["JetMET0"]
    assert not {k for k in histogram_keys(out) if filled(out[k])}
    assert filled(out["trigger_measurement"])
    assert set(out["trigger_measurement"].axes["channel"]) <= {f"trigger-{f}" for f in ("mm", "me", "em", "ee")}
    if "btag_measurement" in out:
        assert not filled(out["btag_measurement"])


@needs_d1
def test_lepton_stream_data_fills_no_measurement(data_outputs):
    out = data_outputs["Muon0"]
    assert not {k for k in MEASUREMENTS if k in out and filled(out[k])}
    assert {k for k in histogram_keys(out) if filled(out[k])}


@needs_m1
def test_mc_measurements(mc_processor, nominal_only):
    as_tt = mc_processor.process(events(M1_FILE, 3000, TT_DATASET, False))[TT_DATASET]
    trigger = as_tt["trigger_measurement"]
    assert filled(trigger) and list(trigger.axes["systematic"]) == ["nominal"]
    itself = mc_processor.process(events(M1_FILE, 3000, "WZto3LNu", False))["WZto3LNu"]
    assert not filled(itself["trigger_measurement"])


@needs_m1
def test_measurement_weight_excludes_sf(mc_processor, nominal_only, recorders):
    out = mc_processor.process(events(M1_FILE, 3000, TT_DATASET, False))[TT_DATASET]
    selection, weights = recorders["selection"], recorders["weights"]
    cleaning = ["lumimask", "goodPV", "metfilter"] + (["jetveto"] if "jetveto" in selection.names else [])
    selected = np.zeros(len(weights.weight()), dtype=bool)
    for flavour in ("mm", "me", "em", "ee"):
        selected |= selection.all(*cleaning, "require-2lep", f"trigger-{flavour}")
    expected = weights.partial_weight(exclude=["triggerSF"])[selected].sum()
    total = out["trigger_measurement"].values(flow=True).sum()
    assert selected.any()
    assert np.isclose(total, expected, rtol=1e-9)
    assert not np.isclose(total, weights.weight()[selected].sum(), rtol=1e-6)
