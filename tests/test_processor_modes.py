import os

import awkward as ak
import numpy as np
import pytest

import qawa.process.wztau2lnu_inclusive as wzmod

M1_FILE = ("/Volumes/WDMac/SMQawaTestFiles/store/mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/"
           "NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/test.root")
needs_m1 = pytest.mark.skipif(not os.path.exists(M1_FILE), reason="local 2024 WZto3LNu test file not available")

FAKE_TERMS = ("sm", "lin_cW", "quad_cW")


class FakeParameterisation:
    terms = FAKE_TERMS

    def decompose(self, x):
        return x

    def validate(self, theta):
        return []

    def active_terms(self, theta=None):
        return list(self.terms)


def fake_theta(event, points):
    gw = ak.to_numpy(event.genWeight).astype(np.float64)
    return np.stack([gw * (1.0 + 0.1 * t) for t in range(len(points))], axis=1)


def m1_events(n):
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    return NanoEventsFactory.from_root(
        {M1_FILE: "Events"}, schemaclass=NanoAODSchema, mode="virtual", entry_start=0, entry_stop=n,
        metadata={"dataset": "WZto3LNu", "is_data": False, "is_signal": True, "is_flashsim": False},
    ).events()


@pytest.fixture
def fake_eft(monkeypatch):
    monkeypatch.setattr(wzmod, "points_from_events", lambda event: (FakeParameterisation(), FAKE_TERMS))
    monkeypatch.setattr(wzmod, "stack_named_weights", fake_theta)
    process_shift = wzmod.wzinclusive_processor.process_shift

    def nominal_only(self, event, shift_name=''):
        return process_shift(self, event, shift_name) if shift_name is None else {}
    monkeypatch.setattr(wzmod.wzinclusive_processor, "process_shift", nominal_only)


@needs_m1
def test_pol_with_eft_raises(fake_eft):
    proc = wzmod.wzinclusive_processor(era="2024", version="v15", pol=True)
    with pytest.raises(NotImplementedError, match="EFT"):
        proc.process(m1_events(500))


@needs_m1
def test_eft_without_pol_fills(fake_eft):
    proc = wzmod.wzinclusive_processor(era="2024", version="v15", pol=False)
    out = proc.process(m1_events(500))["WZto3LNu"]
    eft_keys = [k for k in out if k.endswith("_eft")]
    assert eft_keys
    assert any(len(out[k].axes[0]) for k in eft_keys)
    assert set(out["dilep_deta_eft"].axes[2]) == set(FAKE_TERMS)
