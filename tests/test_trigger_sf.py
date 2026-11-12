"""Di-lepton trigger scale factors (inc-WZ): correctionlib-only consumer.

The Run 2 jsons are the flavour-only conversions of histo_triggerEff_sel0_<era>.root
(convert_trigger_sf.py, inverse-variance combination of the four eta regions).
The processor's _add_trigger_sf evaluates them on (lead pt, trail pt) only.
"""
import inspect
import os
from types import SimpleNamespace

import awkward as ak
import correctionlib
import numpy as np
import pytest
import uproot
from coffea.analysis_tools import Weights

from conftest import CVMFS_HEAD, REPO, input_path
from convert_trigger_sf import CORRECTION_NAME, ETA_REGIONS, FLAVORS, combine_regions, root_path
from qawa.jsoncorrections import CorrectionlibHandler
from qawa.measurements import TRIGGER_SF_FLAVOURS, TRIGGER_SF_SYSTEMATICS, trigger_sf_payload, write_payload
import qawa.process.wztau2lnu_inclusive as wzmod

CVMFS, needs_cvmfs = input_path(CVMFS_HEAD, "cms-griddata.cern.ch")
DATA_DIR = os.path.join(REPO, "src", "qawa", "data", "trigger_sf")
ERAS = ("2016", "2017", "2018")
PDG = {"E": 11, "M": 13}
PT_EDGES = [20, 25, 30, 35, 40, 50, 60, 70]
PT_CENTERS = [22.5, 27.5, 32.5, 37.5, 45.0, 55.0, 65.0]


class ClibShim:
    def __init__(self, path):
        self.cset = correctionlib.CorrectionSet.from_file(str(path))

    def keys(self):
        return ["trigger_sf"]

    def getCorrectionSet(self, key):
        assert key == "trigger_sf"
        return self.cset


def processor_shim(path, alpha=False):
    return SimpleNamespace(clibhandler=ClibShim(path), _trigger_sf_alpha=alpha)


def leptons(lead_pdg, subl_pdg, lead_pt, subl_pt, lead_eta=0.5, subl_eta=0.5):
    n = len(lead_pt)

    def record(pdg, pt, eta):
        return ak.zip({"pt": np.asarray(pt, dtype=float), "eta": np.full(n, eta),
                       "pdgId": np.full(n, pdg, dtype=np.int64)})
    return record(lead_pdg, lead_pt, lead_eta), record(subl_pdg, subl_pt, subl_eta)


def trigger_weights(shim, lead, subl):
    weights = Weights(len(lead), storeIndividual=True)
    wzmod.wzinclusive_processor._add_trigger_sf(shim, weights, lead, subl)
    return (np.asarray(weights.weight()), np.asarray(weights.weight(modifier="triggerSFUp")),
            np.asarray(weights.weight(modifier="triggerSFDown")))


@needs_cvmfs
@pytest.mark.parametrize("version", ["v9", "v15"])
@pytest.mark.parametrize("era,isAPV", [("2016", True), ("2016", False), ("2017", False), ("2018", False)])
def test_run2_trigger_sf_registered(era, isAPV, version):
    handler = CorrectionlibHandler(era=era, analysis="inc-WZ", isAPV=isAPV, nanoAODversion=version,
                                   cvmfs_head=CVMFS_HEAD)
    assert handler.getPath("trigger_sf").name == f"triggerSF_{era}.json"


@needs_cvmfs
def test_missing_trigger_sf_raises(monkeypatch):
    keys = CorrectionlibHandler.keys
    monkeypatch.setattr(CorrectionlibHandler, "keys", lambda self: [k for k in keys(self) if k != "trigger_sf"])
    with pytest.raises(ValueError, match="trigger_sf"):
        wzmod.wzinclusive_processor(era="2018", version="v15")
    assert "clibhandler" not in inspect.signature(wzmod.wzinclusive_processor._add_trigger_sf).parameters


@pytest.mark.parametrize("era", ERAS)
def test_json_matches_root_combination(era):
    shim = processor_shim(os.path.join(DATA_DIR, f"triggerSF_{era}.json"))
    lead_idx, subl_idx = np.array([(i, j) for i in range(7) for j in range(i + 1)]).T
    lead_pt, subl_pt = np.take(PT_CENTERS, lead_idx), np.take(PT_CENTERS, subl_idx)
    with uproot.open(root_path(era, DATA_DIR)) as root_file:
        for flavor in FLAVORS:
            hists = [root_file[f"trgSF{flavor}{region}"] for region in ETA_REGIONS]
            sf, sigma = combine_regions([h.values() for h in hists], [np.sqrt(h.variances()) for h in hists])
            nominal, up, down = trigger_weights(shim, *leptons(PDG[flavor[0]], PDG[flavor[1]], lead_pt, subl_pt))
            expected = sf[lead_idx, subl_idx]
            assert np.allclose(nominal, expected, rtol=0, atol=1e-9), f"{era} {CORRECTION_NAME[flavor]}"
            assert np.allclose(up, expected + sigma[lead_idx, subl_idx], rtol=0, atol=1e-9)
            assert np.allclose(down, expected - sigma[lead_idx, subl_idx], rtol=0, atol=1e-9)
            clamped, _, _ = trigger_weights(shim, *leptons(PDG[flavor[0]], PDG[flavor[1]], [500.0], [65.0]))
            assert np.allclose(clamped, sf[6, 6], rtol=0, atol=1e-9)


@needs_cvmfs
def test_flavour_masks_use_pdgid_only():
    handler = CorrectionlibHandler(era="2018", analysis="inc-WZ", nanoAODversion="v15", cvmfs_head=CVMFS_HEAD)
    shim = SimpleNamespace(clibhandler=handler, _trigger_sf_alpha=False)
    for lead_pdg, subl_pdg in [(13, 13), (13, 11), (11, 13), (11, 11)]:
        negative = trigger_weights(shim, *leptons(lead_pdg, subl_pdg, [45.0], [32.5], -2.0, -2.0))
        positive = trigger_weights(shim, *leptons(lead_pdg, subl_pdg, [45.0], [32.5], 2.0, 2.0))
        assert np.array_equal(np.stack(negative), np.stack(positive))


def variant_payload(tmp_path):
    shape = (len(PT_EDGES) - 1, len(PT_EDGES) - 1)
    offsets = {"nominal": 0.0, "up": 0.01, "down": -0.01, "alpha_up": 0.05, "alpha_down": -0.05}
    grids = {flavour: {key: np.full(shape, 0.9 + offsets[key]) for key in TRIGGER_SF_SYSTEMATICS}
             for flavour in TRIGGER_SF_FLAVOURS}
    path = tmp_path / "triggerSF.json"
    write_payload(trigger_sf_payload(grids, (PT_EDGES, PT_EDGES), "test"), path)
    return path


@needs_cvmfs
def test_trigger_sf_variant(tmp_path):
    path = variant_payload(tmp_path)
    lead, subl = leptons(13, 11, [45.0, 65.0], [32.5, 22.5])
    stat = trigger_weights(processor_shim(path, alpha=False), lead, subl)
    alpha = trigger_weights(processor_shim(path, alpha=True), lead, subl)
    assert np.allclose(stat[0], 0.9) and np.array_equal(stat[0], alpha[0])
    assert np.allclose(stat[1], 0.91) and np.allclose(stat[2], 0.89)
    assert np.allclose(alpha[1], 0.95) and np.allclose(alpha[2], 0.85)
    with pytest.raises(ValueError, match="alpha_up"):
        wzmod.wzinclusive_processor(era="2018", version="v15", trigger_sf_alpha=True)
