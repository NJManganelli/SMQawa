"""2025 data-taking: the handler payloads, the JME tags and the processor build for 2025 data and Summer24 MC."""
import os
from pathlib import Path

import pytest
import yaml

from conftest import CVMFS_HEAD, input_path

CVMFS, needs_cvmfs = input_path(CVMFS_HEAD, "cms-griddata.cern.ch")
DATA_PATH = Path(__file__).resolve().parent.parent / "src" / "qawa" / "data"

JEC_TAG = "Summer24Prompt25_V3"
JER_TAG = "Summer24Prompt25_JRV2"
JETVETO_TAG = "Summer24Prompt25_RunCDEFG_V1"
JET_TYPE = "AK4PFPuppi"


def handler():
    from qawa.jsoncorrections import CorrectionlibHandler
    return CorrectionlibHandler(era="2025", analysis="inc-WZ", nanoAODversion="v15", cvmfs_head=CVMFS_HEAD)


@needs_cvmfs
def test_handler_paths_exist():
    h = handler()
    expected = {"jet_jerc", "fatJet_jerc", "jetid", "jetvetomaps", "btagging", "electron", "electronSS_EtDependent",
                "muon_Z", "muon_scalesmearing", "tau", "puWeights", "dddy", "trigger_sf"}
    assert expected <= set(h.keys())
    for key in h.keys():
        assert Path(h.getPath(key)).exists(), f"{key}: {h.getPath(key)}"
    for key in ("jet_jerc", "jetid", "jetvetomaps", "electron", "muon_Z", "tau", "puWeights"):
        assert "Run3-25Prompt-Summer24-NanoAODv15" in str(h.getPath(key)), key
    assert Path(h.getPath("Cert_Collisions2025_391658_398903_Golden")).exists()


@needs_cvmfs
def test_jme_tags_resolve():
    h = handler()
    jerc = h.getCorrectionSet("jet_jerc")
    for data_type in ("MC", "DATA"):
        assert f"{JEC_TAG}_{data_type}_L1L2L3Res_{JET_TYPE}" in jerc.compound
        assert f"{JEC_TAG}_{data_type}_L1FastJet_{JET_TYPE}" in jerc
    assert "run" in [i.name for i in jerc[f"{JEC_TAG}_DATA_L2L3Residual_{JET_TYPE}"].inputs]
    assert f"{JEC_TAG}_MC_Regrouped_Absolute_2025_{JET_TYPE}" in jerc
    for name in ("PtResolution", "ScaleFactor", "SFUncertainty"):
        assert f"{JER_TAG}_MC_{name}_{JET_TYPE}" in jerc
    assert JETVETO_TAG in h.getCorrectionSet("jetvetomaps")
    assert "AK4PUPPI_TightLeptonVeto" in h.getCorrectionSet("jetid")


@needs_cvmfs
@pytest.mark.parametrize("run_period", ["", "C", "E", "G"])
def test_processor_builds(run_period):
    import qawa.process.wztau2lnu_inclusive as wzmod
    kwargs = {"run_period": run_period} if run_period else {}
    proc = wzmod.wzinclusive_processor(era="2025", version="v15", **kwargs)
    assert proc._use_clib_jme
    assert proc._jetveto is not None
    assert proc._jetid.cset is not None
    assert proc._lumimask_path.endswith("Cert_Collisions2025_391658_398903_Golden.json")


@needs_cvmfs
@pytest.mark.parametrize("era", ["2022", "2023"])
def test_unimplemented_run3_eras_raise(era):
    import qawa.process.wztau2lnu_inclusive as wzmod
    with pytest.raises(NotImplementedError):
        wzmod.wzinclusive_processor(era=era, version="v15")


def test_trigger_rules_keys():
    with open(os.path.join(DATA_PATH, "2025-trigger-rules.yaml")) as f:
        rules = yaml.load(f, Loader=yaml.FullLoader)
    assert set(rules) == {"Run2025.Muon", "Run2025.EGamma", "Run2025.MuonEG", "Run2025.MC", "Run2025.Reference"}
    with open(os.path.join(DATA_PATH, "2024-trigger-rules.yaml")) as f:
        rules_2024 = yaml.load(f, Loader=yaml.FullLoader)
    assert {k.replace("2025", "2024"): v for k, v in rules.items()} == rules_2024
