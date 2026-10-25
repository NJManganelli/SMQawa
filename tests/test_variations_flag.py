import os

import awkward as ak
import numpy as np
import pytest

import qawa.process.wztau2lnu_inclusive as wzmod
from qawa.tauSF import tauIDScaleFactors

STORE = "/Volumes/WDMac/SMQawaTestFiles/store"
M1_FILE = (f"{STORE}/mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/"
           "NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/test.root")
M9_FILE = (f"{STORE}/mc/RunIISummer20UL18NanoAODv9/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/"
           "NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2500000/test.root")
NEVENTS = 2000
WPS = ("VTight", "Tight", "Loose")

V15_ID = {
    "e": ("idDeepTau2018v2p5VSe", {"VVLoose": 2, "Tight": 6, "VTight": 7}),
    "mu": ("idDeepTau2018v2p5VSmu", {"Tight": 4}),
    "j": ("idDeepTau2018v2p5VSjet", {"Loose": 4, "Tight": 6, "VTight": 7}),
    "max_eta": 2.5,
}
V9_ID = {
    "e": ("idDeepTau2017v2p1VSe", {"VVLoose": 2, "Tight": 32, "VTight": 64}),
    "mu": ("idDeepTau2017v2p1VSmu", {"Tight": 8}),
    "j": ("idDeepTau2017v2p1VSjet", {"Loose": 8, "Tight": 32, "VTight": 64}),
    "max_eta": 2.3,
}
CASES = [
    pytest.param(M1_FILE, "v15", V15_ID, marks=pytest.mark.skipif(not os.path.exists(M1_FILE), reason="local 2024 file not available")),
    pytest.param(M9_FILE, "v9", V9_ID, marks=pytest.mark.skipif(not os.path.exists(M9_FILE), reason="local 2018 file not available")),
]


def load_events(path):
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    return NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="eager", entry_start=0, entry_stop=NEVENTS,
    ).events()


def independent_taus(tau, lep, ids, wp_e, wp_mu, wp_j):
    branch_e, cuts_e = ids["e"]
    branch_mu, cuts_mu = ids["mu"]
    branch_j, cuts_j = ids["j"]
    mask = (
        (tau.pt > 20)
        & (np.abs(tau.eta) < ids["max_eta"])
        & (np.abs(tau.dz) < 0.2)
        & (tau.decayMode != 2) & (tau.decayMode != 5) & (tau.decayMode != 6)
        & (tau[branch_e] >= cuts_e[wp_e])
        & (tau[branch_mu] >= cuts_mu[wp_mu])
        & (tau[branch_j] >= cuts_j[wp_j])
        & ak.all(tau.metric_table(lep) > 0.4, axis=2)
    )
    return tau[mask]


def same_taus(a, b):
    assert ak.to_list(ak.num(a)) == ak.to_list(ak.num(b))
    for field in ("pt", "eta", "phi"):
        assert np.array_equal(ak.to_numpy(ak.flatten(a[field])), ak.to_numpy(ak.flatten(b[field])))


@pytest.mark.parametrize("path, version, ids", CASES)
def test_build_htaus_multi_wp(path, version, ids):
    events = load_events(path)
    lep = events.Muon
    taus = wzmod.build_htaus(events.Tau, lep, nanoAODversion=version, tauIDvsj_wps=WPS,
                             tauIDvse_wp="VTight", tauIDvsmu_wp="Tight")
    assert list(taus) == list(WPS)
    assert ak.sum(ak.num(taus["Loose"])) > ak.sum(ak.num(taus["VTight"])) > 0
    for wp in WPS:
        same_taus(taus[wp], independent_taus(events.Tau, lep, ids, "VTight", "Tight", wp))
        single = wzmod.build_htaus(events.Tau, lep, nanoAODversion=version, tauIDvsj_wps=(wp,),
                                   tauIDvse_wp="Tight", tauIDvsmu_wp="Tight")
        assert list(single) == [wp]
        multi = wzmod.build_htaus(events.Tau, lep, nanoAODversion=version, tauIDvsj_wps=WPS,
                                  tauIDvse_wp="Tight", tauIDvsmu_wp="Tight")
        same_taus(single[wp], multi[wp])
        same_taus(single[wp], independent_taus(events.Tau, lep, ids, "Tight", "Tight", wp))


@pytest.mark.parametrize("path, version, ids", CASES)
def test_build_htaus_api_errors(path, version, ids):
    events = load_events(path)
    with pytest.raises(TypeError):
        wzmod.build_htaus(events.Tau, events.Muon, nanoAODversion=version, tauIDvsj_wp="VTight")
    with pytest.raises(ValueError, match="VVTight"):
        wzmod.build_htaus(events.Tau, events.Muon, nanoAODversion=version, tauIDvsj_wps=("Tight", "Bogus"))
    assert not hasattr(wzmod, "build_htaus_wps")


class RaisingCorrection:
    @property
    def inputs(self):
        raise AssertionError("tau energy scale evaluated in getSF")

    def evaluate(self, *args):
        raise AssertionError("tau energy scale evaluated in getSF")


def direct_sf(corr, inputs, ntaus):
    sf = corr.evaluate(*[inputs[k.name] for k in corr.inputs])
    return ak.prod(ak.unflatten(ak.fill_none(sf, 1.0), ntaus), axis=-1)


@pytest.mark.skipif(not os.path.exists(M9_FILE), reason="local 2018 file not available")
def test_getsf_skips_tau_energy_scale():
    events = load_events(M9_FILE)
    taus = wzmod.build_htaus(events.Tau, events.Muon, nanoAODversion="v9", tauIDvsj_wps=("Loose",),
                             tauIDvse_wp="VTight", tauIDvsmu_wp="Tight")["Loose"]
    assert ak.sum(ak.num(taus)) > 0
    sf = tauIDScaleFactors(era="2018", vsjet_wp="VTight", vse_wp="VVLoose", vsmu_wp="Tight")
    sf.corr_enscale = RaisingCorrection()
    for syst in ("nom", "up", "down"):
        vsjet, vse, vsmu = sf.getSF(taus, syst=syst)
        flat = ak.flatten(taus)
        pt, eta = np.array(flat.pt), np.array(flat.eta)
        dm, gen = np.array(flat.decayMode), np.array(flat.genPartFlav)
        ntaus = ak.num(taus)
        exp_vsjet = direct_sf(sf.corr_vsjet, {"pt": pt, "dm": dm, "genmatch": gen, "syst": syst, "wp": "VTight",
                                              "wp_VSe": "VVLoose", "flag": "dm"}, ntaus)
        exp_vse = direct_sf(sf.corr_vse, {"eta": eta, "dm": dm, "genmatch": gen, "wp": "VVLoose", "syst": syst}, ntaus)
        exp_vsmu = direct_sf(sf.corr_vsmu, {"eta": eta, "genmatch": gen, "wp": "Tight", "wp_VSjet": "Tight",
                                            "wp_VSe": "VVLoose", "syst": syst}, ntaus)
        assert np.array_equal(ak.to_numpy(vsjet), ak.to_numpy(exp_vsjet))
        assert np.array_equal(ak.to_numpy(vse), ak.to_numpy(exp_vse))
        assert np.array_equal(ak.to_numpy(vsmu), ak.to_numpy(exp_vsmu))
