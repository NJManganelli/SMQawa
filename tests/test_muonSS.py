import importlib.util
import inspect
import os
import warnings
from functools import partial

import numpy as np
import awkward as ak
import pytest
import scipy.stats
import correctionlib.schemav2 as schemav2
from coffea.lookup_tools.doublecrystalball import doublecrystalball
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import qawa
from qawa.jsoncorrections import CorrectionlibHandler
from qawa.muonSS import muon_pt_scare

expected_root = os.environ.get("QAWA_EXPECTED_ROOT")
if expected_root and not os.path.realpath(qawa.__file__).startswith(os.path.realpath(expected_root)):
    pytest.exit(f"qawa imported from {qawa.__file__}, expected under {expected_root}", returncode=3)

warnings.filterwarnings("ignore")
NanoAODSchema.warn_missing_crossrefs = False

MUMC = "/Volumes/WDMac/SMQawaTestFiles/store/mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/test.root"
MUDATA = "/Volumes/WDMac/SMQawaTestFiles/store/data/Run2024C/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root"

needs_mc = pytest.mark.skipif(not os.path.exists(MUMC), reason=f"missing {MUMC}")
needs_data = pytest.mark.skipif(not os.path.exists(MUDATA), reason=f"missing {MUDATA}")

HANDLER_ERAS = [("2016", True, "v15"), ("2016", False, "v15"), ("2017", False, "v15"), ("2018", False, "v15"),
                ("2024", False, "v15"), ("2025", False, "v15"), ("2018", False, "v9")]


def f(x):
    return ak.to_numpy(ak.flatten(x))


def load(path, start=None, stop=20000, mode="eager"):
    return NanoEventsFactory.from_root({path: "Events"}, schemaclass=NanoAODSchema, mode=mode,
                                       entry_start=start, entry_stop=stop).events()


def kit_random_smearing():
    corr = schemav2.Correction(
        name="RandomSmearing", version=1,
        inputs=[schemav2.Variable(name="evtNr", type="int"), schemav2.Variable(name="lumiNr", type="int"),
                schemav2.Variable(name="phi", type="real")],
        output=schemav2.Variable(name="random_number", type="real"),
        data=schemav2.HashPRNG(nodetype="hashprng", inputs=["evtNr", "lumiNr", "phi"], distribution="stdflat"))
    return corr.to_evaluator()


def muon_inputs(events):
    mu = events.Muon
    n_mu = ak.num(mu)
    return {
        "pt": f(mu.pt), "eta": f(mu.eta), "phi": f(mu.phi), "charge": f(mu.charge), "nL": f(mu.nTrackerLayers),
        "event": np.repeat(np.asarray(events.event), n_mu), "lumi": np.repeat(np.asarray(events.luminosityBlock), n_mu),
        "counts": ak.to_numpy(n_mu),
    }


def u_kit(ref):
    return kit_random_smearing().evaluate(ref["event"], ref["lumi"], ref["phi"])


def u_prod(handler, ref):
    return handler.getCorrectionSet("muon_smear_random")["RandomSmearing"].evaluate(ref["event"], ref["lumi"], ref["phi"])


def corrected_with_systematics(events, handler):
    muon_pt_scare(None, events=events, is_correction=True, clibhandler=handler)
    muon = events.Muon
    muon.add_systematic("scale_m", "UpDownSystematic", "pt", partial(muon_pt_scare, events=events, unc_type="Scale", is_correction=False, clibhandler=handler))
    muon.add_systematic("res_m", "UpDownSystematic", "pt", partial(muon_pt_scare, events=events, unc_type="Resolution", is_correction=False, clibhandler=handler))
    return muon


def muon_arrays(muon):
    return {
        "pt": f(muon.pt), "pt_scalecorr": f(muon.pt_scalecorr), "pt_scarecorr": f(muon.pt_scarecorr),
        "scale_up": f(muon.systematics.scale_m.up.pt), "scale_down": f(muon.systematics.scale_m.down.pt),
        "res_up": f(muon.systematics.res_m.up.pt), "res_down": f(muon.systematics.res_m.down.pt),
    }


@pytest.fixture(scope="module")
def handler():
    return CorrectionlibHandler(era="2024", subera=None, analysis="inc-WZ", nanoAODversion="v15", cvmfs_head="/cvmfs/")


@pytest.fixture(scope="module")
def mumc_events():
    return load(MUMC)


@pytest.fixture(scope="module")
def mudata_events():
    return load(MUDATA)


def test_mu_payload():
    with_scalesmearing = 0
    for era, is_apv, version in HANDLER_ERAS:
        h = CorrectionlibHandler(era=era, subera=None, isAPV=is_apv, analysis="inc-WZ", nanoAODversion=version, cvmfs_head="/cvmfs/")
        assert "muon_smear_random" in h.keys(), (era, is_apv, version)
        assert str(h["muon_smear_random"]).endswith("qawa/data/muo/muon_smear_random.json"), h["muon_smear_random"]
        cset = h.getCorrectionSet("muon_smear_random")
        assert cset is not None
        assert "RandomSmearing" in cset
        with_scalesmearing += "muon_scalesmearing" in h.keys()
    assert with_scalesmearing >= 1
    h = CorrectionlibHandler(era="2024", subera=None, analysis="inc-WZ", nanoAODversion="v15", cvmfs_head="/cvmfs/")
    with open(str(h["muon_smear_random"])) as fin:
        model = schemav2.CorrectionSet.model_validate_json(fin.read())
    assert [c.name for c in model.corrections] == ["RandomSmearing"]
    corr = model.corrections[0]
    assert corr.data.nodetype == "hashprng"
    assert corr.data.distribution == "stdflat"
    assert list(corr.data.inputs) == ["evtNr", "lumiNr", "phi"]
    assert [v.name for v in corr.inputs] == ["evtNr", "lumiNr", "phi"]
    assert [v.type for v in corr.inputs] == ["int", "int", "real"]
    assert str(h["electron_smear_random"]).endswith("qawa/data/egm/electron_smear_random.json"), h["electron_smear_random"]


@needs_mc
def test_mu_kit_u(handler, mumc_events):
    ref = muon_inputs(ak.Array(mumc_events))
    assert len(ref["phi"]) > 10000
    assert np.array_equal(u_prod(handler, ref), u_kit(ref))


@needs_mc
def test_mu_uniform(handler, mumc_events):
    ref = muon_inputs(ak.Array(mumc_events))
    u = u_prod(handler, ref)
    n = len(u)
    assert np.all((u >= 0) & (u < 1))
    assert len(np.unique(u)) >= 0.999 * n
    assert abs(u.mean() - 0.5) < 5 * np.sqrt(1 / (12 * n)), u.mean()
    assert abs(u.var() - 1 / 12) < 5 * np.sqrt((1 / 80 - 1 / 144) / n), u.var()
    assert scipy.stats.kstest(u, "uniform").pvalue > 1e-4
    for name in ("pt", "eta", "phi"):
        assert abs(np.corrcoef(u, ref[name])[0, 1]) < 5 / np.sqrt(n), name
    uj = ak.unflatten(u, ref["counts"])
    uj = uj[ak.num(uj) >= 2]
    n_pairs = len(uj)
    assert n_pairs > 100, n_pairs
    assert abs(np.corrcoef(ak.to_numpy(uj[:, 0]), ak.to_numpy(uj[:, 1]))[0, 1]) < 5 / np.sqrt(n_pairs)


@needs_mc
def test_mu_source(handler, mumc_events):
    ref = muon_inputs(ak.Array(mumc_events))
    ev = ak.Array(mumc_events)
    muon_pt_scare(None, events=ev, is_correction=True, clibhandler=handler)
    pt, ps, scare = f(ev.Muon.pt), f(ev.Muon.pt_scalecorr), f(ev.Muon.pt_scarecorr)
    cset = handler.getCorrectionSet("muon_scalesmearing")
    abs_eta, nL = np.abs(ref["eta"]), ref["nL"]
    m, s, n, a = (cset["cb_params"].evaluate(abs_eta, nL, i) for i in range(4))
    valid = (np.isfinite(s) & (s > 0) & np.isfinite(a) & (a > 0) & np.isfinite(n) & (n > 1.001) & np.isfinite(m))
    cbv = np.zeros(len(ps))
    cbv[valid] = doublecrystalball(a[valid], a[valid], n[valid], n[valid], m[valid], s[valid]).ppf(u_kit(ref)[valid])
    p0, p1, p2 = (cset["poly_params"].evaluate(abs_eta, nL, i) for i in range(3))
    std = np.maximum(p0 + p1 * ps + p2 * ps * ps, 0)
    k_data, k_mc = cset["k_data"].evaluate(abs_eta, "nom"), cset["k_mc"].evaluate(abs_eta, "nom")
    k = np.where(k_mc < k_data, np.sqrt(np.maximum(k_data**2 - k_mc**2, 0)), 0)
    cand = ps * (1 + k * std * cbv)
    cand = np.where((ps < 26) | (ps > 200), ps, cand)
    cand = np.where(np.isnan(cand), ps, cand)
    cand = np.where((cand / ps > 2) | (cand / ps < 0.1) | (cand < 0), ps, cand)
    assert not np.any(np.isnan(pt))
    np.testing.assert_allclose(pt, cand, rtol=1e-12, atol=0)
    assert np.array_equal(scare, pt)
    assert (pt != ps).mean() >= 0.3, (pt != ps).mean()


@needs_mc
def test_mu_chunk(handler):
    a_ev = load(MUMC, stop=3000)
    b_ev = load(MUMC, start=1234, stop=3000)
    offset = int(ak.sum(ak.num(a_ev.Muon[:1234])))
    eta_a, eta_b = f(a_ev.Muon.eta), f(b_ev.Muon.eta)
    seed_a = abs(np.float32(eta_a[0]).view("int32"))
    seed_b = abs(np.float32(eta_b[0]).view("int32"))
    u_a = np.random.default_rng(seed_a).random(len(eta_a))[offset:]
    u_b = np.random.default_rng(seed_b).random(len(eta_b))
    assert seed_a != seed_b
    assert len(u_a) == len(u_b)
    assert not np.array_equal(u_a, u_b)
    a = muon_arrays(corrected_with_systematics(a_ev, handler))
    b = muon_arrays(corrected_with_systematics(b_ev, handler))
    for name in a:
        assert np.array_equal(a[name][offset:], b[name]), name
    smeared = int((b["pt"] != b["pt_scalecorr"]).sum())
    assert smeared >= 100, smeared


@needs_mc
def test_mu_repeat(handler):
    eager = muon_arrays(corrected_with_systematics(load(MUMC, stop=3000, mode="eager"), handler))
    virtual = muon_arrays(corrected_with_systematics(load(MUMC, stop=3000, mode="virtual"), handler))
    for name in eager:
        assert np.array_equal(eager[name], virtual[name]), name


@needs_mc
def test_mu_syst(handler, mumc_events):
    ref = muon_inputs(ak.Array(mumc_events))
    prod = muon_arrays(corrected_with_systematics(ak.Array(mumc_events), handler))
    cset = handler.getCorrectionSet("muon_scalesmearing")
    pt, ps = prod["pt"], prod["pt_scalecorr"]
    abs_eta = np.abs(ref["eta"])
    k_unc, k_nom = cset["k_mc"].evaluate(abs_eta, "stat"), cset["k_mc"].evaluate(abs_eta, "nom")
    has_k = k_nom > 0
    for direction, sign in (("up", 1), ("down", -1)):
        expected = ps * (1 + (k_nom + sign * k_unc) * (pt / ps - 1) / np.where(has_k, k_nom, 1))
        expected = np.where(has_k, expected, pt)
        expected = np.where((expected / ps > 2) | (expected / ps < 0.1) | (expected < 0), ps, expected)
        np.testing.assert_allclose(prod[f"res_{direction}"], expected, rtol=1e-10, atol=0)
    unsmeared = pt == ps
    assert unsmeared.sum() > 100
    assert np.array_equal(prod["res_up"][unsmeared], pt[unsmeared])
    assert np.array_equal(prod["res_down"][unsmeared], pt[unsmeared])
    stat_a = cset["a_mc"].evaluate(ref["eta"], ref["phi"], "stat")
    stat_m = cset["m_mc"].evaluate(ref["eta"], ref["phi"], "stat")
    stat_rho = cset["m_mc"].evaluate(ref["eta"], ref["phi"], "rho_stat")
    q = ref["charge"]
    unc = pt * pt * (stat_m * stat_m / (pt * pt) + stat_a * stat_a + 2 * q * stat_rho * stat_m / pt * stat_a) ** 0.5
    np.testing.assert_allclose(prod["scale_up"], pt + unc, rtol=1e-12, atol=0)
    np.testing.assert_allclose(prod["scale_down"], pt - unc, rtol=1e-12, atol=0)


@needs_data
def test_mu_data(handler, mudata_events):
    ref = muon_inputs(ak.Array(mudata_events))
    ev = ak.Array(mudata_events)
    muon_pt_scare(None, events=ev, is_correction=True, clibhandler=handler)
    pt, ps = f(ev.Muon.pt), f(ev.Muon.pt_scalecorr)
    assert np.array_equal(pt, ps)
    assert "pt_scarecorr" not in ev.Muon.fields
    cset = handler.getCorrectionSet("muon_scalesmearing")
    a_data = cset["a_data"].evaluate(ref["eta"], ref["phi"], "nom")
    m_data = cset["m_data"].evaluate(ref["eta"], ref["phi"], "nom")
    raw = ref["pt"]
    expected = 1 / (m_data / raw + ref["charge"] * a_data)
    expected = np.where((raw < 26) | (raw > 200), raw, expected)
    expected = np.where(np.isnan(expected), raw, expected)
    np.testing.assert_allclose(pt, expected, rtol=1e-12, atol=0)


@needs_mc
def test_mu_empty(handler, mumc_events):
    ev = ak.Array(mumc_events)
    sub = ev[ak.num(ev.Muon) == 0][:50]
    assert len(sub) == 50
    muon_pt_scare(None, events=sub, is_correction=True, clibhandler=handler)
    assert "pt_scarecorr" in sub.Muon.fields


class KitCset:
    def __init__(self, base):
        self.base = base
        self.random_smearing = kit_random_smearing()

    def get(self, name):
        return self.random_smearing if name == "RandomSmearing" else self.base[name]

    def __getitem__(self, name):
        return self.get(name)


@needs_mc
def test_mu_kit(handler, mumc_events):
    kit_path = os.environ.get("MUONSCAREKIT_PATH")
    if not kit_path:
        pytest.skip("MUONSCAREKIT_PATH not set: point it at a local muonscarekit scripts/MuonScaRe.py (commit d3933320 or later) to check production against the official kit")
    assert os.path.isfile(kit_path), kit_path
    spec = importlib.util.spec_from_file_location("muonscarekit_reference", kit_path)
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    assert "RandomSmearing" in inspect.getsource(kit.get_rndm)

    cset = handler.getCorrectionSet("muon_scalesmearing")
    ref = ak.Array(mumc_events)
    mu = ref.Muon

    def U(x):
        return ak.unflatten(x, ak.num(mu))

    flat = muon_inputs(ref)
    nL = flat["nL"].astype(float)
    ks = kit.pt_scale(0, flat["pt"], flat["eta"], flat["phi"], flat["charge"], cset, nested=False)
    kr = kit.pt_resol(U(ks), mu.eta, mu.phi, U(nL), ref.event, ref.luminosityBlock, KitCset(cset), nested=True)
    kit_out = {
        "pt": f(kr),
        "scale_up": f(kit.pt_scale_var(kr, mu.eta, mu.phi, mu.charge, "up", cset, nested=True)),
        "scale_down": f(kit.pt_scale_var(kr, mu.eta, mu.phi, mu.charge, "dn", cset, nested=True)),
        "res_up": f(kit.pt_resol_var(U(ks), kr, mu.eta, "up", cset, nested=True)),
        "res_down": f(kit.pt_resol_var(U(ks), kr, mu.eta, "dn", cset, nested=True)),
    }
    prod = muon_arrays(corrected_with_systematics(ak.Array(mumc_events), handler))

    for name, values in kit_out.items():
        assert not np.any(np.isnan(values)), name
        assert not np.any(np.isnan(prod[name])), name
    delta = {name: np.abs(prod[name] - kit_out[name]) for name in kit_out}
    nominal_max = float(delta["pt"].max())
    nominal_frac = float((delta["pt"] > 1e-6).mean())
    print(f"nominal max|dpt| = {nominal_max:.3e} GeV, fraction > 1e-6 GeV = {nominal_frac:.5f}")
    for name in ("scale_up", "scale_down", "res_up", "res_down"):
        print(f"{name} max|dpt| = {float(delta[name].max()):.3e} GeV")
    assert nominal_max <= 1e-5, f"nominal max|dpt| = {nominal_max:.3e} GeV, fraction > 1e-6 = {nominal_frac:.5f}, fraction > 0 = {float((delta['pt'] > 0).mean()):.5f}"
    assert nominal_frac <= 1e-3, f"fraction > 1e-6 GeV = {nominal_frac:.5f}"
    for name in ("scale_up", "scale_down", "res_up", "res_down"):
        assert float(delta[name].max()) <= 1e-4, f"{name} max|dpt| = {float(delta[name].max()):.3e} GeV"
    smeared = float((kit_out["pt"] != ks).mean())
    assert smeared >= 0.3, smeared
