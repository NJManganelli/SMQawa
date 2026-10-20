from __future__ import annotations

import functools
import gzip
import inspect
import json
from pathlib import Path

import numpy as np
import awkward as ak
import correctionlib
import pytest
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import (
    FactorizedJetCorrector, CorrectedJetsFactory, CorrectedMETFactory, CorrectionLibJEC, CorrectionLibJECStack,
    CorrectionLibJER, CorrectionLibJERSF,
)

import qawa
import qawa.process.wztau2lnu_inclusive as wzproc
from qawa import uproot_float16_patch
from qawa.flashsim_shim import flashsim_schema
from qawa.jme import (
    JMEUncertainty, clib_name_map, add_jme_variables, _prepare_corrt1_jets, _jersmear_jets, _jersmear_corrt1_jets,
    _jer_rho_max, jetmet_mitigations, abs_eta_in_regions, forward_jet_pt_mask, _jersmear_factors, _ResidualPtFloorJEC, _nearest_gen_pt,
    JETMET_RUN3_MITIGATIONS,
)
from qawa.jsoncorrections import CorrectionlibHandler
from qawa.met_shim import prepare_met_for_factory

MC_FILE = "/Volumes/WDMac/SMQawaTestFiles/store/mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/test.root"
DATA_C_FILE = "/Volumes/WDMac/SMQawaTestFiles/store/data/Run2024C/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root"
DATA_I_FILE = "/Volumes/WDMac/SMQawaTestFiles/store/data/Run2024I/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root"
FLASHSIM_FILE = "/Volumes/WDMac/eft-studies/flashsim/fullsim/WZto3LNu_powheg_pythia8_RunIII2024Summer24_flashsim_Student_000.root"
FILES = {"mc": MC_FILE, "dataC": DATA_C_FILE, "dataI": DATA_I_FILE}

TXT_DIR = Path(qawa.__file__).resolve().parent / "data" / "jme" / "2024"

JEC_TAG = "Summer24Prompt24_V2"
JER_TAG = "Summer23BPixPrompt23_RunD_JRV1"
TXT_MC_TAG = "Summer24Prompt24_V3_MC"
TXT_JER_TAG = "Summer24Prompt24_JRV1_MC"
JET_TYPE = "AK4PFPuppi"
VARIATIONS = ("nom", "up", "down")

HE_IND = ((2.5, 3.0),)
FLOOR_ETA_IND = ((2.0, 2.5),)
FLOOR_PT_IND = 30.0

NIBS = (
    ("Cnib1", 379412, 380253),
    ("Dnib1", 380253, 380948),
    ("Enib1", 380948, 381944),
    ("Fnib1", 381944, 382298),
    ("Fnib2", 382298, 383247),
    ("Fnib3", 383247, 383780),
    ("Gnib1", 383780, 384933),
    ("Gnib2", 384933, 385814),
    ("Hnib1", 385814, 386409),
    ("Inib1", 386409, 387121),
)

JES_SOURCES = (
    "Total", "Absolute", "Absolute_2024", "BBEC1", "BBEC1_2024", "EC2", "EC2_2024",
    "FlavorQCD", "HF", "HF_2024", "RelativeBal", "RelativeSample_2024",
)

PROCESSOR_SHIFT_JES = (
    "JES_Total", "JES_Absolute", "JES_BBEC1", "JES_EC2", "JES_FlavorQCD", "JES_HF", "JES_RelativeBal",
    "JES_BBEC1_2024", "JES_Absolute_2024", "JES_EC2_2024", "JES_HF_2024", "JES_RelativeSample_2024",
)

CORRT1_FIELDS = {"rawPt", "eta", "phi", "area", "EmEF", "muonSubtrFactor", "muonSubtrDeltaPhi"}


def _require_txt():
    if not TXT_DIR.is_dir():
        pytest.skip(f"[txt] {TXT_DIR} missing")


def _load(path, entry_stop, schema=NanoAODSchema, metadata=None, entry_start=None):
    return NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=schema, entry_start=entry_start, entry_stop=entry_stop, metadata=metadata or {},
    ).events()


@functools.lru_cache(maxsize=None)
def _events(key):
    return _load(FILES[key], 2000)


@functools.lru_cache(maxsize=None)
def _handler():
    return CorrectionlibHandler(era="2024", analysis="inc-WZ", nanoAODversion="v15")


@functools.lru_cache(maxsize=None)
def _cset():
    return correctionlib.CorrectionSet.from_file(str(_handler().getPath("jet_jerc")))


@functools.lru_cache(maxsize=None)
def _smear_corr():
    return correctionlib.CorrectionSet.from_file(str(_handler().getPath("jer_smear")))["JERSmear"]


@functools.lru_cache(maxsize=None)
def _rho_max_ind():
    with gzip.open(str(_handler().getPath("jet_jerc")), "rt") as f:
        payload = json.load(f)
    data = [c for c in payload["corrections"] if c["name"] == f"{JER_TAG}_MC_PtResolution_{JET_TYPE}"][0]["data"]
    assert data["input"] == "JetEta", f"unexpected PtResolution layout: {data['input']}"
    assert all(n["input"] == "Rho" for n in data["content"]), "PtResolution eta bins are not Rho binnings"
    edges = {n["edges"][-1] for n in data["content"]}
    assert edges == {52.05}, f"upper Rho edges {edges}"
    return np.nextafter(52.05, -np.inf)


@functools.lru_cache(maxsize=None)
def _clib(is_mc, doJER):
    return JMEUncertainty(JEC_TAG, JER_TAG, era="2024", is_mc=is_mc, doJER=doJER, clibhandler=_handler(), version="v15")


@functools.lru_cache(maxsize=None)
def _txt(tag, is_mc):
    return JMEUncertainty(tag, TXT_JER_TAG, era="2024", is_mc=is_mc, clibhandler=None, version="v15")


def _txt_data_tag(nib):
    return f"Summer24Prompt24_Run{nib}_V3_DATA"


@functools.lru_cache(maxsize=None)
def _txt_type1_factory(tag, is_mc):
    levels = ["L1FastJet", "L2Relative", "L3Absolute"] + ([] if is_mc else ["L2L3Residual"])
    names = [f"{tag}_{lvl}_{JET_TYPE}" for lvl in levels]
    paths = [TXT_DIR / f"{n}.jec.txt" for n in names]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(str(p))
    ext = extractor()
    ext.add_weight_sets([f"* * {p}" for p in paths])
    ext.finalize()
    ev = ext.make_evaluator()
    jec_L123 = FactorizedJetCorrector(**{n: ev[n] for n in names})
    jec_L1 = FactorizedJetCorrector(**{names[0]: ev[names[0]]})
    return CorrectedMETFactory(clib_name_map, jec_L1L2L3=jec_L123, jec_L1=jec_L1)


def _rho(events):
    return events.Rho.fixedGridRhoFastjetAll


def _flat(x):
    return np.asarray(ak.to_numpy(ak.flatten(x, axis=None)), dtype=np.float64)


def _flat_i64(x):
    return np.asarray(ak.to_numpy(ak.flatten(x, axis=None)), dtype=np.int64)


def _np(x):
    return np.asarray(ak.to_numpy(x), dtype=np.float64)


def _const_run(events, run):
    return ak.full_like(events.run, run)


def _with_run(coll, run):
    return ak.with_field(coll, ak.broadcast_arrays(run, coll.pt if "pt" in coll.fields else coll.rawPt)[0], "run")


def _prepare_corrt1(events, run=None):
    ct1 = events.CorrT1METJet
    ct1 = ak.with_field(ct1, ak.broadcast_arrays(_rho(events), ct1.rawPt)[0], "rhoFixedGridFastJetAll")
    if run is not None:
        ct1 = ak.with_field(ct1, ak.broadcast_arrays(run, ct1.rawPt)[0], "run")
    return ct1


def _jet_raw(jets):
    return (1 - jets.rawFactor) * jets.pt


def _eval_named(corr, values):
    return corr.evaluate(*[values[i.name] for i in corr.inputs])


def _regions_ind(eta, regions):
    mask = np.zeros(np.shape(eta), dtype=bool) if isinstance(eta, np.ndarray) else ak.zeros_like(eta, dtype=bool)
    for lo, hi in regions:
        mask = mask | ((np.abs(eta) >= lo) & (np.abs(eta) < hi))
    return mask


def _jec_values(coll, rho, run=None, corrt1=False):
    raw = coll.rawPt if corrt1 else _jet_raw(coll)
    values = {
        "JetA": coll.area,
        "JetEta": coll.eta,
        "JetPt": raw,
        "Rho": ak.broadcast_arrays(rho, raw)[0],
        "JetPhi": coll.phi,
    }
    if run is not None:
        values["run"] = ak.broadcast_arrays(run, raw)[0]
    return raw, values


def _floor_ind(coll, rho, run, corrt1=False):
    cset = _cset()
    raw, values = _jec_values(coll, rho, run=run, corrt1=corrt1)
    f_mc = None
    for lvl in ("L1FastJet", "L2Relative", "L3Absolute"):
        f = _eval_named(cset[f"{JEC_TAG}_DATA_{lvl}_{JET_TYPE}"], values)
        f_mc = f if f_mc is None else f_mc * f
        values["JetPt"] = values["JetPt"] * f
    pt_mc = values["JetPt"]
    residual = cset[f"{JEC_TAG}_DATA_L2L3Residual_{JET_TYPE}"]
    a = _regions_ind(coll.eta, FLOOR_ETA_IND) & (pt_mc < FLOOR_PT_IND)
    res_pt = _eval_named(residual, values)
    res_30 = _eval_named(residual, {**values, "JetPt": ak.full_like(pt_mc, FLOOR_PT_IND)})
    return {
        "f_mc": f_mc, "pt_mc": pt_mc, "A": a, "res_pt": res_pt, "res_30": res_30,
        "f_floor": f_mc * ak.where(a, res_30, res_pt), "f_chain": f_mc * res_pt,
    }


def _clib_factors(coll, rho, is_mc, run=None, corrt1=False, floor=True):
    cset = _cset()
    data_type = "MC" if is_mc else "DATA"
    _, values = _jec_values(coll, rho, run=run, corrt1=corrt1)
    if not is_mc and floor:
        f123 = _floor_ind(coll, rho, run, corrt1=corrt1)["f_floor"]
    else:
        f123 = _eval_named(cset.compound[f"{JEC_TAG}_{data_type}_L1L2L3Res_{JET_TYPE}"], values)
    f1 = _eval_named(cset[f"{JEC_TAG}_{data_type}_L1FastJet_{JET_TYPE}"], values)
    return f123, f1


def _type1_factors(events, is_mc, run=None, with_corrt1=True, floor=True, corrt1_floor=None):
    rho = _rho(events)
    f123, f1 = _clib_factors(events.Jet, rho, is_mc, run=run, floor=floor)
    f = {"jet_L123": f123, "jet_L1": f1}
    if with_corrt1:
        c123, c1 = _clib_factors(events.CorrT1METJet, rho, is_mc, run=run, corrt1=True,
                                 floor=floor if corrt1_floor is None else corrt1_floor)
        f["ct1_L123"] = c123
        f["ct1_L1"] = c1
    return f


def _no_floor_events(events, run):
    rho = _rho(events)
    a_jet = _floor_ind(events.Jet, rho, run)["A"]
    a_ct1 = _floor_ind(events.CorrT1METJet, rho, run, corrt1=True)["A"]
    return ~(_np(ak.any(a_jet, axis=1)).astype(bool) | _np(ak.any(a_ct1, axis=1)).astype(bool))


def _pt_jec_ind(events, is_mc, run=None):
    f123, _ = _clib_factors(events.Jet, _rho(events), is_mc, run=run)
    return _jet_raw(events.Jet) * f123


def _type1_sum(raw, msf, dphi, phi, eta, em, f123, f1, scale, jes, sign):
    f123 = ak.values_astype(f123, np.float32)
    f1 = ak.values_astype(f1, np.float32)
    nomu = ak.values_astype(raw * (1.0 - msf), np.float64)
    mu = ak.values_astype(raw * msf, np.float64)
    phip = dphi + phi
    sel = (nomu * f123 > 15.0) & (np.abs(eta) < 5.2) & (em < 0.9)
    pt123 = nomu * f123 + mu
    pt1 = nomu * f1 + mu
    if scale is None and jes is None:
        diff = nomu * f123 - nomu * f1
    else:
        s = 1.0 if scale is None else scale
        delta = 0.0 if jes is None else ak.values_astype(jes.evaluate(eta, pt123), np.float32)
        diff = pt123 * (s + sign * delta) - pt1
    diff = ak.where(sel, diff, 0.0)
    return ak.sum(diff * np.cos(phip), axis=1), ak.sum(diff * np.sin(phip), axis=1)


def reference_type1(raw_met, jets, corrt1, f, jet_scale=None, jes=None, sign=0, no_muon=False, corrt1_scale=None):
    msf = ak.zeros_like(jets.muonSubtrFactor) if no_muon else jets.muonSubtrFactor
    dphi = ak.zeros_like(jets.muonSubtrDeltaPhi) if no_muon else jets.muonSubtrDeltaPhi
    dpx, dpy = _type1_sum(
        _jet_raw(jets), msf, dphi, jets.phi, jets.eta, jets.chEmEF + jets.neEmEF,
        f["jet_L123"], f["jet_L1"], jet_scale, jes, sign,
    )
    if corrt1 is not None:
        cpx, cpy = _type1_sum(
            corrt1.rawPt, corrt1.muonSubtrFactor, corrt1.muonSubtrDeltaPhi, corrt1.phi, corrt1.eta, corrt1.EmEF,
            f["ct1_L123"], f["ct1_L1"], corrt1_scale, jes, sign,
        )
        dpx = dpx + cpx
        dpy = dpy + cpy
    x = _np(raw_met.pt) * np.cos(_np(raw_met.phi)) - _np(dpx)
    y = _np(raw_met.pt) * np.sin(_np(raw_met.phi)) - _np(dpy)
    return np.hypot(x, y), np.arctan2(y, x)


def _dphi(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _met_mismatch(pt, phi, ref_pt, ref_phi, rtol=1e-4, atol=1e-3):
    pt, phi = _np(pt), _np(phi)
    bad_pt = np.abs(pt - ref_pt) > atol + rtol * np.abs(ref_pt)
    bad_phi = _dphi(phi, ref_phi) * ref_pt > atol + rtol * np.abs(ref_pt)
    return bad_pt | bad_phi


def _assert_met_close(met, ref, label, rtol=1e-4, atol=1e-3):
    bad = _met_mismatch(met.pt, met.phi, ref[0], ref[1], rtol=rtol, atol=atol)
    dpt = np.abs(_np(met.pt) - ref[0])
    assert not bad.any(), f"{label}: {bad.sum()}/{len(bad)} events outside tolerance, max |dpt| = {dpt.max():.3g} GeV"
    return float(dpt.max())


def _assert_jets_close(a, b, label, rtol=1e-5):
    x, y = _flat(a), _flat(b)
    assert len(x) == len(y), f"{label}: length mismatch {len(x)} vs {len(y)}"
    ok = np.isclose(x, y, rtol=rtol, atol=0)
    r = np.abs(x / np.where(y != 0, y, 1) - 1)
    assert ok.all(), f"{label}: {(~ok).sum()}/{len(ok)} jets differ, max |ratio-1| = {r.max():.3g}"
    return float(r.max())


def _assert_close(x, y, label, rtol=1e-12):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert len(x) == len(y), f"{label}: length mismatch {len(x)} vs {len(y)}"
    ok = np.isclose(x, y, rtol=rtol, atol=0)
    assert ok.all(), f"{label}: {(~ok).sum()}/{len(ok)} differ, max |x-y| = {np.abs(x - y).max():.3g}"


def _frac_differ(x, y, tol=1e-6):
    d = np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)) > tol
    return float(d.mean()) if len(d) else 0.0, len(d)


def _met_inputs(events):
    return prepare_met_for_factory(events.PuppiMET), events.RawPuppiMET


def _clib_met(jm, events, jets, run=None, corrt1=True):
    met_in, raw = _met_inputs(events)
    return jm.corrected_met(met_in, jets, _rho(events), {}, raw_met=raw,
                            corrt1_jets=events.CorrT1METJet if corrt1 else None, run=run,
                            event_id=events.event, gen_jets=events.GenJet if "GenJet" in events.fields else None)


def _txt_type1_met(tag, is_mc, events):
    jets = _txt(tag, is_mc).corrected_jets_L123_noJER(events.Jet, _rho(events), {})
    met_in, raw = _met_inputs(events)
    return _txt_type1_factory(tag, is_mc).build(met_in, jets, in_RawMET=raw, in_CorrT1METJets=_prepare_corrt1(events))


def _quantiles(x, qs=(1, 5, 16, 50, 84, 95, 99)):
    v = np.percentile(x, qs)
    return " ".join(f"{q}%={val:.4g}" for q, val in zip(qs, v))


def _gen_pt_dr(jets, max_dr):
    dr = ak.fill_none(jets.delta_r(jets.matched_gen), 999.0)
    return ak.values_astype(ak.where(dr < max_dr, ak.fill_none(jets.matched_gen.pt, 0), 0), np.float32), dr


def _guard(s):
    return np.where((s > 0) & np.isfinite(s), s, 1.0)


def _jersmear_ind(pt_jec, eta, genpt, rho, eid, sigma, sf):
    return _eval_named(_smear_corr(), {
        "JetPt": pt_jec, "JetEta": eta, "GenPt": genpt, "Rho": rho, "EventID": eid, "JER": sigma, "JERSF": sf,
    })


def _smear_inputs(pt_jec, eta, rho, eid, pt_gen, sigma, window):
    cset = _cset()
    sfc = cset[f"{JER_TAG}_MC_ScaleFactor_{JET_TYPE}"]
    sf = {v: sfc.evaluate(eta, pt_jec, v) for v in VARIATIONS}
    matched = (pt_gen > 0) & (np.abs((pt_jec - pt_gen) / pt_jec) < 3 * window)
    genpt = np.where(matched, pt_gen, -1.0)
    s = {v: _jersmear_ind(pt_jec, eta, genpt, rho, eid, sigma, sf[v]) for v in VARIATIONS}
    guard = {v: _guard(s[v]) for v in VARIATIONS}
    g_sets = {v: ~((s[v] > 0) & np.isfinite(s[v])) for v in VARIATIONS}
    z = _jersmear_ind(pt_jec, eta, np.full_like(pt_jec, -1.0), rho, eid, np.ones_like(pt_jec), np.full_like(pt_jec, np.sqrt(2.0))) - 1
    he = _regions_ind(eta, HE_IND)
    final = {v: np.where(he & ~matched, 1.0, guard[v]) for v in VARIATIONS}
    return {
        "pt_jec": pt_jec, "eta": eta, "rho": rho, "eid": eid, "pt_gen": pt_gen, "sigma": sigma, "sf": sf,
        "matched": matched, "genpt": genpt, "s": s, "guard": guard, "G": g_sets, "z": z, "he": he, "final": final,
    }


def _jer_inputs(events):
    cset = _cset()
    jets = events.Jet
    pt_jec_j = _pt_jec_ind(events, True)
    pt_jec = _flat(pt_jec_j)
    eta = _flat(jets.eta)
    rho = _flat(ak.broadcast_arrays(_rho(events), pt_jec_j)[0])
    eid = _flat_i64(ak.broadcast_arrays(ak.values_astype(events.event, np.int64), pt_jec_j)[0])
    sigma = cset[f"{JER_TAG}_MC_PtResolution_{JET_TYPE}"].evaluate(eta, pt_jec, rho)
    pt_gen = _flat(_gen_pt_dr(jets, 0.2)[0])
    out = _smear_inputs(pt_jec, eta, rho, eid, pt_gen, sigma, sigma)
    out["counts"] = ak.num(jets)
    return out


def _ct1_jer_inputs(events):
    cset = _cset()
    ct1 = events.CorrT1METJet
    f123, _ = _clib_factors(ct1, _rho(events), True, corrt1=True)
    pt_jec_j = ct1.rawPt * f123
    pt_jec = _flat(pt_jec_j)
    eta = _flat(ct1.eta)
    rho = _flat(ak.broadcast_arrays(_rho(events), pt_jec_j)[0])
    eid = _flat_i64(ak.broadcast_arrays(ak.values_astype(events.event, np.int64), pt_jec_j)[0])
    v4 = ak.zip({"pt": ct1.rawPt, "eta": ct1.eta, "phi": ct1.phi, "mass": ak.zeros_like(ct1.rawPt)},
                with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    pt_gen = _flat(ak.fill_none(v4.nearest(events.GenJet, threshold=0.2).pt, 0))
    res = cset[f"{JER_TAG}_MC_PtResolution_{JET_TYPE}"]
    sigma = res.evaluate(eta, pt_jec, rho)
    window = res.evaluate(eta, pt_jec, np.minimum(rho, _rho_max_ind()))
    out = _smear_inputs(pt_jec, eta, rho, eid, pt_gen, sigma, window)
    out["counts"] = ak.num(ct1)
    out["v4"] = v4
    out["window"] = window
    return out


def _unflat(x, counts):
    return ak.unflatten(x, counts)


def _coffea_rng_factory(jm):
    cset = _cset()
    stack = CorrectionLibJECStack(
        jec=jm.jec_stack_L123_noJER.jec, junc=jm.jec_stack_L123_noJER.junc,
        jer=CorrectionLibJER(cset[f"{JER_TAG}_MC_PtResolution_{JET_TYPE}"]),
        jersf=CorrectionLibJERSF(cset[f"{JER_TAG}_MC_ScaleFactor_{JET_TYPE}"]),
    )
    return CorrectedJetsFactory(clib_name_map, stack)


def _coffea_rng_jets(jm, events):
    return _coffea_rng_factory(jm).build(add_jme_variables(events.Jet, _rho(events), max_gen_dr=0.2))


def _smeared(events, jm=None):
    jm = jm or _clib(True, True)
    return jm.corrected_jets_L123_JER(events.Jet, _rho(events), {}, event_id=events.event)


def one(x):
    return ak.unflatten(x, [len(x)])


class _stub_btag:
    def tagged_jets(self, jets, wp):
        return {w: ak.zeros_like(jets.pt, dtype=bool) for w in wp}


def test_nib_table_matches_payload():
    with gzip.open(str(_handler().getPath("jet_jerc")), "rt") as f:
        payload = json.load(f)
    corr = [c for c in payload["corrections"] if c["name"] == f"{JEC_TAG}_DATA_L2L3Residual_{JET_TYPE}"]
    assert len(corr) == 1, "data residual correction not found in payload"
    corr = corr[0]
    node = corr["data"]
    while node.get("nodetype") != "binning" or node.get("input") != "run":
        assert "content" in node or "default" in node, f"no run binning found in {corr['name']}"
        node = node["content"][0] if isinstance(node.get("content"), list) else node["content"]
    edges = [int(e) for e in node["edges"]]
    expected = [n[1] for n in NIBS] + [NIBS[-1][2]]
    assert edges == expected, f"run edges {edges} != NIBS {expected}"
    assert [n[2] for n in NIBS[:-1]] == [n[1] for n in NIBS[1:]], "NIBS not contiguous"
    desc = corr.get("description", "")
    pos = [desc.find(n[0]) for n in NIBS]
    assert all(p >= 0 for p in pos), f"nib names missing from description: {desc}"
    assert pos == sorted(pos), f"nib names out of order in description: {desc}"


def test_mc_jec_matches_txt():
    _require_txt()
    ev = _events("mc")
    clib = _clib(True, False).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    txt = _txt(TXT_MC_TAG, True).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    for fld in ("pt", "mass", "pt_jec"):
        m = _assert_jets_close(clib[fld], txt[fld], f"MC {fld}")
        print(f"  MC {fld}: max |clib/txt-1| = {m:.3g}")


def test_mc_jes_sources():
    ev = _events("mc")
    jets = _clib(True, False).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    jes_fields = {f for f in jets.fields if f.startswith("JES_")}
    assert jes_fields == set(PROCESSOR_SHIFT_JES), f"JES fields {sorted(jes_fields)} != processor shift list"
    cset = _cset()
    pt_jec = _pt_jec_ind(ev, True)
    for name in JES_SOURCES:
        delta = cset[f"{JEC_TAG}_MC_Regrouped_{name}_{JET_TYPE}"].evaluate(ev.Jet.eta, pt_jec)
        _assert_jets_close(jets[f"JES_{name}"].up.pt, pt_jec * (1 + delta), f"JES_{name} up")
        _assert_jets_close(jets[f"JES_{name}"].down.pt, pt_jec * (1 - delta), f"JES_{name} down")


def test_mc_jes_sources_match_txt():
    _require_txt()
    ev = _events("mc")
    jets = _clib(True, False).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    txt = _txt(TXT_MC_TAG, True).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    worst = 0.0
    for name in JES_SOURCES:
        for d in ("up", "down"):
            worst = max(worst, _assert_jets_close(jets[f"JES_{name}"][d].pt, txt[f"JES_{name}"][d].pt, f"txt JES_{name} {d}"))
    print(f"  max |clib/txt-1| over 12 sources x up/down = {worst:.3g}")


def test_data_jec_matches_txt_in_nib1_runs():
    _require_txt()
    for key, nib in (("dataC", "Cnib1"), ("dataI", "Inib1")):
        ev = _events(key)
        lo, hi = [(n[1], n[2]) for n in NIBS if n[0] == nib][0]
        runs = _np(ev.run)
        assert ((runs >= lo) & (runs < hi)).all(), f"{key}: runs {runs.min()}-{runs.max()} outside {nib} [{lo},{hi})"
        clib = _clib(False, False).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {}, run=ev.run)
        txt = _txt(_txt_data_tag(nib), False).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
        keep = ~_flat(_floor_ind(ev.Jet, _rho(ev), ev.run)["A"]).astype(bool)
        print(f"  {key}: ~A jets {int(keep.sum())} of {len(keep)}")
        assert keep.sum() >= 2000, f"{key}: only {keep.sum()} jets outside the residual-floor region"
        for fld in ("pt", "mass", "pt_jec"):
            m = _assert_jets_close(_flat(clib[fld])[keep], _flat(txt[fld])[keep], f"{key} {fld}")
            print(f"  {key} vs {nib} {fld} (~A): max |clib/txt-1| = {m:.3g}")
        assert not any(f.startswith("JES_") for f in clib.fields), f"{key}: data jets carry JES fields"
        assert "JER" not in clib.fields, f"{key}: data jets carry JER"


def test_data_jec_run_dependence():
    _require_txt()
    ev = _events("dataC")
    jm = _clib(False, False)
    for nib, lo, hi in NIBS:
        mid = (lo + hi) // 2
        run = _const_run(ev, mid)
        keep = ~_flat(_floor_ind(ev.Jet, _rho(ev), run)["A"]).astype(bool)
        jets_in = _with_run(ev.Jet, run)
        clib = jm.corrected_jets_L123_noJER(jets_in, _rho(ev), {}, run=run)
        same = _txt(_txt_data_tag(nib), False).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
        _assert_jets_close(_flat(clib.pt)[keep], _flat(same.pt)[keep], f"{nib} clib vs same-nib txt")
        nib1 = nib[0] + "nib1"
        old = _txt(_txt_data_tag(nib1), False).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
        r = np.abs(_flat(clib.pt)[keep] / _flat(old.pt)[keep] - 1)
        if nib == nib1:
            _assert_jets_close(_flat(clib.pt)[keep], _flat(old.pt)[keep], f"{nib} clib vs nib1 txt")
        else:
            frac = float(np.mean(r > 1e-3))
            print(f"  {nib} vs {nib1} (~A, N={len(r)}): fraction(|ratio-1|>1e-3) = {frac:.4f}, median = {np.median(r):.4g}, max = {r.max():.4g}")
            assert frac > 0.5, f"{nib}: only {frac:.3f} of jets differ from the {nib1} txt by > 1e-3"


def test_data_run_outside_payload_raises():
    ev = _events("dataC")
    jm = _clib(False, False)
    good = jm.corrected_jets_L123_noJER(_with_run(ev.Jet, _const_run(ev, 380000)), _rho(ev), {})
    _flat(good.pt)
    with pytest.raises(RuntimeError, match="bounds") as info:
        bad = jm.corrected_jets_L123_noJER(_with_run(ev.Jet, _const_run(ev, 379411)), _rho(ev), {})
        _flat(bad.pt)
    print(f"  run 379411 raised {info.type.__name__}: {str(info.value)[:120]}")


def test_type1_met_mc_matches_txt_type1():
    _require_txt()
    ev = _events("mc")
    jets = _clib(True, False).corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    met = _clib_met(_clib(True, False), ev, jets)
    ref = _txt_type1_met(TXT_MC_TAG, True, ev)
    m = _assert_met_close(met, (_np(ref.pt), _np(ref.phi)), "MC nominal")
    for d in ("up", "down"):
        r = ref.MET_UnclusteredEnergy[d]
        _assert_met_close(met.MET_UnclusteredEnergy[d], (_np(r.pt), _np(r.phi)), f"MC unclustered {d}")
    print(f"  MC nominal max |dpt| = {m:.3g} GeV")


def test_type1_met_data_matches_txt_type1_in_nib1_runs():
    _require_txt()
    for key, nib in (("dataC", "Cnib1"), ("dataI", "Inib1")):
        ev = _events(key)
        jm = _clib(False, False)
        jets = jm.corrected_jets_L123_noJER(ev.Jet, _rho(ev), {}, run=ev.run)
        met = _clib_met(jm, ev, jets, run=ev.run)
        ref = _txt_type1_met(_txt_data_tag(nib), False, ev)
        keep = _no_floor_events(ev, ev.run)
        print(f"  {key}: no-floor events {int(keep.sum())} of {len(keep)}")
        assert keep.sum() >= 700, f"{key}: only {keep.sum()} events without residual-floor jets"
        m = _assert_met_close(met[keep], (_np(ref.pt)[keep], _np(ref.phi)[keep]), f"{key} nominal")
        for d in ("up", "down"):
            r = ref.MET_UnclusteredEnergy[d]
            _assert_met_close(met.MET_UnclusteredEnergy[d][keep], (_np(r.pt)[keep], _np(r.phi)[keep]), f"{key} unclustered {d}")
        print(f"  {key} nominal max |dpt| = {m:.3g} GeV")


def test_type1_met_data_run_dependence():
    _require_txt()
    ev = _events("dataC")
    lo, hi = [(n[1], n[2]) for n in NIBS if n[0] == "Fnib2"][0]
    run = _const_run(ev, (lo + hi) // 2)
    jm = _clib(False, False)
    jets = jm.corrected_jets_L123_noJER(_with_run(ev.Jet, run), _rho(ev), {}, run=run)
    met = _clib_met(jm, ev, jets, run=run)
    keep = _no_floor_events(ev, run)
    print(f"  no-floor events {int(keep.sum())} of {len(keep)}")
    assert keep.sum() >= 700, f"only {keep.sum()} events without residual-floor jets"
    same = _txt_type1_met(_txt_data_tag("Fnib2"), False, ev)
    _assert_met_close(met[keep], (_np(same.pt)[keep], _np(same.phi)[keep]), "Fnib2 clib vs Fnib2 txt")
    old = _txt_type1_met(_txt_data_tag("Fnib1"), False, ev)
    d = np.abs(_np(met.pt) - _np(old.pt))[keep]
    frac = float(np.mean(d > 0.01))
    print(f"  Fnib2 clib vs Fnib1 txt |dpt| GeV: {_quantiles(d)}; fraction(>0.01 GeV) = {frac:.4f}")
    assert frac > 0.2, f"only {frac:.3f} of events differ from the Fnib1 txt Type-1 by > 0.01 GeV"


def test_type1_met_matches_reference_recipe():
    for key, is_mc in (("mc", True), ("dataC", False)):
        ev = _events(key)
        run = None if is_mc else ev.run
        jm = _clib(is_mc, False)
        jets = jm.corrected_jets_L123_noJER(ev.Jet, _rho(ev), {}, run=run)
        met = _clib_met(jm, ev, jets, run=run)
        ref = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, _type1_factors(ev, is_mc, run=run))
        m = _assert_met_close(met, ref, f"{key} nominal vs reference")
        print(f"  {key}: max |dpt| vs reference = {m:.3g} GeV")
    ev = _events("mc")
    f = _type1_factors(ev, True, with_corrt1=False)
    f_nano = {"jet_L123": 1.0 / (1.0 - ev.Jet.rawFactor), "jet_L1": f["jet_L1"]}
    puppi = _np(ev.PuppiMET.pt)
    selfcheck = reference_type1(ev.RawPuppiMET, ev.Jet, None, f_nano)
    stat = float(np.percentile(np.abs(selfcheck[0] / puppi - 1), 95))
    nomu = reference_type1(ev.RawPuppiMET, ev.Jet, None, f_nano, no_muon=True)
    stat_nomu = float(np.percentile(np.abs(nomu[0] / puppi - 1), 95))
    print(f"  self-check 95% |rel| vs PuppiMET = {stat:.4g} (threshold < 1e-2); no-muon-subtraction = {stat_nomu:.4g} (threshold > 5e-2)")
    assert stat < 1e-2, f"reference recipe with NanoAOD factors does not reproduce PuppiMET: {stat:.4g}"
    assert stat_nomu > 5e-2, f"negative control (no muon subtraction) too close to PuppiMET: {stat_nomu:.4g}"


def test_type1_met_jes_propagation():
    ev = _events("mc")
    jm = _clib(True, False)
    jets = jm.corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    met = _clib_met(jm, ev, jets)
    f = _type1_factors(ev, True)
    src = _cset()[f"{JEC_TAG}_MC_Regrouped_Total_{JET_TYPE}"]
    for sign, d in ((1, "up"), (-1, "down")):
        ref = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, f, jes=src, sign=sign)
        m = _assert_met_close(met.JES_Total[d], ref, f"JES_Total {d}")
        print(f"  JES_Total {d}: max |dpt| = {m:.3g} GeV")
    nominal = _np(met.pt)
    for name in list(PROCESSOR_SHIFT_JES) + ["MET_UnclusteredEnergy"]:
        assert name in met.fields, f"{name} missing from MET"
        assert (np.abs(_np(met[name].up.pt) - nominal) > 0).any(), f"{name}.up identical to nominal everywhere"


def test_type1_met_jes_propagation_smeared():
    ev = _events("mc")
    jm = _clib(True, True)
    jets = _smeared(ev, jm)
    met = _clib_met(jm, ev, jets)
    f = _type1_factors(ev, True)
    c = _ct1_jer_inputs(ev)
    ct1_s = _unflat(c["final"]["nom"], c["counts"])
    s = jets.jet_energy_resolution_correction
    cset = _cset()
    worst = 0.0
    for name in JES_SOURCES:
        src = cset[f"{JEC_TAG}_MC_Regrouped_{name}_{JET_TYPE}"]
        for sign, d in ((1, "up"), (-1, "down")):
            ref = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, f, jet_scale=s, jes=src, sign=sign, corrt1_scale=ct1_s)
            worst = max(worst, _assert_met_close(met[f"JES_{name}"][d], ref, f"smeared JES_{name} {d}"))
    print(f"  max |dpt| over 12 sources x up/down = {worst:.3g} GeV")
    src = cset[f"{JEC_TAG}_MC_Regrouped_Total_{JET_TYPE}"]
    neg = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, f, jet_scale=None, jes=src, sign=1, corrt1_scale=ct1_s)
    frac = float(np.mean(np.abs(_np(met.JES_Total.up.pt) - neg[0]) > 1e-3))
    print(f"  negative control (Jet s := 1): fraction(|dpt| > 1e-3 GeV) = {frac:.4f} (threshold > 0.1)")
    assert frac > 0.1, f"negative control s := 1 not distinguishable: {frac:.3f}"
    ct1_neg = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, f, jet_scale=s, jes=src, sign=1, corrt1_scale=None)
    n_ct1 = int(_met_mismatch(met.JES_Total.up.pt, met.JES_Total.up.phi, ct1_neg[0], ct1_neg[1]).sum())
    print(f"  CorrT1 control (CorrT1 unsmeared): {n_ct1} events mismatch (threshold > 10)")
    assert n_ct1 > 10, f"CorrT1 smearing control not distinguishable: {n_ct1} events"
    raw = _jet_raw(ev.Jet)
    nomu = ak.values_astype(raw * (1.0 - ev.Jet.muonSubtrFactor), np.float64)
    sel = (nomu * ak.values_astype(f["jet_L123"], np.float32) > 15.0) & (np.abs(ev.Jet.eta) < 5.2) & ((ev.Jet.chEmEF + ev.Jet.neEmEF) < 0.9)
    phip = ev.Jet.phi + ev.Jet.muonSubtrDeltaPhi
    dpt = ak.where(sel, jets.JES_Total.up.pt - jets.pt, 0.0)
    px = _np(met.pt) * np.cos(_np(met.phi)) - _np(ak.sum(dpt * np.cos(phip), axis=1))
    py = _np(met.pt) * np.sin(_np(met.phi)) - _np(ak.sum(dpt * np.sin(phip), axis=1))
    d5 = np.hypot(px - _np(met.JES_Total.up.pt) * np.cos(_np(met.JES_Total.up.phi)),
                  py - _np(met.JES_Total.up.pt) * np.sin(_np(met.JES_Total.up.phi)))
    print(f"  D5 |dMET| (direct JES vs jet-shift propagation) GeV: {_quantiles(d5, (50, 84, 95, 99))}")


def test_jer_hybrid_smearing_matches_recipe():
    ev = _events("mc")
    jets = _smeared(ev)
    J = _jer_inputs(ev)
    _assert_jets_close(jets.pt_jec, _unflat(J["pt_jec"], J["counts"]), "pt_jec vs independent")
    hyb = J["matched"]
    n = int(hyb.sum())
    print(f"  hybrid population: {n} of {len(hyb)} jets")
    assert n >= 1000, f"hybrid population too small: {n}"
    pt_jec, pt_gen = J["pt_jec"], J["pt_gen"]
    for var, obs in (("nom", jets.pt), ("up", jets.JER.up.pt), ("down", jets.JER.down.pt)):
        s = J["s"][var][hyb]
        n_guard = int((~((s > 0) & np.isfinite(s))).sum())
        print(f"  {var}: guard active on hybrid jets = {n_guard}")
        assert n_guard == 0, f"guard acts on {n_guard} hybrid jets ({var})"
        smear = 1 + (J["sf"][var] - 1) * (pt_jec - pt_gen) / pt_jec
        x = _flat(obs)[hyb]
        y = (pt_jec * smear)[hyb]
        ok = np.isclose(x, y, rtol=1e-5, atol=0)
        assert ok.all(), f"hybrid {var}: {(~ok).sum()}/{len(ok)} jets differ"


def test_jer_smearing_matches_jersmear():
    ev = _events("mc")
    jm = _clib(True, True)
    jets = _smeared(ev, jm)
    J = _jer_inputs(ev)
    assert np.array_equal(_flat(jets.pt_jec), J["pt_jec"]), "pt_jec differs bitwise from the independent raw*f_L123"
    obs = {
        "nom": _flat(jets.jet_energy_resolution_correction),
        "up": _flat(jets.JER.up.jet_energy_resolution_correction),
        "down": _flat(jets.JER.down.jet_energy_resolution_correction),
    }
    pts = {"nom": _flat(jets.pt), "up": _flat(jets.JER.up.pt), "down": _flat(jets.JER.down.pt)}
    for v in VARIATIONS:
        _assert_close(obs[v], J["final"][v], f"factor {v} vs independent JERSmear + guard + HE rule")
        _assert_close(pts[v], J["pt_jec"] * J["final"][v], f"pt {v} vs pt_jec * factor")
    he_unm = J["he"] & ~J["matched"]
    for v in VARIATIONS:
        assert np.isfinite(obs[v]).all() and (obs[v] > 0).all(), f"{v}: non-finite or non-positive factor"
        g = J["G"][v]
        assert (obs[v][g] == 1.0).all(), f"{v}: guarded factors are not exactly 1"
        print(f"  guard set {v}: {int(g.sum())} jets, outside HE-unmatched {int((g & ~he_unm).sum())}")
    assert J["G"]["up"].sum() >= 1, "no guarded 'up' factor: the guard is not exercised"
    assert (J["G"]["up"] & ~he_unm).sum() >= 1, "no guarded 'up' factor outside the HE-unmatched set"
    stoch = ~J["matched"]
    print(f"  stochastic population: {int(stoch.sum())}")
    assert stoch.sum() >= 1000, f"stochastic population too small: {stoch.sum()}"
    z = J["z"]
    for v in VARIATIONS:
        expect = 1 + np.sqrt(np.maximum(J["sf"][v] ** 2 - 1, 0)) * J["sigma"] * z
        assert np.allclose(J["s"][v][stoch], expect[stoch], rtol=0, atol=1e-12), f"{v}: JERSmear stochastic branch != 1+sqrt(sf^2-1)*sigma*z"
    N = len(z)
    bound = 5 / np.sqrt(N)
    print(f"  z over all jets: N={N}, mean={z.mean():.4g}, std={z.std():.4g}, bound={bound:.4g}")
    assert abs(z.mean()) < bound, f"mean(z) = {z.mean():.4g}"
    assert abs(z.std() - 1) < bound, f"std(z) = {z.std():.4g}"
    coffea = _coffea_rng_jets(jm, ev)
    pop = stoch & ~J["he"] & (J["sf"]["nom"] > 1)
    frac, n = _frac_differ(_flat(coffea.jet_energy_resolution_correction)[pop], J["guard"]["nom"][pop])
    print(f"  negative control (coffea RNG): fraction differing = {frac:.4f} of N={n} non-HE stochastic jets with sf>1 (threshold > 0.5)")
    assert frac > 0.5, f"coffea-RNG control not distinguishable: {frac:.3f}"


def test_corrt1_jersmear_factors(monkeypatch):
    ev = _events("mc")
    jm = _clib(True, True)
    jets = _smeared(ev, jm)
    recorded = {}
    original = jm.met_factory.build

    def recorder(*args, **kwargs):
        recorded["ct1"] = kwargs.get("in_CorrT1METJets")
        return original(*args, **kwargs)

    monkeypatch.setattr(jm.met_factory, "build", recorder)
    _clib_met(jm, ev, jets)
    ct1 = recorded["ct1"]
    C = _ct1_jer_inputs(ev)
    obs = {
        "nom": _flat(ct1.jet_energy_resolution_correction),
        "up": _flat(ct1.JER.up.jet_energy_resolution_correction),
        "down": _flat(ct1.JER.down.jet_energy_resolution_correction),
    }
    print(f"  CorrT1 jets: {len(obs['nom'])}")
    for v in VARIATIONS:
        _assert_close(obs[v], C["final"][v], f"CorrT1 factor {v}")
    m = C["matched"]
    print(f"  CorrT1 matched {int(m.sum())}, stochastic {int((~m).sum())}")
    assert m.sum() >= 100, f"CorrT1 matched population too small: {m.sum()}"
    assert (~m).sum() >= 1000, f"CorrT1 stochastic population too small: {(~m).sum()}"
    _, metric = C["v4"].nearest(ev.GenJet, return_metric=True)
    metric = _flat(ak.fill_none(metric, 999.0))
    mid = int(((metric >= 0.2) & (metric < 0.4)).sum())
    print(f"  CorrT1 jets with nearest GenJet at 0.2 <= dR < 0.4: {mid}")
    assert mid >= 10, f"too few CorrT1 jets at 0.2 <= dR < 0.4: {mid}"
    he_unm = C["he"] & ~m
    for v in VARIATIONS:
        assert np.isfinite(obs[v]).all() and (obs[v] > 0).all(), f"CorrT1 {v}: non-finite or non-positive factor"
        g = C["G"][v]
        assert (obs[v][g] == 1.0).all(), f"CorrT1 {v}: guarded factors are not exactly 1"
        print(f"  CorrT1 guard set {v}: {int(g.sum())}, outside HE-unmatched {int((g & ~he_unm).sum())}")
    assert C["G"]["up"].sum() >= 1, "no guarded CorrT1 'up' factor"
    assert (C["G"]["up"] & ~he_unm).sum() >= 1, "no guarded CorrT1 'up' factor outside the HE-unmatched set"
    s_nomatch = _jersmear_ind(C["pt_jec"], C["eta"], np.full_like(C["pt_jec"], -1.0), C["rho"], C["eid"], C["sigma"], C["sf"]["nom"])
    pop = m & (C["sf"]["nom"] > 1)
    frac, n = _frac_differ(_guard(s_nomatch)[pop], C["guard"]["nom"][pop])
    print(f"  negative control (no CorrT1 matching): fraction differing = {frac:.4f} of N={n} (threshold > 0.5)")
    assert frac > 0.5, f"no-matching control not distinguishable: {frac:.3f}"
    assert clib_name_map["CorrT1JetJERSmearFactor"] == "jet_energy_resolution_correction"
    near = _flat(ak.fill_none(ev.Jet.nearest(ev.GenJet, threshold=0.2).pt, 0))
    idx = _flat(_gen_pt_dr(ev.Jet, 0.2)[0])
    agree = float(np.mean(near == idx))
    print(f"  Jet nearest-GenJet(dR<=0.2) vs genJetIdx+dR<0.2 agreement: {agree:.5f} (threshold >= 0.99)")
    assert agree >= 0.99, f"nearest-GenJet and genJetIdx matching agree for only {agree:.4f}"
    print(f"  MET-selected unmatched HE CorrT1 jets: {_met_selected_corrt1(ev, he_unm)}")
    n_he_unm = int(he_unm.sum())
    pop_sf = he_unm & (C["sf"]["nom"] > 1)
    he_m = C["he"] & m
    print(f"  CorrT1 HE unmatched {n_he_unm}, with sf>1 {int(pop_sf.sum())}, HE matched {int(he_m.sum())}")
    assert n_he_unm >= 500, f"CorrT1 HE-unmatched population too small: {n_he_unm}"
    for v in VARIATIONS:
        assert (obs[v][he_unm] == 1.0).all(), f"CorrT1 HE-unmatched factor {v} != 1"
    assert pop_sf.sum() >= 90, f"CorrT1 HE-unmatched with sf>1 too small: {pop_sf.sum()}"
    frac, n = _frac_differ(C["guard"]["nom"][pop_sf], np.ones(int(pop_sf.sum())))
    print(f"  CorrT1 HE negative control (no rule): fraction differing from 1 = {frac:.4f} of N={n}")
    assert frac > 0.5, f"CorrT1 HE-rule control not distinguishable: {frac:.3f}"
    assert he_m.sum() >= 10, f"CorrT1 HE matched population too small: {he_m.sum()}"
    for v in VARIATIONS:
        assert (obs[v][he_m] == C["guard"][v][he_m]).all(), f"CorrT1 HE matched factor {v} changed by the rule"
    for jm_other, ev_other, label in ((_clib(False, False), _events("dataC"), "data C"), (_clib(True, False), ev, "MC doJER=False")):
        rec = {}
        orig = jm_other.met_factory.build

        def rec_build(*args, _rec=rec, _orig=orig, **kwargs):
            _rec["ct1"] = kwargs.get("in_CorrT1METJets")
            return _orig(*args, **kwargs)

        monkeypatch.setattr(jm_other.met_factory, "build", rec_build)
        run = ev_other.run if label == "data C" else None
        js = jm_other.corrected_jets_L123_JER(ev_other.Jet, _rho(ev_other), {}, run=run)
        _clib_met(jm_other, ev_other, js, run=run)
        assert "jet_energy_resolution_correction" not in rec["ct1"].fields, f"{label}: CorrT1 carries a smear factor"
        assert "JER" not in rec["ct1"].fields, f"{label}: CorrT1 carries a JER record"


def _met_selected_corrt1(ev, mask):
    ct1 = ev.CorrT1METJet
    f123, _ = _clib_factors(ct1, _rho(ev), True, corrt1=True)
    nomu = _flat(ak.values_astype(ct1.rawPt * (1.0 - ct1.muonSubtrFactor), np.float64))
    sel = (nomu * _flat(ak.values_astype(f123, np.float32)) > 15.0) & (np.abs(_flat(ct1.eta)) < 5.2) & (_flat(ct1.EmEF) < 0.9)
    return int((sel & mask).sum())


def test_corrt1_match_window_rho_clamp():
    jm = _clib(True, True)
    ev = _events("mc")
    rho_max = _rho_max_ind()
    edge = _jer_rho_max(str(_handler().getPath("jet_jerc")), f"{JER_TAG}_MC_PtResolution_{JET_TYPE}")
    assert edge == rho_max, f"_jer_rho_max {edge!r} != {rho_max!r}"
    assert jm._corrt1_match_rho_max == rho_max, f"stored clamp {jm._corrt1_match_rho_max!r} != {rho_max!r}"
    res = _cset()[f"{JER_TAG}_MC_PtResolution_{JET_TYPE}"]
    assert res.evaluate(1.0, 15.0, 52.05) == 1.0, "PtResolution at rho = 52.05 is not the 1.0 overflow"
    assert res.evaluate(1.0, 15.0, float(rho_max)) != 1.0, "PtResolution just below the edge is 1.0"
    rho = _rho(ev)
    ct1 = _prepare_corrt1_jets(ev.CorrT1METJet, rho)
    C = _ct1_jer_inputs(ev)
    hi = C["rho"] > 52.05
    print(f"  high-rho CorrT1 jets: {int(hi.sum())}, with sf_nom > 1: {int((hi & (C['sf']['nom'] > 1)).sum())}")
    assert hi.sum() >= 40, f"too few high-rho CorrT1 jets: {hi.sum()}"
    assert (hi & (C["sf"]["nom"] > 1)).sum() >= 30, "too few high-rho CorrT1 jets with sf_nom > 1"
    sigma_un = C["sigma"]
    sigma_cl = res.evaluate(C["eta"], C["pt_jec"], np.minimum(C["rho"], rho_max))
    assert (sigma_un[hi] == 1.0).all(), "payload sigma is not 1.0 above the rho edge"
    assert (sigma_cl[hi] < 1.0).all(), "clamped sigma is not below 1.0"
    hi_j = _unflat(hi, C["counts"])
    he = C["he"]

    def gen_for(r):
        pt_gen = C["pt_jec"] * (1 + r)
        return ak.zip({
            "eta": ct1.eta[hi_j], "phi": ct1.phi[hi_j], "pt": _unflat(pt_gen, C["counts"])[hi_j],
        }), pt_gen

    def factors(gen):
        out = _jersmear_corrt1_jets(ct1, ev.event, gen, jm.met_factory.jec_L1L2L3, jm._max_gen_dr, *jm._jer_smear[:3],
                                    jm._corrt1_match_rho_max, jm._jer_smear[4])
        return {
            "nom": _flat(out.jet_energy_resolution_correction),
            "up": _flat(out.JER.up.jet_energy_resolution_correction),
            "down": _flat(out.JER.down.jet_energy_resolution_correction),
        }

    def ref(genpt, sigma, v):
        return _guard(_jersmear_ind(C["pt_jec"], C["eta"], genpt, C["rho"], C["eid"], sigma, C["sf"][v]))

    r_out = (3 * sigma_cl + 3) / 2
    assert ((r_out[hi] >= 3 * sigma_cl[hi]) & (r_out[hi] < 3 * sigma_un[hi])).all(), "r_out not in [3 sigma_cl, 3 sigma_un)"
    gen_out, pt_gen_out = gen_for(r_out)
    f_out = factors(gen_out)
    minus1 = np.full_like(C["pt_jec"], -1.0)
    for v in VARIATIONS:
        expect = np.where(he, 1.0, ref(minus1, sigma_un, v))
        _assert_close(f_out[v][hi], expect[hi], f"gen_out {v}: unmatched, unclamped smear sigma")
    pop = hi & (C["sf"]["nom"] > 1)
    frac, n = _frac_differ(ref(pt_gen_out, sigma_un, "nom")[pop], f_out["nom"][pop])
    print(f"  negative control 1 (unclamped window would match): {frac:.4f} of N={n}")
    assert frac > 0.5, f"unclamped-window control not distinguishable: {frac:.3f}"
    pop2 = hi & ~he & (C["sf"]["nom"] > 1)
    print(f"  high-rho non-HE CorrT1 jets with sf_nom > 1: {int(pop2.sum())}")
    assert pop2.sum() >= 30, f"too few high-rho non-HE CorrT1 jets with sf_nom > 1: {pop2.sum()}"
    frac, n = _frac_differ(ref(minus1, sigma_cl, "nom")[pop2], f_out["nom"][pop2])
    print(f"  negative control 2 (clamped smear sigma): {frac:.4f} of N={n}")
    assert frac > 0.5, f"clamped-smear-sigma control not distinguishable: {frac:.3f}"
    gen_in, pt_gen_in = gen_for(1.5 * sigma_cl)
    f_in = factors(gen_in)
    for v in VARIATIONS:
        _assert_close(f_in[v][hi], ref(pt_gen_in, sigma_un, v)[hi], f"gen_in {v}: matched")
    J = _jer_inputs(ev)
    injets = add_jme_variables(ev.Jet, rho, max_gen_dr=0.2)
    jets = jm.jec_factory_L123_JER.build(injets)
    hi_jet = J["rho"] > 52.05
    sig_cl_jet = res.evaluate(J["eta"], J["pt_jec"], np.minimum(J["rho"], rho_max))
    r_jet = (3 * sig_cl_jet + 3) / 2
    new_gen = J["pt_jec"] * (1 + r_jet)
    pt_gen_mod = np.where(hi_jet, new_gen, _flat(jets.pt_gen))
    jets_mod = ak.with_field(jets, _unflat(pt_gen_mod, J["counts"]), "pt_gen")
    out = _jersmear_jets(injets, jets_mod, ev.event, *jm._jer_smear)
    nom = _flat(out.jet_energy_resolution_correction)
    pop_j = hi_jet & (J["sf"]["nom"] > 1)
    print(f"  high-rho Jets: {int(hi_jet.sum())}, with sf_nom > 1: {int(pop_j.sum())}")
    assert hi_jet.sum() >= 100, f"too few high-rho Jets: {hi_jet.sum()}"
    assert pop_j.sum() >= 90, f"too few high-rho Jets with sf_nom > 1: {pop_j.sum()}"
    hyb = _guard(_jersmear_ind(J["pt_jec"], J["eta"], pt_gen_mod, J["rho"], J["eid"], J["sigma"], J["sf"]["nom"]))
    _assert_close(nom[hi_jet], hyb[hi_jet], "high-rho Jets keep the payload window (hybrid)")
    sto = np.where(J["he"], 1.0, _guard(_jersmear_ind(J["pt_jec"], J["eta"], np.full_like(J["pt_jec"], -1.0), J["rho"], J["eid"], J["sigma"], J["sf"]["nom"])))
    frac, n = _frac_differ(sto[pop_j], nom[pop_j])
    print(f"  Jet stochastic alternative differs: {frac:.4f} of N={n}")
    assert frac > 0.5, f"Jet window control not distinguishable: {frac:.3f}"


def test_jer_he_rule_jets():
    jm = _clib(True, True)
    ev = _events("mc")
    jets = _smeared(ev, jm)
    J = _jer_inputs(ev)
    he, m = J["he"], J["matched"]
    obs = {
        "nom": _flat(jets.jet_energy_resolution_correction),
        "up": _flat(jets.JER.up.jet_energy_resolution_correction),
        "down": _flat(jets.JER.down.jet_energy_resolution_correction),
    }
    h_unm = he & ~m
    print(f"  HE unmatched jets: {int(h_unm.sum())} of {len(he)}")
    assert h_unm.sum() >= 1000, f"HE-unmatched population too small: {h_unm.sum()}"
    pt_jec = _flat(jets.pt_jec)
    for v in VARIATIONS:
        assert (obs[v][h_unm] == 1.0).all(), f"HE-unmatched factor {v} != 1"
    for label, x in (("pt", jets.pt), ("JER.up.pt", jets.JER.up.pt), ("JER.down.pt", jets.JER.down.pt)):
        assert np.array_equal(_flat(x)[h_unm], pt_jec[h_unm]), f"HE-unmatched {label} != pt_jec"
    delta = _cset()[f"{JEC_TAG}_MC_Regrouped_Total_{JET_TYPE}"].evaluate(J["eta"], pt_jec)
    _assert_close(_flat(jets.JES_Total.up.pt)[h_unm], (pt_jec * (1 + delta))[h_unm], "HE-unmatched JES_Total up")
    _assert_close(_flat(jets.JES_Total.down.pt)[h_unm], (pt_jec * (1 - delta))[h_unm], "HE-unmatched JES_Total down")
    for v in ("nom", "up"):
        pop = h_unm & (J["sf"][v] > 1)
        frac, n = _frac_differ(J["guard"][v][pop], np.ones(int(pop.sum())))
        print(f"  negative control {v} (no rule): {frac:.4f} of N={n} HE-unmatched jets with sf_{v} > 1")
        assert n >= 900, f"HE-unmatched with sf_{v} > 1 too small: {n}"
        assert frac > 0.5, f"HE-rule control {v} not distinguishable: {frac:.3f}"
    n_down = int((np.abs(J["guard"]["down"][h_unm] - 1) > 1e-6).sum())
    print(f"  HE-unmatched jets whose down factor the rule changes: {n_down}")
    assert n_down >= 100, f"too few HE-unmatched down factors changed: {n_down}"
    h_m = he & m
    print(f"  HE matched jets: {int(h_m.sum())}")
    assert h_m.sum() >= 100, f"HE-matched population too small: {h_m.sum()}"
    for v in VARIATIONS:
        hyb = 1 + (J["sf"][v] - 1) * (J["pt_jec"] - J["pt_gen"]) / J["pt_jec"]
        _assert_close(obs[v][h_m], hyb[h_m], f"HE matched {v} hybrid")
    pop = h_m & (J["sf"]["nom"] > 1)
    frac, n = _frac_differ(obs["nom"][pop], np.ones(int(pop.sum())))
    print(f"  HE matched with sf>1 differing from 1: {frac:.4f} of N={n}")
    assert frac > 0.5, f"HE matched jets look unsmeared: {frac:.3f}"
    nh_unm = ~he & ~m
    print(f"  non-HE unmatched jets: {int(nh_unm.sum())}")
    assert nh_unm.sum() >= 800, f"non-HE unmatched population too small: {nh_unm.sum()}"
    for v in VARIATIONS:
        _assert_close(obs[v][nh_unm], J["guard"][v][nh_unm], f"non-HE unmatched {v}")
    pop = nh_unm & (J["sf"]["nom"] > 1)
    frac, n = _frac_differ(obs["nom"][pop], np.ones(int(pop.sum())))
    print(f"  non-HE unmatched with sf>1 differing from 1: {frac:.4f} of N={n}")
    assert frac > 0.5, f"non-HE unmatched jets look unsmeared: {frac:.3f}"
    eta = np.array([1.0, np.nextafter(np.float32(2.5), np.float32(0)), 2.5, -2.5, 2.75,
                    np.nextafter(np.float32(3.0), np.float32(0)), 3.0, -3.0, 3.2], dtype=np.float32)
    assert eta[1] != np.float32(2.5) and eta[5] != np.float32(3.0), "near-boundary eta collapsed onto the boundary"
    n_pts = len(eta)
    pt = np.full(n_pts, 40.0, dtype=np.float64)
    rho_s = np.full(n_pts, 20.0, dtype=np.float32)
    eid = ak.Array([123456789])
    in_region = np.array([False, False, True, True, True, True, False, False, False])
    res = _cset()[f"{JER_TAG}_MC_PtResolution_{JET_TYPE}"]
    sfc = _cset()[f"{JER_TAG}_MC_ScaleFactor_{JET_TYPE}"]
    sig = res.evaluate(eta.astype(np.float64), pt, rho_s.astype(np.float64))
    sf = {v: sfc.evaluate(eta.astype(np.float64), pt, v) for v in VARIATIONS}
    eid_flat = np.full(n_pts, 123456789, dtype=np.int64)
    sto = {v: _guard(_jersmear_ind(pt, eta.astype(np.float64), np.full(n_pts, -1.0), rho_s.astype(np.float64), eid_flat, sig, sf[v])) for v in VARIATIONS}
    for regions in (jm._jer_smear[4], ()):
        _, s_unm = _jersmear_factors(one(pt), one(eta), one(rho_s), eid, one(np.zeros(n_pts, dtype=np.float32)), *jm._jer_smear[:3], regions)
        _, s_m = _jersmear_factors(one(pt), one(eta), one(rho_s), eid, one(np.full(n_pts, 38.0, dtype=np.float32)), *jm._jer_smear[:3], regions)
        for v in VARIATIONS:
            got = _flat(s_unm[v])
            if regions:
                assert (got[in_region] == 1.0).all(), f"synthetic in-region unmatched {v} != 1: {got}"
                _assert_close(got[~in_region], sto[v][~in_region], f"synthetic out-of-region {v}")
            else:
                _assert_close(got, sto[v], f"synthetic no-rule {v}")
            hyb = 1 + (sf[v] - 1) * (40.0 - 38.0) / 40.0
            _assert_close(_flat(s_m[v]), hyb, f"synthetic matched {v} ({regions})")
        got = {v: _flat(s_unm[v]) for v in VARIATIONS}
        if regions:
            for v in ("nom", "up"):
                assert (np.abs(got[v][~in_region] - 1) > 1e-6).all(), f"out-of-region {v} factor is 1: {got[v]}"
            assert (np.abs(got["down"][[6, 7]] - 1) > 1e-6).all(), f"down factor at +-3.0 is 1: {got['down']}"
        else:
            for v in ("nom", "up"):
                assert (np.abs(got[v][in_region] - 1) > 1e-6).all(), f"no-rule in-region {v} factor is 1: {got[v]}"
            assert (np.abs(got["down"][[4, 5]] - 1) > 1e-6).all(), f"no-rule down at 2.75 / 3.0- is 1: {got['down']}"
            assert (got["down"][[2, 3]] == 1.0).all(), f"no-rule down at +-2.5 is not 1: {got['down']}"
    rho = _rho(ev)
    injets = add_jme_variables(ev.Jet, rho, max_gen_dr=0.2)
    jec_only = jm.jec_factory_L123_JER.build(injets)
    jets_nr = _jersmear_jets(injets, jec_only, ev.event, *jm._jer_smear[:4], ())
    ct1_nr = _jersmear_corrt1_jets(_prepare_corrt1_jets(ev.CorrT1METJet, rho), ev.event, ev.GenJet, jm.met_factory.jec_L1L2L3,
                                   jm._max_gen_dr, *jm._jer_smear[:3], jm._corrt1_match_rho_max, ())
    met_in, raw = _met_inputs(ev)
    met_nr = jm.met_factory.build(met_in, jets_nr, in_RawMET=raw, in_CorrT1METJets=ct1_nr)
    met = _clib_met(jm, ev, jets)
    n_nom = int(_met_mismatch(met.pt, met.phi, _np(met_nr.pt), _np(met_nr.phi)).sum())
    n_up = int(_met_mismatch(met.JER.up.pt, met.JER.up.phi, _np(met_nr.JER.up.pt), _np(met_nr.JER.up.phi)).sum())
    print(f"  MET rule vs no rule: nominal {n_nom}, JERUp {n_up} events mismatch (threshold > 100)")
    for label, a, b in (("nominal", met, met_nr), ("JERUp", met.JER.up, met_nr.JER.up), ("JERDown", met.JER.down, met_nr.JER.down),
                        ("JES_Total up", met.JES_Total.up, met_nr.JES_Total.up)):
        d = np.abs(_np(a.pt) - _np(b.pt))
        print(f"  |dMET| rule vs no rule, {label}: changed {int((d > 1e-3).sum())}; {_quantiles(d, (50, 84, 95, 99, 100))}; mean shift {np.mean(_np(a.pt) - _np(b.pt)):.4g} GeV")
    assert n_nom > 100, f"HE rule changes nominal MET in only {n_nom} events"
    assert n_up > 100, f"HE rule changes JERUp MET in only {n_up} events"


def test_pt_gen_requires_dr_match():
    ev = _events("mc")
    jets = _clib(True, True).corrected_jets_L123_JER(ev.Jet, _rho(ev), {}, run=ev.run, event_id=ev.event)
    matched_pt = _flat(ak.fill_none(ev.Jet.matched_gen.pt, -1.0))
    _, dr = _gen_pt_dr(ev.Jet, 0.2)
    dr = _flat(dr)
    pt_gen = _flat(jets.pt_gen)
    close = dr < 0.2
    far = ~close
    assert np.array_equal(pt_gen[close], matched_pt[close]), "pt_gen != matched_gen.pt for dR < 0.2"
    assert (pt_gen[far] == 0).all(), "pt_gen nonzero for dR >= 0.2 or unmatched"
    mid = (dr >= 0.2) & (dr < 0.4)
    print(f"  jets with 0.2 <= dR < 0.4: {int(mid.sum())}")
    assert mid.any(), "no jets with 0.2 <= dR < 0.4"
    assert "run" not in jets.fields, "clib MC jets carry a run field"
    evd = _events("dataC")
    djets = _clib(False, False).corrected_jets_L123_noJER(evd.Jet, _rho(evd), {}, run=evd.run)
    assert "run" in djets.fields, "clib data jets lack the run field"


def test_legacy_jets_unchanged():
    _require_txt()
    ev = _events("mc")
    matched_pt = _flat(ak.fill_none(ev.Jet.matched_gen.pt, -1.0))
    _, dr = _gen_pt_dr(ev.Jet, 0.2)
    dr = _flat(dr)
    mid = (dr >= 0.2) & (dr < 0.4)
    leg = _txt(TXT_MC_TAG, True)
    ljets = leg.corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    lpt_gen = _flat(ljets.pt_gen)
    assert np.array_equal(lpt_gen[mid], matched_pt[mid]), "legacy pt_gen changed for 0.2 <= dR < 0.4"
    ljets_run = leg.corrected_jets_L123_noJER(ev.Jet, _rho(ev), {}, run=ev.run)
    assert ljets_run.fields == ljets.fields, "legacy field list changes when run is passed"
    assert "run" not in ljets_run.fields, "legacy jets gained a run field"
    a = leg.corrected_jets_L123_JER(ev.Jet, _rho(ev), {})
    b = leg.corrected_jets_L123_JER(ev.Jet, _rho(ev), {}, run=ev.run, event_id=ev.event)
    assert a.fields == b.fields, "legacy _JER field list changes with run/event_id"
    assert np.array_equal(_flat(a.pt), _flat(b.pt)), "legacy _JER pt changes with run/event_id"
    met_in = prepare_met_for_factory(ev.PuppiMET)
    m0 = leg.corrected_met(met_in, a, _rho(ev), {})
    m1 = leg.corrected_met(met_in, a, _rho(ev), {}, event_id=ev.event, gen_jets=ev.GenJet)
    assert np.array_equal(_np(m0.pt), _np(m1.pt)), "legacy MET changes with event_id/gen_jets"


def test_jer_requires_event_id():
    ev = _events("mc")
    jm = _clib(True, True)
    with pytest.raises(ValueError, match="event_id"):
        jm.corrected_jets_L123_JER(ev.Jet, _rho(ev), {})
    jets = _smeared(ev, jm)
    met_in, raw = _met_inputs(ev)
    with pytest.raises(ValueError, match="gen_jets"):
        jm.corrected_met(met_in, jets, _rho(ev), {}, raw_met=raw, corrt1_jets=ev.CorrT1METJet, event_id=ev.event)
    with pytest.raises(ValueError, match="gen_jets"):
        jm.corrected_met(met_in, jets, _rho(ev), {}, raw_met=raw, corrt1_jets=ev.CorrT1METJet, gen_jets=ev.GenJet)
    met = jm.corrected_met(met_in, jets, _rho(ev), {}, raw_met=raw, corrt1_jets=None)
    assert "JER" in met.fields, "MET without CorrT1 lacks JER"
    for jm_other, evo, run in ((_clib(True, False), ev, None), (_clib(False, False), _events("dataC"), _events("dataC").run)):
        js = jm_other.corrected_jets_L123_JER(evo.Jet, _rho(evo), {}, run=run)
        assert "JER" not in js.fields, "unsmeared clib jets carry JER"
        mi, rw = _met_inputs(evo)
        jm_other.corrected_met(mi, js, _rho(evo), {}, raw_met=rw, corrt1_jets=evo.CorrT1METJet, run=run)


def test_jer_smearing_chunk_independent():
    jm = _clib(True, True)
    ev_a = _events("mc")
    ev_b = _load(MC_FILE, 2000, entry_start=1000)
    a = _smeared(ev_a, jm)
    a2 = _smeared(ev_a, jm)
    b = _smeared(ev_b, jm)
    assert np.array_equal(_flat(a.pt), _flat(a2.pt)), "second build of A differs"
    ov = a[1000:]
    for label, x, y in (
        ("pt", ov.pt, b.pt), ("mass", ov.mass, b.mass),
        ("jet_energy_resolution_correction", ov.jet_energy_resolution_correction, b.jet_energy_resolution_correction),
        ("JER.up.pt", ov.JER.up.pt, b.JER.up.pt), ("JER.down.pt", ov.JER.down.pt, b.JER.down.pt),
        ("JES_Total.up.pt", ov.JES_Total.up.pt, b.JES_Total.up.pt),
    ):
        assert np.array_equal(_flat(x), _flat(y)), f"{label} depends on the chunking"
    met_a = _clib_met(jm, ev_a, a)[1000:]
    met_b = _clib_met(jm, ev_b, b)
    for label, x, y in (("MET", met_a, met_b), ("MET JER.up", met_a.JER.up, met_b.JER.up),
                        ("MET JES_Total.up", met_a.JES_Total.up, met_b.JES_Total.up)):
        assert np.array_equal(_np(x.pt), _np(y.pt)), f"{label} pt depends on the chunking"
    J = _jer_inputs(ev_a)
    off = int(ak.sum(ak.num(ev_a.Jet)[:1000]))
    pop = (~J["matched"] & (J["sf"]["nom"] > 1))[off:]
    ca = _flat(_coffea_rng_jets(jm, ev_a).pt)[off:]
    cb = _flat(_coffea_rng_jets(jm, ev_b).pt)
    frac, n = _frac_differ(ca[pop], cb[pop])
    print(f"  negative control (coffea RNG): fraction of overlap stochastic jets with a different pt = {frac:.4f} of N={n}")
    assert frac > 0.5, f"coffea-RNG chunking control not distinguishable: {frac:.3f}"


def test_jersmear_jets_match_factory_contract():
    jm = _clib(True, True)
    ev = _events("mc")
    ours = _smeared(ev, jm)
    coffea = _coffea_rng_jets(jm, ev)
    expected_missing = {"jet_resolution_rand_gauss", "jet_energy_resolution_scale_factor"} | {f"jet_energy_uncertainty_{n}" for n in JES_SOURCES}
    assert set(coffea.fields) - set(ours.fields) == expected_missing, f"missing fields {sorted(set(coffea.fields) - set(ours.fields))}"
    assert not set(ours.fields) - set(coffea.fields), f"extra fields {sorted(set(ours.fields) - set(coffea.fields))}"
    for d in ("up", "down"):
        assert ours.JER[d].fields == coffea.JER[d].fields, f"JER.{d} fields differ"
        for n in JES_SOURCES:
            assert ours[f"JES_{n}"][d].fields == coffea[f"JES_{n}"][d].fields, f"JES_{n}.{d} fields differ"
    assert ours.layout.content.parameters == coffea.layout.content.parameters, "top-level record parameters differ"
    print(f"  top-level parameters: {ours.layout.content.parameters}")
    assert str(ak.type(ours.JER)) == str(ak.type(coffea.JER)), "JER type differs"
    assert ak.parameters(ours.JER) == ak.parameters(coffea.JER), "JER parameters differ"
    print(f"  JER type: {ak.type(ours.JER)}; parameters {ak.parameters(ours.JER)}")
    for n in JES_SOURCES:
        assert str(ak.type(ours[f"JES_{n}"])) == str(ak.type(coffea[f"JES_{n}"])), f"JES_{n} type differs"
        assert ak.parameters(ours[f"JES_{n}"]) == ak.parameters(coffea[f"JES_{n}"]), f"JES_{n} parameters differ"
    print(f"  JES_Total type: {ak.type(ours.JES_Total)}; parameters {ak.parameters(ours.JES_Total)}")
    J = _jer_inputs(ev)
    pop = J["matched"]
    print(f"  P_pop = matched_ind: {int(pop.sum())}, of which HE {int((pop & J['he']).sum())}")
    assert pop.sum() >= 1000, f"P_pop too small: {pop.sum()}"
    assert (pop & J["he"]).sum() >= 100, f"HE part of P_pop too small: {(pop & J['he']).sum()}"
    for v in VARIATIONS:
        s = J["s"][v][pop]
        assert ((s > 0) & np.isfinite(s)).all(), f"guard acts on P_pop ({v})"
    fields = ["jet_energy_resolution_correction", "pt", "mass", "pt_jer", "mass_jer"]
    for f in fields:
        _assert_close(_flat(ours[f])[pop], _flat(coffea[f])[pop], f"parity {f}")
    for d in ("up", "down"):
        for f in ("pt", "mass", "jet_energy_resolution_correction"):
            _assert_close(_flat(ours.JER[d][f])[pop], _flat(coffea.JER[d][f])[pop], f"parity JER.{d}.{f}")
        for n in JES_SOURCES:
            for f in ("pt", "mass"):
                _assert_close(_flat(ours[f"JES_{n}"][d][f])[pop], _flat(coffea[f"JES_{n}"][d][f])[pop], f"parity JES_{n}.{d}.{f}")
    pt_jer = _flat(ours.pt_jer)
    eta = _flat(ours.eta)
    for n in JES_SOURCES:
        delta = _cset()[f"{JEC_TAG}_MC_Regrouped_{n}_{JET_TYPE}"].evaluate(eta, pt_jer)
        _assert_close(_flat(ours[f"JES_{n}"].up.pt), pt_jer * (1 + delta), f"JES_{n} up at smeared pt")
        _assert_close(_flat(ours[f"JES_{n}"].down.pt), pt_jer * (1 - delta), f"JES_{n} down at smeared pt")


def test_smeared_type1_nominal():
    ev = _events("mc")
    jm = _clib(True, True)
    jets = _smeared(ev, jm)
    met = _clib_met(jm, ev, jets)
    jm0 = _clib(True, False)
    met0 = _clib_met(jm0, ev, jm0.corrected_jets_L123_noJER(ev.Jet, _rho(ev), {}))
    ndiff = int((np.abs(_np(met.pt) - _np(met0.pt)) > 1e-3).sum())
    print(f"  events with smeared != unsmeared nominal MET: {ndiff}/{len(ev)}")
    assert ndiff > 0, "smeared nominal MET identical to unsmeared"
    f = _type1_factors(ev, True)
    C = _ct1_jer_inputs(ev)
    ref = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, f, jet_scale=jets.jet_energy_resolution_correction,
                          corrt1_scale=_unflat(C["final"]["nom"], C["counts"]))
    m = _assert_met_close(met, ref, "smeared nominal")
    print(f"  smeared nominal max |dpt| = {m:.3g} GeV")
    neg = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, f, jet_scale=jets.jet_energy_resolution_correction)
    n_ct1 = int(_met_mismatch(met.pt, met.phi, neg[0], neg[1]).sum())
    print(f"  CorrT1 control (unsmeared CorrT1): {n_ct1} events mismatch (threshold > 10)")
    assert n_ct1 > 10, f"CorrT1 smearing control not distinguishable: {n_ct1}"


def test_met_jer_variations():
    ev = _events("mc")
    jm = _clib(True, True)
    jets = _smeared(ev, jm)
    met = _clib_met(jm, ev, jets)
    f = _type1_factors(ev, True)
    pt_jec = _pt_jec_ind(ev, True)
    C = _ct1_jer_inputs(ev)
    ct1 = {v: _unflat(C["final"][v], C["counts"]) for v in VARIATIONS}
    refs = {}
    for d in ("up", "down"):
        scale = jets.JER[d].pt / pt_jec
        _assert_jets_close(scale, jets.JER[d].jet_energy_resolution_correction, f"JER {d} scale", rtol=1e-6)
        refs[d] = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, f, jet_scale=scale, corrt1_scale=ct1[d])
        m = _assert_met_close(met.JER[d], refs[d], f"JER {d}")
        frac = float(np.mean(np.abs(_np(met.JER[d].pt) - _np(met.pt)) > 1e-3))
        print(f"  JER {d}: max |dpt| vs reference = {m:.3g} GeV; fraction(|MET_JER - MET| > 1e-3) = {frac:.4f} (threshold > 0.3)")
        assert frac > 0.3, f"JER {d} differs from nominal in only {frac:.3f} of events"
        ctl = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, f, jet_scale=scale, corrt1_scale=ct1["nom"])
        n_ct1 = int(_met_mismatch(met.JER[d].pt, met.JER[d].phi, ctl[0], ctl[1]).sum())
        print(f"  CorrT1 control JER {d} (nominal CorrT1 factors): {n_ct1} events mismatch")
        if d == "up":
            assert n_ct1 > 10, f"CorrT1 JER-up control not distinguishable: {n_ct1}"
    mf = jm.met_factory
    unsmeared = CorrectedMETFactory(clib_name_map, jec_L1L2L3=mf.jec_L1L2L3, jec_L1=mf.jec_L1,
                                    jes_uncertainties=mf.jes_uncertainties, is_t1_smeared_met=False)
    met_in, raw = _met_inputs(ev)
    neg = unsmeared.build(met_in, jets, in_RawMET=raw, in_CorrT1METJets=_prepare_corrt1(ev))
    frac = float(np.mean(_met_mismatch(neg.JER.up.pt, neg.JER.up.phi, refs["up"][0], refs["up"][1])))
    print(f"  negative control (is_t1_smeared_met=False): fraction differing = {frac:.4f} (threshold > 0.1)")
    assert frac > 0.1, f"negative control not distinguishable: {frac:.3f}"
    nojer = jm.corrected_jets_L123_noJER(ev.Jet, _rho(ev), {})
    miswired = _clib_met(jm, ev, ak.with_field(nojer, jets.JER, "JER"))
    frac = float(np.mean(np.abs(_np(miswired.pt) - _np(met.pt)) > 1e-3))
    print(f"  mis-wiring (noJER jets + JER record): fraction nominal differing = {frac:.4f} (threshold > 0.1)")
    assert frac > 0.1, f"mis-wired nominal not distinguishable: {frac:.3f}"


def test_legacy_vs_type1_characterisation():
    _require_txt()
    for key, is_mc, tag in (("mc", True, TXT_MC_TAG), ("dataC", False, _txt_data_tag("Cnib1"))):
        ev = _events(key)
        run = None if is_mc else ev.run
        jm = _clib(is_mc, False)
        met = _clib_met(jm, ev, jm.corrected_jets_L123_noJER(ev.Jet, _rho(ev), {}, run=run), run=run)
        leg = _txt(tag, is_mc)
        lmet = leg.corrected_met(prepare_met_for_factory(ev.PuppiMET), leg.corrected_jets_L123_JER(ev.Jet, _rho(ev), {}), _rho(ev), {})
        rel = _np(met.pt) / _np(lmet.pt) - 1
        relp = _np(met.pt) / _np(ev.PuppiMET.pt) - 1
        print(f"  {key} (Type-1 - legacy)/legacy: {_quantiles(rel)}")
        print(f"  {key} (Type-1 - PuppiMET)/PuppiMET: {_quantiles(relp)}")
        med = float(np.median(rel))
        print(f"  {key} median (Type-1 - legacy)/legacy = {med:.4g} (threshold |median| < 0.05)")
        assert abs(med) < 0.05, f"{key}: |median| = {abs(med):.4g}"


def _install_recorder(proc):
    original = proc._jmeu.corrected_met
    sig = inspect.signature(JMEUncertainty.corrected_met)
    record = {}

    def recorder(*args, **kwargs):
        bound = sig.bind(proc._jmeu, *args, **kwargs)
        bound.apply_defaults()
        record["args"] = dict(bound.arguments)
        out = original(*args, **kwargs)
        record["out"] = out
        return out

    proc._jmeu.corrected_met = recorder
    return original, record


def _systematics(out, dataset):
    return set(out[dataset]["met_pt"].axes["systematic"])


def test_processor_type1_wiring(monkeypatch):
    from qawa.process.wztau2lnu_inclusive import wzinclusive_processor
    eras = []
    original_build_jets = wzproc.build_jets

    def spy(*args, **kwargs):
        eras.append(kwargs.get("era"))
        return original_build_jets(*args, **kwargs)

    monkeypatch.setattr(wzproc, "build_jets", spy)
    cases = (
        ("", "WZto3LNu", MC_FILE, False),
        ("C", "Run2024C__Muon0", DATA_C_FILE, True),
    )
    for run_period, dataset, path, is_data in cases:
        eras.clear()
        proc = wzinclusive_processor(era="2024", version="v15", run_period=run_period)
        original, record = _install_recorder(proc)
        meta = {"dataset": dataset, "is_data": is_data, "is_signal": False, "is_flashsim": False}
        out = proc.process(_load(path, 500, metadata=meta))
        args = record["args"]
        ev = _load(path, 500, metadata=meta)
        print(f"  {dataset}: build_jets calls {len(eras)}, eras {sorted(set(map(str, eras)))}")
        assert eras and all(e == "2024" for e in eras), f"{dataset}: build_jets era arguments {eras}"
        assert args["raw_met"] is not None, f"{dataset}: raw_met not passed"
        assert np.array_equal(_np(args["raw_met"].pt), _np(ev.RawPuppiMET.pt)), f"{dataset}: raw_met is not RawPuppiMET"
        assert args["corrt1_jets"] is not None, f"{dataset}: corrt1_jets not passed"
        assert CORRT1_FIELDS <= set(args["corrt1_jets"].fields), f"{dataset}: corrt1_jets fields {args['corrt1_jets'].fields}"
        assert args["run"] is not None and np.array_equal(_np(args["run"]), _np(ev.run)), f"{dataset}: run not passed"
        assert np.array_equal(_np(args["event_id"]), _np(ev.event)), f"{dataset}: event_id is not event.event"
        rho = _rho(ev)
        jets = proc._jmeu.corrected_jets_L123_JER(ev.Jet, rho, {}, run=ev.run, event_id=ev.event)
        direct = original(prepare_met_for_factory(ev.PuppiMET), jets, rho, {}, raw_met=ev.RawPuppiMET,
                          corrt1_jets=ev.CorrT1METJet, run=ev.run, event_id=ev.event,
                          gen_jets=ev.GenJet if "GenJet" in ev.fields else None)
        rec = record["out"]
        assert np.array_equal(_np(rec.pt), _np(direct.pt)), f"{dataset}: recorded MET pt != direct call"
        assert np.array_equal(_np(rec.phi), _np(direct.phi)), f"{dataset}: recorded MET phi != direct call"
        assert proc._jmeu.met_factory.type1_mode, f"{dataset}: MET factory not in Type-1 mode"
        if not is_data:
            assert isinstance(proc._jmeu.jec_stack_L123_noJER.jec, CorrectionLibJEC), f"{dataset}: not a correctionlib JEC"
            assert np.array_equal(_flat(args["jets_L123"].pt), _flat(jets.pt)), "recorded jets_L123 pt != direct smeared jets"
            assert args["gen_jets"] is not None, "MC gen_jets not passed"
            assert np.array_equal(_np(ak.num(args["gen_jets"])), _np(ak.num(ev.GenJet))), "MC gen_jets is not event.GenJet"
            assert "JER" in args["jets_L123"].fields, "MC jets_L123 not smeared (no JER)"
            assert "jet_energy_resolution_correction" in args["jets_L123"].fields, "MC jets_L123 lack JER correction"
            assert np.array_equal(_np(rec.JER.up.pt), _np(direct.JER.up.pt)), "recorded JER.up != direct"
            assert np.array_equal(_np(rec.JES_Total.up.pt), _np(direct.JES_Total.up.pt)), "recorded JES_Total.up != direct"
            syst = _systematics(out, dataset)
            expected = {"JERUp", "JERDown", "JESUp", "JESDown"}
            for name in PROCESSOR_SHIFT_JES[1:]:
                expected |= {name.replace("_2024", "2024") + "Up", name.replace("_2024", "2024") + "Down"}
            missing = expected - syst
            assert not missing, f"MC systematic axis lacks {sorted(missing)}"
        else:
            assert "JER" not in args["jets_L123"].fields, "data jets carry JER"
            assert args["gen_jets"] is None, "data gen_jets is not None"
            assert isinstance(proc._jmeu.jec_stack_L123_JER.jec, _ResidualPtFloorJEC), "data JEC is not the residual-floor JEC"
            assert proc._jmeu.met_factory.jec_L1L2L3 is proc._jmeu.jec_stack_L123_JER.jec, "MET and jets use different data JECs"


def test_processor_flashsim_wiring():
    from qawa.process.wztau2lnu_inclusive import wzinclusive_processor
    proc = wzinclusive_processor(era="2024", version="v15", run_period="")
    original, record = _install_recorder(proc)
    meta = {"dataset": "WZto3LNu_flashsim", "is_data": False, "is_signal": False, "is_flashsim": True}
    uproot_float16_patch.enable()
    try:
        events = _load(FLASHSIM_FILE, 2000, schema=flashsim_schema(), metadata=meta)
        out = proc.process(events)
        args = record["args"]
        ref = _load(FLASHSIM_FILE, 2000, schema=flashsim_schema(), metadata=meta)
        assert args["corrt1_jets"] is None, "FlashSim corrt1_jets is not None"
        assert args["raw_met"] is not None, "FlashSim raw_met not passed"
        assert np.array_equal(_np(args["raw_met"].pt), _np(ref.RawPuppiMET.pt)), "FlashSim raw_met is not RawPuppiMET"
        assert "JER" in args["jets_L123"].fields, "FlashSim jets_L123 lack JER"
        assert "JER" in record["out"].fields, "FlashSim MET lacks JER"
        rho = ref.Rho.fixedGridRhoFastjetAll if "Rho" in ref.fields else ref.fixedGridRhoFastjetAll
        direct = proc._jmeu.corrected_jets_L123_JER(ref.Jet, rho, {}, run=ref.run, event_id=ref.event)
        assert np.array_equal(_flat(args["jets_L123"].pt), _flat(direct.pt)), "FlashSim recorded jets_L123 pt != direct"
        nominal = out["WZto3LNu_flashsim"]["met_pt"][{"systematic": "nominal"}].sum(flow=True).value
        syst = _systematics(out, "WZto3LNu_flashsim")
        print(f"  FlashSim nominal weighted sum = {nominal:.4g}; systematic categories = {len(syst)}")
        assert nominal > 0, "no FlashSim event passed selection in 2000 events"
        assert {"JERUp", "JERDown"} <= syst, f"FlashSim systematic axis lacks JER: {sorted(syst)}"
    finally:
        uproot_float16_patch.disable()


def test_jetmet_mitigation_table():
    run3 = {"jer_genmatched_only_eta": ((2.5, 3.0),), "jet_pt_min": (((2.5, 3.0), 50.0), ((3.0, 5.0), 50.0)), "residual_pt_floor": None}
    assert jetmet_mitigations("2022") == run3
    assert jetmet_mitigations("2023") == run3
    assert jetmet_mitigations("2024") == {
        "jer_genmatched_only_eta": ((2.5, 3.0),), "jet_pt_min": (((2.5, 3.0), 50.0),), "residual_pt_floor": (((2.0, 2.5),), 30.0),
    }
    for era in ("2016", "2017", "2018", "2025", "2026", None):
        assert jetmet_mitigations(era) == {"jer_genmatched_only_eta": (), "jet_pt_min": (), "residual_pt_floor": None}, era
    assert set(JETMET_RUN3_MITIGATIONS) == {"2022", "2023", "2024"}
    pt = np.array([20, 20, 60, 20, 20, 50.0, np.nextafter(50.0, np.inf), 20, 20, 20, 20], dtype=np.float64)
    eta = np.array([0.5, 2.7, -2.8, 3.5, -2.2, 2.6, 2.6, 2.5, np.nextafter(np.float32(2.5), np.float32(0)), 3.0, 5.0], dtype=np.float32)
    assert pt[6] != 50.0 and eta[8] != np.float32(2.5), "near-boundary point collapsed onto the boundary"
    jets = ak.zip({"pt": ak.unflatten(pt, [len(pt)]), "eta": ak.unflatten(eta, [len(eta)])})
    T, F = True, False
    expected = {
        "2022": [T, F, T, F, T, F, T, F, T, F, T],
        "2023": [T, F, T, F, T, F, T, F, T, F, T],
        "2024": [T, F, T, T, T, F, T, F, T, T, T],
        "2018": [T] * 11, "2025": [T] * 11, None: [T] * 11,
    }
    for era, exp in expected.items():
        assert ak.to_list(forward_jet_pt_mask(jets, era)[0]) == exp, f"{era}: {ak.to_list(forward_jet_pt_mask(jets, era)[0])}"
    eta_j = ak.unflatten(eta, [len(eta)])
    assert not ak.any(abs_eta_in_regions(eta_j, ())), "empty regions select jets"
    assert ak.to_list(abs_eta_in_regions(eta_j, ((2.5, 3.0),))) == ak.to_list(_regions_ind(eta_j, HE_IND))
    assert _clib(True, True)._jer_smear[4] == ((2.5, 3.0),)


def _keyed(jets):
    return ak.with_field(ak.with_field(ak.with_field(jets, ak.values_astype(ak.full_like(jets.pt, 6), np.int32), "jetId"),
                                       ak.local_index(jets.pt), "idx"),
                         ak.broadcast_arrays(ak.local_index(jets.pt, axis=0), jets.pt)[0], "evt")


def _keys(jets):
    return set(zip(_flat_i64(jets.evt).tolist(), _flat_i64(jets.idx).tolist()))


def _build(jets, ev, era):
    return wzproc.build_jets(jets, ev.Muon[:, :0], ev.Tau[:, :0], "M", _stub_btag(), nanoAODversion="v15", era=era)[0]


def test_build_jets_forward_pt_cut():
    ev = _events("mc")
    jets = _keyed(_smeared(ev))
    k18 = _build(jets, ev, "2018")
    base = jets[(jets.pt > 30) & (np.abs(jets.eta) < 4.7) & (jets.jetId >= 6)]
    for f in ("evt", "idx"):
        assert np.array_equal(_flat_i64(k18[f]), _flat_i64(base[f])), f"2018 selection {f} differs from the pre-existing mask"
    assert np.array_equal(_flat(k18.pt), _flat(base.pt)), "2018 selection pt differs from the pre-existing mask"
    he_low = base[_regions_ind(base.eta, HE_IND) & (base.pt <= 50)]
    hf_low = base[_regions_ind(base.eta, ((3.0, 5.0),)) & (base.pt <= 50)]
    removed24 = _keys(k18) - _keys(_build(jets, ev, "2024"))
    print(f"  MC 2024 removals: {len(removed24)}; 2022 extra HF removals: {len(_keys(hf_low))}")
    assert removed24 == _keys(he_low), "2024 removals != HE jets with pt <= 50"
    assert len(removed24) >= 80, f"too few MC HE removals: {len(removed24)}"
    assert _keys(_build(jets, ev, "2022")) == _keys(k18) - _keys(he_low) - _keys(hf_low), "2022 selection != HE+HF cut"
    assert len(_keys(hf_low)) >= 90, f"too few HF removals: {len(_keys(hf_low))}"
    evd = _events("dataC")
    djets = _keyed(_clib(False, False).corrected_jets_L123_JER(evd.Jet, _rho(evd), {}, run=evd.run))
    removed_d = _keys(_build(djets, evd, "2018")) - _keys(_build(djets, evd, "2024"))
    print(f"  data C 2024 removals: {len(removed_d)}")
    assert len(removed_d) >= 25, f"too few data C HE removals: {len(removed_d)}"
    jes = _keyed(_smeared(ev).JES_Total.up)
    j18 = _build(jes, ev, "2018")
    removed_jes = _keys(j18) - _keys(_build(jes, ev, "2024"))
    assert removed_jes == _keys(j18[_regions_ind(j18.eta, HE_IND) & (j18.pt <= 50)]), "JES-up removals != HE JES-up jets with pt <= 50"
    sym = removed_jes ^ removed24
    print(f"  JES_Total.up removals {len(removed_jes)}; symmetric difference with nominal {len(sym)}")
    assert len(sym) >= 1, "the cut does not follow the shifted pT"
    with pytest.raises(TypeError, match=r"missing 1 required keyword-only argument: 'era'"):
        wzproc.build_jets(jets, ev.Muon[:, :0], ev.Tau[:, :0], "M", _stub_btag(), nanoAODversion="v15")


def test_data_residual_pt_floor():
    with gzip.open(str(_handler().getPath("jet_jerc")), "rt") as f:
        payload = json.load(f)
    comp = [c for c in payload["compound_corrections"] if c["name"] == f"{JEC_TAG}_DATA_L1L2L3Res_{JET_TYPE}"]
    assert len(comp) == 1, "data compound not found"
    comp = comp[0]
    assert comp["stack"] == [f"{JEC_TAG}_DATA_{lvl}_{JET_TYPE}" for lvl in ("L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual")], comp["stack"]
    assert comp["inputs_update"] == ["JetPt"] and comp["input_op"] == "*" and comp["output_op"] == "*", comp
    cset = _cset()
    compound = cset.compound[f"{JEC_TAG}_DATA_L1L2L3Res_{JET_TYPE}"]
    jm = _clib(False, False)
    for key in ("dataC", "dataI"):
        ev = _events(key)
        rho = _rho(ev)
        fj = _floor_ind(ev.Jet, rho, ev.run)
        fc = _floor_ind(ev.CorrT1METJet, rho, ev.run, corrt1=True)
        _, vj = _jec_values(ev.Jet, rho, run=ev.run)
        _, vc = _jec_values(ev.CorrT1METJet, rho, run=ev.run, corrt1=True)
        comp_j = _flat(_eval_named(compound, vj))
        comp_c = _flat(_eval_named(compound, vc))
        assert np.array_equal(_flat(fj["f_chain"]), comp_j), f"{key}: Jet chain != compound"
        assert np.array_equal(_flat(fc["f_chain"]), comp_c), f"{key}: CorrT1 chain != compound"
        jets = jm.corrected_jets_L123_noJER(ev.Jet, rho, {}, run=ev.run)
        jec = _flat(jets.jet_energy_correction)
        a = _flat(fj["A"]).astype(bool)
        print(f"  {key}: A Jets {int(a.sum())} of {len(a)}")
        assert a.sum() >= 150, f"{key}: too few A jets: {a.sum()}"
        _assert_close(jec[a], _flat(fj["f_floor"])[a], f"{key}: A jets use the residual at 30 GeV")
        frac, n = _frac_differ(jec[a], comp_j[a])
        assert frac > 0.5, f"{key}: A jets equal the compound: {frac:.3f}"
        assert np.array_equal(jec[~a], comp_j[~a]), f"{key}: non-A jets differ from the compound"
        rel = jec[a] / comp_j[a] - 1
        print(f"  {key}: floor/compound - 1 on A: {_quantiles(np.abs(rel), (50, 84, 99))}; signed median {np.median(rel):.4g}")
        jets_jer = jm.corrected_jets_L123_JER(ev.Jet, rho, {}, run=ev.run)
        assert np.array_equal(_flat(jets_jer.pt), _flat(jets.pt)), f"{key}: _JER and _noJER data jets differ"
        ct1_f = _flat(jm.met_factory.jec_L1L2L3.getCorrection(**{k: vc[k] for k in jm.met_factory.jec_L1L2L3.signature}))
        _assert_close(ct1_f, _flat(fc["f_floor"]), f"{key}: CorrT1 adapter vs independent floor")
        ac = _flat(fc["A"]).astype(bool)
        print(f"  {key}: A CorrT1 {int(ac.sum())}, MET-selected {_met_selected_corrt1_data(ev, ac)}")
        assert ac.sum() >= 150, f"{key}: too few A CorrT1 jets: {ac.sum()}"
        frac, n = _frac_differ(ct1_f[ac], comp_c[ac])
        assert frac > 0.5, f"{key}: A CorrT1 factors equal the compound: {frac:.3f}"
        met = _clib_met(jm, ev, jets, run=ev.run)
        ref = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, _type1_factors(ev, False, run=ev.run))
        _assert_met_close(met, ref, f"{key}: Type-1 with floor")
        neg = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, _type1_factors(ev, False, run=ev.run, floor=False))
        n_neg = int(_met_mismatch(met.pt, met.phi, neg[0], neg[1]).sum())
        d = np.abs(_np(met.pt) - neg[0])
        print(f"  {key}: compound control mismatches {n_neg}; |dMET| floor vs compound {_quantiles(d, (84, 95, 99, 100))}; mean {np.mean(_np(met.pt) - neg[0]):.4g} GeV")
        assert n_neg > 100, f"{key}: compound control not distinguishable: {n_neg}"
        ctl = reference_type1(ev.RawPuppiMET, ev.Jet, ev.CorrT1METJet, _type1_factors(ev, False, run=ev.run, corrt1_floor=False))
        n_ctl = int(_met_mismatch(met.pt, met.phi, ctl[0], ctl[1]).sum())
        print(f"  {key}: CorrT1 control (compound CorrT1) mismatches {n_ctl}")
        if key == "dataC":
            assert n_ctl > 50, f"{key}: CorrT1 floor control not distinguishable: {n_ctl}"
    assert jm.met_factory.jec_L1L2L3 is jm.jec_stack_L123_JER.jec is jm.jec_stack_L123_noJER.jec, "data JEC objects differ"
    for mc in (_clib(True, False), _clib(True, True)):
        assert type(mc.jec_stack_L123_JER.jec) is CorrectionLibJEC, "MC JEC is not the plain compound"
        assert type(mc.met_factory.jec_L1L2L3) is CorrectionLibJEC, "MC MET JEC is not the plain compound"


def _met_selected_corrt1_data(ev, mask):
    ct1 = ev.CorrT1METJet
    f123, _ = _clib_factors(ct1, _rho(ev), False, run=ev.run, corrt1=True)
    nomu = _flat(ak.values_astype(ct1.rawPt * (1.0 - ct1.muonSubtrFactor), np.float64))
    sel = (nomu * _flat(ak.values_astype(f123, np.float32)) > 15.0) & (np.abs(_flat(ct1.eta)) < 5.2) & (_flat(ct1.EmEF) < 0.9)
    return int((sel & mask).sum())


def test_empty_chunk_no_genjet_no_corrt1(monkeypatch):
    for jm, key in ((_clib(True, True), "mc"), (_clib(False, False), "dataC")):
        ev = _events(key)[:0]
        run = ev.run if key == "dataC" else None
        rec = {}
        orig = jm.met_factory.build

        def rec_build(*args, _rec=rec, _orig=orig, **kwargs):
            _rec["ct1"] = kwargs.get("in_CorrT1METJets")
            return _orig(*args, **kwargs)

        monkeypatch.setattr(jm.met_factory, "build", rec_build)
        jets = jm.corrected_jets_L123_JER(ev.Jet, _rho(ev), {}, run=run, event_id=ev.event)
        met = _clib_met(jm, ev, jets, run=run)
        assert len(jets) == 0 and jets.ndim == 2, f"{key}: empty-chunk jets {jets.type}"
        assert len(met) == 0 and len(met.pt) == 0, f"{key}: empty-chunk MET {met.type}"
        assert len(rec["ct1"]) == 0 and rec["ct1"].ndim == 2, f"{key}: empty-chunk CorrT1 {rec['ct1'].type}"
        if key == "mc":
            assert "JER" in jets.fields and "jet_energy_resolution_correction" in rec["ct1"].fields, "MC empty chunk lost JER"
            assert len(met.JER.up.pt) == 0 and len(met.JES_Total.up.pt) == 0, "MC empty-chunk MET variations"
            gen = _nearest_gen_pt(ev.CorrT1METJet, ev.GenJet, 0.2)
            assert len(gen) == 0 and gen.ndim == 2, f"empty-chunk _nearest_gen_pt {gen.type}"
        monkeypatch.undo()
    ev = _events("mc")[:200]
    nogen = ev.GenJet[:, :0]
    gen = _nearest_gen_pt(ev.CorrT1METJet, nogen, 0.2)
    assert np.array_equal(_np(ak.num(gen)), _np(ak.num(ev.CorrT1METJet))), "no-GenJet _nearest_gen_pt counts"
    assert np.array_equal(_flat(gen), np.zeros(int(ak.sum(ak.num(ev.CorrT1METJet))))), "no-GenJet _nearest_gen_pt is not 0"
    jm = _clib(True, True)
    rec = {}
    orig = jm.met_factory.build

    def rec_build(*args, **kwargs):
        rec["ct1"] = kwargs.get("in_CorrT1METJets")
        return orig(*args, **kwargs)

    monkeypatch.setattr(jm.met_factory, "build", rec_build)
    jets = _smeared(ev, jm)
    met_in, raw = _met_inputs(ev)
    met = jm.corrected_met(met_in, jets, _rho(ev), {}, raw_met=raw, corrt1_jets=ev.CorrT1METJet, event_id=ev.event, gen_jets=nogen)
    monkeypatch.undo()
    assert len(met) == len(ev) and np.isfinite(_np(met.pt)).all(), "no-GenJet MET"
    C = _ct1_jer_inputs(ev)
    minus1 = np.full_like(C["pt_jec"], -1.0)
    obs = {
        "nom": _flat(rec["ct1"].jet_energy_resolution_correction),
        "up": _flat(rec["ct1"].JER.up.jet_energy_resolution_correction),
        "down": _flat(rec["ct1"].JER.down.jet_energy_resolution_correction),
    }
    assert len(obs["nom"]) >= 100, f"too few CorrT1 jets: {len(obs['nom'])}"
    for v in VARIATIONS:
        s = _jersmear_ind(C["pt_jec"], C["eta"], minus1, C["rho"], C["eid"], C["sigma"], C["sf"][v])
        _assert_close(obs[v], np.where(C["he"], 1.0, _guard(s)), f"no-GenJet CorrT1 factor {v}")
    for jm, key in ((_clib(True, True), "mc"), (_clib(False, False), "dataC")):
        ev = _events(key)[:200]
        run = ev.run if key == "dataC" else None
        jets = jm.corrected_jets_L123_JER(ev.Jet, _rho(ev), {}, run=run, event_id=ev.event)
        met_in, raw = _met_inputs(ev)
        gen_jets = ev.GenJet if key == "mc" else None
        empty = jm.corrected_met(met_in, jets, _rho(ev), {}, raw_met=raw, corrt1_jets=ev.CorrT1METJet[:, :0], run=run,
                                 event_id=ev.event, gen_jets=gen_jets)
        none = jm.corrected_met(met_in, jets, _rho(ev), {}, raw_met=raw, corrt1_jets=None, run=run)
        assert np.array_equal(_np(empty.pt), _np(none.pt)), f"{key}: no-CorrT1 MET differs from corrt1_jets=None"
        assert np.array_equal(_np(empty.phi), _np(none.phi)), f"{key}: no-CorrT1 MET phi differs from corrt1_jets=None"
        if key == "mc":
            assert np.array_equal(_np(empty.JER.up.pt), _np(none.JER.up.pt)), "MC no-CorrT1 JERUp MET differs"
