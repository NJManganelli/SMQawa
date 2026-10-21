import os
import warnings

import numpy as np
import awkward as ak
import pytest
import scipy.stats
import correctionlib.schemav2 as schemav2
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import qawa
import qawa.egammaSS
import qawa.process.wztau2lnu_inclusive as wzproc
from qawa.jsoncorrections import CorrectionlibHandler

expected_root = os.environ.get("QAWA_EXPECTED_ROOT")
if expected_root and not os.path.realpath(qawa.__file__).startswith(os.path.realpath(expected_root)):
    pytest.exit(f"qawa imported from {qawa.__file__}, expected under {expected_root}", returncode=3)

warnings.filterwarnings("ignore")
NanoAODSchema.warn_missing_crossrefs = False

MC = "/Volumes/WDMac/SMQawaTestFiles/store/mc/RunIII2024Summer24NanoAODv15/DYto2E-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/110000/test.root"
DATA = "/Users/nmangane/vz-main/SMQawaTestFiles/store/data/Run2024G/EGamma0/NANOAOD/MINIv6NANOv15-v2/120000/test.root"

needs_mc = pytest.mark.skipif(not os.path.exists(MC), reason=f"missing {MC}")
needs_data = pytest.mark.skipif(not os.path.exists(DATA), reason=f"missing {DATA}")

V15_ERAS = [("2016", True), ("2016", False), ("2017", False), ("2018", False), ("2024", False), ("2025", False)]
MC_FIELDS = ["pt", "energyErr", "pt_smear_up", "pt_smear_down", "pt_scale_up", "pt_scale_down",
             "energyErr_smear_up", "energyErr_smear_down", "energyErr_scale_up", "energyErr_scale_down"]


def f(x):
    return ak.to_numpy(ak.flatten(x))


def load(path, start=None, stop=20000, mode="eager", is_data=False):
    metadata = {"dataset": "EGamma0" if is_data else "DY", "is_data": is_data, "is_signal": False}
    return NanoEventsFactory.from_root({path: "Events"}, schemaclass=NanoAODSchema, mode=mode,
                                       entry_start=start, entry_stop=stop, metadata=metadata).events()


def egm(events, handler, restriction=None):
    return qawa.egammaSS.EGM_scale_and_smear(None, events=events, is_correction=True, unc_type=None, restriction=restriction,
                                             is_electron=True, clibhandler=handler)


def reference(events):
    el = events.Electron
    sc = el.superclusterEta if "superclusterEta" in el.fields else el.eta + el.deltaEtaSC
    return {
        "pt_raw": f(el.pt), "eta": f(el.eta), "phi": f(el.phi), "r9": f(el.r9), "sc": f(sc),
        "energy": f(el.energy), "energyErr": f(el.energyErr), "gain": f(el.seedGain), "mva": f(el.mvaIso_WP90),
        "run": f(ak.broadcast_arrays(events.run, el.pt)[0]), "event": f(ak.broadcast_arrays(events.event, el.pt)[0]),
        "counts": ak.to_numpy(ak.num(el)),
    }


def corrected(events, fields):
    return {name: f(events.Electron[name]) for name in fields}


def widths(handler, ref):
    sas = handler.getCorrectionSet("electronSS_EtDependent")["SmearAndSyst"]
    args = (ref["pt_raw"], ref["r9"], ref["sc"])
    return {name: sas.evaluate(name, *args) for name in ("smear", "smear_up", "smear_down", "scale_up", "scale_down")}


def data_scale(handler, ref):
    cset = handler.getCorrectionSet("electronSS_EtDependent")
    s = cset.compound["Scale"].evaluate("scale", ref["run"], ref["sc"], ref["r9"], ref["pt_raw"], ref["gain"])
    rho = cset["SmearAndSyst"].evaluate("smear", ref["pt_raw"] * s, ref["r9"], ref["sc"])
    return s, rho


def hash_z(handler, ref):
    node = handler.getCorrectionSet("electron_smear_random")["electron_smear_random"]
    return node.evaluate(ref["event"], ref["run"], ref["pt_raw"], ref["eta"], ref["phi"])


@pytest.fixture(scope="module")
def handler():
    return CorrectionlibHandler(era="2024", subera=None, analysis="inc-WZ", nanoAODversion="v15", cvmfs_head="/cvmfs/")


@pytest.fixture(scope="module")
def mc_events():
    return load(MC)


@pytest.fixture(scope="module")
def data_events():
    return load(DATA, is_data=True)


def test_payload():
    met = 0
    for era, is_apv in V15_ERAS:
        h = CorrectionlibHandler(era=era, subera=None, isAPV=is_apv, analysis="inc-WZ", nanoAODversion="v15", cvmfs_head="/cvmfs/")
        if "electronSS_EtDependent" not in h.keys():
            continue
        met += 1
        assert "electron_smear_random" in h.keys(), (era, is_apv)
        assert "qawa/data/egm" in str(h["electron_smear_random"]), h["electron_smear_random"]
        cset = h.getCorrectionSet("electron_smear_random")
        assert cset is not None
        assert "electron_smear_random" in cset
    assert met >= 6, met
    h = CorrectionlibHandler(era="2024", subera=None, analysis="inc-WZ", nanoAODversion="v15", cvmfs_head="/cvmfs/")
    path = str(h["electron_smear_random"])
    assert os.path.exists(path), path
    with open(path) as fin:
        model = schemav2.CorrectionSet.model_validate_json(fin.read())
    assert [c.name for c in model.corrections] == ["electron_smear_random"]
    corr = model.corrections[0]
    assert corr.data.nodetype == "hashprng"
    assert corr.data.distribution == "normal"
    assert [v.name for v in corr.inputs] == ["event", "run", "pt", "eta", "phi"]
    assert list(corr.data.inputs) == ["event", "run", "pt", "eta", "phi"]


@needs_mc
def test_z_source(handler, mc_events):
    ref = reference(ak.Array(mc_events))
    ev = egm(ak.Array(mc_events), handler)
    out = corrected(ev, ["pt"])
    app = ref["pt_raw"] >= 15.0
    sigma = widths(handler, ref)["smear"]
    z_h = hash_z(handler, ref)
    np.testing.assert_allclose(out["pt"][app], (ref["pt_raw"] * (1 + sigma * z_h))[app], rtol=1e-12, atol=0)


@needs_mc
def test_z_normal(handler, mc_events):
    ref = reference(ak.Array(mc_events))
    ev = egm(ak.Array(mc_events), handler)
    out = corrected(ev, ["pt"])
    app = ref["pt_raw"] >= 15.0
    sigma = widths(handler, ref)["smear"]
    z_all = np.where(app, (out["pt"] / ref["pt_raw"] - 1) / sigma, np.nan)
    z = z_all[app]
    n = len(z)
    assert n > 1000, n
    assert len(np.unique(np.round(z, 9))) >= 0.99 * n
    assert abs(z.mean()) < 5 / np.sqrt(n), z.mean()
    assert abs(z.std() - 1) < 5 / np.sqrt(2 * n), z.std()
    assert scipy.stats.kstest(z, "norm").pvalue > 1e-4
    zj = ak.unflatten(z_all, ref["counts"])
    zj = zj[ak.num(zj) >= 2]
    z1, z2 = ak.to_numpy(zj[:, 0]), ak.to_numpy(zj[:, 1])
    both = np.isfinite(z1) & np.isfinite(z2)
    n_pairs = int(both.sum())
    assert n_pairs > 100, n_pairs
    assert abs(np.corrcoef(z1[both], z2[both])[0, 1]) < 5 / np.sqrt(n_pairs)


@needs_mc
def test_chunk(handler):
    a_ev = load(MC, stop=3000)
    b_ev = load(MC, start=1234, stop=3000)
    offset = int(ak.sum(ak.num(a_ev.Electron[:1234])))
    sc_a = reference(a_ev)["sc"]
    sc_b = reference(b_ev)["sc"]
    seed_a = abs(np.float32(sc_a[0]).view("int32"))
    seed_b = abs(np.float32(sc_b[0]).view("int32"))
    z_a = np.random.default_rng(seed_a).normal(size=len(sc_a))[offset:]
    z_b = np.random.default_rng(seed_b).normal(size=len(sc_b))
    assert seed_a != seed_b
    assert len(z_a) == len(z_b)
    assert not np.array_equal(z_a, z_b)
    fields = ["pt", "pt_smear_up", "pt_smear_down", "pt_scale_up", "pt_scale_down"]
    a = corrected(egm(a_ev, handler), fields)
    b = corrected(egm(b_ev, handler), fields)
    for name in fields:
        assert np.array_equal(a[name][offset:], b[name]), name


@needs_mc
def test_repeat(handler):
    first = corrected(egm(load(MC, stop=3000), handler), MC_FIELDS)
    second = corrected(egm(load(MC, stop=3000), handler), MC_FIELDS)
    for name in MC_FIELDS:
        assert np.array_equal(first[name], second[name]), name


@needs_mc
def test_smear_var(handler, mc_events):
    ref = reference(ak.Array(mc_events))
    out = corrected(egm(ak.Array(mc_events), handler), MC_FIELDS)
    app = ref["pt_raw"] >= 15.0
    w = widths(handler, ref)
    assert w["smear"][app].min() > 0
    z = (out["pt"] / ref["pt_raw"] - 1) / w["smear"]
    for direction in ("up", "down"):
        s = w[f"smear_{direction}"]
        factor = 1 + s * z
        np.testing.assert_allclose(out[f"pt_smear_{direction}"][app], (ref["pt_raw"] * factor)[app], rtol=1e-10, atol=0)
        expected = np.sqrt(ref["energyErr"].astype(np.float64)**2 + (ref["energy"] * s)**2) * factor
        np.testing.assert_allclose(out[f"energyErr_smear_{direction}"][app], expected[app], rtol=1e-10, atol=0)
    shift = (out["pt_smear_up"] - out["pt"])[app]
    assert (shift > 0).mean() > 0.3, (shift > 0).mean()
    assert (shift < 0).mean() > 0.3, (shift < 0).mean()


@needs_mc
def test_scale_var(handler, mc_events):
    ref = reference(ak.Array(mc_events))
    out = corrected(egm(ak.Array(mc_events), handler), MC_FIELDS)
    app = ref["pt_raw"] >= 15.0
    w = widths(handler, ref)
    for direction in ("up", "down"):
        s = w[f"scale_{direction}"]
        np.testing.assert_allclose(out[f"pt_scale_{direction}"][app], (out["pt"] * s)[app], rtol=1e-12, atol=0)
        np.testing.assert_allclose(out[f"energyErr_scale_{direction}"][app], (out["energyErr"] * s)[app], rtol=1e-12, atol=0)


@needs_data
def test_data_eerr(handler, data_events):
    ref = reference(ak.Array(data_events))
    out = corrected(egm(ak.Array(data_events), handler), ["pt", "energyErr"])
    app = ref["pt_raw"] >= 15.0
    s, rho = data_scale(handler, ref)
    assert np.max(np.abs(s[app] - 1)) > 1e-3
    np.testing.assert_allclose(out["pt"][app], (ref["pt_raw"] * s)[app], rtol=1e-12, atol=0)
    expected = np.sqrt(ref["energyErr"].astype(np.float64)**2 + (ref["energy"] * rho)**2) * s
    np.testing.assert_allclose(out["energyErr"][app], expected[app], rtol=1e-9, atol=0)


@needs_mc
def test_restrict(handler, mc_events):
    ref = reference(ak.Array(mc_events))
    ev = ak.Array(mc_events)
    in_region = np.abs(ref["sc"]) < 1.4442
    ev["Electron", "isScEtaEB"] = ak.unflatten(in_region, ref["counts"])
    out = corrected(egm(ev, handler, restriction="EB"), MC_FIELDS)
    unrestricted = corrected(egm(ak.Array(mc_events), handler), MC_FIELDS)
    outside = ~in_region
    assert outside.sum() > 100 and in_region.sum() > 100
    for direction in ("up", "down"):
        assert np.array_equal(out[f"pt_scale_{direction}"][outside], out["pt"][outside]), direction
        assert np.array_equal(out[f"energyErr_scale_{direction}"][outside], out["energyErr"][outside]), direction
        assert np.array_equal(out[f"pt_smear_{direction}"][outside], out["pt"][outside]), direction
    for name in MC_FIELDS:
        assert np.array_equal(out[name][in_region], unrestricted[name][in_region]), name


@needs_mc
@needs_data
def test_guard(handler, mc_events, data_events):
    mc_ref = reference(ak.Array(mc_events))
    mc = corrected(egm(ak.Array(mc_events), handler), MC_FIELDS)
    data_ref = reference(ak.Array(data_events))
    data = corrected(egm(ak.Array(data_events), handler), ["pt", "energyErr"])

    mc_raw = mc_ref["pt_raw"]
    below = mc_raw < 15.0
    window = (mc_raw >= 14.0) & (mc_raw < 15.0)
    assert window.sum() >= 100, window.sum()
    for mask in (window, below):
        assert np.array_equal(mc["pt"][mask], mc_raw[mask])
        np.testing.assert_allclose(mc["energyErr"][mask], mc_ref["energyErr"][mask], rtol=1e-7, atol=0)
        for name in ("pt_smear_up", "pt_smear_down", "pt_scale_up", "pt_scale_down"):
            assert np.array_equal(mc[name][mask], mc["pt"][mask]), name
        for name in ("energyErr_smear_up", "energyErr_smear_down", "energyErr_scale_up", "energyErr_scale_down"):
            assert np.array_equal(mc[name][mask], mc["energyErr"][mask]), name

    data_raw = data_ref["pt_raw"]
    below = data_raw < 15.0
    window = (data_raw >= 14.0) & (data_raw < 15.0)
    assert window.sum() >= 100, window.sum()
    for mask in (window, below):
        assert np.array_equal(data["pt"][mask], data_raw[mask])
        np.testing.assert_allclose(data["energyErr"][mask], data_ref["energyErr"][mask], rtol=1e-7, atol=0)

    app = data_raw >= 15.0
    window = (data_raw >= 15.0) & (data_raw < 16.0)
    assert window.sum() >= 100, window.sum()
    s, rho = data_scale(handler, data_ref)
    for mask in (window, app):
        np.testing.assert_allclose(data["pt"][mask], (data_raw * s)[mask], rtol=1e-12, atol=0)
        assert (data["pt"][mask] != data_raw[mask]).mean() >= 0.99

    app = mc_raw >= 15.0
    window = (mc_raw >= 15.0) & (mc_raw < 16.0)
    assert window.sum() >= 100, window.sum()
    z_h = hash_z(handler, mc_ref)
    sigma = widths(handler, mc_ref)["smear"]
    for mask in (window, app):
        np.testing.assert_allclose(mc["pt"][mask], (mc_raw * (1 + sigma * z_h))[mask], rtol=1e-12, atol=0)
        assert np.all(mc["pt"][mask] != mc_raw[mask])
        assert np.all(mc["pt_scale_up"][mask] != mc["pt"][mask])


@needs_mc
def test_guard_edge(handler, mc_events, monkeypatch):
    raw = reference(ak.Array(mc_events))["pt_raw"]
    k = int(np.flatnonzero(raw == raw[raw >= 25.0].min())[0])
    j = int(np.flatnonzero(raw == raw[raw < raw[k]].max())[0])
    assert raw[j] > 15
    monkeypatch.setattr(qawa.egammaSS, "EGM_SS_MIN_PT", float(raw[k]), raising=False)
    pt = corrected(egm(ak.Array(mc_events), handler), ["pt"])["pt"]
    assert pt[k] != raw[k]
    assert pt[j] == raw[j]


@needs_mc
@needs_data
def test_guard_yield(handler, mc_events, data_events, monkeypatch):
    samples = {"MC": mc_events, "DATA": data_events}
    refs = {name: reference(ak.Array(events)) for name, events in samples.items()}
    runs = {name: {} for name in samples}
    for name, events in samples.items():
        runs[name]["G"] = corrected(egm(ak.Array(events), handler), ["pt"])["pt"]
    for name, events in samples.items():
        with monkeypatch.context() as m:
            m.setattr(qawa.egammaSS, "EGM_SS_MIN_PT", -1.0, raising=False)
            runs[name]["N"] = corrected(egm(ak.Array(events), handler), ["pt"])["pt"]
        with monkeypatch.context() as m:
            m.setattr(qawa.egammaSS, "EGM_SS_MIN_PT", 20.0, raising=False)
            runs[name]["G20"] = corrected(egm(ak.Array(events), handler), ["pt"])["pt"]
    for name in samples:
        ref = refs[name]
        abs_sc = np.abs(ref["sc"])
        tight_like = ref["mva"] & ((abs_sc < 1.4442) | ((abs_sc > 1.5660) & (abs_sc < 2.5)))
        counts = {}
        for run, pt in runs[name].items():
            tight_low_bin = (pt > 20) & (pt < 25)
            counts[run] = int((tight_low_bin & tight_like).sum())
        print(f"{name}: tight-like 20-25 GeV electrons G={counts['G']} N={counts['N']} G20={counts['G20']} "
              f"G/N-1={counts['G'] / counts['N'] - 1:+.4f} G20/N-1={counts['G20'] / counts['N'] - 1:+.4f}")
    for name in samples:
        g, n = runs[name]["G"], runs[name]["N"]
        assert np.array_equal(g > 10, refs[name]["pt_raw"] > 10), name
        assert np.array_equal((g > 20) & (g < 25), (n > 20) & (n < 25)), name
    assert getattr(qawa.egammaSS, "EGM_SS_MIN_PT", None) == 15.0


@needs_mc
def test_empty(handler, mc_events):
    ev = ak.Array(mc_events)
    sub = ev[ak.num(ev.Electron) == 0][:50]
    assert len(sub) == 50
    out = egm(sub, handler)
    for name in ["pt_orig", "energyErr_orig"] + MC_FIELDS:
        assert name in out.Electron.fields, name


def run_processor(events, run_period):
    p = wzproc.wzinclusive_processor(era="2024", run_period=run_period, version="v15")
    seen = {}

    def record(event, shift_name=""):
        seen[shift_name] = event
        return {}

    p.process_shift = record
    p.process(events)
    return seen


@needs_mc
@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_proc_mc(handler, mode):
    seen = run_processor(load(MC, stop=500, mode=mode), "")
    refev = egm(load(MC, stop=500, mode=mode), handler)
    ref = corrected(refev, MC_FIELDS)
    nominal = seen[None].Electron
    assert np.array_equal(f(nominal.pt), ref["pt"])
    for name in ["pt_orig", "pt_scale_up", "pt_scale_down", "pt_smear_up", "pt_smear_down"]:
        assert name in nominal.fields, name
    electron_shifts = {"scale_eUp", "scale_eDown", "res_eUp", "res_eDown"}
    assert electron_shifts <= set(seen)
    for name, event in seen.items():
        if name not in electron_shifts:
            assert np.array_equal(f(event.Electron.pt), ref["pt"]), name
    for shift, field in [("scale_eUp", "scale_up"), ("scale_eDown", "scale_down"), ("res_eUp", "smear_up"), ("res_eDown", "smear_down")]:
        kind, direction = field.split("_")
        assert np.array_equal(f(seen[shift].Electron.pt), ref[f"pt_{kind}_{direction}"]), shift
        assert np.array_equal(f(seen[shift].Electron.energyErr), ref[f"energyErr_{kind}_{direction}"]), shift
    met_nom = seen[None].MET
    for shift, field in [("scale_eUp", "pt_scale_up"), ("res_eUp", "pt_smear_up")]:
        dpt = refev.Electron[field] - refev.Electron.pt
        x = met_nom.pt * np.cos(met_nom.phi) - ak.sum(dpt * np.cos(refev.Electron.phi), axis=1)
        y = met_nom.pt * np.sin(met_nom.phi) - ak.sum(dpt * np.sin(refev.Electron.phi), axis=1)
        np.testing.assert_allclose(ak.to_numpy(seen[shift].MET.pt), ak.to_numpy(np.hypot(x, y)), rtol=1e-5, atol=1e-4)
    below = f(nominal.pt_orig) < 15
    for name, event in seen.items():
        assert np.array_equal(f(event.Electron.pt)[below], f(nominal.pt)[below]), name


@needs_data
def test_proc_data(handler):
    seen = run_processor(load(DATA, stop=500, mode="virtual", is_data=True), "G")
    assert list(seen) == [None]
    ref = reference(load(DATA, stop=500, is_data=True))
    nominal = seen[None].Electron
    pt, energy_err = f(nominal.pt), f(nominal.energyErr)
    app = ref["pt_raw"] >= 15.0
    s, rho = data_scale(handler, ref)
    np.testing.assert_allclose(pt[app], (ref["pt_raw"] * s)[app], rtol=1e-12, atol=0)
    expected = np.sqrt(ref["energyErr"].astype(np.float64)**2 + (ref["energy"] * rho)**2) * s
    np.testing.assert_allclose(energy_err[app], expected[app], rtol=1e-9, atol=0)
    assert np.array_equal(pt[~app], ref["pt_raw"][~app])


@needs_data
def test_proc_data_met(monkeypatch):
    first = run_processor(load(DATA, stop=500, mode="virtual", is_data=True), "G")
    monkeypatch.setattr(wzproc, "EGM_scale_and_smear", lambda *a, **k: k["events"])
    second = run_processor(load(DATA, stop=500, mode="virtual", is_data=True), "G")
    assert np.array_equal(ak.to_numpy(first[None].MET.pt), ak.to_numpy(second[None].MET.pt))
    assert np.array_equal(ak.to_numpy(first[None].MET.phi), ak.to_numpy(second[None].MET.phi))
