"""BTVCorrector: BTV fixed-WP method 1a b-tag SF weights read with efficiencies from a btag_eff payload."""
import awkward as ak
import numpy as np
import pytest
from coffea.analysis_tools import Weights
from coffea.util import coffea_console

from conftest import CVMFS_HEAD, STORE, input_path
from qawa.btag import BTVCorrector, btv_shift_systematic, method1a_weight
from qawa.jsoncorrections import CorrectionlibHandler
from qawa.measurements import BTAG_EFF_FLAVOURS, BTAG_EFF_SYSTEMATICS, btag_eff_payload, write_payload
import qawa.process.wztau2lnu_inclusive as wzmod

CVMFS, needs_cvmfs = input_path(CVMFS_HEAD, "cms-griddata.cern.ch")
pytestmark = needs_cvmfs
WPS = ("L", "M", "T", "XT", "XXT")
NOMINAL_EFF = {5: (0.9, 0.8, 0.7, 0.6, 0.5), 4: (0.5, 0.3, 0.2, 0.1, 0.05), 0: (0.3, 0.1, 0.05, 0.02, 0.01)}
PAIRED_NAMES = {"bc_type3", "bc_as", "bc_bdecays", "bc_bfragmentation", "bc_fsrdef", "bc_hdamp", "bc_isrdef",
                "bc_mass", "bc_muf", "bc_mur", "bc_pdf", "bc_pileup", "bc_ttbar"}


def constant_eff(systematic, wp, flavour):
    shift = {"nominal": 0., "up": 0.02, "down": -0.02}[systematic]
    return NOMINAL_EFF[flavour][WPS.index(wp)] + shift


def eff_payload(path, value=constant_eff):
    edges = (wzmod.BTAG_PT_EDGES, wzmod.BTAG_ABSETA_EDGES)
    shape = (len(edges[0]) - 1, len(edges[1]) - 1)
    maps = {"inclusive": {syst: {wp: {flavour: np.full(shape, value(syst, wp, flavour)) for flavour in BTAG_EFF_FLAVOURS}
                                 for wp in WPS} for syst in BTAG_EFF_SYSTEMATICS}}
    write_payload(btag_eff_payload(maps, edges, WPS), path)
    return path


def handler(era, version="v15", payload=None):
    clib = CorrectionlibHandler(era=era, analysis="inc-WZ", nanoAODversion=version, cvmfs_head=CVMFS_HEAD)
    if payload is not None:
        clib["btag_eff"] = payload
    return clib


def corrector(era, tmp_path, value=constant_eff, jes_shifts=None, **kwargs):
    clib = handler(era, payload=eff_payload(tmp_path / f"btagEff_{era}.json", value))
    shifts = wzmod.jes_shift_names(era) if jes_shifts is None else jes_shifts
    return BTVCorrector(era=era, tagger="UParTAK4", clibhandler=clib, jes_shifts=shifts, **kwargs)


def threshold(btv, wp):
    return btv.clib["UParTAK4_wp_values"].evaluate(wp)


def jets(*events):
    """events: lists of (pt, eta, hadronFlavour, discriminant)."""
    return ak.Array([[dict(pt=pt, eta=eta, hadronFlavour=flav, btagUParTAK4B=disc) for pt, eta, flav, disc in event]
                     for event in events])


def sf(btv, correction, systematic, wp, flavour, pt, eta):
    return btv.clib[f"UParTAK4_{correction}"].evaluate(systematic, wp, flavour, abs(eta), pt)


def weights_for(btv, events, shift_name=None, variations=True, process="TTto2L2Nu"):
    weights = Weights(len(events), storeIndividual=True)
    btv.append_btag_sf(events, weights, process, shift_name=shift_name, variations=variations)
    return weights


def variation_names(weights):
    return {name[len("btag_sf_"):-len("Up")] for name in weights.variations if name.endswith("Up")}


def classic_event(btv, bc="central", light="central"):
    """A tagged b jet and an untagged light jet, single loose WP."""
    tagged = sf(btv, "comb", bc, "L", 5, 60., 0.5)
    lsf = sf(btv, "light", light, "L", 0, 45., -1.0)
    eps = NOMINAL_EFF[0][0]
    return tagged * (1 - lsf * eps) / (1 - eps)


def classic_jets(btv):
    return jets([(60., 0.5, 5, 0.999), (45., -1.0, 0, 0.5 * threshold(btv, "L"))])


def test_nominal_applied_once(tmp_path):
    btv = corrector("2024", tmp_path)
    weights = weights_for(btv, classic_jets(btv))
    expected = classic_event(btv)
    assert np.allclose(weights.weight(), expected, rtol=1e-12)
    assert np.allclose(weights.partial_weight(include=["btag_sf"]), expected, rtol=1e-12)


def test_variation_ratio(tmp_path):
    btv = corrector("2024", tmp_path)
    weights = weights_for(btv, classic_jets(btv))
    nominal = weights.weight()
    assert np.allclose(weights.weight("btag_sf_bc_massUp") / nominal,
                       classic_event(btv, bc="up_mass") / classic_event(btv), rtol=1e-12)
    assert np.allclose(weights.weight("btag_sf_light_correlatedUp") / nominal,
                       classic_event(btv, light="up_correlated") / classic_event(btv), rtol=1e-12)


@pytest.mark.parametrize("era", ["2024", "2025"])
def test_paired_variation_set(era, tmp_path):
    btv = corrector(era, tmp_path)
    assert btv.paired
    names = variation_names(weights_for(btv, classic_jets(btv)))
    assert names == PAIRED_NAMES | {f"bc_statistic_{era}", "light_correlated", f"light_uncorrelated_{era}", f"effstat_{era}"}
    assert not {n for n in names if "jes" in n or "jer" in n or n in ("bc_correlated", "bc_up", "bc_down")}
    assert not {n for n in names if n.startswith("bc_uncorrelated")}


def fallback_names(era):
    return {"bc_correlated", f"bc_uncorrelated_{era}", "light_correlated", f"light_uncorrelated_{era}", f"effstat_{era}"}


def test_empty_jes_shifts_not_paired(tmp_path):
    with coffea_console.capture() as captured:
        btv = corrector("2024", tmp_path, jes_shifts=())
    assert "not paired" in captured.get()
    assert variation_names(weights_for(btv, classic_jets(btv))) == fallback_names("2024")
    with pytest.raises(TypeError):
        BTVCorrector(era="2024", tagger="UParTAK4", clibhandler=handler("2024"))


def test_unmatched_scheme_falls_back(tmp_path):
    with coffea_console.capture() as captured:
        btv = corrector("2018", tmp_path)
    assert captured.get().count("not paired") == 1
    assert not btv.paired
    events = jets([(60., 0.5, 5, 0.999)])
    shifted = weights_for(btv, events, shift_name="JES_AbsoluteUp", variations=False)
    assert np.allclose(shifted.weight(), sf(btv, "comb", "central", "L", 5, 60., 0.5), rtol=1e-12)
    assert variation_names(weights_for(btv, classic_jets(btv))) == fallback_names("2018")
    with pytest.raises(ValueError):
        corrector("2018", tmp_path, bc_correction="mujets")


def test_jes_pairing(tmp_path):
    btv = corrector("2024", tmp_path)
    events = classic_jets(btv)
    for shift_name, systematic in [("JES_Absolute2024Up", "up_jes_absolute_2024"), ("JESUp", "central"),
                                   ("UESUp", "central")]:
        weights = weights_for(btv, events, shift_name=shift_name, variations=False)
        assert np.allclose(weights.weight(), classic_event(btv, bc=systematic), rtol=1e-12), shift_name
        assert not variation_names(weights)
    assert {name: btv_shift_systematic(name) for name in ("JES_BBEC1Up", "JES_BBEC12024Up", "JES_RelativeSample2025Down",
                                                        "JERDown", "HEMUp")} == {
        "JES_BBEC1Up": "up_jes_bbec1", "JES_BBEC12024Up": "up_jes_bbec1_2024",
        "JES_RelativeSample2025Down": "down_jes_relativesample_2025", "JERDown": "down_jer", "HEMUp": None}


def test_multi_wp_formula(tmp_path):
    wps = ("L", "M", "T")
    sf_ = {"L": np.full(3, 1.1), "M": np.full(3, 1.2), "T": np.full(3, 1.3)}
    eff = {"L": np.full(3, 0.6), "M": np.full(3, 0.4), "T": np.full(3, 0.2)}
    passed = {"L": np.array([False, True, True]), "M": np.array([False, False, True]), "T": np.array([False, False, True])}
    factor, guarded = method1a_weight(sf_, eff, passed, wps)
    assert np.allclose(factor, [(1 - 1.1 * 0.6) / (1 - 0.6), (1.1 * 0.6 - 1.2 * 0.4) / (0.6 - 0.4), 1.3])
    assert not guarded.any()
    single, _ = method1a_weight({"L": sf_["L"]}, {"L": eff["L"]}, {"L": passed["L"]}, ("L",))
    assert np.allclose(single, [(1 - 1.1 * 0.6) / (1 - 0.6), 1.1, 1.1])

    btv = corrector("2024", tmp_path, wps=wps)
    between = jets([(60., 0.5, 5, 0.5 * (threshold(btv, "L") + threshold(btv, "M")))])
    eps_l, eps_m = NOMINAL_EFF[5][:2]
    sf_l, sf_m = (sf(btv, "comb", "central", wp, 5, 60., 0.5) for wp in ("L", "M"))
    assert np.allclose(weights_for(btv, between).weight(), (sf_l * eps_l - sf_m * eps_m) / (eps_l - eps_m), rtol=1e-12)


def test_efficiency_guard(tmp_path):
    def degenerate(systematic, wp, flavour):
        if flavour == 0 and wp == "L":
            return 1.0
        if flavour == 5 and wp == "M":
            return constant_eff(systematic, "L", flavour)
        return constant_eff(systematic, wp, flavour)
    btv = corrector("2024", tmp_path, value=degenerate, wps=("L", "M"))
    failing = jets([(45., -1.0, 0, 0.5 * threshold(btv, "L"))])
    between = jets([(60., 0.5, 5, 0.5 * (threshold(btv, "L") + threshold(btv, "M")))])
    for events in (failing, between):
        weights = weights_for(btv, events)
        assert np.allclose(weights.weight(), 1.0)
        assert np.allclose(weights.weight("btag_sf_effstat_2024Up") / weights.weight(), 1.0)
    assert btv.guard_counts

    light_sf = sf(btv, "light", "central", "L", 0, 45., -1.0)
    eps_nominal, eps_up = 0.5 / light_sf, (1 / light_sf + 1) / 2

    def up_guarded(systematic, wp, flavour):
        if flavour == 0 and wp == "L":
            return {"nominal": eps_nominal, "up": eps_up, "down": eps_nominal - 0.02}[systematic]
        return constant_eff(systematic, wp, flavour)
    btv = corrector("2024", tmp_path, value=up_guarded)
    with coffea_console.capture() as captured:
        weights = weights_for(btv, jets([(45., -1.0, 0, 0.5 * threshold(btv, "L"))]))
    expected = (1 - light_sf * eps_nominal) / (1 - eps_nominal)
    assert np.allclose(weights.weight(), expected, rtol=1e-12)
    assert np.allclose(weights.weight("btag_sf_effstat_2024Up"), expected, rtol=1e-12)
    assert np.all(weights.weight("btag_sf_effstat_2024Up") != 0)
    assert btv.guard_counts and "non-positive" in captured.get()


def test_light_correction_name(tmp_path):
    btv = corrector("2024", tmp_path)
    events = jets([(45., -1.0, 0, 0.999)])
    assert np.allclose(weights_for(btv, events, variations=False).weight(), sf(btv, "light", "central", "L", 0, 45., -1.0),
                       rtol=1e-12)


def test_v9_unsupported():
    btv = BTVCorrector(era="2018", tagger="deepJet", clibhandler=handler("2018", "v9"), jes_shifts=())
    assert btv.efficiency_available is False
    with pytest.raises(NotImplementedError):
        btv.append_btag_sf(jets([(60., 0.5, 5, 0.9)]), Weights(1, storeIndividual=True), "TTto2L2Nu")
    with pytest.raises(ValueError, match="CorrectionlibHandler"):
        BTVCorrector(era="2018", tagger="deepJet", jes_shifts=())


def test_acceptance(tmp_path):
    btv = corrector("2018", tmp_path)
    weights = weights_for(btv, jets([(60., 2.45, 5, 0.999), (60., -2.45, 0, 0.0)]))
    assert np.allclose(weights.weight(), 1.0)


M1_FILE, needs_m1 = input_path(STORE, "mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/"
                                       "NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/test.root")


def m1_events(n=2000):
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    return NanoEventsFactory.from_root(
        {M1_FILE: "Events"}, schemaclass=NanoAODSchema, mode="virtual", entry_start=0, entry_stop=n,
        metadata={"dataset": "WZto3LNu", "is_data": False, "is_signal": False, "is_flashsim": False},
    ).events()


@pytest.fixture
def recorded_pass(monkeypatch):
    """Runs only the chosen pass and records its Weights, event selection and the good jets built in it."""
    recorded = {}

    class RecordedWeights(wzmod.Weights):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorded["weights"] = self

    class RecordedSelection(wzmod.PackedSelection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorded.setdefault("selection", self)

    def recorded_build_jets(*args, **kwargs):
        good, bjets = build_jets(*args, **kwargs)
        recorded["good_jets"] = good[ak.argsort(good.pt, axis=1, ascending=False)]
        return good, bjets
    build_jets = wzmod.build_jets
    monkeypatch.setattr(wzmod, "Weights", RecordedWeights)
    monkeypatch.setattr(wzmod, "PackedSelection", RecordedSelection)
    monkeypatch.setattr(wzmod, "build_jets", recorded_build_jets)
    process_shift = wzmod.wzinclusive_processor.process_shift

    def run(proc, wanted):
        def only(self, event, shift_name=''):
            return process_shift(self, event, shift_name) if shift_name == wanted else {}
        monkeypatch.setattr(wzmod.wzinclusive_processor, "process_shift", only)
        recorded["out"] = proc.process(m1_events())["WZto3LNu"]
        return recorded
    return run


def processor_with_payload(monkeypatch, tmp_path):
    payload = eff_payload(tmp_path / "btagEff_2024_v15.json")
    monkeypatch.setattr("qawa.jsoncorrections.btag_eff_path", lambda era_label, version: payload)
    return wzmod.wzinclusive_processor(era="2024", version="v15")


def btag_labels(out):
    return {label for h in out.values() if isinstance(h, wzmod.hist.Hist) and "systematic" in h.axes.name
            for label in h.axes["systematic"] if label.startswith("btag_sf")}


@needs_m1
def test_processor_nominal_pass(monkeypatch, tmp_path, recorded_pass):
    proc = processor_with_payload(monkeypatch, tmp_path)
    recorded = recorded_pass(proc, None)
    names = PAIRED_NAMES | {"bc_statistic_2024", "light_correlated", "light_uncorrelated_2024", "effstat_2024"}
    assert btag_labels(recorded["out"]) == {f"btag_sf_{n}{d}" for n in names for d in ("Up", "Down")}
    weights = recorded["weights"]
    alone = Weights(len(weights.weight()), storeIndividual=True)
    proc._btag.append_btag_sf(recorded["good_jets"], alone, "WZto3LNu", variations=False)
    assert np.allclose(weights.weight(), weights.partial_weight(exclude=["btag_sf"]) * alone.weight(), rtol=1e-12)
    assert not np.allclose(alone.weight(), 1.0)
    selection = recorded["selection"]
    cleaning = ["lumimask", "goodPV", "metfilter"] + (["jetveto"] if "jetveto" in selection.names else [])
    selected = selection.all("triggers", *cleaning, "require-2lep")
    good_jets = recorded["good_jets"]
    accepted = ak.to_numpy(ak.num(good_jets[np.abs(good_jets.eta) < wzmod.BTAG_MAX_ABSETA]))[selected]
    total = recorded["out"]["btag_measurement"].values(flow=True).sum()
    assert accepted.sum() > 0
    assert np.isclose(total, (accepted * weights.partial_weight(exclude=["btag_sf"])[selected]).sum(), rtol=1e-9)
    assert not np.isclose(total, (accepted * weights.weight()[selected]).sum(), rtol=1e-6)


@needs_m1
def test_processor_jes_pass(monkeypatch, tmp_path, recorded_pass):
    proc = processor_with_payload(monkeypatch, tmp_path)
    recorded = recorded_pass(proc, "JES_Absolute2024Up")
    assert not btag_labels(recorded["out"])
    factor = recorded["weights"].partial_weight(include=["btag_sf"])
    paired, central = (Weights(len(factor), storeIndividual=True) for _ in range(2))
    proc._btag.append_btag_sf(recorded["good_jets"], paired, "WZto3LNu", shift_name="JES_Absolute2024Up", variations=False)
    proc._btag.append_btag_sf(recorded["good_jets"], central, "WZto3LNu", variations=False)
    assert np.allclose(factor, paired.weight(), rtol=1e-12)
    assert not np.allclose(factor, central.weight(), rtol=1e-12)


@needs_m1
def test_processor_without_payload(recorded_pass):
    proc = wzmod.wzinclusive_processor(era="2024", version="v15")
    with coffea_console.capture() as captured:
        recorded = recorded_pass(proc, None)
    assert "B-TAG SFs DISABLED" in captured.get()
    assert "btag_sf" not in recorded["weights"].weightStatistics
    assert not btag_labels(recorded["out"])


@needs_m1
@pytest.mark.parametrize("era", ["2024", "2025"])
def test_jes_shift_names_match_process(era, monkeypatch):
    built = []
    process_shift = wzmod.wzinclusive_processor.process_shift

    def record(self, event, shift_name=''):
        built.append(shift_name)
        return process_shift(self, event, shift_name) if shift_name is None else {}
    monkeypatch.setattr(wzmod.wzinclusive_processor, "process_shift", record)
    wzmod.wzinclusive_processor(era=era, version="v15").process(m1_events(200))
    assert [name for name in built if name and (name.startswith("JES_") or name.startswith("JER"))] == \
        list(wzmod.jes_shift_names(era))
