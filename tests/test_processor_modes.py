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


ROOT = os.environ.get("QAWA_EXPECTED_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_FILE = os.path.join(ROOT, "benchmarks", "rename_map_wz_inclusive.tsv")
KINDS = {"hist", "axis", "ntuple", "selection"}
KEPT_OVERLAP = {"met_pt", "met_pt_fv", "dilep_m_fv", "dilep_pt_fv"}
T24_ROWS = {"tau_pt_gen_vs_tau_vtight_pt", "tau_pt_gen_vs_tau_vtight_pt_tau_loosetovtight_matched"}
U4_REMOVED = {"dilep_tau_pt", "dilep_loose_tau_pt", "dilep_loose_tau_met_dphi", "dilep_loose_tau_phi", "dilep_tau_phi"}
WPREFS = ("wZW", "wZWp", "wZWm", "wWZ", "wWpZ", "wWmZ")
POLS = ("long", "left", "right")

D1_FILE = "/Volumes/WDMac/SMQawaTestFiles/store/data/Run2024C/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root"
needs_d1 = pytest.mark.skipif(not os.path.exists(D1_FILE), reason="local 2024 Muon0 data test file not available")

ALLOWED = {
    ("met_pt", "low_met_pt"), ("met_pt", "medium_met_pt"), ("met_pt", "high_met_pt"),
    ("njets", "0njets"), ("njets", "1njets"), ("njets", "1njets_only"),
    ("nbjets", "0nbjets"),
} | {(f"nhtaus_{wp}", f"{n}nhtaus_{wp}") for wp in ("loose", "tight", "vtight") for n in ("1", "2plus")}

# build_histos booked 75 keys at 76984ce; U4 removes 5. The pol builder booked 35 (107 with the signal clones).
BOOKED_COUNTS = {"standard": 70, "pol": 35, "pol_signal": 107}


def read_map():
    import csv
    with open(MAP_FILE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert reader.fieldnames == ["old", "new", "kinds", "tau_wp"]
        rows = [dict(r, kinds=set(r["kinds"].split(","))) for r in reader]
    return rows


def test_n_minus_one_collisions():
    assert wzmod.n_minus_one_collisions(["dilep_dphi"], ["dilep_dphi_met"]) == [("dilep_dphi", "dilep_dphi_met")]
    assert wzmod.n_minus_one_collisions(["tau_vtight_dphi_dilep"], ["tau_vtight_dphi_dilep"]) == []
    variables = ["njets", "met_pt", "dilep_m"]
    selections = ["1njets", "high_met_pt", "0njets", "dilep_m", "low_met_pt"]
    expected = [("met_pt", "high_met_pt"), ("met_pt", "low_met_pt"), ("njets", "0njets"), ("njets", "1njets")]
    assert wzmod.n_minus_one_collisions(variables, selections) == expected
    assert wzmod.n_minus_one_collisions(variables[::-1], selections[::-1]) == expected


def test_rename_map_consistency():
    import inspect
    import re
    from qawa.polarization import POL_TEMPLATE_VARS, pol_wpref_for
    rows = read_map()
    for r in rows:
        assert r["kinds"] and r["kinds"] <= KINDS, r
    output = [r for r in rows if r["kinds"] & {"hist", "axis"}]
    selection = [r for r in rows if "selection" in r["kinds"]]
    for cls in (output, selection):
        olds = [r["old"] for r in cls]
        news = [r["new"] for r in cls]
        assert len(set(olds)) == len(olds)
        assert len(set(news)) == len(news)
        assert not set(news) & set(olds)

    source = inspect.getsource(wzmod)
    value_names = set(re.findall(r"ntuple\['(\w+)'", source)) | set(wzmod.STANDARD_FILL_VARIABLES)
    selection_only = {r["old"] for r in rows if r["kinds"] == {"selection"}}
    assert selection_only & value_names == KEPT_OVERLAP

    tau_rows = [r for r in rows if r["tau_wp"] != "-"]
    assert len(tau_rows) == 73
    for r in tau_rows:
        tags = r["tau_wp"].split("+")
        for tag in tags:
            assert re.search(rf"(^|_)tau_{tag}_", r["new"]), r
        if r["new"].startswith("tau_") and r["new"] not in T24_ROWS:
            assert r["new"].startswith(f"tau_{tags[0]}_"), r
    assert {r["new"] for r in tau_rows if not r["new"].startswith(f"tau_{r['tau_wp'].split('+')[0]}_")} == T24_ROWS
    assert all(r["tau_wp"] != "-" for r in rows if "tau" in r["old"])

    met = {r["old"]: r["new"] for r in selection if "met_pt" in r["old"]}
    for old, new in met.items():
        if old.endswith("_fv"):
            assert "met_pt" not in new, (old, new)
        else:
            assert "met_pt" in new, (old, new)

    for r in rows:
        assert pol_wpref_for(r["new"]) == pol_wpref_for(r["old"]), r

    assert set(WPREFS) == {pol_wpref_for(v) for v in POL_TEMPLATE_VARS}
    suffixes = ["_eft"] + [f"_{w}_{p}" for w in WPREFS for p in POLS] + [f"_{w}_{p}_eft" for w in WPREFS for p in POLS]
    assert len(suffixes) == 37
    hist_rows = [r for r in rows if "hist" in r["kinds"]]
    for side in ("old", "new"):
        bases = {r[side] for r in hist_rows}
        assert not [(a, s) for a in bases for s in suffixes if a + s in bases]
    for key, side in (("tau_loosetovtight_mT_WZ_eft", "new"), ("mT_WZ_eft", "old")):
        bases = {r[side] for r in hist_rows}
        assert [(b, s) for b in bases for s in suffixes if b + s == key] == [(key[:-len("_eft")], "_eft")]


def mode_events(path, n, dataset, is_data, is_signal):
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    return NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="virtual", entry_start=0, entry_stop=n,
        metadata={"dataset": dataset, "is_data": is_data, "is_signal": is_signal, "is_flashsim": False},
    ).events()


MODES = {
    "standard": (dict(), (M1_FILE, "WZto3LNu", False, True)),
    "pol": (dict(pol=True), (M1_FILE, "WZto3LNu", False, True)),
    "split_FV": (dict(split_FV=True), (M1_FILE, "WZto3LNu", False, True)),
    "split_by_charge": (dict(split_by_charge=True), (M1_FILE, "WZto3LNu", False, True)),
    "data": (dict(run_period="C"), (D1_FILE, "Muon0_Run2024C", True, False)),
}


def booked_names(histos):
    axes = {ax.name for h in histos.values() for ax in h.axes if not isinstance(ax, wzmod.hist.axis.StrCategory)}
    return set(histos), axes


@pytest.fixture(scope="module")
def inventory():
    result = {}
    process_shift = wzmod.wzinclusive_processor.process_shift
    n_minus_one_key = wzmod.n_minus_one_key
    channel_block = wzmod.channel_block
    for mode, (kwargs, (path, dataset, is_data, is_signal)) in MODES.items():
        excludes, selections = set(), set()

        def recorded_key(requirements, exclude):
            excludes.add(tuple(exclude))
            return n_minus_one_key(requirements, exclude)

        def recorded_block(exclude, pass_state):
            selections.update(pass_state["selection"].names)
            return channel_block(exclude, pass_state)

        def nominal_only(self, event, shift_name=''):
            return process_shift(self, event, shift_name) if shift_name is None else {}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(wzmod, "n_minus_one_key", recorded_key)
            mp.setattr(wzmod, "channel_block", recorded_block)
            mp.setattr(wzmod.wzinclusive_processor, "process_shift", nominal_only)
            proc = wzmod.wzinclusive_processor(era="2024", version="v15", **kwargs)
            out = proc.process(mode_events(path, 1000, dataset, is_data, is_signal))[dataset]
        if kwargs.get("pol"):
            builders = {"pol": proc.build_histos_pol(), "pol_signal": proc._build_histos_with_pol()}
        else:
            builders = {"standard": proc.build_histos()}
        keys, axes = set(), set()
        for histos in builders.values():
            k, a = booked_names(histos)
            keys |= k
            axes |= a
        variables = {v for e in excludes for v in e} | keys | axes
        result[mode] = dict(excludes=excludes, selections=selections, keys=keys, axes=axes, variables=variables,
                            output=set(out), counts={b: len(h) for b, h in builders.items()})
    return result


@needs_m1
@needs_d1
def test_n_minus_one_inventory(inventory):
    import warnings
    observed = set()
    for rec in inventory.values():
        assert rec["selections"]
        observed |= set(wzmod.n_minus_one_collisions(rec["variables"], rec["selections"]))
    assert observed <= ALLOWED, sorted(observed - ALLOWED)
    registered = set().union(*(rec["selections"] for rec in inventory.values()))
    for pair in sorted(ALLOWED):
        if pair[1] in registered:
            assert pair in observed, pair
        else:
            warnings.warn(f"dormant N-1 pair {pair}: {pair[1]} is registered in no mode", UserWarning)
    standard_1d = {e[0] for e in inventory["standard"]["excludes"] if len(e) == 1}
    assert standard_1d == set(wzmod.STANDARD_FILL_VARIABLES)


@needs_m1
@needs_d1
def test_rename_inventory(inventory):
    rows = read_map()
    selections = set().union(*(rec["selections"] for rec in inventory.values()))
    keys = set().union(*(rec["keys"] for rec in inventory.values()))
    axes = set().union(*(rec["axes"] for rec in inventory.values()))
    outputs = set().union(*(rec["output"] for rec in inventory.values()))
    for r in rows:
        if "selection" in r["kinds"]:
            assert r["old"] not in selections, r
            assert r["new"] in selections, r
        if "hist" in r["kinds"]:
            assert r["old"] not in keys, r
            assert r["new"] in keys, r
        if "axis" in r["kinds"]:
            assert r["old"] not in axes, r
            assert r["new"] in axes, r
    selection_only = {r["old"] for r in rows if r["kinds"] == {"selection"}}
    assert selection_only & (keys | axes) == {"met_pt"}
    for mode, rec in inventory.items():
        for builder, n in rec["counts"].items():
            assert n == BOOKED_COUNTS[builder], (mode, builder, n)
    assert not U4_REMOVED & keys
    assert not U4_REMOVED & {k[:-len("_eft")] if k.endswith("_eft") else k for k in outputs}


RESTORED = "tau_loosetovtight_mT_WZ"


@pytest.fixture(scope="module")
def restored_outputs():
    result = {}
    for mode, n in (("standard", 1000), ("data", 60000)):
        kwargs, (path, dataset, is_data, is_signal) = MODES[mode]
        proc = wzmod.wzinclusive_processor(era="2024", version="v15", **kwargs)
        result[mode] = proc.process(mode_events(path, n, dataset, is_data, is_signal))[dataset]
    return result


def category_labels(h):
    return [set(ax) for ax in h.axes if isinstance(ax, wzmod.hist.axis.StrCategory)]


def label_sums(h):
    if h is None:
        return {}
    channels, systematics = [ax for ax in h.axes if isinstance(ax, wzmod.hist.axis.StrCategory)]
    view = h.view(flow=True)
    values = view["value"] if view.dtype.names else view
    return {(c, s): (np.sum(values[i, j]), np.sum(np.abs(values[i, j][np.isfinite(values[i, j])])))
            for i, c in enumerate(channels) for j, s in enumerate(systematics)}


def deta_sums(out):
    sums = label_sums(out["dilep_deta"])
    assert any(np.isfinite(a) and a > 0 for _, a in sums.values())
    return sums


def test_restored_in_fill_tuple():
    variables = wzmod.STANDARD_FILL_VARIABLES
    assert RESTORED in variables
    assert variables[variables.index(RESTORED) - 1] == "tau_loosetovtight_mT_W"


@needs_m1
@needs_d1
@pytest.mark.parametrize("mode", ["standard", "data"])
def test_restored_has_labels(restored_outputs, mode):
    out = restored_outputs[mode]
    deta_sums(out)
    assert label_sums(out.get(RESTORED)), mode


@needs_m1
@needs_d1
@pytest.mark.parametrize("mode", ["standard", "data"])
def test_restored_label_sets(restored_outputs, mode):
    out = restored_outputs[mode]
    deta_sums(out)
    assert RESTORED in out and category_labels(out[RESTORED]) == category_labels(out["dilep_deta"]), mode


@needs_m1
@needs_d1
@pytest.mark.parametrize("mode", ["standard", "data"])
def test_restored_sums(restored_outputs, mode):
    out = restored_outputs[mode]
    reference = deta_sums(out)
    restored = label_sums(out.get(RESTORED))
    for label, (b, b_scale) in reference.items():
        a, a_scale = restored.get(label, (0.0, 0.0))
        same = (np.isnan(a) and np.isnan(b)) or a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b), a_scale + b_scale)
        assert same, (mode, label, a, b)
