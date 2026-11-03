import collections.abc
import gc

import awkward as ak
import hist
import numpy as np
import pytest
from coffea.analysis_tools import PackedSelection

import qawa.process.wztau2lnu_inclusive as wzmod
import qawa.utils as utils


N = 3000
SEL_NAMES = ("triggers", "met_pt", "low_met_pt", "dilep_dphi_met", "require-2lep", "empty")


def booked(*axes, eft=False):
    cats = [hist.axis.StrCategory([], name="channel", growth=True),
            hist.axis.StrCategory([], name="systematic", growth=True)]
    if eft:
        cats.append(hist.axis.StrCategory([], name="eft", growth=True))
    return hist.Hist(*cats, *axes, hist.storage.Weight())


def make_selection(seed=1, n=N):
    rng = np.random.default_rng(seed)
    selection = PackedSelection(dtype="uint64")
    selection.add("triggers", rng.random(n) < 0.9)
    selection.add("met_pt", rng.random(n) < 0.5)
    selection.add("low_met_pt", rng.random(n) < 0.3)
    selection.add("dilep_dphi_met", rng.random(n) < 0.6)
    selection.add("require-2lep", rng.random(n) < 0.7)
    selection.add("empty", np.zeros(n, dtype=bool))
    return selection


CHANNELS = {
    "chA": ["triggers", "met_pt", "dilep_dphi_met", "require-2lep"],
    "chB": ["triggers", "~low_met_pt", "require-2lep"],
    "chEmpty": ["triggers", "empty"],
    "chAll": [],
}


def pass_state_for(selection, channels=CHANNELS):
    return {"selection": selection, "channels": channels, "block_cache": {}}


def make_weight_syst_stack(k=4, n=N, seed=2):
    rng = np.random.default_rng(seed)
    weight_syst_stack = rng.uniform(0.2, 1.8, (k, n))
    for i in range(k):
        weight_syst_stack[i, rng.integers(0, n, 5)] = np.nan
        weight_syst_stack[i, rng.integers(0, n, 5)] = np.inf
        weight_syst_stack[i, rng.integers(0, n, 5)] = -np.inf
    return weight_syst_stack


def make_values(n=N, seed=3):
    rng = np.random.default_rng(seed)
    met = rng.uniform(-20, 700, n)
    met[rng.integers(0, n, 20)] = np.nan
    dphi = rng.uniform(-3.5, 3.5, n)
    njets = rng.integers(0, 12, n).astype(np.float64)
    return {"met_pt": met, "dilep_dphi": dphi, "njets": njets}


def cut_for(selection, requirements, exclude):
    sel_args = {
        s.replace('~', ''): (False if '~' in s else True) for s in requirements if all(e not in s for e in exclude)
    }
    return selection.require(**sel_args)


def naive_fill(h, selection, channels, values, weight_syst_stack, labels, exclude, weight_fn):
    for ch, reqs in channels.items():
        cut = cut_for(selection, reqs, exclude)
        for k, label in enumerate(labels):
            w = weight_fn(k, cut)
            h.fill(channel=ch, **dict(zip([ax.name for ax in h.axes][1:1 + len(label)], label)),
                   **{name: v[cut] for name, v in values.items()}, weight=w)
    return h


def standard_ref(weight_syst_stack):
    return lambda k, cut: ak.nan_to_num(weight_syst_stack[k][cut], nan=1.0, posinf=1.0, neginf=1.0)


def assert_same(ref, out):
    assert [type(a) for a in ref.axes] == [type(a) for a in out.axes]
    assert [a.name for a in ref.axes] == [a.name for a in out.axes]
    assert [a.__dict__ for a in ref.axes] == [a.__dict__ for a in out.axes]
    assert ref.__dict__ == out.__dict__
    for ra, oa in zip(ref.axes, out.axes):
        assert ra.traits == oa.traits
        if isinstance(ra, hist.axis.StrCategory):
            assert set(ra) == set(oa)
        else:
            np.testing.assert_array_equal(ra.edges, oa.edges)
    rv = ref.view(flow=True)
    ov = out.view(flow=True)
    for i, (ra, oa) in enumerate(zip(ref.axes, out.axes)):
        if isinstance(ra, hist.axis.StrCategory):
            ov = np.take(ov, [list(oa).index(x) for x in ra], axis=i)
    assert np.array_equal(rv["value"], ov["value"], equal_nan=True)
    assert np.array_equal(rv["variance"], ov["variance"], equal_nan=True)


def axes_for(var):
    return {
        "met_pt": hist.axis.Regular(60, 0, 600, name="met_pt", label="MET"),
        "dilep_dphi": hist.axis.Variable([-3.2, -2, -1, 0, 0.5, 1, 2, 3.2], name="dilep_dphi", label="dphi"),
        "njets": hist.axis.Regular(10, 0, 10, name="njets", label="njets"),
    }[var]


class Counting:
    def __init__(self, weight):
        self.weight = weight
        self.calls = 0

    def __call__(self, syst_index, selected_event_indices):
        self.calls += 1
        return self.weight(syst_index, selected_event_indices)


@pytest.mark.parametrize("var", ["met_pt", "dilep_dphi", "njets"])
def test_fill_histogram_matches_naive(var):
    selection = make_selection()
    weight_syst_stack = make_weight_syst_stack()
    values = make_values()
    labels = [(f"s{k}",) for k in range(len(weight_syst_stack))]
    histos = {var: booked(axes_for(var))}
    utils.fill_histogram(histos, var, {var: values[var]}, labels, utils.syst_weight(weight_syst_stack), (var,),
                         pass_state_for(selection))
    ref = naive_fill(booked(axes_for(var)), selection, CHANNELS, {var: values[var]}, weight_syst_stack, labels, (var,),
                     standard_ref(weight_syst_stack))
    assert_same(ref, histos[var])
    assert "chEmpty" not in histos[var].axes[0]
    if var == "met_pt":
        cut_wrong = cut_for(selection, CHANNELS["chA"], ())
        cut_right = cut_for(selection, CHANNELS["chA"], ("met_pt",))
        assert cut_wrong.sum() != cut_right.sum()


def test_substring_exclusion_dilep_dphi():
    selection = make_selection()
    assert cut_for(selection, CHANNELS["chA"], ("dilep_dphi",)).sum() != cut_for(selection, CHANNELS["chA"], ()).sum()


def test_all_empty_target_untouched():
    selection = make_selection()
    weight_syst_stack = make_weight_syst_stack()
    channels = {"e1": ["empty"], "e2": ["empty", "triggers"]}
    target = booked(axes_for("met_pt"))
    histos = {"met_pt": target}
    counting = Counting(utils.syst_weight(weight_syst_stack))
    utils.fill_histogram(histos, "met_pt", {"met_pt": make_values()["met_pt"]}, [("s0",), ("s1",)], counting,
                         ("met_pt",), pass_state_for(selection, channels))
    assert counting.calls == 0
    assert histos["met_pt"] is target
    assert len(target.axes[0]) == 0 and len(target.axes[1]) == 0


def matched_extra(n=N, seed=4):
    rng = np.random.default_rng(seed)
    dr = ak.mask(ak.Array(rng.uniform(0, 1, n)), rng.random(n) < 0.8)
    return ak.values_astype(ak.fill_none(dr < 0.4, False), np.float32)


def test_2d_extra_weight():
    selection = make_selection()
    weight_syst_stack = make_weight_syst_stack(k=1)
    values = make_values()
    extra = matched_extra()
    factor = ak.to_numpy(ak.fill_none(extra, 0.0))
    assert factor.dtype == np.float32
    axes = (axes_for("met_pt"), axes_for("njets"))
    histos = {"m": booked(*axes)}
    utils.fill_histogram(histos, "m", {"met_pt": values["met_pt"], "njets": values["njets"]}, [("nominal",)],
                         utils.syst_weight(weight_syst_stack, factor), ("met_pt", "njets"), pass_state_for(selection))

    def ref_weight(k, cut):
        weight = weight_syst_stack[k][cut] * ak.to_numpy(ak.fill_none(extra, 0.0))[cut]
        return ak.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
    ref = naive_fill(booked(*axes), selection, CHANNELS, {"met_pt": values["met_pt"], "njets": values["njets"]},
                     weight_syst_stack, [("nominal",)], ("met_pt", "njets"), ref_weight)
    assert_same(ref, histos["m"])


def test_one_bin_many_entries():
    n = 100000
    rng = np.random.default_rng(5)
    selection = PackedSelection(dtype="uint64")
    selection.add("triggers", np.ones(n, dtype=bool))
    channels = {"c": ["triggers"]}
    weight_syst_stack = rng.uniform(0.1, 3.0, (2, n))
    values = {"met_pt": np.full(n, 42.0)}
    histos = {"met_pt": booked(axes_for("met_pt"))}
    labels = [("a",), ("b",)]
    utils.fill_histogram(histos, "met_pt", values, labels, utils.syst_weight(weight_syst_stack), ("met_pt",),
                         pass_state_for(selection, channels))
    ref = naive_fill(booked(axes_for("met_pt")), selection, channels, values, weight_syst_stack, labels, ("met_pt",),
                     standard_ref(weight_syst_stack))
    assert_same(ref, histos["met_pt"])


def selected_events(selection):
    cut = cut_for(selection, CHANNELS["chA"], ())
    return cut, np.flatnonzero(cut)


def test_syst_weight():
    selection = make_selection()
    weight_syst_stack = make_weight_syst_stack()
    cut, selected_event_indices = selected_events(selection)
    weight = utils.syst_weight(weight_syst_stack)
    for k in range(len(weight_syst_stack)):
        ref = ak.to_numpy(ak.nan_to_num(weight_syst_stack[k][cut], nan=1.0, posinf=1.0, neginf=1.0))
        w = weight(k, selected_event_indices)
        assert np.array_equal(w, ref) and w.dtype == ref.dtype
        assert np.isfinite(w).all()


def test_syst_weight_scale_2d_extra():
    selection = make_selection()
    weight_syst_stack = make_weight_syst_stack()
    extra = matched_extra()
    factor = ak.to_numpy(ak.fill_none(extra, 0.0))
    assert factor.dtype == np.float32
    cut, selected_event_indices = selected_events(selection)
    j = int(np.flatnonzero(factor[selected_event_indices] == 0)[0])
    weight_syst_stack[0, selected_event_indices[j]] = np.nan
    weight = utils.syst_weight(weight_syst_stack, factor)
    for k in range(len(weight_syst_stack)):
        product = weight_syst_stack[k][cut] * ak.to_numpy(ak.fill_none(extra, 0.0))[cut]
        ref = ak.to_numpy(ak.nan_to_num(product, nan=1.0, posinf=1.0, neginf=1.0))
        w = weight(k, selected_event_indices)
        assert np.array_equal(w, ref)
        assert w.dtype == ref.dtype
    assert weight(0, selected_event_indices)[j] == 1.0


def test_syst_weight_eft_struct_const():
    selection = make_selection()
    weight_syst_stack = make_weight_syst_stack()
    cut, selected_event_indices = selected_events(selection)
    rng = np.random.default_rng(6)
    eft_struct_consts = [rng.normal(0, 1, N) for _ in range(3)]
    eft_struct_consts[1][selected_event_indices[:4]] = np.nan
    n_terms = len(eft_struct_consts)
    weight = utils.syst_weight_eft_struct_const(weight_syst_stack, eft_struct_consts)
    for k in range(len(weight_syst_stack)):
        w_eft = ak.nan_to_num(weight_syst_stack[k][cut], nan=1.0, posinf=1.0, neginf=1.0)
        for t in range(n_terms):
            ref = ak.to_numpy(w_eft * eft_struct_consts[t][cut])
            w = weight(k * n_terms + t, selected_event_indices)
            assert np.array_equal(w, ref, equal_nan=True) and w.dtype == ref.dtype
            assert np.isnan(w).any() == (t == 1)
    assert np.isfinite(weight(0, selected_event_indices)).all()


def pol_case():
    n = 10
    selection = PackedSelection(dtype="uint64")
    selection.add("triggers", np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=bool))
    channels = {"c": ["triggers"]}
    weight_syst_stack = np.ones((3, n))
    weight_syst_stack[0, [0, 1, 2]] = [np.nan, np.inf, -np.inf]
    weight_syst_stack[1, [3, 4, 5]] = [np.nan, np.inf, -np.inf]
    weight_syst_stack[2, [6, 7, 9]] = [np.nan, np.inf, -np.inf]
    weight_syst_stack *= np.linspace(0.5, 2.0, n)[None, :]
    pol_w = ak.values_astype(ak.Array([0.0, 0.5, None, 0.0, 0.5, None, 0.0, 0.5, 1.0, None]), np.float32)
    values = {"x": np.linspace(-0.9, 0.9, n)}
    return selection, channels, weight_syst_stack, pol_w, values


def pol_reference(weight_syst_stack, pol_w, cut, k):
    pw = ak.to_numpy(ak.fill_none(pol_w[cut], 1.0))
    w = weight_syst_stack[k][cut] * pw
    return ak.to_numpy(ak.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0))


@pytest.mark.parametrize("factor_kind", ["pol", "t27"])
def test_pol_clone_operation_order(factor_kind):
    selection, channels, weight_syst_stack, pol_w, values = pol_case()
    if factor_kind == "t27":
        rescued = ak.Array([0.0, 1.0, None, 0.0, None, 1.0, 0.0, 1.0, 1.0, None])
        pol_w = ak.values_astype(rescued < 0.5, np.float32)
    factor = ak.to_numpy(ak.fill_none(pol_w, 1.0))
    cut = selection.require(triggers=True)
    selected_event_indices = np.flatnonzero(cut)
    weight = utils.syst_weight(weight_syst_stack, factor)
    for k in range(3):
        ref = pol_reference(weight_syst_stack, pol_w, cut, k)
        w = weight(k, selected_event_indices)
        assert np.array_equal(w, ref) and w.dtype == ref.dtype
    axis = hist.axis.Regular(4, -1, 1, name="x")
    histos = {"x_w": booked(axis)}
    labels = [("a",), ("b",), ("c",)]
    utils.fill_histogram(histos, "x_w", values, labels, utils.syst_weight(weight_syst_stack, factor), ("x",),
                         pass_state_for(selection, channels))
    ref = booked(axis)
    for k, (label,) in enumerate(labels):
        ref.fill(channel="c", systematic=label, x=values["x"][cut], weight=pol_reference(weight_syst_stack, pol_w, cut, k))
    assert_same(ref, histos["x_w"])
    if factor_kind == "pol":
        wrong = np.nan_to_num(weight_syst_stack[0][selected_event_indices], nan=1.0, posinf=1.0, neginf=1.0) * factor[selected_event_indices]
        assert not np.array_equal(wrong, pol_reference(weight_syst_stack, pol_w, cut, 0))


def eft_case():
    selection = make_selection()
    weight_syst_stack = make_weight_syst_stack(k=4)
    rng = np.random.default_rng(7)
    terms = ["sm", "lin_cW", "quad_cW"]
    eft_struct_consts = [rng.normal(1, 0.5, N) for _ in terms]
    eft_struct_consts[1][rng.integers(0, N, 30)] = np.nan
    labels = [(f"s{k}", t) for k in range(len(weight_syst_stack)) for t in terms]
    return selection, weight_syst_stack, terms, eft_struct_consts, labels


def eft_reference(selection, weight_syst_stack, terms, eft_struct_consts, values):
    ref = booked(axes_for("met_pt"), eft=True)
    for ch, reqs in CHANNELS.items():
        cut = cut_for(selection, reqs, ("met_pt",))
        for k in range(len(weight_syst_stack)):
            w_eft = ak.nan_to_num(weight_syst_stack[k][cut], nan=1.0, posinf=1.0, neginf=1.0)
            for t, term in enumerate(terms):
                ref.fill(channel=ch, systematic=f"s{k}", eft=term, met_pt=values["met_pt"][cut],
                         weight=w_eft * eft_struct_consts[t][cut])
    return ref


def test_eft_target_matches_naive():
    selection, weight_syst_stack, terms, eft_struct_consts, labels = eft_case()
    values = make_values()
    histos = {"met_pt_eft": booked(axes_for("met_pt"), eft=True)}
    utils.fill_histogram(histos, "met_pt_eft", {"met_pt": values["met_pt"]}, labels,
                         utils.syst_weight_eft_struct_const(weight_syst_stack, eft_struct_consts), ("met_pt",), pass_state_for(selection))
    out = histos["met_pt_eft"]
    assert_same(eft_reference(selection, weight_syst_stack, terms, eft_struct_consts, values), out)
    assert np.isnan(out.view(flow=True)["value"]).any()
    nan_terms = {list(out.axes[2])[i[2]] for i in np.argwhere(np.isnan(out.view(flow=True)["value"]))}
    assert nan_terms == {"lin_cW"}


def test_n_minus_one_key():
    reqs = ["triggers", "met_pt", "low_met_pt", "~low_met_pt", "dilep_dphi_met", "~require-2lep"]
    for exclude in [(), ("met_pt",), ("dilep_dphi",), ("met_pt", "dilep_dphi")]:
        sel_args = {s.replace('~', ''): (False if '~' in s else True) for s in reqs if all(e not in s for e in exclude)}
        key = utils.n_minus_one_key(reqs, exclude)
        assert key == tuple(sorted(sel_args.items()))
        selection = make_selection()
        assert np.array_equal(selection.require(**dict(key)), selection.require(**sel_args))
    assert dict(utils.n_minus_one_key(reqs, ()))["low_met_pt"] is False
    assert utils.n_minus_one_key(["met_pt"], ("met_pt",)) == ()
    assert utils.n_minus_one_key([], ("x",)) == ()
    assert make_selection().require(**dict(())).all()


MUTABLE = (dict, list, set, collections.abc.MutableMapping, collections.abc.MutableSet,
           collections.abc.MutableSequence)
ENGINE = ("fill_histogram", "channel_selection_cache", "n_minus_one_key", "syst_weight",
          "syst_weight_eft_struct_const")


def container_sizes():
    sizes = {}
    for module in (wzmod, utils):
        for name, value in vars(module).items():
            if isinstance(value, MUTABLE):
                sizes[(module.__name__, name)] = len(value)
            if hasattr(value, "cache_info"):
                sizes[("cache", module.__name__, name)] = value.cache_info().currsize
    for fname in ENGINE:
        f = getattr(utils, fname)
        assert not hasattr(f, "cache_info")
        for name, value in vars(f).items():
            if isinstance(value, MUTABLE):
                sizes[(fname, "vars", name)] = len(value)
        for i, value in enumerate(f.__defaults__ or ()):
            if isinstance(value, MUTABLE):
                sizes[(fname, "defaults", i)] = len(value)
        for name, value in (f.__kwdefaults__ or {}).items():
            if isinstance(value, MUTABLE):
                sizes[(fname, "kwdefaults", name)] = len(value)
        for i, cell in enumerate(f.__closure__ or ()):
            value = cell.cell_contents
            if isinstance(value, MUTABLE):
                sizes[(fname, "closure", i)] = len(value)
    return sizes


def fill_one(histos, key, selection, channels, weight_syst_stack, values, state, weight=None):
    labels = [(f"s{k}",) for k in range(len(weight_syst_stack))]
    histos[key] = booked(axes_for("met_pt"))
    utils.fill_histogram(histos, key, {"met_pt": values["met_pt"]}, labels, weight or utils.syst_weight(weight_syst_stack),
                         ("met_pt",), state)
    return histos[key]


def assert_blocks_hold_event_indices(state):
    for block in state["block_cache"].values():
        assert isinstance(block, tuple)
        for pair in block:
            assert isinstance(pair, tuple) and len(pair) == 2
            name, selected_event_indices = pair
            assert isinstance(name, str)
            assert isinstance(selected_event_indices, np.ndarray) and np.issubdtype(selected_event_indices.dtype, np.integer)


def test_cache_safety():
    before = container_sizes()
    weight_syst_stack = make_weight_syst_stack(k=2)
    values = make_values()

    selection = make_selection()
    state = pass_state_for(selection)
    histos = {}
    out2 = fill_one(histos, "a", selection, CHANNELS, weight_syst_stack, values, state, utils.syst_weight(weight_syst_stack, np.full(N, 2.0)))
    out3 = fill_one(histos, "b", selection, CHANNELS, weight_syst_stack, values, state, utils.syst_weight(weight_syst_stack, np.full(N, 3.0)))
    for out, f in ((out2, 2.0), (out3, 3.0)):
        ref = naive_fill(booked(axes_for("met_pt")), selection, CHANNELS, {"met_pt": values["met_pt"]}, weight_syst_stack,
                         [(f"s{k}",) for k in range(len(weight_syst_stack))], ("met_pt",),
                         lambda k, cut, f=f: ak.nan_to_num(weight_syst_stack[k][cut] * np.full(N, f)[cut], nan=1.0, posinf=1.0,
                                                           neginf=1.0))
        assert_same(ref, out)
    assert not np.array_equal(out2.view(flow=True)["value"], out3.view(flow=True)["value"])

    renamed = {f"{k}_x": v for k, v in CHANNELS.items()}
    out_a = fill_one({}, "c", selection, CHANNELS, weight_syst_stack, values, pass_state_for(selection))
    out_b = fill_one({}, "c", selection, renamed, weight_syst_stack, values, pass_state_for(selection, renamed))
    assert set(out_b.axes[0]) == {f"{c}_x" for c in out_a.axes[0]}

    state = pass_state_for(selection)
    first = utils.channel_selection_cache(("leading_lep_pt",), state)
    second = utils.channel_selection_cache(("dilep_deta",), state)
    assert len(state["block_cache"]) == 1 and first is second
    utils.channel_selection_cache(("met_pt",), state)
    assert len(state["block_cache"]) == 2
    assert_blocks_hold_event_indices(state)

    sel1, sel2 = make_selection(seed=11), make_selection(seed=12)
    s1, s2 = pass_state_for(sel1), pass_state_for(sel2)
    o1 = fill_one({}, "d", sel1, CHANNELS, weight_syst_stack, values, s1)
    o2 = fill_one({}, "d", sel2, CHANNELS, weight_syst_stack, values, s2)
    for sel, state_, out in ((sel1, s1, o1), (sel2, s2, o2)):
        ref = naive_fill(booked(axes_for("met_pt")), sel, CHANNELS, {"met_pt": values["met_pt"]}, weight_syst_stack,
                         [(f"s{k}",) for k in range(len(weight_syst_stack))], ("met_pt",), standard_ref(weight_syst_stack))
        assert_same(ref, out)
        block = utils.channel_selection_cache(("met_pt",), state_)
        expected_event_indices = np.concatenate([np.flatnonzero(cut_for(sel, r, ("met_pt",))) for r in CHANNELS.values()])
        assert np.array_equal(np.concatenate([selected_event_indices for _, selected_event_indices in block]), expected_event_indices)
        assert_blocks_hold_event_indices(state_)
    assert not np.array_equal(o1.view(flow=True)["value"], o2.view(flow=True)["value"])

    sel3 = make_selection(seed=13)
    s3 = pass_state_for(sel3)
    o3 = fill_one({}, "e", sel3, CHANNELS, weight_syst_stack, values, s3)
    del sel3, s3
    gc.collect()
    sel4 = make_selection(seed=14)
    s4 = pass_state_for(sel4)
    o4 = fill_one({}, "e", sel4, CHANNELS, weight_syst_stack, values, s4)
    ref4 = naive_fill(booked(axes_for("met_pt")), sel4, CHANNELS, {"met_pt": values["met_pt"]}, weight_syst_stack,
                      [(f"s{k}",) for k in range(len(weight_syst_stack))], ("met_pt",), standard_ref(weight_syst_stack))
    assert_same(ref4, o4)
    assert not np.array_equal(o3.view(flow=True)["value"], o4.view(flow=True)["value"])

    assert container_sizes() == before


def filled(labels, seed, eft=False, metadata=None):
    rng = np.random.default_rng(seed)
    h = booked(hist.axis.Regular(5, 0, 10, name="x"), eft=eft)
    if metadata is not None:
        h.metadata = metadata
    for label in labels:
        n = 50
        extra = {"eft": rng.choice(["t0", "t1"], n)} if eft else {}
        h.fill(channel=label[0], systematic=label[1], x=rng.normal(5, 4, n), weight=rng.normal(1, 0.3, n), **extra)
    return h


def merge_inputs():
    pass_a = {"ds": {
        "disjoint": filled([("chA", "nominal")], 1, metadata={"tag": "a"}),
        "overlap": filled([("chA", "nominal"), ("chB", "nominal")], 2),
        "untouched": filled([], 3),
        "three": filled([("chA", "nominal")], 4, eft=True),
    }}
    pass_b = {"ds": {
        "disjoint": filled([("chB", "jesUp")], 5, metadata={"tag": "b"}),
        "overlap": filled([("chB", "nominal"), ("chA", "jesUp")], 6),
        "untouched": filled([], 7),
        "three": filled([("chB", "jesUp"), ("chA", "nominal")], 8, eft=True),
        "single": filled([("chA", "jesUp")], 9),
    }, "other": {"only": filled([("chC", "nominal")], 10)}}
    pass_c = {"ds": {
        "disjoint": filled([("chC", "nominal")], 11),
        "overlap": filled([("chA", "nominal")], 12),
        "untouched": filled([], 13),
        "three": filled([("chA", "jesDown")], 14, eft=True),
    }}
    return [pass_a, {}, pass_b, pass_c]


def assert_same_merge(merged, ref):
    assert list(merged) == list(ref)
    for dataset in ref:
        assert list(merged[dataset]) == list(ref[dataset])
        for key, r in ref[dataset].items():
            o = merged[dataset][key]
            assert o.__dict__ == r.__dict__, key
            assert len(o.axes) == len(r.axes)
            for oa, ra in zip(o.axes, r.axes):
                assert type(oa) is type(ra) and oa.__dict__ == ra.__dict__ and oa.traits == ra.traits
                assert list(oa) == list(ra) if isinstance(ra, hist.axis.StrCategory) else oa == ra
            assert o.storage_type == r.storage_type
            for field in ("value", "variance"):
                assert np.array_equal(o.view(flow=True)[field], r.view(flow=True)[field]), (key, field)


def test_merge_histogram_sets_matches_accumulate():
    from coffea import processor
    merged = utils.merge_histogram_sets(merge_inputs())
    ref = processor.accumulate(merge_inputs())
    assert ref["ds"]["overlap"]._variance_known is False
    assert ref["ds"]["single"]._variance_known is True
    assert ref["other"]["only"]._variance_known is True
    assert_same_merge(merged, ref)
    assert list(merged["ds"]["disjoint"].axes["channel"]) == ["chA", "chB", "chC"]

    one = merge_inputs()[2]
    assert_same_merge(utils.merge_histogram_sets([one]), processor.accumulate([one]))


def test_merge_histogram_sets_empty_cases():
    assert utils.merge_histogram_sets([]) == {}
    assert utils.merge_histogram_sets([{}, {}]) == {}
    assert utils.merge_histogram_sets([{"ds": {}}, {"ds": {}}]) == {"ds": {}}
    assert utils.merge_histogram_sets([{}, {"ds": {}}, {}]) == {"ds": {}}
