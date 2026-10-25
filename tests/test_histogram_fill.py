import collections.abc
import gc

import awkward as ak
import hist
import numpy as np
import pytest
from coffea.analysis_tools import PackedSelection

import qawa.process.wztau2lnu_inclusive as wzmod


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


def make_raw(k=4, n=N, seed=2):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.2, 1.8, (k, n))
    for i in range(k):
        raw[i, rng.integers(0, n, 5)] = np.nan
        raw[i, rng.integers(0, n, 5)] = np.inf
        raw[i, rng.integers(0, n, 5)] = -np.inf
    return raw


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


def naive_fill(h, selection, channels, values, raw, labels, exclude, weight_fn):
    for ch, reqs in channels.items():
        cut = cut_for(selection, reqs, exclude)
        for k, label in enumerate(labels):
            w = weight_fn(k, cut)
            h.fill(channel=ch, **dict(zip([ax.name for ax in h.axes][1:1 + len(label)], label)),
                   **{name: v[cut] for name, v in values.items()}, weight=w)
    return h


def standard_ref(raw):
    return lambda k, cut: ak.nan_to_num(raw[k][cut], nan=1.0, posinf=1.0, neginf=1.0)


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

    def __call__(self, k, rows):
        self.calls += 1
        return self.weight(k, rows)


@pytest.mark.parametrize("var", ["met_pt", "dilep_dphi", "njets"])
def test_fill_histogram_matches_naive(var):
    selection = make_selection()
    raw = make_raw()
    values = make_values()
    labels = [(f"s{k}",) for k in range(len(raw))]
    histos = {var: booked(axes_for(var))}
    wzmod.fill_histogram(histos, var, {var: values[var]}, labels, wzmod.standard_weights(raw), (var,),
                         pass_state_for(selection))
    ref = naive_fill(booked(axes_for(var)), selection, CHANNELS, {var: values[var]}, raw, labels, (var,),
                     standard_ref(raw))
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
    raw = make_raw()
    channels = {"e1": ["empty"], "e2": ["empty", "triggers"]}
    target = booked(axes_for("met_pt"))
    histos = {"met_pt": target}
    counting = Counting(wzmod.standard_weights(raw))
    wzmod.fill_histogram(histos, "met_pt", {"met_pt": make_values()["met_pt"]}, [("s0",), ("s1",)], counting,
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
    raw = make_raw(k=1)
    values = make_values()
    extra = matched_extra()
    factor = ak.to_numpy(ak.fill_none(extra, 0.0))
    assert factor.dtype == np.float32
    axes = (axes_for("met_pt"), axes_for("njets"))
    histos = {"m": booked(*axes)}
    wzmod.fill_histogram(histos, "m", {"met_pt": values["met_pt"], "njets": values["njets"]}, [("nominal",)],
                         wzmod.scaled_weights(raw, factor), ("met_pt", "njets"), pass_state_for(selection))

    def ref_weight(k, cut):
        weight = raw[k][cut] * ak.to_numpy(ak.fill_none(extra, 0.0))[cut]
        return ak.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
    ref = naive_fill(booked(*axes), selection, CHANNELS, {"met_pt": values["met_pt"], "njets": values["njets"]},
                     raw, [("nominal",)], ("met_pt", "njets"), ref_weight)
    assert_same(ref, histos["m"])


def test_one_bin_many_entries():
    n = 100000
    rng = np.random.default_rng(5)
    selection = PackedSelection(dtype="uint64")
    selection.add("triggers", np.ones(n, dtype=bool))
    channels = {"c": ["triggers"]}
    raw = rng.uniform(0.1, 3.0, (2, n))
    values = {"met_pt": np.full(n, 42.0)}
    histos = {"met_pt": booked(axes_for("met_pt"))}
    labels = [("a",), ("b",)]
    wzmod.fill_histogram(histos, "met_pt", values, labels, wzmod.standard_weights(raw), ("met_pt",),
                         pass_state_for(selection, channels))
    ref = naive_fill(booked(axes_for("met_pt")), selection, channels, values, raw, labels, ("met_pt",),
                     standard_ref(raw))
    assert_same(ref, histos["met_pt"])


def selected_rows(selection):
    cut = cut_for(selection, CHANNELS["chA"], ())
    return cut, np.flatnonzero(cut)


def test_standard_weights():
    selection = make_selection()
    raw = make_raw()
    cut, rows = selected_rows(selection)
    weight = wzmod.standard_weights(raw)
    for k in range(len(raw)):
        ref = ak.to_numpy(ak.nan_to_num(raw[k][cut], nan=1.0, posinf=1.0, neginf=1.0))
        w = weight(k, rows)
        assert np.array_equal(w, ref) and w.dtype == ref.dtype
        assert np.isfinite(w).all()


def test_scaled_weights_2d_extra():
    selection = make_selection()
    raw = make_raw()
    extra = matched_extra()
    factor = ak.to_numpy(ak.fill_none(extra, 0.0))
    assert factor.dtype == np.float32
    cut, rows = selected_rows(selection)
    j = int(np.flatnonzero(factor[rows] == 0)[0])
    raw[0, rows[j]] = np.nan
    weight = wzmod.scaled_weights(raw, factor)
    for k in range(len(raw)):
        product = raw[k][cut] * ak.to_numpy(ak.fill_none(extra, 0.0))[cut]
        ref = ak.to_numpy(ak.nan_to_num(product, nan=1.0, posinf=1.0, neginf=1.0))
        w = weight(k, rows)
        assert np.array_equal(w, ref)
        assert w.dtype == ref.dtype
    assert weight(0, rows)[j] == 1.0


def test_eft_term_weights():
    selection = make_selection()
    raw = make_raw()
    cut, rows = selected_rows(selection)
    rng = np.random.default_rng(6)
    columns = [rng.normal(0, 1, N) for _ in range(3)]
    columns[1][rows[:4]] = np.nan
    n_terms = len(columns)
    weight = wzmod.eft_term_weights(raw, columns)
    for k in range(len(raw)):
        w_eft = ak.nan_to_num(raw[k][cut], nan=1.0, posinf=1.0, neginf=1.0)
        for t in range(n_terms):
            ref = ak.to_numpy(w_eft * columns[t][cut])
            w = weight(k * n_terms + t, rows)
            assert np.array_equal(w, ref, equal_nan=True) and w.dtype == ref.dtype
            assert np.isnan(w).any() == (t == 1)
    assert np.isfinite(weight(0, rows)).all()


def pol_case():
    n = 10
    selection = PackedSelection(dtype="uint64")
    selection.add("triggers", np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=bool))
    channels = {"c": ["triggers"]}
    raw = np.ones((3, n))
    raw[0, [0, 1, 2]] = [np.nan, np.inf, -np.inf]
    raw[1, [3, 4, 5]] = [np.nan, np.inf, -np.inf]
    raw[2, [6, 7, 9]] = [np.nan, np.inf, -np.inf]
    raw *= np.linspace(0.5, 2.0, n)[None, :]
    pol_w = ak.values_astype(ak.Array([0.0, 0.5, None, 0.0, 0.5, None, 0.0, 0.5, 1.0, None]), np.float32)
    values = {"x": np.linspace(-0.9, 0.9, n)}
    return selection, channels, raw, pol_w, values


def pol_reference(raw, pol_w, cut, k):
    pw = ak.to_numpy(ak.fill_none(pol_w[cut], 1.0))
    w = raw[k][cut] * pw
    return ak.to_numpy(ak.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0))


@pytest.mark.parametrize("factor_kind", ["pol", "t27"])
def test_pol_clone_operation_order(factor_kind):
    selection, channels, raw, pol_w, values = pol_case()
    if factor_kind == "t27":
        rescued = ak.Array([0.0, 1.0, None, 0.0, None, 1.0, 0.0, 1.0, 1.0, None])
        pol_w = ak.values_astype(rescued < 0.5, np.float32)
    factor = ak.to_numpy(ak.fill_none(pol_w, 1.0))
    cut = selection.require(triggers=True)
    rows = np.flatnonzero(cut)
    weight = wzmod.scaled_weights(raw, factor)
    for k in range(3):
        ref = pol_reference(raw, pol_w, cut, k)
        w = weight(k, rows)
        assert np.array_equal(w, ref) and w.dtype == ref.dtype
    axis = hist.axis.Regular(4, -1, 1, name="x")
    histos = {"x_w": booked(axis)}
    labels = [("a",), ("b",), ("c",)]
    wzmod.fill_histogram(histos, "x_w", values, labels, wzmod.scaled_weights(raw, factor), ("x",),
                         pass_state_for(selection, channels))
    ref = booked(axis)
    for k, (label,) in enumerate(labels):
        ref.fill(channel="c", systematic=label, x=values["x"][cut], weight=pol_reference(raw, pol_w, cut, k))
    assert_same(ref, histos["x_w"])
    if factor_kind == "pol":
        wrong = np.nan_to_num(raw[0][rows], nan=1.0, posinf=1.0, neginf=1.0) * factor[rows]
        assert not np.array_equal(wrong, pol_reference(raw, pol_w, cut, 0))


def eft_case():
    selection = make_selection()
    raw = make_raw(k=4)
    rng = np.random.default_rng(7)
    terms = ["sm", "lin_cW", "quad_cW"]
    columns = [rng.normal(1, 0.5, N) for _ in terms]
    columns[1][rng.integers(0, N, 30)] = np.nan
    labels = [(f"s{k}", t) for k in range(len(raw)) for t in terms]
    return selection, raw, terms, columns, labels


def eft_reference(selection, raw, terms, columns, values):
    ref = booked(axes_for("met_pt"), eft=True)
    for ch, reqs in CHANNELS.items():
        cut = cut_for(selection, reqs, ("met_pt",))
        for k in range(len(raw)):
            w_eft = ak.nan_to_num(raw[k][cut], nan=1.0, posinf=1.0, neginf=1.0)
            for t, term in enumerate(terms):
                ref.fill(channel=ch, systematic=f"s{k}", eft=term, met_pt=values["met_pt"][cut],
                         weight=w_eft * columns[t][cut])
    return ref


def test_eft_target_matches_naive():
    selection, raw, terms, columns, labels = eft_case()
    values = make_values()
    histos = {"met_pt_eft": booked(axes_for("met_pt"), eft=True)}
    wzmod.fill_histogram(histos, "met_pt_eft", {"met_pt": values["met_pt"]}, labels,
                         wzmod.eft_term_weights(raw, columns), ("met_pt",), pass_state_for(selection))
    out = histos["met_pt_eft"]
    assert_same(eft_reference(selection, raw, terms, columns, values), out)
    assert np.isnan(out.view(flow=True)["value"]).any()
    nan_terms = {list(out.axes[2])[i[2]] for i in np.argwhere(np.isnan(out.view(flow=True)["value"]))}
    assert nan_terms == {"lin_cW"}


def test_n_minus_one_key():
    reqs = ["triggers", "met_pt", "low_met_pt", "~low_met_pt", "dilep_dphi_met", "~require-2lep"]
    for exclude in [(), ("met_pt",), ("dilep_dphi",), ("met_pt", "dilep_dphi")]:
        sel_args = {s.replace('~', ''): (False if '~' in s else True) for s in reqs if all(e not in s for e in exclude)}
        key = wzmod.n_minus_one_key(reqs, exclude)
        assert key == tuple(sorted(sel_args.items()))
        selection = make_selection()
        assert np.array_equal(selection.require(**dict(key)), selection.require(**sel_args))
    assert dict(wzmod.n_minus_one_key(reqs, ()))["low_met_pt"] is False
    assert wzmod.n_minus_one_key(["met_pt"], ("met_pt",)) == ()
    assert wzmod.n_minus_one_key([], ("x",)) == ()
    assert make_selection().require(**dict(())).all()


MUTABLE = (dict, list, set, collections.abc.MutableMapping, collections.abc.MutableSet,
           collections.abc.MutableSequence)
ENGINE = ("fill_histogram", "channel_block", "n_minus_one_key", "standard_weights", "scaled_weights",
          "eft_term_weights")


def container_sizes():
    sizes = {}
    for name, value in vars(wzmod).items():
        if isinstance(value, MUTABLE):
            sizes[("module", name)] = len(value)
        if hasattr(value, "cache_info"):
            sizes[("cache", name)] = value.cache_info().currsize
    for fname in ENGINE:
        f = getattr(wzmod, fname)
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


def fill_one(histos, key, selection, channels, raw, values, state, weight=None):
    labels = [(f"s{k}",) for k in range(len(raw))]
    histos[key] = booked(axes_for("met_pt"))
    wzmod.fill_histogram(histos, key, {"met_pt": values["met_pt"]}, labels, weight or wzmod.standard_weights(raw),
                         ("met_pt",), state)
    return histos[key]


def assert_blocks_hold_rows(state):
    for block in state["block_cache"].values():
        assert isinstance(block, tuple)
        for pair in block:
            assert isinstance(pair, tuple) and len(pair) == 2
            name, rows = pair
            assert isinstance(name, str)
            assert isinstance(rows, np.ndarray) and np.issubdtype(rows.dtype, np.integer)


def test_cache_safety():
    before = container_sizes()
    raw = make_raw(k=2)
    values = make_values()

    selection = make_selection()
    state = pass_state_for(selection)
    histos = {}
    out2 = fill_one(histos, "a", selection, CHANNELS, raw, values, state, wzmod.scaled_weights(raw, np.full(N, 2.0)))
    out3 = fill_one(histos, "b", selection, CHANNELS, raw, values, state, wzmod.scaled_weights(raw, np.full(N, 3.0)))
    for out, f in ((out2, 2.0), (out3, 3.0)):
        ref = naive_fill(booked(axes_for("met_pt")), selection, CHANNELS, {"met_pt": values["met_pt"]}, raw,
                         [(f"s{k}",) for k in range(len(raw))], ("met_pt",),
                         lambda k, cut, f=f: ak.nan_to_num(raw[k][cut] * np.full(N, f)[cut], nan=1.0, posinf=1.0,
                                                           neginf=1.0))
        assert_same(ref, out)
    assert not np.array_equal(out2.view(flow=True)["value"], out3.view(flow=True)["value"])

    renamed = {f"{k}_x": v for k, v in CHANNELS.items()}
    out_a = fill_one({}, "c", selection, CHANNELS, raw, values, pass_state_for(selection))
    out_b = fill_one({}, "c", selection, renamed, raw, values, pass_state_for(selection, renamed))
    assert set(out_b.axes[0]) == {f"{c}_x" for c in out_a.axes[0]}

    state = pass_state_for(selection)
    first = wzmod.channel_block(("leading_lep_pt",), state)
    second = wzmod.channel_block(("dilep_deta",), state)
    assert len(state["block_cache"]) == 1 and first is second
    wzmod.channel_block(("met_pt",), state)
    assert len(state["block_cache"]) == 2
    assert_blocks_hold_rows(state)

    sel1, sel2 = make_selection(seed=11), make_selection(seed=12)
    s1, s2 = pass_state_for(sel1), pass_state_for(sel2)
    o1 = fill_one({}, "d", sel1, CHANNELS, raw, values, s1)
    o2 = fill_one({}, "d", sel2, CHANNELS, raw, values, s2)
    for sel, state_, out in ((sel1, s1, o1), (sel2, s2, o2)):
        ref = naive_fill(booked(axes_for("met_pt")), sel, CHANNELS, {"met_pt": values["met_pt"]}, raw,
                         [(f"s{k}",) for k in range(len(raw))], ("met_pt",), standard_ref(raw))
        assert_same(ref, out)
        block = wzmod.channel_block(("met_pt",), state_)
        expected_rows = np.concatenate([np.flatnonzero(cut_for(sel, r, ("met_pt",))) for r in CHANNELS.values()])
        assert np.array_equal(np.concatenate([rows for _, rows in block]), expected_rows)
        assert_blocks_hold_rows(state_)
    assert not np.array_equal(o1.view(flow=True)["value"], o2.view(flow=True)["value"])

    sel3 = make_selection(seed=13)
    s3 = pass_state_for(sel3)
    o3 = fill_one({}, "e", sel3, CHANNELS, raw, values, s3)
    del sel3, s3
    gc.collect()
    sel4 = make_selection(seed=14)
    s4 = pass_state_for(sel4)
    o4 = fill_one({}, "e", sel4, CHANNELS, raw, values, s4)
    ref4 = naive_fill(booked(axes_for("met_pt")), sel4, CHANNELS, {"met_pt": values["met_pt"]}, raw,
                      [(f"s{k}",) for k in range(len(raw))], ("met_pt",), standard_ref(raw))
    assert_same(ref4, o4)
    assert not np.array_equal(o3.view(flow=True)["value"], o4.view(flow=True)["value"])

    assert container_sizes() == before
