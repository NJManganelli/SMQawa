import importlib.util
import os

import hist
import numpy as np
import pytest

import qawa.process.wztau2lnu_inclusive as wzmod

ROOT = os.environ.get("QAWA_EXPECTED_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fill_engines = load("fill_engines", os.path.join(ROOT, "benchmarks", "fill_engines.py"))
helpers = load("histogram_fill_helpers", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "test_histogram_fill.py"))

CHANNELS = {
    "chA": ["triggers", "met_pt", "dilep_dphi_met", "require-2lep"],
    "chB": ["triggers", "~low_met_pt", "require-2lep"],
    "chEmpty": ["triggers", "empty"],
}


def assert_close(ref, out):
    assert [a.name for a in ref.axes] == [a.name for a in out.axes]
    assert [a.__dict__ for a in ref.axes] == [a.__dict__ for a in out.axes]
    assert ref.__dict__ == out.__dict__
    for ra, oa in zip(ref.axes, out.axes):
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
    r, o = rv["variance"], ov["variance"]
    ok = (np.isnan(r) & np.isnan(o)) | (r == o) | (
        np.isfinite(r) & np.isfinite(o) & (np.abs(r - o) <= 1e-9 * np.maximum(np.abs(r), np.abs(o))))
    assert ok.all()


def standard_case():
    selection = helpers.make_selection()
    raw = helpers.make_raw(k=3)
    values = {"met_pt": helpers.make_values()["met_pt"]}
    labels = [(f"s{k}",) for k in range(len(raw))]
    return selection, raw, values, labels


def fill_with(engine, key, axes, values, labels, weight, selection, eft=False):
    histos = {key: helpers.booked(*axes, eft=eft)}
    state = helpers.pass_state_for(selection, CHANNELS)
    engine(histos, key, values, labels, weight, ("met_pt",), state)
    return histos[key], state


@pytest.mark.parametrize("name", sorted(fill_engines.ENGINES))
def test_engine_equals_c1(name):
    selection, raw, values, labels = standard_case()
    axis = helpers.axes_for("met_pt")
    c1, _ = fill_with(wzmod.fill_histogram, "met_pt", (axis,), values, labels, wzmod.standard_weights(raw),
                      selection)
    out, _ = fill_with(fill_engines.ENGINES[name], "met_pt", (axis,), values, labels, wzmod.standard_weights(raw),
                       selection)
    assert "chEmpty" not in out.axes[0]
    assert_close(c1, out)


@pytest.mark.parametrize("name", sorted(fill_engines.ENGINES))
def test_engine_equals_c1_eft(name):
    selection, raw, terms, columns, labels = helpers.eft_case()
    values = {"met_pt": helpers.make_values()["met_pt"]}
    axis = helpers.axes_for("met_pt")
    weight = wzmod.eft_term_weights(raw, columns)
    c1, _ = fill_with(wzmod.fill_histogram, "met_pt_eft", (axis,), values, labels, weight, selection, eft=True)
    out, _ = fill_with(fill_engines.ENGINES[name], "met_pt_eft", (axis,), values, labels, weight, selection, eft=True)
    assert np.isnan(c1.view(flow=True)["value"]).any()
    assert_close(c1, out)


def test_transposed_write_back_is_detected():
    selection, raw, values, labels = standard_case()
    axis = helpers.axes_for("met_pt")
    c1, _ = fill_with(wzmod.fill_histogram, "met_pt", (axis,), values, labels, wzmod.standard_weights(raw),
                      selection)
    out, _ = fill_with(fill_engines.multicell, "met_pt", (axis,), values, labels, wzmod.standard_weights(raw),
                       selection)
    view = out.view(flow=True)
    view[:, :] = view[:, ::-1]
    with pytest.raises(AssertionError):
        assert_close(c1, out)


def test_multicell_vs_chunkedhist():
    selection, raw, values, labels = standard_case()
    axis = helpers.axes_for("met_pt")
    naive = helpers.naive_fill(helpers.booked(axis), selection, CHANNELS, values, raw, labels, ("met_pt",),
                               helpers.standard_ref(raw))
    multicell, _ = fill_with(fill_engines.multicell, "met_pt", (axis,), values, labels, wzmod.standard_weights(raw),
                             selection)
    chunked, _ = fill_with(fill_engines.chunked, "met_pt", (axis,), values, labels, wzmod.standard_weights(raw),
                           selection)
    assert_close(multicell, chunked)
    assert_close(naive, multicell)
    assert_close(naive, chunked)


@pytest.mark.xfail(strict=True, reason="growth-axis resize zeroes existing MultiCell content (boost-histogram 1.8.1); "
                                       "if this XPASSes, revisit D2a pre-declaration")
def test_multicell_growth_canary():
    h = hist.Hist(hist.axis.StrCategory([], name="c", growth=True), hist.axis.Regular(4, 0, 1, name="x"),
                  storage=hist.storage.MultiCell(2))
    h.fill(c="a", x=[0.5], weight=np.array([[1.0, 2.0]]))
    h.fill(c="b", x=[0.5], weight=np.array([[3.0, 4.0]]))
    v = h.view(flow=True)
    assert v[0, 0].sum() == 1.0 and v[1, 0].sum() == 2.0


@pytest.mark.parametrize("name", sorted(fill_engines.ENGINES))
def test_engines_add_nothing_to_processor_module(name):
    assert not [a for a in vars(wzmod) if "multicell" in a.lower() or "buffer" in a.lower()]
    selection, raw, values, labels = standard_case()
    _, state = fill_with(fill_engines.ENGINES[name], "met_pt", (helpers.axes_for("met_pt"),), values, labels,
                         wzmod.standard_weights(raw), selection)
    added = set(state) - {"selection", "channels", "block_cache"}
    assert all(key.startswith(name) for key in added)
    assert not [a for a in vars(wzmod) if "multicell" in a.lower() or "buffer" in a.lower()]
