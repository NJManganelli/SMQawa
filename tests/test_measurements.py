"""Derivation of the trigger SF and b-tag efficiency payloads from synthetic measurement histograms."""
import correctionlib
import numpy as np
import pytest
from coffea.util import coffea_console
from scipy import stats

from conftest import CVMFS_HEAD, input_path
from qawa import measurements
import qawa.process.wztau2lnu_inclusive as wzmod

CVMFS, needs_cvmfs = input_path(CVMFS_HEAD, "cms-griddata.cern.ch")
COVERAGE = stats.norm.cdf(1) - stats.norm.cdf(-1)
WPS = ("L", "M", "T", "XT", "XXT")


def cp(k, n):
    lo = stats.beta.ppf((1 - COVERAGE) / 2, k, n - k + 1) if k > 0 else 0.
    hi = stats.beta.ppf((1 + COVERAGE) / 2, k + 1, n - k) if k < n else 1.
    return lo, hi


def test_clopper_pearson_efficiency():
    eff, lo, hi = measurements.efficiency_interval([0, 5, 10], [10, 10, 10])
    assert np.allclose(eff, [0, 0.5, 1])
    expected = np.array([cp(k, 10) for k in (0, 5, 10)])
    assert np.allclose(lo, expected[:, 0]) and np.allclose(hi, expected[:, 1])
    assert measurements.efficiency_interval([0], [0])[0] == 0


def test_effective_counts():
    k, n = np.array([3., 7.]), np.array([10., 10.])
    eff, k_eff, n_eff, filled, clipped = measurements.effective_counts(2 * k, 2 * n, 4 * n)
    assert np.allclose(eff, k / n) and np.allclose(k_eff, k) and np.allclose(n_eff, n)
    assert filled.all() and not clipped.any()
    # weights 1, 1, 2 (pass: the weight-2 entry): W = 4, V = 6
    eff, k_eff, n_eff, _, _ = measurements.effective_counts([2.], [4.], [6.])
    assert np.allclose(eff, 0.5) and np.allclose(n_eff, 16 / 6) and np.allclose(k_eff, 8 / 6)
    eff, k_eff, n_eff, filled, clipped = measurements.effective_counts([-1., 5., 1.], [4., 4., -2.], [6., 6., 6.])
    assert np.allclose(eff, [0., 1., 0.]) and clipped.tolist() == [True, True, False] and filled.tolist() == [True, True, False]
    _, lo, hi = measurements.efficiency_interval(k_eff, n_eff)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


@pytest.fixture(scope="module")
def booked():
    if not CVMFS:
        pytest.skip("cvmfs not available")
    return wzmod.wzinclusive_processor(era="2024", version="v15").build_histos


def fill_trigger(h, channel, lead, trail, reference, lepton, count, weight=1.0):
    h.fill(channel=channel, systematic="nominal", trigger_lead_pt=np.full(count, lead), trigger_trail_pt=np.full(count, trail),
           trigger_reference=np.full(count, reference), trigger_lepton=np.full(count, lepton), weight=np.full(count, weight))


def expected_bin(n10, n11, w_cells):
    """w_cells[(ref, lep)] are the MC weight sums, filled with weight-2 entries."""
    eps_d = n11 / (n10 + n11)
    lo_d, hi_d = cp(n11, n10 + n11)
    w1 = w_cells[1, 0] + w_cells[1, 1]
    eps_m = w_cells[1, 1] / w1
    n_eff = w1 ** 2 / (2 * w1)
    lo_m, hi_m = cp(eps_m * n_eff, n_eff)
    total = sum(w_cells.values())
    alpha = (w_cells[0, 1] + w_cells[1, 1]) * w1 / (w_cells[1, 1] * total)
    sf = eps_d / eps_m
    d_up = sf * np.hypot((hi_d - eps_d) / eps_d, (eps_m - lo_m) / eps_m)
    d_dn = sf * np.hypot((eps_d - lo_d) / eps_d, (hi_m - eps_m) / eps_m)
    a = sf * abs(1 - alpha)
    return {"nominal": sf, "up": sf + d_up, "down": sf - d_dn, "alpha_up": sf + np.hypot(d_up, a),
            "alpha_down": sf - np.hypot(d_dn, a)}, alpha, (hi_d - eps_d, eps_m - lo_m)


def test_trigger_sf_arithmetic(booked, tmp_path):
    data, mc = booked()["trigger_measurement"], booked()["trigger_measurement"]
    bins = {"first": ((45., 32.5), (4, 2), (1, 199), {(0, 0): 30., (0, 1): 630., (1, 0): 30., (1, 1): 170.}),
            "second": ((100., 45.), (6, 4), (10, 90), {(0, 0): 100., (0, 1): 100., (1, 0): 20., (1, 1): 180.})}
    for (lead, trail), _, (n10, n11), w_cells in bins.values():
        fill_trigger(data, "trigger-mm", lead, trail, 1, 0, n10)
        fill_trigger(data, "trigger-mm", lead, trail, 1, 1, n11)
        for (ref, lep), w in w_cells.items():
            fill_trigger(mc, "trigger-mm", 65. if lead > 70 else lead, trail, ref, lep, int(w / 2), 2.0)
    fill_trigger(data, "trigger-mm", 37.5, 22.5, 1, 0, 5)
    fill_trigger(mc, "trigger-mm", 37.5, 22.5, 1, 1, 10, 2.0)
    fill_trigger(data, "trigger-mm", 37.5, 27.5, 1, 1, 5)
    fill_trigger(mc, "trigger-mm", 37.5, 27.5, 1, 0, 10, 2.0)

    grids, edges = measurements.derive_trigger_sf([data], [mc])
    mm = grids["mm"]
    alphas = []
    for _, (i, j), (n10, n11), w_cells in bins.values():
        expected, alpha, (sigma_d_up, sigma_m_down) = expected_bin(n10, n11, w_cells)
        alphas.append(alpha)
        for key, value in expected.items():
            assert np.isclose(mm[key][i, j], value, rtol=1e-10), (i, j, key)
    first = expected_bin(1, 199, bins["first"][3])
    assert first[2][0] < 0.1 * first[2][1]
    assert mm["up"][4, 2] - mm["nominal"][4, 2] > 0.9 * mm["nominal"][4, 2] * first[2][1] / (170 / 200)
    assert not np.isclose(*alphas)
    assert all(mm[key][0, 1] == 1.0 for key in mm)
    for i, j in [(2, 0), (3, 0), (3, 1)]:
        assert [mm[key][i, j] for key in measurements.TRIGGER_SF_SYSTEMATICS] == [1., 2., 0., 2., 0.], (i, j)
    path = tmp_path / "trigger.json"
    measurements.write_payload(measurements.trigger_sf_payload(grids, edges, "test"), path)
    cset = correctionlib.schemav2.CorrectionSet.model_validate_json(path.read_text())
    assert all(np.all(np.isfinite(c.data.content[k].value.content)) for c in cset.corrections for k in range(5))


def records_with(h, dataset, kind):
    return {dataset: {"hist": {"trigger_measurement": h}, "sumw": 1.0, "kind": kind}}


def test_trigger_guard(booked):
    filled = booked()["trigger_measurement"]
    fill_trigger(filled, "trigger-mm", 45., 32.5, 1, 1, 3)
    good = {**records_with(filled, "JetMET0_Run2024C", "data"), **records_with(filled, "TTto2L2Nu_x", "mc")}
    assert measurements.trigger_inputs(good, "2024")
    for dataset, kind in [("Muon0_Run2024C", "data"), ("WZto3LNu_x", "mc")]:
        with pytest.raises(ValueError, match=dataset):
            measurements.trigger_inputs({**good, **records_with(filled, dataset, kind)}, "2024")


def fill_btag(h, flavour, pt, abseta, wp_counts, weight=1.0):
    for npassed, count in enumerate(wp_counts):
        h.fill(channel="btag-jets", systematic="nominal", btag_flavour=np.full(count, flavour),
               btag_pt=np.full(count, pt), btag_abseta=np.full(count, abseta), btag_wp=np.full(count, npassed),
               weight=np.full(count, weight))


def test_btag_efficiency(booked):
    first, second = booked()["btag_measurement"], booked()["btag_measurement"]
    fill_btag(first, 5, 40., 0.3, [10, 10, 20, 20, 20, 20])
    fill_btag(first, 5, 2000., 0.3, [0, 0, 0, 0, 0, 5])
    fill_btag(second, 5, 40., 0.3, [50, 0, 0, 0, 0, 50])
    records = {"TTto2L2Nu_x": {"hist": {"btag_measurement": first}, "sumw": 10.0, "kind": "mc"},
               "WZto3LNu_x": {"hist": {"btag_measurement": second}, "sumw": 100.0, "kind": "signal"}}
    xsections = {"TTto2L2Nu_x": {"xsec": 2.0, "br": 1.0, "kr": 1.0}, "WZto3LNu_x": {"xsec": 5.0, "br": 1.0, "kr": 2.0}}
    with coffea_console.capture() as captured:
        maps, edges, wps = measurements.derive_btag_eff(records, "v15", xsections)
    assert wps == WPS and set(maps) == {"TTto2L2Nu_x", "WZto3LNu_x", "inclusive"}
    pt_bin, eta_bin = 1, 0
    counts = np.array([10, 10, 20, 20, 20, 20])
    for k, wp in enumerate(WPS):
        assert np.isclose(maps["TTto2L2Nu_x"]["nominal"][wp][5][pt_bin, eta_bin], counts[k + 1:].sum() / counts.sum())
        assert maps["TTto2L2Nu_x"]["nominal"][wp][5][-1, eta_bin] == 1.0
    scale_tt, scale_wz = 2.0 / 10.0, 10.0 / 100.0
    expected = (scale_tt * 90 + scale_wz * 50) / (scale_tt * 100 + scale_wz * 100)
    assert np.isclose(maps["inclusive"]["nominal"]["L"][5][pt_bin, eta_bin], expected)
    empty = [maps["TTto2L2Nu_x"][s]["L"][0][2, 2] for s in ("nominal", "down", "up")]
    assert empty == [0., 0., 1.]
    n_cells = 3 * (len(edges[0]) - 2) * (len(edges[1]) - 1) * len(WPS)
    assert f"btag_eff TTto2L2Nu_x: {n_cells - 2 * len(WPS)} empty" in captured.get()


def test_cli_requires_outputs(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        measurements.main(["--era", "2024", "--version", "v15", "-i", str(tmp_path / "x.pkl.gz"), "--derive", "trigger"])
    assert exit_info.value.code == 2
