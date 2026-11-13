"""Round trip of the trigger-SF and b-tag-efficiency correctionlib payload writers."""
import correctionlib
import numpy as np

from qawa.measurements import (BTAG_EFF_FLAVOURS, BTAG_EFF_SYSTEMATICS, TRIGGER_SF_FLAVOURS, TRIGGER_SF_SYSTEMATICS,
                               btag_eff_payload, trigger_sf_payload, write_payload)

TRIGGER_EDGES = [20, 25, 30, 35, 40, 50, 60, 70]
BTAG_PT_EDGES = [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000]
BTAG_ABSETA_EDGES = [0, 0.6, 1.2, 1.8, 2.4]
WPS = ("L", "M", "T")


def centres(edges):
    edges = np.asarray(edges, dtype=float)
    return (edges[1:] + edges[:-1]) / 2


def test_trigger_payload_round_trip(tmp_path):
    n = len(TRIGGER_EDGES) - 1
    grid = np.arange(n * n, dtype=float).reshape(n, n)
    grids = {flavour: {key: grid + 100 * i + 1000 * k for k, key in enumerate(TRIGGER_SF_SYSTEMATICS)}
             for i, flavour in enumerate(TRIGGER_SF_FLAVOURS)}
    path = tmp_path / "trigger.json"
    write_payload(trigger_sf_payload(grids, (TRIGGER_EDGES, TRIGGER_EDGES), "test"), path)
    cset = correctionlib.CorrectionSet.from_file(str(path))
    lead, trail = np.meshgrid(centres(TRIGGER_EDGES), centres(TRIGGER_EDGES), indexing="ij")
    for flavour in TRIGGER_SF_FLAVOURS:
        for key in TRIGGER_SF_SYSTEMATICS:
            values = cset[f"SF_{flavour}"].evaluate(lead.ravel(), trail.ravel(), key)
            assert np.array_equal(values.reshape(n, n), grids[flavour][key])
    assert cset["SF_mm"].evaluate(55., 32.5, "nominal") == grids["mm"]["nominal"][5, 2]
    assert cset["SF_mm"].evaluate(85., 32.5, "up") == grids["mm"]["up"][6, 2]


def test_btag_payload_round_trip(tmp_path):
    npt, neta = len(BTAG_PT_EDGES) - 1, len(BTAG_ABSETA_EDGES) - 1
    grid = np.arange(npt * neta, dtype=float).reshape(npt, neta) / 100

    def value(p, s, w, f):
        return grid + 1 * p + 10 * s + 100 * w + 1000 * f

    processes = ("TTto2L2Nu", "inclusive")
    maps = {process: {syst: {wp: {flavour: value(p, s, w, f) for f, flavour in enumerate(BTAG_EFF_FLAVOURS)}
                             for w, wp in enumerate(WPS)}
                      for s, syst in enumerate(BTAG_EFF_SYSTEMATICS)}
            for p, process in enumerate(processes)}
    path = tmp_path / "btag.json"
    write_payload(btag_eff_payload(maps, (BTAG_PT_EDGES, BTAG_ABSETA_EDGES), WPS), path)
    correction = correctionlib.CorrectionSet.from_file(str(path))["btag_eff"]
    pt, abseta = np.meshgrid(centres(BTAG_PT_EDGES), centres(BTAG_ABSETA_EDGES), indexing="ij")
    for process in processes:
        for syst in BTAG_EFF_SYSTEMATICS:
            for wp in WPS:
                for flavour in BTAG_EFF_FLAVOURS:
                    values = correction.evaluate(process, syst, wp, flavour, pt.ravel(), abseta.ravel())
                    # correctionlib's json reader may round the last bit
                    assert np.allclose(values.reshape(npt, neta), maps[process][syst][wp][flavour], rtol=1e-15, atol=0)
    assert np.isclose(correction.evaluate("inclusive", "up", "T", 5, 2000., 3.0), maps["inclusive"]["up"]["T"][5][-1, -1],
                      rtol=1e-15, atol=0)
