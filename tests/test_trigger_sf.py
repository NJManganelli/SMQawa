#!/usr/bin/env python
"""Unit test: ROOT vs correctionlib di-lepton trigger scale factors (inc-WZ).

For each pre-2024 era (2016/2017/2018) this exercises the *actual* trigger-SF
code of the WZ inclusive processor,
:meth:`qawa.process.wztau2lnu_inclusive.wzinclusive_processor._add_trigger_sf`,
along both branches:

* the **legacy ROOT path** -- ``digitize`` + ``trig_sf_map`` lookup, which is
  eta-dependent (one of the 16 ``trgSF<flavor><region>`` histograms per event);
* the **correctionlib path** -- ``SF_xx.evaluate(lead_pt, subl_pt, syst)`` on the
  JSON produced by ``convert_trigger_sf.py``, exactly as used for 2024.

The correctionlib file collapses eta by inverse-variance combination of the four
regions (see ``convert_trigger_sf.py``). The test therefore drives the legacy
path once per eta region, combines the four results with the same
inverse-variance rule (:func:`convert_trigger_sf.combine_regions`), and asserts
the correctionlib output matches -- nominal, up and down -- bin for bin.

Run directly (no pytest required)::

    python test_trigger_sf.py

or under pytest if available::

    pytest test_trigger_sf.py
"""
from __future__ import annotations

import os

import numpy as np
import awkward as ak
import uproot
from coffea.analysis_tools import Weights

from convert_trigger_sf import (
    ETA_REGIONS, FLAVORS, CORRECTION_NAME, combine_regions, root_path, convert,
)
from qawa.process.wztau2lnu_inclusive import wzinclusive_processor

import correctionlib

ERAS = ("2016", "2017", "2018")
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "src", "qawa", "data", "trigger_sf")

# Unbound method -> call as _add_trigger_sf(shim_self, weights, lead, subl, clib).
_add_trigger_sf = wzinclusive_processor._add_trigger_sf

# pdgId of (leading, subleading) lepton for each flavor pair (first=lead).
_PDG = {"E": 11, "M": 13}
# eta used to place a lepton in the barrel ("B", <1.5) or endcap ("E", >1.5)
# region. The processor uses raw eta (not |eta|), so positive values are used.
_ETA = {"B": 0.5, "E": 2.0}

# Bin centres of the [20,25,30,35,40,50,60,70] pt axis, plus a high-pt point
# (85) that probes the correctionlib "clamp" overflow against the legacy open
# last bin. Only physical pairs (lead_pt >= subl_pt) are tested.
_PT_CENTERS = [22.5, 27.5, 32.5, 37.5, 45.0, 55.0, 65.0, 85.0]


class _TrigSfMapShim:
    """Minimal stand-in for the processor, carrying only ``trig_sf_map``."""

    def __init__(self, era: str):
        # Mirror wztau2lnu_inclusive.py:391-394 exactly.
        with uproot.open(root_path(era, _DATA_DIR)) as _fn:
            _hvalue = np.dstack([_fn[_hn].values() for _hn in _fn.keys()] + [np.ones((7, 7))])
            _herror = np.dstack([np.sqrt(_fn[_hn].variances()) for _hn in _fn.keys()] + [np.zeros((7, 7))])
            self.trig_sf_map = np.stack([_hvalue, _herror], axis=-1)


class _ClibShim:
    """Minimal CorrectionlibHandler stand-in exposing one ``trigger_sf`` set."""

    def __init__(self, cset):
        self._cset = cset

    def keys(self):
        return ["trigger_sf"]

    def getCorrectionSet(self, key):
        assert key == "trigger_sf"
        return self._cset


def _make_leptons(lead_flavor: str, subl_flavor: str, lead_region: str,
                  subl_region: str, lead_pt, subl_pt):
    """Build awkward (lead, subl) lepton records as the processor expects."""
    n = len(lead_pt)
    lead = ak.zip({
        "pt": ak.Array(np.asarray(lead_pt, dtype=float)),
        "eta": ak.Array(np.full(n, _ETA[lead_region])),
        "pdgId": ak.Array(np.full(n, _PDG[lead_flavor], dtype=np.int64)),
    })
    subl = ak.zip({
        "pt": ak.Array(np.asarray(subl_pt, dtype=float)),
        "eta": ak.Array(np.full(n, _ETA[subl_region])),
        "pdgId": ak.Array(np.full(n, _PDG[subl_flavor], dtype=np.int64)),
    })
    return lead, subl


def _legacy_lookup(shim, flavor, region, lead_pt, subl_pt):
    """Drive the real legacy branch; return (nominal, sigma) arrays."""
    lead, subl = _make_leptons(flavor[0], flavor[1], region[0], region[1],
                               lead_pt, subl_pt)
    weights = Weights(len(lead_pt), storeIndividual=True)
    _add_trigger_sf(shim, weights, lead, subl, clibhandler=None)
    nominal = np.asarray(weights.weight())
    up = np.asarray(weights.weight(modifier="triggerSFUp"))
    return nominal, up - nominal


def _clib_lookup(clib_shim, flavor, lead_pt, subl_pt):
    """Drive the real correctionlib branch; return (nominal, up, down) arrays."""
    # eta is irrelevant on this branch; barrel is fine.
    lead, subl = _make_leptons(flavor[0], flavor[1], "B", "B", lead_pt, subl_pt)
    weights = Weights(len(lead_pt), storeIndividual=True)
    _add_trigger_sf(clib_shim, weights, lead, subl, clibhandler=clib_shim)
    return (np.asarray(weights.weight()),
            np.asarray(weights.weight(modifier="triggerSFUp")),
            np.asarray(weights.weight(modifier="triggerSFDown")))


def _physical_points():
    """All physical (lead_pt >= subl_pt) bin-centre pairs as two arrays."""
    lead, subl = [], []
    for i, lpt in enumerate(_PT_CENTERS):
        for jpt in _PT_CENTERS[:i + 1]:
            lead.append(lpt)
            subl.append(jpt)
    return np.array(lead), np.array(subl)


def _check_era(era: str):
    lead_pt, subl_pt = _physical_points()
    shim = _TrigSfMapShim(era)
    cset = correctionlib.CorrectionSet.from_file(
        os.path.join(_DATA_DIR, f"triggerSF_{era}.json"))
    clib_shim = _ClibShim(cset)

    for flavor in FLAVORS:
        # Legacy lookup for each of the four eta regions ...
        noms, sigs = [], []
        for region in ETA_REGIONS:
            nominal, sigma = _legacy_lookup(shim, flavor, region, lead_pt, subl_pt)
            noms.append(nominal)
            sigs.append(sigma)
        # ... combined with the same rule the converter used.
        comb_nom, comb_sig = combine_regions(noms, sigs)

        # Real correctionlib branch on the converted JSON.
        clib_nom, clib_up, clib_dn = _clib_lookup(clib_shim, flavor, lead_pt, subl_pt)

        name = CORRECTION_NAME[flavor]
        assert np.allclose(clib_nom, comb_nom, rtol=0, atol=1e-9), \
            f"{era} {name} nominal mismatch:\n{clib_nom}\nvs\n{comb_nom}"
        assert np.allclose(clib_up, comb_nom + comb_sig, rtol=0, atol=1e-9), \
            f"{era} {name} up mismatch"
        assert np.allclose(clib_dn, comb_nom - comb_sig, rtol=0, atol=1e-9), \
            f"{era} {name} down mismatch"
    print(f"[{era}] OK: legacy(ROOT, 4 eta regions combined) == correctionlib JSON "
          f"for {', '.join(CORRECTION_NAME[f] for f in FLAVORS)} "
          f"({len(lead_pt)} physical pt pairs each)")


def test_trigger_sf_root_vs_correctionlib():
    """Pytest entry point: validate every pre-2024 era."""
    for era in ERAS:
        _check_era(era)


def test_clamp_overflow_matches_last_bin():
    """A very-high-pt pair must clamp to the last pt bin on both paths."""
    for era in ERAS:
        shim = _TrigSfMapShim(era)
        cset = correctionlib.CorrectionSet.from_file(
            os.path.join(_DATA_DIR, f"triggerSF_{era}.json"))
        clib_shim = _ClibShim(cset)
        # 500 GeV (well past 70) and 65 GeV (last real centre) -> both last bin.
        lead_pt = np.array([500.0, 65.0])
        subl_pt = np.array([65.0, 65.0])
        for flavor in FLAVORS:
            n0, _ = _legacy_lookup(shim, flavor, "BB", np.array([65.0, 65.0]),
                                   np.array([65.0, 65.0]))
            clib_nom, _, _ = _clib_lookup(clib_shim, flavor, lead_pt, subl_pt)
            # both entries clamp to (lead_bin=6, subl_bin=6); legacy 65/65 is the
            # same bin, so correctionlib at 500/65 must equal it.
            assert np.allclose(clib_nom[0], clib_nom[1], atol=1e-9), \
                f"{era} {CORRECTION_NAME[flavor]} clamp inconsistent"


def main():
    # Regenerate the JSONs from the ROOT files, then validate.
    for era in ERAS:
        convert(era, outdir=_DATA_DIR, data_dir=_DATA_DIR)
    test_trigger_sf_root_vs_correctionlib()
    test_clamp_overflow_matches_last_bin()
    print("All trigger-SF conversion tests passed.")


if __name__ == "__main__":
    main()
