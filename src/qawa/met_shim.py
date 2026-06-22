"""
met_shim.py  --  NanoAOD v9 -> v15 MET unclustered-energy shims for coffea
==========================================================================

Background
----------
coffea's ``CorrectedMETFactory`` builds the "unclustered energy" up/down MET
variations from two per-event fields on the MET record:

    MET[name_map["UnClusteredEnergyDeltaX"]]   # default mapping -> "MetUnclustEnUpDeltaX"
    MET[name_map["UnClusteredEnergyDeltaY"]]   # default mapping -> "MetUnclustEnUpDeltaY"

It then forms, per event, the symmetric variation
    up   = nominal_corrected_met + (dx, dy)
    down = nominal_corrected_met - (dx, dy)
(see ``corrected_polar_met(..., positive=True/False, dx=dx, dy=dy)``).

NanoAOD v9 stores these directly:

    MET_MetUnclustEnUpDeltaX, MET_MetUnclustEnUpDeltaY     (PF CHS "MET")

NanoAOD v15 *renames and restructures* the MET collections:

    * the old PF-CHS "MET"  ->  "PFMET"
    * the new default       ->  "PuppiMET"   (PUPPI)
    * raw variants          ->  "RawPFMET", "RawPuppiMET"

and neither "PuppiMET" nor "PFMET" carries ``MetUnclustEnUpDeltaX/Y`` anymore.
Instead v15 stores the *already-shifted* unclustered MET as four fields plus a
covariance breakdown:

    <col>_ptUnclusteredUp,  <col>_phiUnclusteredUp
    <col>_ptUnclusteredDown,<col>_phiUnclusteredDown
    <col>_covXX, <col>_covXY, <col>_covYY
    <col>_sumPtUnclustered, <col>_significance

Empirically (verified on the SMQawaTestFiles v15 samples) the Up/Down shifts are
symmetric about the nominal, i.e. the Cartesian delta reconstructed from "Up"
equals minus the one reconstructed from "Down" to rounding precision. So we can
recover exactly the quantity coffea wants:

    DeltaX = ptUnclusteredUp*cos(phiUnclusteredUp) - pt*cos(phi)
    DeltaY = ptUnclusteredUp*sin(phiUnclusteredUp) - pt*sin(phi)

A second, independent v15 wrinkle: coffea's ``NanoAODSchema`` (as of 2025.12)
maps the collection *names* it knows ("MET", "PuppiMET", "RawMET",
"RawPuppiMET", "CaloMET", "ChsMET", "TkMET", "GenMET") to the ``MissingET``
behavior, but it does NOT know the new v15 names "PFMET", "RawPFMET", "TrkMET",
"FiducialMET" -- those load as a plain ``NanoCollection`` and therefore lack the
two-vector / ``add_systematic`` methods. ``PuppiMET`` *is* recognized, so if you
stay on PUPPI MET you only need the field shim; if you want PF-CHS MET in v15
you also need the behavior shim ``as_missinget``.

This module provides small, dependency-light shims that turn a v15 MET record
into something ``CorrectedMETFactory`` (and the rest of the existing SMQawa code)
can consume unchanged. Nothing here mutates inputs in place: every function
returns a new awkward array, mirroring the ``ak.with_field`` style already used
in ``qawa.jme``.

Intended call site (qawa/process/wztau2lnu_inclusive.py, just before
``self._jmeu.corrected_met(...)``)::

    from met_shim import prepare_met_for_factory
    met_to_correct = prepare_met_for_factory(
        event.MET if is_v9 else event.PuppiMET
    )

The default ``jec_name_map`` in ``qawa.jme`` already maps
``UnClusteredEnergyDeltaX/Y -> MetUnclustEnUpDeltaX/Y`` so no map change is
needed once the fields are present.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

# Field names coffea's CorrectedMETFactory expects (the v9 spelling).
_DELTA_X = "MetUnclustEnUpDeltaX"
_DELTA_Y = "MetUnclustEnUpDeltaY"

# v15 precomputed unclustered-shift fields.
_V15_REQUIRED = ("ptUnclusteredUp", "phiUnclusteredUp")
_V15_OPTIONAL_DOWN = ("ptUnclusteredDown", "phiUnclusteredDown")


def has_v9_deltas(met) -> bool:
    """True if the record already carries the v9-style DeltaX/DeltaY fields."""
    f = met.fields
    return _DELTA_X in f and _DELTA_Y in f


def has_v15_unclustered(met) -> bool:
    """True if the record carries the v15 precomputed unclustered Up fields."""
    f = met.fields
    return all(name in f for name in _V15_REQUIRED)


def ensure_unclustered_deltas(met, symmetrize: bool = False):
    """Return ``met`` guaranteed to have ``MetUnclustEnUpDeltaX/Y`` fields.

    * If the fields already exist (NanoAOD v9 ``MET``) the record is returned
      untouched.
    * If only the v15 precomputed ``pt/phiUnclusteredUp`` (and optionally Down)
      fields exist, the Cartesian deltas are reconstructed and attached.

    Parameters
    ----------
    met : awkward record array
        A single MET object (one record per event) with at least ``pt`` and
        ``phi``.
    symmetrize : bool, default False
        If True and the Down fields are present, use the average of the Up delta
        and the negated Down delta:  ``delta = (deltaUp - deltaDown) / 2``.
        This removes the tiny per-field rounding asymmetry seen in the stored
        branches. If False (default) the Up variation alone defines the delta,
        which is the exact analogue of the single v9 ``...UpDelta...`` branch and
        of what coffea applies symmetrically.

    Notes
    -----
    coffea applies these deltas symmetrically (up = nominal + delta,
    down = nominal - delta), so storing only the Up-derived delta reproduces the
    Run 2 behavior bit-for-bit in spirit.
    """
    if has_v9_deltas(met):
        return met

    if not has_v15_unclustered(met):
        raise ValueError(
            "MET record has neither v9 'MetUnclustEnUpDeltaX/Y' nor v15 "
            f"'pt/phiUnclusteredUp'. Available fields: {met.fields}"
        )

    px = met.pt * np.cos(met.phi)
    py = met.pt * np.sin(met.phi)

    pxU = met.ptUnclusteredUp * np.cos(met.phiUnclusteredUp)
    pyU = met.ptUnclusteredUp * np.sin(met.phiUnclusteredUp)
    dX = pxU - px
    dY = pyU - py

    if symmetrize and all(name in met.fields for name in _V15_OPTIONAL_DOWN):
        pxD = met.ptUnclusteredDown * np.cos(met.phiUnclusteredDown)
        pyD = met.ptUnclusteredDown * np.sin(met.phiUnclusteredDown)
        # down delta is (D - nominal); a symmetric estimate averages up and -down
        dX = 0.5 * (dX - (pxD - px))
        dY = 0.5 * (dY - (pyD - py))

    out = ak.with_field(met, ak.values_astype(dX, np.float32), _DELTA_X)
    out = ak.with_field(out, ak.values_astype(dY, np.float32), _DELTA_Y)
    return out


def as_missinget(met):
    """Re-stamp a MET record with coffea's ``MissingET`` behavior.

    Needed only for v15 collections whose *name* coffea's NanoAODSchema does not
    recognize as MET (notably ``PFMET``, ``RawPFMET``, ``TrkMET``,
    ``FiducialMET``), which otherwise load as a plain ``NanoCollection`` and lack
    the two-vector / ``add_systematic`` methods. ``PuppiMET`` is already
    recognized and needs no re-stamping.

    The original ``behavior`` dict is reused so that the ``MissingET`` mixin
    resolves correctly.
    """
    from coffea.nanoevents.methods import nanoaod

    behavior = getattr(met, "behavior", None) or nanoaod.behavior
    return ak.with_name(met, "MissingET", behavior=behavior)


def prepare_met_for_factory(met, force_missinget_behavior: bool = False,
                            symmetrize: bool = False):
    """One-stop shim: ensure MissingET behavior (optional) + Delta fields.

    Use this immediately before handing a MET record to
    ``CorrectedMETFactory.build`` (i.e. ``JMEUncertainty.corrected_met``).

    Parameters
    ----------
    met : awkward record array
        e.g. ``event.MET`` (v9) or ``event.PuppiMET`` / ``event.PFMET`` (v15).
    force_missinget_behavior : bool, default False
        Set True when passing a v15 ``PFMET``-family record that loaded as a bare
        NanoCollection. Harmless (idempotent) for records already typed as
        MissingET.
    symmetrize : bool, default False
        Forwarded to :func:`ensure_unclustered_deltas`.
    """
    if force_missinget_behavior:
        met = as_missinget(met)
    return ensure_unclustered_deltas(met, symmetrize=symmetrize)
