"""Compatibility shim for CMS FlashSim fast-simulation NanoAOD.

FlashSim emits a subset of NanoAOD: the generative models produce the object
collections they were trained on, and everything else -- trigger bits, MET
filters, some ID working points, gen-matching flavour codes -- is simply absent.
This module injects the fields the analysis needs so that a FlashSim file can be
read by the same processor as a FullSim file.

Every insertion is guarded on the field being MISSING.  That is deliberate: as
the FlashSim models gain outputs, the corresponding patch silently becomes a
no-op rather than overwriting real model output with a surrogate.  Call
``patch_flashsim`` once, at the top of ``process()``, before anything touches the
collections.

The patches are NOT all equal in quality.  ``PatchReport.surrogates`` lists the
ones that invent information rather than recover it; those must be understood
before any result derived from them is trusted.  See ``FLASHSIM_CAVEATS``.

SEPARATE PROBLEM, EARLIER IN THE CHAIN: a FlashSim file cannot be READ by stock
uproot at all -- 332 of 671 branches raise UnboundLocalError before any of this
module runs.  See ``qawa.uproot_float16_patch``; it must be enabled at the point
the file is opened, not here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field as _dcfield

import awkward as ak
import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "data")

# MET filters the WZ/ZZ processors cut on.  FlashSim models no detector
# pathology at all (HCAL noise, beam halo, bad PF muons), so there is nothing for
# it to flag: the physically correct value for a fast-sim event is "passes".
MET_FILTERS = (
    "globalSuperTightHalo2016Filter",
    "HBHENoiseFilter",
    "HBHENoiseIsoFilter",
    "EcalDeadCellTriggerPrimitiveFilter",
    "goodVertices",
    "eeBadScFilter",
    "globalTightHalo2016Filter",
    "BadChargedCandidateFilter",
    "BadPFMuonFilter",
)

# Placeholder thresholds on the raw Electron MVA scores, used only when the
# WP boolean branches are absent.  These are NOT the official EGM working
# points -- they are a stand-in so the selection runs.  Override via
# ``patch_flashsim(..., mva_wp={"mvaIso_WP90": ..., "mvaNoIso_WP90": ...})``
# with the real pt/SC-eta-dependent parameterisation from the EGM POG.
DEFAULT_MVA_WP = {"mvaIso_WP90": 0.90, "mvaNoIso_WP90": 0.90}

FLASHSIM_CAVEATS = """\
Electron_mvaIso_WP90 / mvaNoIso_WP90 are recomputed from the raw MVA score with a
  flat placeholder threshold, not the official pt/SC-eta-dependent EGM working
  point.  Electron ID efficiency is therefore approximate.
Flag_* MET filters are all set True.  Correct for fast sim (nothing to flag), but
  it means the data/fast-sim treatment of event cleaning is asymmetric.
HLT paths absent from the FlashSim model are set True, so any trigger requirement
  is effectively only as selective as the subset FlashSim does emit, and the
  trigger_rules PD-veto logic cannot be evaluated faithfully.
Tau_genPartFlav is reconstructed by dR gen-matching rather than read from
  NanoAOD; the prompt/tau-decay/jet-fake split is an approximation of the
  official definition.
"""


@dataclass
class PatchReport:
    """What ``patch_flashsim`` actually changed, for logging and provenance."""

    added: list = _dcfield(default_factory=list)
    skipped: list = _dcfield(default_factory=list)
    surrogates: list = _dcfield(default_factory=list)

    def __str__(self):
        return (
            f"[flashsim_shim] added {len(self.added)}: {', '.join(self.added) or '-'}\n"
            f"[flashsim_shim] already present, not touched {len(self.skipped)}: "
            f"{', '.join(self.skipped) or '-'}\n"
            f"[flashsim_shim] SURROGATE (invented, not recovered): "
            f"{', '.join(self.surrogates) or '-'}"
        )


def _has(record, name):
    return record is not None and name in record.fields


def _trigger_paths(era):
    """HLT path names (no HLT_ prefix) referenced by the era's trigger rules."""
    path = os.path.join(_DATA, f"{era}-trigger-rules.yaml")
    if not os.path.exists(path):
        return []
    # Imported lazily: this module is useful without PyYAML as long as you are
    # not patching HLT, and the rules file is a flat list-of-strings that the
    # regex fallback below parses exactly.
    try:
        import yaml
    except ImportError:
        import re

        with open(path) as fh:
            return sorted({m.strip() for m in re.findall(r"^\s*-\s+(\S+)\s*$", fh.read(), re.M)})
    with open(path) as fh:
        rules = yaml.load(fh, Loader=yaml.FullLoader)
    out = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            out.add(node)

    walk(rules)
    return sorted(out)


def _delta_r2(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return deta**2 + dphi**2


def _tau_gen_part_flav(event, dr_max=0.3):
    """Reconstruct the NanoAOD Tau_genPartFlav code by gen-matching.

    Codes follow the NanoAOD convention:
        1 prompt e, 2 prompt mu, 3 tau->e, 4 tau->mu, 5 tau_h, 0 unmatched (jet).

    Priority is leptonic first, then hadronic, matching the official ordering:
    a reconstructed tau overlapping a gen electron is an electron fake, not a
    tau_h, regardless of whether a GenVisTau also happens to be nearby.

    CLOSURE, measured against the real Tau_genPartFlav branch in FullSim
    WZto3LNu RunIII2024Summer24NanoAODv15 (114,464 reco taus): 95.1% exact
    agreement overall.  Per-category recall:

        prompt e   96.1%    tau->e   96.4%    tau_h          99.5%
        prompt mu  99.4%    tau->mu  99.4%    jet/unmatched  80.1%

    The jet/unmatched category is the weak one: 14.8% of true jet fakes are
    reconstructed here as prompt electrons, because a jet-faked tau sitting near
    a prompt electron is indistinguishable by dR alone.  That category is
    exactly the one a data-driven tau fake-rate estimate depends on, so do not
    use this surrogate to derive or validate fake rates.
    """
    taus = event.Tau
    flav = ak.zeros_like(taus.pt, dtype=np.int32)

    # --- leptonic: prompt and tau-decay electrons/muons from GenDressedLepton
    if "GenDressedLepton" in event.fields:
        gl = event.GenDressedLepton
        pairs = ak.cartesian({"t": taus, "g": gl}, axis=1, nested=True)
        dr2 = _delta_r2(pairs.t.eta, pairs.t.phi, pairs.g.eta, pairs.g.phi)
        close = dr2 < dr_max**2
        absid = abs(pairs.g.pdgId)
        from_tau = pairs.g.hasTauAnc if "hasTauAnc" in gl.fields else ak.zeros_like(absid, dtype=bool)
        for code, sel in (
            (1, close & (absid == 11) & ~from_tau),
            (2, close & (absid == 13) & ~from_tau),
            (3, close & (absid == 11) & from_tau),
            (4, close & (absid == 13) & from_tau),
        ):
            flav = ak.where((flav == 0) & ak.any(sel, axis=-1), code, flav)

    # --- hadronic: GenVisTau
    if "GenVisTau" in event.fields:
        gv = event.GenVisTau
        pairs = ak.cartesian({"t": taus, "g": gv}, axis=1, nested=True)
        dr2 = _delta_r2(pairs.t.eta, pairs.t.phi, pairs.g.eta, pairs.g.phi)
        flav = ak.where((flav == 0) & ak.any(dr2 < dr_max**2, axis=-1), 5, flav)

    return ak.values_astype(flav, np.uint8)


def flashsim_schema():
    """NanoAODSchema subclass tolerant of two FlashSim output defects.

    Both kill the read in coffea's ``_build_collections`` with

        KeyError: 'content'   (nanoaod.py)

    before any analysis code runs, so neither can be handled downstream.

    (a) ORPHAN COUNTERS.  ``nFakeJets`` and ``nSVFromPU`` are emitted with no
        ``FakeJets_*`` / ``SVFromPU_*`` fields and no same-named array branch.
        Note ``nLHEPdfWeight``, ``nPSWeight``, ``nLHEScaleWeight`` and
        ``nLHEReweightingWeight`` are NOT orphans -- each has an array branch of
        the same name minus the ``n`` -- and are left alone.

    (b) MIXED JAGGED/SCALAR COLLECTIONS.  FlashSim writes the ``PVBS``
        collection twice over: a jagged copy indexed by the counter
        (``PVBS_x[nPVBS]``) AND a scalar copy of every field alongside it
        (``PVBS_x_scalar``).  coffea sees ``nPVBS``, builds offsets, and tries
        to zip the whole collection as jagged; the scalar members make
        ``zip_forms`` return a bare RecordArray with no ``content``.  The
        ``_scalar`` duplicates look like writer scaffolding that should not
        have been emitted.  Dropped here; nothing in the analysis reads PVBS.

    Both rules are written generically against the form rather than against
    those branch names, so FlashSim output gaining or losing such branches
    needs no change here.
    """
    import re as _re

    from coffea.nanoevents import NanoAODSchema

    def _is_list(content):
        return "list" in str(content.get("class", "")).lower()

    class FlashSimNanoAODSchema(NanoAODSchema):
        def __init__(self, base_form, *args, **kwargs):
            fields = list(base_form["fields"])
            contents = list(base_form["contents"])
            counted = {f[1:] for f in fields if _re.match(r"^n[A-Z]", f)}

            keep, dropped_orphans, dropped_scalars = [], [], []
            for i, f in enumerate(fields):
                # (a) a counter that counts nothing
                if _re.match(r"^n[A-Z]", f):
                    name = f[1:]
                    if not any(g == name or g.startswith(name + "_") for g in fields):
                        dropped_orphans.append(f)
                        continue
                # (b) a non-jagged member of a counted collection
                head = f.split("_")[0]
                if head in counted and "_" in f and not _is_list(contents[i]):
                    dropped_scalars.append(f)
                    continue
                keep.append(i)

            if dropped_orphans or dropped_scalars:
                base_form = dict(base_form)
                base_form["fields"] = [fields[i] for i in keep]
                base_form["contents"] = [contents[i] for i in keep]
            self.dropped_orphan_counters = dropped_orphans
            self.dropped_scalar_members = dropped_scalars
            super().__init__(base_form, *args, **kwargs)

    return FlashSimNanoAODSchema


def patch_flashsim(event, era, mva_wp=None, tau_match_dr=0.3, verbose=True):
    """Add NanoAOD fields absent from a FlashSim file.  Returns (event, report).

    Idempotent and purely additive: a field that already exists is never
    overwritten, so this is safe to run on a FullSim file (it becomes a no-op)
    and degrades gracefully as FlashSim's outputs grow.
    """
    mva_wp = {**DEFAULT_MVA_WP, **(mva_wp or {})}
    rep = PatchReport()

    # ---- 1. Muon_pdgId ---------------------------------------------------
    # Not physics: fully determined by the charge.  Critically, build_leptons
    # does ak.concatenate([muons, electrons]), and awkward silently returns only
    # the INTERSECTION of fields.  Electron_pdgId exists and Muon_pdgId does not,
    # so without this patch pdgId vanishes from the MERGED lepton collection with
    # no exception raised -- and pdgId drives the OSSF pairing and the trigger SF
    # flavour masks.
    if "Muon" in event.fields:
        if not _has(event.Muon, "pdgId"):
            pdg = ak.values_astype(-13 * event.Muon.charge, np.int32)
            event = ak.with_field(event, ak.with_field(event.Muon, pdg, "pdgId"), "Muon")
            rep.added.append("Muon_pdgId")
        else:
            rep.skipped.append("Muon_pdgId")

    # ---- 2. lepton field parity for the concatenate ----------------------
    # These are not used by the physics, but their absence on one flavour and
    # not the other shrinks the Muon-Electron intersection and silently drops
    # them from the merged collection.  Sentinel -1 so they cannot be mistaken
    # for real information.
    for coll, fields in (("Muon", ("jetNDauCharged",)), ("Electron", ("fsrPhotonIdx",))):
        if coll not in event.fields:
            continue
        rec = event[coll]
        for f in fields:
            if not _has(rec, f):
                rec = ak.with_field(rec, ak.values_astype(ak.zeros_like(rec.pt) - 1, np.int32), f)
                rep.added.append(f"{coll}_{f}")
        event = ak.with_field(event, rec, coll)

    # ---- 3. MET filters --------------------------------------------------
    flag = event.Flag if "Flag" in event.fields else None
    missing_flags = [f for f in MET_FILTERS if not _has(flag, f)]
    if missing_flags:
        ones = ak.ones_like(event.PV.npvs, dtype=bool) if "PV" in event.fields else \
               ak.Array(np.ones(len(event), dtype=bool))
        payload = {f: (flag[f] if _has(flag, f) else ones) for f in MET_FILTERS}
        if flag is not None:
            for f in flag.fields:
                payload.setdefault(f, flag[f])
        event = ak.with_field(event, ak.zip(payload), "Flag")
        rep.added.extend(f"Flag_{f}" for f in missing_flags)
        rep.surrogates.append(f"Flag_* ({len(missing_flags)} set True)")

    # ---- 4. Electron MVA working points ----------------------------------
    if "Electron" in event.fields:
        el = event.Electron
        for wp, raw in (("mvaIso_WP90", "mvaIso"), ("mvaNoIso_WP90", "mvaNoIso")):
            if _has(el, wp):
                rep.skipped.append(f"Electron_{wp}")
                continue
            if not _has(el, raw):
                continue
            el = ak.with_field(el, el[raw] > mva_wp[wp], wp)
            rep.added.append(f"Electron_{wp}")
            rep.surrogates.append(f"Electron_{wp} (flat cut {mva_wp[wp]} on {raw})")
        event = ak.with_field(event, el, "Electron")

    # ---- 5. Tau_genPartFlav ----------------------------------------------
    if "Tau" in event.fields and not _has(event.Tau, "genPartFlav"):
        if "GenVisTau" in event.fields or "GenDressedLepton" in event.fields:
            flav = _tau_gen_part_flav(event, dr_max=tau_match_dr)
            event = ak.with_field(event, ak.with_field(event.Tau, flav, "genPartFlav"), "Tau")
            rep.added.append("Tau_genPartFlav")
            rep.surrogates.append(f"Tau_genPartFlav (dR<{tau_match_dr} gen-match)")

    # ---- 6. HLT paths ----------------------------------------------------
    # Dynamic: only the paths this era's trigger rules actually reference, and
    # only those FlashSim does not already provide.
    wanted = _trigger_paths(era)
    if wanted:
        hlt = event.HLT if "HLT" in event.fields else None
        missing_hlt = [p for p in wanted if not _has(hlt, p)]
        if missing_hlt:
            ones = ak.Array(np.ones(len(event), dtype=bool))
            payload = {p: (hlt[p] if _has(hlt, p) else ones) for p in wanted}
            if hlt is not None:
                for p in hlt.fields:
                    payload.setdefault(p, hlt[p])
            event = ak.with_field(event, ak.zip(payload), "HLT")
            rep.added.extend(f"HLT_{p}" for p in missing_hlt)
            rep.surrogates.append(
                f"HLT_* ({len(missing_hlt)}/{len(wanted)} set True; "
                f"{len(wanted) - len(missing_hlt)} genuinely from FlashSim)"
            )

    if verbose:
        print(str(rep))
    return event, rep
