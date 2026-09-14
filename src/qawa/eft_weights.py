"""SMEFT reweighting: point discovery, structure-constant decomposition.

Generator- and simulation-agnostic.  It reads only the EFT weight branches, so
the same code path serves FlashSim, FullSim and NanoGEN samples, and samples
from different productions can be combined as long as they share a
parameterisation.

WHAT THIS COMPUTES
A dim-6 reweighted sample carries, per event, a weight at each of several
Wilson-coefficient points.  The physical content is the quadratic form

    w(c) = S + sum_i A_i c_i + sum_i B_ii c_i^2 + sum_{i<j} C_ij c_i c_j

so for N operators there are (N+1)(N+2)/2 independent structure constants.
The map from point weights to structure constants is a FIXED matrix set only by
the coefficient values at each point -- it has no data dependence.

WHY EVENT-LEVEL RATHER THAN PER-BIN
Histogram filling is linear, so filling one histogram per structure constant
gives identical central values to filling one per reweight point and inverting
bin by bin.  But it avoids a per-bin matrix inversion (no noise amplification
where the templates are individually poorly determined), it gives correct
Weight() variances that account for the point weights of a single event being
fully correlated, and the result is already in the form combine's
InterferenceModel wants.

NAMED vs UNNAMED WEIGHTS
Named weights (LHEWeight_<point>, produced by setting genWeightsTable
namedWeightIDs/namedWeightLabels) carry the point identity in the branch name,
so the parameterisation can be auto-detected.  This is strongly preferred.
The standard unnamed route (the LHEReweightingWeight vector) carries only an
ordinal index, so the point definitions must be supplied externally -- see
``UnnamedParameterisation``.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

# ``<op>_p2`` -> +2, ``<op>_m1`` -> -1.  The sign prefix lets a production add
# negative points, which are what break the degeneracy between a large negative
# interference and a large positive quadratic term.
_SCAN_RE = re.compile(r"^(?P<op>.+?)_(?P<sign>[pm])(?P<val>\d+(?:p\d+)?)$")

SM_ALIASES = ("SM", "sm", "SM_point")


def _num(sign, val):
    return (1.0 if sign == "p" else -1.0) * float(val.replace("p", "."))


def parse_point_name(name, operators):
    """Point name -> {operator: value}, following the reweight-card convention.

        SM                 all coefficients zero
        <op>_p2 / <op>_m1  that operator at +2 / -1, all others zero
        <opA>_<opB>        both operators at 1, all others zero

    Returns None if the name does not match, so callers can fall back to an
    explicit map.

    VERIFY THIS AGAINST THE REWEIGHT CARD ONCE PER PRODUCTION.  MadGraph applies
    the ``set param_card`` commands of a reweight point CUMULATIVELY from the
    generated point, so a point that does not explicitly reset an operator
    INHERITS its generated value.  In the WZ dim-6 sample the generated point is
    cW = 1 and the cross points deliberately leave LHA 2 alone, so ``cW_cHWB``
    really is (cW=1, cHWB=1) and the convention above happens to be exact.  A
    card written differently would silently produce a wrong design matrix and
    plausible-looking but incorrect structure constants -- ``validate()`` is the
    cheap guard against that.
    """
    if name in SM_ALIASES:
        return {o: 0.0 for o in operators}

    m = _SCAN_RE.match(name)
    if m and m.group("op") in operators:
        out = {o: 0.0 for o in operators}
        out[m.group("op")] = _num(m.group("sign"), m.group("val"))
        return out

    # pairwise cross term: longest-first so ceWRe33 is not shadowed by ceW
    for a in sorted(operators, key=len, reverse=True):
        if not name.startswith(a + "_"):
            continue
        b = name[len(a) + 1:]
        if b in operators:
            out = {o: 0.0 for o in operators}
            out[a] = out[b] = 1.0
            return out
    return None


def discover_operators(point_names):
    """Infer the operator list from the set of point names.

    Operators are the stems of the ``<op>_p<N>`` scan points; a production that
    scans every operator it varies (the usual case) is fully described by them.
    """
    ops = []
    for n in point_names:
        m = _SCAN_RE.match(n)
        if m:
            op = m.group("op")
            if op not in ops:
                ops.append(op)
    return ops


class Parameterisation:
    """Design matrix and structure-constant decomposition for a point set."""

    def __init__(self, operators, points, null_tol=1e-6):
        self.operators = list(operators)
        self.points = dict(points)              # name -> {op: value}
        self.point_names = list(self.points)
        self.null_tol = null_tol

        n = len(self.operators)
        self.terms = (
            ["sm"]
            + [f"lin_{o}" for o in self.operators]
            + [f"quad_{o}" for o in self.operators]
            + [f"mix_{a}_{b}" for a, b in itertools.combinations(self.operators, 2)]
        )
        n_terms = (n + 1) * (n + 2) // 2
        if len(self.terms) != n_terms:                      # defensive
            raise RuntimeError(f"term count {len(self.terms)} != {n_terms}")
        if len(self.point_names) < n_terms:
            raise ValueError(
                f"{len(self.point_names)} reweight points cannot determine "
                f"{n_terms} structure constants for {n} operators"
            )

        self.design = np.array([self._row(self.points[p]) for p in self.point_names])
        rank = np.linalg.matrix_rank(self.design)
        if rank < n_terms:
            raise ValueError(
                f"reweight points are degenerate: design matrix rank {rank} < {n_terms}. "
                "Two points probably describe the same coefficient vector."
            )
        # pinv covers the over-determined case (more points than terms), where
        # it is the least-squares solution; for a square system it is the inverse.
        self.solver = np.linalg.pinv(self.design)
        self.condition = float(np.linalg.cond(self.design))
        self._null_terms = None

    def _row(self, coeffs):
        c = np.array([coeffs[o] for o in self.operators], dtype=float)
        cross = [c[i] * c[j] for i, j in itertools.combinations(range(len(c)), 2)]
        return np.concatenate([[1.0], c, c**2, cross])

    # ------------------------------------------------------------------
    def decompose(self, point_weights):
        """(n_events, n_points) -> (n_events, n_terms), in ``terms`` order."""
        return np.asarray(point_weights, dtype=np.float64) @ self.solver.T

    def predict(self, theta, coeffs):
        """Event weight at an arbitrary coefficient point.

        ``coeffs`` may be a dict or a sequence in ``operators`` order.
        """
        if not isinstance(coeffs, dict):
            coeffs = dict(zip(self.operators, coeffs))
        coeffs = {o: float(coeffs.get(o, 0.0)) for o in self.operators}
        return np.asarray(theta) @ self._row(coeffs)

    # ------------------------------------------------------------------
    def null_terms(self, theta):
        """Structure constants consistent with zero, i.e. not worth storing.

        Physics, not just bookkeeping: a chirality-flipping operator (the lepton
        dipoles) has no interference with the chirality-preserving SM amplitude
        when the corresponding fermion is massless, so its linear term and its
        cross terms with non-dipole operators vanish identically.  Detecting
        that here typically removes a third of the templates.
        """
        theta = np.asarray(theta)
        scale = np.abs(theta[:, 0]).mean() or 1.0
        mag = np.abs(theta).mean(axis=0) / scale
        return [t for t, m in zip(self.terms, mag) if m < self.null_tol]

    def active_terms(self, theta=None):
        if theta is not None:
            self._null_terms = self.null_terms(theta)
        return [t for t in self.terms if t not in (self._null_terms or [])]

    def validate(self, theta):
        """Sanity checks that catch a mis-specified design matrix.

        The pure-quadratic terms are |M_6|^2 and must be non-negative event by
        event.  A wrong coefficient assignment usually shows up here first.
        Returns a list of human-readable problems; empty means healthy.
        """
        theta = np.asarray(theta)
        problems = []
        scale = np.abs(theta[:, 0]).mean() or 1.0
        for i, t in enumerate(self.terms):
            if not t.startswith("quad_"):
                continue
            neg = theta[:, i] < -self.null_tol * scale
            if neg.mean() > 1e-4:
                problems.append(
                    f"{t}: {100 * neg.mean():.2f}% of events negative "
                    f"(min {theta[:, i].min():.3g}) -- it is |M6|^2 and cannot be"
                )
        if self.condition > 1e6:
            problems.append(f"design matrix poorly conditioned (cond={self.condition:.3g})")
        return problems

    def __repr__(self):
        return (
            f"Parameterisation(operators={self.operators}, "
            f"points={len(self.point_names)}, terms={len(self.terms)}, "
            f"cond={self.condition:.1f})"
        )


# ----------------------------------------------------------------------
def named_weight_points(fields):
    """EFT point names from a field list, excluding the bookkeeping weights.

    Accepts either layout: flat uproot branch names (``LHEWeight_cW_p1``) or
    the subfields of a coffea NanoEvents ``LHEWeight`` collection (``cW_p1``),
    since NanoAODSchema groups the ``LHEWeight_*`` branches into a record.
    """
    skip = {"originalXWGTUP"}
    flat = [f[len("LHEWeight_"):] for f in fields if f.startswith("LHEWeight_")]
    names = flat if flat else [f for f in fields if f not in ("LHEWeight",)]
    return [n for n in names if n not in skip]


def points_from_events(events):
    """Auto-detect on a coffea NanoEvents record, handling either layout."""
    if "LHEWeight" in events.fields:
        return from_named_weights(events.LHEWeight.fields)
    return from_named_weights(events.fields)


def from_named_weights(fields, operators=None, point_map=None):
    """Auto-detect the parameterisation from named LHEWeight_* branches.

    Returns (Parameterisation, point_names) or (None, []) when the sample
    carries no usable EFT weights -- so a caller can simply skip EFT filling.

    ``point_map`` overrides the name convention for productions that do not
    follow it; supply {point_name: {operator: value}}.
    """
    names = named_weight_points(fields)
    if len(names) < 3:
        return None, []

    if point_map:
        pts = {n: point_map[n] for n in names if n in point_map}
        ops = operators or sorted({o for v in pts.values() for o in v})
        return Parameterisation(ops, pts), list(pts)

    ops = operators or discover_operators(names)
    if not ops:
        return None, []

    pts, unparsed = {}, []
    for n in names:
        c = parse_point_name(n, ops)
        (pts.__setitem__(n, c) if c is not None else unparsed.append(n))
    if unparsed:
        raise ValueError(
            f"EFT point names not understood: {unparsed}. Operators inferred as "
            f"{ops}. Pass an explicit point_map={{name: {{op: value}}}}."
        )
    return Parameterisation(ops, pts), list(pts)


def stack_named_weights(events, point_names):
    """Pull the per-point weight columns off an events record, in order.

    Handles both the coffea nested layout (``events.LHEWeight.cW_p1``) and flat
    ``LHEWeight_cW_p1`` keys.
    """
    import awkward as ak

    src = events.LHEWeight if "LHEWeight" in getattr(events, "fields", []) else None
    if src is not None:
        cols = [ak.to_numpy(src[p]).astype(np.float64) for p in point_names]
    else:
        cols = [ak.to_numpy(events[f"LHEWeight_{p}"]).astype(np.float64) for p in point_names]
    return np.stack(cols, axis=1)


# ----------------------------------------------------------------------
class UnnamedParameterisation(Parameterisation):
    """STUB -- EFT weights delivered as the unnamed LHEReweightingWeight vector.

    This is the default CMSSW route when genWeightsTable namedWeightIDs is not
    configured: every reweight point lands in a single variable-length vector
    with nothing but an ordinal index, so nothing about the physics can be
    recovered from the file alone.  The point definitions MUST come from the
    gridpack's reweight card, in the order MadGraph emitted them.

    Prefer named weights.  Setting

        process.genWeightsTable.namedWeightIDs   = [...]
        process.genWeightsTable.namedWeightLabels = [...]

    at NanoAOD time makes the sample self-describing and removes the ordering
    assumption entirely.  It also sidesteps a real failure mode: CMSSW's
    <weightgroup> regexes require DOUBLE-quoted attributes, while MadGraph's
    reweight module writes name='mg_reweighting' with SINGLE quotes, so
    LHEReweightingWeight silently comes out empty for such samples.

    TO IMPLEMENT: confirm the index->coefficient ordering against the reweight
    card, verify nLHEReweightingWeight equals len(ordered_points) on a real
    file, then decompose exactly as for named weights.
    """

    def __init__(self, operators, ordered_points, null_tol=1e-6):
        if not ordered_points:
            raise ValueError(
                "UnnamedParameterisation requires the reweight points in MadGraph "
                "order; they cannot be recovered from the NanoAOD file."
            )
        self.order = list(ordered_points)
        super().__init__(operators, dict(enumerate(ordered_points)), null_tol=null_tol)

    @staticmethod
    def stack(events, n_points):
        import awkward as ak

        w = events["LHEReweightingWeight"]
        n = ak.to_numpy(ak.num(w))
        if not (n == n_points).all():
            raise ValueError(
                f"nLHEReweightingWeight is not uniformly {n_points} "
                f"(saw {sorted(set(n.tolist()))[:5]}). If it is 0, the LHE header "
                "was written with single-quoted <weightgroup> attributes and CMSSW "
                "dropped the weights -- regenerate with namedWeightIDs instead."
            )
        return ak.to_numpy(ak.unflatten(ak.flatten(w), n_points, axis=0)).astype(np.float64)

    def decompose_events(self, events):
        raise NotImplementedError(
            "Unnamed-weight decomposition is a stub. Supply the ordered point "
            "definitions and validate the ordering against the reweight card first; "
            "see the class docstring."
        )
