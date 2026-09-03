from __future__ import annotations
import numpy as np
import awkward as ak
from coffea.nanoevents.methods import candidate, vector

ak.behavior.update(vector.behavior)

MW = 80.379


def cos_theta_helicity(boson_p4, daughter_p4):
    """cos(theta*) of daughter_p4 in the helicity frame of boson_p4.
    Boost daughter into boson rest frame, project onto boson lab direction.
    Result clipped to [-1, 1]. Guards both magnitudes against zero.
    """
    eps = 1e-12
    bx, by, bz = boson_p4.px, boson_p4.py, boson_p4.pz
    bmag2 = bx * bx + by * by + bz * bz
    bmag = np.sqrt(ak.where(bmag2 < eps * eps, eps * eps, bmag2))
    d = daughter_p4.boost(-boson_p4.boostvec)
    dx, dy, dz = d.px, d.py, d.pz
    dmag2 = dx * dx + dy * dy + dz * dz
    dmag = np.sqrt(ak.where(dmag2 < eps * eps, eps * eps, dmag2))
    cos_t = (dx * bx + dy * by + dz * bz) / (dmag * bmag)
    return ak.where(cos_t > 1.0, 1.0, ak.where(cos_t < -1.0, -1.0, cos_t))


def neutrino_pz(tau, met_pt, met_phi, mW=MW):
    """Solve M_W^2 = (p_tau + p_nu)^2 for p_z^nu (quadratic).
    tau must already be the corrected 4-vector. disc<0 (mT>mW, in the
    massless-tau limit) signals no real solution; see
    neutrino_pz_transvmlv for how that case is handled in this module.
    """
    pt_tau = tau.pt
    pz_tau = tau.pt * np.sinh(tau.eta)
    E_tau = np.sqrt(tau.pt ** 2 + pz_tau ** 2 + tau.mass ** 2)

    dphi = tau.phi - met_phi
    pt_dot = pt_tau * met_pt * np.cos(dphi)

    mu = (mW ** 2 - tau.mass ** 2) / 2.0 + pt_dot

    a = E_tau ** 2 - pz_tau ** 2
    b = -2.0 * mu * pz_tau
    c = E_tau ** 2 * met_pt ** 2 - mu ** 2

    disc = b ** 2 - 4.0 * a * c
    sqrt_disc = np.sqrt(ak.where(disc < 0, 0.0, disc))

    pz1 = (-b + sqrt_disc) / (2.0 * a)
    pz2 = (-b - sqrt_disc) / (2.0 * a)
    return pz1, pz2, pz_tau, disc


def pick_neutrino_pz(pz1, pz2, pz_tau):
    """Smaller |pz| solution (prefers central / less-boosted neutrino), used
    when a real solution exists (disc>=0). Matches CMS-SMP-20-014
    (arXiv:2110.11231): 'the one resulting in a lower magnitude of the
    longitudinal momentum of the neutrino is chosen.'"""
    return ak.where(np.abs(pz1) <= np.abs(pz2), pz1, pz2)


def neutrino_pz_transvmlv(tau, met_pt, met_phi):
    """Used when disc<0 (no real solution to the MW-based quadratic):
    substitute the ACTUAL (measured) transverse mass mT for MW in the same
    quadratic, which forces a real solution by construction -- closed form,
    no iteration, no rescaling of tau or MET. Matches arXiv:1907.04722
    (Polarized VBS WZ/ZZ, Appendix A.2, option [transvMlv]) -- reported
    there to work "slightly better" than the take-the-real-part convention
    in the same channel class, and confirmed in this analysis: resolution
    (gen-reco RMS) came out lower for transvmlv than the take-real-part
    alternative in every tested cos(theta*_W) bin (Pol_reg, ~5% relative
    improvement), with no directional bias artifact in the angular
    distributions or pure-state templates.

    Derivation (tau mass kept exact; the paper's own formula,
    pz = pz_lep*pT_nu/pT_lep, drops it since electrons/muons are ~massless):
    substituting mT^2 = 2*pt_tau*met_pt*(1-cos(dphi)) for mW^2 in
    mu = (mW^2-m_tau^2)/2 + pt_tau*met_pt*cos(dphi) simplifies to
    mu' = pt_tau*met_pt - m_tau^2/2, and since disc=0 forces pz = -b/(2a) =
    mu'*pz_tau/a with a = pt_tau^2+m_tau^2 (exact):
        pz_nu = (pt_tau*met_pt - m_tau^2/2) * pz_tau / (pt_tau^2 + m_tau^2)

    IMPORTANT: this does NOT force W_mass to MW for these events. By
    construction, the resulting invariant mass equals mT itself (whatever
    that was for this event), not 80 GeV -- verified both algebraically and
    against real MC. It's a different, differently-motivated choice of pz
    for the ambiguous case, not a mass-forcing technique. See the
    literature survey in this analysis's methodology summary for why
    forcing a sharp mass peak isn't standard practice regardless of method.
    """
    pt_tau = tau.pt
    pz_tau = tau.pt * np.sinh(tau.eta)
    mu_prime = pt_tau * met_pt - tau.mass ** 2 / 2.0
    a = pt_tau ** 2 + tau.mass ** 2   # = E_tau^2 - pz_tau^2, exact
    return mu_prime * pz_tau / ak.where(np.abs(a) > 1e-12, a, 1e-12)


def build_W(tau_corr, met_pt, met_phi, pz_nu):
    """Reconstruct full W 4-vector from corrected tau + neutrino."""
    px_nu = met_pt * np.cos(met_phi)
    py_nu = met_pt * np.sin(met_phi)
    E_nu = np.sqrt(met_pt ** 2 + pz_nu ** 2)
    nu = ak.zip(
        {"x": px_nu, "y": py_nu, "z": pz_nu, "t": E_nu},
        with_name="LorentzVector", behavior=vector.behavior,
    )
    tau_lv = ak.zip(
        {"x": tau_corr.px, "y": tau_corr.py, "z": tau_corr.pz, "t": tau_corr.energy},
        with_name="LorentzVector", behavior=vector.behavior,
    )
    return tau_lv.add(nu)


def reconstruct_W_tau_nu(tau, p4_met, slope=None, intercept=None):
    """End-to-end W->tau_nu reconstruction, single method: [transvMlv].

    tau_corr (the tau-pT correction) and the pre-rescue MET rebalancing are
    computed once; neither is ever modified afterward -- the neutrino-pz
    quadratic is solved once against these fixed inputs, and only the
    CHOICE of pz_nu differs between the real-solution and no-real-solution
    cases (both closed-form, no iteration):
      - disc >= 0 (real solution exists): smaller-|pz| root (matches
        CMS-SMP-20-014 for this case -- both conventions agree here).
      - disc <  0 (no real solution)    : neutrino_pz_transvmlv -- see its
        docstring for exactly what this does and does not achieve.
    """
    if slope is None:
        raise ValueError(
            "reconstruct_W_tau_nu: slope must be provided explicitly "
            "(e.g. from the processor's era-conditioned config) -- no "
            "default or era-based lookup is used."
        )
    if intercept is None:
        raise ValueError(
            "reconstruct_W_tau_nu: intercept must be provided explicitly "
            "(e.g. from the processor's era-conditioned config) -- no "
            "default or era-based lookup is used."
        )
    eps = 1e-12

    # ---- tau-pT correction (inverted form: reco-on-gen fit, applied inverse) ----
    tau_corr_pt = (tau.pt - intercept) / slope
    tau_corr_pt = ak.where(tau_corr_pt < tau.pt, tau.pt, tau_corr_pt)  # active safeguard; see docstring
    scale = tau_corr_pt / ak.where(tau.pt > eps, tau.pt, eps)
    tau_corr = ak.zip(
        {"pt": tau_corr_pt, "eta": tau.eta, "phi": tau.phi, "mass": tau.mass * scale},
        with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior,
    )

    # ---- MET rebalancing (never modified afterward) ----
    delta_px = (tau_corr.px - tau.px)
    delta_py = (tau_corr.py - tau.py)

    met_px_new = p4_met.pt * np.cos(p4_met.phi) - delta_px
    met_py_new = p4_met.pt * np.sin(p4_met.phi) - delta_py
    met_pt_new = np.sqrt(met_px_new ** 2 + met_py_new ** 2)
    met_phi_new = np.arctan2(met_py_new, met_px_new)

    dphi_pre = tau_corr.phi - met_phi_new
    mT_pre = np.sqrt(2.0 * tau_corr.pt * met_pt_new * (1.0 - np.cos(dphi_pre)))
    rescued = ak.values_astype(mT_pre > MW, np.float32)   # "no real solution" flag (diagnostic)

    # ---- single solve, method choice only affects which pz is picked ----
    pz1, pz2, pz_tau, disc = neutrino_pz(tau_corr, met_pt_new, met_phi_new, mW=MW)
    pz_pole = pick_neutrino_pz(pz1, pz2, pz_tau)
    pz_transv = neutrino_pz_transvmlv(tau_corr, met_pt_new, met_phi_new)
    nu_pz = ak.where(disc < 0, pz_transv, pz_pole)

    W = build_W(tau_corr, met_pt_new, met_phi_new, nu_pz)

    dphi = tau_corr.phi - met_phi_new
    mT = np.sqrt(2.0 * tau_corr.pt * met_pt_new * (1.0 - np.cos(dphi)))
    W_mass = np.sqrt(np.abs(W.t ** 2 - W.x ** 2 - W.y ** 2 - W.z ** 2))

    return {
        "tau_corr": tau_corr, "nu_pz": nu_pz, "pz1": pz1, "pz2": pz2,
        "disc": disc, "W": W, "W_px": W.x, "W_py": W.y, "W_pz": W.z,
        "W_E": W.t, "W_mass": W_mass, "mT": mT,
        "mT_pre": mT_pre, "rescued": rescued,
    }


# =============================================================================
# 2. Reweighting: angular distributions, pure-state weights.
# =============================================================================
#
# Formulas below follow CMS-SMP-20-014 / arXiv:2110.11231 (JHEP 07 (2022) 032),
# Eqs. 9-10. W: raw (non-charge-flipped) cos(theta*_W) with fL/fR swapping
# roles between W+ and W-; equivalently -- and how this code does it -- a
# charge-signed angle qcos_theta_W = cos(theta*_W) * q_W fed into a single
# fL/fR-labelled formula (these are algebraically identical). Z: the
# "(negatively) charged lepton" convention with the asymmetry coefficient
#   c = (cL^2 - cR^2)/(cL^2 + cR^2),  cL = -1/2 + sin^2(theta_eff), cR = sin^2(theta_eff)
# which reduces, with _C_Z = 1 - 4 sin^2(theta_eff), to c = 2*_C_Z/(1+_C_Z^2).
#
# NOTE: generator fractions (f0/fL/fR per boson/charge) and the tau-pT-response
# slope/intercept are analysis CONFIGURATION, not physics -- they live in the
# processor, keyed off self._era, and are passed into
# compute_polarization_quantities / compute_reco_angles as gen_frac=/slope=/
# intercept=. This module has no era knowledge.

SIN2_THETAW = 0.23122
_C_Z = 1.0 - 4.0 * SIN2_THETAW                 # ~ 0.0751
ALPHA_Z = 2.0 * _C_Z / (1.0 + _C_Z * _C_Z)     # ~ 0.149 (c = 2*_C_Z/(1+_C_Z^2))
W_CLIP = 10.0                                  # safety clip on pol weights

POL_WEIGHT_KEYS = [
    "wZ_long", "wZ_left", "wZ_right",
    "wW_long", "wW_left", "wW_right",
    "wWp_long", "wWp_left", "wWp_right",
    "wWm_long", "wWm_left", "wWm_right",
]

# Variables that receive a polarization-weighted template for every key above.
POL_TEMPLATE_VARS = [
    "cos_theta_z_gen", "cos_theta_w_gen", "qcos_theta_w_gen",
    "cos_theta_wp_gen", "cos_theta_wm_gen",
    "cos_theta_z_reco", "cos_theta_w_reco", "qcos_theta_w_reco",
    "cos_theta_wp_reco", "cos_theta_wm_reco",
    "W_mass", "mT", "W_E", "nu_pz", "tau_corr_pt", 'dilep_tau_loose_met_hadron_mt',
]


def angular_dist_W(cos_theta, f0, fL, fR):
    """W (pure V-A): (3/8)[ f0*2(1-x^2) + fL*(1-x)^2 + fR*(1+x)^2 ].
    Input: charge-signed cos(theta*_W) [i.e. qcos_theta_w = cos_theta_w * q_W].
    Matches CMS-SMP-20-014 Eq. 9 evaluated at the raw angle with fL/fR swapped
    for W- (algebraically identical to evaluating this formula at qcos_theta_w
    for both charges -- verified)."""
    x = cos_theta
    return (3.0 / 8.0) * (f0 * 2.0 * (1.0 - x ** 2) + fL * (1.0 - x) ** 2 + fR * (1.0 + x) ** 2)


def angular_dist_Z(cos_theta, f0, fL, fR, alpha=ALPHA_Z):
    """Z (mixed V/A): (3/8)[ f0*2(1-x^2) + fL*(1-2ax+x^2) + fR*(1+2ax+x^2) ].
    Input: UNSIGNED cos(theta*_Z), i.e. w.r.t. the negatively charged lepton.
    Matches CMS-SMP-20-014 Eq. 10 with alpha = c = 2*_C_Z/(1+_C_Z^2)."""
    x = cos_theta
    return (3.0 / 8.0) * (f0 * 2.0 * (1.0 - x ** 2)
                          + fL * (1.0 - 2.0 * alpha * x + x ** 2)
                          + fR * (1.0 + 2.0 * alpha * x + x ** 2))


def weight_W(cos_theta, f0_t, fL_t, fR_t, f0_g, fL_g, fR_g, clip=W_CLIP):
    """Per-event W polarization reweight: P(target)/P(gen)."""
    eps = 1e-12
    num = angular_dist_W(cos_theta, f0_t, fL_t, fR_t)
    den = angular_dist_W(cos_theta, f0_g, fL_g, fR_g)
    den = ak.where(np.abs(den) < eps, eps, den)
    w = num / den
    return ak.where(w > clip, clip, ak.where(w < 0, 0.0, w))


def weight_Z(cos_theta, f0_t, fL_t, fR_t, f0_g, fL_g, fR_g, alpha=ALPHA_Z, clip=W_CLIP):
    """Per-event Z polarization reweight: P(target)/P(gen)."""
    eps = 1e-12
    num = angular_dist_Z(cos_theta, f0_t, fL_t, fR_t, alpha)
    den = angular_dist_Z(cos_theta, f0_g, fL_g, fR_g, alpha)
    den = ak.where(np.abs(den) < eps, eps, den)
    w = num / den
    return ak.where(w > clip, clip, ak.where(w < 0, 0.0, w))


def _p4(parts):
    return ak.zip(
        {"pt": parts.pt, "eta": parts.eta, "phi": parts.phi, "mass": parts.mass,
         "charge": ak.zeros_like(parts.pt)},
        with_name="PtEtaPhiMCandidate", behavior=candidate.behavior,
    )


def compute_reco_angles(
    p4_met, lead_tau, dilep_p4, lead_lep, subl_lep, z_cand_mask,
    slope=None, intercept=None,
):
    reco_W = reconstruct_W_tau_nu(lead_tau, p4_met, slope=slope, intercept=intercept)

    lminus_reco = ak.where(lead_lep.charge < 0, lead_lep, subl_lep)
    w_charge_reco = lead_tau.charge
    z_mask_evt = ak.firsts(z_cand_mask)

    cos_w_reco = cos_theta_helicity(reco_W["W"], lead_tau)
    cos_z_reco = cos_theta_helicity(dilep_p4, lminus_reco)
    qcos_w_reco = cos_w_reco * w_charge_reco

    cos_w_reco = ak.where(z_mask_evt, cos_w_reco, np.nan)
    cos_z_reco = ak.where(z_mask_evt, cos_z_reco, np.nan)
    qcos_w_reco = ak.where(z_mask_evt, qcos_w_reco, np.nan)
    cos_wp_reco = ak.where(z_mask_evt & (w_charge_reco > 0), qcos_w_reco, np.nan)
    cos_wm_reco = ak.where(z_mask_evt & (w_charge_reco < 0), qcos_w_reco, np.nan)

    angles = {
        "cos_theta_w_reco": cos_w_reco,
        "cos_theta_z_reco": cos_z_reco,
        "qcos_theta_w_reco": qcos_w_reco,
        "cos_theta_wp_reco": cos_wp_reco,
        "cos_theta_wm_reco": cos_wm_reco,
    }
    extras = {
        "tau_corr_pt": reco_W["tau_corr"].pt,
        "nu_pz": reco_W["nu_pz"], "pz1": reco_W["pz1"], "pz2": reco_W["pz2"],
        "disc": reco_W["disc"], "W_E": reco_W["W_E"],
        "W_mass": reco_W["W_mass"], "mT": reco_W["mT"],
        "mT_pre": reco_W["mT_pre"], "rescued": reco_W["rescued"],
    }
    extras["disc_sign"] = ak.where(reco_W["disc"] > 0, 0.5, -0.5)
    return {"angles": angles, "extras": extras, "reco_W": reco_W}


def compute_polarization_quantities(
    event, gp, p4_met, lead_tau, dilep_p4, lead_lep, subl_lep,
    z_cand_mask, zmass, gen_frac=None, slope=None, intercept=None,
):
    if gen_frac is None:
        raise ValueError(
            "compute_polarization_quantities: gen_frac must be provided "
            "explicitly (e.g. from the processor's era-conditioned config) "
            "-- no default or era-based lookup is used."
        )
    fZ = gen_frac["Z"]
    fW = gen_frac["W"]
    fWp = gen_frac.get("Wp", fW)
    fWm = gen_frac.get("Wm", fW)

    # ---- gen-level W (tau + nu) and Z (OSSF leptons) ----
    is_fromW = gp.hasFlags(["fromHardProcess"]) & (np.abs(gp.distinctParent.pdgId) == 24)
    is_fromZ = gp.hasFlags(["fromHardProcess"]) & (np.abs(gp.distinctParent.pdgId) == 23)
    taus_W = gp[is_fromW & (np.abs(gp.pdgId) == 15)]
    nus_W = gp[is_fromW & (np.abs(gp.pdgId) == 16)]
    leps_Z = gp[is_fromZ & ((np.abs(gp.pdgId) == 11) | (np.abs(gp.pdgId) == 13))]

    taus_W = taus_W[ak.argsort(taus_W.pt, ascending=False)]
    nus_W = nus_W[ak.argsort(nus_W.pt, ascending=False)]
    lead_tau_W = ak.firsts(taus_W)
    # No gen-level pT cut here (tried and reverted -- a pT>20 requirement on
    # gen_valid restricted which events enter qcos_theta_w_gen based on tau
    # pT, distorting the shape used to fit f0/fL/fR).
    tau_pt_gen = ak.fill_none(lead_tau_W.pt, -99)
    lead_nu_W = ak.firsts(nus_W)

    tau_p4 = _p4(lead_tau_W)
    nu_p4 = _p4(lead_nu_W)
    w_p4 = tau_p4 + nu_p4
    tau_charge = ak.where(lead_tau_W.pdgId > 0, -1, +1)
    w_charge_gen = tau_charge

    gen_valid = (ak.num(taus_W) > 0) & (ak.num(nus_W) > 0)

    leps_Z = leps_Z[ak.argsort(leps_Z.pt, ascending=False)]
    leps_Z = ak.with_field(leps_Z, ak.where(leps_Z.pdgId > 0, -1, 1), "charge")
    pairs = ak.combinations(leps_Z, 2, axis=1, fields=["l1", "l2"])
    ossf = (np.abs(pairs.l1.pdgId) == np.abs(pairs.l2.pdgId)) & \
           ((pairs.l1.charge + pairs.l2.charge) == 0)
    pairs = pairs[ossf]
    m_pair = (pairs.l1 + pairs.l2).mass
    idx = ak.argmin(np.abs(m_pair - zmass), axis=1, keepdims=True)
    best = ak.pad_none(pairs[idx], 1, axis=1)
    l1 = ak.firsts(best.l1)
    l2 = ak.firsts(best.l2)
    z_p4_gen = l1 + l2
    lminus = ak.where(l1.charge < 0, l1, l2)
    z_valid_gen = ak.fill_none(z_p4_gen.mass > 0, False) & gen_valid

    cos_w_gen = cos_theta_helicity(w_p4, tau_p4)
    cos_z_gen = cos_theta_helicity(z_p4_gen, lminus)
    qcos_w_gen = cos_w_gen * w_charge_gen
    cos_w_gen = ak.where(z_valid_gen, cos_w_gen, np.nan)
    cos_z_gen = ak.where(z_valid_gen, cos_z_gen, np.nan)
    qcos_w_gen = ak.where(z_valid_gen, qcos_w_gen, np.nan)
    cos_wp_gen = ak.where(z_valid_gen & (w_charge_gen > 0), qcos_w_gen, np.nan)
    cos_wm_gen = ak.where(z_valid_gen & (w_charge_gen < 0), qcos_w_gen, np.nan)

    # ---- reco-level W reconstruction + angles (shared with bkg/data path) ----
    reco = compute_reco_angles(
        p4_met, lead_tau, dilep_p4, lead_lep, subl_lep, z_cand_mask,
        slope=slope, intercept=intercept,
    )
    reco_W = reco["reco_W"]
    cos_w_reco = reco["angles"]["cos_theta_w_reco"]
    cos_z_reco = reco["angles"]["cos_theta_z_reco"]
    qcos_w_reco = reco["angles"]["qcos_theta_w_reco"]
    cos_wp_reco = reco["angles"]["cos_theta_wp_reco"]
    cos_wm_reco = reco["angles"]["cos_theta_wm_reco"]

    # ---- pure-state weights ----
    cos_z_safe = ak.fill_none(ak.nan_to_none(cos_z_gen), 0.0)
    qcos_w_safe = ak.fill_none(ak.nan_to_none(qcos_w_gen), 0.0)

    wZ_long = ak.where(z_valid_gen, weight_Z(cos_z_safe, 1, 0, 0, fZ["f0"], fZ["fL"], fZ["fR"]), 1.0)
    wZ_left = ak.where(z_valid_gen, weight_Z(cos_z_safe, 0, 1, 0, fZ["f0"], fZ["fL"], fZ["fR"]), 1.0)
    wZ_right = ak.where(z_valid_gen, weight_Z(cos_z_safe, 0, 0, 1, fZ["f0"], fZ["fL"], fZ["fR"]), 1.0)
    wW_long = ak.where(z_valid_gen, weight_W(qcos_w_safe, 1, 0, 0, fW["f0"], fW["fL"], fW["fR"]), 1.0)
    wW_left = ak.where(z_valid_gen, weight_W(qcos_w_safe, 0, 1, 0, fW["f0"], fW["fL"], fW["fR"]), 1.0)
    wW_right = ak.where(z_valid_gen, weight_W(qcos_w_safe, 0, 0, 1, fW["f0"], fW["fL"], fW["fR"]), 1.0)

    wp_mask = z_valid_gen & (w_charge_gen > 0)
    wWp_long = ak.where(wp_mask, weight_W(qcos_w_safe, 1, 0, 0, fWp["f0"], fWp["fL"], fWp["fR"]), 1.0)
    wWp_left = ak.where(wp_mask, weight_W(qcos_w_safe, 0, 1, 0, fWp["f0"], fWp["fL"], fWp["fR"]), 1.0)
    wWp_right = ak.where(wp_mask, weight_W(qcos_w_safe, 0, 0, 1, fWp["f0"], fWp["fL"], fWp["fR"]), 1.0)
    wm_mask = z_valid_gen & (w_charge_gen < 0)
    wWm_long = ak.where(wm_mask, weight_W(qcos_w_safe, 1, 0, 0, fWm["f0"], fWm["fL"], fWm["fR"]), 1.0)
    wWm_left = ak.where(wm_mask, weight_W(qcos_w_safe, 0, 1, 0, fWm["f0"], fWm["fL"], fWm["fR"]), 1.0)
    wWm_right = ak.where(wm_mask, weight_W(qcos_w_safe, 0, 0, 1, fWm["f0"], fWm["fL"], fWm["fR"]), 1.0)

    angles = {
        "cos_theta_z_gen": cos_z_gen,
        "cos_theta_w_gen": cos_w_gen,
        "qcos_theta_w_gen": qcos_w_gen,
        "cos_theta_wp_gen": cos_wp_gen,
        "cos_theta_wm_gen": cos_wm_gen,
        "cos_theta_w_reco": cos_w_reco,
        "cos_theta_z_reco": cos_z_reco,
        "qcos_theta_w_reco": qcos_w_reco,
        "cos_theta_wp_reco": cos_wp_reco,
        "cos_theta_wm_reco": cos_wm_reco,
    }
    weights = {
        "wZ_long": wZ_long, "wZ_left": wZ_left, "wZ_right": wZ_right,
        "wW_long": wW_long, "wW_left": wW_left, "wW_right": wW_right,
        "wWp_long": wWp_long, "wWp_left": wWp_left, "wWp_right": wWp_right,
        "wWm_long": wWm_long, "wWm_left": wWm_left, "wWm_right": wWm_right,
    }
    extras = dict(reco["extras"])
    extras["tau_pt_gen"] = tau_pt_gen

    _dphi = tau_p4.phi - lead_tau.phi
    _dphi = (_dphi + np.pi) % (2 * np.pi) - np.pi
    _deta = tau_p4.eta - lead_tau.eta
    _dr = np.sqrt(_deta ** 2 + _dphi ** 2)
    extras["tau_gen_dR"] = ak.fill_none(_dr, 99.0)
    extras["tau_matched"] = ak.values_astype(ak.fill_none(_dr < 0.4, False), np.float32)
    extras["W_pt_gen"] = ak.fill_none(w_p4.pt, -99)
    extras["gen_valid"] = ak.values_astype(ak.fill_none(z_valid_gen, False), np.float32)
    return {"angles": angles, "weights": weights, "reco_W": reco_W, "extras": extras}