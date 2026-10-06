from __future__ import annotations
import numpy as np
import awkward as ak
from coffea.nanoevents.methods import candidate, vector

ak.behavior.update(vector.behavior)

MW = 80.379
M_TAU = 1.77686   

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

# def cos_theta_helicity(boson_p4, daughter_p4, wz_system_p4):
#     """
#     cos(theta*) of daughter_p4 in the helicity frame of boson_p4,
#     where the z-axis is defined by the boson's direction in the WZ CoM frame.
#     """
#     eps = 1e-12
    
#     # 1. Boost both the boson and the daughter into the WZ CoM frame
#     wz_boost = -wz_system_p4.boostvec
#     boson_in_wz = boson_p4.boost(wz_boost)
#     daughter_in_wz = daughter_p4.boost(wz_boost)
    
#     # 2. In the WZ CoM frame, the z-axis is the direction of the boson
#     bx, by, bz = boson_in_wz.px, boson_in_wz.py, boson_in_wz.pz
#     bmag2 = bx**2 + by**2 + bz**2
#     bmag = np.sqrt(ak.where(bmag2 < eps*eps, eps*eps, bmag2))
    
#     # 3. Now get the daughter in the Boson's rest frame 
#     # (boosting directly from the WZ CoM frame copy to stay consistent)
#     d_rest = daughter_in_wz.boost(-boson_in_wz.boostvec)
#     dx, dy, dz = d_rest.px, d_rest.py, d_rest.pz
#     dmag2 = dx**2 + dy**2 + dz**2
#     dmag = np.sqrt(ak.where(dmag2 < eps*eps, eps*eps, dmag2))
    
#     # 4. Calculate the alignment angle
#     cos_t = (dx * bx + dy * by + dz * bz) / (dmag * bmag)
#     return ak.where(cos_t > 1.0, 1.0, ak.where(cos_t < -1.0, -1.0, cos_t))


def transverse_mass(vis_pt, vis_mass, dphi, met_pt):
    """mT of (visible, massless invisible) system:
    mT^2 = m_vis^2 + 2 (E_T^vis * MET - pT^vis * MET * cos(dphi)),  E_T^vis = sqrt(pT^2 + m^2).
    Exactly the minimum of M(vis + nu) over pz^nu, so M_W >= mT for a real solution."""
    et = np.sqrt(vis_pt ** 2 + vis_mass ** 2)
    mt2 = vis_mass ** 2 + 2.0 * met_pt * (et - vis_pt * np.cos(dphi))
    return np.sqrt(ak.where(mt2 > 0, mt2, 0.0))


def neutrino_pz(tau, met_pt, met_phi, mW=MW):
    pt_tau = tau.pt
    pz_tau = tau.pt * np.sinh(tau.eta)
    et2 = tau.pt ** 2 + tau.mass ** 2                     
    E_tau = np.sqrt(et2 + pz_tau ** 2)
    pt_dot = pt_tau * met_pt * np.cos(tau.phi - met_phi)
    mu = (mW ** 2 - tau.mass ** 2) / 2.0 + pt_dot
    a = ak.where(et2 > 1e-12, et2, 1e-12)
    b = -2.0 * mu * pz_tau
    c = E_tau ** 2 * met_pt ** 2 - mu ** 2
    disc = b ** 2 - 4.0 * a * c
    sqrt_disc = np.sqrt(ak.where(disc < 0, 0.0, disc))
    pz1 = (-b + sqrt_disc) / (2.0 * a)
    pz2 = (-b - sqrt_disc) / (2.0 * a)
    return pz1, pz2, pz_tau, disc


def pick_neutrino_pz(pz1, pz2, pz_tau):
    """Smaller |pz| solution. Matches CMS-SMP-20-014 (arXiv:2110.11231)."""
    return ak.where(np.abs(pz1) <= np.abs(pz2), pz1, pz2)


def neutrino_pz_transvmlv(tau, met_pt, met_phi):
    pz_tau = tau.pt * np.sinh(tau.eta)
    et = np.sqrt(tau.pt ** 2 + tau.mass ** 2)
    return met_pt * pz_tau / ak.where(et > 1e-12, et, 1e-12)


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


def check_pol_ratio(slope, intercept, pt_lo=20.0, pt_hi=800.0):
    """Config-time sanity check on the tau-pT correction (LINEAR form).

    The correction is tau_corr_pt = slope*pt_reco + intercept, i.e. a linear
    fit of <pt_gen | pt_reco>. Validates it over the pT range where it is
    actually used and returns the implied ratio tau_corr_pt/pt_reco at both
    ends. This is deliberately NOT a per-event check: a bad fit is a
    configuration error, so it should fail once at setup rather than raise
    inside the event loop.
    """
    c_lo = slope * pt_lo + intercept
    c_hi = slope * pt_hi + intercept
    if min(c_lo, c_hi) <= 0:
        raise ValueError(
            f"corrected tau pT goes non-positive over [{pt_lo}, {pt_hi}] GeV: "
            f"tau_corr_pt = {c_lo:.1f} .. {c_hi:.1f} GeV (slope={slope}, intercept={intercept}). "
            "This would flip or collapse the tau pT."
        )
    r_lo, r_hi = c_lo / pt_lo, c_hi / pt_hi
    if min(r_lo, r_hi) < 0.9 or max(r_lo, r_hi) > 4.0:
        print(
            f"WARNING: implied tau-pT ratio tau_corr_pt/pt_reco = {r_lo:.3f} .. {r_hi:.3f} "
            f"over [{pt_lo}, {pt_hi}] GeV is outside the expected ~[0.9, 4] band. "
            "Check that slope/intercept come from the LINEAR fit <pt_gen | pt_reco> "
            "(slope ~1, intercept in GeV) and not from the old ratio parameterisation."
        )
    return r_lo, r_hi


def reconstruct_W_tau_nu(tau, p4_met, slope=None, intercept=None):
    
    if slope is None or intercept is None:
        raise ValueError(
            "reconstruct_W_tau_nu: slope and intercept must both be provided "
            "explicitly (e.g. from the processor's era-conditioned config)."
        )
    eps = 1e-12

    # mT of the uncorrected visible tau + reco MET (monitoring only)
    mT_pre = transverse_mass(tau.pt, tau.mass, tau.phi - p4_met.phi, p4_met.pt)

    # ---- 1. tau-pT correction (linear form) ----
    tau_corr_pt = slope * tau.pt + intercept
    tau_corr_pt = ak.where(tau_corr_pt > eps, tau_corr_pt, eps)   # never flip the sign or collapse pT
    tau_corr = ak.zip(
        {"pt": tau_corr_pt, "eta": tau.eta, "phi": tau.phi, "mass": ak.full_like(tau_corr_pt, M_TAU)},
        with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior,
    )

    # ---- 2. MET rebalancing (never modified afterward) ----
    met_px_new = p4_met.pt * np.cos(p4_met.phi) - (tau_corr.px - tau.px)
    met_py_new = p4_met.pt * np.sin(p4_met.phi) - (tau_corr.py - tau.py)
    met_pt_new = np.sqrt(met_px_new ** 2 + met_py_new ** 2)
    met_phi_new = np.arctan2(met_py_new, met_px_new)

    # ---- 3. nu pz: pole solution if kinematically allowed, transvMlv otherwise ----
    mT = transverse_mass(tau_corr.pt, tau_corr.mass, tau_corr.phi - met_phi_new, met_pt_new)
    rescued = mT > MW                     # no real solution <=> mT > M_W (exact, massive tau)
    pz1, pz2, pz_tau, disc = neutrino_pz(tau_corr, met_pt_new, met_phi_new, mW=MW)
    pz_pole = pick_neutrino_pz(pz1, pz2, pz_tau)
    pz_transv = neutrino_pz_transvmlv(tau_corr, met_pt_new, met_phi_new)
    nu_pz = ak.where(rescued, pz_transv, pz_pole)

    W = build_W(tau_corr, met_pt_new, met_phi_new, nu_pz)
    W_mass = np.sqrt(np.abs(W.t ** 2 - W.x ** 2 - W.y ** 2 - W.z ** 2))

    return {
        "tau_corr": tau_corr, "nu_pz": nu_pz, "pz1": pz1, "pz2": pz2,
        "disc": disc, "W": W, "W_px": W.x, "W_py": W.y, "W_pz": W.z,
        "W_E": W.t, "W_mass": W_mass, "mT": mT,
        "mT_pre": mT_pre, "rescued": ak.values_astype(rescued, np.float32),
    }


# =============================================================================
# 2. Reweighting: angular distributions, pure-state weights.
# =============================================================================

SIN2_THETAW = 0.23122
_C_Z = 1.0 - 4.0 * SIN2_THETAW                 # ~ 0.0751
ALPHA_Z = 2.0 * _C_Z / (1.0 + _C_Z * _C_Z)     # ~ 0.149
W_CLIP = 10.0

# Naming convention (first symbol = the boson being polarized):
#   wWZ_*  = Z polarization, inclusive WZ        -> cos_theta_z_*
#   wWpZ_* = Z polarization, W+Z only            -> cos_theta_wpz_*
#   wWmZ_* = Z polarization, W-Z only            -> cos_theta_wmz_*
#   wZW_*  = W polarization, inclusive WZ        -> qcos_theta_w_*
#   wZWp_* = W+ polarization                     -> cos_theta_wp_*
#   wZWm_* = W- polarization                     -> cos_theta_wm_*
POL_WEIGHT_KEYS = [
    "wWZ_long", "wWZ_left", "wWZ_right",
    "wWpZ_long", "wWpZ_left", "wWpZ_right",
    "wWmZ_long", "wWmZ_left", "wWmZ_right",
    "wZW_long", "wZW_left", "wZW_right",
    "wZWp_long", "wZWp_left", "wZWp_right",
    "wZWm_long", "wZWm_left", "wZWm_right",
]

POL_TEMPLATE_VARS = [
    "cos_theta_z_gen", "cos_theta_wpz_gen", "cos_theta_wmz_gen",
    "qcos_theta_w_gen", "cos_theta_wp_gen", "cos_theta_wm_gen",
    "cos_theta_z_reco", "cos_theta_wpz_reco", "cos_theta_wmz_reco",
    "qcos_theta_w_reco", "cos_theta_wp_reco", "cos_theta_wm_reco",
    "W_mass", "mT", "W_E", "nu_pz", "tau_corr_pt",
    "dilep_tau_loose_met_hadron_mt",
]


def pol_wpref_for(var):
    if "cos_theta_wpz" in var:
        return "wWpZ"
    if "cos_theta_wmz" in var:
        return "wWmZ"
    if "cos_theta_wp" in var:
        return "wZWp"
    if "cos_theta_wm" in var:
        return "wZWm"
    if "cos_theta_z" in var:
        return "wWZ"
    return "wZW"


def angular_dist_W(cos_theta, f0, fL, fR):
    """W (pure V-A): (3/8)[ f0*2(1-x^2) + fL*(1-x)^2 + fR*(1+x)^2 ].
    Input: charge-signed cos(theta*_W). CMS-SMP-20-014 Eq. 9."""
    x = cos_theta
    return (3.0 / 8.0) * (f0 * 2.0 * (1.0 - x ** 2) + fL * (1.0 - x) ** 2 + fR * (1.0 + x) ** 2)


def angular_dist_Z(cos_theta, f0, fL, fR, alpha=ALPHA_Z):
    """Z (mixed V/A): (3/8)[ f0*2(1-x^2) + fL*(1-2ax+x^2) + fR*(1+2ax+x^2) ].
    Input: UNSIGNED cos(theta*_Z) w.r.t. the negatively charged lepton.
    CMS-SMP-20-014 Eq. 10 with alpha = 2*_C_Z/(1+_C_Z^2)."""
    x = cos_theta
    return (3.0 / 8.0) * (f0 * 2.0 * (1.0 - x ** 2)
                          + fL * (1.0 - 2.0 * alpha * x + x ** 2)
                          + fR * (1.0 + 2.0 * alpha * x + x ** 2))


def weight_W(cos_theta, f0_t, fL_t, fR_t, f0_g, fL_g, fR_g, clip=W_CLIP):
    eps = 1e-12
    num = angular_dist_W(cos_theta, f0_t, fL_t, fR_t)
    den = angular_dist_W(cos_theta, f0_g, fL_g, fR_g)
    den = ak.where(np.abs(den) < eps, eps, den)
    w = num / den
    return ak.where(w > clip, clip, ak.where(w < 0, 0.0, w))


def weight_Z(cos_theta, f0_t, fL_t, fR_t, f0_g, fL_g, fR_g, alpha=ALPHA_Z, clip=W_CLIP):
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


def compute_reco_angles(p4_met, lead_tau, dilep_p4, lead_lep, subl_lep,
                        z_cand_mask, slope=None, intercept=None):
    """Reco-only angles/kinematics -- no gen info, so this also serves the
    data/background path. The W+Z / W-Z split here uses the RECO tau charge
    (a genuine reco-level observable), NOT the gen W charge, so these
    variables remain defined for data."""
    reco_W = reconstruct_W_tau_nu(lead_tau, p4_met, slope=slope, intercept=intercept)
    reco_wz_system = reco_W["W"].add(dilep_p4)

    lminus_reco = ak.where(lead_lep.charge < 0, lead_lep, subl_lep)
    w_charge_reco = lead_tau.charge
    z_mask_evt = ak.firsts(z_cand_mask)

    # cos_w_reco = cos_theta_helicity(reco_W["W"], reco_W["tau_corr"], reco_wz_system)
    # cos_z_reco = cos_theta_helicity(dilep_p4, lminus_reco, reco_wz_system)
    cos_w_reco = cos_theta_helicity(reco_W["W"], reco_W["tau_corr"])
    cos_z_reco = cos_theta_helicity(dilep_p4, lminus_reco)
    qcos_w_reco = cos_w_reco * w_charge_reco

    cos_w_reco = ak.where(z_mask_evt, cos_w_reco, np.nan)
    cos_z_reco = ak.where(z_mask_evt, cos_z_reco, np.nan)
    qcos_w_reco = ak.where(z_mask_evt, qcos_w_reco, np.nan)

    wp_evt = z_mask_evt & (w_charge_reco > 0)
    wm_evt = z_mask_evt & (w_charge_reco < 0)
    # Z angle split by W charge (FIXED: second line assigned cos_wpz twice)
    cos_wpz_reco = ak.where(wp_evt, cos_z_reco, np.nan)
    cos_wmz_reco = ak.where(wm_evt, cos_z_reco, np.nan)
    # W angle split by W charge
    cos_wp_reco = ak.where(wp_evt, qcos_w_reco, np.nan)
    cos_wm_reco = ak.where(wm_evt, qcos_w_reco, np.nan)

    angles = {
        "cos_theta_w_reco": cos_w_reco,          
        "cos_theta_z_reco": cos_z_reco,
        "cos_theta_wpz_reco": cos_wpz_reco,
        "cos_theta_wmz_reco": cos_wmz_reco,
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
            "explicitly (e.g. from the processor's era-conditioned config)."
        )
    fWZ = gen_frac["WZ"]; fWpZ = gen_frac["WpZ"]; fWmZ = gen_frac["WmZ"]
    fZW = gen_frac["ZW"]; fZWp = gen_frac["ZWp"]; fZWm = gen_frac["ZWm"]

    is_from_hard_process_first = gp.hasFlags(["fromHardProcess", "isFirstCopy"])
    from_W = is_from_hard_process_first & (np.abs(gp.distinctParent.pdgId) == 24)
    from_Z = is_from_hard_process_first & (np.abs(gp.distinctParent.pdgId) == 23)
 
    taus_W = gp[from_W & (np.abs(gp.pdgId) == 15)]
    nus_W  = gp[from_W & (np.abs(gp.pdgId) == 16)]

    leps_Z = gp[from_Z & ((np.abs(gp.pdgId) == 11) | (np.abs(gp.pdgId) == 13))]

    is_W_gen = (gp.hasFlags(["fromHardProcess", "isLastCopy"]) & (np.abs(gp.pdgId) == 24))
    is_Z_gen = (gp.hasFlags(["fromHardProcess", "isLastCopy"]) & (gp.pdgId == 23))

    W_bosons_gen = gp[is_W_gen]
    Z_bosons_gen = gp[is_Z_gen]

    W_bosons_gen = W_bosons_gen[ak.argsort(W_bosons_gen.pt, ascending=False)]
    Z_bosons_gen = Z_bosons_gen[ak.argsort(Z_bosons_gen.pt, ascending=False)]

    lead_W_gen = ak.firsts(W_bosons_gen)
    lead_Z_gen = ak.firsts(Z_bosons_gen)

    w_boson_found = ~ak.is_none(lead_W_gen.pt)
    z_boson_found = ~ak.is_none(lead_Z_gen.pt)

    w_p4 = _p4(lead_W_gen)
    z_p4_gen = _p4(lead_Z_gen)
 
    taus_W = taus_W[ak.argsort(taus_W.pt, ascending=False)]
    nus_W  = nus_W[ak.argsort(nus_W.pt, ascending=False)]
    leps_Z = leps_Z[ak.argsort(leps_Z.pt, ascending=False)]
 
    lead_tau_W = ak.firsts(taus_W)
    lead_nu_W  = ak.firsts(nus_W)
    tau_pt_gen = ak.fill_none(lead_tau_W.pt, -99)
 
    tau_p4 = _p4(lead_tau_W)
    nu_p4  = _p4(lead_nu_W)
    tau_found = ~ak.is_none(lead_tau_W.pt)
    nu_found = ~ak.is_none(lead_nu_W.pt)
    w_charge_gen = ak.fill_none(ak.where(lead_tau_W.pdgId > 0, -1, +1), 0)

    leps_Z = ak.with_field(leps_Z, ak.where(leps_Z.pdgId > 0, -1, +1), "charge")
    pairs = ak.combinations(leps_Z, 2, axis=1, fields=["l1", "l2"])
    ossf = (np.abs(pairs.l1.pdgId) == np.abs(pairs.l2.pdgId)) & ((pairs.l1.charge + pairs.l2.charge) == 0)
    pairs = pairs[ossf]
    m_pair = (pairs.l1 + pairs.l2).mass
    idx = ak.argmin(np.abs(m_pair - zmass), axis=1, keepdims=True)
    best = ak.pad_none(pairs[idx], 1, axis=1)
    l1 = ak.firsts(best.l1)
    l2 = ak.firsts(best.l2)
    z_sum = l1 + l2
    z_pair_found = ~ak.is_none(l1.pt)
    z_mass_cut = ak.fill_none(np.abs(z_sum.mass - zmass) < 15, False)
    lminus = ak.where(l1.charge < 0, l1, l2)

    gen_valid = (w_boson_found & z_boson_found & tau_found & nu_found & z_pair_found & z_mass_cut)
    wz_system_p4 = w_p4 + z_p4_gen

    # ---- gen-level angles ----
    # cos_w_gen = cos_theta_helicity(w_p4, tau_p4, wz_system_p4)   
    # cos_z_gen = cos_theta_helicity(z_p4_gen, _p4(lminus), wz_system_p4)
    cos_w_gen = cos_theta_helicity(w_p4, tau_p4)   
    cos_z_gen = cos_theta_helicity(z_p4_gen, _p4(lminus))
    qcos_w_gen = cos_w_gen * w_charge_gen

    gen_wp = gen_valid & (w_charge_gen > 0)
    gen_wm = gen_valid & (w_charge_gen < 0)

    cos_z_gen = ak.where(gen_valid, cos_z_gen, np.nan)
    qcos_w_gen = ak.where(gen_valid, qcos_w_gen, np.nan)
    cos_wpz_gen = ak.where(gen_wp, cos_z_gen, np.nan)     # Z angle in W+Z
    cos_wmz_gen = ak.where(gen_wm, cos_z_gen, np.nan)     # Z angle in W-Z
    cos_wp_gen = ak.where(gen_wp, qcos_w_gen, np.nan)     # W+ angle
    cos_wm_gen = ak.where(gen_wm, qcos_w_gen, np.nan)     # W- angle

    # ---- reco-level angles (shared with the data/background path) ----
    reco = compute_reco_angles(p4_met, lead_tau, dilep_p4, lead_lep, subl_lep,
                                z_cand_mask, slope=slope, intercept=intercept)
    reco_W = reco["reco_W"]

    # ---- pure-state weights ----
    cos_z_safe = ak.fill_none(ak.nan_to_none(cos_z_gen), 0.0)
    qcos_w_safe = ak.fill_none(ak.nan_to_none(qcos_w_gen), 0.0)

    # Z polarization (weight_Z, on the UNSIGNED Z angle)
    wWZ_long  = ak.where(gen_valid, weight_Z(cos_z_safe, 1, 0, 0, fWZ["f0"], fWZ["fL"], fWZ["fR"]), 1.0)
    wWZ_left  = ak.where(gen_valid, weight_Z(cos_z_safe, 0, 1, 0, fWZ["f0"], fWZ["fL"], fWZ["fR"]), 1.0)
    wWZ_right = ak.where(gen_valid, weight_Z(cos_z_safe, 0, 0, 1, fWZ["f0"], fWZ["fL"], fWZ["fR"]), 1.0)

    wWpZ_long  = ak.where(gen_wp, weight_Z(cos_z_safe, 1, 0, 0, fWpZ["f0"], fWpZ["fL"], fWpZ["fR"]), 1.0)
    wWpZ_left  = ak.where(gen_wp, weight_Z(cos_z_safe, 0, 1, 0, fWpZ["f0"], fWpZ["fL"], fWpZ["fR"]), 1.0)
    wWpZ_right = ak.where(gen_wp, weight_Z(cos_z_safe, 0, 0, 1, fWpZ["f0"], fWpZ["fL"], fWpZ["fR"]), 1.0)

    wWmZ_long  = ak.where(gen_wm, weight_Z(cos_z_safe, 1, 0, 0, fWmZ["f0"], fWmZ["fL"], fWmZ["fR"]), 1.0)
    wWmZ_left  = ak.where(gen_wm, weight_Z(cos_z_safe, 0, 1, 0, fWmZ["f0"], fWmZ["fL"], fWmZ["fR"]), 1.0)
    wWmZ_right = ak.where(gen_wm, weight_Z(cos_z_safe, 0, 0, 1, fWmZ["f0"], fWmZ["fL"], fWmZ["fR"]), 1.0)

    # W polarization (weight_W, on the CHARGE-SIGNED W angle)
    wZW_long  = ak.where(gen_valid, weight_W(qcos_w_safe, 1, 0, 0, fZW["f0"], fZW["fL"], fZW["fR"]), 1.0)
    wZW_left  = ak.where(gen_valid, weight_W(qcos_w_safe, 0, 1, 0, fZW["f0"], fZW["fL"], fZW["fR"]), 1.0)
    wZW_right = ak.where(gen_valid, weight_W(qcos_w_safe, 0, 0, 1, fZW["f0"], fZW["fL"], fZW["fR"]), 1.0)

    wZWp_long  = ak.where(gen_wp, weight_W(qcos_w_safe, 1, 0, 0, fZWp["f0"], fZWp["fL"], fZWp["fR"]), 1.0)
    wZWp_left  = ak.where(gen_wp, weight_W(qcos_w_safe, 0, 1, 0, fZWp["f0"], fZWp["fL"], fZWp["fR"]), 1.0)
    wZWp_right = ak.where(gen_wp, weight_W(qcos_w_safe, 0, 0, 1, fZWp["f0"], fZWp["fL"], fZWp["fR"]), 1.0)

    wZWm_long  = ak.where(gen_wm, weight_W(qcos_w_safe, 1, 0, 0, fZWm["f0"], fZWm["fL"], fZWm["fR"]), 1.0)
    wZWm_left  = ak.where(gen_wm, weight_W(qcos_w_safe, 0, 1, 0, fZWm["f0"], fZWm["fL"], fZWm["fR"]), 1.0)
    wZWm_right = ak.where(gen_wm, weight_W(qcos_w_safe, 0, 0, 1, fZWm["f0"], fZWm["fL"], fZWm["fR"]), 1.0)

    angles = {
        "cos_theta_z_gen": cos_z_gen,
        "cos_theta_wpz_gen": cos_wpz_gen,
        "cos_theta_wmz_gen": cos_wmz_gen,
        "qcos_theta_w_gen": qcos_w_gen,
        "cos_theta_wp_gen": cos_wp_gen,
        "cos_theta_wm_gen": cos_wm_gen,
    }
    angles.update(reco["angles"])   # all six reco angles, incl. cos_theta_w_reco

    weights = {
        "wWZ_long": wWZ_long, "wWZ_left": wWZ_left, "wWZ_right": wWZ_right,
        "wWpZ_long": wWpZ_long, "wWpZ_left": wWpZ_left, "wWpZ_right": wWpZ_right,
        "wWmZ_long": wWmZ_long, "wWmZ_left": wWmZ_left, "wWmZ_right": wWmZ_right,
        "wZW_long": wZW_long, "wZW_left": wZW_left, "wZW_right": wZW_right,
        "wZWp_long": wZWp_long, "wZWp_left": wZWp_left, "wZWp_right": wZWp_right,
        "wZWm_long": wZWm_long, "wZWm_left": wZWm_left, "wZWm_right": wZWm_right,
    }

    extras = dict(reco["extras"])
    extras["tau_pt_gen"] = tau_pt_gen

    eps = 1e-9
    _reco_pt_safe = ak.where(lead_tau.pt > eps, lead_tau.pt, eps)
    extras["tau_pt_ratio"] = ak.fill_none(ak.where(tau_pt_gen > 0, tau_pt_gen / _reco_pt_safe, -99.0), -99.0)

    _dphi = tau_p4.phi - lead_tau.phi
    _dphi = (_dphi + np.pi) % (2 * np.pi) - np.pi
    _deta = tau_p4.eta - lead_tau.eta
    _dr = np.sqrt(_deta ** 2 + _dphi ** 2)

    extras["tau_gen_dR"] = ak.fill_none(_dr, 99.0)
    extras["tau_matched"] = ak.values_astype(ak.fill_none(_dr < 0.4, False), np.float32)
    extras["W_pt_gen"] = ak.fill_none(w_p4.pt, -99)
    extras["w_charge_gen"] = w_charge_gen
    extras["W_gen_found"] = ak.values_astype(ak.fill_none(w_boson_found, False), np.float32)
    extras["Z_gen_found"] = ak.values_astype(ak.fill_none(z_boson_found, False), np.float32)
    extras["tau_gen_found"] = ak.values_astype(ak.fill_none(tau_found, False), np.float32)
    extras["nu_gen_found"] = ak.values_astype(ak.fill_none(nu_found, False), np.float32)
    extras["Z_ll_gen_found"] = ak.values_astype(ak.fill_none(z_pair_found, False), np.float32)
    extras["gen_valid"] = ak.values_astype(ak.fill_none(gen_valid, False), np.float32)

    return {"angles": angles, "weights": weights, "reco_W": reco_W, "extras": extras}