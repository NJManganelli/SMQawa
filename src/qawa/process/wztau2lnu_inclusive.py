import awkward as ak
import scipy.interpolate as interp
from scipy import stats as st
from scipy.special import expit
import hist
import yaml
import os
import re
import onnxruntime as rt
import numpy.lib.recfunctions as rf
import uproot
import pickle
import numpy as np
import pandas as pd
import math
from functools import partial
import copy

from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from coffea.util import coffea_console
from qawa.roccor import rochester_correction
from qawa.muonSS import muon_pt_scare
from qawa.egammaSS import EGM_scale_and_smear_v9, EGM_scale_and_smear
from qawa.leptonsSF import LeptonScaleFactors
from qawa.jetPU import jetPUScaleFactors
from qawa.tauSF import tauIDScaleFactors, tau_energy_scale
from qawa.btag import BTVCorrector
from qawa.jme import JMEUncertainty, update_collection
from qawa.gen_match import find_best_match
from qawa.datadriven import DataDrivenEventReweight
from qawa.common import pileup_weights, ewk_corrector, met_phi_xy_correction, theory_ps_weight, theory_pdf_weight, trigger_rules, transverse_energy, propagate_shift_to_met
from qawa.jsoncorrections import CorrectionlibHandler
from qawa.met_shim import prepare_met_for_factory

from qawa.polarization import (
    POL_WEIGHT_KEYS,
    POL_TEMPLATE_VARS,
    compute_polarization_quantities,
    compute_reco_angles,
)


# ─────────────────────────────────────────────────────────────────────────
# Polarization config: era -> generator fractions / tau-pT-response slope
# and intercept. Analysis configuration, not physics -- lives here, not in
# qawa.polarization, which is pure physics/formulas and takes gen_frac=,
# slope=, intercept= as required arguments with no era knowledge of its own.
# ─────────────────────────────────────────────────────────────────────────
_POL_ERA_KEY = {"2016": "Run2", "2016APV": "Run2", "2017": "Run2", "2018": "Run2",
                "2024": "2024"}

 


POL_GEN_FRACTIONS_BY_ERA = {
    "Run2": {
        "Z":  dict(f0=0.24483, fL=0.22700, fR=0.52817),
        "W":  dict(f0=0.25230, fL=0.47953, fR=0.26817),
        "Wp": dict(f0=0.24966, fL=0.52017, fR=0.23017),
        "Wm": dict(f0=0.25669, fL=0.41433, fR=0.32899),
    },
    "2024": {   # TODO: placeholder copied from Run2 -- refit on 2024 MC
        "Z":  dict(f0=0.25622, fL=0.32269, fR=0.42109),
        "W":  dict(f0=0.09579, fL=0.59350, fR=0.31070),
        "Wp": dict(f0=0.07308, fL=0.68563, fR=0.24129),
        "Wm": dict(f0=0.14875, fL=0.45029, fR=0.40096),
    },
}
POL_GEN_FRACTIONS_2024_VERIFIED = False

POL_SLOPE_BY_ERA = {
    "Run2": 0.5080,
    "2024": 0.4675,
}
POL_Int_BY_ERA = {
    "Run2": 11.27,
    "2024": 14.58,
}


def _pol_era_key(era):
    key = _POL_ERA_KEY.get(str(era))
    if key is None:
        raise KeyError(f"unknown era {era!r} for polarization config; "
                       f"known: {sorted(_POL_ERA_KEY)}")
    return key


def get_pol_gen_fractions(era, normalize=True):
    key = _pol_era_key(era)
    if key == "2024" and not POL_GEN_FRACTIONS_2024_VERIFIED:
        coffea_console.print(
            "[yellow]WARNING: PLACEHOLDER 2024 polarization generator fractions "
            "in use (copied from Run 2). Refit POL_GEN_FRACTIONS_BY_ERA['2024'] "
            "before trusting these results.[/yellow]"
        )
    out = {b: dict(v) for b, v in POL_GEN_FRACTIONS_BY_ERA[key].items()}
    if normalize:
        for b, f in out.items():
            t = f["f0"] + f["fL"] + f["fR"]
            for k in ("f0", "fL", "fR"):
                f[k] /= t
    return out


def get_pol_slope(era):
    return POL_SLOPE_BY_ERA[_pol_era_key(era)]


def get_pol_intercept(era):
    return POL_Int_BY_ERA[_pol_era_key(era)]


def _wh(*axes):
    """A weighted histogram with the standard (channel, systematic) categorical
    axes prepended and Weight() storage. `axes` are the value axes."""
    return hist.Hist(
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        *axes,
        hist.storage.Weight(),
    )

def _pol_wpref_for(var):
    if "cos_theta_wp" in var:
        return "wWp"
    if "cos_theta_wm" in var:
        return "wWm"
    if "cos_theta_z" in var:
        return "wZ"
    return "wW"

def build_leptons_fv(muons, electrons):
    # select tight/loose muons
    tight_muons_mask = (
        (muons.pt             >  20.0) & 
        (np.abs(muons.eta)    <  2.4) & 
        (~muons.hasTauAnc) #avoid mu coming from tau
    )
    tight_muons = muons[tight_muons_mask]
    # select tight/loose electron
    SCeta = np.abs(electrons.eta)
    tight_electrons_mask = (
        (electrons.pt > 20.0) & 
        ((np.abs(electrons.eta) < 1.4442) | ((np.abs(electrons.eta) > 1.5660) & (np.abs(electrons.eta) < 2.5))) &
        (~electrons.hasTauAnc) #avoid e coming from tau
    )
    tight_electrons = electrons[tight_electrons_mask]
    # contruct a lepton object
    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')

    tight_sorted_index = ak.argsort(tight_leptons.pt,ascending=False)

    tight_leptons = tight_leptons[tight_sorted_index]

    return tight_leptons

def build_leptons_fv_loose(muons, electrons):
    """
    Returns gen-level loose leptons that are NOT already in the tight FV
    collection (pt > 10, |eta| < 2.5, no tau ancestor).  Used to veto events
    with a third visible lepton at generator level, matching the reco
    nloose_lep == 0 requirement.
    """
    loose_muons = muons[
        (muons.pt > 10.0) & (muons.pt < 20.0) &
        (np.abs(muons.eta) <  2.4)  &
        (~muons.hasTauAnc)
    ]
    loose_electrons = electrons[
        (electrons.pt > 10.0) & (electrons.pt < 20.0) &
        (np.abs(electrons.eta) <  2.5)  &
        (~electrons.hasTauAnc)
    ]
    loose_leptons = ak.with_name(
        ak.concatenate([loose_muons, loose_electrons], axis=1),
        'PtEtaPhiMCandidate'
    )
    return loose_leptons

def build_htaus_fv(tau, nanoAODversion="v9"):
    if nanoAODversion in [f"v{V}" for V in range(12)]:
        tau_max_eta = 2.3
    elif nanoAODversion in ["v15"]:
        tau_max_eta = 2.5
    tight_taus_mask = (
        (tau.pt             >  20.0) & 
        (np.abs(tau.eta) < tau_max_eta ) 
    )

    tight_taus = tau[tight_taus_mask]
    tight_sorted_index = ak.argsort(tight_taus.pt,ascending=False)
    tight_taus = tight_taus[tight_sorted_index]
    tight_taus = ak.with_name(tight_taus, "PtEtaPhiMCandidate", )

    return tight_taus    
def build_leptons(muons, electrons, nanoAODversion="v9", analysisID="inc-WZ-baseline", leptonIDs=None):
    # We need IDs for SFs later, since we make a tag per-analysis, we return the concrete IDs used for later SF application
    # select tight/loose muons
    if analysisID == "inc-WZ-baseline":
        # use simple cutbased IDs
        if nanoAODversion == "v9":
            assert leptonIDs["muonID"] == "Tight"
            assert leptonIDs["muonISO"] == "TightRelIso"
            assert leptonIDs["electronID"] == "wp90iso"
        elif nanoAODversion == "v15":
            assert leptonIDs["muonID"] == "Tight"
            assert leptonIDs["muonISO"] == "TightPFIso"
            assert leptonIDs["electronID"] == "wp90iso"
        else:
            raise NotImplementedError
        muons_tightID = muons.tightId # this corresponds to "Tight" in the Run 2 cutbased naming convention, do not allow them to become desynchronized
        muons_looseID = muons.looseId
        muons_tightISO = (muons.pfRelIso04_all <= 0.15) # this corresponds to 'TightRelIso' in the Run 2 cutbased naming convention, do not allow them to become desynchronized
        muons_looseISO = (muons.pfRelIso04_all <= 0.25)
        muons_nonISO = (muons.pfRelIso04_all >  0.25)
        electrons_tightISOID = electrons.mvaFall17V2Iso_WP90 if "mvaFall17V2Iso_WP90" in electrons.fields else electrons.mvaIso_WP90
        electrons_looseISOID = electrons.mvaFall17V2Iso_WPL  if "mvaFall17V2Iso_WPL"  in electrons.fields else electrons.mvaIso_WP90
        if "mvaFall17V2noIso_WPL" in electrons.fields:
            electrons_nonISO = electrons.mvaFall17V2noIso_WPL
        else:
            electrons_nonISO = electrons.mvaNoIso_WP90
    elif analysisID == "inc-WZ-enhanced":
        raise NotImplementedError


    tight_muons_mask = (
        (muons.pt             >  20. ) &
        (np.abs(muons.eta)    <  2.4 ) &
        (np.abs(muons.dxy)    <  0.045) &
        (np.abs(muons.dz )    <  0.2 ) &
        muons_tightISO &
        muons_tightID
    )
    tight_muons = muons[tight_muons_mask]
    loose_muons = muons[
        ~tight_muons_mask &
        (muons.pt            >  10. ) &
        (np.abs(muons.eta)   <  2.4 ) &
        muons_looseISO &
        muons_looseID
    ]
    non_iso_muons_mask = (
        ~tight_muons_mask & # cross-cleaning against loose-iso muons not needed, isolation inversion guarantee below
        (muons.pt            >  10. ) &
        (np.abs(muons.eta)   <  2.4 ) &
        muons_nonISO &
        muons_looseID
    )
    non_iso_muons = muons[non_iso_muons_mask]
    # select tight/loose electron
    # https://cms-talk.web.cern.ch/t/clarification-on-ee-eb-gap-veto/133256
    electron_superclusterEta = electrons.superclusterEta if "suberclusterEta" in electrons.fields else (electrons.eta + electrons.deltaEtaSC) #v15 has the superclusterEta branch directly for MC?
    tight_electrons_mask = (
        (electrons.pt           > 20.) &
        ((np.abs(electron_superclusterEta) < 1.4442) | ((np.abs(electron_superclusterEta) > 1.5660) & (np.abs(electron_superclusterEta)  < 2.5)))  &
        electrons_tightISOID
    )
    tight_electrons = electrons[tight_electrons_mask]

    loose_electrons_mask = (
        ~tight_electrons_mask & 
        (electrons.pt           > 10.) &
        (np.abs(electrons.eta)  < 2.5)  &
        electrons_looseISOID
    )
    loose_electrons = electrons[loose_electrons_mask]

    non_iso_electrons_mask = (
        ~tight_electrons_mask &
        ~loose_electrons_mask &
        (electrons.pt           > 10.) &
        (np.abs(electrons.eta)  < 2.5)  &
        electrons_nonISO
    )

    non_iso_electrons = electrons[non_iso_electrons_mask]

    # contruct a lepton object
    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')
    loose_leptons = ak.with_name(ak.concatenate([loose_muons, loose_electrons], axis=1), 'PtEtaPhiMCandidate')
    non_iso_leptons = ak.with_name(ak.concatenate([non_iso_muons, non_iso_electrons], axis=1), 'PtEtaPhiMCandidate')

    return tight_leptons, loose_leptons, non_iso_leptons

def build_htaus(tau, lepton, nanoAODversion="v9", tauIDvsj_wp="VTight", tauIDvse_wp="VTight", tauIDvsmu_wp="Tight"):
    tau_e_branch = None
    tau_e_subid = None
    tau_mu_subid = None
    tau_mu_branch = None
    tau_j_branch = None
    if nanoAODversion in [f"v{V}" for V in range(12)]:
        tau_max_eta = 2.3
        # v9::byDeepTau2017v2p1VSe ID working points (deepTau2017v2p1): bitmask 1 = VVVLoose, 2 = VVLoose, 4 = VLoose, 8 = Loose, 16 = Medium, 32 = Tight, 64 = VTight, 128 = VVTight
        tau_e_id_cuts = {"VVVLoose": 1, "VVLoose": 2, "VLoose": 4, "Loose": 8, "Medium": 16, "Tight": 32, "VTight": 64, "VVTight": 128}
        tau_e_branch = tau.idDeepTau2017v2p1VSe
        # v9::byDeepTau2017v2p1VSmu ID working points (deepTau2017v2p1): bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight
        tau_mu_id_cuts = {"VLoose": 1, "Loose": 2, "Medium": 4, "Tight": 8}
        tau_mu_branch = tau.idDeepTau2017v2p1VSmu
        # v9::byDeepTau2017v2p1VSjet ID working points (deepTau2017v2p1): bitmask 1 = VVVLoose, 2 = VVLoose, 4 = VLoose, 8 = Loose, 16 = Medium, 32 = Tight, 64 = VTight, 128 = VVTight
        tau_j_id_cuts = {"VVVLoose": 1, "VVLoose": 2, "VLoose": 4, "Loose": 8, "Medium": 16, "Tight": 32, "VTight": 64, "VVTight": 128}
        tau_j_branch = tau.idDeepTau2017v2p1VSjet
    elif nanoAODversion in ["v15"]:
        tau_max_eta = 2.5
        # v15::byDeepTau2018v2p5VSe ID working points (deepTau2018v2p5): 1 = VVVLoose, 2 = VVLoose, 3 = VLoose, 4 = Loose, 5 = Medium, 6 = Tight, 7 = VTight, 8 = VVTight
        tau_e_id_cuts = {"VVVLoose": 1, "VVLoose": 2, "VLoose": 3, "Loose": 4, "Medium": 5, "Tight": 6, "VTight": 7, "VVTight": 8}
        tau_e_branch = tau.idDeepTau2018v2p5VSe
        # v15::byDeepTau2018v2p5VSmu ID working points (deepTau2018v2p5): 1 = VLoose, 2 = Loose, 3 = Medium, 4 = Tight
        tau_mu_id_cuts = {"VLoose": 1, "Loose": 2, "Medium": 3, "Tight": 4}
        tau_mu_branch = tau.idDeepTau2018v2p5VSmu
        # v15::byDeepTau2018v2p5VSjet ID working points (deepTau2018v2p5): 1 = VVVLoose, 2 = VVLoose, 3 = VLoose, 4 = Loose, 5 = Medium, 6 = Tight, 7 = VTight, 8 = VVTight
        tau_j_id_cuts = {"VVVLoose": 1, "VVLoose": 2, "VLoose": 3, "Loose": 4, "Medium": 5, "Tight": 6, "VTight": 7, "VVTight": 8}
        tau_j_branch = tau.idDeepTau2018v2p5VSjet
    else:
        raise NotImplementedError

    if tauIDvse_wp not in tau_e_id_cuts:
        raise ValueError(f"Available levels for tauIDvse_wp: {list(tauIDvse_wp.keys())}")
    else:
        tau_e_subid = (tau_e_branch >= tau_e_id_cuts[tauIDvse_wp])
    if tauIDvsmu_wp not in tau_mu_id_cuts:
        raise ValueError(f"Available levels for tauIDvmu_wp: {list(tauIDvsmu_wp.keys())}")
    else:
        tau_mu_subid = (tau_mu_branch >= tau_mu_id_cuts[tauIDvsmu_wp])
    if tauIDvsj_wp not in tau_j_id_cuts:
        raise ValueError(f"Available levels for tauIDvj_wp: {list(tauIDvsj_wp.keys())}")
    else:
        tau_j_subid = (tau_j_branch >= tau_j_id_cuts[tauIDvsj_wp])

    base_selection = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta) < tau_max_eta ) &
        (np.abs(tau.dz)< 0.2 ) &
        (tau.decayMode != 5   ) &
        (tau.decayMode != 6   ) &
        tau_e_subid & tau_mu_subid & tau_j_subid
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )

    return tau[base_selection & ~overlap_leptons]

def build_jets(jets, tight_leptons, taus_loose, btag_wp, btag_corrector, nanoAODversion="v9"):

    overlap_leptons = ak.any(jets.metric_table(tight_leptons) <= 0.4, axis=2)
    overlap_taus = ak.any(jets.metric_table(taus_loose) <= 0.4, axis=2)


    jet_mask = (
            ~overlap_leptons & 
            ~overlap_taus &
            (jets.pt>30.0) & 
            (np.abs(jets.eta) < 4.7) & 
            (jets.jetId >= 6) # tight JetID 7(2016) and 6(2017/8)
        )
    if nanoAODversion in [f"v{V}" for V in range(12)] and "puId" in jets.fields:
        jet_mask = jet_mask & ((jets.puId >= 6) | (jets.puId == 3) | (jets.pt >= 50)) # medium puID https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL 3,7 for 16and 16APV; 6,7 for 17,18

    assert hasattr(btag_corrector, "tagged_jets")
    jet_btag = (btag_corrector.tagged_jets(jets, wp=["L", "M", "T"])[btag_wp]) & (np.abs(jets.eta)<2.4) #FIXME: update eta restriction if appropriate for RunIII

    return jets[jet_mask], jets[jet_mask & jet_btag]

def apply_hem_uncertainty(jets, met):
    
    phi_mask = (
        (jets.phi > -1.57) &
        (jets.phi < -0.87)
    )
    eta_mask_20 = (jets.eta > -2.5) & (jets.eta < -1.3)
    eta_mask_35 = (jets.eta > -3.0) & (jets.eta < -2.5)
    
    mask_20 = phi_mask & eta_mask_20
    mask_35 = phi_mask & eta_mask_35

    scale = ak.ones_like(jets.pt)
    scale = ak.where(mask_20, 0.80, scale)
    scale = ak.where(mask_35, 0.65, scale)

    scaled_jets = ak.with_field(jets, jets.pt * scale, 'pt')
    scaled_jets = ak.with_field(scaled_jets, jets.mass * scale, 'mass')

    delta_px = ak.sum((jets.pt - scaled_jets.pt) * np.cos(jets.phi), axis=1, mask_identity=False) #delta is negative vector sum of corrected jet_pt minus old jet_pt
    delta_py = ak.sum((jets.pt - scaled_jets.pt) * np.sin(jets.phi), axis=1, mask_identity=False)

    met_px = met.pt * np.cos(met.phi)
    met_py = met.pt * np.sin(met.phi)
    shifted_px = met_px + delta_px
    shifted_py = met_py + delta_py
    shifted_pt = np.sqrt(shifted_px**2 + shifted_py**2)
    shifted_phi = np.arctan2(shifted_py, shifted_px)

    scaled_met = ak.with_field(met, shifted_pt, 'pt')
    scaled_met = ak.with_field(scaled_met, shifted_phi, 'phi')

    return scaled_jets, scaled_met


class wzinclusive_processor(processor.ProcessorABC):
    # TODOS:  modify purw and JMEUncertainty appropriately, also the tauID, update coffea/correctionlib versions and utilize the new clib JECs... also consider the JER, deterministic smearing
    # NEED
    #    JER Smearing broken in txt format, need to update to clib, unclear if coffea version fully ready with L1 and new MET stuff
    #    Verify electroweak corrections are working as intended
    #    Update inc-ZZ to utilize the same interfaces
    # NICE TO HAVE / DEFER
    #    Switch to correctionlib JECS + Type1 MET when coffea implementation for MET and JME implementation of "L1" only correction can be loaded/built
    def __init__(self, era: str ='2018', ewk_process_name=None, run_period: str = '', split_by_charge:bool = False, split_FV:bool = False, pol:bool = False, version="v9",):
        self._era = era
        self._split_by_charge = split_by_charge
        self._split_FV = split_FV
        self._pol = pol
        self._ver = version
        
        self._isAPV  = False
        self._isEE   = False
        self._isBPix = False
        if 'apv' in self._era.lower():
            self._isAPV = True
            self._era = re.findall(r'\d+', self._era)[0]
        elif 'ee' in self._era.lower():
            self._isEE = True
            self._era = re.findall(r'\d+', self._era)[0]
        elif 'bpix' in self._era.lower():
            self._isBPix = True
            self._era = re.findall(r'\d+', self._era)[0]

        if self._era in ["2024", "2025"]:
            assert self._ver not in [f"v{V}" for V in range(15)] # minimum version we will consider is 15
        if self._era.lower() in ["2016", "2016apv", "2017", "2018"]:
            self.com = 13.0
        elif self._era.lower() in ["2022", "2022ee", "2023", "2023bpix", "2024", "2025", "2026"]:
            self.com = 13.6
        else:
            raise NotImplementedError("Unhandled era for center of mass (com) energy")
        self._process_shift_warnings = True

        
        
        jec_tag = ''
        jer_tag = ''
        jetveto_tag = None
        if len(run_period)==0:
            if self._era == '2016':
                if self._isAPV:
                    jec_tag = 'Summer19UL16APV_V7_MC'
                    jer_tag = 'Summer20UL16APV_JRV3_MC'
                else:
                    jec_tag = 'Summer19UL16_V7_MC'
                    jer_tag = 'Summer20UL16_JRV3_MC'
            elif self._era == '2017':
                jec_tag = 'Summer19UL17_V5_MC'
                jer_tag = 'Summer19UL17_JRV2_MC'
            elif self._era == '2018':
                jec_tag = 'Summer19UL18_V5_MC'
                jer_tag = 'Summer19UL18_JRV2_MC'
            elif self._era == '2024':
                jec_tag = 'Summer24Prompt24_V3_MC'
                jer_tag = 'Summer24Prompt24_JRV1_MC' #FIXME, temporary until new JER tag available in jsons... these are 50% copies of Summer19UL18 anyway
                jetveto_tag = 'Summer24Prompt24_RunBCDEFGHI_V1'
            elif self._era in ["2022", "2023", "2025"]:
                raise NotImplementedError(f"{self._era} is not yet implemented")
            else:
                raise ValueError(f"Unexpected era={self._era}")
        else:
            if self._era == '2016':
                if self._isAPV:
                    if run_period in ['B', 'C', 'D']:
                        jec_tag = 'Summer19UL16APV_RunBCD_V7_DATA'
                    else:
                        jec_tag = 'Summer19UL16APV_RunEF_V7_DATA'
                else:
                    jec_tag = 'Summer19UL16_RunFGH_V7_DATA'
            elif self._era == '2017':
                jec_tag = f'Summer19UL17_Run{run_period}_V5_DATA'
            elif self._era == '2018':
                jec_tag = f'Summer19UL18_Run{run_period}_V5_DATA'
            elif self._era == '2024':
                # FIXME: WRONG TAG, but we don't have the code to split by different nibs, so we'll take nib1 always until correctionlib version is compatible
                jec_tag = f'Summer24Prompt24_Run{run_period}nib1_V3_DATA'
                jetveto_tag = 'Summer24Prompt24_RunBCDEFGHI_V1'
            elif self._era in ["2022", "2023", "2025"]:
                raise NotImplementedError(f"{self._era} is not yet implemented")
            else:
                raise ValueError(f"Unexpected era={self._era}")
        
        self.btag_wp = 'L'
        self.btag_tagger = "deepJet" if self._ver in [f"v{V}" for V in range(12)] else "UParTAK4"
        self.jetPU_wp = 'M'
        self.electronID = "wp90iso"
        self.muonID = "Tight"
        self.muonISO = "TightRelIso" if self._ver in [f"v{V}" for V in range(12)] else "TightPFIso"
        self.tauIDvsjet_wp = 'VTight' #This wp refers to the vtight tau we are selecting but it will be overwritten by VTigh and Loose wp for the Vtight and Loose tau SF in the tau_SF.py
        self.tauIDvse_wp = 'Tight'
        # Run 2 and 2024 (Run 3?)doesn't have VTight in SFs so fall back to Tight in the lookup alone, for if vse WP is nominally set to VTight above
        self.tauIDvse_wp_for_sfs = 'Tight'
        self.tauIDvsmu_wp = 'Tight'
        self.zmass = 91.1873 # GeV
        self.clibhandler = CorrectionlibHandler(era = self._era, subera=None, isAPV=self._isAPV, isEE=self._isEE, isBPix=self._isBPix, analysis="inc-WZ", nanoAODversion=self._ver, cvmfs_head="/cvmfs/")
        self.clibhandler.printStatus(console=coffea_console)
        self._btag = BTVCorrector(era=self._era, wp=self.btag_wp, tagger=self.btag_tagger, isAPV=self._isAPV, isEE=self._isEE, isBPix=self._isBPix, clibhandler=self.clibhandler)
        # FIXME: Need JER updates according to https://cms-talk.web.cern.ch/t/new-jer-smearing-inputs-available-for-2024-and-2025/145723/1
        # FIXME: JMEUncertainty class not ready for Correctionlib yet... need upstream updates
        self._jmeu = JMEUncertainty(jec_tag, jer_tag, era=self._era, is_mc=(len(run_period)==0), clibhandler=None, version=self._ver) 
        try:
            self._jetid = self.clibhandler.getCorrectionSet("jetid")
        except KeyError as ke:
            self._jetid = None
        self._purw = pileup_weights(era=self._era, clibhandler=self.clibhandler, clibkey=None) #auto key mode until we find a case that doesn't work
        self._leSF = LeptonScaleFactors(era=self._era, isAPV=self._isAPV, isEE=self._isEE, isBPix=self._isBPix, clibhandler=self.clibhandler,
                                        electronID=self.electronID, muonID=self.muonID, muonISO=self.muonISO)
        if (int(self._era) < 2022) and self._ver in [f"v{V}" for V in range(12)]:
            self._jpSF = jetPUScaleFactors(era=self._era, wp=self.jetPU_wp, isAPV=self._isAPV, isEE=self._isEE, isBPix=self._isBPix)
        else:
            #FIXME: might still need with AK4PUPPI
            self._jpSF = None
        self._tauID= tauIDScaleFactors(era=self._era, vsjet_wp=self.tauIDvsjet_wp, vse_wp=self.tauIDvse_wp_for_sfs, vsmu_wp=self.tauIDvsmu_wp,
                                       isAPV=self._isAPV, isEE=self._isEE, isBPix=self._isBPix, clibhandler=self.clibhandler)
        self._dd   = DataDrivenEventReweight(era=self._era, isAPV=self._isAPV, isEE=self._isEE, isBPix=self._isBPix, override_stat_check=(self._era == "2024"), clibhandler=self.clibhandler, estimator="LNTTau_VTTau_DDDY_Estimate") #FIXME: remove override

        ### polarization part ####
        # era-aware generator fractions: Run-2 eras -> measured UL set,
        # 2024 -> GEN_FRACTIONS_BY_ERA["2024"] (placeholder until fitted).
        self.gen_fractions = get_pol_gen_fractions(self._era)
        self.pol_slope = get_pol_slope(self._era)
        self.pol_intercept = get_pol_intercept(self._era)

        # ── Polarization weight keys + templated variables (from polarization.py) ──

        

        self.pol_weight_keys   = list(POL_WEIGHT_KEYS)
        self.pol_template_vars = [
            v for v in POL_TEMPLATE_VARS
            if v not in ("cos_theta_w_reco", "cos_theta_w_gen")
        ]
        self.pol_reco_vars = [
            "cos_theta_w_reco", "cos_theta_z_reco", "qcos_theta_w_reco",
            "cos_theta_wp_reco", "cos_theta_wm_reco",
            "W_mass", "mT", "W_E", "nu_pz", "tau_corr_pt", 'dilep_tau_loose_met_hadron_mt',
            "disc", "rescued", "disc_sign", "tau_pt_vtight", "mT_pre"
        ]
        
        self.pol_gen_vars = [
            "cos_theta_z_gen", "cos_theta_w_gen", "cos_theta_wm_gen",
            "cos_theta_wp_gen", "qcos_theta_w_gen",
            "tau_pt_gen", "W_pt_gen", "tau_gen_dR",
        ]
       
        ### polarization part ####

        _data_path = 'qawa/data'
        _data_path = os.path.join(os.path.dirname(__file__), '../data')
        self._lumimask_path = {
            '2018': f'{_data_path}/json/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
            '2017': f'{_data_path}/json/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
            '2016': f'{_data_path}/json/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
            '2022': str(self.clibhandler.getPath('Cert_Collisions2022_355100_362760_Golden', fallbackNone=True)),
            '2023': str(self.clibhandler.getPath('Cert_Collisions2023_366442_370790_Golden', fallbackNone=True)),
            '2024': str(self.clibhandler.getPath('Cert_Collisions2024_378981_386951_Golden', fallbackNone=True)),
            '2025': str(self.clibhandler.getPath('Cert_Collisions2025_391658_398903_Golden', fallbackNone=True)),
        }[self._era]
        self._lumimask = LumiMask(self._lumimask_path)
        if jetveto_tag is None:
            self._jetveto = None
        else:
            self._jetveto = self.clibhandler.getCorrectionSet('jetvetomaps')[jetveto_tag]
        with open(f'{_data_path}/{self._era}-trigger-rules.yaml') as ftrig:
            self._triggers = yaml.load(ftrig, Loader=yaml.FullLoader)

        with open(f'{_data_path}/eft-names.dat') as eft_file:
            self._eftnames = [n.strip() for n in eft_file.readlines()]

        if self.clibhandler is not None and "trigger_sf" in self.clibhandler.keys():
            # We have trigger_sf in correctionlib, skip loading from root files
            pass
        else:
            with uproot.open(f'{_data_path}/trigger_sf/histo_triggerEff_sel0_{self._era}.root') as _fn:
                _hvalue = np.dstack([_fn[_hn].values() for _hn in _fn.keys()] + [np.ones((7,7))])
                _herror = np.dstack([np.sqrt(_fn[_hn].variances()) for _hn in _fn.keys()] + [np.zeros((7,7))])
                self.trig_sf_map = np.stack([_hvalue, _herror], axis=-1)

        self.ewk_process_name = ewk_process_name
        self.beam_energy = self.com * 1000 / 2
        if self.ewk_process_name is not None:
            # FIXME: Do we need to adjust inputs or derive new corrections for Run III? For now we'll put in the beam_energy
            self.ewk_corr = ewk_corrector(process=ewk_process_name, beam_energy=self.beam_energy)

        # Target (coarse) tau fake-rate binning of the hard-coded/derived correction set. The
        # tau_pt / tau_pt_loose histograms below are now filled at fine 5-GeV granularity
        # (Regular(120, 0, 600)) so the data-driven derivation can rebin down to exactly this
        # scheme via `hist.rebin(groups=[4, 1, 1, 1, 1, 4, 4, 4, 100])` (see datadriven configs).
        # self.ABCD_tau_bins = [20,25,30,35,40,60,80,100,1000]
        #to change this in tau_pt_vtight use this in tau histogram hist.axis.Variable(self.ABCD_tau_bins, name="tau_pt_vtight", label=r"$p_{T}^{tau_vtight}$ (GeV)")

        self.build_histos = lambda: {
            'dilep_mt_llnunu': _wh(
                hist.axis.Regular(100, 0, 1000, name="dilep_mt_llnunu", label=r"$M_{T}^{\ell\ell}$ (GeV)"),
            ), 
            'dilep_pt': _wh(
                hist.axis.Regular(60, 0, 600, name="dilep_pt", label=r"$p_{T}^{\ell\ell}$ (GeV)"),
            ),
            'pt_llt': _wh(
                hist.axis.Regular(60, 0, 600, name="pt_llt", label=r"$p_{T}^{\ell\ell \tau}$ (GeV)"),
            ),
            'ptv_llt': _wh(
                hist.axis.Regular(60, 0, 600, name="ptv_llt", label=r"$|\vec{p}_{T}^{\,\ell\ell\tau}|$ (GeV)"),
            ), 
            'dilep_deta': _wh(
                hist.axis.Regular(20, 0, 8, name="dilep_deta", label=r"$Delta\eta_{\ell\ell}$"),
            ),
            'HTl': _wh(
                hist.axis.Regular(100, 0, 1000, name="HTl", label=r"$H_{Tl}$ (GeV)"),
            ), 
            'ST': _wh(
                hist.axis.Regular(100, 0, 1000, name="ST", label=r"$S_{T}$ (GeV)"),
            ), 
            'dilep_tau_pt': _wh(
                hist.axis.Regular(60, 0, 600, name="dilep_tau_pt", label=r"$p_{T}^{\tau \ell\ell}$ (GeV)"),
            ), 
            'dilep_loose_tau_pt': _wh(
                hist.axis.Regular(60, 0, 600, name="dilep_loose_tau_pt", label=r"$p_{T}^{\ell\ell, \tau}$ (GeV)"),
            ),
            'dilep_m': _wh(
                hist.axis.Regular(60, 0, 180, name="dilep_m", label=r"$M_{\ell\ell}$ (GeV)"),
            ), 
            'met_pt': _wh(
                hist.axis.Regular(60, 0, 600, name="met_pt", label=r"$p_{T}^{miss}$ (GeV)"),
            ),
            'mT_W': _wh(
                hist.axis.Regular(100, 0, 1000, name="mT_W", label=r"$m_{T}^{W}$ (GeV)"),
            ),
            'dilep_tau_loose_met_hadron_mt': _wh(
                hist.axis.Regular(150, 0, 1500, name="dilep_tau_loose_met_hadron_mt", label=r"$M_{T}^{WZ}$ (GeV)"),
            ),
            'mT_WZ': _wh(
                hist.axis.Regular(150, 0, 1500, name="mT_WZ", label=r"$m_{T}^{WZ}$ (GeV)"),
            ),
            'inv_m_WZ': _wh(
                hist.axis.Regular(150, 0, 1500, name="inv_m_WZ", label=r"$m_{inv}^{WZ}$ (GeV)"),
            ),
            'm_lep': _wh(
                hist.axis.Regular(150, 0, 1500, name="m_lep", label=r"$m^{lep}$ (GeV)"),
            ),
            'tau_pt_vtight': _wh(
                hist.axis.Regular(120, 0, 600, name="tau_pt_vtight", label=r"$p_{T}^{tau_vtight}$ (GeV)"), 
            ),
            'tau_pt_tight': _wh(
                hist.axis.Regular(120, 0, 600, name="tau_pt_tight", label=r"$p_{T}^{tau_tight}$ (GeV)"), 
            ),
            'taus_eta': _wh(
                hist.axis.Regular(50, -5, 5, name="taus_eta", label=r"$\eta(\tau)$"),
            ),
            'taus_phi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="taus_phi", label=r"$\phi(\tau)$"),
            ),
            'tau_pt_loose': _wh(
                hist.axis.Regular(120, 0, 600, name="tau_pt_loose", label=r"$p_{T}^{tau_loose}$ (GeV)"), 
            ),
            'taus_eta_loose': _wh(
                hist.axis.Regular(50, -5, 5, name="taus_eta_loose", label=r"$\eta(\tau loose)$"),
            ),
            'taus_phi_loose': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="taus_phi_loose", label=r"$\phi(\tau loose)$"),
            ),
            'met_phi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="met_phi", label=r"$\phi(p_{T}^{miss})$"),
            ),
            'delta_tau_met_phi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="delta_tau_met_phi", label=r"$\Delta \phi(\tau, p_{T}^{miss})$"),
            ),
            'deep_tau_jet': _wh(
                hist.axis.Regular(100, 0, 1, name="deep_tau_jet", label=r"$deep_{\tau}^{jet}$"),
            ),
            'deep_tau_e': _wh(
                hist.axis.Regular(100, 0, 1, name="deep_tau_e", label=r"$deep_{\tau}^{e}$"),
            ),
            'deep_tau_mu': _wh(
                hist.axis.Regular(100, 0, 1, name="deep_tau_mu", label=r"$deep_{\tau}^{mu}$"),
            ),
            'dphi_met_ll': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="dphi_met_ll", label=r"$\Delta \phi(\ell\ell,p_{T}^{miss})$"),
            ),
            'dphi_jet_met': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="dphi_jet_met", label=r"$\Delta \phi(j,p_{T}^{miss})$"),
            ),
            'dilep_dphi_tau': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_dphi_tau", label=r"$\Delta \phi(\ell\ell,\tau)$"),
            ),
            'dilep_loose_tau_met_dphi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_loose_tau_met_dphi", label=r"$\Delta \phi(\ell\ell\tau, p_{T}^{miss})$"),
            ),
            'dilep_loose_tau_phi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_loose_tau_phi", label=r"$\phi(\ell\ell\tau)$"),
            ),
            'dilep_tau_phi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_tau_phi", label=r"$\phi(\ell\ell\tau)$"),
            ),
            'dilep_dphi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="dilep_dphi", label=r"$\Delta \phi(\ell\ell)$"),
            ),
            'baseweight': _wh(
                hist.axis.Regular(400, -100, 100, name="baseweight", label=r"baseweight"),
            ),
            'delta_R': _wh(
                hist.axis.Regular(50, 0, 5, name="delta_R", label=r"$\Delta R (ll, \tau)$"),
            ),
            'lead_jet_pt': _wh(
                hist.axis.Regular(50, 30, 530, name="lead_jet_pt", label="$p_T^{j_1}$ (GeV)"),
            ),
            'lead_jet_phi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="lead_jet_phi", label=r"$\phi($p_T^{j_1})$"),
            ),
            'lead_jet_eta': _wh(
                hist.axis.Regular(50, -5, 5, name="lead_jet_eta", label=r"$\eta(j_1)$"),
            ), 
            'delta_R_jet_dilep': _wh(
                hist.axis.Regular(50, 0, np.pi, name="delta_R_jet_dilep", label=r"$\Delta R(\ell\ell,j)$"),
            ),
            'delta_R_jet_tau': _wh(
                hist.axis.Regular(50, 0, np.pi, name="delta_R_jet_tau", label=r"$\Delta R(\tau,j)$"),
            ),
            'dilep_dR': _wh(
                hist.axis.Regular(50, 0, 5, name="dilep_dR", label=r"$\Delta R(\ell\ell)$"),
            ),
            'leading_lep_pt': _wh(
                hist.axis.Regular(50, 30, 530, name="leading_lep_pt", label="$p_T^{l_1}$ (GeV)"),
            ), 
            'trailing_lep_pt': _wh(
                hist.axis.Regular(50, 30, 530, name="trailing_lep_pt", label=r"$p_T^{l_2}$ (GeV)"),
            ),
            'leading_lep_eta': _wh(
                hist.axis.Regular(50, -5, 5, name="leading_lep_eta", label=r"$\eta(l_1)$"),
            ), 
            'trailing_lep_eta': _wh(
                hist.axis.Regular(50, -5, 5, name="trailing_lep_eta", label=r"$\eta(l_2)$"),
            ),
            'leading_lep_phi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="leading_lep_phi", label=r"$\phi^(l_1)$"),
            ), 
            'trailing_lep_phi': _wh(
                hist.axis.Regular(50, -np.pi, np.pi, name="trailing_lep_phi", label=r"$\phi^(l_2)$"),
            ),
            'njets': _wh(
                hist.axis.Regular(5, 0, 5, name="njets", label=r"$N_{jet}$ ($p_{T}>30$ GeV)"),
            ), 
            'nbjets': _wh(
                hist.axis.Regular(5, 0, 5, name="nbjets", label=r"$N_{b-jet}$ ($p_{T}>30$ GeV)"),
            ),
            'nhtaus_vtight': _wh(
                hist.axis.Regular(5, 0, 5, name="nhtaus_vtight", label=r"$N_{taus}$ (vtight)"),
            ), 
            'nhtaus_tight': _wh(
                hist.axis.Regular(5, 0, 5, name="nhtaus_tight", label=r"$N_{taus}$ (tight)"),
            ),
            'nhtaus_loose': _wh(
                hist.axis.Regular(5, 0, 5, name="nhtaus_loose", label=r"$N_{taus}$ (loose)"),
            ),
            'delta_R_non_iso_lep_loose_tau': _wh(
                hist.axis.Regular(50, 0, np.pi, name="delta_R_non_iso_lep_loose_tau", label=r"$\Delta R(\tau (loose),non_iso_lep)$"),
            ),
            'delta_R_non_iso_lep_vtight_tau': _wh(
                hist.axis.Regular(50, 0, np.pi, name="delta_R_non_iso_lep_vtight_tau", label=r"$\Delta R(\tau (loose),tightlep)$"),
            ),
            'delta_R_non_iso_lep_tight_tau': _wh(
                hist.axis.Regular(50, 0, np.pi, name="delta_R_non_iso_lep_tight_tau", label=r"$\Delta R(\tau (loose),looselep)$"),
            ),
        }
        self.build_histos_pol = lambda: {
            'cos_theta_z_gen': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_z_gen", label=r"$\cos\theta_{Helicity}$ (Z_gen)"),
            ),
            'cos_theta_w_gen': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_w_gen", label=r"$\cos\theta_{Helicity}$ (w_gen)"),
            ),
            'cos_theta_wm_gen': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_wm_gen", label=r"$\cos\theta_{Helicity}$ (wm_gen)"),
            ),
            'cos_theta_wp_gen': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_wp_gen", label=r"$\cos\theta_{Helicity}$ (wp_gen)"),
            ),
            'qcos_theta_w_gen': _wh(
                hist.axis.Regular(20, -1, 1, name="qcos_theta_w_gen", label=r"$\cos\theta_{Helicity}$ (qW_gen)"),
            ),
            'qcos_theta_w_gen_onshell': _wh(
                hist.axis.Regular(20, -1, 1, name="qcos_theta_w_gen", label=r"$\cos\theta_{Helicity}$ (qW_gen_onshell)"),
            ),
            'W_mass': _wh(
                hist.axis.Regular(100, 0, 500, name="W_mass", label=r"$M_W^{reco}$ (GeV)"),
            ),
            'mT': _wh(
                hist.axis.Regular(100, 0, 500, name="mT", label=r"$m_T^{W}$ (GeV)"),
            ),
            'mT_pre': _wh(
                hist.axis.Regular(100, 0, 500, name="mT_pre", label=r"$m_T^{W}$ (GeV)"),
            ),
            'W_E': _wh(
                hist.axis.Regular(100, 0, 500, name="W_E", label=r"$E_W^{reco}$ (GeV)"),
            ),
            'nu_pz': _wh(
                hist.axis.Regular(100, -500, 500, name="nu_pz", label=r"$p_z^\nu$ (GeV)"),
            ),
            'tau_corr_pt': _wh(
                hist.axis.Regular(60, 0, 600, name="tau_corr_pt", label=r"$p_T^{\tau,corr}$ (GeV)"),
            ),
            'cos_theta_w_reco': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_w_reco", label=r"$\cos\theta^*_W$ (reco)"),
            ),
            'cos_theta_z_reco': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_z_reco", label=r"$\cos\theta^*_Z$ (reco)"),
            ),
            'qcos_theta_w_reco': _wh(
                hist.axis.Regular(20, -1, 1, name="qcos_theta_w_reco", label=r"$q_W\cos\theta^*_W$ (reco)"),
            ),
            'cos_theta_wp_reco': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_wp_reco", label=r"$\cos\theta^*_{W^+}$ (reco)"),
            ),
            'cos_theta_wm_reco': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_wm_reco", label=r"$\cos\theta^*_{W^-}$ (reco)"),
            ),
            'qcos_theta_w_reco_gen': _wh(
                hist.axis.Regular(20, -1, 1,  name="qcos_theta_w_gen", label=r"$\cos\theta_{Helicity}$ (qW_gen)"),
                hist.axis.Regular(20, -1, 1,  name="qcos_theta_w_reco", label=r"$q_W\cos\theta^*_W$ (reco)"),
            ),
            'cos_theta_z_reco_gen': _wh(
                hist.axis.Regular(20, -1, 1, name="cos_theta_z_gen",  label=r"$\cos\theta^*_Z$ (gen)"),
                hist.axis.Regular(20, -1, 1, name="cos_theta_z_reco", label=r"$\cos\theta^*_Z$ (reco)"),
            ),
            'dilep_tau_loose_met_hadron_mt': _wh(
                hist.axis.Regular(60, 0, 1200, name="dilep_tau_loose_met_hadron_mt", label=r"$M_{T}^{WZ}$ (GeV)"),
            ),
            'tau_pt_gen': _wh(
                hist.axis.Regular(100, 0, 800, name="tau_pt_gen", label=r"$p_{T}^{\tau}$ (GeV, gen)"),
            ),
            'tau_pt_gen_vs_reco': _wh(
                hist.axis.Regular(100, 0, 800, name="tau_pt_gen", label=r"$p_{T}^{\tau}$ (GeV, gen)"),
                hist.axis.Regular(100, 0, 800, name="tau_pt_vtight",     label=r"$p_{T}^{\tau}$ (GeV, reco)"),
            ),
            'W_pt_gen': _wh(
                hist.axis.Regular(60, 0, 600, name="W_pt_gen", label=r"$p_{T}^{W}$ (GeV, gen)"),
            ),
            'disc': _wh(
                hist.axis.Regular(80, -2e5, 2e5, name="disc", label=r"$\nu\ p_z$ discriminant"),
            ),
            'disc_sign': _wh(
                hist.axis.Regular(2, -1, 1, name="disc_sign", label="sign of the $\\nu$ $p_z$ discriminant"),
            ),
            'rescued': _wh(
                hist.axis.Regular(2, 0, 2, name="rescued", label="off-shell (no real nu_pz solution)"),
            ),
            'tau_gen_dR': _wh(
                hist.axis.Regular(50, 0, 5, name="tau_gen_dR", label=r"$\Delta R(\tau_{reco},\tau_{gen})$"),
            ),
            'tau_pt_gen_vs_reco_matched': _wh(
                hist.axis.Regular(100, 0, 800, name="tau_pt_gen", label=r"$p_{T}^{\tau}$ (GeV, gen)"),
                hist.axis.Regular(100, 0, 800, name="tau_pt_vtight",     label=r"$p_{T}^{\tau}$ (GeV, reco)"),
            ),
            'qcos_w_vs_tau_pt': _wh(
                hist.axis.Regular(20, -1, 1,  name="qcos_theta_w_reco", label=r"$q\cos\theta^*_W$ (reco)"),
                hist.axis.Regular(20, 0, 400, name="tau_corr_pt",       label=r"$p_T^{\tau,corr}$ (GeV)"),
            ),
        }
        
    def _build_histos_with_pol(self):
        histos = self.build_histos_pol()
        # monitoring histos for any weight key not already defined explicitly
        for k in self.pol_weight_keys:
            if k not in histos:
                histos[k] = hist.Hist(
                    hist.axis.StrCategory([], name="channel", growth=True),
                    hist.axis.StrCategory([], name="systematic", growth=True),
                    hist.axis.Regular(100, 0.0, 10.0, name=k, label=f"{k} weight"),
                    hist.storage.Weight(),
                )
        # polarization-weighted clones of every templated variable
        for v in self.pol_template_vars:
            if v not in histos:           # not a 1D pol histo -> skip safely
                continue
            base = histos[v]
            wpref = _pol_wpref_for(v)
            for pol in ("long", "left", "right"):
                histos[f'{v}_{wpref}_{pol}'] = copy.deepcopy(base)
        return histos
    
    def _add_trigger_sf(self, weights, lead_lep, subl_lep, clibhandler=None):
        mask_BB = ak.fill_none((lead_lep.eta <= 1.5) & (subl_lep.eta <= 1.5), False)
        mask_EB = ak.fill_none((lead_lep.eta >= 1.5) & (subl_lep.eta <= 1.5), False)
        mask_BE = ak.fill_none((lead_lep.eta <= 1.5) & (subl_lep.eta >= 1.5), False)
        mask_EE = ak.fill_none((lead_lep.eta >= 1.5) & (subl_lep.eta >= 1.5), False)

        mask_mm = ak.fill_none((np.abs(lead_lep.pdgId)==13) & (np.abs(subl_lep.pdgId)==13), False)
        mask_ee = ak.fill_none((np.abs(lead_lep.pdgId)==11) & (np.abs(subl_lep.pdgId)==11), False)

        mask_me = (~mask_mm & ~mask_ee) & (np.abs(lead_lep.pdgId) == 13)
        mask_em = (~mask_mm & ~mask_ee) & (np.abs(lead_lep.pdgId) == 11)

        # Correctionlib path
        if clibhandler is not None and "trigger_sf" in clibhandler.keys():
            trigger_sf = clibhandler.getCorrectionSet("trigger_sf")
            # Nominally we'd need to mask each input where invalid, but here we can mix "wrong" SFs in and take care of it in the where statement at the end
            sf_mm_nom = trigger_sf["SF_mm"].evaluate(lead_lep.pt, subl_lep.pt, "nominal")
            sf_me_nom = trigger_sf["SF_me"].evaluate(lead_lep.pt, subl_lep.pt, "nominal")
            sf_em_nom = trigger_sf["SF_em"].evaluate(lead_lep.pt, subl_lep.pt, "nominal")
            sf_ee_nom = trigger_sf["SF_ee"].evaluate(lead_lep.pt, subl_lep.pt, "nominal")
            sf_nom = ak.where(mask_mm, sf_mm_nom, ak.where(mask_me, sf_me_nom, ak.where(mask_em, sf_em_nom, ak.where(mask_ee, sf_ee_nom, ak.ones_like(sf_ee_nom) ) ) ) )
            sf_mm_up = trigger_sf["SF_mm"].evaluate(lead_lep.pt, subl_lep.pt, "up")
            sf_me_up = trigger_sf["SF_me"].evaluate(lead_lep.pt, subl_lep.pt, "up")
            sf_em_up = trigger_sf["SF_em"].evaluate(lead_lep.pt, subl_lep.pt, "up")
            sf_ee_up = trigger_sf["SF_ee"].evaluate(lead_lep.pt, subl_lep.pt, "up")
            sf_up = ak.where(mask_mm, sf_mm_up, ak.where(mask_me, sf_me_up, ak.where(mask_em, sf_em_up, ak.where(mask_ee, sf_ee_up, ak.ones_like(sf_ee_up) ) ) ) )
            sf_mm_down = trigger_sf["SF_mm"].evaluate(lead_lep.pt, subl_lep.pt, "down")
            sf_me_down = trigger_sf["SF_me"].evaluate(lead_lep.pt, subl_lep.pt, "down")
            sf_em_down = trigger_sf["SF_em"].evaluate(lead_lep.pt, subl_lep.pt, "down")
            sf_ee_down = trigger_sf["SF_ee"].evaluate(lead_lep.pt, subl_lep.pt, "down")
            sf_down = ak.where(mask_mm, sf_mm_down, ak.where(mask_me, sf_me_down, ak.where(mask_em, sf_em_down, ak.where(mask_ee, sf_ee_down, ak.ones_like(sf_ee_down) ) ) ) )
            weights.add(
                'triggerSF',
                sf_nom,
                sf_up,
                sf_down,
            )

        # Legacy code path
        else:
            lept_pt_bins = [20, 25, 30, 35, 40, 50, 60, 100000]
            lep_1_bin = np.digitize(lead_lep.pt.to_numpy(), lept_pt_bins) - 1
            lep_2_bin = np.digitize(subl_lep.pt.to_numpy(), lept_pt_bins) - 1
            trigg_bin = np.select([
                (mask_ee & mask_BB).to_numpy(),
                (mask_ee & mask_BE).to_numpy(),
                (mask_ee & mask_EB).to_numpy(),
                (mask_ee & mask_EE).to_numpy(),

                (mask_em & mask_BB).to_numpy(),
                (mask_em & mask_BE).to_numpy(),
                (mask_em & mask_EB).to_numpy(),
                (mask_em & mask_EE).to_numpy(),

                (mask_me & mask_BB).to_numpy(),
                (mask_me & mask_BE).to_numpy(),
                (mask_me & mask_EB).to_numpy(),
                (mask_me & mask_EE).to_numpy(),

                (mask_mm & mask_BB).to_numpy(),
                (mask_mm & mask_BE).to_numpy(),
                (mask_mm & mask_EB).to_numpy(),
                (mask_mm & mask_EE).to_numpy()
            ], np.arange(0,16), 16)

            # this is to avoid cases were two
            # leptons are not in the event
            lep_1_bin[lep_1_bin>6] = -1
            lep_2_bin[lep_2_bin>6] = -1
            center_value = self.trig_sf_map[lep_1_bin,lep_2_bin,trigg_bin,0]
            errors_value = self.trig_sf_map[lep_1_bin,lep_2_bin,trigg_bin,1]

            weights.add(
                'triggerSF', 
                center_value, 
                center_value + errors_value,
                center_value - errors_value
            )


    def process_shift(self, event, shift_name:str=''):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        dataset = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        is_signal = event.metadata.get('is_signal')
        selection = PackedSelection(dtype="uint64")
        weights = Weights(len(event), storeIndividual=True)
        
        if self._pol:
            if is_signal and (not is_data):
                histos = self._build_histos_with_pol()
            else:
                histos = self.build_histos_pol()
        else:
            histos = self.build_histos()
        
        if is_data:
            selection.add('lumimask', self._lumimask(event.run, event.luminosityBlock))
            selection.add('triggers', trigger_rules(event, self._triggers, self._era))
        else:
            selection.add('lumimask', np.ones(len(event), dtype='bool'))
            selection.add('triggers', np.ones(len(event), dtype='bool'))
        if self._jetveto is not None:
            selection.add('jetveto', ~ak.any(
                self._jetveto.evaluate('jetvetomap', event.OrigJet.eta, event.OrigJet.phi), axis=1
            )) # must veto on ALL jets, not selected jets, that are in the veto regions
        # MET filters
        if is_data:
            selection.add(
                'metfilter',
                #event.Flag.METFilters &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter &
                event.Flag.goodVertices &
                event.Flag.eeBadScFilter &
                event.Flag.globalTightHalo2016Filter &
                event.Flag.BadChargedCandidateFilter & 
                event.Flag.BadPFMuonFilter
            )
        else:
            selection.add(
                'metfilter',
                #event.Flag.METFilters &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter & 
                event.Flag.goodVertices &
                event.Flag.eeBadScFilter &
                event.Flag.globalTightHalo2016Filter &
                event.Flag.BadChargedCandidateFilter & 
                event.Flag.BadPFMuonFilter
            )

        ### FV variable defined ###
        if is_signal:

            gen_leptons = event.GenDressedLepton
            # GenDressedLepton_hasTauAnc is true if leptons are coming from tau
            gen_leptons["charge"] = ak.where(gen_leptons.pdgId < 0, 1, -1)
            gen_muons = gen_leptons[abs(gen_leptons.pdgId) == 13]
            gen_electrons = gen_leptons[abs(gen_leptons.pdgId) == 11]
            # gen_taus = gen_leptons[abs(gen_leptons.pdgId) == 15]
            gen_taus = event.GenVisTau
            tight_lep_fv = build_leptons_fv(
                gen_muons,
                gen_electrons
            )
            # tight_lep_fv = tight_lep_fv[~tight_lep_fv.hasTauAnc]
            ntight_lep_fv = ak.num(tight_lep_fv)

            tight_tau_fv = build_htaus_fv(gen_taus, nanoAODversion=self._ver)
            lead_tau_fv = ak.firsts(tight_tau_fv)
            ntight_tau_fv = ak.num(tight_tau_fv)
            tau_pt_fv = lead_tau_fv.pt

            loose_lep_fv  = build_leptons_fv_loose(gen_muons, gen_electrons)
            nloose_lep_fv = ak.num(loose_lep_fv)


            pairs_fv = ak.combinations(tight_lep_fv, 2, axis=1, fields=['l1','l2'])
            # OSSF mask: opposite sign, same flavour (pdgId sum == 0)
            ossf_mask_fv = ak.fill_none((pairs_fv.l1.pdgId + pairs_fv.l2.pdgId) == 0, False)
            pairs_fv_ossf = pairs_fv[ossf_mask_fv]
            mass_fv = (pairs_fv_ossf.l1 + pairs_fv_ossf.l2).mass
            best_idx_fv = ak.argmin(np.abs(mass_fv - self.zmass), axis=1, keepdims=True)
            pairs_fv_best = pairs_fv_ossf[best_idx_fv]

            lead_lep_fv = ak.firsts(pairs_fv_best.l1)
            subl_lep_fv = ak.firsts(pairs_fv_best.l2)

            
            met=event.GenMET
            p4_met_fv = ak.zip(
                {
                    "pt": met.pt,
                    "eta": ak.zeros_like(met.pt),
                    "phi": met.phi,
                    "mass": ak.zeros_like(met.pt),
                    "charge": ak.zeros_like(met.pt),
                },
                with_name="PtEtaPhiMCandidate",
                behavior=candidate.behavior,
            )
            dilep_p4_fv = (lead_lep_fv + subl_lep_fv)
            dilep_m_fv  = dilep_p4_fv.mass
            dilep_pt_fv = dilep_p4_fv.pt
            

            genjets = event.GenJet

            overlap = ak.any(genjets.metric_table(tight_lep_fv) <= 0.4, axis=2)
            overlap_taus_fv = ak.any(genjets.metric_table(tight_tau_fv) <= 0.4, axis=2)

            jet_mask_fv = (
                ~overlap &
                ~overlap_taus_fv &
                (genjets.pt>30.0) & 
                (np.abs(genjets.eta) < 4.7) 
            )
            
             
            good_jets_fv = genjets[jet_mask_fv]
            sorted_indices = np.argsort(-good_jets_fv.pt)
            good_jets_fv = good_jets_fv[sorted_indices]
            
            lead_jet_fv = ak.firsts(good_jets_fv)
            subl_jet_fv = ak.firsts(good_jets_fv[lead_jet_fv.delta_r(good_jets_fv)>0.01])
            third_jet_fv = ak.firsts(good_jets_fv[(lead_jet_fv.delta_r(good_jets_fv)>0.01) & (subl_jet_fv.delta_r(good_jets_fv)>0.01)])
            ngood_jets_fv  = ak.num(good_jets_fv)

            dilep_tau_met_p4_fv = dilep_p4_fv + lead_tau_fv + p4_met_fv
            dilep_tau_met_hadron_mt_fv = np.sqrt((transverse_energy(lead_lep_fv) + transverse_energy(subl_lep_fv) + transverse_energy(lead_tau_fv) + p4_met_fv.pt) ** 2 - dilep_tau_met_p4_fv.pt**2)
            
        ### FV variables ###

        # Electrons and Muons and Taus
        # Adding scale factors to Muon and Electron fields, post-Scale/Smearing
        muonSFs = self._leSF.muonSF(event.Muon)
        elecSFs = self._leSF.electronSF(event.Electron)
        # keys: nominal, eff_m_(id|iso)(Up|Down) (Muon) or eff_e_(reco|id)(Up|Down) (Electron)
        for k, v in muonSFs.items():
            event["Muon", k] = v
        for k, v in elecSFs.items():
            event["Electron", k] = v
        tight_lep, loose_lep, non_iso_leptons = build_leptons(
            event.Muon,
            event.Electron,
            nanoAODversion=self._ver,
            analysisID="inc-WZ-baseline",
            leptonIDs={"electronID": self.electronID, "muonID": self.muonID, "muonISO": self.muonISO},
        )
        tight_sorter = ak.argsort(tight_lep.pt, axis=1, ascending=False)
        tight_lep = tight_lep[tight_sorter]
        ntight_lep = ak.num(tight_lep)
        nloose_lep = ak.num(loose_lep)

        
        had_taus = build_htaus(event.Tau, tight_lep, nanoAODversion=self._ver, tauIDvsj_wp="VTight", tauIDvse_wp=self.tauIDvse_wp, tauIDvsmu_wp=self.tauIDvsmu_wp)
        had_taus_tight = build_htaus(event.Tau, tight_lep, nanoAODversion=self._ver, tauIDvsj_wp="Tight", tauIDvse_wp=self.tauIDvse_wp, tauIDvsmu_wp=self.tauIDvsmu_wp)
        had_taus_loose = build_htaus(event.Tau, tight_lep, nanoAODversion=self._ver, tauIDvsj_wp="Loose", tauIDvse_wp=self.tauIDvse_wp, tauIDvsmu_wp=self.tauIDvsmu_wp)

        # sort tau collections
        vtight_tau_sorter = ak.argsort(had_taus.pt, axis=1, ascending=False)
        had_taus = had_taus[vtight_tau_sorter]
        had_taus_vtight_plus_mask = (had_taus.charge == 1)
        had_taus_vtight_minus_mask = (had_taus.charge == -1)
        nhtaus_lep_vtight = ak.num(had_taus)
        nhtaus_lep_vtight_plus = ak.sum(had_taus_vtight_plus_mask, axis=1)
        nhtaus_lep_vtight_minus = ak.sum(had_taus_vtight_minus_mask, axis=1)

        tight_tau_sorter = ak.argsort(had_taus_tight.pt, axis=1, ascending=False)
        had_taus_tight = had_taus_tight[tight_tau_sorter]
        had_taus_tight_plus_mask = (had_taus_tight.charge == 1)
        had_taus_tight_minus_mask = (had_taus_tight.charge == -1)
        nhtaus_lep_tight = ak.num(had_taus_tight)
        nhtaus_lep_tight_plus = ak.sum(had_taus_tight_plus_mask, axis=1)
        nhtaus_lep_tight_minus = ak.sum(had_taus_tight_minus_mask, axis=1)

        loose_tau_sorter = ak.argsort(had_taus_loose.pt, axis=1, ascending=False)
        # loose_tau_sorter = ak.argsort(had_taus_loose.idDeepTau2017v2p1VSjet, axis=1, ascending=False) #not needed if we always use lead_tau_for_vars
        had_taus_loose = had_taus_loose[loose_tau_sorter]
        had_taus_loose_plus_mask = (had_taus_loose.charge == 1)
        had_taus_loose_minus_mask = (had_taus_loose.charge == -1)
        nhtaus_lep_loose = ak.num(had_taus_loose)
        nhtaus_lep_loose_plus = ak.sum(had_taus_loose_plus_mask, axis=1)
        nhtaus_lep_loose_minus = ak.sum(had_taus_loose_minus_mask, axis=1)




        tau_mask_vtight = (nhtaus_lep_vtight >= 1)
        tau_mask_tight = (nhtaus_lep_tight >= 1)
        tau_mask_loose = (nhtaus_lep_loose >= 1)
        lead_tau_for_vars = ak.firsts(
            ak.where(
                tau_mask_vtight,
                had_taus,
                ak.where(tau_mask_tight,
                         had_taus_tight,
                         had_taus_loose
                         )
            )
        )
        lead_tau_plus_mask = lead_tau_for_vars.charge == 1
        lead_tau_minus_mask = lead_tau_for_vars.charge == -1
        lead_tau_plus_tag = ak.fill_none(lead_tau_plus_mask, False)
        lead_tau_minus_tag = ak.fill_none(lead_tau_minus_mask, False)

        #definig plus and minus lead tau for W plus/minus
        # tau_plus  = had_taus[had_taus.charge == 1]
        # tau_minus = had_taus[had_taus.charge == -1]

        # tau_loose_plus =had_taus_loose[had_taus_loose.charge == 1]
        # tau_loose_minus =had_taus_loose[had_taus_loose.charge == -1]

        lead_tau_vtight = ak.firsts(had_taus)
        lead_tau_tight = ak.firsts(had_taus_tight)
        lead_tau_loose = ak.firsts(had_taus_loose)

        tau_pt_vtight = lead_tau_vtight.pt
        tau_pt_tight = lead_tau_tight.pt
        taus_eta = lead_tau_vtight.eta
        taus_phi = lead_tau_vtight.phi
        tau_E = lead_tau_vtight.E

        tau_pt_loose = lead_tau_loose.pt
        taus_eta_loose = lead_tau_loose.eta
        taus_phi_loose = lead_tau_loose.phi
        tau_E_loose = lead_tau_loose.E

        lead_non_iso_lep = ak.firsts(non_iso_leptons)
        lead_lep_tight = ak.firsts(tight_lep)
        lead_lep_loose = ak.firsts(loose_lep)

        delta_R_non_iso_lep_loose_tau = lead_non_iso_lep.delta_r(lead_tau_loose)
        delta_R_non_iso_lep_vtight_tau = lead_lep_tight.delta_r(lead_tau_loose)
        delta_R_non_iso_lep_tight_tau = lead_lep_loose.delta_r(lead_tau_loose)

        if self._ver in [f"v{V}" for V in range(12)]:
            deep_tau_e = lead_tau_loose.rawDeepTau2017v2p1VSe
            deep_tau_mu = lead_tau_loose.rawDeepTau2017v2p1VSmu
            deep_tau_jet = lead_tau_loose.rawDeepTau2017v2p1VSjet
        elif self._ver in ["v15"]:
            deep_tau_e = lead_tau_loose.rawDeepTau2018v2p5VSe
            deep_tau_mu = lead_tau_loose.rawDeepTau2018v2p5VSmu
            deep_tau_jet = lead_tau_loose.rawDeepTau2018v2p5VSjet

        if "jetId" not in event.Jet.fields:
            if self._ver in [f"v{V}" for V in range(12)]:
                jet_type = "AK4CHS"
            else:
                jet_type = "AK4PUPPI"
            # Match 2017-2018 encoding, no Loose bit active, with following inputs
            jet_args = (event.Jet.eta,                                      # ('eta', 'real', 'pseudorapidity of the jet')
                        event.Jet.chHEF,                                    # ('chHEF', 'real', 'charged Hadron Energy Fraction')
                        event.Jet.neHEF,                                    # ('neHEF', 'real', 'neutral Hadron Energy Fraction')
                        event.Jet.chEmEF,                                   # ('chEmEF', 'real', 'charged Electromagnetic Energy Fraction')
                        event.Jet.neEmEF,                                   # ('neEmEF', 'real', 'neutral Electromagnetic Energy Fraction')
                        event.Jet.muEF,                                     # ('muEF', 'real', 'muon Energy Fraction')
                        event.Jet.chMultiplicity,                           # ('chMultiplicity', 'int', 'charged Multiplicity')
                        event.Jet.neMultiplicity,                           # ('neMultiplicity', 'int', 'neutral Multiplicity')
                        event.Jet.chMultiplicity + event.Jet.neMultiplicity # ('multiplicity', 'int', 'charged Multiplicity + neutral Multiplicity')]
                        )
            event["Jet", "jetId"] = (
                4 * self._jetid[f"{jet_type}_TightLeptonVeto"].evaluate(*jet_args) + 
                2 * self._jetid[f"{jet_type}_Tight"].evaluate(*jet_args)
                )
        jets = event.Jet
        good_jets, good_bjets = build_jets(jets, tight_lep, had_taus_loose, self.btag_wp, self._btag, nanoAODversion=self._ver)
        jet_sorter = ak.argsort(good_jets.pt, axis=1, ascending=False)
        good_jets = good_jets[jet_sorter]

        ngood_jets  = ak.num(good_jets)
        ngood_bjets = ak.num(good_bjets)

        event['ngood_bjets'] = ngood_bjets
        event['ngood_jets']  = ngood_jets

        # lepton quantities
        def z_lepton_pair(leptons):
            pair = ak.combinations(leptons, 2, axis=1, fields=['l1', 'l2'])
            mass = (pair.l1 + pair.l2).mass
            cand = ak.local_index(mass, axis=1) == ak.argmin(np.abs(mass - self.zmass), axis=1)

            extra_lepton = leptons[(
                ~ak.any(leptons.metric_table(pair[cand].l1) <= 0.01, axis=2) & 
                ~ak.any(leptons.metric_table(pair[cand].l2) <= 0.01, axis=2) )
            ]
            return pair[cand], extra_lepton, cand
        dilep, extra_lep, z_cand_mask = z_lepton_pair(tight_lep)
        # this doesn't work in coffea 202X, so instead we sort leptons to guarantee the first is the higher in the pair
        # lead_lep = ak.firsts(ak.where(dilep.l1.pt >  dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        # subl_lep = ak.firsts(ak.where(dilep.l1.pt <= dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        lead_lep = ak.firsts(dilep.l1)
        subl_lep = ak.firsts(dilep.l2)


        dilep_p4 = (lead_lep + subl_lep)
        dilep_m  = dilep_p4.mass
        dilep_pt = dilep_p4.pt
        dilep_E = dilep_p4.t

        # high level observables
        p4_met = ak.zip(
            {
                "pt": event.MET.pt,
                "eta": ak.zeros_like(event.MET.pt),
                "phi": event.MET.phi,
                "mass": ak.zeros_like(event.MET.pt),
                "charge": ak.zeros_like(event.MET.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        
        emu_met = ak.firsts(extra_lep, axis=1) + p4_met
        reco_met_pt = ak.where(ntight_lep==2, p4_met.pt, emu_met.pt)
        reco_met_phi = ak.where(ntight_lep==2, p4_met.phi, emu_met.phi)
        
        ptmiss_sigma = event.MET.significance
        met_px = reco_met_pt * np.cos(reco_met_phi)
        met_py = reco_met_pt * np.sin(reco_met_phi)
        numerator = (
            met_px**2 * event.MET.covXX +
            2 * met_px * met_py * event.MET.covXY +
            met_py**2 * event.MET.covYY
        )
        ptmiss_unc = np.sqrt(numerator / (reco_met_pt**2))


        # this definition is not correct as it doesn't include the mass of the second Z
        dilep_et_ll = np.sqrt(dilep_pt**2 + dilep_m**2)
        dilep_et_nunu = np.sqrt(reco_met_pt**2 + self.zmass**2)
        dilep_mt_llnunu = ak.where(
                ntight_lep==3,
                # mT = m^2 + px^2 + py^2 = E^2 - pz^2
                np.sqrt((dilep_et_ll + dilep_et_nunu)**2 - ((dilep_p4.pvec + emu_met.pvec).pt)**2),
                np.sqrt((dilep_et_ll + dilep_et_nunu)**2 - ((dilep_p4.pvec +  p4_met.pvec).pt)**2)
        )

        dilep_dphi = lead_lep.delta_phi(subl_lep)
        dilep_deta = np.abs(lead_lep.eta - subl_lep.eta)
        dilep_dR   = lead_lep.delta_r(subl_lep)

        delta_R = ak.where(ntight_lep==2, dilep_p4.delta_r(lead_tau_vtight), dilep_p4.delta_r(lead_tau_vtight))
        dilep_dphi_met  = ak.where(ntight_lep==2, dilep_p4.delta_phi(p4_met), dilep_p4.delta_phi(emu_met))
        #scalar_balance = ak.where(ntight_lep==3, emu_met.pt/dilep_p4.pt, p4_met.pt/dilep_p4.pt)
        delta_tau_met_phi = ak.where(ntight_lep==2, lead_tau_vtight.delta_phi(p4_met), lead_tau_vtight.delta_phi(emu_met))
        dilep_dphi_tau = ak.where(ntight_lep==2, dilep_p4.delta_phi(lead_tau_vtight), dilep_p4.delta_phi(lead_tau_vtight))
        delta_tau_loose_met_phi = ak.where(ntight_lep==2, lead_tau_loose.delta_phi(p4_met), lead_tau_loose.delta_phi(emu_met))
        dilep_dphi_tau_loose = ak.where(ntight_lep==2, dilep_p4.delta_phi(lead_tau_loose), dilep_p4.delta_phi(lead_tau_loose))



        #Transverse WZ mass system 
        # Building 4 vector for tranverse mass calculation
        # .t is synonym for energy but there is a bug when we add option types of arrays of two leptons
        dilep_loose_tau_met_p4 = dilep_p4 + lead_tau_for_vars + p4_met
        mT_WZ_square = ((dilep_loose_tau_met_p4.t**2) - (dilep_loose_tau_met_p4.pz**2))
        ptv_llt = (dilep_p4 + lead_tau_for_vars).pt
        pt_llt = (lead_lep.pt + subl_lep.pt + lead_tau_for_vars.pt)
        mT_WZ = np.sqrt(np.maximum(0, mT_WZ_square))


        dilep_tau_loose_met_hadron_mt = np.sqrt((transverse_energy(lead_lep) + transverse_energy(subl_lep) + transverse_energy(lead_tau_for_vars) + p4_met.pt) ** 2 - dilep_loose_tau_met_p4.pt**2)
        m_lep = (dilep_p4 +lead_tau_for_vars).mass
        inv_m_WZ = (dilep_loose_tau_met_p4).mass


        #tranverse W mass
        tau_loose_met_p4 = lead_tau_for_vars + p4_met
        mT_W_square = ((tau_loose_met_p4.t)**2 - (tau_loose_met_p4.pz)**2)
        mT_W = np.sqrt(np.maximum(0, mT_W_square))

        #HT, scalar sum of jet pt, and HTl, HT + lepton pt
        HT = ak.sum(good_jets.pt, axis=1)
        HTl = HT + lead_lep.pt + subl_lep.pt + lead_tau_loose.pt

        # ST, scalar sum of all object pts
        ST = HTl + p4_met.pt


        # ── Polarization angles (only when running the pol part) ──────────────
        if self._pol:
            if (not is_data) and is_signal:
                polq = compute_polarization_quantities(event, event.GenPart, p4_met, lead_tau_for_vars, dilep_p4, lead_lep, subl_lep, z_cand_mask, self.zmass, gen_frac=self.gen_fractions, slope=self.pol_slope, intercept = self.pol_intercept,)
                for _d in ("angles", "extras", "weights"):
                    for _k, _v in polq[_d].items():
                        event[_k] = _v
            else:
                reco = compute_reco_angles(p4_met, lead_tau_for_vars, dilep_p4, lead_lep, subl_lep, z_cand_mask, slope=self.pol_slope, intercept = self.pol_intercept,)
                for _d in ("angles", "extras"):
                    for _k, _v in reco[_d].items():
                        event[_k] = _v
        # ── Polarization block end ────────────────────────────────────────────



        # 2jet and vbs related variables
        lead_jet = ak.firsts(good_jets)
        subl_jet = ak.firsts(good_jets[lead_jet.delta_r(good_jets)>0.01])
        third_jet = ak.firsts(good_jets[(lead_jet.delta_r(good_jets)>0.01) & (subl_jet.delta_r(good_jets)>0.01)])
        delta_R_jet_dilep = ak.where(ntight_lep==2, dilep_p4.delta_r(lead_jet), dilep_p4.delta_r(lead_jet))
        delta_R_jet_tau = ak.where(ntight_lep==2, lead_tau_vtight.delta_r(lead_jet), lead_tau_vtight.delta_r(lead_jet))
        dphi_jet_met = ak.where(ntight_lep==2, lead_jet.delta_phi(p4_met), lead_jet.delta_phi(emu_met))

        dijet_mass = (lead_jet + subl_jet).mass
        dijet_deta = np.abs(lead_jet.eta - subl_jet.eta)
        event['dijet_mass'] = dijet_mass
        event['dijet_deta'] = dijet_deta 

        min_dphi_met_j = ak.min(np.abs(
            ak.where(
                ntight_lep==3, 
                good_jets.delta_phi(emu_met), 
                good_jets.delta_phi(p4_met)
            )
        ), axis=1)
        event['min_dphi_met_j'] = min_dphi_met_j

        ### FV selections ####
        if is_signal:
            selection.add(
                "require-ossf-fv",
                (ntight_lep_fv==2) & (nloose_lep_fv==0) &
                (ak.firsts(tight_lep_fv).pt>25) &
                ak.fill_none((lead_lep_fv.pdgId + subl_lep_fv.pdgId)==0, False)
            )
            selection.add(
                "require-2lep_fv",
                (ntight_lep_fv==2) & (nloose_lep_fv==0) &
                (ak.firsts(tight_lep_fv).pt>25) 
            )
            selection.add('met_pt_fv', ak.fill_none((p4_met_fv.pt > 30), False)) 
            selection.add('low_met_pt_fv', ak.fill_none((p4_met_fv.pt < 20) & (p4_met_fv.pt > 0), False))
            selection.add('dilep_m_fv', ak.fill_none(np.abs(dilep_m_fv - self.zmass) < 15, False))
            selection.add('dilep_pt_fv', ak.fill_none(dilep_pt_fv>30, False))

            selection.add('0njets_fv' , ngood_jets_fv  == 0 )
            selection.add('1njets_fv' , ngood_jets_fv  <= 1 )
            selection.add('1njets_only_fv' , ngood_jets_fv  == 1 )
            selection.add('1nhtaus_fv', ntight_tau_fv  == 1 )

            # selection.add('select_FV01', (ntight_lep_fv==2) & (nloose_lep_fv==0) & (ak.firsts(tight_lep_fv).pt>25) & (ak.fill_none((lead_lep_fv.pdgId + subl_lep_fv.pdgId)==0, False)) & (ntight_tau_fv  == 1) & (ak.fill_none(np.abs(dilep_m_fv - self.zmass) < 15, False)) & (ak.fill_none(dilep_pt_fv>30, False) & (ngood_jets_fv  <= 1)))
            select_FV01 = ['require-ossf-fv', 'require-2lep_fv', '1nhtaus_fv', 'dilep_m_fv', 'dilep_pt_fv', '1njets_fv']
            selection.add('inside_FV01', selection.all(*select_FV01))
        #### FV selections ####

        # define basic selection
        selection.add(
            "require-ossf",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        # FIXME: this isn't osof, it's either sssf or osof together, FIX!
        selection.add(
            "require-osof",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
           ak.fill_none(np.abs(lead_lep.pdgId) != np.abs(subl_lep.pdgId), False)
        )

        selection.add(
            "require-2lep",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) 
        )

        selection.add('met_pt', ak.fill_none((reco_met_pt > 30), False))
        selection.add('low_met_pt', ak.fill_none((reco_met_pt < 20) & (reco_met_pt > 0), False))
        selection.add('val_met_pt', ak.fill_none((reco_met_pt < 30) & (reco_met_pt > 20), False))
        selection.add('dilep_m'   , ak.fill_none(np.abs(dilep_m - self.zmass) < 15, False))
        selection.add('dilep_pt', ak.fill_none(dilep_pt > 30, False))
        selection.add('high_mTwz', ak.fill_none(dilep_tau_loose_met_hadron_mt >= 250, False))
        selection.add("dilep_dphi_met", ak.fill_none(np.abs(dilep_dphi_met)>1.0, False))
        selection.add("dilep_dphi_tau", ak.fill_none(np.abs(dilep_dphi_tau)>1.0, False))
        selection.add("delta_tau_met_phi", ak.fill_none(np.abs(delta_tau_met_phi)>1.0, False))
        selection.add("dilep_dphi_tau_loose", ak.fill_none(np.abs(dilep_dphi_tau_loose)>1.0, False))
        selection.add("delta_tau_loose_met_phi", ak.fill_none(np.abs(delta_tau_loose_met_phi)>1.0, False))
        selection.add(
            "min_dphi_met_j",
            ak.where(
                ngood_jets <= 1, 
                ak.fill_none(np.abs(min_dphi_met_j)>0.25, False), 
                ak.fill_none(np.abs(min_dphi_met_j)>0.5, False), 
            )
        )
        # jet demography
        selection.add('0njets' , ngood_jets  == 0 )
        selection.add('1njets' , ngood_jets  <= 1 )
        selection.add('1njets_only' , ngood_jets  == 1 )
        # selection.add('0nbjets', ngood_bjets == 0 )
        # selection.add('nbjets', ngood_bjets >= 1 )
        selection.add('1nhtaus_vtight', nhtaus_lep_vtight  == 1 )
        selection.add('1nhtaus_tight', nhtaus_lep_tight  == 1 )
        selection.add('1nhtaus_loose', nhtaus_lep_loose  == 1 )
        selection.add('2plusnhtaus_vtight', nhtaus_lep_vtight  >= 2 )
        selection.add('2plusnhtaus_tight', nhtaus_lep_tight  >= 2 )
        selection.add('2plusnhtaus_loose', nhtaus_lep_loose  >= 2 )
        selection.add('lead_tau_plus', lead_tau_plus_tag )
        selection.add('lead_tau_minus', lead_tau_minus_tag )
        # selection.add('1nhtaus_plus', nhtaus_plus == 1)
        # selection.add('1nhtaus_minus', nhtuaus_minus == 1)
        # selection.add('1nhtaus_loose_plus', nhtaus_loose_plus == 1)
        # selection.add('1nhtaus_loose_minus', nhtaus_loose_minus == 1)

        ### FV adding FV fields to the events ####
        if is_signal:
            event['ngood_jets_fv'  ] = ak.fill_none(ngood_jets_fv,-99)
            event['dilep_tau_met_hadron_mt_fv'  ] = ak.fill_none(dilep_tau_met_hadron_mt_fv,-99)
            event['met_sigma'  ] = ak.fill_none(ptmiss_sigma,-99)
            event['met_uncertainty'  ] = ak.fill_none(ptmiss_unc,-99)
            event['met_pt_fv'  ] = ak.fill_none(p4_met_fv.pt,-99)
            event['dilep_m_fv'] = ak.fill_none(dilep_m_fv,-99)
            event['dilep_pt_fv'] = ak.fill_none(dilep_pt_fv,-99)
            event['leading_lep_pt_fv'] = ak.fill_none(lead_lep_fv.pt,-99)
            event['trailing_lep_pt_fv'] = ak.fill_none(subl_lep_fv.pt,-99)
            event['tau_pt_fv'] = ak.fill_none(tau_pt_fv,-99)
            event['lead_jet_pt_fv'] = ak.fill_none(lead_jet_fv.pt,-99)
            event['trail_jet_pt_fv'] = ak.fill_none(subl_jet_fv.pt,-99)

        # Define all variables for the BDT
        event['met_pt'  ] = ak.fill_none(reco_met_pt,-99)
        event['met_phi'  ] = ak.fill_none(reco_met_phi,-99)
        event['mT_W'  ] = ak.fill_none(mT_W,-99)
        event['m_lep'  ] = ak.fill_none(m_lep,-99)
        event['mT_WZ'  ] = ak.fill_none(mT_WZ,-99)
        event['pt_llt'] = ak.fill_none(pt_llt,-99)
        event['ptv_llt'] = ak.fill_none(ptv_llt,-99)
        event['inv_m_WZ'  ] = ak.fill_none(inv_m_WZ,-99)
        event['dilep_tau_loose_met_hadron_mt'  ] = ak.fill_none(dilep_tau_loose_met_hadron_mt,-99)
        event['met_phi' ] = ak.fill_none(reco_met_phi,-99)
        event['dilep_mt_llnunu'] = ak.fill_none(dilep_mt_llnunu,-99)
        event['dilep_m'] = ak.fill_none(dilep_m,-99)
        event['dilep_pt'] = ak.fill_none(dilep_pt,-99)
        event['HTl'] = ak.fill_none(HTl,-99)
        event['ST'] = ak.fill_none(ST,-99)
        event['dilep_dphi'] = ak.fill_none(dilep_dphi,-99)
        event['njets'   ] = ak.fill_none(ngood_jets,-99)
        # event['nbjets'   ] = ak.fill_none(ngood_bjets,-99)
        event['nhtaus_vtight'   ] = ak.fill_none(nhtaus_lep_vtight,-99)
        event['nhtaus_tight'   ] = ak.fill_none(nhtaus_lep_tight,-99)
        event['nhtaus_loose'   ] = ak.fill_none(nhtaus_lep_loose,-99)
        event['dphi_met_ll'] = ak.fill_none(dilep_dphi_met,-99)
        event['dilep_dphi_tau'] = ak.fill_none(dilep_dphi_tau,-99)
        event['dijet_mass'] = ak.fill_none(dijet_mass,-99)
        event['dijet_deta'] = ak.fill_none(dijet_deta,-99)
        event['min_dphi_met_j'] = ak.fill_none(min_dphi_met_j,-99)
        event['tau_pt_vtight'] = ak.fill_none(tau_pt_vtight,-99)
        event['tau_pt_tight'] = ak.fill_none(tau_pt_tight,-99)
        event['taus_phi'] = ak.fill_none(taus_phi,-99)
        event['taus_eta'] = ak.fill_none(taus_eta,-99)
        event['delta_R'] = ak.fill_none(delta_R,-99)
        event['dilep_dR'] = ak.fill_none(dilep_dR,-99)
        event['dilep_deta'] = ak.fill_none(dilep_deta,-99)
        event['delta_tau_met_phi'] = ak.fill_none(delta_tau_met_phi,-99)
        event['tau_pt_loose'] = ak.fill_none(tau_pt_loose,-99)
        event['taus_phi_loose'] = ak.fill_none(taus_phi_loose,-99)
        event['taus_eta_loose'] = ak.fill_none(taus_eta_loose,-99)
        event['leading_lep_pt'  ] = ak.fill_none(lead_lep.pt,-99)
        event['leading_lep_eta' ] = ak.fill_none(lead_lep.eta,-99)
        event['leading_lep_phi' ] = ak.fill_none(lead_lep.phi,-99)
        event['trailing_lep_pt' ] = ak.fill_none(subl_lep.pt,-99)
        event['trailing_lep_eta'] = ak.fill_none(subl_lep.eta,-99)
        event['trailing_lep_phi'] = ak.fill_none(subl_lep.phi,-99)       
        event['lead_jet_pt'  ] = ak.fill_none(lead_jet.pt,-99)
        event['lead_jet_eta' ] = ak.fill_none(lead_jet.eta,-99)
        event['lead_jet_phi' ] = ak.fill_none(lead_jet.phi,-99)
        event['delta_R_jet_tau'] = ak.fill_none(delta_R_jet_tau,-99)
        event['delta_R_jet_dilep'] = ak.fill_none(delta_R_jet_dilep,-99)
        event['dphi_jet_met'] = ak.fill_none(dphi_jet_met,-99)
        event['deep_tau_e'] = ak.fill_none(deep_tau_e,-99)
        event['deep_tau_mu'] = ak.fill_none(deep_tau_mu,-99)
        event['deep_tau_jet'] = ak.fill_none(deep_tau_jet,-99)
        event['delta_R_non_iso_lep_loose_tau'] = ak.fill_none(delta_R_non_iso_lep_loose_tau,-99)
        event['delta_R_non_iso_lep_vtight_tau'] = ak.fill_none(delta_R_non_iso_lep_vtight_tau,-99)
        event['delta_R_non_iso_lep_tight_tau'] = ak.fill_none(delta_R_non_iso_lep_tight_tau,-99)


        # Now adding weights
        _ones = np.ones(len(weights.weight()))
        if not is_data:
            weights.add('genweight', event.genWeight)
            # self._btag.append_btag_sf(jets, weights)
            # FIXME: jpSF only valid for Run2 currently, to be fixed for AK4PUPPI or never needed?
            if self._jpSF is not None:
                self._jpSF.append_jetPU_sf(good_jets, weights)
            else:
                if self._process_shift_warnings:
                    coffea_console.print("[red]JET PU ID SFs DISABLED[/red]")
            self._purw.append_pileup_weight(weights, event.Pileup.nTrueInt) # fix: https://github.com/9GaoHong/SMQawa_update/commit/d6cdebda4856593162c03365eb9d9a91ceb1a185
            self._tauID.append_tauID_multiwp_sf(had_taus, had_taus_tight, had_taus_loose,
                                                tau_mask_vtight, tau_mask_tight, tau_mask_loose,
                                                weights
                                                )
            # self._tauID.append_tauID_sf(had_taus, weights)
            self._add_trigger_sf(weights, lead_lep, subl_lep, clibhandler=self.clibhandler)
            self._leSF.append_lepton_sf(lead_lep, subl_lep, weights)
            if self.ewk_process_name:
                self.ewk_corr.get_weight(
                        event.GenPart,
                        event.Generator.x1,
                        event.Generator.x2,
                        weights
                )
            else:
                weights.add("kEW", _ones, _ones, _ones)

            if "PSWeight" in event.fields:
                theory_ps_weight(weights, event.PSWeight)
            else:
                theory_ps_weight(weights, None)

            if "LHEPdfWeight" in event.fields:
                theory_pdf_weight(weights, event.LHEPdfWeight)
            else:
                theory_pdf_weight(weights, None)

            if ('LHEScaleWeight' in event.fields) and (len(event.LHEScaleWeight[0]) > 0):
                if len(event.LHEScaleWeight[0]) == 9:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 7])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 5])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 8])
                elif len(event.LHEScaleWeight[0]) == 8:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 6])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 4])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 7])
                elif len(event.LHEScaleWeight[0]) == 18:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 2], event.LHEScaleWeight[:, 14])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 6], event.LHEScaleWeight[:, 10])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 16])
                else:
                    if self._process_shift_warnings:
                        coffea_console.print("WARNING: QCD scale variation type not recongnised ... ")

            if 'LHEReweightingWeight' in event.fields and 'aQGC' in dataset:
                for i in range(1057):
                    weights.add(f"eft_{self._eftnames[i]}", event.LHEReweightingWeight[:, i])

            # 2017 Prefiring correction weight
            if 'L1PreFiringWeight' in event.fields:
                # Doesn't appear to be calculated (by default) in Run3 v15 NanoAOD
                weights.add("prefiring_weight", event.L1PreFiringWeight.Nom, event.L1PreFiringWeight.Dn, event.L1PreFiringWeight.Up)
        else:
            # If systematic variations are needed, they must be manually inserted here to give different DD estimates; they should be picked up later for histos.
            weights.add("datadriven_DDDYNominal", _ones, self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, systematic="nominal"), self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, systematic="nominal"))  #added nominal value twice to avoid getting 1/up for the nominaldown
            if self._dd.stat_systematics:
                # Per-era statistical nuisance(s): DataDrivenEventReweight restricts stat_systematics
                # to the single stat_{era} matching the era/subera being processed, so the other
                # eras' stat nuisances stay at nominal for these events (keeping them decorrelated).
                for _stat in self._dd.stat_systematics:
                    weights.add(f"datadriven_{_stat}", _ones, self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, f"{_stat}Up"), self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, f"{_stat}Down"))
            elif self._dd.has_legacy_dddy:
                # Legacy estimate: a single combined statistical nuisance
                weights.add("datadriven_DDDY",_ones, self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, "DDDYUp"), self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, "DDDYDown"))
            # Propagated MC systematics from the non-DY subtraction, added under their bare source
            # names so they correlate with the same-named analysis nuisances on the MC.
            for _mcsyst in self._dd.mc_systematics:
                weights.add(_mcsyst, _ones, self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, f"{_mcsyst}Up"), self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, f"{_mcsyst}Down"))
        # selections (delta_tau_met_phi cut is removed from SR)

        common_sel = ['triggers', 'lumimask', 'metfilter']
        if "jetveto" in selection.names:
            common_sel += ["jetveto"]
        channels = {
            "inc-SR0": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'met_pt',
        ],
            "inc-SR1": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_vtight' ,'met_pt',
        ],
            "inc-SR01": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'met_pt'
        ],
            "inc-VR0": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'val_met_pt'
        ],
            "inc-VR1": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_vtight' , 'val_met_pt'
        ],
            "inc-VR01": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_vtight' , 'val_met_pt'
        ],
            "inc-VB0": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'val_met_pt'
        ],

            "inc-VB1": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'val_met_pt'
        ],
            "inc-VB01": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'val_met_pt'
        ],
            "inc-B0": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'met_pt'
        ],
            "inc-B1": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'met_pt'
        ],
            "inc-B01": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'met_pt'
        ],
            "inc-C0": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', 'low_met_pt', '1nhtaus_loose', '0njets', '~1nhtaus_tight', '~1nhtaus_vtight'
        ],
            "inc-C01": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', 'low_met_pt', '1nhtaus_loose', '1njets', '~1nhtaus_tight', '~1nhtaus_vtight'
        ],
            "inc-C1": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', 'low_met_pt', '1nhtaus_loose', '1njets_only', '~1nhtaus_tight', '~1nhtaus_vtight'
        ],
            "inc-D0": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'low_met_pt'
        ],
            "inc-D01": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'low_met_pt'
        ],
            "inc-D1": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_vtight', '~2plusnhtaus_tight', 'low_met_pt'
        ],
            "inc-IR0L": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'low_met_pt'
        ],
            "inc-IR01L": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'low_met_pt'
        ],
            "inc-IR1L": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'low_met_pt'
        ],
            "inc-IR0M": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'val_met_pt'
        ],

            "inc-IR1M": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'val_met_pt'
        ],
             "inc-IR01M": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'val_met_pt'
        ],
            "inc-IR0H": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'met_pt'
        ],
            "inc-IR01H": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'met_pt'
        ],

            "inc-IR1H": common_sel + [
                'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'met_pt'
        ],            

        }


        if self._split_by_charge:
            charged_channels = {}
            for channel, reqs in channels.items():
                reqs_plus = reqs + ["lead_tau_plus"]
                reqs_minus = reqs + ["lead_tau_minus"]
                charged_channels[f"{channel}_p"] = reqs_plus
                charged_channels[f"{channel}_m"] = reqs_minus
            # Replace channels with charge-separated channels
            channels = charged_channels


        if is_signal and self._split_FV:
            FV_channels = {}
            for channel, reqs in channels.items():
                reqs_in_FV = reqs + ["inside_FV01"]
                reqs_out_FV = reqs + ["~inside_FV01"]
                FV_channels[f"{channel}_IFV"] = reqs_in_FV
                FV_channels[f"{channel}_OFV"] = reqs_out_FV
            # Replace channels with charge-separated channels
            channels = FV_channels

        if self._pol:
            channels = {
            "inc-SR01": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'met_pt', 'high_mTwz'],
            "inc-B01": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'met_pt', 'high_mTwz'],
            "Pol_reg": []
            }
            
            

        def _format_variable(variable, cut):
            if cut is None:
                vv = ak.to_numpy(ak.fill_none(variable, np.nan))
                if np.isnan(np.any(vv)):
                    coffea_console.print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable, np.nan))
            else:
                vv = ak.to_numpy(ak.fill_none(variable[cut], np.nan))
                if np.isnan(np.any(vv)):
                    coffea_console.print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable[cut], np.nan))

        def collection_printer(collection):
            longest_field = max([len(field) for field in collection.fields])
            for field in collection.fields:
                coffea_console.print(f"\t{field:<{longest_field}}={getattr(collection, field)}")

        def _histogram_filler(ch, syst, var, _weight=None, pol_weights=None):
            sel_ = channels[ch]
            sel_args_ = {
                s.replace('~',''): (False if '~' in s else True) for s in sel_ if var not in s
            }
            cut =  selection.require(**sel_args_)

            systname = 'nominal' if syst is None else syst

            if syst in weights.variations:
                weight = weights.weight(modifier=syst)[cut]
            else:
                weight = weights.weight()[cut]
            if _weight is not None:
                weight = weight * _weight[cut]

            fill_var = _format_variable(event[var], cut)

            if pol_weights is None:
                histos[var].fill(
                    **{
                        "channel": ch,
                        "systematic": systname,
                        var: fill_var,
                        "weight": ak.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
                    }
                )
            else:
                # one weight + one fill-value computation reused across all the
                # pure-state clones; pol weight is gen-level / detector-stable.
                for postfix, pol_w in pol_weights.items():
                    pw = ak.to_numpy(ak.fill_none(pol_w[cut], 1.0))
                    w = weight * pw
                    histos[var + postfix].fill(
                        **{
                            "channel": ch,
                            "systematic": systname,
                            var: fill_var,
                            "weight": ak.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
                        }
                    )


        def _hist2d_filler(ch, syst, xvar, yvar, hist_key, extra_weight=None):
            # gen-vs-reco migration map; hist_key picks the explicit 2D histo
            # (axis names = xvar, yvar). Used for qcos_theta_w_reco_gen etc.
            sel_ = channels[ch]
            sel_args_ = {
                s.replace('~',''): (False if '~' in s else True) for s in sel_ if xvar not in s and yvar not in s
            }
            cut = selection.require(**sel_args_)
            systname = 'nominal' if syst is None else syst
            if syst in weights.variations:
                weight = weights.weight(modifier=syst)[cut]
            else:
                weight = weights.weight()[cut]
            if extra_weight is not None:
                # e.g. a 0/1 gen-matching flag -> fills only matched pairs
                weight = weight * ak.to_numpy(ak.fill_none(extra_weight, 0.0))[cut]
            histos[hist_key].fill(
                **{
                    "channel": ch,
                    "systematic": systname,
                    xvar: _format_variable(event[xvar], cut),
                    yvar: _format_variable(event[yvar], cut),
                    "weight": ak.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
                }
            )

        def _histogram_filler2D(ch, syst, var1, var2, _weight=None):
            sel_ = channels[ch]
            sel_args_ = {
                s.replace('~',''): (False if '~' in s else True) for s in sel_ if var1 not in s and var2 not in s
            }
            cut =  selection.require(**sel_args_)

            systname = 'nominal' if syst is None else syst

            if _weight is None: 
                if syst in weights.variations:
                    weight = weights.weight(modifier=syst)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * _weight[cut]

            vv = ak.to_numpy(ak.fill_none(weight, np.nan))
            if np.isnan(np.any(vv)):
                coffea_console.print(f" - {syst} weight contains invalid values:", vv[np.isnan(vv)], vv[np.isinf(vv)])

            histos[var1+"_2D_"+var2].fill(
                **{
                    "channel": ch, 
                    "systematic": systname, 
                    var1: _format_variable(event[var1], cut),
                    var2: _format_variable(event[var2], cut), 
                    "weight": ak.nan_to_num(weight,nan=1.0, posinf=1.0, neginf=1.0)
                }
            )
        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]

        if self._pol:
            for ch in channels:
                for sys in systematics:
                    for var in self.pol_reco_vars:
                        if var in histos:
                            _histogram_filler(ch, sys, var)

            
            if is_signal and (not is_data):
                for ch in channels:
                    for sys in systematics:
                        gv = event['gen_valid']
                        for var in self.pol_template_vars:
                            wpref = _pol_wpref_for(var)
                            if f'{var}_{wpref}_long' not in histos:
                                continue
                            wdict = {f'_{wpref}_{pol}': event[f'{wpref}_{pol}'] * gv
                                     for pol in ('long', 'left', 'right')}
                            _histogram_filler(ch, sys, var, pol_weights=wdict)
                        for var in self.pol_gen_vars:
                            if var in histos:
                                _histogram_filler(ch, sys, var)
                        _histogram_filler(ch, sys, 'qcos_theta_w_gen',
                                           pol_weights={'_onshell': ak.values_astype(event['rescued'] < 0.5, np.float32)})
                        if sys is None:
                            for wk in self.pol_weight_keys:
                                _histogram_filler(ch, None, wk)
                            _hist2d_filler(ch, None, 'qcos_theta_w_gen', 'qcos_theta_w_reco', 'qcos_theta_w_reco_gen')
                            _hist2d_filler(ch, None, 'cos_theta_z_gen',  'cos_theta_z_reco',  'cos_theta_z_reco_gen')
                            # tau pT response (slope) and the resolution handle for f0
                            _hist2d_filler(ch, None, 'tau_pt_gen', 'tau_pt_vtight', 'tau_pt_gen_vs_reco')
                            # matched-only version: fit the tau pT slope on THIS one
                            _hist2d_filler(ch, None, 'tau_pt_gen', 'tau_pt_vtight', 'tau_pt_gen_vs_reco_matched', extra_weight=event['tau_matched'])
                            _hist2d_filler(ch, None, 'qcos_theta_w_reco', 'tau_corr_pt', 'qcos_w_vs_tau_pt')
        else:
            # ── standard inclusive analysis ────────────────────────────────────
            for ch in channels:
                for sys in systematics:
                    _histogram_filler(ch, sys, 'leading_lep_pt')
                    _histogram_filler(ch, sys, 'leading_lep_phi')
                    _histogram_filler(ch, sys, 'leading_lep_eta')
                    _histogram_filler(ch, sys, 'trailing_lep_pt')
                    _histogram_filler(ch, sys, 'trailing_lep_phi')
                    _histogram_filler(ch, sys, 'trailing_lep_eta')
                    _histogram_filler(ch, sys, 'met_pt')
                    _histogram_filler(ch, sys, 'met_phi')
                    _histogram_filler(ch, sys, 'tau_pt_vtight')
                    _histogram_filler(ch, sys, 'tau_pt_tight')
                    _histogram_filler(ch, sys, 'taus_phi')
                    _histogram_filler(ch, sys, 'taus_eta')
                    _histogram_filler(ch, sys, 'tau_pt_loose')
                    _histogram_filler(ch, sys, 'taus_phi_loose')
                    _histogram_filler(ch, sys, 'taus_eta_loose')
                    _histogram_filler(ch, sys, 'lead_jet_pt')
                    _histogram_filler(ch, sys, 'lead_jet_phi')
                    _histogram_filler(ch, sys, 'lead_jet_eta')
                    _histogram_filler(ch, sys, 'njets')
                    _histogram_filler(ch, sys, 'nhtaus_loose')
                    _histogram_filler(ch, sys, 'nhtaus_tight')
                    _histogram_filler(ch, sys, 'nhtaus_vtight')
                    _histogram_filler(ch, sys, 'dilep_pt')
                    _histogram_filler(ch, sys, 'pt_llt')
                    _histogram_filler(ch, sys, 'ptv_llt')
                    _histogram_filler(ch, sys, 'dilep_dphi')
                    _histogram_filler(ch, sys, 'dilep_deta')
                    _histogram_filler(ch, sys, 'dilep_m')
                    _histogram_filler(ch, sys, 'dilep_dR')
                    _histogram_filler(ch, sys, 'delta_R')
                    _histogram_filler(ch, sys, 'dphi_met_ll')
                    _histogram_filler(ch, sys, 'dilep_dphi_tau')
                    _histogram_filler(ch, sys, 'dphi_jet_met')
                    _histogram_filler(ch, sys, 'mT_W')
                    _histogram_filler(ch, sys, 'inv_m_WZ')
                    _histogram_filler(ch, sys, 'm_lep')
                    _histogram_filler(ch, sys, 'dilep_tau_loose_met_hadron_mt')
                    _histogram_filler(ch, sys, 'dilep_mt_llnunu')
                    _histogram_filler(ch, sys, 'HTl')
                    _histogram_filler(ch, sys, 'ST')

        self._process_shift_warnings = False # this will only print warning for first process shift
        return {dataset: histos}

    def process(self, event):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        is_signal = event.metadata.get("is_signal")


        # JES/JER corrections
        Rho = event.Rho.fixedGridRhoFastjetAll if "Rho" in event.fields else event.fixedGridRhoFastjetAll # if self._ver in [f"v{V}" for V in range(12)]
        cache = {}

        raw_met = event.RawMET if self._ver in [f"v{V}" for V in range(12)] else event.RawPuppiMET
        met_to_correct = event.MET if self._ver in [f"v{V}" for V in range(12)] else event.PuppiMET
        met_to_correct = prepare_met_for_factory(met_to_correct) # TODO: remove this shim when coffea properly handles the deltas vs final pt/phi variations
        if ak.parameters(met_to_correct).get("__record__") != "MissingET":
            met_to_correct = as_missinget(met_to_correct) # TODO: remove this behavior shim when coffea annotates all the new MET variants as MissingET type
       
        jets = self._jmeu.corrected_jets_L123_JER(event.Jet, Rho, cache)
        jets_to_correct_met = self._jmeu.corrected_jets_L123_noJER(event.Jet, Rho, cache) # For case without smeared L123 jets ONLY
        met = self._jmeu.corrected_met(met_to_correct, jets, Rho, cache) # we are adding fully smeared L123 jets

        event = ak.with_field(event, event.Jet, 'OrigJet')
        event = ak.with_field(event, met_to_correct, 'OrigMET')
        event = ak.with_field(event, jets, 'Jet')
        event = ak.with_field(event, jets_to_correct_met, 'JetforMET')
        event = ak.with_field(event, met, 'MET')


        run = event.run 
        npv = event.PV.npvs

        # We apply met xy correction for AK4PFCHS jets only
        if self._ver in [f"v{V}" for V in range(12)]:
            met = met_phi_xy_correction(
                event.MET, run, npv, 
                is_mc=not is_data, 
                era=self._era
            )
            event = ak.with_field(event, met, 'MET')
        
        # Apply Muon rochester_correction or Scale and Resolution KIT corrections
        if "muon_scalesmearing" in self.clibhandler.keys():
            # Main pt scale and smearing on the central value, updates event.Muon automatically
            muon_pt_scare(sink=None, events=event, unc_type=None, is_correction=True, clibhandler=self.clibhandler)
            # Adhere to CMS systematics convention https://gitlab.cern.ch/cms-analysis/general/systematics
            muon = event.Muon
            if not is_data:
                muon.add_systematic("scale_m", "UpDownSystematic", "pt", partial(muon_pt_scare, events=event, unc_type="Scale", is_correction=False, clibhandler=self.clibhandler))
                muon.add_systematic("res_m", "UpDownSystematic", "pt", partial(muon_pt_scare, events=event, unc_type="Resolution", is_correction=False, clibhandler=self.clibhandler))
                met.add_systematic("scale_m", "UpDownMultiSystematic", ("pt", "phi"),
                                   partial(propagate_shift_to_met, events=event, met=met, shifted_collection=muon, unc_type="Scale", is_correction=False)
                                   )
                met.add_systematic("res_m", "UpDownMultiSystematic", ("pt", "phi"),
                                   partial(propagate_shift_to_met, events=event, met=met, shifted_collection=muon, unc_type="Resolution", is_correction=False)
                                   )
        # Rochester corrections path
        else:
            muon = event.Muon
            muonEnUp = event.Muon
            muonEnDown = event.Muon
            muon_pt, muon_pt_up, muon_pt_down = rochester_correction(is_data).apply_rochester_correction(muon)

            muon['pt'] = muon_pt
            muonEnUp['pt'] = muon_pt_up
            muonEnDown['pt'] = muon_pt_down
        event = ak.with_field(event, muon, 'Muon')

        # Electron corrections
        electron = event.Electron
        if "electronSS_EtDependent" in self.clibhandler.keys():
            # Main pt/energy scale correction needed in data, updates event.Electron pt and energyErr fields and adds *_orig, plus the scale and smear variations to MC
            EGM_scale_and_smear(None, events=event, is_correction=True, unc_type=None, restriction=None, is_electron=True, clibhandler=self.clibhandler)
            # Unlike Muon ScaRe KIT corrections, these systematics calls just extract the already embedded pt_(scale|smear)_(up|down) pt and energyErr for casting to systematics (if MC)
            if not is_data:
                electron.add_systematic("scale_e", "UpDownMultiSystematic", ("pt", "energyErr"),
                                        partial(EGM_scale_and_smear, events=event, is_correction=False, unc_type="Scale", restriction=None, is_electron=True, clibhandler=self.clibhandler)
                                        )
                electron.add_systematic("res_e", "UpDownMultiSystematic", ("pt", "energyErr"),
                                        partial(EGM_scale_and_smear, events=event, is_correction=False, unc_type="Smear", restriction=None, is_electron=True, clibhandler=self.clibhandler)
                                        )
                met.add_systematic("scale_e", "UpDownMultiSystematic", ("pt", "phi"),
                                   partial(propagate_shift_to_met, events=event, met=met, shifted_collection=electron, unc_type="Scale", is_correction=False)
                                   )
                met.add_systematic("res_e", "UpDownMultiSystematic", ("pt", "phi"),
                                   partial(propagate_shift_to_met, events=event, met=met, shifted_collection=electron, unc_type="Resolution", is_correction=False)
                                   )
        else:
            if not is_data:
                electron.add_systematic("scale_e", "UpDownMultiSystematic", ("pt", "energyErr"),
                                        partial(EGM_scale_and_smear_v9, events=event,is_correction=False, unc_type="Scale", restriction=None, is_electron=True)
                                        )
                electron.add_systematic("res_e", "UpDownMultiSystematic", ("pt", "energyErr"),
                                        partial(EGM_scale_and_smear_v9, events=event, is_correction=False, unc_type="Smear", restriction=None, is_electron=True)
                                        )
                met.add_systematic("scale_e", "UpDownMultiSystematic", ("pt", "phi"),
                                   partial(propagate_shift_to_met, events=event, met=met, shifted_collection=electron, unc_type="Scale", is_correction=False)
                                   )
                met.add_systematic("res_e", "UpDownMultiSystematic", ("pt", "phi"),
                                   partial(propagate_shift_to_met, events=event, met=met, shifted_collection=electron, unc_type="Resolution", is_correction=False)
                                   )
            
        event = ak.with_field(event, electron, 'Electron')

        # IF is_data is True, shortcut with process_shift for nominal
        if is_data:
            return self.process_shift(event, None)

        #Tau corrections are applied only to Monte Carlo, not to data, hence post-process_shift
        try:
            tau_energy_scale(None, event, tagger=self._tauID.tagger, wp_VSjet="VTight", wp_VSe="Tight",
                             unc_type=None, is_correction=True, clibhandler=self.clibhandler)
            # Access tau only post-correction
            tau = event.Tau
            tau.add_systematic("scale_t", "UpDownMultiSystematic", ("pt", "mass"),
                               partial(tau_energy_scale, events=event, tagger=self._tauID.tagger, wp_VSjet="VTight", wp_VSe="Tight",
                                       unc_type="Scale", is_correction=False, clibhandler=self.clibhandler)
                               )
        except IndexError:
            tau_energy_scale(None, event, tagger=self._tauID.tagger, wp_VSjet="VTight", wp_VSe="Tight",
                             unc_type=None, is_correction=True, dm2IndexErrorWorkaround=True, clibhandler=self.clibhandler)
            # Access tau only post-correction
            tau = event.Tau
            tau.add_systematic("scale_t", "UpDownMultiSystematic", ("pt", "mass"),
                               partial(tau_energy_scale, events=event, tagger=self._tauID.tagger, wp_VSjet="VTight", wp_VSe="Tight",
                                       unc_type="Scale", is_correction=False, dm2IndexErrorWorkaround=True,clibhandler=self.clibhandler)
                               )
        met.add_systematic("scale_t", "UpDownMultiSystematic", ("pt", "phi"),
                           partial(propagate_shift_to_met, events=event, met=met, shifted_collection=tau, unc_type="Scale", is_correction=False)
                           )
        event = ak.with_field(event, tau, 'Tau')

        # One final update of MET for good measure
        event = ak.with_field(event, met, 'MET')
    
        # define all the shifts
        shifts = [
            # Jets
            ({"Jet": jets                             , "MET": met                               }, None                  ),
            ({"Jet": jets.JES_Total.up                , "MET": met.JES_Total.up                  }, "JESUp"               ),
            ({"Jet": jets.JES_Total.down              , "MET": met.JES_Total.down                }, "JESDown"             ),
            ({"Jet": jets.JES_Absolute.up             , "MET": met.JES_Absolute.up               }, "JES_AbsoluteUp"      ),
            ({"Jet": jets.JES_Absolute.down           , "MET": met.JES_Absolute.down             }, "JES_AbsoluteDown"    ),
            ({"Jet": jets.JES_BBEC1.up                , "MET": met.JES_BBEC1.up                  }, "JES_BBEC1Up"         ),
            ({"Jet": jets.JES_BBEC1.down              , "MET": met.JES_BBEC1.down                }, "JES_BBEC1Down"       ),
            ({"Jet": jets.JES_EC2.up                  , "MET": met.JES_EC2.up                    }, "JES_EC2Up"           ),
            ({"Jet": jets.JES_EC2.down                , "MET": met.JES_EC2.down                  }, "JES_EC2Down"         ),
            ({"Jet": jets.JES_FlavorQCD.up            , "MET": met.JES_FlavorQCD.up              }, "JES_FlavorQCDUp"     ),
            ({"Jet": jets.JES_FlavorQCD.down          , "MET": met.JES_FlavorQCD.down            }, "JES_FlavorQCDDown"   ),
            ({"Jet": jets.JES_HF.up                   , "MET": met.JES_HF.up                     }, "JES_HFUp"            ),
            ({"Jet": jets.JES_HF.down                 , "MET": met.JES_HF.down                   }, "JES_HFDown"          ),
            ({"Jet": jets.JES_RelativeBal.up          , "MET": met.JES_RelativeBal.up            }, "JES_RelativeBalUp"   ),
            ({"Jet": jets.JES_RelativeBal.down        , "MET": met.JES_RelativeBal.down          }, "JES_RelativeBalDown" ),
            ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.up      }, "UESUp"               ),
            ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.down    }, "UESDown"             ), 
              ##year dependent systematics
            ({"Jet": getattr(jets,f'JES_BBEC1_{self._era}').up     , "MET": getattr(met,f'JES_BBEC1_{self._era}').up      }, f"JES_BBEC1{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_BBEC1_{self._era}').down   , "MET": getattr(met,f'JES_BBEC1_{self._era}').down    }, f"JES_BBEC1{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_Absolute_{self._era}').up  , "MET": getattr(met,f'JES_Absolute_{self._era}').up   }, f"JES_Absolute{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_Absolute_{self._era}').down, "MET": getattr(met,f'JES_Absolute_{self._era}').down }, f"JES_Absolute{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_EC2_{self._era}').up       , "MET": getattr(met,f'JES_EC2_{self._era}').up        }, f"JES_EC2{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_EC2_{self._era}').down     , "MET": getattr(met,f'JES_EC2_{self._era}').down      }, f"JES_EC2{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_HF_{self._era}').up        , "MET": getattr(met,f'JES_HF_{self._era}').up         }, f"JES_HF{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_HF_{self._era}').down      , "MET": getattr(met,f'JES_HF_{self._era}').down       }, f"JES_HF{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_RelativeSample_{self._era}').up  , "MET": getattr(met,f'JES_RelativeSample_{self._era}').up   }, f"JES_RelativeSample{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_RelativeSample_{self._era}').down, "MET": getattr(met,f'JES_RelativeSample_{self._era}').down }, f"JES_RelativeSample{self._era}Down"),
        ]
        if "JER" in jets.fields and "JER" in met.fields:
            shifts += [
                ({"Jet": jets.JER.up                      , "MET": met.JER.up                        }, "JERUp"               ),
                ({"Jet": jets.JER.down                    , "MET": met.JER.down                      }, "JERDown"             ),
                ]
        else:
            if self._process_shift_warnings:
                coffea_console.print(f"WARNING: JER variation missing in jets, met, or both: JER in\n\tjets... {'JER' in jets.fields}\n\tmet... {'JER' in met.fields}")
        if "scale_e" in event.Electron.systematics.fields:
            shifts += [
                ({"Electron": event.Electron.systematics.scale_e.up  , "MET": met.systematics.scale_e.up}, "scale_eUp"  ),
                ({"Electron": event.Electron.systematics.scale_e.down, "MET": met.systematics.scale_e.down}, "scale_eDown"),
                ({"Electron": event.Electron.systematics.res_e.up  , "MET": met.systematics.res_e.up}, "res_eUp"  ),
                ({"Electron": event.Electron.systematics.res_e.down, "MET": met.systematics.res_e.down}, "res_eDown"),
            ]
        else:
            shifts += [            
                # Electrons + MET shift (FIXME: shift to be added)
                ({"Electron": electronEnUp  }, "scale_eUp"),
                ({"Electron": electronEnDown}, "scale_eDown"),                
            ]
        if "scale_m" in event.Muon.systematics.fields:
            shifts += [
                ({"Muon": event.Muon.systematics.scale_m.up, "MET": met.systematics.scale_m.up}, "scale_mUp"),
                ({"Muon": event.Muon.systematics.scale_m.down, "MET": met.systematics.scale_m.down}, "scale_mDown"),
                ({"Muon": event.Muon.systematics.res_m.up, "MET": met.systematics.res_m.up}, "res_mUp"),
                ({"Muon": event.Muon.systematics.res_m.down, "MET": met.systematics.res_m.down}, "res_mDown"),
            ]
        else:
            shifts += [
                # Muon + MET shifts
                ({"Muon": muonEnUp  }, "scale_mUp"),
                ({"Muon": muonEnDown}, "scale_mDown"),
            ]
        if "scale_t" in event.Tau.systematics.fields:
            shifts += [
                ({"Tau": event.Tau.systematics.scale_t.up, "MET": met.systematics.scale_t.up}, "scale_tUp"),
                ({"Tau": event.Tau.systematics.scale_t.down, "MET": met.systematics.scale_t.down}, "scale_tDown"),
            ]
        else:
            raise NotImplementedError("Legacy tau energy scale method has been deprecated")

        # the not is_data requirement is technically redundant if the process_shift is applied with the rochester correction for data path above, but it's helpful for clarity
        if (self._era == '2018') and (not is_data):
            hem_jets, hem_met = apply_hem_uncertainty(
                event.Jet,
                event.MET
            )
            shifts.append(({"Jet": hem_jets, "MET": hem_met}, "HEMDown"))
            shifts.append(({"Jet": event.Jet, "MET": event.MET}, "HEMUp"))

        shifts = [
            self.process_shift(
                update_collection(event, collections), 
                name
            ) for collections, name in shifts
        ]
        return processor.accumulate(shifts)



    def postprocess(self, accumulator):
        return accumulator