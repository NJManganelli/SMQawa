import awkward as ak
import numpy as np
import scipy.interpolate as interp
from scipy import stats as st
import uproot
import pickle
import hist
import yaml
import copy
import os
import re
import gzip
from coffea import processor
from coffea import nanoevents
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import nanoaod
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import PackedSelection
from qawa.roccor import rochester_correction
from qawa.leptonsSF import LeptonScaleFactors
from qawa.jetPU import jetPUScaleFactors
from qawa.tauSF import tauIDScaleFactors
from qawa.btag import BTVCorrector, btag_id
from qawa.jme import JMEUncertainty, update_collection
from qawa.gen_match import delta_r2, find_best_match
from qawa.datadriven import DataDrivenEventReweight
from qawa.common import pileup_weights, ewk_corrector, met_phi_xy_correction, theory_ps_weight, theory_pdf_weight, trigger_rules, transverse_energy

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
        (SCeta  < 2.4) & 
        (~electrons.hasTauAnc) #avoid e coming from tau

    )
    tight_electrons = electrons[tight_electrons_mask]
    # contruct a lepton object
    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')

    tight_sorted_index = ak.argsort(tight_leptons.pt,ascending=False)

    tight_leptons = tight_leptons[tight_sorted_index]

    return tight_leptons

def build_leptons(muons, electrons):
    # select tight/loose muons
    tight_muons_mask = (
        (muons.pt             >  20.0) &
        (np.abs(muons.eta)    <  2.4) &
        (np.abs(muons.dxy)    <  0.045) &
        (np.abs(muons.dz )    <  0.2) &
        (muons.pfRelIso04_all <= 0.15) & 
        muons.tightId
    )
    tight_muons = muons[tight_muons_mask]
    loose_muons = muons[
        ~tight_muons_mask &
        (muons.pt            >  10.0) &
        (np.abs(muons.eta)   <  2.4) &
        (muons.pfRelIso04_all<= 0.25) &
        muons.looseId   
    ]
    # select tight/loose electron
    electron_superclusterEta = electrons.eta + electrons.deltaEtaSC
    tight_electrons_mask = (
        (electrons.pt           > 20.0) &
        ((np.abs(electron_superclusterEta) < 1.4442) | ((np.abs(electron_superclusterEta) > 1.5660) & (np.abs(electron_superclusterEta)  < 2.5)))  &
        electrons.mvaFall17V2Iso_WP90
    )
    tight_electrons = electrons[tight_electrons_mask]
    loose_electrons = electrons[
        ~tight_electrons_mask &
        (electrons.pt           > 10.0) &
        (np.abs(electrons.eta)  < 2.5) &
        electrons.mvaFall17V2Iso_WPL
    ]
    # contruct a lepton object
    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')
    loose_leptons = ak.with_name(ak.concatenate([loose_muons, loose_electrons], axis=1), 'PtEtaPhiMCandidate')

    return tight_leptons, loose_leptons


def build_htaus_fv(tau):
    tight_taus_mask = (
        (tau.pt             >  20.0) & 
        (np.abs(tau.eta)    <  2.3) 
    )

    tight_taus = tau[tight_taus_mask]
    tight_sorted_index = ak.argsort(tight_taus.pt,ascending=False)
    tight_taus = tight_taus[tight_sorted_index]

    return tight_taus

def build_htaus(tau, lepton):
    #print(dir(tau))
    #print(tau.__dict__)
    
    base_selection = (
        (tau.pt         > 20.0) & 
        (np.abs(tau.eta)< 2.3) &
        (np.abs(tau.dz)< 0.2) &
        (tau.decayMode != 5) & 
        (tau.decayMode != 6) &
        (tau.idDeepTau2017v2p1VSe >= 2) &
        (tau.idDeepTau2017v2p1VSmu >= 1) &
        (tau.idDeepTau2017v2p1VSjet >= 64)
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
   
    return tau[base_selection & ~overlap_leptons]

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
    # EWK corrections process has to be define before hand, it has to change when we move to dask
    def __init__(self, era: str ='2018', ewk_process_name=None, run_period: str = ''): 
        self._era = era
        if 'APV' in self._era:
            self._isAPV = True
            self._era = re.findall(r'\d+', self._era)[0] 
            #print(f"[YACINE DEBUG] era={self._era} APV={self._isAPV}")
        else:
            self._isAPV = False

        
        jec_tag = ''
        jer_tag = ''
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
            else:
                print('error')
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
            else:
                print('error')
        
        self.btag_wp = 'L'
        self.jetPU_wp = 'M'
        self.tauIDvsjet_wp = 'VTight' #Medium is working
        self.tauIDvse_wp = 'VVLoose'
        self.tauIDvsmu_wp = 'VLoose'
        self.zmass = 91.1873 # GeV 
        self._btag = BTVCorrector(era=self._era, wp=self.btag_wp, isAPV=self._isAPV)
        self._jmeu = JMEUncertainty(jec_tag, jer_tag, era=self._era, is_mc=(len(run_period)==0))
        self._purw = pileup_weights(era=self._era)
        self._leSF = LeptonScaleFactors(era=self._era, isAPV=self._isAPV)
        self._jpSF = jetPUScaleFactors(era=self._era, wp=self.jetPU_wp, isAPV=self._isAPV)
        self._tauID= tauIDScaleFactors(era=self._era, vsjet_wp=self.tauIDvsjet_wp,vse_wp=self.tauIDvse_wp, vsmu_wp=self.tauIDvsmu_wp, isAPV=self._isAPV)
        self._dd   = DataDrivenEventReweight(era=self._era)
        
        _data_path = 'qawa/data'
        _data_path = os.path.join(os.path.dirname(__file__), '../data')
        self._json = {
            '2018': LumiMask(f'{_data_path}/json/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt'),
            '2017': LumiMask(f'{_data_path}/json/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
            '2016': LumiMask(f'{_data_path}/json/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
        }
        with open(f'{_data_path}/{self._era}-trigger-rules.yaml') as ftrig:
            self._triggers = yaml.load(ftrig, Loader=yaml.FullLoader)
            
        with open(f'{_data_path}/eft-names.dat') as eft_file:
            self._eftnames = [n.strip() for n in eft_file.readlines()]

        with uproot.open(f'{_data_path}/trigger_sf/histo_triggerEff_sel0_{self._era}.root') as _fn:
            _hvalue = np.dstack([_fn[_hn].values() for _hn in _fn.keys()] + [np.ones((7,7))])
            _herror = np.dstack([np.sqrt(_fn[_hn].variances()) for _hn in _fn.keys()] + [np.zeros((7,7))])
            self.trig_sf_map = np.stack([_hvalue, _herror], axis=-1)
        
        self.ewk_process_name = ewk_process_name
        if self.ewk_process_name is not None:
            self.ewk_corr = ewk_corrector(process=ewk_process_name)

        self.build_histos = lambda: {
            'met_pt_fv_vs_reco': hist.Hist(
                hist.axis.StrCategory([], name="channel",    growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(100, 0, 800,     name="met_pt_fv", label=r"$p_{T}^{miss}$ (GeV, truth)"),
                hist.axis.Regular(100, 0, 800,  name="met_pt",    label=r"$p_{T}^{miss}$ (GeV, reco)"),
                hist.storage.Weight()
            ),
            'tau_pt_fv_vs_reco': hist.Hist(
                hist.axis.StrCategory([], name="channel",    growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(100, 0, 800,     name="tau_pt_fv", label=r"$p_{T}^{tau}$ (GeV, truth)"),
                hist.axis.Regular(100, 0, 800,  name="tau_pt",    label=r"$p_{T}^{tau}$ (GeV, reco)"),
                hist.storage.Weight()
            ),
            'dilep_pt_fv_vs_reco': hist.Hist(
                hist.axis.StrCategory([], name="channel",    growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(60, 0, 600,     name="dilep_pt_fv", label=r"$p_{T}^{\ell\ell}$ (GeV, truth)"),
                hist.axis.Regular(60, 0, 600,  name="dilep_pt",    label=r"$p_{T}^{\ell\ell}$ (GeV, reco)"),
                hist.storage.Weight()
            ),
            'dilep_m_fv_vs_reco': hist.Hist(
                hist.axis.StrCategory([], name="channel",    growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(60, 0, 120,     name="dilep_m_fv", label=r"$M_{\ell\ell}$ (GeV, truth)"),
                hist.axis.Regular(60, 0, 120,  name="dilep_m",    label=r"$M_{\ell\ell}$ (GeV, reco)"),
                hist.storage.Weight()
            ),
            'dilep_tau_met_hadron_mt_fv_vs_reco': hist.Hist(
                hist.axis.StrCategory([], name="channel",    growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(150, 0, 1500,     name="dilep_tau_met_hadron_mt_fv", label=r"$M_{T}^{WZ}$ (GeV, truth)"),
                hist.axis.Regular(150, 0, 1500,  name="dilep_tau_met_hadron_mt",    label=r"$M_{T}^{WZ}$ (GeV, reco)"),
                hist.storage.Weight()
            ),
	        'dilep_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="dilep_pt", label=r"$p_{T}^{\ell\ell}$ (GeV, reco)"),
                hist.storage.Weight()
            ), 
            'dilep_pt_fv': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="dilep_pt_fv", label=r"$p_{T}^{\ell\ell}$ (GeV, truth)"),
                hist.storage.Weight()
            ), 
	        'dilep_m': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 120, name="dilep_m", label=r"$M_{\ell\ell}$ (GeV, reco)"),
                hist.storage.Weight()
            ), 
            'dilep_m_fv': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 120, name="dilep_m_fv", label=r"$M_{\ell\ell}$ (GeV, truth)"),
                hist.storage.Weight()
            ), 
            'met_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="met_pt", label=r"$p_{T}^{miss}$ (GeV, reco)"),
                hist.storage.Weight()
            ), 
            'met_pt_fv': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="met_pt_fv", label=r"$p_{T}^{miss}$ (GeV, truth)"),
                hist.storage.Weight()
            ),
            'tau_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="tau_pt", label=r"$p_{T}^{tau}$ (GeV, reco)"),
                hist.storage.Weight()
            ),
            'tau_pt_fv': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="tau_pt_fv", label=r"$p_{T}^{tau}$ (GeV, truth)"),
                hist.storage.Weight()
            ),
            'dilep_tau_met_hadron_mt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(150, 0, 1500, name="dilep_tau_met_hadron_mt", label=r"$M_{T}^{WZ}$ (GeV, reco)"),
                hist.storage.Weight()
            ),
            'dilep_tau_met_hadron_mt_fv': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(150, 0, 1500, name="dilep_tau_met_hadron_mt_fv", label=r"$M_{T}^{WZ}$ (GeV, truth)"),
                hist.storage.Weight()
            ),
            'met_uncertainty': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, 100, name="met_uncertainty", label=r"$p_{T}^{miss} uncertainty$ (GeV)"),
                hist.storage.Weight()
            ),
            'met_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="met_phi", label=r"$\phi^{miss}$"),
                hist.storage.Weight()
            ),
            'njets': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="njets", label=r"$N_{jet}$ ($p_{T}>30$ GeV)"),
                hist.storage.Weight()
            ), 
            'ngood_jets_fv': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="ngood_jets_fv", label=r"$N_{jet}$ ($p_{T}>30$ GeV)"),
                hist.storage.Weight()
            ), 
            'ngood_jets_fv_noclean': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="ngood_jets_fv_noclean", label=r"$N_{jet}$ ($p_{T}>30$ GeV)"),
                hist.storage.Weight()
            ), 
            'dphi_met_ll': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="dphi_met_ll", label=r"$\Delta \phi(\ell\ell,p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            "leading_lep_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="leading_lep_pt", label="$p_T^{l_1}$ (GeV)"),
                hist.storage.Weight()
            ), 
            "trailing_lep_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="trailing_lep_pt", label=r"$p_T^{l_2}$ (GeV)"),
                hist.storage.Weight()
            ),
            "leading_lep_pt_fv": hist.Hist(
                hist.axis.StrCategory([], name="channel", growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(50, 30, 530, name="leading_lep_pt_fv", label=r"$p_T^{\ell_1}$ (GeV, truth)"),
                hist.storage.Weight()
            ),
            "trailing_lep_pt_fv": hist.Hist(
                hist.axis.StrCategory([], name="channel", growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True),
                hist.axis.Regular(50, 30, 530, name="trailing_lep_pt_fv", label=r"$p_T^{\ell_2}$ (GeV, truth)"),
                hist.storage.Weight()
            ),
            "third_lep_pt": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 30, 530, name="third_lep_pt", label=r"$p_T^{l_2}$ (GeV)"),
                hist.storage.Weight()
            ),
            "leading_lep_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="leading_lep_eta", label=r"$\eta(l_1)$"),
                hist.storage.Weight()
            ), 
            "trailing_lep_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="trailing_lep_eta", label=r"$\eta(l_2)$"),
                hist.storage.Weight()
            ),
            "third_lep_eta": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="third_lep_eta", label=r"$\eta(l_3)$"),
                hist.storage.Weight()
            ),
            "leading_lep_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="leading_lep_phi", label=r"$\phi^(l_1)$"),
                hist.storage.Weight()
            ), 
            "trailing_lep_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="trailing_lep_phi", label=r"$\phi^(l_2)$"),
                hist.storage.Weight()
            ),
            "third_lep_phi": hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -np.pi, np.pi, name="third_lep_phi", label=r"$\phi^(l_3)$"),
                hist.storage.Weight()
            ),
            
        }

    
    def _add_trigger_sf(self, weights, lead_lep, subl_lep):
        mask_BB = ak.fill_none((lead_lep.eta <= 1.5) & (subl_lep.eta <= 1.5), False)
        mask_EB = ak.fill_none((lead_lep.eta >= 1.5) & (subl_lep.eta <= 1.5), False)
        mask_BE = ak.fill_none((lead_lep.eta <= 1.5) & (subl_lep.eta >= 1.5), False)
        mask_EE = ak.fill_none((lead_lep.eta >= 1.5) & (subl_lep.eta >= 1.5), False)

        mask_mm = ak.fill_none((np.abs(lead_lep.pdgId)==13) & (np.abs(subl_lep.pdgId)==13), False)
        mask_ee = ak.fill_none((np.abs(lead_lep.pdgId)==11) & (np.abs(subl_lep.pdgId)==11), False)
       
        mask_me = (~mask_mm & ~mask_ee) & (np.abs(lead_lep.pdgId) == 13)
        mask_em = (~mask_mm & ~mask_ee) & (np.abs(lead_lep.pdgId) == 11)

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
        dataset = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        selection = PackedSelection()
        weights = Weights(len(event), storeIndividual=True)
        
        histos = self.build_histos()
        
        if is_data:
            selection.add('lumimask', self._json[self._era](event.run, event.luminosityBlock))
            selection.add('triggers', trigger_rules(event, self._triggers, self._era))
        else:
            selection.add('lumimask', np.ones(len(event), dtype='bool'))
            selection.add('triggers', np.ones(len(event), dtype='bool'))
        
        # MET filters
        if "2016" in self._era:
            selection.add(
                'metfilter',
                #event.Flag.METFilters &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter &
                event.Flag.goodVertices &
                event.Flag.eeBadScFilter &
                event.Flag.BadPFMuonFilter &
                event.Flag.BadPFMuonDzFilter
            )
        else:
            selection.add(
                'metfilter',
                event.Flag.goodVertices &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter & 
                event.Flag.BadPFMuonFilter &
                event.Flag.BadPFMuonDzFilter &
                event.Flag.eeBadScFilter &
                event.Flag.ecalBadCalibFilter 
            )

        
        tight_lep, loose_lep = build_leptons(
            event.Muon,
            event.Electron
        )
        
        had_taus = build_htaus(event.Tau, tight_lep)
        nhtaus_lep = ak.num(had_taus)
        lead_tau = ak.firsts(had_taus)
        tau_pt = lead_tau.pt

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

        tight_tau_fv = build_htaus_fv(gen_taus)
        lead_tau_fv = ak.firsts(tight_tau_fv)
        ntight_tau_fv = ak.num(tight_tau_fv)
        tau_pt_fv = lead_tau_fv.pt


        pairs_fv     = ak.combinations(tight_lep_fv, 2, axis=1, fields=['l1','l2'])
        pairs_fv    = ak.pad_none(pairs_fv, 1, axis=1)

        lead_lep_fv = pairs_fv.l1[:, 0]
        subl_lep_fv = pairs_fv.l2[:, 0]
        
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
        
        dijet_mass_fv = (lead_jet_fv + subl_jet_fv).mass
        dijet_deta_fv = np.abs(lead_jet_fv.eta - subl_jet_fv.eta)
        
        had_taus = build_htaus(event.Tau, tight_lep)
        
        ntight_lep = ak.num(tight_lep)
        nloose_lep = ak.num(loose_lep)
        nhtaus_lep = ak.num(had_taus)
        
        jets = event.Jet
        overlap_leptons = ak.any(
            jets.metric_table(tight_lep) <= 0.4,
            axis=2
        )
        overlap_taus = ak.any(
            jets.metric_table(had_taus) <= 0.4,
            axis=2
        )

        jet_mask = (
            ~overlap_leptons & 
            ~overlap_taus &
            (jets.pt>30.0) & 
            (np.abs(jets.eta) < 4.7) & 
            (jets.jetId >= 6)& # tight JetID 7(2016) and 6(2017/8)
            ((jets.puId >= 6) | (jets.puId == 3) | (jets.pt >= 50)) # medium puID https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL 3,7 for 16and 16APV; 6,7 for 17,18
        )

        jet_mask_PUID = (
            ~overlap_leptons & 
            ~overlap_taus &
            (jets.pt>30.0) & 
            (np.abs(jets.eta) < 4.7) & 
            (jets.jetId >= 6) # tight JetID 7(2016) and 6(2017/18)
        )
        
        jet_btag = (
                (event.Jet.btagDeepFlavB > btag_id(
                    self.btag_wp, 
                    self._era + 'APV' if self._isAPV else self._era
                )) &
                (np.abs(jets.eta)<2.5)
        )
        good_jets = jets[jet_mask]
        good_bjet = jets[jet_btag & jet_mask & (np.abs(jets.eta)<2.5)]
        good_jets_forBtag = jets[jet_mask & (np.abs(jets.eta) < (2.4 if "2016" in self._era else 2.5))]
        pu_good_jets = jets[jet_mask_PUID]

            
        ngood_jets  = ak.num(good_jets)
        ngood_bjets = ak.num(good_bjet)
        
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
        
        lead_lep = ak.firsts(ak.where(dilep.l1.pt >  dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        subl_lep = ak.firsts(ak.where(dilep.l1.pt <= dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        
        dilep_p4 = (lead_lep + subl_lep)
        dilep_m  = dilep_p4.mass
        dilep_pt = dilep_p4.pt
        
        third_lep = ak.firsts(extra_lep, axis=1)

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
        # reco_met_pt = event.MET.pt
        # reco_met_phi = event.MET.phi

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
        dilep_et_met = np.sqrt(reco_met_pt**2 + self.zmass**2)
        
        # new version
        dilep_mt = ak.where(  
                ntight_lep==3,
                np.sqrt((dilep_et_ll + dilep_et_met)**2 - ((dilep_p4.pvec + emu_met.pvec).pt)**2),
                np.sqrt((dilep_et_ll + dilep_et_met)**2 - ((dilep_p4.pvec +  p4_met.pvec).pt)**2))
        
        dilep_dphi_met  = dilep_p4.delta_phi(p4_met)
        
        sorted_indices = np.argsort(-good_jets.pt)
        good_jets = good_jets[sorted_indices]
        lead_jet = ak.firsts(good_jets)
        subl_jet = ak.firsts(good_jets[lead_jet.delta_r(good_jets)>0.01])
        third_jet = ak.firsts(good_jets[(lead_jet.delta_r(good_jets)>0.01) & (subl_jet.delta_r(good_jets)>0.01)])

        #Transverse WZ mass system 
        # Building 4 vector for tranverse mass calculation
        # .t is synonym for energy but there is a bug when we add option types of arrays of two leptons
        dilep_tau_met_p4 = dilep_p4 + lead_tau + p4_met
        dilep_tau_met_p4_fv = dilep_p4_fv + lead_tau_fv + p4_met_fv
        mT_WZ_square = ((dilep_tau_met_p4.t**2) - (dilep_tau_met_p4.pz**2))
        mT_WZ = np.sqrt(np.maximum(0, mT_WZ_square))


        dilep_tau_met_hadron_mt = np.sqrt((transverse_energy(lead_lep) + transverse_energy(subl_lep) + transverse_energy(lead_tau) + p4_met.pt) ** 2 - dilep_tau_met_p4.pt**2)
        dilep_tau_met_hadron_mt_fv = np.sqrt((transverse_energy(lead_lep_fv) + transverse_energy(subl_lep_fv) + transverse_energy(lead_tau_fv) + p4_met_fv.pt) ** 2 - dilep_tau_met_p4_fv.pt**2)

        inv_m_WZ = (dilep_tau_met_p4).mass

        #HT, scalar sum of jet pt, and HTl, HT + lepton pt
        HT = ak.sum(good_jets.pt, axis=1)
        HTl = HT + lead_lep.pt + subl_lep.pt + lead_tau.pt

        # ST, scalar sum of all object pts
        ST = HTl + p4_met.pt



        # 2jet and vbs related variables
        dijet_mass = (lead_jet + subl_jet).mass
        dijet_deta = np.abs(lead_jet.eta - subl_jet.eta)
        event['dijet_mass'] = dijet_mass
        event['dijet_deta'] = dijet_deta 

        leadbjet_score = lead_jet.btagDeepFlavB
        sublbjet_score = subl_jet.btagDeepFlavB
        event['leadbjet_score'] = ak.fill_none(leadbjet_score,np.nan)
        event['sublbjet_score'] = ak.fill_none(sublbjet_score,np.nan)
        
        
        # define basic selection
        selection.add(
            "require-ossf",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        selection.add(
            "require-ossf-fv",
            (ntight_lep_fv==2) &
            (ak.firsts(tight_lep_fv).pt>25) &
            ak.fill_none((lead_lep_fv.pdgId + subl_lep_fv.pdgId)==0, False)
        )
        selection.add(
            "require-2lep",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        

        selection.add('met_pt', ak.fill_none((reco_met_pt > 30), False))
        selection.add('met_pt_fv', ak.fill_none((p4_met_fv.pt > 30), False)) # loose met cut in fv from 30 to 20 and then to 10
        selection.add('low_met_pt_fv', ak.fill_none((p4_met_fv.pt < 20) & (p4_met_fv.pt > 0), False))
        selection.add('low_met_pt', ak.fill_none((reco_met_pt < 20) & (reco_met_pt > 0), False))
        # selection.add('medium_ptmiss', ak.fill_none((reco_met_pt > 70), False))
        selection.add('dilep_m'   , ak.fill_none(np.abs(dilep_m - self.zmass) < 15, False))
        selection.add('dilep_m_fv', ak.fill_none(np.abs(dilep_m_fv - self.zmass) < 15, False))
        selection.add('dilep_pt_fv', ak.fill_none(dilep_pt_fv>30, False))
        selection.add('dilep_pt', ak.fill_none(dilep_pt > 30, False))

        selection.add("dilep_dphi_met", ak.fill_none(np.abs(dilep_dphi_met)>1.0, False))
        # jet demography

        selection.add('0njets' , ngood_jets  == 0 )
        selection.add('0njets_fv' , ngood_jets_fv  == 0 )
        selection.add('1njets' , ngood_jets  <= 1 )
        selection.add('1njets_fv' , ngood_jets_fv  <= 1 )
        selection.add('1njets_only' , ngood_jets  == 1 )
        selection.add('1njets_only_fv' , ngood_jets_fv  == 1 )
        
        selection.add('1nhtaus', nhtaus_lep  == 1 )
        selection.add('1nhtaus_fv', ntight_tau_fv  == 1 )
        
        
        # selection.add('dijet_deta_fv', ak.fill_none(dijet_deta_fv > 2.5, False))
        # selection.add('dijet_deta', ak.fill_none(dijet_deta > 2.5, False))
        # selection.add('dijet_mass_400_fv' , ak.fill_none(dijet_mass_fv >  400, False))
        # selection.add('dijet_mass_400' , ak.fill_none(dijet_mass >  400, False))


        # Define all variables for the GNN
        event['ngood_jets_fv'  ] = ak.fill_none(ngood_jets_fv,-99)
        event['dilep_tau_met_hadron_mt'  ] = ak.fill_none(dilep_tau_met_hadron_mt,-99)
        event['dilep_tau_met_hadron_mt_fv'  ] = ak.fill_none(dilep_tau_met_hadron_mt_fv,-99)
        event['met_sigma'  ] = ak.fill_none(ptmiss_sigma,-99)
        event['met_uncertainty'  ] = ak.fill_none(ptmiss_unc,-99)
        event['met_pt'  ] = ak.fill_none(reco_met_pt,-99)
        event['met_pt_fv'  ] = ak.fill_none(p4_met_fv.pt,-99)
        event['met_phi' ] = ak.fill_none(reco_met_phi,-99)
        
        event['dilep_mt'] = ak.fill_none(dilep_mt,-99)
        event['dilep_m'] = ak.fill_none(dilep_m,-99)
        event['dilep_pt'] = ak.fill_none(dilep_pt,-99)
        event['dilep_m_fv'] = ak.fill_none(dilep_m_fv,-99)
        event['dilep_pt_fv'] = ak.fill_none(dilep_pt_fv,-99)
        event['njets'   ] = ak.fill_none(ngood_jets,-99)
        event['dphi_met_ll'] = ak.fill_none(dilep_dphi_met,-99)
        

        event['leading_lep_pt'  ] = ak.fill_none(lead_lep.pt,-99)
        event['leading_lep_eta' ] = ak.fill_none(lead_lep.eta,-99)
        event['leading_lep_phi' ] = ak.fill_none(lead_lep.phi,-99)
        event['trailing_lep_pt' ] = ak.fill_none(subl_lep.pt,-99)
        event['trailing_lep_eta'] = ak.fill_none(subl_lep.eta,-99)
        event['trailing_lep_phi'] = ak.fill_none(subl_lep.phi,-99)
        event['leading_lep_pt_fv'] = ak.fill_none(lead_lep_fv.pt,-99)
        event['trailing_lep_pt_fv'] = ak.fill_none(subl_lep_fv.pt,-99)
        event['tau_pt_fv'] = ak.fill_none(tau_pt_fv,-99)
        event['tau_pt'] = ak.fill_none(tau_pt,-99)
        event['third_lep_pt'  ] = ak.fill_none(third_lep.pt,-99)
        event['third_lep_eta' ] = ak.fill_none(third_lep.eta,-99)
        event['third_lep_phi' ] = ak.fill_none(third_lep.phi,-99)

        event['lead_jet_pt'  ] = ak.fill_none(lead_jet.pt,-99)
        event['lead_jet_eta' ] = ak.fill_none(lead_jet.eta,-99)
        event['lead_jet_phi' ] = ak.fill_none(lead_jet.phi,-99)
        event['trail_jet_pt' ] = ak.fill_none(subl_jet.pt,-99)
        event['trail_jet_eta'] = ak.fill_none(subl_jet.eta,-99)
        event['trail_jet_phi'] = ak.fill_none(subl_jet.phi,-99)
        event['lead_jet_pt_fv'] = ak.fill_none(lead_jet_fv.pt,-99)
        event['trail_jet_pt_fv'] = ak.fill_none(subl_jet_fv.pt,-99)

        # Now adding weights
        if not is_data:
            weights.add('genweight', event.genWeight)
            # self._btag.append_btag_sf(good_jets_forBtag, weights)#good_jets_forBtag
            self._jpSF.append_jetPU_sf(pu_good_jets, weights)

            self._purw.append_pileup_weight(weights, event.Pileup.nTrueInt)
            self._tauID.append_tauID_sf(had_taus, weights)
            self._add_trigger_sf(weights, lead_lep, subl_lep)

    
            weights.add (
                    'LeptonSF', 
                    lead_lep.SF*subl_lep.SF, 
                    lead_lep.SF_up*subl_lep.SF_up, 
                    lead_lep.SF_down*subl_lep.SF_down
            )
            _ones = np.ones(len(weights.weight()))
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
                    print("WARNING: QCD scale variation type not recongnised ... ")

            # 2017 Prefiring correction weight
            if 'L1PreFiringWeight' in event.fields:
                weights.add("prefiring_weight", event.L1PreFiringWeight.Nom, event.L1PreFiringWeight.Dn, event.L1PreFiringWeight.Up)

        # else:
        #     # If systematic variations are needed, they must be manually inserted here to give different DD estimates; they should be picked up later for histos.
        #     weights.add("datadriven_DDDYNominal", _ones, self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, systematic="nominal"), self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, systematic="nominal"))  #added nominal value twice to avoid getting 1/up for the nominaldown
        #     weights.add("datadriven_DDDY",_ones, self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, "DDDYUp"), self._dd.estimate_dd_DY(ngood_jets, tau_pt_loose, "DDDYDown"))
            

        # selections
        #'require-ossf','1nhtaus', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '1njets', 'met_pt'
        #'require-ossf-fv', '1nhtaus_fv', 'dilep_m_fv', 'dilep_pt_fv', '1njets_fv', 'met_pt_fv'

        common_sel = ['triggers', 'lumimask', 'metfilter']

        reco_SR0 = common_sel + [
            'require-ossf','1nhtaus', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '0njets', 'met_pt',
        ]

        truth_FV0 = [
            'require-ossf-fv', '1nhtaus_fv', 'dilep_m_fv', 'dilep_pt_fv', '0njets_fv', 'met_pt_fv'
        ]
        reco_SR1 = common_sel + [
            'require-ossf','1nhtaus', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '1njets_only', 'met_pt',
        ]

        truth_FV1 = [
            'require-ossf-fv', '1nhtaus_fv', 'dilep_m_fv', 'dilep_pt_fv', '1njets_only_fv', 'met_pt_fv'
        ]
        reco_SR01 = common_sel + [
           'require-ossf', '1nhtaus','dilep_m', 'dilep_pt', 'dilep_dphi_met', '1njets', 'met_pt'
        ]

        truth_FV01 = [
            'require-ossf-fv', '1nhtaus_fv', 'dilep_m_fv', 'dilep_pt_fv', '1njets_fv', 'met_pt_fv'
        ]
        def bits_to_kwargs(bits):
            """
            Convert list like ['a','b','~c'] into kwargs dict
            for PackedSelection.require, e.g. {'a':True, 'b':True, 'c':False}
            """
            kw = {}
            for b in bits:
                neg = b.startswith("~")
                name = b[1:] if neg else b
                kw[name] = not neg
            return kw
        # 而不是对 triggers/lumimask/metfilter 单独 ~。
        pass_reco0  = selection.require(**bits_to_kwargs(reco_SR0))
        pass_truth0 = selection.require(**bits_to_kwargs(truth_FV0))
        pass_reco1  = selection.require(**bits_to_kwargs(reco_SR1))
        pass_truth1 = selection.require(**bits_to_kwargs(truth_FV1))
        pass_reco01  = selection.require(**bits_to_kwargs(reco_SR01))
        pass_truth01 = selection.require(**bits_to_kwargs(truth_FV01))

        channels = {
            "both-SR0":       pass_reco0 & pass_truth0,
            "truth-only0": ~pass_reco0 & pass_truth0,
            "reco-only0": pass_reco0 & ~pass_truth0,
            "none0":       ~pass_reco0 & ~pass_truth0,
            "reco0":       pass_reco0,
            "truth0":      pass_truth0,
            "both-SR1":       pass_reco1 & pass_truth1,
            "truth-only1": ~pass_reco1 & pass_truth1,
            "reco-only1": pass_reco1 & ~pass_truth1,
            "none1":       ~pass_reco1 & ~pass_truth1,
            "reco1":       pass_reco1,
            "truth1":      pass_truth1,
            "both-SR01":       pass_reco01 & pass_truth01,
            "truth-only01": ~pass_reco01 & pass_truth01,
            "reco-only01": pass_reco01 & ~pass_truth01,
            "none01":       ~pass_reco01 & ~pass_truth01,
            "reco01":       pass_reco01,
            "truth01":      pass_truth01,
        }


            
        def _format_variable(variable, cut):
            if cut is None:
                vv = ak.to_numpy(ak.fill_none(variable, -99))
                if np.isnan(np.any(vv)):
                    print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable, -99))
            else:
                vv = ak.to_numpy(ak.fill_none(variable[cut], -99))
                if np.isnan(np.any(vv)):
                    print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable[cut], -99))
        
        def _histogram_filler(ch, syst, var, _weight=None):
            # sel_ = channels[ch]
            cut = channels[ch]
            # sel_args_ = {
            #     s.replace('~',''): (False if '~' in s else True) for s in sel_ if var not in s
            # }
            # cut =  selection.require(**sel_args_)
            systname = 'nominal' if syst is None else syst
            if _weight is None: 
                if syst in weights.variations:
                    weight = weights.weight(modifier=syst)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * _weight[cut]
          
            vv = ak.to_numpy(ak.fill_none(weight, -99))
            if np.isnan(np.any(vv)):
                print(f" - {syst} weight nan/inf:", vv[np.isnan(vv)], vv[np.isinf(vv)])
            histos[var].fill(
                **{
                    "channel": ch, 
                    "systematic": systname, 
                    var: _format_variable(event[var], cut), 
                    "weight": ak.nan_to_num(weight,nan=1.0, posinf=1.0, neginf=1.0)
                        #ak.ones_like(weight)
                        #ak.nan_to_num(weight,nan=1.0, posinf=1.0, neginf=1.0)
                }
            )
            
        def _hist2d_filler(ch, syst, xvar, yvar, _weight=None, hist_key='dijet2d_fv_vs_reco'):
            cut = channels[ch]
            systname = 'nominal' if syst is None else syst
            if _weight is None:
                weight = weights.weight(modifier=syst)[cut] if (syst in weights.variations) else weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * _weight[cut]

            histos[hist_key].fill(
                channel=ch,
                systematic=systname,
                **{
                    xvar: _format_variable(event[xvar], cut),
                    yvar: _format_variable(event[yvar], cut),
                    "weight": ak.nan_to_num(weight) #ak.ones_like(weight)  # 或用真实权重：ak.nan_to_num(weight)
                }
            )

        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]
        for ch in channels:
            for sys in systematics:
                for var in (
                    'met_pt',
                    'njets',
                    'ngood_jets_fv',
                    'met_pt_fv',
                    'dilep_m',
                    'dilep_m_fv',
                    'dilep_pt',
                    'dilep_pt_fv',
                    'tau_pt',
                    'tau_pt_fv',
                    'dilep_tau_met_hadron_mt',
                    'dilep_tau_met_hadron_mt_fv',

                ):
                    _histogram_filler(ch, sys, var)
                _hist2d_filler(ch, None, 'met_pt_fv', 'met_pt',hist_key='met_pt_fv_vs_reco')
                _hist2d_filler(ch, None, 'tau_pt_fv', 'tau_pt',hist_key='tau_pt_fv_vs_reco')
                _hist2d_filler(ch, None, 'dilep_pt_fv', 'dilep_pt',hist_key='dilep_pt_fv_vs_reco')
                _hist2d_filler(ch, None, 'dilep_m_fv', 'dilep_m',hist_key='dilep_m_fv_vs_reco')
                _hist2d_filler(ch, None, 'dilep_tau_met_hadron_mt_fv', 'dilep_tau_met_hadron_mt',hist_key='dilep_tau_met_hadron_mt_fv_vs_reco')


                
        return {dataset: histos}
        
    def process(self, event):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        

        # JES/JER corrections
        rho = event.fixedGridRhoFastjetAll
        cache = event.caches[0]

        
        raw_met = event.RawMET
        met_to_correct = event.MET
       
        jets = self._jmeu.corrected_jets_L123_JER(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
        jets_to_correct_met = self._jmeu.corrected_jets_L123_noJER(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
        met = self._jmeu.corrected_met(met_to_correct, jets, event.fixedGridRhoFastjetAll, event.caches[0]) # we are adding fully smeared L123 jets
        

        event = ak.with_field(event, event.Jet, 'OrigJet')
        event = ak.with_field(event, event.MET, 'OrigMET')
        event = ak.with_field(event, jets, 'Jet')
        event = ak.with_field(event, jets_to_correct_met, 'JetforMET')
        event = ak.with_field(event, met, 'MET')
        

        run = event.run 
        npv = event.PV.npvs
        
        met = met_phi_xy_correction(
            event.MET, run, npv, 
            is_mc=not is_data, 
            era=self._era
        )
        event = ak.with_field(event, met, 'MET')


        if is_data:
            # Apply rochester_correction
            muon = event.Muon 
            muonEnUp=event.Muon
            muonEnDown=event.Muon
            muon_pt,muon_pt_roccorUp,muon_pt_roccorDown=rochester_correction(is_data).apply_rochester_correction (muon)

            muon['pt'] = muon_pt
            muonEnUp['pt'] = muon_pt_roccorUp
            muonEnDown['pt'] = muon_pt_roccorDown
            event = ak.with_field(event, muon, 'Muon')
               
            return self.process_shift(event, None)
        
		
        # Adding scale factors to Muon and Electron fields
        muon = event.Muon 
        electron = event.Electron
        muonSF_nom, muonSF_up, muonSF_down = self._leSF.muonSF(muon)
        elecSF_nom, elecSF_up, elecSF_down = self._leSF.electronSF(electron)
        
        muon['SF'] = muonSF_nom
        muon['SF_up'] = muonSF_up
        muon['SF_down'] = muonSF_down

        electron['SF'] = elecSF_nom
        electron['SF_up'] = elecSF_up
        electron['SF_down'] = elecSF_down

        event = ak.with_field(event, muon, 'Muon')
        event = ak.with_field(event, electron, 'Electron')

        # Apply rochester_correction
        muon=event.Muon
        muonEnUp=event.Muon
        muonEnDown=event.Muon
        muon_pt,muon_pt_roccorUp,muon_pt_roccorDown=rochester_correction(is_data).apply_rochester_correction (muon)
        
        muon['pt'] = muon_pt
        muonEnUp['pt'] = muon_pt_roccorUp
        muonEnDown['pt'] = muon_pt_roccorDown
        event = ak.with_field(event, muon, 'Muon')
        
        # Electron corrections
        electronEnUp=event.Electron
        electronEnDown=event.Electron

        electronEnUp  ['pt'] = event.Electron['pt'] + event.Electron.energyErr/np.cosh(event.Electron.eta)
        electronEnDown['pt'] = event.Electron['pt'] - event.Electron.energyErr/np.cosh(event.Electron.eta)	


        tau = event.Tau[(event.Tau.decayMode != 5) & (event.Tau.decayMode != 6)]
        tauEnUp = event.Tau[(event.Tau.decayMode != 5) & (event.Tau.decayMode != 6)]
        tauEnDown = event.Tau[(event.Tau.decayMode != 5) & (event.Tau.decayMode != 6)]
        tau_pt,tau_pt_EnUp,tau_pt_EnDown,tau_mass,tau_mass_EnUp,tau_mass_EnDown=self._tauID.tau_energy_scale_correction(tau)

        tau['pt'] = tau_pt
        tau['mass'] = tau_mass
        tauEnUp['pt'] = tau_pt_EnUp
        tauEnUp['mass'] = tau_mass_EnUp
        tauEnDown['pt'] = tau_pt_EnDown
        tauEnDown['mass'] = tau_mass_EnDown
        event = ak.with_field(event, tau, 'Tau')
	
        
        # define all the shifts
        shifts = [
            # Jets
           ({"Jet": event.Jet                        , "MET": event.MET                         }, None                  ),
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
           ({"Jet": jets.JER.up                      , "MET": met.JER.up                        }, "JERUp"               ),
           ({"Jet": jets.JER.down                    , "MET": met.JER.down                      }, "JERDown"             ),
           ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.up      }, "UESUp"               ),
           ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.down    }, "UESDown"             ), 
           # year dependent systematics
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

           
           # Electrons + MET shift (FIXME: shift to be added)
           ({"Electron": electronEnUp  }, "ElectronEnUp"  ),
           ({"Electron": electronEnDown}, "ElectronEnDown"),
           # Muon + MET shifts
           ({"Muon": muonEnUp  }, "MuonRocUp"),
           ({"Muon": muonEnDown}, "MuonRocDown"),
           # Tau + MET shifts
           ({"Tau": tauEnUp  }, "TauEnUp"),
           ({"Tau": tauEnDown}, "TauEnDown"),
        ]

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

