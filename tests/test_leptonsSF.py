import awkward as ak
import numpy as np
import hist
import matplotlib.pyplot as plt
from functools import partial
from qawa.jsoncorrections import CorrectionlibHandler
from qawa.leptonsSF import LeptonScaleFactors
from qawa.muonSS import muon_pt_scare
from qawa.egammaSS import EGM_scale_and_smear, EGM_scale_and_smear_v9
from qawa.tauSF import tauIDScaleFactors, tau_energy_scale
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection

def build_htaus(tau, lepton, nanoAODversion="v9", analysisID="inc-WZ-vtight", tauIDvse_wp="VTight", tauIDvsmu_wp="Tight"):
    tau_e_branch = None
    tau_e_subid = None
    tau_mu_subid = None
    tau_mu_branch = None
    tau_j_branch = None
    if nanoAODversion in [f"v{V}" for V in range(12)]:
        tau_e_branch = tau.idDeepTau2017v2p1VSe
        tau_mu_branch = tau.idDeepTau2017v2p1VSmu
        tau_j_branch = tau.idDeepTau2017v2p1VSjet
    elif nanoAODversion in ["v15"]:
        tau_e_branch = tau.idDeepTau2018v2p5VSe
        tau_mu_branch = tau.idDeepTau2018v2p5VSmu
        tau_j_branch = tau.idDeepTau2018v2p5VSjet
    else:
        raise NotImplementedError

    match tauIDvse_wp:
        case "VTight":
            tau_e_subid = (tau_e_branch >= 64)
        case "Tight":
            tau_e_subid = (tau_e_branch >= 32)
        case "Medium":
            tau_e_subid = (tau_e_branch >= 16)
        case "Loose":
            tau_e_subid = (tau_e_branch >= 8)
        case "VLoose":
            tau_e_subid = (tau_e_branch >= 4)
        case "VVLoose":
            tau_e_subid = (tau_e_branch >= 2)
        case "VVVLoose":
            tau_e_subid = (tau_e_branch >= 1)
    match tauIDvsmu_wp:
        case "Tight":
            tau_mu_subid = (tau_mu_branch >= 8)
        case "Medium":
            tau_mu_subid = (tau_mu_branch >= 4)
        case "Loose":
            tau_mu_subid = (tau_mu_branch >= 2)
        case "VLoose":
            tau_mu_subid = (tau_mu_branch >= 1)
        case _:
            raise NotImplementedError
    if analysisID == "inc-WZ-VTight":
        tau_ID = tau_e_subid & tau_mu_subid & (tau_j_branch >= 64)
    elif analysisID == "inc-WZ-Tight":
        tau_ID = tau_e_subid & tau_mu_subid & (tau_j_branch >= 32)
    elif analysisID == "inc-WZ-Loose":
        tau_ID = tau_e_subid & tau_mu_subid & (tau_j_branch >= 8)
    else:
        raise NotImplementedError
    base_selection = (
        (tau.pt         > 20 ) &
        (np.abs(tau.eta)< 2.3 ) &
        (np.abs(tau.dz)< 0.2 ) &
        (tau.decayMode != 5   ) &
        (tau.decayMode != 6   ) &
        tau_ID
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )

    return tau[base_selection & ~overlap_leptons]

clibhandler9 = CorrectionlibHandler(era="2018", subera=None, analysis="inc-WZ", nanoAODversion="v9", cvmfs_head="/Users/nmangane/Downloads/cvmfs")
lsf = LeptonScaleFactors(era="2018", clibhandler=clibhandler9, electronID="wp90iso", muonID="Tight", muonISO="TightRelIso")
leg = LeptonScaleFactors(era="2018", clibhandler=None, electronID="wp90iso", muonID="Tight", muonISO="TightRelIso")
e15 = NanoEventsFactory.from_root({"WZ_TuneCP5_13p6TeV_pythia8.root": "Events"}, schemaclass=NanoAODSchema).events()
d15 = NanoEventsFactory.from_root({"Run2024G_Muon0_PromptReco-v1.root": "Events"}, schemaclass=NanoAODSchema).events()
e9 = NanoEventsFactory.from_root({"WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8.root": "Events"}, schemaclass=NanoAODSchema).events()
d9 = NanoEventsFactory.from_root({"DoubleMuon_UL2018_NanoAODv9.root": "Events"}, schemaclass=NanoAODSchema).events()

clibhandler15 = CorrectionlibHandler(era="2024", subera=None, analysis="inc-WZ", nanoAODversion="v15", cvmfs_head="/Users/nmangane/Downloads/cvmfs")
def _add_trigger_sf(weights, lead_lep, subl_lep, clibhandler=None):
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
    else:
        raise ValueError

e15twomu = e15[ak.num(e15.Muon, axis=1) == 2]; e15twomu_lead = e15twomu.Muon[:, 0]; e15twomu_sublead = e15twomu.Muon[:, 1]
e15twomu_weights = Weights(len(e15twomu), storeIndividual=True)
e15twoe = e15[ak.num(e15.Electron, axis=1) == 2]; e15twoe_lead = e15twoe.Electron[:, 0]; e15twoe_sublead = e15twoe.Electron[:, 1]
e15twoe_weights = Weights(len(e15twoe), storeIndividual=True)
breakpoint()
_add_trigger_sf(e15twomu_weights, e15twomu_lead, e15twomu_sublead, clibhandler=clibhandler15)
_add_trigger_sf(e15twoe_weights, e15twoe_lead, e15twoe_sublead, clibhandler=clibhandler15)
# e15twoemu = e15[(ak.num(e15.Electron, axis=1) == 1) & (ak.num(e15.Muon, axis=1) == 1)]


from qawa.common import met_phi_xy_correction
px_old = e9.MET.px
py_old = e9.MET.py
# met_func = met_phi_xy_correction(e9.MET, e9.run, e9.PV.npvs, is_mc=True, era="2018")
mnew = met_phi_xy_correction(e9.MET, e9.run, e9.PV.npvs, is_mc=True, era="2018")
dnew = met_phi_xy_correction(d9.MET, d9.run, d9.PV.npvs, is_mc=False, era="2018")
# mnew = met_phi_xy_correction(e9.MET, e9.run, e9.PV.npvs, is_mc=True, era="2018", jet_type="AK4CHS", clibhandler=clibhandler9)
# dnew = met_phi_xy_correction(d9.MET, d9.run, d9.PV.npvs, is_mc=False, era="2018", jet_type="AK4CHS", clibhandler=clibhandler9)

sumWeights = ak.sum(e9.genWeight)
sumEvents = ak.num(d9.event, axis=0)
h = hist.Hist.new.Regular(100, -3.14, 3.14, name="phi", circular=True).Regular(100, 0, 500, name="pt").StrCategory([], name="source", growth=True).Weight()
h.fill(pt=e9.MET.pt, phi=e9.MET.phi, source="MC_uncorrected", weight=e9.genWeight/sumWeights)
h.fill(pt=mnew.pt, phi=mnew.phi, source="MC_corrected", weight=e9.genWeight/sumWeights)
h.fill(pt=d9.MET.pt, phi=d9.MET.phi, source="DATA_uncorrected", weight=1/sumEvents)
h.fill(pt=dnew.pt, phi=dnew.phi, source="DATA_corrected", weight=1/sumEvents)

# fig, ax = plt.subplots(1, 1)

# h[{"pt": hist.tag.Slicer()[::sum], "phi": hist.tag.Slicer()[::hist.rebin(5)]}].plot1d(overlay="source")
# h[{"phi": hist.tag.Slicer()[::sum], "pt": hist.tag.Slicer()[::hist.rebin(5)]}].plot1d(overlay="source")
# fig.show()
# ax.legend()

print("px:", mnew.px,
      "py:", mnew.py,
      )
print("pxdiff:", ak.mean(np.abs(mnew.px - px_old)),
      "pydiff:", ak.mean(np.abs(mnew.py - py_old)),
      )
print("closure mnew:", ak.mean(np.sqrt(mnew.px**2 + mnew.py**2) - mnew.pt),
      "closure orig:", ak.mean(np.sqrt(e9.MET.px**2 + e9.MET.py**2) - e9.MET.pt),
      )

# Add main correction and smearing
muon_pt_scare(None, e15, unc_type=None, is_correction=True, clibhandler=clibhandler15)
muon = e15.Muon
muon.add_systematic("scale_m", "UpDownSystematic", "pt", partial(muon_pt_scare, events=e15, unc_type="Scale", is_correction=False, clibhandler=clibhandler15))
muon.add_systematic("res_m", "UpDownSystematic", "pt", partial(muon_pt_scare, events=e15, unc_type="Resolution", is_correction=False, clibhandler=clibhandler15))
EGM_scale_and_smear(None, d15, is_correction=True, unc_type=None, restriction=None, is_electron=True, clibhandler=clibhandler15)
EGM_scale_and_smear(None, e15, is_correction=True, unc_type=None, restriction=None, is_electron=True, clibhandler=clibhandler15)
electron = e15.Electron

electron.add_systematic("scale_e", "UpDownMultiSystematic", ("pt", "energyErr"), partial(EGM_scale_and_smear, events=e15, is_correction=False, unc_type="Scale", restriction=None, is_electron=True, clibhandler=clibhandler15))
electron.add_systematic("res_e", "UpDownMultiSystematic", ("pt", "energyErr"), partial(EGM_scale_and_smear, events=e15, is_correction=False, unc_type="Smear", restriction=None, is_electron=True, clibhandler=clibhandler15))

# EGM_scale_and_smear_v9(None, e9, is_correction=True, unc_type=None, restriction=None, is_electron=True) # Not needed, this is a reminder
electron9 = e9.Electron
electron9.add_systematic("scale_e", "UpDownMultiSystematic", ("pt", "energyErr"), partial(EGM_scale_and_smear_v9, events=e9,is_correction=False, unc_type="Scale", restriction=None, is_electron=True))
electron9.add_systematic("res_e", "UpDownMultiSystematic", ("pt", "energyErr"), partial(EGM_scale_and_smear_v9, events=e9, is_correction=False, unc_type="Smear", restriction=None, is_electron=True))

# Tau energy scale

tau9leg = e9.Tau[(e9.Tau.decayMode != 5) & (e9.Tau.decayMode != 6)]
_tauID= tauIDScaleFactors(era="2018", vsjet_wp="VTight", vse_wp="Tight", vsmu_wp="Tight",
                                       isAPV=False, isEE=False, isBPix=False, clibhandler=None)
tau9leg_pt,tau9leg_pt_EnUp,tau9leg_pt_EnDown,tau9leg_mass,tau9leg_mass_EnUp,tau9leg_mass_EnDown=_tauID.tau_energy_scale_correction(tau9leg)
try:
    print("Default v9")
    tau_energy_scale(None, e9, "DeepTau2017v2p1", wp_VSjet="VTight", wp_VSe="VTight", unc_type=None, is_correction=True, clibhandler=clibhandler9)
except IndexError:
    print("dm2IndexErrorWorkaround=True v9")
    tau_energy_scale(None, e9, "DeepTau2017v2p1", wp_VSjet="VTight", wp_VSe="VTight", unc_type=None, is_correction=True, dm2IndexErrorWorkaround=True, clibhandler=clibhandler9)
tau9 = e9.Tau
tau9.add_systematic("scale_t", "UpDownMultiSystematic", ("pt", "mass"),
                    partial(tau_energy_scale, events=e9, tagger="DeepTau2017v2p1", wp_VSjet="VTight", wp_VSe="VTight", unc_type="Scale", is_correction=False, clibhandler=clibhandler9))

try:
    print("Default v15")
    tau_energy_scale(None, e15, "DeepTau2018v2p5", wp_VSjet="VTight", wp_VSe="Tight", unc_type=None, is_correction=True, clibhandler=clibhandler15)
except IndexError:
    print("dm2IndexErrorWorkaround=True v15")
    tau_energy_scale(None, e15, "DeepTau2018v2p5", wp_VSjet="VTight", wp_VSe="Tight", unc_type=None, is_correction=True, dm2IndexErrorWorkaround=True, clibhandler=clibhandler15)
tau15 = e15.Tau
tau15.add_systematic("scale_t", "UpDownMultiSystematic", ("pt", "mass"),
                      partial(tau_energy_scale, events=e15, tagger="DeepTau2018v2p5", wp_VSjet="VTight", wp_VSe="Tight", unc_type="Scale", is_correction=False, dm2IndexErrorWorkaround=True, clibhandler=clibhandler15))

had_taus = build_htaus(e9.Tau, e9.Muon, nanoAODversion="v9", analysisID="inc-WZ-VTight", tauIDvse_wp='VTight', tauIDvsmu_wp='Tight')
from coffea.jetmet_tools.CorrectedMETFactory import corrected_polar_met

def propagate_scale_to_met(sink, events, met, shifted_collection, unc_type=None, is_correction=True):
    if is_correction:
        raise NotImplementedError("propagation of scale shifts to MET for central variation has not been implemented, consider implications of doing so carefully including prop to JES/JER/UES variations")
    else:
        systematic = []
        match unc_type:
            case "Scale":
                systematic = [name for name in shifted_collection.systematics.fields if name.startswith("scale")]
            case "Resolution":
                systematic = [name for name in shifted_collection.systematics.fields if name.startswith("res")]
            case _:
                raise NotImplementedError
        if len(systematic) > 0:
            systematic = systematic[0]
        else:
            raise ValueError(f"No matching systematic found in set for unc_type={unc_type}: {shifted_collection.systematics.fields}")

        shifted_up = getattr(shifted_collection.systematics, systematic).up
        shifted_down = getattr(shifted_collection.systematics, systematic).down
        up = corrected_polar_met(
            met.pt, met.phi, shifted_up.pt, shifted_up.phi, shifted_collection.pt, positive=None, dx=None, dy=None
        )
        down = corrected_polar_met(
            met.pt, met.phi, shifted_down.pt, shifted_down.phi, shifted_collection.pt, positive=None, dx=None, dy=None
        )
        return ak.zip({
            "pt": np.concatenate((up[:, None].pt, down[:, None].pt), axis=1),
            "phi": np.concatenate((up[:, None].phi, down[:, None].phi), axis=1),
        }, depth_limit=1)

met = e9.MET
met.add_systematic("scale_t", "UpDownMultiSystematic", ("pt", "phi"),
                   partial(propagate_scale_to_met, events=e9, met=e9.MET, shifted_collection=tau9, unc_type="Scale", is_correction=False)
                   )
# met.add_systematic("scale_m", "UpDownMultiSystematic", ("pt", "phi"),
#                    partial(propagate_scale_to_met, events=e9, met=e9.MET, shifted_collection=, unc_type="Scale", is_correction=False)
#                    )
met.add_systematic("scale_e", "UpDownMultiSystematic", ("pt", "phi"),
                   partial(propagate_scale_to_met, events=e9, met=e9.MET, shifted_collection=electron9, unc_type="Scale", is_correction=False)
                   )
met.add_systematic("res_e", "UpDownMultiSystematic", ("pt", "phi"),
                   partial(propagate_scale_to_met, events=e9, met=e9.MET, shifted_collection=electron9, unc_type="Resolution", is_correction=False)
                   )
# finalboss = propagate_scale_to_met(None, e9, e9.MET, tau9, unc_type="Scale", is_correction=False)
