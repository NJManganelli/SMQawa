import os.path
from coffea.lookup_tools import extractor, dense_lookup
import awkward as ak
import numpy as np
import uproot 


class LeptonScaleFactors:
    def __init__(self,  era:str='2018', isAPV:bool=False, isEE:bool=False, isBPix:bool=False, clibhandler=None, electronID:str = None, muonID:str = None, muonISO:str = None):
        self.muonID = muonID
        self.muonISO = muonISO
        self.electronID = electronID
        if clibhandler is not None:
            self.__init_clib__(era=era, isAPV=isAPV, isEE=isEE, isBPix=isBPix, clibhandler=clibhandler)
        else:
            self.__init_legacy__(era=era, isAPV=isAPV, isEE=isEE, isBPix=isBPix)
            self.clib_electrons = None
            self.clib_muons = None

    def __init_clib__(self, era:str='2018', isAPV:bool=False, isEE:bool=False, isBPix:bool=False, clibhandler=None):
        self._erasubera = era
        if isAPV:
            self._era = era + 'APV'
        else:
            self._era = era 
        if isAPV:
            self._erasubera += "_APV"
        elif isEE:
            self._erasubera += "_EE"
        elif isBPix:
            self._erasubera += "_BPix"
        self._erasuberanousc = self._erasubera.replace("_", "")
        self.clib_ver = clibhandler._ver
        self.clib_electrons = clibhandler.getCorrectionSet("electron")
        self.clib_muons = clibhandler.getCorrectionSet("muon_Z")
        # signature sig0_eta, sig1_pt, sig2_syst
        # main systematics: "nominal","systup","systdown"
        self.cset_m_id_key = f"NUM_{self.muonID}ID_DEN_TrackerMuons"
        self.cset_m_iso_key = f"NUM_{self.muonISO}_DEN_{self.muonID}ID{'andIPCut' if self.muonID in ['Tight', 'HighPt', 'TrkHighPt'] else ''}"
        self.cset_m_iso_key_fallback = f"NUM_{self.muonISO}_DEN_{self.muonID}ID" # Because there's a fucking exception for no good god damned reason
        self.cset_e_key = None
        self.sig0_year = None
        match self._erasubera:
            case "2024":
                # MUON
                if not hasattr(self, "muonIDxISOAvailable"):
                    self.mask_m_pt = 10.0
                    self.muonIDxISOAvailable = [
                        ("Loose", "LooseMiniIso"), ("Loose", "LoosePFIso"),
                        ("Medium", "LoosePFIso"), ("Medium", "MediumPFIso"), ("Medium", "TightPFIso"),
                        ("Medium", "LooseMiniIso"), ("Medium", "MediumMiniIso"), ("Medium", "TightMiniIso"),
                        ("Medium", "promptMVA_WP64"), # not really ISO?, but for simplicity...
                        ("MediumPrompt", "LoosePFIso"), ("MediumPrompt", "TightPFIso"),
                        ("Tight", "LoosePFIso"), ("Tight", "TightPFIso"),
                        ("Tight", "TightMiniIso"),
                        ("Tight", "promptMVA_WP64"), # not really ISO?, but for simplicity...
                        ("HighPt", "LooseRelTkIso"), ("HighPt", "TightRelTkIso"),
                        ("TrkHighPt", "LooseRelTkIso"), ("TrkHighPt", "TightRelTkIso"),
                        ]
                    if (self.muonID, self.muonISO) not in self.muonIDxISOAvailable and self.muonID not in ["Soft"]:
                        assert False, f"Available muon ID x ISO pairs: {self.muonIDxISOAvailable}"
                    available = list(self.clib_muons.keys())
                    assert self.cset_m_id_key in available, f"{self.cset_m_id_key} not valid key, set {[x for x in self.clib_muons.keys() if 'id' in x.lower()]}"
                    if self.cset_m_iso_key not in available and self.cset_m_iso_key_fallback in available:
                        self.cset_m_iso_key = self.cset_m_iso_key_fallback
                    assert self.cset_m_iso_key in available, f"{self.cset_m_iso_key} not valid key, set {[x for x in self.clib_muons.keys() if ('iso' in x.lower() or 'promptmva' in x.lower())]}"
                # ELECTRON
                if not hasattr(self, "electronIDAvailable"):
                    self.electronIDAvailable = ["Veto", "Loose", "Medium", "Tight", "wp80iso", "wp90iso", "wp80noiso", "wp90noiso", "PromptMVA-Medium",  "PromptMVA-Tight"]
                    assert self.electronID in self.electronIDAvailable, f"Available electron IDs: {self.electronIDAvailable}"
                self.cset_e_key = "Electron-ID-SF"
                self.sig0_year = "2024Prompt"
                self.reco_e_high = "RecoAbove75"
                self.reco_e_mid = "Reco20to75"
                self.reco_e_low = "RecoBelow20"
            case "2016_APV" | "2016" | "2017" | "2018":
                # MUON
                if not hasattr(self, "muonIDxISOAvailable"):
                    self.mask_m_pt = 15.0
                    # if self.clib_ver in ["v9", "v15"]: # no specialization needed, they appear to be the same with the latest tested tag, but different iso naming convention from RunIII
                    self.muonIDxISOAvailable = [
                        ("Loose", "LooseRelIso"),
                        ("Medium", "LooseRelIso"), ("Medium", "TightRelIso"),
                        ("MediumPrompt", "LooseRelIso"), ("MediumPrompt", "TightRelIso"),
                        ("Tight", "LooseRelIso"), ("Tight", "TightRelIso"),
                        ("HighPt", "LooseRelTkIso"), ("HighPt", "TightRelTkIso"),
                        ("TrkHighPt", "LooseRelTkIso"), ("TrkHighPt", "TightRelTkIso"),
                    ]
                    if (self.muonID, self.muonISO) not in self.muonIDxISOAvailable and self.muonID not in ["Soft"]:
                        assert False, f"{(self.muonID, self.muonISO)} not in available muon ID x ISO pairs: {self.muonIDxISOAvailable}"
                    assert self.cset_m_id_key in self.clib_muons.keys(), f"{self.cset_m_id_key} not valid key, set {[x for x in self.clib_muons.keys() if 'id' in x.lower()]}"
                    assert self.cset_m_iso_key in self.clib_muons.keys(), f"{self.cset_m_iso_key} not valid key, set {[x for x in self.clib_muons.keys() if ('iso' in x.lower() or 'promptmva' in x.lower())]}"
                # ELECTRON
                if not hasattr(self, "electronIDAvailable"):
                    self.electronIDAvailable = ["Veto", "Loose", "Medium", "Tight", "wp80iso", "wp90iso", "wp80noiso", "wp90noiso"]
                    assert self.electronID in self.electronIDAvailable, f"Available electron IDs: {self.electronIDAvailable}"
                    self.reco_e_high = "RecoAbove75" "RecoAbove20" "Reco20to75" "RecoBelow20"
                self.cset_e_key = "UL-Electron-ID-SF"
                if self._erasubera == "2016_APV":
                    self.sig0_year = "2016preVFP"
                elif self._erasubera == "2016":
                    self.sig0_year = "2016postVFP"
                else:
                    self.sig0_year = self._erasubera
                self.reco_e_high = "RecoAbove20"
                self.reco_e_mid = "RecoAbove20"
                self.reco_e_low = "RecoBelow20"
            case _:
                raise NotImplementedError(f"Unmatched era + subera in LeptonSF implementation: {self._erasubera}")
        
    def __init_legacy__(self, era:str='2018', isAPV:bool=False, isEE:bool=False, isBPix:bool=False):
        raise NotImplementedError("Legacy path deprecated due to evolution in clib")

    def muonSF(self, muons: ak.Array):
        if self.clib_muons:
            return self.muonSF_clib(muons)
        else:
            raise NotImplementedError("Legacy path deprecated due to evolution in clib")

    def muonSF_clib(self, muons: ak.Array):
        maskEtaPt = (np.abs(muons.eta) <= 2.4) & (muons.pt >= self.mask_m_pt)
        # Alternative systematics are additive (i.e. must be added to nominal to bound it): "AltBkg", "AltSig","massBin","massRange","stat","syst","tagIso"
        id_sfs = {sig2_name: ak.where(maskEtaPt, 
                                      # correction
                                       self.clib_muons[self.cset_m_id_key].evaluate(ak.mask(muons.eta, maskEtaPt, valid_when=True),
                                                                                    ak.mask(muons.pt,  maskEtaPt, valid_when=True),
                                                                                    sig2_syst
                                                                                    ),
                                      ak.ones_like(muons.pt)
                                      ) for sig2_name, sig2_syst in {"nominal": "nominal", "up": "systup", "down": "systdown"}.items()
                  }
        id_sfs = {k: ak.fill_none(v, 1.0, axis=1) for k, v in id_sfs.items()}
        iso_sfs = {sig2_name: ak.where(maskEtaPt, 
                                       # correction
                                       self.clib_muons[self.cset_m_iso_key].evaluate(ak.mask(muons.eta, maskEtaPt, valid_when=True),
                                                                                     ak.mask(muons.pt,  maskEtaPt, valid_when=True),
                                                                                     sig2_syst
                                                                                     ),
                                       ak.ones_like(muons.pt)
                                       ) for sig2_name, sig2_syst in {"nominal": "nominal", "up": "systup", "down": "systdown"}.items()
                  }
        iso_sfs = {k: ak.fill_none(v, 1.0, axis=1) for k, v in iso_sfs.items()}
        _ones = ak.ones_like(muons.pt)
        return {
            "eff_m_idNominal":   id_sfs["nominal"],
            "eff_m_idUp":        id_sfs["up"],
            "eff_m_idDown":      id_sfs["down"],
            "eff_m_isoNominal":  iso_sfs["nominal"],
            "eff_m_isoUp":       iso_sfs["up"],
            "eff_m_isoDown":     iso_sfs["down"],
            "eff_e_recoNominal": _ones,
            "eff_e_recoUp":      _ones,
            "eff_e_recoDown":    _ones,
            "eff_e_idNominal":   _ones,
            "eff_e_idUp":        _ones,
            "eff_e_idDown":      _ones,
        }

    def electronSF(self, electrons: ak.Array):
        if self.clib_electrons:
            return self.electronSF_clib(electrons)
        else:
            raise NotImplementedError("Legacy path deprecated due to evolution in clib")

    def electronSF_clib(self, electrons: ak.Array):
        # 0	{ name: "year", type: "string", description: "year/scenario: example, 2017, 2022FG etc" }
        # 1	{ name: "ValType", type: "string", description: "sf/ sfup / sfdown / effData / effMC / err_stat /err_statData / err_statMC / err_syst (sfup = sf + syst, sfdown = sf - syst) " }
        # 2	{ name: "WorkingPoint", type: "string", description: "Working Point of choice : Loose, Medium etc." }
        # 3	{ name: "eta", type: "real", description: "supercluster eta" }
        # 4	{ name: "pt", type: "real", description: "electron pT" }
        # Non-optimized, but contained, implementation for the electron SFs
        # self.cset_e_key
        # self.sig0_year
        # sig1_valtype should cycle between sf, sfup (sf + syst) and sfdown (sf - syst), or additional types like eff, errStat
        # sig2 must vary between 3 Reco maps for reconstruction efficiency and a fixed WP for the ID itself: self.electronID
        sig3_electrons_sceta = electrons.eta + electrons.deltaEtaSC
        sig4_electrons_pt = electrons.pt
        maskAbove75 = sig4_electrons_pt >= 75.0
        mask20to75 = (sig4_electrons_pt >= 20.0) & (sig4_electrons_pt < 75.0)
        maskBelow20 = (sig4_electrons_pt >= 10.0) & (sig4_electrons_pt < 20.0)
        reco_sfs = {sig1_name: ak.where(maskAbove75,
                                        # correction RecoAbove75
                                        self.clib_electrons[self.cset_e_key].evaluate(self.sig0_year,
                                                                                      sig1_syst,
                                                                                      self.reco_e_high,
                                                                                      sig3_electrons_sceta,
                                                                                      ak.mask(sig4_electrons_pt, maskAbove75, valid_when=True)
                                                                                      ),
                                        ak.where(mask20to75,
                                                 # correction between 20 and 75
                                                 self.clib_electrons[self.cset_e_key].evaluate(self.sig0_year,
                                                                                               sig1_syst,
                                                                                               self.reco_e_mid,
                                                                                               sig3_electrons_sceta,
                                                                                               ak.mask(sig4_electrons_pt, mask20to75, valid_when=True)
                                                                                               ),
                                                 ak.where(maskBelow20,
                                                          # correction from 10 to 20
                                                          self.clib_electrons[self.cset_e_key].evaluate(self.sig0_year,
                                                                                                        sig1_syst,
                                                                                                        self.reco_e_low,
                                                                                                        sig3_electrons_sceta,
                                                                                                        ak.mask(sig4_electrons_pt, maskBelow20, valid_when=True)
                                                                                                        ),
                                                          # fallback: 1.0
                                                          ak.ones_like(sig4_electrons_pt)
                                                          )
                                                 )
                                        ) for sig1_name, sig1_syst in {"nominal": "sf", "up": "sfup", "down": "sfdown"}.items()
                    }
        reco_sfs = {k: ak.fill_none(v, 1.0, axis=1) for k, v in reco_sfs.items()}
        maskAbove10 = sig4_electrons_pt >= 10.0
        id_sfs = {sig1_name: ak.where(maskAbove10,
                                      # correction for valid ID range above 10 GeV
                                      self.clib_electrons[self.cset_e_key].evaluate(self.sig0_year,
                                                                                    sig1_syst,
                                                                                    self.electronID,
                                                                                    sig3_electrons_sceta,
                                                                                    ak.mask(sig4_electrons_pt, maskAbove10, valid_when=True)
                                                                                    ),
                                      # fallback: 1.0
                                      ak.ones_like(sig4_electrons_pt)
                                      ) for sig1_name, sig1_syst in {"nominal": "sf", "up": "sfup", "down": "sfdown"}.items()
                  }
        id_sfs = {k: ak.fill_none(v, 1.0, axis=1) for k, v in id_sfs.items()}
        _ones = ak.ones_like(electrons.pt)
        return {
            "eff_m_idNominal":   _ones,
            "eff_m_idUp":        _ones,
            "eff_m_idDown":      _ones,
            "eff_m_isoNominal":  _ones,
            "eff_m_isoUp":       _ones,
            "eff_m_isoDown":     _ones,
            "eff_e_recoNominal": reco_sfs["nominal"],
            "eff_e_recoUp":      reco_sfs["up"],
            "eff_e_recoDown":    reco_sfs["down"],
            "eff_e_idNominal":   id_sfs["nominal"],
            "eff_e_idUp":        id_sfs["up"],
            "eff_e_idDown":      id_sfs["down"],
        }

    def append_lepton_sf(self, lead_lep, subl_lep, weights):
        for key in ["eff_m_id", "eff_m_iso", "eff_e_reco", "eff_e_id"]:
            weights.add(
                key,
                getattr(lead_lep, f"{key}Nominal") * getattr(subl_lep, f"{key}Nominal"),
                getattr(lead_lep, f"{key}Up") * getattr(subl_lep, f"{key}Up"),
                getattr(lead_lep, f"{key}Down") * getattr(subl_lep, f"{key}Down"),
            )
