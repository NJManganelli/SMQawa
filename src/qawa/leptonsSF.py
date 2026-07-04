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
        extLepSF = extractor()

        _data_path = os.path.join(os.path.dirname(__file__), 'data/LeptonSF')

        muonSelectionTag     = "TightWP_" + self._erasuberanousc
        electronSelectionTag = "GPMVA90_" + self._erasuberanousc

        if muonSelectionTag=="TightWP_2016":
            mu_f=["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.root"]
            mu_h = ["NUM_TightID_DEN_TrackerMuons_abseta_pt",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
            mu_f_up = ["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID_variations.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO_variations.root"]
            mu_h_up = ['NUM_LooseID_DEN_TrackerMuons_abseta_pt_up;1',
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_up;1"]
            mu_f_down = ["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ID_variations.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO_variations.root"]
            mu_h_down = ["NUM_LooseID_DEN_TrackerMuons_abseta_pt_down;1",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_down"]
            
        elif muonSelectionTag=="TightWP_2016APV":
            mu_f=["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.root"]
            mu_h = ["NUM_TightID_DEN_TrackerMuons_abseta_pt",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
            mu_f_up = ["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID_variations.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO_variations.root"]
            mu_h_up = ['NUM_LooseID_DEN_TrackerMuons_abseta_pt_up;1',
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_up;1"]
            mu_f_down = ["2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID_variations.root",
                  "2016/Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO_variations.root"]
            mu_h_down = ["NUM_LooseID_DEN_TrackerMuons_abseta_pt_down;1",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_down"]
        elif muonSelectionTag=="TightWP_2017":
            mu_f=["2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.root",
                  "2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.root"]
            mu_h = ["NUM_TightID_DEN_TrackerMuons_abseta_pt",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
            mu_f_up = ["2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID_variations.root",
                  "2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO_variations.root"]
            mu_h_up = ['NUM_LooseID_DEN_TrackerMuons_abseta_pt_up;1',
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_up;1"]
            mu_f_down = ["2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ID_variations.root",
                  "2017/Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO_variations.root"]
            mu_h_down = ["NUM_LooseID_DEN_TrackerMuons_abseta_pt_down;1",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_down"]
        elif muonSelectionTag=="TightWP_2018":
            mu_f=["2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.root",
                  "2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.root"]
            mu_h = ["NUM_TightID_DEN_TrackerMuons_abseta_pt",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt"]
            mu_f_up = ["2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID_variations.root",
                  "2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO_variations.root"]
            mu_h_up = ['NUM_LooseID_DEN_TrackerMuons_abseta_pt_up;1',
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_up;1"]
            mu_f_down = ["2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ID_variations.root",
                  "2018/Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO_variations.root"]
            mu_h_down = ["NUM_LooseID_DEN_TrackerMuons_abseta_pt_down;1",
                    "NUM_TightRelIso_DEN_TightIDandIPCut_abseta_pt_down"]
        else:
            print (f'wrong era: {muonSelectionTag}')


        if electronSelectionTag=="GPMVA90_2016":
            el_f = ["2016/egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D.root",
                    "2016/electron_RecoSF_UL2016postVFP.root"]
            el_h = ["EGamma_SF2D","EGamma_SF2D"]
            el_f_up = ["2016/egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D_variations.root",
                    "2016/electron_RecoSF_UL2016postVFP.root"]
            el_h_up = ["EGamma_SF2D_up","EGamma_SF2D"]
            el_f_down = ["2016/egammaEffi.txt_Ele_wp90iso_postVFP_EGM2D_variations.root",
                    "2016/electron_RecoSF_UL2016postVFP.root"]
            el_h_down = ["EGamma_SF2D_down","EGamma_SF2D"]
        elif electronSelectionTag=="GPMVA90_2016APV":
            el_f = ["2016/egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D.root",
                    "2016/electron_RecoSF_UL2016preVFP.root"]
            el_h = ["EGamma_SF2D","EGamma_SF2D"]
            el_f_up = ["2016/egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D_variations.root",
                    "2016/electron_RecoSF_UL2016preVFP.root"]
            el_h_up = ["EGamma_SF2D_up","EGamma_SF2D"]
            el_f_down = ["2016/egammaEffi.txt_Ele_wp90iso_preVFP_EGM2D_variations.root",
                    "2016/electron_RecoSF_UL2016preVFP.root"]
            el_h_down = ["EGamma_SF2D_down","EGamma_SF2D"]
        elif electronSelectionTag=="GPMVA90_2017":
            el_f = ["2017/egammaEffi.txt_EGM2D_MVA90iso_UL17.root",
                    "2017/electron_RecoSF_UL2017.root"]
            el_h = ["EGamma_SF2D","EGamma_SF2D"]
            el_f_up = ["2017/egammaEffi.txt_EGM2D_MVA90iso_UL17_variations.root",
                    "2017/electron_RecoSF_UL2017.root"]
            el_h_up = ["EGamma_SF2D_up","EGamma_SF2D"]
            el_f_down = ["2017/egammaEffi.txt_EGM2D_MVA90iso_UL17_variations.root",
                    "2017/electron_RecoSF_UL2017.root"]
            el_h_down = ["EGamma_SF2D_down","EGamma_SF2D"]
        elif electronSelectionTag=="GPMVA90_2018":
            el_f = ["2018/egammaEffi.txt_Ele_wp90iso_EGM2D.root",
                    "2018/electron_RecoSF_UL2018.root"]
            el_h = ["EGamma_SF2D","EGamma_SF2D"]
            el_f_up = ["2018/egammaEffi.txt_Ele_wp90iso_EGM2D_variations.root",
                    "2018/electron_RecoSF_UL2018.root"]
            el_h_up = ["EGamma_SF2D_up","EGamma_SF2D"]
            el_f_down = ["2018/egammaEffi.txt_Ele_wp90iso_EGM2D_variations.root",
                    "2018/electron_RecoSF_UL2018.root"]
            el_h_down = ["EGamma_SF2D_down","EGamma_SF2D"]
        else:
            print (f'wrong era: {electronSelectionTag}')
            
        self.maps_nom = {}
        self.maps_up = {}
        self.maps_down = {}
        for i, _fname in enumerate(mu_f):
            with uproot.open(f"{_data_path}/{_fname}") as _fn:
                _hist = _fn[mu_h[i]].to_hist()
                _hnom = _hist.values()
                tag = f"MuonIso{era}" if "ISO" in _fname else ""
                tag = f"MuonId{era}"  if "ID" in _fname else tag
                self.maps_nom[tag] = dense_lookup.dense_lookup(_hnom,[ax.edges for ax in _hist.axes])
            with uproot.open(f"{_data_path}/{mu_f_up[i]}") as _fn:
                _hist = _fn[mu_h_up[i]].to_hist()
                _herr_up = _hist.values()
                tag = f"MuonIso{era}" if "ISO" in _fname else ""
                tag = f"MuonId{era}"  if "ID" in _fname else tag
                self.maps_up[tag] = dense_lookup.dense_lookup(_herr_up,[ax.edges for ax in _hist.axes])
            with uproot.open(f"{_data_path}/{mu_f_down[i]}") as _fn:
                _hist = _fn[mu_h_down[i]].to_hist()
                _herr_down = _hist.values()
                tag = f"MuonIso{era}" if "ISO" in _fname else ""
                tag = f"MuonId{era}"  if "ID" in _fname else tag
                self.maps_down[tag] = dense_lookup.dense_lookup(_herr_down,[ax.edges for ax in _hist.axes])
                
        for i, _fname in enumerate(el_f):
            with uproot.open(f"{_data_path}/{_fname}") as _fn:
                _hist = _fn[el_h[i]].to_hist()
                _hnom = _hist.values()
                tag = f"ElectronIso{era}" if "iso" in _fname else ""
                tag = f"ElectronReco{era}" if "Reco" in _fname else tag
                self.maps_nom[tag] = dense_lookup.dense_lookup(_hnom,[ax.edges for ax in _hist.axes])
            with uproot.open(f"{_data_path}/{el_f_up[i]}") as _fn:
                _hist = _fn[el_h_up[i]].to_hist()
                _herr_up = _hist.values()
                tag = f"ElectronIso{era}" if "iso" in _fname else ""
                tag = f"ElectronReco{era}" if "Reco" in _fname else tag
                self.maps_up[tag] = dense_lookup.dense_lookup(_herr_up,[ax.edges for ax in _hist.axes])
            with uproot.open(f"{_data_path}/{el_f_down[i]}") as _fn:
                _hist = _fn[el_h_down[i]].to_hist()
                _herr_down = _hist.values()
                tag = f"ElectronIso{era}" if "iso" in _fname else ""
                tag = f"ElectronReco{era}" if "Reco" in _fname else tag
                self.maps_down[tag] = dense_lookup.dense_lookup(_herr_down,[ax.edges for ax in _hist.axes])

    def muonSF(self, muons: ak.Array, variations=True):
        if self.clib_muons:
            return self.muonSF_clib(muons, variations=variations)
        else:
            # Hardcoded TightID and
            assert self.muonID == "Tight", "Legacy muonSF method is hardcoded for TightID [ID]"
            assert self.muonISO == "TightRelIso", "Legacy muonSF method is hardcoded for Tight RelIso [ISO]"
            return self.muonSF_legacy(muons, variations=variations)

    def muonSF_legacy(self, muons: ak.Array, variations=True):
        # sf_nom  = 1.0 # ak.ones_like(muons.pt)
        # sf_up   = 1.0 # ak.ones_like(muons.pt)
        # sf_down = 1.0 # ak.ones_like(muons.pt)

        id_sfs = {}
        iso_sfs = {}
        for n in self.maps_nom.keys():
            if 'Muon' not in n: continue
            _nom = self.maps_nom[n](muons.pt, np.abs(muons.eta))
            if variations:
                _up = self.maps_up[n](muons.pt, np.abs(muons.eta))
                _down = self.maps_down[n](muons.pt, np.abs(muons.eta))
            if "Id" in n:
                id_sfs["nominal"] = _nom
                if variations:
                    id_sfs["up"] = _up
                    id_sfs["down"] = _down
            else:
                iso_sfs["nominal"] = _nom
                if variations:
                    iso_sfs["up"] = _up
                    iso_sfs["down"] = _down
            # sf_nom = sf_nom * _nom
            # sf_up = sf_up * _up
            # sf_down = sf_down * _down

        # return sf_nom, sf_up, sf_down
        _ones = ak.ones_like(muons.pt)
        if not variations:
            _nom = id_sfs["nominal"] * iso_sfs["nominal"]
            # variation fields aliased to nominal (unused when variations=False)
            return {"nominal":        _nom,
                    "eff_m_idUp":     _nom,
                    "eff_m_idDown":   _nom,
                    "eff_m_isoUp":    _nom,
                    "eff_m_isoDown":  _nom,
                    "eff_e_recoUp":   _ones,
                    "eff_e_recoDown": _ones,
                    "eff_e_idUp":     _ones,
                    "eff_e_idDown":   _ones,
                    }
        return {"nominal":        id_sfs["nominal"] * iso_sfs["nominal"],
                "eff_m_idUp":     id_sfs["up"]      * iso_sfs["nominal"],
                "eff_m_idDown":   id_sfs["down"]    * iso_sfs["nominal"],
                "eff_m_isoUp":    id_sfs["nominal"] * iso_sfs["up"],
                "eff_m_isoDown":  id_sfs["nominal"] * iso_sfs["down"],
                "eff_e_recoUp":   _ones,
                "eff_e_recoDown": _ones,
                "eff_e_idUp":     _ones,
                "eff_e_idDown":   _ones,
                }

    def muonSF_clib(self, muons: ak.Array, variations=True):
        maskEtaPt = (np.abs(muons.eta) <= 2.4) & (muons.pt >= self.mask_m_pt)
        # restrict to the nominal systematic when up/down variations are not requested
        _muon_systs = {"nominal": "nominal", "up": "systup", "down": "systdown"} if variations else {"nominal": "nominal"}
        # Alternative systematics are additive (i.e. must be added to nominal to bound it): "AltBkg", "AltSig","massBin","massRange","stat","syst","tagIso"
        id_sfs = {sig2_name: ak.where(maskEtaPt,
                                      # correction
                                       self.clib_muons[self.cset_m_id_key].evaluate(ak.mask(muons.eta, maskEtaPt, valid_when=True),
                                                                                    ak.mask(muons.pt,  maskEtaPt, valid_when=True),
                                                                                    sig2_syst
                                                                                    ),
                                      ak.ones_like(muons.pt)
                                      ) for sig2_name, sig2_syst in _muon_systs.items()
                  }
        id_sfs = {k: ak.fill_none(v, 1.0, axis=1) for k, v in id_sfs.items()}
        iso_sfs = {sig2_name: ak.where(maskEtaPt,
                                       # correction
                                       self.clib_muons[self.cset_m_iso_key].evaluate(ak.mask(muons.eta, maskEtaPt, valid_when=True),
                                                                                     ak.mask(muons.pt,  maskEtaPt, valid_when=True),
                                                                                     sig2_syst
                                                                                     ),
                                       ak.ones_like(muons.pt)
                                       ) for sig2_name, sig2_syst in _muon_systs.items()
                  }
        iso_sfs = {k: ak.fill_none(v, 1.0, axis=1) for k, v in iso_sfs.items()}
        _ones = ak.ones_like(muons.pt)
        if not variations:
            _nom = id_sfs["nominal"] * iso_sfs["nominal"]
            # variation fields aliased to nominal (unused when variations=False)
            return {"nominal":        _nom,
                    "eff_m_idUp":     _nom,
                    "eff_m_idDown":   _nom,
                    "eff_m_isoUp":    _nom,
                    "eff_m_isoDown":  _nom,
                    "eff_e_recoUp":   _ones,
                    "eff_e_recoDown": _ones,
                    "eff_e_idUp":     _ones,
                    "eff_e_idDown":   _ones,
                    }
        return {"nominal":        id_sfs["nominal"] * iso_sfs["nominal"],
                "eff_m_idUp":     id_sfs["up"]      * iso_sfs["nominal"],
                "eff_m_idDown":   id_sfs["down"]    * iso_sfs["nominal"],
                "eff_m_isoUp":    id_sfs["nominal"] * iso_sfs["up"],
                "eff_m_isoDown":  id_sfs["nominal"] * iso_sfs["down"],
                "eff_e_recoUp":   _ones,
                "eff_e_recoDown": _ones,
                "eff_e_idUp":     _ones,
                "eff_e_idDown":   _ones,
                }

    def electronSF(self, electrons: ak.Array, variations=True):
        if self.clib_electrons:
            return self.electronSF_clib(electrons, variations=variations)
        else:
            assert self.electronID == "wp90iso", "Legacy electronSF method is hardcoded for wp90iso [ID+ISO]"
            return self.electronSF_legacy(electrons, variations=variations)

    def electronSF_legacy(self, electrons: ak.Array, variations=True):
        # sf_nom  = 1.0 # ak.ones_like(muons.pt)
        # sf_up   = 1.0 # ak.ones_like(muons.pt)
        # sf_down = 1.0 # ak.ones_like(muons.pt)

        reco_sfs = {}
        id_sfs = {}
        for n in self.maps_nom.keys():
            if 'Electron' not in n: continue
            _nom = self.maps_nom[n](electrons.pt, np.abs(electrons.eta))
            if variations:
                _up = self.maps_up[n](electrons.pt, np.abs(electrons.eta))
                _down = self.maps_down[n](electrons.pt, np.abs(electrons.eta))
            if "Reco" in n:
                assert "nominal" not in reco_sfs.keys()
                reco_sfs["nominal"] = _nom
                if variations:
                    reco_sfs["up"] = _up
                    reco_sfs["down"] = _down
            else:
                assert "nominal" not in id_sfs.keys()
                # "Iso" means nothing in electron MVA IDs, it's a combination ID/ISO SF...
                id_sfs["nominal"] = _nom
                if variations:
                    id_sfs["up"] = _up
                    id_sfs["down"] = _down
            # sf_nom = sf_nom * _nom
            # sf_up = sf_up * _up
            # sf_down = sf_down * _down

        # return sf_nom, sf_up, sf_down
        _ones = ak.ones_like(electrons.pt)
        if not variations:
            _nom = reco_sfs["nominal"] * id_sfs["nominal"]
            # variation fields aliased to nominal/ones (unused when variations=False)
            return {"nominal":        _nom,
                    "eff_m_idUp":     _ones,
                    "eff_m_idDown":   _ones,
                    "eff_m_isoUp":    _ones,
                    "eff_m_isoDown":  _ones,
                    "eff_e_recoUp":   _nom,
                    "eff_e_recoDown": _nom,
                    "eff_e_idUp":     _nom,
                    "eff_e_idDown":   _nom,
                    }
        return {"nominal":        reco_sfs["nominal"] * id_sfs["nominal"],
                "eff_m_idUp":     _ones,
                "eff_m_idDown":   _ones,
                "eff_m_isoUp":    _ones,
                "eff_m_isoDown":  _ones,
                "eff_e_recoUp":   reco_sfs["up"]      * id_sfs["nominal"],
                "eff_e_recoDown": reco_sfs["down"]    * id_sfs["nominal"],
                "eff_e_idUp":     reco_sfs["nominal"] * id_sfs["up"],
                "eff_e_idDown":   reco_sfs["nominal"] * id_sfs["down"],
                }

    def electronSF_clib(self, electrons: ak.Array, variations=True):
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
        # restrict to the nominal systematic when up/down variations are not requested
        _electron_systs = {"nominal": "sf", "up": "sfup", "down": "sfdown"} if variations else {"nominal": "sf"}
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
                                        ) for sig1_name, sig1_syst in _electron_systs.items()
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
                                      ) for sig1_name, sig1_syst in _electron_systs.items()
                  }
        id_sfs = {k: ak.fill_none(v, 1.0, axis=1) for k, v in id_sfs.items()}
        _ones = ak.ones_like(electrons.pt)
        if not variations:
            _nom = reco_sfs["nominal"] * id_sfs["nominal"]
            # variation fields aliased to nominal/ones (unused when variations=False)
            return {"nominal":        _nom,
                    "eff_m_idUp":     _ones,
                    "eff_m_idDown":   _ones,
                    "eff_m_isoUp":    _ones,
                    "eff_m_isoDown":  _ones,
                    "eff_e_recoUp":   _nom,
                    "eff_e_recoDown": _nom,
                    "eff_e_idUp":     _nom,
                    "eff_e_idDown":   _nom,
                    }
        return {"nominal":        reco_sfs["nominal"] * id_sfs["nominal"],
                "eff_m_idUp":     _ones,
                "eff_m_idDown":   _ones,
                "eff_m_isoUp":    _ones,
                "eff_m_isoDown":  _ones,
                "eff_e_recoUp":   reco_sfs["up"]      * id_sfs["nominal"],
                "eff_e_recoDown": reco_sfs["down"]    * id_sfs["nominal"],
                "eff_e_idUp":     reco_sfs["nominal"] * id_sfs["up"],
                "eff_e_idDown":   reco_sfs["nominal"] * id_sfs["down"],
                }

    def append_lepton_sf(self, lead_lep, subl_lep, weights, variations=True):
        # FIXME: the same reco/id nominal SF (lead_lep.nominal*subl_lep.nominal) is
        # added once per key, so the nominal weight is effectively raised to the 4th
        # power. Preserved here bit-for-bit; a central application of the reco nominal
        # SF (with all-ones nominals on the variation entries) is deferred to a future
        # PR to keep this variations change bit-identical.
        for key in ["eff_m_id", "eff_m_iso", "eff_e_reco", "eff_e_id"]:
            if variations:
                weights.add(
                    key,
                    lead_lep.nominal*subl_lep.nominal,
                    getattr(lead_lep, f"{key}Up") * getattr(subl_lep, f"{key}Up"),
                    getattr(lead_lep, f"{key}Down") * getattr(subl_lep, f"{key}Down"),
                )
            else:
                weights.add(
                    key,
                    lead_lep.nominal*subl_lep.nominal,
                )
