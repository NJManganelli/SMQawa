import correctionlib
import pickle
import hist
import gzip
import os
from hist.intervals import clopper_pearson_interval
from correctionlib import _core

from coffea.analysis_tools import Weights

import awkward as ak
import numpy as np
import uproot



class jetPUScaleFactors:
    def __init__(self, era:str='2018', wp:str='M', isAPV=False, isEE=False, isBPix=False):
        if isEE or isBPix:
            raise NotImplementedError("The implementation for 2022 and 2023 eras is complete for jetPUScaleFactors, ensure they are applicable and if so complete this method, or disable them entirely")
        self._era = era
        self._wp  = wp
        self.isAPV = isAPV
        _data_path = os.path.join(os.path.dirname(__file__), 'data')
        with gzip.open(
                f"{_data_path}/jet_PU_SF/jmar_{era+'_APV' if isAPV else era}_UL.json.gz", 
                'rt') as ifile:
            data = ifile.read().strip()
            #self.evaluator = _core.CorrectionSet.from_string(data)
            self.evaluator = correctionlib.CorrectionSet.from_string(data)


    def getSF(self, jet, syst="nom"):
        njets    = ak.num(jet)
        jets     = ak.flatten(jet)
        jet_eta  = np.array(jets.eta)
        jet_pt   = np.array(jets.pt)
        jet_puid = np.array(jets.puId)
        jet_sf = self.evaluator["PUJetID_eff"].evaluate(
            jet_eta,
            jet_pt,
            syst, self._wp
            )
        jet_eff = self.evaluator["PUJetID_eff"].evaluate(
            jet_eta,
            jet_pt,
            'MCEff', self._wp
            )
        if   self._wp == 'M':
            # tagged = (jet_puid>=6) | (jet_puid == 3)
            tagged = np.where((jet_puid>=6) | (jet_puid == 3),True, False)
        elif self._wp == 'T':
            tagged = jet_puid>=7
        elif self._wp == 'L':
            tagged = jet_puid>=1
        tagged_sf   = np.minimum(jet_sf*jet_eff, 1)/jet_eff
        untagged_sf = np.maximum(1.- jet_sf*jet_eff,0)/(1-jet_eff)
        tagged_sf   = ak.unflatten(tagged_sf  , njets)
        untagged_sf = ak.unflatten(untagged_sf, njets)
        
        tagged      = ak.unflatten(tagged     , njets)
        
        tagged_sf   = ak.prod(tagged_sf[tagged]  , axis=-1)
        untagged_sf = ak.prod(untagged_sf[~tagged], axis=-1)

        return ak.fill_none(tagged_sf * untagged_sf, 1.)

    


    def append_jetPU_sf(self, jets: ak.Array, weights: Weights, variations=True):
        jets = jets[(jets.genJetIdx != -1) & (jets.pt <= 50) & (np.abs(jets.eta) <= 5) & (jets.pt >= 30)]
        sf_nom  = self.getSF(jets, 'nom')
        if variations:
            sf_up   = self.getSF(jets, 'up')
            sf_down = self.getSF(jets, 'down')
            weights.add('jetPUid_sf'  , sf_nom, sf_up, sf_down)
        else:
            sf_up, sf_down = None, None
            weights.add('jetPUid_sf'  , sf_nom)



        return sf_nom, sf_up, sf_down
