import correctionlib
import pickle
import hist
import gzip
import os
from pathlib import Path
from hist.intervals import clopper_pearson_interval

from coffea.lookup_tools import dense_lookup
from coffea.lookup_tools import dense_lookup
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.analysis_tools import Weights

import awkward as ak
import numpy as np
import uproot

def btag_disc_wps(tagger:str='deepJet'):
    _disc_wps = {
        "deepJet": {
            "discriminant": "btagDeepFlavB",
            "workingpoints": {
                "2016"   : {"L": 0.0508, "M": 0.2598, "T": 0.6502},
                "2016APV": {"L": 0.0480, "M": 0.2489, "T": 0.6377},
                "2017"   : {"L": 0.0532, "M": 0.3040, "T": 0.7476},
                "2018"   : {"L": 0.0490, "M": 0.2783, "T": 0.7100}
            },
        },
        "UParTAK4": {
            "discriminant": "btagUParTAK4B",
            "workingpoints": None # dynamically loaded from btagging.json.gz instead
        },
    }
    return _disc_wps[tagger]
    
class BTVCorrector:
    def __init__(self, era:str='2018', wp:str='L', tagger:str='deepJet', isAPV=False, isEE=False, isBPix=False, clibhandler=None):
        assert tagger in ["deepJet", "UParTAK4"]
        assert wp in ["L", "M", "T", "XT", "XXT"]
        self._era = era
        self._wp  = wp
        self.isAPV = isAPV
        self.isEE = isEE
        self.isBPix = isBPix
        self._tagger = tagger
        self._disc = btag_disc_wps(tagger=tagger)["discriminant"]
        self.eff_hist = None
        _data_path = Path(os.path.dirname(__file__)) / 'data'
        self._erasubera = era
        if isAPV:
            self._erasubera += "_APV"
        elif isEE:
            self._erasubera += "_EE"
        elif isBPix:
            self._erasubera += "_BPix"
        self._erasuberanousc = self._erasubera.replace("_", "")
        eff_path = Path(f"{_data_path}/btv/{self._erasubera}_UL/eff-btag-{self._erasubera}.pkl.gz") # Run 2 compatible version
        if eff_path.exists():
            with gzip.open(eff_path, 'rb') as ifile:
                self.eff_hist = pickle.loads(ifile.read())
    
            b_tag = self.eff_hist[{'tagger':self._tagger, 'WP': wp, 'passWP': 1  }].values()
            b_all = self.eff_hist[{'tagger':self._tagger, 'WP': wp, 'passWP': sum}].values()
    
            nom = b_tag / np.maximum(b_all, 1.)
            dw, up = clopper_pearson_interval(b_tag, b_all)
    
            self.eff        = dense_lookup.dense_lookup(nom,[ax.edges for ax in self.eff_hist.axes[3:]])
            self.eff_statUp = dense_lookup.dense_lookup(up ,[ax.edges for ax in self.eff_hist.axes[3:]])
            self.eff_statDw = dense_lookup.dense_lookup(dw ,[ax.edges for ax in self.eff_hist.axes[3:]])

        # FIXME This will need better correctionlib handling in the future, akin to __init_legacy__ vs __init_clib__ in other functions perhaps
        if clibhandler is not None:
            self.clib = clibhandler.getCorrectionSet("btagging")
        else:
            self.clib = correctionlib.CorrectionSet.from_file(
                f"{_data_path}/btv/{self._erasubera}_UL/"
                f"btagging.json.gz"
            )

        
    def lightSF(self, jet, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        njets = ak.num(jet)
        jet  = ak.flatten(jet)
        sf = self.clib[f"{self._tagger}_incl"].evaluate(
            syst, self._wp,
            np.array(jet.hadronFlavour), 
            np.array(abs(jet.eta)), 
            np.array(jet.pt)
        )
        return ak.unflatten(sf, njets)
    
    def btagSF(self, jet, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        # the jets have to be split between light and c/b quarks
        njets = ak.num(jet)
        jet  = ak.flatten(jet)
        sf = self.clib[f"{self._tagger}_comb"].evaluate(
            syst, self._wp,
            np.array(jet.hadronFlavour),
            np.array(abs(jet.eta)),
            np.array(jet.pt)
        )
        return ak.unflatten(sf, njets)

    def combine(self, eff, sf, b_tagged):
        # tagged SF = SF*eff / eff = SF
        tagged_sf = ak.prod(sf[b_tagged], axis=-1)
        # untagged SF = (1 - SF*eff) / (1 - eff)
        untagged_sf = ak.prod(((1 - sf*eff) / (1 - eff))[~b_tagged], axis=-1)
        return ak.fill_none(tagged_sf * untagged_sf, 1.)

    def tagged_jets(self, jets: ak.Array, wp=None):
        # permits possible multi-WP expansion in the future
        if wp is None:
            wps = [self._wp]
        elif isinstance(wp, str):
            wps = [wp]
        elif isinstance(wp, list):
            wps = wp
        else:
            raise ValueError(f"Unhandled tagged_jets logic: self._wp={self._wp} ; wp={wp} ; wps={wps}")
        return {wp: (getattr(jets, self._disc)  > self.clib[f"{self._tagger}_wp_values"].evaluate(wp)) for wp in wps}

    def append_btag_sf(self, jets: ak.Array, weights: Weights):
        if not hasattr(self, "eff"):
            raise ValueError("The efficiency lookups were not loaded for the BTVCorrector, possibly due to lack of correctionlib implementation or failure to find the path")
        # Eventually we may be able to migrate to a smarter compound lookup: https://cms-talk.web.cern.ch/t/suggestion-on-the-scope-of-provided-correctionlib-files/141772/4
        eta_max = 2.4 if "2016" in self._era else 2.5
        li_jets = jets[(jets.hadronFlavour==0) & (np.abs(jets.eta) <= eta_max)]
        bc_jets = jets[(jets.hadronFlavour!=0) & (np.abs(jets.eta) <= eta_max)]
        
        b_tagged_li = (getattr(li_jets, self._disc)  > self.clib[f"{self._tagger}_wp_values"].evaluate(self._wp))
        b_tagged_bc = (getattr(bc_jets, self._disc)  > self.clib[f"{self._tagger}_wp_values"].evaluate(self._wp))

        eff_li_nom = self.eff(li_jets.hadronFlavour, li_jets.pt, np.abs(li_jets.eta))
        eff_bc_nom = self.eff(bc_jets.hadronFlavour, bc_jets.pt, np.abs(bc_jets.eta))
        eff_li_up  = self.eff_statUp(li_jets.hadronFlavour, li_jets.pt, np.abs(li_jets.eta))
        eff_bc_up  = self.eff_statUp(bc_jets.hadronFlavour, bc_jets.pt, np.abs(bc_jets.eta))
        eff_li_dw  = self.eff_statDw(li_jets.hadronFlavour, li_jets.pt, np.abs(li_jets.eta))
        eff_bc_dw  = self.eff_statDw(bc_jets.hadronFlavour, bc_jets.pt, np.abs(bc_jets.eta))

        sf_nom_li = self.combine(eff_li_nom, self.lightSF(li_jets, 'central'), b_tagged_li)
        sf_nom_bc = self.combine(eff_bc_nom, self.btagSF(bc_jets, 'central'), b_tagged_bc)
        sf_nom = sf_nom_li * sf_nom_bc

        # nominal scale factors
        weights.add('btag_sf_light', sf_nom_li)
        weights.add('btag_sf_bc'   , sf_nom_bc)
        weights.add('btag_sf_nom'  , sf_nom)

        # statistical uncertainties
        weights.add(
            f'btag_sf_stat',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(
                eff_li_up, self.lightSF(li_jets, 'central'), b_tagged_li
            ) * self.combine(
                eff_bc_up, self.btagSF(bc_jets, 'central'), b_tagged_bc
            ) / (sf_nom_li*sf_nom_bc),
            weightDown= self.combine(
                eff_li_dw, self.lightSF(li_jets, 'central'), b_tagged_li
            ) * self.combine(
                eff_bc_dw, self.btagSF(bc_jets, 'central'), b_tagged_bc
            ) / (sf_nom_li*sf_nom_bc)
        )

        # uncorrelated uncertainties
        weights.add(
            f'btag_sf_light_{self._erasuberanousc}',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(eff_li_nom, self.lightSF(li_jets, 'up'), b_tagged_li),
            weightDown= self.combine(eff_li_nom, self.lightSF(li_jets, 'down'), b_tagged_li)
        )
        weights.add(
            f'btag_sf_bc_{self._erasuberanousc}',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(eff_bc_nom, self.btagSF(bc_jets, 'up'), b_tagged_bc),
            weightDown= self.combine(eff_bc_nom, self.btagSF(bc_jets, 'down'), b_tagged_bc)
        )

        # correlated uncertainties
        weights.add(
            'btag_sf_light_correlated',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(eff_li_nom, self.lightSF(li_jets, 'up_correlated'), b_tagged_li),
            weightDown= self.combine(eff_li_nom, self.lightSF(li_jets, 'down_correlated'), b_tagged_li)
        )
        weights.add(
            'btag_sf_bc_correlated',
            np.ones(len(sf_nom)),
            weightUp  = self.combine(eff_bc_nom, self.btagSF(bc_jets, 'up_correlated'), b_tagged_bc),
            weightDown= self.combine(eff_bc_nom, self.btagSF(bc_jets, 'down_correlated'), b_tagged_bc)
        )

        return sf_nom
