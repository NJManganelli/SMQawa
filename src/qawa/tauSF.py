import correctionlib
import pickle
import hist
import gzip
import os

from coffea.analysis_tools import Weights

import awkward as ak
import numpy as np
import uproot

# https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun3#Corrections_for_the_DeepTauv2p5

class tauIDScaleFactors:
    def __init__(self, era:str='2018', vsjet_wp:str='VTight', vse_wp:str='VVLoose', vsmu_wp:str='VLoose',
                 isAPV=False, isEE=False, isBPix=False, clibhandler=None):

        config = {"2016": {"tagger": "DeepTau2017v2p1"},
                  "2017": {"tagger": "DeepTau2017v2p1"},
                  "2018": {"tagger": "DeepTau2017v2p1"},
                  "2022": {"tagger": "DeepTau2018v2p5"},
                  "2023": {"tagger": "DeepTau2018v2p5"}, # Maybe we want UParT tagger or Run 3, however!
                  "2024": {"tagger": "DeepTau2018v2p5"},
                  "2025": {"tagger": "DeepTau2018v2p5"},
                  }
        self._era = era
        self.vsjet_wp = vsjet_wp
        self.vse_wp = vse_wp
        self.vsmu_wp = vsmu_wp
        self.isAPV = isAPV
        self.isEE = isEE
        self.isBPix = isBPix
        assert sum([self.isAPV, self.isEE, self.isBPix]) <= 1, "Multiple incompatible suberas selected as active"
        assert self.vsmu_wp in ["Tight"], "Tau vsmu_wp may not be in calibrated WPs, double-check and update accordingly"
        assert self.vse_wp in ["VVLoose", "Tight"], "Tau vse_wp may not be in calibrated WPs, double-check and update accordingly"
        self._erasubera = self._era

        if isAPV:
            self._subera = "_APV"
        elif isEE:
            self._subera = "_EE"
        elif isBPix:
            self._subera = "_BPix"
        else:
            self._subera = ""
        self._erasubera += self._subera
        self.tagger = config[self._era]["tagger"]
        _data_path = os.path.join(os.path.dirname(__file__), 'data')
        # Load CorrectionSet
        if clibhandler is not None:
            cset = clibhandler.getCorrectionSet("tau")
        else:
            fname = f"{_data_path}/tau_ID_SF/{self._erasubera}/tau.json.gz"
            with gzip.open(fname,'rt') as file:
                data = file.read().strip()
                cset = correctionlib.CorrectionSet.from_string(data)
                # cset = correctionlib.CorrectionSet.from_file(file)

        #Load Correction Objects :
        self.corr_vsjet = cset[f"{self.tagger}VSjet"]
        self.corr_vse = cset[f"{self.tagger}VSe"]
        self.corr_vsmu = cset[f"{self.tagger}VSmu"]
        self.corr_enscale = cset["tau_energy_scale"]


    def getSF(self, tau, syst="nom"):
        ntaus = ak.num(tau)
        taus  = ak.flatten(tau)
        tau_eta = np.array(taus.eta)
        tau_pt  = np.array(taus.pt)
        tau_dm  = np.array(taus.decayMode)
        tau_genmatch = np.array(taus.genPartFlav)

        inputs_vsjet = {"pt": tau_pt, "dm": tau_dm, "genmatch": tau_genmatch, "syst": syst, "wp": self.vsjet_wp, "wp_VSe": self.vse_wp, "flag": "dm"} #flag "pt" alone also possible, dm = pt(param) + dm (binned)
        inputs_vse =   {"eta": tau_eta, "dm": tau_dm, "genmatch": tau_genmatch, "wp": self.vse_wp, "syst": syst}
        inputs_vsmu =  {"eta": tau_eta, "genmatch": tau_genmatch, "wp": self.vsmu_wp, "wp_VSjet": self.vsjet_wp, "wp_VSe": self.vse_wp, "syst": syst}
        if inputs_vsmu["wp_VSjet"] in ['VTight', 'VVTight']:
            inputs_vsmu["wp_VSjet"] = 'Tight'

        # {self.tagger}VSjet
        # sf_vsjet = self.corr_vsjet.evaluate(tau_pt,tau_dm,tau_genmatch, self.vsjet_wp, self.vse_wp, syst,"pt")
        # DeepTau2018v2p5VSjet ['pt', 'dm', 'genmatch', 'wp', 'wp_VSe', 'syst', 'flag'] # ('flag', "Flag: 'pt' = pT-dependent SFs, 'dm' = DM-dependent SFs")]
        # DeepTau2017v2p1VSjet ['pt', 'dm', 'genmatch', 'wp', 'wp_VSe', 'syst', 'flag']
        sf_vsjet = self.corr_vsjet.evaluate(*({k.name: inputs_vsjet[k.name] for k in self.corr_vsjet.inputs}.values()))
        sf_vsjet = ak.fill_none(sf_vsjet, 1.0)
        sf_vsjet = ak.unflatten(sf_vsjet, ntaus)
        sf_vsjet = ak.prod(sf_vsjet, axis=-1)

        # {self.tagger}VSe
        # DeepTau2018v2p5VSe ['eta', 'dm', 'genmatch', 'wp', 'syst']
        # DeepTau2017v2p1VSe ['eta', 'genmatch', 'wp', 'syst']
        sf_vse = self.corr_vse.evaluate(*({k.name: inputs_vse[k.name] for k in self.corr_vse.inputs}.values()))
        sf_vse = ak.fill_none(sf_vse, 1.)
        sf_vse = ak.unflatten(sf_vse, ntaus)
        sf_vse = ak.prod(sf_vse, axis=-1)

        # {self.tagger}VSmu
        # DeepTau2018v2p5VSmu ['eta', 'genmatch', 'wp', 'wp_VSe', 'wp_VSjet', 'syst'] # BUT wp_VSjet cannot be higher than 'Tight' in Run 3
        # DeepTau2017v2p1VSmu ['eta', 'genmatch', 'wp', 'syst']
        sf_vsmu = self.corr_vsmu.evaluate(*({k.name: inputs_vsmu[k.name] for k in self.corr_vsmu.inputs}.values()))
        sf_vsmu = ak.fill_none(sf_vsmu, 1.)
        sf_vsmu = ak.unflatten(sf_vsmu, ntaus)
        sf_vsmu = ak.prod(sf_vsmu, axis=-1)

        return  sf_vsjet, sf_vse, sf_vsmu


    def apply_function_flattened_masked(self,func, *args, flatten_axis=1, valid_where=None):
        maybe_flat_args = []
        num = None
        for arg in args:
            if isinstance(arg, ak.Array):
                if valid_where is None:
                    valid_where = ak.ones_like(arg, dtype=bool)
                if num is None:
                    num = ak.num(arg)
        maybe_flat_args = [ak.flatten(ak.mask(arg, valid_where), axis=flatten_axis) if isinstance(arg, ak.Array) else arg for arg in args]
        return ak.unflatten(
            func(*maybe_flat_args),
            num
        )

    def tau_energy_scale_correction(self, tau):

        # Usage, lets pretend only the dm 5 and 6 had to be avoided but all other inputs were valid
        valid_tau_enscale = (tau.decayMode != 5) & (tau.decayMode != 6)

        # ntaus = ak.num(tau)
        # taus  = ak.flatten(tau)
        tau_eta = tau.eta
        tau_pt  = tau.pt
        tau_mass = tau.mass
        tau_dm  = tau.decayMode
        tau_genmatch = tau.genPartFlav

        enscale_nom_with_none = self.apply_function_flattened_masked(
            self.corr_enscale.evaluate,
            tau_pt,
            tau_eta,
            tau_dm,
            tau_genmatch,
            self.tagger,
            "nom",
            valid_where = valid_tau_enscale
        )
        enscale_up_with_none = self.apply_function_flattened_masked(
            self.corr_enscale.evaluate,
            tau_pt,
            tau_eta,
            tau_dm,
            tau_genmatch,
            self.tagger,
            "up",
            valid_where = valid_tau_enscale
        )
        enscale_down_with_none = self.apply_function_flattened_masked(
            self.corr_enscale.evaluate,
            tau_pt,
            tau_eta,
            tau_dm,
            tau_genmatch,
            self.tagger,
            "down",
            valid_where = valid_tau_enscale
        )
        #now enscale_nom should have a SF where the valid_tau_enscale is true, and "None" elsewhere, so fill_none it with 1.0 to not alter the scale
        enscale_nom = ak.fill_none(enscale_nom_with_none, 1.0)
        enscale_up = ak.fill_none(enscale_up_with_none, 1.0)
        enscale_down = ak.fill_none(enscale_down_with_none, 1.0)

        # ak.num(enscale_nom, axis=1) == ak.num(tau.pt, axis=1)
        # ak.num(enscale_up, axis=1) == ak.num(tau.pt, axis=1)
        # ak.num(enscale_down, axis=1) == ak.num(tau.pt, axis=1)

        tau_pt = enscale_nom*tau_pt
        tau_pt_EnUp = enscale_up*tau_pt
        tau_pt_EnDown = enscale_down*tau_pt

        tau_mass = enscale_nom*tau_mass
        tau_mass_EnUp = enscale_up*tau_mass
        tau_mass_EnDown = enscale_down*tau_mass

        return tau_pt, tau_pt_EnUp, tau_pt_EnDown, tau_mass, tau_mass_EnUp, tau_mass_EnDown

    def append_tauID_sf(self, taus: ak.Array, weights: Weights):

        sf_vsjet_nom, sf_vse_nom, sf_vsmu_nom = self.getSF(taus, syst="nom")
        sf_vsjet_up, sf_vse_up, sf_vsmu_up = self.getSF(taus, syst="up")
        sf_vsjet_down, sf_vse_down, sf_vsmu_down = self.getSF(taus, syst="down")


        weights.add('tauIDvsjet_sf', sf_vsjet_nom, sf_vsjet_up, sf_vsjet_down)
        weights.add('tauIDvse_sf'  , sf_vse_nom, sf_vse_up, sf_vse_down)
        weights.add('tauIDvsmu_sf' , sf_vsmu_nom, sf_vsmu_up, sf_vsmu_down)

        return sf_vsjet_nom, sf_vsjet_up, sf_vsjet_down, sf_vse_nom, sf_vse_up, sf_vse_down, sf_vsmu_nom, sf_vsmu_up, sf_vsmu_down

    def append_tauID_multiwp_sf(self,
                                taus_vtight: ak.Array, taus_tight: ak.Array, taus_loose: ak.Array,
                                mask_vtight: ak.Array, mask_tight: ak.Array, mask_loose: ak.Array,
                                weights: Weights,
                                variations=True
                                ):
        # store original WP to prevent statefulness bug
        original_WP = self.vsjet_wp

        self.vsjet_wp = "VTight"
        vtight_vsjet_nom, vtight_vse_nom, vtight_vsmu_nom = self.getSF(taus_vtight, syst="nom")
        if variations:
            vtight_vsjet_up,  vtight_vse_up,  vtight_vsmu_up  = self.getSF(taus_vtight, syst="up")
            vtight_vsjet_dn,  vtight_vse_dn,  vtight_vsmu_dn  = self.getSF(taus_vtight, syst="down")

        self.vsjet_wp = "Tight"
        tight_vsjet_nom, tight_vse_nom, tight_vsmu_nom = self.getSF(taus_tight, syst="nom")
        if variations:
            tight_vsjet_up,  tight_vse_up,  tight_vsmu_up  = self.getSF(taus_tight, syst="up")
            tight_vsjet_dn,  tight_vse_dn,  tight_vsmu_dn  = self.getSF(taus_tight, syst="down")

        self.vsjet_wp = "Loose"
        loose_vsjet_nom, loose_vse_nom, loose_vsmu_nom = self.getSF(taus_loose, syst="nom")
        if variations:
            loose_vsjet_up,  loose_vse_up,  loose_vsmu_up  = self.getSF(taus_loose, syst="up")
            loose_vsjet_dn,  loose_vse_dn,  loose_vsmu_dn  = self.getSF(taus_loose, syst="down")

        # restore original WP, we do not want to introduce a statefulness bug with usage
        self.vsjet_wp = original_WP

        vtight_vsjet_nom = ak.fill_none(vtight_vsjet_nom, 1.0)
        tight_vsjet_nom = ak.fill_none(vtight_vsjet_nom, 1.0)
        loose_vsjet_nom = ak.fill_none(loose_vsjet_nom, 1.0)

        sf_vsjet_nom = ak.where(mask_vtight, vtight_vsjet_nom, ak.where(mask_tight, tight_vsjet_nom, loose_vsjet_nom))
        sf_vse_nom   = ak.where(mask_vtight, vtight_vse_nom, ak.where(mask_tight, tight_vse_nom, loose_vse_nom))
        sf_vsmu_nom  = ak.where(mask_vtight, vtight_vsmu_nom, ak.where(mask_tight, tight_vsmu_nom, loose_vsmu_nom))

        if not variations:
            weights.add('tauIDvsjet_sf', sf_vsjet_nom)
            weights.add('tauIDvse_sf',   sf_vse_nom)
            weights.add('tauIDvsmu_sf',  sf_vsmu_nom)
            return sf_vsjet_nom, None, None, sf_vse_nom, None, None, sf_vsmu_nom, None, None

        vtight_vsjet_up  = ak.fill_none(vtight_vsjet_up , 1.0)
        vtight_vsjet_dn  = ak.fill_none(vtight_vsjet_dn , 1.0)

        tight_vsjet_up  = ak.fill_none(vtight_vsjet_up , 1.0)
        tight_vsjet_dn  = ak.fill_none(vtight_vsjet_dn , 1.0)

        loose_vsjet_up  = ak.fill_none(loose_vsjet_up , 1.0)
        loose_vsjet_dn  = ak.fill_none(loose_vsjet_dn , 1.0)

        sf_vsjet_up  = ak.where(mask_vtight, vtight_vsjet_up , ak.where(mask_tight, tight_vsjet_up , loose_vsjet_up ))
        sf_vsjet_dn  = ak.where(mask_vtight, vtight_vsjet_dn , ak.where(mask_tight, tight_vsjet_dn , loose_vsjet_dn ))

        sf_vse_up    = ak.where(mask_vtight, vtight_vse_up , ak.where(mask_tight, tight_vse_up , loose_vse_up ))
        sf_vse_dn    = ak.where(mask_vtight, vtight_vse_dn , ak.where(mask_tight, tight_vse_dn , loose_vse_dn ))

        sf_vsmu_up   = ak.where(mask_vtight, vtight_vsmu_up , ak.where(mask_tight, tight_vsmu_up , loose_vsmu_up ))
        sf_vsmu_dn   = ak.where(mask_vtight, vtight_vsmu_dn , ak.where(mask_tight, tight_vsmu_dn , loose_vsmu_dn ))

        weights.add('tauIDvsjet_sf', sf_vsjet_nom, sf_vsjet_up, sf_vsjet_dn)
        weights.add('tauIDvse_sf',   sf_vse_nom,   sf_vse_up,   sf_vse_dn)
        weights.add('tauIDvsmu_sf',  sf_vsmu_nom,  sf_vsmu_up,  sf_vsmu_dn)

        return sf_vsjet_nom, sf_vsjet_up, sf_vsjet_dn, sf_vse_nom, sf_vse_up, sf_vse_dn, sf_vsmu_nom, sf_vsmu_up, sf_vsmu_dn

def tau_energy_scale(sink, events, tagger, wp_VSjet, wp_VSe, unc_type=None, is_correction=True, dm2IndexErrorWorkaround=False, clibhandler=None):
    """Alternative implementation to tauSF.tau_energy_scale_correction function to make compatible with systematics"""
    # FIXME: This function probably needs a multi-WP variation the same as the append_tauID_sf method now has...
    if clibhandler is not None:
        cset = clibhandler.getCorrectionSet("tau")
    else:
        raise ValueError("clibhandler required for tau_energy_scale function")

    corr_enscale = cset["tau_energy_scale"]

    # evaluator = clibhandler.getCorrectionSet("muon_scalesmearing")
    # for later unflattening:
    counts = ak.num(events.Tau.pt)
    # decide if the process data
    is_data = False if hasattr(events, "genWeight") else True

    taus_jagged = events.Tau
    taus = ak.flatten(taus_jagged)

    valid_tau_enscale = (taus.decayMode != 5) & (taus.decayMode != 6)
    if dm2IndexErrorWorkaround:
        valid_tau_enscale = valid_tau_enscale & (taus.decayMode != 2)
    taus_masked = ak.mask(taus, valid_tau_enscale, valid_when=True)
    input_superset = {
        "pt": taus_masked.pt,
        "eta": taus_masked.eta,
        "mass": taus_masked.mass,
        "dm":taus_masked.decayMode,
        "genmatch": taus_masked.genPartFlav,
        "wp": wp_VSjet, # Needed in Run3 corrections, but not UL Run2, but that may change with NanoAODv15
        "wp_VSe": wp_VSe, # Needed in Run3 corrections, but not UL Run2, but that may change with NanoAODv15
        "id": tagger,
        "syst": None, #override in is_correction if-else
        }
    inputs = {k: input_superset[k] for k in [k.name for k in corr_enscale.inputs]} # Automatic ordering and input subset selection
    if is_correction:
        inputs["syst"] = "nom"
        taus["pt_nanoaod"] = taus.pt
        taus["mass_nanoaod"] = taus.mass
        # * Data only need scale correction
        enscale_nom_with_none = corr_enscale.evaluate(*inputs.values())
        # enscale_nom = ak.where(ak.is_none(enscale_nom_with_none), taus.pt, enscale_nom_with_none)
        enscale_nom = ak.fill_none(enscale_nom_with_none, 1.0)
        taus["pt"] = enscale_nom * taus.pt_nanoaod
        taus["mass"] = enscale_nom * taus.pt_nanoaod
        events["Tau"] = ak.unflatten(taus, counts)
        return events
    else:
        if not hasattr(events, "genWeight"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        if unc_type:
            if unc_type == "Scale":
                if not hasattr(taus, "pt_nanoaod"):
                    raise ValueError(f"Tau collection missing pt_nanoaod field from applying is_correction of tau_energy_scale")
                inputs["syst"] = "up"
                enscale_up_with_none = corr_enscale.evaluate(*inputs.values())
                enscale_up = ak.fill_none(enscale_up_with_none, 1.0)
                # enscale_up = ak.where(ak.is_none(enscale_up_with_none), taus.pt, enscale_up_with_none)
                inputs["syst"] = "down"
                enscale_down_with_none = corr_enscale.evaluate(*inputs.values())
                enscale_down = ak.fill_none(enscale_down_with_none, 1.0)
                # enscale_down = ak.where(ak.is_none(enscale_down_with_none), taus.pt, enscale_down_with_none)
                # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
                return ak.zip({
                    "pt": np.concatenate(
                        (
                            (enscale_up * taus.pt_nanoaod)[:, None],
                            (enscale_down * taus.pt_nanoaod)[:, None],
                        ),
                        axis=1,
                    ),
                    "mass": np.concatenate(
                        (
                            (enscale_up * taus.mass_nanoaod)[:, None],
                            (enscale_down * taus.mass_nanoaod)[:, None],
                        ),
                        axis=1,
                    ),
                }, depth_limit=1)
            else:
                raise ValueError(f"unc_type must be Scale, got {unc_type}")
