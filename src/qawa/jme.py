from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JECStack
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import CorrectedJetsFactory
from coffea.jetmet_tools import CorrectedMETFactory

from coffea.lookup_tools import dense_lookup
import awkward as ak
import numpy as np
import os
import warnings


jec_name_map = {
    'JetPt': 'pt',
    'JetMass': 'mass',
    'JetEta': 'eta',
    'JetA': 'area',
    'ptRaw': 'pt_raw',
    'massRaw': 'mass_raw',
    'Rho': 'rhoFixedGridFastJetAll',
    'METpt': 'pt',
    'METphi': 'phi',
    'JetPhi': 'phi',
    'UnClusteredEnergyDeltaX': 'MetUnclustEnUpDeltaX',
    'UnClusteredEnergyDeltaY': 'MetUnclustEnUpDeltaY',
    'ptGenJet' : 'pt_gen'
}

def update_collection(event, coll):
    out = event
    for name, value in coll.items():
        out = ak.with_field(out, value, name)
    return out

def add_jme_variables(jets, events_rho, pt_gen=None):
    # Protect against re-deriving raw from wrong pt by only adding fields the first time they're definitively missing
    if 'pt_raw' not in jets.fields:
        jets['pt_raw'  ] = (1 - jets.rawFactor) * jets.pt
    if 'mass_raw' not in jets.fields:
        jets['mass_raw'] = (1 - jets.rawFactor) * jets.mass
    if 'pt_gen' not in jets.fields:
        if hasattr(jets, 'matched_gen'):
            jets['pt_gen'  ] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        elif pt_gen is not None:
            jets['pt_gen'] = ak.values_astype(ak.fill_none(pt_gen, 0), np.float32)
        else:
            jets['pt_gen'] = ak.Array(np.zeros(len(jets), dtype=np.float32))
    if 'rhoFixedGridFastJetAll' not in jets.fields:
        jets['rhoFixedGridFastJetAll'] = ak.broadcast_arrays(events_rho, jets.pt)[0]
    return jets

class JMEUncertainty:
    def __init__(
            self,
            jec_tag: str = 'Summer19UL18_V5_MC',
            jer_tag: str = 'Summer19UL18_JRV2_MC',
            era: str = "2018",
            is_mc: bool = True,
            doJER: bool = True,
            clibhandler = None,
            version: str = "v9",
    ):
        if clibhandler is not None:
            self.__init_clib__(jec_tag=jec_tag,
                               jer_tag=jer_tag,
                               era=era,
                               is_mc=is_mc,
                               doJER=doJER,
                               clibhandler=clibhandler,
                               version=version,
                               )
        else:
            self.__init_legacy__(jec_tag=jec_tag,
                                 jer_tag=jer_tag,
                                 era=era,
                                 is_mc=is_mc,
                                 doJER=doJER,
                                 version=version,
                                 )
    def __init_legacy__(
            self,
            jec_tag: str = 'Summer19UL18_V5_MC',
            jer_tag: str = 'Summer19UL18_JRV2_MC',
            era: str = "2018",
            is_mc: bool = True,
            doJER: bool = True,
            version: str = "v9",
    ):
        jet_type = "AK4PFchs" if version in [f"v{V}" for V in range(12)]  else "AK4PFPuppi" # Use Puppi for v12 and later
        _data_path = os.path.join(os.path.dirname(__file__), 'data/jme/')
        extract_L1 = extractor()
        extract_L123_noJER = extractor()
        extract_L123_JER = extractor()

        if is_mc:
            correction_list_L123 = [
                # Jet Energy Correction
                f'* * {_data_path}/{era}/{jec_tag}_L1FastJet_{jet_type}.jec.txt',
                f'* * {_data_path}/{era}/{jec_tag}_L2Relative_{jet_type}.jec.txt',
                f'* * {_data_path}/{era}/{jec_tag}_L3Absolute_{jet_type}.jec.txt',
                f'* * {_data_path}/{era}/RegroupedV2_{jec_tag}_UncertaintySources_{jet_type}.junc.txt',
            ]
        else:
            correction_list_L123 = [
                # Jet Energy Correction
                f'* * {_data_path}/{era}/{jec_tag}_L1FastJet_{jet_type}.jec.txt',
                #f'* * {_data_path}/{era}/{jec_tag}_L2L3Residual_{jet_type}.jec.txt',
                f'* * {_data_path}/{era}/{jec_tag}_L2Relative_{jet_type}.jec.txt',
                f'* * {_data_path}/{era}/{jec_tag}_L3Absolute_{jet_type}.jec.txt',
                f'* * {_data_path}/{era}/{jec_tag}_L2L3Residual_{jet_type}.jec.txt',
            ]

        correction_list_L1 = [
            # Jet Energy Correction
            f'* * {_data_path}/{era}/{jec_tag}_L1FastJet_{jet_type}.jec.txt',
        ]
        # these two have to start in sync, then we add JER to the latter
        correction_list_L123_noJER = [cor for cor in correction_list_L123]
        correction_list_L123_JER = [cor for cor in correction_list_L123]
        if is_mc and jet_type != "AK4PFPuppi":
            common_files = [
                # Jet Energy Resolution
                f'* * {_data_path}/{era}/{jer_tag}_PtResolution_{jet_type}.jr.txt',
                f'* * {_data_path}/{era}/{jer_tag}_SF_{jet_type}.jersf.txt',
            ]
            correction_list_L123_JER += common_files
        if jet_type == "AK4PFPuppi":
            warnings.warn("JER disabled for AK4PFPuppi jets due to broken text file format, need correctionlib version to replace it")


        extract_L1.add_weight_sets(correction_list_L1)
        extract_L1.finalize()
        evaluator_L1 = extract_L1.make_evaluator()
        jec_inputs_L1 = {
            name: evaluator_L1[name] for name in dir(evaluator_L1)
        }
        self.jec_stack_L1 = JECStack(jec_inputs_L1)
        self.jec_factory_L1 = CorrectedJetsFactory(jec_name_map, self.jec_stack_L1)

        extract_L123_JER.add_weight_sets(correction_list_L123_JER)
        extract_L123_JER.finalize()
        evaluator_L123_JER = extract_L123_JER.make_evaluator()
        jec_inputs_L123_JER = {
            name: evaluator_L123_JER[name] for name in dir(evaluator_L123_JER)
        }
        self.jec_stack_L123_JER = JECStack(jec_inputs_L123_JER)
        self.jec_factory_L123_JER = CorrectedJetsFactory(jec_name_map, self.jec_stack_L123_JER)

        extract_L123_noJER.add_weight_sets(correction_list_L123_noJER)
        extract_L123_noJER.finalize()
        evaluator_L123_noJER = extract_L123_noJER.make_evaluator()
        jec_inputs_L123_noJER = {
            name: evaluator_L123_noJER[name] for name in dir(evaluator_L123_noJER)
        }
        self.jec_stack_L123_noJER = JECStack(jec_inputs_L123_noJER)
        self.jec_factory_L123_noJER = CorrectedJetsFactory(jec_name_map, self.jec_stack_L123_noJER)

        self.met_factory = CorrectedMETFactory(jec_name_map)


    def __init_clib__(self,
                      jec_tag: str = 'Summer19UL18_V5_MC',
                      jer_tag: str = 'Summer19UL18_JRV2_MC',
                      era: str = "2018",
                      is_mc: bool = True,
                      doJER: bool = True,
                      version: str = "v9",
                      clibhandler=None,
                      ):
        if clibhandler is None:
            raise ValueError("Must provide clibhandler for non-legacy path")
        jet_type = "AK4PFchs" if version in [f"v{V}" for V in range(12)]  else "AK4PFPuppi" # Use Puppi for v12 and later
        # Regrouped set of uncertainties
        unc_sources_regrouped = [
            f"Regrouped_Absolute_{era}",
            "Regrouped_Absolute",
            f"Regrouped_BBEC1_{era}",
            "Regrouped_BBEC1",
            f"Regrouped_EC2_{era}",
            "Regrouped_EC2",
            "Regrouped_FlavorQCD",
            f"Regrouped_HF_{era}",
            "Regrouped_HF",
            "Regrouped_RelativeBal",
            f"Regrouped_RelativeSample_{era}",
        ]
        from coffea.jetmet_tools import CorrectionLibJECStack
        # Use the central code added here: https://github.com/scikit-hep/coffea/pull/1521
        self.jec_stack_L123_JER = CorrectionLibJECStack.from_file(
            clibhandler.getPath("jet_jerc"),
            jec_tag=jet_tag,
            data_type="MC" if is_mc else "DATA",
            jet_type=jet_type,
            jet_level="L1L2L3Res",
            unc_sources=unc_sources_regrouped,
            jer_tag=jer_tag,
        )
        self.jec_stack_L123_noJER = CorrectionLibJECStack.from_file(
            clibhandler.getPath("jet_jerc"),
            jec_tag=jet_tag,
            data_type="MC" if is_mc else "DATA",
            jet_type=jet_type,
            jet_level="L1L2L3Res",
            unc_sources=unc_sources_regrouped,
            jer_tag=None,
        )
        # This doesn't yet work due to lack of CorrectionSet compound corrections, currently limited to the MC and DATA variations of L1L2L3Res...
        # self.jec_stack_L1 = CorrectionLibJECStack.from_file(
        #     clibhandler.getPath("jet_jerc"),
        #     jec_tag=jet_tag,
        #     data_type="MC" if is_mc else "DATA",
        #     jet_type=jet_type,
        #     jet_level="L1",
        #     unc_sources=unc_sources_regrouped,
        #     jer_tag=None,
        # )
        raise NotImplementedError("JMEUncertainty with Correctionlib requires further adaptations to include L1 only corrections and propagation to MET; L123Res is usable alone")
        

    # This function is all confused, remove now that vbs-zz is finished
    # def corrected_jets_L123(self, jets, event_rho, lazy_cache, pt_gen=None):
    #     jet_pt_L123 = self.jec_factory_L123_noJER.build(
    #         add_jme_variables(jets, event_rho, pt_gen),
    #         # lazy_cache=lazy_cache
    #     )
    #     emFraction = jet_pt_L123.chEmEF + jet_pt_L123.neEmEF
    #     mask_jec = (jet_pt_L123['pt'] > 15) & (emFraction <= 0.9)
    #     selected_jets_L123 = ak.mask(jet_pt_L123, mask_jec)
    #     # selected_jets_L123 = jet_pt_L123[mask_jec]
    #     selected_jets_L123['pt'] = selected_jets_L123['pt'] * (1 - selected_jets_L123.muonSubtrFactor)
    #     return selected_jets_L123
    
    def corrected_jets_L1(self, jets, event_rho, lazy_cache, pt_gen=None):
        jet_pt_L1 = self.jec_factory_L1.build(
            add_jme_variables(jets, event_rho, pt_gen),
            # lazy_cache=lazy_cache
        )
        jet_pt_L1['pt'] = jet_pt_L1['pt'] * (1 - jet_pt_L1.muonSubtrFactor)
        return jet_pt_L1
    
    def corrected_jets_L123_JER(self, jets, event_rho, lazy_cache):
        jets = add_jme_variables(jets, event_rho)
        return self.jec_factory_L123_JER.build(
            jets,
            # lazy_cache
        )

    def corrected_jets_L123_noJER(self, jets, event_rho, lazy_cache):
        jets = add_jme_variables(jets, event_rho)
        #jets['pt'] = jets['pt'] * (1 - jets.muonSubtrFactor)
        return self.jec_factory_L123_noJER.build(
            jets,
            # lazy_cache
        )
      #remove jets_L1
    # Replace this with the coffea updated implementation eventually: https://github.com/scikit-hep/coffea/pull/1537
    def corrected_met(self, met, jets_L123, event_rho, lazy_cache):
        emFraction = jets_L123.chEmEF + jets_L123.neEmEF
        mask_jec = (jets_L123['pt'] > 15) & (emFraction <= 0.9)
        jets_L123_cleaned_for_MET = jets_L123[mask_jec]
        #jets_L123_cleaned_for_MET['pt'] = jets_L123_cleaned_for_MET['pt'] * (1 - jets_L123_cleaned_for_MET.muonSubtrFactor)
        return self.met_factory.build(
            met,
            jets_L123_cleaned_for_MET,
            # lazy_cache=lazy_cache
        )
