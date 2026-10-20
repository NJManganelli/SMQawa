from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JECStack
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import CorrectedJetsFactory
from coffea.jetmet_tools import CorrectedMETFactory
from coffea.jetmet_tools import CorrectionLibJEC, CorrectionLibJUNC, CorrectionLibJECStack

from coffea.lookup_tools import dense_lookup
from coffea.util import coffea_console
import awkward as ak
import numpy as np
import os


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

clib_name_map = {
    **jec_name_map,
    'run': 'run',
    'RawMETpt': 'pt',
    'RawMETphi': 'phi',
    'JetMuonSubtrFactor': 'muonSubtrFactor',
    'JetMuonSubtrDeltaPhi': 'muonSubtrDeltaPhi',
    'JetChEmEF': 'chEmEF',
    'JetNeEmEF': 'neEmEF',
    'CorrT1JetPt': 'rawPt',
    'CorrT1JetPhi': 'phi',
    'CorrT1JetEta': 'eta',
    'CorrT1JetArea': 'area',
    'CorrT1JetMuonSubtrFactor': 'muonSubtrFactor',
    'CorrT1JetMuonSubtrDeltaPhi': 'muonSubtrDeltaPhi',
    'CorrT1JetEmEF': 'EmEF',
    'CorrT1JetJERSmearFactor': 'jet_energy_resolution_correction',
}

# JetMET Run 3 performance (A. Benecke, 4 Dec 2025), slide 14: forward-jet mitigations per era
# https://indico.cern.ch/event/1615783/contributions/6811120/attachments/3186812/5672346/20251204_JetMET_PerformanceRun3.pdf#page=14
JETMET_RUN3_MITIGATIONS = {
    '2022': {
        'jer_genmatched_only_eta': ((2.5, 3.0),),
        'jet_pt_min': (((2.5, 3.0), 50.0), ((3.0, 5.0), 50.0)),
        'residual_pt_floor': None,
    },
    '2023': {
        'jer_genmatched_only_eta': ((2.5, 3.0),),
        'jet_pt_min': (((2.5, 3.0), 50.0), ((3.0, 5.0), 50.0)),
        'residual_pt_floor': None,
    },
    '2024': {
        'jer_genmatched_only_eta': ((2.5, 3.0),),
        'jet_pt_min': (((2.5, 3.0), 50.0),),
        'residual_pt_floor': (((2.0, 2.5),), 30.0),
    },
}

NO_MITIGATIONS = {'jer_genmatched_only_eta': (), 'jet_pt_min': (), 'residual_pt_floor': None}

JER_RESOLUTION_RHO_MAX = {"Summer23BPixPrompt23_RunD_JRV1": 52.05}

def jetmet_mitigations(era):
    return JETMET_RUN3_MITIGATIONS.get(era, NO_MITIGATIONS)

def abs_eta_in_regions(eta, regions):
    abs_eta = np.abs(eta)
    mask = ak.zeros_like(abs_eta, dtype=bool)
    for eta_min, eta_max in regions:
        mask = mask | ((abs_eta >= eta_min) & (abs_eta < eta_max))
    return mask

def forward_jet_pt_mask(jets, era):
    mask = ak.ones_like(jets.pt, dtype=bool)
    for region, pt_min in jetmet_mitigations(era)['jet_pt_min']:
        mask = mask & ~(abs_eta_in_regions(jets.eta, (region,)) & (jets.pt <= pt_min))
    return mask

class ResidualPtFloorJEC:
    def __init__(self, levels, residual, eta_regions, pt_min):
        self.levels = levels
        self.residual = residual
        self.eta_regions = eta_regions
        self.pt_min = pt_min
        self.signature = list(dict.fromkeys(inp.name for corr in [*levels, residual] for inp in corr.inputs))

    def getCorrection(self, **kwargs):
        values = dict(kwargs)
        total = None
        for corr in self.levels:
            factor = evaluate(corr, values)
            total = factor if total is None else total * factor
            values['JetPt'] = values['JetPt'] * factor
        floored = abs_eta_in_regions(values['JetEta'], self.eta_regions) & (values['JetPt'] < self.pt_min)
        values['JetPt'] = ak.where(floored, self.pt_min, values['JetPt'])
        return total * evaluate(self.residual, values)

def update_collection(event, coll):
    out = event
    for name, value in coll.items():
        out = ak.with_field(out, value, name)
    return out

def add_jme_variables(jets, events_rho, pt_gen=None, run=None, max_gen_dr=None):
    # Protect against re-deriving raw from wrong pt by only adding fields the first time they're definitively missing
    if 'pt_raw' not in jets.fields:
        jets['pt_raw'  ] = (1 - jets.rawFactor) * jets.pt
    if 'mass_raw' not in jets.fields:
        jets['mass_raw'] = (1 - jets.rawFactor) * jets.mass
    if 'pt_gen' not in jets.fields:
        if hasattr(jets, 'matched_gen'):
            if max_gen_dr is None:
                jets['pt_gen'  ] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            else:
                jets['pt_gen'] = ak.values_astype(ak.fill_none(ak.where(jets.delta_r(jets.matched_gen) < max_gen_dr, jets.matched_gen.pt, 0), 0), np.float32)
        elif pt_gen is not None:
            jets['pt_gen'] = ak.values_astype(ak.fill_none(pt_gen, 0), np.float32)
        else:
            jets['pt_gen'] = ak.Array(np.zeros(len(jets), dtype=np.float32))
    if 'rhoFixedGridFastJetAll' not in jets.fields:
        jets['rhoFixedGridFastJetAll'] = ak.broadcast_arrays(events_rho, jets.pt)[0]
    if run is not None and 'run' not in jets.fields:
        jets['run'] = ak.broadcast_arrays(run, jets.pt)[0]
    return jets

def prepare_corrt1_jets(corrt1_jets, events_rho, run=None):
    corrt1_jets = ak.with_field(corrt1_jets, ak.broadcast_arrays(events_rho, corrt1_jets.rawPt)[0], 'rhoFixedGridFastJetAll')
    if run is not None:
        corrt1_jets = ak.with_field(corrt1_jets, ak.broadcast_arrays(run, corrt1_jets.rawPt)[0], 'run')
    return corrt1_jets

def evaluate(correction, values):
    return correction.evaluate(*[values[inp.name] for inp in correction.inputs])

def shift_jets(jets, factor, pt, mass):
    return ak.with_field(ak.with_field(jets, factor * pt, 'pt'), factor * mass, 'mass')

def jersmear_factors(pt_jec, eta, rho, event_id, pt_gen, jer, jersf, jer_smear, genmatched_only_eta, match_sigma=None):
    values = {
        'JetPt': pt_jec,
        'JetEta': eta,
        'Rho': rho,
        'EventID': ak.without_parameters(ak.broadcast_arrays(ak.values_astype(event_id, np.int64), pt_jec)[0]),
    }
    values['JER'] = evaluate(jer, values)
    window = values['JER'] if match_sigma is None else match_sigma
    matched = (pt_gen > 0) & (np.abs((pt_jec - pt_gen) / pt_jec) < 3 * window)
    values['GenPt'] = ak.where(matched, pt_gen, -1.0)
    unsmeared = abs_eta_in_regions(eta, genmatched_only_eta) & ~matched
    smear = {}
    for variation in ('nom', 'up', 'down'):
        values['systematic'] = variation
        values['JERSF'] = evaluate(jersf, values)
        factor = evaluate(jer_smear, values)
        smear[variation] = ak.where(unsmeared | ~((factor > 0) & np.isfinite(factor)), 1.0, factor)
    return values['JER'], smear

def jersmear_jets(injets, jets, event_id, jer, jersf, jer_smear, jes_sources, genmatched_only_eta):
    sigma, smear = jersmear_factors(jets.pt_jec, jets.eta, jets.rhoFixedGridFastJetAll, event_id, jets.pt_gen, jer, jersf, jer_smear, genmatched_only_eta)
    template = ak.with_field(ak.with_field(injets, injets.pt, 'pt_orig'), injets.mass, 'mass_orig')
    out = ak.with_field(jets, sigma, 'jet_energy_resolution')
    out = ak.with_field(out, smear['nom'], 'jet_energy_resolution_correction')
    out = shift_jets(out, smear['nom'], jets.pt_jec, jets.mass_jec)
    out = ak.with_field(out, out.pt, 'pt_jer')
    out = ak.with_field(out, out.mass, 'mass_jer')
    jer_variants = {
        d: shift_jets(ak.with_field(template, smear[d], 'jet_energy_resolution_correction'), smear[d], jets.pt_jec, jets.mass_jec)
        for d in ('up', 'down')
    }
    out = ak.with_field(out, ak.zip(jer_variants, depth_limit=2, with_name='JetSystematic'), 'JER')
    for name, source in jes_sources.items():
        delta = evaluate(source, {'JetEta': out.eta, 'JetPt': out.pt_jer})
        variants = {
            'up': shift_jets(template, 1 + delta, out.pt_jer, out.mass_jer),
            'down': shift_jets(template, 1 - delta, out.pt_jer, out.mass_jer),
        }
        out = ak.with_field(out, ak.zip(variants, depth_limit=2, with_name='JetSystematic'), f'JES_{name}')
    return out

def nearest_gen_pt(jets, gen_jets, max_dr):
    reco, gen = ak.unzip(ak.cartesian(
        [ak.zip({'eta': jets.eta, 'phi': jets.phi}), ak.zip({'eta': gen_jets.eta, 'phi': gen_jets.phi, 'pt': gen_jets.pt})],
        nested=True,
    ))
    dr = np.hypot(reco.eta - gen.eta, (reco.phi - gen.phi + np.pi) % (2 * np.pi) - np.pi)
    best = ak.argmin(dr, axis=2, keepdims=True)
    return ak.fill_none(ak.firsts(ak.where(dr[best] < max_dr, gen.pt[best], 0.0), axis=2), 0.0)

def jersmear_corrt1_jets(corrt1_jets, event_id, gen_jets, jec, max_gen_dr, jer, jersf, jer_smear, match_rho_max, genmatched_only_eta):
    inputs = {
        'JetA': corrt1_jets.area,
        'JetEta': corrt1_jets.eta,
        'JetPt': corrt1_jets.rawPt,
        'Rho': corrt1_jets.rhoFixedGridFastJetAll,
        'JetPhi': corrt1_jets.phi,
    }
    pt_jec = corrt1_jets.rawPt * jec.getCorrection(**{name: inputs[name] for name in jec.signature})
    pt_gen = nearest_gen_pt(corrt1_jets, gen_jets, max_gen_dr)
    match_sigma = evaluate(jer, {'JetEta': corrt1_jets.eta, 'JetPt': pt_jec, 'Rho': np.minimum(corrt1_jets.rhoFixedGridFastJetAll, match_rho_max)})
    _, smear = jersmear_factors(pt_jec, corrt1_jets.eta, corrt1_jets.rhoFixedGridFastJetAll, event_id, pt_gen, jer, jersf, jer_smear, genmatched_only_eta, match_sigma)
    out = ak.with_field(corrt1_jets, smear['nom'], 'jet_energy_resolution_correction')
    variants = {d: ak.zip({'jet_energy_resolution_correction': smear[d]}) for d in ('up', 'down')}
    return ak.with_field(out, ak.zip(variants, depth_limit=2), 'JER')

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
            coffea_console.print("[red]JER disabled for AK4PFPuppi jets due to broken text file format, need correctionlib version to replace it[/red]")


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
        self.max_gen_dr = None
        self.jer_smear = None
        self.jec_needs_run = False


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
        data_type = "MC" if is_mc else "DATA"
        mitigations = jetmet_mitigations(era)
        if clibhandler.getPath("jet_jerc", fallbackNone=True) is None:
            raise ValueError("jet_jerc correction set not registered in the CorrectionlibHandler")
        cset = clibhandler.getCorrectionSet("jet_jerc")
        if cset is None:
            raise ValueError(f"jet_jerc correction set could not be loaded from {str(clibhandler.getPath('jet_jerc'))}")
        # Use the central code added here: https://github.com/scikit-hep/coffea/pull/1521
        floor = None if is_mc else mitigations['residual_pt_floor']
        if floor is None:
            jec = CorrectionLibJEC(cset.compound[f"{jec_tag}_{data_type}_L1L2L3Res_{jet_type}"])
        else:
            jec = ResidualPtFloorJEC(
                [cset[f"{jec_tag}_DATA_{level}_{jet_type}"] for level in ("L1FastJet", "L2Relative", "L3Absolute")],
                cset[f"{jec_tag}_DATA_L2L3Residual_{jet_type}"],
                *floor,
            )
        jec_L1 = CorrectionLibJEC(cset[f"{jec_tag}_{data_type}_L1FastJet_{jet_type}"])
        # Regrouped set of uncertainties
        if is_mc:
            sources = {
                name: cset[f"{jec_tag}_MC_Regrouped_{name}_{jet_type}"]
                for name in [
                    "Total",
                    "Absolute",
                    f"Absolute_{era}",
                    "BBEC1",
                    f"BBEC1_{era}",
                    "EC2",
                    f"EC2_{era}",
                    "FlavorQCD",
                    "HF",
                    f"HF_{era}",
                    "RelativeBal",
                    f"RelativeSample_{era}",
                ]
            }
            junc = CorrectionLibJUNC(list(sources.items()))
            jes_uncertainties = {f"JES_{name}": CorrectionLibJEC(corr) for name, corr in sources.items()}
        else:
            junc = None
            jes_uncertainties = None
        if is_mc and doJER:
            if clibhandler.getPath("jer_smear", fallbackNone=True) is None:
                raise ValueError("jer_smear correction set not registered in the CorrectionlibHandler")
            smear_cset = clibhandler.getCorrectionSet("jer_smear")
            if smear_cset is None:
                raise ValueError(f"jer_smear correction set could not be loaded from {str(clibhandler.getPath('jer_smear'))}")
            self.jer_smear = (
                cset[f"{jer_tag}_MC_PtResolution_{jet_type}"],
                cset[f"{jer_tag}_MC_ScaleFactor_{jet_type}"],
                smear_cset["JERSmear"],
                sources,
                mitigations['jer_genmatched_only_eta'],
            )
            self.corrt1_match_rho_max = np.nextafter(np.float64(JER_RESOLUTION_RHO_MAX[jer_tag]), -np.inf)
        else:
            self.jer_smear = None
        self.jec_stack_L123_JER = CorrectionLibJECStack(jec=jec, junc=junc if self.jer_smear is None else None)
        self.jec_factory_L123_JER = CorrectedJetsFactory(clib_name_map, self.jec_stack_L123_JER)
        self.jec_stack_L123_noJER = CorrectionLibJECStack(jec=jec, junc=junc)
        self.jec_factory_L123_noJER = CorrectedJetsFactory(clib_name_map, self.jec_stack_L123_noJER)
        self.met_factory = CorrectedMETFactory(
            clib_name_map,
            jec_L1L2L3=jec,
            jec_L1=jec_L1,
            jes_uncertainties=jes_uncertainties,
            # Nominal Type-1 MET includes the JER smearing of both Jet and CorrT1METJet jets
            is_t1_smeared_met=self.jer_smear is not None,
        )
        # JER hybrid method: gen-jet match requires dR < R/2 (AK4); NanoAOD genJetIdx matches up to dR=0.4
        self.max_gen_dr = 0.2
        self.jec_needs_run = "run" in jec.signature
        

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
    
    def corrected_jets_L123_JER(self, jets, event_rho, lazy_cache, run=None, event_id=None):
        if self.jer_smear is not None and event_id is None:
            raise ValueError("event_id is required for JERSmear jet smearing")
        jets = add_jme_variables(jets, event_rho, run=run if self.jec_needs_run else None, max_gen_dr=self.max_gen_dr)
        corrected = self.jec_factory_L123_JER.build(
            jets,
            # lazy_cache
        )
        if self.jer_smear is None:
            return corrected
        return jersmear_jets(jets, corrected, event_id, *self.jer_smear)

    def corrected_jets_L123_noJER(self, jets, event_rho, lazy_cache, run=None):
        jets = add_jme_variables(jets, event_rho, run=run if self.jec_needs_run else None, max_gen_dr=self.max_gen_dr)
        #jets['pt'] = jets['pt'] * (1 - jets.muonSubtrFactor)
        return self.jec_factory_L123_noJER.build(
            jets,
            # lazy_cache
        )
      #remove jets_L1
    # Replace this with the coffea updated implementation eventually: https://github.com/scikit-hep/coffea/pull/1537
    def corrected_met(self, met, jets_L123, event_rho, lazy_cache, raw_met=None, corrt1_jets=None, run=None, event_id=None, gen_jets=None):
        if self.met_factory.type1_mode:
            if corrt1_jets is not None:
                if self.jer_smear is not None and (event_id is None or gen_jets is None):
                    raise ValueError("event_id and gen_jets are required for JERSmear CorrT1METJet smearing")
                corrt1_jets = prepare_corrt1_jets(corrt1_jets, event_rho, run if self.jec_needs_run else None)
                if self.jer_smear is not None:
                    jer, jersf, jer_smear, _, genmatched_only_eta = self.jer_smear
                    corrt1_jets = jersmear_corrt1_jets(corrt1_jets, event_id, gen_jets, self.met_factory.jec_L1L2L3, self.max_gen_dr, jer, jersf, jer_smear, self.corrt1_match_rho_max, genmatched_only_eta)
            return self.met_factory.build(met, jets_L123, in_RawMET=raw_met, in_CorrT1METJets=corrt1_jets)
        emFraction = jets_L123.chEmEF + jets_L123.neEmEF
        mask_jec = (jets_L123['pt'] > 15) & (emFraction <= 0.9)
        jets_L123_cleaned_for_MET = jets_L123[mask_jec]
        #jets_L123_cleaned_for_MET['pt'] = jets_L123_cleaned_for_MET['pt'] * (1 - jets_L123_cleaned_for_MET.muonSubtrFactor)
        return self.met_factory.build(
            met,
            jets_L123_cleaned_for_MET,
            # lazy_cache=lazy_cache
        )
