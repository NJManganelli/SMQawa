from coffea.lookup_tools import dense_lookup
from coffea.analysis_tools import Weights
from coffea import processor
from scipy import interpolate
from coffea.nanoevents.methods import candidate
from coffea.jetmet_tools.CorrectedMETFactory import corrected_polar_met

import awkward as ak
import numpy as np
import uproot
import os
import re

# These are all the pdf sets we got in all our UL samples
_lhapdf_config = {
    306000: {
        "variation_type": "hessian",
        "central_pdf_id": 0,
        "alphas_indices": [101, 102]
    }, 
    325300: {
        "variation_type": "hessian", 
        "central_pdf_id": 0,
        "alphas_indices": [101, 102]
    },
    320900: {
        "variation_type": "mc", 
        "central_pdf_id": 0,
        "alphas_indices": []
    },
    325500: {
        "variation_type": "hessian", 
        "central_pdf_id": 0,
        "alphas_indices": []
    }

}
NANOAOD_VERSION_RE = re.compile(r"NANO(?:AOD)?(?:APV)?v(\d+)", re.IGNORECASE)
def nanoaod_version(path: str) -> str | None:
    m = NANOAOD_VERSION_RE.search(path)
    if m:
        return f"v{m.group(1)}"
    if re.search(r"Run2025[A-Z]/.+/NANOAOD/PromptReco", path):
        return "v15"  # 2025 prompt NanoAOD is produced as v15
    return None    


def met_phi_xy_correction(met, run, npv, is_mc:bool=False, era:str='2016', jet_type:str=None, clibhandler=None):
    if clibhandler is not None:
        corrector = clibhandler.getCorrectionSet("met")
        pt_key, phi_key = None, None
        if jet_type == "AK4CHS":
            pt_key = f"pt_metphicorr_pfmet_{'mc' if is_mc else 'data'}"
            phi_key = f"pt_metphicorr_pfmet_{'mc' if is_mc else 'data'}"
        elif jet_type == "AK4PUPPI":
            pt_key = f"pt_metphicorr_puppimet_{'mc' if is_mc else 'data'}"
            phi_key = f"pt_metphicorr_puppimet_{'mc' if is_mc else 'data'}"
        else:
            raise ValueError(f"Unexpected jet_type in met_phi_xy_correction when clibhandler has been enabled: {jet_type}")
        pt_ = corrector[pt_key].evaluate(met.pt, met.phi, npv, run)
        phi_ = corrector[phi_key].evaluate(met.pt, met.phi, npv, run)
        met['pt'] = pt_
        met['phi'] = phi_
        return met
        
    xcor = ak.ones_like(run)
    ycor = ak.ones_like(run)
    
    # UL2016
    if is_mc:
        if '2016' in era:
            if('APV' in era):  
                xcor = -(-0.188743*npv +0.136539)
                ycor = -(0.0127927*npv +0.117747);
            else:
                xcor = -(-0.15349700*npv -0.231751)
                ycor = -( 0.00731978*npv +0.243323)
        if '2018' in era:
            xcor = -(0.183518*npv +0.546754)
            ycor = -(0.192263*npv -0.421210)
        if '2017' in era:
            xcor = -(-0.300155*npv +1.90608)
            ycor = -( 0.300213*npv -2.02232)
            
    else:
        # UL2016
        xcor = ak.where((run >= 272007) & (run <= 275376), -(-0.0214894*npv -0.188255), xcor) # UL2016B
        ycor = ak.where((run >= 272007) & (run <= 275376), -( 0.0876624*npv +0.812885), ycor) # UL2016B
        xcor = ak.where((run >= 275657) & (run <= 276283), -(-0.0322090*npv +0.067288), xcor) # UL2016C
        ycor = ak.where((run >= 275657) & (run <= 276283), -( 0.1139170*npv +0.743906), ycor) # UL2016C
        xcor = ak.where((run >= 276315) & (run <= 276811), -(-0.0293663*npv +0.211060), xcor) # UL2016D
        ycor = ak.where((run >= 276315) & (run <= 276811), -( 0.1133100*npv +0.815787), ycor) # UL2016D
        xcor = ak.where((run >= 276831) & (run <= 277420), -(-0.0132046*npv +0.200730), xcor) # UL2016E
        ycor = ak.where((run >= 276831) & (run <= 277420), -( 0.1348090*npv +0.679068), ycor) # UL2016E
        xcor = ak.where((run >= 277772) & (run <= 278768) & (run==278770), -(-0.0543566*npv +0.816597), xcor) # UL2016F
        ycor = ak.where((run >= 277772) & (run <= 278768) & (run==278770),-( 0.1142250*npv +1.172660), ycor) # UL2016F
        xcor = ak.where(
            (run >= 278801) & (run <= 278808) & (run==278769),
            -( 0.1346160*npv -0.899650), xcor) # UL2016Flate
        ycor = ak.where(
            (run >= 278801) & (run <= 278808) & (run==278769),
            -( 0.0397736*npv +1.038500), ycor) # UL2016Flate
        xcor = ak.where((run >= 278820) & (run <= 280385), -( 0.1218090*npv -0.584893), xcor) #UL2016G
        ycor = ak.where((run >= 278820) & (run <= 280385), -( 0.0558974*npv +0.891234), ycor) #UL2016G
        xcor = ak.where((run >= 280919) & (run <= 284044), -( 0.0868828*npv -0.703489), xcor) #UL2016H
        ycor = ak.where((run >= 280919) & (run <= 284044), -( 0.0888774*npv +0.902632), ycor) #UL2016H

        # UL2017
        xcor = ak.where((run >= 297020) & (run <= 299329), -(-0.211161*npv +0.419333), xcor)
        ycor = ak.where((run >= 297020) & (run <= 299329), -( 0.251789*npv -1.280890), ycor)
        xcor = ak.where((run >= 299337) & (run <= 302029), -(-0.185184*npv -0.164009), xcor)
        ycor = ak.where((run >= 299337) & (run <= 302029), -( 0.200941*npv -0.568530), ycor)
        xcor = ak.where((run >= 302030) & (run <= 303434), -(-0.201606*npv +0.426502), xcor)
        ycor = ak.where((run >= 302030) & (run <= 303434), -( 0.188208*npv -0.583130), ycor)
        xcor = ak.where((run >= 303435) & (run <= 316995), -(-0.162472*npv +0.176329), xcor)
        ycor = ak.where((run >= 303435) & (run <= 316995), -( 0.138076*npv -0.250239), ycor)
        xcor = ak.where((run >= 304911) & (run <= 316995), -(-0.210639*npv +0.729340), xcor)
        ycor = ak.where((run >= 304911) & (run <= 316995), -( 0.198626*npv +1.028000), ycor)
        
        # UL2018
        xcor = ak.where((run >= 315252) & (run <= 316995), -(0.263733*npv -1.91115), xcor)
        ycor = ak.where((run >= 315252) & (run <= 316995), -(0.0431304*npv -0.112043), ycor)
        xcor = ak.where((run >= 316998) & (run <= 319312), -(0.400466*npv -3.05914), xcor)
        ycor = ak.where((run >= 316998) & (run <= 319312), -(0.146125*npv -0.533233), ycor)
        xcor = ak.where((run >= 319313) & (run <= 320393), -(0.430911*npv -1.42865), xcor)
        ycor = ak.where((run >= 319313) & (run <= 320393), -(0.0620083*npv -1.46021), ycor)
        xcor = ak.where((run >= 320394) & (run <= 325273), -(0.457327*npv -1.56856), xcor)
        ycor = ak.where((run >= 320394) & (run <= 325273), -(0.0684071*npv -0.928372), ycor)
        
    shifts_met = [item for item in dir(met) if 'JES' in item or 'JE' in item or 'MET' in item]
    for s in shifts_met:
        met_shift = getattr(met, s, None)
        metx_down=getattr(met, s, None).down.pt * np.cos(met.phi)+xcor
        mety_down=getattr(met, s, None).down.pt * np.sin(met.phi)+ycor
        pt_down = np.sqrt((metx_down**2)+(mety_down**2))
        phi_down = np.arctan2(mety_down,metx_down)
        met_shift = ak.with_field(met_shift, pt_down, ['down', 'pt'])
        met_shift = ak.with_field(met_shift, phi_down, ['down', 'phi'])
        
        metx_up=getattr(met, s, None).up.pt * np.cos(met.phi)+xcor
        mety_up=getattr(met, s, None).up.pt * np.sin(met.phi)+ycor
        pt_up = np.sqrt((metx_up**2)+(mety_up**2))
        phi_up = np.arctan2(mety_up,metx_up)
        met_shift = ak.with_field(met_shift, pt_up, ['up', 'pt'])
        met_shift = ak.with_field(met_shift, phi_up, ['up', 'phi'])
        met = ak.with_field(met, met_shift, s)
    metx_ = met.pt * np.cos(met.phi)+xcor
    mety_ = met.pt * np.sin(met.phi)+ycor

    pt_ = np.sqrt((metx_**2)+(mety_**2))
    phi_ = np.arctan2(mety_,metx_)
    
    met['pt'] = pt_
    met['phi'] = phi_
    return met

def trigger_rules(event, rules:dict, era:str='2018'):
    ds_names_ = {
        '2016' : ['DoubleMuon', 'SingleMuon', 'DoubleEG', 'SingleElectron', 'MuonEG'],
        '2017' : ['DoubleMuon', 'SingleMuon', 'DoubleEG', 'SingleElectron', 'MuonEG'],
        '2018' : ['DoubleMuon', 'SingleMuon', 'EGamma', 'MuonEG'],
        '2024' : ['Muon', 'EGamma', 'MuonEG'],
        '2025' : ['Muon', 'EGamma', 'MuonEG'],
        '2026' : ['Muon', 'EGamma', 'MuonEG'],
    }
    
    _pass = np.zeros(len(event), dtype='bool')
    _veto = np.zeros(len(event), dtype='bool')
    
    _ds = event.metadata['dataset']
    ds_name = ''
    for s in ds_names_[era]:
        if s in _ds:
            ds_name = s
            break
        
    for t in rules[f'Run{era}.{ds_name}']['pass']:
        if t in event.HLT.fields:
            _pass = _pass | event.HLT[t]
        else:
            print(f'{t} not present in the data')
            
    for t in rules[f'Run{era}.{ds_name}']['veto']:
        if t in event.HLT.fields:
            _veto =  _veto | event.HLT[t]
            
    # passing triggers and vetoing triggers from other datasets
    return _pass & ~_veto

# def theory_pdf_weight(weights, pdf_weight):
#     _nm = np.ones(len(weights.weight()))
#     _up = np.ones(len(weights.weight()))
#     _dw = np.ones(len(weights.weight()))

#     if pdf_weight is not None and (("306" in pdf_weight.__doc__) or ("325" in pdf_weight.__doc__)):
#         arg = pdf_weight[:, 1:-2] - np.ones((len(weights.weight()), 100))
#         summed  = ak.sum(np.square(arg), axis=1)
#         pdf_unc = np.sqrt((1. / 99.) * summed)
#         weights.add('PDF_weight', _nm, pdf_unc + _nm)

#         # alpha_S weights
#         # Eq. 27 of same ref
#         as_unc = 0.5 * (pdf_weight[:, 102] - pdf_weight[:, 101])
#         weights.add('aS_weight', _nm, as_unc + _nm)

#         # PDF + alpha_S weights
#         # Eq. 28 of same ref
#         pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
#         weights.add('PDFaS_weight', _nm, pdfas_unc + _nm)
#     else:
#         weights.add('aS_weight'   , _nm, _up, _dw)
#         weights.add('PDF_weight'  , _nm, _up, _dw)
#         weights.add('PDFaS_weight', _nm, _up, _dw)

def theory_pdf_weight(weights, pdf_weight):
    _nm = np.ones(len(weights.weight()))
    _up = np.ones(len(weights.weight()))
    _dw = np.ones(len(weights.weight()))
    
    match = re.findall(r"(LHA\s+IDs\s+(\d+)\s*-\s*(\d+)\b)", pdf_weight.__doc__)
    if len(match) == 0:
        print("[warning] sample has no PDF weights stored. setting aS/PDF_weight to 1.0")
        weights.add('aS_weight'   , _nm, _up, _dw)
        weights.add('PDF_weight'  , _nm, _up, _dw)
        return

    lhapdf_id_range = np.array(match[0][1:]).astype(int)
    lhapdf_id = lhapdf_id_range[0]
    
    variation_type = _lhapdf_config[lhapdf_id]['variation_type']
    alphas_indices = _lhapdf_config[lhapdf_id]['alphas_indices']
    
    n = np.diff(lhapdf_id_range)[0]-2 if len(alphas_indices) else np.diff(lhapdf_id_range)[0]
    
    weight = pdf_weight[:, 1:-2] if len(alphas_indices) else pdf_weight[:, 1:]
    weight = weight * np.ones((len(weights.weight()), n))
        
    mean = ak.sum(weight, axis=1)/n
    sumw2 = ak.sum(np.square(weight), axis=1)
    sumdiff2 = sumw2 - n * np.power(mean, 2)

    if variation_type == "hessian":
        weights.add('PDF_weight', _nm, np.sqrt(sumdiff2) + _nm)
        if len(alphas_indices):
            as_unc = 0.5 * (pdf_weight[:, alphas_indices[1]] - pdf_weight[:, alphas_indices[0]])
            weights.add('aS_weight', _nm, as_unc + _nm)
        else:
            weights.add('aS_weight', _nm, _up, _dw)
        return 
            
    else:
        weights.add('PDF_weight', _nm, np.sqrt(sumdiff2/(n - 1)) + _nm)
        weights.add('aS_weight', _nm, _up, _dw)
        return
        

def theory_ps_weight(weights, ps_weight):
    nominal= np.ones(len(weights.weight()))
    up_isr = np.ones(len(weights.weight()))
    up_fsr = np.ones(len(weights.weight()))
    dw_isr = np.ones(len(weights.weight()))
    dw_fsr = np.ones(len(weights.weight()))

    if ps_weight is not None:
        if len(ps_weight[0]) == 4:
            up_isr = ps_weight[:, 0]
            dw_isr = ps_weight[:, 2]
            up_fsr = ps_weight[:, 1]
            dw_fsr = ps_weight[:, 3]

    weights.add('UEPS_ISR', nominal, up_isr, dw_isr)
    weights.add('UEPS_FSR', nominal, up_fsr, dw_fsr)
    

def transverse_energy(fourmomentum):
    # Direct computation
    if hasattr(fourmomentum, "et"):
        # scikit-hep/vector
        return fourmomentum.et
    # E_t = E * pt / p  (but guard against p==0).
    if hasattr(fourmomentum, "E") and hasattr(fourmomentum, "pt") and hasattr(fourmomentum, "p"):
        # coffea.vector
        p = fourmomentum.p
        # avoid division by zero; where p==0 use pt (so E*pt/p -> pt)
        return fourmomentum.E * fourmomentum.pt / ak.where(p == 0, ak.ones_like(p), p)
    if hasattr(fourmomentum, "t") and hasattr(fourmomentum, "pt") and hasattr(fourmomentum, "p"):
        # buggy coffea.vector
        p = fourmomentum.p
        # avoid division by zero; where p==0 use pt (so E*pt/p -> pt)
        return fourmomentum.t * fourmomentum.pt / ak.where(p == 0, ak.ones_like(p), p)
    raise AttributeError("Cannot compute transverse energy from object; "
                         "no .E/.pt/.p attributes found")


class pileup_weights:
    def __init__(self, do_syst:bool=True, era:str='2018', clibhandler=None, clibkey:str=None):
        if clibhandler is None:
            self.__init_legacy__(do_syst=do_syst, era=era)
            self.clib = None
        else:
            self.__init_clib__(era=era, clibhandler=clibhandler, clibkey=clibkey)

    def __init_clib__(self, era:str='2018', clibhandler=None, clibkey=None):
        cset = clibhandler.getCorrectionSet("puWeights")
        if clibkey is None:
            all_keys = list(cset.keys())
            if len(all_keys) == 1:
                clibkey = all_keys[0]
            else:
                raise ValueError(f"clibkey must be specified, available set of corrections: {all_keys} [CompoundCorrections: {list(cset.compound.keys())}")
        self.clib = cset[clibkey]
        
    def __init_legacy__(self, do_syst:bool=True, era:str='2018'):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/PU/')
        files_  = {
            "Nom" : f"{_data_path}/PileupHistogram-goldenJSON-13tev-{era}-69200ub-99bins.root",
            "Up"  : f"{_data_path}/PileupHistogram-goldenJSON-13tev-{era}-72400ub-99bins.root",
            "Down": f"{_data_path}/PileupHistogram-goldenJSON-13tev-{era}-66000ub-99bins.root"
        }

        hist_norm = lambda x: np.divide(x, x.sum())

        self.simu_pu = None 
        with uproot.open(f'{_data_path}/mcPileupUL{era}.root') as ifile:
            self.simu_pu = ifile['pu_mc'].values()
            self.simu_pu = np.insert(self.simu_pu,0,0.)
            self.simu_pu = self.simu_pu[:-2]

        mask = self.simu_pu > 0

        self.corrections = {}
        for var,pfile in files_.items():
            with uproot.open(pfile) as ifile:
                data_pu = hist_norm(ifile["pileup"].values())
                edges = ifile["pileup"].axis().edges()

                corr = np.divide(data_pu, self.simu_pu, out=np.ones_like(data_pu),where=mask)
                pileup_corr = dense_lookup.dense_lookup(corr, edges)
                self.corrections['puWeight' if 'Nom' in var else f'puWeight{var}'] = pileup_corr
    
    
    def append_pileup_weight(self, weights, pu):
        weights.add(
            'pileup_weight',
            self.clib.evaluate(pu, 'nominal') if self.clib else self.corrections['puWeight'    ](pu),
            self.clib.evaluate(pu, 'up'     ) if self.clib else self.corrections['puWeightUp'  ](pu),
            self.clib.evaluate(pu, 'down'   ) if self.clib else self.corrections['puWeightDown'](pu),
        )
        return weights
    
class LinearNDInterpolatorExt(object):
    """ 
    LinearNDInterpolator with possibility to extralopate
    outside the range by using Nearest neighbour.
    """
    def __init__(self, points, values):
        self.funcinterp = interpolate.LinearNDInterpolator (points, values)
        self.funcneares = interpolate.NearestNDInterpolator(points, values)
    
    def __call__(self, *args):
        z = self.funcinterp(*args)
        h = self.funcneares(*args)
        chk = np.isnan(z)
        if chk.any():
            return np.where(chk, h, z)
        else:
            return z
    
class ewk_corrector:
    def __init__(self, process:str='ZZ', beam_energy:float = 6500, use_inv_masses:bool=False):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        self.corr_file = f"{_data_path}/ewk/data_{process}_EwkCorrections.dat"
        self.corr_data = np.loadtxt(self.corr_file)
        self.energy = beam_energy
        self.process = process
        self.use_inv_masses = use_inv_masses
        self.m_z = 91.1876
        self.m_w = 80.385 if process=='WZ' else self.m_z
        
        self.m_z2 = self.m_z**2
        self.m_w2 = self.m_w**2
        
        self.exterp = []
        for iv in range(3):
            self.exterp.append(
               LinearNDInterpolatorExt(
                   list(zip(self.corr_data[:,0], self.corr_data[:,1])), 
                   self.corr_data[:,2 + iv]
                )
            )
            
        self.corrNNLO = np.array([
            1.513834489150, #  0
            1.541738780180, #  1
            1.497829632510, #  2
            1.534956782920, #  3
            1.478217033060, #  4
            1.504330859290, #  5
            1.520626246850, #  6
            1.507013090030, #  7
            1.494243156250, #  8
            1.450536096150, #  9
            1.460812521660, # 10
            1.471603622200, # 11
            1.467700038200, # 12
            1.422408690640, # 13
            1.397184022730, # 14
            1.375593447520, # 15
            1.391901318370, # 16
            1.368564350560, # 17
            1.317884804290, # 18
            1.314019950800, # 19
            1.274641749910, # 20
            1.242346606820, # 21
            1.244727403840, # 22
            1.146259351670, # 23
            1.107804993520, # 24
            1.042053646740, # 25
            0.973608545141, # 26
            0.872169942668, # 27
            0.734505279177, # 28
            1.163152837230, # 29
            1.163152837230, # 30
            1.163152837230  # 31
        ])
        
        
    def get_weight(self, gen_coll, x1, x2, weights=None):
        # float32 precision is insufficient and leads to infinities in calculations, replace key p4/Energy fields with float64 parameters
        x1 = x1 * np.float64([1.0])
        x2 = x2 * np.float64([1.0])
        gen_coll = ak.with_field(gen_coll, gen_coll.pt * np.float64([1.0]), "pt")
        gen_coll = ak.with_field(gen_coll, gen_coll.eta * np.float64([1.0]), "eta")
        gen_coll = ak.with_field(gen_coll, gen_coll.phi * np.float64([1.0]), "phi")
        gen_coll = ak.with_field(gen_coll, gen_coll.mass * np.float64([1.0]), "mass")

        id_q1 = np.abs(gen_coll.pdgId[:,0])
        id_q2 = np.abs(gen_coll.pdgId[:,1])
        
        # make sure we have quark in the initial state
        init_q_mask = (
            (((id_q1>=1) & (id_q1<=6)) | (id_q1==21)) &
            (((id_q2>=1) & (id_q2<=6)) | (id_q2==21))
        )
        # make sure we have vector bosons in the events   
        vect_v_mask = (
            ((np.abs(gen_coll.pdgId[:,2])==23) | (np.abs(gen_coll.pdgId[:,2]) == 24)) &
            ((np.abs(gen_coll.pdgId[:,3])==23) | (np.abs(gen_coll.pdgId[:,3]) == 24))
        )
        # apply corrections only on same flavour and non-gluon initiate processes        
        q1 = ak.zip({
                "x":  ak.zeros_like(x1), 
                "y":  ak.zeros_like(x1),
                "z":  x1 * self.energy,
                "t":  x1 * self.energy
            },
            with_name="Candidate",
            behavior=candidate.behavior
        )

        q2 = ak.zip(
            {
                "x":  ak.zeros_like(x2), 
                "y":  ak.zeros_like(x2),
                "z": -x2 * self.energy,
                "t":  x2 * self.energy
            },
            with_name="Candidate",
            behavior=candidate.behavior,
        )

        v1 = gen_coll[:,2]
        v2 = gen_coll[:,3]
        
        # getting the s hat
        vv = v1 + v2
        shat = vv.mass2

        # getting the t hat
        # Boost to the VV center of mass frame
        b_q1 = q1.boost( -vv.boostvec )
        b_q2 = q2.boost( -vv.boostvec )
        b_v1 = v1.boost( -vv.boostvec )

        # Unitary vectors
        uq1 = (b_q1.pvec / b_q1.pvec.p )
        uq2 = (b_q2.pvec / b_q2.pvec.p )
        uv1 = (b_v1.pvec / b_v1.pvec.p )

        # effectiuve beam axis
        diff_q = uq1 - uq2
        beam_eff_axis = diff_q/diff_q.p

        # cos theta for the effective beam axis
        costheta = beam_eff_axis.dot(uv1)
        that = 0.0 
        if self.use_inv_masses:
            b = 1./2./np.sqrt(shat) * np.sqrt(
                np.power(shat-v1.mass2-v2.mass2,2) - 4*v1.mass2*v2.mass2
            )
            a = np.sqrt(b*b + v1.mass2)
            that = v1.mass2 - np.sqrt(shat) * (a - b * costheta)
        else:
            b = 1./2./np.sqrt(shat) * np.sqrt(
                np.abs(
                    np.power(shat-self.m_z2-self.m_w2,2) - 4*self.m_w2*self.m_z2
                )
            );
            a = np.sqrt(b*b + self.m_z2);
            that = self.m_z2 - np.sqrt(shat) * (a - b * costheta)
    
    
        onshell_mask = np.sqrt(shat) < self.m_w + self.m_z
        quark_type = np.zeros_like(shat)
        
        quark_type = np.minimum(id_q1, id_q2)

        shat = ak.nan_to_num(shat, 1.0)
        that = ak.nan_to_num(that, 1.0)

        corr_0 = 1 + self.exterp[0](np.sqrt(shat.to_numpy()), that.to_numpy())
        corr_1 = 1 + self.exterp[1](np.sqrt(shat.to_numpy()), that.to_numpy())
        corr_2 = 1 + self.exterp[2](np.sqrt(shat.to_numpy()), that.to_numpy())
        
        # make sure the corrections are only on the on-shell
        corr_mask = (
            init_q_mask & 
            vect_v_mask &
            onshell_mask
        )

        corr_0 = np.where(corr_mask, np.ones_like(corr_0), corr_0)
        corr_1 = np.where(corr_mask, np.ones_like(corr_0), corr_1)
        corr_2 = np.where(corr_mask, np.ones_like(corr_0), corr_2)
    
        # correction by flavour
        weight = np.where(
            (quark_type == 1) | (quark_type == 3), 
            corr_1,
            np.where(
                (quark_type == 2) | (quark_type == 4), 
                corr_0,
                np.where(
                    (quark_type == 5), 
                    corr_2,
                    1.0
                )
            )
        )
        
        # NNLO weight
        knnlo = ak.ones_like(corr_0)
        if self.process == 'ZZ':
            delta_phi_vv = v1.delta_phi(v2)
            nnlo_ibin = np.round(delta_phi_vv.to_numpy() * 10.).astype(np.int32)
            knnlo =  np.where(
                init_q_mask & vect_v_mask, 
                self.corrNNLO[nnlo_ibin], 
                1.0
            )
            
        # Average QCD NLO k factors from arXiv:1105.0020
        leptons = gen_coll[ 
            ((gen_coll.statusFlags & 128)!=0) & 
            (np.abs(gen_coll.pdgId) >= 11) & 
            (np.abs(gen_coll.pdgId) <= 16)
        ]
        rhovv = leptons.sum().pt/ak.sum(leptons.pt, axis=1)
        
        qcd_kfactor = ak.zeros_like(rhovv)
        if self.process == 'ZZ':
            qcd_kfactor = 15.99/ 9.89 - ak.ones_like(rhovv)# ZZ from arXiv1105.0020
        else:
            qcd_kfactor = np.where(
                (gen_coll.pdgId[:,2]*gen_coll.pdgId[:,3]) > 0, 
                28.55 / 15.51 - ak.ones_like(rhovv), # W+Z
                18.19 /  9.53 - ak.ones_like(rhovv), # W-Z
            )

        ewk_uncert = np.where(
            rhovv < 0.3, 1 + np.abs((weight-1.0)*(qcd_kfactor-1.0)), np.abs(weight)
        )
        
        # WZ: gamma-induced contribution
        if self.process=='WZ':
            gamma_ind_corr = np.where(
                ((gen_coll.pdgId[:,2]*gen_coll.pdgId[:,3]) > 0), 
                (1 + 0.00559445 - 5.17082e-6 * np.sqrt(shat) + 3.63331e-8 * shat), 
                (1 + 0.00174737 + 1.70668e-5 * np.sqrt(shat) + 2.26398e-8 * shat) 
            )
            weight = weight * np.where(corr_mask, gamma_ind_corr, 1.0)
            
        # filling the weights
        if weights is not None:
            weights.add('kEW', weight, weight*ewk_uncert, weight/ewk_uncert)
            weights.add('kNNLO', knnlo)
            
def propagate_shift_to_met(sink, events, met, shifted_collection, unc_type=None, is_correction=True):
    """Designed originally for propagating the lepton scale and resolution systematic uncertainties to the MET for proper correlation, utilizing the corrected_polar_met function of coffea"""
    if is_correction:
        raise NotImplementedError("propagation of scale shifts to MET for central variation not implemented, consider implications of doing so carefully including prop to JES/JER/UES variations")
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
