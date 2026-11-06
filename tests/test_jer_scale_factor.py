import os
import subprocess
import sys

import awkward as ak
import correctionlib.schemav2 as cs

from qawa.jme import CorrectionLibJERSF, jersmear_factors

ETA_EDGES = [-5.0, -2.5, 0.0, 2.5, 5.0]
NOM = [1.25, 1.125, 1.0625, 1.5]
UNC = [0.125, 0.0625, 0.25, 0.5]
VARIATIONS = {'nom': 0.0, 'up': 1.0, 'down': -1.0}


def correction(name, inputs, data):
    return cs.Correction(
        name=name, version=1, output=cs.Variable(name='sf', type='real'),
        inputs=[cs.Variable(name=inp, type='string' if inp == 'systematic' else 'real') for inp in inputs],
        data=data,
    ).to_evaluator()


def eta_binning(content):
    return cs.Binning(nodetype='binning', input='JetEta', edges=ETA_EDGES, content=content, flow='clamp')


def systematic_layout():
    content = [
        cs.Category(nodetype='category', input='systematic', content=[
            cs.CategoryItem(key=key, value=nom + sign * unc) for key, sign in VARIATIONS.items()
        ])
        for nom, unc in zip(NOM, UNC)
    ]
    return correction('sf', ['JetEta', 'systematic'], eta_binning(content))


def split_layout():
    sf = correction('sf', ['JetEta', 'JetPt'], eta_binning(NOM))
    unc = correction('unc', ['JetEta', 'JetPt'], eta_binning(UNC))
    return sf, unc


def jets():
    eta = ak.Array([[-3.0, -1.0, 0.5], [], [2.0, 4.0]])
    pt = ak.Array([[40.0, 60.0, 80.0], [], [30.0, 120.0]])
    return pt, eta


def factors(jersf):
    pt, eta = jets()
    jer = correction('jer', ['JetEta', 'JetPt', 'Rho'], 0.1)
    jer_smear = correction('smear', ['JERSF'], cs.Formula(nodetype='formula', expression='x', parser='TFormula', variables=['JERSF']))
    return jersmear_factors(pt, eta, ak.full_like(pt, 20.0), ak.Array([1, 2, 3]), ak.full_like(pt, -1.0), jer, jersf, jer_smear, ())[1]


def test_systematic_and_split_layouts_give_identical_factors():
    expected = {
        key: [[NOM[0] + sign * UNC[0], NOM[1] + sign * UNC[1], NOM[2] + sign * UNC[2]], [], [NOM[2] + sign * UNC[2], NOM[3] + sign * UNC[3]]]
        for key, sign in VARIATIONS.items()
    }
    from_systematic = factors(CorrectionLibJERSF(systematic_layout()))
    from_split = factors(CorrectionLibJERSF(*split_layout()))
    for key in VARIATIONS:
        assert ak.to_list(from_systematic[key]) == expected[key]
        assert ak.to_list(from_split[key]) == expected[key]


def test_import_guard_needs_uncertainty_parameter():
    script = (
        "import coffea.jetmet_tools as jetmet_tools\n"
        "class CorrectionLibJERSF:\n"
        "    def __init__(self, correction):\n"
        "        pass\n"
        "jetmet_tools.CorrectionLibJERSF = CorrectionLibJERSF\n"
        "import qawa.jme\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert all(word in result.stderr for word in ('ImportError', 'uncertainty', 'f11301a', 'scikit-hep/coffea'))
