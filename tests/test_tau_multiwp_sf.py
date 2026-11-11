import awkward as ak
import numpy as np
import pytest
from coffea.analysis_tools import Weights

from conftest import CVMFS_HEAD, input_path

CVMFS, needs_cvmfs = input_path(CVMFS_HEAD, "cms-griddata.cern.ch")


@needs_cvmfs
@pytest.mark.parametrize("era", ["2024", "2025"])
def test_each_tau_gets_its_own_wp_sf(era):
    from qawa.jsoncorrections import CorrectionlibHandler
    from qawa.tauSF import tauIDScaleFactors
    handler = CorrectionlibHandler(era=era, analysis="inc-WZ", nanoAODversion="v15", cvmfs_head=CVMFS_HEAD)
    tau_id = tauIDScaleFactors(era=era, vsjet_wp="VTight", vse_wp="Tight", vsmu_wp="Tight", clibhandler=handler,
                               nanoAODversion="v15")
    taus = ak.Array([[{"pt": pt, "eta": eta, "decayMode": 1, "genPartFlav": 5}]
                     for pt, eta in ((35.0, 0.5), (45.0, -1.2), (60.0, 1.8))])
    vtight = np.array([True, False, False])
    tight = np.array([False, True, False])
    loose = np.array([False, False, True])
    weights = Weights(len(taus), storeIndividual=True)
    tau_id.append_tauID_multiwp_sf(taus, taus, taus, vtight, tight, loose, weights)
    nominal = weights.partial_weight(include=["tauIDvsjet_sf"])
    for index, wp in enumerate(("VTight", "Tight", "Loose")):
        tau_id.vsjet_wp = wp
        want = {syst: ak.to_numpy(tau_id.getSF(taus, syst=syst)[0])[index] for syst in ("nom", "up", "down")}
        assert np.isclose(nominal[index], want["nom"]), wp
        for syst, modifier in (("up", "tauIDvsjet_sfUp"), ("down", "tauIDvsjet_sfDown")):
            assert np.isclose(weights.weight(modifier)[index] / weights.weight()[index] * want["nom"], want[syst]), (wp, syst)
