import contextlib
import io

import awkward as ak
import numpy as np
import pytest

from qawa.jetid import JET_ID_RECIPES, JetID, jet_id_recipe, recipe_jet_id


def synthetic_jets(n=200_000, seed=3):
    rng = np.random.default_rng(seed)

    def fraction():
        x = rng.random(n)
        for value in (0.0, 0.01, 0.4, 0.8, 0.9, 0.98, 0.99, 0.9999, 1.0):
            x[rng.random(n) < 0.01] = value
        return x

    eta = rng.uniform(-5.5, 5.5, n)
    for edge in (2.4, 2.6, 2.7, 3.0, 5.2):
        on_edge = rng.random(n) < 0.01
        eta[on_edge] = edge * np.sign(rng.standard_normal(on_edge.sum()))
    flat = ak.zip({"eta": eta, "neHEF": fraction(), "neEmEF": fraction(), "chHEF": fraction(), "chEmEF": fraction(),
                   "muEF": fraction(), "chMultiplicity": rng.integers(0, 6, n), "neMultiplicity": rng.integers(0, 6, n)})
    return ak.unflatten(flat, np.full(n // 4, 4))


def handler(era, version):
    from qawa.jsoncorrections import CorrectionlibHandler
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return CorrectionlibHandler(era=era, subera=None, analysis="inc-WZ", nanoAODversion=version, cvmfs_head="/cvmfs/")
    except ValueError as error:
        pytest.skip(f"correctionlib payloads not available: {error}")


@pytest.mark.parametrize("recipe", sorted(JET_ID_RECIPES))
def test_recipe_encoding(recipe):
    codes = recipe_jet_id(synthetic_jets(), recipe)
    assert set(np.unique(ak.flatten(codes))) <= {0, 2, 6}


def test_summer24_recipe_matches_correctionlib():
    jet_id = JetID("2024", "v15", handler("2024", "v15"))
    assert jet_id.cset is not None
    jets = synthetic_jets()
    assert ak.all(jet_id(jets) == recipe_jet_id(jets, "Summer24PUPPI"))


@pytest.mark.parametrize("era, recipe", [("2016", "UL16PUPPI"), ("2017", "UL1718PUPPI"), ("2018", "UL1718PUPPI")])
def test_run2_nanoaodv15_uses_recipe(era, recipe):
    jet_id = JetID(era, "v15", handler(era, "v15"))
    assert jet_id.cset is None and jet_id.recipe == recipe


def test_stored_jet_id_before_nanoaodv12():
    jets = ak.Array([[{"jetId": 6}, {"jetId": 2}]])
    assert JetID("2018", "v9").stored
    assert ak.all(JetID("2018", "v9")(jets) == jets.jetId)


def test_unknown_era_raises():
    with pytest.raises(NotImplementedError):
        jet_id_recipe("2025", "v15")
