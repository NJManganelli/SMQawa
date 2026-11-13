from pathlib import Path

import correctionlib
import numpy as np
import pytest

import qawa

PT_EDGES = [20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0]
PT_CENTERS = [(low + high) / 2 for low, high in zip(PT_EDGES[:-1], PT_EDGES[1:])]
LEAD, TRAIL = (np.array(axis).ravel() for axis in np.meshgrid(PT_CENTERS, PT_CENTERS, indexing="ij"))


@pytest.fixture(scope="module")
def trigger_sf():
    return correctionlib.CorrectionSet.from_file(
        str(Path(qawa.__file__).parent / "data" / "trigger_sf" / "triggerSF_2024.json"))


@pytest.mark.parametrize("name", ("SF_mm", "SF_me", "SF_em", "SF_ee"))
@pytest.mark.parametrize("syst", ("nominal", "up", "down"))
def test_measured_values_sit_at_lead_pt_above_trail_pt(trigger_sf, name, syst):
    sf = trigger_sf[name].evaluate(LEAD, TRAIL, syst)
    assert np.all(sf[LEAD < TRAIL] == 1.0)
    assert np.all(sf[LEAD > TRAIL] != 1.0)
