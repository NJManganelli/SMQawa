"""inc-ZZ processor: importable, with b-tag SF weights refused (no correctionlib handler)."""
import pytest


def test_import():
    import qawa.process.zz2l2nu_inclusive as zzmod
    assert zzmod.zzinc_processor


def test_btag_sf_refused():
    from qawa.process.zz2l2nu_inclusive import zzinc_processor
    with pytest.raises(ValueError, match="no correctionlib handler"):
        zzinc_processor(era="2018", apply_btag_sf=True)
