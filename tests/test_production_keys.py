import ast
import hashlib
import itertools
import sys

import pytest

import qawa.production_keys as pk

WZ_DAS = "/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
DY_DAS = ("/DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/"
          "RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v1/NANOAODSIM")
MUON_DAS = "/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD"

M1_PATH = ("/Volumes/WDMac/SMQawaTestFiles/store/mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/"
           "NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/test.root")
M3_PATH = ("/store/mc/RunIII2024Summer24NanoAODv15/DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/"
           "NANOAODSIM/150X_mcRun3_2024_realistic_v2-v1/90000/test.root")
D1_PATH = "root://cmsxrootd.fnal.gov//store/data/Run2024C/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root"
PROMPT_PATH = "/store/data/Run2025C/Muon0/NANOAOD/PromptReco-v1/000/392/123/00000/x.root"

LABEL = "3fa2c81b09de"
FLAGS = ("pol", "split_by_charge", "split_FV", "eft", "dd")


def variation_subsets():
    for n in range(len(FLAGS) + 1):
        for names in itertools.combinations(FLAGS, n):
            yield {name: (LABEL if name == "dd" else True) for name in names}


def test_das_parts_and_primary():
    assert pk.das_parts(WZ_DAS) == ("WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8",
                                    "RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2", "NANOAODSIM")
    assert pk.primary_dataset(DY_DAS) == "DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8"
    assert pk.primary_dataset(MUON_DAS) == "Muon0"
    for bad in ("/Muon0/Run2024*/NANOAOD", "/Muon0/NANOAOD", "/Muon0/Run2024C-v1/USER", MUON_DAS + "/"):
        with pytest.raises(ValueError):
            pk.das_parts(bad)


def test_das_name_from_path():
    assert pk.das_name_from_path(M1_PATH) == WZ_DAS
    assert pk.das_name_from_path(M3_PATH) == DY_DAS
    assert pk.das_name_from_path(D1_PATH) == MUON_DAS
    assert pk.das_name_from_path(PROMPT_PATH) == "/Muon0/Run2025C-PromptReco-v1/NANOAOD"
    assert pk.das_name_from_path("/Users/me/DoubleMuon_UL2018_NanoAODv9.root") is None
    assert pk.das_name_from_path("/store/user/me/WZ/NANOAODSIM/v1/0000/x.root") is None


def test_sample_key_round_trip():
    keys = []
    for variations in variation_subsets():
        key = pk.sample_key(WZ_DAS, **variations)
        assert pk.sample_key(WZ_DAS, **dict(reversed(list(variations.items())))) == key
        assert pk.parse_sample_key(key) == (WZ_DAS, variations)
        tokens = key.partition("#")[2].split("+") if variations else []
        assert [t.partition(":")[0] for t in tokens] == [n for n in pk.VARIATION_ORDER if n in variations]
        keys.append(key)
    assert len(set(keys)) == 32
    stems = [pk.sample_file_stem(key) for key in keys]
    assert len(set(stems)) == 32
    assert not any(c in stem for stem in stems for c in "/#:+")
    assert pk.sample_file_stem(pk.sample_key(MUON_DAS, dd=LABEL)) == "Muon0__Run2024C-MINIv6NANOv15-v1__NANOAOD__dd-3fa2c81b09de"
    assert pk.sample_key(MUON_DAS, dd=LABEL) != pk.sample_key(MUON_DAS)
    assert pk.sample_key(WZ_DAS) != pk.sample_key(DY_DAS)
    for bad in ("3FA2C81B09DE", "3fa2c81b09d", "3fa2c81b09dex", "3fa2c81b09dg"):
        with pytest.raises(ValueError):
            pk.sample_key(MUON_DAS, dd=bad)
    for bad in (WZ_DAS + "#eft+pol", WZ_DAS + "#pol+pol", WZ_DAS + "#tau", WZ_DAS + "#", "WZ#pol"):
        with pytest.raises(ValueError):
            pk.parse_sample_key(bad)


def test_dd_label(tmp_path):
    path = tmp_path / "dd.json"
    payload = b"x" * (3 << 20) + b"tail"
    path.write_bytes(payload)
    assert pk.file_sha256(path) == hashlib.sha256(payload).hexdigest()
    assert pk.dd_label(path) == hashlib.sha256(payload).hexdigest()[:12]


def test_standard_library_imports_only():
    tree = ast.parse(open(pk.__file__).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names == {"hashlib", "os", "re"}
    assert names <= set(sys.stdlib_module_names)


def test_final_names():
    assert pk.variation_suffix({}) == ""
    assert pk.variation_suffix({"pol": True}) == "__pol"
    assert pk.variation_suffix({"split_by_charge": True}) == "__split_by_charge"
    assert pk.variation_suffix({"dd": LABEL}) == "__dd-3fa2c81b09de"
    assert pk.variation_suffix({"eft": True}) == ""
    assert (pk.final_file_stem("t", "2024", "WZ", "data", {"dd": LABEL})
            == "merged-histogram-Inc-t-2024-WZ__dd-3fa2c81b09de-Data")
    assert pk.final_file_stem("t", "2024", "WZ", "mc", {}) == "merged-histogram-Inc-t-2024-WZ-MC-backgrounds"
    assert pk.final_file_stem("t", "2024", "WZ", "signal", {"pol": True}) == "merged-histogram-Inc-t-2024-WZ__pol-MC-signal"
    assert pk.final_file_stem("t", "2024", "WZ", "eft", {"eft": True}) == "merged-histogram-Inc-t-2024-WZ-EFT"
    assert set(pk.KIND_FILE_LABELS) == {"data", "mc", "signal", "eft"}
    stems = set()
    count = 0
    for variations in variation_subsets():
        for kind in pk.KIND_FILE_LABELS:
            if bool(variations.get("eft")) != (kind == "eft") or ("dd" in variations and kind != "data"):
                continue
            stems.add(pk.final_file_stem("t", "2024", "WZ", kind, variations))
            count += 1
    assert len(stems) == count
    assert not any("eft" in pk.variation_suffix(v) for v in variation_subsets())
