from pathlib import Path
import warnings
import rich

import correctionlib

_runIII_v15_pogs_tags_pogtags_csets = (
    ("DC", "Collisions22", "2026-02-26", "Cert_Collisions2022_355100_362760_Golden.json"),
    ("DC", "Collisions23", "2026-02-26", "Cert_Collisions2023_366442_370790_Golden.json"),
    ("DC", "Collisions24", "2026-02-25", "Cert_Collisions2024_378981_386951_Golden.json"),
    ("DC", "Collisions25", "2026-02-04", "Cert_Collisions2025_391658_398903_Golden.json"),
    ("LUM", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-02", "puWeights_BCDEFGHI.json.gz"),

    ("BTV", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2026-03-10", "btagging.json.gz"),
    ("BTV", "Run3-25Prompt-Summer24-NanoAODv15", "add_2025_b_and_c_WPs", "btagging.json.gz"),

    # MISSING: met_xyCorrections, 2025 jetid, ...
    ("JME", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-02", "fatJet_jerc.json.gz"),
    ("JME", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-02", "jet_jerc.json.gz"),
    ("JME", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-02", "jetid.json.gz"),
    ("JME", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-02", "jetvetomaps.json.gz"),
    ("JME", "Run3-25Prompt-Winter25-NanoAODv15", "2026-02-09", "fatJet_jerc.json.gz"),
    ("JME", "Run3-25Prompt-Winter25-NanoAODv15", "2026-02-09", "jet_jerc.json.gz"),
    ("JME", "Run3-25Prompt-Winter25-NanoAODv15", "2026-02-09", "jetvetomaps.json.gz"),


    ("EGM", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-15", "electron.json.gz"),
    ("EGM", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-15", "photon.json.gz"),
    ("EGM", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-15", "electronSS_EtDependent.json.gz"),
    ("EGM", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-12-15", "photonSS_EtDependent.json.gz"),    
    ("EGM", "Run3-25Prompt-Summer24-NanoAODv15", "2026-01-22", "electronSS_EtDependent.json.gz"),
    ("EGM", "Run3-25Prompt-Summer24-NanoAODv15", "2026-01-22", "photonSS_EtDependent.json.gz"),

    # Missing  muon_JPsi.json.gz in 2025
    # ("MUO", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-11-27", "muon_HighPt.json.gz"),
    ("MUO", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-11-27", "muon_JPsi.json.gz"),
    ("MUO", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-11-27", "muon_Z.json.gz"),
    ("MUO", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2025-11-27", "muon_scalesmearing.json.gz"),
    # ("MUO", "Run3-25Prompt-Summer24-NanoAODv15", "2026-04-13", "muon_HighPt.json.gz"),
    # ("MUO", "Run3-25Prompt-Summer24-NanoAODv15", "2026-04-13", "muon_Z.json.gz"), # Causes a crash in correctionlib 2.7.0, not likely fixed in 2.8.0 but perhaps...

    ("TAU", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2026-01-14", "tau.json.gz"),
)
_runII_v15_pogs_tags_pogtags_csets = (
    ("BTV", "Run2-2016preVFP-UL-NanoAODv15", "add_btagging_wps", "btagging.json.gz"),
    ("BTV", "Run2-2016postVFP-UL-NanoAODv15", "add_b_tagging_WPs", "btagging.json.gz"),
    ("BTV", "Run2-2017-UL-NanoAODv15", "add_b_tagging_WPs", "btagging.json.gz"),
    ("BTV", "Run2-2018-UL-NanoAODv15", "add_b_tagging_WPs", "btagging.json.gz"),

    ("EGM", "Run2-2016preVFP-UL-NanoAODv15", "2025-12-05", "electron.json.gz"),
    ("EGM", "Run2-2016preVFP-UL-NanoAODv15", "2025-12-05", "electronSS_EtDependent.json.gz"),
    ("EGM", "Run2-2016preVFP-UL-NanoAODv15", "2025-12-05", "photon.json.gz"),
    ("EGM", "Run2-2016preVFP-UL-NanoAODv15", "2025-12-05", "photonSS_EtDependent.json.gz"),
    ("EGM", "Run2-2016postVFP-UL-NanoAODv15", "2025-12-05", "electron.json.gz"),
    ("EGM", "Run2-2016postVFP-UL-NanoAODv15", "2025-12-05", "electronSS_EtDependent.json.gz"),
    ("EGM", "Run2-2016postVFP-UL-NanoAODv15", "2025-12-05", "photon.json.gz"),
    ("EGM", "Run2-2016postVFP-UL-NanoAODv15", "2025-12-05", "photonSS_EtDependent.json.gz"),
    ("EGM", "Run2-2017-UL-NanoAODv15", "2025-12-05", "electron.json.gz"),
    ("EGM", "Run2-2017-UL-NanoAODv15", "2025-12-05", "electronSS_EtDependent.json.gz"),
    ("EGM", "Run2-2017-UL-NanoAODv15", "2025-12-05", "photon.json.gz"),
    ("EGM", "Run2-2017-UL-NanoAODv15", "2025-12-05", "photonSS_EtDependent.json.gz"),
    ("EGM", "Run2-2018-UL-NanoAODv15", "2025-12-05", "electron.json.gz"),
    ("EGM", "Run2-2018-UL-NanoAODv15", "2025-12-05", "electronSS_EtDependent.json.gz"),
    ("EGM", "Run2-2018-UL-NanoAODv15", "2025-12-05", "photon.json.gz"),
    ("EGM", "Run2-2018-UL-NanoAODv15", "2025-12-05", "photonSS_EtDependent.json.gz"),
    
    ("TAU", "Run2-2016postVFP-UL-NanoAODv15", "2025-11-27", "tau.json.gz"),
    ("TAU", "Run2-2016preVFP-UL-NanoAODv15", "2025-11-27", "tau.json.gz"),
    ("TAU", "Run2-2017-UL-NanoAODv15", "2025-11-27", "tau.json.gz"),
    ("TAU", "Run2-2018-UL-NanoAODv15", "2025-11-27", "tau.json.gz"),    
)

_runII_v9_pogs_tags_pogtags_csets = (
    ("LUM", "Run2-2016postVFP-UL-NanoAODv9", "2021-09-10", "puWeights.json.gz"),
    ("LUM", "Run2-2016preVFP-UL-NanoAODv9", "2021-09-10", "puWeights.json.gz"),
    ("LUM", "Run2-2017-UL-NanoAODv9", "2021-09-10", "puWeights.json.gz"),
    ("LUM", "Run2-2018-UL-NanoAODv9", "2021-09-10", "puWeights.json.gz"),

    ("BTV", "Run2-2016postVFP-UL-NanoAODv9", "2025-08-19", "btagging.json.gz"),
    ("BTV", "Run2-2016preVFP-UL-NanoAODv9", "2025-08-19", "btagging.json.gz"),
    ("BTV", "Run2-2017-UL-NanoAODv9", "2025-08-19", "btagging.json.gz"),
    ("BTV", "Run2-2018-UL-NanoAODv9", "2025-08-19", "btagging.json.gz"),

    ("JME", "Run2-2016postVFP-UL-NanoAODv9", "2025-04-11", "jet_jerc.json.gz"),
    ("JME", "Run2-2016postVFP-UL-NanoAODv9", "2025-04-11", "jetvetomaps.json.gz"),
    ("JME", "Run2-2016postVFP-UL-NanoAODv9", "2025-04-11", "jmar.json.gz"),
    ("JME", "Run2-2016postVFP-UL-NanoAODv9", "2025-04-11", "met.json.gz"),
    ("JME", "Run2-2016preVFP-UL-NanoAODv9", "2025-04-11", "jet_jerc.json.gz"),
    ("JME", "Run2-2016preVFP-UL-NanoAODv9", "2025-04-11", "jetvetomaps.json.gz"),
    ("JME", "Run2-2016preVFP-UL-NanoAODv9", "2025-04-11", "jmar.json.gz"),
    ("JME", "Run2-2016preVFP-UL-NanoAODv9", "2025-04-11", "met.json.gz"),
    ("JME", "Run2-2017-UL-NanoAODv9", "2025-04-11", "jet_jerc.json.gz"),
    ("JME", "Run2-2017-UL-NanoAODv9", "2025-04-11", "jetvetomaps.json.gz"),
    ("JME", "Run2-2017-UL-NanoAODv9", "2025-04-11", "jmar.json.gz"),
    ("JME", "Run2-2017-UL-NanoAODv9", "2025-04-11", "met.json.gz"),
    ("JME", "Run2-2018-UL-NanoAODv9", "2025-04-11", "jet_jerc.json.gz"),
    ("JME", "Run2-2018-UL-NanoAODv9", "2025-04-11", "jetvetomaps.json.gz"),
    ("JME", "Run2-2018-UL-NanoAODv9", "2025-04-11", "jmar.json.gz"),
    ("JME", "Run2-2018-UL-NanoAODv9", "2025-04-11", "met.json.gz"),

    ("EGM", "Run2-2016postVFP-UL-NanoAODv9", "2024-07-02", "electron.json.gz"),
    ("EGM", "Run2-2016postVFP-UL-NanoAODv9", "2024-07-02", "photon.json.gz"),
    ("EGM", "Run2-2016preVFP-UL-NanoAODv9", "2024-07-02", "electron.json.gz"),
    ("EGM", "Run2-2016preVFP-UL-NanoAODv9", "2024-07-02", "photon.json.gz"),
    ("EGM", "Run2-2017-UL-NanoAODv9", "2024-07-02", "electron.json.gz"),
    ("EGM", "Run2-2017-UL-NanoAODv9", "2024-07-02", "photon.json.gz"),
    ("EGM", "Run2-2018-UL-NanoAODv9", "2024-07-02", "electron.json.gz"),
    ("EGM", "Run2-2018-UL-NanoAODv9", "2024-07-02", "photon.json.gz"),

    ("MUO", "Run2-2016postVFP-UL-NanoAODv9", "2024-07-02", "muon_JPsi.json.gz"),
    ("MUO", "Run2-2016postVFP-UL-NanoAODv9", "2024-07-02", "muon_Z.json.gz"),
    ("MUO", "Run2-2016preVFP-UL-NanoAODv9", "2024-07-02", "muon_JPsi.json.gz"),
    ("MUO", "Run2-2016preVFP-UL-NanoAODv9", "2024-07-02", "muon_Z.json.gz"),
    ("MUO", "Run2-2017-UL-NanoAODv9", "2024-07-02", "muon_JPsi.json.gz"),
    ("MUO", "Run2-2017-UL-NanoAODv9", "2024-07-02", "muon_Z.json.gz"),
    ("MUO", "Run2-2018-UL-NanoAODv9", "2024-07-02", "muon_JPsi.json.gz"),
    ("MUO", "Run2-2018-UL-NanoAODv9", "2024-07-02", "muon_Z.json.gz"),

    ("TAU", "Run2-2016postVFP-UL-NanoAODv9", "2026-03-25", "tau.json.gz"),
    ("TAU", "Run2-2016preVFP-UL-NanoAODv9", "2026-03-25", "tau.json.gz"),
    ("TAU", "Run2-2017-UL-NanoAODv9", "2026-03-25", "tau.json.gz"),
    ("TAU", "Run2-2018-UL-NanoAODv9", "2026-03-25", "tau.json.gz"),
)

class CorrectionlibHandler:
    def __init__(self, era: str, analysis: str, subera: str | None = None, isAPV:bool = False, isEE:bool = False, isBPix:bool = False, nanoAODversion: str = "v15", cvmfs_head: str = "/cvmfs/"):
        self._era = era
        self._subera = subera
        match self._era:
            case "2016":
                self._run = "Run2"
                if self._subera is None:
                    if isAPV:
                        self._subera = "preVFP"
                    else:
                        self._subera = "postVFP"
            case "2017" | "2018":
                self._run = "Run2"
            case "2022":
                self._run = "Run3"
                if self._subera is None:
                    if isEE:
                        self._subera = "EE"
                    else:
                        self._subera = None
            case "2023":
                self._run = "Run3"
                if self._subera is None:
                    if isBPix:
                        self._subera = "BPix"
                    else:
                        self._subera = None
            case "2024" | "2025" | "2026":
                self._run = "Run3"
                self._subera = None
            case _:
                assert False, f"{self._era} not in known choices"
        self._analysis = analysis
        self._ver = nanoAODversion
        self._head = Path(cvmfs_head)
        if not self._head.exists():
            # local fallback for testing only
            self._head = Path("/Users/nmangane/Downloads/cvmfs")
        if not self._head.exists():
            raise ValueError(f"{str(self._head)} is not a valid cvmfs head directory for fetching CAT POG corrections")
        self._poghead = self._head / "cms-griddata.cern.ch/cat/metadata/"
        if not self._poghead.exists():
            raise ValueError(f"{str(self._poghead)} is not a valid CAT POG directory for fetching CAT POG corrections")
        self._knownsuberamap = {None: None, # for all other unmapped eras
                                "preVFP": "2016", "postVFP": "2016",
                                "EE": "2022", "preEE": "2022", "postEE": "2022",
                                "BPix": "2023", "preBPix": "2023", "postBPix": "2023",
                                }
        assert self._subera in list(self._knownsuberamap.keys()), f"{self._subera} not in known choices: {self._knownsuberamap}"
        self._knownpogs = ["BTV", "DC", "EGM", "JME", "LUM", "MUO", "TAU"]
        # self._extrapogs = dynamically look into the _pogs directory and compare entries against the _knownpogs maps
        self._paths = {}
        self._csets = {}
        self._notcsets = {}

        self._loadCentralPaths()
        self._loadCentralExceptions()
        self._loadUserPaths(self._analysis)

    def _loadCentralPaths(self):
        pogs_tags_pogtags_csets = None
        if self._era in ["2016", "2017", "2018", "2018"]:
            if self._ver == "v15":
                pogs_tags_pogtags_csets = _runII_v15_pogs_tags_pogtags_csets
            elif self._ver == "v9":
                pogs_tags_pogtags_csets = _runII_v9_pogs_tags_pogtags_csets
            else:
                raise ValueError(f"Unconfigured NanoAOD version requested {self._ver} in CorrectionlibHandler")                
        elif self._era in ["2022", "2023", "2024", "2025", "2026"]:
            if self._ver == "v15":
                pogs_tags_pogtags_csets = _runIII_v15_pogs_tags_pogtags_csets
            else:
                raise ValueError(f"Unconfigured NanoAOD version requested {self._ver} in CorrectionlibHandler")
        else:
            raise ValueError(f"Unexpected era {self._era} in CorrectionlibHandler")                
        
        for pog, tag, pogtag, cset in pogs_tags_pogtags_csets:
            # handle Golden JSONs separately, for convenience...
            if pog == "DC":
                if f"Collisions{self._era[-2:]}" not in tag:
                    continue
                path = self._poghead / pog / tag / pogtag / cset
                if path.exists():
                    key = cset.split(".")[0] if "." in cset else cset # strip filetype/compression
                    self._notcsets[key] = path
            # skip unmatched year, but sometimes the tag only contains the last 2 digits
            if f"{self._run}-{self._era}" not in tag and f"{self._run}-{self._era[-2:]}" not in tag:
                continue
            if self._subera is not None and self._subera not in tag:
                continue
            if f"NanoAOD{self._ver}" not in tag:
                continue
            path = self._poghead / pog / tag / pogtag / cset
            if path.exists():
                key = cset.split(".")[0] if "." in cset else cset # strip filetype/compression
                key = "puWeights" if "puWeights_" in key else key# strip era specifier from weights
                if key in self._paths:
                    warnings.warn(f"Unexpectedly overwriting {key} [{self._paths[key]}] with {path}")
                self._paths[key] = path
            else:
                warnings.warn(f"Failed to load expected Central Path: {path}")

    def _loadCentralExceptions(self):
        # Because there's always a special little exception to add misery to our lives
        if "jer_smear" not in self._paths:
            self._paths["jer_smear"] = self._poghead / "JME" / "JER-Smearing" / "2025-11-03" / "jer_smear.json.gz"

    def _loadUserPaths(self, analysis: str):
        # Where to load HLT, DDDY, etc corrections...
        if analysis == "inc-WZ":
            match self._era:
                case "2024":
                    self._paths["trigger_sf"] = Path("src/qawa/data/trigger_sf/triggerSF_2024.json")
                case _:
                    pass
        elif analysis == "inc-ZZ":
            pass
        else:
            raise ValueError

    def keys(self):
        return self._paths.keys()

    def values(self):
        return self._paths.values()

    def items(self):
        return self._paths.items()
    
    def __getitem__(self, key):
        return self._paths[key]

    def __setitem__(self, key, value):
        assert value.exists()
        if key in self._paths:
            warnings.warn(f"Unexpectedly overwriting {key} [{self._paths[key]}] with {path}")
        self._paths[key] = value

    def getPath(self, lookup: str, fallbackNone: bool = False):
        if fallbackNone and (lookup not in self.keys() and lookup not in self._notcsets.keys()):
            return None
        if lookup in self.keys():
            return self[lookup]
        elif lookup in self._notcsets.keys():
            return self._notcsets[lookup]

    def getCorrectionSet(self, key: str):
        if key not in self._csets and key not in self._notcsets:
            try:
                self._csets[key] = correctionlib.CorrectionSet.from_file(str(self[key]))
            except Exception as e:
                self._notcsets[key] = self._paths[key]
        if key not in self._notcsets:
            return self._csets[key]
        else:
            return None

    def clearCorrectionSets(self):
        self._csets = {}
        

__all__ = ["CorrectionlibPathDict"]

if __name__ == "__main__":
    import rich
    all_pairs = [("2017", None), ("2018", None), ("2024", None), ("2025", None)]
    for subera, era in {
                        "preVFP": "2016", "postVFP": "2016",
                        # "EE": "2022", "preEE": "2022", "postEE": "2022"
                        # "BPix": "2023", "preBPix": "2023", "postBPix": "2023",
                        }.items():
        all_pairs.append((era, subera))
    for era, subera in sorted(all_pairs):
        # v15 corrections
        rich.print(era, subera)
        ch = CorrectionlibHandler(era=era, subera=subera, analysis="inc-WZ", nanoAODversion="v15", cvmfs_head="/cvmfs")
        expected = 0
        for k, v in ch.items():
            expected += 1
            test = ch.getCorrectionSet(k)
            rich.print("\t", k, v, len(test.keys()), len(test.compound.keys()))
        rich.print(f"\texpected={expected}, [green]actual={len(ch._csets.keys())}[/green]")
    for era, subera in sorted(all_pairs):
        # v9 corrections
        if int(era) > 2018:
            continue
        rich.print(era, subera)
        ch = CorrectionlibHandler(era=era, subera=subera, analysis="inc-WZ", nanoAODversion="v9", cvmfs_head="/cvmfs")
        expected = 0
        for k, v in ch.items():
            expected += 1
            test = ch.getCorrectionSet(k)
            rich.print("\t", k, v, len(test.keys()), len(test.compound.keys()))
        rich.print(f"\texpected={expected}, [green]actual={len(ch._csets.keys())}[/green]")
