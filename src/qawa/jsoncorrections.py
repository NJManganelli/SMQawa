import os
import datetime
from pathlib import Path
import rich
from rich.table import Table
from rich.markup import escape
from coffea.util import coffea_console
import correctionlib

# Directory of the installed `qawa` package (.../src/qawa). Anchoring on
# __file__ makes data lookups independent of the current working directory,
# so they resolve inside Singularity/HTCondor where CWD is not the repo root.
_QAWA_DIR = Path(__file__).resolve().parent
_QAWA_DATA = _QAWA_DIR / "data"

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
    ("LUM", "Run2-2016postVFP-UL-NanoAODv9", "2021-09-10", "puWeights.json.gz"),
    ("LUM", "Run2-2016preVFP-UL-NanoAODv9", "2021-09-10", "puWeights.json.gz"),
    ("LUM", "Run2-2017-UL-NanoAODv9", "2021-09-10", "puWeights.json.gz"),
    ("LUM", "Run2-2018-UL-NanoAODv9", "2021-09-10", "puWeights.json.gz"),

    ("BTV", "Run2-2016preVFP-UL-NanoAODv15", "add_btagging_wps", "btagging.json.gz"),
    ("BTV", "Run2-2016postVFP-UL-NanoAODv15", "add_b_tagging_WPs", "btagging.json.gz"),
    ("BTV", "Run2-2017-UL-NanoAODv15", "add_b_tagging_WPs", "btagging.json.gz"),
    ("BTV", "Run2-2018-UL-NanoAODv15", "add_b_tagging_WPs", "btagging.json.gz"),

    ("JME", "Run2-2016postVFP-UL-NanoAODv15", "2026-06-05", "jet_jerc.json.gz"),
    ("JME", "Run2-2016postVFP-UL-NanoAODv15", "2026-06-05", "jetvetomaps.json.gz"),
    # ("JME", "Run2-2016postVFP-UL-NanoAODv15", "2026-06-05", "jmar.json.gz"),
    ("JME", "Run2-2016postVFP-UL-NanoAODv15", "22026-06-05", "met.json.gz"),
    ("JME", "Run2-2016preVFP-UL-NanoAODv15", "2026-06-05", "jet_jerc.json.gz"),
    ("JME", "Run2-2016preVFP-UL-NanoAODv15", "2026-06-05", "jetvetomaps.json.gz"),
    # ("JME", "Run2-2016preVFP-UL-NanoAODv15", "2026-06-05", "jmar.json.gz"),
    ("JME", "Run2-2016preVFP-UL-NanoAODv15", "2026-06-051", "met.json.gz"),
    ("JME", "Run2-2017-UL-NanoAODv15", "2026-06-05", "jet_jerc.json.gz"),
    ("JME", "Run2-2017-UL-NanoAODv15", "2026-06-05", "jetvetomaps.json.gz"),
    # ("JME", "Run2-2017-UL-NanoAODv15", "2026-06-05", "jmar.json.gz"),
    ("JME", "Run2-2017-UL-NanoAODv15", "2026-06-05", "met.json.gz"),
    ("JME", "Run2-2018-UL-NanoAODv15", "2026-06-05", "jet_jerc.json.gz"),
    ("JME", "Run2-2018-UL-NanoAODv15", "2026-06-05", "jetvetomaps.json.gz"),
    # ("JME", "Run2-2018-UL-NanoAODv15", "2026-06-05", "jmar.json.gz"),
    ("JME", "Run2-2018-UL-NanoAODv15", "2026-06-05", "met.json.gz"),

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

    ("MUO", "Run2-2016postVFP-UL-NanoAODv9", "2024-07-02", "muon_JPsi.json.gz"),
    ("MUO", "Run2-2016postVFP-UL-NanoAODv9", "2024-07-02", "muon_Z.json.gz"),
    ("MUO", "Run2-2016preVFP-UL-NanoAODv9", "2024-07-02", "muon_JPsi.json.gz"),
    ("MUO", "Run2-2016preVFP-UL-NanoAODv9", "2024-07-02", "muon_Z.json.gz"),
    ("MUO", "Run2-2017-UL-NanoAODv9", "2024-07-02", "muon_JPsi.json.gz"),
    ("MUO", "Run2-2017-UL-NanoAODv9", "2024-07-02", "muon_Z.json.gz"),
    ("MUO", "Run2-2018-UL-NanoAODv9", "2024-07-02", "muon_JPsi.json.gz"),
    ("MUO", "Run2-2018-UL-NanoAODv9", "2024-07-02", "muon_Z.json.gz"),
    
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
        if self._era in ["2016", "2017", "2018"]:
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
                if pog == "LUM" and self._era in ["2016", "2017", "2018"]:
                    # exception for no PU reweighting for v15 Run2
                    pass
                if pog == "MUO" and self._era in ["2016", "2017", "2018"]:
                    # exception for no PU reweighting for v15 Run2
                    pass
                else:
                    continue
            path = self._poghead / pog / tag / pogtag / cset
            if path.exists():
                key = cset.split(".")[0] if "." in cset else cset # strip filetype/compression
                key = "puWeights" if "puWeights_" in key else key# strip era specifier from weights
                if key in self._paths:
                    coffea_console.print(f"[red]Unexpectedly overwriting {key}[/red] [yellow]{escape(str(self._paths[key]))}[/yellow] with [blue]{escape(str(path))}[/blue]")
                self._paths[key] = path
            else:
                coffea_console.print(f"Failed to load expected Central Path: {path}")

    def _loadCentralExceptions(self):
        # Because there's always a special little exception to add misery to our lives
        if "jer_smear" not in self._paths:
            self._paths["jer_smear"] = self._poghead / "JME" / "JER-Smearing" / "2025-11-03" / "jer_smear.json.gz"

    def _loadUserPaths(self, analysis: str):
        # Where to load HLT, DDDY, etc corrections...
        if analysis == "inc-WZ":
            match self._era:
                case "2024":
                    # PLACEHOLDER - once Run 3 dddy (2024 only to start with) is derived, replace with Run3/...Run3.json
                    self._paths["dddy"] = _QAWA_DATA / "dd" / "Run3" / "WZ_inclusive_data_driven_Run3.json"
                    self._paths["trigger_sf"] = _QAWA_DATA / "trigger_sf" / "triggerSF_2024.json"
                # Correctionlib conversions of the legacy ROOT trigger SFs
                # (histo_triggerEff_sel0_<era>.root), produced by
                # convert_trigger_sf.py. These collapse the eta dependence of the
                # source histograms via per-bin inverse-variance combination, so
                # they are NOT yet a drop-in replacement for the eta-dependent
                # legacy lookup. Uncomment a case to route that era through the
                # correctionlib path in wztau2lnu_inclusive._add_trigger_sf.
                case "2016":
                    self._paths["dddy"] = _QAWA_DATA / "dd" / "Run2" / "WZ_inclusive_data_driven_Run2.json"
                #     self._paths["trigger_sf"] = Path("src/qawa/data/trigger_sf/triggerSF_2016.json")
                case "2017":
                    self._paths["dddy"] = _QAWA_DATA / "dd" / "Run2" / "WZ_inclusive_data_driven_Run2.json"
                #     self._paths["trigger_sf"] = Path("src/qawa/data/trigger_sf/triggerSF_2017.json")
                case "2018":
                    self._paths["dddy"] = _QAWA_DATA / "dd" / "Run2" / "WZ_inclusive_data_driven_Run2.json"
                #     self._paths["trigger_sf"] = Path("src/qawa/data/trigger_sf/triggerSF_2018.json")
                case _:
                    pass
        elif analysis == "inc-ZZ":
            pass
        else:
            raise ValueError

    # ------------------------------------------------------------------
    # Status reporting / validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _isValidDate(name: str) -> bool:
        """A tag counts as a real version only if it is an ISO date (YYYY-MM-DD).

        The moving 'latest' symlink and hand-written tags like
        'add_b_tagging_WPs' are deliberately excluded: we never want to pin to
        a moving target, and non-date tags cannot be ordered chronologically.
        """
        try:
            datetime.date.fromisoformat(name)
            return True
        except (ValueError, TypeError):
            return False

    def _isCvmfsPath(self, path: Path) -> bool:
        """True if the path lives under the CAT POG metadata tree on /cvmfs."""
        try:
            return self._poghead in Path(path).resolve().parents or self._poghead in Path(path).parents
        except (OSError, RuntimeError):
            return self._poghead in Path(path).parents

    def _countNewerTags(self, path):
        """Count how many newer date-tagged versions of a correction file exist.

        Returns a ``(count, kind)`` tuple where ``kind`` is:
          * ``"ok"``      currently-selected tag is a valid date; ``count`` is the
                          number of strictly-newer date tags that still contain
                          this file.
          * ``"nondate"`` selected tag is not a date (e.g. 'add_b_tagging_WPs' or
                          'latest'); ``count`` is the number of date-tagged
                          versions available, none of which we can order against.
          * ``"na"``      not a /cvmfs POG path, so version tracking does not
                          apply (user-supplied JSON, local fallbacks, ...).
        """
        p = Path(path)
        if not self._isCvmfsPath(p):
            return (None, "na")
        selected = p.parent.name          # the <date-or-tag> directory
        tagdir = p.parent.parent          # poghead/POG/<tag>
        cset = p.name                     # the correction file itself
        if not tagdir.is_dir():
            return (None, "na")
        versions = []
        for child in tagdir.iterdir():
            if not child.is_dir() or child.name == "latest":
                continue
            if not self._isValidDate(child.name):
                continue
            # only count siblings that actually provide the same file
            if not (child / cset).exists():
                continue
            versions.append(child.name)
        if self._isValidDate(selected):
            # ISO date strings sort chronologically as plain strings
            newer = [v for v in versions if v > selected]
            return (len(newer), "ok")
        return (len(versions), "nondate")

    @staticmethod
    def _newerTagColor(count: int) -> str:
        """Interpolate green (0 newer) -> yellow -> red (>=5 newer)."""
        frac = min(max(count, 0), 5) / 5.0
        green, yellow, red = (0, 180, 0), (220, 200, 0), (200, 0, 0)
        if frac <= 0.5:
            t = frac / 0.5
            c = tuple(int(green[i] + t * (yellow[i] - green[i])) for i in range(3))
        else:
            t = (frac - 0.5) / 0.5
            c = tuple(int(yellow[i] + t * (red[i] - yellow[i])) for i in range(3))
        return f"rgb({c[0]},{c[1]},{c[2]})"

    @staticmethod
    def _discoverConsole():
        """Return the live coffea console if it is already around, else a fresh one.

        Reusing coffea's console means the status table shares the same output
        stream as any in-flight progress bars, so printing it does not corrupt
        or fight with them.
        """
        import sys
        mod = sys.modules.get("coffea.util")
        if mod is not None and hasattr(mod, "coffea_console"):
            return mod.coffea_console
        try:
            from coffea.util import coffea_console as _cc
            return _cc
        except Exception:
            from rich.console import Console
            return Console()

    def _relForDisplay(self, path):
        """Anchor a path against a known root so common prefixes collapse away.

        Returns the path's components relative to the best-matching root: the
        CAT POG tree on /cvmfs (components start at the POG), the packaged qawa
        data dir (prefixed with a ``~`` marker), or the filesystem root for
        anything else. The leading absolute separator is folded into the first
        component so tree joins never produce a doubled slash.
        """
        p = Path(path)
        if self._poghead in p.parents:
            return p.relative_to(self._poghead).parts
        if _QAWA_DATA in p.parents:
            return ("~",) + p.relative_to(_QAWA_DATA).parts
        parts = p.parts
        if len(parts) > 1 and parts[0] == os.sep:
            parts = (parts[0] + parts[1],) + parts[2:]
        return parts

    def printStatus(self, console=None, title: str | None = None, print_it: bool = True):
        """Render a rich table summarising every tracked correction path.

        Columns: the path (first) rendered as a tree — directory prefixes shared
        by more than one entry are lifted onto their own line and the members
        indented beneath — then the shorthand name (the key for
        ``getCorrectionSet``), whether the file exists on disk, whether a
        CorrectionSet has been loaded, and how many newer date-tagged versions
        exist on /cvmfs (green = up to date, shading toward red for 5+ versions
        behind; magenta = a non-date tag is pinned and cannot be version-checked).

        Pass ``console`` to reuse an existing rich console (e.g. coffea's), or
        leave it ``None`` to auto-discover the coffea console if it is imported.
        """
        console = console if console is not None else self._discoverConsole()

        # Merge the loadable paths with the "not a cset" entries (Golden JSONs,
        # or files whose load attempt failed) so nothing is hidden.
        entries = dict(self._paths)
        for k, v in self._notcsets.items():
            entries.setdefault(k, v)

        # Precompute every cell up front, keyed by the display path components.
        display = []
        for key, path in entries.items():
            p = Path(path)
            exists_cell = "[green]✔[/green]" if p.exists() else "[red]✘[/red]"
            if key in self._csets:
                loaded_cell = "[green]✔[/green]"
            elif key in self._notcsets:
                loaded_cell = "[yellow]not a cset[/yellow]"
            else:
                loaded_cell = "[dim]–[/dim]"
            count, kind = self._countNewerTags(p)
            if kind == "na":
                newer_cell = "[dim]–[/dim]"
            elif kind == "nondate":
                newer_cell = f"[magenta]{count}⚠[/magenta]"
            else:
                color = self._newerTagColor(count)
                suffix = " (current)" if count == 0 else ""
                newer_cell = f"[{color}]{count}{suffix}[/{color}]"
            display.append({
                "key": key,
                "parts": self._relForDisplay(p),
                "exists": exists_cell,
                "loaded": loaded_cell,
                "newer": newer_cell,
            })
        display.sort(key=lambda e: e["parts"])

        # Build a trie over the path components so shared directory prefixes can
        # be factored out. Each node holds sub-directories and terminal files.
        def _new_node():
            return {"dirs": {}, "files": []}

        root = _new_node()
        for e in display:
            node = root
            for seg in e["parts"][:-1]:
                node = node["dirs"].setdefault(seg, _new_node())
            node["files"].append((e["parts"][-1], e))

        def _count_leaves(node):
            return len(node["files"]) + sum(_count_leaves(c) for c in node["dirs"].values())

        def _single_leaf(node):
            # Called only on subtrees holding exactly one file; return its
            # remaining path (relative to node) and the entry.
            if node["files"]:
                return node["files"][0]
            seg = next(iter(node["dirs"]))
            tail, e = _single_leaf(node["dirs"][seg])
            return f"{seg}/{tail}", e

        def _fmt_leaf(indent, s):
            # Dim the directory portion, leave the filename at full contrast.
            if "/" in s:
                d, f = s.rsplit("/", 1)
                return f"{indent}[dim]{escape(d)}/[/dim]{escape(f)}"
            return f"{indent}{escape(s)}"

        rows = []  # (path_cell, entry_or_None); None marks a shared-prefix header

        def _walk(node, depth):
            indent = "  " * depth
            for seg in sorted(node["dirs"]):
                child = node["dirs"][seg]
                # Collapse a chain of single-child directories into one segment.
                segs = [seg]
                while len(child["dirs"]) == 1 and not child["files"]:
                    only = next(iter(child["dirs"]))
                    segs.append(only)
                    child = child["dirs"][only]
                compressed = "/".join(segs)
                if _count_leaves(child) > 1:
                    # Shared by more than one entry: give it its own line.
                    rows.append((f"{indent}[bold dim]{escape(compressed)}/[/bold dim]", None))
                    _walk(child, depth + 1)
                else:
                    # Only one file below: keep it inline, no header line.
                    tail, e = _single_leaf(child)
                    rows.append((_fmt_leaf(indent, f"{compressed}/{tail}"), e))
            for fname, e in sorted(node["files"], key=lambda t: t[0]):
                rows.append((_fmt_leaf(indent, fname), e))

        _walk(root, 0)

        table = Table(
            title=title or f"CorrectionlibHandler status  [dim]({self._run} {self._era}"
            + (f"-{self._subera}" if self._subera else "")
            + f", NanoAOD{self._ver}, {self._analysis})[/dim]",
            title_justify="left",
            expand=False,
            header_style="bold",
            caption=(
                f"[dim]cvmfs POG root:[/dim] {self._poghead}\n"
                f"[dim]qawa data root (~):[/dim] {_QAWA_DATA}\n"
                "[dim]newer tags: [/dim][rgb(0,180,0)]0 (current)[/] .. "
                "[rgb(200,0,0)]>=5 behind[/]   "
                "[magenta]magenta = non-date tag pinned[/magenta]"
            ),
            caption_justify="left",
        )
        table.add_column("path", overflow="fold")
        table.add_column("name", style="bold cyan", no_wrap=True)
        table.add_column("exists", justify="center")
        table.add_column("loaded", justify="center")
        table.add_column("newer tags", justify="right")

        for path_cell, e in rows:
            if e is None:
                table.add_row(path_cell, "", "", "", "")
            else:
                table.add_row(path_cell, e["key"], e["exists"], e["loaded"], e["newer"])

        if print_it:
            console.print(table)
        return table

    def keys(self):
        return self._paths.keys()

    def values(self):
        return self._paths.values()

    def items(self):
        return self._paths.items()
    
    def __getitem__(self, key):
        return self._paths[key]

    def __setitem__(self, key, value):
        value = Path(value)
        if not value.exists():
            raise ValueError(f"Refusing to set {key}: path does not exist: {value}")
        if key in self._paths:
            coffea_console.print(f"[red]Unexpectedly overwriting {key}[/red] [yellow]{escape(str(self._paths[key]))}[/yellow] with [blue]{escape(str(value))}[/blue]")
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
        ch.printStatus()
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
