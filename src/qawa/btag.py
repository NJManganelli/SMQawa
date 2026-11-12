import re

import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights
from coffea.util import coffea_console

from qawa.jsoncorrections import category_keys

FIXED_WPS = {"deepJet": ("L", "M", "T"), "UParTAK4": ("L", "M", "T", "XT", "XXT")}
BTAG_MAX_ABSETA = 2.4
BC_CORRECTIONS = ("comb", "mujets")
SUMMARY_SYSTEMATICS = {"central", "up", "down", "up_correlated", "down_correlated", "up_uncorrelated", "down_uncorrelated"}
JES_SHIFT_PATTERN = re.compile(r"^JES_([A-Za-z]+\d?)(\d{4})?(Up|Down)$")


def btag_disc_wps(tagger:str='deepJet'):
    _disc_wps = {
        "deepJet": {
            "discriminant": "btagDeepFlavB",
            "workingpoints": {
                "2016"   : {"L": 0.0508, "M": 0.2598, "T": 0.6502},
                "2016APV": {"L": 0.0480, "M": 0.2489, "T": 0.6377},
                "2017"   : {"L": 0.0532, "M": 0.3040, "T": 0.7476},
                "2018"   : {"L": 0.0490, "M": 0.2783, "T": 0.7100}
            },
        },
        "UParTAK4": {
            "discriminant": "btagUParTAK4B",
            "workingpoints": None # dynamically loaded from btagging.json.gz instead
        },
    }
    return _disc_wps[tagger]


def btv_shift_systematic(shift_name):
    """BTV systematic paired with a processor JES-source or JER shift, None for any other shift."""
    if shift_name in ("JERUp", "JERDown"):
        return f"{shift_name[3:].lower()}_jer"
    match = JES_SHIFT_PATTERN.match(shift_name or "")
    if match is None:
        return None
    source, year, direction = match.groups()
    return f"{direction.lower()}_jes_{source.lower()}" + (f"_{year}" if year else "")


def method1a_weight(sf, eff, passed, wps):
    """Per-jet BTV fixed-WP method 1a factors for the WPs `wps` (loose to tight).

    sf, eff and passed map each WP to flat per-jet arrays. Returns (factor, guarded): a jet whose
    factor has a non-positive numerator or denominator is guarded and gets factor 1.
    """
    sf_eff = np.stack([np.ones_like(eff[wps[0]])] + [sf[wp] * eff[wp] for wp in wps])
    eps = np.stack([np.ones_like(eff[wps[0]])] + [eff[wp] for wp in wps])
    npassed = np.sum([np.asarray(passed[wp], dtype=int) for wp in wps], axis=0)
    tight = npassed == len(wps)
    lower = np.minimum(npassed, len(wps) - 1)
    jets = np.arange(len(npassed))
    numerator = sf_eff[lower, jets] - np.where(tight, 0., sf_eff[lower + 1, jets])
    denominator = eps[lower, jets] - np.where(tight, 0., eps[lower + 1, jets])
    guarded = ~tight & ((numerator <= 0) | (denominator <= 0))
    factor = np.where(tight, sf[wps[-1]], np.where(guarded, 1., numerator / np.where(guarded, 1., denominator)))
    return factor, guarded


class BTVCorrector:
    weight_names = ("btag_sf",)

    def __init__(self, era:str='2018', wp:str='L', tagger:str='deepJet', isAPV=False, isEE=False, isBPix=False,
                 clibhandler=None, wps=None, bc_correction="comb", *, jes_shifts):
        if clibhandler is None:
            raise ValueError("BTVCorrector needs a CorrectionlibHandler")
        assert tagger in FIXED_WPS
        assert wp in FIXED_WPS[tagger]
        self._era = era
        self._wp  = wp
        self.isAPV = isAPV
        self.isEE = isEE
        self.isBPix = isBPix
        self._tagger = tagger
        self._disc = btag_disc_wps(tagger=tagger)["discriminant"]
        self._erasubera = era
        if isAPV:
            self._erasubera += "_APV"
        elif isEE:
            self._erasubera += "_EE"
        elif isBPix:
            self._erasubera += "_BPix"
        self._erasuberanousc = self._erasubera.replace("_", "")
        self.wps = (wp,) if wps is None else tuple(wps)
        order = [FIXED_WPS[tagger].index(w) for w in self.wps if w in FIXED_WPS[tagger]]
        if len(order) != len(self.wps) or order != sorted(set(order)):
            raise ValueError(f"wps {self.wps} must be distinct {tagger} fixed WPs from loose to tight {FIXED_WPS[tagger]}")
        self.clib = clibhandler.getCorrectionSet("btagging")
        if bc_correction not in BC_CORRECTIONS or f"{tagger}_{bc_correction}" not in list(self.clib):
            raise ValueError(f"b/c correction {tagger}_{bc_correction} is not in {clibhandler.getPath('btagging')}")
        self._bc_name = f"{tagger}_{bc_correction}"
        self._light_name = f"{tagger}_light"
        self.efficiency_available = tagger == "UParTAK4" and "btag_eff" in clibhandler.keys()
        self.guard_counts = {}
        self._inclusive_reported = set()
        if tagger != "UParTAK4":
            return
        bc_systematics = category_keys(clibhandler.getPath("btagging"), self._bc_name)
        self._bc_systematics = set(bc_systematics)
        unmatched = [s for s in jes_shifts if btv_shift_systematic(s) not in self._bc_systematics]
        self.paired = bool(jes_shifts) and not unmatched
        if not self.paired:
            coffea_console.print(f"[BTVCorrector] {self._bc_name} has no systematics for the shifts "
                                 f"{unmatched or 'none given'}: b-tag SFs are not paired with JES/JER passes, "
                                 "using correlated/uncorrelated in the nominal pass", markup=False)
        era_name = self._erasuberanousc
        if self.paired:
            sources = dict.fromkeys(s.split("_", 1)[1] for s in bc_systematics
                                    if s not in SUMMARY_SYSTEMATICS and "jes" not in s and "jer" not in s)
            self.variations = {(f"bc_{src}_{era_name}" if src == "statistic" else f"bc_{src}"): ("bc", src)
                               for src in sources}
        else:
            self.variations = {"bc_correlated": ("bc", "correlated"), f"bc_uncorrelated_{era_name}": ("bc", "uncorrelated")}
        self.variations.update({"light_correlated": ("light", "correlated"),
                                f"light_uncorrelated_{era_name}": ("light", "uncorrelated"),
                                f"effstat_{era_name}": ("eff", None)})
        if self.efficiency_available:
            self._eff = clibhandler.getCorrectionSet("btag_eff")["btag_eff"]
            self._eff_processes = set(category_keys(clibhandler.getPath("btag_eff"), "btag_eff"))

    def tagged_jets(self, jets: ak.Array, wp=None):
        # permits possible multi-WP expansion in the future
        if wp is None:
            wps = [self._wp]
        elif isinstance(wp, str):
            wps = [wp]
        elif isinstance(wp, list):
            wps = wp
        else:
            raise ValueError(f"Unhandled tagged_jets logic: self._wp={self._wp} ; wp={wp} ; wps={wps}")
        return {wp: (getattr(jets, self._disc)  > self.clib[f"{self._tagger}_wp_values"].evaluate(wp)) for wp in wps}

    def scale_factor(self, jets, correction, systematic, wp):
        return self.clib[correction].evaluate(systematic, wp, np.asarray(jets.hadronFlavour), np.abs(np.asarray(jets.eta)),
                                              np.asarray(jets.pt))

    def efficiency(self, jets, process, wp, systematic):
        if process not in self._eff_processes:
            process_key = "inclusive"
            if process not in self._inclusive_reported:
                self._inclusive_reported.add(process)
                coffea_console.print(f"[BTVCorrector] no b-tag efficiency for {process}, using inclusive", markup=False)
        else:
            process_key = process
        return self._eff.evaluate(process_key, systematic, wp, np.asarray(jets.hadronFlavour), np.asarray(jets.pt),
                                  np.abs(np.asarray(jets.eta)))

    def jet_factors(self, jets, light, process, bc_systematic, light_systematic, eff_systematic):
        sf, eff = {}, {}
        for wp in self.wps:
            sf[wp] = np.ones(len(light))
            sf[wp][light] = self.scale_factor(jets[light], self._light_name, light_systematic, wp)
            sf[wp][~light] = self.scale_factor(jets[~light], self._bc_name, bc_systematic, wp)
            eff[wp] = self.efficiency(jets, process, wp, eff_systematic)
        passed = {wp: np.asarray(tagged) for wp, tagged in self.tagged_jets(jets, wp=list(self.wps)).items()}
        return method1a_weight(sf, eff, passed, self.wps)

    def append_btag_sf(self, jets: ak.Array, weights: Weights, process, shift_name=None, variations=True):
        if self._tagger != "UParTAK4":
            raise NotImplementedError("b-tag SF weights are implemented for UParTAK4 (NanoAODv15) only")
        accepted = jets[np.abs(jets.eta) < BTAG_MAX_ABSETA]
        counts = ak.num(accepted)
        flat = ak.flatten(accepted)
        flat = ak.zip({field: np.asarray(flat[field]) for field in ("pt", "eta", "hadronFlavour", self._disc)})
        light = np.asarray(flat.hadronFlavour) == 0
        bc_nominal = "central"
        if self.paired and btv_shift_systematic(shift_name) is not None:
            bc_nominal = btv_shift_systematic(shift_name)

        def event_weight(factor):
            return ak.to_numpy(ak.prod(ak.unflatten(factor, counts), axis=1))

        nominal, guarded = self.jet_factors(flat, light, process, bc_nominal, "central", "nominal")
        self.report_guarded(shift_name, int(guarded.sum()))
        if not variations:
            weights.add("btag_sf", event_weight(nominal))
            return
        names, ups, downs = [], [], []
        for name, (group, source) in self.variations.items():
            shifted = []
            for direction in ("up", "down"):
                bc_syst, light_syst, eff_syst = bc_nominal, "central", "nominal"
                if group == "bc":
                    bc_syst = f"{direction}_{source}"
                elif group == "light":
                    light_syst = f"{direction}_{source}"
                else:
                    eff_syst = direction
                factor, guarded_var = self.jet_factors(flat, light, process, bc_syst, light_syst, eff_syst)
                self.report_guarded(shift_name, int(guarded_var.sum()))
                shifted.append(event_weight(np.where(guarded_var | guarded, nominal, factor)))
            names.append(name)
            ups.append(shifted[0])
            downs.append(shifted[1])
        weights.add_multivariation("btag_sf", event_weight(nominal), names, ups, downs)

    def report_guarded(self, pass_name, count):
        if count == 0:
            return
        first = pass_name not in self.guard_counts
        self.guard_counts[pass_name] = self.guard_counts.get(pass_name, 0) + count
        if first:
            coffea_console.print(f"[BTVCorrector] {count} jets with a non-positive method-1a numerator or denominator "
                                 f"in pass {pass_name}: factor 1 in the nominal, nominal factor in variations",
                                 markup=False)
