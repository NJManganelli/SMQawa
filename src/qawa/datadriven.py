import copy
import correctionlib
import json
import os
import re
import awkward as ak
from pathlib import Path
from coffea.util import coffea_console
import numpy as np
import hist
import rich
from hist.intervals import ratio_uncertainty
from typing import Dict, List, Tuple, Optional


def discover_systematics(json_path):
    """Scan a correctionlib JSON for the systematic variations carried by the correction.

    Returns ``(stat_systematics, mc_systematics, has_legacy_dddy)``:

    - ``stat_systematics``: sorted per-era statistical nuisance base names (paired Up/Down),
      e.g. ["stat_2016", "stat_2018"]. These are unique to the data-driven estimate and
      decorrelated across eras. A single-era estimate still carries its own (single) entry,
      e.g. ["stat_2024"].
    - ``mc_systematics``: every other paired variation, e.g. ["JES", "pileup_weight", ...].
      These are the underlying MC systematics propagated through the non-DY subtraction and
      should be correlated with the same-named analysis nuisances, so they keep their bare names.
    - ``has_legacy_dddy``: whether the legacy single combined statistical nuisance "DDDY"
      (paired DDDYUp/DDDYDown) is present. It is excluded from ``mc_systematics``.
    """
    with open(json_path) as f:
        data = json.load(f)
    keys = set()

    def _walk(node):
        if isinstance(node, dict):
            if node.get("nodetype") == "category":
                for item in node.get("content", []):
                    if isinstance(item, dict) and "key" in item:
                        keys.add(item["key"])
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(data)
    pattern = re.compile(r"^(.+)(Up|Down)$")
    ups, downs = set(), set()
    for k in keys:
        m = pattern.match(k)
        if not m:
            continue
        (ups if m.group(2) == "Up" else downs).add(m.group(1))
    paired = ups & downs
    stat_systematics = sorted(b for b in paired if b.startswith("stat_"))
    mc_systematics = sorted(b for b in paired if not b.startswith("stat_") and b != "DDDY")
    has_legacy_dddy = "DDDY" in paired
    return stat_systematics, mc_systematics, has_legacy_dddy


class DataDrivenEventReweight:
    """Reweight events using the data-driven fake-tau / P(DY) correction set.

    Loads the requested compound correction (the DDDY estimate by default) and, when loaded
    from a JSON file, exposes the per-era statistical nuisances (``stat_systematics``) and the
    propagated MC systematics (``mc_systematics``) the consumer should fold into its weights.
    """

    def __init__(self, era: str = "2018", estimator: str = "LNTTau_VTTau_DDDY_Estimate", path=None, clibhandler=None,
                 isAPV: bool = False, isEE: bool = False, isBPix: bool = False):
        # Era/subera label (same "era + _subera" convention used elsewhere, e.g. leptonsSF) used to
        # match the per-era statistical nuisance (stat_{erasubera}) of the (possibly multi-era
        # averaged) estimate. The stat keys carry no underscore (stat_2016APV), so it is stripped
        # when matching; see combine_era_corrections for how these are derived.
        self.erasubera = era
        if isAPV:
            self.erasubera += "_APV"
        elif isEE:
            self.erasubera += "_EE"
        elif isBPix:
            self.erasubera += "_BPix"

        if clibhandler is not None:
            self.dd_estimator = clibhandler.getCorrectionSet("dddy").compound[estimator]
            # Systematic discovery needs the raw JSON category keys, which the cached correctionlib
            # CorrectionSet does not expose; getPath + bare json.load (in discover_systematics) does
            # not re-open the file through correctionlib, so it does not defeat the handler's cache.
            _data_path = clibhandler.getPath("dddy", fallbackNone=True)
            if _data_path is not None and Path(_data_path).exists():
                self.stat_systematics, self.mc_systematics, self.has_legacy_dddy = discover_systematics(_data_path)
            else:
                self.stat_systematics = []
                self.mc_systematics = []
                self.has_legacy_dddy = False
        else:
            if path is None:
                path = Path(os.path.dirname(__file__)) / f"data/dd/{era}/WZ_inclusive_data_driven_{era}.json"
            _data_path = Path(path)
            assert _data_path.exists(), f"DataDrivenEventReweight could not find the expected json file: {str(_data_path)}"
            cset = correctionlib.CorrectionSet.from_file(str(_data_path))
            if estimator not in cset.compound:
                raise KeyError(f"Must select a valid CompoundCorrection from the available set: {list(cset.compound.keys())}")
            self.dd_estimator = cset.compound[estimator]
            self.stat_systematics, self.mc_systematics, self.has_legacy_dddy = discover_systematics(_data_path)

        # Every estimate carries a statistical component. A modern estimate exposes it as one
        # decorrelated nuisance per era it was built from (stat_2016, stat_2016APV, ..., or just
        # stat_2024 for a single-era build); a legacy estimate exposes one combined "DDDY". Only
        # the per-era stat matching the era/subera being processed applies to these events (the
        # others stay at nominal, keeping the per-era stat nuisances decorrelated in the fit).
        self.all_stat_systematics = list(self.stat_systematics)
        self.stat_systematics = [s for s in self.all_stat_systematics if s == f"stat_{self.erasubera.replace('_', '')}"]
        # Sanity check: the file must carry the statistical component for the era being processed,
        # either as its matching per-era stat nuisance or (legacy) as the combined DDDY. Otherwise
        # the wrong correction file is in use for this era (e.g. a Run2-only average for 2024).
        if not self.stat_systematics and not self.has_legacy_dddy:
            raise RuntimeError(
                f"data-driven estimate '{estimator}' from {str(_data_path)} carries no statistical "
                f"nuisance for era/subera '{self.erasubera}': expected 'stat_{self.erasubera.replace('_', '')}' "
                f"among {self.all_stat_systematics} or a legacy 'DDDY'. Wrong correction file for this era?"
            )

    def estimate_dd_DY(self, jet_multiplicity, tau_pt, systematic: str = None):
        systematic = "nominal" if systematic is None else systematic
        return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0.0), ak.fill_none(tau_pt, 0.0), systematic)

def histogram_extractor(config,
                        variable,
                        channel,
                        rebin=1,
                        xlim=[],
                        blind=False,
                        era="someyear",
                        checksyst=True,
                        remap_replacement_types = None,
                        logy=True,
                        logx=False,
                        bin_width_norm=None,
                        no_ratios=False) -> None:
    if remap_replacement_types is None:
        remap_replacement_types = [] #expected args: "datadriven", "validation"
    datasets:Dict = dict()
    color_cycle:List = []

    if bin_width_norm is None and "bin_width_norm" in config:
        bin_width_norm = config.bin_width_norm

    for ng, name in enumerate(config.groups):
        # handle the plotting of histograms directly from SMQawa
        histograms = dict(
            filter(
                lambda _n: _n[0] in config.groups[name].processes,
                config.boosthist.items()
            )
        )
        p = datagroup(
            histograms       = histograms,
            ptype            = config.groups[name].type,
            observable       = variable,
            name             = name,
            xsections        = config.xsections,
            channel          = channel,
            luminosity       = config.luminosity.value,
            rebin            = rebin,
            remap_class_name = config.groups[name].remap_class_name if "remap_class_name" in config.groups[name] else None,
        )
        #remap_replacement_types lets us control whether we replace a given process with a remap type, such as a datadriven estimate. 
        # The remap_class should have a method which returns a tuple of the config group name for which a remapped group replaces, and what type it is categorized as
        # for example, in WZ, we have a data driven estimate for SR0 and SR1 derived from B0 and B1, and these are called "datadriven" to indicate they are for full replacement
        # of the DY MonteCarlo
        # Meanwhile, we can do some crossvalidation/closure tests by looking at the datadriven etimate derived for other regions, so their type is "validation"
        # to toggle datadriven types and/or validation types (or any other type name you choose) to replace the given process, just add it to the remap_replacement_types list
        if p.remap_replace_group_name is not None:
            if p.remap_replace_type in remap_replacement_types:
                print(f"Overwriting: channel: {p.channel} type: {p.remap_replace_type}, {p.remap_replace_group_name} replaced by {p.name}")
                # overwrite a previously defined dataset in the dictionary. This requires the remap types to be after ALL MC in the config file (and still before the real data)
                datasets[p.remap_replace_group_name] = p
                if hasattr(config.groups[name], "color") and len(p.to_boost().shape):
                    # must replace the previous color cycler...
                    index = list(datasets.keys()).index(p.remap_replace_group_name)
                    color_cycle[index] = config.groups[name].color
            else:
                print(f"Skipping: channel: {p.channel} type: {p.remap_replace_type}, {p.remap_replace_group_name} would have been replaced by {p.name}")
                # this process is ignored / not added to the stack
                continue
        else:
            # nominal path for MC/data which doesn't have a remap_class and
            datasets[p.name] = p
            # add the new color to the color cycler...
            if hasattr(config.groups[name], "color") and len(p.to_boost().shape):
                color_cycle.append(config.groups[name].color)
        if p.ptype == "signal":
            signal = p.name

    return {"datasets": datasets,
            "color_cycle": color_cycle,
            }

def inc_WZ_dddy_tf_ratio_and_uncertainty(data_hists, non_dy_mc_hists, data_syst:str = "", non_dy_mc_syst:str = "", numerator_key:str = "", denominator_key:str = ""):
    numerator = data_hists[numerator_key][{"systematic": data_syst}] + (-1)*non_dy_mc_hists[numerator_key][{"systematic": non_dy_mc_syst}]
    denominator = data_hists[denominator_key][{"systematic": data_syst}] + (-1)*non_dy_mc_hists[denominator_key][{"systematic": non_dy_mc_syst}]
    num_v = numerator.values()
    den_v = denominator.values()
    # Guard empty bins (e.g. unpopulated high-tau_pt bins): a zero denominator must not bake
    # NaN/inf into the correction content (correctionlib rejects non-finite values).
    ratio = np.divide(num_v, den_v, out=np.zeros_like(den_v, dtype=float), where=den_v != 0)
    ratio_unc_up, ratio_unc_down = ratio_uncertainty(num_v, den_v, uncertainty_type="poisson")
    ratio_unc_up = np.nan_to_num(ratio_unc_up, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_unc_down = np.nan_to_num(ratio_unc_down, nan=0.0, posinf=0.0, neginf=0.0)
    return numerator, denominator, ratio, (ratio_unc_up, ratio_unc_down)

def inc_WZ_dddy_probdy_ratio_and_uncertainty(data_hists, dy_mc_hists, non_dy_mc_hists, data_syst:str = "", mc_syst:str = "", channel:str = "", probdy_method:str = "data_subtraction"):
    match probdy_method:
        case "data_subtraction":
            numerator = data_hists[channel][{"systematic": data_syst}] + (-1)*non_dy_mc_hists[channel][{"systematic": mc_syst}]
            denominator = data_hists[channel][{"systematic": data_syst}]
        case "pure_mc":
            numerator = dy_mc_hists[channel][{"systematic": mc_syst}]
            denominator = dy_mc_hists[channel][{"systematic": mc_syst}] + non_dy_mc_hists[channel][{"systematic": mc_syst}]
        case _:
            raise NotImplementedError(f"Unhandled probdy_method {probdy_method}")
    # P(DY) is a probability in [0, 1]. A negative *total* non-DY MC in a bin (from negative
    # MC weights / over-subtraction) drives the raw numerator above the denominator, which is
    # unphysical and illegal for the binomial (efficiency) interval. Clamp the counts to
    # 0 <= numerator <= denominator before forming the ratio and its uncertainty (such bins
    # become P(DY) -> 1, i.e. effectively all Drell-Yan).
    num_v = np.clip(numerator.values(), 0.0, None)
    den_v = np.clip(denominator.values(), 0.0, None)
    num_v = np.minimum(num_v, den_v)
    ratio = np.divide(num_v, den_v, out=np.zeros_like(den_v, dtype=float), where=den_v > 0)
    ratio_unc_up, ratio_unc_down = ratio_uncertainty(num_v, den_v, uncertainty_type="efficiency")
    ratio_unc_up = np.nan_to_num(ratio_unc_up, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_unc_down = np.nan_to_num(ratio_unc_down, nan=0.0, posinf=0.0, neginf=0.0)
    return numerator, denominator, ratio, (ratio_unc_up, ratio_unc_down)

def _prepare_method_hists(histogroups, ddcorrconfig):
    """Build per-channel data / DY / non-DY MC / template boost histograms for a single
    era's correction, plus the sorted list of MC systematic variations present."""
    from dctools import update_axes_meta
    data_hists, dy_mc_hists, non_dy_mc_hists, template_hists = {}, {}, {}, {}
    mc_systematics = None
    for channel in histogroups:
        data_hists[channel] = histogroups[channel]["datasets"]["data"].to_boost()
        dy_mc_hists[channel] = histogroups[channel]["datasets"]["DY"].to_boost()
        # Drop empty (0-dim) group histograms before summing: a background group that has no
        # entries in this region contributes zero, but summing a 0-dim hist (especially first)
        # collapses the result and discards the systematic axis. Skipping them makes the sum
        # order-independent and robust to a single background missing a region.
        non_dy_components = [v.to_boost() for k, v in histogroups[channel]["datasets"].items() if k not in ["data", "DY"]]
        non_dy_components = [h for h in non_dy_components if len(h.axes) > 0]
        if not non_dy_components:
            raise RuntimeError(f"No non-DY MC backgrounds with entries found in region '{channel}' for this config; "
                               f"cannot perform the data-driven subtraction (the region appears to be missing in the histograms)")
        non_dy_mc_hists[channel] = sum(non_dy_components)
        template_hists[channel] = update_axes_meta(
            data_hists[channel].project(*[ax.name for ax in data_hists[channel].axes if ax.name != "systematic"]).copy().reset(),
            ddcorrconfig.update_axes_meta,
        )
        if mc_systematics is None:
            mc_systematics = sorted([x for x in non_dy_mc_hists[channel].axes["systematic"]])
    return data_hists, dy_mc_hists, non_dy_mc_hists, template_hists, mc_systematics


def compute_era_correction(histogroups, ddcorrconfig, probdy_method):
    """Compute the per-bin ratios for a single era / config for one Correction.

    Returns a dict::

        {
            "ratios": { axis_key: { syst: {"ratio": .., "unc_up": .., "unc_down": ..} } },
            "template": <1D template hist over the observable>,
            "is_probdy": bool,
        }

    Only the ``"nominal"`` syst entry carries ``unc_up``/``unc_down`` (the statistical
    uncertainty of the ratio); MC systematic entries carry only ``"ratio"``. The
    averaging / stat handling across eras is done later in ``combine_era_corrections``.
    """
    data_hists, dy_mc_hists, non_dy_mc_hists, template_hists, mc_systematics = _prepare_method_hists(histogroups, ddcorrconfig)
    ratios = {}
    is_probdy = False
    match ddcorrconfig.method:
        case "inc_WZ_DYDD_TransferFactor":
            for axis_key, num_den_dict in ddcorrconfig.channel_ratios.items():
                numerator_key, denominator_key = num_den_dict["numerator"], num_den_dict["denominator"]
                per_syst = {}
                for mc_syst in mc_systematics:
                    _, _, ratio, (ratio_unc_up, ratio_unc_down) = inc_WZ_dddy_tf_ratio_and_uncertainty(
                        data_hists,
                        non_dy_mc_hists,
                        data_syst="nominal",
                        non_dy_mc_syst=mc_syst,
                        numerator_key=numerator_key,
                        denominator_key=denominator_key,
                    )
                    if mc_syst == "nominal":
                        per_syst["nominal"] = {"ratio": ratio, "unc_up": ratio_unc_up, "unc_down": ratio_unc_down}
                    else:
                        per_syst[mc_syst] = {"ratio": ratio}
                ratios[axis_key] = per_syst
        case "inc_WZ_DYDD_ProbDY":
            is_probdy = True
            for axis_key, channel in ddcorrconfig.channel_absolutes.items():
                per_syst = {}
                for mc_syst in mc_systematics:
                    _, _, ratio, (ratio_unc_up, ratio_unc_down) = inc_WZ_dddy_probdy_ratio_and_uncertainty(
                        data_hists,
                        dy_mc_hists,
                        non_dy_mc_hists,
                        data_syst="nominal",
                        mc_syst=mc_syst,
                        channel=channel,
                        probdy_method=probdy_method,
                    )
                    if mc_syst == "nominal":
                        per_syst["nominal"] = {"ratio": ratio, "unc_up": ratio_unc_up, "unc_down": ratio_unc_down}
                    else:
                        per_syst[mc_syst] = {"ratio": ratio}
                ratios[axis_key] = per_syst
        case _:
            raise NotImplementedError(f"Unknown method for data driven correction {ddcorrconfig.method}")
    # All channel templates share binning after update_axes_meta; any one suffices for projection.
    template = next(iter(template_hists.values()))
    return {"ratios": ratios, "template": template, "is_probdy": is_probdy}


def _axes_signature(h):
    return [(ax.name, tuple(np.asarray(getattr(ax, "edges", [])).tolist())) for ax in h.axes]


def combine_era_corrections(per_era_ratios, eras, lumis, templates, is_probdy):
    """Luminosity-weighted average of per-era ratios into a single Correction's slices.

    - nominal      : sum_e(L_e * r_e) / sum_e(L_e), with the (quadrature) combined stat
                     stored in the variance slot for reference.
    - stat_{era}   : the combination recomputed with only that era's nominal ratio shifted
                     by +/- its own statistical uncertainty (uncorrelated across eras).
    - MC syst      : luminosity-weighted average over the union of MC systematics across
                     eras; eras missing a given variation fall back to their nominal ratio.

    For P(DY) corrections the upward stat slice is clipped to <= 1.0 (a probability).
    Returns ``(dict_for_hist_axis, reference_template)``.
    """
    from dctools import dict_to_hist_axis
    ref_template = templates[0]
    ref_sig = _axes_signature(ref_template)
    for era, t in zip(eras[1:], templates[1:]):
        assert _axes_signature(t) == ref_sig, (
            f"Inconsistent binning for era {era} vs {eras[0]}; rebin configs to a common "
            f"scheme before averaging across eras"
        )
    L = np.asarray(lumis, dtype=float)
    wsum = L.sum()
    n = len(per_era_ratios)
    axis_keys = list(per_era_ratios[0].keys())
    dict_for_hist_axis = {}
    for axis_key in axis_keys:
        nominal = [per_era_ratios[i][axis_key]["nominal"] for i in range(n)]
        nom_ratios = [d["ratio"] for d in nominal]
        unc_up = [d["unc_up"] for d in nominal]
        unc_down = [d["unc_down"] for d in nominal]

        comb_nominal = sum(L[i] * nom_ratios[i] for i in range(n)) / wsum
        comb_stat = np.sqrt(sum((L[i] / wsum * unc_up[i]) ** 2 for i in range(n)))

        dict_for_syst_axis = {}
        # https://github.com/scikit-hep/boost-histogram/issues/421 - stack [value, variance] on the view
        h_nom = ref_template.copy()
        h_nom[...] = np.stack([comb_nominal, comb_stat], axis=-1)
        dict_for_syst_axis["nominal"] = h_nom

        # Per-era statistical uncertainty: recombine with only era i shifted (decorrelated across eras)
        for i, era in enumerate(eras):
            up_ratios = list(nom_ratios)
            up_ratios[i] = nom_ratios[i] + unc_up[i]
            down_ratios = list(nom_ratios)
            down_ratios[i] = nom_ratios[i] - unc_down[i]
            comb_up = sum(L[j] * up_ratios[j] for j in range(n)) / wsum
            comb_down = sum(L[j] * down_ratios[j] for j in range(n)) / wsum
            if is_probdy:
                comb_up = np.clip(comb_up, a_min=None, a_max=1.0)
            h_up = ref_template.copy()
            h_up[...] = np.stack([comb_up, np.zeros_like(comb_up)], axis=-1)
            h_down = ref_template.copy()
            h_down[...] = np.stack([comb_down, np.zeros_like(comb_down)], axis=-1)
            dict_for_syst_axis[f"stat_{era}Up"] = h_up
            dict_for_syst_axis[f"stat_{era}Down"] = h_down

        # MC systematics: union across eras, falling back to per-era nominal where disjoint
        mc_systs = sorted(set().union(*[set(per_era_ratios[i][axis_key].keys()) for i in range(n)]) - {"nominal"})
        for mc_syst in mc_systs:
            syst_ratios = []
            for i in range(n):
                entry = per_era_ratios[i][axis_key].get(mc_syst)
                syst_ratios.append(entry["ratio"] if entry is not None else nom_ratios[i])
            comb = sum(L[i] * syst_ratios[i] for i in range(n)) / wsum
            h_s = ref_template.copy()
            h_s[...] = np.stack([comb, np.zeros_like(comb)], axis=-1)
            dict_for_syst_axis[mc_syst] = h_s

        dict_for_hist_axis[axis_key] = dict_to_hist_axis(
            histos=dict_for_syst_axis,
            axis_name="systematic",
            axis_label="Systematic Variation",
            axis_type="StrCategory",
            axis_storage=hist.storage.Weight(),
            axis_args={"growth": False},
        )
    return dict_for_hist_axis, ref_template


if __name__ == "__main__":
    import argparse
    import dctools
    from dctools import datagroup, dict_to_hist_axis, update_axes_meta

    parser = argparse.ArgumentParser(description='SMQawa DataDriven Derivation')
    parser.add_argument("-i", "--input", type=str, nargs="+", default=[],
                        help="Input YAML config file(s). Pass one config per era to average across eras.")
    parser.add_argument("-y", "--era", type=str, default='2018',
                        help="Fallback era label, used only for a single config that omits a top-level 'era' key.")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Output correctionlib JSON path for the (averaged) correction set. "
                             "Defaults to data/dd/{era}/WZ_inclusive_data_driven_{era}.json for a single config.")
    parser.add_argument("--derive", action="store_true", help="Derive corrections from histograms")
    parser.add_argument("--skip_json", action="store_true", help="Skip writing correctionlib JSON")
    parser.add_argument("--probdy_method", type=str, default="pure_mc",
                        choices=["pure_mc", "data_subtraction"],
                        help="P(DY) method: 'pure_mc' or 'data_subtraction'")
    parser.add_argument("--plot", action="store_true", help="Generate PDF plots")
    parser.add_argument("--plot_dir", type=str, default=".", help="Output directory for plots")
    parser.add_argument("--n_systematics", type=int, default=5,
                        help="Number of top systematics to show in plots")
    options = parser.parse_args()

    config_paths = options.input if isinstance(options.input, list) else ([options.input] if options.input else [])
    configs = [dctools.read_config(p) for p in config_paths]

    # Derive corrections
    derived_results = None
    if options.derive:
        from correctionlib import convert
        from correctionlib import schemav2 as cs
        new_corrections = []
        new_compoundcorrections = []
        if not configs:
            raise RuntimeError("--derive requires at least one --input config file")
        # Resolve the era label for each config (used for the per-era stat_{era} nuisance names)
        eras = []
        for c in configs:
            if "era" in c:
                eras.append(str(c.era))
            elif len(configs) == 1:
                eras.append(str(options.era))
            else:
                raise RuntimeError("When averaging across multiple configs, each config must declare a top-level 'era' key")
        lumis = [float(c.luminosity.value) for c in configs]
        print(f"=== Deriving data-driven corrections averaged over eras {eras} (lumi weights {lumis}) ===")

        base = configs[0]
        for ddconfig_key in base.datadriven:
            ddconfig = base.datadriven[ddconfig_key]
            for ddcorrconfig_key in ddconfig["Corrections"]:
                # Compute the per-bin ratios independently for each era / config
                per_era_ratios, templates, is_probdy = [], [], False
                for config in configs:
                    ddcorrconfig = config.datadriven[ddconfig_key]["Corrections"][ddcorrconfig_key]
                    # Per-correction fake-rate binning override (independent of the plotting
                    # `rebin`), falling back to a global config.rebin then to 1 (no rebinning).
                    ddrebin = ddcorrconfig.rebin if "rebin" in ddcorrconfig else (config.rebin if "rebin" in config else 1)
                    histogroups = {}
                    for channel, variable in ddcorrconfig.channels.items():
                        histogroups[channel] = histogram_extractor(
                            config,
                            variable,
                            channel,
                            rebin=ddrebin,
                            xlim=[],
                            blind=False,
                            era="someyear",
                            checksyst=False,
                            remap_replacement_types=None,
                        )
                    era_res = compute_era_correction(histogroups, ddcorrconfig, options.probdy_method)
                    per_era_ratios.append(era_res["ratios"])
                    templates.append(era_res["template"])
                    is_probdy = era_res["is_probdy"]

                # Average across eras: nominal is lumi-weighted, statistics become per-era
                # stat_{era} nuisances, MC systematics are correlated (union, nominal fallback).
                dict_for_hist_axis, template = combine_era_corrections(per_era_ratios, eras, lumis, templates, is_probdy)

                dthac = base.datadriven[ddconfig_key]["Corrections"][ddcorrconfig_key].dict_to_hist_axis
                final_hist = dict_to_hist_axis(
                    histos=dict_for_hist_axis,
                    axis_name=dthac.axis_name,
                    axis_label=dthac.axis_label,
                    axis_type=dthac.axis_type,
                    axis_storage=dthac.axis_storage,
                    axis_iter_override=dthac.axis_iter_override if "axis_iter_override" in dthac else None,
                # Order inputs as (jet_multiplicity, tau_pt, systematic) to match how the
                # consumer (DataDrivenEventReweight.estimate_dd_DY) calls evaluate(); from_histogram
                # uses axis order for the declared input order.
                ).project(dthac.axis_name, template.axes[0].name, "systematic")
                # Set the correctionset (name) and output node (label)
                ddcorrconfig = base.datadriven[ddconfig_key]["Corrections"][ddcorrconfig_key]
                final_hist.name = ddcorrconfig_key
                final_hist.label = ddcorrconfig.correctionset_output_label
                # Set the overflow behavior and descriptions
                corr = correctionlib.convert.from_histogram(final_hist)
                corr.data.flow = ddcorrconfig.correctionset_flow
                corr.description = ddcorrconfig.correctionset_top_description
                corr.output.description = ddcorrconfig.correctionset_output_description
                new_corrections.append(corr)

            for ddcompcorrconfig_key in ddconfig["CompoundCorrections"]:
                ddcompcorrconfig = ddconfig["CompoundCorrections"][ddcompcorrconfig_key]
                ddcompstack = ddcompcorrconfig.stack
                assert (isinstance(ddcompstack, list) and len(ddcompstack) > 0)
                # Use a dictionary
                ddcompinputs = {}
                for stk_key in ddcompstack:
                    try:
                        corrinputs = [x.inputs for x in new_corrections if x.name == stk_key][0]
                    except IndexError as e:
                        coffea_console.print(f"Failed to find matching input for CompoundCorrection [red]{ddcompcorrconfig_key}[/red]: {stk_key}"
                                      f"Available corrections are: {[x.name for x in new_corrections]}"
                                      )
                        raise e
                    for corrinp in corrinputs:
                        ddcompinputs[corrinp.name] = corrinp
                comp = cs.CompoundCorrection(
                    name=ddcompcorrconfig_key,
                    description=ddcompcorrconfig.correctionset_top_description,
                    inputs = ddcompinputs.values(),
                    output=cs.Variable(name=ddcompcorrconfig.correctionset_output_label,
                                       type=ddcompcorrconfig.correctionset_output_type,
                                       description=ddcompcorrconfig.correctionset_output_description),
                    inputs_update=ddcompcorrconfig.inputs_update,
                    input_op=ddcompcorrconfig.input_op,
                    output_op=ddcompcorrconfig.output_op,
                    stack=ddcompstack,
                )
                new_compoundcorrections.append(comp)
        new_cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            corrections=new_corrections,
            compound_corrections=new_compoundcorrections,
        )
        if options.output:
            outfile = Path(options.output)
        else:
            outfile = Path(os.path.dirname(__file__)) / f"data/dd/{eras[0]}/WZ_inclusive_data_driven_{eras[0]}.json"
        if not options.skip_json:
            outfile.parent.mkdir(parents=True, exist_ok=True)
            with open(outfile, "w") as fout:
                fout.write(new_cset.model_dump_json(exclude_unset=True))
            print(f"Wrote to {outfile}"
                  f"\n\tCorrections: {[x.name for x in new_corrections]}\n\tCompoundCorrections: {[x.name for x in new_compoundcorrections]}"
                  )
            rich.print(new_cset)
        else:
            print(f"Skipping write of new correctionset json: with {len(new_corrections)} Corrections and {len(new_compoundcorrections)} CompoundCorrections to {outfile}"
                  f"\nWould have respectivelywritten \n\tCorrections: {[x.name for x in new_corrections]}\n\tCompoundCorrections: {[x.name for x in new_compoundcorrections]}"
                  )
        if options.plot:
            raise NotImplementedError

    # Test
    print("\n=== Testing DataDrivenEventReweight ===")
    test_path = options.output if options.output else None
    try:
        test = DataDrivenEventReweight(era=options.era, estimator="LNTTau_VTTau_DDDY_Estimate", path=test_path)
        test_njets = np.array([0, 0, 0, 1, 1])
        test_tau_pt = np.array([25, 80, 55, 33, 110])
        print(f"njets: {test_njets}, tau_pt: {test_tau_pt}")
        print(f"weights (nominal): {test.estimate_dd_DY(test_njets, test_tau_pt)}")
        test2 = DataDrivenEventReweight(era=options.era, estimator="LNTTau_VTTau_DDDY_Closure", path=test_path)
        print(f"Closure weights (nominal): {test2.estimate_dd_DY(test_njets, test_tau_pt)}")
    except AssertionError as e:
        print(f"Skipping test: {e}")
