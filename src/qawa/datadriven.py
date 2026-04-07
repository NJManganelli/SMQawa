import copy
import correctionlib
import os
import awkward as ak
from pathlib import Path
import numpy as np
import hist
from typing import Dict, List, Tuple, Optional
import warnings


class DataDrivenEventReweight:
    """Reweight events using data-driven fake tau rate and P(DY) corrections."""

    def __init__(self, era: str = "2018", estimator=None):
        _data_path = Path(os.path.dirname(__file__)) / f"data/dd/{era}/WZ_inclusive_data_driven_{era}.json"
        assert _data_path.exists(), f"DataDrivenEventReweight could not find the expected json file: {str(_data_path)}"
        self.dd_estimator = correctionlib.CorrectionSet.from_file(str(_data_path))
        if estimator is None:
            raise KeyError(f"Must select valid CompoundCorrection from the available set: {[x for x in self.dd_estimator.compound.keys()]}")
        else:
            self.dd_estimator = self.dd_estimator.compound[estimator]

    def estimate_dd_DY(self, jet_multiplicity, tau_pt, systematic=None):
        if systematic is not None:
            return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0), ak.fill_none(tau_pt, 0.0), systematic)
        return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0), ak.fill_none(tau_pt, 0.0), "nominal")

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
    ratio = numerator.values() / denominator.values()
    ratio_unc = ratio_uncertainty(numerator.values(), denominator.values(), uncertainty_type="poisson")
    return numerator, denominator, ratio, ratio_unc

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
    ratio = numerator.values() / denominator.values()
    ratio_unc = ratio_uncertainty(numerator.values(), denominator.values(), uncertainty_type="efficiency")
    return numerator, denominator, ratio, ratio_unc

if __name__ == "__main__":
    import argparse
    import dctools
    from dctools import datagroup, dict_to_hist_axis, update_axes_meta

    parser = argparse.ArgumentParser(description='SMQawa DataDriven Derivation')
    parser.add_argument("-i", "--input", type=str, default="", help="Input YAML config file")
    parser.add_argument("-y", "--era", type=str, default='2018', help="Era (2016/2017/2018)")
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

    config = dctools.read_config(options.input) if options.input else None

    # Derive corrections
    derived_results = None
    if options.derive:
        from correctionlib import convert
        from correctionlib import schemav2 as cs
        from hist.intervals import ratio_uncertainty
        new_corrections = []
        new_compoundcorrections = []
        if config is None:
            raise RuntimeError("--derive requires --input config file")
        else:
            print(f"=== Deriving data-driven corrections for era {options.era} ===")
            datadriven_configs = config.datadriven
            for ddconfig_key in config.datadriven:
                import yaml
                ddconfig = config.datadriven[ddconfig_key]
                for ddcorrconfig_key in ddconfig["Corrections"]:
                    ddcorrconfig = ddconfig["Corrections"][ddcorrconfig_key]
                    histogroups = {}
                    for channel, variable in ddcorrconfig.channels.items():
                        histogroups[channel] = histogram_extractor(config,
                                                                   variable,
                                                                   channel,
                                                                   rebin=config.rebin if "rebin" in config else 1,
                                                                   xlim=[],
                                                                   blind=False,
                                                                   era="someyear",
                                                                   checksyst=False,
                                                                   remap_replacement_types = None,
                                                                   )
                    match ddcorrconfig.method:
                        case "inc_WZ_DYDD_TransferFactor":
                            data_hists = {}
                            non_dy_mc_hists = {}
                            dy_mc_hists = {}
                            template_hists = {}
                            mc_systematics = None
                            for channel in histogroups:
                                data_hists[channel] = histogroups[channel]["datasets"]["data"].to_boost()
                                dy_mc_hists[channel] = histogroups[channel]["datasets"]["DY"].to_boost()
                                non_dy_mc_hists[channel] = sum([v.to_boost() for k, v in histogroups[channel]["datasets"].items() if k not in ["data", "DY"]])
                                template_hists[channel] = update_axes_meta(
                                    data_hists[channel].project(*[ax.name for ax in data_hists[channel].axes if ax.name != "systematic"]).copy().reset(),
                                    ddcorrconfig.update_axes_meta
                                    )
                                if mc_systematics is None:
                                    mc_systematics = sorted([x for x in non_dy_mc_hists[channel].axes["systematic"]])
                            channel_ratios = {}
                            dict_for_hist_axis = {}
                            for dict_to_hist_axis_key, num_den_dict in ddcorrconfig.channel_ratios.items():
                                numerator_key, denominator_key = num_den_dict["numerator"], num_den_dict["denominator"]
                                # key = f"{numerator_key}/{denominator_key}"
                                dict_for_syst_axis = {}
                                for mc_syst in mc_systematics:
                                    # projections of fake rates by systematic
                                    num, den, ratio, (ratio_unc_up, ratio_unc_down) = inc_WZ_dddy_tf_ratio_and_uncertainty(
                                        data_hists,
                                        non_dy_mc_hists,
                                        data_syst="nominal",
                                        non_dy_mc_syst=mc_syst,
                                        numerator_key=numerator_key,
                                        denominator_key=denominator_key,
                                    )    
                                    if mc_syst == "nominal":
                                        # https://github.com/scikit-hep/boost-histogram/issues/421 - can also use h.view().value/variance, but MUST be on view, not h.value!!!
                                        dict_for_syst_axis["nominal"] = template_hists[channel].copy()
                                        dict_for_syst_axis["nominal"][...] = np.stack([ratio, ratio_unc_up], axis=-1)
                                        dict_for_syst_axis["DDDYUp"] = template_hists[channel].copy()
                                        dict_for_syst_axis["DDDYUp"][...] = np.stack([ratio + ratio_unc_up, np.zeros_like(ratio)], axis=-1)
                                        dict_for_syst_axis["DDDYDown"] = template_hists[channel].copy()
                                        dict_for_syst_axis["DDDYDown"][...] = np.stack([ratio - ratio_unc_down, np.zeros_like(ratio)], axis=-1)
                                    else:
                                        dict_for_syst_axis[mc_syst] = template_hists[channel].copy()
                                        dict_for_syst_axis[mc_syst][...] = np.stack([ratio, np.zeros_like(ratio)], axis=-1)
                                dict_for_hist_axis[dict_to_hist_axis_key] = dict_to_hist_axis(
                                    # first promote the systematics axis for which we computed all slices
                                    histos=dict_for_syst_axis,
                                    axis_name="systematic",
                                    axis_label="Systematic Variation",
                                    axis_type="StrCategory",
                                    axis_storage=hist.storage.Weight(),
                                    axis_args={"growth": False},
                                )
                            dthac = ddcorrconfig.dict_to_hist_axis
                            final_hist = dict_to_hist_axis(
                                histos=dict_for_hist_axis,
                                axis_name=dthac.axis_name,
                                axis_label=dthac.axis_label,
                                axis_type=dthac.axis_type,
                                axis_storage=dthac.axis_storage,
                                axis_iter_override=dthac.axis_iter_override if "axis_iter_override" in dthac else None,
                                ).project(template_hists[channel].axes[0].name, dthac.axis_name, "systematic")
                            # Set the correcitonset (name) and outputnode (label)
                            final_hist.name = ddcorrconfig_key
                            final_hist.label = ddcorrconfig.correctionset_output_label
                            # Set the overflow behavior and the correctionset description (pairing with name above) as well as the output description
                            corr = correctionlib.convert.from_histogram(final_hist)
                            corr.data.flow = ddcorrconfig.correctionset_flow
                            corr.description = ddcorrconfig.correctionset_top_description
                            corr.output.description = ddcorrconfig.correctionset_output_description
                            new_corrections.append(corr)
                            print("TODO: Finish plotting")
                        case "inc_WZ_DYDD_ProbDY":
                            data_hists = {}
                            non_dy_mc_hists = {}
                            dy_mc_hists = {}
                            template_hists = {}
                            mc_systematics = None
                            for channel in histogroups:
                                data_hists[channel] = histogroups[channel]["datasets"]["data"].to_boost()
                                dy_mc_hists[channel] = histogroups[channel]["datasets"]["DY"].to_boost()
                                non_dy_mc_hists[channel] = sum([v.to_boost() for k, v in histogroups[channel]["datasets"].items() if k not in ["data", "DY"]])
                                template_hists[channel] = update_axes_meta(
                                    data_hists[channel].project(*[ax.name for ax in data_hists[channel].axes if ax.name != "systematic"]).copy().reset(),
                                    ddcorrconfig.update_axes_meta
                                    )
                                if mc_systematics is None:
                                    mc_systematics = sorted([x for x in non_dy_mc_hists[channel].axes["systematic"]])
                            dict_for_hist_axis = {}
                            for dict_to_hist_axis_key, channel in ddcorrconfig.channel_absolutes.items():
                                dict_for_syst_axis = {}
                                for mc_syst in mc_systematics:
                                    # projections of fake rates by systematic
                                    num, den, ratio, (ratio_unc_up, ratio_unc_down) = inc_WZ_dddy_probdy_ratio_and_uncertainty(
                                        data_hists,
                                        dy_mc_hists,
                                        non_dy_mc_hists,
                                        data_syst="nominal",
                                        mc_syst=mc_syst,
                                        channel=channel,
                                        probdy_method = options.probdy_method,
                                    )
                                    if mc_syst == "nominal":
                                        # https://github.com/scikit-hep/boost-histogram/issues/421 - can also use h.view().value/variance, but MUST be on view, not h.value!!!
                                        dict_for_syst_axis["nominal"] = template_hists[channel].copy()
                                        dict_for_syst_axis["nominal"][...] = np.stack([ratio, ratio_unc_up], axis=-1)
                                        dict_for_syst_axis["DDDYUp"] = template_hists[channel].copy()
                                        dict_for_syst_axis["DDDYUp"][...] = np.stack([np.clip(ratio + ratio_unc_up, a_min=None, a_max=1.0), np.zeros_like(ratio)], axis=-1)
                                        dict_for_syst_axis["DDDYDown"] = template_hists[channel].copy()
                                        dict_for_syst_axis["DDDYDown"][...] = np.stack([ratio - ratio_unc_down, np.zeros_like(ratio)], axis=-1)
                                    else:
                                        dict_for_syst_axis[mc_syst] = template_hists[channel].copy()
                                        dict_for_syst_axis[mc_syst][...] = np.stack([ratio, np.zeros_like(ratio)], axis=-1)
                                dict_for_hist_axis[dict_to_hist_axis_key] = dict_to_hist_axis(
                                    # first promote the systematics axis for which we computed all slices
                                    histos=dict_for_syst_axis,
                                    axis_name="systematic",
                                    axis_label="Systematic Variation",
                                    axis_type="StrCategory",
                                    axis_storage=hist.storage.Weight(),
                                    axis_args={"growth": False},
                                )
                            dthac = ddcorrconfig.dict_to_hist_axis
                            final_hist = dict_to_hist_axis(
                                histos=dict_for_hist_axis,
                                axis_name=dthac.axis_name,
                                axis_label=dthac.axis_label,
                                axis_type=dthac.axis_type,
                                axis_storage=dthac.axis_storage,
                                axis_iter_override=dthac.axis_iter_override if "axis_iter_override" in dthac else None,
                                ).project(template_hists[channel].axes[0].name, dthac.axis_name, "systematic")
                            # Set the correcitonset (name) and outputnode (label)
                            final_hist.name = ddcorrconfig_key
                            final_hist.label = ddcorrconfig.correctionset_output_label
                            # Set the overflow behavior and the correctionset description (pairing with name above) as well as the output description
                            corr = correctionlib.convert.from_histogram(final_hist)
                            corr.data.flow = ddcorrconfig.correctionset_flow
                            corr.description = ddcorrconfig.correctionset_top_description
                            corr.output.description = ddcorrconfig.correctionset_output_description
                            new_corrections.append(corr)
                        case _:
                            raise NotImplementedError(f"Unknown method for data driven correction configuration {ddconfig_key} {ddcorrconfig_key} {ddcorrconfig.method} [from config {ddcorrconfig}]")                    
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
                            warnings.warn(f"Failed to find matching input for CompoundCorrection {ddcompcorrconfig_key}: {stk_key}"
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
            outfile = Path(os.path.dirname(__file__)) / f"data/dd/{options.era}/WZ_inclusive_data_driven_{options.era}.json"
            if not options.skip_json:
                with open(outfile, "w") as fout:
                    fout.write(new_cset.model_dump_json(exclude_unset=True))
                print(f"Wrote to {outfile}"
                      f"\n\tCorrections: {[x.name for x in new_corrections]}\n\tCompoundCorrections: {[x.name for x in new_compoundcorrections]}"
                      )
            else:
                print(f"Skipping write of new correctionset json: with {len(new_corrections)} Corrections and {len(new_compoundcorrections)} CompoundCorrections to {outfile}"
                      f"\nWould have respectivelywritten \n\tCorrections: {[x.name for x in new_corrections]}\n\tCompoundCorrections: {[x.name for x in new_compoundcorrections]}"
                      )
            if options.plot:
                raise NotImplementedError

    # Test
    print("\n=== Testing DataDrivenEventReweight ===")
    try:
        test = DataDrivenEventReweight(era=options.era, estimator="LNTTau_VTTau_DDDY_Estimate")
        test_njets = np.array([0, 0, 0, 1, 1])
        test_tau_pt = np.array([25, 80, 55, 33, 110])
        print(f"njets: {test_njets}, tau_pt: {test_tau_pt}")
        print(f"weights (nominal): {test.estimate_dd_DY(test_njets, test_tau_pt)}")
        print(f"weights (DDDYUp): {test.estimate_dd_DY(test_njets, test_tau_pt, systematic='DDDYUp')}")
        test2 = DataDrivenEventReweight(era=options.era, estimator="LNTTau_VTTau_DDDY_Closure")
        print(f"Closure weights (nominal): {test2.estimate_dd_DY(test_njets, test_tau_pt)}")
    except AssertionError as e:
        print(f"Skipping test: {e}")
