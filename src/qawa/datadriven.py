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

def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Safe division returning 0 where denominator is <= 0."""
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def ratio_with_error(num: np.ndarray, num_var: np.ndarray,
                     den: np.ndarray, den_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ratio and propagated error: σ(a/b) = (a/b) * sqrt((σa/a)² + (σb/b)²)."""
    ratio = safe_divide(num, den)
    rel_err_num = safe_divide(np.sqrt(num_var), np.abs(num))
    rel_err_den = safe_divide(np.sqrt(den_var), np.abs(den))
    err = ratio * np.sqrt(rel_err_num**2 + rel_err_den**2)
    return np.nan_to_num(ratio, nan=0.0), np.nan_to_num(err, nan=0.0)


def subtract_with_variance(a: np.ndarray, a_var: np.ndarray,
                           b: np.ndarray, b_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract arrays and propagate variance: var(a-b) = var(a) + var(b)."""
    return a - b, a_var + b_var


REGION_CONFIG = {
    'B0': {'channel': 'inc-B0', 'observable': 'tau_pt_loose'},
    'B1': {'channel': 'inc-B1', 'observable': 'tau_pt_loose'},
    'C0': {'channel': 'inc-C0', 'observable': 'tau_pt_loose'},
    'C1': {'channel': 'inc-C1', 'observable': 'tau_pt_loose'},
    'D0': {'channel': 'inc-D0', 'observable': 'tau_pt'},
    'D1': {'channel': 'inc-D1', 'observable': 'tau_pt'},
}


def categorize_processes(config) -> Dict[str, List[str]]:
    """Extract data, DY, and non-DY MC process lists from config."""
    categories = {'data': [], 'dy': [], 'all_mc': [], 'non_dy_mc': []}

    if hasattr(config, 'groups'):
        for group_name in config.groups:
            group = config.groups[group_name]
            group_type = getattr(group, 'type', 'background').lower()
            processes = list(getattr(group, 'processes', []))

            if group_type == 'data':
                categories['data'].extend(processes)
            else:
                categories['all_mc'].extend(processes)
                if group_name == 'DY':
                    categories['dy'].extend(processes)
                else:
                    categories['non_dy_mc'].extend(processes)
    else:
        # Fallback
        categories['data'] = ['MuonEG', 'DoubleMuon', 'SingleMuon', 'EGamma']
        categories['dy'] = [p for p in config.boosthist.keys() if 'DYJets' in p]
        categories['all_mc'] = [p for p in config.boosthist.keys() if p not in categories['data']]
        categories['non_dy_mc'] = [p for p in categories['all_mc'] if p not in categories['dy']]

    return categories


class HistogramExtractor:
    """Extract and sum histogram yields from config with cross-section scaling."""

    def __init__(self, config, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.xsections = getattr(config, 'xsections', None)
        self.luminosity = getattr(getattr(config, 'luminosity', None), 'value', 1.0)

        # Get available axes from first histogram
        first_proc = list(config.boosthist.keys())[0]
        proc_hists = config.boosthist[first_proc]['hist']
        # sample_h = first_hist[list(first_hist.keys())[0]] if isinstance(first_hist, dict) else first_hist

        # self.available_channels = list(sample_h.axes['channel'])
        # self.available_systematics = list(sample_h.axes['systematic'])

        # Get tau_pt bin edges
        if isinstance(proc_hists, dict):
            test_obs_loose = 'tau_pt_loose'
            test_obs_vtight = 'tau_pt'
            test_hist_loose = proc_hists[test_obs_loose]
            test_hist_vtight = proc_hists[test_obs_vtight]
            self.tau_pt_edges = np.array(test_hist_loose.axes[test_obs_loose].edges)
            assert np.all(np.array(test_hist_vtight.axes[test_obs_vtight].edges) == self.tau_pt_edges)
            self.available_channels_loose = list(test_hist_loose.axes['channel'])
            self.available_channels_vtight = list(test_hist_vtight.axes['channel'])
            self.available_systematics = [x for x in list(test_hist_vtight.axes['systematic']) if x != 'nominal']
            loose_systematics = [x for x in list(test_hist_loose.axes['systematic']) if x != "nominal"]
            assert all([x in self.available_systematics for x in loose_systematics]) and all([x in loose_systematics for x in self.available_systematics])
        else:
            raise NotImplementedError
        # else:
        #     self.tau_pt_edges = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 1000.0])

        self.n_bins = len(self.tau_pt_edges) - 1

    def get_xs_scale(self, proc: str, sumw: float) -> float:
        """Compute cross-section scale factor for a process."""
        if self.xsections is None or proc not in self.xsections:
            return 1.0
        xs = self.xsections[proc]
        xsec = xs.xsec * xs.kr * xs.br * 1000.0  # convert to fb
        return xsec * self.luminosity / sumw if sumw > 0 else 0.0

    def get_yields(self, region: str, processes: List[str], systematic: str,
                   apply_xsec_scale: bool = False, is_data: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get summed yields and variances for processes in a region."""
        cfg = REGION_CONFIG[region]
        channel, observable = cfg['channel'], cfg['observable']
        effective_syst = 'nominal' if is_data else systematic

        if channel not in self.available_channels_loose and channel not in self.available_channels_vtight:
            if self.verbose:
                print(f"  Warning: Channel {channel} not available for observable {observable}, systematic {systematic}, processes = {processes}")
            return np.zeros(self.n_bins), np.zeros(self.n_bins)

        values = np.zeros(self.n_bins)
        variances = np.zeros(self.n_bins)

        print(f"  Computing yields for channel {channel} and observable {observable}, systematic {systematic}, processes = {processes}")
        for proc in processes:
            if proc not in self.config.boosthist:
                continue

            proc_data = self.config.boosthist[proc]
            hist_data = proc_data['hist']
            sumw = proc_data.get('sumw', 1.0)

            h = hist_data.get(observable) # if isinstance(hist_data, dict) else hist_data
            if h is None or channel not in h.axes['channel'] or effective_syst not in h.axes['systematic']:
                continue

            try:
                sliced = h[{'channel': channel, 'systematic': effective_syst}]
                v, var = sliced.values(), sliced.variances()
                # var = var if var is not None else v

                if apply_xsec_scale:
                    
                    scale = self.get_xs_scale(proc, sumw)
                    v, var = v * scale, var * scale**2

                values += v
                variances += var
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Error processing {proc}: {e}")

        return values, variances


def create_result_histogram(tau_pt_edges: np.ndarray, systematics: List[str],
                            name: str = "fake_rate") -> hist.Hist:
    """Create a hist.Hist for storing results with systematic variations."""
    return hist.Hist(
        hist.axis.StrCategory(systematics, name="systematic", growth=False),
        hist.axis.IntCategory([0, 1], name="jet_bin", label="Jet multiplicity bin"),
        hist.axis.Variable(tau_pt_edges, name="tau_pt", label=r"$\tau p_T$ [GeV]"),
        storage=hist.storage.Weight(),
        name=name,
    )


# =============================================================================
# Main derivation function
# =============================================================================

def derive_datadriven_corrections(config, era: str = "2018", verbose: bool = True,
                                   pdy_method: str = "pure_mc") -> dict:
    """
    Derive data-driven corrections for fake tau rates and P(DY) from histograms.

    Computes:
    1. Fake Rate: (Data - non-DY MC)_D / (Data - non-DY MC)_C
    2. MC Fake Rate (closure): DY_D / DY_C
    3. P(DY): DY fraction (pure_mc or data_subtraction method)

    Returns dict with hist.Hist objects for each quantity and systematic axis.
    """
    if pdy_method not in ["pure_mc", "data_subtraction"]:
        raise ValueError(f"pdy_method must be 'pure_mc' or 'data_subtraction'")

    # Initialize
    procs = categorize_processes(config)
    extractor = HistogramExtractor(config, verbose)

    if verbose:
        print(f"P(DY) method: {pdy_method}")
        print(f"Data: {len(procs['data'])}, DY: {len(procs['dy'])}, non-DY MC: {len(procs['non_dy_mc'])}")
        print(f"Channels: \n\t{extractor.available_channels_loose} [loose] \n\t {extractor.available_channels_vtight} [vtight]")
        print(f"Systematics: {extractor.available_systematics}")
        print(f"Tau pT edges: {extractor.tau_pt_edges}")

    # Create result histograms
    systs = extractor.available_systematics
    h_fake_rate = create_result_histogram(extractor.tau_pt_edges, systs, "fake_rate")
    h_mc_fake_rate = create_result_histogram(extractor.tau_pt_edges, systs, "mc_fake_rate")
    h_pDY = create_result_histogram(extractor.tau_pt_edges, systs, "pDY")

    # Also keep raw arrays for compatibility
    results = {
        'fake_rate_0': {}, 'fake_rate_1': {}, 'fake_rate_0_err': {}, 'fake_rate_1_err': {},
        'mc_fake_rate_0': {}, 'mc_fake_rate_1': {}, 'mc_fake_rate_0_err': {}, 'mc_fake_rate_1_err': {},
        'pDY_0': {}, 'pDY_1': {}, 'pDY_0_err': {}, 'pDY_1_err': {},
        'tau_pt_edges': extractor.tau_pt_edges.tolist(),
        'available_systematics': systs,
        'histograms': {'fake_rate': h_fake_rate, 'mc_fake_rate': h_mc_fake_rate, 'pDY': h_pDY},
    }

    # Process each systematic
    for syst in systs:
        if verbose:
            print(f"\nProcessing: {syst}" + (" (data uses nominal)" if syst != 'nominal' else ""))

        # Get yields for all regions
        yields = {}
        for region in ['B0', 'B1', 'C0', 'C1', 'D0', 'D1']:
            yields[f'dy_{region}'] = extractor.get_yields(region, procs['dy'], syst, apply_xsec_scale=True)
            yields[f'data_{region}'] = extractor.get_yields(region, procs['data'], syst, is_data=True)
            yields[f'non_dy_{region}'] = extractor.get_yields(region, procs['non_dy_mc'], syst, apply_xsec_scale=True)
            yields[f'all_mc_{region}'] = extractor.get_yields(region, procs['all_mc'], syst, apply_xsec_scale=True)

        # Compute fake rates for both jet bins
        for jet_bin, suffix in [(0, '0'), (1, '1')]:
            # Data-driven fake rate: (Data - non-DY)_D / (Data - non-DY)_C
            num, num_var = subtract_with_variance(
                yields[f'data_D{suffix}'][0], yields[f'data_D{suffix}'][1],
                yields[f'non_dy_D{suffix}'][0], yields[f'non_dy_D{suffix}'][1]
            )
            den, den_var = subtract_with_variance(
                yields[f'data_C{suffix}'][0], yields[f'data_C{suffix}'][1],
                yields[f'non_dy_C{suffix}'][0], yields[f'non_dy_C{suffix}'][1]
            )
            fr, fr_err = ratio_with_error(num, num_var, den, den_var)

            # MC-only fake rate: DY_D / DY_C
            mc_fr, mc_fr_err = ratio_with_error(
                yields[f'dy_D{suffix}'][0], yields[f'dy_D{suffix}'][1],
                yields[f'dy_C{suffix}'][0], yields[f'dy_C{suffix}'][1]
            )

            # P(DY) computation
            if pdy_method == "pure_mc":
                pdy, pdy_err = ratio_with_error(
                    yields[f'dy_B{suffix}'][0], yields[f'dy_B{suffix}'][1],
                    yields[f'all_mc_B{suffix}'][0], yields[f'all_mc_B{suffix}'][1]
                )
            else:  # data_subtraction
                pdy_num, pdy_num_var = subtract_with_variance(
                    yields[f'data_B{suffix}'][0], yields[f'data_B{suffix}'][1],
                    yields[f'non_dy_B{suffix}'][0], yields[f'non_dy_B{suffix}'][1]
                )
                pdy, pdy_err = ratio_with_error(
                    pdy_num, pdy_num_var,
                    yields[f'data_B{suffix}'][0], yields[f'data_B{suffix}'][1]
                )

            # Fill histograms (weighted fill with variance)
            bin_centers = (extractor.tau_pt_edges[:-1] + extractor.tau_pt_edges[1:]) / 2
            h_fake_rate.fill(systematic=syst, jet_bin=jet_bin, tau_pt=bin_centers, weight=fr)
            h_mc_fake_rate.fill(systematic=syst, jet_bin=jet_bin, tau_pt=bin_centers, weight=mc_fr)
            h_pDY.fill(systematic=syst, jet_bin=jet_bin, tau_pt=bin_centers, weight=pdy)

            # Store in dict (for backward compatibility)
            results[f'fake_rate_{suffix}'][syst] = fr.tolist()
            results[f'fake_rate_{suffix}_err'][syst] = fr_err.tolist()
            results[f'mc_fake_rate_{suffix}'][syst] = mc_fr.tolist()
            results[f'mc_fake_rate_{suffix}_err'][syst] = mc_fr_err.tolist()
            results[f'pDY_{suffix}'][syst] = pdy.tolist()
            results[f'pDY_{suffix}_err'][syst] = pdy_err.tolist()

        if verbose:
            print(f"  FR 0-jet: {np.round(results['fake_rate_0'][syst], 4)}")
            print(f"  FR 1-jet: {np.round(results['fake_rate_1'][syst], 4)}")
            print(f"  P(DY) 0-jet: {np.round(results['pDY_0'][syst], 4)}")

    return results


def plot_corrections(results: dict, output_dir: str = ".", prefix: str = "dd_corrections",
                     n_top_systematics: int = 5) -> List[str]:
    """
    Plot fake rates and P(DY) with systematic variations using mplhep.

    Parameters
    ----------
    results : dict
        Output from derive_datadriven_corrections()
    output_dir : str
        Directory for output PDFs
    prefix : str
        Filename prefix
    n_top_systematics : int
        Number of most significant systematics to show (beyond nominal)

    Returns
    -------
    List[str]
        Paths to generated PDF files
    """
    import matplotlib.pyplot as plt
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    tau_pt_edges = np.array(results['tau_pt_edges'])
    bin_centers = (tau_pt_edges[:-1] + tau_pt_edges[1:]) / 2
    bin_widths = tau_pt_edges[1:] - tau_pt_edges[:-1]
    systs = results['available_systematics']

    def find_top_variations(nominal: np.ndarray, all_systs: dict, n: int) -> List[str]:
        """Find systematics with largest deviations from nominal."""
        deviations = {}
        for s, vals in all_systs.items():
            if s != 'nominal':
                deviations[s] = np.max(np.abs(np.array(vals) - nominal))
        return sorted(deviations.keys(), key=lambda x: deviations[x], reverse=True)[:n]

    # Plot configurations
    plot_configs = [
        ('fake_rate', '0', 'Data-Driven Fake Rate (0-jet)', r'Fake Rate $F_{\tau}$'),
        ('fake_rate', '1', 'Data-Driven Fake Rate (1-jet)', r'Fake Rate $F_{\tau}$'),
        ('mc_fake_rate', '0', 'MC Closure Fake Rate (0-jet)', r'Fake Rate $F_{\tau}^{MC}$'),
        ('mc_fake_rate', '1', 'MC Closure Fake Rate (1-jet)', r'Fake Rate $F_{\tau}^{MC}$'),
        ('pDY', '0', 'P(DY) (0-jet)', r'$P_{DY}$'),
        ('pDY', '1', 'P(DY) (1-jet)', r'$P_{DY}$'),
    ]

    for var_name, jet_bin, title, ylabel in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 8))

        key = f'{var_name}_{jet_bin}'
        err_key = f'{key}_err'

        nominal = np.array(results[key].get('nominal', []))
        nominal_err = np.array(results[err_key].get('nominal', []))

        if len(nominal) == 0:
            continue

        # Plot nominal with error band
        ax.errorbar(bin_centers, nominal, xerr=bin_widths/2, yerr=nominal_err,
                    fmt='o', color='black', label='Nominal', markersize=8, capsize=3, zorder=10)

        # Find and plot top systematic variations
        top_systs = find_top_variations(nominal, results[key], n_top_systematics)

        colors = plt.cm.tab10(np.linspace(0, 1, len(top_systs)))
        for syst, color in zip(top_systs, colors):
            vals = np.array(results[key].get(syst, []))
            if len(vals) > 0:
                # Use step plot for systematic variations
                ax.step(tau_pt_edges[:-1], vals, where='post', color=color,
                        alpha=0.7, linewidth=2, label=syst)

        ax.set_xlabel(r'$\tau$ $p_T$ [GeV]', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(tau_pt_edges[0], min(tau_pt_edges[-1], 150))

        if 'pDY' in var_name:
            ax.set_ylim(0.8, 1.05)
        else:
            ax.set_ylim(0, None)

        ax.grid(True, alpha=0.3)
        hep.cms.label(ax=ax, data=True, lumi=59.83, year=2018)

        # Save
        outpath = os.path.join(output_dir, f"{prefix}_{var_name}_jet{jet_bin}.pdf")
        fig.savefig(outpath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        output_files.append(outpath)
        print(f"Saved: {outpath}")

    # Combined comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (jet_bin, ax_row) in enumerate(zip(['0', '1'], axes)):
        for col, (var, ylabel, title_part) in enumerate([
            ('fake_rate', r'Fake Rate', 'Data-Driven FR'),
            ('pDY', r'$P_{DY}$', 'P(DY)')
        ]):
            ax = ax_row[col]
            key = f'{var}_{jet_bin}'
            nominal = np.array(results[key].get('nominal', []))

            if len(nominal) > 0:
                ax.errorbar(bin_centers, nominal, xerr=bin_widths/2,
                            yerr=np.array(results[f'{key}_err'].get('nominal', [])),
                            fmt='o', color='black', label='Nominal', capsize=3)

                # Envelope of all systematics
                all_vals = [np.array(results[key].get(s, nominal)) for s in systs if s != 'nominal']
                if all_vals:
                    all_vals = np.array([v for v in all_vals if len(v) == len(nominal)])
                    if len(all_vals) > 0:
                        lo = np.min(all_vals, axis=0)
                        hi = np.max(all_vals, axis=0)
                        ax.fill_between(bin_centers, lo, hi, alpha=0.3, color='blue',
                                        label='Systematic envelope')

            ax.set_xlabel(r'$\tau$ $p_T$ [GeV]')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title_part} ({jet_bin}-jet)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(tau_pt_edges[0], min(tau_pt_edges[-1], 150))

    plt.tight_layout()
    outpath = os.path.join(output_dir, f"{prefix}_summary.pdf")
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    output_files.append(outpath)
    print(f"Saved: {outpath}")

    return output_files


def format_for_correctionlib(results: dict, prepend_zero: bool = True) -> dict:
    """Format results for correctionlib MultiBinning with [0-jet..., 1-jet...] layout."""
    formatted = {k: {} for k in ['fake_rate', 'fake_rate_err', 'mc_fake_rate',
                                  'mc_fake_rate_err', 'pDY', 'pDY_err']}

    for syst in results['available_systematics']:
        for var in ['fake_rate', 'mc_fake_rate', 'pDY']:
            v0 = list(results.get(f'{var}_0', {}).get(syst, []))
            v1 = list(results.get(f'{var}_1', {}).get(syst, []))
            e0 = list(results.get(f'{var}_0_err', {}).get(syst, []))
            e1 = list(results.get(f'{var}_1_err', {}).get(syst, []))

            if prepend_zero:
                v0, v1 = [0.0] + v0, [0.0] + v1
                e0, e1 = [0.0] + e0, [0.0] + e1

            formatted[var][syst] = v0 + v1
            formatted[f'{var}_err'][syst] = e0 + e1

    return formatted

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

    # Generate plots
    if options.plot and derived_results is not None:
        print("\n=== Generating plots ===")
        plot_corrections(derived_results, output_dir=options.plot_dir,
                         n_top_systematics=options.n_systematics)

    # Generate correctionlib JSON
    # if not options.skip_json:
    #     from correctionlib import schemav2 as cs

    #     new_cset = correctionlib.schemav2.CorrectionSet(
    #         schema_version=2,
    #         corrections=[
    #             cs.Correction(
    #                 name="LNTTau_to_TTau_TransferFactor",
    #                 version=1,
    #                 inputs=[
    #                     cs.Variable(name="jet_multiplicity", type="real",
    #                                 description="Number of jets"),
    #                     cs.Variable(name="tau_pt", type="real",
    #                                 description="Tau pT [GeV]"),
    #                 ],
    #                 output=cs.Variable(name="weight", type="real",
    #                                    description="Fake tau transfer factor"),
    #                 data=cs.MultiBinning(
    #                     nodetype="multibinning",
    #                     inputs=["jet_multiplicity", "tau_pt"],
    #                     edges=[[0, 1, 2], [0, 20, 25, 30, 35, 40, 60, 80, 100, 110]],
    #                     content=[0.0, 0.05079561, 0.09419116, 0.11183215, 0.1351457,
    #                              0.17628764, 0.19667385, 0.19257761, 0.18835731,
    #                              0.0, 0.01081895, 0.00904362, 0.00930078, 0.00792693,
    #                              0.01008214, 0.01083654, 0.01113167, 0.02650382],
    #                     flow="clamp",
    #                 ),
    #             ),
    #             cs.Correction(
    #                 name="LNTTau_HighMET_DY_to_Data_estimate",
    #                 version=1,
    #                 inputs=[
    #                     cs.Variable(name="jet_multiplicity", type="real",
    #                                 description="Number of jets"),
    #                     cs.Variable(name="tau_pt", type="real",
    #                                 description="Tau pT [GeV]"),
    #                 ],
    #                 output=cs.Variable(name="weight", type="real",
    #                                    description="P(DY) weight"),
    #                 data=cs.MultiBinning(
    #                     nodetype="multibinning",
    #                     inputs=["jet_multiplicity", "tau_pt"],
    #                     edges=[[0, 1, 2], [0, 20, 25, 30, 35, 40, 60, 80, 100, 110]],
    #                     content=[0.0, 0.97531364, 0.96670784, 0.9585013, 0.94991049,
    #                              0.92440385, 0.88846139, 0.87653405, 0.88508274,
    #                              0.0, 0.97916137, 0.98144067, 0.98084848, 0.98033434,
    #                              0.97477991, 0.9655559, 0.95486799, 0.93447709],
    #                     flow="clamp",
    #                 ),
    #             ),
    #         ],
    #         compound_corrections=[
    #             cs.CompoundCorrection(
    #                 name="LNTTau_TTau_DD_Estimate",
    #                 description="Data-driven DY background estimate",
    #                 inputs=[
    #                     cs.Variable(name="jet_multiplicity", type="real",
    #                                 description="Number of jets"),
    #                     cs.Variable(name="tau_pt", type="real",
    #                                 description="Tau pT [GeV]"),
    #                 ],
    #                 output=cs.Variable(name="weight", type="real",
    #                                    description="DD estimate weight"),
    #                 inputs_update=[],
    #                 input_op="*",
    #                 output_op="*",
    #                 stack=["LNTTau_to_TTau_TransferFactor", "LNTTau_HighMET_DY_to_Data_estimate"],
    #             ),
    #         ],
    #     )

    #     outfile = f"WZ_inclusive_data_driven_{options.era}.json"
    #     with open(outfile, "w") as fout:
    #         fout.write(new_cset.model_dump_json(exclude_unset=True))
    #     print(f"Wrote {outfile}")

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
