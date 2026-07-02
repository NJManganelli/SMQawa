"""Benchmark of histogram-fill strategies mimicking wztau2lnu_inclusive.process_shift.

Run with:  python benchmarks/bench_histogram_fill.py

Clones the fill-stage logic of wzinc_processor.process_shift (channel definitions,
PackedSelection cuts with substring-based N-1 removal, coffea Weights with 38
variations, 47 variables, 100k-event chunk) and times each optimization strategy
in isolation and combined, verifying bin-level equality (values and variances)
against the baseline.

Variants:
  V0  baseline           : verbatim clone of current _histogram_filler triple loop
  V1  weight-hoist       : precompute weights.weight(modifier) once per syst; loop order
                           ch -> var -> syst with cut+values computed once per (ch, var)
  V2  cut-cache          : memoize selection.require + formatted variables across the
                           original triple loop (no reordering); weights still per-call
  V12 both               : V1 + V2 (cuts shared across variables with identical N-1 sets)
  V3  region-broadcast   : one fill per (var, syst) with concatenated channels (StrCategory
                           array fill); builds on V12-style caching
  V4  multicell          : one fill per (ch, var) with (n_sel, 2*S) weight matrix into
                           MultiCell storage (slots = S values + S sumw2), then convert
                           back to standard (channel, systematic, var) Weight hists
  V5  multicell+broadcast: one fill per var, channels concatenated, MultiCell weights

All variants are checked for bin-level equality (value and variance) against V0.
"""
import time
import numpy as np
import awkward as ak
import hist
from coffea.analysis_tools import Weights, PackedSelection

rng = np.random.default_rng(1234)
N = 100_000

# ----------------------------------------------------------------------------
# Synthetic underlying event quantities (WZ-MC-like rates so channel yields are
# realistic: SRs ~0.5-2% of the chunk, CRs bigger)
# ----------------------------------------------------------------------------
ntight_lep = rng.choice([0, 1, 2, 3, 4], size=N, p=[0.30, 0.30, 0.25, 0.13, 0.02])
nloose_lep = rng.choice([0, 1], size=N, p=[0.9, 0.1])
ossf = rng.random(N) < 0.85
lead_pt = rng.exponential(40, N) + 10
met = rng.exponential(45, N)
dilep_m_val = rng.normal(91, 12, N)
dilep_pt_val = rng.exponential(60, N)
dphi_met = rng.uniform(-np.pi, np.pi, N)
njets = rng.poisson(0.8, N)
nhtaus_vtight = rng.choice([0, 1, 2], size=N, p=[0.55, 0.42, 0.03])
nhtaus_tight = np.clip(nhtaus_vtight + rng.choice([0, 1], size=N, p=[0.8, 0.2]), 0, 3)
nhtaus_loose = np.clip(nhtaus_tight + rng.choice([0, 1], size=N, p=[0.7, 0.3]), 0, 3)

selection = PackedSelection(dtype="uint64")
selection.add("lumimask", np.ones(N, dtype=bool))
selection.add("triggers", np.ones(N, dtype=bool))
selection.add("metfilter", rng.random(N) < 0.999)
selection.add("require-ossf", (ntight_lep == 2) & (nloose_lep == 0) & (lead_pt > 25) & ossf)
selection.add("require-osof", (ntight_lep == 2) & (nloose_lep == 0) & (lead_pt > 25) & ~ossf)
selection.add("require-2lep", (ntight_lep == 2) & (nloose_lep == 0) & (lead_pt > 25))
selection.add("met_pt", met > 30)
selection.add("low_met_pt", (met < 20) & (met > 0))
selection.add("val_met_pt", (met < 30) & (met > 20))
selection.add("dilep_m", np.abs(dilep_m_val - 91.1873) < 15)
selection.add("dilep_pt", dilep_pt_val > 30)
selection.add("dilep_dphi_met", np.abs(dphi_met) > 1.0)
selection.add("dilep_dphi_tau", rng.random(N) < 0.8)
selection.add("delta_tau_met_phi", rng.random(N) < 0.8)
selection.add("dilep_dphi_tau_loose", rng.random(N) < 0.8)
selection.add("delta_tau_loose_met_phi", rng.random(N) < 0.8)
selection.add("min_dphi_met_j", rng.random(N) < 0.9)
selection.add("0njets", njets == 0)
selection.add("1njets", njets <= 1)
selection.add("1njets_only", njets == 1)
selection.add("1nhtaus_vtight", nhtaus_vtight == 1)
selection.add("1nhtaus_tight", nhtaus_tight == 1)
selection.add("1nhtaus_loose", nhtaus_loose == 1)
selection.add("2plusnhtaus_vtight", nhtaus_vtight >= 2)
selection.add("2plusnhtaus_tight", nhtaus_tight >= 2)
selection.add("2plusnhtaus_loose", nhtaus_loose >= 2)
selection.add("lead_tau_plus", rng.random(N) < 0.5)
selection.add("lead_tau_minus", rng.random(N) < 0.5)

common_sel = ["triggers", "lumimask", "metfilter"]
channels = {
    "inc-SR0": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'met_pt'],
    "inc-SR1": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_vtight', 'met_pt'],
    "inc-VR0": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'val_met_pt'],
    "inc-VR1": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_vtight', 'val_met_pt'],
    "inc-VB0": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'val_met_pt'],
    "inc-VB1": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'val_met_pt'],
    "inc-B0": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'met_pt'],
    "inc-B1": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_loose', '~1nhtaus_tight', '~1nhtaus_vtight', 'met_pt'],
    "inc-C0": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', 'low_met_pt', '1nhtaus_loose', '0njets', '~1nhtaus_tight', '~1nhtaus_vtight'],
    "inc-C1": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', 'low_met_pt', '1nhtaus_loose', '1njets_only', '~1nhtaus_tight', '~1nhtaus_vtight'],
    "inc-D0": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_vtight', '~2plusnhtaus_tight', 'low_met_pt'],
    "inc-D1": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_vtight', '~2plusnhtaus_tight', 'low_met_pt'],
    "inc-IR0L": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'low_met_pt'],
    "inc-IR1L": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'low_met_pt'],
    "inc-IR0H": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '0njets', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'met_pt'],
    "inc-IR1H": common_sel + ['require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'dilep_pt', '1njets_only', '1nhtaus_tight', '~2plusnhtaus_loose', '~1nhtaus_vtight', 'met_pt'],
}

# ----------------------------------------------------------------------------
# Variables (47, names verbatim) + axis specs matching build_histos
# ----------------------------------------------------------------------------
PI = np.pi
axis_specs = {
    'dilep_mt_llnunu': (100, 0, 1000), 'dilep_pt': (60, 0, 600), 'dilep_deta': (20, 0, 8),
    'HTl': (100, 0, 1000), 'ST': (100, 0, 1000), 'dilep_m': (60, 0, 180),
    'met_pt': (60, 0, 600), 'mT_W': (100, 0, 1000),
    'dilep_tau_loose_met_hadron_mt': (150, 0, 1500), 'mT_WZ': (150, 0, 1500),
    'inv_m_WZ': (150, 0, 1500), 'tau_pt_vtight': (120, 0, 600), 'taus_eta': (50, -5, 5),
    'taus_phi': (50, -PI, PI), 'tau_pt_loose': (120, 0, 600), 'tau_pt_tight': (120, 0, 600),
    'taus_eta_loose': (50, -5, 5), 'taus_phi_loose': (50, -PI, PI), 'met_phi': (50, -PI, PI),
    'delta_tau_met_phi': (50, -PI, PI), 'deep_tau_jet': (100, 0, 1), 'deep_tau_e': (100, 0, 1),
    'deep_tau_mu': (100, 0, 1), 'dphi_met_ll': (50, -PI, PI), 'dphi_jet_met': (50, -PI, PI),
    'dilep_dphi_tau': (50, -PI, PI), 'dilep_dphi': (50, -PI, PI), 'delta_R': (50, 0, 5),
    'lead_jet_pt': (50, 30, 530), 'lead_jet_phi': (50, -PI, PI), 'lead_jet_eta': (50, -5, 5),
    'delta_R_jet_dilep': (50, 0, PI), 'delta_R_jet_tau': (50, 0, PI), 'dilep_dR': (50, 0, 5),
    'leading_lep_pt': (50, 30, 530), 'trailing_lep_pt': (50, 30, 530),
    'leading_lep_eta': (50, -5, 5), 'trailing_lep_eta': (50, -5, 5),
    'leading_lep_phi': (50, -PI, PI), 'trailing_lep_phi': (50, -PI, PI),
    'njets': (5, 0, 5), 'nhtaus_vtight': (5, 0, 5), 'nhtaus_tight': (5, 0, 5),
    'nhtaus_loose': (5, 0, 5), 'delta_R_non_iso_lep_loose_tau': (50, 0, PI),
    'delta_R_non_iso_lep_vtight_tau': (50, 0, PI), 'delta_R_non_iso_lep_tight_tau': (50, 0, PI),
}
variables = list(axis_specs)
assert len(variables) == 47

event = {}
for v, (nb, lo, hi) in axis_specs.items():
    vals = rng.uniform(lo, hi + 0.2 * (hi - lo), N)  # some overflow
    miss = rng.random(N) < 0.15
    vals[miss] = -99.0
    event[v] = ak.Array(vals)
# tie the N-1 variables to the actual cut quantities so N-1 histograms are meaningful
event['met_pt'] = ak.Array(met)
event['dilep_m'] = ak.Array(dilep_m_val)
event['dilep_pt'] = ak.Array(dilep_pt_val)
event['njets'] = ak.Array(njets.astype(np.float64))
event['nhtaus_vtight'] = ak.Array(nhtaus_vtight.astype(np.float64))
event['nhtaus_tight'] = ak.Array(nhtaus_tight.astype(np.float64))
event['nhtaus_loose'] = ak.Array(nhtaus_loose.astype(np.float64))

# ----------------------------------------------------------------------------
# Weights: genweight + 19 up/down pairs -> 38 variations (matches MC nominal pass)
# ----------------------------------------------------------------------------
weights = Weights(N, storeIndividual=True)
weights.add('genweight', rng.normal(1, 0.1, N))
pair_names = ['jetPUid_sf', 'puWeight', 'tauIDvsjet_sf', 'tauIDvse_sf', 'tauIDvsmu_sf',
              'triggerSF', 'eff_m_id', 'eff_m_iso', 'eff_e_reco', 'eff_e_id', 'kEW',
              'UEPS_ISR', 'UEPS_FSR', 'PDF_weight', 'aS_weight', 'QCDScale0w',
              'QCDScale1w', 'QCDScale2w', 'prefiring_weight']
for name in pair_names:
    nom = np.ones(N)
    weights.add(name, nom, rng.normal(1.05, 0.02, N), rng.normal(0.95, 0.02, N))

SYSTS_NOMINAL_PASS = [None] + list(weights.variations)   # 39
SYSTS_SHIFT_PASS = ["JESUp"]                             # object-shift re-run

console_lines = []
def fake_console_print(*args, **kwargs):
    console_lines.append(args)

def build_histos():
    return {
        v: hist.Hist(
            hist.axis.StrCategory([], name="channel", growth=True),
            hist.axis.StrCategory([], name="systematic", growth=True),
            hist.axis.Regular(*axis_specs[v], name=v),
            hist.storage.Weight(),
        ) for v in variables
    }

# ============================================================================
# V0: baseline, verbatim logic
# ============================================================================
def run_baseline(systematics):
    histos = build_histos()

    def _format_variable(variable, cut):
        if cut is None:
            vv = ak.to_numpy(ak.fill_none(variable, np.nan))
            if np.isnan(np.any(vv)):
                fake_console_print(" - vv with nan:", vv)
            return ak.to_numpy(ak.fill_none(variable, np.nan))
        else:
            vv = ak.to_numpy(ak.fill_none(variable[cut], np.nan))
            if np.isnan(np.any(vv)):
                fake_console_print(" - vv with nan:", vv)
            return ak.to_numpy(ak.fill_none(variable[cut], np.nan))

    def _histogram_filler(ch, syst, var, _weight=None):
        sel_ = channels[ch]
        sel_args_ = {
            s.replace('~', ''): (False if '~' in s else True) for s in sel_ if var not in s
        }
        cut = selection.require(**sel_args_)
        systname = 'nominal' if syst is None else syst
        if _weight is None:
            if syst in weights.variations:
                weight = weights.weight(modifier=syst)[cut]
            else:
                weight = weights.weight()[cut]
        else:
            weight = weights.weight()[cut] * _weight[cut]
        vv = ak.to_numpy(ak.fill_none(weight, np.nan))
        if np.isnan(np.any(vv)):
            fake_console_print(f" - {syst} weight contains invalid values:", vv)
        histos[var].fill(**{
            "channel": ch, "systematic": systname,
            var: _format_variable(event[var], cut),
            "weight": ak.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0),
        })

    for ch in channels:
        for sys_ in systematics:
            for var in variables:
                _histogram_filler(ch, sys_, var)
    return histos

# ----------------------------------------------------------------------------
# shared helpers for optimized variants
# ----------------------------------------------------------------------------
def sel_key(ch, var):
    sel_ = channels[ch]
    return tuple(sorted(
        (s.replace('~', ''), '~' not in s) for s in sel_ if var not in s
    ))

def hoisted_weights(systematics):
    out = {}
    for syst in systematics:
        if syst in weights.variations:
            out[syst] = weights.weight(modifier=syst)
        else:
            out[syst] = weights.weight()
    return out

def format_np(variable, cut):
    return ak.to_numpy(ak.fill_none(variable[cut], np.nan))

# ============================================================================
# V1: weight hoisting + loop reorder (cut/values once per (ch, var)); no
#     cross-variable cut sharing
# ============================================================================
def run_v1(systematics):
    histos = build_histos()
    w_by_syst = hoisted_weights(systematics)
    for ch in channels:
        for var in variables:
            sel_ = channels[ch]
            sel_args_ = {s.replace('~', ''): ('~' not in s) for s in sel_ if var not in s}
            cut = selection.require(**sel_args_)
            vv = format_np(event[var], cut)
            h = histos[var]
            for syst in systematics:
                systname = 'nominal' if syst is None else syst
                w = w_by_syst[syst][cut]
                h.fill(**{
                    "channel": ch, "systematic": systname, var: vv,
                    "weight": np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0),
                })
    return histos

# ============================================================================
# V2: cut + variable memoization only; original loop order, weights per-call
# ============================================================================
def run_v2(systematics):
    histos = build_histos()
    cut_cache = {}
    vv_cache = {}
    for ch in channels:
        for syst in systematics:
            systname = 'nominal' if syst is None else syst
            for var in variables:
                key = sel_key(ch, var)
                cut = cut_cache.get(key)
                if cut is None:
                    cut = selection.require(**dict(key))
                    cut_cache[key] = cut
                vkey = (var, key)
                vv = vv_cache.get(vkey)
                if vv is None:
                    vv = format_np(event[var], cut)
                    vv_cache[vkey] = vv
                if syst in weights.variations:
                    w = weights.weight(modifier=syst)[cut]
                else:
                    w = weights.weight()[cut]
                histos[var].fill(**{
                    "channel": ch, "systematic": systname, var: vv,
                    "weight": np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0),
                })
    return histos

# ============================================================================
# V12: weight hoisting + cut/variable/selected-weight caching
# ============================================================================
def run_v12(systematics):
    histos = build_histos()
    w_by_syst = hoisted_weights(systematics)
    cut_cache = {}
    for ch in channels:
        # group variables by their N-1 cut signature
        groups = {}
        for var in variables:
            groups.setdefault(sel_key(ch, var), []).append(var)
        for key, vars_in_group in groups.items():
            cut = cut_cache.get(key)
            if cut is None:
                cut = selection.require(**dict(key))
                cut_cache[key] = cut
            w_sel = {}
            for syst in systematics:
                w = w_by_syst[syst][cut]
                w_sel[syst] = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
            for var in vars_in_group:
                vv = format_np(event[var], cut)
                h = histos[var]
                for syst in systematics:
                    systname = 'nominal' if syst is None else syst
                    h.fill(**{"channel": ch, "systematic": systname, var: vv,
                              "weight": w_sel[syst]})
    return histos

# ============================================================================
# V3: region broadcast -- concatenate channels, one fill per (var, syst)
# ============================================================================
def run_v3(systematics):
    histos = build_histos()
    w_by_syst = hoisted_weights(systematics)
    cut_cache = {}
    # per variable: concatenated values + channel labels + per-syst weights
    for var in variables:
        vv_parts, ch_parts = [], []
        w_parts = {syst: [] for syst in systematics}
        for ch in channels:
            key = sel_key(ch, var)
            cut = cut_cache.get(key)
            if cut is None:
                cut = selection.require(**dict(key))
                cut_cache[key] = cut
            vv = format_np(event[var], cut)
            vv_parts.append(vv)
            ch_parts.append(np.full(len(vv), ch, dtype=object))
            for syst in systematics:
                w_parts[syst].append(w_by_syst[syst][cut])
        vv_all = np.concatenate(vv_parts)
        ch_all = np.concatenate(ch_parts)
        h = histos[var]
        for syst in systematics:
            systname = 'nominal' if syst is None else syst
            w = np.concatenate(w_parts[syst])
            h.fill(**{"channel": ch_all, "systematic": systname, var: vv_all,
                      "weight": np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)})
    return histos

# ============================================================================
# V4: MultiCell -- one fill per (ch, var); slots = [values x S, sumw2 x S];
#     converted back to standard Weight hists afterwards (cost included)
# ============================================================================
def build_multicell_histos(n_slots):
    # NOTE: channel categories MUST be pre-declared. boost-histogram 1.7.1
    # multi_cell storage loses existing contents when a growth axis resizes.
    return {
        v: hist.Hist(
            hist.axis.StrCategory(list(channels), name="channel", growth=True),
            hist.axis.Regular(*axis_specs[v], name=v),
            storage=hist.storage.MultiCell(n_slots),
        ) for v in variables
    }

def multicell_to_standard(histos_mc, systematics):
    """Expand MultiCell slots back into a (channel, systematic, var) Weight hist."""
    S = len(systematics)
    systnames = ['nominal' if s is None else s for s in systematics]
    out = {}
    for var, hmc in histos_mc.items():
        ch_names = list(hmc.axes[0])
        h = hist.Hist(
            hist.axis.StrCategory(ch_names, name="channel", growth=True),
            hist.axis.StrCategory(systnames, name="systematic", growth=True),
            hist.axis.Regular(*axis_specs[var], name=var),
            hist.storage.Weight(),
        )
        view_mc = hmc.view(flow=True)          # (2S, n_ch, n_bins+2)
        view = h.view(flow=True)               # (n_ch, S, n_bins+2) structured
        view["value"] = np.moveaxis(view_mc[:S], 0, 1)
        view["variance"] = np.moveaxis(view_mc[S:], 0, 1)
        out[var] = h
    return out

def run_v4(systematics):
    S = len(systematics)
    histos_mc = build_multicell_histos(2 * S)
    w_by_syst = hoisted_weights(systematics)
    W_full = np.stack([np.nan_to_num(w_by_syst[s], nan=1.0, posinf=1.0, neginf=1.0)
                       for s in systematics], axis=1)   # (N, S)
    cut_cache = {}
    for ch in channels:
        groups = {}
        for var in variables:
            groups.setdefault(sel_key(ch, var), []).append(var)
        for key, vars_in_group in groups.items():
            cut = cut_cache.get(key)
            if cut is None:
                cut = selection.require(**dict(key))
                cut_cache[key] = cut
            W = W_full[cut]                          # (n_sel, S)
            Wmat = np.concatenate([W, W * W], axis=1)  # (n_sel, 2S)
            for var in vars_in_group:
                vv = format_np(event[var], cut)
                histos_mc[var].fill(**{"channel": ch, var: vv}, weight=Wmat)
    return multicell_to_standard(histos_mc, systematics)

# ============================================================================
# V5: MultiCell + region broadcast -- one fill per var
# ============================================================================
def run_v5(systematics):
    S = len(systematics)
    histos_mc = build_multicell_histos(2 * S)
    w_by_syst = hoisted_weights(systematics)
    W_full = np.stack([np.nan_to_num(w_by_syst[s], nan=1.0, posinf=1.0, neginf=1.0)
                       for s in systematics], axis=1)
    cut_cache = {}
    for var in variables:
        vv_parts, ch_parts, w_parts = [], [], []
        for ch in channels:
            key = sel_key(ch, var)
            cut = cut_cache.get(key)
            if cut is None:
                cut = selection.require(**dict(key))
                cut_cache[key] = cut
            vv = format_np(event[var], cut)
            vv_parts.append(vv)
            ch_parts.append(np.full(len(vv), ch, dtype=object))
            w_parts.append(W_full[cut])
        vv_all = np.concatenate(vv_parts)
        ch_all = np.concatenate(ch_parts)
        W = np.concatenate(w_parts, axis=0)
        Wmat = np.concatenate([W, W * W], axis=1)
        histos_mc[var].fill(**{"channel": ch_all, var: vv_all}, weight=Wmat)
    return multicell_to_standard(histos_mc, systematics)

# ============================================================================
# comparison + timing
# ============================================================================
def compare(href, htest, systematics, label):
    systnames = ['nominal' if s is None else s for s in systematics]
    for var in variables:
        for ch in channels:
            for sn in systnames:
                a = href[var][ch, sn, :].view(flow=True)
                b = htest[var][ch, sn, :].view(flow=True)
                if not (np.allclose(a["value"], b["value"], rtol=1e-9, atol=1e-9)
                        and np.allclose(a["variance"], b["variance"], rtol=1e-9, atol=1e-9)):
                    print(f"  MISMATCH {label}: var={var} ch={ch} syst={sn}")
                    return False
    return True

def bench(fn, systematics, reps=2):
    best = np.inf
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(systematics)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, result

if __name__ == "__main__":
    # channel yields for context
    print("channel yields (of {} events):".format(N))
    for ch in channels:
        args = {s.replace('~', ''): ('~' not in s) for s in channels[ch]}
        print(f"  {ch:10s} {selection.require(**args).sum():6d}")

    variants = [("V0-baseline", run_baseline), ("V1-weight-hoist", run_v1),
                ("V2-cut-cache", run_v2), ("V12-both", run_v12),
                ("V3-broadcast", run_v3), ("V4-multicell", run_v4),
                ("V5-mc+bcast", run_v5)]

    for passname, systs in [("NOMINAL pass (39 systs)", SYSTS_NOMINAL_PASS),
                            ("SHIFT pass (1 syst)", SYSTS_SHIFT_PASS)]:
        print(f"\n=== {passname} ===")
        ref = None
        t_base = None
        for label, fn in variants:
            t, res = bench(fn, systs, reps=2)
            if ref is None:
                ref, t_base = res, t
                ok = "ref"
            else:
                ok = "OK" if compare(ref, res, systs, label) else "FAIL"
            print(f"  {label:18s} {t:8.2f} s   x{t_base/t:6.1f}   [{ok}]")
