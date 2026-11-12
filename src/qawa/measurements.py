"""Trigger scale factor and b-tag efficiency measurements: correctionlib payloads and their derivation.

The inclusive WZ processor fills trigger_measurement and btag_measurement; this module turns merged
(hist-merger --splitByKind) outputs into per-era correctionlib payloads.
"""
import argparse
import gzip
import os
import pickle

import numpy as np
import yaml
from coffea.util import coffea_console
from correctionlib import schemav2 as cs
from hist.intervals import clopper_pearson_interval

from qawa.btag import FIXED_WPS
from qawa.common import is_reference_stream, is_trigger_mc_dataset

TRIGGER_SF_FLAVOURS = ("mm", "me", "em", "ee")
TRIGGER_SF_SYSTEMATICS = ("nominal", "up", "down", "alpha_up", "alpha_down")
BTAG_EFF_SYSTEMATICS = ("nominal", "up", "down")
BTAG_EFF_FLAVOURS = (0, 4, 5)


def multibinning(inputs, edges, content):
    return cs.MultiBinning(
        nodetype="multibinning",
        inputs=list(inputs),
        edges=[[float(e) for e in axis] for axis in edges],
        content=[float(v) for v in np.asarray(content, dtype=float).ravel(order="C")],
        flow="clamp",
    )


def trigger_sf_payload(grids, edges, description):
    """grids[flavour][systematic] is an (n_lead, n_trail) array over edges = (lead_edges, trail_edges)."""
    corrections = []
    for flavour in TRIGGER_SF_FLAVOURS:
        corrections.append(cs.Correction(
            name=f"SF_{flavour}",
            description=f"Di-lepton trigger scale factor (leading={flavour[0]}, trailing={flavour[1]})",
            version=1,
            inputs=[
                cs.Variable(name="lead_pt", type="real", description="Leading lepton pT [GeV]"),
                cs.Variable(name="trail_pt", type="real", description="Trailing lepton pT [GeV]"),
                cs.Variable(name="systematic", type="string", description=" / ".join(TRIGGER_SF_SYSTEMATICS)),
            ],
            output=cs.Variable(name="weight", type="real", description="Scale factor weight"),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[cs.CategoryItem(key=systematic,
                                         value=multibinning(("lead_pt", "trail_pt"), edges, grids[flavour][systematic]))
                         for systematic in TRIGGER_SF_SYSTEMATICS],
            ),
        ))
    return cs.CorrectionSet(schema_version=2, description=description, corrections=corrections)


def btag_eff_payload(maps, edges, wps):
    """maps[process][systematic][wp][flavour] is an (n_pt, n_abseta) array over edges = (pt_edges, abseta_edges)."""
    def flavour_node(by_flavour):
        return cs.Category(
            nodetype="category",
            input="hadronFlavour",
            content=[cs.CategoryItem(key=flavour, value=multibinning(("pt", "abseta"), edges, by_flavour[flavour]))
                     for flavour in BTAG_EFF_FLAVOURS],
        )

    def category(name, keys, value):
        return cs.Category(nodetype="category", input=name,
                           content=[cs.CategoryItem(key=key, value=value(key)) for key in keys])

    data = category("process", sorted(maps), lambda process: category(
        "systematic", BTAG_EFF_SYSTEMATICS, lambda systematic: category(
            "working_point", wps, lambda wp: flavour_node(maps[process][systematic][wp]))))
    correction = cs.Correction(
        name="btag_eff",
        description="b-tagging efficiency per process, working point and hadron flavour",
        version=1,
        inputs=[
            cs.Variable(name="process", type="string", description="dataset name, or inclusive"),
            cs.Variable(name="systematic", type="string", description=" / ".join(BTAG_EFF_SYSTEMATICS)),
            cs.Variable(name="working_point", type="string", description="fixed working point"),
            cs.Variable(name="hadronFlavour", type="int", description="0 (light), 4 (c), 5 (b)"),
            cs.Variable(name="pt", type="real", description="jet pT [GeV]"),
            cs.Variable(name="abseta", type="real", description="jet |eta|"),
        ],
        output=cs.Variable(name="efficiency", type="real", description="tagging efficiency"),
        data=data,
    )
    return cs.CorrectionSet(schema_version=2, description="b-tagging efficiencies", corrections=[correction])


def write_payload(cset, path):
    with open(path, "w") as fout:
        fout.write(cset.model_dump_json(exclude_unset=True))
    coffea_console.print(f"Wrote {path}. A payload at the jsoncorrections location under src/qawa/data "
                         "is picked up by every later run of that era.", markup=False)


def efficiency_interval(k, n):
    """Clopper-Pearson (one standard deviation) efficiency k/n and its bounds; cells with n <= 0 give 0."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    filled = n > 0
    safe_n = np.where(filled, n, 1.)
    safe_k = np.where(filled, k, 0.)
    lo, hi = clopper_pearson_interval(safe_k, safe_n)
    eff = np.where(filled, safe_k / safe_n, 0.)
    return eff, np.where(filled, lo, 0.), np.where(filled, hi, 0.)


def effective_counts(w_pass, w_total, v_total):
    """Weighted pass/total sums to (eff, k_eff, n_eff, filled, clipped) with n_eff = W^2/V.

    Cells with W <= 0 or V <= 0 are not filled. eff is clipped to [0, 1] (negative weights); `clipped`
    flags the filled cells where that changed it.
    """
    w_pass, w_total, v_total = (np.asarray(a, dtype=float) for a in (w_pass, w_total, v_total))
    filled = (w_total > 0) & (v_total > 0)
    raw = np.where(filled, w_pass / np.where(filled, w_total, 1.), 0.)
    eff = np.clip(raw, 0., 1.)
    n_eff = np.where(filled, w_total ** 2 / np.where(filled, v_total, 1.), 0.)
    k_eff = np.clip(eff * n_eff, 0., n_eff)
    return eff, k_eff, n_eff, filled, filled & (raw != eff)


def load_records(paths):
    records = {}
    for path in paths:
        with gzip.open(path, "rb") as fin:
            content = pickle.load(fin)
        for dataset, record in content.items():
            if "kind" not in record:
                raise ValueError(f"{path}: {dataset} has no kind (write the inputs with hist-merger --splitByKind)")
            if dataset in records:
                raise ValueError(f"{dataset} is in more than one input file")
            records[dataset] = record
    return records


def nonzero(h):
    return bool(np.any(h.values(flow=True) != 0))


def trigger_rules_for(era):
    year = era.replace("APV", "")
    with open(os.path.join(os.path.dirname(__file__), "data", f"{year}-trigger-rules.yaml")) as fin:
        return yaml.load(fin, Loader=yaml.FullLoader), year


def trigger_inputs(records, era):
    """trigger_measurement of the reference-stream data and of the trigger MC; raises for any other filler."""
    rules, year = trigger_rules_for(era)
    data, mc, unexpected = [], [], []
    for dataset, record in records.items():
        h = record["hist"].get("trigger_measurement")
        if h is None or not nonzero(h):
            continue
        if record["kind"] == "data" and is_reference_stream(dataset, rules, year):
            data.append(h)
        elif record["kind"] in ("mc", "signal") and is_trigger_mc_dataset(dataset):
            mc.append(h)
        else:
            unexpected.append(dataset)
    if unexpected:
        raise ValueError(f"trigger_measurement is filled for datasets that must not fill it: {sorted(unexpected)}")
    if not data or not mc:
        raise ValueError("trigger derivation needs reference-stream data and TTto2L2Nu MC with trigger_measurement")
    return data, mc


def fold_pt_flow(array):
    """Fold the under/overflow of the two leading (pt) axes of a flow=True array into the first/last bins."""
    for axis in (0, 1):
        array = np.moveaxis(array, axis, 0)
        array = np.concatenate([array[1:2] + array[0:1], array[2:-2], array[-2:-1] + array[-1:]])
        array = np.moveaxis(array, 0, axis)
    return array


def trigger_cells(hists, flavour):
    """Summed (value, variance) of the trigger-<flavour> channel, shape (lead, trail, reference, lepton)."""
    shape = tuple(len(hists[0].axes[name]) for name in ("trigger_lead_pt", "trigger_trail_pt")) + (2, 2)
    value, variance = np.zeros(shape), np.zeros(shape)
    for h in hists:
        if f"trigger-{flavour}" in h.axes["channel"]:
            view = h[{"channel": f"trigger-{flavour}", "systematic": "nominal"}].view(flow=True)
            value += fold_pt_flow(view["value"])
            variance += fold_pt_flow(view["variance"])
    return value, variance


def derive_trigger_sf(data_hists, mc_hists):
    """SF grids per flavour and systematic (TRIGGER_SF_SYSTEMATICS) from trigger_measurement histograms."""
    lead_edges = data_hists[0].axes["trigger_lead_pt"].edges
    trail_edges = data_hists[0].axes["trigger_trail_pt"].edges
    kinematically_empty = lead_edges[1:][:, None] <= trail_edges[:-1][None, :]
    grids = {}
    for flavour in TRIGGER_SF_FLAVOURS:
        n, _ = trigger_cells(data_hists, flavour)
        w, v = trigger_cells(mc_hists, flavour)
        eps_d, lo_d, hi_d = efficiency_interval(n[..., 1, 1], n[..., 1, :].sum(-1))
        eps_m, k_eff, n_eff, filled_m, clipped = effective_counts(w[..., 1, 1], w[..., 1, :].sum(-1), v[..., 1, :].sum(-1))
        _, lo_m, hi_m = efficiency_interval(k_eff, n_eff)
        total = w.sum((-1, -2))
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = w[..., :, 1].sum(-1) * w[..., 1, :].sum(-1) / (w[..., 1, 1] * total)
            sf = eps_d / eps_m
            d_up = sf * np.hypot((hi_d - eps_d) / eps_d, (eps_m - lo_m) / eps_m)
            d_dn = sf * np.hypot((eps_d - lo_d) / eps_d, (hi_m - eps_m) / eps_m)
            a = sf * np.abs(1 - alpha)
            values = {"nominal": sf, "up": sf + d_up, "down": sf - d_dn,
                      "alpha_up": sf + np.hypot(d_up, a), "alpha_down": sf - np.hypot(d_dn, a)}
        empty = ~kinematically_empty & ((n[..., 1, :].sum(-1) <= 0) | ~filled_m | (eps_d == 0) | (eps_m == 0))
        valid = ~kinematically_empty & ~empty
        empty_values = {"nominal": 1., "up": 2., "down": 0., "alpha_up": 2., "alpha_down": 0.}
        grids[flavour] = {key: np.where(valid, values[key], np.where(empty, empty_values[key], 1.))
                          for key in TRIGGER_SF_SYSTEMATICS}
        coffea_console.print(f"SF_{flavour}: {int(clipped.sum())} MC cells with a clipped efficiency; "
                             f"empty (lead, trail) bins: {[tuple(map(int, ij)) for ij in np.argwhere(empty)]}", markup=False)
        for i, j in np.argwhere(valid):
            coffea_console.print(f"SF_{flavour} lead [{lead_edges[i]:g},{lead_edges[i + 1]:g}) trail "
                                 f"[{trail_edges[j]:g},{trail_edges[j + 1]:g}): eps_data={eps_d[i, j]:.4f} "
                                 f"eps_mc={eps_m[i, j]:.4f} SF={sf[i, j]:.4f} alpha={alpha[i, j]:.4f}", markup=False)
    return grids, (lead_edges, trail_edges)


def xsection_scale(dataset, xsections, sumw):
    key = dataset if dataset in xsections else dataset.split("__")[0]
    if key not in xsections:
        raise ValueError(f"no cross section for {dataset} in the --xsections file")
    entry = xsections[key]
    return entry["xsec"] * entry.get("br", 1.0) * entry.get("kr", 1.0) / sumw


def btag_cells(h):
    view = h[{"channel": "btag-jets", "systematic": "nominal"}].view(flow=True)
    pt_axis = h.axes.name.index("btag_pt") - 2

    def fold(array):
        array = np.moveaxis(array, pt_axis, 0)
        array = np.concatenate([array[1:-2], array[-2:-1] + array[-1:]])
        return np.moveaxis(array, 0, pt_axis)[:, :, 1:-1, :]
    return fold(view["value"]), fold(view["variance"])


def derive_btag_eff(records, version, xsections):
    """Efficiency maps per process (and inclusive) from btag_measurement, for every fixed WP of the era's tagger."""
    wps = FIXED_WPS["deepJet" if version in [f"v{v}" for v in range(12)] else "UParTAK4"]
    histograms, inclusive = {}, None
    for dataset, record in records.items():
        if record["kind"] not in ("mc", "signal"):
            continue
        h = record["hist"].get("btag_measurement")
        if h is None or not nonzero(h):
            continue
        histograms[dataset] = h
        scaled = h * xsection_scale(dataset, xsections, record["sumw"])
        inclusive = scaled if inclusive is None else inclusive + scaled
    if inclusive is None:
        raise ValueError("b-tag derivation needs MC with btag_measurement")
    histograms["inclusive"] = inclusive
    edges = (inclusive.axes["btag_pt"].edges, inclusive.axes["btag_abseta"].edges)
    flavours = {flavour: index for index, flavour in enumerate((0, 4, 5))}
    maps = {}
    for process, h in histograms.items():
        w, v = btag_cells(h)
        maps[process] = {syst: {wp: {} for wp in wps} for syst in BTAG_EFF_SYSTEMATICS}
        n_empty = n_clipped = 0
        for k, wp in enumerate(wps):
            eff, k_eff, n_eff, filled, clipped = effective_counts(w[..., k + 1:].sum(-1), w.sum(-1), v.sum(-1))
            _, lo, hi = efficiency_interval(k_eff, n_eff)
            n_clipped += int(clipped.sum())
            n_empty += int((~filled)[:, 1:, :].sum())
            for flavour, index in flavours.items():
                maps[process]["nominal"][wp][flavour] = np.where(filled[index], eff[index], 0.)
                maps[process]["down"][wp][flavour] = np.where(filled[index], lo[index], 0.)
                maps[process]["up"][wp][flavour] = np.where(filled[index], hi[index], 1.)
        coffea_console.print(f"btag_eff {process}: {n_empty} empty and {n_clipped} clipped (flavour, pt, |eta|, WP) "
                             "cells above 30 GeV", markup=False)
    return maps, edges, wps


def main(argv=None):
    parser = argparse.ArgumentParser(description="Derive trigger SF and b-tag efficiency correctionlib payloads from "
                                                 "merged inclusive WZ outputs.")
    parser.add_argument("--era", required=True, help="era as given to the processor, e.g. 2016APV, 2018, 2024")
    parser.add_argument("--version", required=True, choices=["v9", "v15"], help="NanoAOD version")
    parser.add_argument("-i", "--input", nargs="+", required=True, help="hist-merger --splitByKind .pkl.gz files")
    parser.add_argument("--derive", nargs="+", required=True, choices=["trigger", "btag"])
    parser.add_argument("--xsections", help="DCTools xsections yaml (required for btag)")
    parser.add_argument("--trigger-output", help="trigger SF json; jsoncorrections.trigger_sf_path(era, version) is "
                                                 "the location the processor picks up")
    parser.add_argument("--btag-output", help="b-tag efficiency json; jsoncorrections.btag_eff_path(era, version) is "
                                              "the location the processor picks up")
    options = parser.parse_args(argv)
    if "trigger" in options.derive and not options.trigger_output:
        parser.error("--derive trigger needs --trigger-output")
    if "btag" in options.derive and not (options.btag_output and options.xsections):
        parser.error("--derive btag needs --btag-output and --xsections")
    records = load_records(options.input)
    if "trigger" in options.derive:
        grids, edges = derive_trigger_sf(*trigger_inputs(records, options.era))
        write_payload(trigger_sf_payload(grids, edges, f"Di-lepton trigger SFs, {options.era} {options.version}, "
                                                       "orthogonal (MET reference) method, Clopper-Pearson"),
                      options.trigger_output)
    if "btag" in options.derive:
        with open(options.xsections) as fin:
            xsections = yaml.safe_load(fin)
        maps, edges, wps = derive_btag_eff(records, options.version, xsections)
        write_payload(btag_eff_payload(maps, edges, wps), options.btag_output)


if __name__ == "__main__":
    main()
