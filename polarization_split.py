"""
Split the WZ polarization templates into standalone DCTools datasets.

Six boson combinations x three helicity states = 18 templates:

    boson   observable            weight prefix   meaning
    W       tau_loosetovtight_qcos_theta_w_reco   wZW             W polarization, inclusive WZ
    Wp      tau_loosetovtight_cos_theta_wp_reco   wZWp            W+ polarization
    Wm      tau_loosetovtight_cos_theta_wm_reco   wZWm            W- polarization
    Z       cos_theta_z_reco                      wWZ             Z polarization, inclusive WZ
    Z_WpZ   tau_loosetovtight_cos_theta_wpz_reco  wWpZ            Z polarization in W+Z
    Z_WmZ   tau_loosetovtight_cos_theta_wmz_reco  wWmZ            Z polarization in W-Z

Each template <obs>_<wpref>_<pol> (pol = long, left, right) is written as its
own dataset <wz_dataset>_<boson>_<pol>, holding one histogram under the key <obs>.

Usage
-----
python polarization_split.py -i merged.pkl.gz -o merged_polsplit.pkl.gz
"""

from __future__ import annotations

import argparse
import copy
import gzip
import pickle
from typing import Dict, List, Optional

POLS = ("long", "left", "right")

# boson -> observable and weight prefix, as named in the processor output
BOSON_CONFIG = {
    "W":     dict(obs="tau_loosetovtight_qcos_theta_w_reco",  wpref="wZW"),
    "Wp":    dict(obs="tau_loosetovtight_cos_theta_wp_reco",  wpref="wZWp"),
    "Wm":    dict(obs="tau_loosetovtight_cos_theta_wm_reco",  wpref="wZWm"),
    "Z":     dict(obs="cos_theta_z_reco",                     wpref="wWZ"),
    "Z_WpZ": dict(obs="tau_loosetovtight_cos_theta_wpz_reco", wpref="wWpZ"),
    "Z_WmZ": dict(obs="tau_loosetovtight_cos_theta_wmz_reco", wpref="wWmZ"),
}


def _template_key(obs: str, wpref: str, pol: str) -> str:
    return f"{obs}_{wpref}_{pol}"


def _find_wz_dataset(accumulator: Dict) -> Optional[str]:
    """The single WZ signal entry (Run 2 amcatnlo 'WZTo3LNu' or Run 3 powheg 'WZto3LNu')."""
    cands = [k for k in accumulator
             if k.startswith(("WZTo3LNu", "WZto3LNu")) and not k.endswith(POLS)]
    return cands[0] if len(cands) == 1 else None


def split_polarization_datasets(
    accumulator: Dict,
    bosons: List[str] = tuple(BOSON_CONFIG),
    wz_dataset: Optional[str] = None,
    keep_inclusive: bool = True,
    verbose: bool = True,
) -> Dict:
    wz_dataset = wz_dataset or _find_wz_dataset(accumulator)
    if wz_dataset not in accumulator:
        raise KeyError(f"[pol-split] WZ dataset {wz_dataset!r} not found; "
                       f"available: {list(accumulator)[:5]}... (pass --wz-dataset)")

    wz_entry = accumulator[wz_dataset]
    wz_hist_dict = wz_entry["hist"]
    wz_sumw = wz_entry["sumw"]

    written, missing = 0, []
    for boson in bosons:
        cfg = BOSON_CONFIG[boson]
        obs, wpref = cfg["obs"], cfg["wpref"]
        for pol in POLS:
            src_key = _template_key(obs, wpref, pol)
            if src_key not in wz_hist_dict:
                missing.append(src_key)
                continue
            h = copy.deepcopy(wz_hist_dict[src_key])
            new_dataset = f"{wz_dataset}_{boson}_{pol}"
            accumulator[new_dataset] = {"hist": {obs: h}, "sumw": wz_sumw}
            written += 1
            if verbose:
                print(f"[pol-split] wrote {new_dataset}  (obs={obs}, from {src_key})")

    expected = len(bosons) * len(POLS)
    if verbose:
        print(f"[pol-split] {written}/{expected} templates written")
    if missing:
        raise KeyError(f"[pol-split] {len(missing)} template(s) missing from '{wz_dataset}': {missing}")

    if not keep_inclusive:
        if verbose:
            print(f"[pol-split] dropping inclusive '{wz_dataset}'")
        del accumulator[wz_dataset]

    return accumulator


def _main():
    ap = argparse.ArgumentParser(description="Split WZ polarization templates "
                                             "into standalone DCTools datasets.")
    ap.add_argument("-i", "--input", required=True, help="merged .pkl.gz")
    ap.add_argument("-o", "--output", required=True, help="output .pkl.gz")
    ap.add_argument("-b", "--bosons", nargs="+", default=list(BOSON_CONFIG),
                    choices=list(BOSON_CONFIG))
    ap.add_argument("--wz-dataset", default=None,
                    help="signal dataset key (default: the single WZTo3LNu/WZto3LNu entry)")
    ap.add_argument("--drop-inclusive", action="store_true",
                    help="remove the original inclusive WZ dataset from output")
    args = ap.parse_args()

    open_fn = gzip.open if args.input.endswith(".gz") else open
    with open_fn(args.input, "rb") as f:
        data = pickle.load(f)

    data = split_polarization_datasets(
        data,
        bosons=args.bosons,
        wz_dataset=args.wz_dataset,
        keep_inclusive=not args.drop_inclusive,
        verbose=True,
    )

    out_fn = gzip.open if args.output.endswith(".gz") else open
    with out_fn(args.output, "wb") as f:
        pickle.dump(data, f)
    print(f"[pol-split] wrote {args.output}")


if __name__ == "__main__":
    _main()
