"""
Usage
-----
python polarization_split.py -i merged.pkl.gz -o merged_polsplit.pkl.gz
"""

from __future__ import annotations

import argparse
import copy
import gzip
import pickle
from typing import Dict, List

# Default signal dataset key as it appears in the accumulator
WZ_DATASET = "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8"

BOSON_CONFIG = {
    "W":  dict(obs="qcos_theta_w_reco", wpref="wW"),
    "Wp": dict(obs="cos_theta_wp_reco", wpref="wWp"),
    "Wm": dict(obs="cos_theta_wm_reco", wpref="wWm"),
    "Z":  dict(obs="cos_theta_z_reco",  wpref="wZ"),
}


def _template_key(obs: str, wpref: str, pol: str) -> str:
    return f"{obs}_{wpref}_{pol}"


def split_polarization_datasets(
    accumulator: Dict,
    bosons: List[str] = ("W", "Wp", "Wm", "Z"),
    wz_dataset: str = WZ_DATASET,
    keep_inclusive: bool = True,
    verbose: bool = True,
) -> Dict:
    if wz_dataset not in accumulator:
        if verbose:
            print(f"[pol-split] WZ dataset '{wz_dataset}' not in accumulator; "
                  f"available: {list(accumulator.keys())[:5]}...")
        return accumulator

    wz_entry = accumulator[wz_dataset]
    wz_hist_dict = wz_entry["hist"]
    wz_sumw = wz_entry["sumw"]

    for boson in bosons:
        if boson not in BOSON_CONFIG:
            if verbose:
                print(f"[pol-split] unknown boson '{boson}', skipping")
            continue
        cfg = BOSON_CONFIG[boson]
        obs, wpref = cfg["obs"], cfg["wpref"]
        for pol in ("long", "left", "right"):
            src_key = _template_key(obs, wpref, pol)
            if src_key not in wz_hist_dict:
                if verbose:
                    print(f"[pol-split] {boson}: missing template '{src_key}', skipping")
                continue
            h = copy.deepcopy(wz_hist_dict[src_key])
            new_dataset = f"{wz_dataset}_{boson}_{pol}"
            accumulator[new_dataset] = {"hist": {obs: h}, "sumw": wz_sumw}
            if verbose:
                print(f"[pol-split] wrote {new_dataset}  (obs={obs}, from {src_key})")

    if not keep_inclusive and wz_dataset in accumulator:
        if verbose:
            print(f"[pol-split] dropping inclusive '{wz_dataset}'")
        del accumulator[wz_dataset]

    return accumulator


def _main():
    ap = argparse.ArgumentParser(description="Split WZ polarization templates "
                                             "into standalone DCTools datasets.")
    ap.add_argument("-i", "--input", required=True, help="merged .pkl.gz")
    ap.add_argument("-o", "--output", required=True, help="output .pkl.gz")
    ap.add_argument("-b", "--bosons", nargs="+", default=["W", "Wp", "Wm", "Z"],
                    choices=["W", "Wp", "Wm", "Z"])
    ap.add_argument("--wz-dataset", default=WZ_DATASET)
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
