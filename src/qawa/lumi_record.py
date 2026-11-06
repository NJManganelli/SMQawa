import argparse
import gzip
import json
import os
import pickle
import sys

import numpy

LUMI_KEY = "lumi"

DESCRIPTION = """Report the lumi records of data datasets in histogram pickles.

A data dataset record may carry "lumi": the sorted (run, luminosityBlock) pairs, an (n, 2) uint32
array, of the events that passed the golden-JSON LumiMask. The record is optional: a pickle or a
dataset without one is listed as "no lumi record" and is not an error.

  python -m qawa.lumi_record report PICKLE... [--lumi-csv CSV] [--json OUT] [--export-json DIR]

The report gives, per pickle and dataset, the runs, lumi sections and first and last run, and, per
pair of datasets of one pickle, the lumi sections only in A, only in B and in both. Streams are
reported separately and never summed. The integrated luminosity needs a brilcalc CSV: write each
dataset's record as a CMS JSON with --export-json DIR, run
  brilcalc lumi -i DIR/<dataset>_lumis.json --byls --output-style csv -u /pb > lumi.csv
and give the CSV back with --lumi-csv.
"""

CSV_HINT = ("no --lumi-csv: luminosities are not computed; write the records with --export-json DIR, run "
            "`brilcalc lumi -i DIR/<dataset>_lumis.json --byls --output-style csv -u /pb > lumi.csv` and pass "
            "--lumi-csv lumi.csv")


def lumi_pairs(runs, lumis, mask):
    mask = numpy.asarray(mask, dtype=bool)
    stacked = numpy.stack([numpy.asarray(runs)[mask], numpy.asarray(lumis)[mask]], axis=1)
    return {(int(run), int(lumi)) for run, lumi in numpy.unique(stacked, axis=0)}


def as_array(pairs):
    return numpy.unique(numpy.asarray(list(pairs), dtype=numpy.uint32).reshape(-1, 2), axis=0)


def union(*arrays):
    return numpy.unique(numpy.concatenate([as_array(array) for array in arrays] + [as_array(())]), axis=0)


def to_cms_json(array):
    ranges = {}
    for run, lumi in as_array(array).tolist():
        spans = ranges.setdefault(str(run), [])
        if spans and spans[-1][1] == lumi - 1:
            spans[-1][1] = lumi
        else:
            spans.append([lumi, lumi])
    return ranges


def read_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def records_of(content):
    return {dataset: as_array(record[LUMI_KEY]) for dataset, record in content.items()
            if isinstance(record, dict) and LUMI_KEY in record}


def report(paths, lumi_csv=None):
    lumi_data = None
    if lumi_csv is not None:
        from coffea.lumi_tools import LumiData
        lumi_data = LumiData(lumi_csv)

    def luminosity(array):
        if lumi_data is None:
            return None
        return float(lumi_data.get_lumi(array)) if len(array) else 0.0

    out = {}
    for path in paths:
        content = read_pickle(path)
        records = records_of(content)
        datasets = {}
        for dataset, array in records.items():
            runs = numpy.unique(array[:, 0])
            datasets[dataset] = {"runs": int(len(runs)), "lumi_sections": int(len(array)),
                                 "first_run": int(runs[0]) if len(runs) else None,
                                 "last_run": int(runs[-1]) if len(runs) else None, "lumi": luminosity(array)}
        pairs = []
        names = sorted(records)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                set_a = {tuple(pair) for pair in records[a].tolist()}
                set_b = {tuple(pair) for pair in records[b].tolist()}
                parts = {"only_a": set_a - set_b, "only_b": set_b - set_a, "both": set_a & set_b}
                pair = {"a": a, "b": b, **{name: len(part) for name, part in parts.items()}}
                pair["lumi"] = None if lumi_data is None else {name: luminosity(as_array(part))
                                                               for name, part in parts.items()}
                pairs.append(pair)
        out[path] = {"datasets": datasets, "no_lumi_record": sorted(set(content) - set(records)), "pairs": pairs}
    return out


def print_report(result, lumi_csv):
    for path, entry in result.items():
        print(path)
        if not entry["datasets"]:
            print("  no lumi record")
        for dataset, row in sorted(entry["datasets"].items()):
            print(f"  {dataset}: runs {row['runs']} ({row['first_run']}-{row['last_run']}), "
                  f"lumi sections {row['lumi_sections']}, lumi: {row['lumi']}")
        for dataset in entry["no_lumi_record"]:
            print(f"  {dataset}: no lumi record")
        for pair in entry["pairs"]:
            print(f"  {pair['a']} vs {pair['b']}: only A {pair['only_a']}, only B {pair['only_b']}, "
                  f"both {pair['both']}, lumi: {pair['lumi']}")
    if lumi_csv is None:
        print(CSV_HINT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report", help="per-dataset and pairwise lumi report of histogram pickles")
    p.add_argument("pickles", nargs="+", metavar="PICKLE")
    p.add_argument("--lumi-csv", help="brilcalc --byls --output-style csv output for the integrated luminosity")
    p.add_argument("--json", help="also write the report as JSON to this path")
    p.add_argument("--export-json", metavar="DIR", help="write <dataset>_lumis.json (CMS JSON) per dataset record")
    args = parser.parse_args(argv)
    result = report(args.pickles, args.lumi_csv)
    print_report(result, args.lumi_csv)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1, sort_keys=True)
    if args.export_json:
        os.makedirs(args.export_json, exist_ok=True)
        for path in args.pickles:
            for dataset, array in records_of(read_pickle(path)).items():
                with open(os.path.join(args.export_json, f"{dataset}_lumis.json"), "w") as f:
                    json.dump(to_cms_json(array), f, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
