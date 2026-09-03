import gzip
import pickle
import argparse
import os
import glob
import math
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _read_pickle_gz(filename):
    # one big read is much friendlier to a network FS (EOS) than gzip.open's
    # many small buffered reads
    with open(filename, "rb") as f:
        raw = f.read()
    return pickle.loads(gzip.decompress(raw))


def _accumulate(acc, key, value):
    """Add value into acc[key], in place for numpy arrays."""
    cur = acc.get(key)
    if cur is None:
        # keep a private copy so later in-place adds don't touch loaded data
        acc[key] = value.copy() if hasattr(value, "copy") else value
        return
    try:
        cur += value            # in-place for ndarray, rebinds for scalars
        acc[key] = cur
    except Exception:
        acc[key] = cur + value  # fallback (dtype mismatch / exotic accumulators)


def merge_chunk(files):
    """Load a list of files and reduce them to one partial result in the worker.

    Doing the merge here keeps the per-bin addition parallel and drastically
    cuts how much data is shipped back to the parent (one partial per chunk
    instead of one dict per file).
    """
    combined_hist = {}
    combined_sumw = {}
    for filename in files:
        try:
            if os.path.getsize(filename) == 0:
                continue
            data = _read_pickle_gz(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        for s, v in data.items():
            if s not in combined_hist:
                combined_hist[s] = {}
                combined_sumw[s] = 0.0
            acc = combined_hist[s]
            for k, val in v["hist"].items():
                if isinstance(val, dict):      # skip nested dicts, as before
                    continue
                _accumulate(acc, k, val)
            combined_sumw[s] += v["sumw"]
    return combined_hist, combined_sumw


def merge_into(combined_hist, combined_sumw, partial_hist, partial_sumw):
    for s, acc in partial_hist.items():
        if s not in combined_hist:
            combined_hist[s] = acc                 # adopt worker's dict directly
            combined_sumw[s] = partial_sumw[s]
        else:
            dst = combined_hist[s]
            for k, val in acc.items():
                _accumulate(dst, k, val)
            combined_sumw[s] += partial_sumw[s]


def merger():
    parser = argparse.ArgumentParser(description='Fast Histogram Merger (EOS)')
    parser.add_argument("-t", '--tag', type=str, default="algiers", help="Tag name used at submission")
    parser.add_argument('--era', type=str, default="2018", help="Era used at submission")
    parser.add_argument("-f", '--filename', type=str, default="latest_DD", help="output file name suffix")
    parser.add_argument("--eosdir", type=str, default=None,
                        help="EOS base directory holding the outputs (default: "
                             "/eos/user/<u>/<user>/WZtotau2lnu). Histograms are read from "
                             "<eosdir>/<tag>/<era>/<sample>/histogram_*.pkl.gz")
    parser.add_argument("--sample", type=str, default="*",
                        help="glob pattern to merge only matching sample directories (default: all)")
    parser.add_argument("--legacy-jobsdir", action="store_true",
                        help="merge from the old local jobs_* directories instead of EOS")
    parser.add_argument("--nprocs", type=int, default=cpu_count(),
                        help="worker processes (bump above core count if EOS reads are I/O bound)")
    parser.add_argument("--compresslevel", type=int, default=4,
                        help="gzip level for the OUTPUT file (1=fast/larger .. 9=slow/smaller)")
    options = parser.parse_args()

    if options.legacy_jobsdir:
        file_list = glob.glob(f'*{options.tag}*_{options.era}_*/*.pkl.gz')
    else:
        if options.eosdir is None:
            user_name = os.environ['USER']
            options.eosdir = f"/eos/user/{user_name[0]}/{user_name}/WZtotau2lnu"
        pattern = os.path.join(options.eosdir, options.tag, options.era,
                               options.sample, "histogram_*.pkl.gz")
        print(f"globbing: {pattern}")
        file_list = glob.glob(pattern)

    if not file_list:
        raise SystemExit("no input pickle files found -- check --eosdir/--tag/--era "
                         "(and that /eos is mounted on this machine)")

    n_samples = len({os.path.dirname(fn) for fn in file_list})
    print(f"found {len(file_list)} files from {n_samples} sample directories")

    nprocs = max(1, options.nprocs)
    # more chunks than workers -> better load balancing, but bounded so the
    # parent's final reduce stays cheap
    n_chunks = min(len(file_list), nprocs * 4)
    chunk_size = math.ceil(len(file_list) / n_chunks)
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    combined_hist = {}
    combined_sumw = {}
    with Pool(processes=nprocs) as pool:
        for partial_hist, partial_sumw in tqdm(
                pool.imap_unordered(merge_chunk, chunks),
                total=len(chunks), desc="Merging", ncols=75):
            merge_into(combined_hist, combined_sumw, partial_hist, partial_sumw)

    combined_dict = {
        s: {"hist": h, "sumw": combined_sumw[s]}
        for s, h in combined_hist.items()
    }

    output_file = f"merged-histogram-Inc-{options.tag}-{options.era}-{options.filename}.pkl.gz"
    with gzip.open(output_file, "wb", compresslevel=options.compresslevel) as f:
        pickle.dump(combined_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Done. Output written to: {output_file}")


if __name__ == "__main__":
    merger()