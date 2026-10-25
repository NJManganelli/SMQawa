import argparse
import collections
import fnmatch
import getpass
import glob
import gzip
import itertools
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import psutil
import rich
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.text import Text

LARGE_INPUT_MIB = 10


class MemoryUsageColumn(TextColumn):
    warn_threshold = 256
    severe_threshold = 1024

    def render(self, task):
        process = psutil.Process()
        memory_info = process.memory_info()
        # Convert bytes to MB for readability
        memory_mb = memory_info.rss / (1024 * 1024)
        if memory_mb > self.severe_threshold:
            style = "bold red underline"
        elif memory_mb > self.warn_threshold:
            style = "yellow underline"
        else:
            style = "green"
        return Text(f"Mem: {memory_mb:.2f}MB", style=style)


def progress_bar():
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        MemoryUsageColumn(""),
        TimeElapsedColumn(),
    )


def find_files(options):
    match = options.match or options.tag
    if options.eos:
        pattern = os.path.join(options.dir, match, options.era, options.sample, "histogram_*.pkl.gz")
    else:
        pattern = os.path.join(options.dir, f"jobs_{match}_{options.era}_{options.sample}", "histogram_*.pkl.gz")
    return pattern, sorted(glob.glob(pattern))


def read_gzip(filename):
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def read_whole_file(filename):
    with open(filename, "rb") as f:
        raw = f.read()
    return pickle.loads(gzip.decompress(raw))


def add_into(accumulator, key, value):
    if key in accumulator:
        accumulator[key] += value
    else:
        accumulator[key] = value.copy() if hasattr(value, "copy") else value


def merge_files(filenames, merged, load):
    empty = []
    for filename in filenames:
        if os.path.getsize(filename) == 0:
            empty.append(filename)
            continue
        key = None
        try:
            for dataset, content in load(filename).items():
                target = merged.setdefault(dataset, {"hist": {}, "sumw": 0.0})
                for name, histogram in content["hist"].items():
                    if isinstance(histogram, dict):
                        continue
                    key = f"{dataset}/{name}"
                    add_into(target["hist"], name, histogram)
                    key = None
                target["sumw"] += content["sumw"]
        except Exception as err:
            if key is None:
                raise RuntimeError(f"cannot merge {filename}: {err}") from err
            raise RuntimeError(f"cannot add {key} from {filename} to the histograms merged before it: {err}") from err
    return empty


def merge_chunk(filenames, load):
    merged = {}
    empty = merge_files(filenames, merged, load)
    return merged, empty


def merge_into(merged, partial, source):
    for dataset, content in partial.items():
        if dataset not in merged:
            merged[dataset] = content
            continue
        for name, histogram in content["hist"].items():
            try:
                add_into(merged[dataset]["hist"], name, histogram)
            except Exception as err:
                raise RuntimeError(f"cannot add {dataset}/{name} from {source} to the histograms merged "
                                   f"before it: {err}") from err
        merged[dataset]["sumw"] += content["sumw"]


def make_chunks(filenames, ncores):
    size = math.ceil(len(filenames) / ncores)
    chunks = []
    for _, group in itertools.groupby(filenames, key=os.path.dirname):
        group = list(group)
        chunks.extend(group[i:i + size] for i in range(0, len(group), size))
    return chunks


def merge_parallel(chunks, executor, ncores, load, live, done):
    merged, empty = {}, []
    chunks = iter(chunks)
    pending = collections.deque((chunk, executor.submit(merge_chunk, chunk, load))
                                for chunk in itertools.islice(chunks, ncores))
    with live:
        while pending:
            chunk, future = pending.popleft()
            try:
                partial, chunk_empty = future.result()
            except BrokenProcessPool as err:
                raise RuntimeError("a worker process died (for example killed for using too much memory); no "
                                   "output was written; rerun with fewer --ncores") from err
            empty.extend(chunk_empty)
            merge_into(merged, partial, chunk[0] if len(chunk) == 1 else f"files {chunk[0]} to {chunk[-1]}")
            done(chunk)
            for chunk in itertools.islice(chunks, 1):
                pending.append((chunk, executor.submit(merge_chunk, chunk, load)))
    return merged, empty


def main():
    parser = argparse.ArgumentParser(description="Histogram Merger")
    parser.add_argument("-t", "--tag", type=str, default="algiers",
                        help="tag used at submission; names the output and is matched exactly in "
                             "jobs_<tag>_<era>_<sample>")
    parser.add_argument("--match", type=str, default=None,
                        help="glob for the tag part of the input directories instead of the exact tag, e.g. '*WZ*'; "
                             "the output name still uses -t")
    parser.add_argument("--era", type=str, default="2018", help="")
    parser.add_argument("--dir", type=str, default=None,
                        help="base directory holding the outputs (default '.', with --eos "
                             "/eos/user/<u>/<user>/WZtotau2lnu)")
    parser.add_argument("--eos", action="store_true",
                        help="read <dir>/<tag>/<era>/<sample>/histogram_*.pkl.gz instead of "
                             "<dir>/jobs_<tag>_<era>_<sample>/histogram_*.pkl.gz")
    parser.add_argument("--sample", type=str, default="*",
                        help="glob pattern to merge only matching sample directories (default: all)")
    parser.add_argument("--outdir", type=str, default=".",
                        help="path to directory where files should be written, defaults to '.'")
    parser.add_argument("-f", "--filename", type=str, default="latest", help="output file name suffix")
    parser.add_argument("--ncores", type=int, default=None,
                        help="number of worker processes (default 4, or 1 when the largest input is at least 10 MiB); "
                             "1 or less merges in the main process; memory grows with --ncores")
    parser.add_argument("--verbose", action="store_true", help="list the matched directories with file counts")
    options = parser.parse_args()

    if set("*?[") & set(options.tag):
        parser.error("-t/--tag names the output and must not contain *, ? or [; use --match for a glob")
    if not os.path.isdir(options.outdir):
        parser.error(f"--outdir {options.outdir} is not a directory")
    if options.match and not fnmatch.fnmatchcase(options.tag, options.match):
        print(f"warning: -t {options.tag} does not match --match {options.match}; the output is still named after "
              f"{options.tag}")

    if options.dir is None:
        if options.eos:
            user = getpass.getuser()
            options.dir = f"/eos/user/{user[0]}/{user}/WZtotau2lnu"
        else:
            options.dir = "."
    pattern, filenames = find_files(options)
    print(f"globbing: {pattern}")
    if not filenames:
        raise SystemExit("no input pickle files found -- check --dir/--tag/--era/--sample "
                         "(and that /eos is mounted on this machine)")
    print(f"found {len(filenames)} files from {len({os.path.dirname(f) for f in filenames})} sample directories")
    if options.verbose:
        for directory, n in collections.Counter(os.path.dirname(f) for f in filenames).items():
            print(f"{directory}: {n} files")

    load = read_whole_file if options.eos else read_gzip
    largest = max(os.path.getsize(f) for f in filenames) / 2**20
    if options.ncores is None:
        ncores = 1 if largest >= LARGE_INPUT_MIB else 4
        if ncores == 1:
            rich.print(f"[yellow]largest input is {largest:.0f} MiB: merging in the main process to bound memory; "
                       f"pass --ncores to use workers anyway")
    else:
        ncores = max(1, options.ncores)
        if ncores > 1 and largest >= LARGE_INPUT_MIB:
            rich.print(f"[yellow]largest input is {largest:.0f} MiB: each of the {ncores} workers holds a partial "
                       f"merge and a decompressed job file (with --eos also the raw file contents), so memory grows "
                       f"with --ncores")
    progress = progress_bar()
    task = progress.add_task("[cyan]Merging histograms...", total=len(filenames))
    if ncores == 1:
        merged, empty = {}, []
        with progress:
            for filename in filenames:
                empty += merge_files([filename], merged, load)
                progress.advance(task)
    else:
        chunks = make_chunks(filenames, ncores)
        with ProcessPoolExecutor(max_workers=min(ncores, len(chunks))) as executor:
            merged, empty = merge_parallel(chunks, executor, ncores, load, progress,
                                           lambda chunk: progress.advance(task, len(chunk)))

    for directory, n in collections.Counter(os.path.dirname(f) for f in empty).items():
        rich.print(f"[red]{os.path.basename(directory)} contains {n} empty pkl.gz files!")

    output = os.path.join(options.outdir, f"merged-histogram-Inc-{options.tag}-{options.era}-{options.filename}.pkl.gz")
    with gzip.open(output + ".tmp", "wb") as f:
        pickle.dump(merged, f)
    os.replace(output + ".tmp", output)
    print(output)


if __name__ == "__main__":
    main()
