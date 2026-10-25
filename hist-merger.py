import argparse
import collections
import contextlib
import fnmatch
import getpass
import glob
import gzip
import itertools
import math
import os
import pickle
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy
import psutil
import rich
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, ProgressBar, ProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Column, Table
from rich.text import Text

LARGE_INPUT_MIB = 10
LABEL_WIDTH = 20
BAR_WIDTH = 24
KIND_FILES = {"data": "Data", "mc": "MC-backgrounds", "signal": "MC-signal", "eft": "EFT"}


def tree_memory(processes):
    rss, count = 0, 0
    for process in processes:
        try:
            rss += process.memory_info().rss
        except psutil.Error:
            continue
        count += 1
    return rss, count


class MemoryBarColumn(ProgressColumn):
    max_refresh = 0.5

    def __init__(self, workers):
        super().__init__()
        self.workers = workers
        self.total = psutil.virtual_memory().total
        self.parent = psutil.Process()
        self.processes, self.listed = [self.parent], None

    def current_processes(self):
        if self.workers and (self.listed is None or time.monotonic() - self.listed >= 5):
            self.processes = [self.parent, *self.parent.children(recursive=True)]
            self.listed = time.monotonic()
        return self.processes

    def render(self, task):
        try:
            rss, count = tree_memory(self.current_processes())
            available = psutil.virtual_memory().available
        except (psutil.Error, OSError):
            return Text("memory unavailable")
        fraction = rss / self.total
        low = available < 0.1 * self.total
        style = "bold red" if fraction >= 0.8 or low else "yellow" if fraction >= 0.5 else "green"
        grid = Table.grid(padding=(0, 1))
        grid.add_row(ProgressBar(total=self.total, completed=rss, width=BAR_WIDTH, complete_style=style,
                                 finished_style=style),
                     Text(f"{rss / 2**30:.1f}/{self.total / 2**30:.1f} GiB, {available / 2**30:.1f} free, "
                          f"{count} proc", style=style, no_wrap=True, overflow="ellipsis"))
        return grid


class MergeDisplay:
    def __init__(self, filenames, nchunks, enabled, era, console=None):
        self.era = era
        self.counts = collections.Counter(os.path.dirname(filename) for filename in filenames)
        self.current, self.done, self.finished = None, 0, 0
        label = Column(width=LABEL_WIDTH, no_wrap=True, overflow="ellipsis")
        self.progress = Progress(TextColumn("{task.description}", table_column=label),
                                 BarColumn(bar_width=BAR_WIDTH), MofNCompleteColumn(), TimeElapsedColumn(),
                                 TimeRemainingColumn(), console=console)
        self.directories = self.progress.add_task(self.directories_label(), total=len(filenames))
        self.files = self.progress.add_task("files", total=None)
        self.chunks = self.progress.add_task("chunks", total=nchunks, visible=nchunks > 0)
        memory = Progress(TextColumn("memory", table_column=Column(width=LABEL_WIDTH)), MemoryBarColumn(nchunks > 0),
                          console=console)
        memory.add_task("memory", total=None)
        self.live = (Live(Group(self.progress, memory), console=console, refresh_per_second=4) if enabled
                     else contextlib.nullcontext())

    def directories_label(self):
        return f"dirs {self.finished}/{len(self.counts)}"

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, *exc):
        return self.live.__exit__(*exc)

    def files_done(self, filenames):
        for directory, group in itertools.groupby(filenames, key=os.path.dirname):
            if directory != self.current:
                self.current, self.done = directory, 0
                sample = os.path.basename(directory).split(f"_{self.era}_", 1)[-1]
                self.progress.reset(self.files, total=self.counts[directory], description=f"  {sample}")
            n = len(list(group))
            self.done += n
            self.progress.advance(self.files, n)
            if self.done == self.counts[directory]:
                self.finished += 1
            self.progress.update(self.directories, advance=n, description=self.directories_label())

    def chunk_done(self, chunk):
        self.progress.advance(self.chunks)
        self.files_done(chunk)


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


def add_lumi(target, lumi):
    blocks = numpy.asarray(lumi).reshape(-1, 2)
    if "lumi" in target:
        if len(blocks) == 0:
            return
        if len(target["lumi"]):
            blocks = numpy.concatenate([target["lumi"], blocks])
    target["lumi"] = numpy.unique(blocks, axis=0)


def add_kind(target, kind, new):
    if kind is not None and kind not in KIND_FILES:
        raise ValueError(f"unknown kind {kind!r}, expected one of {', '.join(KIND_FILES)}")
    if new:
        if kind is not None:
            target["kind"] = kind
    elif target.get("kind") != kind:
        raise ValueError(f"kind {kind!r} disagrees with kind {target.get('kind')!r}")


def merge_files(filenames, merged, load):
    empty = []
    for filename in filenames:
        if os.path.getsize(filename) == 0:
            empty.append(filename)
            continue
        key = None
        try:
            for dataset, content in load(filename).items():
                new = dataset not in merged
                target = merged.setdefault(dataset, {"hist": {}, "sumw": 0.0})
                for name, histogram in content["hist"].items():
                    if isinstance(histogram, dict):
                        continue
                    key = f"{dataset}/{name}"
                    add_into(target["hist"], name, histogram)
                    key = None
                target["sumw"] += content["sumw"]
                if "lumi" in content:
                    key = f"{dataset}/lumi"
                    add_lumi(target, content["lumi"])
                    key = None
                key = f"{dataset}/kind"
                add_kind(target, content.get("kind"), new)
                key = None
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
        if "lumi" in content:
            try:
                add_lumi(merged[dataset], content["lumi"])
            except Exception as err:
                raise RuntimeError(f"cannot add {dataset}/lumi from {source} to the histograms merged "
                                   f"before it: {err}") from err
        try:
            add_kind(merged[dataset], content.get("kind"), False)
        except Exception as err:
            raise RuntimeError(f"cannot add {dataset}/kind from {source} to the histograms merged "
                               f"before it: {err}") from err


def split_by_kind(merged):
    parts = {"Data": {}, "MC-backgrounds": {}, "MC-signal": {}}
    for dataset, content in merged.items():
        kind = content.get("kind")
        if kind is None:
            if content["sumw"] >= 0:
                raise SystemExit(f"--splitByKind: dataset {dataset} has no kind and sumw {content['sumw']} >= 0, so "
                                 f"it is not data; cannot tell signal from background without kind")
            kind = "data"
        parts.setdefault(KIND_FILES[kind], {})[dataset] = content
    return parts


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


def peak_rss(who):
    peak = resource.getrusage(who).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main():
    start = time.monotonic()
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
    parser.add_argument("--compressionLevel", type=int, default=4,
                        help="gzip level for the OUTPUT file (1=fast/larger .. 9=slow/smaller)")
    parser.add_argument("--splitByKind", action="store_true",
                        help="write the Data, MC-backgrounds and MC-signal datasets (and EFT ones, if any) to "
                             "separate files ...-<filename>-<part>.pkl.gz, by each dataset's kind; without kind, "
                             "a negative sumw means data")
    parser.add_argument("--verbose", action="store_true", help="list the matched directories with file counts")
    parser.add_argument("--noProgress", action="store_true",
                        help="do not show the progress display; it is also off when stdout is not a terminal")
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
    ndirectories = len({os.path.dirname(f) for f in filenames})
    print(f"found {len(filenames)} files from {ndirectories} sample directories")
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
    enabled = not options.noProgress and Console().is_terminal
    if ncores == 1:
        merged, empty = {}, []
        with MergeDisplay(filenames, 0, enabled, options.era) as display:
            for filename in filenames:
                empty += merge_files([filename], merged, load)
                display.files_done([filename])
    else:
        chunks = make_chunks(filenames, ncores)
        display = MergeDisplay(filenames, len(chunks), enabled, options.era)
        with ProcessPoolExecutor(max_workers=min(ncores, len(chunks))) as executor:
            merged, empty = merge_parallel(chunks, executor, ncores, load, display, display.chunk_done)

    for directory, n in collections.Counter(os.path.dirname(f) for f in empty).items():
        rich.print(f"[red]{os.path.basename(directory)} contains {n} empty pkl.gz files!")

    name = f"merged-histogram-Inc-{options.tag}-{options.era}-{options.filename}"
    if options.splitByKind:
        outputs = {}
        for part, content in split_by_kind(merged).items():
            basename = f"{name}-{part}.pkl.gz"
            if not content and os.path.exists(os.path.join(options.outdir, basename)):
                print(f"{basename}: no {part} datasets in this merge, so the existing file is kept")
            else:
                outputs[basename] = content
    else:
        for content in merged.values():
            content.pop("kind", None)
        outputs = {f"{name}.pkl.gz": merged}
    paths = [os.path.join(options.outdir, basename) for basename in outputs]
    try:
        for output, content in zip(paths, outputs.values()):
            with gzip.open(output + ".tmp", "wb", compresslevel=options.compressionLevel) as f:
                pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException:
        for output in paths:
            with contextlib.suppress(FileNotFoundError):
                os.remove(output + ".tmp")
        raise
    for output in paths:
        os.replace(output + ".tmp", output)
        print(output)
    summary = (f"merged {len(filenames)} files from {ndirectories} directories in {time.monotonic() - start:.1f} s; "
               f"peak RSS: main process {peak_rss(resource.RUSAGE_SELF) / 2**30:.2f} GiB")
    if ncores > 1:
        summary += f", largest worker {peak_rss(resource.RUSAGE_CHILDREN) / 2**30:.2f} GiB"
    print(summary)


if __name__ == "__main__":
    main()
