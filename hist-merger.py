import argparse
import collections
import fnmatch
import glob
import gzip
import os
import pickle

import psutil
import rich
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.text import Text


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
    pattern = os.path.join(options.dir, f"jobs_{match}_{options.era}_*", "histogram_*.pkl.gz")
    return sorted(glob.glob(pattern))


def read_gzip(filename):
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def merge_files(filenames, merged):
    empty = []
    for filename in filenames:
        if os.path.getsize(filename) == 0:
            empty.append(filename)
            continue
        key = None
        try:
            for dataset, content in read_gzip(filename).items():
                target = merged.setdefault(dataset, {"hist": {}, "sumw": 0.0})
                hists = target["hist"]
                for name, histogram in content["hist"].items():
                    if isinstance(histogram, dict):
                        continue
                    key = f"{dataset}/{name}"
                    hists[name] = hists[name] + histogram if name in hists else histogram
                    key = None
                target["sumw"] += content["sumw"]
        except Exception as err:
            if key is None:
                raise RuntimeError(f"cannot merge {filename}: {err}") from err
            raise RuntimeError(f"cannot add {key} from {filename} to the histograms merged before it: {err}") from err
    return empty


def main():
    parser = argparse.ArgumentParser(description="Histogram Merger")
    parser.add_argument("-t", "--tag", type=str, default="algiers",
                        help="tag used at submission; names the output and is matched exactly in "
                             "jobs_<tag>_<era>_<sample>")
    parser.add_argument("--match", type=str, default=None,
                        help="glob for the tag part of the input directories instead of the exact tag, e.g. '*WZ*'; "
                             "the output name still uses -t")
    parser.add_argument("--era", type=str, default="2018", help="")
    parser.add_argument("--dir", type=str, default=".", help="path to directory where files are stored, defaults to '.'")
    parser.add_argument("--outdir", type=str, default=".",
                        help="path to directory where files should be written, defaults to '.'")
    parser.add_argument("-f", "--filename", type=str, default="latest_DD", help="output file name suffix")
    parser.add_argument("--verbose", action="store_true", help="list the matched directories with file counts")
    options = parser.parse_args()

    if set("*?[") & set(options.tag):
        parser.error("-t/--tag names the output and must not contain *, ? or [; use --match for a glob")
    if not os.path.isdir(options.outdir):
        parser.error(f"--outdir {options.outdir} is not a directory")
    if options.match and not fnmatch.fnmatchcase(options.tag, options.match):
        print(f"warning: -t {options.tag} does not match --match {options.match}; the output is still named after "
              f"{options.tag}")

    filenames = find_files(options)
    if options.verbose:
        for directory, n in collections.Counter(os.path.dirname(f) for f in filenames).items():
            print(f"{directory}: {n} files")

    merged, empty = {}, []
    with progress_bar() as progress:
        task = progress.add_task("[cyan]Merging histograms...", total=len(filenames))
        for filename in filenames:
            empty += merge_files([filename], merged)
            progress.advance(task)

    for directory, n in collections.Counter(os.path.dirname(f) for f in empty).items():
        rich.print(f"[red]{os.path.basename(directory)} contains {n} empty pkl.gz files!")

    output = os.path.join(options.outdir, f"merged-histogram-Inc-{options.tag}-{options.era}-{options.filename}.pkl.gz")
    with gzip.open(output + ".tmp", "wb") as f:
        pickle.dump(merged, f)
    os.replace(output + ".tmp", output)
    print(output)


if __name__ == "__main__":
    main()
