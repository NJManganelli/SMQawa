import gzip 
import pickle
import argparse
import collections
import functools
import itertools
import os, glob
from tqdm import tqdm

import psutil
import time
import rich
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.live import Live
from rich.text import Text

class MemoryUsageColumn(TextColumn):
    warn_threshold = 256
    severe_threshold = 1024
    def render(self, task: "Task") -> Text:
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

def merge_hists_sumw(dataset_dict, dataset, hist_dict):
    if dataset not in dataset_dict:
        dataset_dict[dataset] = {'hist': hist_dict['hist'],
                                 'sumw': hist_dict['sumw']
                                 }
    else:
        for histo_name, histo in hist_dict['hist'].items():
            if histo_name not in dataset_dict[dataset]['hist']:
                dataset_dict[dataset]['hist'][histo_name] = histo
            else:
                dataset_dict[dataset]['hist'][histo_name] = dataset_dict[dataset]['hist'][histo_name] + histo
        dataset_dict[dataset]['sumw'] = dataset_dict[dataset]['sumw'] + hist_dict['sumw']
    return dataset_dict


def ram_merger(options):
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        MemoryUsageColumn(""), # Custom memory column
        TimeElapsedColumn(),
    ) as progress:
        filenames = glob.glob(os.path.join(options.dir, f'*{options.tag}*_{options.era}_*', '*.pkl.gz'))
        split_filenames = {name: name.split("/") for name in filenames}

        dirs_to_merge = {}
        task1 = progress.add_task("[cyan]Parsing filenames...", total=len(split_filenames))
        for name, split_name in split_filenames.items():
            dir_name = split_name[-2]
            if dir_name not in dirs_to_merge:
                dirs_to_merge[dir_name] = [name]
            else:
                dirs_to_merge[dir_name].append(name)
            progress.update(task1, advance=1)
        progress.remove_task(task1)

        # dataset_merged_results = {}
        # dataset_sumw = {}
        skipped_sub_histograms = []
        written_sub_histograms = []
        empty_filenames = {}
        # task2 = progress.add_task("[cyan]Merging Directory Histograms...", total=len(dirs_to_merge))
        task2 = progress.add_task("[cyan]Merging into directory histograms...", total=len(filenames))
        for dir_name, dir_filenames in dirs_to_merge.items():
            sub_histograms_file = options.outdir + "/" + dir_name + "_histograms_v2.pkl.gz"
            if os.path.exists(sub_histograms_file) and not options.force:
                skipped_sub_histograms.append(sub_histograms_file)
                if options.verbose:
                    rich.print(f"[green]Skipping creation of sub-histograms for {sub_histograms_file}... use --force to override")
                progress.update(task2, advance=len(dir_filenames))
                continue
            dataset_histos = {} #both 'hist' and 'sumw' in one dictionary
            merged_hists = None
            merged_sumw = None
            empty_filenames[dir_name] = []
            task3 = progress.add_task(f"\t[green]Merging {dir_name}...", total=len(dir_filenames))
            for filename in dir_filenames:
                if os.path.getsize(filename) == 0:
                    empty_filenames[dir_name].append(filename)
                    progress.update(task3, advance=1)
                    progress.update(task2, advance=1)
                else:
                    with gzip.open(filename, 'rb') as f:
                        _data = pickle.load(f)
                        for dataset, raw_results in _data.items():
                            these_dataset_hists = dict(filter(lambda n: not isinstance(n[1], dict), raw_results['hist'].items()))
                            dataset_histos = merge_hists_sumw(dataset_histos, dataset, {'hist': these_dataset_hists, 'sumw': raw_results['sumw']})
                        f.close()
                        del f
                        del _data
                progress.update(task3, advance=1)
                progress.update(task2, advance=1) # counts by filenames so gets incremented at the same time as task3 directory merging
            with gzip.open(sub_histograms_file, "wb") as fo:
                pickle.dump(dataset_histos, fo)
            written_sub_histograms.append(sub_histograms_file)
            progress.remove_task(task3)
            # progress.update(task2, advance=1)
        for dir_name, empty_files in empty_filenames.items():
            rich.print(f"[red]{dir_name} contains {len(empty_files)} empty pkl.gz files!")

        combined_dict = {}
        rich.print("[yellow]Loading already sub-merged histograms")
        all_sub_histogram_filenames = skipped_sub_histograms + written_sub_histograms
        task4 = progress.add_task(f"[cyan]Loading directory histograms...", total=len(all_sub_histogram_filenames))
        for sub_histogram_filename in all_sub_histogram_filenames:
            with gzip.open(sub_histogram_filename, "rb") as sf:
                _data = pickle.load(sf)
                for reloaded_dataset, sub_merged_results in _data.items():
                    combined_dict = merge_hists_sumw(combined_dict, reloaded_dataset, sub_merged_results)
                sf.close()
                del sf
                del _data
            progress.update(task4, advance=1)

        with gzip.open(options.outdir + f"/merged-histogram-{options.tag}-{options.era}-Inc_WZ_latest_rawdeeptau.pkl.gz", "wb") as ff:
            pickle.dump(combined_dict, ff)
            ff.close()

def merger(options):
    # parser = argparse.ArgumentParser(description='Famous Submitter')
    # parser.add_argument("-t", '--tag', type=str, default="algiers", help="")
    # parser.add_argument('--era', type=str, default="2018"   , help="")
    # options = parser.parse_args()
    

    def updated(c, items):
        c.update(items)
        return c


    all_hists = []
    for filename in tqdm(glob.glob(f'*{options.tag}*_{options.era}_*/*.pkl.gz'), desc="reading", ascii=False, ncols=75):
        if os.path.getsize(filename) == 0: 
            print(f"{filename} is empty !! ")
        else:
            with gzip.open(filename, 'rb') as f:
                _data = pickle.load(f)
                all_hists.append(_data)
                f.close()
                del f 
                del _data
            
    combined_hist = {}
    combined_sumw = {}
    
    for i in tqdm(all_hists, desc="format", ascii=False, ncols=75):
        for s, v in i.items():
            v_hist = dict(filter(lambda n: not isinstance(n[1], dict), v['hist'].items()))
            if s in combined_hist:
                combined_hist[s].append(v_hist)
                combined_sumw[s].append(v['sumw'])
            else:
                combined_hist[s] = [v_hist]
                combined_sumw[s] = [v["sumw"]]
    
    del all_hists

    combined_dict = {}
    for s, h in tqdm(combined_hist.items(), desc="merging", ascii=False, ncols=75):
        hist_ = dict(functools.reduce(updated, h, collections.Counter()))
        sumw_ = sum(combined_sumw[s])
        combined_dict[s] = {"hist": hist_, "sumw": sumw_}

    with gzip.open(options.outdir + f"/merged-histogram-{options.tag}-{options.era}-Inc_WZ_latest_rawdeeptau.pkl.gz", "wb") as ff:
        pickle.dump(combined_dict, ff)
        ff.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Histogram Merger')
    parser.add_argument("-t", '--tag', type=str, default="algiers", help="")
    parser.add_argument('--era', type=str, default="2018"   , help="")
    parser.add_argument('--dir', type=str, default="."      , help="path to directory where files are stored, defaults to '.'")
    parser.add_argument('--outdir', type=str, default="."   , help="path to directory where files should be written, defaults to '.'")
    parser.add_argument('--force', action='store_true'      , help="force overwrite of sub-histogram files and final histogram file")
    parser.add_argument('--verbose', action='store_true'    , help="Print more information about merging process")
    parser.add_argument('--legacy', action='store_true', help="Merge with the original merger method, must be able to load all histos to RAM")
    options = parser.parse_args()

    if options.legacy:
        merger(options)
    else:
        ram_merger(options)
