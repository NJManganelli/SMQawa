from coffea import processor
from coffea import nanoevents
from coffea.nanoevents import NanoAODSchema, BaseSchema
from coffea.util import coffea_console
from qawa.process.coffea_sumw import coffea_sumw
import argparse
import pickle
import gzip
import re
import sys
import os
import traceback
import uproot
import numpy as np
import traceback
import tempfile

NanoAODSchema.warn_missing_crossrefs = False

np.seterr(all='ignore')

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
uproot.open.defaults["timeout"] = 650 # wait more


def validate_input_file(nanofile):
    pfn = nanofile
    pfn=re.sub("\n","",pfn)
    aliases = [
        "root://cms-xrd-global.cern.ch/",
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global.cern.ch/",
        "root://cmsxrootd.fnal.gov/"
    ]

    valid = False
    for alias in aliases:
        testfile = None
        try:
            testfile=uproot.open(alias + pfn)
        except:
            pass
        if testfile:
            nanofile=alias + pfn
            coffea_console.print(f'--> {alias} OK')
            valid = True
            break
        else:
            coffea_console.print(f'--> {alias} FAILED')

        if valid==False:
            # all faild force AAA anyways
            nanofile = aliases[-1] + pfn
    return nanofile

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--analysis',  type=str, default='inc-WZ', help="Processor name to apply to datasets, and parent folder for config files")
    parser.add_argument('--jobNum' ,   type=int, default=1     , help="")
    parser.add_argument('--era'    ,   type=str, default="2018", help="")
    parser.add_argument('--isMC'   ,   type=int, default=1     , help="")
    parser.add_argument('--infile' ,   type=str, default=None  , help="input root file")
    parser.add_argument('--dataset',   type=str, default=None  , help="dataset name. need to specify if file is not in EOS")
    parser.add_argument('--runperiod', type=str, default=None)
    parser.add_argument('--executor' , type=str, default="FuturesExecutor", help="Executor to use, one of IterativeExecutor (good for debugging), FuturesExecutor (multithreaded), or other coffea option")
    parser.add_argument('--copyInput', action='store_true'     , help="xrdcp a file to the worker node before executing the coffea processor on it")
    parser.add_argument('--maxChunks', '--maxchunks', type=int, default= -1, help="limit number of chunks per-file to this number at most, default '-1' to process all")

    options = parser.parse_args()
    split_args = options.infile.split('/')
    auto_isMC = None  #if neither data or mc tag is found, keep as None
    auto_dataset = None
    auto_runperiod = ""
    if "NANOAODSIM" in split_args:
        auto_isMC = True
    elif "NANOAOD" in split_args:
        auto_isMC = False
    else:
        pass
    if auto_isMC is not None:
        assert ((options.isMC==1) == auto_isMC), f"auto MC detection is not consistent with isMC command line option: (isMC==1)={options.isMC==1} :: auto_isMC={auto_isMC}"
        try:
            tier_index = split_args.index("NANOAODSIM" if auto_isMC else "NANOAOD")
            auto_dataset = split_args[tier_index - 1]
            auto_runperiod = split_args[tier_index - 2].replace(f"Run{options.era}", "")
        except ValueError as ve:
            coffea_console.print("couldn't auto-parse dataset and runperiod from filename:)")
            coffea_console.print(ve)


    if options.dataset is None:
        options.dataset = auto_dataset
    executor = None
    if options.executor == "FuturesExecutor":
        executor = processor.FuturesExecutor(workers=8,)
    elif options.executor == "IterativeExecutor":
        executor = processor.IterativeExecutor()
    elif options.executor == "DaskExecutor":
        executor = processor.DaskExecutor()
    else:
        raise ValueError(f"Invalid Executor option {options.executor}")

    era=options.era
    is_data = not options.isMC

    failed = True
    ixrd = 0
    local_file_name = None
    aliases = [
        "root://cms-xrd-global.cern.ch/",
        "root://eoscms.cern.ch/",
        "root://llrxrd-redir.in2p3.fr/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global01.cern.ch/", 
        "root://cms-xrd-global02.cern.ch/",
        "root://cmsxrootd.fnal.gov/",
        "root://xrootd-cms-redir-int.cr.cnaf.infn.it/",
        "root://xrootd-redic.pi.infn.it/"
    ]
    while failed:
        try:
            file_name = options.infile
            if '/store/' in options.infile:
                if options.infile.startswith("root://"):
                    pass
                else:
                    file_name = aliases[ixrd] + options.infile
            else:
                coffea_console.print(f"Skipping redirector prepending for file {file_name}")

            if options.copyInput:
                if local_file_name is None and file_name.startswith("root://"):
                    try:
                        split_name = file_name.split("//")
                        local_file_name = str(tempfile.gettempdir()) + "/" + split_name[-1]
                        deepest_name = local_file_name.split("/")[-1]
                        local_file_nested_dir = local_file_name.replace(deepest_name, "")
                        if not os.path.isdir(local_file_nested_dir):
                            coffea_console.print(f"making directory... {local_file_nested_dir}")
                            os.makedirs(local_file_nested_dir, exist_ok=True)
                        if not os.path.isfile(local_file_name):
                            coffea_console.print(f"xrdcp file {file_name} {local_file_name}")
                            os.system(f"xrdcp {file_name} {local_file_name}")
                        if not os.path.isfile(local_file_name):
                            raise RuntimeError(f"Failed to download the file locally for processing: {file_name} -> {local_file_name}")
                    except Exception as le:
                        local_file_name = None
                        coffea_console.print(le)
                else:
                    if local_file_name:
                        coffea_console.print(f"File loaded to local directory: {local_file_name} (existence-test: {os.path.isfile(local_file_name)}")
                    else:
                        coffea_console.print(f"local_file_name not set ({local_file_name}), probably due to file_name ({file_name}) not indicating an xrdcp-able path by starting with root://")


                file_name = aliases[ixrd] + options.infile
            else:
                file_name = options.infile 

            metadata = {
                'era': era,
                'is_data': is_data
            }
            runs_files = {local_file_name: "Runs"} if local_file_name else {file_name: "Runs"}
            runs_samples ={
                options.dataset:{
                    'files': runs_files,
                    'metadata': metadata
                }
            }
            events_files = {fn: "Events" for fn in runs_files.keys()}
            events_samples ={
                options.dataset:{
                    'files': events_files,
                    'metadata': metadata
                }
            }
            sumw_runner = processor.Runner(
                executor=executor,
                schema=BaseSchema,
                format="root",
                savemetrics=True,
            )
            sumw_out, sumw_metrics = sumw_runner(
                runs_samples,
                processor_instance=coffea_sumw(),
            )
            
            ewk_flag = None
            if "ZZTo" in options.infile and "GluGluTo" not in options.infile and "ZZJJ" not in options.infile:
                ewk_flag= 'ZZ'
            if "WZTo" in options.infile and "GluGluTo" not in options.infile:
                ewk_flag = 'WZ'

            # extarct the run period
            if is_data:
                if 'Run20' in options.infile:
                    # options.runperiod = file_name.split('/store/data/')[1].split('/')[0].replace(f'Run{options.era}','')
                    options.runperiod = auto_runperiod.replace(f'Run{options.era}','')
            else:
                options.runperiod = ''

            coffea_console.print(
                f"""---------------------------
                -- options   = {options}
                -- analysis  = {options.analysis}
                -- isMC      = {options.isMC}
                -- jobNum    = {options.jobNum}
                -- era       = {options.era}
                -- infile    = {options.infile}
                --> {list(events_files.keys())[0]}
                -- dataset   = {options.dataset}
                -- period    = {options.runperiod}
                -- executor  = {options.executor}
                -- copyInput = {options.copyInput}
                -- maxChunks = {options.maxChunks if options.maxChunks > 0 else "None"}

                ---------------------------"""
            )
            if options.maxChunks > 0:
                coffea_console.print("WARNING: maxChunks will limit processing of MC or data, "
                      "for MC the normalization will not be stored for the number of events processed and the scaling will be incorrect, ",
                      "for data there will similarly not be an appropriate scaling of MC to match the processed luminosity in data."
                      )
            if options.analysis in ["inc-WZ"]:
                from qawa.process.wztau2lnu_inclusive import wzinclusive_processor
                coffea_console.print(" --- wztau2lnu_inclusive main code processor ... ")
                proc_configured = wzinclusive_processor(
                    era=options.era,
                    ewk_process_name=ewk_flag,
                    run_period=options.runperiod if is_data else ''
                )
            elif options.analysis in ["inc-WZ-Fxsec"]:
                from qawa.process.Fxsec import wzinclusive_processor # Fiducial XSec test processor for inc-WZ
                coffea_console.print(" --- wztau2lnu_inclusive FV Xsec processor ... ")
                proc_configured = wzinclusive_processor(
                    era=options.era,
                    ewk_process_name=ewk_flag,
                    run_period=options.runperiod if is_data else ''
                )
            elif options.analysis in ["trig-eff"]:
                coffea_console.print(" --- wztau2lnu_inclusive trigger efficiency processor ... ")
                from qawa.process.trig_eff import trig_processor
                proc_configured = trig_processor(
                    isMC=options.isMC,
                    era=options.era)
            else:
                raise NotImplementedError(f"{options.analysis} does not have hooks for loading a processor, please update the code to point appropriately to it, along with any necessary init configuration options.")

            # coffea_console.print(" --- wztau2lnu_inclusive processor ... ")
            events_runner = processor.Runner(
                executor=executor,
                schema=NanoAODSchema,
                chunksize=100000,
                maxchunks = options.maxChunks if options.maxChunks > 0 else None,
                format="root",
                savemetrics=True
            )
            events_out, event_metrics = events_runner(
                events_samples,
                processor_instance=proc_configured,
            )
            bh_output = {}
            for key, content in events_out.items():
                bh_output[key] = {
                    "hist": content,
                    "sumw": sumw_out[key],
            }
            with gzip.open("histogram_%s.pkl.gz" % str(options.jobNum), "wb") as f:
                pickle.dump(bh_output, f)
            failed=False
        except Exception as err:
            coffea_console.print(f"[WARNING] {aliases[ixrd]} failed with the following error : ")
            coffea_console.print(f"Unexpected {err=}, {type(err)=}")
            coffea_console.print("printing Exception err:")
            coffea_console.print(err)
            # print(traceback.format_exc())
            coffea_console.print("printing traceback.print_exc():")
            traceback.print_exc()
            coffea_console.print("-------------------------------------------")
            failed=True
            ixrd += 1

if __name__ == "__main__":
    main()
