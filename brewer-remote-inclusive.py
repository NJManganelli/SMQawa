from coffea import processor
from coffea import nanoevents
from coffea.nanoevents import NanoAODSchema, BaseSchema
from coffea.util import coffea_console
from qawa.process.coffea_sumw import coffea_sumw
from qawa.common import nanoaod_version
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
        "root://eoscms.cern.ch/",
        "root://cmsxrootd.fnal.gov/"
        "root://cms-xrd-global.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global.cern.ch/",
        
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


def read_input_manifest(infile):
    """Expand --infile into the list of ROOT paths to process.

    A .txt argument is a manifest: one path per line, with blank lines and
    '#' comments ignored. Anything else is a single ROOT file. Either way the
    caller gets a list, so there is exactly one downstream code path.
    """
    if not infile:
        raise ValueError("--infile is required")
    if not infile.endswith(".txt"):
        return [infile]
    with open(infile) as manifest:
        paths = [
            line.strip() for line in manifest
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not paths:
        raise ValueError(f"--infile manifest {infile} listed no paths")
    coffea_console.print(f"--infile is a manifest: {len(paths)} files from {infile}")
    return paths


def resolve_input_path(path, alias, copy_input):
    """Map one input path to the name coffea should open.

    A remote /store/ path gets the current redirector prepended; with
    --copyInput an xrdcp-able path is first copied to a local temporary file
    and that copy is opened instead. Falls back to the remote name if the copy
    fails, so a bad local disk degrades to streaming rather than killing the
    job.

    A file that exists on local disk is always opened in place, even when it
    sits under a mirrored /store/ tree -- testing os.path.isfile rather than
    the '/store/' substring is what keeps local mirrors and site POSIX mounts
    working, since their paths contain /store/ but must not be redirected.
    """
    remote_name = path
    if path.startswith("root://") or path.startswith("file:"):
        pass
    elif os.path.isfile(path):
        coffea_console.print(f"Opening local file in place: {path}")
        return path
    elif '/store/' in path:
        remote_name = alias + path
    else:
        coffea_console.print(f"Skipping redirector prepending for file {remote_name}")

    if not copy_input:
        return remote_name
    if not remote_name.startswith("root://"):
        coffea_console.print(
            f"--copyInput requested but {remote_name} does not start with root:// "
            "and so is not xrdcp-able; opening it in place"
        )
        return remote_name
    try:
        local_name = str(tempfile.gettempdir()) + "/" + remote_name.split("//")[-1]
        local_dir = os.path.dirname(local_name)
        if not os.path.isdir(local_dir):
            coffea_console.print(f"making directory... {local_dir}")
            os.makedirs(local_dir, exist_ok=True)
        if not os.path.isfile(local_name):
            coffea_console.print(f"xrdcp file {remote_name} {local_name}")
            os.system(f"xrdcp {remote_name} {local_name}")
        if not os.path.isfile(local_name):
            raise RuntimeError(
                f"Failed to download the file locally for processing: {remote_name} -> {local_name}"
            )
        coffea_console.print(f"File loaded to local directory: {local_name}")
        return local_name
    except Exception as le:
        coffea_console.print(le)
        coffea_console.print(f"falling back to streaming {remote_name}")
        return remote_name


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--analysis' , type=str, default='inc-WZ'         , help="Processor name to apply to datasets, and parent folder for config files")
    parser.add_argument('--jobNum'   , type=int, default=1                , help="")
    parser.add_argument('--era'      , type=str, default="2018"           , help="")
    parser.add_argument('--isMC'     , type=int, default=1                , help="")
    parser.add_argument('--infile'   , type=str, default=None             , help="input root file, OR a .txt manifest listing one root path per line. A manifest is processed as a SINGLE coffea fileset, so chunks parallelise across every file in it")
    parser.add_argument('--dataset'  , type=str, default=None             , help="dataset name. need to specify if file is not in EOS")
    parser.add_argument('--runperiod', type=str, default=None             , help="run period, can be auto-parse if full dataset paths are utilized")
    parser.add_argument('--version'  , type=str, default=None             , help="NanoAOD version (e.g. v15) override, for files whose path does not encode it -- local files, private production, FlashSim output")
    parser.add_argument("--zzdd"     , type=str, default="onlySR"         , help="For vbs-ZZ and/or inc-ZZ analyses DataDriven",
                        choices=["onlySR", "DYSR", "MC"])
    parser.add_argument("--split_by_charge", action='store_true'          , help="split templates by tau charge")
    parser.add_argument("--split_FV", action='store_true'          , help="split signal into IFV and OFV")
    parser.add_argument("--pol", action='store_true'          , help="Fill the histogram with polarization weights")
    parser.add_argument('--executor' , type=str, default="FuturesExecutor", help="Executor to use, one of IterativeExecutor (good for debugging), FuturesExecutor (multithreaded), or other coffea option")
    parser.add_argument('--copyInput', action='store_true'     , help="xrdcp a file to the worker node before executing the coffea processor on it")
    parser.add_argument('--maxChunks', '--maxchunks', type=int, default= -1, help="limit number of chunks per-file to this number at most, default '-1' to process all")

    options = parser.parse_args()
    # Every file in a manifest belongs to the same dataset, so the first entry
    # is the reference path for all of the auto-parsing below.
    input_paths = read_input_manifest(options.infile)
    ref_infile = input_paths[0]
    split_args = ref_infile.split('/')
    auto_isMC = None  #if neither data or mc tag is found, keep as None
    auto_dataset = None
    auto_runperiod = ""
    # An explicit --version wins: nanoaod_version() parses the tier out of a
    # /store/ path, which private and FlashSim outputs do not have.
    auto_ver = options.version or nanoaod_version(ref_infile)
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
            # Resolve every input path through the SAME logic, whether --infile
            # named one root file or a .txt manifest of them. aliases[ixrd] is the
            # redirector currently being tried, so a failure anywhere below retries
            # the whole sample against the next one.
            input_files = [
                resolve_input_path(path, aliases[ixrd], options.copyInput)
                for path in input_paths
            ]

            if options.maxChunks > 0:
                coffea_console.print("WARNING: maxChunks will limit processing of MC or data, "
                      "for MC the normalization will not be stored for the number of events processed and the scaling will be incorrect, ",
                      "for data there will similarly not be an appropriate scaling of MC to match the processed luminosity in data."
                      )

            if is_data:
                if 'Run20' in ref_infile:
                    options.runperiod = auto_runperiod.replace(f'Run{options.era}','')
            else:
                options.runperiod = ''


            ewk_flag = None
            if "ZZTo" in ref_infile and "GluGluTo" not in ref_infile and "ZZJJ" not in ref_infile:
                ewk_flag= 'ZZ'
            if "WZTo" in ref_infile and "GluGluTo" not in ref_infile:
                ewk_flag = 'WZ'
            dy_flag = False
            if "DYJetsToLL" in ref_infile:
                dy_flag = True

            # extarct the run period
            
            
            
            is_signal = None
            if options.analysis in ["inc-WZ"]:
                from qawa.process.wztau2lnu_inclusive import wzinclusive_processor
                coffea_console.print(" --- wztau2lnu_inclusive main code processor ... ")
                proc_configured = wzinclusive_processor(
                    era=options.era,
                    split_by_charge = options.split_by_charge,
                    split_FV = options.split_FV,
                    pol = options.pol,
                    ewk_process_name=ewk_flag,
                    run_period=options.runperiod if is_data else '',
                    version=auto_ver,
                )
                is_signal = True if ('WZto3LNu' in options.dataset or 'WZTo3LNu' in options.dataset) else False
            elif options.analysis in ["inc-WZ-Fxsec"]:
                from qawa.process.Fxsec import wzinclusive_processor 
                coffea_console.print(" --- wztau2lnu_inclusive FV Xsec processor ... ")
                proc_configured = wzinclusive_processor(
                    era=options.era,
                    ewk_process_name=ewk_flag,
                    run_period=options.runperiod if is_data else '',
                    version=auto_ver,
                )
            elif options.analysis in ["trig-eff"]:
                coffea_console.print(" --- wztau2lnu_inclusive trigger efficiency processor ... ")
                from qawa.process.trig_eff import trig_processor
                proc_configured = trig_processor(
                    isMC=options.isMC,
                    era=options.era)
            elif options.analysis in ["inc-ZZ"]:
                from qawa.process.zz2l2nu_inclusive import zzinc_processor
                coffea_console.print(" --- zz2l2nu_inclusive main code processor ... ")
                proc_configured = zzinc_processor(
                    era=options.era,
                    isDY=dy_flag,
                    dd = options.zzdd,
                    ewk_process_name=ewk_flag,
                    # dump_gnn_array=options.dumpgnn,
                    run_period=options.runperiod if is_data else ''
                )
            elif options.analysis in ["vbs-ZZ"]:
                from qawa.process.zz2l2nu_vbs import zzinc_processor
                coffea_console.print(" --- zz2l2nu_vbs main code processor ... ")
                proc_configured = zzinc_processor(
                    era=options.era,
                    isDY=dy_flag,
                    dd = options.zzdd,
                    ewk_process_name=ewk_flag,
                    # dump_gnn_array=options.dumpgnn,
                    run_period=options.runperiod if is_data else ''
                )
            else:
                raise NotImplementedError(f"{options.analysis} does not have hooks for loading a processor, please update the code to point appropriately to it, along with any necessary init configuration options.")

            # FlashSim samples are flagged by dataset name. Two things follow:
            # the processor needs the field shim (metadata), and uproot needs
            # the Float16 read patch installed BEFORE the Runner opens any file
            # -- 332 of 671 branches in a FlashSim NanoAOD are otherwise
            # unreadable. The patch is opt-in so non-FlashSim jobs keep stock
            # uproot behaviour and stock uproot errors.
            is_flashsim = 'flashsim' in options.dataset.lower()
            if is_flashsim:
                from qawa import uproot_float16_patch
                if uproot_float16_patch.enable_for_workers():
                    coffea_console.print(
                        "[yellow]FlashSim sample: uproot jagged-Float16 read patch enabled[/yellow]"
                    )

            metadata = {
                'era': era,
                'is_data': is_data,
                'is_signal': is_signal,
                'is_flashsim': is_flashsim,
            }
            runs_files = {fn: "Runs" for fn in input_files}
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

            coffea_console.print(
                f"""---------------------------
                -- options   = {options}
                -- analysis  = {options.analysis}
                -- isMC      = {options.isMC}
                -- split_by_charge = {options.split_by_charge}
                -- split_FV = {options.split_FV}
                -- pol      = {options.pol}
                -- jobNum    = {options.jobNum}
                -- era       = {options.era}
                -- infile    = {options.infile}
                -- nfiles    = {len(input_files)}
                --> {list(events_files.keys())[0]}
                -- dataset   = {options.dataset}
                -- period    = {options.runperiod}
                -- version   = {auto_ver}
                -- executor  = {options.executor}
                -- copyInput = {options.copyInput}
                -- maxChunks = {options.maxChunks if options.maxChunks > 0 else "None"}
                -- zzdd    = {options.zzdd if options.analysis in ['vbs-ZZ', 'inc-ZZ'] else 'N/A'}

                ---------------------------"""
            )

            # FlashSim needs a schema that tolerates its orphan counter
            # branches (nFakeJets / nSVFromPU have nothing behind them, and
            # stock NanoAODSchema dies building a collection for them).
            events_schema = NanoAODSchema
            if is_flashsim:
                from qawa.flashsim_shim import flashsim_schema
                events_schema = flashsim_schema()

            events_runner = processor.Runner(
                executor=executor,
                schema=events_schema,
                chunksize=50000,
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
            if ixrd >= len(aliases):
                raise RuntimeError(
                    f"all {len(aliases)} redirectors failed for --infile {options.infile}"
                ) from err

if __name__ == "__main__":
    main()
