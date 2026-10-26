import os
import argparse
import json
import re
import shlex
import logging
import subprocess
import shutil
import time
import importlib.metadata

logging.basicConfig(level=logging.DEBUG)

#qawa_version = importlib.metadata.version('qawa')
qawa_version = '0.0.7'


script_TEMPLATE = """#!/bin/bash
export X509_USER_PROXY={proxy}
export XRD_REQUESTTIMEOUT=6400
export XRD_REDIRECTLIMIT=64
export INSTALL_LOC_EXTERNAL={install_loc_external}
export COFFEA_IMAGE={coffea_image}
export FULL_IMAGE={full_image}

voms-proxy-info -all
voms-proxy-info -all -file {proxy}

echo "----- COFFEA_IMAGE :"
echo COFFEA_IMAGE $COFFEA_IMAGE
echo FULL_IMAGE $FULL_IMAGE

echo "----- Sourcing virtual environment :"
echo source $INSTALL_LOC_EXTERNAL/.env/bin/activate
source $INSTALL_LOC_EXTERNAL/.env/bin/activate
echo "which python3"
which python3
echo '$SHELL'
echo $SHELL
echo '$PYTHONPATH'
echo $PYTHONPATH
echo '$PYTHON3PATH'
echo $PYTHON3PATH
echo '$PYTHONHOME'
echo $PYTHONHOME
echo '$PATH'
echo $PATH
echo awkward, uproot, coffea, qawa versions:
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show awkward
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show uproot
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show coffea
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show qawa
echo import and print __file__ for coffea
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -c "import coffea; print(coffea.__version__); print(coffea.__file__)"
echo import and print __file__ for qawa
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -c "import qawa; print(qawa.__file__)"


echo "----- JOB STARTS @" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- X509_USER_PROXY    : $X509_USER_PROXY"
echo "----- XRD_REDIRECTLIMIT  : $XRD_REDIRECTLIMIT"
echo "----- XRD_REQUESTTIMEOUT : $XRD_REQUESTTIMEOUT"
ls -lthr

echo "----- processing the files : "

$INSTALL_LOC_EXTERNAL/.env/bin/python3 brewer-remote-inclusive.py --jobNum=$1 --isMC={ismc} --era={era} --analysis={analysis} --zzdd={zzdd} --infile=$2 --executor={executor} --copyInput {split_by_charge}{dd}{histserv}

echo "----- directory after running :"
ls -lthr
if [ ! -f "{done_file}" ]; then
  echo "{done_message}";
  exit 1;
fi
echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 10000000

executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobfn)
# use_x509userproxy     = True
transfer_input_files  = {transfer_file}
{transfer_output}
should_transfer_files = YES
WhenToTransferOutput  = ON_EXIT_OR_EVICT
initialdir            = {jobdir}

output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log

on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
max_retries           = 3{histserv_hold}
requirements          = Machine =!= LastRemoteHost
# MY.XRDCP_CREATE_DIR   = True
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/{coffea_image}"
+JobFlavour           = "{queue}"

queue jobfn from {jobdir}/inputfiles.dat
"""

DEFAULT_TRANSFER_OUTPUT = "# transfer_output_files = histogram_$(ProcId).pkl.gz "
DD_DIRECTORY_PATTERN = re.compile(r"_dd-([0-9a-f]{12})$")


def sample_name_of(sample, dd_label=None):
    auto_isMC = 1 * (sample.split("/")[-1] == "NANOAODSIM")
    # sample_name = sample.split("/")[1] if options.isMC else '_'.join(sample.split("/")[1:3])
    sample_name = sample.split("/")[1] if auto_isMC else '_'.join(sample.split("/")[1:3])
    sample_name = sample_name.replace("*", "")
    if dd_label is not None:
        sample_name += f"_dd-{dd_label}"
    return auto_isMC, sample_name


def check_pass_directories(tag, era, dd_label):
    submission = "default" if dd_label is None else f"dd-{dd_label}"
    for name in sorted(os.listdir(".")):
        if not name.startswith("jobs_") or f"_{era}_" not in name[len("jobs_"):] or not os.path.isdir(name):
            continue
        existing_tag, existing_sample = name[len("jobs_"):].split(f"_{era}_", 1)
        match = DD_DIRECTORY_PATTERN.search(existing_sample)
        existing = f"dd-{match.group(1)}" if match else "default"
        if (existing_tag in tag or tag in existing_tag) and existing != submission:
            raise SystemExit(f"{name} holds the {existing} pass of tag {existing_tag}, and this submission is the {submission} pass "
                             f"of tag {tag}: each pass needs a tag of its own, because hist-merger.py --legacy matches tags as substrings")


def check_registry(registry, samples, options, dd_label=None):
    from qawa import production_keys
    wanted = {"era": options.era, "split_by_charge": options.split_by_charge, "pol": False, "split_FV": False}
    wrong = {name: registry["options"][name] for name, value in wanted.items() if registry["options"][name] != value}
    if wrong:
        raise SystemExit(f"registry options {wrong} do not match this submission {wanted}")
    variations = {"pol": False, "split_by_charge": options.split_by_charge, "split_FV": False, "dd": dd_label}
    dd_sha256 = None if dd_label is None else production_keys.file_sha256(options.dd_file)
    problems = []
    names = {}
    keys_of = {}
    for sample in samples:
        if '#' in sample or len(sample.split("/")) <= 1:
            continue
        if '*' in sample:
            raise SystemExit(f"C4 needs one DAS dataset per line: expand the wildcard in {sample}")
        try:
            production_keys.das_parts(sample)
        except ValueError as error:
            problems.append(str(error))
            continue
        auto_isMC, sample_name = sample_name_of(sample, dd_label)
        if sample_name in names:
            raise SystemExit(f"{names[sample_name]} and {sample} share the job directory of {sample_name}: "
                             "in C4 mode one job directory must hold one registry entry")
        names[sample_name] = sample
        keys = [key for key, entry in registry["samples"].items()
                if entry["das"] == sample and all(entry["variations"][name] == value for name, value in variations.items())]
        if len(keys) != 1:
            problems.append(f"{sample}: {len(keys)} registered entries with {variations}")
            continue
        keys_of[sample] = keys[0]
        kind = registry["samples"][keys[0]]["kind"]
        if registry["samples"][keys[0]].get("dd_sha256") != dd_sha256:
            problems.append(f"{sample}: the registered dd sha256 differs from {options.dd_file}")
        wz_signal = 'WZto3LNu' in sample.split("/")[1] or 'WZTo3LNu' in sample.split("/")[1]
        kind_matches = {"data": not auto_isMC, "signal": auto_isMC and wz_signal, "mc": auto_isMC and not wz_signal,
                        "eft": auto_isMC}
        if not kind_matches[kind]:
            problems.append(f"{sample}: registered kind {kind} contradicts the sample")
    if problems:
        raise SystemExit("C4 registry check failed: " + "; ".join(problems))
    return keys_of


def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-a"   , "--analysis", type=str, default="inc-WZ"       , help="Analysis to run", required=True)
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt"       , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="atakour"        , help="production tag", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1                , help="")
    parser.add_argument("--zzdd"           , type=str, default="onlySR"         , help="For vbs-ZZ and/or inc-ZZ analyses DataDriven",
                        choices=["onlySR", "DYSR", "MC"])
    parser.add_argument("--split_by_charge", action="store_true", help="split templates by tau charge")
    parser.add_argument("-q"   , "--queue" , type=str, default="longlunch"      , help="")
    parser.add_argument("-e"   , "--era"   , type=str, default="2018"           , help="")
    parser.add_argument("-f"   , "--force" , action="store_true"                , help="recreate files and jobs")
    parser.add_argument("-s"   , "--submit", action="store_true"                , help="submit only")
    parser.add_argument("-dry" , "--dryrun", action="store_true"                , help="running without submission")
    parser.add_argument("--redo-proxy"     , action="store_true"                , help="redo the voms proxy")
    parser.add_argument("-ex", "--executor", type=str, default="FuturesExecutor", help="coffea executor to use",
                        choices=["FuturesExecutor","IterativeExecutor","DaskExecutor"])
    parser.add_argument("--histserv-registry", type=str, default=None,
                        help="C4: fill the central histserv server of this registry JSON (python -m qawa.histserv_output register) instead of job pickles")
    parser.add_argument("--dd-file", type=str, default=None,
                        help="data-driven estimate (correctionlib JSON) replacing the era default, for re-processing data with updated estimates (inc-WZ data only)")
    options = parser.parse_args()

    with open(options.input, 'r') as stream:
        samples = stream.read().split('\n')
    dd_label = None
    if options.dd_file is not None:
        from qawa import datadriven, production_keys
        mc_lines = [sample for sample in samples if '#' not in sample and len(sample.split("/")) > 1 and sample_name_of(sample)[0]]
        if mc_lines:
            raise SystemExit(f"--dd-file re-processes data only (NANOAOD), got {mc_lines}")
        try:
            datadriven.check_dd_file(options.dd_file)
        except ValueError as error:
            raise SystemExit(str(error))
        dd_label = production_keys.dd_label(options.dd_file)
    check_pass_directories(options.tag, options.era, dd_label)
    histserv_registry = None
    if options.histserv_registry is not None:
        with open(options.histserv_registry, 'rb') as f:
            registry_bytes = f.read()
        histserv_registry = json.loads(registry_bytes)
        keys_of = check_registry(histserv_registry, samples, options, dd_label)
    collect_manifests = []

    # Making sure that the proxy is good
    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    user_name  = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    tag = options.tag
    # eosbase = f"/eos/user/{user_name[0]}/{user_name}/WZtotau2lnu/" + "{tag}/{sample}/"
    coffea_image = os.environ['COFFEA_IMAGE']
    full_image = os.environ['FULL_IMAGE']
    install_loc_external = os.environ['INSTALL_LOC_EXTERNAL']
    brewer_loc_external = os.path.join(os.environ['INSTALL_LOC_EXTERNAL'], "SMQawa", "brewer-remote-inclusive.py")

    regenerate_proxy = False
    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
        regenerate_proxy = True
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )
        print (lifetime)
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        logging.info("--- proxy lifetime is {} hours".format(lifetime))
        if lifetime < 10.0:
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system('voms-proxy-init -voms cms')
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        shutil.copyfile('/tmp/'+proxy_base,  proxy_copy)


    captured_env = os.environ.copy()
    if "bash" in captured_env['SHELL']:
        to_source = os.path.join(captured_env['INSTALL_LOC'], ".bashrc")
    elif "zsh" in captured_env['SHELL']:
        to_source = os.path.join(captured_env['INSTALL_LOC'], ".zshrc")
    else:
        raise NotImplementedError("neither bash or zsh detected in the shell env variable, something has gone wrong; contents=", captured_env['SHELL'])
    for sample in samples:
        if '#' in sample: continue
        split_sample = sample.split("/")
        if len(split_sample) <= 1: continue
        auto_isMC, sample_name = sample_name_of(sample, dd_label)
        jobs_dir = '_'.join(['jobs', options.tag, options.era, sample_name])
        jobs_dir_external = os.path.join(os.environ['INSTALL_LOC_EXTERNAL'], os.path.relpath(os.path.normpath(jobs_dir), os.environ['INSTALL_LOC']))
        print("jobs_dir:", jobs_dir, "\njobs_dir_external:", jobs_dir_external)
        logging.info("-- sample_name : " + sample)

        if os.path.isdir(jobs_dir):
            if not options.force:
                logging.error(" " + jobs_dir + " already exist !")
                continue
            else:
                logging.warning(" " + jobs_dir + " already exists, forcing its deletion!")
                shutil.rmtree(jobs_dir)
                os.mkdir(jobs_dir)
        else:
            os.mkdir(jobs_dir)

        if not options.submit:
            sample_files = []
            if '*' in sample:
                sample_with_ext = subprocess.check_output(
                    ['dasgoclient', '--query', f"dataset={sample}"]
                )
                print(" --- found these samples : ")
                print(sample_with_ext.decode('UTF-8'))
                print("and these are the files : ")
                for sample_ in sample_with_ext.decode("UTF-8").split("\n")[:-1]:
                    output_ = subprocess.check_output(
                        ['dasgoclient', '--query', f'file dataset={sample_}']
                    )
                    sample_files += list(filter(lambda x: x != '', output_.decode('UTF-8').split('\n')))
            else:
                output_ = subprocess.check_output(
                    ['dasgoclient','--query', f"file dataset={sample}"]
                )
                sample_files = list(filter(lambda x: x != '', output_.decode('UTF-8').split('\n')))

            time.sleep(1)
            with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                for fn in sample_files:
                    infiles.write(fn)
                    infiles.write('\n')
                infiles.close()
        time.sleep(2)
        histserv = ""
        histserv_hold = ""
        done_file = "histogram_$1.pkl.gz"
        done_message = "No output histogram pickle file found"
        transfer_output = DEFAULT_TRANSFER_OUTPUT
        transfer_files = [brewer_loc_external]
        dd = ""
        if dd_label is not None:
            shutil.copyfile(options.dd_file, os.path.join(jobs_dir, "dd_estimate.json"))
            os.chmod(os.path.join(jobs_dir, "dd_estimate.json"), 0o644)
            transfer_files.append(os.path.join(jobs_dir_external, "dd_estimate.json"))
            dd = " --dd-file=dd_estimate.json"
        if histserv_registry is not None:
            fd = os.open(os.path.join(jobs_dir, "histserv_registry.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(registry_bytes)
            transfer_files.append(os.path.join(jobs_dir_external, "histserv_registry.json"))
            code = histserv_registry["poison_exit_code"]
            max_chunks = f" --maxChunks={histserv_registry['max_chunks']}" if histserv_registry["max_chunks"] is not None else ""
            histserv = f" --histserv-registry=histserv_registry.json --das-name={sample}{max_chunks}\nrc=$?\nif [ $rc -eq {code} ]; then exit {code}; fi"
            histserv_hold = (f"\non_exit_hold          = (ExitBySignal == False) && (ExitCode == {code})"
                             "\non_exit_hold_reason   = \"C4 poison: do not release; re-register under a new tag and re-run\"")
            done_file = "histserv_done_$1.json"
            done_message = "No C4 done file found"
            transfer_output = "# transfer_output_files = histserv_done_$(ProcId).json"
            collect_manifests.append(keys_of[sample] + "=" + os.path.abspath(jobs_dir) + "/inputfiles.dat")
        # eosoutdir =  eosbase.format(tag=options.tag,sample=sample_name)
        # # crete a directory
        # os.system("mkdir -p {}".format(eosoutdir))

        with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
            script = script_TEMPLATE.format(
                proxy=proxy_copy,
                analysis=options.analysis,
                ismc=auto_isMC,
                era=options.era,
                zzdd=options.zzdd,
                split_by_charge = "--split_by_charge" if options.split_by_charge else "",
                dd=dd,
                qawa_version=qawa_version,
                coffea_image=coffea_image,
                full_image=full_image,
                install_loc_external=install_loc_external,
                executor=options.executor,
                histserv=histserv,
                done_file=done_file,
                done_message=done_message,
            )
            scriptfile.write(script)
            scriptfile.close()

        with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
            condor = condor_TEMPLATE.format(
                transfer_file= ",".join(transfer_files),
                jobdir=str(jobs_dir_external), #use the external path so call_host condor_submit can find it
                queue=options.queue,
                coffea_image=coffea_image,
                transfer_output=transfer_output,
                histserv_hold=histserv_hold,
            )
            condorfile.write(condor)
            condorfile.close()
        if options.dryrun:
            continue

        try:
            cmd = f"cd {captured_env['INSTALL_LOC']} && source {to_source} && condor_submit {os.path.join(jobs_dir_external, 'condor.sub')}"
            logging.info(f"condor command : {cmd}")
            htc = subprocess.Popen(
                cmd,
                shell      = True,
                executable = captured_env['SHELL'],
                env        = captured_env,
                stdin      = subprocess.PIPE,
                stdout     = subprocess.PIPE,
                stderr     = subprocess.PIPE,
                close_fds  =True
            )

            htc_out, htc_err = htc.communicate()
            exit_status = htc.returncode
            logging.info(f"condor submission status : {exit_status}")
            logging.info(f"condor communicate stdout : {htc_out}")
            logging.info(f"condor communicate stderr : {htc_err}")
        except Exception as e:
            print(f"HTCondor submission error: {e}")
    if histserv_registry is not None:
        logging.info("C4 collect, once no job is running, held or failed: python -m qawa.histserv_output collect --registry "
                     + shlex.quote(os.path.abspath(options.histserv_registry))
                     + "".join(" --manifest " + shlex.quote(manifest) for manifest in collect_manifests) + " --outdir <outdir>")

if __name__ == "__main__":
    main()
