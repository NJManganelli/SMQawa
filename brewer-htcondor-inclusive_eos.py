import os
import argparse
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
export EOSOUTDIR={eosoutdir}

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
echo "----- EOSOUTDIR          : $EOSOUTDIR"
ls -lthr

echo "----- processing the files : "

$INSTALL_LOC_EXTERNAL/.env/bin/python3 brewer-remote-inclusive.py --jobNum=$1 --isMC={ismc} --era={era} --analysis={analysis} --zzdd={zzdd} --infile=$2 --executor={executor} --copyInput {split_by_charge} {split_FV} {pol}

echo "----- directory after running :"
ls -lthr
if [ ! -f "histogram_$1.pkl.gz" ]; then
  echo "No output histogram pickle file found";
  exit 1;
fi

echo "----- copying the output histogram to EOS :"
echo "xrdcp -f histogram_$1.pkl.gz root://eosuser.cern.ch/$EOSOUTDIR/histogram_$1.pkl.gz"
xrdcp -f -p histogram_$1.pkl.gz "root://eosuser.cern.ch/$EOSOUTDIR/histogram_$1.pkl.gz"
XRDEXIT=$?
if [ $XRDEXIT -ne 0 ]; then
  echo "Failed to xrdcp the output histogram to EOS (exit $XRDEXIT)";
  exit $XRDEXIT;
fi
# remove the local copy so condor does not transfer it back to the submit dir
rm -f histogram_$1.pkl.gz
echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 10000000
request_memory        = {memory}

executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobfn)
# use_x509userproxy     = True
transfer_input_files  = {transfer_file}
should_transfer_files = YES
WhenToTransferOutput  = ON_EXIT_OR_EVICT
initialdir            = {jobdir}

# ---- everything condor produces goes to EOS ----
# stdout/stderr are shipped to EOS by the xrdcp transfer plugin; the histogram
# is xrdcp'ed by script.sh into {eosoutdir}. Only the (tiny) condor event log
# below stays on AFS: HTCondor requires file locking for it, which EOS does
# not support, so 'log' CANNOT point to EOS.
output_destination    = root://eosuser.cern.ch/{eosoutdir}/logs/
MY.XRDCP_CREATE_DIR   = True
output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log

on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
max_retries           = 3
requirements          = Machine =!= LastRemoteHost
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/{coffea_image}"
+JobFlavour           = "{queue}"

queue jobfn from {jobdir}/inputfiles.dat
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-a"   , "--analysis", type=str, default="inc-WZ"       , help="Analysis to run", required=True)
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt"       , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="atakour"        , help="production tag", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1                , help="")
    parser.add_argument("--zzdd"           , type=str, default="onlySR"         , help="For vbs-ZZ and/or inc-ZZ analyses DataDriven",
                        choices=["onlySR", "DYSR", "MC"])
    parser.add_argument("--split_by_charge", action="store_true", help="split templates by tau charge")
    parser.add_argument("--split_FV", action='store_true'          , help="split signal into IFV and OFV")
    parser.add_argument("--pol", action='store_true'          , help="Fill the histogram with polarization weights")
    parser.add_argument("-q"   , "--queue" , type=str, default="longlunch"      , help="")
    parser.add_argument("-e"   , "--era"   , type=str, default="2018"           , help="")
    parser.add_argument("-f"   , "--force" , action="store_true"                , help="recreate files and jobs")
    parser.add_argument("-s"   , "--submit", action="store_true"                , help="submit only")
    parser.add_argument("-dry" , "--dryrun", action="store_true"                , help="running without submission")
    parser.add_argument("--redo-proxy"     , action="store_true"                , help="redo the voms proxy")
    parser.add_argument("-ex", "--executor", type=str, default="FuturesExecutor", help="coffea executor to use",
                        choices=["FuturesExecutor","IterativeExecutor","DaskExecutor"])
    parser.add_argument("-m", "--memory", type=int, default=8000,
                        help="request_memory in MB per job (default 8000; peak coffea RSS ~6 GB)")
    parser.add_argument("--eosdir", type=str, default=None,
                        help="EOS base directory for the output histograms "
                             "(default: /eos/user/<u>/<user>/WZtotau2lnu). "
                             "Outputs land in <eosdir>/<tag>/<era>/<sample>/histogram_<N>.pkl.gz")
    options = parser.parse_args()

    # Making sure that the proxy is good
    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    user_name  = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    tag = options.tag
    # ---- EOS output base: <eosdir>/<tag>/<era>/<sample>/ ----
    if options.eosdir is None:
        options.eosdir = f"/eos/user/{user_name[0]}/{user_name}/WZtotau2lnu"
    eosbase = os.path.join(options.eosdir, "{tag}", "{era}", "{sample}")
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


    with open(options.input, 'r') as stream:
        captured_env = os.environ.copy()
        if "bash" in captured_env['SHELL']:
            to_source = os.path.join(captured_env['INSTALL_LOC'], ".bashrc")
        elif "zsh" in captured_env['SHELL']:
            to_source = os.path.join(captured_env['INSTALL_LOC'], ".zshrc")
        else:
            raise NotImplementedError("neither bash or zsh detected in the shell env variable, something has gone wrong; contents=", captured_env['SHELL'])
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            split_sample = sample.split("/")
            if len(split_sample) <= 1: continue
            auto_isMC = 1 * (split_sample[-1] == "NANOAODSIM")
            # sample_name = sample.split("/")[1] if options.isMC else '_'.join(sample.split("/")[1:3])
            sample_name = sample.split("/")[1] if auto_isMC else '_'.join(sample.split("/")[1:3])
            sample_name = sample_name.replace("*", "")
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

            # ---- per-sample EOS output directory ----
            eosoutdir = eosbase.format(tag=options.tag, era=options.era, sample=sample_name)
            os.system("mkdir -p {}".format(eosoutdir))
            os.system("mkdir -p {}".format(os.path.join(eosoutdir, "logs")))
            logging.info("-- EOS output dir : " + eosoutdir)

            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                script = script_TEMPLATE.format(
                    proxy=proxy_copy,
                    analysis=options.analysis,
                    ismc=auto_isMC,
                    era=options.era,
                    zzdd=options.zzdd,
                    split_by_charge = "--split_by_charge" if options.split_by_charge else "",
                    split_FV = "--split_FV" if options.split_FV else "",
                    pol = "--pol" if options.pol else "",
                    qawa_version=qawa_version,
                    coffea_image=coffea_image,
                    full_image=full_image,
                    install_loc_external=install_loc_external,
                    executor=options.executor,
                    eosoutdir=eosoutdir,
                )
                scriptfile.write(script)
                scriptfile.close()

            with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
                condor = condor_TEMPLATE.format(
                    transfer_file= ",".join([
                        brewer_loc_external,
                    ]),
                    jobdir=str(jobs_dir_external), #use the external path so call_host condor_submit can find it
                    queue=options.queue,
                    coffea_image=coffea_image,
                    eosoutdir=eosoutdir,
                    memory=options.memory,
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

if __name__ == "__main__":
    main()