import os
import argparse
import shutil
import logging
import subprocess
import rich
# from pandas.core.internals.array_manager import new_block
# from termcolor import colored
import importlib.metadata
qawa_version = "0.0.7"

logging.basicConfig(level=logging.INFO)

rerun_script_header = f"""#!/bin/bash
# cd /srv/
# python -m venv --without-pip --system-site-packages jobenv
# source jobenv/bin/activate
# python -m pip install scipy --upgrade --no-cache-dir
# python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

echo "... start job at" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- directory before running:"
ls -lthr
"""


resub_script_header = """#!/bin/bash
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
{command}

echo "----- directory after running :"
ls -lthr
if [ ! -f "histogram_{jobid}.pkl.gz" ]; then
  echo "No output histogram pickle file found";
  exit 1;
fi

echo "----- copying the output histogram to EOS :"
echo "xrdcp -f histogram_{jobid}.pkl.gz root://eosuser.cern.ch/$EOSOUTDIR/histogram_{jobid}.pkl.gz"
xrdcp -f -p histogram_{jobid}.pkl.gz "root://eosuser.cern.ch/$EOSOUTDIR/histogram_{jobid}.pkl.gz"
XRDEXIT=$?
if [ $XRDEXIT -ne 0 ]; then
  echo "Failed to xrdcp the output histogram to EOS (exit $XRDEXIT)";
  exit $XRDEXIT;
fi
rm -f histogram_{jobid}.pkl.gz
echo " ------ THE END (everyone dies !) ----- "
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-a"   , "--analysis", type=str, default="inc-WZ"       , help="Analysis to run", required=True)
    parser.add_argument("-i"   , "--input" , type=str, default="input"  , required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="algiers", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1        , help="")
    parser.add_argument("--split_by_charge", action="store_true", help="split templates by tau charge")
    parser.add_argument("--split_FV", action='store_true'          , help="split signal into IFV and OFV")
    parser.add_argument("--pol", action='store_true'          , help="Fill the histogram with polarization weights")
    parser.add_argument("-e"   , "--era"   , type=str, default="2018"   , help="")
    parser.add_argument("--runlocal", action="store_true")
    parser.add_argument("--resubmit", action="store_true", help="resubmit failed jobs")
    parser.add_argument("--dryrun"  , action="store_true")
    parser.add_argument('--executor' , type=str, default="FuturesExecutor", help="Executor to use, one of IterativeExecutor (good for debugging), FuturesExecutor (multithreaded), or other coffea option")
    parser.add_argument('--copyInput', action='store_true'     , help="xrdcp a file to the worker node before executing the coffea processor on it")
    parser.add_argument('--verbose'  , action='store_true'     , help="verbose output printing status of running, finished, and failed files per job")
    parser.add_argument("-m", "--memory", type=int, default=8000,
                        help="request_memory in MB for the resubmitted jobs (default 8000; "
                             "bump higher, e.g. --memory 12000, for jobs that keep hitting the cgroup limit)")
    parser.add_argument("--eosdir", type=str, default=None,
                        help="EOS base directory holding the output histograms "
                             "(must match the one used at submission; default: "
                             "/eos/user/<u>/<user>/WZtotau2lnu). Job completion is "
                             "checked in <eosdir>/<tag>/<era>/<sample>/histogram_<N>.pkl.gz")
    options = parser.parse_args()


    captured_env = os.environ.copy()
    if "bash" in captured_env['SHELL']:
        to_source = os.path.join(captured_env['INSTALL_LOC'], ".bashrc")
    elif "zsh" in captured_env['SHELL']:
        to_source = os.path.join(captured_env['INSTALL_LOC'], ".zshrc")
    else:
        raise NotImplementedError("neither bash or zsh detected in the shell env variable, something has gone wrong; contents=", captured_env['SHELL'])

    condor_stat_cmd = f"cd {captured_env['INSTALL_LOC']} && source {to_source} && condor_q -nobatch"
    logging.info(f"condor command : {condor_stat_cmd}")
    htc = subprocess.Popen(
        condor_stat_cmd,
        shell      = True,
        executable = captured_env['SHELL'],
        env        = captured_env,
        stdin      = subprocess.PIPE,
        stdout     = subprocess.PIPE,
        stderr     = subprocess.PIPE,
        close_fds  = True
    )
    condor_status, htc_err = htc.communicate()
    condor_status = str(condor_status) # need to convert bytes object to string
    exit_status = htc.returncode
    logging.info(f"condor q -nobatch status : {exit_status}")
    logging.info(f"condor q -nobatch stderr : {htc_err}")
    print("condor_status:", condor_status)

    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    user_name  = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    coffea_image = os.environ['COFFEA_IMAGE']
    full_image = os.environ['FULL_IMAGE']
    install_loc_external = os.environ['INSTALL_LOC_EXTERNAL']
    brewer_loc_external = os.path.join(os.environ['INSTALL_LOC_EXTERNAL'], "SMQawa", "brewer-remote-inclusive.py")

    # ---- EOS output base (must match the submitter) ----
    if options.eosdir is None:
        options.eosdir = f"/eos/user/{user_name[0]}/{user_name}/WZtotau2lnu"
    eosbase = os.path.join(options.eosdir, "{tag}", "{era}", "{sample}")

    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )    
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        logging.info("--- proxy lifetime is {} hours".format(lifetime))
        if lifetime < 10.0: # we want at least 10 hours
            logging.warning("--- proxy has expired !")


    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            split_sample = sample.split('/')
            if len(split_sample) <= 1: continue
            # per-sample MC/data auto-detection (same as the submitter), so a
            # mixed data+MC input list resolves the same jobs_* directories
            auto_isMC = 1 * (split_sample[-1] == "NANOAODSIM")
            sample_name = sample.split("/")[1] if auto_isMC else '_'.join(sample.split("/")[1:3])
            sample_name = sample_name.replace("*", "")
            jobs_dir = '_'.join(['jobs', options.tag, options.era, sample_name])
            jobs_dir_external = os.path.join(os.environ['INSTALL_LOC_EXTERNAL'], os.path.relpath(os.path.normpath(jobs_dir), os.environ['INSTALL_LOC']))

            # ---- EOS directory where this sample's histograms land ----
            eosoutdir = eosbase.format(tag=options.tag, era=options.era, sample=sample_name)

            input_root_files = list(open(jobs_dir + "/" + "inputfiles.dat").read().splitlines())

            n_jobs = len(input_root_files)

            job_running = []
            job_failed = []
            job_finished = []
            resubmit_list = {}
            for idf, rfn in enumerate(input_root_files):
                if rfn in str(condor_status):
                    if options.verbose:
                        rich.print(f"Debug found job in [green]running status: {idf} -  {rfn}[/green]")
                    job_running.append(rfn)
                elif os.path.exists(os.path.join(eosoutdir, f'histogram_{idf}.pkl.gz')):
                    # finished: the histogram made it to EOS
                    if options.verbose:
                        rich.print(f"Debug [blue]found histogram on EOS: {idf} -  {rfn} - {eosoutdir}/histogram_{idf}.pkl.gz[/blue]")
                    job_finished.append(rfn)
                elif os.path.exists(jobs_dir_external + f'/histogram_{idf}.pkl.gz'):
                    # legacy fallback: histogram in the jobs dir (pre-EOS submissions)
                    if options.verbose:
                        rich.print(f"Debug [blue]found histogram (legacy jobs dir): {idf} -  {rfn}[/blue]")
                    job_finished.append(rfn)
                else:
                    if options.verbose:
                        rich.print(f"Debug [red]classifying job as failed: {idf} -  {rfn}[/red]")
                    job_failed.append(rfn)
                    resubmit_list[idf] = rfn
            logging.info(
                "-- {:62s}".format((sample_name[:60] + '..') if len(sample_name)>60 else sample_name) +
                (
                    f" --> {n_jobs:5d} : completed" if n_jobs==len(job_finished) else
                    f" --> {n_jobs:5d} : {len(job_running):5d} {n_jobs-len(job_failed)-len(job_running):5d} {len(job_failed):5d}"
                )
            )

            if len(job_running)>0:
                for rfn in job_running:
                    logging.debug(f'running : {rfn}')
            if len(job_failed)>0:
                for rfn in job_failed:
                    logging.debug(f'failed  : {rfn}')

            if options.resubmit and len(job_failed)>0:
                local_rerun_lines = [rerun_script_header]
                for jid,infile in resubmit_list.items():
                    if options.runlocal:
                        assert options.era != "", f"please specify the era you are rerunning ... example: --era=2018"
                        local_rerun_lines.append(
                            f"$INSTALL_LOC_EXTERNAL/.env/bin/python3 brewer-remote-inclusive.py --jobNum={jid} --isMC={auto_isMC} {'--split_by_charge' if options.split_by_charge else ''} {"--split_FV" if options.split_FV else ""} {"--pol" if options.pol else ""} --era={options.era} --infile={infile} --executor={options.executor} {'--copyInput' if options.copyInput else ''}\n"
                        )
                        continue

                    # ---- resubmit by REUSING the sample's original script.sh ----
                    # script.sh already carries the correct env, isMC flag, and the
                    # xrdcp-to-EOS block, and takes ($1=jobNum, $2=infile). We only
                    # override the arguments (bump the flavour, bump request_memory)
                    # -- no separate resub script that can drift out of sync.
                    #
                    # All resub artifacts are read from / written to jobs_dir_external
                    # (the host-visible AFS path condor_submit is actually handed),
                    # NOT the container-relative jobs_dir -- otherwise the host-side
                    # condor_submit gets "No such file or directory".
                    condor_sub = open(os.path.join(jobs_dir_external, "condor.sub")).readlines()
                    found_memory = False
                    for il, line in enumerate(condor_sub):
                        sline = line.strip().lower()
                        if sline.startswith('arguments'):
                            condor_sub[il] = f"arguments             = {jid} {infile}\n"
                        elif sline.startswith('+jobflavour'):
                            condor_sub[il] = '+JobFlavour           = "workday"\n'
                        elif sline.startswith('request_memory'):
                            condor_sub[il] = f"request_memory        = {options.memory}\n"
                            found_memory = True
                        elif sline.startswith('queue'):
                            condor_sub[il] = "queue\n"
                    # If the base condor.sub predates the request_memory template,
                    # inject it now (before writing) so the resub actually gets the bump.
                    if not found_memory:
                        for il, line in enumerate(condor_sub):
                            if line.strip().lower().startswith('request_disk'):
                                condor_sub.insert(il + 1, f"request_memory        = {options.memory}\n")
                                break

                    resub_sub_path = os.path.join(jobs_dir_external, f'condor_resub_{jid}.sub')
                    with open(resub_sub_path, 'w') as new_condor:
                        new_condor.writelines(condor_sub)
                        new_condor.close()

                    if not options.dryrun:
                        assert os.path.exists(resub_sub_path), f"resub sub not written where condor looks: {resub_sub_path}"
                        cmd = f"cd {captured_env['INSTALL_LOC']} && source {to_source} && condor_submit {resub_sub_path}"
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

                if options.runlocal:
                    raise NotImplementedError("runlocal has not been modified to run inside the container yet, properly referencing the bash/zsh-shell bootstrap and environment variables pointing to the image used 'FULL_IMAGE'")
                    with open(os.path.join(jobs_dir, f"rerun-script.sh"), "w") as _stream:
                        _stream.writelines(local_rerun_lines)

                    coffea_image = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
                    os.system(f"cp dist/Qawa-0.0.7-py2.py3-none-any.whl {jobs_dir}")
                    if not options.dryrun:
                        htc = os.popen(f"singularity exec -B {jobs_dir}:/srv/ {coffea_image} bash /srv/rerun-script.sh").read()
                        print(htc)




if __name__ == "__main__":
    main()