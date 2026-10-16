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

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Fallback request_memory (MB) used when --memory is not given explicitly.
# Bumped from the old 8000 since these brewer jobs are memory-hungry. Jobs that
# were OOM-evicted (held) usually need much more -- pass --memory explicitly.
DEFAULT_MEMORY = 12000

rerun_script_header = f"""#!/bin/bash
echo "----- JOB STARTS @" `date "+%Y-%m-%d %H:%M:%S"`
{{command}}
if [ ! -f "histogram_{{jobid}}.pkl.gz" ]; then
    echo "ERROR: No output histogram pickle file found"
    exit 1
fi
xrdcp -f -p histogram_{{jobid}}.pkl.gz \
    "root://eosuser.cern.ch/$EOSOUTDIR/histogram_{{jobid}}.pkl.gz"
XRDEXIT=$?
if [ $XRDEXIT -ne 0 ]; then
    echo "ERROR: Failed to copy histogram to EOS (exit $XRDEXIT)"
    exit $XRDEXIT
fi
rm -f histogram_{{jobid}}.pkl.gz
echo "----- JOB FINISHED @" `date "+%Y-%m-%d %H:%M:%S"`
"""

resub_script_header = """#!/bin/bash
export X509_USER_PROXY={proxy}
export XRD_REQUESTTIMEOUT=6400
export XRD_REDIRECTLIMIT=64
export INSTALL_LOC_EXTERNAL={install_loc_external}
export COFFEA_IMAGE={coffea_image}
export FULL_IMAGE={full_image}
export EOSOUTDIR={eosoutdir}
source $INSTALL_LOC_EXTERNAL/.env/bin/activate
echo "----- JOB STARTS @" `date "+%Y-%m-%d %H:%M:%S"`
{command}
if [ ! -f "histogram_{jobid}.pkl.gz" ]; then
    echo "ERROR: No output histogram pickle file found"
    exit 1
fi
xrdcp -f -p histogram_{jobid}.pkl.gz \
    "root://eosuser.cern.ch/$EOSOUTDIR/histogram_{jobid}.pkl.gz"
XRDEXIT=$?
if [ $XRDEXIT -ne 0 ]; then
    echo "ERROR: Failed to copy histogram to EOS (exit $XRDEXIT)"
    exit $XRDEXIT
fi
rm -f histogram_{jobid}.pkl.gz
echo "----- JOB FINISHED @" `date "+%Y-%m-%d %H:%M:%S"`
"""


def run_condor(cmd_suffix, captured_env, to_source):
    """Run a condor command through the same shell bootstrap the rest of the
    script uses (cd INSTALL_LOC && source .bashrc/.zshrc && ...), returning
    (returncode, stdout_str, stderr_str). Centralises the Popen boilerplate so
    the added condor_q -hold / condor_rm calls stay consistent with condor_q /
    condor_submit."""
    cmd = f"cd {captured_env['INSTALL_LOC']} && source {to_source} && {cmd_suffix}"
    htc = subprocess.Popen(
        cmd,
        shell      = True,
        executable = captured_env['SHELL'],
        env        = captured_env,
        stdin      = subprocess.PIPE,
        stdout     = subprocess.PIPE,
        stderr     = subprocess.PIPE,
        close_fds  = True,
    )
    out, err = htc.communicate()
    return htc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def get_held_jobs(captured_env, to_source):
    """Return (held_blob, held_procs) for jobs currently in the HELD (H) state.

    held_blob  : raw text containing the held jobs' arguments (for substring
                 detection of an infile).
    held_procs : {infile: "cluster.proc"} so a held job can be condor_rm'd
                 before being resubmitted fresh (picking up the bumped
                 request_memory).

    Held jobs still appear in `condor_q -nobatch`, so without this an OOM-evicted
    job gets misclassified as RUNNING and is never resubmitted -- it just sits in
    H forever. We ask for both Arguments and Args to be robust to the condor
    version's arguments-syntax; the infile is picked out as the token that looks
    like a root file, so 'undefined' fields are harmless.
    """
    rc, out, err = run_condor(
        'condor_q -hold -af ClusterId ProcId Arguments Args',
        captured_env, to_source,
    )
    if rc != 0:
        logging.warning(f"could not query held jobs (condor_q -hold rc={rc}): {err.strip()}")
        return "", {}

    held_procs = {}
    for line in out.splitlines():
        toks = line.split()
        if len(toks) < 3:
            continue
        cluster, proc = toks[0], toks[1]
        infile = next(
            (t for t in toks[2:] if t.startswith("root://") or ".root" in t),
            None,
        )
        if infile:
            held_procs[infile] = f"{cluster}.{proc}"
    return out, held_procs


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
    parser.add_argument("--resubmit", action="store_true", help="resubmit failed AND held jobs")
    parser.add_argument("--dryrun"  , action="store_true")
    parser.add_argument('--executor' , type=str, default="FuturesExecutor", help="Executor to use, one of IterativeExecutor (good for debugging), FuturesExecutor (multithreaded), or other coffea option")
    parser.add_argument('--copyInput', action='store_true'     , help="xrdcp a file to the worker node before executing the coffea processor on it")
    parser.add_argument('--verbose'  , action='store_true'     , help="verbose output printing status of running, finished, and failed files per job")
    parser.add_argument("-m", "--memory", type=int, default=None,
                        help=f"request_memory in MB for the resubmitted jobs (default {DEFAULT_MEMORY}). "
                             f"HELD jobs are usually OOM-evicted -- pass a larger value "
                             f"(e.g. --memory 36000) or they will be held again.")
    parser.add_argument("--keep-held", action="store_true",
                        help="do NOT condor_rm held jobs before resubmitting them "
                             "(default is to remove the held job first so it does not "
                             "linger in H alongside the fresh resubmission)")
    parser.add_argument("--eosdir", type=str, default=None,
                        help="EOS base directory holding the output histograms "
                             "(must match the one used at submission; default: "
                             "/eos/user/<u>/<user>/WZtotau2lnu). Job completion is "
                             "checked in <eosdir>/<tag>/<era>/<sample>/histogram_<N>.pkl.gz")
    options = parser.parse_args()

    # resolve request_memory: explicit value wins, else the documented default
    req_mem = options.memory if options.memory is not None else DEFAULT_MEMORY

    captured_env = os.environ.copy()
    if "bash" in captured_env['SHELL']:
        to_source = os.path.join(captured_env['INSTALL_LOC'], ".bashrc")
    elif "zsh" in captured_env['SHELL']:
        to_source = os.path.join(captured_env['INSTALL_LOC'], ".zshrc")
    else:
        raise NotImplementedError("neither bash or zsh detected in the shell env variable, something has gone wrong; contents=", captured_env['SHELL'])

    exit_status, condor_status, htc_err = run_condor(
        "condor_q -nobatch", captured_env, to_source
    )
    if exit_status != 0:
        logging.warning(f"condor_q failed: {htc_err.strip()}")

    # ---- held (H-state) jobs: OOM-evicted brewer jobs land here and, because
    # they still show up in condor_q -nobatch, would otherwise be miscounted as
    # running and never resubmitted. ----
    held_blob, held_procs = get_held_jobs(captured_env, to_source)
    if held_procs:
        logging.info(f"found {len(held_procs)} held job(s) across the queue "
                     f"(will be resubmitted with --resubmit)")

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
            job_held = []
            job_finished = []
            resubmit_list = {}
            for idf, rfn in enumerate(input_root_files):
                # NOTE: completion (EOS/legacy histogram) is checked BEFORE the
                # condor_q membership check on purpose. `condor_q -nobatch` is run
                # non-interactively (piped through subprocess), so HTCondor does
                # NOT truncate the command line -- the full Arguments (including
                # this infile) are present in `condor_status` for ANY job still
                # sitting in the schedd's queue, not just ones actually R/Idle.
                # A job that already finished and copied its histogram to EOS can
                # still linger in the queue in Completed (C) state for a while
                # before being purged, so checking `rfn in condor_status` first
                # would misclassify it as "running" and mask a real completion --
                # which is exactly what produces run=N/done=0 even when the
                # histograms already exist on EOS.
                #
                # held is still checked first: a genuinely held job cannot have
                # produced a finished histogram, so there's no ordering conflict.
                if rfn in held_blob:
                    if options.verbose:
                        rich.print(f"Debug [magenta]found HELD job (resubmit): {idf} -  {rfn}[/magenta]")
                    job_held.append(rfn)
                    resubmit_list[idf] = rfn
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
                elif rfn in condor_status:
                    if options.verbose:
                        rich.print(f"Debug found job in [green]running status: {idf} -  {rfn}[/green]")
                    job_running.append(rfn)
                else:
                    if options.verbose:
                        rich.print(f"Debug [red]classifying job as failed: {idf} -  {rfn}[/red]")
                    job_failed.append(rfn)
                    resubmit_list[idf] = rfn

            logging.info(
                "-- {:62s}".format((sample_name[:60] + '..') if len(sample_name)>60 else sample_name) +
                (
                    f" --> {n_jobs:5d} : completed" if n_jobs==len(job_finished) else
                    f" --> {n_jobs:5d} : run={len(job_running):<5d} done={len(job_finished):<5d} "
                    f"fail={len(job_failed):<5d} held={len(job_held):<5d}"
                )
            )
            if len(job_running)>0:
                for rfn in job_running:
                    logging.debug(f'running : {rfn}')
            if len(job_held)>0:
                for rfn in job_held:
                    logging.debug(f'held    : {rfn}')
            if len(job_failed)>0:
                for rfn in job_failed:
                    logging.debug(f'failed  : {rfn}')

            # warn once per sample if held jobs are being resubmitted without an
            # explicit memory bump -- these were most likely OOM-evicted.
            if options.resubmit and len(job_held) > 0 and options.memory is None:
                logging.warning(
                    f"{len(job_held)} HELD job(s) in {sample_name} being resubmitted at "
                    f"request_memory={req_mem} MB (default). Held jobs are usually killed "
                    f"for exceeding the cgroup memory limit -- if so, pass a larger --memory "
                    f"(e.g. --memory 36000) or they will be held again."
                )

            if options.resubmit and (len(job_failed)>0 or len(job_held)>0):
                local_rerun_lines = [rerun_script_header]
                for jid,infile in resubmit_list.items():
                    if options.runlocal:
                        assert options.era != "", f"please specify the era you are rerunning ... example: --era=2018"
                        local_rerun_lines.append(
                            f"$INSTALL_LOC_EXTERNAL/.env/bin/python3 brewer-remote-inclusive.py --jobNum={jid} --isMC={auto_isMC} {'--split_by_charge' if options.split_by_charge else ''} {'--split_FV' if options.split_FV else ''} {'--pol' if options.pol else ''} --era={options.era} --infile={infile} --executor={options.executor} {'--copyInput' if options.copyInput else ''}\n"
                        )
                        continue

                    # ---- if this job is HELD, remove it first so the fresh
                    # resubmission (with the bumped request_memory) is the only
                    # copy in the queue. Otherwise condor_release-style behaviour
                    # would just re-run the same job under the same limit. ----
                    held_id = held_procs.get(infile)
                    if held_id and not options.keep_held:
                        if options.dryrun:
                            logging.info(f"[dryrun] would condor_rm held job {held_id} ({infile})")
                        else:
                            rc, rout, rerr = run_condor(f"condor_rm {held_id}", captured_env, to_source)
                            if rc != 0:
                                logging.warning(f"condor_rm {held_id} failed: {rerr.strip()}")

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
                            condor_sub[il] = f"request_memory        = {req_mem}\n"
                            found_memory = True
                        elif sline.startswith('queue'):
                            condor_sub[il] = "queue\n"

                    # If the base condor.sub predates the request_memory template,
                    # inject it now (before writing) so the resub actually gets the bump.
                    if not found_memory:
                        for il, line in enumerate(condor_sub):
                            if line.strip().lower().startswith('request_disk'):
                                condor_sub.insert(il + 1, f"request_memory        = {req_mem}\n")
                                break

                    resub_sub_path = os.path.join(jobs_dir_external, f'condor_resub_{jid}.sub')
                    with open(resub_sub_path, 'w') as new_condor:
                        new_condor.writelines(condor_sub)
                        new_condor.close()

                    if not options.dryrun:
                        assert os.path.exists(resub_sub_path), f"resub sub not written where condor looks: {resub_sub_path}"
                        exit_status, htc_out, htc_err = run_condor(
                            f"condor_submit {resub_sub_path}", captured_env, to_source)
                        if exit_status == 0:
                            logging.info(f"resubmitted job {jid}")
                        else:
                            logging.warning(
                                f"failed to resubmit job {jid}: {htc_err.strip() or htc_out.strip()}"
                            )

                if options.runlocal:
                    raise NotImplementedError("runlocal has not been modified to run inside the container yet, properly referencing the bash/zsh-shell bootstrap and environment variables pointing to the image used 'FULL_IMAGE'")
                    with open(os.path.join(jobs_dir, f"rerun-script.sh"), "w") as _stream:
                        _stream.writelines(local_rerun_lines)
                    coffea_image = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
                    os.system(f"cp dist/Qawa-0.0.7-py2.py3-none-any.whl {jobs_dir}")
                    if not options.dryrun:
                        htc = os.popen(f"singularity exec -B {jobs_dir}:/srv/ {coffea_image} bash /srv/rerun-script.sh").read()


if __name__ == "__main__":
    main()