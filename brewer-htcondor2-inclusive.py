#!/usr/bin/env python3
"""Phase-1 bindings-based submitter (htcondor2), behavior-parity replacement for
brewer-htcondor-inclusive.py. Same CLI + ``--submit-via``. The old script is left
untouched until Phase 2 is validated (plan Phase 1, step 4).

  --submit-via=bindings  (default) -> htcondor2.Schedd().submit(sub, itemdata=...)
  --submit-via=call_host           -> write condor.sub and print the call_host
                                      condor_submit command (S0.1 escape hatch)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil

from qawa.condor import config as cfgmod
from qawa.condor.config import SubmissionConfig
from qawa.condor import datasets as ds
from qawa.condor import proxy as proxymod
from qawa.condor import submit as submitmod

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("brewer-htcondor2")


def parse_args():
    p = argparse.ArgumentParser(description="Famous Submitter (htcondor2)")
    p.add_argument("-a", "--analysis", type=str, default="inc-WZ", required=True)
    p.add_argument("-i", "--input", type=str, default="data.txt", required=True)
    p.add_argument("-t", "--tag", type=str, default="atakour", required=True)
    p.add_argument("-isMC", "--isMC", type=int, default=1)
    p.add_argument("--zzdd", type=str, default="onlySR", choices=["onlySR", "DYSR", "MC"])
    p.add_argument("--split_by_charge", action="store_true")
    p.add_argument("-q", "--queue", type=str, default="longlunch")
    p.add_argument("-e", "--era", type=str, default="2018")
    p.add_argument("-f", "--force", action="store_true", help="recreate files and jobs")
    p.add_argument("-s", "--submit", action="store_true", help="submit only (reuse inputfiles.dat)")
    p.add_argument("-dry", "--dryrun", action="store_true", help="build files, do not submit")
    p.add_argument("--redo-proxy", action="store_true")
    p.add_argument("-ex", "--executor", type=str, default="FuturesExecutor",
                   choices=["FuturesExecutor", "IterativeExecutor", "DaskExecutor"])
    p.add_argument("--submit-via", type=str, default="bindings",
                   choices=["bindings", "call_host"],
                   help="bindings: htcondor2.Schedd().submit; call_host: write condor.sub for the shim")
    return p.parse_args()


def main():
    opts = parse_args()

    proxy_path = proxymod.ensure_proxy(redo=opts.redo_proxy)

    brewer_loc = os.path.join(os.environ.get("INSTALL_LOC", os.getcwd()),
                              "SMQawa", "brewer-remote-inclusive.py")

    schedd = None
    if opts.submit_via == "bindings" and not opts.dryrun:
        cfgmod.ensure_condor_config()
        import htcondor2
        schedd = htcondor2.Schedd()

    with open(opts.input) as stream:
        lines = stream.read().split("\n")

    for line in lines:
        sample = ds.parse_sample_line(line)
        if sample is None:
            continue

        cfg = SubmissionConfig.from_env(
            analysis=opts.analysis, tag=opts.tag, era=opts.era,
            isMC=sample.isMC, zzdd=opts.zzdd, split_by_charge=opts.split_by_charge,
            queue=opts.queue, executor=opts.executor, proxy_path=proxy_path,
        )

        jobdir = ds.jobs_dir_name(opts.tag, opts.era, sample.name)
        logger.info("-- sample: %s -> %s", sample.query, jobdir)

        if os.path.isdir(jobdir):
            if not opts.force and not opts.submit:
                logger.error("%s already exists (use --force to recreate)", jobdir)
                continue
            if opts.force:
                logger.warning("%s exists; forcing deletion", jobdir)
                shutil.rmtree(jobdir)
                os.makedirs(jobdir)
        else:
            os.makedirs(jobdir)

        if not opts.submit:
            files = ds.resolve_files(sample)
            ds.write_inputfiles(jobdir, files)
        else:
            files = ds.read_inputfiles(jobdir)

        submitmod.render_worker_script(cfg, jobdir)
        sub = submitmod.build_worker_submit(cfg, jobdir, transfer_files=[brewer_loc]) \
            if opts.submit_via == "bindings" or opts.dryrun else None

        if opts.dryrun:
            # Always leave a submit file behind for inspection.
            if sub is None:
                sub = submitmod.build_worker_submit(cfg, jobdir, transfer_files=[brewer_loc])
            submitmod.write_submit_file(sub, jobdir)
            logger.info("dryrun: wrote %s/condor.sub (%d files)", jobdir, len(files))
            continue

        if opts.submit_via == "bindings":
            result = submitmod.submit_sample(sub, files, jobdir, cfg, schedd=schedd)
            logger.info("cluster %d: %d jobs (%s)", result.cluster, result.n_jobs, jobdir)
        else:
            sub = submitmod.build_worker_submit(cfg, jobdir, transfer_files=[brewer_loc])
            subfile = submitmod.write_submit_file(sub, jobdir)
            ext = cfg.external(subfile)
            print(f"call_host condor_submit {ext}")
            logger.info("call_host mode: submit with -> call_host condor_submit %s", ext)


if __name__ == "__main__":
    main()
