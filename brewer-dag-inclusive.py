#!/usr/bin/env python3
"""Phase-2 DAGMan submitter (htcondor2): one DAG per (tag, era) spanning all
samples, rooted at jobs_<tag>_<era>/dag/. Same CLI as brewer-htcondor2-inclusive.py
plus --dag, --resume, and --histserv (plan 6.4).

  fresh submit : resolve datasets -> render per-sample script.sh -> build DAG ->
                 htcondor2.Submit.from_dag(..., force=True) -> Schedd().submit
  --resume     : rerun Submit.from_dag on the SAME workflow.dag WITHOUT force so
                 DAGMan picks up workflow.dag.rescue### and reruns only failed nodes
  --histserv   : Phase-3 flag; threaded through to WorkflowDAG(histserv=...) so
                 Phase 3 only flips behavior, not plumbing. Errors out for now.

--resume and fresh submit are mutually exclusive (force deletes rescue files).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil

from qawa.condor.config import SubmissionConfig
from qawa.condor import datasets as ds
from qawa.condor import proxy as proxymod
from qawa.condor import submit as submitmod
from qawa.condor.dag import WorkflowDAG, SampleJobs

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("brewer-dag")


def parse_args():
    p = argparse.ArgumentParser(description="Famous Submitter (htcondor2 DAGMan)")
    p.add_argument("-a", "--analysis", type=str, default="inc-WZ", required=True)
    p.add_argument("-i", "--input", type=str, default="data.txt", required=True)
    p.add_argument("-t", "--tag", type=str, default="atakour", required=True)
    p.add_argument("-isMC", "--isMC", type=int, default=1)
    p.add_argument("--zzdd", type=str, default="onlySR", choices=["onlySR", "DYSR", "MC"])
    p.add_argument("--split_by_charge", action="store_true")
    p.add_argument("-q", "--queue", type=str, default="longlunch")
    p.add_argument("-e", "--era", type=str, default="2018")
    p.add_argument("-f", "--force", action="store_true", help="recreate files and jobs")
    p.add_argument("-dry", "--dryrun", action="store_true", help="build DAG, do not submit")
    p.add_argument("--redo-proxy", action="store_true")
    p.add_argument("-ex", "--executor", type=str, default="FuturesExecutor",
                   choices=["FuturesExecutor", "IterativeExecutor", "DaskExecutor"])
    p.add_argument("--dag", action="store_true",
                   help="(default behavior of this entry point) build+submit a DAG")
    p.add_argument("--resume", action="store_true",
                   help="rerun Submit.from_dag on the existing DAG WITHOUT force so "
                        "DAGMan reruns only the failed nodes from the rescue file")
    p.add_argument("--histserv", action="store_true",
                   help="Phase-3 histserv mode (not implemented yet)")
    p.add_argument("--maxidle", type=int, default=500,
                   help="DAGMan maxidle throttle passed to Submit.from_dag")
    opts = p.parse_args()
    if opts.resume and opts.force:
        p.error("--resume and --force are mutually exclusive (--force deletes rescue files)")
    return opts


def _dagdir(tag: str, era: str) -> str:
    return os.path.join("_".join(["jobs", tag, era]), "dag")


def _resolve_samples(opts, proxy_path, brewer_loc):
    """Resolve datasets, create per-sample jobdirs, render script.sh, return
    (cfg, list[SampleJobs]). cfg is a representative config for the shared knobs."""
    with open(opts.input) as stream:
        lines = stream.read().split("\n")

    samples: list[SampleJobs] = []
    rep_cfg = None
    for line in lines:
        sample = ds.parse_sample_line(line)
        if sample is None:
            continue

        cfg = SubmissionConfig.from_env(
            analysis=opts.analysis, tag=opts.tag, era=opts.era,
            isMC=sample.isMC, zzdd=opts.zzdd, split_by_charge=opts.split_by_charge,
            queue=opts.queue, executor=opts.executor, proxy_path=proxy_path,
        )
        if rep_cfg is None:
            rep_cfg = cfg

        jobdir = ds.jobs_dir_name(opts.tag, opts.era, sample.name)
        logger.info("-- sample: %s -> %s", sample.query, jobdir)

        if os.path.isdir(jobdir):
            if not opts.force:
                # Abort rather than skip: a silently partial DAG is worse than the
                # old per-sample skip (the DAG spans ALL samples for this tag/era).
                raise SystemExit(f"{jobdir} already exists (use --force to recreate, "
                                 "or --resume to continue a previous DAG)")
            logger.warning("%s exists; forcing deletion", jobdir)
            shutil.rmtree(jobdir)
        os.makedirs(jobdir, exist_ok=True)

        files = ds.resolve_files(sample)
        ds.write_inputfiles(jobdir, files)
        submitmod.render_worker_script(cfg, jobdir)

        samples.append(SampleJobs(
            name=sample.name,
            jobdir=jobdir,
            jobdir_external=cfg.external(jobdir),
            script=cfg.external(os.path.join(jobdir, "script.sh")),
            transfer_files=[cfg.external(brewer_loc)],
            files=files,
        ))
    return rep_cfg, samples


def _make_dag(opts, cfg, samples) -> WorkflowDAG:
    dagdir = _dagdir(opts.tag, opts.era)
    merge_script = os.path.join(os.environ.get("INSTALL_LOC", os.getcwd()),
                                "SMQawa", "hist-merger.py")
    # hist-merger.py globs f'{dir}*{tag}*_{era}_*/*.pkl.gz'; the sample jobdirs sit
    # in the current working directory, so --dir is that directory's external path
    # with a trailing separator.
    merge_dir = cfg.external(os.getcwd()) + os.sep
    venv_python = os.path.join(cfg.install_loc_external or os.getcwd(),
                               ".env", "bin", "python3")
    return WorkflowDAG(
        tag=opts.tag, era=opts.era, dagdir=dagdir, cfg=cfg, samples=samples,
        merge_dir=merge_dir, merge_outdir=merge_dir,
        merge_script=cfg.external(merge_script), python_exe=venv_python,
        histserv=opts.histserv,
    )


def main():
    opts = parse_args()

    if opts.histserv:
        # Plumbing is threaded (WorkflowDAG(histserv=True)); behavior is Phase 3.
        raise SystemExit("Phase 3 not implemented: --histserv is accepted but the "
                         "SERVICE/server_ready/PRE-reset/snapshot pieces are not built "
                         "yet. Run without --histserv for the Phase-2 local-hist DAG.")

    proxy_path = proxymod.ensure_proxy(redo=opts.redo_proxy)
    brewer_loc = os.path.join(os.environ.get("INSTALL_LOC", os.getcwd()),
                              "SMQawa", "brewer-remote-inclusive.py")
    dagdir = _dagdir(opts.tag, opts.era)

    if opts.resume:
        dag = WorkflowDAG(tag=opts.tag, era=opts.era, dagdir=dagdir,
                          histserv=opts.histserv)
        if not os.path.isfile(dag.dag_path()):
            raise SystemExit(f"--resume: no DAG at {dag.dag_path()} (nothing to resume)")
        logger.info("resume: re-submitting %s (DAGMan will use the rescue file)",
                    dag.dag_path())
        cluster = dag.submit(resume=True, maxidle=opts.maxidle)
        logger.info("resubmitted DAGMan cluster %d (rescue-driven)", cluster)
        return

    cfg, samples = _resolve_samples(opts, proxy_path, brewer_loc)
    if not samples:
        raise SystemExit("no samples parsed from input list")

    dag = _make_dag(opts, cfg, samples)
    dag_file = dag.build()
    logger.info("built DAG %s: %d samples, %d worker nodes",
                dag_file, len(samples), sum(len(s.files) for s in samples))

    if opts.dryrun:
        logger.info("dryrun: DAG assembled at %s (not submitted)", dag_file)
        return

    cluster = dag.submit(resume=False, maxidle=opts.maxidle)
    logger.info("submitted DAGMan cluster %d for %s", cluster, dag_file)


if __name__ == "__main__":
    main()
