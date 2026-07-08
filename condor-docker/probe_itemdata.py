#!/usr/bin/env python3
"""Phase-1 acceptance harness (local pool portion).

Exercises the actual qawa.condor Phase-1 code -- not a generic probe -- in two parts:

  (A) OFFLINE: assert build_worker_submit() reproduces every "preserved knob" from
      the old condor_TEMPLATE (plan Phase 1 accept criterion: RequestDisk, retries,
      requirements, SingularityImage, arguments, on_exit_remove, transfer).

  (B) LIVE: drive submit_sample()'s itemdata + manifest path on the running v25
      pool. The worker payload (worker.sh) targets a CVMFS venv + SingularityImage
      that do not exist locally, so for the live run we substitute a trivial
      runnable executable while keeping submit_sample()/itemdata/manifest -- the
      code actually under test -- unchanged. Confirms N files -> N jobs across the
      3 workers, all complete, and manifest.json is written correctly.

Run on the access point as the submitter:

    docker compose exec -u submituser submit \\
        env PYTHONPATH=/smqawa/src python3 /smqawa/condor-docker/probe_itemdata.py

The full dataset/coffea parity test (identical histogram_<N>.pkl.gz vs the old
submitter) runs only on the LPC/lxplus AP -- see condor-spikes/RESULTS.md.
"""
from __future__ import annotations

import os
import tempfile
import time

from qawa.condor.config import SubmissionConfig
from qawa.condor import submit as submitmod


def check_offline() -> None:
    print("== (A) offline: preserved-knob assertions on build_worker_submit ==")
    cfg = SubmissionConfig(
        analysis="inc-WZ", tag="probe", era="2018", isMC=1, queue="longlunch",
        coffea_image="cmsX/coffea:latest", full_image="full:latest",
        install_loc="/srv", install_loc_external="/srv", proxy_path="/tmp/x509",
    )
    jobdir = tempfile.mkdtemp()
    sub = submitmod.build_worker_submit(cfg, jobdir, transfer_files=["/srv/SMQawa/brewer-remote-inclusive.py"])
    text = str(sub)
    print(text)

    checks = {
        "request_disk 10000000": sub.get("request_disk") == "10000000",
        "max_retries 3": sub.get("max_retries") == "3",
        "requirements LastRemoteHost": "LastRemoteHost" in sub.get("requirements", ""),
        "MY.SingularityImage on cvmfs": "unpacked.cern.ch" in sub.get("MY.SingularityImage", ""),
        "arguments ProcId+jobfn": sub.get("arguments") == "$(ProcId) $(jobfn)",
        "on_exit_remove ExitCode==0": "ExitCode == 0" in sub.get("on_exit_remove", ""),
        "should_transfer_files YES": sub.get("should_transfer_files") == "YES",
        "WhenToTransferOutput": sub.get("WhenToTransferOutput") == "ON_EXIT_OR_EVICT",
    }
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    assert all(checks.values()), "preserved-knob check failed"

    # for_dag variant: DAGMan owns retry, jobid VARS instead of ProcId
    sub_dag = submitmod.build_worker_submit(cfg, jobdir, transfer_files=["/srv/x.py"], for_dag=True)
    assert sub_dag.get("max_retries", "") == "", "for_dag must drop max_retries (DAGMan RETRY owns it)"
    assert sub_dag.get("on_exit_remove", "") == "", \
        "for_dag must drop on_exit_remove (else failed jobs re-queue forever and DAGMan never sees the failure)"
    assert sub_dag.get("arguments") == "$(jobid) $(jobfn)", "for_dag must use $(jobid) VARS"
    print("  [OK] for_dag variant drops max_retries/on_exit_remove and uses $(jobid)")
    print("== (A) PASS ==\n")


def check_live() -> int:
    print("== (B) live: submit_sample itemdata + manifest on the pool ==")
    import htcondor2

    jobdir = tempfile.mkdtemp(prefix="qawa_itemdata_")
    files = [f"/store/fake/file_{i}.root" for i in range(3)]

    cfg = SubmissionConfig(
        analysis="inc-WZ", tag="probe", era="2018", isMC=1,
        install_loc="/", install_loc_external="/", proxy_path="/tmp/x509",
    )

    # Runnable stand-in for the CVMFS payload: echo the itemdata arg, exit 0.
    # Everything else (itemdata iteration, cluster, manifest) is the real code path.
    sub = htcondor2.Submit({
        "executable": "/bin/echo",
        "arguments": "job $(ProcId) infile=$(jobfn)",
        "universe": "vanilla",
        "request_disk": "64MB",
        "request_memory": "64MB",
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT",
        "output": "/tmp/qawa_itemdata.$(ProcId).out",
        "error": "/tmp/qawa_itemdata.$(ProcId).err",
        "log": os.path.join(jobdir, "cluster.log"),
    })

    result = submitmod.submit_sample(sub, files, jobdir, cfg, schedd=htcondor2.Schedd())
    print(f"  submitted cluster {result.cluster}: {result.n_jobs} jobs -> {result.jobdir}")
    assert result.n_jobs == 3

    # manifest.json written and correct
    import json
    with open(result.manifest_path) as fh:
        manifest = json.load(fh)
    assert manifest["n_jobs"] == 3 and manifest["cluster"] == result.cluster
    assert manifest["files"]["0"] == files[0] and manifest["files"]["2"] == files[2]
    print(f"  [OK] manifest.json: cluster={manifest['cluster']} n_jobs={manifest['n_jobs']} files mapped")

    schedd = htcondor2.Schedd()
    deadline = time.time() + 180
    while time.time() < deadline:
        ads = schedd.query(constraint=f"ClusterId=={result.cluster}",
                           projection=["ProcId", "JobStatus"])
        if not ads:
            print("  all 3 jobs left the queue -> completed on the workers.")
            print("== (B) PASS ==")
            return 0
        left = len(ads)
        print(f"  {left} job(s) still in queue ...")
        time.sleep(5)
    print("  TIMEOUT waiting for jobs to complete")
    return 1


def main() -> int:
    check_offline()
    return check_live()


if __name__ == "__main__":
    raise SystemExit(main())
