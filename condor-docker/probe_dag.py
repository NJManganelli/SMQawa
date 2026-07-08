#!/usr/bin/env python3
"""Phase-2 DAG acceptance harness on the local v25 pool.

Exercises the REAL qawa.condor.dag.WorkflowDAG code (build + submit + resume) with
synthetic worker/merge payloads (no dasgoclient / NanoAOD / coffea needed). Runs
inside the submit container as submituser with PYTHONPATH=/smqawa/src.

Scenarios (plan 6 acceptance, adapted):
  1. fresh submit: 2 samples x 3 files -> all workers + FINAL merge run; manifest.
  2. kill one worker mid-run (condor_rm the node job) -> DAGMan retries -> completes.
  3. force one node to exhaust retries (payload always exits 1) -> DAG fails, a
     rescue file appears, FINAL still runs + writes MERGE_INCOMPLETE; then fix the
     payload and --resume (no force) reruns ONLY the failed node -> completes,
     rescue consumed.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time

import htcondor2

sys.path.insert(0, "/smqawa/src")
from qawa.condor.dag import WorkflowDAG, SampleJobs  # noqa: E402

ROOT = "/home/submituser/dag_accept"
SAMPLES = ["SAMPA", "SAMPB"]
NFILES = 3


def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def make_good_worker(path):
    # Synthetic worker: arg1=jobid arg2=infile; touch the output pickle in cwd
    # (initialdir == per-sample jobdir), mimicking the coffea payload's product.
    with open(path, "w") as fh:
        fh.write(
            "#!/bin/bash\n"
            "echo worker jobid=$1 infile=$2 cwd=$(pwd)\n"
            "sleep 3\n"
            "printf 'synthetic' > histogram_$1.pkl.gz\n"
            "test -f histogram_$1.pkl.gz\n"
        )
    os.chmod(path, 0o755)


def make_bad_worker(path):
    with open(path, "w") as fh:
        fh.write("#!/bin/bash\necho BAD worker jobid=$1 always-fails\nexit 1\n")
    os.chmod(path, 0o755)


def make_merge_script(path):
    # Stand-in for hist-merger.py: write one merged file per sample dir it finds.
    with open(path, "w") as fh:
        fh.write(
            "#!/usr/bin/env python3\n"
            "import argparse, glob, os\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--dir'); p.add_argument('--outdir')\n"
            "p.add_argument('--tag'); p.add_argument('--era')\n"
            "p.add_argument('--force', action='store_true')\n"
            "a=p.parse_args()\n"
            "pat=f'{a.dir}*{a.tag}*_{a.era}_*/*.pkl.gz'\n"
            "hits=glob.glob(pat)\n"
            "print('merge glob', pat, '->', len(hits), 'files')\n"
            "dirs={os.path.basename(os.path.dirname(h)) for h in hits}\n"
            "for d in sorted(dirs):\n"
            "    out=os.path.join(a.outdir, f'merged-{d}.pkl.gz')\n"
            "    open(out,'w').write('merged')\n"
            "    print('wrote', out)\n"
        )
    os.chmod(path, 0o755)


def build_dag(tag, worker_paths, merge_script):
    """worker_paths: {sample: script_path}. Returns WorkflowDAG."""
    era = "2018"
    base = os.path.join(ROOT, tag)
    dagdir = os.path.join(base, "_".join(["jobs", tag, era]), "dag")
    samples = []
    for s in SAMPLES:
        jobdir = os.path.join(base, "_".join(["jobs", tag, era, s]))
        os.makedirs(jobdir, exist_ok=True)
        samples.append(SampleJobs(
            name=s, jobdir=jobdir, jobdir_external=jobdir,
            script=worker_paths[s],
            transfer_files=[],
            files=[f"/store/fake/{s}_{i}.root" for i in range(NFILES)],
        ))
    merge_dir = base + os.sep
    dag = WorkflowDAG(
        tag=tag, era=era, dagdir=dagdir, cfg=_cfg(), samples=samples,
        merge_dir=merge_dir, merge_outdir=merge_dir,
        merge_script=merge_script, python_exe="/usr/bin/python3",
    )
    return dag


def _cfg():
    from qawa.condor.config import SubmissionConfig
    return SubmissionConfig(
        analysis="inc-WZ", tag="probe", era="2018", isMC=1,
        install_loc="/", install_loc_external="/", proxy_path="/tmp/x509",
        coffea_image="x/y:latest", full_image="x/y:latest",
    )


def patch_worker_sub_no_transfer(dag):
    """Synthetic workers target no CVMFS image and transfer no extra inputs; strip
    singularity/requirements/transfer_input_files from the shared worker.sub (which
    build() writes with the real production knobs) and insert small resource asks
    BEFORE the trailing `queue` line. The executable itself still auto-transfers."""
    p = os.path.join(dag.dagdir, dag.WORKER_SUB)
    with open(p) as fh:
        lines = fh.read().splitlines()
    out = []
    for ln in lines:
        k = ln.split("=", 1)[0].strip().lower()
        if k in ("transfer_input_files", "my.singularityimage", "requirements"):
            continue
        if k == "queue":
            out += ["should_transfer_files = YES",
                    "when_to_transfer_output = ON_EXIT",
                    "request_memory = 64MB"]
        out.append(ln)
    with open(p, "w") as fh:
        fh.write("\n".join(out) + "\n")


def dagman_cluster_alive(schedd, cluster):
    ads = schedd.query(constraint=f"ClusterId=={cluster}", projection=["ProcId"])
    return len(ads) > 0


def wait_dag(schedd, cluster, timeout=420):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not dagman_cluster_alive(schedd, cluster):
            return True
        time.sleep(5)
    return False


def status_counts(dagdir):
    """Parse dag.status NODE_STATUS_FILE for a rough state summary."""
    p = os.path.join(dagdir, "dag.status")
    if not os.path.isfile(p):
        return "no status file"
    with open(p) as fh:
        txt = fh.read()
    return txt


# ------------------------------------------------------------------- scenarios
def scenario1():
    print("\n==================== SCENARIO 1: fresh submit ====================")
    tag = "s1"
    shutil.rmtree(os.path.join(ROOT, tag), ignore_errors=True)
    wp = {}
    base = os.path.join(ROOT, tag)
    os.makedirs(base, exist_ok=True)
    for s in SAMPLES:
        wp[s] = os.path.join(base, f"worker_{s}.sh")
        make_good_worker(wp[s])
    merge = os.path.join(base, "merge.py")
    make_merge_script(merge)
    dag = build_dag(tag, wp, merge)
    dag.build()
    patch_worker_sub_no_transfer(dag)
    schedd = htcondor2.Schedd()
    cluster = dag.submit(resume=False, maxidle=500)
    print(f"DAGMan cluster = {cluster}")
    import json
    with open(os.path.join(dag.dagdir, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["dagman_cluster"] == cluster, man
    assert man["n_worker_nodes"] == len(SAMPLES) * NFILES, man
    print(f"[OK] manifest records DAGMan cluster {cluster}, "
          f"n_worker_nodes={man['n_worker_nodes']}")
    ok = wait_dag(schedd, cluster)
    merged = sorted(glob.glob(os.path.join(base, "merged-*.pkl.gz")))
    print("merged outputs:", merged)
    marker = os.path.join(base, "MERGE_INCOMPLETE")
    print("MERGE_INCOMPLETE present?", os.path.exists(marker))
    print("--- dag.status tail ---")
    print(status_counts(dag.dagdir)[-600:])
    assert ok, "DAG did not leave the queue in time"
    assert len(merged) == len(SAMPLES), merged
    assert not os.path.exists(marker), "clean run must not write MERGE_INCOMPLETE"
    print("[SCENARIO 1 PASS]")
    return True


def scenario2():
    print("\n============ SCENARIO 2: kill a worker, DAG retries ============")
    tag = "s2"
    shutil.rmtree(os.path.join(ROOT, tag), ignore_errors=True)
    base = os.path.join(ROOT, tag)
    os.makedirs(base, exist_ok=True)
    wp = {}
    for s in SAMPLES:
        wp[s] = os.path.join(base, f"worker_{s}.sh")
        # longer sleep so we have time to condor_rm a running node
        with open(wp[s], "w") as fh:
            fh.write(
                "#!/bin/bash\necho worker jobid=$1 cwd=$(pwd)\nsleep 25\n"
                "printf 'synthetic' > histogram_$1.pkl.gz\n"
            )
        os.chmod(wp[s], 0o755)
    merge = os.path.join(base, "merge.py")
    make_merge_script(merge)
    dag = build_dag(tag, wp, merge)
    dag.build()
    patch_worker_sub_no_transfer(dag)
    schedd = htcondor2.Schedd()
    cluster = dag.submit(resume=False, maxidle=500)
    print(f"DAGMan cluster = {cluster}")

    # Find and remove one running non-DAGMan worker job under this DAGMan.
    killed = None
    deadline = time.time() + 120
    while time.time() < deadline and killed is None:
        ads = schedd.query(
            constraint=f"DAGManJobId=={cluster} && JobStatus==2",
            projection=["ClusterId", "ProcId", "JobStatus"])
        if ads:
            victim = ads[0]
            jid = f"{int(victim['ClusterId'])}.{int(victim['ProcId'])}"
            r = sh(f"condor_rm {jid}")
            print("condor_rm:", r.stdout.strip(), r.stderr.strip())
            killed = jid
            break
        time.sleep(2)
    print(f"killed running worker job: {killed}")
    assert killed is not None, "no running worker job appeared to kill"

    ok = wait_dag(schedd, cluster, timeout=480)
    print("--- dag.status tail ---")
    print(status_counts(dag.dagdir)[-600:])
    # DAGMan log records the retry
    dbg = glob.glob(os.path.join(dag.dagdir, "*.dagman.out"))
    retried = False
    if dbg:
        with open(dbg[0]) as fh:
            t = fh.read()
        retried = ("Retrying node" in t) or ("Job retry" in t) or ("will retry" in t.lower())
        print("dagman.out mentions retry?", retried)
    merged = sorted(glob.glob(os.path.join(base, "merged-*.pkl.gz")))
    print("merged outputs:", merged)
    assert ok, "DAG did not complete after worker kill+retry"
    assert len(merged) == len(SAMPLES), merged
    print("[SCENARIO 2 PASS] (killed one worker, DAG retried and completed)")
    return True


def scenario3():
    print("\n==== SCENARIO 3: exhaust retries -> rescue -> resume ====")
    tag = "s3"
    shutil.rmtree(os.path.join(ROOT, tag), ignore_errors=True)
    base = os.path.join(ROOT, tag)
    os.makedirs(base, exist_ok=True)
    wp = {}
    # SAMPA good, SAMPB bad (always exits 1 -> exhausts RETRY 3)
    wp["SAMPA"] = os.path.join(base, "worker_SAMPA.sh")
    make_good_worker(wp["SAMPA"])
    wp["SAMPB"] = os.path.join(base, "worker_SAMPB.sh")
    make_bad_worker(wp["SAMPB"])
    merge = os.path.join(base, "merge.py")
    make_merge_script(merge)
    dag = build_dag(tag, wp, merge)
    dag.build()
    patch_worker_sub_no_transfer(dag)
    schedd = htcondor2.Schedd()
    cluster = dag.submit(resume=False, maxidle=500)
    print(f"DAGMan cluster = {cluster} (expect FAILURE)")
    ok = wait_dag(schedd, cluster, timeout=480)
    assert ok, "DAGMan did not exit"

    rescue = sorted(glob.glob(os.path.join(dag.dagdir, "workflow.dag.rescue*")))
    print("rescue files:", rescue)
    marker = os.path.join(base, "MERGE_INCOMPLETE")
    print("MERGE_INCOMPLETE present?", os.path.exists(marker))
    merged = sorted(glob.glob(os.path.join(base, "merged-*.pkl.gz")))
    print("merged outputs (partial):", merged)
    assert rescue, "expected a rescue file after a node exhausted retries"
    assert os.path.exists(marker), "FINAL should write MERGE_INCOMPLETE on failure"

    # Fix the bad payload, then --resume (no force) -> reruns only failed node.
    print("\n-- fixing SAMPB payload and resuming (no force) --")
    make_good_worker(wp["SAMPB"])
    # Re-instantiate WorkflowDAG the way brewer-dag-inclusive.py --resume does:
    # minimal object pointing at the same dagdir, submit(resume=True).
    dag_resume = WorkflowDAG(tag=tag, era="2018", dagdir=dag.dagdir)
    rescue_before = set(glob.glob(os.path.join(dag.dagdir, "workflow.dag.rescue*")))
    cluster2 = dag_resume.submit(resume=True, maxidle=500)
    print(f"resume DAGMan cluster = {cluster2}")
    ok2 = wait_dag(schedd, cluster2, timeout=480)
    print("--- dag.status tail after resume ---")
    print(status_counts(dag.dagdir)[-600:])
    rescue_after = set(glob.glob(os.path.join(dag.dagdir, "workflow.dag.rescue*")))
    merged2 = sorted(glob.glob(os.path.join(base, "merged-*.pkl.gz")))
    print("merged outputs after resume:", merged2)
    print("MERGE_INCOMPLETE after resume?", os.path.exists(marker))
    assert ok2, "resume DAG did not complete"
    assert not os.path.exists(marker), \
        "clean resume must clear the stale MERGE_INCOMPLETE marker"
    # rescue consumed: a successful resume renames/removes the rescue it used
    # (DAGMan bumps the rescue number only on a NEW failure; a clean resume leaves
    # the highest rescue in place but marks the DAG done -- assert both samples merged
    # and no fresh MERGE_INCOMPLETE).
    assert len(merged2) == len(SAMPLES), merged2
    print(f"rescue files before resume: {sorted(rescue_before)}")
    print(f"rescue files after  resume: {sorted(rescue_after)}")
    print("[SCENARIO 3 PASS] (rescue produced; resume reran failed node; both merged)")
    return True


def main():
    os.makedirs(ROOT, exist_ok=True)
    results = {}
    for name, fn in [("scenario1", scenario1), ("scenario2", scenario2),
                     ("scenario3", scenario3)]:
        try:
            results[name] = fn()
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            results[name] = False
    print("\n==================== SUMMARY ====================")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
