"""DAGMan orchestration + histserv SERVICE-node hooks (Phases 2 & 3).

Phase 2 is implemented: :class:`WorkflowDAG` builds and submits one DAG per
(tag, era) -- shared ``worker.sub``, one JOB+RETRY per input file, a FINAL
local-universe merge node, and rescue-driven ``--resume``. Phase 3 (histserv)
remains a set of precise hooks (``write_histserv_service_sub``,
``write_pre_reset_script_line``, ``resume_edits_for_rescue``, plus the
``_histserv_dag_lines``/``_final_node_lines`` seams inside the builder) so it
drops in without re-litigating the design. :func:`histserv_unique_id` -- the
concrete double-fill-avoidance hook (see the module notes) -- is implemented
eagerly because it is small, testable, and must be threaded through the worker
fill path the moment histserv mode exists.

Target DAG (per HTCONDOR_PYTHON_PLAN.md 2 & 7.2), rooted at jobs_<tag>_<era>/dag/:

    SERVICE histserv      histserv.sub       # long-running server, UP BEFORE workers
    JOB     server_ready  server_ready.sub   # universe=local: wait for addr + register
    JOB     w_<s>_<i>     worker.sub  VARS…   # one node per input file -> per-node RETRY
    PARENT  server_ready  CHILD w_*
    FINAL   snapshot      snapshot.sub        # snapshot server -> merged pickles, always runs

Non-histserv mode is the same DAG minus SERVICE/server_ready and with FINAL running
plain hist-merger.py, so both modes share one code path.

================================ DOUBLE-FILL HOOKS ==============================
histserv (validated at v0.1.9) DOES implement hash-based double-fill protection,
but it is NOT wired in the prototype and it does NOT survive a server restart.
Both facts drive the hooks below. Concretely (see condor-spikes/RESULTS.md):

  * Native idempotency: ``RemoteHist.fill(unique_id=...)`` /
    ``fill_many(unique_id=...)``. The server keeps ``entry.unique_ids: set[bytes]``
    per histogram and REJECTS a fill whose id is already present with
    ``grpc.StatusCode.ALREADY_EXISTS`` (histserv/service.py::_reject_duplicate_unique_id).
    The id is ``sha256(json.dumps(unique_id, sort_keys=True))`` (serialize.py).
    => HOOK 1: the histserv worker fill path (the ``_RemoteFillProxy.fill`` that
       today passes NO unique_id) MUST pass a deterministic key so a RETRIED worker
       replaying its fills is deduplicated server-side. Use histserv_unique_id().

  * Restart gap: ``entry.unique_ids`` is in-memory only; it is ``.clear()``-ed on
    snapshot-with-delete (service.py) and Flush does NOT persist it. So after a
    checkpoint reload the server has forgotten which fills it applied.
    => HOOK 2: native dedup covers worker-retry-while-server-alive ONLY. The
       server-restart case still needs the plan's PRE-script namespace reset
       (reset_namespace.py resets the (dataset, jobid, *) hists before a retry runs)
       + resume_planner.py (reruns workers that completed after the last checkpoint).
       Namespacing MUST therefore include jobid: keys are (dataset, jobid, var).
===============================================================================
"""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field


# ---- DAG node-name conventions (single source of truth) --------------------
SERVICE_NODE = "histserv"
SERVER_READY_NODE = "server_ready"
FINAL_NODE = "snapshot"


def worker_node_name(sample: str, jobid: int) -> str:
    return f"w_{sample}_{jobid}"


# ---- HOOK 1: idempotency key for histserv fills ----------------------------
def histserv_unique_id(dataset: str, jobid: int, var: str,
                       chunk_index: int, systematic: str = "nominal") -> list:
    """Deterministic idempotency key for a single remote fill.

    Passed as ``RemoteHist.fill(unique_id=histserv_unique_id(...))`` from the
    histserv worker fill path. histserv sha256-hashes the JSON of this value, so
    a worker that dies and is retried replays the *same* keys and the server drops
    the duplicates -> no double counting while the server stays alive (HOOK 1).

    ``jobid`` is part of both this key AND the server-side histogram namespace so
    the PRE-script reset (HOOK 2) can wipe exactly one worker's contribution on the
    server-restart path without touching sibling jobs. ``chunk_index`` must be a
    monotonic per-(dataset, jobid, var, systematic) fill counter inside the worker.
    """
    return [dataset, int(jobid), var, str(systematic), int(chunk_index)]


def namespace_key(dataset: str, jobid: int, var: str) -> tuple:
    """Server-side histogram namespace key (extends the prototype's (dataset, var)).

    The extra ``jobid`` is what makes retries resettable per HOOK 2 (plan 7.3).
    """
    return (dataset, int(jobid), var)


# ---- Phase 2/3 builder (hook skeleton) -------------------------------------
@dataclass
class SampleJobs:
    """One sample's contribution to the DAG: its worker nodes and merge inputs.

    ``jobdir`` is the in-container per-sample directory (``jobs_<tag>_<era>_<name>``)
    where output pickles land; ``jobdir_external`` is its host spelling (what
    condor's ``initialdir`` must use). ``script`` is the sample's rendered
    ``script.sh`` (external path) -- data vs MC bake different flags into it, so the
    executable is genuinely per-sample even though the submit *description* is shared.
    ``files`` is the resolved input-file list (one worker node per file).
    """

    name: str
    jobdir: str
    jobdir_external: str
    script: str
    transfer_files: list  # external paths already resolved
    files: list


@dataclass
class WorkflowDAG:
    """Builds one DAG per (tag, era) spanning all samples, rooted at the dagdir.

    Phase 2 (implemented): the non-histserv DAG -- shared ``worker.sub`` + one
    ``JOB``/``RETRY`` per input file + a ``FINAL`` local-universe merge node.
    Phase 3 (still hooks below): the SERVICE/server_ready/PRE-reset/snapshot pieces
    are added when ``histserv`` is set. ``build()`` keeps a clear seam for them
    (``_histserv_dag_lines`` / ``_final_node_lines``) so Phase 3 only flips behavior.

    See the module docstring for the node layout and the two double-fill hooks.

    Shared-vs-per-sample worker.sub decision (plan 6.1 left this open):
    ONE shared ``worker.sub`` is written. Phase 1's ``build_worker_submit`` bakes
    per-sample paths (executable, initialdir, log, transfer_input_files) into the
    description; here those four keys are rewritten to ``$(...)`` macros fed by each
    node's ``VARS`` (``script``, ``jobdir_ext``, ``transfer``). This honours "writes
    worker.sub once" while still letting each sample's own ``script.sh`` (which bakes
    data-vs-MC flags, so it is inherently per-sample) be the executable via a VAR.
    """

    tag: str
    era: str
    dagdir: str
    cfg: object = None          # a SubmissionConfig (any sample's) for the shared knobs
    samples: list = field(default_factory=list)   # list[SampleJobs]
    merge_dir: str = "."        # hist-merger.py --dir prefix (parent of the jobdirs)
    merge_outdir: str = "."     # hist-merger.py --outdir
    merge_script: str = "hist-merger.py"          # external path to hist-merger.py
    python_exe: str = "python3"                   # interpreter for the FINAL merge
    histserv: bool = False
    checkpoint_interval: int = 300  # seconds; shorter -> smaller resume-rerun set
    retries: int = 3

    # -- filenames inside the dagdir ----------------------------------------
    DAG_FILE = "workflow.dag"
    WORKER_SUB = "worker.sub"
    MERGE_SUB = "merge.sub"
    STATUS_FILE = "dag.status"

    def dag_path(self) -> str:
        return os.path.join(self.dagdir, self.DAG_FILE)

    # -- shared worker.sub (macro-parametrised) -----------------------------
    def _worker_sub_text(self) -> str:
        """Return the text of the single shared ``worker.sub``.

        Built from ``build_worker_submit(..., for_dag=True)`` (so ``max_retries`` is
        dropped -- DAGMan owns RETRY -- and ``arguments`` use ``$(jobid)``), then the
        four per-sample keys are rewritten to node-supplied ``$(...)`` macros. Lazily
        imports htcondor2 (bindings live only in the container); the pure-text DAG and
        merge assembly below do NOT need it, so they stay testable off-pool.
        """
        from .submit import build_worker_submit

        # Any sample's cfg carries the shared knobs; paths get overridden to macros.
        rep = self.samples[0]
        sub = build_worker_submit(self.cfg, rep.jobdir,
                                  transfer_files=rep.transfer_files, for_dag=True)
        sub["executable"] = "$(script)"
        sub["initialdir"] = "$(jobdir_ext)"
        sub["log"] = "$(jobdir_ext)/cluster.log"
        sub["transfer_input_files"] = "$(transfer)"
        return str(sub)

    def write_worker_sub(self) -> str:
        path = os.path.join(self.dagdir, self.WORKER_SUB)
        with open(path, "w") as fh:
            fh.write(self._worker_sub_text())
        return path

    # -- FINAL merge node ----------------------------------------------------
    def _merge_sub_text(self) -> str:
        """Local-universe FINAL merge node running hist-merger.py for every sample.

        FINAL runs even when the DAG fails, so the merge script must not fail the
        node. ``merge_wrapper.py`` runs hist-merger.py, then writes a
        ``MERGE_INCOMPLETE`` marker (and still exits 0) if DAGMan reports any failed
        node -- $(DAG_STATUS)/$(FAILED_COUNT) are threaded in as VARS.
        """
        wrapper = os.path.join(self.dagdir, "merge_wrapper.py")
        args = (
            f"{shlex.quote(wrapper)} "
            f"--merge-script {shlex.quote(self.merge_script)} "
            f"--python {shlex.quote(self.python_exe)} "
            f"--dir {shlex.quote(self.merge_dir)} "
            f"--outdir {shlex.quote(self.merge_outdir)} "
            f"--tag {shlex.quote(self.tag)} --era {shlex.quote(self.era)} "
            "--dag-status $(DAG_STATUS) --failed-count $(FAILED_COUNT)"
        )
        lines = [
            "universe = local",
            f"executable = {self.python_exe}",
            f"arguments = {args}",
            "should_transfer_files = NO",
            "getenv = True",
            f"output = {os.path.join(self.dagdir, 'merge.out')}",
            f"error = {os.path.join(self.dagdir, 'merge.err')}",
            f"log = {os.path.join(self.dagdir, 'merge.log')}",
            "queue",
            "",
        ]
        return "\n".join(lines)

    def write_merge_sub(self) -> str:
        path = os.path.join(self.dagdir, self.MERGE_SUB)
        with open(path, "w") as fh:
            fh.write(self._merge_sub_text())
        # Ship the merge wrapper alongside the sub.
        wrapper = os.path.join(self.dagdir, "merge_wrapper.py")
        with open(wrapper, "w") as fh:
            fh.write(MERGE_WRAPPER_SRC)
        os.chmod(wrapper, 0o755)
        return path

    # -- DAG text ------------------------------------------------------------
    def _worker_node_lines(self) -> list:
        worker_sub = os.path.abspath(os.path.join(self.dagdir, self.WORKER_SUB))
        lines: list = []
        for smp in self.samples:
            for i, fn in enumerate(smp.files):
                node = worker_node_name(smp.name, i)
                lines.append(f"JOB {node} {worker_sub}")
                transfer = ",".join(smp.transfer_files)
                lines.append(
                    f'VARS {node} jobid="{i}" jobfn="{fn}" sample="{smp.name}" '
                    f'script="{smp.script}" jobdir_ext="{smp.jobdir_external}" '
                    f'transfer="{transfer}"'
                )
                lines.append(f"RETRY {node} {self.retries}")
                # Phase-3 seam: PRE-reset per worker node (no-op until histserv).
                if self.histserv:
                    lines.append(self.write_pre_reset_script_line(smp.name, i))
        return lines

    def _histserv_dag_lines(self) -> list:
        """Phase-3 seam: SERVICE + server_ready nodes and the PARENT/CHILD gate.

        Empty in Phase 2 (non-histserv). Filling this in is the *only* structural
        change Phase 3 makes to the DAG body besides the PRE-reset lines and the
        snapshot FINAL node -- the worker/merge machinery is unchanged.
        """
        if not self.histserv:
            return []
        raise NotImplementedError(
            "Phase 3 hook: SERVICE histserv / JOB server_ready / PARENT server_ready "
            "CHILD w_* (see plan 7.2 and write_histserv_service_sub)."
        )

    def _final_node_lines(self) -> list:
        """FINAL node lines. Phase 2: plain hist-merger merge. Phase 3 swaps this for
        the snapshot_and_merge FINAL (seam: overridden when self.histserv)."""
        if self.histserv:
            raise NotImplementedError(
                "Phase 3 hook: FINAL snapshot snapshot.sub (see plan 7.2)."
            )
        merge_sub = os.path.abspath(os.path.join(self.dagdir, self.MERGE_SUB))
        return [f"FINAL merge {merge_sub}"]

    def _dag_text(self) -> str:
        # All paths in the DAG are absolute: DAGMan submitted via Submit.from_dag +
        # Schedd().submit does NOT chdir into the DAG's directory the way
        # condor_submit_dag does, so relative sub/status paths resolve against the
        # submit-time cwd (validated on the local v25 pool).
        status_file = os.path.abspath(os.path.join(self.dagdir, self.STATUS_FILE))
        lines: list = [
            f"# workflow.dag for tag={self.tag} era={self.era} "
            f"({'histserv' if self.histserv else 'local-hist'} mode)",
            f"NODE_STATUS_FILE {status_file} 60",
            "",
        ]
        lines += self._histserv_dag_lines()
        lines += self._worker_node_lines()
        lines.append("")
        lines += self._final_node_lines()
        lines.append("")
        return "\n".join(lines)

    def write_dag(self) -> str:
        path = self.dag_path()
        with open(path, "w") as fh:
            fh.write(self._dag_text())
        return path

    def build(self) -> str:
        """Assemble the DAG dir: worker.sub, merge.sub (+wrapper), workflow.dag.

        Returns the path to ``workflow.dag``. Does not submit -- see ``submit()``.
        """
        os.makedirs(self.dagdir, exist_ok=True)
        self.write_worker_sub()
        self.write_merge_sub()
        return self.write_dag()

    def submit(self, *, resume: bool = False, maxidle: int = 500, schedd=None):
        """Submit the DAG via ``htcondor2.Submit.from_dag`` + ``Schedd().submit``.

        Fresh submit passes ``force=True`` (overwrites prior products). ``resume``
        omits ``force`` so DAGMan picks up ``workflow.dag.rescue###`` and reruns only
        failed nodes (``force`` would delete the rescue files). Records the DAGMan
        cluster in the dagdir manifest. Returns the DAGMan cluster id.
        """
        from .config import ensure_condor_config
        ensure_condor_config()
        import htcondor2

        opts = {"maxidle": str(maxidle)}
        if not resume:
            opts["force"] = True
        sub = htcondor2.Submit.from_dag(self.dag_path(), opts)
        if schedd is None:
            from .submit import locate_schedd
            schedd = locate_schedd()
        result = schedd.submit(sub)
        cluster = result.cluster()
        self.write_manifest(cluster, resume=resume)
        return cluster

    def write_manifest(self, cluster: int, *, resume: bool = False) -> str:
        """Record the DAGMan cluster in ``<dagdir>/manifest.json`` (extends the
        Phase-1 per-sample manifest shape with DAG-level fields).

        On ``--resume`` the WorkflowDAG is constructed without ``samples`` (only the
        dagdir matters), so the existing manifest is merged into: the fresh-submit
        sample bookkeeping is preserved and the resume appends to ``resume_clusters``.
        """
        import time
        path = os.path.join(self.dagdir, "manifest.json")
        manifest = {}
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    manifest = json.load(fh)
            except (OSError, ValueError):
                manifest = {}

        manifest.update({
            "kind": "dag",
            "dagman_cluster": cluster,
            "tag": self.tag,
            "era": self.era,
            "histserv": self.histserv,
            "resume": resume,
            "dag_file": self.dag_path(),
            "node_status_file": os.path.abspath(
                os.path.join(self.dagdir, self.STATUS_FILE)),
            "submit_time": time.time(),
        })
        if self.samples or not resume:
            manifest.update({
                "n_samples": len(self.samples),
                "n_worker_nodes": sum(len(s.files) for s in self.samples),
                "samples": {s.name: len(s.files) for s in self.samples},
            })
        if resume:
            manifest.setdefault("resume_clusters", []).append(cluster)

        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        return path

    def write_histserv_service_sub(self) -> str:  # pragma: no cover - Phase 3
        """SERVICE node: histserv comes up BEFORE workers and is condor_rm'd only
        when the DAG finishes (i.e. after every worker has shipped data). On
        startup it reloads ckpt_latest before advertising readiness (restart path).
        """
        raise NotImplementedError("Phase 3 hook: histserv.sub (see plan 7.2).")

    def write_pre_reset_script_line(self, sample: str, jobid: int) -> str:  # Phase 3
        """PRE script per worker node (HOOK 2): resets the (sample, jobid, *) server
        namespace so a retried worker's partial fills are discarded on the
        server-restart path.  ``SCRIPT PRE w_<s>_<i> reset_namespace.py …``
        """
        raise NotImplementedError("Phase 3 hook: reset_namespace.py PRE script.")


def resume_edits_for_rescue(checkpoint_meta_path: str, node_status_path: str) -> list:  # Phase 3
    """HOOK 2 (resume): return DONE lines to un-mark in the rescue DAG.

    Compares each done-worker's completion time against ckpt_latest.meta.json; any
    worker that finished AFTER the last checkpoint has data not in the reload and
    must be rerun (its PRE reset makes the rerun safe). Phase-3 stub.
    """
    raise NotImplementedError("Phase 3 hook: resume_planner (see plan 7.3).")


def load_checkpoint_meta(path: str) -> dict:
    """Read ckpt_latest.meta.json ({"time":…, "flushed_hist_keys":[…]}). Implemented
    now because both the service checkpoint loop and resume planner need it."""
    with open(path) as fh:
        return json.load(fh)


# ---- FINAL merge wrapper (shipped into the dagdir by write_merge_sub) -------
# Runs hist-merger.py for every sample, then -- if DAGMan reports any failed node
# via $(DAG_STATUS)/$(FAILED_COUNT) -- writes a MERGE_INCOMPLETE marker while still
# exiting 0. FINAL nodes run even when the DAG fails; the marker (not a nonzero
# exit) is what signals a partial merge so the rescue path stays clean.
MERGE_WRAPPER_SRC = r'''#!/usr/bin/env python3
"""FINAL merge wrapper (generated by qawa.condor.dag). Do not edit by hand."""
import argparse
import os
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--merge-script", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--era", required=True)
    p.add_argument("--dag-status", default="")
    p.add_argument("--failed-count", default="")
    a = p.parse_args()

    cmd = [a.python, a.merge_script, "--dir", a.dir, "--outdir", a.outdir,
           "--tag", a.tag, "--era", a.era, "--force"]
    print("merge wrapper: running", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    print(f"merge wrapper: hist-merger exit={rc}", flush=True)

    def _as_int(s):
        try:
            return int(str(s).strip())
        except (TypeError, ValueError):
            return None

    failed = _as_int(a.failed_count)
    # DAG_STATUS: 0 == OK; anything else means the DAG did not fully succeed.
    status = _as_int(a.dag_status)
    incomplete = (failed is not None and failed > 0) or (status is not None and status != 0)
    marker = os.path.join(a.outdir, "MERGE_INCOMPLETE")
    if incomplete:
        with open(marker, "w") as fh:
            fh.write(f"dag_status={a.dag_status} failed_count={a.failed_count}\n")
        print(f"merge wrapper: wrote {marker} (partial merge)", flush=True)
    elif os.path.exists(marker):
        # A previous (failed) run left the marker; this run merged cleanly
        # (e.g. after a rescue-driven --resume), so clear it.
        os.remove(marker)
        print(f"merge wrapper: removed stale {marker}", flush=True)
    # FINAL must not mask the rescue file: always exit 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
