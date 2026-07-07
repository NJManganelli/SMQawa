"""DAGMan orchestration + histserv SERVICE-node hooks (Phases 2 & 3).

This module is intentionally a **hook skeleton**: Phase 1 (config/datasets/proxy/
submit) is implemented and usable now; the DAG builder and histserv wiring below
are stubs with precise contracts so Phases 2-3 drop in without re-litigating the
design. The one thing implemented eagerly is :func:`histserv_unique_id` -- the
concrete double-fill-avoidance hook (see the module notes) -- because it is small,
testable, and must be threaded through the worker fill path the moment histserv
mode exists.

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
from dataclasses import dataclass


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
class WorkflowDAG:
    """Builds one DAG per (tag, era). Phase 2/3 -- not yet implemented.

    See the module docstring for the node layout and the two double-fill hooks
    that MUST be honored when the histserv path is filled in.
    """

    tag: str
    era: str
    dagdir: str
    histserv: bool = False
    checkpoint_interval: int = 300  # seconds; shorter -> smaller resume-rerun set

    def build(self) -> str:  # pragma: no cover - Phase 2
        raise NotImplementedError(
            "WorkflowDAG.build is a Phase-2 hook. Implementation must: write shared "
            "worker.sub via str(build_worker_submit(..., for_dag=True)); emit one "
            "JOB + RETRY per input file; add NODE_STATUS_FILE; and, when histserv is "
            "set, add the SERVICE/server_ready/PRE-reset/FINAL-snapshot pieces and "
            "thread histserv_unique_id() into the worker fill path (see hooks)."
        )

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
