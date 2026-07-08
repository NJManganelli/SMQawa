# condor-spikes / RESULTS.md

Phase-0 spike testbed and recorded findings for the htcondor2 + DAGMan + histserv
migration (see `../HTCONDOR_PYTHON_PLAN.md`). Update the per-spike Accept lines as
each is run against the local pool and, later, LPC/lxplus.

## Local test pool (spike harness)

`../condor-docker/` stands up an HTCondor **v25** pool in Docker for S0.1 / S0.3:

* `cm` — central manager (COLLECTOR + NEGOTIATOR)
* `submit` — access point (SCHEDD); repo mounted at `/smqawa`
* `execute` ×3 — worker nodes (STARTD)
* `histserv` — long-running server node (profile `histserv`) — DAGMan SERVICE-node stand-in

Bring up / tear down:

```bash
cd ../condor-docker
./setup-condor-docker.sh            # cm + submit + 3 workers (pool-password auth)
./setup-condor-docker.sh --histserv # also the histserv service node
docker compose exec -u submituser submit python3 /smqawa/condor-docker/probe_submit.py  # S0.1
./teardown-condor-docker.sh
```

Images `htcondor/{cm,submit,execute}:25.0-el9` (override `CONDOR_VERSION`). Auth is a
shared pool password created once into the `secrets` volume by the setup script,
plus FS auth for co-located client->schedd so the submitter is a real unix user.

**Submit as `submituser`, not root.** Local tools authenticate via FS as the unix
user; root maps to the pool identity `condor@submit`, which the schedd rejects
under the HTCondor >= 23 user-record policy. `submituser` (uid 1000) is the image's
designated submitter and the schedd auto-adds its user record on first submit.

## Spike status

| id | question | status |
|---|---|---|
| S0.1 | htcondor2 submit+query from inside the container | **PASS on local pool** (htcondor2 25.0.11: submit -> idle -> running -> completed via `probe_submit.py`, as `submituser`). Re-run on LPC/lxplus where auth is FS/IDTOKENS/Kerberos. |
| S0.2 | worker↔server reachability (SERVICE-node universe) | pending (histserv node reachable on the bridge; decides local vs vanilla) |
| S0.3 | DAGMan SERVICE-node start/kill/rescue semantics | pending (use local pool) |
| S0.4 | condor_chirp `HistServAddress` from a worker | pending |
| **S0.5** | **histserv durability + double-fill primitives** | **DONE — see below** |

## Phase 1 acceptance (local pool portion) — PASS

`../condor-docker/probe_itemdata.py` exercises the real Phase-1 code on the running
v25 pool (run as `submituser`, `PYTHONPATH=/smqawa/src`):

* **(A) preserved-knob parity** — `build_worker_submit()` reproduces every knob from
  the old `condor_TEMPLATE`: `request_disk=10000000`, `max_retries=3`,
  `requirements = Machine =!= LastRemoteHost`, `MY.SingularityImage` (cvmfs),
  `arguments = $(ProcId) $(jobfn)`, `on_exit_remove ... ExitCode==0`,
  `should_transfer_files=YES`, `WhenToTransferOutput=ON_EXIT_OR_EVICT`. The
  `for_dag=True` variant correctly drops `max_retries` (DAGMan RETRY owns it) and
  uses `$(jobid)` VARS. PASS.
* **(B) itemdata + manifest** — `submit_sample()` submitted 3 files -> 3 jobs via
  itemdata; all completed across the 3 workers; `manifest.json` written with the
  cluster id, `n_jobs=3`, and the procid->infile map. PASS.

Still requires the LPC/lxplus AP (not reproducible locally): the full dataset
parity test — old vs new submitter producing identical `histogram_<N>.pkl.gz`
sets — because it needs dasgoclient, real NanoAOD, the CVMFS SingularityImage, and
the coffea payload venv.

## Phase 2 acceptance (local pool portion) — PASS

`../condor-docker/probe_dag.py` exercises the real `WorkflowDAG` code (build +
`Submit.from_dag` + resume) on the v25 pool with synthetic worker/merge payloads
(run as `submituser`, `PYTHONPATH=/smqawa/src`, DAG root under
`/home/submituser` — see the /tmp gotcha below). Three scenarios, all PASS:

1. **Fresh submit** — 2 samples x 3 files -> 6 worker nodes + FINAL merge; all ran,
   per-sample merged outputs produced, `manifest.json` records the DAGMan cluster,
   `dag.status` NODE_STATUS_FILE shows every node `STATUS_DONE`.
2. **Kill a worker mid-run** — `condor_rm` of a running node job -> DAGMan logged
   the retry, node reran, DAG completed, both samples merged.
3. **Exhaust retries -> rescue -> resume** — always-failing payload exhausted
   `RETRY 3` -> DAG failed, `workflow.dag.rescue001` written (3 nodes premarked
   DONE, 3 failed), FINAL merge still ran and wrote `MERGE_INCOMPLETE`; after
   fixing the payload, `submit(resume=True)` (`Submit.from_dag` WITHOUT `force`)
   picked up the rescue ("Number of pre-completed nodes: 3"), reran ONLY the
   failed nodes + FINAL, exited 0, and the merge wrapper cleared the stale
   `MERGE_INCOMPLETE` marker.

Recorded Phase-2 findings (they shaped the implementation):

* **DAGMan submitted via `Submit.from_dag` + `Schedd().submit` does NOT chdir into
  the DAG's directory** (unlike `condor_submit_dag`): relative `JOB <node> worker.sub`
  and `NODE_STATUS_FILE dag.status` paths resolve against the *submit-time cwd* and
  fail/land elsewhere. All paths inside the generated DAG are therefore absolute.
* **`on_exit_remove` must be dropped in DAG mode** (now done by
  `build_worker_submit(for_dag=True)`): with the old
  `(ExitBySignal==False)&&(ExitCode==0)` expression a nonzero-exit job is re-queued
  by the schedd forever (observed as endless ULOG_JOB_EVICTED/idle cycles), so
  DAGMan never sees the node fail and RETRY/rescue never trigger. In DAG mode retry
  ownership lives exclusively in DAGMan's `RETRY`.
* **DAGMAN_USE_STRICT gotcha:** a DAG whose node log lands in `/tmp` is a *fatal*
  warning under the default strict setting — keep DAG dirs out of `/tmp`.

Still requires the LPC/lxplus AP: the real coffea payload end-to-end (dasgoclient,
CVMFS image, venv) and confirming `universe=local` FINAL-merge behavior on a real
AP where the schedd runs outside the analysis container.

## S0.5 — histserv double-fill / hash-checking (DONE, v0.1.9)

**Question the user asked: does histserv implement the right hash-checking to avoid
double-filling, and what hooks must be utilized?**

**Answer: YES, natively — but it is not wired in our prototype, and it does not
survive a server restart. Two hooks must be utilized.**

Evidence (paths under the histserv-lab venv site-packages):

* `RemoteHist.fill(unique_id=...)` and `fill_many(unique_id=...)` accept an
  idempotency key (`client.py:359,409`).
* The server hashes it: `serialize_unique_id = sha256(json.dumps(key, sort_keys=True))`
  (`serialize.py:188`).
* Per-histogram dedup set `entry.unique_ids: set[bytes]` (`service.py:132`); a fill
  whose id is already present is **rejected** with `grpc ALREADY_EXISTS`
  (`service.py:242-260`, `_reject_duplicate_unique_id` / `_remember_unique_id`).
* **Restart gap:** the set is in-memory only, is `.clear()`-ed on
  snapshot-with-delete (`service.py:620`), and `Flush` does NOT persist it. So after
  a checkpoint reload the server has forgotten which fills it applied.
* **Prototype gap:** `histserv-lab/wz_histserv_processor.py::_RemoteFillProxy.fill`
  passes **no** `unique_id` today → there is currently **zero** double-fill
  protection in the histserv path.

### Hooks that MUST be utilized (implemented as code hooks in `src/qawa/condor/dag.py`)

1. **HOOK 1 — pass a deterministic `unique_id` on every remote fill.** Use
   `qawa.condor.dag.histserv_unique_id(dataset, jobid, var, chunk_index, systematic)`.
   This makes a **retried worker (server still alive)** replay identical keys → the
   server drops the duplicates. `chunk_index` must be a monotonic per-(dataset,
   jobid, var, systematic) counter inside the worker. Wire this into the port of
   `_RemoteFillProxy.fill` (plan 7.1 →
   `src/qawa/process/wztau2lnu_inclusive_histserv.py`).

2. **HOOK 2 — jobid-namespaced hists + PRE-script reset + resume planner.** Because
   HOOK 1 is lost on restart, the **server-restart** path relies on:
   * namespacing server hists by `(dataset, jobid, var)` — `dag.namespace_key(...)`
     (extends the prototype's `(dataset, var)`);
   * a DAGMan `SCRIPT PRE` per worker that resets the `(dataset, jobid, *)` hists
     before a retry runs (`reset_namespace.py`, Phase 3) — so a rerun after reload
     starts clean;
   * `resume_planner` re-running workers that completed *after* the last checkpoint
     (`dag.resume_edits_for_rescue`, Phase 3), since their data is not in the reload.

Together: HOOK 1 covers the common worker-retry case cheaply; HOOK 2 covers server
death/checkpoint-reload. Namespacing by `jobid` is the linchpin of both.

### histserv SERVICE-node lifecycle hooks (user request)

The long-running server must **come up before workers and stay until every worker
has shipped its data, across restarts.** Encoded as hooks in `dag.py`
(`WorkflowDAG.write_histserv_service_sub`, node names `SERVICE_NODE`/`FINAL_NODE`):

* `SERVICE histserv` starts before/with the first jobs; DAGMan `condor_rm`s it only
  when the DAG finishes (after all workers ship) — confirm exact semantics in S0.3.
* On startup the service **reloads `ckpt_latest` before advertising readiness**
  (restart path); it runs a `--checkpoint-interval` flush loop (default 300 s).
* `server_ready` (universe=local) gates workers on address + hist registration;
  `FINAL snapshot` always runs (even on failure) to snapshot successful namespaces
  into merged pickles.

Open decision (S0.3): what happens if the *service* job itself dies mid-DAG — does
the DAG keep running, and can `ABORT-DAG-ON` attach? Plan 7.3 currently assumes
retries-then-rescue-then-`--resume`; confirm and record here.
