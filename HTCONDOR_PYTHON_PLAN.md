# HTCondor Python-bindings migration + histserv DAG hook — implementation plan

Branch: `htcondor_python`. Execution plan written for a coding agent (target: Opus 4.8);
every step names files, functions, and acceptance criteria. Read this whole file before
starting — Phase 3's design constraints (retry idempotency, readiness, checkpointing)
back-propagate into choices made in Phases 1–2.

---

## 1. Context — what exists today

| artifact | role | key facts |
|---|---|---|
| `SMQawa/brewer-htcondor-inclusive.py` | submitter | Python writes a bash `script.sh` (from `script_TEMPLATE`) + `condor.sub` (from `condor_TEMPLATE`) per sample into `jobs_<tag>_<era>_<sample>/`, resolves files with `dasgoclient`, then runs `condor_submit` via `subprocess` |
| `SMQawa/call_host.zsh` / `.sh` | host escape hatch | analysis runs inside an Apptainer container that has **no condor CLI**; `condor_*` commands are proxied to the host through named pipes (`$HOSTPIPE`/`$CONTPIPE`/`$EXITPIPE`). The submitter's `condor_submit` call rides this. |
| `SMQawa/monitor.py` | monitoring/resubmit | parses `condor_q -nobatch` text output, diffs `inputfiles.dat` against produced `histogram_<N>.pkl.gz`, hand-edits copies of `condor.sub` per failed job (`condor_resub_<jid>.sub`) and resubmits |
| worker payload | `brewer-remote-inclusive.py --jobNum=$1 --infile=$2 ...` | wrapper `script.sh` sources the venv from `$INSTALL_LOC_EXTERNAL/.env`, exports the proxy, runs the payload, and fails the job if `histogram_$1.pkl.gz` is missing (exit 1 → condor `max_retries=3` + `on_exit_remove` handles retry) |
| `SMQawa/hist-merger.py` | merge | merges per-job `histogram_<N>.pkl.gz` into per-sample merged pickles |
| `../histserv-lab/` | prototype | `wzinclusive_processor_histserv` subclass validated bin-for-bin vs stock. Key API: `client.init(template) → RemoteHist` (picklable), `connection_info()` dict for reconnect, `snapshot_histos(delete_from_server=...)`, `per_dataset` mode (one server hist per `(dataset, var)`), `skip_empty_fills=True`. Server: `histserv --port <p>` (gRPC). `flush(destination=...)` can checkpoint to disk. |

Current submit-description facts to preserve (from `condor_TEMPLATE`):
`universe=vanilla`, `request_disk=10000000`, `arguments = $(ProcId) $(jobfn)`,
`queue jobfn from inputfiles.dat`, `transfer_input_files` = external path of
`brewer-remote-inclusive.py`, `should_transfer_files=YES`,
`WhenToTransferOutput=ON_EXIT_OR_EVICT`, `on_exit_remove=(ExitBySignal==False)&&(ExitCode==0)`,
`max_retries=3`, `requirements = Machine =!= LastRemoteHost`,
`+SingularityImage=/cvmfs/unpacked.cern.ch/...`, `+JobFlavour` (lxplus-ism; ignored at LPC).

## 2. Target architecture

```
brewer-dag-inclusive.py  (runs in the container venv, htcondor2 bindings)
  ├─ resolve datasets (dasgoclient)            [unchanged logic, isolated]
  ├─ build per-tag DAG under jobs_<tag>_<era>/dag/
  │    SERVICE  histserv        histserv.sub        # long-running server, up before workers,
  │    JOB      server_ready    server_ready.sub    # local universe: wait for address + register hists
  │    JOB      w_<sample>_<i>  worker.sub  VARS ...# one node per input file → per-node RETRY
  │    PARENT   server_ready CHILD <all workers>
  │    FINAL    snapshot        snapshot.sub        # snapshot_histos + hist-merger; runs even on failure
  └─ htcondor2.Submit.from_dag(...) → Schedd().submit(...)
resume = resubmit same DAG → DAGMan rescue file reruns only failed nodes
```

Non-histserv mode is the same DAG minus `SERVICE`/`server_ready`/snapshot-from-server
(FINAL node just runs `hist-merger.py`), so both modes share one code path.

## 3. Design decisions (locked in up front)

1. **`htcondor2` + `classad2` modules only** (the version-2 bindings; the classic
   `htcondor` module is deprecated and not to be used). Minimum HTCondor 25 on the
   access point (AP); pip package `htcondor>=25` in the venv provides `htcondor2`.
2. **Keep the bash wrapper `script.sh`.** The *worker payload* legitimately needs a
   shell wrapper (venv activation, proxy export, env dump). What we eliminate is
   Python writing *condor submit files* and shelling out to `condor_submit`/`condor_q`.
   Move the wrapper template into package data (`src/qawa/condor/templates/worker.sh`)
   rather than a module-level string.
3. **DAG files are declarative artifacts, not "scripts writing scripts".** Generate
   `workflow.dag` + per-role `.sub` content from `htcondor2.Submit` objects
   (`str(Submit)` round-trips) with a thin writer in `src/qawa/condor/dag.py`.
   (`htcondor.dags` python module exists only for the classic bindings — do not
   depend on it; verify in Phase 0 whether `htcondor2` has grown an equivalent, and
   use it if so.)
4. **One node per input file** (not one cluster per sample) so `RETRY` granularity
   matches today's per-file jobs and rescue DAGs give free resubmission. Throttle
   with `Submit.from_dag(dag, {"maxidle": 500, "maxjobs": ...})` instead of relying
   on cluster batching.
5. **histserv retry idempotency via per-job namespacing** (see §7.3). A worker that
   dies mid-file has already streamed fills to the server; blind retry double-counts.
   Every worker fills hists registered under a `(dataset, jobid)` namespace and a
   retry **resets that namespace first**. This is the single most important
   correctness constraint in the whole plan.

## 4. Phase 0 — spikes / validation (do first, ~each is a small standalone script under `SMQawa/condor-spikes/`)

Each spike has a binary acceptance criterion. Record results in
`condor-spikes/RESULTS.md`. If a spike fails, the fallback is named.

- **S0.1 — bindings submit from inside the container.**
  `pip install 'htcondor>=25'` into the venv. From inside the Apptainer session on
  the AP (LPC first, then lxplus), run a probe that does
  `htcondor2.Schedd().submit(Submit({"executable": "/bin/sleep", "arguments": "60"}))`
  and then `schedd.query(constraint=f"ClusterId=={r.cluster()}")`.
  Likely blockers and fixes: the container must see the host condor config and
  auth — bind-mount `/etc/condor` (add to `APPTAINER_BIND` in `bootstrap.zsh` /
  shell scripts) or set `CONDOR_CONFIG=ONLY_ENV` plus `_CONDOR_SCHEDD_HOST` /
  `_CONDOR_COLLECTOR_HOST`; auth is FS on LPC (same host, needs `/tmp` visibility)
  and IDTOKENS/Kerberos on lxplus (bind `~/.condor/`).
  *Accept:* job submitted, queried, and completes; from inside the container; no
  `call_host` involvement. *Fallback:* keep `call_host condor_submit` as a shim
  behind `--submit-via={bindings,call_host}` (implemented as one function swap in
  `submit.py`), but all file generation still moves to Phase 1 structure.
- **S0.2 — worker→server network reachability.** Submit a trivial gRPC/TCP
  listener job (or histserv itself) to a worker node; have (a) a second worker job
  and (b) the AP connect to its `host:port`. Test both placements:
  server-on-worker-node (vanilla universe) and server-on-AP (**local universe**).
  *Accept:* at least one placement where workers can connect. *Decision output:*
  which universe `histserv.sub` uses. (Expectation: at LPC worker→AP connectivity
  is more likely than worker→worker; if only AP works, the SERVICE node becomes a
  `universe=local` job on the AP — DAGMan supports that.)
- **S0.3 — DAGMan SERVICE-node semantics on the site's version.** Tiny DAG:
  SERVICE node = `sleep`-forever wrapper, two JOB nodes, FINAL node. Confirm:
  service starts before/with first jobs, is `condor_rm`'d when the DAG finishes,
  rescue-DAG behavior when a JOB node fails, and what happens when the *service*
  job dies mid-DAG (does the DAG keep running? can `ABORT-DAG-ON` be attached?).
  *Accept:* documented answers to those four questions in RESULTS.md — Phase 3's
  server-restart story depends on them.
- **S0.4 — condor_chirp from a worker.** In a probe job with `MY.WantIOProxy=true`,
  run `condor_chirp set_job_attr HistServAddress '"host:port"'` and read it back
  from the AP via `schedd.query`. *Accept:* attribute visible. *Fallback:* server
  writes its address to EOS via `xrdcp` and `server_ready` polls that path.
- **S0.5 — histserv durability primitives.** In histserv-lab: exercise
  `flush(destination=...)`, kill the server, restart, and confirm reload/reconnect
  semantics (does `RemoteHist.from_connection_info` survive a server restart at the
  same address? does flush write everything needed to resume?). Also verify a
  "reset/delete one histogram" operation exists (needed for §7.3 retry reset;
  `snapshot(delete_from_server=True)` proves deletion exists — confirm it's usable
  as a standalone reset). *Accept:* a written recipe: checkpoint → kill → restart
  → reload → clients keep filling.

## 5. Phase 1 — bindings-based submitter with behavior parity (no DAG yet)

New package `src/qawa/condor/`:

```
src/qawa/condor/
├── __init__.py
├── config.py      # SubmissionConfig dataclass: analysis, tag, era, isMC, zzdd,
│                  #   split_by_charge, queue, executor, images, install locs, proxy path
├── datasets.py    # dasgoclient resolution (lift lines 192–215 of the old brewer verbatim),
│                  #   sample-name mangling, inputfiles.dat read/write
├── proxy.py       # voms proxy check/renew (lift lines 122–156), returns proxy path
├── submit.py      # build_worker_submit(cfg, jobdir) -> htcondor2.Submit
│                  # submit_sample(schedd, sub, files) -> SubmitResult (itemdata path)
├── dag.py         # Phase 2
├── monitor.py     # Phase 4 (bindings-based status/retry)
└── templates/worker.sh   # the old script_TEMPLATE, unchanged semantics
```

Steps:

1. **`submit.py::build_worker_submit`** returns `htcondor2.Submit` built from a
   dict mirroring `condor_TEMPLATE` exactly, with `+X` spellings converted:
   `"MY.SingularityImage": f'"/cvmfs/.../{coffea_image}"'`,
   `"MY.JobFlavour": f'"{queue}"'`. Keep `arguments = "$(ProcId) $(jobfn)"`.
2. **Itemdata instead of `queue ... from`:**
   `schedd.submit(sub, itemdata=iter([{"jobfn": f} for f in files]))`.
   Still *also* write `inputfiles.dat` (monitoring, resume, and humans rely on it).
3. **Manifest.** After submit, write `jobs_.../manifest.json`:
   `{cluster, schedd_name, n_jobs, files: {procid: infile}, tag, era, sample,
   submit_time, user_log}`. Set `log = <jobdir>/cluster.log` (one shared user log
   per sample) — this is what `JobEventLog` monitoring keys off in Phase 4.
4. **New entry point `brewer-htcondor2-inclusive.py`** with the *same CLI* as the
   old script (argparse block lifted verbatim, plus `--submit-via`). The old
   `brewer-htcondor-inclusive.py` is left untouched until Phase 2 is validated,
   then deleted in Phase 5.
5. **Drop from the old script:** `subprocess.Popen(condor_submit ...)`, the
   shell-sourcing dance (`to_source`, `captured_env`), `jobs_dir_external`
   path-remapping for the submit file (bindings submit from wherever Python runs;
   only paths *inside* the submit description still need external-path spelling —
   keep the `INSTALL_LOC`→`INSTALL_LOC_EXTERNAL` translation as a helper in
   `config.py` and unit-test it).

*Accept (parity test):* for one small MC dataset and one data dataset, submit with
old and new submitters (different tags); jobs run, produce identical
`histogram_<N>.pkl.gz` sets; `condor_q`-visible ads match on the preserved knobs
(`RequestDisk`, retries, requirements, SingularityImage).

## 6. Phase 2 — DAGMan orchestration (still local-hist workers)

1. **`dag.py::WorkflowDAG`** — builds one DAG **per (tag, era)** spanning all
   samples in the input list, rooted at `jobs_<tag>_<era>/dag/`:
   - writes `worker.sub` once (shared by all worker nodes) using
     `str(build_worker_submit(...))` + `arguments = "$(jobid) $(jobfn)"`;
     per-node `VARS w_<sample>_<i> jobid="<i>" jobfn="<file>" sample="<sample>"`.
     Worker `initialdir` stays the per-sample jobdir so output pickles land where
     `hist-merger.py` expects them.
   - `RETRY w_<sample>_<i> 3` for every worker node (replaces `max_retries` in the
     submit file — remove it there to keep retry ownership in one place: DAGMan).
   - `FINAL merge merge.sub` — local-universe job running `hist-merger.py
     --dir=... --tag=... --era=...` for every sample; a FINAL node runs even when
     the DAG fails, so partial merges are produced and the rescue path stays clean.
     Guard: merge script exits 0 but writes `MERGE_INCOMPLETE` marker if any
     worker node failed (query `$DAG_STATUS` via the FINAL node's
     `$(DAG_STATUS)`/`$(FAILED_COUNT)` VARS — DAGMan provides both).
   - `NODE_STATUS_FILE dag.status 60` for cheap machine-readable progress.
2. **Submission:** `sub = htcondor2.Submit.from_dag("workflow.dag",
   {"force": True, "maxidle": 500})`; `Schedd().submit(sub)`. Record DAGMan
   cluster in `manifest.json`.
3. **Resume story (replaces `monitor.py --resubmit`):** rerunning
   `brewer-dag-inclusive.py --resume --tag=... --era=...` calls `Submit.from_dag`
   on the same DAG file; DAGMan auto-detects `workflow.dag.rescue###` and reruns
   only failed nodes. `--force` must NOT be passed on resume (it deletes rescue
   files) — make `--resume` and fresh-submit mutually exclusive in the CLI.
4. New entry point `brewer-dag-inclusive.py` (same CLI + `--dag`, `--resume`,
   `--histserv` for Phase 3). Phase 1's non-DAG path stays available via
   `brewer-htcondor2-inclusive.py` for one-off debugging.

*Accept:* full small-scale production (2 samples × ~5 files) via DAG; kill one
worker mid-run (`condor_rm` a node job) → node retries; force one job to exhaust
retries (bogus infile) → DAG fails, rescue file appears, `--resume` reruns only
that node; FINAL merge produces merged pickles both times.

## 7. Phase 3 — histserv service integration

### 7.1 Prerequisite refactor in SMQawa (small, upstreamable)

- Factor histogram-template construction out of processor `__init__` so
  registration doesn't need the full correction stack:
  `wzinclusive_processor.histogram_templates(era, ...) -> dict[str, hist.Hist]`
  (classmethod or module function extracted from the existing `build_histos`
  lambda contents). The `server_ready` registration step (below) runs on the AP
  and must not require `/cvmfs` correction loading. Stock processor behavior is
  unchanged (its `build_histos` now calls the classmethod).
- Move `histserv-lab/wz_histserv_processor.py` into
  `src/qawa/process/wztau2lnu_inclusive_histserv.py` (guarded import of
  `histserv`), keeping `per_dataset` mode, `skip_empty_fills=True`,
  `connection_info` reconnect, `snapshot_histos`. **Extend the namespace key**
  from `(dataset, var)` to `(dataset, jobid, var)` — see 7.3. Add
  `histserv>=<pinned>` to an extras group in `pyproject` (`qawa[histserv]`).
- `brewer-remote-inclusive.py`: add `--histserv-connection=<path to json>` and
  `--histserv-jobid=<N>`. When set: construct the histserv processor with
  `connection_info=json.load(...)` filtered to this job's namespace, run the
  runner, then instead of pickling runner output, write a **success sentinel**
  `filled_<jobNum>.done` (the wrapper's existence check switches from
  `histogram_$1.pkl.gz` to the sentinel in histserv mode). No per-job pickle.

### 7.2 New DAG pieces (in `dag.py`, enabled by `--histserv`)

```
SERVICE histserv      histserv.sub
JOB     server_ready  server_ready.sub      # universe=local on the AP
PARENT  server_ready  CHILD w_*             # workers gated on readiness+registration
FINAL   snapshot      snapshot.sub          # replaces plain merge FINAL
```

- **`histserv.sub`** — placement per S0.2 (expected: `universe=local` on the AP;
  else vanilla on a worker node with `MY.WantIOProxy=true`). Executable
  `histserv_service.sh`: pick a free port, start `histserv --port $PORT`,
  advertise `host:port` (chirp job-ad attr `HistServAddress` per S0.4, or
  local-universe: just write `<dagdir>/histserv_address.txt` — local universe
  runs on the AP filesystem, no chirp needed), then run the **checkpoint loop**:
  every `--checkpoint-interval` (default 300 s) call `flush(destination=
  <dagdir>/checkpoints/ckpt_latest)` (atomic: write tmp + rename), stamping
  `ckpt_latest.meta.json` with `{"time": ..., "flushed_hist_keys": [...]}`.
  On startup, if a checkpoint exists, **reload it before advertising readiness**
  (recipe from S0.5) — this is the server-resume path.
- **`server_ready.sub`** (`universe=local`) — script `server_ready.py`:
  1. poll for the address (file or chirped attr) with timeout (default 30 min →
     exit 1 → `RETRY server_ready 2`);
  2. health-check: gRPC connect + trivial round-trip;
  3. **register** all hists: for each `(sample, jobid)` in the manifest and each
     var from `histogram_templates(era)`, `client.init(...)` — *unless* resuming
     and the key is already on the server (reload from checkpoint covers it);
     idempotency rule: try reconnect-first, init-on-missing;
  4. write `<dagdir>/connection_info.json` (the prototype's `connection_info()`
     dict, extended with the `(dataset, jobid, var)` keying and the address).
  Workers receive it via `transfer_input_files = <dagdir>/connection_info.json`
  — safe because file transfer happens at worker *job start*, and DAG ordering
  guarantees `server_ready` finished writing it first.
- **`worker.sub` deltas** (histserv mode): add the two new payload flags, the
  connection-info transfer, and a **PRE script** per worker node:
  `SCRIPT PRE w_<s>_<i> reset_namespace.py <connection_info> <sample> <i>` —
  resets/deletes the `(sample, i, *)` server hists (no-op if empty). PRE scripts
  run on the AP, so this needs only the venv + network to the server. This makes
  retries idempotent (see 7.3). Success = sentinel file, so
  `on_exit_remove`/DAG RETRY semantics are unchanged.
- **`snapshot.sub`** (FINAL, `universe=local`) — script `snapshot_and_merge.py`:
  1. read `dag.status` / `$(FAILED_COUNT)`: build the set of **successful** jobids
     per sample;
  2. `snapshot_histos()` restricted to successful `(sample, jobid)` namespaces;
     sum over jobid → `{sample: {var: hist.Hist}}`; write per-sample
     `merged-histogram-...pkl.gz` directly (bypasses `hist-merger.py`'s
     per-job-pickle globbing; reuse its output naming so DCTools is untouched);
  3. if any workers failed: do NOT `delete_from_server`; force a final
     `flush(...)` checkpoint, write `SNAPSHOT_PARTIAL.json` listing missing
     jobids, exit 0 (FINAL must not mask the rescue file);
  4. if all succeeded: snapshot with `delete_from_server=True`, write
     `SNAPSHOT_COMPLETE`, optionally also emit per-job-shaped pickles for
     backward-compat debugging (`--emit-perjob-pickles` flag, default off).

### 7.3 Failure & resume semantics (the core of the hook)

Failure matrix and the mechanism that handles each:

| failure | handling |
|---|---|
| worker dies mid-fill | DAG `RETRY` reruns node; **PRE script resets the `(sample, jobid)` namespace** so partial fills from the dead attempt are discarded → no double counting. This is why namespacing includes `jobid`. |
| worker exhausts retries | DAG fails → rescue file; FINAL snapshot still runs, checkpoints server state, reports partial. `--resume` reruns only failed workers. |
| histserv server dies mid-DAG | workers start failing fill RPCs → they exit non-zero → retries begin failing too. Per S0.3: if SERVICE-node death doesn't halt the DAG, cap damage with worker-side fail-fast (bounded gRPC retry ~2 min then exit 75) and DAG-level `ABORT-DAG-ON w_* 75` is **not** used (any worker would kill the DAG); instead rely on retries exhausting → rescue. Data since last flush is lost **only for jobs that completed in that window** — the resume planner (below) handles them. |
| resume (`--resume --histserv`) | rescue DAG reruns `server_ready` dependents that failed. Server SERVICE node starts fresh → reloads `ckpt_latest`. **Gap:** workers that succeeded *after* the last flush have data not in the checkpoint. `resume_planner.py` (called by `--resume` before submitting): compare each done-worker's sentinel/NodeStatus completion time against `ckpt_latest.meta.json` time; for workers finished after the checkpoint, rewrite their `DONE` status in the rescue file back to not-done (rescue files are line-edited: remove/comment the `DONE w_<s>_<i>` line) so DAGMan reruns them; their PRE reset makes the rerun safe. |
| AP reboot / DAGMan itself dies | `condor_submit_dag`-style recovery: resubmitting the DAG picks up the `.dag.lock`/rescue state; document `--resume` as the single user action for *every* interruption class. |
| stale server hists from an abandoned tag | `brewer-dag-inclusive.py --histserv --force` runs a namespace sweep (reset all `(sample, *, *)` keys for this tag's samples) before fresh registration. |

Checkpoint-interval tradeoff: shorter interval → smaller resume-rerun set;
default 300 s, flag `--checkpoint-interval`.

### 7.4 sumw

`coffea_sumw` stays file-local per job (as noted in `HISTSERV_PLAN.md`, it is
independent of the hist backend). In histserv mode each worker still pickles the
tiny sumw-only payload as `sumw_<jobNum>.pkl.gz`; `snapshot_and_merge.py` merges
these classically. (Do not push sumw through histserv — it's a scalar accumulator
and per-job files make the successful-jobid bookkeeping trivial.)

*Accept (Phase 3):*
1. 2-sample × 5-file DAG in histserv mode reproduces the Phase-2 (local-hist)
   merged pickles **bin-for-bin** (reuse the comparison logic from
   `histserv-lab/run_inc_wz_histserv.py::compare`).
2. Kill one worker mid-run → retried node's namespace shows no double counting
   (assert merged yields equal the clean run).
3. Kill the histserv server mid-run → DAG fails → `--resume` → final merged
   output still bin-for-bin identical to the clean run (this exercises checkpoint
   reload + resume-planner rerun of post-checkpoint completions).
4. `--resume` after a worker exhausts retries reruns only that node.

## 8. Phase 4 — monitoring: retire `monitor.py` text-parsing

- `src/qawa/condor/monitor.py`: progress from (a) `htcondor2.JobEventLog` over the
  per-sample `cluster.log` / DAGMan's `.nodes.log`, (b) the DAG `NODE_STATUS_FILE`,
  (c) `schedd.query(projection=[...])` for live states — no `condor_q -nobatch`
  parsing, no `call_host`.
- New CLI `monitor2.py --tag --era [--watch]`: table of per-sample
  done/running/idle/held/failed + histserv server address & last-checkpoint age in
  histserv mode. (Optional: Rich table — keep it dependency-light otherwise.)
- Resubmission subcommand is deliberately **not** reimplemented: print the
  `brewer-dag-inclusive.py --resume` invocation instead.
- Keep old `monitor.py` for the legacy submitter until Phase 5.

## 9. Phase 5 — cleanup & docs

- Delete `brewer-htcondor-inclusive.py` (and `brewer-htcondor.py` if the
  non-inclusive flow migrates too), old `monitor.py`; drop the `condor_submit`
  path from `call_host` docs (call_host stays for `eos*` and interactive use).
- Update `CLAUDE.md` HTCondor-batch section and `bootstrap.zsh` (condor config /
  token bind mounts from S0.1; `pip install 'htcondor>=25'` and `qawa[histserv]`
  in the venv build).
- Fold `condor-spikes/RESULTS.md` conclusions into this file's §4 as recorded
  decisions; move histserv-lab prototype notes reference into SMQawa docs.

## 10. Risks / open questions (resolve via Phase 0, do not guess)

1. **Bindings auth from inside Apptainer** (S0.1) — highest-risk item; the
   `--submit-via=call_host` shim is the escape hatch and keeps every later phase
   intact (only the final `Schedd().submit` call differs; `Submit` objects
   serialize to submit files for the shim path).
2. **Worker↔server connectivity** (S0.2) decides SERVICE-node universe; if
   *neither* placement is reachable, histserv mode is blocked at that site —
   document and stop Phase 3 for that site (DAG migration Phases 1–2 still land).
3. **SERVICE-node kill/retry semantics** (S0.3) — if DAGMan does not restart a
   dead service node, the "server dies" row of §7.3 degrades to
   rescue-then-resume only (acceptable; note it).
4. **histserv reset/delete + flush-reload API details** (S0.5) — the PRE-reset and
   checkpoint designs assume both exist; if reset-by-key is missing, fallback is
   per-attempt namespacing `(sample, jobid, attempt, var)` with attempt number
   from DAGMan's `$(RETRY)` VARS and snapshot keeping only the last attempt.
5. **Scale:** per-(dataset, jobid, var) hists multiply server hist count by
   n_jobs (~10³–10⁴ hists × 54 vars). Prototype showed histserv handles many
   hists, but memory-check at realistic scale early in Phase 3 (one full sample);
   if it's heavy, collapse the namespace to `(dataset, var)` + per-attempt reset
   is impossible → switch to worker-side local fill with single end-of-job bulk
   push (loses streaming, keeps everything else).

## 11. Suggested commit sequence

1. `condor-spikes/` + RESULTS.md (Phase 0)
2. `src/qawa/condor/{config,datasets,proxy,submit}.py` + `brewer-htcondor2-inclusive.py` + parity test note
3. `dag.py` + `brewer-dag-inclusive.py` (--dag/--resume) + FINAL merge
4. `histogram_templates` refactor in `wztau2lnu_inclusive.py` (isolated, easy review)
5. `wztau2lnu_inclusive_histserv.py` port + `brewer-remote-inclusive.py` flags + sumw split
6. DAG histserv pieces (`histserv_service.sh`, `server_ready.py`, `reset_namespace.py`, `snapshot_and_merge.py`, `resume_planner.py`)
7. `monitor2.py`
8. cleanup + docs
