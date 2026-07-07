"""Bindings-based submission (Phase 1): build an ``htcondor2.Submit`` and submit.

Replaces the old flow (write condor.sub string, shell out to condor_submit). The
worker *payload* wrapper (script.sh) is still a shell script -- it legitimately
needs venv activation + proxy export -- but it is rendered from package data, not
a module-level string, and the submit description is an ``htcondor2.Submit`` object.

``htcondor2``/``classad2`` are imported lazily so this module (and its non-condor
helpers) import cleanly on hosts without the bindings (e.g. macOS dev boxes; the
bindings ship inside the Linux submit container / on the AP).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from importlib import resources

from .config import SubmissionConfig

logger = logging.getLogger(__name__)


def render_worker_script(cfg: SubmissionConfig, jobdir: str) -> str:
    """Render templates/worker.sh into ``<jobdir>/script.sh``; return its path."""
    template = resources.files("qawa.condor").joinpath("templates/worker.sh").read_text()
    script = template.format(
        proxy=cfg.proxy_path,
        analysis=cfg.analysis,
        ismc=cfg.isMC,
        era=cfg.era,
        zzdd=cfg.zzdd,
        split_by_charge="--split_by_charge" if cfg.split_by_charge else "",
        coffea_image=cfg.coffea_image,
        full_image=cfg.full_image,
        install_loc_external=cfg.install_loc_external,
        executor=cfg.executor,
    )
    path = os.path.join(jobdir, "script.sh")
    with open(path, "w") as fh:
        fh.write(script)
    os.chmod(path, 0o755)
    return path


def build_worker_submit(cfg: SubmissionConfig, jobdir: str, *, transfer_files: list[str],
                        for_dag: bool = False):
    """Return an ``htcondor2.Submit`` mirroring the old condor_TEMPLATE exactly.

    ``+X`` submit-file spellings become ``MY.X`` dict keys. When ``for_dag`` is
    True, ``max_retries`` is omitted (DAGMan owns retry via ``RETRY`` -- plan 6.1)
    and ``arguments`` use ``$(jobid)`` VARS instead of ``$(ProcId)``.
    """
    import htcondor2  # lazy: only needed at actual submit/build time

    jobdir_ext = cfg.external(jobdir)
    script_ext = os.path.join(jobdir_ext, "script.sh")

    desc = {
        "universe": "vanilla",
        "request_disk": cfg.request_disk,
        "executable": script_ext,
        "arguments": "$(jobid) $(jobfn)" if for_dag else "$(ProcId) $(jobfn)",
        "transfer_input_files": ",".join(cfg.external(f) for f in transfer_files),
        "should_transfer_files": "YES",
        "WhenToTransferOutput": "ON_EXIT_OR_EVICT",
        "initialdir": jobdir_ext,
        "output": "$(ClusterId).$(ProcId).out",
        "error": "$(ClusterId).$(ProcId).err",
        "log": os.path.join(jobdir_ext, "cluster.log"),
        "on_exit_remove": "(ExitBySignal == False) && (ExitCode == 0)",
        "requirements": "Machine =!= LastRemoteHost",
        "MY.SingularityImage": f'"{cfg.singularity_image}"',
        "MY.JobFlavour": f'"{cfg.queue}"',
    }
    if not for_dag:
        desc["max_retries"] = str(cfg.max_retries)

    return htcondor2.Submit(desc)


@dataclass
class SubmitResult:
    cluster: int
    n_jobs: int
    jobdir: str
    manifest_path: str


def submit_sample(sub, files: list[str], jobdir: str, cfg: SubmissionConfig, *,
                  schedd=None) -> SubmitResult:
    """Submit one sample's jobs via itemdata (one job per input file).

    ``queue jobfn from inputfiles.dat`` becomes ``schedd.submit(sub, itemdata=...)``
    (plan Phase 1, step 2). Writes ``manifest.json`` keyed for Phase-4 monitoring.
    """
    import htcondor2

    schedd = schedd if schedd is not None else htcondor2.Schedd()
    itemdata = [{"jobfn": f} for f in files]
    result = schedd.submit(sub, itemdata=iter(itemdata))
    cluster = result.cluster()

    manifest_path = write_manifest(jobdir, cfg, cluster, files, schedd=schedd)
    logger.info("submitted %d jobs for %s as cluster %d", len(files), cfg.tag, cluster)
    return SubmitResult(cluster=cluster, n_jobs=len(files), jobdir=jobdir,
                        manifest_path=manifest_path)


def write_manifest(jobdir: str, cfg: SubmissionConfig, cluster: int, files: list[str],
                   *, schedd=None, sample_name: str | None = None) -> str:
    """Write jobs_.../manifest.json (plan Phase 1, step 3)."""
    schedd_name = ""
    try:
        if schedd is not None:
            import classad2  # noqa
            loc = schedd.location  # type: ignore[attr-defined]
            schedd_name = getattr(loc, "name", "") or str(loc)
    except Exception:
        schedd_name = os.environ.get("_CONDOR_SCHEDD_NAME", "")

    manifest = {
        "cluster": cluster,
        "schedd_name": schedd_name,
        "n_jobs": len(files),
        "files": {str(i): f for i, f in enumerate(files)},
        "tag": cfg.tag,
        "era": cfg.era,
        "sample": sample_name,
        "analysis": cfg.analysis,
        "submit_time": time.time(),
        "user_log": os.path.join(cfg.external(jobdir), "cluster.log"),
    }
    path = os.path.join(jobdir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def write_submit_file(sub, jobdir: str) -> str:
    """Serialize an ``htcondor2.Submit`` to <jobdir>/condor.sub.

    Used by the ``--submit-via=call_host`` fallback shim (plan 4/S0.1, 10.1):
    str(Submit) round-trips to a valid submit file that the call_host
    condor_submit escape hatch can consume unchanged.
    """
    path = os.path.join(jobdir, "condor.sub")
    with open(path, "w") as fh:
        fh.write(str(sub))
        fh.write("\nqueue jobfn from inputfiles.dat\n")
    return path
