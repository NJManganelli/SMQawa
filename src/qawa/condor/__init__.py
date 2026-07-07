"""HTCondor v25 (``htcondor2``/``classad2``) submission for SMQawa.

This package replaces the old "Python writes bash + condor.sub, then shells out to
condor_submit" flow (``brewer-htcondor-inclusive.py``) with the version-2 Python
bindings. See ``HTCONDOR_PYTHON_PLAN.md`` for the full migration plan.

Phase 1 (implemented here): ``config``, ``datasets``, ``proxy``, ``submit`` —
bindings-based per-sample submission with behavior parity to the old submitter.

Phase 2/3 (hooks only, see ``dag``): DAGMan orchestration and the long-running
``histserv`` SERVICE node. The double-fill-avoidance hooks that MUST be wired for
histserv mode are documented in ``dag`` and ``condor-spikes/RESULTS.md``.
"""

from .config import SubmissionConfig, to_external_path

__all__ = ["SubmissionConfig", "to_external_path"]
