#!/usr/bin/env python3
"""Phase-0 spike S0.1 probe: submit + query a job with the htcondor2 bindings.

Run from the access point (the `submit` container) as the submitter account
(root maps to the pool identity condor@submit, which the schedd rejects):

    docker compose exec -u submituser submit python3 /smqawa/condor-docker/probe_submit.py

Accept criterion (per HTCONDOR_PYTHON_PLAN.md 4, S0.1): the job is submitted,
visible via schedd.query, and completes on one of the worker nodes -- all through
the version-2 bindings, with no condor_submit / call_host involvement.
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    import htcondor2  # noqa: this is the whole point of the probe
    import classad2   # noqa

    print(f"htcondor2 {htcondor2.version()}")

    sub = htcondor2.Submit({
        "executable": "/bin/sleep",
        "arguments": "20",
        "universe": "vanilla",
        "request_disk": "64MB",
        "request_memory": "64MB",
        "output": "/tmp/qawa_probe.$(ClusterId).$(ProcId).out",
        "error": "/tmp/qawa_probe.$(ClusterId).$(ProcId).err",
        "log": "/tmp/qawa_probe.log",
    })

    schedd = htcondor2.Schedd()
    result = schedd.submit(sub, count=1)
    cluster = result.cluster()
    print(f"submitted cluster {cluster}")

    deadline = time.time() + 300
    while time.time() < deadline:
        ads = schedd.query(
            constraint=f"ClusterId=={cluster}",
            projection=["ClusterId", "ProcId", "JobStatus"],
        )
        if not ads:
            print("job left the queue -> completed.")
            return 0
        status = {1: "idle", 2: "running", 5: "held"}.get(int(ads[0]["JobStatus"]), str(ads[0]["JobStatus"]))
        print(f"cluster {cluster} status: {status}")
        time.sleep(5)

    print("TIMEOUT waiting for job to complete", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
