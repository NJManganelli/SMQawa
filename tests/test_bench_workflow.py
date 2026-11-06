import importlib.util
import os
import sys

ROOT = os.environ.get("QAWA_EXPECTED_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
spec = importlib.util.spec_from_file_location("bench_workflow", os.path.join(ROOT, "benchmarks", "bench_workflow.py"))
bench_workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_workflow)


def sample(t, *procs):
    return {"t": t, "procs": [{"pid": pid, "role": role, "cpu": cpu} for pid, role, cpu in procs]}


def test_cpu_percent_series_ignores_exited_processes():
    procs = [
        sample(0.0, (1, "job", 0.0), (2, "job", 0.0)),
        sample(1.0, (1, "job", 1.0), (2, "job", 0.5)),
        sample(2.0, (2, "job", 1.5)),
        sample(3.0, (2, "job", 2.0), (3, "merge", 0.25)),
    ]
    t, percent = bench_workflow.cpu_percent_series(procs, 0.0)
    assert t == [1.0, 2.0, 3.0]
    assert percent == [150.0, 100.0, 75.0]
    assert min(percent) >= 0.0


def test_cpu_percent_series_by_role():
    procs = [
        sample(10.0, (1, "job", 5.0), (2, "merge", 0.0)),
        sample(12.0, (1, "job", 6.0), (2, "merge", 2.0)),
        sample(14.0, (2, "merge", 3.0)),
    ]
    assert bench_workflow.cpu_percent_series(procs, 10.0, "job") == ([2.0, 4.0], [50.0, 0.0])
    assert bench_workflow.cpu_percent_series(procs, 10.0, "merge") == ([2.0, 4.0], [100.0, 50.0])
