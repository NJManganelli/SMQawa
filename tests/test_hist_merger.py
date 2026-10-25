import collections
import contextlib
import gzip
import importlib.util
import io
import multiprocessing
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import hist
import numpy
import psutil
import pytest
from rich.console import Console

SCRIPT = Path(__file__).resolve().parents[1] / "hist-merger.py"
FAILING_DUMP = (
    "import pickle, runpy, sys\n"
    "def dump(obj, f, *args, **kwargs):\n"
    "    f.write(b'partial')\n"
    "    raise OSError('disk full')\n"
    "pickle.dump = dump\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)
PROTOCOL_RUNNER = ("import pickle, runpy, sys; pickle.HIGHEST_PROTOCOL = pickle.DEFAULT_PROTOCOL - 1; "
                   "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
POOL_SPY = (
    "import concurrent.futures, runpy, sys, threading\n"
    "real = concurrent.futures.ProcessPoolExecutor\n"
    "class Spy(real):\n"
    "    def __init__(self, max_workers=None, *args, **kwargs):\n"
    "        print(f'pool {max_workers}', flush=True)\n"
    "        self.first = True\n"
    "        super().__init__(max_workers, *args, **kwargs)\n"
    "    def submit(self, *args, **kwargs):\n"
    "        if self.first:\n"
    "            print(f'threads {threading.active_count()}', flush=True)\n"
    "            self.first = False\n"
    "        return super().submit(*args, **kwargs)\n"
    "concurrent.futures.ProcessPoolExecutor = Spy\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)
KILLER = (
    "import gzip, multiprocessing, os, signal\n"
    "real = gzip.open\n"
    "def killing_open(filename, *args, **kwargs):\n"
    "    if multiprocessing.parent_process() is not None and str(filename).endswith('histogram_2.pkl.gz'):\n"
    "        os.kill(os.getpid(), signal.SIGKILL)\n"
    "    return real(filename, *args, **kwargs)\n"
    "gzip.open = killing_open\n"
)


def make_hist(entries, nbins=4):
    h = hist.Hist(hist.axis.StrCategory([], growth=True, name="syst"), hist.axis.Regular(nbins, 0, 4),
                  storage=hist.storage.Weight())
    for syst, x, weight in entries:
        h.fill(syst, x, weight=weight)
    return h


def write_job(directory, index, content):
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"histogram_{index}.pkl.gz", "wb") as f:
        pickle.dump(content, f)


def run_merger(cwd, *args, runner=None, env=None, ncores=1, timeout=None):
    prefix = [sys.executable] if runner is None else [sys.executable, "-c", runner]
    extra = [] if ncores is None else ["--ncores", str(ncores)]
    return subprocess.run([*prefix, str(SCRIPT), *args, *extra], cwd=cwd, capture_output=True, text=True, env=env,
                          timeout=timeout)


def load_module():
    spec = importlib.util.spec_from_file_location("hist_merger", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merged_output(outdir):
    (path,) = Path(outdir).glob("merged-histogram-*.pkl.gz")
    with gzip.open(path) as f:
        return pickle.load(f)


def assert_same(h, expected):
    assert h.variances() is not None
    assert sorted(h.axes[0]) == sorted(expected.axes[0])
    for label in expected.axes[0]:
        numpy.testing.assert_array_equal(h[{"syst": label}].values(flow=True),
                                         expected[{"syst": label}].values(flow=True))
        numpy.testing.assert_array_equal(h[{"syst": label}].variances(flow=True),
                                         expected[{"syst": label}].variances(flow=True))


def dy_job(weight, sumw, nbins=4):
    return {"DY": {"hist": {"x": make_hist([("nominal", 0.5, weight)], nbins)}, "sumw": sumw}}


def three_files(base):
    for i, weight in enumerate([10, 20, 30]):
        write_job(base / "jobs_tagX_2024_DY", i, dy_job(weight, 100 * (i + 1)))


def merge(cwd, base, *extra):
    out = cwd / "out"
    out.mkdir(exist_ok=True)
    r = run_merger(cwd, "--dir", str(base), "-t", "tagX", "--era", "2024", "--outdir", str(out), *extra)
    assert r.returncode == 0, r.stderr
    return merged_output(out)


def assert_three_files(merged):
    assert merged["DY"]["sumw"] == 600
    assert sorted(merged["DY"]) == ["hist", "sumw"]
    assert_same(merged["DY"]["hist"]["x"], make_hist([("nominal", 0.5, w) for w in (10, 20, 30)]))


def test_three_files_one_directory_sums_every_file(tmp_path):
    three_files(tmp_path)
    assert_three_files(merge(tmp_path, tmp_path))


def test_dataset_split_across_directories(tmp_path):
    for directory in ("jobs_tagX_2024_DY", "jobs_tagX_2024_DY_ext"):
        for i, weight in enumerate([10, 20]):
            write_job(tmp_path / directory, i, dy_job(weight, 100))
    merged = merge(tmp_path, tmp_path)
    assert sorted(merged) == ["DY"]
    assert sorted(merged["DY"]) == ["hist", "sumw"] and merged["DY"]["sumw"] == 400
    assert_same(merged["DY"]["hist"]["x"], make_hist([("nominal", 0.5, w) for w in (10, 20, 10, 20)]))


def test_multi_dataset_directory_keeps_later_directories(tmp_path):
    for i in range(2):
        write_job(tmp_path / "jobs_tagX_2024_AB", i, {"A": {"hist": {"x": make_hist([("nominal", 0.5, 1)])}, "sumw": 1},
                                                      "B": {"hist": {"x": make_hist([("nominal", 1.5, 2)])}, "sumw": 2}})
    three_files(tmp_path)
    merged = merge(tmp_path, tmp_path)
    assert sorted(merged) == ["A", "B", "DY"]
    assert merged["A"]["sumw"] == 2 and merged["B"]["sumw"] == 4
    assert_same(merged["A"]["hist"]["x"], make_hist([("nominal", 0.5, 1)] * 2))
    assert_same(merged["B"]["hist"]["x"], make_hist([("nominal", 1.5, 2)] * 2))
    assert_three_files(merged)


def test_dir_without_trailing_separator(tmp_path):
    base = tmp_path / "base"
    three_files(base)
    assert not str(base).endswith("/")
    assert_three_files(merge(tmp_path, base))


def test_default_dir_is_cwd(tmp_path):
    three_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "-t", "tagX", "--era", "2024", "--outdir", str(out))
    assert r.returncode == 0, r.stderr
    assert_three_files(merged_output(out))


def test_stale_cache_from_earlier_version_is_not_reused(tmp_path):
    three_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with gzip.open(out / "jobs_tagX_2024_DY_histograms.pkl.gz", "wb") as f:
        pickle.dump({"DY": {"hist": {"x": make_hist([("nominal", 0.5, 30)])}, "sumw": 600,
                            "x": make_hist([("nominal", 0.5, 50)])}}, f)
    assert_three_files(merge(tmp_path, tmp_path))


def outputs(outdir):
    return sorted(p.name for p in Path(outdir).iterdir())


def tag_layout(tmp_path):
    write_job(tmp_path / "jobs_tagX_2024_DY", 0, dy_job(10, 100))
    write_job(tmp_path / "jobs_tagX_DD_2024_DY", 0, dy_job(20, 100))
    out = tmp_path / "out"
    out.mkdir()
    return ("--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out)), out


def test_tag_does_not_match_longer_tag(tmp_path):
    args, out = tag_layout(tmp_path)
    r = run_merger(tmp_path, *args)
    assert r.returncode == 0, r.stderr
    merged = merged_output(out)
    assert merged["DY"]["sumw"] == 100
    assert_same(merged["DY"]["hist"]["x"], make_hist([("nominal", 0.5, 10)]))


def test_match_glob_keeps_literal_output_name(tmp_path):
    args, out = tag_layout(tmp_path)
    r = run_merger(tmp_path, *args, "--match", "tagX*")
    assert r.returncode == 0, r.stderr
    names = outputs(out)
    assert len(names) == 1 and "-tagX-2024-" in names[0] and not set("*?[") & set(names[0])
    assert "does not match" not in r.stdout
    merged = merged_output(out)
    assert merged["DY"]["sumw"] == 200
    assert_same(merged["DY"]["hist"]["x"], make_hist([("nominal", 0.5, 10), ("nominal", 0.5, 20)]))


def test_match_not_covering_tag_warns(tmp_path):
    args, out = tag_layout(tmp_path)
    r = run_merger(tmp_path, *args, "--match", "tagX_DD")
    assert r.returncode == 0, r.stderr
    assert "does not match --match" in r.stdout and len(outputs(out)) == 1
    assert_same(merged_output(out)["DY"]["hist"]["x"], make_hist([("nominal", 0.5, 20)]))


def test_missing_outdir_fails_before_merging(tmp_path):
    write_job(tmp_path / "jobs_tagX_2024_DY", 0, dy_job(10, 100))
    missing = tmp_path / "missing"
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(missing))
    assert r.returncode == 2 and "--outdir" in r.stderr and "Traceback" not in r.stderr
    assert not missing.exists()


def test_tag_with_glob_is_rejected(tmp_path):
    args, out = tag_layout(tmp_path)
    r = run_merger(tmp_path, *args[:3], "tagX*", *args[4:])
    assert r.returncode != 0 and "--match" in r.stderr and outputs(out) == []


def test_no_cache_files_written(tmp_path):
    three_files(tmp_path)
    out = tmp_path / "out"
    merge(tmp_path, tmp_path)
    assert len(outputs(out)) == 1 and outputs(out)[0].startswith("merged-histogram-")


@pytest.mark.parametrize("ncores", [1, 2])
def test_unreadable_file_fails(tmp_path, ncores):
    write_job(tmp_path / "jobs_tagX_2024_DY", 0, dy_job(10, 100))
    (tmp_path / "jobs_tagX_2024_DY" / "histogram_1.pkl.gz").write_bytes(gzip.compress(b"not a pickle")[:12])
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out),
                   ncores=ncores)
    assert r.returncode != 0 and "histogram_1.pkl.gz" in r.stderr and outputs(out) == []


def test_unmergeable_histogram_names_the_file(tmp_path):
    for i in range(17):
        write_job(tmp_path / "jobs_tagX_2024_DY", i, dy_job(10, 100, nbins=5 if i == 1 else 4))
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out))
    assert r.returncode != 0 and "histogram_1.pkl.gz" in r.stderr and "DY/x" in r.stderr
    assert outputs(out) == []


def test_failed_write_keeps_previous_output(tmp_path):
    three_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    args = ("--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out))
    assert run_merger(tmp_path, *args).returncode == 0
    (name,) = [n for n in outputs(out) if n.startswith("merged-histogram-")]
    before = outputs(out), (out / name).read_bytes()
    r = run_merger(tmp_path, *args, runner=FAILING_DUMP)
    assert r.returncode != 0 and "disk full" in r.stderr
    assert [n for n in outputs(out) if not n.endswith(".tmp")] == before[0] and (out / name).read_bytes() == before[1]


def test_no_empty_line_without_empty_files(tmp_path):
    three_files(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out))
    assert r.returncode == 0 and "empty pkl.gz" not in r.stdout


def test_verbose_lists_directories(tmp_path):
    dir_a, dir_b = tmp_path / "jobs_tagX_2024_A", tmp_path / "jobs_tagX_2024_B"
    for i in range(2):
        write_job(dir_a, i, dy_job(10, 100))
    write_job(dir_b, 0, dy_job(10, 100))
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out), "--verbose")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert f"{dir_a}: 2 files" in lines and f"{dir_b}: 1 files" in lines


def test_jobs_layout_ignores_other_pickles(tmp_path):
    three_files(tmp_path)
    write_job(tmp_path / "jobs_tagX_2024_DY", "old", dy_job(1000, 1000))
    (tmp_path / "jobs_tagX_2024_DY" / "histogram_old.pkl.gz").rename(
        tmp_path / "jobs_tagX_2024_DY" / "merged-histogram-old.pkl.gz")
    assert_three_files(merge(tmp_path, tmp_path))


def test_several_datasets(tmp_path):
    three_files(tmp_path)
    for i in range(2):
        write_job(tmp_path / "jobs_tagX_2024_WZ", i, {"WZ": {"hist": {"x": make_hist([("nominal", 2.5, 3)])},
                                                             "sumw": 5}})
    merged = merge(tmp_path, tmp_path)
    assert_three_files(merged)
    assert merged["WZ"]["sumw"] == 10
    assert_same(merged["WZ"]["hist"]["x"], make_hist([("nominal", 2.5, 3)] * 2))


def test_two_datasets_in_one_directory(tmp_path):
    for i in range(3):
        write_job(tmp_path / "jobs_tagX_2024_AB", i, {"A": {"hist": {"x": make_hist([("nominal", 0.5, i + 1)])}, "sumw": 1},
                                                      "B": {"hist": {"x": make_hist([("up", 3.5, 2)])}, "sumw": 2}})
    merged = merge(tmp_path, tmp_path)
    assert merged["A"]["sumw"] == 3 and merged["B"]["sumw"] == 6
    assert_same(merged["A"]["hist"]["x"], make_hist([("nominal", 0.5, w) for w in (1, 2, 3)]))
    assert_same(merged["B"]["hist"]["x"], make_hist([("up", 3.5, 2)] * 3))


@pytest.mark.parametrize("ncores", [1, 2])
def test_empty_file_is_skipped_and_reported(tmp_path, ncores):
    three_files(tmp_path)
    (tmp_path / "jobs_tagX_2024_DY" / "histogram_3.pkl.gz").write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out),
                   ncores=ncores)
    assert r.returncode == 0, r.stderr
    assert "1 empty" in r.stdout
    assert_three_files(merged_output(out))


def test_histograms_missing_in_some_files(tmp_path):
    names = [("h1", "h2"), ("h1",), ("h2", "h3")]
    for i, keys in enumerate(names):
        write_job(tmp_path / "jobs_tagX_2024_DY", i,
                  {"DY": {"hist": {k: make_hist([("nominal", 0.5, i + 1)]) for k in keys}, "sumw": 1}})
    merged = merge(tmp_path, tmp_path)
    assert sorted(merged["DY"]["hist"]) == ["h1", "h2", "h3"] and merged["DY"]["sumw"] == 3
    assert_same(merged["DY"]["hist"]["h1"], make_hist([("nominal", 0.5, 1), ("nominal", 0.5, 2)]))
    assert_same(merged["DY"]["hist"]["h2"], make_hist([("nominal", 0.5, 1), ("nominal", 0.5, 3)]))
    assert_same(merged["DY"]["hist"]["h3"], make_hist([("nominal", 0.5, 3)]))


def test_nested_dict_entries_are_dropped(tmp_path):
    for i in range(2):
        write_job(tmp_path / "jobs_tagX_2024_DY", i,
                  {"DY": {"hist": {"x": make_hist([("nominal", 0.5, 1)]), "nested": {"y": 1}}, "sumw": 1}})
    merged = merge(tmp_path, tmp_path)
    assert sorted(merged["DY"]["hist"]) == ["x"]


def test_growing_categories_across_files(tmp_path):
    entries = [[("nominal", 0.5, 1)], [("up", 1.5, 2), ("nominal", 0.5, 3)], [("down", 2.5, 4)]]
    for i, e in enumerate(entries):
        write_job(tmp_path / "jobs_tagX_2024_DY", i, {"DY": {"hist": {"x": make_hist(e)}, "sumw": 1}})
    merged = merge(tmp_path, tmp_path)
    assert_same(merged["DY"]["hist"]["x"], make_hist([x for e in entries for x in e]))


def test_output_name(tmp_path):
    three_files(tmp_path)
    out = tmp_path / "out"
    merge(tmp_path, tmp_path)
    assert outputs(out) == ["merged-histogram-Inc-tagX-2024-latest.pkl.gz"]
    (out / "merged-histogram-Inc-tagX-2024-latest.pkl.gz").unlink()
    merge(tmp_path, tmp_path, "-f", "mysuffix")
    assert outputs(out) == ["merged-histogram-Inc-tagX-2024-mysuffix.pkl.gz"]


def test_eos_layout_and_sample_filter(tmp_path):
    base = tmp_path / "eosbase"
    for i, weight in enumerate([10, 20]):
        write_job(base / "tagX" / "2024" / "DY", i, dy_job(weight, 100))
        write_job(base / "tagX" / "2024" / "WZ", i, {"WZ": dy_job(weight, 100)["DY"]})
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--eos", "--dir", str(base), "-t", "tagX", "--era", "2024", "--sample", "DY",
                   "--outdir", str(out))
    assert r.returncode == 0, r.stderr
    assert "found 2 files from 1 sample directories" in r.stdout
    merged = merged_output(out)
    assert sorted(merged) == ["DY"] and merged["DY"]["sumw"] == 200
    assert_same(merged["DY"]["hist"]["x"], make_hist([("nominal", 0.5, 10), ("nominal", 0.5, 20)]))


def test_sample_filter_jobs_layout(tmp_path):
    three_files(tmp_path)
    write_job(tmp_path / "jobs_tagX_2024_WZ", 0, {"WZ": dy_job(5, 5)["DY"]})
    merged = merge(tmp_path, tmp_path, "--sample", "D*")
    assert sorted(merged) == ["DY"]
    assert_three_files(merged)


def test_no_input_files_fails(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out))
    assert r.returncode != 0 and "no input pickle files found" in r.stderr and outputs(out) == []


def test_eos_default_dir_without_user_variable(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("USER", "LOGNAME")}
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--eos", "-t", "tagX", "--era", "2024", "--outdir", str(out), env=env)
    assert r.returncode != 0 and "globbing: /eos/user/" in r.stdout
    assert "no input pickle files found" in r.stderr and "KeyError" not in r.stderr


def merged_values(merged):
    return {dataset: (content["sumw"],
                      {name: {label: (h[{"syst": label}].values(flow=True).tolist(),
                                      h[{"syst": label}].variances(flow=True).tolist()) for label in h.axes[0]}
                       for name, h in content["hist"].items()})
            for dataset, content in merged.items()}


def test_parallel_equals_serial(tmp_path):
    for dataset in ("DY", "WZ"):
        for i in range(10):
            write_job(tmp_path / f"jobs_tagX_2024_{dataset}", i,
                      {dataset: {"hist": {"x": make_hist([(f"c{i % 3}", i % 4 + 0.5, i + 1)]),
                                          "y": make_hist([("nominal", 1.5, 2 * i + 1)])}, "sumw": i + 1}})
    results = {}
    for ncores in (1, 2, 4):
        out = tmp_path / f"out{ncores}"
        out.mkdir()
        r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out),
                       ncores=ncores)
        assert r.returncode == 0, r.stderr
        results[ncores] = merged_values(merged_output(out))
    assert results[1] == results[2] == results[4]
    assert results[1]["DY"][0] == 55 and results[1]["WZ"][0] == 55


@pytest.mark.parametrize("ncores", [1, 2])
def test_shared_reference_in_one_file_is_not_double_counted(tmp_path, ncores):
    shared = make_hist([("nominal", 0.5, 1)])
    write_job(tmp_path / "jobs_tagX_2024_DY", 0, {"DY": {"hist": {"h1": shared, "h2": shared}, "sumw": 1}})
    write_job(tmp_path / "jobs_tagX_2024_DY", 1, {"DY": {"hist": {"h1": make_hist([("nominal", 0.5, 1)]),
                                                                  "h2": make_hist([("nominal", 0.5, 1)])}, "sumw": 1}})
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out),
                   ncores=ncores)
    assert r.returncode == 0, r.stderr
    merged = merged_output(out)
    assert merged["DY"]["hist"]["h1"].sum().value == 2 and merged["DY"]["hist"]["h2"].sum().value == 2


def layout(tmp_path, big=False):
    for dataset, n in (("DY", 3), ("WZ", 4)):
        for i in range(n):
            content = {dataset: dy_job(10 + i, 100)["DY"]}
            if big and i == 0 and dataset == "DY":
                payload = numpy.random.default_rng(0).integers(0, 256, 11 * 2**20, numpy.uint8)
                content[dataset]["hist"]["blob"] = {"payload": payload}
            write_job(tmp_path / f"jobs_tagX_2024_{dataset}", i, content)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return ("--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out)), out


def has_pool_line(stdout):
    return re.search(r"^pool \d+$", stdout, re.M) is not None


def test_chunks_follow_directories():
    module = load_module()
    files = [f"/d/{directory}/histogram_{i}.pkl.gz" for directory, n in (("a", 10), ("b", 2), ("c", 1))
             for i in range(n)]
    chunks = module.make_chunks(files, 4)
    assert [(c[0].split("/")[2], len(c)) for c in chunks] == [("a", 4), ("a", 4), ("a", 2), ("b", 2), ("c", 1)]
    assert [f for c in chunks for f in c] == files


class FakeFuture:
    def __init__(self, executor, fn, args):
        self.executor, self.fn, self.args = executor, fn, args

    def result(self):
        self.executor.outstanding -= 1
        return self.fn(*self.args)


class FakeExecutor:
    def __init__(self):
        self.outstanding, self.peak = 0, 0

    def submit(self, fn, *args):
        self.outstanding += 1
        self.peak = max(self.peak, self.outstanding)
        return FakeFuture(self, fn, args)


def test_window_bounds_in_flight_and_keeps_order(tmp_path):
    module = load_module()
    for i in range(9):
        write_job(tmp_path / "jobs_tagX_2024_DY", i, {"DY": {"hist": {"x": make_hist([(f"c{i}", 0.5, 1)])}, "sumw": 1}})
    files = sorted(str(p) for p in (tmp_path / "jobs_tagX_2024_DY").glob("*.pkl.gz"))
    executor = FakeExecutor()
    merged, empty = module.merge_parallel([[f] for f in files], executor, 3, module.read_gzip,
                                          contextlib.nullcontext(), lambda *args: None)
    assert executor.peak == 3 and empty == []
    assert list(merged["DY"]["hist"]["x"].axes[0]) == [f"c{i}" for i in range(9)]


def test_unmergeable_partials_name_the_chunk(tmp_path):
    write_job(tmp_path / "jobs_tagX_2024_DY", 0, dy_job(10, 100))
    write_job(tmp_path / "jobs_tagX_2024_DY", 1, dy_job(10, 100, nbins=5))
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out), ncores=2)
    assert r.returncode != 0 and "histogram_1.pkl.gz" in r.stderr and "DY/x" in r.stderr
    assert outputs(out) == []


def test_parallel_category_order_matches_serial(tmp_path):
    for i in range(8):
        write_job(tmp_path / "jobs_tagX_2024_DY", i, {"DY": {"hist": {"x": make_hist([(f"c{i}", 0.5, 1)])}, "sumw": 1}})
    orders = {}
    for ncores in (1, 2, 4):
        out = tmp_path / f"out{ncores}"
        out.mkdir()
        r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out),
                       ncores=ncores)
        assert r.returncode == 0, r.stderr
        orders[ncores] = list(merged_output(out)["DY"]["hist"]["x"].axes[0])
    assert orders[1] == orders[2] == orders[4] == [f"c{i}" for i in range(8)]


def test_killed_worker_aborts_the_merge(tmp_path):
    args, out = layout(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(KILLER)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")])}
    r = run_merger(tmp_path, *args, ncores=2, env=env, timeout=60)
    assert r.returncode != 0 and "a worker process died" in r.stderr and outputs(out) == []


def test_workers_start_before_the_display(tmp_path):
    args, out = layout(tmp_path)
    env = {**os.environ, "FORCE_COLOR": "1", "COLUMNS": "80"}
    r = run_merger(tmp_path, *args, ncores=2, runner=POOL_SPY, env=env)
    assert r.returncode == 0, r.stderr
    assert has_pool_line(r.stdout) and re.search(r"^threads 1$", r.stdout, re.M), r.stdout


def test_eft_size_input_falls_back_to_one_core(tmp_path):
    args, out = layout(tmp_path, big=True)
    r = run_merger(tmp_path, *args, ncores=None, runner=POOL_SPY)
    assert r.returncode == 0, r.stderr
    assert "merging in the main process" in " ".join(r.stdout.split()) and not has_pool_line(r.stdout)
    r = run_merger(tmp_path, *args, ncores=2, runner=POOL_SPY)
    assert r.returncode == 0, r.stderr
    assert re.search(r"^pool 2$", r.stdout, re.M) and "memory grows with --ncores" in " ".join(r.stdout.split())


def test_small_inputs_keep_four_workers(tmp_path):
    args, out = layout(tmp_path)
    r = run_merger(tmp_path, *args, ncores=None, runner=POOL_SPY)
    assert r.returncode == 0, r.stderr
    assert re.search(r"^pool 4$", r.stdout, re.M) and "merging in the main process" not in " ".join(r.stdout.split())


@pytest.mark.parametrize("ncores", [1, 2])
def test_display_does_not_change_results(tmp_path, ncores):
    args, out = layout(tmp_path)
    plain = run_merger(tmp_path, *args, ncores=ncores)
    assert plain.returncode == 0, plain.stderr
    assert "\u2501" not in plain.stdout and "\x1b[" not in plain.stdout and " free, " not in plain.stdout
    reference = merged_values(merged_output(out))
    env = {**os.environ, "FORCE_COLOR": "1", "COLUMNS": "120"}
    shown = run_merger(tmp_path, *args, ncores=ncores, env=env)
    assert shown.returncode == 0, shown.stderr
    assert "dirs " in shown.stdout and " free, " in shown.stdout and ("chunks" in shown.stdout) == (ncores > 1)
    assert merged_values(merged_output(out)) == reference
    quiet = run_merger(tmp_path, *args, "--noProgress", ncores=ncores, env=env)
    assert quiet.returncode == 0, quiet.stderr
    assert " free, " not in quiet.stdout and merged_values(merged_output(out)) == reference


@pytest.mark.parametrize("ncores", [1, 2])
def test_summary_line_in_every_mode(tmp_path, ncores):
    args, out = layout(tmp_path)
    r = run_merger(tmp_path, *args, ncores=ncores)
    worker = r", largest worker \d+\.\d\d GiB" if ncores > 1 else ""
    pattern = rf"^merged 7 files from 2 directories in \d+\.\d s; peak RSS: main process \d+\.\d\d GiB{worker}$"
    assert r.returncode == 0 and re.search(pattern, r.stdout, re.M), r.stdout


def test_display_counts_directories_files_and_chunks(tmp_path):
    module = load_module()
    files = sorted(str(tmp_path / f"jobs_tagX_2024_{d}" / f"histogram_{i}.pkl.gz")
                   for d, n in (("A", 10), ("B", 2), ("C", 1)) for i in range(n))
    chunks = module.make_chunks(files, 4)
    display = module.MergeDisplay(files, len(chunks), False, "2024")
    for chunk in chunks[:2]:
        display.chunk_done(chunk)
    tasks = {t.description.strip(): t for t in display.progress.tasks}
    assert tasks["dirs 0/3"].completed == 8 and tasks["dirs 0/3"].total == 13 and tasks["chunks"].completed == 2
    assert tasks["A"].completed == 8 and tasks["A"].total == 10
    for chunk in chunks[2:]:
        display.chunk_done(chunk)
    tasks = {t.description.strip(): t for t in display.progress.tasks}
    assert tasks["dirs 3/3"].completed == 13 and tasks["chunks"].completed == len(chunks)
    assert tasks["C"].completed == 1


def test_rows_fit_80_columns(monkeypatch):
    module = load_module()
    files = sorted(f"/d/jobs_WZ_DD_2024_{d}/histogram_{i}.pkl.gz"
                   for d, n in (("WZto3LNu_EFT_madgraph_pythia8", 10), ("TTto2L2Nu", 4)) for i in range(n))
    chunks = module.make_chunks(files, 4)
    monkeypatch.setattr(module, "tree_memory", lambda processes: (int(114.4 * 2**30), 5))
    display = module.MergeDisplay(files, len(chunks), True, "2024")
    display.live.renderable.renderables[1].columns[1].total = 256 * 2**30
    for chunk in chunks[:3]:
        display.chunk_done(chunk)
    console = Console(file=io.StringIO(), width=80, force_terminal=True, color_system=None)
    with console.capture() as capture:
        console.print(display.live.renderable)
    lines = capture.get().splitlines()
    assert len(lines) == 4 and lines[1].startswith("  WZto3LNu"), lines
    assert lines[-1].startswith("memory") and "\u2501" in lines[-1] and "proc" in lines[-1], lines


@pytest.mark.parametrize("fraction, free, style", [(0.4, 0.5, "green"), (0.6, 0.3, "yellow"), (0.9, 0.05, "bold red"),
                                                   (0.2, 0.05, "bold red")])
def test_memory_colours(monkeypatch, fraction, free, style):
    module = load_module()
    total = 64 * 2**30
    monkeypatch.setattr(module, "tree_memory", lambda processes: (int(fraction * total), 3))
    memory = collections.namedtuple("memory", "total available")
    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: memory(total, int(free * total)))
    grid = module.MemoryBarColumn(True).render(None)
    assert list(grid.columns[1].cells)[0].style == style


def test_memory_sampling_error_does_not_abort(monkeypatch):
    module = load_module()

    def denied(self, recursive=False):
        raise psutil.AccessDenied(os.getpid())

    monkeypatch.setattr(psutil.Process, "children", denied)
    files = [f"/d/jobs_t_2024_A/histogram_{i}.pkl.gz" for i in range(4)]
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    with module.MergeDisplay(files, 2, True, "2024", console=console) as display:
        for filename in files:
            time.sleep(0.3)
            display.files_done([filename])
    assert "memory unavailable" in console.file.getvalue()


def test_serial_display_does_not_scan_processes(monkeypatch):
    module = load_module()
    calls = []
    monkeypatch.setattr(psutil.Process, "children", lambda self, recursive=False: calls.append(1) or [])
    module.MemoryBarColumn(False).render(None)
    assert calls == []
    column = module.MemoryBarColumn(True)
    column.render(None)
    column.render(None)
    assert calls == [1]


def hold(ready):
    block = os.urandom(200 * 2**20)
    ready.set()
    time.sleep(60 + len(block) * 0)


def test_tree_memory_counts_children():
    module = load_module()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    child = context.Process(target=hold, args=(ready,))
    child.start()
    try:
        assert ready.wait(30)
        own = psutil.Process().memory_info().rss
        rss, count = module.tree_memory(module.MemoryBarColumn(True).current_processes())
        assert rss >= own + 150 * 2**20 and count >= 2
    finally:
        child.kill()
        child.join()


def test_output_compression_and_protocol(tmp_path):
    three_files(tmp_path)
    out = tmp_path / "out"
    name = out / "merged-histogram-Inc-tagX-2024-latest.pkl.gz"
    merge(tmp_path, tmp_path)
    assert name.read_bytes()[8] == 0
    merge(tmp_path, tmp_path, "--compressionLevel", "9")
    assert name.read_bytes()[8] == 2
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out),
                   runner=PROTOCOL_RUNNER)
    assert r.returncode == 0, r.stderr
    assert gzip.decompress(name.read_bytes())[1] == pickle.DEFAULT_PROTOCOL - 1
