import gzip
import pickle
import subprocess
import sys
from pathlib import Path

import hist
import numpy

SCRIPT = Path(__file__).resolve().parents[1] / "hist-merger.py"


def make_hist(entries):
    h = hist.Hist(hist.axis.StrCategory([], growth=True, name="syst"), hist.axis.Regular(4, 0, 4),
                  storage=hist.storage.Weight())
    for syst, x, weight in entries:
        h.fill(syst, x, weight=weight)
    return h


def write_job(directory, index, content):
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"histogram_{index}.pkl.gz", "wb") as f:
        pickle.dump(content, f)


def run_merger(cwd, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True)


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


def dy_job(weight, sumw):
    return {"DY": {"hist": {"x": make_hist([("nominal", 0.5, weight)])}, "sumw": sumw}}


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
