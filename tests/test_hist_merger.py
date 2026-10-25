import gzip
import pickle
import subprocess
import sys
from pathlib import Path

import hist
import numpy

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


def run_merger(cwd, *args, runner=None):
    prefix = [sys.executable] if runner is None else [sys.executable, "-c", runner]
    return subprocess.run([*prefix, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True)


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


def test_unreadable_file_fails(tmp_path):
    write_job(tmp_path / "jobs_tagX_2024_DY", 0, dy_job(10, 100))
    (tmp_path / "jobs_tagX_2024_DY" / "histogram_1.pkl.gz").write_bytes(gzip.compress(b"not a pickle")[:12])
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out))
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


def test_empty_file_is_skipped_and_reported(tmp_path):
    three_files(tmp_path)
    (tmp_path / "jobs_tagX_2024_DY" / "histogram_3.pkl.gz").write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()
    r = run_merger(tmp_path, "--dir", str(tmp_path), "-t", "tagX", "--era", "2024", "--outdir", str(out))
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
