"""Dataset resolution via dasgoclient + sample-name mangling + inputfiles.dat I/O.

Logic lifted from brewer-htcondor-inclusive.py (192-215 and the sample-name
handling around 167-176), isolated so it is reusable and testable without a
submit round-trip.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """A resolved sample: its DAS query string, mangled name, and MC flag."""

    query: str          # the raw line from the input list (may contain '*')
    name: str           # mangled sample name used for jobs_<tag>_<era>_<name>
    isMC: int           # auto-detected from the DAS path tier


def parse_sample_line(line: str) -> "Sample | None":
    """Parse one line of the input dataset list; return None to skip it.

    Skips comments ('#') and non-dataset lines, auto-detects MC from the tier
    (``NANOAODSIM`` -> MC), and mangles the sample name exactly as the old brewer:
    MC -> the primary dataset name; data -> ``<primary>_<era-ish>`` (fields 1-2).
    """
    if "#" in line:
        return None
    split_sample = line.split("/")
    if len(split_sample) <= 1:
        return None
    is_mc = 1 * (split_sample[-1] == "NANOAODSIM")
    name = line.split("/")[1] if is_mc else "_".join(line.split("/")[1:3])
    name = name.replace("*", "")
    return Sample(query=line, name=name, isMC=is_mc)


def resolve_files(sample: Sample) -> list[str]:
    """Resolve a sample's input ROOT files via dasgoclient.

    Handles wildcard dataset queries (expand datasets first, then files) exactly
    like the old brewer.
    """
    files: list[str] = []
    if "*" in sample.query:
        datasets = subprocess.check_output(
            ["dasgoclient", "--query", f"dataset={sample.query}"]
        ).decode("UTF-8")
        logger.info("--- expanded wildcard to datasets:\n%s", datasets)
        for ds in datasets.split("\n")[:-1]:
            out = subprocess.check_output(
                ["dasgoclient", "--query", f"file dataset={ds}"]
            ).decode("UTF-8")
            files += [f for f in out.split("\n") if f != ""]
    else:
        out = subprocess.check_output(
            ["dasgoclient", "--query", f"file dataset={sample.query}"]
        ).decode("UTF-8")
        files = [f for f in out.split("\n") if f != ""]
    return files


def write_inputfiles(jobdir: str, files: list[str]) -> str:
    """Write inputfiles.dat (one file per line). Returns its path.

    Still written even though the bindings submit via itemdata: monitoring,
    resume, and humans all rely on it (plan Phase 1, step 2).
    """
    path = os.path.join(jobdir, "inputfiles.dat")
    with open(path, "w") as fh:
        for fn in files:
            fh.write(fn + "\n")
    return path


def read_inputfiles(jobdir: str) -> list[str]:
    path = os.path.join(jobdir, "inputfiles.dat")
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def jobs_dir_name(tag: str, era: str, sample_name: str) -> str:
    return "_".join(["jobs", tag, era, sample_name])
