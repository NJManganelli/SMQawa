import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STORE = os.environ.get("QAWA_TEST_STORE", str(REPO.parent / "SMQawaTestFiles" / "store"))
DATA = os.environ.get("QAWA_TEST_DATA", str(REPO))
CVMFS_HEAD = os.environ.get("QAWA_CVMFS_HEAD", "/cvmfs")


def input_path(base, relpath):
    path = os.path.join(base, relpath)
    return path, pytest.mark.skipif(not os.path.exists(path), reason=f"test input not available: {path}")


def require_input(path):
    if not os.path.exists(path):
        pytest.skip(f"test input not available: {path}", allow_module_level=True)
