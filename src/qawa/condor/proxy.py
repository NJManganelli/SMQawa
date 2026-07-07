"""VOMS proxy check / renewal (lifted from brewer-htcondor-inclusive.py 122-156).

Kept as an isolated, importable function so both the Phase-1 submitter and the
Phase-2 DAG builder share one implementation.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def ensure_proxy(min_hours: float = 10.0, redo: bool = False) -> str:
    """Ensure a valid VOMS proxy exists in $HOME; return its path.

    Behavior matches the old brewer: copy of ``x509up_u<uid>`` in $HOME, renewed
    with ``voms-proxy-init -voms cms`` when missing, near-expiry, or ``redo``.
    """
    proxy_base = "x509up_u{}".format(os.getuid())
    home_base = os.environ["HOME"]
    proxy_copy = os.path.join(home_base, proxy_base)

    regenerate = redo
    if not os.path.isfile(proxy_copy):
        logger.warning("--- proxy file does not exist")
        regenerate = True
    elif not redo:
        lifetime = subprocess.check_output(
            ["voms-proxy-info", "--file", proxy_copy, "--timeleft"]
        )
        lifetime = float(lifetime) / (60 * 60)
        logger.info("--- proxy lifetime is %s hours", lifetime)
        if lifetime < min_hours:
            logger.warning("--- proxy has expired !")
            regenerate = True

    if regenerate:
        redone = False
        while not redone:
            status = os.system("voms-proxy-init -voms cms")
            if os.WEXITSTATUS(status) == 0:
                redone = True
        shutil.copyfile("/tmp/" + proxy_base, proxy_copy)

    return proxy_copy
