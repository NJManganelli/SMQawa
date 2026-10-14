"""Opt-in fix for uproot's mis-parse of JAGGED Float16_t / Double32_t branches.

THE BUG
uproot locates the ``[min,max,nbits]`` range spec of a Float16_t/Double32_t leaf
with::

    left  = title.index("[")
    right = title.index("]")

``index`` finds the FIRST bracket.  For a scalar leaf, titled e.g.

    GenVtx_x/f[0,0,10]

that is the range spec and everything works.  For a jagged leaf, titled

    Electron_eta[nElectron]/f[0,0,12]

the first bracket is the ARRAY-COUNT bracket.  The parsed spec is then
non-numeric, the code falls through to ``num_bits = "no brackets"`` WITHOUT
binding ``low``/``high``, and the following ``AsFloat16(low, high, ...)`` raises

    UnboundLocalError: cannot access local variable 'low' ...

Verified still present on uproot5 ``main`` (September 2026), so upgrading does
not help.

WHO THIS AFFECTS
Not a corrupt file: ``name[count]/f[0,0,12]`` is valid ROOT and ROOT reads it.
CMS FlashSim output hits it because its writer runs on ROOT 6.40 outside CMSSW
and emits genuine Float16_t leaves, whereas CMSSW NanoAOD reduces precision by
truncating the mantissa of a plain float32 and so never produces this title
form.  On a FlashSim NanoAOD that is 332 of 671 branches -- including Jet_pt,
Jet_eta, Electron_eta and the DeepTau raw scores.

SCOPE OF THE PATCH
Deliberately minimal.  The wrapper calls the original function first and only
intervenes on ``UnboundLocalError``, which is the precise and complete
signature of this bug -- every title uproot handles today keeps its existing
interpretation, and every other uproot error propagates untouched.  It is
opt-in rather than applied at import, so a run that does not involve FlashSim
gets stock uproot behaviour and stock uproot errors.

    from qawa import uproot_float16_patch
    if uproot_float16_patch.file_is_affected(path):   # or: if sample is FlashSim
        uproot_float16_patch.enable()

This is a stopgap.  The durable fixes are a one-line upstream change (use the
bracket group following ``/f`` or ``/d`` rather than the first one) and having
FlashSim write plain ``/F`` like CMSSW NanoAOD does.
"""

from __future__ import annotations

import os
import re

import uproot
import uproot.interpretation.identify as _identify

# the range spec is the bracket group immediately after the /f or /d type code
_RANGE_RE = re.compile(r"/(?P<code>[fd])\[(?P<spec>[^\]]*)\]\s*$")

_ORIGINAL = None


def _title_of(branch, leaf):
    if leaf.classname in ("TLeafF16", "TLeafD32"):
        return leaf.member("fTitle")
    if branch.streamer is not None:
        return branch.streamer.title
    return ""


def _parse_range(title):
    """-> (low, high, num_bits) or None if the title has no trailing range spec."""
    m = _RANGE_RE.search(title or "")
    if m is None:
        return None
    parts = [p.strip() for p in m.group("spec").split(",")]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if len(vals) == 2:
        return vals[0], vals[1], None
    if len(vals) == 3:
        return vals[0], vals[1], int(vals[2])
    return None


def _patched_float16_or_double32(branch, context, leaf, is_float16, dims):
    try:
        return _ORIGINAL(branch, context, leaf, is_float16, dims)
    except UnboundLocalError as exc:
        # exactly this bug: low/high were never bound. Anything else propagates.
        # Hold the exception rather than bare-raising later: by the time we know
        # the title is unparseable the except block has exited, so a bare raise
        # would report "No active exception to reraise" and hide the real error.
        original_error = exc

    parsed = _parse_range(_title_of(branch, leaf))
    if parsed is None:
        raise original_error
    low, high, num_bits = parsed

    # mirror uproot's own defaults for an out-of-range or absent bit count
    if num_bits is None or not 2 <= num_bits <= 32:
        num_bits = 32
    numerical = uproot.interpretation.numerical
    cls = numerical.AsFloat16 if is_float16 else numerical.AsDouble32
    # the caller wraps this in AsJagged when the leaf has an fLeafCount, so we
    # return only the inner numeric interpretation
    return cls(low, high, num_bits, dims)


def enable():
    """Install the patch. Idempotent; returns True if it was newly installed."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        return False
    _ORIGINAL = _identify._float16_or_double32
    _identify._float16_or_double32 = _patched_float16_or_double32
    return True


def disable():
    """Restore stock uproot. Returns True if the patch had been installed."""
    global _ORIGINAL
    if _ORIGINAL is None:
        return False
    _identify._float16_or_double32 = _ORIGINAL
    _ORIGINAL = None
    return True


def is_enabled():
    return _ORIGINAL is not None


def file_is_affected(path, tree="Events"):
    """True if `path` contains a jagged Float16_t/Double32_t branch.

    Cheap: reads only the branch metadata, never the baskets.
    """
    try:
        t = uproot.open(path)[tree]
    except Exception:
        return False
    for name in t.keys():
        branch = t[name]
        leaves = branch.member("fLeaves")
        if not len(leaves):
            continue
        leaf = leaves[0]
        if leaf.classname not in ("TLeafF16", "TLeafD32"):
            continue
        if "[" in (branch.title or "").split("/")[0]:
            return True
    return False


def enable_if_needed(path, tree="Events"):
    """Install the patch only if this file actually needs it."""
    if is_enabled():
        return False
    if file_is_affected(path, tree=tree):
        enable()
        return True
    return False


# --------------------------------------------------------------------------
# Worker propagation.
#
# A monkeypatch installed in the parent process does NOT reach coffea's
# FuturesExecutor workers: macOS spawns rather than forks, so each worker is a
# fresh interpreter. Environment variables ARE inherited, so the parent sets
# ENV_FLAG and every process that imports this module patches itself.
#
# The processor module imports this module unconditionally, and a worker must
# import the processor to unpickle it, so the patch is in place before the
# worker's first read. Without this, a FlashSim job under FuturesExecutor fails
# in the worker with the very UnboundLocalError this module exists to prevent.
ENV_FLAG = "QAWA_UPROOT_FLOAT16_PATCH"

if os.environ.get(ENV_FLAG) == "1":
    enable()


def enable_for_workers():
    """Enable here AND in any child process spawned from this one."""
    os.environ[ENV_FLAG] = "1"
    return enable()
