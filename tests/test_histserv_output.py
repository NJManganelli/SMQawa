import gzip
import hashlib
import itertools
import json
import os
import pickle
import socket
import subprocess
import sys
import textwrap
import time
import uuid

import pytest

histserv = pytest.importorskip("histserv")

import awkward as ak
import grpc
import hist
import numpy as np
from coffea.analysis_tools import PackedSelection
from coffea.processor.executor import WorkItem

import qawa.datadriven as datadriven
import qawa.histserv_output as ho
import qawa.process.wztau2lnu_inclusive as wzmod
import qawa.production_keys as pk
from qawa import lumi_record

ROOT = os.environ.get("QAWA_EXPECTED_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAPS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "NUMEXPR_MAX_THREADS": "4"}
N = 3000
DAS_S = "/S/RunX-v1/NANOAODSIM"
DAS_X = "/X/RunX-v1/NANOAODSIM"
DAS_C = "/Muon0/Run2024C-v1/NANOAOD"
DAS_D = "/Muon0/Run2024D-v1/NANOAOD"
OPTIONS = {"era": "2024", "version": "v15", "pol": False, "split_by_charge": False, "split_FV": False}
CHANNELS = {"chA": ["triggers", "met_pt"], "chB": ["triggers", "~low"], "chEmpty": ["triggers", "empty"]}
LABELS = [("nominal",), ("up",), ("down",)]


def final_name(kind_label, markers="", tag="t", filename="latest", suffix=""):
    return f"merged-histogram-Inc-{tag}-2024-{filename}{suffix}-{kind_label}{markers}.pkl.gz"


MC_FILE = final_name("MC-backgrounds")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    ready = tmp_path_factory.mktemp("server") / "ready.json"
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "benchmarks", "loopback_histserv.py"), "--port", "0",
                             "--ready-file", str(ready)], env={**os.environ, **CAPS},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            if time.monotonic() > deadline or proc.poll() is not None:
                raise RuntimeError("loopback histserv server did not start")
            time.sleep(0.05)
        address = json.loads(ready.read_text())["address"]
        assert address.startswith("127.0.0.1:")
        yield address
    finally:
        proc.terminate()
        proc.wait(30)


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    ho.ATTEMPTS.clear()
    ho.POISONED.clear()
    monkeypatch.delenv(ho.POISON_FILE_ENV, raising=False)
    yield
    ho.ATTEMPTS.clear()
    ho.POISONED.clear()


def template():
    return hist.Hist(hist.axis.StrCategory([], name="channel", growth=True),
                     hist.axis.StrCategory([], name="systematic", growth=True),
                     hist.axis.Regular(20, -20, 700, name="met_pt"), hist.storage.Weight())


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(ho, "make_processor", lambda options: None)
    monkeypatch.setattr(ho, "templates", lambda proc, kind: {"met_pt": template()})
    monkeypatch.setattr(ho, "nondeterministic_reason", lambda proc, kind: None)


def synthetic_registry(address, samples=None, max_chunks=None, pol=False, tag="t"):
    options = dict(OPTIONS, pol=pol)
    return ho.register(address, tag, options, samples or {DAS_S: "mc"}, uuid.uuid4().hex, max_chunks=max_chunks)


def fill_inputs():
    rng = np.random.default_rng(1)
    selection = PackedSelection(dtype="uint64")
    selection.add("triggers", rng.random(N) < 0.9)
    selection.add("met_pt", rng.random(N) < 0.5)
    selection.add("low", rng.random(N) < 0.3)
    selection.add("empty", np.zeros(N, dtype=bool))
    raw = rng.uniform(0.2, 1.8, (len(LABELS), N))
    raw[:, rng.integers(0, N, 5)] = np.nan
    raw[:, rng.integers(0, N, 5)] = np.inf
    met = rng.uniform(-20, 700, N)
    return selection, {"met_pt": met}, wzmod.standard_weights(raw)


SELECTION, VALUES, WEIGHT = fill_inputs()


def pass_state(ctx=None):
    state = {"selection": SELECTION, "channels": CHANNELS, "block_cache": {}}
    if ctx is not None:
        state["histserv"] = ctx
    return state


def file_uuid(i):
    return str(uuid.UUID(int=i + 1))


def meta(reg, key, fileuuid, start, stop):
    entry = reg["samples"][key]
    return {"dataset": entry["dataset"], "fileuuid": fileuuid, "entrystart": start, "entrystop": stop,
            "is_data": entry["kind"] == "data", "is_signal": entry["kind"] == "signal"}


def send_pass(reg, key, md, pass_name="nominal", end=True):
    ctx = ho.pass_context(reg, key, md, None if pass_name == "nominal" else pass_name,
                          reg["samples"][key]["kind"] == "eft")
    state = pass_state(ctx)
    result = ho.send_histogram({"met_pt": template()}, "met_pt", VALUES, LABELS, WEIGHT, ("met_pt",), state)
    if end:
        ho.end_pass(state)
    return result


def run_chunk(reg, key, md, sumw=1.0, passes=("nominal",), lumis=None):
    ho.begin_chunk(reg, key, md, list(passes))
    for pass_name in passes:
        send_pass(reg, key, md, pass_name)
    is_data = reg["samples"][key]["kind"] == "data"
    return ho.end_chunk(reg, key, md, md["entrystop"] - md["entrystart"], sumw=None if is_data else sumw, lumis=lumis)


def file_record(i, events=20, runs_sumw=None, runs_count=None, data=False):
    return {"fileuuid": file_uuid(i), "events": events, "runs_sumw": None if data else runs_sumw,
            "runs_count": None if data else (events if runs_count is None else runs_count)}


def production(reg, key, n_files=2, sumws=(1.25, 2.25), run=True, base=0, lumis=None):
    data = reg["samples"][key]["kind"] == "data"
    manifest = []
    total = 0.0
    for i in range(base, base + n_files):
        name = f"f{i}"
        chunks = [ho.chunk_id(file_uuid(i), 0, 10), ho.chunk_id(file_uuid(i), 10, 20)]
        ho.begin_job(reg, key, {name: chunks}, {name: file_record(i, runs_sumw=sum(sumws), data=data)})
        if run:
            for j, ((start, stop), sumw) in enumerate(zip(((0, 10), (10, 20)), sumws)):
                chunk_lumis = None if lumis is None else lumis[2 * (i - base) + j]
                run_chunk(reg, key, meta(reg, key, file_uuid(i), start, stop), sumw=sumw, lumis=chunk_lumis)
        manifest.append(name)
        total += sum(sumws)
    return manifest, total


def cells(h):
    cats = [ax for ax in h.axes if isinstance(ax, hist.axis.StrCategory)]
    view = h.view(flow=True)
    return {labels: view[tuple(ax.index(label) for ax, label in zip(cats, labels))]
            for labels in itertools.product(*[list(ax) for ax in cats])}


def assert_same_hist(a, b):
    ca, cb = cells(a), cells(b)
    assert set(ca) == set(cb)
    for labels in ca:
        assert np.array_equal(ca[labels]["value"], cb[labels]["value"])
        assert np.array_equal(ca[labels]["variance"], cb[labels]["variance"])
    assert [ax.__dict__ for ax in a.axes] == [ax.__dict__ for ax in b.axes]
    assert a.__dict__ == b.__dict__


def local_fill():
    local = template()
    wzmod.fill_histogram({"met_pt": local}, "met_pt", VALUES, LABELS, WEIGHT, ("met_pt",), pass_state())
    return local


def remote_hist(reg, key, name="met_pt"):
    return ho.remote(reg, reg["samples"][key]["hists"][name])


def ledger_items(reg, key, state=None):
    snap = ho.remote(reg, reg["samples"][key]["ledger"]).snapshot()
    return [k for k, v in snap.items() if v["value"][0] > 0 and (state is None or k[2] == state)]


def load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class FakeRpcError(grpc.RpcError):
    def __init__(self, status, text="fake details"):
        self.status = status
        self.text = text

    def code(self):
        return self.status

    def details(self):
        return self.text


REAL_FILL_MANY = histserv.RemoteHist.fill_many


def wrap_fill_many(monkeypatch, behaviour):
    calls = []

    def wrapper(self, fills, **kwargs):
        fills = list(fills)
        calls.append((self.hist_id, kwargs))
        return behaviour(self, fills, kwargs, len(calls))
    monkeypatch.setattr(histserv.RemoteHist, "fill_many", wrapper)
    return calls


def fail_target(reg, key, status, before=True):
    target = reg["samples"][key]["hists"]["met_pt"]

    def behaviour(self, fills, kwargs, n):
        if self.hist_id == target:
            if not before:
                REAL_FILL_MANY(self, fills, **kwargs)
            raise FakeRpcError(status)
        return REAL_FILL_MANY(self, fills, **kwargs)
    return behaviour


def open_chunk(reg, key, i=0, start=0, stop=10, events=10):
    chunk = ho.chunk_id(file_uuid(i), start, stop)
    ho.begin_job(reg, key, {f"f{i}": [chunk]}, {f"f{i}": file_record(i, events=events, runs_sumw=1.0)})
    md = meta(reg, key, file_uuid(i), start, stop)
    ho.begin_chunk(reg, key, md, ["nominal"])
    return md


def problems_of(reg, key, manifest):
    return {(category, cls): names for category, cls, names in ho.audit(reg, key, manifest)[0]}


def test_send_equals_local_fill(server, synthetic):
    reg = synthetic_registry(server)
    md = open_chunk(reg, DAS_S)
    assert send_pass(reg, DAS_S, md, end=False) == ho.SENT
    local = local_fill()
    snapshot = remote_hist(reg, DAS_S).snapshot()
    assert_same_hist(ho.rebuild(template(), snapshot, True), local)
    other = ho.rebuild(template(), snapshot, False)
    assert other._variance_known is False
    assert {k: v for k, v in other.__dict__.items() if k != "_variance_known"} == \
        {k: v for k, v in local.__dict__.items() if k != "_variance_known"}
    assert snapshot.to_hist().__dict__ != template().__dict__
    with pytest.raises(grpc.RpcError) as error:
        histserv.Client(server).connect(reg["samples"][DAS_S]["hists"]["met_pt"])
    assert error.value.code() == grpc.StatusCode.NOT_FOUND


def test_three_way_result(server, synthetic):
    reg = synthetic_registry(server)
    md = open_chunk(reg, DAS_S)
    ctx = ho.pass_context(reg, DAS_S, md, None, False)
    empty_state = pass_state(ctx)
    empty_state["channels"] = {"chEmpty": ["triggers", "empty"]}
    assert ho.send_histogram({"met_pt": template()}, "met_pt", VALUES, LABELS, WEIGHT, ("met_pt",), empty_state) == ho.EMPTY
    assert list(remote_hist(reg, DAS_S).snapshot().items()) == []
    ho.ATTEMPTS.clear()
    ho.begin_chunk(reg, DAS_S, md, ["nominal"])
    assert send_pass(reg, DAS_S, md, end=False) == ho.SENT
    first = cells(ho.rebuild(template(), remote_hist(reg, DAS_S).snapshot(), True))
    ho.begin_chunk(reg, DAS_S, md, ["nominal"])
    assert send_pass(reg, DAS_S, md, end=False) == ho.DUPLICATE
    again = cells(ho.rebuild(template(), remote_hist(reg, DAS_S).snapshot(), True))
    assert again.keys() == first.keys() and all(np.array_equal(again[k], first[k]) for k in first)
    unique_id = [reg["tag"], DAS_S, md["fileuuid"], 0, 10, "nominal", "met_pt"]
    assert remote_hist(reg, DAS_S).was_filled_with_unique_id(unique_id)
    md2 = meta(reg, DAS_S, md["fileuuid"], 10, 20)
    ho.begin_chunk(reg, DAS_S, md2, ["nominal"])
    assert send_pass(reg, DAS_S, md2, end=False) == ho.SENT
    doubled = cells(ho.rebuild(template(), remote_hist(reg, DAS_S).snapshot(), True))
    for labels, view in first.items():
        assert np.allclose(doubled[labels]["value"], 2 * view["value"], rtol=1e-12)


def test_repeat_rules(server, synthetic, monkeypatch):
    reg = synthetic_registry(server)
    md = open_chunk(reg, DAS_S)
    ctx = ho.pass_context(reg, DAS_S, md, None, False)
    state = pass_state(ctx)
    state["channels"] = {"chEmpty": ["triggers", "empty"]}
    assert ho.send_histogram({"met_pt": template()}, "met_pt", VALUES, LABELS, WEIGHT, ("met_pt",), state) == ho.EMPTY
    with pytest.raises(RuntimeError):
        ho.send_histogram({"met_pt": template()}, "met_pt", VALUES, LABELS, WEIGHT, ("met_pt",), state)
    with pytest.raises(RuntimeError):
        ho.begin_chunk(reg, DAS_S, md, ["nominal", "nominal"])
    with pytest.raises(RuntimeError):
        ho.begin_chunk(reg, DAS_S, md, ["nominal", "started"])
    with pytest.raises(RuntimeError):
        ho.end_chunk(reg, DAS_S, md, 10, sumw=1.0)
    ho.end_pass(state)
    calls = wrap_fill_many(monkeypatch, lambda self, fills, kwargs, n: REAL_FILL_MANY(self, fills, **kwargs))
    state2 = dict(state, histserv=dict(ctx, keys=set()))
    with pytest.raises(RuntimeError):
        ho.send_histogram({"met_pt": template()}, "met_pt", VALUES, LABELS, WEIGHT, ("met_pt",), state2)
    assert calls == []
    ho.ATTEMPTS.clear()
    with pytest.raises(RuntimeError):
        ho.pass_context(reg, DAS_S, md, None, False)


def test_poison_half_merge(server, synthetic, monkeypatch):
    reg = synthetic_registry(server)
    target = reg["samples"][DAS_S]["hists"]["met_pt"]

    def behaviour(self, fills, kwargs, n):
        if self.hist_id == target:
            REAL_FILL_MANY(self, fills[:len(fills) // 2], timeout=kwargs["timeout"])
            raise FakeRpcError(grpc.StatusCode.INTERNAL)
        return REAL_FILL_MANY(self, fills, **kwargs)
    wrap_fill_many(monkeypatch, behaviour)
    md = open_chunk(reg, DAS_S)
    with pytest.raises(ho.HistservPoisoned) as error:
        send_pass(reg, DAS_S, md, end=False)
    assert isinstance(error.value.__cause__, FakeRpcError)
    assert "poison recorded" in str(error.value)
    chunk = ho.chunk_id(file_uuid(0), 0, 10)
    assert ledger_items(reg, DAS_S, "poisoned") == [("-", chunk, "poisoned", "nominal")]
    calls = wrap_fill_many(monkeypatch, lambda self, fills, kwargs, n: REAL_FILL_MANY(self, fills, **kwargs))
    for call in (lambda: send_pass(reg, DAS_S, md, end=False), lambda: ho.begin_chunk(reg, DAS_S, md, ["nominal"]),
                 lambda: open_chunk(reg, DAS_S, i=1)):
        with pytest.raises(ho.HistservPoisoned):
            call()
    assert calls == []
    ho.POISONED.clear()
    with pytest.raises(ho.HistservPoisoned, match=chunk):
        ho.check_poison(reg, DAS_S)
    with pytest.raises(ho.HistservPoisoned, match=chunk):
        open_chunk(reg, DAS_S, i=1)
    assert ("poisoned", "inconsistent") in problems_of(reg, DAS_S, ["f0"])


@pytest.mark.parametrize("status", [s for s in grpc.StatusCode if s != grpc.StatusCode.OK], ids=lambda s: s.name)
def test_status_classification(server, synthetic, monkeypatch, status):
    reg = synthetic_registry(server)
    md = open_chunk(reg, DAS_S)
    wrap_fill_many(monkeypatch, fail_target(reg, DAS_S, status))
    if status == grpc.StatusCode.ALREADY_EXISTS:
        assert send_pass(reg, DAS_S, md, end=False) == ho.DUPLICATE
    elif status in ho.NON_POISONING_CODES:
        with pytest.raises(FakeRpcError) as error:
            send_pass(reg, DAS_S, md, end=False)
        assert error.value.code() == status
        assert ho.POISONED == {}
        assert ledger_items(reg, DAS_S, "poisoned") == []
    else:
        with pytest.raises(ho.HistservPoisoned):
            send_pass(reg, DAS_S, md, end=False)
        assert len(ledger_items(reg, DAS_S, "poisoned")) == 1


def test_never_reached_and_lost_response(server, synthetic, monkeypatch):
    local = cells(local_fill())
    for before, retry in ((True, ho.SENT), (False, ho.DUPLICATE)):
        reg = synthetic_registry(server)
        md = open_chunk(reg, DAS_S)
        status = grpc.StatusCode.UNAVAILABLE if before else grpc.StatusCode.DEADLINE_EXCEEDED
        wrap_fill_many(monkeypatch, fail_target(reg, DAS_S, status, before=before))
        with pytest.raises(FakeRpcError):
            send_pass(reg, DAS_S, md, end=False)
        assert ho.POISONED == {}
        monkeypatch.setattr(histserv.RemoteHist, "fill_many", REAL_FILL_MANY)
        ho.begin_chunk(reg, DAS_S, md, ["nominal"])
        assert send_pass(reg, DAS_S, md, end=False) == retry
        got = cells(ho.rebuild(template(), remote_hist(reg, DAS_S).snapshot(), True))
        assert all(np.array_equal(got[labels]["value"], local[labels]["value"]) for labels in local)


def test_read_only_failures_never_poison(server, synthetic, monkeypatch):
    reg = synthetic_registry(server)
    calls = wrap_fill_many(monkeypatch, lambda self, fills, kwargs, n: REAL_FILL_MANY(self, fills, **kwargs))

    def failing_snapshot(self, **kwargs):
        raise FakeRpcError(grpc.StatusCode.INTERNAL)
    real_snapshot = histserv.client.RemoteHistSlice.snapshot
    monkeypatch.setattr(histserv.client.RemoteHistSlice, "snapshot", failing_snapshot)
    with pytest.raises(FakeRpcError):
        ho.check_poison(reg, DAS_S)
    assert calls == [] and ho.POISONED == {}
    monkeypatch.setattr(histserv.client.RemoteHistSlice, "snapshot", real_snapshot)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    closed = dict(reg, address=f"127.0.0.1:{port}")
    start = time.monotonic()
    with pytest.raises(grpc.RpcError) as error:
        ho.begin_job(closed, DAS_S, {"f0": []}, {"f0": file_record(0, runs_sumw=1.0)})
    assert error.value.code() == grpc.StatusCode.UNAVAILABLE
    assert time.monotonic() - start < 1.0
    assert ho.POISONED == {}


def test_poison_keys(server, synthetic, monkeypatch):
    def fail_sumw(reg, key):
        sumw = reg["samples"][key]["sumw"]

        def behaviour(self, fills, kwargs, n):
            if self.hist_id == sumw:
                raise FakeRpcError(grpc.StatusCode.INTERNAL)
            return REAL_FILL_MANY(self, fills, **kwargs)
        return behaviour

    reg = synthetic_registry(server)
    wrap_fill_many(monkeypatch, fail_sumw(reg, DAS_S))
    with pytest.raises(ho.HistservPoisoned):
        open_chunk(reg, DAS_S)
    assert ledger_items(reg, DAS_S, "poisoned") == [("f0", file_uuid(0), "poisoned", "begin_job")]
    assert ("poisoned", "inconsistent") in problems_of(reg, DAS_S, ["f0"])
    monkeypatch.setattr(histserv.RemoteHist, "fill_many", REAL_FILL_MANY)
    ho.POISONED.clear()
    reg = synthetic_registry(server)
    chunk = ho.chunk_id(file_uuid(0), 0, 10)
    ho.begin_job(reg, DAS_S, {"f0": [chunk]}, {"f0": file_record(0, events=10, runs_sumw=1.0)})
    ledger = reg["samples"][DAS_S]["ledger"]

    def fail_started(self, fills, kwargs, n):
        if self.hist_id == ledger and fills[0]["state"] == "started":
            raise FakeRpcError(grpc.StatusCode.INTERNAL)
        return REAL_FILL_MANY(self, fills, **kwargs)
    wrap_fill_many(monkeypatch, fail_started)
    md = meta(reg, DAS_S, file_uuid(0), 0, 10)
    with pytest.raises(ho.HistservPoisoned):
        ho.begin_chunk(reg, DAS_S, md, ["nominal"])
    assert ledger_items(reg, DAS_S, "poisoned") == [("-", chunk, "poisoned", "started")]
    monkeypatch.setattr(histserv.RemoteHist, "fill_many", REAL_FILL_MANY)
    ho.POISONED.clear()
    reg = synthetic_registry(server)
    md = open_chunk(reg, DAS_S)
    send_pass(reg, DAS_S, md)
    wrap_fill_many(monkeypatch, fail_sumw(reg, DAS_S))
    with pytest.raises(ho.HistservPoisoned):
        ho.end_chunk(reg, DAS_S, md, 10, sumw=1.0)
    assert ledger_items(reg, DAS_S, "poisoned") == [("-", chunk, "poisoned", "done")]
    assert any(chunk in name for name in problems_of(reg, DAS_S, ["f0"])[("poisoned", "inconsistent")])


def test_poison_write_backoff(server, synthetic, monkeypatch):
    sleeps = []
    monkeypatch.setattr(ho.time, "sleep", sleeps.append)

    def ledger_behaviour(reg, ledger_fails):
        target, ledger = reg["samples"][DAS_S]["hists"]["met_pt"], reg["samples"][DAS_S]["ledger"]
        state = {"n": 0}

        def behaviour(self, fills, kwargs, n):
            if self.hist_id == target:
                raise FakeRpcError(grpc.StatusCode.INTERNAL)
            if self.hist_id == ledger and fills[0]["state"] == "poisoned":
                state["n"] += 1
                return ledger_fails(self, fills, kwargs, state["n"])
            return REAL_FILL_MANY(self, fills, **kwargs)
        return behaviour

    def twice_unavailable(self, fills, kwargs, n):
        if n <= 2:
            raise FakeRpcError(grpc.StatusCode.UNAVAILABLE)
        return REAL_FILL_MANY(self, fills, **kwargs)

    def merged_then_lost(self, fills, kwargs, n):
        if n == 1:
            REAL_FILL_MANY(self, fills, **kwargs)
            raise FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
        return REAL_FILL_MANY(self, fills, **kwargs)

    def always(self, fills, kwargs, n):
        raise FakeRpcError(grpc.StatusCode.UNAVAILABLE)

    for ledger_fails, recorded, delays in ((twice_unavailable, True, ho.POISON_WRITE_DELAYS[:3]),
                                           (merged_then_lost, True, ho.POISON_WRITE_DELAYS[:2]),
                                           (always, False, ho.POISON_WRITE_DELAYS)):
        ho.POISONED.clear()
        sleeps.clear()
        reg = synthetic_registry(server)
        md = open_chunk(reg, DAS_S)
        wrap_fill_many(monkeypatch, ledger_behaviour(reg, ledger_fails))
        with pytest.raises(ho.HistservPoisoned) as error:
            send_pass(reg, DAS_S, md, end=False)
        assert tuple(sleeps) == tuple(delays)
        if recorded:
            assert "poison recorded" in str(error.value)
            monkeypatch.setattr(histserv.RemoteHist, "fill_many", REAL_FILL_MANY)
            assert len(ledger_items(reg, DAS_S, "poisoned")) == 1
        else:
            assert "poison NOT recorded on the server" in str(error.value)


def test_poisoned_pickles_and_is_poison():
    error = ho.HistservPoisoned("x")
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is ho.HistservPoisoned and str(copy) == "x"
    for build in ("cause", "context", "deep"):
        outer = RuntimeError("outer")
        if build == "cause":
            outer.__cause__ = error
        elif build == "context":
            outer.__context__ = error
        else:
            middle = ValueError("middle")
            middle.__cause__ = error
            outer.__context__ = middle
        assert ho.is_poison(outer)
    loop = RuntimeError("loop")
    loop.__context__ = loop
    assert not ho.is_poison(loop)
    assert ho.is_poison(Exception("Failed processing file: WorkItem(...). The error was: HistservPoisoned('x')."))
    assert not ho.is_poison(RuntimeError("x"))


def test_poison_marker(server, synthetic, monkeypatch, tmp_path):
    marker = tmp_path / "m.jsonl"
    monkeypatch.setenv(ho.POISON_FILE_ENV, str(marker))
    monkeypatch.setattr(ho.time, "sleep", lambda s: None)
    reg = synthetic_registry(server)
    md = open_chunk(reg, DAS_S)
    target = reg["samples"][DAS_S]["hists"]["met_pt"]

    def internal(self, fills, kwargs, n):
        if self.hist_id == target:
            raise FakeRpcError(grpc.StatusCode.INTERNAL, "unexpected server error during FillMany: boom")
        return REAL_FILL_MANY(self, fills, **kwargs)
    wrap_fill_many(monkeypatch, internal)
    with pytest.raises(ho.HistservPoisoned) as error:
        send_pass(reg, DAS_S, md, end=False)
    assert "unexpected server error during FillMany: boom" in str(error.value)
    records = ho.read_poison_marker()
    assert [r["state"] for r in records] == ["pending", "recorded"]
    assert marker.stat().st_mode & 0o777 == 0o600
    reason = ho.poison_reason(Exception("unrelated"))
    assert DAS_S in reason and "nominal" in reason
    marker.unlink()
    ho.POISONED.clear()
    reg = synthetic_registry(server)
    md = open_chunk(reg, DAS_S)
    target = reg["samples"][DAS_S]["hists"]["met_pt"]
    ledger = reg["samples"][DAS_S]["ledger"]
    timeouts = []

    def all_fail(self, fills, kwargs, n):
        if self.hist_id == target:
            raise FakeRpcError(grpc.StatusCode.INTERNAL)
        if self.hist_id == ledger and fills[0]["state"] == "poisoned":
            timeouts.append(kwargs["timeout"])
            raise FakeRpcError(grpc.StatusCode.UNAVAILABLE)
        return REAL_FILL_MANY(self, fills, **kwargs)
    wrap_fill_many(monkeypatch, all_fail)
    with pytest.raises(ho.HistservPoisoned):
        send_pass(reg, DAS_S, md, end=False)
    final = ho.read_poison_marker()[-1]
    assert final["state"] == "not recorded" and final["attempts"] == len(ho.POISON_WRITE_DELAYS)
    assert tuple(timeouts) == ho.POISON_WRITE_TIMEOUTS
    text = marker.read_text().splitlines()
    marker.write_text(text[0] + "\n" + text[1][:15] + "\n")
    assert ho.poison_reason(None)
    monkeypatch.delenv(ho.POISON_FILE_ENV)
    for env in (None, str(tmp_path)):
        if env:
            monkeypatch.setenv(ho.POISON_FILE_ENV, env)
        ho.POISONED.clear()
        reg = synthetic_registry(server)
        md = open_chunk(reg, DAS_S)
        wrap_fill_many(monkeypatch, fail_target(reg, DAS_S, grpc.StatusCode.INTERNAL))
        with pytest.raises(ho.HistservPoisoned) as error:
            send_pass(reg, DAS_S, md, end=False)
        if env is None:
            assert not marker.exists() or marker.read_text().count("\n") == 2
            assert ho.poison_reason(Exception("unrelated")) is None
        else:
            assert "; poison marker not written: " in str(error.value)


LOKY_DRIVER = """
import json, os, sys
import numpy, uproot
from coffea import processor
from coffea.nanoevents import BaseSchema
import qawa.datadriven as datadriven
import qawa.histserv_output as ho
path = sys.argv[1]
with uproot.recreate(os.path.join(path, "events.root")) as f:
    f["Events"] = {"x": numpy.arange(200)}
os.environ[ho.POISON_FILE_ENV] = os.path.join(path, "marker.jsonl")
import toyproc
runner = processor.Runner(processor.FuturesExecutor(workers=2), schema=BaseSchema, chunksize=50)
err = None
try:
    runner({"S": [os.path.join(path, "events.root")]}, toyproc.Toy(), treename="Events")
except Exception as error:
    err = error
chain, e = [], err
while e is not None and len(chain) < 10:
    chain.append(type(e).__name__)
    e = e.__cause__ or e.__context__
records = ho.read_poison_marker()
out = {"chain": chain, "is_poison": ho.is_poison(err), "reason": ho.poison_reason(err), "records": records,
       "pid": os.getpid()}
os.remove(os.environ[ho.POISON_FILE_ENV])
out["reason_text"] = ho.poison_reason(err)
out["unrelated"] = ho.poison_reason(Exception("unrelated"))
print("RESULT " + json.dumps(out))
"""

LOKY_TOY = """
import json, os
import grpc, histserv, numpy
from coffea import processor
import qawa.datadriven as datadriven
import qawa.histserv_output as ho
REG = json.load(open(os.path.join(os.path.dirname(__file__), "reg.json")))
TARGET = REG["samples"]["{das}"]["hists"]["met_pt"]
REAL = histserv.RemoteHist.fill_many
FAIL_LEDGER = {fail_ledger}


class Fake(grpc.RpcError):
    def __init__(self, status):
        self.status = status

    def code(self):
        return self.status

    def details(self):
        return "toy"


def wrapper(self, fills, **kwargs):
    if self.hist_id == TARGET:
        raise Fake(grpc.StatusCode.INTERNAL)
    if FAIL_LEDGER:
        raise Fake(grpc.StatusCode.UNAVAILABLE)
    return REAL(self, fills, **kwargs)


histserv.RemoteHist.fill_many = wrapper


class Toy(processor.ProcessorABC):
    def process(self, events):
        if FAIL_LEDGER:
            ho.time.sleep = lambda s: None
        if events.metadata["entrystart"] == 0:
            fill = {{"channel": "c", "systematic": "nominal", "met_pt": numpy.zeros(1), "weight": numpy.ones(1)}}
            ho.send(REG, "{das}", TARGET, [fill], ["toy", 0], ("-", "c0", "nominal"))
        return {{}}

    def postprocess(self, accumulator):
        return accumulator
"""


@pytest.mark.parametrize("fail_ledger", [False, True], ids=["recorded", "not-recorded"])
def test_poison_across_loky_pool(server, synthetic, tmp_path, fail_ledger):
    reg = synthetic_registry(server)
    ho.write_registry(str(tmp_path / "reg.json"), reg)
    (tmp_path / "toyproc.py").write_text(LOKY_TOY.format(das=DAS_S, fail_ledger=fail_ledger))
    (tmp_path / "driver.py").write_text(LOKY_DRIVER)
    env = {**os.environ, **CAPS, "PYTHONPATH": f"{tmp_path}:{os.path.join(ROOT, 'src')}"}
    run = subprocess.run([sys.executable, str(tmp_path / "driver.py"), str(tmp_path)], env=env, capture_output=True,
                         text=True, timeout=300)
    lines = [line for line in run.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, run.stdout[-2000:] + run.stderr[-4000:]
    out = json.loads(lines[-1][len("RESULT "):])
    assert "HistservPoisoned" not in out["chain"]
    expected = "poison NOT recorded" if fail_ledger else "poison recorded"
    assert expected in out["reason"]
    assert [r["state"] for r in out["records"]] == ["pending", "not recorded" if fail_ledger else "recorded"]
    assert out["records"][0]["pid"] != out["pid"]
    assert out["reason_text"] and out["unrelated"] is None


def test_file_of_mapping(server, synthetic):
    fileuuid = uuid.UUID(int=7).bytes
    items = [WorkItem("S", "/tmp/store/x.root", "Events", 0, 10, fileuuid),
             WorkItem("S", "/tmp/store/x.root", "Events", 10, 20, fileuuid)]
    mapping = {"/tmp/store/x.root": "/store/x.root"}
    expected = ho.expected_chunks(items, mapping)
    assert list(expected) == ["/store/x.root"] and len(expected["/store/x.root"]) == 2
    records = ho.file_records(items, mapping, {"/store/x.root": (1.0, 20)})
    reg = synthetic_registry(server)
    ho.begin_job(reg, DAS_S, expected, records)
    assert {k[0] for k in ledger_items(reg, DAS_S)} == {"/store/x.root"}
    with pytest.raises(ValueError):
        ho.begin_job(reg, DAS_S, {"/tmp/store/x.root": expected["/store/x.root"]}, records)
    with pytest.raises(ValueError, match="/tmp/store/x.root"):
        ho.expected_chunks(items, {})
    with pytest.raises(ValueError):
        ho.file_records(items[1:], mapping, None)


def test_non_grpc_error_does_not_poison(server, synthetic, monkeypatch):
    reg = synthetic_registry(server)

    def raising(self, fills, kwargs, n):
        raise ValueError("not an RPC error")
    wrap_fill_many(monkeypatch, raising)
    target = reg["samples"][DAS_S]["hists"]["met_pt"]
    with pytest.raises(ValueError, match="not an RPC error"):
        ho.send(reg, DAS_S, target, [{"channel": "c"}], ["x"], ("-", "c0", "nominal"))
    assert ho.POISONED == {}


def test_consistency_errors(server, synthetic):
    reg = synthetic_registry(server, samples={DAS_S: "mc", DAS_C: "data", "/E/RunX-v1/NANOAODSIM": "eft"})
    md = open_chunk(reg, DAS_S)
    for change, eft in ((dict(dataset="other"), False), (dict(fileuuid=""), False), ({}, True),
                        (dict(is_signal=True), False)):
        with pytest.raises(ValueError):
            ho.pass_context(reg, DAS_S, dict(md, **change), None, eft)
    with pytest.raises(ValueError):
        ho.pass_context(reg, DAS_C, dict(md, dataset="Muon0", is_data=False), None, False)
    eft_key = pk.sample_key("/E/RunX-v1/NANOAODSIM", eft=True)
    with pytest.raises(ValueError):
        ho.pass_context(reg, eft_key, dict(md, dataset="E"), None, False)
    with pytest.raises(ValueError):
        ho.begin_job(reg, DAS_S, {"f0": []}, {"f1": file_record(1, runs_sumw=1.0)})
    with pytest.raises(ValueError):
        ho.begin_job(reg, "/U/RunX-v1/NANOAODSIM", {"f0": []}, {"f0": file_record(0, runs_sumw=1.0)})


def test_nondeterministic_reason():
    era2018 = ho.make_processor(dict(OPTIONS, era="2018", version="v9"))
    era2024 = ho.make_processor(OPTIONS)
    for kind in ("mc", "signal", "eft"):
        reason = ho.nondeterministic_reason(era2018, kind)
        assert "roccor.py:31" in reason and "N15" in reason
        assert ho.nondeterministic_reason(era2024, kind) is None
    assert ho.nondeterministic_reason(era2018, "data") is None
    assert ho.nondeterministic_reason(era2024, "data") is None


def hand_registry(kind, options=OPTIONS, key=DAS_S, variations=None):
    variations = variations or {name: options[name] for name in ("pol", "split_by_charge", "split_FV")}
    return {"options": dict(options), "samples": {key: {"kind": kind, "variations": dict(variations, eft=False, dd=None)}}}


def test_constructor_consistency_errors():
    reg = hand_registry("mc")
    for kwargs in (dict(histserv_registry=reg), dict(histserv_sample=DAS_S),
                   dict(histserv_registry=reg, histserv_sample=DAS_X),
                   dict(histserv_registry=hand_registry("mc", dict(OPTIONS, split_FV=True)), histserv_sample=DAS_S),
                   dict(histserv_registry=hand_registry("mc", variations=dict(pol=True, split_by_charge=False,
                                                                                split_FV=False)),
                        histserv_sample=DAS_S)):
        with pytest.raises(ValueError):
            wzmod.wzinclusive_processor(era="2024", version="v15", **kwargs)
    proc = wzmod.wzinclusive_processor(era="2024", version="v15", histserv_registry=reg, histserv_sample=DAS_S)
    assert proc.histserv_registry is reg and proc.histserv_sample == DAS_S


def test_constructor_refuses_nondeterministic():
    options = dict(OPTIONS, era="2018", version="v9")
    with pytest.raises(ValueError, match="N15"):
        wzmod.wzinclusive_processor(era="2018", version="v9", histserv_registry=hand_registry("signal", options),
                                    histserv_sample=DAS_S)
    wzmod.wzinclusive_processor(era="2018", version="v9", run_period="C", histserv_registry=hand_registry("data", options),
                                histserv_sample=DAS_S)


M1_FILE = ("/Volumes/WDMac/SMQawaTestFiles/store/mc/RunIII2024Summer24NanoAODv15/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/"
           "NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/100000/test.root")
M3_FILE = ("/Volumes/WDMac/SMQawaTestFiles/store/mc/RunIII2024Summer24NanoAODv15/DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_"
           "amcatnloFXFX-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v1/90000/test.root")
D1_FILE = "/Volumes/WDMac/SMQawaTestFiles/store/data/Run2024C/Muon0/NANOAOD/MINIv6NANOv15-v1/2530000/test.root"
FAKE_TERMS = ("sm", "lin_cW", "quad_cW")


def needs(*paths):
    return pytest.mark.skipif(not all(os.path.exists(p) for p in paths), reason="local 2024 test files not available")


class FakeParameterisation:
    terms = FAKE_TERMS

    def decompose(self, x):
        return x

    def validate(self, theta):
        return []

    def active_terms(self, theta=None):
        return list(self.terms)


def fake_theta(event, points):
    gw = ak.to_numpy(event.genWeight).astype(np.float64)
    return np.stack([gw * (1.0 + 0.1 * t) for t in range(len(points))], axis=1)


def file_events(path, n, dataset, is_data, is_signal):
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    return NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="virtual", entry_start=0, entry_stop=n,
        metadata={"dataset": dataset, "is_data": is_data, "is_signal": is_signal, "is_flashsim": False},
    ).events()


KEY_MODES = {
    "mc": (M3_FILE, "mc", {}, {}),
    "signal": (M1_FILE, "signal", {}, {}),
    "data": (D1_FILE, "data", {}, dict(run_period="C")),
    "pol-signal": (M1_FILE, "signal", dict(pol=True), {}),
    "pol-mc": (M3_FILE, "mc", dict(pol=True), {}),
    "eft": (M1_FILE, "eft", {}, {}),
}


@needs(M1_FILE, M3_FILE, D1_FILE)
@pytest.mark.parametrize("mode", sorted(KEY_MODES))
def test_registered_keys_equal_output_keys(mode, monkeypatch):
    path, kind, option_changes, kwargs = KEY_MODES[mode]
    options = dict(OPTIONS, **option_changes)
    process_shift = wzmod.wzinclusive_processor.process_shift
    monkeypatch.setattr(wzmod.wzinclusive_processor, "process_shift",
                        lambda self, event, shift_name='': process_shift(self, event, shift_name) if shift_name is None else {})
    if kind == "eft":
        monkeypatch.setattr(wzmod, "points_from_events", lambda event: (FakeParameterisation(), FAKE_TERMS))
        monkeypatch.setattr(wzmod, "stack_named_weights", fake_theta)
    proc = wzmod.wzinclusive_processor(**options, **kwargs)
    dataset = "Muon0_Run2024C" if kind == "data" else "D"
    out = proc.process(file_events(path, 300, dataset, kind == "data", kind in ("signal", "eft")))[dataset]
    assert out[wzmod.KIND_KEY] == {kind}
    assert set(out) - {lumi_record.LUMI_KEY, wzmod.KIND_KEY} == set(ho.templates(ho.make_processor(options), kind))


def assert_close_hist(a, b):
    ca, cb = cells(a), cells(b)
    assert set(ca) == set(cb)
    for labels in ca:
        for field in ("value", "variance"):
            assert np.allclose(ca[labels][field], cb[labels][field], rtol=1e-9, atol=0, equal_nan=True), (labels, field)
    assert [ax.__dict__ for ax in a.axes] == [ax.__dict__ for ax in b.axes]
    assert a.__dict__ == b.__dict__


@needs(M1_FILE)
def test_processor_c4_equals_c1(server, tmp_path, monkeypatch):
    import uproot
    from coffea import processor
    from coffea.nanoevents import NanoAODSchema
    das = pk.das_name_from_path(M1_FILE)
    dataset = pk.primary_dataset(das)
    assert dataset == "WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8"
    reg = ho.register(server, "t", OPTIONS, {das: "signal"}, uuid.uuid4().hex)
    fileset = {dataset: {"files": {M1_FILE: "Events"}, "metadata": {"is_data": False, "is_signal": True,
                                                                     "is_flashsim": False}}}
    runner = processor.Runner(processor.IterativeExecutor(), chunksize=250, schema=NanoAODSchema, format="root")
    items = list(itertools.islice(runner.preprocess(fileset), 2))
    assert len(items) == 2
    with uproot.open(M1_FILE) as f:
        runs = f["Runs"].arrays(["genEventSumw", "genEventCount"], library="np")
        record = {"fileuuid": str(uuid.UUID(bytes=items[0].fileuuid)), "events": f["Events"].num_entries,
                  "runs_sumw": float(runs["genEventSumw"].sum()), "runs_count": int(runs["genEventCount"].sum())}
        harness_sumw = sum(float(np.sum(f["Events"]["genWeight"].array(library="np", entry_start=item.entrystart,
                                                                         entry_stop=item.entrystop), dtype=np.float64))
                           for item in items)
    ho.begin_job(reg, das, ho.expected_chunks(items, {M1_FILE: M1_FILE}), {M1_FILE: record})
    c1 = runner(items, processor_instance=wzmod.wzinclusive_processor(era="2024", version="v15"))[dataset]
    assert c1.pop(wzmod.KIND_KEY) == {"signal"}
    c4_proc = wzmod.wzinclusive_processor(era="2024", version="v15", histserv_registry=reg, histserv_sample=das)
    results = []
    real_send = ho.send_histogram

    def counting(*args):
        result = real_send(*args)
        results.append(result)
        return result
    monkeypatch.setattr(ho, "send_histogram", counting)
    assert runner(items, processor_instance=c4_proc) == {dataset: {}}
    first = list(results)
    assert ho.collect(reg, {das: [M1_FILE]}, str(tmp_path / "A"), per_entry=True) == 0
    name = pk.sample_file_stem(das) + "_histograms.pkl.gz"
    content = load(tmp_path / "A" / name)[dataset]
    assert set(content["hist"]) == set(c1)
    for key, h in c1.items():
        assert content["hist"][key]._variance_known is False and h._variance_known is False
        assert_close_hist(content["hist"][key], h)
    assert np.isclose(content["sumw"], harness_sumw, rtol=1e-9, atol=0)
    results.clear()
    assert runner(items, processor_instance=c4_proc) == {dataset: {}}
    assert ho.DUPLICATE in results and results == [ho.DUPLICATE if r == ho.SENT else r for r in first]
    assert ho.collect(reg, {das: [M1_FILE]}, str(tmp_path / "B"), per_entry=True) == 0
    with gzip.open(tmp_path / "A" / name) as first_file, gzip.open(tmp_path / "B" / name) as second_file:
        assert first_file.read() == second_file.read()


def run_cli(*args, env=None):
    return subprocess.run([sys.executable, "-m", "qawa.histserv_output", *args], capture_output=True, text=True,
                          env={**os.environ, **CAPS, **(env or {}), "PYTHONPATH": os.path.join(ROOT, "src")}, timeout=300)


def test_register_cli(server, tmp_path):
    output = tmp_path / "reg.json"
    (tmp_path / "reg.json.tmp").write_text("stale")
    os.chmod(tmp_path / "reg.json.tmp", 0o644)
    run = run_cli("register", "--address", server, "--tag", "t", "--era", "2024", "--version", "v15",
                  "--sample", f"{DAS_X}=mc", "--output", str(output))
    assert run.returncode == 0, run.stderr
    reg = ho.read_registry(str(output))
    assert output.stat().st_mode & 0o777 == 0o600
    entry = reg["samples"][DAS_X]
    assert set(entry["hists"]) == set(ho.make_processor(OPTIONS).build_histos())
    assert entry["dataset"] == "X" and entry["das"] == DAS_X and reg["max_chunks"] is None
    assert entry["variations"] == {"pol": False, "split_by_charge": False, "split_FV": False, "eft": False, "dd": None}
    env = {k: v for k, v in os.environ.items() if k != "HISTSERV_ADDRESS"}
    run = subprocess.run([sys.executable, "-m", "qawa.histserv_output", "register", "--tag", "t", "--era", "2024",
                          "--version", "v15", "--sample", f"{DAS_X}=mc", "--output", str(tmp_path / "none.json")],
                         capture_output=True, text=True, env={**env, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert run.returncode == 2 and not (tmp_path / "none.json").exists()
    count = histserv.Client(server).stats()["histogram_count"]
    run = run_cli("register", "--address", server, "--tag", "t", "--era", "2018", "--version", "v9",
                  "--sample", f"{DAS_X}=mc", "--output", str(tmp_path / "r2018.json"))
    assert run.returncode == 2 and "roccor.py:31" in run.stderr and "N15" in run.stderr
    assert not (tmp_path / "r2018.json").exists()
    assert histserv.Client(server).stats()["histogram_count"] == count
    run = run_cli("register", "--address", server, "--tag", "t", "--era", "2018", "--version", "v9",
                  "--sample", "/D/Run2018C-v1/NANOAOD=data", "--output", str(tmp_path / "d2018.json"))
    assert run.returncode == 0, run.stderr
    assert ho.read_registry(str(tmp_path / "d2018.json"))["poison_exit_code"] == 87


@pytest.mark.parametrize("samples,tag", [(["/X/Run*/NANOAODSIM=mc"], "t"), ([f"{DAS_X}=data"], "t"),
                                         ([f"{DAS_X}=eft", f"{DAS_X}=mc"], "t"),
                                         ([f"{DAS_X}=mc", "/X/RunY-v1/NANOAODSIM=signal"], "t"),
                                         ([f"{DAS_X}=mc"], "a/b"), ([f"{DAS_X}=mc"], "a.b")])
def test_register_refusals(server, tmp_path, samples, tag):
    args = ["register", "--address", server, "--tag", tag, "--era", "2024", "--version", "v15",
            "--output", str(tmp_path / "reg.json")]
    for sample in samples:
        args += ["--sample", sample]
    assert ho.main(args) == 2
    assert not (tmp_path / "reg.json").exists()


def test_complete_production(server, synthetic, tmp_path, monkeypatch):
    reg = synthetic_registry(server, samples={DAS_X: "mc"})
    manifest, total = production(reg, DAS_X)
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(ho.os, "replace", lambda a, b: (replaced.append(os.path.basename(b)), real_replace(a, b)))
    out = tmp_path / "D"
    assert ho.collect(reg, {DAS_X: manifest}, str(out), registry_path="reg.json") == 0
    monkeypatch.setattr(ho.os, "replace", real_replace)
    assert replaced[-1] == "collect.json"
    content = load(out / MC_FILE)
    assert list(content) == ["X"] and content["X"]["kind"] == "mc" and content["X"]["sumw"] == total
    assert set(content["X"]) == {"hist", "sumw", "kind"}
    assert content["X"]["hist"]["met_pt"]._variance_known is False
    assert load(out / final_name("Data")) == {} and load(out / final_name("MC-signal")) == {}
    summary = json.loads((out / "collect.json").read_text())
    assert summary["format"] == 3 and summary["layout"] == "final" and summary["class"] == "complete"
    assert summary["truncated"] is False
    assert summary["files"][MC_FILE] == {"kind": "mc", "variations": "", "datasets": {"X": [DAS_X]}}
    out2 = tmp_path / "D2"
    assert ho.collect(reg, {DAS_X: manifest}, str(out2), per_entry=True) == 0
    per_entry = load(out2 / "X__RunX-v1__NANOAODSIM_histograms.pkl.gz")
    assert sorted(os.listdir(out2)) == ["X__RunX-v1__NANOAODSIM_histograms.pkl.gz", "collect.json"]
    assert per_entry["X"]["sumw"] == content["X"]["sumw"]
    assert_same_hist(per_entry["X"]["hist"]["met_pt"], content["X"]["hist"]["met_pt"])
    assert json.loads((out2 / "collect.json").read_text())["layout"] == "per-entry"
    names = {name: sha(out / name) for name in os.listdir(out)}
    assert ho.collect(reg, {DAS_X: manifest}, str(out)) == 2
    assert {name: sha(out / name) for name in os.listdir(out)} == names
    for stray in (MC_FILE + ".tmp", "X__RunX-v1__NANOAODSIM_histograms.pkl.gz"):
        other = tmp_path / ("E" + stray[:3])
        other.mkdir()
        (other / stray).write_bytes(b"x")
        assert ho.collect(reg, {DAS_X: manifest}, str(other)) == 2
        assert (other / stray).read_bytes() == b"x" and os.listdir(other) == [stray]
    monkeypatch.chdir(tmp_path)
    with gzip.open(tmp_path / "histogram_99.pkl.gz", "wb") as f:
        pickle.dump({"X": {"sumw": 1e9}}, f)
    assert ho.collect(reg, {DAS_X: manifest}, str(tmp_path / "D3")) == 0
    assert load(tmp_path / "D3" / MC_FILE)["X"]["sumw"] == total


def refusal_case(server, case):
    reg = synthetic_registry(server, samples={DAS_X: "mc"})
    key = DAS_X
    manifests = {key: ["f0"]}
    c0, c1 = ho.chunk_id(file_uuid(0), 0, 10), ho.chunk_id(file_uuid(0), 10, 20)
    record = file_record(0, runs_sumw=2.0)
    ledger = ho.remote(reg, reg["samples"][key]["ledger"])
    sumw = ho.remote(reg, reg["samples"][key]["sumw"])
    if case == "a":
        manifests[key] = ["f0", "f9"]
    if case == "i":
        manifests = {}
    if case == "j":
        manifests[key] = []
    if case == "k":
        sumw.fill_many([ho.sumw_fill("f7", "-", "events", 10)])
    if case == "l":
        ledger.fill_many([ho.marker_fill("f0", file_uuid(5), "file", "-")])
    if case == "g":
        ledger.fill_many([ho.marker_fill("f8", file_uuid(0), "file", "-")])
        manifests[key] = ["f0", "f8"]
        sumw.fill_many([ho.sumw_fill("f8", "-", "events", 20)])
    if case == "f":
        ledger.fill_many([ho.marker_fill("f6", file_uuid(6), "file", "-"),
                          ho.marker_fill("-", ho.chunk_id(file_uuid(6), 0, 10), "started", "nominal")])
        manifests[key] = ["f0", "f6"]
    ho.begin_job(reg, key, {"f0": [c0, c1]}, {"f0": record})
    run_chunk(reg, key, meta(reg, key, file_uuid(0), 0, 10), sumw=1.0)
    md1 = meta(reg, key, file_uuid(0), 10, 20)
    if case == "b":
        return reg, manifests
    if case == "c":
        ho.begin_chunk(reg, key, md1, ["nominal"])
        send_pass(reg, key, md1, end=False)
        return reg, manifests
    if case == "d":
        ho.begin_chunk(reg, key, md1, ["nominal", "up"])
        send_pass(reg, key, md1)
        ledger.fill_many([ho.marker_fill("-", c1, "done", "-")])
        sumw.fill_many([ho.sumw_fill("-", c1, "entries", 10), ho.sumw_fill("-", c1, "genweight", 1.0)])
        return reg, manifests
    run_chunk(reg, key, md1, sumw=1.0)
    if case == "e":
        ledger.fill_many([ho.marker_fill("-", c1, "poisoned", "nominal")])
    if case == "h":
        extra = ho.chunk_id(file_uuid(0), 20, 30)
        ledger.fill_many([ho.marker_fill("-", extra, "done", "-")])
    if case == "m":
        reg = json.loads(json.dumps(reg))
        reg["samples"][key]["hists"]["extra"] = reg["samples"][key]["hists"]["met_pt"]
    return reg, manifests


REFUSALS = {"a": "manifest file not recorded", "b": "expected, not started", "c": "started, not done",
            "d": "done, missing passes", "e": "poisoned", "f": "sumw missing for a file",
            "g": "fileuuid shared by several files", "h": "done, not expected", "i": "no manifest",
            "j": "recorded file not in manifest", "k": "sumw for an unrecorded file", "l": "file with several fileuuids",
            "m": "hist keys differ from the booking"}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_refusals(server, synthetic, tmp_path, capsys, case):
    reg, manifests = refusal_case(server, case)
    out = tmp_path / "D"
    assert ho.collect(reg, manifests, str(out)) == 1
    assert not out.exists() or os.listdir(out) == []
    printed = capsys.readouterr().out
    assert REFUSALS[case] in printed
    if case == "d":
        assert "'up'" in printed


def test_straggler_and_override(server, synthetic, tmp_path, monkeypatch):
    for allow in (False, True):
        reg = synthetic_registry(server, samples={DAS_X: "mc"})
        manifest, _ = production(reg, DAS_X, n_files=1)
        real_rebuild = ho.rebuild
        state = {"n": 0}

        def straggler(template_hist, snapshot, variance_known):
            state["n"] += 1
            if state["n"] == 1:
                ho.begin_chunk(reg, DAS_X, meta(reg, DAS_X, file_uuid(0), 20, 30), ["nominal"])
            return real_rebuild(template_hist, snapshot, variance_known)
        monkeypatch.setattr(ho, "rebuild", straggler)
        out = tmp_path / f"N{allow}"
        assert ho.collect(reg, {DAS_X: manifest}, str(out), allow_incomplete=allow) == 1
        assert os.listdir(out) == []
        monkeypatch.setattr(ho, "rebuild", real_rebuild)
    reg, manifests = refusal_case(server, "c")
    out = tmp_path / "O"
    assert ho.collect(reg, manifests, str(out), allow_incomplete=True) == 3
    names = sorted(os.listdir(out))
    assert final_name("MC-backgrounds", ".INCOMPLETE") in names
    report = json.loads((out / final_name("MC-backgrounds", ".INCOMPLETE").replace(".pkl.gz", ".json")).read_text())
    assert report["warning"].startswith("not for physics results")
    assert ho.chunk_id(file_uuid(0), 10, 20) in json.dumps(report["problems"])
    assert all("INCOMPLETE" in name or name == "collect.json" for name in names)


def test_force_replaces_incomplete_set(server, synthetic, tmp_path, monkeypatch):
    reg, manifests = refusal_case(server, "c")
    out = tmp_path / "D"
    assert ho.collect(reg, manifests, str(out), allow_incomplete=True) == 3
    before = {name: sha(out / name) for name in os.listdir(out)}
    reg_c, manifests_c = refusal_case(server, "c")
    assert ho.collect(reg_c, manifests_c, str(out), force=True) == 1
    assert {name: sha(out / name) for name in os.listdir(out)} == before
    md1 = meta(reg, DAS_X, file_uuid(0), 10, 20)
    ho.end_pass(pass_state(ho.pass_context(reg, DAS_X, md1, None, False)))
    ho.end_chunk(reg, DAS_X, md1, 10, sumw=1.0)
    events = []
    real_replace, real_remove = os.replace, os.remove
    monkeypatch.setattr(ho.os, "replace", lambda a, b: (events.append(("replace", os.path.basename(b))),
                                                        real_replace(a, b)))
    monkeypatch.setattr(ho.os, "remove", lambda a: (events.append(("remove", os.path.basename(a))), real_remove(a)))
    assert ho.collect(reg, manifests, str(out), force=True) == 0
    monkeypatch.setattr(ho.os, "replace", real_replace)
    monkeypatch.setattr(ho.os, "remove", real_remove)
    assert events[0] == ("remove", "collect.json") and events[-1] == ("replace", "collect.json")
    assert not any("INCOMPLETE" in name for name in os.listdir(out))
    assert json.loads((out / "collect.json").read_text())["class"] == "complete"


def test_variance_known(server, synthetic, tmp_path):
    reg = synthetic_registry(server, samples={DAS_X: "mc", "/X/RunY-v1/NANOAODSIM": "mc"})
    manifests = {}
    for i, key in enumerate(reg["samples"]):
        chunk = ho.chunk_id(file_uuid(i), 0, 10)
        ho.begin_job(reg, key, {f"f{i}": [chunk]}, {f"f{i}": file_record(i, events=10, runs_sumw=1.0)})
        run_chunk(reg, key, meta(reg, key, file_uuid(i), 0, 10), sumw=1.0)
        manifests[key] = [f"f{i}"]
    assert ho.collect(reg, manifests, str(tmp_path / "F")) == 0
    assert load(tmp_path / "F" / MC_FILE)["X"]["hist"]["met_pt"]._variance_known is False
    assert ho.collect(reg, manifests, str(tmp_path / "P"), per_entry=True) == 0
    for name in ("X__RunX-v1__NANOAODSIM_histograms.pkl.gz", "X__RunY-v1__NANOAODSIM_histograms.pkl.gz"):
        assert load(tmp_path / "P" / name)["X"]["hist"]["met_pt"]._variance_known is True
    single = {DAS_X: reg["samples"][DAS_X]}
    assert ho.collect(dict(reg, samples=single), {DAS_X: manifests[DAS_X]}, str(tmp_path / "S")) == 0
    assert load(tmp_path / "S" / MC_FILE)["X"]["hist"]["met_pt"]._variance_known is True
    reg4 = synthetic_registry(server, samples={DAS_X: "mc"})
    manifest, _ = production(reg4, DAS_X)
    assert ho.collect(reg4, {DAS_X: manifest}, str(tmp_path / "Q")) == 0
    assert load(tmp_path / "Q" / MC_FILE)["X"]["hist"]["met_pt"]._variance_known is False


def test_two_run_periods_and_duplicate_file(server, synthetic, tmp_path):
    reg = synthetic_registry(server, samples={DAS_C: "data", DAS_D: "data"})
    assert {e["dataset"] for e in reg["samples"].values()} == {"Muon0"}
    m_c, _ = production(reg, DAS_C, n_files=1, base=0)
    m_d, _ = production(reg, DAS_D, n_files=1, base=1)
    assert ho.begin_job(reg, DAS_C, {"f0": [ho.chunk_id(file_uuid(0), 0, 10), ho.chunk_id(file_uuid(0), 10, 20)]},
                        {"f0": file_record(0, data=True)}) == [ho.DUPLICATE, ho.DUPLICATE]
    manifests = {DAS_C: m_c, DAS_D: m_d}
    assert ho.collect(reg, manifests, str(tmp_path / "P"), per_entry=True) == 0
    c = load(tmp_path / "P" / "Muon0__Run2024C-v1__NANOAOD_histograms.pkl.gz")
    d = load(tmp_path / "P" / "Muon0__Run2024D-v1__NANOAOD_histograms.pkl.gz")
    assert list(c) == ["Muon0"] and list(d) == ["Muon0"] and c["Muon0"]["sumw"] == -1.0
    assert ho.collect(reg, manifests, str(tmp_path / "F")) == 0
    final = load(tmp_path / "F" / final_name("Data"))
    assert list(final) == ["Muon0"] and final["Muon0"]["sumw"] == -2.0
    merged = wzmod.merge_histogram_sets([{"m": c["Muon0"]["hist"]}, {"m": d["Muon0"]["hist"]}])["m"]["met_pt"]
    merged._variance_known = False
    assert_same_hist(final["Muon0"]["hist"]["met_pt"], merged)
    local = cells(local_fill())
    got = cells(c["Muon0"]["hist"]["met_pt"])
    assert all(np.allclose(got[labels]["value"], 2 * local[labels]["value"], rtol=1e-12) for labels in local)


def test_per_chunk_sumw(server, synthetic):
    def one(record_change=None, entries=10, data=False):
        reg = synthetic_registry(server, samples={DAS_C: "data"} if data else {DAS_S: "mc"})
        key = DAS_C if data else DAS_S
        c1, c2 = ho.chunk_id(file_uuid(0), 0, 10), ho.chunk_id(file_uuid(0), 10, 20)
        record = dict(file_record(0, runs_sumw=3.5, data=data), **(record_change or {}))
        ho.begin_job(reg, key, {"f1": [c1, c2]}, {"f1": record})
        for (start, stop), sumw in (((0, 10), 1.25), ((10, 20), 2.25)):
            md = meta(reg, key, file_uuid(0), start, stop)
            ho.begin_chunk(reg, key, md, ["nominal"])
            send_pass(reg, key, md)
            ho.end_chunk(reg, key, md, entries if start == 0 else 10, sumw=None if data else sumw)
        problems, sumw_total, contributions, _, _ = ho.audit(reg, key, ["f1"])
        return {category for category, _, _ in problems}, sumw_total
    assert one() == (set(), 3.5)
    assert one(entries=9)[0] == {"entries differ from the chunk range"}
    assert one({"runs_count": 21})[0] == {"Events and Runs disagree"}
    assert one({"runs_sumw": 3.5 * (1 + 2e-6)})[0] == {"Events and Runs disagree"}
    assert one({"runs_sumw": 3.5 * (1 + 5e-7)})[0] == set()
    assert one({"events": 30, "runs_count": 30})[0] == set()
    assert one(data=True) == (set(), -1.0)
    reg = synthetic_registry(server, samples={DAS_C: "data"})
    md = open_chunk(reg, DAS_C)
    send_pass(reg, DAS_C, md)
    with pytest.raises(RuntimeError):
        ho.end_chunk(reg, DAS_C, md, 10, sumw=1.0)
    reg = synthetic_registry(server)
    md = open_chunk(reg, DAS_S)
    send_pass(reg, DAS_S, md)
    with pytest.raises(RuntimeError):
        ho.end_chunk(reg, DAS_S, md, 10)


def test_output_classes(server, synthetic, tmp_path, capsys):
    reg = synthetic_registry(server, samples={DAS_X: "mc"})
    c1, c2 = ho.chunk_id(file_uuid(0), 0, 10), ho.chunk_id(file_uuid(0), 10, 20)
    ho.begin_job(reg, DAS_X, {"f0": [c1, c2]}, {"f0": file_record(0, runs_sumw=3.5)})
    run_chunk(reg, DAS_X, meta(reg, DAS_X, file_uuid(0), 0, 10), sumw=1.25)
    assert ho.collect(reg, {DAS_X: ["f0"]}, str(tmp_path / "R")) == 1
    assert "expected, not started [missing]" in capsys.readouterr().out
    assert ho.collect(reg, {DAS_X: ["f0"]}, str(tmp_path / "R"), allow_incomplete=True) == 3
    names = os.listdir(tmp_path / "R")
    assert final_name("MC-backgrounds", ".PARTIAL") in names and not any("INCOMPLETE" in n for n in names)
    assert final_name("MC-backgrounds", ".PARTIAL").replace(".pkl.gz", ".json") in names
    record = load(tmp_path / "R" / final_name("MC-backgrounds", ".PARTIAL"))["X"]
    assert record["sumw"] == 1.25 and record["production"]["class"] == "partial"
    local = cells(local_fill())
    got = cells(record["hist"]["met_pt"])
    assert all(np.array_equal(got[labels]["value"], local[labels]["value"]) for labels in local)
    ho.begin_chunk(reg, DAS_X, meta(reg, DAS_X, file_uuid(0), 10, 20), ["nominal"])
    assert ho.collect(reg, {DAS_X: ["f0"]}, str(tmp_path / "S"), allow_incomplete=True) == 3
    report = json.loads((tmp_path / "S" / final_name("MC-backgrounds", ".INCOMPLETE").replace(".pkl.gz", ".json"))
                        .read_text())
    assert report["warning"] == ho.INCOMPLETE_WARNING
    reg_t, manifests = refusal_case(server, "a")
    assert problems_of(reg_t, DAS_X, manifests[DAS_X]) == {("manifest file not recorded", "missing"): ["f9"]}
    reg_t, manifests = refusal_case(server, "j")
    assert problems_of(reg_t, DAS_X, manifests[DAS_X]) == {("recorded file not in manifest", "inconsistent"): ["f0"]}


def test_truncation(server, synthetic, tmp_path):
    reg = synthetic_registry(server, samples={DAS_X: "mc"}, max_chunks=1)
    c1 = ho.chunk_id(file_uuid(0), 0, 10)
    with pytest.raises(ValueError):
        ho.begin_job(reg, DAS_X, {"f1": [c1]}, {"f1": file_record(0, runs_sumw=1.0)})
    ho.begin_job(reg, DAS_X, {"f1": [c1]}, {"f1": file_record(0, runs_sumw=1.0)}, truncated={"f1": (1, 2)})
    run_chunk(reg, DAS_X, meta(reg, DAS_X, file_uuid(0), 0, 10), sumw=0.5)
    out = tmp_path / "U"
    assert ho.collect(reg, {DAS_X: ["f1"]}, str(out)) == 0
    name = final_name("MC-backgrounds", ".TRUNCATED")
    assert sorted(n for n in os.listdir(out) if "MC-backgrounds" in n) == [name]
    summary = json.loads((out / "collect.json").read_text())
    assert summary["truncated"] is True and summary["max_chunks"] == 1 and summary["class"] == "complete"
    assert load(out / name)["X"]["production"] == {"truncated": True, "max_chunks": 1, "class": "complete",
                                                   "chunks_done": 1, "chunks_expected": 1, "entries": [DAS_X],
                                                   "files": {"f1": {"kept": 1, "total": 2}}}
    plain = synthetic_registry(server, samples={DAS_X: "mc"})
    with pytest.raises(ValueError):
        ho.begin_job(plain, DAS_X, {"f1": [c1]}, {"f1": file_record(0, runs_sumw=1.0)}, truncated={"f1": (1, 2)})
    bad = synthetic_registry(server, samples={DAS_X: "mc"}, max_chunks=1)
    ho.begin_job(bad, DAS_X, {"f1": [c1]}, {"f1": file_record(0, runs_sumw=1.0)}, truncated={"f1": (1, 2)})
    ho.remote(bad, bad["samples"][DAS_X]["ledger"]).fill_many([ho.marker_fill("f1", file_uuid(0), "truncated", "2/2")])
    assert ("truncation differs from the registry", "inconsistent") in problems_of(bad, DAS_X, ["f1"])
    a, b = uuid.UUID(int=11).bytes, uuid.UUID(int=12).bytes
    items = [WorkItem("S", "A", "Events", 0, 10, a), WorkItem("S", "A", "Events", 10, 20, a),
             WorkItem("S", "B", "Events", 0, 10, b)]
    mapping = {"A": "A", "B": "B"}
    assert ho.truncation(items, items[:1], mapping) == {"A": (1, 2), "B": (0, 1)}
    with pytest.raises(ValueError):
        ho.truncation(items, items[1:2], mapping)
    partial = synthetic_registry(server, samples={DAS_X: "mc"}, max_chunks=2)
    chunks = [c1, ho.chunk_id(file_uuid(0), 10, 20)]
    ho.begin_job(partial, DAS_X, {"f1": chunks}, {"f1": file_record(0, runs_sumw=1.0)}, truncated={"f1": (2, 3)})
    run_chunk(partial, DAS_X, meta(partial, DAS_X, file_uuid(0), 0, 10), sumw=0.5)
    assert ho.collect(partial, {DAS_X: ["f1"]}, str(tmp_path / "X"), allow_incomplete=True) == 3
    assert final_name("MC-backgrounds", ".TRUNCATED.PARTIAL") in os.listdir(tmp_path / "X")


def test_pol_keys_do_not_mix(server, synthetic, tmp_path):
    nominal = synthetic_registry(server, samples={DAS_X: "mc"})
    pol = synthetic_registry(server, samples={DAS_X: "mc"}, pol=True)
    assert list(pol["samples"]) == [DAS_X + "#pol"]
    m1, _ = production(nominal, DAS_X, n_files=1)
    m2, _ = production(pol, DAS_X + "#pol", n_files=1, sumws=(3.0, 4.0))
    out = tmp_path / "D"
    assert ho.collect(nominal, {DAS_X: m1}, str(out)) == 0
    assert ho.collect(pol, {DAS_X + "#pol": m2}, str(out), force=True) == 0
    assert load(out / MC_FILE)["X"]["sumw"] == 3.5
    assert load(out / final_name("MC-backgrounds", suffix="__pol"))["X"]["sumw"] == 7.0


def test_interrupted_begin_job(server, synthetic, tmp_path, monkeypatch):
    reg = synthetic_registry(server, samples={DAS_X: "mc"})
    sumw_id = reg["samples"][DAS_X]["sumw"]

    def kill_between(self, fills, kwargs, n):
        if self.hist_id == sumw_id:
            raise FakeRpcError(grpc.StatusCode.UNAVAILABLE)
        return REAL_FILL_MANY(self, fills, **kwargs)
    wrap_fill_many(monkeypatch, kill_between)
    chunks = [ho.chunk_id(file_uuid(0), 0, 10), ho.chunk_id(file_uuid(0), 10, 20)]
    with pytest.raises(FakeRpcError):
        ho.begin_job(reg, DAS_X, {"f0": chunks}, {"f0": file_record(0, runs_sumw=2.0)})
    monkeypatch.setattr(histserv.RemoteHist, "fill_many", REAL_FILL_MANY)
    assert problems_of(reg, DAS_X, ["f0"]) == {
        ("begin_job interrupted: file recorded without sumw", "missing"): ["f0"],
        ("expected, not started", "missing"): sorted(chunks)}
    assert ho.collect(reg, {DAS_X: ["f0"]}, str(tmp_path / "P"), allow_incomplete=True) == 3
    assert ho.begin_job(reg, DAS_X, {"f0": chunks}, {"f0": file_record(0, runs_sumw=2.0)}) == [ho.DUPLICATE, ho.SENT]
    for start in (0, 10):
        run_chunk(reg, DAS_X, meta(reg, DAS_X, file_uuid(0), start, start + 10), sumw=1.0)
    assert ho.collect(reg, {DAS_X: ["f0"]}, str(tmp_path / "C")) == 0
    reg = synthetic_registry(server, samples={DAS_X: "mc"})
    ledger = ho.remote(reg, reg["samples"][DAS_X]["ledger"])
    ledger.fill_many([ho.marker_fill("f0", file_uuid(0), "file", "-"), ho.marker_fill("f0", chunks[0], "expected", "-"),
                      ho.marker_fill("-", chunks[0], "started", "nominal")])
    assert ("sumw missing for a file", "inconsistent") in problems_of(reg, DAS_X, ["f0"])


def test_final_files(server, synthetic, tmp_path):
    samples = {DAS_C: "data", DAS_D: "data", "/W/RunX-v1/NANOAODSIM": "signal", DAS_X: "mc",
               "/X/RunX_ext1-v1/NANOAODSIM": "mc", "/E/RunX-v1/NANOAODSIM": "eft"}
    reg = synthetic_registry(server, samples=samples)
    manifests = {}
    for i, key in enumerate(reg["samples"]):
        manifests[key], _ = production(reg, key, n_files=1, base=i)
    for filename in ("latest", "v2"):
        out = tmp_path / filename
        assert ho.collect(reg, manifests, str(out), filename=filename) == 0
        labels = {"Data": "Muon0", "MC-signal": "W", "MC-backgrounds": "X", "EFT": "E"}
        assert sorted(os.listdir(out)) == sorted([final_name(k, filename=filename) for k in labels] + ["collect.json"])
        for label, dataset in labels.items():
            content = load(out / final_name(label, filename=filename))
            assert list(content) == [dataset]
            assert pk.KIND_FILE_LABELS[content[dataset]["kind"]] == label
        assert load(out / final_name("MC-backgrounds", filename=filename))["X"]["hist"]["met_pt"]._variance_known is False
        summary = json.loads((out / "collect.json").read_text())
        assert summary["files"][final_name("MC-backgrounds", filename=filename)]["datasets"] == {
            "X": [DAS_X, "/X/RunX_ext1-v1/NANOAODSIM"]}
    with pytest.raises(ValueError):
        ho.collect(reg, manifests, str(tmp_path / "bad"), filename="x.y")
    path = tmp_path / "reg.json"
    ho.write_registry(str(path), reg)
    assert ho.main(["collect", "--registry", str(path), "--outdir", str(tmp_path / "bad"), "-f", "x.y"]) == 2


def test_partial_datasets(server, synthetic, tmp_path):
    reg = synthetic_registry(server, samples={DAS_X: "mc", "/X/RunX_ext1-v1/NANOAODSIM": "mc", DAS_C: "data"})
    manifests = {}
    manifests[DAS_X], _ = production(reg, DAS_X, n_files=1, base=0)
    manifests["/X/RunX_ext1-v1/NANOAODSIM"], _ = production(reg, "/X/RunX_ext1-v1/NANOAODSIM", n_files=1, base=1,
                                                            run=False)
    manifests[DAS_C], _ = production(reg, DAS_C, n_files=1, base=2, run=False)
    out = tmp_path / "D"
    assert ho.collect(reg, manifests, str(out), allow_incomplete=True) == 3
    assert load(out / final_name("MC-backgrounds", ".PARTIAL"))["X"]["sumw"] == 3.5
    assert final_name("Data", ".PARTIAL") not in os.listdir(out)
    report = json.loads((out / final_name("Data", ".PARTIAL").replace(".pkl.gz", ".json")).read_text())
    assert report["datasets_without_done_chunks"] == ["Muon0"]


def test_write_done_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fileuuid = uuid.UUID(int=3).bytes
    items = [WorkItem("S", "x.root", "Events", 10, 20, fileuuid), WorkItem("S", "x.root", "Events", 0, 10, fileuuid)]
    path = ho.write_done_file(7, {"tag": "t"}, DAS_S, items)
    assert path == "histserv_done_7.json" and os.stat(path).st_mode & 0o777 == 0o644
    content = json.loads(open(path).read())
    assert content == {"format": 1, "tag": "t", "key": DAS_S,
                       "chunks": sorted(ho.chunk_id(str(uuid.UUID(int=3)), a, b) for a, b in ((0, 10), (10, 20)))}
    assert os.listdir(tmp_path) == ["histserv_done_7.json"]


def test_cli_help_has_no_pickle_input():
    text = textwrap.dedent(run_cli("collect", "--help").stdout)
    assert "--registry" in text and "pickle" not in text.replace("Job pickles are never read", "")


DD_RUN2 = os.path.join(ROOT, "src", "qawa", "data", "dd", "Run2", "WZ_inclusive_data_driven_Run2.json")
DD_RUN3 = os.path.join(ROOT, "src", "qawa", "data", "dd", "Run3", "WZ_inclusive_data_driven_Run3.json")


def test_dd_register_and_collect(server, synthetic, tmp_path):
    label = pk.dd_label(DD_RUN2)
    data = {DAS_C: "data", DAS_D: "data"}
    first = synthetic_registry(server, samples=data)
    manifests = {key: production(first, key, n_files=1, base=i)[0] for i, key in enumerate(first["samples"])}
    assert ho.collect(first, manifests, str(tmp_path / "out1")) == 0
    stage1 = load(tmp_path / "out1" / final_name("Data"))
    results = {}
    for name, factors in (("dd", (1.0, 2.0)), ("nominal", (1 + 1e-6, 1.0))):
        reg = ho.register(server, "t", OPTIONS, data, uuid.uuid4().hex, dd_file=DD_RUN2)
        assert sorted(reg["samples"]) == sorted(f"{das}#dd:{label}" for das in data)
        for entry in reg["samples"].values():
            assert entry["variations"]["dd"] == label and entry["dd_sha256"] == pk.file_sha256(DD_RUN2)
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(globals(), "WEIGHT", lambda k, rows, w=WEIGHT, f=factors: w(k, rows) * (f[0] if k == 0 else f[1]))
            manifests = {key: production(reg, key, n_files=1, base=i)[0] for i, key in enumerate(reg["samples"])}
        out = tmp_path / f"out_{name}"
        assert ho.collect(reg, manifests, str(out)) == 0
        assert sorted(os.listdir(out)) == ["collect.json", final_name("Data", suffix=f"__dd-{label}")]
        results[name] = load(out / final_name("Data", suffix=f"__dd-{label}"))
    assert datadriven.check_reprocessed(stage1, results["dd"]) == []
    old, new = cells(stage1["Muon0"]["hist"]["met_pt"]), cells(results["dd"]["Muon0"]["hist"]["met_pt"])
    assert np.array_equal(old[("chA", "nominal")], new[("chA", "nominal")])
    assert not np.array_equal(old[("chA", "up")], new[("chA", "up")])
    problems = datadriven.check_reprocessed(stage1, results["nominal"])
    assert len(problems) == 1 and "Muon0 met_pt: nominal differs" in problems[0]


def test_dd_register_refusals(server, tmp_path):
    gz = tmp_path / "dd.json.gz"
    with open(DD_RUN2, "rb") as f:
        gz.write_bytes(gzip.compress(f.read()))
    for sample, path in ((f"{DAS_X}=mc", DD_RUN2), (f"{DAS_C}=data", str(gz))):
        args = ["register", "--address", server, "--tag", "t", "--era", "2024", "--version", "v15",
                "--sample", sample, "--dd-file", path, "--output", str(tmp_path / "reg.json")]
        assert ho.main(args) == 2
        assert not (tmp_path / "reg.json").exists()


def test_constructor_dd_checks(tmp_path):
    label = pk.dd_label(DD_RUN3)
    key = f"{DAS_C}#dd:{label}"
    reg = hand_registry("data", key=key)
    reg["samples"][key]["variations"]["dd"] = label
    reg["samples"][key]["dd_sha256"] = pk.file_sha256(DD_RUN3)
    other = tmp_path / "other.json"
    other.write_text(open(DD_RUN3).read() + "\n")
    kwargs = dict(era="2024", version="v15", run_period="C")
    for dd_file, message in ((None, "no dd_file is given"), (str(other), "another sha256")):
        with pytest.raises(ValueError, match=message):
            wzmod.wzinclusive_processor(histserv_registry=reg, histserv_sample=key, dd_file=dd_file, **kwargs)
    with pytest.raises(ValueError, match="registered without a dd file"):
        wzmod.wzinclusive_processor(histserv_registry=hand_registry("data", key=DAS_C), histserv_sample=DAS_C,
                                    dd_file=DD_RUN3, **kwargs)
    proc = wzmod.wzinclusive_processor(histserv_registry=reg, histserv_sample=key, dd_file=DD_RUN3, **kwargs)
    assert proc.dd_file == DD_RUN3 and proc.histserv_sample == key


LOCAL_2018 = "/Users/nmangane/vz-main/SMQawa/DoubleMuon_UL2018_NanoAODv9.root"


def lumi_fills(monkeypatch, reg, key):
    counts = []

    def recording(self, fills, **kwargs):
        if self.hist_id == reg["samples"][key]["lumi"]:
            counts.append(len(fills))
        return REAL_FILL_MANY(self, fills, **kwargs)
    monkeypatch.setattr(histserv.RemoteHist, "fill_many", recording)
    return counts


def test_lumi_record_done_chunks(server, synthetic, tmp_path, capsys):
    reg = synthetic_registry(server, samples={DAS_C: "data"})
    chunks = [{(379000, 1), (379000, 2)}, {(379000, 2), (379000, 3)}, {(379001, 1)}, set()]
    manifest, _ = production(reg, DAS_C, lumis=chunks)
    straggler = ho.chunk_id(file_uuid(9), 0, 10)
    ho.send(reg, DAS_C, reg["samples"][DAS_C]["lumi"], [ho.marker_fill(straggler, "379002", "5", names=("chunk", "run", "lumi"))],
            ["t", DAS_C, "lumi", straggler], ("-", straggler, "done"))
    assert ho.collect(reg, {DAS_C: manifest}, str(tmp_path / "F")) == 0
    record = load(tmp_path / "F" / final_name("Data"))["Muon0"]["lumi"]
    assert record.dtype == np.uint32 and record.tolist() == [[379000, 1], [379000, 2], [379000, 3], [379001, 1]]
    out = capsys.readouterr().out
    assert "lumi records of 1 chunks that are not done" in out and straggler in out


def test_lumi_record_absent_or_empty(server, synthetic, tmp_path, monkeypatch, capsys):
    reg = synthetic_registry(server, samples={DAS_C: "data"})
    manifest, _ = production(reg, DAS_C, n_files=1)
    assert ho.collect(reg, {DAS_C: manifest}, str(tmp_path / "N")) == 0
    assert "lumi" not in load(tmp_path / "N" / final_name("Data"))["Muon0"]
    assert "2 done chunks without a lumi record" in capsys.readouterr().out
    reg = synthetic_registry(server, samples={DAS_C: "data"})
    counts = lumi_fills(monkeypatch, reg, DAS_C)
    manifest, _ = production(reg, DAS_C, n_files=1, lumis=[set(), set()])
    assert counts == [1, 1]
    assert ho.collect(reg, {DAS_C: manifest}, str(tmp_path / "E")) == 0
    record = load(tmp_path / "E" / final_name("Data"))["Muon0"]["lumi"]
    expected = lumi_record.as_array(set())
    assert record.shape == (0, 2) and record.dtype == expected.dtype
    reg = synthetic_registry(server, samples={DAS_C: "data"})
    manifest, _ = production(reg, DAS_C, n_files=1, lumis=[{(379000, 4)}, set()])
    assert ho.collect(reg, {DAS_C: manifest}, str(tmp_path / "M")) == 0
    assert load(tmp_path / "M" / final_name("Data"))["Muon0"]["lumi"].tolist() == [[379000, 4]]
    assert "without a lumi record" not in capsys.readouterr().out


def test_lumi_record_mc_and_entries(server, synthetic, tmp_path):
    reg = synthetic_registry(server, samples={DAS_S: "mc"})
    assert "lumi" not in reg["samples"][DAS_S]
    md = meta(reg, DAS_S, file_uuid(0), 0, 10)
    ho.begin_job(reg, DAS_S, {"f0": [ho.chunk_id(file_uuid(0), 0, 10)]}, {"f0": file_record(0, events=10, runs_sumw=1.0)})
    ho.begin_chunk(reg, DAS_S, md, ["nominal"])
    send_pass(reg, DAS_S, md)
    for lumis in (set(), {(379000, 1)}):
        with pytest.raises(RuntimeError, match="lumis are recorded for data only"):
            ho.end_chunk(reg, DAS_S, md, 10, sumw=1.0, lumis=lumis)
    reg = synthetic_registry(server, samples={DAS_C: "data", DAS_D: "data"})
    m_c, _ = production(reg, DAS_C, n_files=1, base=0, lumis=[{(379000, 1)}, {(379000, 2)}])
    m_d, _ = production(reg, DAS_D, n_files=1, base=1, lumis=[{(380000, 1)}, {(379000, 2)}])
    assert ho.collect(reg, {DAS_C: m_c, DAS_D: m_d}, str(tmp_path / "F")) == 0
    record = load(tmp_path / "F" / final_name("Data"))["Muon0"]["lumi"]
    assert record.tolist() == [[379000, 1], [379000, 2], [380000, 1]]


@pytest.mark.skipif(not os.path.exists(LOCAL_2018), reason="local 2018 data file not available")
def test_process_passes_lumis(server, tmp_path, monkeypatch):
    import uproot
    from coffea import processor
    from coffea.lumi_tools import LumiMask
    from coffea.nanoevents import NanoAODSchema
    das = "/DoubleMuon/Run2018C-E2ELocal-v1/NANOAOD"
    options = dict(OPTIONS, era="2018", version="v9")
    reg = ho.register(server, "t", options, {das: "data"}, uuid.uuid4().hex)
    fileset = {"DoubleMuon": {"files": {LOCAL_2018: "Events"}, "metadata": {"is_data": True, "is_signal": False,
                                                                              "is_flashsim": False}}}
    runner = processor.Runner(processor.IterativeExecutor(), chunksize=50000, schema=NanoAODSchema, format="root")
    items = list(runner.preprocess(fileset, treename="Events"))
    with uproot.open(LOCAL_2018) as f:
        events = f["Events"].num_entries
        arrays = f["Events"].arrays(["run", "luminosityBlock"], library="np")
    record = {"fileuuid": str(uuid.UUID(bytes=items[0].fileuuid)), "events": events, "runs_sumw": None,
              "runs_count": None}
    ho.begin_job(reg, das, ho.expected_chunks(items, {LOCAL_2018: LOCAL_2018}), {LOCAL_2018: record})
    proc = wzmod.wzinclusive_processor(era="2018", version="v9", run_period="C", histserv_registry=reg,
                                       histserv_sample=das)
    mask = LumiMask(proc._lumimask_path)(arrays["run"], arrays["luminosityBlock"])
    independent = {(int(r), int(lumi)) for r, lumi in zip(arrays["run"][mask], arrays["luminosityBlock"][mask])}
    calls = []
    real_end_chunk = ho.end_chunk

    def recording(*args, **kwargs):
        calls.append(kwargs.get("lumis"))
        return real_end_chunk(*args, **kwargs)
    monkeypatch.setattr(ho, "end_chunk", recording)
    assert runner(items, processor_instance=proc) == {"DoubleMuon": {}}
    assert calls and all(lumis is not None for lumis in calls)
    assert set().union(*calls) == independent
    assert ho.collect(reg, {das: [LOCAL_2018]}, str(tmp_path / "F"), per_entry=True) == 0
    content = load(tmp_path / "F" / (pk.sample_file_stem(das) + "_histograms.pkl.gz"))["DoubleMuon"]
    assert content["lumi"].tolist() == sorted(map(list, independent))
