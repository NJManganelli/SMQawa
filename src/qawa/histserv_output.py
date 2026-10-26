import argparse
import gzip
import json
import os
import pickle
import re
import secrets
import sys
import time
import uuid

import grpc
import hist
import numpy
from histserv import Client

import qawa.datadriven as datadriven
import qawa.lumi_record as lumi_record
import qawa.process.wztau2lnu_inclusive as wzmod
import qawa.production_keys as production_keys

KINDS = ("data", "mc", "signal", "eft")
OPTION_NAMES = ("era", "version", "pol", "split_by_charge", "split_FV")
TIMEOUT = 600
REGISTRY_FORMAT = 2
COLLECT_FORMAT = 3
DONE_FILE_FORMAT = 1
SENT = "sent"
EMPTY = "empty"
DUPLICATE = "duplicate"
CLASSES = ("complete", "partial", "incomplete")
MISSING_CATEGORIES = ("manifest file not recorded", "expected, not started",
                      "begin_job interrupted: file recorded without sumw")
CHANGED_CATEGORY = "changed during collect"
POISON_EXIT_CODE = 87
NON_POISONING_CODES = frozenset({grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.CANCELLED,
                                 grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.NOT_FOUND})
POISON_WRITE_DELAYS = (0, 1, 4, 16)
POISON_WRITE_TIMEOUTS = (10, 10, 60, 60)
RESERVED_PASS_NAMES = ("begin_job", "started", "done")
POISON_FILE_ENV = "QAWA_HISTSERV_POISON_FILE"
POISON_FILE_NAME = "histserv_poisoned.jsonl"
DONE_FILE_TEMPLATE = "histserv_done_{}.json"
COLLECT_FILE = "collect.json"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
STANDARD_KINDS = ("data", "mc", "signal")
EMPTY_LUMI_LABEL = "-"
MAX_LISTED_NAMES = 20
INCOMPLETE_WARNING = ("not for physics results: the hists hold partial contributions of chunks that are not done or were "
                      "poisoned, which the sumw does not count")
PARTIAL_STATEMENT = ("whole files or chunks were never processed: the hists and sumw cover exactly the done chunks, so MC is "
                     "normalized to what this output holds and data to its lumi record; a lumi section split across a done "
                     "and a missing chunk is listed whole in the lumi record")

CLIENTS = {}
REMOTES = {}
ATTEMPTS = {}
POISONED = {}
PROCESSORS = {}


class HistservPoisoned(RuntimeError):
    pass


def is_poison(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, HistservPoisoned) or "HistservPoisoned(" in str(error):
            return True
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return False


def first_poison(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, HistservPoisoned):
            return error
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return None


def write_poison_marker(record):
    path = os.environ.get(POISON_FILE_ENV)
    if not path:
        return None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(record) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as error:
        return str(error)
    return None


def read_poison_marker():
    path = os.environ.get(POISON_FILE_ENV)
    if not path or not os.path.isfile(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                records.append({"state": "unreadable", "line": line.rstrip("\n")})
    return records


def poison_reason(error):
    records = read_poison_marker()
    if records:
        groups = {}
        for record in records:
            groups.setdefault((record.get("pid"), json.dumps(record.get("key"))), []).append(record)
        lines = []
        for group in groups.values():
            final = [record for record in group if record.get("message")]
            if final:
                lines.append(final[-1]["message"])
            elif group[0].get("state") == "unreadable":
                lines.append(f"unreadable poison marker line {group[0]['line']!r} (the poison write did not finish)")
            else:
                pending = group[0]
                lines.append(f"C4 poison pending for {pending.get('sample')} {pending.get('key')} "
                             f"({pending.get('code')}: {pending.get('details')}) (the poison write did not finish)")
        return "\n".join(lines)
    if error is not None and is_poison(error):
        poisoned = first_poison(error)
        return str(poisoned) if poisoned is not None else str(error)
    return None


def nondeterministic_reason(proc, kind):
    if kind == "data" or "muon_scalesmearing" in proc.clibhandler.keys():
        return None
    return (f"C4 needs deterministic chunk processing: {proc._era} MC uses the Rochester smearing, which draws unseeded "
            f"random numbers (roccor.py:31); see user decision N15")


def client(address):
    pid_address = (os.getpid(), address)
    if pid_address not in CLIENTS:
        CLIENTS[pid_address] = Client(address)
    return CLIENTS[pid_address]


def remote(registry, hist_id):
    key = (os.getpid(), registry["address"], registry["token"], hist_id)
    if key not in REMOTES:
        REMOTES[key] = client(registry["address"]).connect(hist_id, token=registry["token"], timeout=TIMEOUT)
    return REMOTES[key]


def make_processor(options):
    cache_key = json.dumps(options, sort_keys=True)
    if cache_key not in PROCESSORS:
        PROCESSORS[cache_key] = wzmod.wzinclusive_processor(**options)
    return PROCESSORS[cache_key]


def templates(proc, kind):
    if proc._pol:
        if kind == "eft":
            raise NotImplementedError("polarization templates with EFT weights are not supported")
        histos = proc._build_histos_with_pol() if kind == "signal" else proc.build_histos_pol()
    else:
        histos = proc.build_histos()
    if kind == "eft":
        for var in wzmod.STANDARD_FILL_VARIABLES:
            histos[var + "_eft"] = wzmod._wh_eft(*list(histos[var].axes)[2:])
    return histos


def entry_axis():
    return hist.axis.Integer(0, 1, underflow=False, overflow=False, name="entry")


def ledger_template():
    return hist.Hist(*(hist.axis.StrCategory([], name=name, growth=True) for name in ("file", "chunk", "state", "pass")),
                     entry_axis(), storage=hist.storage.Weight())


def sumw_template():
    return hist.Hist(*(hist.axis.StrCategory([], name=name, growth=True) for name in ("file", "chunk", "quantity")),
                     entry_axis(), storage=hist.storage.Weight())


def lumi_template():
    return hist.Hist(*(hist.axis.StrCategory([], name=name, growth=True) for name in ("chunk", "run", "lumi")),
                     entry_axis(), storage=hist.storage.Weight())


def chunk_id(fileuuid, entrystart, entrystop):
    if not fileuuid:
        raise ValueError("C4 needs the chunk's fileuuid, which is empty")
    return f"{fileuuid}:{entrystart}:{entrystop}"


def parse_chunk_id(chunk):
    fileuuid, start, stop = chunk.rsplit(":", 2)
    return fileuuid, int(start), int(stop)


def item_file(item, file_of):
    if item.filename not in file_of:
        raise ValueError(f"WorkItem file {item.filename!r} is not one of the job's input files {sorted(file_of)}")
    return file_of[item.filename]


def item_chunk(item):
    return chunk_id(str(uuid.UUID(bytes=item.fileuuid)), item.entrystart, item.entrystop)


def expected_chunks(workitems, file_of):
    expected = {}
    for item in workitems:
        expected.setdefault(item_file(item, file_of), []).append(item_chunk(item))
    return expected


def file_records(all_items, file_of, runs):
    grouped = {}
    for item in all_items:
        grouped.setdefault(item_file(item, file_of), []).append(item)
    records = {}
    for file, items in grouped.items():
        ranges = sorted((item.entrystart, item.entrystop) for item in items)
        events = max(stop for _, stop in ranges)
        if ranges[0][0] != 0 or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError(f"the WorkItems of {file} do not tile [0, {events})")
        runs_sumw, runs_count = runs[file] if runs is not None else (None, None)
        records[file] = {"fileuuid": str(uuid.UUID(bytes=items[0].fileuuid)), "events": events,
                         "runs_sumw": runs_sumw, "runs_count": runs_count}
    return records


def truncation(all_items, items, file_of):
    full = expected_chunks(all_items, file_of)
    kept = expected_chunks(items, file_of)
    unknown = set(kept) - set(full)
    if unknown:
        raise ValueError(f"truncated WorkItems name files that are not in the full list: {sorted(unknown)}")
    result = {}
    for file, chunks in full.items():
        mine = kept.get(file, [])
        if chunks[:len(mine)] != mine:
            raise ValueError(f"the kept chunks of {file} are not a prefix of its chunks")
        result[file] = (len(mine), len(chunks))
    return result


def write_registry(path, registry):
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(registry, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_registry(path):
    with open(path) as f:
        registry = json.load(f)
    if registry.get("format") != REGISTRY_FORMAT:
        raise ValueError(f"{path}: registry format {registry.get('format')!r}, expected {REGISTRY_FORMAT}")
    return registry


def register(address, tag, options, samples, token, max_chunks=None, dd_file=None):
    if not NAME_PATTERN.match(tag):
        raise ValueError(f"tag {tag!r} must match {NAME_PATTERN.pattern}")
    if set(options) != set(OPTION_NAMES):
        raise ValueError(f"options must be exactly {OPTION_NAMES}, got {sorted(options)}")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")
    problems = []
    for das, kind in samples.items():
        try:
            tier = production_keys.das_parts(das)[2]
        except ValueError as error:
            problems.append(str(error))
            continue
        if kind not in KINDS:
            problems.append(f"{das}: unknown kind {kind!r}; known: {KINDS}")
        elif (kind == "data") != (tier == "NANOAOD"):
            problems.append(f"{das}: kind {kind} contradicts the tier {tier}")
        elif dd_file is not None and kind != "data":
            problems.append(f"{das}: --dd-file re-processes data only, got kind {kind}")
    if problems:
        raise ValueError("; ".join(problems))
    kinds_of = {}
    for das, kind in samples.items():
        kinds_of.setdefault(production_keys.primary_dataset(das), set()).add(kind)
    mixed = sorted(dataset for dataset, kinds in kinds_of.items() if len(kinds) > 1)
    if mixed:
        raise ValueError(f"datasets registered with several kinds (they would go to two files): {mixed}")
    dd, dd_sha256 = None, None
    if dd_file is not None:
        datadriven.check_dd_file(dd_file)
        dd, dd_sha256 = production_keys.dd_label(dd_file), production_keys.file_sha256(dd_file)
    proc = make_processor(options)
    reasons = {das: nondeterministic_reason(proc, kind) for das, kind in samples.items()}
    if any(reasons.values()):
        raise ValueError("; ".join(f"{das}: {reason}" for das, reason in reasons.items() if reason))
    entries = {}
    stems = {}
    for das, kind in samples.items():
        variations = {"pol": options["pol"], "split_by_charge": options["split_by_charge"],
                      "split_FV": options["split_FV"], "eft": kind == "eft", "dd": dd}
        key = production_keys.sample_key(das, **variations)
        stem = production_keys.sample_file_stem(key)
        if stem in stems:
            raise ValueError(f"sample keys {stems[stem]} and {key} give one file stem {stem}")
        stems[stem] = key
        entries[key] = (das, kind, variations, templates(proc, kind))
    hist_client = client(address)
    registered = {}
    for key, (das, kind, variations, booked) in entries.items():
        registered[key] = {
            "das": das, "kind": kind, "dataset": production_keys.primary_dataset(das), "variations": variations,
            "dd_sha256": dd_sha256,
            "hists": {name: hist_client.init(h, token=token, timeout=TIMEOUT).hist_id for name, h in booked.items()},
            "ledger": hist_client.init(ledger_template(), token=token, timeout=TIMEOUT).hist_id,
            "sumw": hist_client.init(sumw_template(), token=token, timeout=TIMEOUT).hist_id,
        }
        if kind == "data":
            registered[key]["lumi"] = hist_client.init(lumi_template(), token=token, timeout=TIMEOUT).hist_id
    return {"format": REGISTRY_FORMAT, "address": address, "token": token, "tag": tag, "options": dict(options),
            "max_chunks": max_chunks, "poison_exit_code": POISON_EXIT_CODE, "samples": registered}


def resolve_key(registry, das, pol, split_by_charge, split_FV, dd):
    wanted = {"pol": pol, "split_by_charge": split_by_charge, "split_FV": split_FV, "dd": dd}
    keys = [key for key, entry in registry["samples"].items()
            if entry["das"] == das and all(entry["variations"][name] == value for name, value in wanted.items())]
    if len(keys) != 1:
        known = sorted(key for key, entry in registry["samples"].items() if entry["das"] == das)
        raise ValueError(f"no single registered entry for {das} with {wanted}; registered keys of {das}: {known}")
    return keys[0]


def entry_of(registry, key):
    if key not in registry["samples"]:
        raise ValueError(f"sample key {key!r} is not registered; registered: {sorted(registry['samples'])}")
    return registry["samples"][key]


def marker_fill(*labels, names=("file", "chunk", "state", "pass"), weight=1.0):
    fill = dict(zip(names, labels))
    fill["entry"] = numpy.zeros(1, dtype=numpy.int64)
    fill["weight"] = numpy.array([weight], dtype=numpy.float64)
    return fill


def sumw_fill(file, chunk, quantity, weight):
    return marker_fill(file, chunk, quantity, names=("file", "chunk", "quantity"), weight=weight)


def raise_if_poisoned():
    message = POISONED.get(os.getpid())
    if message is not None:
        raise HistservPoisoned(message)


def send(registry, key, hist_id, fills, unique_id, where):
    raise_if_poisoned()
    if not fills:
        return EMPTY
    try:
        remote(registry, hist_id).fill_many(fills, unique_id=unique_id, timeout=TIMEOUT)
    except grpc.RpcError as error:
        code = error.code()
        if code == grpc.StatusCode.ALREADY_EXISTS:
            return DUPLICATE
        if code in NON_POISONING_CODES:
            raise
        poison(registry, key, where, error)
    return SENT


def error_code_name(error):
    code = error.code() if callable(getattr(error, "code", None)) else None
    return code.name if isinstance(code, grpc.StatusCode) else type(error).__name__


def poison(registry, key, where, error):
    details = ""
    if callable(getattr(error, "details", None)):
        details = (str(error.details() or "").splitlines() or [""])[0][:200]
    code = error_code_name(error)
    pid = os.getpid()
    POISONED[pid] = f"C4 poison pending for {key} {tuple(where)} ({code}: {details})"
    pending = {"pid": pid, "sample": key, "key": list(where), "code": code, "details": details, "state": "pending"}
    marker_errors = [write_poison_marker(pending)]
    fills = [marker_fill(where[0], where[1], "poisoned", where[2])]
    unique_id = [registry["tag"], key, "poisoned", *where, secrets.token_hex(8)]
    ledger_id = entry_of(registry, key)["ledger"]
    recorded = False
    attempts = 0
    for delay, timeout in zip(POISON_WRITE_DELAYS, POISON_WRITE_TIMEOUTS):
        attempts += 1
        time.sleep(delay)
        try:
            remote(registry, ledger_id).fill_many(fills, unique_id=unique_id, timeout=timeout)
        except grpc.RpcError as write_error:
            if write_error.code() == grpc.StatusCode.ALREADY_EXISTS:
                recorded = True
                break
        except Exception:
            pass
        else:
            recorded = True
            break
    if recorded:
        message = (f"C4 poison recorded for {key} {tuple(where)} ({code}: {details}): re-register under a new tag and "
                   f"re-run the production")
    else:
        message = (f"C4 poison NOT recorded on the server for {key} {tuple(where)} ({code}: {details}) after {attempts} "
                   f"attempts: do not retry or release this job; re-register under a new tag and re-run the production")
    final = dict(pending, state="recorded" if recorded else "not recorded", attempts=attempts, message=message)
    marker_errors.append(write_poison_marker(final))
    failed = [text for text in marker_errors if text]
    if failed:
        message += f"; poison marker not written: {failed[0]}"
        final["message"] = message
    POISONED[pid] = message
    raise HistservPoisoned(message) from error


def check_poison(registry, key):
    raise_if_poisoned()
    ledger = remote(registry, entry_of(registry, key)["ledger"])
    poisoned = [chunk_key for chunk_key, _ in ledger[{"state": "poisoned"}].snapshot(timeout=TIMEOUT).items()]
    if poisoned:
        raise HistservPoisoned(f"C4 sample {key} is poisoned at {poisoned}: re-register under a new tag and re-run the "
                               f"production")


def begin_job(registry, key, expected, files, truncated=None):
    raise_if_poisoned()
    entry = entry_of(registry, key)
    if (truncated is not None) != (registry["max_chunks"] is not None):
        raise ValueError(f"begin_job: truncated must be given exactly when the registry has max_chunks "
                         f"({registry['max_chunks']})")
    if set(expected) != set(files) or (truncated is not None and set(truncated) != set(files)):
        raise ValueError(f"begin_job: the files of expected {sorted(expected)}, files {sorted(files)} and truncated "
                         f"{sorted(truncated or files)} differ")
    for file, chunks in expected.items():
        fileuuid = files[file]["fileuuid"]
        wrong = [chunk for chunk in chunks if parse_chunk_id(chunk)[0] != fileuuid]
        if not fileuuid or wrong:
            raise ValueError(f"begin_job: chunks {wrong} of {file} do not carry its fileuuid {fileuuid!r}")
    check_poison(registry, key)
    tag = registry["tag"]
    results = []
    for file, chunks in expected.items():
        record = files[file]
        fileuuid = record["fileuuid"]
        where = (file, fileuuid, "begin_job")
        fills = [marker_fill(file, fileuuid, "file", "-")]
        fills += [marker_fill(file, chunk, "expected", "-") for chunk in chunks]
        if truncated is not None:
            fills.append(marker_fill(file, fileuuid, "truncated", "{}/{}".format(*truncated[file])))
        results.append(send(registry, key, entry["ledger"], fills, [tag, key, "expected", file], where))
        sums = [sumw_fill(file, "-", "events", record["events"])]
        if entry["kind"] == "data":
            sums.append(sumw_fill(file, "-", "data_file", -1.0))
        else:
            sums.append(sumw_fill(file, "-", "runs_sumw", record["runs_sumw"]))
            sums.append(sumw_fill(file, "-", "runs_count", record["runs_count"]))
        results.append(send(registry, key, entry["sumw"], sums, [tag, key, "sumw_file", file], where))
    return results


def metadata_chunk(metadata):
    return chunk_id(metadata.get("fileuuid"), metadata["entrystart"], metadata["entrystop"])


def begin_chunk(registry, key, metadata, pass_names):
    raise_if_poisoned()
    entry = entry_of(registry, key)
    if len(set(pass_names)) != len(pass_names) or set(pass_names) & set(RESERVED_PASS_NAMES):
        raise RuntimeError(f"begin_chunk: pass names {list(pass_names)} have duplicates or one of {RESERVED_PASS_NAMES}")
    chunk = metadata_chunk(metadata)
    ATTEMPTS[(os.getpid(), key, chunk)] = {"passes": list(pass_names), "ended": set()}
    fills = [marker_fill("-", chunk, "started", name) for name in pass_names]
    return send(registry, key, entry["ledger"], fills, [registry["tag"], key, "started", chunk], ("-", chunk, "started"))


def kind_problem(kind, is_data, is_signal, eft):
    if (kind == "data") != bool(is_data):
        return f"kind {kind} but is_data={is_data}"
    if kind == "signal" and not is_signal:
        return "kind signal but is_signal is not set"
    if kind == "mc" and is_signal:
        return "kind mc but is_signal is set"
    if (kind == "eft") != bool(eft):
        return f"kind {kind} but the chunk {'has' if eft else 'has no'} EFT weights"
    return None


def pass_context(registry, key, metadata, shift_name, eft):
    entry = entry_of(registry, key)
    if metadata.get("dataset") != entry["dataset"]:
        raise ValueError(f"chunk dataset {metadata.get('dataset')!r} is not the dataset {entry['dataset']!r} of {key}")
    problem = kind_problem(entry["kind"], metadata.get("is_data"), metadata.get("is_signal"), eft)
    if problem:
        raise ValueError(f"{key}: {problem}")
    if not metadata.get("fileuuid") or "entrystart" not in metadata or "entrystop" not in metadata:
        raise ValueError(f"{key}: the chunk metadata needs a non-empty fileuuid, entrystart and entrystop")
    chunk = metadata_chunk(metadata)
    pass_name = "nominal" if shift_name is None else shift_name
    attempt = ATTEMPTS.get((os.getpid(), key, chunk))
    if attempt is None:
        raise RuntimeError(f"pass_context: begin_chunk did not open chunk {chunk} of {key}")
    if pass_name not in attempt["passes"] or pass_name in attempt["ended"]:
        raise RuntimeError(f"pass_context: pass {pass_name} of chunk {chunk} is not an open pass of {attempt['passes']}")
    unique_id = [registry["tag"], key, metadata["fileuuid"], metadata["entrystart"], metadata["entrystop"], pass_name]
    return {"registry": registry, "key": key, "chunk": chunk, "pass": pass_name, "unique_id": unique_id, "keys": set()}


def open_attempt(ctx):
    attempt = ATTEMPTS.get((os.getpid(), ctx["key"], ctx["chunk"]))
    if attempt is None:
        raise RuntimeError(f"no open attempt for chunk {ctx['chunk']} of {ctx['key']}")
    return attempt


def send_histogram(histos, key, values, labels, weight, exclude, pass_state):
    ctx = pass_state["histserv"]
    if ctx["pass"] in open_attempt(ctx)["ended"]:
        raise RuntimeError(f"send_histogram: {key} sent after pass {ctx['pass']} of chunk {ctx['chunk']} ended")
    if key in ctx["keys"]:
        raise RuntimeError(f"send_histogram: {key} sent twice in pass {ctx['pass']} of chunk {ctx['chunk']}")
    ctx["keys"].add(key)
    target = histos[key]
    slot_names = [ax.name for ax in target.axes[1:] if isinstance(ax, hist.axis.StrCategory)]
    fills = []
    for channel, rows in wzmod.channel_block(exclude, pass_state):
        dense = {name: v[rows] for name, v in values.items()}
        for k, label in enumerate(labels):
            fills.append({"channel": channel, **dict(zip(slot_names, label)), **dense, "weight": weight(k, rows)})
    registry = ctx["registry"]
    hist_id = registry["samples"][ctx["key"]]["hists"][key]
    return send(registry, ctx["key"], hist_id, fills, ctx["unique_id"] + [key], ("-", ctx["chunk"], ctx["pass"]))


def end_pass(pass_state):
    ctx = pass_state["histserv"]
    registry, key, chunk, pass_name = ctx["registry"], ctx["key"], ctx["chunk"], ctx["pass"]
    attempt = open_attempt(ctx)
    result = send(registry, key, entry_of(registry, key)["ledger"], [marker_fill("-", chunk, "pass", pass_name)],
                  [registry["tag"], key, "pass", chunk, pass_name], ("-", chunk, pass_name))
    attempt["ended"].add(pass_name)
    return result


def end_chunk(registry, key, metadata, entries, sumw=None, lumis=None):
    raise_if_poisoned()
    entry = entry_of(registry, key)
    chunk = metadata_chunk(metadata)
    attempt = ATTEMPTS.get((os.getpid(), key, chunk))
    if attempt is None or attempt["ended"] != set(attempt["passes"]):
        raise RuntimeError(f"end_chunk: chunk {chunk} of {key} has passes that did not end")
    if (sumw is None) != (entry["kind"] == "data"):
        raise RuntimeError(f"end_chunk: sumw must be given exactly for non-data kinds ({key} is {entry['kind']})")
    if lumis is not None and entry["kind"] != "data":
        raise RuntimeError(f"end_chunk: lumis are recorded for data only ({key} is {entry['kind']})")
    where = ("-", chunk, "done")
    tag = registry["tag"]
    sums = [sumw_fill("-", chunk, "entries", entries)]
    if sumw is not None:
        sums.append(sumw_fill("-", chunk, "genweight", sumw))
    results = [send(registry, key, entry["sumw"], sums, [tag, key, "sumw_chunk", chunk], where)]
    if lumis is not None:
        names = ("chunk", "run", "lumi")
        fills = [marker_fill(chunk, str(run), str(lumi), names=names) for run, lumi in sorted(lumis)]
        fills = fills or [marker_fill(chunk, EMPTY_LUMI_LABEL, EMPTY_LUMI_LABEL, names=names)]
        results.append(send(registry, key, entry["lumi"], fills, [tag, key, "lumi", chunk], where))
    results.append(send(registry, key, entry["ledger"], [marker_fill("-", chunk, "done", "-")],
                        [tag, key, "done", chunk], where))
    del ATTEMPTS[(os.getpid(), key, chunk)]
    return results


def write_done_file(job_num, registry, key, items):
    path = DONE_FILE_TEMPLATE.format(job_num)
    tmp = path + ".tmp"
    content = {"format": DONE_FILE_FORMAT, "tag": registry["tag"], "key": key,
               "chunks": sorted(item_chunk(item) for item in items)}
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(content, f)
    os.replace(tmp, path)
    return path


def rebuild(template, snapshot, variance_known):
    axes = list(template.axes)
    n = 0
    while n < len(axes) and isinstance(axes[n], hist.axis.StrCategory):
        n += 1
    categories, dense_axes = axes[:n], axes[n:]
    assert tuple(snapshot.chunk_axis_names) == tuple(ax.name for ax in categories)
    assert not any(isinstance(ax, hist.axis.StrCategory) for ax in dense_axes)
    assert tuple(snapshot.dense_view_shape) == hist.Hist(*dense_axes, storage=hist.storage.Weight()).view(flow=True).shape
    items = list(snapshot.items())
    new_axes = []
    for i, ax in enumerate(categories):
        labels = list(dict.fromkeys(chunk_key[i] for chunk_key, _ in items))
        new_ax = hist.axis.StrCategory(labels, name=ax.name, label=ax.label, growth=ax.traits.growth,
                                       overflow=ax.traits.overflow)
        new_ax.__dict__.update(ax.__dict__)
        new_axes.append(new_ax)
    out = hist.Hist(*new_axes, *dense_axes, storage=hist.storage.Weight())
    out.__dict__.update(template.__dict__)
    out._variance_known = variance_known
    view = numpy.asarray(out.view(flow=True))
    for chunk_key, dense in items:
        view[tuple(new_axes[i].index(label) for i, label in enumerate(chunk_key))] = dense
    return out


def snapshot_values(registry, hist_id):
    return {chunk_key: float(dense["value"][0]) for chunk_key, dense in
            remote(registry, hist_id).snapshot(timeout=TIMEOUT).items()}


def entry_state(registry, key):
    entry = entry_of(registry, key)
    state = {"ledger": snapshot_values(registry, entry["ledger"]), "sumw": snapshot_values(registry, entry["sumw"])}
    if "lumi" in entry:
        state["lumi"] = snapshot_values(registry, entry["lumi"])
    return state


def booking(options, kind):
    return set(templates(make_processor(options), kind))


def audit(registry, key, manifest):
    entry = entry_of(registry, key)
    state = entry_state(registry, key)
    found = {}

    def problem(category, name):
        found.setdefault(category, []).append(name)

    file_uuids, expected, truncated_records, started, passes, done, poisoned = {}, {}, {}, {}, {}, [], []
    for (file, chunk, marker, pass_name), value in state["ledger"].items():
        if value <= 0:
            continue
        if marker == "file":
            file_uuids.setdefault(file, set()).add(chunk)
        elif marker == "expected":
            expected.setdefault(chunk, file)
        elif marker == "truncated":
            truncated_records.setdefault(file, []).append(pass_name)
        elif marker == "started":
            started.setdefault(chunk, set()).add(pass_name)
        elif marker == "pass":
            passes.setdefault(chunk, set()).add(pass_name)
        elif marker == "done":
            done.append(chunk)
        elif marker == "poisoned":
            poisoned.append(f"{file} {chunk} {pass_name}")
    per_file, per_chunk = {}, {}
    for (file, chunk, quantity), value in state["sumw"].items():
        if chunk == "-":
            per_file.setdefault(file, {})[quantity] = value
        else:
            per_chunk.setdefault(chunk, {})[quantity] = value
    recorded = set(file_uuids)
    if manifest is None:
        problem("no manifest", key)
    else:
        for file in sorted(set(manifest) - recorded):
            problem("manifest file not recorded", file)
        for file in sorted(recorded - set(manifest)):
            problem("recorded file not in manifest", file)
    file_of_uuid = {}
    for file, uuids in sorted(file_uuids.items()):
        if len(uuids) > 1:
            problem("file with several fileuuids", f"{file}: {sorted(uuids)}")
        for fileuuid in uuids:
            file_of_uuid.setdefault(fileuuid, []).append(file)
    for fileuuid, files in sorted(file_of_uuid.items()):
        if len(files) > 1:
            problem("fileuuid shared by several files", f"{fileuuid}: {files}")

    def chunk_file(chunk):
        files = file_of_uuid.get(parse_chunk_id(chunk)[0], [])
        return files[0] if len(files) == 1 else None

    is_data = entry["kind"] == "data"
    for file in sorted(recorded):
        if file not in per_file:
            if any(chunk_file(chunk) == file for chunk in started):
                problem("sumw missing for a file", file)
            else:
                problem("begin_job interrupted: file recorded without sumw", file)
    for file in sorted(set(per_file) - recorded):
        problem("sumw for an unrecorded file", file)
    done_set = set(done)
    bad_chunks = set()
    for chunk in done:
        record = per_chunk.get(chunk, {})
        if "entries" not in record or (not is_data and "genweight" not in record):
            problem("chunk sumw missing", chunk)
            bad_chunks.add(chunk)
        elif record["entries"] != parse_chunk_id(chunk)[2] - parse_chunk_id(chunk)[1]:
            problem("entries differ from the chunk range", f"{chunk}: entries {record['entries']}")
            bad_chunks.add(chunk)
    for chunk in sorted(set(per_chunk) - done_set):
        problem("chunk sumw for a chunk that is not done", chunk)
    if not is_data:
        for file in sorted(recorded & set(per_file)):
            record = per_file[file]
            chunks = [chunk for chunk in done if chunk_file(chunk) == file]
            ranges = sorted(parse_chunk_id(chunk)[1:] for chunk in chunks)
            covered = (bool(ranges) and ranges[0][0] == 0 and ranges[-1][1] == record.get("events")
                       and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
            if not covered or bad_chunks & set(chunks):
                continue
            entries = sum(per_chunk[chunk]["entries"] for chunk in chunks)
            genweight = sum(per_chunk[chunk]["genweight"] for chunk in chunks)
            runs_sumw, runs_count = record.get("runs_sumw"), record.get("runs_count")
            if (entries != runs_count or runs_sumw is None
                    or abs(genweight - runs_sumw) > production_keys.SUMW_RUNS_TOLERANCE * abs(runs_sumw)):
                problem("Events and Runs disagree", f"{file}: skimmed or filtered file? chunk genWeight sum {genweight!r} "
                        f"vs Runs {runs_sumw!r}, entries {entries} vs Runs {runs_count}")
    expected_set = set(expected)
    for chunk in sorted(expected_set - set(started) - done_set):
        problem("expected, not started", chunk)
    for chunk in sorted((expected_set & set(started)) - done_set):
        problem("started, not done", chunk)
    for chunk in sorted(set(started) - expected_set):
        problem("started, not expected", chunk)
    for chunk in sorted(done_set - expected_set):
        problem("done, not expected", chunk)
    for chunk in sorted(done_set):
        if not started.get(chunk) or passes.get(chunk, set()) != started[chunk]:
            missing = sorted(started.get(chunk, set()) - passes.get(chunk, set()))
            problem("done, missing passes", f"{chunk}: {missing}")
    for name in poisoned:
        problem("poisoned", name)
    if set(entry["hists"]) != booking(registry["options"], entry["kind"]):
        problem("hist keys differ from the booking", key)
    files = {}
    for file in sorted(recorded):
        n_expected = sum(1 for chunk, owner in expected.items() if owner == file)
        records = truncated_records.get(file, [])
        if registry["max_chunks"] is None:
            if records:
                problem("truncation differs from the registry", f"{file}: {records} on an untruncated registry")
            files[file] = {"kept": n_expected, "total": n_expected}
        else:
            if len(records) != 1 or int(records[0].split("/")[0]) != n_expected:
                problem("truncation differs from the registry", f"{file}: {records}, {n_expected} expected chunks")
            kept, total = map(int, records[0].split("/")) if records else (0, 0)
            files[file] = {"kept": kept, "total": total}
    problems = [(category, "missing" if category in MISSING_CATEGORIES else "inconsistent", names)
                for category, names in found.items()]
    if is_data:
        sumw_total = sum(per_file[file]["data_file"] for file in per_file if "data_file" in per_file[file])
    else:
        sumw_total = sum(per_chunk[chunk]["genweight"] for chunk in done if "genweight" in per_chunk.get(chunk, {}))
    contributions = sum(len(passes.get(chunk, ())) for chunk in done)
    coverage = {"chunks_done": len(done_set), "chunks_expected": len(expected_set), "files": files, "lumi": None}
    lumi_chunks = {}
    for (chunk, run, lumi), value in state.get("lumi", {}).items():
        if value > 0:
            pairs = lumi_chunks.setdefault(chunk, set())
            if run != EMPTY_LUMI_LABEL:
                pairs.add((int(run), int(lumi)))
    without = sorted(done_set - set(lumi_chunks))
    not_done = sorted(set(lumi_chunks) - done_set)
    if is_data and without:
        print(f"{key}: information: {len(without)} done chunks without a lumi record")
    if not_done:
        print(f"{key}: information: lumi records of {len(not_done)} chunks that are not done, left out: "
              + ", ".join(not_done[:MAX_LISTED_NAMES]) + (" ..." if len(not_done) > MAX_LISTED_NAMES else ""))
    recorded = [lumi_chunks[chunk] for chunk in sorted(done_set) if chunk in lumi_chunks]
    if recorded:
        coverage["lumi"] = lumi_record.as_array(set().union(*recorded))
    return problems, sumw_total, contributions, state, coverage


def output_names(stem, cls, truncated):
    name = stem + (".TRUNCATED" if truncated else "") + {"complete": "", "partial": ".PARTIAL",
                                                        "incomplete": ".INCOMPLETE"}[cls]
    return name + ".pkl.gz", name + ".json"


def production_class(problems):
    if not problems:
        return "complete"
    return "partial" if all(cls == "missing" for _, cls, _ in problems) else "incomplete"


def print_problems(key, problems):
    for category, cls, names in problems:
        listed = ", ".join(names[:MAX_LISTED_NAMES]) + (" ..." if len(names) > MAX_LISTED_NAMES else "")
        print(f"{key}: {category} [{cls}] ({len(names)}): {listed}")


def write_pickle(path, content):
    with gzip.open(path, "wb") as f:
        pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(path, content):
    with open(path, "w") as f:
        json.dump(content, f, indent=1, sort_keys=True)


def output_groups(registry, filename, per_entry, cls="complete"):
    tag, era = registry["tag"], registry["options"]["era"]
    groups = {}
    if per_entry:
        for key, entry in registry["samples"].items():
            groups[production_keys.sample_file_stem(key) + "_histograms"] = {"kind": entry["kind"], "keys": [key],
                                                                             "variations": entry["variations"]}
        return groups, {}
    for key, entry in registry["samples"].items():
        stem = production_keys.final_file_stem(tag, era, filename, entry["kind"], entry["variations"])
        groups.setdefault(stem, {"kind": entry["kind"], "keys": [], "variations": entry["variations"]})["keys"].append(key)
    placeholders = {}
    if cls != "complete" or registry["max_chunks"] is not None:
        return groups, placeholders
    for group in list(groups.values()):
        if group["variations"]["dd"] is not None:
            continue
        for kind in STANDARD_KINDS:
            stem = production_keys.final_file_stem(tag, era, filename, kind, group["variations"])
            if stem not in groups:
                placeholders[stem] = {"kind": kind, "keys": [], "variations": group["variations"]}
    return groups, placeholders


def collect(registry, manifests, outdir, filename="latest", per_entry=False, allow_incomplete=False, force=False,
            registry_path=None):
    if not NAME_PATTERN.match(filename):
        raise ValueError(f"filename {filename!r} must match {NAME_PATTERN.pattern}")
    kinds_of = {}
    for entry in registry["samples"].values():
        kinds_of.setdefault(entry["dataset"], set()).add(entry["kind"])
    mixed = sorted(dataset for dataset, kinds in kinds_of.items() if len(kinds) > 1)
    if mixed:
        raise ValueError(f"datasets registered with several kinds: {mixed}")
    truncated = registry["max_chunks"] is not None
    groups, _ = output_groups(registry, filename, per_entry)
    other_layout, _ = output_groups(registry, filename, not per_entry)
    variants = [name for stem in list(groups) + list(other_layout) for cls in CLASSES for flag in (False, True)
                for name in output_names(stem, cls, flag)]
    taken = [name for name in variants + [name + ".tmp" for name in variants] + [COLLECT_FILE]
             if os.path.exists(os.path.join(outdir, name))]
    if taken and not force:
        print(f"collect: refusing to mix outputs; these exist in {outdir} (use --force to replace them):")
        for name in taken:
            print(f"  {name}")
        return 2
    audits = {}
    all_problems = []
    for key in registry["samples"]:
        audits[key] = audit(registry, key, manifests.get(key))
        print_problems(key, audits[key][0])
        all_problems += audits[key][0]
    cls = production_class(all_problems)
    placeholders = output_groups(registry, filename, per_entry, cls)[1]
    if all_problems and not allow_incomplete:
        print(f"collect: refused, the production is {cls}; nothing was written")
        return 1
    os.makedirs(outdir, exist_ok=True)
    tmps = []
    files = {}
    reports = []

    def refuse(message):
        for tmp in tmps:
            if os.path.exists(tmp):
                os.remove(tmp)
        print(message)
        return 1

    for stem, group in groups.items():
        pkl_name, json_name = output_names(stem, cls, truncated)
        content = {}
        datasets = {}
        contributions = {}
        left_out = []
        for key in group["keys"]:
            entry = registry["samples"][key]
            problems, sumw_total, n_contributions, state, coverage = audits[key]
            dataset = entry["dataset"]
            if n_contributions == 0:
                left_out.append(key)
                continue
            booked = templates(make_processor(registry["options"]), entry["kind"])
            hists = {name: rebuild(booked[name], remote(registry, hist_id).snapshot(timeout=TIMEOUT), False)
                     for name, hist_id in entry["hists"].items()}
            if entry_state(registry, key) != state:
                return refuse(f"collect: refused, {CHANGED_CATEGORY}: {key}; nothing was written")
            if dataset in content:
                record = content[dataset]
                record["hist"] = wzmod.merge_histogram_sets([{dataset: record["hist"]}, {dataset: hists}])[dataset]
                record["sumw"] += sumw_total
                if coverage["lumi"] is not None:
                    record["lumi"] = (coverage["lumi"] if "lumi" not in record
                                      else lumi_record.union(record["lumi"], coverage["lumi"]))
                if "production" in record:
                    production = record["production"]
                    production["chunks_done"] += coverage["chunks_done"]
                    production["chunks_expected"] += coverage["chunks_expected"]
                    production["files"].update(coverage["files"])
                    production["entries"].append(key)
            else:
                record = {"hist": hists, "sumw": sumw_total, "kind": entry["kind"]}
                if coverage["lumi"] is not None:
                    record["lumi"] = coverage["lumi"]
                if truncated or cls != "complete":
                    record["production"] = {"truncated": truncated, "max_chunks": registry["max_chunks"], "class": cls,
                                            "chunks_done": coverage["chunks_done"],
                                            "chunks_expected": coverage["chunks_expected"],
                                            "files": dict(coverage["files"])}
                    if not per_entry:
                        record["production"]["entries"] = [key]
                content[dataset] = record
            datasets.setdefault(dataset, []).append(key)
            contributions[dataset] = contributions.get(dataset, 0) + n_contributions
        for dataset, record in content.items():
            for h in record["hist"].values():
                h._variance_known = contributions[dataset] == 1
        missing_datasets = sorted({registry["samples"][key]["dataset"] for key in left_out} - set(content))
        if content or cls == "complete":
            path = os.path.join(outdir, pkl_name) + ".tmp"
            tmps.append(path)
            write_pickle(path, content)
            files[pkl_name] = {"kind": group["kind"], "variations": production_keys.variation_suffix(group["variations"]),
                               "datasets": datasets}
        if cls != "complete":
            report = {"class": cls, "file": pkl_name if content else None, "datasets_without_done_chunks": missing_datasets,
                      "problems": {key: [{"category": c, "class": k, "names": n} for c, k, n in audits[key][0]]
                                   for key in group["keys"]}}
            if cls == "incomplete":
                report["warning"] = INCOMPLETE_WARNING
            else:
                report["statement"] = PARTIAL_STATEMENT
            path = os.path.join(outdir, json_name) + ".tmp"
            tmps.append(path)
            write_json(path, report)
            reports.append(json_name)
    for stem, group in placeholders.items():
        pkl_name = output_names(stem, cls, truncated)[0]
        if os.path.exists(os.path.join(outdir, pkl_name)):
            print(f"{pkl_name}: no {production_keys.KIND_FILE_LABELS[group['kind']]} entries in this registry, "
                  f"so the existing file is kept")
            continue
        path = os.path.join(outdir, pkl_name) + ".tmp"
        tmps.append(path)
        write_pickle(path, {})
        files[pkl_name] = {"kind": group["kind"], "variations": production_keys.variation_suffix(group["variations"]),
                           "datasets": {}}
    collect_path = os.path.join(outdir, COLLECT_FILE)
    if os.path.exists(collect_path):
        os.remove(collect_path)
    written = {os.path.basename(tmp)[:-len(".tmp")] for tmp in tmps}
    for name in variants:
        path = os.path.join(outdir, name)
        if name not in written and os.path.exists(path):
            os.remove(path)
    for tmp in tmps:
        os.replace(tmp, tmp[:-len(".tmp")])
    exit_code = 0 if cls == "complete" else 3
    summary = {"format": COLLECT_FORMAT, "tag": registry["tag"], "era": registry["options"]["era"],
               "registry": os.path.abspath(registry_path) if registry_path else None,
               "layout": "per-entry" if per_entry else "final", "filename": filename, "files": files,
               "reports": reports, "class": cls, "truncated": truncated, "max_chunks": registry["max_chunks"],
               "exit_code": exit_code}
    write_json(collect_path + ".tmp", summary)
    os.replace(collect_path + ".tmp", collect_path)
    return exit_code


def read_manifest(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


REGISTER_DESCRIPTION = """Register a C4 production on a central histserv server and write its registry JSON.

The server is the operator's. Start it with
  python -m histserv --port 0 --prune-after-seconds <seconds >= the production's lifetime>
and take the bound port from its log. The default prune interval of one day deletes idle
histograms before a longer production collects them. That server listens on every interface,
so keep the registry (it holds the access token) private.

--sample takes the full DAS dataset name (/primary/processed/NANOAOD or NANOAODSIM) and the kind
(data, mc, signal or eft); private datasets (tier USER) are refused. MC on the Rochester smearing
path (Run 2) is refused, because C4 needs deterministic chunk processing (roccor.py:31, N15).
--max-chunks makes a truncated test production: every output name is marked .TRUNCATED.
"""

COLLECT_DESCRIPTION = """Audit a C4 production and write its final histogram files.

Collect only when no job is running, held or failed; re-run a failed job first (deduplication
makes that safe). A refusal names what is missing; a "poisoned" refusal means re-registering
under a new tag and re-running the production. The output is point-in-time consistent: a
production that changes while it is collected is refused.

The final files are merged-histogram-Inc-<tag>-<era>-<filename><variation>-<KIND>.pkl.gz, one per
kind (Data, MC-signal, MC-backgrounds, EFT) and variation. They replace the merge step: DCTools
lists the -Data, -MC-signal and -MC-backgrounds files. --per-entry writes one
<stem>_histograms.pkl.gz per registry entry instead (hist-merger.py writes
jobs_<tag>_<era>_<sample>_histograms.pkl.gz). --allow-incomplete writes .PARTIAL output (whole
chunks missing; MC normalized to what it holds, data with its lumi record) or .INCOMPLETE output
("not for physics results"). Job pickles are never read.
"""


def parse_pairs(values, option):
    pairs = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or not name or not rest:
            raise ValueError(f"{option} needs NAME=VALUE, got {value!r}")
        if name in pairs:
            raise ValueError(f"{option} names {name} twice")
        pairs[name] = rest
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m qawa.histserv_output",
                                     description="C4: fill one central histserv server instead of job pickles.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("register", description=REGISTER_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address", default=os.environ.get("HISTSERV_ADDRESS"),
                   help="HOST:PORT of the server (default: $HISTSERV_ADDRESS; one of the two is required)")
    p.add_argument("--tag", required=True, help="production tag, [A-Za-z0-9_-]+")
    p.add_argument("--era", required=True)
    p.add_argument("--version", required=True, help="NanoAOD version, e.g. v9 or v15")
    p.add_argument("--pol", action="store_true")
    p.add_argument("--split-by-charge", action="store_true")
    p.add_argument("--split-FV", action="store_true")
    p.add_argument("--max-chunks", type=int, default=None, help="truncated test production of N chunks per dataset")
    p.add_argument("--dd-file", default=None, help="data-driven estimate (correctionlib JSON) of a data re-processing production")
    p.add_argument("--sample", action="append", default=[], required=True, help="DAS=KIND, repeatable")
    p.add_argument("--output", required=True, help="registry JSON to write (mode 0600)")
    c = sub.add_parser("collect", description=COLLECT_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_argument("--registry", required=True)
    c.add_argument("--manifest", action="append", default=[],
                   help="KEY=PATH, repeatable: the input files of one registry key, one per line (# and blank lines "
                        "ignored); HTCondor: <jobs_dir>/inputfiles.dat")
    c.add_argument("--outdir", required=True)
    c.add_argument("-f", "--filename", default="latest", help="name part of the final files (default: latest)")
    c.add_argument("--per-entry", action="store_true", help="write one file per registry entry (debugging)")
    c.add_argument("--allow-incomplete", action="store_true", help="write .PARTIAL or .INCOMPLETE output")
    c.add_argument("--force", action="store_true",
                   help="replace an earlier output set; a set is valid only with the collect.json naming it")
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            if not args.address:
                raise ValueError("register needs --address or HISTSERV_ADDRESS")
            options = {"era": args.era, "version": args.version, "pol": args.pol,
                       "split_by_charge": args.split_by_charge, "split_FV": args.split_FV}
            registry = register(args.address, args.tag, options, parse_pairs(args.sample, "--sample"),
                                secrets.token_hex(16), max_chunks=args.max_chunks, dd_file=args.dd_file)
            write_registry(args.output, registry)
            print(f"registry written to {args.output} ({len(registry['samples'])} entries)")
            return 0
        registry = read_registry(args.registry)
        manifests = {key: read_manifest(path) for key, path in parse_pairs(args.manifest, "--manifest").items()}
        return collect(registry, manifests, args.outdir, filename=args.filename, per_entry=args.per_entry,
                       allow_incomplete=args.allow_incomplete, force=args.force, registry_path=args.registry)
    except (ValueError, NotImplementedError) as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
