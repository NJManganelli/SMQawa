import hashlib
import os
import re

TIERS = ("NANOAOD", "NANOAODSIM")
VARIATION_ORDER = ("pol", "split_by_charge", "split_FV", "eft", "dd")
SUMW_RUNS_TOLERANCE = 1e-6
DAS_PATTERN = re.compile(r"^/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/(NANOAOD|NANOAODSIM)$")
DD_LABEL_PATTERN = re.compile(r"^[0-9a-f]{12}$")
KIND_FILE_LABELS = {"data": "Data", "signal": "MC-signal", "mc": "MC-backgrounds", "eft": "EFT"}


def das_parts(das):
    match = DAS_PATTERN.match(das) if isinstance(das, str) else None
    if match is None:
        raise ValueError(f"not a DAS dataset name /primary/processed/{'|'.join(TIERS)}: {das!r}")
    return match.groups()


def primary_dataset(das):
    return das_parts(das)[0]


def das_name_from_path(path):
    parts = os.fspath(path).split("/")
    for i in range(len(parts) - 5):
        if parts[i] == "store" and parts[i + 1] in ("mc", "data"):
            campaign, primary, tier, processed = parts[i + 2:i + 6]
            if tier not in TIERS:
                return None
            das = f"/{primary}/{campaign}-{processed}/{tier}"
            try:
                das_parts(das)
            except ValueError:
                return None
            return das
    return None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dd_label(path):
    return file_sha256(path)[:12]


def variation_tokens(variations):
    unknown = set(variations) - set(VARIATION_ORDER)
    if unknown:
        raise ValueError(f"unknown variations {sorted(unknown)}; known: {VARIATION_ORDER}")
    tokens = [name for name in VARIATION_ORDER[:-1] if variations.get(name)]
    dd = variations.get("dd")
    if dd is not None:
        if not isinstance(dd, str) or DD_LABEL_PATTERN.match(dd) is None:
            raise ValueError(f"dd label must be 12 lowercase hex digits: {dd!r}")
        tokens.append(f"dd:{dd}")
    return tokens


def sample_key(das, pol=False, split_by_charge=False, split_FV=False, eft=False, dd=None):
    das_parts(das)
    tokens = variation_tokens({"pol": pol, "split_by_charge": split_by_charge, "split_FV": split_FV, "eft": eft, "dd": dd})
    return das + "#" + "+".join(tokens) if tokens else das


def parse_sample_key(key):
    das, sep, rest = key.partition("#")
    das_parts(das)
    variations = {}
    if sep:
        for token in rest.split("+"):
            name, colon, value = token.partition(":")
            if name not in VARIATION_ORDER or name in variations or bool(colon) != (name == "dd"):
                raise ValueError(f"bad variation token {token!r} in sample key {key!r}")
            variations[name] = value if name == "dd" else True
    if sample_key(das, **variations) != key:
        raise ValueError(f"sample key {key!r} is not in canonical form")
    return das, variations


def sample_file_stem(key):
    das, variations = parse_sample_key(key)
    return "__".join(list(das_parts(das)) + [token.replace(":", "-") for token in variation_tokens(variations)])


def variation_suffix(variations):
    tokens = [token.replace(":", "-") for token in variation_tokens(variations) if token != "eft"]
    return "".join("__" + token for token in tokens)


def final_file_stem(tag, era, filename, kind, variations):
    return f"merged-histogram-Inc-{tag}-{era}-{filename}{variation_suffix(variations)}-{KIND_FILE_LABELS[kind]}"
