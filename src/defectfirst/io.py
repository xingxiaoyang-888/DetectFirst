"""Portable, hashed artifacts. JSON never silently serializes NaN as a result."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def source_fingerprint() -> str:
    directory = Path(__file__).parent
    # Normalize line endings so Git's Windows checkout conversion is not a code change.
    return digest(
        {
            path.relative_to(directory).as_posix(): hashlib.sha256(
                path.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            for path in sorted(directory.rglob("*.py"))
        }
    )


def sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(digest(list(parts))[:16], 16) % (2**63 - 1)


def atomic_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: str | Path, value: Any) -> None:
    atomic_bytes(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    atomic_bytes(path, b"".join(canonical_bytes(row) + b"\n" for row in rows))


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def within(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root):
        raise ValueError(f"Path must remain relative to the artifact root: {relative}")
    return candidate


def require_keys(value: dict, required: set[str], optional: set[str] | None = None) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing or extra:
        raise ValueError(f"Invalid fields: missing={sorted(missing)}, unknown={sorted(extra)}")
