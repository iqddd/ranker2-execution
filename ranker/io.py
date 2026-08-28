from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_safe(value: object) -> object:
    """Convert NumPy and Torch values to the JSON values they represent."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON document from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one UTF-8 CSV using the repository's stable dictionary contract."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write indented UTF-8 JSON through the repository's stable tmp suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text with normalized LF newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def canonical_json_digest(value: object) -> str:
    """Hash a JSON-compatible value using the project's canonical encoding."""
    encoded = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str] | None = None,
) -> None:
    """Atomically write rows, preserving either supplied or first-row field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"Refusing to write empty CSV {path} without fields.")
    fieldnames = list(fields) if fields is not None else list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_csv_union(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write rows with fields first seen across all rows."""
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}.")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fields.append(name)
    atomic_csv(path, rows, fields)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write a compressed NPZ archive without changing array dtypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_torch(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically serialize a Torch mapping using the legacy payload contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Return the stable SHA-256 digest used by Step40--Step41 manifests."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_read_only_file(path: Path) -> bool:
    """Return the portable mode/Windows attribute read-only audit."""
    metadata = path.stat()
    mode_read_only = not bool(metadata.st_mode & stat.S_IWRITE)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    attribute_read_only = bool(
        file_attributes & getattr(stat, "FILE_ATTRIBUTE_READONLY", 0)
    )
    return bool(mode_read_only or attribute_read_only)


def parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, accepting the common trailing ``Z`` form."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_summary_json(output_dir: Path, summary_payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_path
