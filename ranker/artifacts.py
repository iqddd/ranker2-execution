"""Reusable artifact loading, finiteness and digest verification."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ranker.io import atomic_json, is_read_only_file, read_csv, read_json, sha256_file


def numeric_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_finite(item) for item in value)
    return True


def verify_artifact(
    path: Path, required_keys: set[str], *, kind: str | None = None
) -> dict[str, Any]:
    exists = path.exists()
    regular = path.is_file()
    nonzero = regular and path.stat().st_size > 0
    load_succeeds = False
    keys_present = False
    finite = False
    error = ""
    try:
        artifact_kind = kind or path.suffix.lstrip(".")
        if artifact_kind == "npz":
            with np.load(path, allow_pickle=False) as payload:
                keys_present = required_keys <= set(payload.files)
                finite = all(numeric_finite(np.asarray(payload[key])) for key in payload.files)
        elif artifact_kind == "pt":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            keys_present = required_keys <= set(payload)
            finite = numeric_finite(payload)
        elif artifact_kind in {"csv", "csv_union"}:
            if kind is None and path.read_text(encoding="utf-8").lstrip().startswith("{"):
                payload = read_json(path)
                keys_present = required_keys <= set(payload)
                finite = numeric_finite(payload)
            else:
                payload = read_csv(path)
                keys_present = bool(payload) and required_keys <= set(payload[0])
                forbidden = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
                finite = all(
                    str(value).strip().lower() not in forbidden
                    for row in payload
                    for value in row.values()
                )
        elif artifact_kind == "json":
            payload = read_json(path)
            keys_present = required_keys <= set(payload)
            finite = numeric_finite(payload)
        elif artifact_kind == "text":
            payload = path.read_text(encoding="utf-8")
            keys_present = not required_keys
            finite = True
        else:
            raise ValueError(f"Unsupported artifact kind: {artifact_kind!r}")
        load_succeeds = True
    except Exception as exception:  # noqa: BLE001 - verification must report loader failures.
        error = f"{type(exception).__name__}: {exception}"
    verified = bool(exists and regular and nonzero and load_succeeds and keys_present and finite)
    return {
        "path": str(path.resolve()),
        "artifact_kind": kind or path.suffix.lstrip("."),
        "exists": exists,
        "regular_file": regular,
        "nonzero_size": nonzero,
        "size_bytes": path.stat().st_size if regular else 0,
        "sha256": sha256_file(path) if regular else "",
        "load_read_succeeds": load_succeeds,
        "required_keys_present": keys_present,
        "all_numeric_arrays_finite": finite,
        "verified": verified,
        "error": error,
    }


def write_verified_artifact_manifest(
    *,
    signature: str,
    output_names: tuple[str, ...],
    output_dir: Path,
    required_keys: Mapping[str, set[str]],
    manifest_name: str,
    enforce_exact_set: bool = False,
) -> dict[str, Any]:
    """Verify an artifact set and write a self-verifying canonical manifest."""
    if set(required_keys) != set(output_names) - {manifest_name}:
        raise ValueError("Artifact verification specs must cover every non-manifest output.")
    planned = list(output_names)
    planned_set = set(planned)
    if len(planned_set) != len(planned):
        raise ValueError("Planned artifact names must be unique.")
    checks = [
        verify_artifact(output_dir / name, required_keys[name])
        for name in output_names
        if name != manifest_name
    ]
    if not all(check["verified"] for check in checks):
        raise RuntimeError("At least one planned artifact failed verification.")
    entries_before_manifest = {path.name for path in output_dir.iterdir()}
    allowed_before_manifest = planned_set - {manifest_name}
    artifact_set_precondition = bool(
        not enforce_exact_set
        or entries_before_manifest == allowed_before_manifest
        or entries_before_manifest == planned_set
    )
    if not artifact_set_precondition:
        unexpected = sorted(entries_before_manifest - planned_set)
        missing = sorted(allowed_before_manifest - entries_before_manifest)
        raise RuntimeError(
            "Artifact directory does not match the planned set before manifest write: "
            f"unexpected={unexpected}, missing={missing}."
        )
    written = [name for name in planned if name != manifest_name] + [manifest_name]
    verified = [
        *[Path(check["path"]).name for check in checks],
        manifest_name,
    ]
    parity = bool(set(planned) == set(written) == set(verified))
    manifest: dict[str, Any] = {
        "signature": signature,
        "planned_artifacts": planned,
        "written_artifacts": written,
        "verified_artifacts": verified,
        "planned_equals_written_equals_verified": parity,
        "artifact_set_equality_pass": artifact_set_precondition,
        "artifact_checks": checks,
        "self_digest_scheme": "sha256 of canonical JSON payload with self_digest omitted",
    }
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    manifest["self_digest"] = hashlib.sha256(canonical).hexdigest()
    manifest_path = output_dir / manifest_name
    atomic_json(manifest_path, manifest)
    self_check = verify_artifact(
        manifest_path,
        {"signature", "planned_artifacts", "written_artifacts", "verified_artifacts", "self_digest"},
    )
    reloaded = read_json(manifest_path)
    saved_digest = reloaded.pop("self_digest")
    replay = json.dumps(
        reloaded, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if not self_check["verified"] or saved_digest != hashlib.sha256(replay).hexdigest():
        raise RuntimeError("Artifact manifest self-verification failed.")
    final_entries = {path.name for path in output_dir.iterdir()}
    if enforce_exact_set and final_entries != planned_set:
        raise RuntimeError(
            "Final artifact directory does not equal the planned set: "
            f"unexpected={sorted(final_entries - planned_set)}, "
            f"missing={sorted(planned_set - final_entries)}."
        )
    if not parity:
        raise RuntimeError("Planned, written and verified artifact sets differ.")
    return manifest


def verify_artifact_manifest(
    manifest_path: Path, *, verify_files: bool = True
) -> dict[str, Any]:
    """Verify either a historical manifest or ``RANKER_ARTIFACT_MANIFEST_V2``.

    Historical manifests stored absolute paths in their checks.  File replay is
    intentionally resolved by basename relative to the manifest so archived
    artifact directories remain portable.
    """
    manifest = read_json(manifest_path)
    version = manifest.get("manifest_version")
    if version not in (None, "RANKER_ARTIFACT_MANIFEST_V2"):
        raise RuntimeError(f"Unsupported artifact manifest version: {version!r}.")
    saved_digest = manifest.get("self_digest")
    if not isinstance(saved_digest, str) or not saved_digest:
        raise RuntimeError("Artifact manifest has no self digest.")
    replay_payload = dict(manifest)
    replay_payload.pop("self_digest", None)
    canonical = json.dumps(
        replay_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if saved_digest != hashlib.sha256(canonical).hexdigest():
        raise RuntimeError("Artifact manifest self-digest replay failed.")
    planned = manifest.get("planned_artifacts")
    written = manifest.get("written_artifacts")
    verified = manifest.get("verified_artifacts")
    if not all(isinstance(values, list) for values in (planned, written, verified)):
        raise RuntimeError("Artifact manifest has invalid artifact-set fields.")
    if (
        len(planned) != len(set(planned))
        or set(planned) != set(written)
        or set(planned) != set(verified)
        or not bool(manifest.get("artifact_set_equality_pass"))
    ):
        raise RuntimeError("Artifact manifest set parity failed.")
    checks = manifest.get("artifact_checks", [])
    if not isinstance(checks, list) or not all(
        isinstance(check, Mapping) and bool(check.get("verified")) for check in checks
    ):
        raise RuntimeError("Artifact manifest contains an unverified artifact check.")
    manifest_name = manifest_path.name
    checked_names = {Path(str(check.get("path", ""))).name for check in checks}
    if checked_names != set(planned) - {manifest_name}:
        raise RuntimeError("Artifact manifest checks do not cover the planned payloads.")
    if verify_files:
        for check in checks:
            stored_path = Path(str(check.get("path", "")))
            candidate = manifest_path.parent / stored_path.name
            if not candidate.is_file():
                raise RuntimeError(f"Manifest artifact is missing: {candidate.name}.")
            expected = str(check.get("sha256", ""))
            if not expected or sha256_file(candidate) != expected:
                raise RuntimeError(f"Manifest artifact digest failed: {candidate.name}.")
        if version == "RANKER_ARTIFACT_MANIFEST_V2":
            freeze_ledger = manifest.get("freeze_ledger", [])
            if not isinstance(freeze_ledger, list):
                raise TypeError("Manifest freeze ledger must be a list.")
            for record in freeze_ledger:
                if not isinstance(record, Mapping):
                    raise TypeError("Manifest freeze ledger records must be objects.")
                candidate = manifest_path.parent / str(record.get("artifact", ""))
                expected = str(record.get("sha256", ""))
                if (
                    not candidate.is_file()
                    or not expected
                    or sha256_file(candidate) != expected
                    or not bool(record.get("read_only"))
                    or not is_read_only_file(candidate)
                ):
                    raise RuntimeError(
                        f"Manifest frozen artifact verification failed: {candidate.name}."
                    )
    return manifest
