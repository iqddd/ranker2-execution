"""Shared execution harness for reproducible, artifact-producing steps."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from ranker.artifacts import verify_artifact, verify_artifact_manifest
from ranker.io import (
    atomic_csv,
    atomic_csv_union,
    atomic_json,
    atomic_npz,
    atomic_text,
    atomic_torch,
    is_read_only_file,
    json_safe,
    read_json,
    sha256_file,
)

ArtifactKind = Literal["json", "csv", "csv_union", "npz", "pt", "text"]
MANIFEST_VERSION = "RANKER_ARTIFACT_MANIFEST_V2"
WORKSPACE_VERSION = "RANKER_WORKSPACE_V1"
RESUMABLE_WORKSPACE_VERSION = "RANKER_WORKSPACE_V2"
JOURNAL_VERSION = "RANKER_JOURNAL_V1"
WorkspaceLifecycle = Literal["delete_on_success", "keep_always", "delete_always"]
ExistingOutputPolicy = Literal["replace", "refuse", "resume"]
_SLOT_COMPONENT = re.compile(r"^[A-Za-z0-9_.=-]+$")


@dataclass(frozen=True)
class ArtifactVariant:
    """Alternate kind/schema for a conditionally shaped artifact."""

    kind: ArtifactKind
    required_keys: frozenset[str]


@dataclass(frozen=True)
class Artifact:
    """One declared non-manifest output of a step."""

    name: str
    kind: ArtifactKind
    required_keys: frozenset[str]
    terminal: bool = False
    variants: Mapping[str, ArtifactVariant] | None = None

    def specification(self, variant: str | None = None) -> ArtifactVariant:
        if variant is None:
            return ArtifactVariant(self.kind, self.required_keys)
        if self.variants is None or variant not in self.variants:
            raise KeyError(f"Unknown variant {variant!r} for artifact {self.name!r}.")
        return self.variants[variant]


@dataclass(frozen=True)
class TimeBudget:
    hard_limit: float
    graceful_abort: float
    writing_reserve: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.graceful_abort <= self.hard_limit):
            raise ValueError("Expected 0 < graceful_abort <= hard_limit.")
        if self.writing_reserve < 0.0:
            raise ValueError("writing_reserve must be non-negative.")


class TimeBudgetExceeded(RuntimeError):
    """Raised before a step can overrun its declared graceful budget."""


@dataclass(frozen=True)
class Workspace:
    """Declared scratch storage with an explicit, safely enforced lifecycle."""

    root: Path
    lifecycle: WorkspaceLifecycle = "delete_on_success"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if self.lifecycle not in ("delete_on_success", "keep_always", "delete_always"):
            raise ValueError(f"Unknown workspace lifecycle {self.lifecycle!r}.")

    @property
    def marker_path(self) -> Path:
        return self.root / ".ranker-workspace.json"

    def _resolved_root(self) -> Path:
        resolved = self.root.resolve()
        forbidden = {Path(resolved.anchor).resolve(), Path.cwd().resolve(), Path.home().resolve()}
        if resolved in forbidden or resolved.parent == resolved:
            raise RuntimeError(f"Refusing to manage broad workspace path: {resolved}")
        return resolved

    def _expected_marker(self, signature: str, resume_fingerprint: str | None = None) -> dict[str, str | None]:
        return {
            "workspace_version": (RESUMABLE_WORKSPACE_VERSION if resume_fingerprint else WORKSPACE_VERSION),
            "signature": signature,
            "root": str(self._resolved_root()),
            "lifecycle": self.lifecycle,
            "resume_fingerprint": resume_fingerprint,
        }

    def _verify_marker(
        self,
        signature: str,
        resume_fingerprint: str | None = None,
        *,
        allow_any_fingerprint: bool = False,
        allow_any_signature: bool = False,
    ) -> None:
        if not self.root.is_dir() or not self.marker_path.is_file():
            raise RuntimeError("Refusing to remove an unmarked workspace.")
        marker = read_json(self.marker_path)
        expected = self._expected_marker(signature, resume_fingerprint)
        common_matches = all(marker.get(name) == expected[name] for name in ("root", "lifecycle")) and (
            allow_any_signature or marker.get("signature") == signature
        )
        version_valid = marker.get("workspace_version") in {WORKSPACE_VERSION, RESUMABLE_WORKSPACE_VERSION}
        fingerprint_matches = allow_any_fingerprint or marker.get("resume_fingerprint") == resume_fingerprint
        if not common_matches or not version_valid or not fingerprint_matches:
            raise RuntimeError("Workspace marker does not match this run contract.")

    def prepare(self, signature: str, *, restart: bool = False, resume_fingerprint: str | None = None) -> bool:
        """Prepare scratch storage and return whether a compatible run was resumed."""
        self._resolved_root()
        if self.root.exists():
            if restart:
                self._verify_marker(signature, resume_fingerprint, allow_any_fingerprint=True)
                shutil.rmtree(self.root)
            elif resume_fingerprint is not None:
                self._verify_marker(signature, resume_fingerprint, allow_any_signature=True)
                atomic_json(self.marker_path, self._expected_marker(signature, resume_fingerprint))
                return True
            else:
                raise RuntimeError(
                    f"Workspace already exists and is not resumable: {self.root}. "
                    "Use the explicit restart option to discard it."
                )
        self.root.mkdir(parents=True, exist_ok=False)
        atomic_json(self.marker_path, self._expected_marker(signature, resume_fingerprint))
        return False

    def slot(self, namespace: str, *, suffix: str = "", **keys: int | str) -> Path:
        """Return a deterministic scratch path composed only from safe components."""
        components = [namespace, suffix, *keys]
        values = [namespace, suffix, *map(str, keys), *map(str, keys.values())]
        if (
            not namespace
            or not keys
            or any(not isinstance(key, str) for key in keys)
            or any(value and not _SLOT_COMPONENT.fullmatch(value) for value in values)
            or any(value in (".", "..") for value in values)
        ):
            raise ValueError(f"Unsafe workspace slot components: {components!r}")
        filename = "__".join(f"{key}={keys[key]}" for key in sorted(keys)) + suffix
        result = self.root / namespace / filename
        if self._resolved_root() not in result.resolve().parents:
            raise RuntimeError("Workspace slot escaped its declared root.")
        return result

    def finish(self, signature: str, *, success: bool, resume_fingerprint: str | None = None) -> None:
        remove = self.lifecycle == "delete_always" or (success and self.lifecycle == "delete_on_success")
        if remove and self.root.exists():
            self._verify_marker(signature, resume_fingerprint)
            shutil.rmtree(self.root)


class Journal:
    """Fingerprint-bound immutable records for independently replayable work units."""

    def __init__(self, workspace: Workspace, fingerprint: str) -> None:
        if not fingerprint:
            raise ValueError("Journal fingerprint must be non-empty.")
        self.workspace = workspace
        self.fingerprint = fingerprint
        self._executed_units: list[dict[str, Any]] = []
        self._resumed_units: list[dict[str, Any]] = []

    @staticmethod
    def _canonical(value: object) -> Any:
        return json.loads(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    @classmethod
    def _unit(cls, kind: str, keys: Mapping[str, object]) -> dict[str, Any]:
        if not kind or any(not isinstance(name, str) or not name for name in keys):
            raise ValueError("Journal units require a kind and non-empty string keys.")
        unit = {"kind": kind, "keys": cls._canonical(dict(keys))}
        if not isinstance(unit["keys"], dict):  # pragma: no cover - defensive.
            raise TypeError("Journal unit keys must serialize to an object.")
        return unit

    @staticmethod
    def _digest(payload: object) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _path(self, unit: Mapping[str, Any]) -> Path:
        return self.workspace.slot("journal", unit=self._digest(unit), suffix=".json")

    def _read_record(self, unit: Mapping[str, Any], *, source_path: Path | None = None) -> dict[str, Any]:
        path = self._path(unit)
        if source_path is not None and source_path.resolve() != path.resolve():
            raise RuntimeError(f"Journal record is stored under a noncanonical name: {source_path.name}")
        if not path.is_file():
            raise KeyError(f"Journal unit is incomplete: {unit!r}")
        record = read_json(path)
        saved_digest = record.pop("record_digest", None)
        if (
            record.get("journal_version") != JOURNAL_VERSION
            or record.get("fingerprint") != self.fingerprint
            or record.get("unit") != unit
            or saved_digest != self._digest(record)
        ):
            raise RuntimeError(f"Journal record verification failed: {path.name}")
        return record

    def is_done(self, kind: str, **keys: object) -> bool:
        unit = self._unit(kind, keys)
        path = self._path(unit)
        if not path.exists():
            return False
        self._read_record(unit)
        return True

    def load(self, kind: str, **keys: object) -> Any:
        unit = self._unit(kind, keys)
        record = self._read_record(unit)
        if unit not in self._resumed_units:
            self._resumed_units.append(unit)
        return self._canonical(record["payload"])

    def record(self, kind: str, payload: object, *, elapsed_seconds: float = 0.0, **keys: object) -> Any:
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError("Journal elapsed_seconds must be finite and non-negative.")
        unit = self._unit(kind, keys)
        path = self._path(unit)
        if path.exists():
            raise RuntimeError(f"Journal unit was already recorded: {unit!r}")
        canonical_payload = self._canonical(payload)
        record: dict[str, Any] = {
            "journal_version": JOURNAL_VERSION,
            "fingerprint": self.fingerprint,
            "unit": unit,
            "elapsed_seconds": float(elapsed_seconds),
            "payload": canonical_payload,
        }
        record["record_digest"] = self._digest(record)
        atomic_json(path, record)
        self._read_record(unit)
        if unit not in self._executed_units:
            self._executed_units.append(unit)
        return self._canonical(canonical_payload)

    def run(self, kind: str, producer: Callable[[], object], **keys: object) -> Any:
        if self.is_done(kind, **keys):
            return self.load(kind, **keys)
        started = time.perf_counter()
        payload = producer()
        return self.record(kind, payload, elapsed_seconds=time.perf_counter() - started, **keys)

    def replay(self, kind: str | None = None) -> tuple[Any, ...]:
        records: list[tuple[str, Any]] = []
        journal_dir = self.workspace.root / "journal"
        if not journal_dir.exists():
            return ()
        paths = sorted(journal_dir.iterdir())
        if any(not path.is_file() or not path.match("unit=*.json") for path in paths):
            raise RuntimeError("Journal directory contains unexpected entries.")
        for path in paths:
            raw = read_json(path)
            unit = raw.get("unit")
            if not isinstance(unit, Mapping):
                raise TypeError(f"Invalid journal unit record: {path.name}")
            record = self._read_record(unit, source_path=path)
            if kind is None or unit.get("kind") == kind:
                records.append((self._digest(unit), self._canonical(record["payload"])))
        return tuple(payload for _, payload in sorted(records))

    def audit(self) -> dict[str, Any]:
        completed: list[dict[str, Any]] = []
        journal_dir = self.workspace.root / "journal"
        if journal_dir.exists():
            paths = sorted(journal_dir.iterdir())
            if any(not path.is_file() or not path.match("unit=*.json") for path in paths):
                raise RuntimeError("Journal directory contains unexpected entries.")
            for path in paths:
                raw = read_json(path)
                unit = raw.get("unit")
                if not isinstance(unit, Mapping):
                    raise TypeError(f"Invalid journal unit record: {path.name}")
                record = self._read_record(unit, source_path=path)
                completed.append(
                    {"unit": unit, "elapsed_seconds": record["elapsed_seconds"], "record_digest": raw["record_digest"]}
                )
        return {
            "journal_version": JOURNAL_VERSION,
            "fingerprint": self.fingerprint,
            "completed_units": completed,
            "executed_units": list(self._executed_units),
            "resumed_units": list(self._resumed_units),
            "executed_unit_count": len(self._executed_units),
            "resumed_unit_count": len(self._resumed_units),
        }


class RuntimeRecord:
    """Monotonic runtime observations shared by budget checks and leaf reports."""

    def __init__(self, *, started: float, time_budget: TimeBudget | None, persist_path: Path | None = None) -> None:
        self.started = started
        self.time_budget = time_budget
        self.persist_path = persist_path
        self._observations: list[dict[str, Any]] = []
        self._budget_checks: list[dict[str, Any]] = []
        self.minimum_remaining_margin_seconds: float | None = None
        self.abort_reason: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def _persist(self) -> None:
        if self.persist_path is not None:
            atomic_json(self.persist_path, self.snapshot())

    def observe(
        self,
        kind: str,
        *,
        seconds: float,
        measures: Mapping[str, int | float] | None = None,
        tags: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        if not kind or not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("Runtime observations require a kind and finite non-negative seconds.")
        numeric = dict(measures or {})
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in numeric.values()
        ):
            raise TypeError("Runtime measures must be finite numbers, not bool values.")
        self._observations.append(
            {"kind": kind, "seconds": float(seconds), "measures": numeric, "tags": dict(tags or {})}
        )
        self._persist()

    def select(self, kind: str | None = None, **tags: object) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "kind": row["kind"],
                "seconds": row["seconds"],
                "measures": dict(row["measures"]),
                "tags": dict(row["tags"]),
            }
            for row in self._observations
            if (kind is None or row["kind"] == kind)
            and all(row["tags"].get(name) == value for name, value in tags.items())
        )

    @staticmethod
    def _value(row: Mapping[str, Any], name: str) -> float:
        if name == "seconds":
            return float(row["seconds"])
        return float(row["measures"][name])

    def sum(self, measure: str = "seconds", *, kind: str | None = None, **tags: object) -> float:
        return sum(self._value(row, measure) for row in self.select(kind, **tags))

    def maximum_rate(self, numerator: str, denominator: str, *, kind: str | None = None, **tags: object) -> float:
        rates = [
            self._value(row, numerator) / self._value(row, denominator)
            for row in self.select(kind, **tags)
            if self._value(row, denominator) > 0.0
        ]
        return max(rates, default=0.0)

    def record_budget_check(
        self, *, projected_seconds: float, remaining_margin_seconds: float, abort_reason: str | None
    ) -> None:
        self.minimum_remaining_margin_seconds = min(
            remaining_margin_seconds,
            self.minimum_remaining_margin_seconds
            if self.minimum_remaining_margin_seconds is not None
            else remaining_margin_seconds,
        )
        self.abort_reason = abort_reason or self.abort_reason
        self._budget_checks.append(
            {
                "elapsed_seconds": self.elapsed_seconds,
                "projected_seconds": projected_seconds,
                "remaining_margin_seconds": remaining_margin_seconds,
                "abort_reason": abort_reason,
            }
        )
        self._persist()

    def note_abort(self, reason: str) -> None:
        self.abort_reason = self.abort_reason or reason
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        budget = None
        if self.time_budget is not None:
            budget = {
                "hard_limit_seconds": self.time_budget.hard_limit,
                "graceful_abort_seconds": self.time_budget.graceful_abort,
                "writing_reserve_seconds": self.time_budget.writing_reserve,
            }
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "time_budget": budget,
            "observation_count": len(self._observations),
            "observations": list(self._observations),
            "budget_check_count": len(self._budget_checks),
            "budget_checks": list(self._budget_checks),
            "minimum_remaining_margin_seconds": self.minimum_remaining_margin_seconds,
            "abort_reason": self.abort_reason,
        }


def _dotted_value(payload: object, dotted_path: str) -> object:
    current = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _stdout_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


class StepRun:
    """Own input immutability, checks, output emission and final manifest commit."""

    def __init__(
        self,
        signature: str,
        output_dir: Path,
        artifacts: Sequence[Artifact],
        *,
        manifest_name: str,
        inputs: Mapping[str, Path],
        expectations: Mapping[str, Mapping[str, Any]] | None = None,
        expected_input_digests: Mapping[str, str] | None = None,
        time_budget: TimeBudget | None = None,
        enforce_exact_output_set: bool = True,
        workspace: Workspace | None = None,
        restart_workspace: bool = False,
        resume_contract: Mapping[str, Any] | None = None,
        on_existing: ExistingOutputPolicy = "replace",
    ) -> None:
        self.signature = signature
        self.output_dir = Path(output_dir)
        self.artifacts = tuple(artifacts)
        self.manifest_name = manifest_name
        self.inputs = {name: Path(path) for name, path in inputs.items()}
        self.time_budget = time_budget
        self.workspace = workspace
        self.resume_contract = dict(resume_contract or {})
        self.on_existing = on_existing
        self.enforce_exact_output_set = enforce_exact_output_set
        self.started = time.perf_counter()
        self.runtime = RuntimeRecord(started=self.started, time_budget=time_budget)
        self._by_name = {artifact.name: artifact for artifact in self.artifacts}
        names = [artifact.name for artifact in self.artifacts]
        if not signature:
            raise ValueError("Step signature must be non-empty.")
        if len(names) != len(set(names)):
            raise ValueError("Artifact names must be unique.")
        if manifest_name in self._by_name:
            raise ValueError("The manifest is managed by StepRun and must not be an Artifact.")
        terminals = [artifact.name for artifact in self.artifacts if artifact.terminal]
        if len(terminals) != 1:
            raise ValueError("Exactly one terminal artifact is required.")
        if self._by_name[terminals[0]].kind != "json":
            raise ValueError("The terminal artifact must be JSON.")
        if set(expected_input_digests or {}) - set(self.inputs):
            raise ValueError("Expected digests reference unknown inputs.")
        if set(expectations or {}) - set(self.inputs):
            raise ValueError("Expectations reference unknown inputs.")
        if on_existing not in ("replace", "refuse", "resume"):
            raise ValueError(f"Unknown existing-output policy {on_existing!r}.")
        if resume_contract is not None and workspace is None:
            raise ValueError("A resume contract requires a workspace.")
        if on_existing == "resume" and resume_contract is None:
            raise ValueError("The resume output policy requires a resume contract.")

        self.terminal_name = terminals[0]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._emitted: list[str] = []
        self._selected_variants: dict[str, str] = {}
        self._freeze_ledger: list[dict[str, Any]] = []
        self._frozen_names: set[str] = set()
        self._terminal_payload: dict[str, Any] | None = None
        self._checks: dict[str, bool] = {}
        self._counts: dict[str, dict[str, Any]] = {}
        self._requirements: dict[str, bool] = {}
        self._finalized = False
        self._workspace_finished = False
        self.resume_fingerprint: str | None = None
        self.workspace_resumed = False
        self.journal: Journal | None = None

        planned_set = set(self.planned_names)
        existing_entries = {path.name for path in self.output_dir.iterdir()}
        existing_planned = existing_entries & planned_set
        unexpected_existing = existing_entries - planned_set
        if on_existing == "refuse" and existing_entries:
            raise FileExistsError(f"Output directory is not empty: {sorted(existing_entries)}")
        if on_existing == "replace" and unexpected_existing and enforce_exact_output_set:
            raise RuntimeError(f"Replace output directory has unexpected entries: {sorted(unexpected_existing)}")
        if (
            on_existing == "replace"
            and resume_contract is not None
            and workspace is not None
            and workspace.root.exists()
            and not restart_workspace
        ):
            raise RuntimeError("Replacing outputs for a resumable run requires restarting its workspace.")
        if on_existing == "resume":
            if unexpected_existing:
                raise RuntimeError(f"Resume output directory has unexpected entries: {sorted(unexpected_existing)}")
            if existing_planned and (workspace is None or not workspace.root.exists()):
                raise FileExistsError(
                    f"Planned outputs exist without a resumable workspace: {sorted(existing_planned)}"
                )
            if existing_planned and restart_workspace:
                raise RuntimeError("Restarting a workspace with existing outputs requires the replace policy.")

        expected_digests = dict(expected_input_digests or {})
        self._input_ledger: dict[str, dict[str, Any]] = {}
        for name, path in self.inputs.items():
            if not path.is_file():
                raise FileNotFoundError(f"Input {name!r} is not a file: {path}")
            digest = sha256_file(path)
            expected = expected_digests.get(name)
            matches_expected = expected is None or digest == expected
            self._input_ledger[name] = {
                "path": str(path.resolve()),
                "sha256_before": digest,
                "sha256_after": None,
                "unchanged": None,
                "expected_sha256": expected,
                "expected_sha256_pass": matches_expected,
            }
            if not matches_expected:
                raise RuntimeError(f"Input digest expectation failed for {name!r}.")

        self._expectation_ledger: list[dict[str, Any]] = []
        for input_name, expected_values in (expectations or {}).items():
            payload = read_json(self.inputs[input_name])
            for dotted_path, expected in expected_values.items():
                try:
                    actual = _dotted_value(payload, dotted_path)
                    passed = actual == expected
                    error = ""
                except KeyError:
                    actual = None
                    passed = False
                    error = "missing dotted path"
                self._expectation_ledger.append(
                    {
                        "input": input_name,
                        "dotted_path": dotted_path,
                        "expected": expected,
                        "actual": actual,
                        "pass": passed,
                        "error": error,
                    }
                )
                if not passed:
                    raise RuntimeError(f"Input expectation failed for {input_name!r}: {dotted_path}.")

        if resume_contract is not None:
            fingerprint_payload = {
                "input_sha256": {name: record["sha256_before"] for name, record in self._input_ledger.items()},
                "compute_contract": json_safe(self.resume_contract),
            }
            canonical = json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            self.resume_fingerprint = hashlib.sha256(canonical).hexdigest()

        if self.workspace is not None:
            self.workspace_resumed = self.workspace.prepare(
                self.signature, restart=restart_workspace, resume_fingerprint=self.resume_fingerprint
            )
            if self.resume_fingerprint is not None:
                self.journal = Journal(self.workspace, self.resume_fingerprint)
            self.runtime.persist_path = self.workspace.slot("runtime", name="runtime-record", suffix=".json")
            self.runtime._persist()
        if on_existing == "replace" and existing_planned:
            for name in sorted(existing_planned):
                path = self.output_dir / name
                if not path.is_file():
                    raise RuntimeError(f"Refusing to replace non-file output entry: {path}")
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                path.unlink()
        if on_existing == "resume" and existing_planned:
            if not self.workspace_resumed:
                raise RuntimeError("Partial outputs require a compatible journal workspace.")
            for name in sorted(existing_planned):
                path = self.output_dir / name
                if not path.is_file():
                    raise RuntimeError(f"Refusing to replace non-file output entry: {path}")
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                path.unlink()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc is not None:
            self.runtime.note_abort(f"exception:{type(exc).__name__}")
        self._finish_workspace(success=exc is None and self._finalized)
        return False

    def _finish_workspace(self, *, success: bool) -> None:
        if self.workspace is not None and not self._workspace_finished:
            self.workspace.finish(self.signature, success=success, resume_fingerprint=self.resume_fingerprint)
            if success or self.workspace.lifecycle == "delete_always":
                self._workspace_finished = True

    @property
    def planned_names(self) -> list[str]:
        return [artifact.name for artifact in self.artifacts] + [self.manifest_name]

    @property
    def report_completeness_pass(self) -> bool:
        return bool(
            all(self._requirements.values())
            and all(self._checks.values())
            and all(record["pass"] for record in self._counts.values())
        )

    @property
    def checks(self) -> dict[str, bool]:
        return dict(self._checks)

    @property
    def counts(self) -> dict[str, dict[str, Any]]:
        return {name: dict(record) for name, record in self._counts.items()}

    @property
    def input_digests(self) -> dict[str, str]:
        return {name: str(record["sha256_before"]) for name, record in self._input_ledger.items()}

    def path(self, artifact_name: str) -> Path:
        if artifact_name not in self._by_name and artifact_name != self.manifest_name:
            raise KeyError(f"Unknown artifact {artifact_name!r}.")
        return self.output_dir / artifact_name

    def require(self, name: str, condition: bool, message: str = "") -> bool:
        if type(condition) is not bool:
            raise TypeError("StepRun.require accepts bool values only.")
        if name in self._requirements:
            raise ValueError(f"Requirement {name!r} was already recorded.")
        self._requirements[name] = condition
        if not condition:
            raise RuntimeError(message or f"Required precondition {name!r} failed.")
        return condition

    def check(self, name: str, value: bool) -> bool:
        if type(value) is not bool:
            raise TypeError("StepRun.check accepts bool values only.")
        if name in self._checks:
            raise ValueError(f"Check {name!r} was already recorded.")
        self._checks[name] = value
        return value

    def count(self, name: str, value: int, *, expected: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("StepRun.count accepts int values only.")
        if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int)):
            raise TypeError("StepRun.count expected value must be int or None.")
        if name in self._counts:
            raise ValueError(f"Count {name!r} was already recorded.")
        self._counts[name] = {"value": value, "expected": expected, "pass": expected is None or value == expected}
        return value

    def _write(self, artifact: Artifact, payload: Any, *, variant: str | None = None) -> None:
        path = self.path(artifact.name)
        specification = artifact.specification(variant)
        if specification.kind == "json":
            atomic_json(path, payload)
        elif specification.kind == "csv":
            atomic_csv(path, payload)
        elif specification.kind == "csv_union":
            atomic_csv_union(path, payload)
        elif specification.kind == "npz":
            atomic_npz(path, payload)
        elif specification.kind == "pt":
            if not isinstance(payload, Mapping):
                raise TypeError("Torch artifacts must be mapping payloads.")
            atomic_torch(path, payload)
        elif specification.kind == "text":
            if not isinstance(payload, str):
                raise TypeError("Text artifacts require a string payload.")
            atomic_text(path, payload)
        else:  # pragma: no cover - Literal makes this defensive only.
            raise ValueError(f"Unsupported artifact kind {specification.kind!r}.")

    def emit(self, artifact_name: str, payload: Any, *, variant: str | None = None) -> None:
        if self._finalized:
            raise RuntimeError("Cannot emit after finalization.")
        if artifact_name not in self._by_name:
            raise KeyError(f"Unknown or harness-managed artifact {artifact_name!r}.")
        if artifact_name in self._emitted or (
            artifact_name == self.terminal_name and self._terminal_payload is not None
        ):
            raise RuntimeError(f"Artifact {artifact_name!r} was emitted more than once.")
        artifact = self._by_name[artifact_name]
        if artifact.terminal:
            if variant is not None:
                raise ValueError("Terminal artifacts do not support variants.")
            if not isinstance(payload, Mapping):
                raise TypeError("Terminal JSON payload must be a mapping.")
            self._terminal_payload = dict(payload)
            self._selected_variants[artifact_name] = "default"
            return
        artifact.specification(variant)
        self._write(artifact, payload, variant=variant)
        self._emitted.append(artifact_name)
        self._selected_variants[artifact_name] = variant or "default"

    def freeze(self, artifact_names: Sequence[str]) -> dict[str, Any]:
        """Freeze already-emitted artifacts and return their measured audit."""
        if self._finalized:
            raise RuntimeError("Cannot freeze artifacts after finalization.")
        names = list(artifact_names)
        if len(names) != len(set(names)):
            raise ValueError("Freeze artifact names must be unique.")
        records: list[dict[str, Any]] = []
        for name in names:
            if name not in self._by_name or name == self.terminal_name:
                raise KeyError(f"Unknown or non-freezable artifact {name!r}.")
            if name not in self._emitted:
                raise RuntimeError(f"Artifact {name!r} must be emitted before freezing.")
            if name in self._frozen_names:
                raise RuntimeError(f"Artifact {name!r} was already frozen.")
            path = self.path(name)
            digest = sha256_file(path)
            os.chmod(path, stat.S_IREAD)
            read_only = is_read_only_file(path)
            if not read_only:
                raise RuntimeError(f"Artifact {name!r} did not become read-only.")
            record = {"artifact": name, "path": str(path.resolve()), "sha256": digest, "read_only": read_only}
            records.append(record)
            self._freeze_ledger.append(record)
            self._frozen_names.add(name)
        return {
            "artifacts": records,
            "frozen_file_sha256": {record["artifact"]: record["sha256"] for record in records},
            "all_frozen_files_read_only": all(record["read_only"] for record in records),
        }

    def tick(self, *, projected_seconds: float = 0.0) -> None:
        if projected_seconds < 0.0:
            raise ValueError("projected_seconds must be non-negative.")
        if self.time_budget is None:
            return
        elapsed = self.runtime.elapsed_seconds
        remaining = self.time_budget.hard_limit - elapsed
        if elapsed >= self.time_budget.graceful_abort:
            reason = "graceful_abort_threshold_reached"
            self.runtime.record_budget_check(
                projected_seconds=projected_seconds, remaining_margin_seconds=remaining, abort_reason=reason
            )
            raise TimeBudgetExceeded("The graceful-abort threshold was reached.")
        if elapsed + projected_seconds + self.time_budget.writing_reserve > self.time_budget.hard_limit:
            reason = "insufficient_budget_for_complete_block"
            self.runtime.record_budget_check(
                projected_seconds=projected_seconds, remaining_margin_seconds=remaining, abort_reason=reason
            )
            raise TimeBudgetExceeded("Insufficient budget for the next complete block.")
        self.runtime.record_budget_check(
            projected_seconds=projected_seconds, remaining_margin_seconds=remaining, abort_reason=None
        )

    def progress(self, **fields: object) -> None:
        rendered = " ".join(f"{key}={_stdout_value(value)}" for key, value in fields.items())
        print(f"progress {rendered}".rstrip(), flush=True)

    def result(self, **fields: object) -> None:
        for key, value in fields.items():
            print(f"{key}={_stdout_value(value)}", flush=True)

    def _finish_input_ledger(self) -> None:
        for name, path in self.inputs.items():
            record = self._input_ledger[name]
            digest = sha256_file(path)
            record["sha256_after"] = digest
            record["unchanged"] = digest == record["sha256_before"]
        changed = [name for name, record in self._input_ledger.items() if not record["unchanged"]]
        if changed:
            raise RuntimeError(f"Inputs changed during the step: {changed}")

    def verify_inputs_unchanged(self) -> bool:
        """Refresh the end digests for a pre-summary audit without weakening finalize."""
        for name, path in self.inputs.items():
            digest = sha256_file(path)
            record = self._input_ledger[name]
            record["sha256_after"] = digest
            record["unchanged"] = digest == record["sha256_before"]
        return all(bool(record["unchanged"]) for record in self._input_ledger.values())

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("StepRun.finalize may only be called once.")
        if self._terminal_payload is None:
            raise RuntimeError(f"Terminal artifact {self.terminal_name!r} was not emitted.")
        expected_nonterminal = [artifact.name for artifact in self.artifacts if not artifact.terminal]
        if set(self._emitted) != set(expected_nonterminal) or len(self._emitted) != len(expected_nonterminal):
            missing = [name for name in expected_nonterminal if name not in self._emitted]
            raise RuntimeError(f"Non-terminal artifact emission order/set failed; missing={missing}.")
        if not self.report_completeness_pass:
            failed_checks = [name for name, passed in self._checks.items() if not passed]
            failed_counts = [name for name, record in self._counts.items() if not record["pass"]]
            raise RuntimeError(f"Step completeness failed: checks={failed_checks}, counts={failed_counts}.")
        self._finish_input_ledger()

        for record in self._freeze_ledger:
            path = self.path(str(record["artifact"]))
            if sha256_file(path) != record["sha256"] or not is_read_only_file(path):
                raise RuntimeError(f"Frozen artifact verification failed: {record['artifact']}.")

        nonterminal_checks: list[dict[str, Any]] = []
        for name in expected_nonterminal:
            variant_name = self._selected_variants.get(name, "default")
            variant = None if variant_name == "default" else variant_name
            specification = self._by_name[name].specification(variant)
            check = verify_artifact(self.path(name), set(specification.required_keys), kind=specification.kind)
            check["artifact_variant"] = variant_name
            check["required_keys"] = sorted(specification.required_keys)
            nonterminal_checks.append(check)
        if not all(check["verified"] for check in nonterminal_checks):
            raise RuntimeError("At least one non-terminal artifact failed verification.")

        planned = self.planned_names
        terminal_payload = dict(self._terminal_payload)
        if isinstance(terminal_payload.get("checks"), Mapping):
            terminal_checks = dict(terminal_payload["checks"])
            terminal_checks.update({"artifact_set_equality_pass": True, "planned_equals_written_equals_verified": True})
            terminal_payload["checks"] = terminal_checks
        terminal_payload.update(
            {
                "planned_artifacts": planned,
                "written_artifacts": planned,
                "verified_artifacts": planned,
                "planned_equals_written_equals_verified": True,
                "artifact_set_equality_pass": True,
                "report_completeness_pass": self.report_completeness_pass,
            }
        )
        terminal_artifact = self._by_name[self.terminal_name]
        self._write(terminal_artifact, terminal_payload)
        self._emitted.append(self.terminal_name)
        terminal_check = verify_artifact(
            self.path(self.terminal_name), set(terminal_artifact.required_keys), kind=terminal_artifact.kind
        )
        terminal_check["artifact_variant"] = "default"
        terminal_check["required_keys"] = sorted(terminal_artifact.required_keys)
        if not terminal_check["verified"]:
            raise RuntimeError("The terminal artifact failed verification.")

        planned_set = set(planned)
        entries_before_manifest = {path.name for path in self.output_dir.iterdir()}
        allowed_before_manifest = planned_set - {self.manifest_name}
        exact_before = entries_before_manifest in (allowed_before_manifest, planned_set)
        if self.enforce_exact_output_set and not exact_before:
            raise RuntimeError(
                "Artifact directory has unexpected entries before manifest commit: "
                f"{sorted(entries_before_manifest - planned_set)}"
            )

        artifact_checks = [*nonterminal_checks, terminal_check]
        written = [*expected_nonterminal, self.terminal_name, self.manifest_name]
        verified = [*[Path(check["path"]).name for check in artifact_checks], self.manifest_name]
        parity = set(planned) == set(written) == set(verified)
        manifest: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "signature": self.signature,
            "planned_artifacts": planned,
            "written_artifacts": written,
            "verified_artifacts": verified,
            "planned_equals_written_equals_verified": parity,
            "artifact_set_equality_pass": parity,
            "output_directory_exact_set_pass": exact_before,
            "artifact_checks": artifact_checks,
            "input_ledger": self._input_ledger,
            "expectation_ledger": self._expectation_ledger,
            "requirements": self._requirements,
            "checks": self._checks,
            "counts": self._counts,
            "artifact_variants": dict(self._selected_variants),
            "freeze_ledger": list(self._freeze_ledger),
            "resume_contract": json_safe(self.resume_contract) if self.resume_contract else None,
            "resume_fingerprint": self.resume_fingerprint,
            "workspace_resumed": self.workspace_resumed,
            "journal_ledger": self.journal.audit() if self.journal is not None else None,
            "report_completeness_pass": self.report_completeness_pass,
            "self_digest_scheme": (
                "sha256 of canonical JSON payload with self_digest omitted; "
                "artifact_checks bind artifact sha256 digests"
            ),
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        manifest["self_digest"] = hashlib.sha256(canonical).hexdigest()
        atomic_json(self.path(self.manifest_name), manifest)
        verify_artifact_manifest(self.path(self.manifest_name), verify_files=True)

        final_entries = {path.name for path in self.output_dir.iterdir()}
        final_exact = final_entries == planned_set
        if self.enforce_exact_output_set and not final_exact:
            raise RuntimeError("Final artifact directory does not equal the planned set.")
        if not parity:
            raise RuntimeError("Planned, written and verified artifact sets differ.")
        self._finalized = True
        return manifest
