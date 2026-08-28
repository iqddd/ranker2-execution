"""Semantic evidence sampling for the append-only project ledger."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ranker.io import read_json, sha256_file
from ranker.project_ledger import read_project_events


def _json_pointer(payload: Any, pointer: str) -> Any:
    current = payload
    for token in pointer.strip("/").split("/") if pointer else []:
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def semantic_ledger_sample(
    *, seed: int = 20260814, sample_size: int = 5
) -> tuple[list[dict[str, Any]], bool]:
    """Sample active events and verify the cited fragment, not just its locator."""
    events = read_project_events()
    superseded = {
        event_id for event in events for event_id in event.get("supersedes_event_ids", [])
    }
    active = sorted(
        (event for event in events if event["event_id"] not in superseded),
        key=lambda event: event["event_id"].encode("utf-8"),
    )
    if len(active) < sample_size:
        raise RuntimeError("Not enough active ledger events for semantic sampling")
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = [
        active[index]
        for index in sorted(rng.choice(len(active), size=sample_size, replace=False).tolist())
    ]
    rows: list[dict[str, Any]] = []
    for event in selected:
        entry = event.get("evidence", [None])[0]
        locator = entry.get("locator", {}) if entry else {}
        supported = False
        fragment = ""
        explanation = "no evidence entry"
        if entry:
            path = Path(entry["artifact_path"])
            digest_ok = path.is_file() and sha256_file(path) == entry["artifact_sha256"]
            if digest_ok:
                try:
                    if locator.get("kind") == "json_pointer":
                        payload = read_json(path)
                        assertions = locator.get("assertions", {})
                        values = {
                            pointer: _json_pointer(payload, pointer)
                            for pointer in assertions
                        }
                        supported = all(
                            values[pointer] == expected
                            for pointer, expected in assertions.items()
                        )
                        fragment = json.dumps(values, ensure_ascii=False, sort_keys=True)
                        explanation = (
                            "all cited JSON assertions match"
                            if supported
                            else "JSON assertion mismatch"
                        )
                    elif locator.get("kind") == "csv_row":
                        key = locator.get("key", {})
                        matched = [
                            row
                            for row in _csv_rows(path)
                            if all(str(row.get(name)) == str(value) for name, value in key.items())
                        ]
                        columns = locator.get("columns", [])
                        fragment = json.dumps(
                            [{name: row.get(name) for name in columns} for row in matched],
                            ensure_ascii=False,
                        )
                        supported = len(matched) == 1
                        explanation = (
                            "unique cited CSV row resolves"
                            if supported
                            else "CSV row is missing or ambiguous"
                        )
                    else:
                        explanation = "unsupported locator kind"
                except Exception as error:  # evidence diagnostics must be reported
                    explanation = f"locator evaluation failed: {error}"
            else:
                explanation = "evidence file missing or digest mismatch"
        rows.append(
            {
                "event_id": event["event_id"],
                "full_statement": event["statement"],
                "evidence_locator": json.dumps(locator, ensure_ascii=False, sort_keys=True),
                "verbatim_evidence_fragment": fragment,
                "support_verdict": supported,
                "support_explanation": explanation,
            }
        )
    return rows, all(bool(row["support_verdict"]) for row in rows)
