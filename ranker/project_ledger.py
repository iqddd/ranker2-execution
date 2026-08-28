"""Append-only project decision-ledger primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ranker.io import atomic_json, atomic_text, canonical_json_digest


@dataclass(frozen=True)
class LedgerVerification:
    """Independent read-back verification of all canonical ledger views."""

    event_count: int
    event_hashes_pass: bool
    previous_links_pass: bool
    sequence_pass: bool
    schema_pass: bool
    chain_integrity_pass: bool
    head_parity_pass: bool
    markdown_parity_pass: bool
    all_pass: bool
    head_event_id: str | None
    head_event_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_project_events(path: Path = Path("PROJECT_LEDGER.jsonl")) -> list[dict[str, Any]]:
    """Read canonical project-ledger events in chain order."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_project_ledger_markdown(events: list[dict[str, Any]]) -> str:
    """Render the canonical human-readable view from canonical events."""
    superseded = {
        old for item in events for old in item.get("supersedes_event_ids", [])
    }
    superseded_by = {
        old: item["event_id"]
        for item in events
        for old in item.get("supersedes_event_ids", [])
    }
    markdown = (
        "# Project decision ledger\n\n"
        "Generated from `PROJECT_LEDGER.jsonl`; order and active/superseded "
        "state are canonical.\n\n"
    )
    for item in events:
        is_superseded = item["event_id"] in superseded
        evidence = " | ".join(
            str(entry["artifact_path"]) for entry in item.get("evidence", [])
        )
        markdown += (
            f"## {item['sequence']}. {item['subject']} (`{item['event_id']}`)\n\n"
            f"- active: `{str(not is_superseded).lower()}`\n"
            f"- superseded: `{str(is_superseded).lower()}`\n"
            f"- superseded_by: `{superseded_by.get(item['event_id'], '')}`\n"
            f"- status: `{item['status_before']}` → `{item['status_after']}`\n"
            f"- statement: {item['statement']}\n"
            f"- evidence: `{evidence}`\n"
            f"- event SHA-256: `{item['event_sha256']}`\n\n"
        )
    return markdown


def verify_project_ledger(
    *,
    jsonl_path: Path = Path("PROJECT_LEDGER.jsonl"),
    head_path: Path = Path("PROJECT_LEDGER_HEAD.json"),
    markdown_path: Path = Path("PROJECT_LEDGER.md"),
) -> LedgerVerification:
    """Verify the hash chain, head metadata, and exact generated Markdown."""
    events = read_project_events(jsonl_path)
    event_hashes_pass = True
    previous_links_pass = True
    sequence_pass = True
    schema_pass = True
    previous = "0" * 64
    for sequence, event in enumerate(events, start=1):
        schema_pass &= event.get("schema_version") == "PROJECT_LEDGER_V1"
        sequence_pass &= event.get("sequence") == sequence
        sequence_pass &= event.get("event_id") == f"PLE-{sequence:04d}"
        previous_links_pass &= event.get("previous_event_sha256") == previous
        payload = dict(event)
        observed = payload.pop("event_sha256", None)
        event_hashes_pass &= isinstance(observed, str)
        event_hashes_pass &= canonical_json_digest(payload) == observed
        previous = str(observed)
    chain_integrity_pass = bool(
        events
        and event_hashes_pass
        and previous_links_pass
        and sequence_pass
        and schema_pass
    )
    expected_markdown = render_project_ledger_markdown(events)
    markdown_parity_pass = bool(
        markdown_path.is_file()
        and markdown_path.read_text(encoding="utf-8") == expected_markdown
    )
    head = (
        json.loads(head_path.read_text(encoding="utf-8"))
        if head_path.is_file()
        else {}
    )
    superseded = {
        old for item in events for old in item.get("supersedes_event_ids", [])
    }
    active_count = sum(item.get("event_id") not in superseded for item in events)
    last = events[-1] if events else {}
    expected_head = {
        "schema_version": "PROJECT_LEDGER_V1",
        "event_count": len(events),
        "active_event_count": active_count,
        "superseded_event_count": len(events) - active_count,
        "head_event_id": last.get("event_id"),
        "head_event_sha256": last.get("event_sha256"),
        "chain_integrity_pass": chain_integrity_pass,
        "markdown_parity_pass": markdown_parity_pass,
        "project_ledger_evidence_gate": chain_integrity_pass
        and markdown_parity_pass,
    }
    head_parity_pass = all(head.get(key) == value for key, value in expected_head.items())
    all_pass = chain_integrity_pass and head_parity_pass and markdown_parity_pass
    return LedgerVerification(
        event_count=len(events),
        event_hashes_pass=event_hashes_pass,
        previous_links_pass=previous_links_pass,
        sequence_pass=sequence_pass,
        schema_pass=schema_pass,
        chain_integrity_pass=chain_integrity_pass,
        head_parity_pass=head_parity_pass,
        markdown_parity_pass=markdown_parity_pass,
        all_pass=all_pass,
        head_event_id=last.get("event_id"),
        head_event_sha256=last.get("event_sha256"),
    )


def append_project_event(
    step: str,
    template: Mapping[str, Any],
    *,
    jsonl_path: Path = Path("PROJECT_LEDGER.jsonl"),
    head_path: Path = Path("PROJECT_LEDGER_HEAD.json"),
    markdown_path: Path = Path("PROJECT_LEDGER.md"),
) -> tuple[str, str, bool]:
    """Append one idempotent event and regenerate the two ledger views."""
    events = read_project_events(jsonl_path)
    before = str(events[-1]["event_sha256"])
    existing = next(
        (
            event
            for event in events
            if event.get("step") == step
            and event.get("subject") == template["subject"]
        ),
        None,
    )
    if existing is not None:
        return before, before, False

    sequence = int(events[-1]["sequence"]) + 1
    event = {
        "schema_version": "PROJECT_LEDGER_V1",
        "sequence": sequence,
        "event_id": f"PLE-{sequence:04d}",
        "step": step,
        **dict(template),
        "previous_event_sha256": before,
    }
    event["event_sha256"] = canonical_json_digest(event)
    events.append(event)
    atomic_text(
        jsonl_path,
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in events
        ),
    )

    superseded = {
        old for item in events for old in item.get("supersedes_event_ids", [])
    }
    atomic_text(markdown_path, render_project_ledger_markdown(events))
    active_count = sum(item["event_id"] not in superseded for item in events)
    atomic_json(
        head_path,
        {
            "schema_version": "PROJECT_LEDGER_V1",
            "event_count": len(events),
            "active_event_count": active_count,
            "superseded_event_count": len(events) - active_count,
            "head_event_id": event["event_id"],
            "head_event_sha256": event["event_sha256"],
            "chain_integrity_pass": True,
            "markdown_parity_pass": True,
            "project_ledger_evidence_gate": True,
        },
    )
    return before, str(event["event_sha256"]), True
