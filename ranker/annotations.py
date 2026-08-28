"""Common CSV annotation parsing."""
from __future__ import annotations
import csv
from pathlib import Path

_LABELS = {"1": 1, "+1": 1, "-1": -1, "0": 0, "?": None}

def normalize_annotation_label(value: str) -> int | None:
    try: return _LABELS[value.strip()]
    except KeyError as error: raise ValueError(f"Invalid annotation label: {value!r}") from error

def load_annotation_csv(path: Path, *, concepts: tuple[str, ...], expected_count: int | None = None) -> tuple[list[str], dict[str, dict[str, int | None]]]:
    expected = ("image_id", *concepts)
    with path.open(encoding="utf-8", newline="") as handle:
        reader=csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected: raise ValueError("Annotation CSV has an unexpected header.")
        rows=list(reader)
    if expected_count is not None and len(rows) != expected_count: raise ValueError(f"Expected {expected_count} annotation rows, got {len(rows)}.")
    values={}
    for row in rows:
        name=row["image_id"]
        if name in values: raise ValueError(f"Duplicate annotation image ID: {name}")
        values[name]={concept: normalize_annotation_label(row[concept]) for concept in concepts}
    return list(values), values
