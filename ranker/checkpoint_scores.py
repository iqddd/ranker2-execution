"""Reusable immutable-journal checkpoint score loading and E3 utilities."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ranker.external_validation import percentile_against
from ranker.io import sha256_file

ScorePanel = dict[tuple[str, str, int], dict[int, np.ndarray]]


def score_array_hash(scores: np.ndarray) -> str:
    """Return the canonical digest used by retained raw-score manifests."""
    values = np.asarray(scores, dtype=np.float64).copy(order="C")
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def load_retained_state_manifest(
    source: Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Read and minimally validate a retained-state manifest once."""
    if isinstance(source, Path):
        manifest = json.loads(source.read_text(encoding="utf-8"))
    else:
        manifest = dict(source)
    states = manifest.get("states")
    if not isinstance(states, list):
        raise TypeError("Retained-state manifest must contain a states list.")
    return manifest


def retained_state_records(
    source: Path | Mapping[str, Any],
    *,
    checkpoint_map: Mapping[str, int | Sequence[int]] | None = None,
    identities: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Select manifest-declared retained states without reconstructing filenames."""
    manifest = load_retained_state_manifest(source)
    allowed_identities = None if identities is None else {str(value) for value in identities}
    allowed_seeds = None if seeds is None else {int(value) for value in seeds}
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    required = {
        "identity_id",
        "seed",
        "checkpoint",
        "path",
        "model_state_SHA256",
        "score_panel_SHA256",
    }
    for item in manifest["states"]:
        if not isinstance(item, Mapping) or not required <= set(item):
            raise ValueError("Retained-state record is missing required manifest fields.")
        identity = str(item["identity_id"])
        seed = int(item["seed"])
        checkpoint = int(item["checkpoint"])
        if allowed_identities is not None and identity not in allowed_identities:
            continue
        if allowed_seeds is not None and seed not in allowed_seeds:
            continue
        if checkpoint_map is not None:
            selected_checkpoints = checkpoint_map.get(identity)
            if selected_checkpoints is None:
                continue
            selected = (
                {int(selected_checkpoints)}
                if isinstance(selected_checkpoints, int)
                else {int(value) for value in selected_checkpoints}
            )
            if checkpoint not in selected:
                continue
        key = (identity, checkpoint, seed)
        if key in seen:
            raise RuntimeError(f"Duplicate retained-state record: {key}")
        seen.add(key)
        records.append(dict(item))
    records.sort(
        key=lambda row: (
            str(row["identity_id"]).encode("utf-8"),
            int(row["checkpoint"]),
            int(row["seed"]),
        )
    )
    if checkpoint_map is not None:
        expected = {
            (str(identity), int(checkpoint), int(seed))
            for identity, checkpoints in checkpoint_map.items()
            for checkpoint in (
                (checkpoints,) if isinstance(checkpoints, int) else checkpoints
            )
            for seed in (allowed_seeds if allowed_seeds is not None else {int(row["seed"]) for row in records})
        }
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise RuntimeError(
                f"Retained-state selection mismatch; missing={missing}, extra={extra}"
            )
    return records


def retained_state_paths(
    source: Path | Mapping[str, Any],
    *,
    checkpoint_map: Mapping[str, int | Sequence[int]],
    seeds: Sequence[int],
) -> dict[tuple[str, int, int], Path]:
    """Return exact manifest-declared state paths for an identity/checkpoint policy."""
    records = retained_state_records(
        source, checkpoint_map=checkpoint_map, seeds=seeds
    )
    paths = {
        (str(row["identity_id"]), int(row["checkpoint"]), int(row["seed"])): Path(
            row["path"]
        )
        for row in records
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def load_retained_score_panel(
    source: Path | Mapping[str, Any],
    *,
    system: str,
    checkpoint_map: Mapping[str, int] | None = None,
    expected_score_count: int = 710,
    retained_score_count: int | None = None,
) -> tuple[ScorePanel, list[dict[str, Any]]]:
    """Load and verify raw scores from manifest-declared retained states.

    The returned panel retains the manifest checkpoint dimension; callers that
    need one selected checkpoint per identity provide ``checkpoint_map``.
    """
    records = retained_state_records(source, checkpoint_map=checkpoint_map)
    panel: ScorePanel = {}
    audit: list[dict[str, Any]] = []
    for row in records:
        path = Path(row["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scores = np.asarray(payload["raw_scores"], dtype=np.float64)
        identity = str(row["identity_id"])
        seed = int(row["seed"])
        checkpoint = int(row["checkpoint"])
        metadata_pass = (
            str(payload["identity_id"]) == identity
            and int(payload["seed"]) == seed
            and int(payload["checkpoint"]) == checkpoint
            and scores.shape == (expected_score_count,)
        )
        file_pass = sha256_file(path) == row["model_state_SHA256"]
        score_pass = score_array_hash(scores) == row["score_panel_SHA256"]
        key = (system, identity, seed)
        trajectory = panel.setdefault(key, {})
        if checkpoint in trajectory:
            raise RuntimeError(f"Duplicate retained score panel: {key} checkpoint {checkpoint}")
        trajectory[checkpoint] = (
            scores.copy()
            if retained_score_count is None
            else scores[:retained_score_count].copy()
        )
        audit.append(
            {
                "record_type": "state",
                "identity_id": identity,
                "seed": seed,
                "checkpoint": checkpoint,
                "file_SHA256_pass": file_pass,
                "score_SHA256_pass": score_pass,
                "metadata_pass": metadata_pass,
                "pass": bool(file_pass and score_pass and metadata_pass),
            }
        )
    return panel, audit


def load_checkpoint_score_panel(
    journal_dir: Path,
    *,
    unit_kind: str,
    score_field: str,
    system_key: str | None,
    fixed_system: str | None = None,
) -> ScorePanel:
    """Load checkpoint score arrays from closed immutable journal payloads."""
    panel: ScorePanel = {}
    for source in sorted(journal_dir.glob("*.json")):
        record = json.loads(source.read_text(encoding="utf-8"))
        if record["unit"]["kind"] != unit_kind:
            continue
        keys = record["unit"]["keys"]
        system = fixed_system if system_key is None else str(keys[system_key])
        if system is None:
            raise ValueError("fixed_system is required when system_key is None")
        key = (system, str(keys["identity"]), int(keys["seed"]))
        if key in panel:
            raise RuntimeError(f"duplicate score-panel trajectory: {key}")
        panel[key] = {
            int(checkpoint): np.asarray(scores, dtype=np.float64)
            for checkpoint, scores in record["payload"][score_field].items()
        }
    return panel


def validate_panel_grid(
    panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    *,
    systems: Sequence[str],
    identities: Sequence[str],
    seeds: Sequence[int],
    checkpoints: Sequence[int],
    score_count: int,
) -> None:
    """Require an exact trajectory/checkpoint grid and finite score arrays."""
    expected_keys = {
        (system, identity, seed)
        for system in systems
        for identity in identities
        for seed in seeds
    }
    if set(panel) != expected_keys:
        missing = sorted(expected_keys - set(panel))
        extra = sorted(set(panel) - expected_keys)
        raise RuntimeError(f"score-panel key mismatch; missing={missing}, extra={extra}")
    expected_checkpoints = set(checkpoints)
    for key, trajectory in panel.items():
        available = set(trajectory)
        if not expected_checkpoints <= available:
            raise RuntimeError(
                f"checkpoint grid incomplete for {key}: "
                f"{sorted(expected_checkpoints - available)}"
            )
        for checkpoint in checkpoints:
            score = np.asarray(trajectory[checkpoint], dtype=np.float64)
            if score.shape != (score_count,) or not np.isfinite(score).all():
                raise RuntimeError(
                    f"invalid score vector for {key} checkpoint {checkpoint}: "
                    f"shape={score.shape}"
                )


def fold_percentile_score_map(
    panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    *,
    system: str,
    identity: str,
    checkpoint: int,
    seed_indices: Sequence[int],
    seeds: Sequence[int],
    image_keys: np.ndarray,
    identity_ids: np.ndarray,
    reference_population_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Build a historical seed-percentile ensemble map for one held-out fold."""
    reference_mask = np.asarray(identity_ids) != identity
    if reference_population_mask is not None:
        population = np.asarray(reference_population_mask, dtype=bool)
        if population.shape != reference_mask.shape:
            raise ValueError("reference population mask shape mismatch")
        reference_mask &= population
    percentiles = []
    for index in seed_indices:
        score = np.asarray(
            panel[(system, identity, int(seeds[index]))][checkpoint],
            dtype=np.float64,
        )
        percentiles.append(percentile_against(score, score[reference_mask]))
    ensemble = np.mean(np.stack(percentiles), axis=0, dtype=np.float64)
    return {
        str(key): float(value)
        for key, value in zip(image_keys, ensemble, strict=True)
    }


def exact_seed_identity_distribution(
    per_seed_triple_identity: np.ndarray,
) -> np.ndarray:
    """Apply exact 4^4 paired identity resampling to 3^3 seed triples."""
    values = np.asarray(per_seed_triple_identity, dtype=np.float64)
    if values.shape != (27, 4):
        raise ValueError(f"expected (27, 4), got {values.shape}")
    identity_samples = np.asarray(
        list(itertools.product(range(4), repeat=4)), dtype=np.int64
    )
    return values[:, identity_samples].mean(axis=2).reshape(-1)
