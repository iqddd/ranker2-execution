"""Reusable frozen-score and feature panels for external-validation workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ranker.frozen_readouts import midrank_percentile
from ranker.s2ow import load_historical_s2_features, load_step26_s2_feature_cache


def percentile_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return empirical midrank percentiles against an explicit reference cohort."""
    return midrank_percentile(
        np.asarray(values, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )


def load_journal_score_payloads(
    path: Path, systems: set[str]
) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Load selected system/seed/fold payloads from an immutable journal."""
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for source in path.glob("*.json"):
        record = json.loads(source.read_text(encoding="utf-8"))
        unit = record["unit"]
        keys = unit["keys"]
        system = str(keys.get("system", ""))
        if unit["kind"] != "trajectory" or system not in systems:
            continue
        key = (system, int(keys["seed"]), str(keys["identity"]))
        result[key] = dict(record["payload"])
    return result


def load_cv_score_panel(
    *,
    tune_identities: Sequence[str],
    seeds: Sequence[int],
    step47a_dir: Path,
    step47b_dir: Path,
    step47c_workspace: Path,
) -> tuple[dict[tuple[str, int, str], np.ndarray], list[Path]]:
    """Load the frozen P3 Prodigy/Adam three-seed CV score panel."""
    if len(seeds) != 3:
        raise ValueError("The historical CV panel requires exactly three seeds.")
    scores: dict[tuple[str, int, str], np.ndarray] = {}
    inputs: list[Path] = []
    for fold, identity in enumerate(tune_identities):
        for system, root, pattern in (
            (
                "P3_PRODIGY",
                step47a_dir,
                f"STEP47A_SELECTED_STATE__system=P3_JOINT4_FROZEN__fold={fold}.pt",
            ),
            (
                "P3_ADAM",
                step47b_dir,
                f"STEP47B_SELECTED_STATE__system=P3_ADAM_MATCHED__fold={fold}.pt",
            ),
        ):
            source = root / pattern
            payload = torch.load(source, map_location="cpu", weights_only=False)
            scores[(system, int(seeds[0]), identity)] = np.asarray(
                payload["scores"], dtype=np.float64
            )
            inputs.append(source)
    journal_dir = step47c_workspace / "journal"
    journal = load_journal_score_payloads(journal_dir, {"P3_PRODIGY", "P3_ADAM"})
    for system in ("P3_PRODIGY", "P3_ADAM"):
        for seed in seeds[1:]:
            for identity in tune_identities:
                key = (system, int(seed), identity)
                if key not in journal:
                    raise RuntimeError(f"Missing STEP47C CV score payload {key}.")
                scores[key] = np.asarray(journal[key]["scores"], dtype=np.float64)
    inputs.extend(sorted(journal_dir.glob("*.json")))
    if len(scores) != 24 or any(
        value.shape != (520,) or not np.isfinite(value).all()
        for value in scores.values()
    ):
        raise RuntimeError("CV score panel is incomplete or non-finite.")
    return scores, inputs


def make_power_pools(
    *,
    scores: Mapping[tuple[str, int, str], np.ndarray],
    seeds: Sequence[int],
    tune4: Sequence[str],
    rank10: Sequence[str],
    folds: Mapping[str, Any],
    image_ids: np.ndarray,
    identities: np.ndarray,
    human_percentile: np.ndarray,
    baseline_percentile: np.ndarray,
) -> tuple[
    dict[str, list[tuple[float, float]]],
    dict[str, list[tuple[float, float]]],
]:
    """Build known/pseudo-unseen paired error pools from frozen CV predictions."""
    del image_ids  # Retained in the API to make population alignment explicit.
    unseen: dict[str, list[tuple[float, float]]] = {}
    for identity in tune4:
        mask = identities == identity
        seed_percentiles = [
            midrank_percentile(
                scores[("P3_PRODIGY", int(seed), identity)][mask],
                scores[("P3_PRODIGY", int(seed), identity)][
                    folds[identity].reference_mask
                ],
            )
            for seed in seeds
        ]
        challenger = np.mean(np.stack(seed_percentiles), axis=0)
        unseen[identity] = list(
            zip(
                np.abs(baseline_percentile[mask] - human_percentile[mask]),
                np.abs(challenger - human_percentile[mask]),
                strict=True,
            )
        )

    known: dict[str, list[tuple[float, float]]] = {}
    for identity in rank10:
        mask = identities == identity
        fold_percentiles = []
        for heldout in tune4:
            seed_percentiles = [
                midrank_percentile(
                    scores[("P3_PRODIGY", int(seed), heldout)][mask],
                    scores[("P3_PRODIGY", int(seed), heldout)][
                        folds[heldout].reference_mask
                    ],
                )
                for seed in seeds
            ]
            fold_percentiles.append(np.mean(np.stack(seed_percentiles), axis=0))
        challenger = np.mean(np.stack(fold_percentiles), axis=0)
        known[identity] = list(
            zip(
                np.abs(baseline_percentile[mask] - human_percentile[mask]),
                np.abs(challenger - human_percentile[mask]),
                strict=True,
            )
        )
    return known, unseen


def _ib_tail_delta(values: Sequence[tuple[float, float]]) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    count = max(1, int(np.ceil(0.1 * len(matrix))))
    return float(
        np.mean(np.sort(matrix[:, 1])[-count:])
        - np.mean(np.sort(matrix[:, 0])[-count:])
    )


def center_power_pool(
    pool: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, np.ndarray]:
    """Null-center a paired power pool using the historical tail endpoint."""
    base = {
        identity: np.asarray([item[0] for item in rows], dtype=np.float64)
        for identity, rows in pool.items()
    }
    raw = {
        identity: np.asarray([item[1] - item[0] for item in rows], dtype=np.float64)
        for identity, rows in pool.items()
    }

    def objective(offset: float) -> float:
        deltas = []
        for identity in sorted(pool):
            challenger = np.maximum(0.0, base[identity] + raw[identity] + offset)
            deltas.append(
                _ib_tail_delta(list(zip(base[identity], challenger, strict=True)))
            )
        return float(np.mean(deltas))

    low, high = -100.0, 100.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if objective(middle) > 0.0:
            high = middle
        else:
            low = middle
    offset = (low + high) / 2.0
    return {
        identity: np.stack([base[identity], raw[identity] + offset], axis=1)
        for identity in pool
    }


def load_raw_score_panels(
    *,
    old_scores: Path,
    new_scores: Path,
    challenger_dir: Path,
    image_ids: np.ndarray,
    seed_suffixes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    """Load the frozen 710-image baseline and challenger raw-score panels."""
    old = np.load(old_scores, allow_pickle=False)
    new = np.load(new_scores, allow_pickle=False)
    if not np.array_equal(old["image_ids"].astype(str), image_ids.astype(str)):
        raise RuntimeError("Historical S2 score order differs from theta520.")
    baseline_ids = np.concatenate(
        [old["image_ids"].astype(str), new["image_id"].astype(str)]
    )
    baseline_raw = np.concatenate(
        [old["S2_OW_score"].astype(np.float64), new["S2_OW_score"].astype(np.float64)]
    )
    challenger_raw: list[np.ndarray] = []
    sources = [old_scores, new_scores]
    for suffix in seed_suffixes:
        source = challenger_dir / f"STEP48A_FULLDATA_SEED{suffix}.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        ids = np.asarray(payload["reference_image_ids"], dtype=str)
        raw = np.asarray(payload["reference_raw_scores"], dtype=np.float64)
        if (
            not np.array_equal(ids, baseline_ids)
            or raw.shape != (710,)
            or not np.isfinite(raw).all()
        ):
            raise RuntimeError(f"Invalid challenger score panel {source.name}.")
        challenger_raw.append(raw)
        sources.append(source)
    if (
        baseline_ids.shape != (710,)
        or len(set(baseline_ids.tolist())) != 710
        or not np.isfinite(baseline_raw).all()
    ):
        raise RuntimeError("Invalid S2_OW 710-image raw-score panel.")
    return baseline_ids, baseline_raw, np.stack(challenger_raw), sources


def load_s2_feature_population(
    *, old_pooled: Path, new_pooled: Path, all_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Load the exact historical 520+190 pooled S2 feature population."""
    old_names = list(all_names[:520])
    new_names = set(all_names[520:])
    _, old, _ = load_historical_s2_features(old_pooled, expected_names=old_names)
    new = load_step26_s2_feature_cache(new_pooled, expected_names=sorted(new_names))
    result = {**old, **new}
    if set(result) != set(all_names):
        raise RuntimeError("S2 pooled-feature population is not exact 520+190.")
    return result
