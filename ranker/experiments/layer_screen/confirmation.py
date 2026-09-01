"""Pure fixed-layer confirmation analysis for the STEP61 family."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ranker.experiments.image_heldout_horizon_transfer.analysis import (
    fold_percentile_maps,
    strict_endpoint_vectors,
)
from ranker.experiments.image_heldout_horizon_transfer.certification import (
    bootstrap_weight_matrix,
    equal_identity_mean,
    per_target_direct,
)
from ranker.experiments.layer_screen.step61_analysis import selected_fold_maps
from ranker.validation_resolution import empirical_sign_mde

TRIPLES = tuple(itertools.product(range(3), repeat=3))
POINT_TRIPLE_INDEX = TRIPLES.index((0, 1, 2))


def fixed_triple_maps(
    raw_scores: np.ndarray,
    assignment: np.ndarray,
    names: Sequence[str],
    *,
    arm_index: int,
    checkpoint_index: int,
    triple: Sequence[int],
) -> list[dict[str, float]]:
    """Build one fixed arm/checkpoint fold map without a selection path."""
    return selected_fold_maps(
        raw_scores,
        assignment,
        names,
        layer_index=arm_index,
        checkpoint_indices=[checkpoint_index] * 5,
        seed_indices=triple,
    )


def fixed_identity_metrics(
    raw_scores: np.ndarray,
    assignment: np.ndarray,
    names: Sequence[str],
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    identity_order: Sequence[str],
    *,
    arm_index: int,
    checkpoint_index: int,
) -> dict[str, np.ndarray]:
    """Return [27, 14] direct/G/gross vectors for a fixed system."""
    rows = {metric: [] for metric in ("direct", "G", "gross")}
    for triple in TRIPLES:
        maps = fixed_triple_maps(
            raw_scores,
            assignment,
            names,
            arm_index=arm_index,
            checkpoint_index=checkpoint_index,
            triple=triple,
        )
        vectors = strict_endpoint_vectors(exposures, image_fold, maps, identity_order)
        for metric, vector in vectors.items():
            rows[metric].append(vector)
    return {metric: np.stack(values) for metric, values in rows.items()}


def fixed_target_direct(
    raw_scores: np.ndarray,
    assignment: np.ndarray,
    names: Sequence[str],
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    *,
    arm_index: int,
    checkpoint_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return [27, target_images] direct values and immutable target metadata."""
    values: list[np.ndarray] = []
    metadata: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for triple in TRIPLES:
        maps = fixed_triple_maps(
            raw_scores,
            assignment,
            names,
            arm_index=arm_index,
            checkpoint_index=checkpoint_index,
            triple=triple,
        )
        row, identities, images, folds = per_target_direct(exposures, image_fold, maps)
        if metadata is None:
            metadata = identities, images, folds
        else:
            if not (
                np.array_equal(metadata[0], identities)
                and np.array_equal(metadata[1], images)
                and np.array_equal(metadata[2], folds)
            ):
                raise RuntimeError("Target metadata changed across seed triples.")
        values.append(row)
    assert metadata is not None
    return np.stack(values), *metadata


def combined_e6_maps(
    discovery_raw: np.ndarray,
    confirmation_raw: np.ndarray,
    assignment: np.ndarray,
    names: Sequence[str],
    *,
    discovery_arm_index: int,
    discovery_checkpoint_index: int,
    confirmation_arm_index: int,
    confirmation_checkpoint_index: int,
) -> list[dict[str, float]]:
    """Average six fixed per-seed percentiles; seeds remain fixed."""
    raw = np.concatenate(
        (
            discovery_raw[:, :, discovery_checkpoint_index, discovery_arm_index],
            confirmation_raw[:, :, confirmation_checkpoint_index, confirmation_arm_index],
        ),
        axis=0,
    )
    return fold_percentile_maps(raw, tuple(range(6)), assignment, names)


def fixed_target_uq(
    values: np.ndarray,
    identities: np.ndarray,
    *,
    resamples: int = 100_000,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Target-only UQ for a fixed ensemble with no seed resampling."""
    point = equal_identity_mean(values, identities)
    weights = bootstrap_weight_matrix(
        identities, resamples=resamples, seed=seed, random_identity=False
    )
    distribution = weights @ np.asarray(values, dtype=np.float64)
    return {
        "point_delta": point,
        "bootstrap_mean": float(np.mean(distribution)),
        "SD": float(np.std(distribution, ddof=1)),
        "CI90_lower": float(np.quantile(distribution, 0.05, method="linear")),
        "CI90_upper": float(np.quantile(distribution, 0.95, method="linear")),
        "P_delta_lt_0": float(np.mean(distribution < 0.0)),
        "negative_sign90_MDE": float(empirical_sign_mde(distribution, -1)),
        "combined_evaluations": int(resamples),
        "seed_resampling": "none; six observed seeds fixed",
    }


def classify_confirmation(
    primary_direct: Mapping[str, Any],
    strict_direct: Mapping[str, Any],
    guards: Sequence[Mapping[str, Any]],
) -> str:
    """Apply the preregistered STEP61B precedence without a margin gate."""
    if any(bool(row["resolved_harm"]) for row in guards):
        return "guard_failed"
    primary_point = float(primary_direct["point_delta"])
    primary_upper = float(primary_direct["CI90_upper"])
    strict_lower = float(strict_direct["CI90_lower"])
    if primary_upper < 0.0 and strict_lower > 0.0:
        return "confirmed_seen_anchor_only_strict_harm"
    if primary_upper < 0.0:
        return "confirmed_layer_improvement"
    if primary_point < 0.0:
        return "direction_replicated_but_unresolved"
    return "screen_lead_not_replicated"
