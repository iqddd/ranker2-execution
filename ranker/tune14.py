"""Reusable analysis helpers for retained TUNE14 single-layer panels."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ranker.checkpoint_scores import fold_percentile_score_map, score_array_hash
from ranker.validation_resolution import (
    contiguous_checkpoint_plateau,
    direct_identity_loss_vector,
    empirical_sign_mde,
)

STEP57_CHECKPOINTS = (
    0,
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    160,
    192,
    256,
    384,
    512,
    768,
    1024,
)

SEEDS = (20260803, 20260804, 20260805)


def development_identity_index(
    names: Sequence[str],
    canonical_names: Sequence[str],
    canonical_identities: Sequence[str],
    training_groups: Sequence[Sequence[Any]],
) -> np.ndarray:
    mapping = {
        str(name): str(identity)
        for name, identity in zip(canonical_names, canonical_identities, strict=True)
    }
    for group in training_groups:
        for item in group:
            previous = mapping.setdefault(item.image_name, item.identity)
            if previous != item.identity:
                raise RuntimeError(f"identity conflict for {item.image_name}")
    missing = [name for name in names if name not in mapping]
    if missing:
        raise RuntimeError(f"identity index misses {len(missing)} development images")
    return np.asarray([mapping[name] for name in names], dtype=str)


def percentile_maps(
    panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    system: str,
    checkpoint_map: Mapping[str, int],
    seed_indices: Sequence[int],
    identities: Sequence[str],
    names: np.ndarray,
    identity_ids: np.ndarray,
) -> dict[str, dict[str, float]]:
    canonical = np.arange(len(names)) < 520
    return {
        identity: fold_percentile_score_map(
            panel,
            system=system,
            identity=identity,
            checkpoint=int(checkpoint_map[identity]),
            seed_indices=seed_indices,
            seeds=SEEDS,
            image_keys=names,
            identity_ids=identity_ids,
            reference_population_mask=canonical,
        )
        for identity in identities
    }


def identity_losses(
    exposures: Sequence[Mapping[str, Any]],
    panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    system: str,
    checkpoint_map: Mapping[str, int],
    seed_indices: Sequence[int],
    identities: Sequence[str],
    names: np.ndarray,
    identity_ids: np.ndarray,
) -> np.ndarray:
    maps = percentile_maps(
        panel, system, checkpoint_map, seed_indices, identities, names, identity_ids
    )
    return direct_identity_loss_vector(exposures, maps, identities)


def select_full_loio(
    checkpoints: Sequence[int], checkpoint_identity_losses: Mapping[int, np.ndarray]
) -> tuple[int, dict[str, Any], dict[int, int]]:
    ordered = tuple(map(int, checkpoints))
    matrix = np.stack(
        [checkpoint_identity_losses[checkpoint] for checkpoint in ordered]
    )
    full_index = int(np.argmin(matrix.mean(axis=1)))
    full_checkpoint = ordered[full_index]
    full = {
        "checkpoint": full_checkpoint,
        "loss": float(matrix[full_index].mean()),
        "plateau_count": int(
            np.sum(matrix.mean(axis=1) <= matrix.mean(axis=1).min() + 0.005)
        ),
        "contiguous_plateau": contiguous_checkpoint_plateau(
            ordered, matrix.mean(axis=1), full_index, 0.005
        ),
        "boundary": full_checkpoint == ordered[-1],
    }
    loio: dict[int, int] = {}
    for heldout in range(matrix.shape[1]):
        train = [index for index in range(matrix.shape[1]) if index != heldout]
        loio[heldout] = ordered[int(np.argmin(matrix[:, train].mean(axis=1)))]
    return full_checkpoint, full, loio


def paired_uq(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int = 20260817,
    resamples: int = 100000,
    identity_draws: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """UQ for arrays shaped [27 seed triples, identities].

    ``identity_draws`` lets related contrasts share one frozen bootstrap matrix.
    """
    if candidate.shape != reference.shape or candidate.shape[0] != 27:
        raise ValueError("paired TUNE14 arrays must have shape [27, identities]")
    contrast = candidate - reference
    standard = list(itertools.product(range(3), repeat=3)).index((0, 1, 2))
    point_identity = contrast[standard]
    if identity_draws is None:
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.integers(
            0,
            contrast.shape[1],
            size=(resamples, contrast.shape[1]),
            endpoint=False,
            dtype=np.int64,
        )
    else:
        draws = np.asarray(identity_draws, dtype=np.int64)
        if draws.ndim != 2 or draws.shape[1] != contrast.shape[1]:
            raise ValueError("Identity-draw width must match the identity vector.")
    distribution = np.concatenate([row[draws].mean(axis=1) for row in contrast])
    negative = empirical_sign_mde(distribution, direction=-1)
    positive = empirical_sign_mde(distribution, direction=1)
    seed_means = contrast.mean(axis=1)
    payload = {
        "point_delta": float(point_identity.mean()),
        "resampling_mean": float(distribution.mean()),
        "bootstrap_mean": float(distribution.mean()),
        "SD": float(distribution.std(ddof=1)),
        "CI90_lower": float(np.quantile(distribution, 0.05, method="linear")),
        "CI90_upper": float(np.quantile(distribution, 0.95, method="linear")),
        "P_delta_lt_0": float(np.mean(distribution < 0)),
        "P_delta_gt_0": float(np.mean(distribution > 0)),
        "negative_sign90_MDE": float(negative),
        "positive_sign90_MDE": float(positive),
        "seed_triple_min": float(seed_means.min()),
        "seed_triple_median": float(np.median(seed_means)),
        "seed_triple_max": float(seed_means.max()),
        "identity_deltas": point_identity.tolist(),
        "combined_evaluations": int(distribution.size),
    }
    return payload, distribution


def parameter_name_hash(model: Any) -> str:
    payload = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def score_hash(scores: np.ndarray) -> str:
    """Backward-compatible name for the shared retained-score digest."""
    return score_array_hash(scores)
