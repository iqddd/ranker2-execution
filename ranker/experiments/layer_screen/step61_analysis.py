"""Pure STEP61A surface, LOFO, guard, and level analysis."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr

from ranker.experiments.image_heldout_horizon_transfer.analysis import (
    fold_percentile_maps,
    level_metric_vectors,
    strict_endpoint_vectors,
)
from ranker.experiments.layer_screen.analysis import (
    aggregate_exposure_rows,
    exposure_loss_rows,
)
from ranker.external_validation import percentile_against


def endpoint_fold_loss(
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    maps: Sequence[Mapping[str, float]],
    fold: int,
) -> float:
    rows = [
        row
        for row in exposure_loss_rows(exposures, image_fold, maps)
        if int(image_fold[str(row["image_key"])]) == fold
    ]
    return aggregate_exposure_rows(rows)[0]


def select_lofo_maps(
    per_fold_direct: np.ndarray,
    layers: Sequence[int],
    checkpoints: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-layer checkpoint maps and joint layer/checkpoint map."""
    values = np.asarray(per_fold_direct, dtype=np.float64)
    layerwise = np.empty((len(layers), 5), dtype=np.int64)
    joint = np.empty((5, 2), dtype=np.int64)
    for layer_index, _layer in enumerate(layers):
        for heldout in range(5):
            train = [fold for fold in range(5) if fold != heldout]
            means = values[layer_index][:, train].mean(axis=1)
            layerwise[layer_index, heldout] = int(np.argmin(means))
    for heldout in range(5):
        train = [fold for fold in range(5) if fold != heldout]
        candidates = [
            (
                float(values[layer_index, checkpoint_index][train].mean()),
                -int(layer),
                int(checkpoint),
                layer_index,
                checkpoint_index,
            )
            for layer_index, layer in enumerate(layers)
            for checkpoint_index, checkpoint in enumerate(checkpoints)
        ]
        *_tie, layer_index, checkpoint_index = min(candidates)
        joint[heldout] = (layer_index, checkpoint_index)
    return layerwise, joint


def selected_fold_maps(
    raw_scores: np.ndarray,
    assignment: np.ndarray,
    names: Sequence[str],
    *,
    layer_index: int,
    checkpoint_indices: Sequence[int],
    seed_indices: Sequence[int],
) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for fold, checkpoint_index in enumerate(checkpoint_indices):
        maps = fold_percentile_maps(
            raw_scores[:, :, int(checkpoint_index), layer_index],
            seed_indices,
            assignment,
            names,
        )
        output.append(maps[fold])
    return output


def level_diagnostics(
    maps: Sequence[Mapping[str, float]],
    names: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    assignment: np.ndarray,
    identity_order: Sequence[str],
) -> dict[str, float]:
    level = level_metric_vectors(
        maps,
        names,
        identities,
        assignment,
        theta,
        identity_order,
        eligible_images=None,
    )
    human_rows: list[np.ndarray] = []
    model_rows: list[np.ndarray] = []
    for fold in range(5):
        reference = assignment != fold
        human = percentile_against(theta, theta[reference])
        selected = assignment == fold
        human_rows.append(human[selected])
        model_rows.append(
            np.asarray([maps[fold][str(name)] for name in names[selected]], dtype=np.float64)
        )
    human = np.concatenate(human_rows)
    model = np.concatenate(model_rows)
    stitched_identity = np.concatenate([identities[assignment == fold] for fold in range(5)])
    human_dev = human.copy()
    model_dev = model.copy()
    for identity in identity_order:
        selected = stitched_identity == identity
        human_dev[selected] -= human[selected].mean()
        model_dev[selected] -= model[selected].mean()
    denominator = float(np.dot(human_dev, human_dev))
    return {
        "IdentityOffsetMAE": float(level["IdentityOffsetMAE"].mean()),
        "IB_FrameLTE10": float(level["IB_FrameLTE10"].mean()),
        "beta": float(np.dot(human_dev, model_dev) / denominator),
        "Pearson": float(pearsonr(human_dev, model_dev).statistic),
        "sigma_model_over_sigma_human": float(
            model_dev.std(ddof=0) / human_dev.std(ddof=0)
        ),
    }


def all_seed_triple_vectors(
    raw_scores: np.ndarray,
    assignment: np.ndarray,
    names: Sequence[str],
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    identity_order: Sequence[str],
    *,
    layer_index: int,
    checkpoint_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    output = {metric: [] for metric in ("direct", "G", "gross")}
    for triple in itertools.product(range(3), repeat=3):
        maps = selected_fold_maps(
            raw_scores,
            assignment,
            names,
            layer_index=layer_index,
            checkpoint_indices=checkpoint_indices,
            seed_indices=triple,
        )
        vectors = strict_endpoint_vectors(
            exposures, image_fold, maps, identity_order
        )
        for metric, vector in vectors.items():
            output[metric].append(vector)
    return {metric: np.stack(rows) for metric, rows in output.items()}
