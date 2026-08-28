"""Reusable loading and evaluation helpers for LEVEL_METRICS_V1 experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ranker.frozen_readouts import aggregate_level_results, heldout_level_result
from ranker.io import read_csv
from ranker.trainer_contract import heldout_fold_metrics


def load_canonical_level_targets(
    path: Path, *, expected_images: int = 520, expected_identities: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the canonical connected level-target population."""
    with np.load(path, allow_pickle=False) as payload:
        image_ids = payload["image_id"].astype(str)
        identities = payload["identity_id"].astype(str)
        theta = payload["theta"].astype(np.float64)
    if len(image_ids) != expected_images:
        raise RuntimeError("Canonical target image-count contract failed.")
    if len(set(identities.tolist())) != expected_identities:
        raise RuntimeError("Canonical target identity-count contract failed.")
    return image_ids, identities, theta


def load_tune_rank_identity_split(path: Path) -> tuple[list[str], list[str]]:
    """Load the frozen four-identity tuning and ten-identity ranking split."""
    rows = read_csv(path)
    tune4 = [str(row["identity_id"]) for row in rows if row["partition"] == "TUNE4"]
    rank10 = [str(row["identity_id"]) for row in rows if row["partition"] == "RANK10"]
    if len(tune4) != 4 or len(rank10) != 10 or set(tune4) & set(rank10):
        raise RuntimeError("TUNE4/RANK10 split contract failed.")
    return tune4, rank10


def level_fold_result(
    image_ids: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    fold: Any,
    scores: np.ndarray,
) -> dict[str, object]:
    """Evaluate one score vector on a held-out identity fold."""
    return heldout_level_result(
        image_ids=image_ids,
        identities=identities,
        theta=theta,
        fold=fold,
        canonical_scores=scores,
    )


def aggregate_level_fold_results(
    results: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    """Aggregate level metrics and the historical mean frame correlations."""
    output = dict(aggregate_level_results(results))
    pearsons: list[float] = []
    spearmans: list[float] = []
    for result in results:
        human = np.asarray(result["human_percentile"], dtype=np.float64)
        model = np.asarray(result["model_percentile"], dtype=np.float64)
        pearsons.append(
            float(pearsonr(human - human.mean(), model - model.mean()).statistic)
        )
        spearmans.append(float(spearmanr(human, model).statistic))
    output["mean_frame_Pearson"] = float(np.mean(pearsons))
    output["mean_frame_Spearman"] = float(np.mean(spearmans))
    return output


def ensemble_percentile_fold_result(
    image_ids: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    fold: Any,
    seed_scores: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Evaluate the historical ensemble-of-seed-percentiles semantics."""
    rows = [
        level_fold_result(image_ids, identities, theta, fold, scores)
        for scores in seed_scores
    ]
    human = np.asarray(rows[0]["human_percentile"], dtype=np.float64)
    model = np.mean(
        np.stack(
            [np.asarray(row["model_percentile"], dtype=np.float64) for row in rows]
        ),
        axis=0,
    )
    ids = np.asarray(rows[0]["image_ids"])
    return {
        "identity": fold.identity,
        "image_ids": ids,
        "identities": np.asarray(rows[0]["identities"]),
        "human_percentile": human,
        "model_percentile": model,
        "metrics": heldout_fold_metrics(
            image_ids=ids,
            model_percentiles=model,
            human_percentiles=human,
        ),
        "maximum_percentile_error": float(np.max(np.abs(model - human))),
    }
