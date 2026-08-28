from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .data import RankedImage
from .metrics import (
    compute_same_id_weighted_pair_acc,
    compute_system_metrics,
    score_map_from_array,
)


DEFAULT_EPS = 1e-12


def choose_decorrelated_lambda(
    train_groups: list[list[RankedImage]],
    train_items: list[RankedImage],
    b_norm_train: np.ndarray,
    u_norm_train: np.ndarray,
    *,
    lambda_grid: Sequence[float],
    eps: float = DEFAULT_EPS,
) -> tuple[float, float, float, list[dict[str, float]]]:
    grid = tuple(float(value) for value in lambda_grid)
    if not grid:
        raise ValueError("lambda_grid must not be empty.")
    b_norm_scores = score_map_from_array(train_items, b_norm_train)
    best_lambda = grid[0]
    best_weighted_pair_acc = float("-inf")
    best_same_id_weighted_pair_acc = float("-inf")
    lambda_rows: list[dict[str, float]] = []

    for blend_lambda in grid:
        order_scores = score_map_from_array(
            train_items,
            b_norm_train + blend_lambda * u_norm_train,
        )
        metrics = compute_system_metrics(train_groups, order_scores, b_norm_scores)
        same_id_weighted_pair_acc = compute_same_id_weighted_pair_acc(
            train_groups,
            order_scores,
        )
        lambda_rows.append(
            {
                "lambda": blend_lambda,
                "train_pair_acc": metrics.pair_acc,
                "train_weighted_pair_acc": metrics.weighted_pair_acc,
                "train_top1_hard": metrics.top1_hard,
                "train_top1_tie_aware": metrics.top1_tie_aware,
                "train_same_id_weighted_pair_acc": same_id_weighted_pair_acc,
            }
        )

        is_better = False
        if metrics.weighted_pair_acc > best_weighted_pair_acc + eps:
            is_better = True
        elif abs(metrics.weighted_pair_acc - best_weighted_pair_acc) <= eps:
            if same_id_weighted_pair_acc > best_same_id_weighted_pair_acc + eps:
                is_better = True
            elif (
                abs(same_id_weighted_pair_acc - best_same_id_weighted_pair_acc) <= eps
                and blend_lambda < best_lambda
            ):
                is_better = True

        if is_better:
            best_lambda = blend_lambda
            best_weighted_pair_acc = metrics.weighted_pair_acc
            best_same_id_weighted_pair_acc = same_id_weighted_pair_acc

    return (
        best_lambda,
        best_weighted_pair_acc,
        best_same_id_weighted_pair_acc,
        lambda_rows,
    )
