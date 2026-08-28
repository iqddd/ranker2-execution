from __future__ import annotations

import math
from statistics import mean, median, pstdev

import numpy as np

from .metrics import SystemMetrics

def percentile_ci(values: np.ndarray | list[float]) -> list[float]:
    return [float(value) for value in np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])]


def summarize_distribution(values: list[float], *, include_bounds: bool = True) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    result = {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "median": float(np.median(array))}
    if include_bounds:
        result.update({"min": float(array.min()), "max": float(array.max())})
    return result


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_bootstrap: int, one_sided: bool = False) -> tuple[list[float], float]:
    array = np.asarray(values, dtype=np.float64)
    samples = array[np.random.default_rng(seed).integers(0, len(array), size=(n_bootstrap, len(array)))].mean(axis=1)
    lower = (1 + np.count_nonzero(samples <= 0.0)) / (n_bootstrap + 1)
    upper = (1 + np.count_nonzero(samples >= 0.0)) / (n_bootstrap + 1)
    return percentile_ci(samples), float(lower if one_sided else min(1.0, 2.0 * min(lower, upper)))


def bootstrap_oof_metric(y_chunks, prediction_chunks, metric, *, seed: int, n_bootstrap: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for index, selected in enumerate(rng.integers(0, len(y_chunks), size=(n_bootstrap, len(y_chunks)))):
        values[index] = metric(np.concatenate([y_chunks[value] for value in selected]), np.concatenate([prediction_chunks[value] for value in selected]))
    return percentile_ci(values)


def build_product_cluster_bootstrap_weights(pairs, *, identity_order, seed: int, n_bootstrap: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = {identity: index for index, identity in enumerate(identity_order)}
    weights = np.empty((n_bootstrap, len(pairs)), dtype=np.float64)
    row = 0
    while row < n_bootstrap:
        counts = np.bincount(rng.integers(0, len(identity_order), size=len(identity_order)), minlength=len(identity_order))
        result = np.asarray([counts[positions[first]] * counts[positions[second]] for first, second in pairs], dtype=np.float64)
        if result.sum() > 0:
            weights[row] = result; row += 1
    return weights


def weighted_bootstrap_ci(values: list[float], bootstrap_weights: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(bootstrap_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(array) or not np.all(np.isfinite(weights)) or np.any(weights.sum(axis=1) <= 0):
        raise ValueError("Bootstrap weights must be finite and have positive row sums.")
    return percentile_ci((weights @ array) / weights.sum(axis=1))


def summarize_system_metrics(metrics: list[SystemMetrics]) -> dict[str, float]:
    return {
        "pair_acc_mean": mean(metric.pair_acc for metric in metrics),
        "pair_acc_std": pstdev(metric.pair_acc for metric in metrics),
        "weighted_pair_acc_mean": mean(metric.weighted_pair_acc for metric in metrics),
        "weighted_pair_acc_std": pstdev(metric.weighted_pair_acc for metric in metrics),
        "top1_hard_mean": mean(metric.top1_hard for metric in metrics),
        "top1_hard_std": pstdev(metric.top1_hard for metric in metrics),
        "top1_tie_aware_mean": mean(metric.top1_tie_aware for metric in metrics),
        "top1_tie_aware_std": pstdev(metric.top1_tie_aware for metric in metrics),
    }


def summarize_scalar(values: list[float], label: str) -> dict[str, float]:
    finite_values = [value for value in values if not math.isnan(value)]
    if not finite_values:
        raise ValueError(f"No values collected for {label}")
    return {
        f"{label}_mean": mean(finite_values),
        f"{label}_std": pstdev(finite_values),
    }


def summarize_group_paired_deltas(
    group_delta_values: list[float],
    *,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    return {
        "n_group_positive": sum(delta > eps for delta in group_delta_values),
        "n_group_zero": sum(abs(delta) <= eps for delta in group_delta_values),
        "n_group_negative": sum(delta < -eps for delta in group_delta_values),
        "mean_group_delta_WPA": mean(group_delta_values),
        "median_group_delta_WPA": median(group_delta_values),
    }
