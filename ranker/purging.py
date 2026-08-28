from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .data import RankedImage


def compute_shrunk_identity_means(
    items: list[RankedImage],
    values: np.ndarray,
    lambda_shrinkage: float,
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    by_identity: dict[str, list[float]] = defaultdict(list)
    for item, value in zip(items, values, strict=True):
        by_identity[item.identity].append(float(value))

    shrunk_means: dict[str, float] = {}
    counts: dict[str, int] = {}
    raw_means: dict[str, float] = {}
    for identity, identity_values in by_identity.items():
        count = len(identity_values)
        raw_mean = float(np.mean(identity_values))
        shrink = count / (count + lambda_shrinkage)
        shrunk_means[identity] = shrink * raw_mean
        counts[identity] = count
        raw_means[identity] = raw_mean
    return shrunk_means, counts, raw_means


def subtract_identity_means(
    items: list[RankedImage],
    values: np.ndarray,
    identity_means: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            float(value) - float(identity_means.get(item.identity, 0.0))
            for item, value in zip(items, values, strict=True)
        ],
        dtype=np.float64,
    )


def compute_identity_offset_rms(
    items: list[RankedImage],
    values: np.ndarray,
) -> float:
    by_identity: dict[str, list[float]] = defaultdict(list)
    for item, value in zip(items, values, strict=True):
        by_identity[item.identity].append(float(value))

    if not by_identity:
        return 0.0

    mean_squares = [
        float(np.mean(identity_values)) ** 2
        for identity_values in by_identity.values()
    ]
    return math.sqrt(float(np.mean(mean_squares)))


def compute_identity_1d_accuracy(
    train_items: list[RankedImage],
    train_values: np.ndarray,
    val_items: list[RankedImage],
    val_values: np.ndarray,
) -> float:
    train_by_identity: dict[str, list[float]] = defaultdict(list)
    for item, value in zip(train_items, train_values, strict=True):
        train_by_identity[item.identity].append(float(value))

    if not train_by_identity:
        return 0.0

    train_identity_means = {
        identity: float(np.mean(identity_values))
        for identity, identity_values in train_by_identity.items()
    }

    correct = 0
    total = 0
    for item, value in zip(val_items, val_values, strict=True):
        predicted_identity = min(
            train_identity_means,
            key=lambda identity: (
                abs(float(value) - train_identity_means[identity]),
                identity,
            ),
        )
        correct += int(predicted_identity == item.identity)
        total += 1

    return 0.0 if total == 0 else correct / total
