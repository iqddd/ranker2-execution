"""Reusable hierarchical sampling and power helpers for external validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def balanced_allocations(image_count: int, identity_count: int) -> np.ndarray:
    """Allocate images as evenly as possible over non-empty identities."""
    if image_count < identity_count or identity_count <= 0:
        raise ValueError("Every identity must receive at least one image.")
    result = np.full(identity_count, image_count // identity_count, dtype=np.int64)
    result[: image_count % identity_count] += 1
    return result


def unseen_allocations(image_count: int, design: str) -> np.ndarray:
    """Return deterministic per-identity counts for U4/U2/U1/HYBRID."""
    if image_count <= 0:
        raise ValueError("Unseen image count must be positive.")
    if design in {"U4", "U2"}:
        width = 4 if design == "U4" else 2
        full, remainder = divmod(image_count, width)
        values = [width] * full + ([remainder] if remainder else [])
    elif design == "U1":
        values = [1] * image_count
    elif design == "HYBRID":
        depth_images = 2 * ((image_count // 2) // 2)
        breadth_images = image_count - depth_images
        values = [1] * breadth_images + [2] * (depth_images // 2)
    else:
        raise ValueError(f"Unknown unseen design {design!r}.")
    result = np.asarray(values, dtype=np.int64)
    if int(result.sum()) != image_count or np.any(result <= 0):
        raise AssertionError("Invalid unseen allocation.")
    return result


def stratum_counts(total_images: int, allocation: str) -> tuple[int, int]:
    """Resolve exact KNOWN/UNSEEN counts while retaining all 14 known identities."""
    if allocation == "50_50":
        known = total_images // 2
    elif allocation == "25_75":
        known = max(14, int(round(total_images * 0.25)))
    else:
        raise ValueError(f"Unknown stratum allocation {allocation!r}.")
    return known, total_images - known


def variance_decomposition(pool: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """ANOVA and random-intercept decomposition of observed paired deltas."""
    groups = [np.asarray(pool[key], dtype=np.float64)[:, 1] for key in sorted(pool)]
    if len(groups) < 2 or any(len(group) == 0 for group in groups):
        raise ValueError("Variance decomposition needs at least two non-empty identities.")
    sizes = np.asarray([len(group) for group in groups], dtype=np.float64)
    values = np.concatenate(groups)
    grand = float(np.mean(values))
    within_ss = float(sum(np.sum((group - np.mean(group)) ** 2) for group in groups))
    between_ss = float(sum(len(group) * (float(np.mean(group)) - grand) ** 2 for group in groups))
    total_ss = within_ss + between_ss
    k = len(groups)
    n = len(values)
    ms_within = within_ss / (n - k)
    ms_between = between_ss / (k - 1)
    n0 = (n - float(np.sum(sizes**2)) / n) / (k - 1)
    between_component = max((ms_between - ms_within) / n0, 0.0)
    within_component = max(ms_within, 0.0)
    denominator = between_component + within_component
    return {
        "source_identity_count": k,
        "source_image_count": n,
        "within_SS": within_ss,
        "between_SS": between_ss,
        "within_fraction_observed": within_ss / total_ss if total_ss else 0.0,
        "between_fraction_observed": between_ss / total_ss if total_ss else 0.0,
        "within_variance_component": within_component,
        "between_identity_variance_component": between_component,
        "intraclass_correlation": between_component / denominator if denominator else 0.0,
        "method": "unbalanced one-way random-intercept method-of-moments on centered per-image paired deltas",
    }


def _tail_delta_batch(base: np.ndarray, challenger: np.ndarray) -> np.ndarray:
    k = int(math.ceil(0.10 * base.shape[1]))
    base_tail = np.partition(base, base.shape[1] - k, axis=1)[:, -k:]
    challenger_tail = np.partition(challenger, challenger.shape[1] - k, axis=1)[:, -k:]
    return np.mean(challenger_tail, axis=1) - np.mean(base_tail, axis=1)


def sample_identity_deltas(
    pool: Mapping[str, np.ndarray],
    allocations: Sequence[int],
    *,
    effect: float,
    replicates: int,
    target_perturbation_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw an R x identity matrix under the hierarchical pseudo-identity model."""
    sources = sorted(pool)
    output = np.empty((replicates, len(allocations)), dtype=np.float64)
    for column, count_value in enumerate(allocations):
        count = int(count_value)
        source_index = rng.integers(0, len(sources), size=replicates)
        values = np.empty(replicates, dtype=np.float64)
        for index, source in enumerate(sources):
            mask = source_index == index
            amount = int(np.sum(mask))
            if not amount:
                continue
            rows = np.asarray(pool[source], dtype=np.float64)
            chosen = rows[rng.integers(0, len(rows), size=(amount, count))]
            base = chosen[:, :, 0]
            challenger = np.maximum(0.0, base + chosen[:, :, 1] + effect)
            values[mask] = _tail_delta_batch(base, challenger)
        output[:, column] = values + rng.normal(0.0, target_perturbation_sd, size=replicates)
    return output


def summarize_scope(
    known: np.ndarray,
    unseen: np.ndarray,
    *,
    scope: str,
    effect: float,
    z_value: float,
) -> dict[str, float]:
    """Compute sign, one-sided detection and clustered-normal precision."""
    known_mean = np.mean(known, axis=1)
    unseen_mean = np.mean(unseen, axis=1)
    known_var = np.var(known, axis=1, ddof=0) / known.shape[1]
    unseen_var = np.var(unseen, axis=1, ddof=0) / unseen.shape[1]
    if scope == "EXT_PRIMARY":
        estimate = 0.5 * (known_mean + unseen_mean)
        variance = 0.25 * (known_var + unseen_var)
    elif scope == "KNOWN_ONLY":
        estimate, variance = known_mean, known_var
    elif scope == "UNSEEN_ONLY":
        estimate, variance = unseen_mean, unseen_var
    else:
        raise ValueError(f"Unknown scope {scope!r}.")
    half = float(z_value) * np.sqrt(np.maximum(variance, 0.0))
    widths = 2.0 * half
    correct = estimate < 0.0 if effect < 0.0 else estimate > 0.0
    detect = estimate + half < 0.0 if effect < 0.0 else estimate - half > 0.0
    return {
        "probability_correct_sign": float(np.mean(correct)),
        "probability_one_sided_90CI_detection": float(np.mean(detect)),
        "median_90CI_width": float(np.median(widths)),
        "P10_90CI_width": float(np.percentile(widths, 10)),
        "P90_90CI_width": float(np.percentile(widths, 90)),
        "median_90CI_half_width": float(np.median(half)),
    }


def top_tail_count(allocations: Sequence[int]) -> int:
    return int(sum(math.ceil(0.10 * int(value)) for value in allocations))


def sample_error_cube(
    pool: Mapping[str, Sequence[tuple[float, float]]],
    *,
    identity_count: int,
    images_per_identity: int,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample baseline errors and raw challenger-minus-baseline errors."""
    sources = sorted(pool)
    base = np.empty((replicates, identity_count, images_per_identity), dtype=np.float64)
    raw = np.empty_like(base)
    for identity_slot in range(identity_count):
        source_index = rng.integers(0, len(sources), size=replicates)
        for index, source in enumerate(sources):
            mask = source_index == index
            amount = int(np.sum(mask))
            if not amount:
                continue
            rows = np.asarray(pool[source], dtype=np.float64)
            chosen = rows[rng.integers(0, len(rows), size=(amount, images_per_identity))]
            base[mask, identity_slot] = chosen[:, :, 0]
            raw[mask, identity_slot] = chosen[:, :, 1] - chosen[:, :, 0]
    return base, raw


def metric_with_offset(
    base: np.ndarray,
    raw_delta: np.ndarray,
    *,
    offset: float,
    endpoint: str,
    threshold: float | None = None,
) -> np.ndarray:
    """Return replicate x identity paired endpoint deltas."""
    challenger = np.maximum(0.0, base + raw_delta + offset)
    if endpoint == "IB_AE":
        return np.mean(challenger - base, axis=2)
    if endpoint in {"IB_LTE10", "IB_MAX4"}:
        k = 1 if endpoint == "IB_MAX4" else int(math.ceil(0.10 * base.shape[2]))
        base_tail = np.partition(base, base.shape[2] - k, axis=2)[:, :, -k:]
        challenger_tail = np.partition(challenger, challenger.shape[2] - k, axis=2)[:, :, -k:]
        return np.mean(challenger_tail, axis=2) - np.mean(base_tail, axis=2)
    if endpoint == "IB_GLE":
        if threshold is None:
            raise ValueError("IB_GLE requires a threshold.")
        return np.mean((challenger >= threshold).astype(np.float64) - (base >= threshold).astype(np.float64), axis=2)
    raise ValueError(f"Unknown endpoint {endpoint!r}.")


def calibrate_offset(
    base: np.ndarray,
    raw_delta: np.ndarray,
    *,
    target_effect: float,
    endpoint: str,
) -> float:
    """Calibrate an additive challenger-error offset to a target mean contrast."""
    low, high = -200.0, 200.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        value = float(np.mean(metric_with_offset(base, raw_delta, offset=middle, endpoint=endpoint)))
        if value > target_effect:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def summarize_cluster_metric(
    values: np.ndarray, *, direction: int, z_value: float
) -> dict[str, float]:
    """Summarize a replicate x identity matrix with a fixed identity count."""
    estimate = np.mean(values, axis=1)
    variance = np.var(values, axis=1, ddof=0) / values.shape[1]
    half = float(z_value) * np.sqrt(np.maximum(variance, 0.0))
    correct = estimate < 0.0 if direction < 0 else estimate > 0.0
    detect = estimate + half < 0.0 if direction < 0 else estimate - half > 0.0
    return {
        "mean_contrast": float(np.mean(estimate)),
        "probability_correct_sign": float(np.mean(correct)),
        "probability_one_sided_90CI_detection": float(np.mean(detect)),
        "median_90CI_width": float(2.0 * np.median(half)),
        "P10_90CI_width": float(2.0 * np.percentile(half, 10)),
        "P90_90CI_width": float(2.0 * np.percentile(half, 90)),
        "median_90CI_half_width": float(np.median(half)),
    }
