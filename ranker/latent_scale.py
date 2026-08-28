"""Canonical latent human-scale estimator introduced in Step 20."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import RankedImage

TARGET_BY_DISTANCE: Mapping[int, float] = MappingProxyType(
    {1: 94 / 107, 2: 90 / 92}
)
TARGET_D_GE_3 = 226 / 227


def target_for_distance(distance: int) -> float:
    """Return the fixed Step-20 soft target for one rank distance."""
    return TARGET_BY_DISTANCE.get(int(distance), TARGET_D_GE_3)


def comparison_arrays(
    groups: Sequence[Sequence[RankedImage]], image_index: Mapping[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact Step-20 weighted pair arrays in stable group order."""
    lower_indices: list[int] = []
    higher_indices: list[int] = []
    targets: list[float] = []
    weights: list[float] = []
    group_ids: list[int] = []
    for group_id, group in enumerate(groups):
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (
                    (first, second) if first.rank < second.rank else (second, first)
                )
                distance = int(higher.rank - lower.rank)
                lower_indices.append(image_index[lower.image_name])
                higher_indices.append(image_index[higher.image_name])
                targets.append(target_for_distance(distance))
                weights.append(float(distance))
                group_ids.append(group_id)
    return (
        np.asarray(lower_indices, dtype=np.int64),
        np.asarray(higher_indices, dtype=np.int64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(group_ids, dtype=np.int64),
    )


def fit_theta(
    lower: np.ndarray,
    higher: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    n_images: int,
    *,
    excluded_group: int | None = None,
) -> tuple[np.ndarray, float, float]:
    """Run the exact float64 LBFGS estimator and mean-zero gauge from Step 20.

    The historical normalization ``84 * n_groups`` is retained literally.  It
    is a constant multiplier and therefore does not change the optimum when a
    variable-size group is present in a later diagnostic fit.
    """
    mask = (
        np.ones(len(groups), dtype=bool)
        if excluded_group is None
        else groups != excluded_group
    )
    lower_t = torch.as_tensor(lower[mask], dtype=torch.int64)
    higher_t = torch.as_tensor(higher[mask], dtype=torch.int64)
    targets_t = torch.as_tensor(targets[mask], dtype=torch.float64)
    weights_t = torch.as_tensor(weights[mask], dtype=torch.float64)
    n_train_groups = int(len(np.unique(groups[mask])))
    theta_raw = torch.zeros(n_images, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [theta_raw],
        max_iter=1000,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def loss_fn() -> torch.Tensor:
        centered = theta_raw - theta_raw.mean()
        logits = centered[higher_t] - centered[lower_t]
        losses = F.binary_cross_entropy_with_logits(
            logits, targets_t, reduction="none"
        )
        return torch.sum(weights_t * losses) / (84.0 * n_train_groups)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.zero_grad()
    final_loss = loss_fn()
    final_loss.backward()
    gradient_norm = float(theta_raw.grad.detach().norm().item())
    theta = (
        theta_raw.detach() - theta_raw.detach().mean()
    ).cpu().numpy().astype(np.float64)
    return theta, float(final_loss.item()), gradient_norm


def fit_theta_with_fixed_anchors(
    *,
    ranks: Sequence[int],
    anchor_mask: Sequence[bool],
    anchor_theta: Sequence[float],
) -> tuple[np.ndarray, float, float]:
    """Fit only non-anchor scores for one ranked external-anchor group.

    This is the external-target analogue of :func:`fit_theta`: it uses the
    exact Step-20 soft targets, distance weights, float64 LBFGS settings and
    pair ordering, while treating the supplied anchor scores as constants.
    There is no free gauge because at least one score is fixed.
    """
    rank_array = np.asarray(ranks, dtype=np.int64)
    fixed = np.asarray(anchor_mask, dtype=bool)
    fixed_values = np.asarray(anchor_theta, dtype=np.float64)
    if (
        rank_array.ndim != 1
        or len(rank_array) < 2
        or sorted(rank_array.tolist()) != list(range(len(rank_array)))
        or fixed.shape != rank_array.shape
        or fixed_values.shape != rank_array.shape
        or not np.any(fixed)
        or np.all(fixed)
        or not np.all(np.isfinite(fixed_values[fixed]))
    ):
        raise ValueError("Invalid fixed-anchor latent-fit inputs.")

    lower: list[int] = []
    higher: list[int] = []
    targets: list[float] = []
    weights: list[float] = []
    for first in range(len(rank_array)):
        for second in range(first + 1, len(rank_array)):
            if rank_array[first] < rank_array[second]:
                lo, hi = first, second
            else:
                lo, hi = second, first
            distance = int(rank_array[hi] - rank_array[lo])
            lower.append(lo)
            higher.append(hi)
            targets.append(target_for_distance(distance))
            weights.append(float(distance))

    variable_positions = np.flatnonzero(~fixed)
    variable_lookup = np.full(len(rank_array), -1, dtype=np.int64)
    variable_lookup[variable_positions] = np.arange(len(variable_positions))
    lower_t = torch.as_tensor(lower, dtype=torch.int64)
    higher_t = torch.as_tensor(higher, dtype=torch.int64)
    targets_t = torch.as_tensor(targets, dtype=torch.float64)
    weights_t = torch.as_tensor(weights, dtype=torch.float64)
    fixed_t = torch.as_tensor(
        np.where(fixed, fixed_values, 0.0), dtype=torch.float64
    )
    variable_lookup_t = torch.as_tensor(variable_lookup, dtype=torch.int64)
    variable = torch.zeros(
        len(variable_positions), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [variable],
        max_iter=1000,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def scores() -> torch.Tensor:
        result = fixed_t.clone()
        mask = variable_lookup_t >= 0
        result[mask] = variable[variable_lookup_t[mask]]
        return result

    def loss_fn() -> torch.Tensor:
        value = scores()
        logits = value[higher_t] - value[lower_t]
        losses = F.binary_cross_entropy_with_logits(
            logits, targets_t, reduction="none"
        )
        return torch.sum(weights_t * losses) / 84.0

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.zero_grad()
    final_loss = loss_fn()
    final_loss.backward()
    gradient_norm = float(variable.grad.detach().norm().item())
    result = fixed_values.copy()
    result[variable_positions] = variable.detach().cpu().numpy()
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Fixed-anchor latent fit produced non-finite scores.")
    return result, float(final_loss.item()), gradient_norm


def fit_theta_multi_exposure_fixed_anchors(
    *,
    variable_count: int,
    exposures: Sequence[Mapping[str, Sequence[float] | Sequence[int]]],
) -> tuple[np.ndarray, float, float]:
    """Jointly fit external-image scores across one or more anchor exposures.

    Each exposure supplies eight (or otherwise equally ranked) members through
    ``variable_indices`` (``-1`` denotes a fixed anchor), ``fixed_theta`` and a
    permutation ``ranks``.  The Step-20 targets, distance weights and LBFGS
    tolerances are retained exactly.
    """
    if variable_count <= 0 or not exposures:
        raise ValueError("A multi-exposure fit needs variables and exposures.")
    lower_variable: list[int] = []
    higher_variable: list[int] = []
    lower_fixed: list[float] = []
    higher_fixed: list[float] = []
    targets: list[float] = []
    weights: list[float] = []
    for exposure in exposures:
        variable = np.asarray(exposure["variable_indices"], dtype=np.int64)
        fixed = np.asarray(exposure["fixed_theta"], dtype=np.float64)
        ranks = np.asarray(exposure["ranks"], dtype=np.int64)
        if (
            variable.ndim != 1
            or fixed.shape != variable.shape
            or ranks.shape != variable.shape
            or sorted(ranks.tolist()) != list(range(len(ranks)))
            or np.any(variable >= variable_count)
            or np.any((variable < 0) & ~np.isfinite(fixed))
        ):
            raise ValueError("Invalid multi-exposure member arrays.")
        for first in range(len(ranks)):
            for second in range(first + 1, len(ranks)):
                lo, hi = (first, second) if ranks[first] < ranks[second] else (second, first)
                distance = int(ranks[hi] - ranks[lo])
                lower_variable.append(int(variable[lo]))
                higher_variable.append(int(variable[hi]))
                lower_fixed.append(float(fixed[lo]))
                higher_fixed.append(float(fixed[hi]))
                targets.append(target_for_distance(distance))
                weights.append(float(distance))
    lower_variable_t = torch.as_tensor(lower_variable, dtype=torch.int64)
    higher_variable_t = torch.as_tensor(higher_variable, dtype=torch.int64)
    lower_fixed_t = torch.as_tensor(lower_fixed, dtype=torch.float64)
    higher_fixed_t = torch.as_tensor(higher_fixed, dtype=torch.float64)
    targets_t = torch.as_tensor(targets, dtype=torch.float64)
    weights_t = torch.as_tensor(weights, dtype=torch.float64)
    variable = torch.zeros(variable_count, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [variable], max_iter=1000, tolerance_grad=1e-12,
        tolerance_change=1e-12, history_size=100, line_search_fn="strong_wolfe",
    )

    def loss_fn() -> torch.Tensor:
        lower = torch.where(
            lower_variable_t >= 0,
            variable[torch.clamp(lower_variable_t, min=0)],
            lower_fixed_t,
        )
        higher = torch.where(
            higher_variable_t >= 0,
            variable[torch.clamp(higher_variable_t, min=0)],
            higher_fixed_t,
        )
        losses = F.binary_cross_entropy_with_logits(
            higher - lower, targets_t, reduction="none"
        )
        return torch.sum(weights_t * losses) / (84.0 * len(exposures))

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    optimizer.step(closure)
    optimizer.zero_grad()
    final_loss = loss_fn()
    final_loss.backward()
    result = variable.detach().cpu().numpy().astype(np.float64)
    gradient_norm = float(variable.grad.detach().norm().item())
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Multi-exposure latent fit produced non-finite scores.")
    return result, float(final_loss.item()), gradient_norm


def graph_components(
    groups: Sequence[Sequence[RankedImage]], image_index: Mapping[str, int]
) -> dict[str, Any]:
    """Count comparison-graph components and basic degree statistics."""
    adjacency = [set() for _ in image_index]
    for group in groups:
        indices = [image_index[item.image_name] for item in group]
        for offset, left in enumerate(indices):
            adjacency[left].update(indices[:offset])
            adjacency[left].update(indices[offset + 1 :])
    unseen = set(range(len(adjacency)))
    components = 0
    component_sizes: list[int] = []
    while unseen:
        components += 1
        stack = [unseen.pop()]
        size = 1
        while stack:
            node = stack.pop()
            neighbours = adjacency[node] & unseen
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
            size += len(neighbours)
        component_sizes.append(size)
    degrees = np.asarray([len(value) for value in adjacency], dtype=np.float64)
    return {
        "n_components": components,
        "component_sizes_descending": sorted(component_sizes, reverse=True),
        "n_edges": int(sum(len(value) for value in adjacency) // 2),
        "min_degree": int(degrees.min()),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(degrees.max()),
    }


def midrank_percentiles(values: Sequence[float]) -> np.ndarray:
    """Return standard 0--100 midrank empirical percentiles."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("Midrank percentiles require one finite nonempty vector.")
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        percentile = 100.0 * (start + 0.5 * (stop - start)) / len(array)
        result[order[start:stop]] = percentile
        start = stop
    return result
