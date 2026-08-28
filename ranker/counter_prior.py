"""Train-fold counter-prior labels and CP3 group-balanced RankNet objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit

from .data import RankedImage
from .group_ranknet import GroupRankNetDiagnostics, ranknet_target


@dataclass(frozen=True)
class CounterPriorWeights:
    """Pair multipliers in stable group/pair order plus fold diagnostics."""

    multipliers_by_group: tuple[np.ndarray, ...]
    counter_masks_by_group: tuple[np.ndarray, ...]
    train_o_counter_pair_rate: float
    counter_loss_mass_before: float
    counter_loss_mass_after_cp3: float


def build_counter_prior_weights(
    train_o: Sequence[Sequence[RankedImage]],
) -> CounterPriorWeights:
    """Build leave-current-group-out priors using only the outer fold's train-O."""
    identities = sorted({item.identity for group in train_o for item in group})
    total_count = {identity: 0 for identity in identities}
    total_sum = {identity: 0.0 for identity in identities}
    for group in train_o:
        for item in group:
            total_count[item.identity] += 1
            total_sum[item.identity] += float(item.z)

    multipliers: list[np.ndarray] = []
    counter_masks: list[np.ndarray] = []
    before_mass: list[float] = []
    after_mass: list[float] = []
    total_counter = 0
    total_pairs = 0
    for group in train_o:
        local_count = {identity: 0 for identity in identities}
        local_sum = {identity: 0.0 for identity in identities}
        for item in group:
            local_count[item.identity] += 1
            local_sum[item.identity] += float(item.z)
        prior: dict[str, float] = {}
        for identity in identities:
            count = total_count[identity] - local_count[identity]
            summed = total_sum[identity] - local_sum[identity]
            prior[identity] = float(summed / (count + 8.0))

        group_counter: list[bool] = []
        distances: list[float] = []
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (first, second) if first.rank < second.rank else (second, first)
                delta = prior[higher.identity] - prior[lower.identity]
                group_counter.append(bool(higher.identity != lower.identity and delta < 0.0))
                distances.append(float(higher.rank - lower.rank))
        counter = np.asarray(group_counter, dtype=bool)
        distance = np.asarray(distances, dtype=np.float64)
        multiplier = np.where(counter, 3.0, 1.0).astype(np.float64)
        multipliers.append(multiplier)
        counter_masks.append(counter)
        total_counter += int(counter.sum())
        total_pairs += len(counter)
        before_mass.append(float(distance[counter].sum() / distance.sum()))
        after_mass.append(float((distance[counter] * multiplier[counter]).sum() / (distance * multiplier).sum()))

    if not multipliers or any(array.shape != (28,) for array in multipliers):
        raise ValueError("CP3 requires non-empty train-O octets with 28 pairs each.")
    return CounterPriorWeights(
        multipliers_by_group=tuple(multipliers),
        counter_masks_by_group=tuple(counter_masks),
        train_o_counter_pair_rate=float(total_counter / total_pairs),
        counter_loss_mass_before=float(np.mean(before_mass)),
        counter_loss_mass_after_cp3=float(np.mean(after_mass)),
    )


def _group_arrays(
    group: Sequence[RankedImage], features: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences: list[np.ndarray] = []
    targets: list[float] = []
    distances: list[float] = []
    for offset, first in enumerate(group):
        for second in group[offset + 1 :]:
            lower, higher = (first, second) if first.rank < second.rank else (second, first)
            distance = higher.rank - lower.rank
            differences.append(
                np.asarray(features[higher.image_name], dtype=np.float64)
                - np.asarray(features[lower.image_name], dtype=np.float64)
            )
            targets.append(ranknet_target(distance))
            distances.append(float(distance))
    return np.stack(differences), np.asarray(targets), np.asarray(distances)


def build_cp3_linear_objective(
    train_o: Sequence[Sequence[RankedImage]],
    train_w: Sequence[Sequence[RankedImage]],
    features: Mapping[str, np.ndarray],
    counter_weights: CounterPriorWeights,
    *,
    l2: float = 10.0,
    options: Mapping[str, float | int],
) -> tuple[Callable[[np.ndarray], tuple[float, np.ndarray]], int]:
    """Exact O-CP3/W-standard objective with production distance weighting.

    The production group loss can be written as mean((distance / 3) * BCE)
    because an octet has 28 pairs and total rank distance 84. CP3 applies the
    specified multiplier-weighted mean to that same per-pair loss.
    """
    o_arrays = [_group_arrays(group, features) for group in train_o]
    w_arrays = [_group_arrays(group, features) for group in train_w]
    if len(o_arrays) != len(counter_weights.multipliers_by_group):
        raise ValueError("One CP3 multiplier vector is required per train-O group.")
    dimension = int(o_arrays[0][0].shape[1])

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        weight = np.asarray(vector, dtype=np.float64)
        o_loss = 0.0
        o_gradient = np.zeros(dimension, dtype=np.float64)
        for (differences, targets, distances), multiplier in zip(
            o_arrays, counter_weights.multipliers_by_group, strict=True
        ):
            logits = differences @ weight
            pair_weight = multiplier * distances / 3.0
            denominator = float(multiplier.sum())
            o_loss += float(np.sum(pair_weight * (np.logaddexp(0.0, logits) - targets * logits)) / denominator)
            o_gradient += differences.T @ (pair_weight * (expit(logits) - targets)) / denominator
        o_loss /= len(o_arrays)
        o_gradient /= len(o_arrays)

        w_loss = 0.0
        w_gradient = np.zeros(dimension, dtype=np.float64)
        for differences, targets, distances in w_arrays:
            logits = differences @ weight
            w_loss += float(np.sum(distances * (np.logaddexp(0.0, logits) - targets * logits)) / 84.0)
            w_gradient += differences.T @ (distances * (expit(logits) - targets)) / 84.0
        w_loss /= len(w_arrays)
        w_gradient /= len(w_arrays)
        data_loss = 0.5 * (o_loss + w_loss)
        gradient = 0.5 * (o_gradient + w_gradient)
        return float(data_loss + l2 * np.dot(weight, weight)), gradient + 2.0 * l2 * weight

    return objective, dimension


def fit_cp3_linear_ranknet(
    train_o: Sequence[Sequence[RankedImage]],
    train_w: Sequence[Sequence[RankedImage]],
    features: Mapping[str, np.ndarray],
    counter_weights: CounterPriorWeights,
    *,
    l2: float = 10.0,
) -> tuple[np.ndarray, GroupRankNetDiagnostics]:
    objective, dimension = build_cp3_linear_objective(
        train_o, train_w, features, counter_weights, l2=l2
    )
    initial = np.zeros(dimension, dtype=np.float64)
    loss_at_zero = float(objective(initial)[0])
    result: OptimizeResult = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options=dict(options),
    )
    weights = np.asarray(result.x, dtype=np.float64)
    final_loss, gradient = objective(weights)
    finite = bool(np.all(np.isfinite(weights)) and np.isfinite(final_loss) and np.all(np.isfinite(gradient)))
    diagnostics = GroupRankNetDiagnostics(
        final_loss=float(final_loss),
        loss_at_zero=loss_at_zero,
        gradient_l2=float(np.linalg.norm(gradient)),
        finite=finite,
        loss_not_worse_than_zero=bool(final_loss <= loss_at_zero),
        scipy_success=bool(result.success),
        optimizer_message=str(result.message),
        optimizer_status=int(result.status),
        optimizer_nit=int(result.nit),
        optimizer_nfev=int(result.nfev),
    )
    return weights, diagnostics
