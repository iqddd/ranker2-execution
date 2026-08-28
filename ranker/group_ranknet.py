"""Float64 group-balanced RankNet fitting shared by strict ranking experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit

from .data import RankedImage


@dataclass(frozen=True)
class GroupRankNetDiagnostics:
    final_loss: float
    loss_at_zero: float
    gradient_l2: float
    finite: bool
    loss_not_worse_than_zero: bool
    scipy_success: bool
    optimizer_message: str
    optimizer_status: int
    optimizer_nit: int
    optimizer_nfev: int

    @property
    def certificate_success(self) -> bool:
        return bool(
            self.finite
            and self.loss_not_worse_than_zero
            and self.gradient_l2 <= 1e-6
        )

    def as_dict(self) -> dict[str, float | int | bool | str]:
        result = asdict(self)
        result["certificate_success"] = self.certificate_success
        return result

    def __getitem__(self, key: str) -> float | int | bool | str:
        """Temporary mapping ergonomics for step artifact adapters."""
        return self.as_dict()[key]


def ranknet_target(distance: int) -> float:
    return 0.5 + 0.45 * ((distance / 7.0) ** 0.7)


def build_ranknet_pair_arrays(
    groups: Sequence[Sequence[RankedImage]],
    feature_block: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences: list[np.ndarray] = []
    targets: list[float] = []
    distances: list[float] = []
    for group in groups:
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (first, second) if first.rank < second.rank else (second, first)
                distance = higher.rank - lower.rank
                differences.append(
                    np.asarray(feature_block[higher.image_name], dtype=np.float64)
                    - np.asarray(feature_block[lower.image_name], dtype=np.float64)
                )
                targets.append(ranknet_target(distance))
                distances.append(float(distance))
    if not differences:
        raise ValueError("RankNet requires at least one ranked group.")
    return np.stack(differences), np.asarray(targets), np.asarray(distances)


# Backwards-compatible internal spelling for the O/W objective below.
_pairs = build_ranknet_pair_arrays


def build_linear_ranknet_objective(
    pair_features: np.ndarray,
    targets: np.ndarray,
    pair_weights: np.ndarray,
    l2: float,
    *,
    normalization: str,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Build the float64 linear RankNet loss with an explicit loss scale."""
    x = np.asarray(pair_features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(pair_weights, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or weights.shape != (len(x),):
        raise ValueError("RankNet pair arrays have incompatible shapes.")
    if normalization == "mean_pair":
        denominator = float(len(y))
    elif normalization == "group_weight":
        # The group-balanced callers use build_group_ranknet_objective, where
        # each octet contributes a total rank distance of 84.
        denominator = float(weights.sum())
    else:
        raise ValueError("normalization must be 'mean_pair' or 'group_weight'.")
    if denominator <= 0.0:
        raise ValueError("RankNet requires at least one non-zero-weight pair.")

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ vector
        loss = np.sum(weights * (np.logaddexp(0.0, logits) - y * logits)) / denominator
        gradient = x.T @ (weights * (expit(logits) - y)) / denominator
        return float(loss + l2 * np.dot(vector, vector)), np.asarray(gradient + 2.0 * l2 * vector, dtype=np.float64)

    return objective


def fit_linear_ranknet(
    groups: Sequence[Sequence[RankedImage]],
    features: Mapping[str, np.ndarray],
    l2: float,
    *, normalization: str = "mean_pair",
    options: Mapping[str, float | int],
) -> tuple[np.ndarray, GroupRankNetDiagnostics]:
    pairs, targets, weights = build_ranknet_pair_arrays(groups, features)
    objective = build_linear_ranknet_objective(pairs, targets, weights, l2, normalization=normalization)
    initial = np.zeros(pairs.shape[1], dtype=np.float64)
    loss_at_zero = float(objective(initial)[0])
    result: OptimizeResult = minimize(
        objective, initial, method="L-BFGS-B", jac=True, options=dict(options)
    )
    vector = np.asarray(result.x, dtype=np.float64)
    final_loss, gradient = objective(vector)
    return vector, GroupRankNetDiagnostics(
        final_loss=float(final_loss), loss_at_zero=loss_at_zero,
        gradient_l2=float(np.linalg.norm(gradient)),
        finite=bool(np.all(np.isfinite(vector)) and np.isfinite(final_loss) and np.all(np.isfinite(gradient))),
        loss_not_worse_than_zero=bool(final_loss <= loss_at_zero), scipy_success=bool(result.success),
        optimizer_message=str(result.message), optimizer_status=int(result.status),
        optimizer_nit=int(result.nit), optimizer_nfev=int(result.nfev),
    )


def score_feature_map(features: Mapping[str, np.ndarray], weights: np.ndarray) -> dict[str, float]:
    return {name: float(np.asarray(vector, dtype=np.float64) @ weights) for name, vector in features.items()}


def score_feature_blocks(
    feature_blocks: Sequence[Mapping[str, np.ndarray]], weights: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if not feature_blocks:
        raise ValueError("At least one feature block is required.")
    sizes = [len(np.asarray(next(iter(block.values())))) for block in feature_blocks]
    if sum(sizes) != len(weights):
        raise ValueError("Weight dimension does not match feature blocks.")
    offsets = np.cumsum([0, *sizes])
    maps = [score_feature_map(block, weights[offsets[index]:offsets[index + 1]]) for index, block in enumerate(feature_blocks)]
    names = maps[0].keys()
    return {name: float(sum(scores[name] for scores in maps)) for name in names}, maps


def build_group_ranknet_objective(
    train_o: Sequence[Sequence[RankedImage]],
    train_w: Sequence[Sequence[RankedImage]],
    feature_blocks: Sequence[Mapping[str, np.ndarray]],
    l2_values: Sequence[float],
    *,
    balanced_ow: bool,
) -> tuple[Callable[[np.ndarray], tuple[float, np.ndarray]], int]:
    """Build the exact per-group O/W-balanced RankNet objective and gradient."""
    if len(feature_blocks) != len(l2_values) or not feature_blocks:
        raise ValueError("Every feature block requires exactly one L2 value.")

    o_arrays = [_pairs(train_o, block) for block in feature_blocks]
    o_targets = o_arrays[0][1]
    o_distances = o_arrays[0][2]
    if any(not np.array_equal(targets, o_targets) or not np.array_equal(distances, o_distances) for _, targets, distances in o_arrays[1:]):
        raise AssertionError("Feature blocks produced different O pair ordering.")

    w_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    if balanced_ow:
        w_arrays = [_pairs(train_w, block) for block in feature_blocks]
        w_targets = w_arrays[0][1]
        w_distances = w_arrays[0][2]
        if any(not np.array_equal(targets, w_targets) or not np.array_equal(distances, w_distances) for _, targets, distances in w_arrays[1:]):
            raise AssertionError("Feature blocks produced different W pair ordering.")
    else:
        w_targets = w_distances = None

    block_sizes = [array[0].shape[1] for array in o_arrays]
    offsets = np.cumsum([0, *block_sizes])
    def mean_group_pair_coefficients(
        groups: Sequence[Sequence[RankedImage]], distances: np.ndarray
    ) -> np.ndarray:
        coefficients: list[np.ndarray] = []
        start = 0
        for group in groups:
            pair_count = len(group) * (len(group) - 1) // 2
            current = np.asarray(distances[start : start + pair_count], dtype=np.float64)
            if len(current) != pair_count or current.sum() <= 0.0:
                raise ValueError("Every RankNet group must contain weighted pairs.")
            coefficients.append(current / (len(groups) * current.sum()))
            start += pair_count
        if start != len(distances):
            raise AssertionError("Group/pair distance boundaries are inconsistent.")
        return np.concatenate(coefficients)

    o_coefficients = mean_group_pair_coefficients(train_o, o_distances)
    w_coefficients = (
        mean_group_pair_coefficients(train_w, w_distances)
        if balanced_ow and w_distances is not None
        else None
    )
    if not len(train_o) or (balanced_ow and not len(train_w)):
        raise ValueError("Each active loss split must contain at least one group.")

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        weights = [np.asarray(vector[offsets[index] : offsets[index + 1]], dtype=np.float64) for index in range(len(block_sizes))]
        o_logits = sum(array[0] @ weight for array, weight in zip(o_arrays, weights, strict=True))
        o_residual = o_coefficients * (expit(o_logits) - o_targets)
        data_loss = np.sum(
            o_coefficients * (np.logaddexp(0.0, o_logits) - o_targets * o_logits)
        )
        gradients = [array[0].T @ o_residual for array in o_arrays]

        if balanced_ow:
            assert w_arrays is not None and w_targets is not None and w_distances is not None
            w_logits = sum(array[0] @ weight for array, weight in zip(w_arrays, weights, strict=True))
            assert w_coefficients is not None
            w_residual = w_coefficients * (expit(w_logits) - w_targets)
            w_loss = np.sum(
                w_coefficients * (np.logaddexp(0.0, w_logits) - w_targets * w_logits)
            )
            data_loss = 0.5 * (data_loss + w_loss)
            gradients = [0.5 * (gradient + array[0].T @ w_residual) for gradient, array in zip(gradients, w_arrays, strict=True)]

        regularization = sum(float(l2) * np.dot(weight, weight) for l2, weight in zip(l2_values, weights, strict=True))
        gradients = [gradient + 2.0 * float(l2) * weight for gradient, l2, weight in zip(gradients, l2_values, weights, strict=True)]
        return float(data_loss + regularization), np.concatenate(gradients)

    return objective, int(offsets[-1])


def fit_group_ranknet(
    train_o: Sequence[Sequence[RankedImage]],
    train_w: Sequence[Sequence[RankedImage]],
    feature_blocks: Sequence[Mapping[str, np.ndarray]],
    l2_values: Sequence[float],
    *,
    balanced_ow: bool,
    initial_weights: np.ndarray | None = None,
    options: Mapping[str, float | int],
) -> tuple[np.ndarray, GroupRankNetDiagnostics]:
    objective, dimension = build_group_ranknet_objective(
        train_o, train_w, feature_blocks, l2_values, balanced_ow=balanced_ow
    )
    initial = (
        np.zeros(dimension, dtype=np.float64)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=np.float64).copy()
    )
    if initial.shape != (dimension,) or not np.all(np.isfinite(initial)):
        raise ValueError(f"Initial weights must be a finite vector of shape ({dimension},).")
    loss_at_zero = float(objective(np.zeros(dimension, dtype=np.float64))[0])
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
