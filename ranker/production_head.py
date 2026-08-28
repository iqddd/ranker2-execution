"""Reusable fixed production-head fitting and calibration primitives."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ranker.attention_evaluation import (
    forward_head_deltas,
    physically_calibrate_attention_head,
)
from ranker.attention_runtime import RuntimeFold
from ranker.attention_training import (
    corrected_head,
    head_optimizer,
    measure_head_train_loss,
    train_head_pass,
    training_state_finite,
)
from ranker.data import RankedImage
from ranker.group_ranknet import fit_group_ranknet, score_feature_map
from ranker.reproducibility import seed_everything


@dataclass(frozen=True)
class BaseS2Model:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    scores: dict[str, float]
    diagnostics: dict[str, Any]


def fit_base_s2_model(
    *,
    train_o: Sequence[Sequence[RankedImage]],
    train_w: Sequence[Sequence[RankedImage]],
    raw_features: Mapping[str, np.ndarray],
    score_names: Sequence[str],
    solver_options: Mapping[str, float | int],
    l2: float = 10.0,
) -> BaseS2Model:
    """Fit the leakage-safe production S2 O/W objective with train-only scaling."""
    train_names = sorted(
        {
            item.image_name
            for group in (*tuple(train_o), *tuple(train_w))
            for item in group
        }
    )
    matrix = np.stack(
        [np.asarray(raw_features[name], dtype=np.float64) for name in train_names]
    )
    mean = matrix.mean(axis=0)
    std = np.maximum(matrix.std(axis=0, ddof=0), 1e-8)
    required = sorted(set(train_names) | set(score_names))
    standardized = {
        name: (np.asarray(raw_features[name], dtype=np.float64) - mean) / std
        for name in required
    }
    weights, diagnostic = fit_group_ranknet(
        train_o,
        train_w,
        [standardized],
        [l2],
        balanced_ow=True,
        options=solver_options,
    )
    scores = score_feature_map(standardized, weights)
    report = {
        **diagnostic.as_dict(),
        "n_train_unique_images": len(train_names),
        "all_parameters_finite": bool(np.all(np.isfinite(weights))),
        "all_scores_finite": all(math.isfinite(value) for value in scores.values()),
    }
    report["pass"] = bool(
        report["all_parameters_finite"]
        and report["all_scores_finite"]
        and float(report["final_loss"]) <= float(report["loss_at_zero"])
        and float(report["gradient_l2"]) <= 5e-6
    )
    return BaseS2Model(weights, mean, std, scores, report)


def apply_base_s2_model(
    model: BaseS2Model, raw_features: Mapping[str, np.ndarray]
) -> dict[str, float]:
    standardized = {
        name: (np.asarray(value, dtype=np.float64) - model.feature_mean)
        / model.feature_std
        for name, value in raw_features.items()
    }
    return score_feature_map(standardized, model.weights)


def train_fixed_production_head(
    *,
    runtime: RuntimeFold,
    frozen_tokens: torch.Tensor,
    seed: int,
    learning_rate: float,
    epochs: int,
    initial_query_norm: float,
    device: torch.device,
    after_epoch: Callable[[int], None] | None = None,
) -> tuple[torch.nn.Module, np.ndarray, dict[str, Any]]:
    """Train, physically calibrate, and replay-check one fixed head trajectory."""
    seed_everything(seed)
    head = corrected_head(seed, device, initial_query_norm=initial_query_norm)
    optimizer = head_optimizer(head, learning_rate)
    initial_train_losses = measure_head_train_loss(
        head=head,
        runtime=runtime,
        frozen_tokens=frozen_tokens,
        epoch_seed=seed,
    )
    zero_delta, _ = forward_head_deltas(
        head=head,
        tokens=frozen_tokens,
        indices=runtime.validation_cache_indices,
    )
    zero_difference = zero_delta
    zero_parity = {
        "max_abs_score_difference": float(np.max(np.abs(zero_difference))),
        "mean_abs_score_difference": float(np.mean(np.abs(zero_difference))),
        "pass": bool(np.max(np.abs(zero_difference)) <= 1e-7),
    }
    training_seconds = 0.0
    last_losses: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        losses, gradient = train_head_pass(
            head=head,
            optimizer=optimizer,
            runtime=runtime,
            frozen_tokens=frozen_tokens,
            learning_rate=learning_rate,
            epoch_seed=seed + epoch,
        )
        training_seconds += time.perf_counter() - started
        last_losses = {**losses, "gradient_l2_before_clip": gradient}
        if not all(
            math.isfinite(float(value)) for value in last_losses.values()
        ) or not training_state_finite(head, optimizer):
            raise RuntimeError("Fixed production-head trajectory became non-finite.")
        if after_epoch is not None:
            after_epoch(epoch)
    final_train_losses = measure_head_train_loss(
        head=head,
        runtime=runtime,
        frozen_tokens=frozen_tokens,
        epoch_seed=seed + epochs + 1,
    )
    calibration = physically_calibrate_attention_head(
        head=head,
        runtime=runtime,
        frozen_tokens=frozen_tokens,
    )
    heldout_delta, _ = forward_head_deltas(
        head=head,
        tokens=frozen_tokens,
        indices=runtime.validation_cache_indices,
    )
    heldout_scores = runtime.fold.validation_base_scores + heldout_delta
    record = {
        "seed": seed,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "gamma": 1.0,
        "zero_initialization_parity": zero_parity,
        "calibration_parity": calibration,
        "last_train_metrics": last_losses,
        "initial_train_metrics": initial_train_losses,
        "final_train_metrics": final_train_losses,
        "train_loss_not_worse_than_initial": bool(
            final_train_losses["train_balanced_loss"]
            <= initial_train_losses["train_balanced_loss"]
        ),
        "finite": bool(np.all(np.isfinite(heldout_scores))),
        "training_seconds": training_seconds,
    }
    return head, heldout_scores, record
