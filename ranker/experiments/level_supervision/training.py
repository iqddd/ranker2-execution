"""Batch-local theta-distillation trajectories with an FP64 gradient cap."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

from ranker.data import RankedImage
from ranker.frozen_readouts import (
    FrozenFeatureRuntime,
    ReadoutSpec,
    StandardReadout,
    TrainingFold,
    configure_determinism,
    load_readout_state,
    make_model,
)
from ranker.prodigy_guard import neutral_schedulefree_evaluation
from ranker.trainer_contract import CyclingGroupSampler, reliability_weighted_group_loss

from .targets import LCCTarget


@dataclass
class GradientCapTrajectoryResult:
    checkpoint_scores: dict[int, np.ndarray]
    checkpoint_metrics: dict[int, dict[str, float]]
    dynamics: dict[str, Any]
    elapsed_seconds: float
    all_finite: bool
    schedule_digest: str


def gradient_cap_result_payload(result: GradientCapTrajectoryResult) -> dict[str, Any]:
    return {
        "checkpoint_scores": {
            str(key): value.tolist() for key, value in result.checkpoint_scores.items()
        },
        "checkpoint_metrics": {
            str(key): value for key, value in result.checkpoint_metrics.items()
        },
        "dynamics": result.dynamics,
        "elapsed_seconds": result.elapsed_seconds,
        "all_finite": result.all_finite,
        "schedule_digest": result.schedule_digest,
    }


def gradient_cap_result_from_payload(
    payload: Mapping[str, Any],
) -> GradientCapTrajectoryResult:
    return GradientCapTrajectoryResult(
        checkpoint_scores={
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in payload["checkpoint_scores"].items()
        },
        checkpoint_metrics={
            int(key): {name: float(value) for name, value in values.items()}
            for key, values in payload["checkpoint_metrics"].items()
        },
        dynamics=dict(payload["dynamics"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        all_finite=bool(payload["all_finite"]),
        schedule_digest=str(payload["schedule_digest"]),
    )


def batch_local_theta_loss(
    o_scores: torch.Tensor,
    w_scores: torch.Tensor,
    o_group: Sequence[RankedImage],
    w_group: Sequence[RankedImage],
    target: LCCTarget,
) -> tuple[torch.Tensor | None, int, int]:
    """Return identity-pair-balanced ALL loss using only already-scored images."""
    losses: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    image_pairs: set[tuple[str, str]] = set()
    for o_index, o_item in enumerate(o_group):
        if o_item.image_name not in target.theta_by_image:
            continue
        for w_index, w_item in enumerate(w_group):
            if (
                w_item.image_name not in target.theta_by_image
                or o_item.identity == w_item.identity
                or o_item.image_name == w_item.image_name
            ):
                continue
            image_key = tuple(
                sorted((o_item.image_name, w_item.image_name), key=str.encode)
            )
            if image_key in image_pairs:
                raise RuntimeError(
                    "Duplicate admissible unordered image pair inside update."
                )
            image_pairs.add(image_key)
            identity_key = tuple(
                sorted((o_item.identity, w_item.identity), key=str.encode)
            )
            logit = o_scores[o_index] - w_scores[w_index]
            theta_delta = (
                target.theta_by_image[o_item.image_name]
                - target.theta_by_image[w_item.image_name]
            )
            q = torch.as_tensor(
                1.0 / (1.0 + math.exp(-theta_delta)),
                dtype=logit.dtype,
                device=logit.device,
            )
            losses[identity_key].append(F.binary_cross_entropy_with_logits(logit, q))
    if not losses:
        return None, 0, 0
    identity_losses = [torch.stack(values).mean() for values in losses.values()]
    return torch.stack(identity_losses).mean(), len(image_pairs), len(losses)


def _sum_squares(
    values: Sequence[torch.Tensor | None],
    *,
    fallback_device: torch.device | None = None,
) -> torch.Tensor:
    device = next(
        (value.device for value in values if value is not None), fallback_device
    )
    if device is None:
        raise ValueError("A fallback device is required for an all-absent gradient.")
    result = torch.zeros((), dtype=torch.float64, device=device)
    for value in values:
        if value is not None:
            result = result + value.detach().double().square().sum()
    return result


def combine_capped_gradients(
    primary: Sequence[torch.Tensor | None],
    auxiliary: Sequence[torch.Tensor | None],
    *,
    cap: float = 0.25,
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Combine detached gradients and enforce the preregistered FP64 gates."""
    if len(primary) != len(auxiliary) or not primary:
        raise ValueError("Gradient lists must be nonempty and aligned.")
    if any(value is None for value in primary):
        raise RuntimeError("Primary gradient is structurally absent.")
    primary_values = [value for value in primary if value is not None]
    primary_sq = _sum_squares(primary)
    auxiliary_sq = _sum_squares(auxiliary, fallback_device=primary_values[0].device)
    primary_norm = torch.sqrt(primary_sq)
    auxiliary_norm = torch.sqrt(auxiliary_sq)
    if not bool(torch.isfinite(primary_norm)) or not bool(primary_norm > 0):
        raise RuntimeError("Primary gradient norm is zero or non-finite.")
    if not bool(torch.isfinite(auxiliary_norm)):
        raise RuntimeError("Auxiliary gradient norm is non-finite.")
    if bool(auxiliary_norm == 0):
        coefficient = torch.zeros_like(primary_norm)
    else:
        coefficient = torch.minimum(
            torch.ones_like(primary_norm),
            cap * primary_norm / (auxiliary_norm + epsilon),
        )
    combined: list[torch.Tensor] = []
    dot = torch.zeros_like(primary_sq)
    for primary_value, auxiliary_value in zip(primary_values, auxiliary, strict=True):
        added = (
            primary_value.detach()
            if auxiliary_value is None
            else primary_value.detach()
            + coefficient.to(primary_value.dtype) * auxiliary_value.detach()
        )
        combined.append(added)
        dot = dot + added.double().mul(primary_value.detach().double()).sum()
    scaled_ratio = coefficient * auxiliary_norm / primary_norm
    alignment = dot / primary_sq
    if not bool(scaled_ratio <= cap + 1e-6):
        raise RuntimeError("Auxiliary gradient cap gate failed.")
    if not bool(alignment >= 0.75 - 1e-6):
        raise RuntimeError("Combined-primary alignment gate failed.")
    cosine = torch.tensor(float("nan"), dtype=torch.float64, device=primary_norm.device)
    if bool(auxiliary_norm > 0):
        raw_dot = torch.zeros_like(primary_sq)
        for primary_value, auxiliary_value in zip(
            primary_values, auxiliary, strict=True
        ):
            if auxiliary_value is not None:
                raw_dot = (
                    raw_dot
                    + primary_value.detach()
                    .double()
                    .mul(auxiliary_value.detach().double())
                    .sum()
                )
        cosine = raw_dot / (primary_norm * auxiliary_norm)
    return combined, {
        "primary_norm": float(primary_norm.item()),
        "auxiliary_norm": float(auxiliary_norm.item()),
        "unscaled_ratio": float((auxiliary_norm / primary_norm).item()),
        "a_t": float(coefficient.item()),
        "scaled_ratio": float(scaled_ratio.item()),
        "cosine": float(cosine.item()),
        "primary_alignment": float(alignment.item()),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            name: float("nan")
            for name in ("mean", "median", "minimum", "maximum", "P10", "P90")
        }
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "P10": float(np.quantile(finite, 0.10)),
        "P90": float(np.quantile(finite, 0.90)),
    }


def run_gradient_cap_trajectory(
    *,
    runtime: FrozenFeatureRuntime,
    fold: TrainingFold,
    target: LCCTarget,
    spec: ReadoutSpec,
    seed: int,
    initial_state: Mapping[str, Any],
    target_indices: torch.Tensor,
    capture_partitions: Sequence[torch.Tensor],
    evaluate_scores: Callable[[np.ndarray], Mapping[str, float]],
    checkpoints: Sequence[int],
    max_updates: int = 1024,
    tick: Callable[[], None] | None = None,
) -> GradientCapTrajectoryResult:
    """Train one JOINT4 readout with batch-local theta and a 0.25 gradient cap."""
    ordered = tuple(map(int, checkpoints))
    if not ordered or ordered[0] != 0 or ordered[-1] != max_updates:
        raise ValueError("Checkpoint grid must span 0..max_updates.")
    configure_determinism(seed)
    model = make_model(runtime, spec, seed)
    if not isinstance(model, StandardReadout):
        raise TypeError("Gradient-cap training requires StandardReadout.")
    load_readout_state(model, initial_state)
    parameters = model.trainable_parameters()
    named = list(model.named_parameters())
    optimizer = ProdigyPlusScheduleFree([{"params": parameters}], lr=1.0)
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("Expected one historical readout optimizer group.")
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)
    score_panel: dict[int, np.ndarray] = {}
    metric_panel: dict[int, dict[str, float]] = {}
    capture = set(ordered)
    schedule: list[tuple[int, int]] = []
    statistics: dict[str, list[float]] = defaultdict(list)
    partition_names = {
        "pooled_linear": {"linear.weight", "linear.bias"},
        "attention_output": {"attention.output.weight", "attention.output.bias"},
    }
    partition_indices = {
        key: [index for index, (name, _parameter) in enumerate(named) if name in names]
        for key, names in partition_names.items()
    }
    used = set(partition_indices["pooled_linear"]) | set(
        partition_indices["attention_output"]
    )
    partition_indices["remaining_attention_mlp"] = [
        index for index in range(len(named)) if index not in used
    ]
    if any(not indices for indices in partition_indices.values()):
        raise RuntimeError("Gradient diagnostic partitions are incomplete.")

    def evaluate(checkpoint: int) -> None:
        with neutral_schedulefree_evaluation(optimizer, parameters, model=model.head):
            with torch.no_grad():
                captured = torch.cat(
                    [
                        model.capture_scores(partition, partition)
                        for partition in capture_partitions
                    ]
                )
                values = captured.detach().cpu().numpy().astype(np.float64)
            metrics = {
                name: float(value) for name, value in evaluate_scores(values).items()
            }
            if not np.all(np.isfinite(values)) or not all(
                map(math.isfinite, metrics.values())
            ):
                raise RuntimeError("Gradient-cap validation became non-finite.")
            score_panel[checkpoint] = values.copy()
            metric_panel[checkpoint] = metrics

    started = time.perf_counter()
    model.train()
    evaluate(0)
    zero_auxiliary = 0
    total_pairs = 0
    total_identity_pairs = 0
    for update in range(1, max_updates + 1):
        if tick is not None:
            tick()
        o_index = o_sampler.next()
        w_index = w_sampler.next()
        schedule.append((o_index, w_index))
        o_group = fold.retained_o[o_index]
        w_group = fold.retained_w[w_index]
        o_scores = model.train_scores(fold.group_indices[("O", o_index)])
        w_scores = model.train_scores(fold.group_indices[("W", w_index)])
        o_loss = reliability_weighted_group_loss(
            o_scores, o_group, "O", pair_weighting="REL", normalization="group_weight"
        )
        w_loss = reliability_weighted_group_loss(
            w_scores, w_group, "W", pair_weighting="REL", normalization="group_weight"
        )
        primary_loss = 0.75 * o_loss + 0.25 * w_loss
        theta_loss, pair_count, identity_pair_count = batch_local_theta_loss(
            o_scores, w_scores, o_group, w_group, target
        )
        has_auxiliary = theta_loss is not None
        primary_gradients = torch.autograd.grad(
            primary_loss, parameters, retain_graph=has_auxiliary, create_graph=False
        )
        if theta_loss is None:
            auxiliary_gradients: tuple[torch.Tensor | None, ...] = tuple(
                None for _ in parameters
            )
            zero_auxiliary += 1
            theta_value = 0.0
        else:
            auxiliary_gradients = torch.autograd.grad(
                theta_loss,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            theta_value = float(theta_loss.detach())
        combined, geometry = combine_capped_gradients(
            primary_gradients, auxiliary_gradients
        )
        for parameter, gradient in zip(parameters, combined, strict=True):
            parameter.grad = gradient
        if not all(
            bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
        ):
            raise RuntimeError("Combined gradient became non-finite.")
        for partition, indices in partition_indices.items():
            p_norm = math.sqrt(
                sum(
                    float(primary_gradients[index].detach().double().square().sum())
                    for index in indices
                )
            )
            t_norm = math.sqrt(
                sum(
                    0.0
                    if auxiliary_gradients[index] is None
                    else float(
                        auxiliary_gradients[index].detach().double().square().sum()
                    )
                    for index in indices
                )
            )
            statistics[f"{partition}_primary_norm"].append(p_norm)
            statistics[f"{partition}_auxiliary_norm"].append(t_norm)
        for key, value in geometry.items():
            statistics[key].append(value)
        statistics["primary_loss"].append(float(primary_loss.detach()))
        statistics["theta_loss"].append(theta_value)
        statistics["pair_count"].append(float(pair_count))
        statistics["identity_pair_count"].append(float(identity_pair_count))
        total_pairs += pair_count
        total_identity_pairs += identity_pair_count
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise RuntimeError("Model parameter became non-finite.")
        if update in capture:
            evaluate(update)
    if set(score_panel) != capture:
        raise RuntimeError("Gradient-cap checkpoint capture is incomplete.")
    dynamics = {name: _summary(values) for name, values in statistics.items()}
    dynamics.update(
        {
            "identity_id": fold.identity,
            "seed": seed,
            "updates": max_updates,
            "zero_auxiliary_updates": zero_auxiliary,
            "nonzero_auxiliary_updates": max_updates - zero_auxiliary,
            "total_pseudo_pairs": total_pairs,
            "total_identity_pair_occurrences": total_identity_pairs,
            "minimum_scaled_ratio_gate": min(statistics["scaled_ratio"]),
            "maximum_scaled_ratio_gate": max(statistics["scaled_ratio"]),
            "minimum_primary_alignment_gate": min(statistics["primary_alignment"]),
        }
    )
    schedule_digest = hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode()
    ).hexdigest()
    elapsed = time.perf_counter() - started
    # Keep closure-bound model/optimizer names valid through function return;
    # normal frame teardown releases them after the result is constructed.
    del named, primary_gradients, auxiliary_gradients, combined
    gc.collect()
    torch.cuda.empty_cache()
    return GradientCapTrajectoryResult(
        checkpoint_scores=score_panel,
        checkpoint_metrics=metric_panel,
        dynamics=dynamics,
        elapsed_seconds=elapsed,
        all_finite=True,
        schedule_digest=schedule_digest,
    )
