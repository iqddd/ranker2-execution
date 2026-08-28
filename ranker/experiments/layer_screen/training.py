"""Provenance-valid simultaneous multi-layer training primitives."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

from ranker.experiments.layer_screen.simultaneous import LayerJointHead, MultiLayerScreen
from ranker.frozen_readouts import (
    FrozenFeatureRuntime,
    ReadoutSpec,
    StandardReadout,
    TrainingFold,
    configure_determinism,
    make_model,
)
from ranker.prodigy_guard import (
    neutral_schedulefree_evaluation,
    neutral_schedulefree_multi_evaluation,
)
from ranker.trainer_contract import CyclingGroupSampler, reliability_weighted_group_loss


def _tree_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
            left, right
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _tree_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _optimizer_group_state(
    optimizer: ProdigyPlusScheduleFree,
    group_index: int,
) -> dict[str, Any]:
    group = optimizer.param_groups[group_index]
    parameters = list(group["params"])
    return {
        "group": {
            key: value
            for key, value in group.items()
            if key not in {"params", "split_groups"}
        },
        "parameter_state": [optimizer.state[parameter] for parameter in parameters],
    }


@dataclass
class MultiLayerTrajectoryResult:
    checkpoint_scores: dict[int, np.ndarray]
    dynamics: list[dict[str, Any]]
    elapsed_seconds: float
    all_finite: bool


def trajectory_payload(result: MultiLayerTrajectoryResult) -> dict[str, Any]:
    return {
        "checkpoint_scores": {
            str(checkpoint): scores.tolist()
            for checkpoint, scores in result.checkpoint_scores.items()
        },
        "dynamics": result.dynamics,
        "elapsed_seconds": result.elapsed_seconds,
        "all_finite": result.all_finite,
    }


def trajectory_from_payload(payload: Mapping[str, Any]) -> MultiLayerTrajectoryResult:
    return MultiLayerTrajectoryResult(
        checkpoint_scores={
            int(checkpoint): np.asarray(scores, dtype=np.float32)
            for checkpoint, scores in payload["checkpoint_scores"].items()
        },
        dynamics=list(payload["dynamics"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        all_finite=bool(payload["all_finite"]),
    )


def _norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("Every multihead trainable parameter must have a gradient.")
        total.add_(parameter.grad.detach().double().square().sum())
    return float(torch.sqrt(total))


def _group_scalar(group: Mapping[str, Any], key: str) -> float:
    value = group.get(key, float("nan"))
    if isinstance(value, torch.Tensor):
        value = value.detach().double().mean().cpu().item()
    return float(value)


def _select_feature_rows(
    values: torch.Tensor, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    selected = indices.to(values.device)
    result = values.index_select(0, selected)
    return result if result.device == device else result.to(device)


def build_screen(
    *, template: torch.nn.Module, seed: int, device: torch.device
) -> tuple[MultiLayerScreen, torch.Tensor, list[torch.Tensor]]:
    """Build identical heads while preserving ordinary one-head RNG history."""
    configure_determinism(seed)
    ordinary = LayerJointHead(seed=seed, device=device)
    shared_state = {
        name: value.detach().clone() for name, value in ordinary.state_dict().items()
    }
    post_factory_cuda_rng = torch.cuda.get_rng_state()
    del ordinary
    screen = MultiLayerScreen(template=template, seed=seed, device=device)
    screen.load_shared_head_state(shared_state)
    torch.cuda.set_rng_state(post_factory_cuda_rng)
    rng_streams = [post_factory_cuda_rng.clone() for _ in screen.heads]
    return screen, post_factory_cuda_rng, rng_streams


def backend_matched_l26_first_update_parity(
    *,
    template: torch.nn.Module,
    residual_l10_cpu: torch.Tensor,
    runtime: FrozenFeatureRuntime,
    fold: TrainingFold,
    seed: int,
    capture_indices: torch.Tensor,
) -> dict[str, Any]:
    """Compare the L26 multihead arm with a one-group STEP60A-style trainer."""
    device = runtime.device
    spec = ReadoutSpec("L26_FINAL", "joint", 4)
    configure_determinism(seed)
    standalone = make_model(runtime, spec, seed)
    if not isinstance(standalone, StandardReadout):
        raise TypeError("Backend-matched L26 shadow requires StandardReadout.")
    screen, _post_factory_rng, head_rng = build_screen(
        template=template, seed=seed, device=device
    )
    l26 = screen.heads[-1]
    assert standalone.linear_weight is not None
    assert standalone.linear_bias is not None
    assert standalone.head is not None
    initial_parameters_exact = bool(
        torch.equal(standalone.linear_weight, l26.linear_weight)
        and torch.equal(standalone.linear_bias, l26.linear_bias)
        and all(
            torch.equal(left, right)
            for left, right in zip(
                standalone.head.parameters(), l26.attention.parameters(), strict=True
            )
        )
    )
    standalone_parameters = standalone.trainable_parameters()
    multi_groups = screen.head_parameter_groups()
    standalone_optimizer = ProdigyPlusScheduleFree(
        [{"params": standalone_parameters}], lr=1.0
    )
    multi_optimizers = [
        ProdigyPlusScheduleFree([group], lr=1.0) for group in multi_groups
    ]
    optimizer_state_before_exact = _tree_exact(
        _optimizer_group_state(standalone_optimizer, 0),
        _optimizer_group_state(multi_optimizers[16], 0),
    )
    screen.train(True)
    standalone.head.train(True)

    def capture_standalone(model: StandardReadout) -> torch.Tensor:
        assert model.head is not None
        was_training = bool(model.head.training)
        model.head.eval()
        rows: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(capture_indices), 8):
                selected = capture_indices[start : start + 8]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    attention, _ = model.head(
                        runtime.tokens.index_select(0, selected)
                    )
                rows.append(model._linear(selected) + attention)
        model.head.train(was_training)
        return torch.cat(rows)

    with neutral_schedulefree_evaluation(
        standalone_optimizer, standalone_parameters, model=standalone.head
    ):
        standalone_pre = capture_standalone(standalone).detach()
    with neutral_schedulefree_multi_evaluation(
        multi_optimizers,
        [group["params"] for group in multi_groups],
        model=screen,
    ):
        multi_pre = torch.from_numpy(
            capture_all_scores(
                screen,
                residual_l10_cpu,
                runtime.pooled,
                capture_indices,
                batch_size=8,
            )[-1]
        ).to(device)
    pre_scores_exact = bool(torch.equal(standalone_pre, multi_pre))

    standalone_optimizer.zero_grad(set_to_none=True)
    for optimizer in multi_optimizers:
        optimizer.zero_grad(set_to_none=True)
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)
    loss_rows: list[tuple[float, float]] = []
    rng_exact = True
    for source, coefficient, sampler, retained in (
        ("O", 0.75, o_sampler, fold.retained_o),
        ("W", 0.25, w_sampler, fold.retained_w),
    ):
        group_index = sampler.next()
        group = retained[group_index]
        indices = fold.group_indices[(source, group_index)]
        initial_rng = head_rng[-1].clone()
        group_residual = _select_feature_rows(residual_l10_cpu, indices, device)
        l26_tokens = screen.capture_l26_tokens(group_residual)
        torch.cuda.set_rng_state(initial_rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            attention_scores, _ = standalone.head(l26_tokens)
        standalone_scores = standalone._linear(indices) + attention_scores
        standalone_loss = reliability_weighted_group_loss(
            standalone_scores, group, source, pair_weighting="REL"
        )
        (coefficient * standalone_loss).backward()
        standalone_next_rng = torch.cuda.get_rng_state().clone()
        callbacks = [
            (
                lambda score, group=group, source=source: reliability_weighted_group_loss(
                    score, group, source, pair_weighting="REL"
                )
            )
            for _ in range(17)
        ]
        multi_losses = screen.score_and_backward(
            group_residual,
            runtime.pooled.index_select(0, indices),
            callbacks,
            coefficient,
            head_rng,
        )
        rng_exact &= torch.equal(standalone_next_rng, head_rng[-1])
        loss_rows.append((float(standalone_loss.detach()), multi_losses[-1]))

    loss_exact = all(left == right for left, right in loss_rows)
    l26_parameters = list(l26.parameters())
    gradients_exact = all(
        left.grad is not None
        and right.grad is not None
        and torch.equal(left.grad, right.grad)
        for left, right in zip(
            standalone_parameters, l26_parameters, strict=True
        )
    )
    standalone_optimizer.step()
    for optimizer in multi_optimizers:
        optimizer.step()
    parameters_after_exact = all(
        torch.equal(left, right)
        for left, right in zip(
            standalone_parameters, l26_parameters, strict=True
        )
    )
    optimizer_state_after_exact = _tree_exact(
        _optimizer_group_state(standalone_optimizer, 0),
        _optimizer_group_state(multi_optimizers[16], 0),
    )

    with neutral_schedulefree_evaluation(
        standalone_optimizer, standalone_parameters, model=standalone.head
    ):
        standalone_eval_parameters = [
            parameter.detach().clone() for parameter in standalone_parameters
        ]
        standalone_cp1 = capture_standalone(standalone).detach()
    with neutral_schedulefree_multi_evaluation(
        multi_optimizers,
        [group["params"] for group in multi_groups],
        model=screen,
    ):
        multi_eval_parameters = [
            parameter.detach().clone() for parameter in l26_parameters
        ]
        multi_cp1 = torch.from_numpy(
            capture_all_scores(
                screen,
                residual_l10_cpu,
                runtime.pooled,
                capture_indices,
                batch_size=8,
            )[-1]
        ).to(device)
    cp1_scores_exact = bool(torch.equal(standalone_cp1, multi_cp1))
    cp1_difference = (standalone_cp1 - multi_cp1).abs()
    eval_parameters_exact = all(
        torch.equal(left, right)
        for left, right in zip(
            standalone_eval_parameters, multi_eval_parameters, strict=True
        )
    )
    result = {
        "reference": "L26_EFFICIENT_SCAN_REFERENCE",
        "initial_parameters_exact": initial_parameters_exact,
        "pre_update_scores_exact": pre_scores_exact,
        "pre_update_losses_exact": loss_exact,
        "gradients_exact": gradients_exact,
        "optimizer_group_state_before_exact": optimizer_state_before_exact,
        "optimizer_group_state_after_exact": optimizer_state_after_exact,
        "parameters_after_update_exact": parameters_after_exact,
        "eval_space_parameters_exact": eval_parameters_exact,
        "checkpoint_1_scores_exact": cp1_scores_exact,
        "checkpoint_1_exact_score_count": int(
            torch.count_nonzero(cp1_difference == 0).item()
        ),
        "checkpoint_1_max_abs_difference": float(cp1_difference.max().item()),
        "dropout_rng_exact": bool(rng_exact),
        "shadow_comparisons": int(screen.shadow_comparisons),
    }
    result["pass"] = all(
        bool(value)
        for key, value in result.items()
        if key.endswith("_exact")
    )
    del standalone, screen, standalone_optimizer, multi_optimizers
    gc.collect()
    torch.cuda.empty_cache()
    return result


def capture_all_scores(
    screen: MultiLayerScreen,
    residual_l10_cpu: torch.Tensor,
    pooled: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int = 8,
) -> np.ndarray:
    """Capture all layer scores using one streamed suffix pass per image batch."""
    rows: list[torch.Tensor] = []
    device = pooled.device
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            residual = _select_feature_rows(residual_l10_cpu, selected, device)
            selected_cuda = selected.to(device)
            rows.append(
                screen.capture_scores(
                    residual,
                    pooled.index_select(0, selected_cuda),
                ).cpu()
            )
    return torch.cat(rows, dim=1).numpy().astype(np.float32)


def run_multihead_trajectory(
    *,
    template: torch.nn.Module,
    residual_l10_cpu: torch.Tensor,
    pooled: torch.Tensor,
    fold: TrainingFold,
    seed: int,
    capture_indices: torch.Tensor,
    checkpoints: Sequence[int],
    max_updates: int,
    tick: Callable[[], None] | None = None,
) -> MultiLayerTrajectoryResult:
    """Train 17 independent JOINT4 arms over one frozen suffix stream."""
    evaluation = set(map(int, checkpoints))
    if not evaluation or min(evaluation) != 0 or max(evaluation) != max_updates:
        raise ValueError("Checkpoint grid must contain 0 and max_updates.")
    device = pooled.device
    screen, _post_factory_rng, head_rng = build_screen(
        template=template, seed=seed, device=device
    )
    groups = screen.head_parameter_groups()
    parameters = screen.trainable_parameters()
    if len(groups) != 17 or len({id(value) for value in parameters}) != len(parameters):
        raise RuntimeError("Multihead optimizer topology is not 17 disjoint groups.")
    optimizers = [ProdigyPlusScheduleFree([group], lr=1.0) for group in groups]
    if len(optimizers) != 17 or any(
        len(optimizer.param_groups) != 1 for optimizer in optimizers
    ):
        raise RuntimeError("PPSF did not create 17 independent optimizers.")
    screen.train(True)
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)
    checkpoint_scores: dict[int, np.ndarray] = {}
    dynamics: list[dict[str, Any]] = []

    def evaluate(checkpoint: int) -> None:
        with neutral_schedulefree_multi_evaluation(
            optimizers,
            [group["params"] for group in groups],
            model=screen,
        ):
            values = capture_all_scores(
                screen, residual_l10_cpu, pooled, capture_indices, batch_size=8
            )
            if not np.all(np.isfinite(values)):
                raise RuntimeError("Multihead evaluation scores became non-finite.")
            checkpoint_scores[checkpoint] = values.copy()

    started = time.perf_counter()
    evaluate(0)
    for update in range(1, max_updates + 1):
        if tick is not None:
            tick()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        source_losses: dict[str, list[float]] = {}
        for source, source_weight, sampler, retained in (
            ("O", 0.75, o_sampler, fold.retained_o),
            ("W", 0.25, w_sampler, fold.retained_w),
        ):
            group_index = sampler.next()
            group = retained[group_index]
            indices = fold.group_indices[(source, group_index)]
            residual = _select_feature_rows(residual_l10_cpu, indices, device)
            selected_pooled = pooled.index_select(0, indices)
            callbacks = [
                (
                    lambda score, group=group, source=source: reliability_weighted_group_loss(
                        score, group, source, pair_weighting="REL"
                    )
                )
                for _ in range(17)
            ]
            source_losses[source] = screen.score_and_backward(
                residual,
                selected_pooled,
                callbacks,
                source_weight,
                head_rng,
            )
        norms = [_norm(list(group["params"])) for group in groups]
        if not all(math.isfinite(value) for value in norms):
            raise RuntimeError("Multihead gradients became non-finite.")
        for optimizer in optimizers:
            optimizer.step()
        if not all(bool(torch.isfinite(value).all()) for value in parameters):
            raise RuntimeError("Multihead parameters became non-finite.")
        if update in evaluation:
            evaluate(update)
            for layer_offset, optimizer in enumerate(optimizers):
                group = optimizer.param_groups[0]
                dynamics.append(
                    {
                        "update": update,
                        "layer": 10 + layer_offset,
                        "d": _group_scalar(group, "d"),
                        "effective_lr": _group_scalar(group, "lr"),
                        "gradient_norm": norms[layer_offset],
                        "training_O_loss": source_losses["O"][layer_offset],
                        "training_W_loss": source_losses["W"][layer_offset],
                    }
                )
    elapsed = time.perf_counter() - started
    gc.collect()
    torch.cuda.empty_cache()
    return MultiLayerTrajectoryResult(
        checkpoint_scores=checkpoint_scores,
        dynamics=dynamics,
        elapsed_seconds=elapsed,
        all_finite=True,
    )
