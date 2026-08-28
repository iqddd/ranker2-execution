"""Reusable Prodigy Schedule-Free last-block trajectory primitives."""

from __future__ import annotations

import copy
import gc
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

from ranker.frozen_readouts import (
    FrozenFeatureRuntime,
    ReadoutSpec,
    StandardReadout,
    TrainingFold,
    configure_determinism,
    export_readout_state,
    load_readout_state,
    make_model,
)
from ranker.io import atomic_torch, sha256_file
from ranker.last_block_finetune import (
    ExactPrefixCache,
    LastBlockCandidate,
    LastBlockReadout,
    capture_scores,
)
from ranker.trainer_contract import (
    CyclingGroupSampler,
    factorial_group_loss,
    reliability_weighted_group_loss,
)


@dataclass
class ProdigyTrajectoryResult:
    """Complete replayable result of one optimizer trajectory."""

    checkpoint_scores: dict[int, np.ndarray]
    checkpoint_metrics: dict[int, dict[str, float]]
    log_rows: list[dict[str, Any]]
    snapshot_rows: list[dict[str, Any]]
    elapsed_seconds: float
    all_finite: bool
    state: dict[str, Any] | None = None
    eval_state: dict[str, Any] | None = None


def trajectory_result_payload(result: ProdigyTrajectoryResult) -> dict[str, Any]:
    """Convert a Prodigy trajectory result into a journal-safe payload."""
    return {
        "checkpoint_scores": {
            str(key): value.tolist() for key, value in result.checkpoint_scores.items()
        },
        "checkpoint_metrics": {
            str(key): value for key, value in result.checkpoint_metrics.items()
        },
        "log_rows": result.log_rows,
        "snapshot_rows": result.snapshot_rows,
        "elapsed_seconds": result.elapsed_seconds,
        "all_finite": result.all_finite,
    }


def trajectory_result_from_payload(
    payload: Mapping[str, Any],
) -> ProdigyTrajectoryResult:
    """Restore a trajectory result and verify all journaled snapshot files."""
    from ranker.io import sha256_file

    snapshots = list(payload["snapshot_rows"])
    for row in snapshots:
        path = Path(str(row["path"]))
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError("Journaled Prodigy snapshot is missing or changed.")
    return ProdigyTrajectoryResult(
        checkpoint_scores={
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in payload["checkpoint_scores"].items()
        },
        checkpoint_metrics={
            int(key): {name: float(value) for name, value in values.items()}
            for key, values in payload["checkpoint_metrics"].items()
        },
        log_rows=list(payload["log_rows"]),
        snapshot_rows=snapshots,
        elapsed_seconds=float(payload["elapsed_seconds"]),
        all_finite=bool(payload["all_finite"]),
    )


def run_frozen_prodigy_trajectory(
    *,
    runtime: FrozenFeatureRuntime,
    fold: TrainingFold,
    spec: ReadoutSpec,
    seed: int,
    initial_state: Mapping[str, Any],
    pair_weighting: str,
    o_coefficient: float,
    w_coefficient: float,
    loss_normalization: str = "group_weight",
    loss_normalization_scale: float | None = None,
    target_kind: str = "HARD",
    target_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    capture_partitions: Sequence[torch.Tensor] | None = None,
    evaluate_scores: Callable[[np.ndarray], Mapping[str, float]],
    evaluation_checkpoints: Sequence[int],
    logging_updates: Sequence[int],
    snapshot_checkpoints: Sequence[int],
    snapshot_directory: Path | None,
    snapshot_include_optimizer: bool = True,
    max_updates: int = 1024,
    retain_state: bool = False,
    retain_eval_state_via_clone: bool = False,
    diagnostic_parameter_groups: Mapping[str, Sequence[str]] | None = None,
    tick: Callable[[], None] | None = None,
) -> ProdigyTrajectoryResult:
    """Train one fresh frozen-encoder readout with default Schedule-Free Prodigy."""
    if runtime.device.type != "cuda":
        raise RuntimeError("Frozen Prodigy trajectories require CUDA.")
    supported_readout = (
        (spec.mode == "linear" and spec.queries == 0)
        or (spec.mode in {"main", "joint"} and spec.queries == 4)
    )
    if not supported_readout:
        raise RuntimeError(
            "Frozen Prodigy supports LINEAR or fresh MAIN4/JOINT4 readouts only."
        )
    legacy_contract = (
        target_kind == "HARD"
        and pair_weighting == "REL"
        and (o_coefficient, w_coefficient) == (0.75, 0.25)
    )
    factorial_contract = (
        target_kind in {"HARD", "SOFT"}
        and pair_weighting in {"REL", "DIST"}
        and (o_coefficient, w_coefficient) in {(0.75, 0.25), (0.5, 0.5)}
    )
    if not (legacy_contract or factorial_contract):
        raise RuntimeError("Unsupported frozen Prodigy loss contract.")
    if loss_normalization not in {
        "group_weight",
        "mean_pair",
        "mean_pair_unit_mass",
    }:
        raise RuntimeError("Unsupported frozen Prodigy loss normalization.")
    effective_loss_normalization = (
        "mean_pair"
        if loss_normalization == "mean_pair_unit_mass"
        else loss_normalization
    )
    if loss_normalization == "mean_pair_unit_mass":
        if loss_normalization_scale is None:
            raise RuntimeError(
                "mean_pair_unit_mass requires a precommitted fold scale"
            )
        loss_scale = float(loss_normalization_scale)
        if not math.isfinite(loss_scale) or loss_scale <= 0.0:
            raise ValueError("loss_normalization_scale must be finite and positive")
    else:
        if loss_normalization_scale is not None:
            raise RuntimeError(
                "loss_normalization_scale is valid only for mean_pair_unit_mass"
            )
        loss_scale = 1.0
    evaluation = set(map(int, evaluation_checkpoints))
    logging = set(map(int, logging_updates))
    snapshots = set(map(int, snapshot_checkpoints))
    if max_updates < 1 or max_updates > 1024:
        raise ValueError("Frozen Prodigy max_updates must be inside [1, 1024].")
    if not evaluation or max(evaluation) != max_updates or min(evaluation) < 0:
        raise ValueError("Frozen Prodigy evaluation must include max_updates.")
    if snapshots - evaluation:
        raise ValueError("Snapshots require a matching evaluation checkpoint.")
    if snapshots and snapshot_directory is None:
        raise ValueError("Snapshot checkpoints require a scratch directory.")

    configure_determinism(seed)
    model = make_model(runtime, spec, seed)
    if not isinstance(model, StandardReadout):
        raise TypeError("Fresh frozen Prodigy requires StandardReadout.")
    load_readout_state(model, initial_state)
    parameters = model.trainable_parameters()
    if not parameters:
        raise RuntimeError("Frozen Prodigy readout parameter group is empty.")
    optimizer = ProdigyPlusScheduleFree([{"params": parameters}], lr=1.0)
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("Frozen Prodigy did not retain exactly one parameter group.")
    named_parameters = dict(model.named_parameters())
    diagnostic_groups: dict[str, list[torch.nn.Parameter]] = {}
    diagnostic_group_names: dict[str, list[str]] = {}
    if diagnostic_parameter_groups is not None:
        seen: set[str] = set()
        for group_name, parameter_names in diagnostic_parameter_groups.items():
            names = list(parameter_names)
            missing = set(names) - set(named_parameters)
            overlap = seen.intersection(names)
            if missing or overlap or not names:
                raise ValueError(
                    f"Invalid diagnostic group {group_name}: missing={sorted(missing)}, "
                    f"overlap={sorted(overlap)}, count={len(names)}"
                )
            diagnostic_groups[group_name] = [named_parameters[name] for name in names]
            diagnostic_group_names[group_name] = names
            seen.update(names)
        if seen != set(named_parameters):
            raise ValueError("Diagnostic parameter groups must partition the readout.")
    initial_parameters = {
        name: parameter.detach().float().clone()
        for name, parameter in named_parameters.items()
    }
    cumulative_update_l2 = {name: 0.0 for name in diagnostic_groups}
    cumulative_update_rms = {name: 0.0 for name in diagnostic_groups}
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)
    checkpoint_scores: dict[int, np.ndarray] = {}
    checkpoint_metrics: dict[int, dict[str, float]] = {}
    snapshot_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    def evaluate(checkpoint: int) -> np.ndarray:
        with neutral_schedulefree_evaluation(optimizer, parameters, model=model.head):
            with torch.no_grad():
                if capture_partitions is None:
                    captured = model.capture_scores(target_indices, reference_positions)
                else:
                    captured = torch.cat(
                        [model.capture_scores(partition, partition) for partition in capture_partitions]
                    )
                values = captured.detach().cpu().numpy().astype(np.float64)
            metrics = {
                name: float(value) for name, value in evaluate_scores(values).items()
            }
            if not np.all(np.isfinite(values)) or not all(
                math.isfinite(value) for value in metrics.values()
            ):
                raise RuntimeError("Frozen Prodigy validation became non-finite.")
            checkpoint_scores[checkpoint] = values
            checkpoint_metrics[checkpoint] = metrics
            if checkpoint in snapshots:
                assert snapshot_directory is not None
                snapshot_directory.mkdir(parents=True, exist_ok=True)
                path = snapshot_directory / f"checkpoint={checkpoint}.pt"
                snapshot_payload = {
                        "candidate_id": spec.candidate_id,
                        "fold": fold.identity,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "optimizer_mode": "eval",
                        "neutral_train_state_restoration": "exact_after_callback",
                        "readout_state": export_readout_state(model),
                        "scores": values,
                        "metrics": metrics,
                    }
                if snapshot_include_optimizer:
                    snapshot_payload["optimizer_state"] = optimizer.state_dict()
                atomic_torch(path, snapshot_payload)
                snapshot_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                    }
                )
        return values

    started = time.perf_counter()
    model.train()
    if 0 in evaluation:
        evaluate(0)
    all_finite = True
    for update in range(1, max_updates + 1):
        if tick is not None:
            tick()
        optimizer.zero_grad(set_to_none=True)
        o_index = o_sampler.next()
        w_index = w_sampler.next()
        o_scores = model.train_scores(fold.group_indices[("O", o_index)])
        o_loss = (
            reliability_weighted_group_loss(
                o_scores,
                fold.retained_o[o_index],
                "O",
                pair_weighting="REL",
                normalization=effective_loss_normalization,
            )
            if legacy_contract
            else factorial_group_loss(
                o_scores, fold.retained_o[o_index], "O",
                target_kind=target_kind, pair_weighting=pair_weighting,
            )
        )
        (loss_scale * o_coefficient * o_loss).backward()
        w_scores = model.train_scores(fold.group_indices[("W", w_index)])
        w_loss = (
            reliability_weighted_group_loss(
                w_scores,
                fold.retained_w[w_index],
                "W",
                pair_weighting="REL",
                normalization=effective_loss_normalization,
            )
            if legacy_contract
            else factorial_group_loss(
                w_scores, fold.retained_w[w_index], "W",
                target_kind=target_kind, pair_weighting=pair_weighting,
            )
        )
        (loss_scale * w_coefficient * w_loss).backward()
        gradient_norm = _gradient_norm(parameters)
        all_finite = all_finite and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        before = (
            [value.detach().float().clone() for value in parameters]
            if update in logging
            else None
        )
        optimizer.step()
        all_finite = all_finite and all(
            bool(torch.isfinite(parameter).all()) for parameter in parameters
        )
        if not all_finite or not math.isfinite(float(o_loss.detach()) + float(w_loss.detach())):
            raise RuntimeError(f"{spec.candidate_id} frozen Prodigy became non-finite.")
        metrics: Mapping[str, float] = {}
        if update in evaluation:
            evaluate(update)
            metrics = checkpoint_metrics[update]
        if update in logging:
            assert before is not None
            group = optimizer.param_groups[0]
            d = _float_group_value(group, "d")
            effective_lr = _float_group_value(group, "effective_lr")
            if diagnostic_groups:
                before_by_id = {
                    id(parameter): reference
                    for parameter, reference in zip(parameters, before)
                }
                for diagnostic_name, diagnostic_parameters in diagnostic_groups.items():
                    diagnostic_before = [
                        before_by_id[id(parameter)] for parameter in diagnostic_parameters
                    ]
                    update_values = [
                        parameter.detach().float() - reference
                        for parameter, reference in zip(
                            diagnostic_parameters, diagnostic_before
                        )
                    ]
                    initial_values = [
                        named_parameters[name].detach().float()
                        - initial_parameters[name]
                        for name in diagnostic_group_names[diagnostic_name]
                    ]
                    update_l2 = _l2_norm(update_values)
                    update_rms = _rms(update_values)
                    cumulative_update_l2[diagnostic_name] += update_l2
                    cumulative_update_rms[diagnostic_name] += update_rms
                    log_rows.append(
                        {
                            "candidate_id": spec.candidate_id,
                            "identity_id": fold.identity,
                            "seed": seed,
                            "update": update,
                            "parameter_group": diagnostic_name,
                            "optimizer_parameter_group": "shared_readout",
                            "d": d,
                            "effective_lr": effective_lr,
                            "d_times_effective_lr": d * effective_lr,
                            "gradient_RMS": _rms(
                                [parameter.grad for parameter in diagnostic_parameters]
                            ),
                            "actual_parameter_update_RMS": update_rms,
                            "actual_parameter_update_L2": update_l2,
                            "cumulative_actual_update_RMS": cumulative_update_rms[
                                diagnostic_name
                            ],
                            "cumulative_actual_update_L2": cumulative_update_l2[
                                diagnostic_name
                            ],
                            "distance_from_initialization_RMS": _rms(initial_values),
                            "train_O_loss": float(o_loss.detach()),
                            "train_W_loss": float(w_loss.detach()),
                            "validation_IB_LTE10": metrics.get("IB_LTE10"),
                            "validation_IB_FrameLTE10": metrics.get("IB_FrameLTE10"),
                            "validation_IdentityOffsetMAE": metrics.get("IdentityOffsetMAE"),
                            "validation_status": "eligible" if update in evaluation else "not_evaluated",
                        }
                    )
            else:
                log_rows.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "identity_id": fold.identity,
                        "seed": seed,
                        "update": update,
                        "parameter_group": "readout",
                        "d": d,
                        "effective_lr": effective_lr,
                        "d_times_effective_lr": d * effective_lr,
                        "parameter_norm": _l2_norm(
                            [value.detach() for value in parameters]
                        ),
                        "update_norm": _update_norm(parameters, before),
                        "gradient_norm": gradient_norm,
                        "train_O_loss": float(o_loss.detach()),
                        "train_W_loss": float(w_loss.detach()),
                        "validation_IB_LTE10": metrics.get("IB_LTE10"),
                        "validation_IB_FrameLTE10": metrics.get("IB_FrameLTE10"),
                        "validation_IdentityOffsetMAE": metrics.get("IdentityOffsetMAE"),
                        "validation_status": "eligible" if update in evaluation else "not_evaluated",
                    }
                )
            del before
    if set(checkpoint_scores) != evaluation:
        raise RuntimeError("Frozen Prodigy checkpoint capture is incomplete.")
    elapsed = time.perf_counter() - started
    eval_state: dict[str, Any] | None = None
    if retain_eval_state_via_clone:
        # Schedule-Free inference weights are an optimizer-derived view.  Clone
        # both train-space model and complete optimizer state, enter eval only
        # on the clone, and serialize standalone eval-space parameters.
        train_model_state = export_readout_state(model)
        train_optimizer_state = copy.deepcopy(optimizer.state_dict())
        clone_model = make_model(runtime, spec, seed)
        if not isinstance(clone_model, StandardReadout):
            raise TypeError("Eval-state rematerialization requires StandardReadout.")
        load_readout_state(clone_model, train_model_state)
        clone_parameters = clone_model.trainable_parameters()
        clone_optimizer = ProdigyPlusScheduleFree(
            [{"params": clone_parameters}], lr=1.0
        )
        clone_optimizer.load_state_dict(train_optimizer_state)
        clone_optimizer.eval()
        with torch.no_grad():
            if capture_partitions is None:
                cloned = clone_model.capture_scores(target_indices, reference_positions)
            else:
                cloned = torch.cat(
                    [clone_model.capture_scores(partition, partition) for partition in capture_partitions]
                )
            cloned_scores = cloned.detach().cpu().numpy().astype(np.float64)
        expected_scores = checkpoint_scores[max_updates]
        if not np.array_equal(cloned_scores, expected_scores):
            raise RuntimeError(
                "Cloned Schedule-Free eval state does not exactly reproduce "
                "the terminal checkpoint scores."
            )
        eval_state = export_readout_state(clone_model)
        del clone_optimizer, clone_model, clone_parameters
        del train_model_state, train_optimizer_state, cloned_scores
    result = ProdigyTrajectoryResult(
        checkpoint_scores=checkpoint_scores,
        checkpoint_metrics=checkpoint_metrics,
        log_rows=log_rows,
        snapshot_rows=snapshot_rows,
        elapsed_seconds=elapsed,
        all_finite=all_finite,
        state=export_readout_state(model) if retain_state else None,
        eval_state=eval_state,
    )
    # Function return releases the closure-bound model and optimizer without
    # invalidating the bindings used by ``evaluate``.
    del initial_parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result


@contextmanager
def neutral_schedulefree_evaluation(
    optimizer: ProdigyPlusScheduleFree,
    parameters: Sequence[torch.nn.Parameter],
    *,
    model: torch.nn.Module | None = None,
) -> Iterator[None]:
    """Enter Schedule-Free eval mode without perturbing the training trajectory.

    ``optimizer.eval()`` and ``optimizer.train()`` are algebraic inverses, but
    their in-place FP32 lerps are not bitwise inverses.  Preserve the exact
    train-space tensors and RNG state around the complete validation callback,
    then restore them after the required optimizer mode transition.
    """
    values = tuple(parameters)
    if not values or len({id(value) for value in values}) != len(values):
        raise ValueError("Neutral evaluation requires unique trainable parameters.")
    if any(not bool(group.get("train_mode", False)) for group in optimizer.param_groups):
        raise RuntimeError("Neutral evaluation must begin in Schedule-Free train mode.")
    saved_parameters = [value.detach().clone() for value in values]
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    module_modes = (
        [(module, bool(module.training)) for module in model.modules()]
        if model is not None
        else []
    )
    optimizer.eval()
    if any(bool(group.get("train_mode", True)) for group in optimizer.param_groups):
        raise RuntimeError("Schedule-Free optimizer failed to enter eval mode.")
    try:
        yield
    finally:
        optimizer.train()
        with torch.no_grad():
            for parameter, saved in zip(values, saved_parameters):
                parameter.copy_(saved)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        for module, training in module_modes:
            module.training = training
        exact = all(
            torch.equal(parameter.detach(), saved)
            for parameter, saved in zip(values, saved_parameters)
        )
        train_mode = all(
            bool(group.get("train_mode", False)) for group in optimizer.param_groups
        )
        del saved_parameters
        if not exact or not train_mode:
            raise RuntimeError(
                "Schedule-Free neutral evaluation failed exact train-state restoration."
            )


@contextmanager
def neutral_schedulefree_multi_evaluation(
    optimizers: Sequence[ProdigyPlusScheduleFree],
    parameter_groups: Sequence[Sequence[torch.nn.Parameter]],
    *,
    model: torch.nn.Module | None = None,
) -> Iterator[None]:
    """Losslessly enter eval-space for independent Schedule-Free optimizers."""
    optimizer_values = tuple(optimizers)
    grouped_values = tuple(tuple(group) for group in parameter_groups)
    if not optimizer_values or len(optimizer_values) != len(grouped_values):
        raise ValueError("Each optimizer requires exactly one parameter collection.")
    values = tuple(parameter for group in grouped_values for parameter in group)
    if not values or len({id(value) for value in values}) != len(values):
        raise ValueError("Neutral evaluation requires unique trainable parameters.")
    for optimizer, group in zip(optimizer_values, grouped_values, strict=True):
        if len(optimizer.param_groups) != 1:
            raise ValueError("Multi-optimizer evaluation requires single-group optimizers.")
        if tuple(map(id, optimizer.param_groups[0]["params"])) != tuple(map(id, group)):
            raise ValueError("Optimizer parameters do not match the declared collection.")
        if not bool(optimizer.param_groups[0].get("train_mode", False)):
            raise RuntimeError("Neutral evaluation must begin in Schedule-Free train mode.")
    saved_parameters = [value.detach().clone() for value in values]
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    module_modes = (
        [(module, bool(module.training)) for module in model.modules()]
        if model is not None
        else []
    )
    for optimizer in optimizer_values:
        optimizer.eval()
    if any(
        bool(group.get("train_mode", True))
        for optimizer in optimizer_values
        for group in optimizer.param_groups
    ):
        raise RuntimeError("Schedule-Free optimizer failed to enter eval mode.")
    try:
        yield
    finally:
        for optimizer in optimizer_values:
            optimizer.train()
        with torch.no_grad():
            for parameter, saved in zip(values, saved_parameters, strict=True):
                parameter.copy_(saved)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        for module, training in module_modes:
            module.training = training
        exact = all(
            torch.equal(parameter.detach(), saved)
            for parameter, saved in zip(values, saved_parameters, strict=True)
        )
        train_mode = all(
            bool(group.get("train_mode", False))
            for optimizer in optimizer_values
            for group in optimizer.param_groups
        )
        del saved_parameters
        if not exact or not train_mode:
            raise RuntimeError(
                "Schedule-Free neutral evaluation failed exact train-state restoration."
            )


def _l2_norm(values: Sequence[torch.Tensor]) -> float:
    if not values:
        raise ValueError("A parameter group may not be empty.")
    total = torch.zeros((), dtype=torch.float64, device=values[0].device)
    for value in values:
        total.add_(value.detach().double().square().sum())
    return float(torch.sqrt(total))


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    gradients: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("Prodigy trainable parameter has no gradient.")
        gradients.append(parameter.grad)
    return _l2_norm(gradients)


def _update_norm(
    parameters: Sequence[torch.nn.Parameter], before: Sequence[torch.Tensor]
) -> float:
    if len(parameters) != len(before):
        raise RuntimeError("Update-norm snapshot has the wrong parameter count.")
    differences = [
        parameter.detach().float() - reference for parameter, reference in zip(parameters, before)
    ]
    return _l2_norm(differences)


def _rms(values: Sequence[torch.Tensor]) -> float:
    """Return a parameter-count weighted RMS for a tensor collection."""
    if not values:
        raise ValueError("An RMS diagnostic group may not be empty.")
    total = torch.zeros((), dtype=torch.float64, device=values[0].device)
    count = 0
    for value in values:
        total.add_(value.detach().double().square().sum())
        count += value.numel()
    return float(torch.sqrt(total / count))


def _float_group_value(group: Mapping[str, Any], key: str) -> float:
    value = group[key]
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)


def run_prodigy_trajectory(
    *,
    cache: ExactPrefixCache,
    fold: TrainingFold,
    candidate: LastBlockCandidate,
    seed: int,
    last_block: torch.nn.Module,
    post_layernorm: torch.nn.Module,
    pooling_head: torch.nn.Module,
    predecessor_state: Mapping[str, Any],
    base_scores: np.ndarray | None,
    pair_weighting: str,
    o_coefficient: float,
    w_coefficient: float,
    target_indices: torch.Tensor,
    device: torch.device,
    evaluate_scores: Callable[[np.ndarray], Mapping[str, float]],
    evaluation_checkpoints: Sequence[int],
    logging_updates: Sequence[int],
    snapshot_checkpoints: Sequence[int],
    snapshot_directory: Path | None,
    evaluate_intermediate: bool = True,
    max_updates: int = 1024,
    retain_state: bool = False,
    tick: Callable[[], None] | None = None,
) -> ProdigyTrajectoryResult:
    """Run one 1024-update two-group Prodigy Schedule-Free trajectory.

    The optimizer construction deliberately has no arguments beyond the exact two
    parameter groups and ``lr=1.0``.  This keeps every other behavior at the
    installed package's defaults.
    """
    if device.type != "cuda":
        raise RuntimeError("Prodigy last-block guard requires CUDA.")
    if pair_weighting != "REL" or (o_coefficient, w_coefficient) != (0.75, 0.25):
        raise RuntimeError("STEP46C loss contract must be REL with O/W 0.75/0.25.")
    evaluation = set(map(int, evaluation_checkpoints))
    logging = set(map(int, logging_updates))
    snapshots = set(map(int, snapshot_checkpoints))
    if max_updates < 1 or max_updates > 1024:
        raise ValueError("Prodigy max_updates must be inside [1, 1024].")
    if not evaluation or max(evaluation) != max_updates or min(evaluation) < 0:
        raise ValueError("Prodigy trajectory evaluation must include max_updates.")
    if not logging or max(logging) != max_updates or min(logging) < 1:
        raise ValueError("Prodigy log schedule must include max_updates.")
    if snapshots - evaluation:
        raise ValueError("Snapshots require a matching evaluation checkpoint.")
    if snapshots and snapshot_directory is None:
        raise ValueError("Snapshot checkpoints require a scratch directory.")

    configure_determinism(seed)
    model = LastBlockReadout(
        candidate=candidate,
        seed=seed,
        last_block=last_block,
        post_layernorm=post_layernorm,
        pooling_head=pooling_head,
        predecessor_state=predecessor_state,
        base_scores=base_scores,
        device=device,
    )
    encoder_parameters = [value for _, value in model.encoder_named_parameters()]
    readout_parameters = [value for _, value in model.readout_named_parameters()]
    if not encoder_parameters or not readout_parameters:
        raise RuntimeError("Both Prodigy parameter groups must be non-empty.")
    parameter_groups = [
        {"params": encoder_parameters},
        {"params": readout_parameters},
    ]
    optimizer = ProdigyPlusScheduleFree(parameter_groups, lr=1.0)
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("Prodigy optimizer did not retain exactly two groups.")

    static_o = torch.empty((8, *cache.token_shape), dtype=cache.original_dtype, device=device)
    static_w = torch.empty_like(static_o)
    static_o_indices = torch.empty(8, dtype=torch.int64, device=device)
    static_w_indices = torch.empty_like(static_o_indices)
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)

    def stage(source: str, group_index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        indices = fold.group_indices[(source, group_index)]
        size = len(indices)
        if not 2 <= size <= 8:
            raise RuntimeError(f"{source} group size is outside [2, 8].")
        values = static_o if source == "O" else static_w
        staged_indices = static_o_indices if source == "O" else static_w_indices
        values[:size].copy_(cache.pre_last(indices, device))
        staged_indices[:size].copy_(indices)
        return values, staged_indices, size

    checkpoint_scores: dict[int, np.ndarray] = {}
    checkpoint_metrics: dict[int, dict[str, float]] = {}
    snapshot_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    def evaluate(checkpoint: int) -> np.ndarray:
        with neutral_schedulefree_evaluation(
            optimizer, [*encoder_parameters, *readout_parameters], model=model
        ):
            values = capture_scores(
                model, cache, target_indices, device=device, batch_size=64
            )
            metrics = {
                name: float(value) for name, value in evaluate_scores(values).items()
            }
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError("Prodigy validation metrics became non-finite.")
            checkpoint_scores[checkpoint] = values
            checkpoint_metrics[checkpoint] = metrics
            if checkpoint in snapshots:
                assert snapshot_directory is not None
                snapshot_directory.mkdir(parents=True, exist_ok=True)
                path = snapshot_directory / f"checkpoint={checkpoint}.pt"
                atomic_torch(
                    path,
                    {
                        "candidate_id": candidate.candidate_id,
                        "fold": fold.identity,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "optimizer_mode": "eval",
                        "neutral_train_state_restoration": "exact_after_callback",
                        "model_state": model.export_state(),
                        "optimizer_state": optimizer.state_dict(),
                        "scores": values,
                        "metrics": metrics,
                    },
                )
                snapshot_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                    }
                )
        return values

    started = time.perf_counter()
    model.train()
    if 0 in evaluation:
        evaluate(0)
    all_finite = True
    last_o_loss = float("nan")
    last_w_loss = float("nan")
    for update in range(1, max_updates + 1):
        if tick is not None:
            tick()
        optimizer.zero_grad(set_to_none=True)
        o_index = o_sampler.next()
        w_index = w_sampler.next()
        o_values, o_indices, o_size = stage("O", o_index)
        w_values, w_indices, w_size = stage("W", w_index)
        o_scores = model(o_values[:o_size], o_indices[:o_size])
        o_loss = reliability_weighted_group_loss(
            o_scores, fold.retained_o[o_index], "O", pair_weighting=pair_weighting
        )
        (o_coefficient * o_loss).backward()
        w_scores = model(w_values[:w_size], w_indices[:w_size])
        w_loss = reliability_weighted_group_loss(
            w_scores, fold.retained_w[w_index], "W", pair_weighting=pair_weighting
        )
        (w_coefficient * w_loss).backward()
        last_o_loss = float(o_loss.detach())
        last_w_loss = float(w_loss.detach())
        parameters_by_group = (encoder_parameters, readout_parameters)
        gradient_norms = [_gradient_norm(values) for values in parameters_by_group]
        all_finite = all_finite and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for values in parameters_by_group
            for parameter in values
        )
        before = (
            [[value.detach().float().clone() for value in values] for values in parameters_by_group]
            if update in logging
            else None
        )
        optimizer.step()
        all_finite = all_finite and all(
            bool(torch.isfinite(parameter).all())
            for values in parameters_by_group
            for parameter in values
        )
        if not all_finite or not math.isfinite(last_o_loss + last_w_loss):
            raise RuntimeError(f"{candidate.candidate_id} Prodigy trajectory became non-finite.")

        metrics: Mapping[str, float] = {}
        should_evaluate = update in evaluation and (
            evaluate_intermediate or update == max(evaluation)
        )
        if should_evaluate:
            evaluate(update)
            metrics = checkpoint_metrics[update]
        if update in logging:
            assert before is not None
            for group_index, (name, parameters) in enumerate(
                (("encoder", encoder_parameters), ("readout", readout_parameters))
            ):
                group = optimizer.param_groups[group_index]
                d = _float_group_value(group, "d")
                effective_lr = _float_group_value(group, "effective_lr")
                log_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "identity_id": fold.identity,
                        "seed": seed,
                        "update": update,
                        "parameter_group": name,
                        "d": d,
                        "effective_lr": effective_lr,
                        "d_times_effective_lr": d * effective_lr,
                        "parameter_norm": _l2_norm([value.detach() for value in parameters]),
                        "update_norm": _update_norm(parameters, before[group_index]),
                        "gradient_norm": gradient_norms[group_index],
                        "train_O_loss": last_o_loss,
                        "train_W_loss": last_w_loss,
                        "validation_IB_LTE10": metrics.get("IB_LTE10"),
                        "validation_IB_FrameLTE10": metrics.get("IB_FrameLTE10"),
                        "validation_IdentityOffsetMAE": metrics.get("IdentityOffsetMAE"),
                        "validation_status": (
                            "eligible" if update >= 256 and should_evaluate
                            else "diagnostic" if should_evaluate
                            else "not_evaluated"
                        ),
                    }
                )
            del before
    if set(checkpoint_scores) != (
        evaluation if evaluate_intermediate else {0, max(evaluation)}
    ):
        raise RuntimeError("Prodigy checkpoint capture is incomplete.")
    elapsed = time.perf_counter() - started
    result = ProdigyTrajectoryResult(
        checkpoint_scores=checkpoint_scores,
        checkpoint_metrics=checkpoint_metrics,
        log_rows=log_rows,
        snapshot_rows=snapshot_rows,
        elapsed_seconds=elapsed,
        all_finite=all_finite,
        state=model.export_state() if retain_state else None,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result
