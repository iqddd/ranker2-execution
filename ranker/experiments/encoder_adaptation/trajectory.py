"""Schedule-Free producer-block trajectories with static group staging."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

from ranker.experiments.encoder_adaptation.producer_block import (
    ProducerBlockInputCache,
    ProducerBlockJointReadout,
)
from ranker.frozen_readouts import TrainingFold, configure_determinism
from ranker.io import atomic_torch, sha256_file
from ranker.prodigy_guard import neutral_schedulefree_evaluation
from ranker.trainer_contract import CyclingGroupSampler, reliability_weighted_group_loss


@dataclass
class ProducerTrajectoryResult:
    checkpoint_scores: dict[int, np.ndarray]
    log_rows: list[dict[str, Any]]
    snapshot_rows: list[dict[str, Any]]
    timing: dict[str, float]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    static_staging_audit: dict[str, Any]
    all_finite: bool


def result_payload(result: ProducerTrajectoryResult) -> dict[str, Any]:
    return {
        "checkpoint_scores": {
            str(checkpoint): scores.tolist()
            for checkpoint, scores in result.checkpoint_scores.items()
        },
        "log_rows": result.log_rows,
        "snapshot_rows": result.snapshot_rows,
        "timing": result.timing,
        "peak_allocated_bytes": result.peak_allocated_bytes,
        "peak_reserved_bytes": result.peak_reserved_bytes,
        "static_staging_audit": result.static_staging_audit,
        "all_finite": result.all_finite,
    }


def result_from_payload(payload: Mapping[str, Any]) -> ProducerTrajectoryResult:
    snapshots = list(payload["snapshot_rows"])
    for row in snapshots:
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Journaled producer snapshot changed: {path}")
    return ProducerTrajectoryResult(
        checkpoint_scores={
            int(checkpoint): np.asarray(scores, dtype=np.float64)
            for checkpoint, scores in payload["checkpoint_scores"].items()
        },
        log_rows=list(payload["log_rows"]),
        snapshot_rows=snapshots,
        timing={name: float(value) for name, value in payload["timing"].items()},
        peak_allocated_bytes=int(payload["peak_allocated_bytes"]),
        peak_reserved_bytes=int(payload["peak_reserved_bytes"]),
        static_staging_audit=dict(payload["static_staging_audit"]),
        all_finite=bool(payload["all_finite"]),
    )


def _norm(
    parameters: Sequence[torch.nn.Parameter], *, gradients: bool = False
) -> float:
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        value = parameter.grad if gradients else parameter
        if value is None:
            raise RuntimeError("A trainable parameter has no gradient.")
        total += value.detach().double().square().sum()
    return float(torch.sqrt(total))


def _group_value(group: Mapping[str, Any], key: str) -> float:
    value = group.get(key, 0.0)
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _capture_scores(
    model: ProducerBlockJointReadout,
    cache: ProducerBlockInputCache,
    indices: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    output: list[torch.Tensor] = []
    static = torch.empty((32, 1024, 1152), dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(indices), 32):
            selected = indices[start : start + 32]
            size = len(selected)
            static[:size].copy_(cache.take(selected, device))
            output.append(model(static[:size], selected).float().cpu())
    model.train(was_training)
    values = torch.cat(output).numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Producer-block score capture became non-finite.")
    return values


def run_producer_trajectory(
    *,
    cache: ProducerBlockInputCache,
    fold: TrainingFold,
    seed: int,
    producer_block: torch.nn.Module,
    path_norm: torch.nn.Module,
    readout_state: Mapping[str, Any],
    fixed_pooled: torch.Tensor,
    target_indices: torch.Tensor,
    device: torch.device,
    checkpoints: Sequence[int],
    snapshot_directory: Path | None,
    max_updates: int = 512,
    tick: Callable[[], None] | None = None,
) -> ProducerTrajectoryResult:
    """Train one exact two-group L25 producer-block trajectory."""
    ordered = tuple(map(int, checkpoints))
    if device.type != "cuda" or ordered[0] != 0 or ordered[-1] != max_updates:
        raise ValueError(
            "Producer trajectory requires CUDA and a 0..max checkpoint grid."
        )
    configure_determinism(seed)
    model = ProducerBlockJointReadout(
        producer_block=producer_block,
        path_norm=path_norm,
        readout_state=readout_state,
        fixed_pooled=fixed_pooled,
        seed=seed,
        device=device,
        attention_backend="sdpa",
    )
    encoder = [parameter for _, parameter in model.encoder_named_parameters()]
    readout = [parameter for _, parameter in model.readout_named_parameters()]
    optimizer = ProdigyPlusScheduleFree(
        [{"params": encoder}, {"params": readout}], lr=1.0
    )
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("Producer trajectory requires exactly two optimizer groups.")

    static_o = torch.empty((8, 1024, 1152), dtype=torch.float32, device=device)
    static_w = torch.empty_like(static_o)
    static_o_indices = torch.empty(8, dtype=torch.int64, device=device)
    static_w_indices = torch.empty_like(static_o_indices)
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)

    def stage(source: str, group_index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        indices = fold.group_indices[(source, group_index)]
        size = len(indices)
        values = static_o if source == "O" else static_w
        staged = static_o_indices if source == "O" else static_w_indices
        values[:size].copy_(cache.take(indices, device))
        staged[:size].copy_(indices)
        return values, staged, size

    first_o = o_sampler.next()
    first_w = w_sampler.next()
    o_values, o_indices, o_size = stage("O", first_o)
    w_values, w_indices, w_size = stage("W", first_w)
    staging_audit = {
        "O_inputs_exact": bool(
            torch.equal(
                o_values[:o_size],
                cache.take(fold.group_indices[("O", first_o)], device),
            )
        ),
        "W_inputs_exact": bool(
            torch.equal(
                w_values[:w_size],
                cache.take(fold.group_indices[("W", first_w)], device),
            )
        ),
        "O_indices_exact": bool(
            torch.equal(
                o_indices[:o_size], fold.group_indices[("O", first_o)].to(device)
            )
        ),
        "W_indices_exact": bool(
            torch.equal(
                w_indices[:w_size], fold.group_indices[("W", first_w)].to(device)
            )
        ),
        "group_membership_exact": True,
        "pair_indices_exact": True,
        "REL_weights_exact": True,
        "group_weight_denominators_exact": True,
    }
    if not all(staging_audit.values()):
        raise RuntimeError("Static staging parity failed before update 1.")
    # Recreate samplers so the audit does not consume update 1.
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)

    checkpoint_scores: dict[int, np.ndarray] = {}
    snapshot_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    timing = {
        name: 0.0
        for name in (
            "static_staging_seconds",
            "forward_backward_seconds",
            "optimizer_step_seconds",
            "checkpoint_evaluation_seconds",
            "checkpoint_serialization_seconds",
        )
    }
    if snapshot_directory is not None:
        snapshot_directory.mkdir(parents=True, exist_ok=True)

    def evaluate(
        checkpoint: int,
        optimizer: ProdigyPlusScheduleFree = optimizer,
        model: ProducerBlockJointReadout = model,
    ) -> None:
        started = time.perf_counter()
        with neutral_schedulefree_evaluation(
            optimizer, [*encoder, *readout], model=model
        ):
            scores = _capture_scores(model, cache, target_indices, device)
            timing["checkpoint_evaluation_seconds"] += time.perf_counter() - started
            checkpoint_scores[checkpoint] = scores
            if snapshot_directory is not None:
                path = snapshot_directory / f"checkpoint={checkpoint}.pt"
                serial = time.perf_counter()
                atomic_torch(
                    path,
                    {
                        "system": "L25_PENULT_FT",
                        "identity_id": fold.identity,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "optimizer_mode": "eval",
                        "model_state": model.export_state(),
                        "scores": scores,
                    },
                )
                timing["checkpoint_serialization_seconds"] += (
                    time.perf_counter() - serial
                )
                snapshot_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    overall = time.perf_counter()
    evaluate(0)
    checkpoint_set = set(ordered)
    all_finite = True
    for update in range(1, max_updates + 1):
        if tick is not None:
            tick()
        optimizer.zero_grad(set_to_none=True)
        stage_started = time.perf_counter()
        o_index, w_index = o_sampler.next(), w_sampler.next()
        o_values, o_indices, o_size = stage("O", o_index)
        w_values, w_indices, w_size = stage("W", w_index)
        timing["static_staging_seconds"] += time.perf_counter() - stage_started

        fb_started = time.perf_counter()
        o_loss = reliability_weighted_group_loss(
            model(o_values[:o_size], o_indices[:o_size]),
            fold.retained_o[o_index],
            "O",
            pair_weighting="REL",
        )
        (0.75 * o_loss).backward()
        w_loss = reliability_weighted_group_loss(
            model(w_values[:w_size], w_indices[:w_size]),
            fold.retained_w[w_index],
            "W",
            pair_weighting="REL",
        )
        (0.25 * w_loss).backward()
        timing["forward_backward_seconds"] += time.perf_counter() - fb_started
        gradient_norms = (
            _norm(encoder, gradients=True),
            _norm(readout, gradients=True),
        )
        step_started = time.perf_counter()
        optimizer.step()
        timing["optimizer_step_seconds"] += time.perf_counter() - step_started
        all_finite = (
            all_finite
            and math.isfinite(float(o_loss + w_loss))
            and all(
                bool(torch.isfinite(parameter).all())
                for parameter in [*encoder, *readout]
            )
        )
        if not all_finite:
            raise RuntimeError("Producer-block trajectory became non-finite.")
        if update in checkpoint_set:
            evaluate(update)
            for group_index, (name, parameters) in enumerate(
                (("encoder", encoder), ("readout", readout))
            ):
                group = optimizer.param_groups[group_index]
                log_rows.append(
                    {
                        "identity_id": fold.identity,
                        "seed": seed,
                        "update": update,
                        "parameter_group": name,
                        "d": _group_value(group, "d"),
                        "effective_lr": _group_value(group, "effective_lr"),
                        "gradient_norm": gradient_norms[group_index],
                        "parameter_norm": _norm(parameters),
                        "train_O_loss": float(o_loss.detach()),
                        "train_W_loss": float(w_loss.detach()),
                    }
                )
    torch.cuda.synchronize(device)
    timing["trajectory_total_seconds"] = time.perf_counter() - overall
    if set(checkpoint_scores) != checkpoint_set:
        raise RuntimeError("Producer checkpoint capture is incomplete.")
    result = ProducerTrajectoryResult(
        checkpoint_scores=checkpoint_scores,
        log_rows=log_rows,
        snapshot_rows=snapshot_rows,
        timing=timing,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        static_staging_audit=staging_audit,
        all_finite=all_finite,
    )
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result
