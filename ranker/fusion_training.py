"""Reusable Schedule-Free training for shared-layer frozen readouts."""

from __future__ import annotations

import gc
import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

from ranker.frozen_readouts import FrozenFeatureRuntime, TrainingFold, configure_determinism
from ranker.fusion_readout import SharedLayerFusionReadout
from ranker.io import atomic_torch, sha256_file
from ranker.prodigy_guard import _float_group_value, neutral_schedulefree_evaluation
from ranker.trainer_contract import CyclingGroupSampler, reliability_weighted_group_loss


@dataclass
class FusionTrajectoryResult:
    """Closed replayable result of one fusion trajectory."""

    checkpoint_scores: dict[int, np.ndarray]
    checkpoint_diagnostics: dict[int, dict[str, Any]]
    log_rows: list[dict[str, Any]]
    snapshot_rows: list[dict[str, Any]]
    elapsed_seconds: float
    all_finite: bool
    sampler_schedule_sha256: str


def fusion_result_payload(result: FusionTrajectoryResult) -> dict[str, Any]:
    return {
        "checkpoint_scores": {
            str(checkpoint): scores.tolist()
            for checkpoint, scores in result.checkpoint_scores.items()
        },
        "checkpoint_diagnostics": {
            str(checkpoint): values
            for checkpoint, values in result.checkpoint_diagnostics.items()
        },
        "log_rows": result.log_rows,
        "snapshot_rows": result.snapshot_rows,
        "elapsed_seconds": result.elapsed_seconds,
        "all_finite": result.all_finite,
        "sampler_schedule_sha256": result.sampler_schedule_sha256,
    }


def fusion_result_from_payload(payload: Mapping[str, Any]) -> FusionTrajectoryResult:
    snapshots = list(payload["snapshot_rows"])
    for row in snapshots:
        path = Path(str(row["path"]))
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError("Journaled fusion snapshot is missing or changed.")
    return FusionTrajectoryResult(
        checkpoint_scores={
            int(checkpoint): np.asarray(scores, dtype=np.float64)
            for checkpoint, scores in payload["checkpoint_scores"].items()
        },
        checkpoint_diagnostics={
            int(checkpoint): dict(values)
            for checkpoint, values in payload["checkpoint_diagnostics"].items()
        },
        log_rows=list(payload["log_rows"]),
        snapshot_rows=snapshots,
        elapsed_seconds=float(payload["elapsed_seconds"]),
        all_finite=bool(payload["all_finite"]),
        sampler_schedule_sha256=str(payload["sampler_schedule_sha256"]),
    )


def sampler_schedule_digest(o_group_count: int, w_group_count: int, seed: int, updates: int) -> str:
    """Hash the exact paired CyclingGroupSampler schedule used by a trajectory."""
    o_sampler = CyclingGroupSampler(o_group_count, seed)
    w_sampler = CyclingGroupSampler(w_group_count, seed + 1)
    digest = hashlib.sha256()
    for _ in range(updates):
        pair = np.asarray([o_sampler.next(), w_sampler.next()], dtype="<i8")
        digest.update(pair.tobytes())
    return digest.hexdigest()


def _gradient_rms(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    count = 0
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("Fusion parameter has no gradient.")
        total.add_(parameter.grad.detach().double().square().sum())
        count += parameter.numel()
    return float(torch.sqrt(total / count))


def run_fusion_prodigy_trajectory(
    *,
    runtime: FrozenFeatureRuntime,
    tokens22: torch.Tensor,
    fold: TrainingFold,
    seed: int,
    initial_common_state: Mapping[str, Any],
    initial_gates: Sequence[float],
    capture_partitions: Sequence[torch.Tensor],
    evaluation_checkpoints: Sequence[int],
    snapshot_directory: Path,
    max_updates: int = 1024,
    tick: Callable[[], None] | None = None,
) -> FusionTrajectoryResult:
    """Train one HARD/REL/M75 fusion trajectory with exact package defaults."""
    checkpoints = set(map(int, evaluation_checkpoints))
    if max(checkpoints) != max_updates or min(checkpoints) < 1:
        raise ValueError("Fusion checkpoint grid must end at max_updates.")
    if runtime.device.type != "cuda":
        raise RuntimeError("Fusion training requires CUDA.")
    configure_determinism(seed)
    model = SharedLayerFusionReadout(runtime, tokens22, seed)
    model.load_common_state(dict(initial_common_state))
    gates = torch.as_tensor(initial_gates, dtype=torch.float32, device=runtime.device)
    if gates.shape != (4,):
        raise ValueError("Fusion initialization requires exactly four gates.")
    with torch.no_grad():
        model.gates.copy_(gates)
    parameters = model.all_parameters()
    optimizer = ProdigyPlusScheduleFree([{"params": parameters}], lr=1.0)
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("Fusion optimizer must contain one parameter group.")
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)
    score_panel: dict[int, np.ndarray] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    logs: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    all_finite = True
    schedule_digest = hashlib.sha256()
    started = time.perf_counter()
    model.train()
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    for update in range(1, max_updates + 1):
        if tick is not None:
            tick()
        optimizer.zero_grad(set_to_none=True)
        o_index = o_sampler.next()
        w_index = w_sampler.next()
        schedule_digest.update(
            np.asarray([o_index, w_index], dtype="<i8").tobytes()
        )
        o_scores = model.scores(fold.group_indices[("O", o_index)])[0]
        o_loss = reliability_weighted_group_loss(
            o_scores, fold.retained_o[o_index], "O", pair_weighting="REL"
        )
        (.75 * o_loss).backward()
        w_scores = model.scores(fold.group_indices[("W", w_index)])[0]
        w_loss = reliability_weighted_group_loss(
            w_scores, fold.retained_w[w_index], "W", pair_weighting="REL"
        )
        (.25 * w_loss).backward()
        gradient_rms = _gradient_rms(parameters)
        all_finite &= all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        optimizer.step()
        all_finite &= all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
        if not all_finite:
            raise RuntimeError("Fusion trajectory became non-finite.")
        if update not in checkpoints:
            continue
        with neutral_schedulefree_evaluation(optimizer, parameters, model=model):
            capture = model.capture_partitions(capture_partitions)
            scores = np.asarray(capture["scores"], dtype=np.float64)
            if not np.isfinite(scores).all():
                raise RuntimeError("Fusion checkpoint scores became non-finite.")
            score_panel[update] = scores
            diagnostics[update] = {
                "gates": np.asarray(capture["gates"], dtype=np.float64).tolist(),
                "z26_RMS": float(capture["z26_RMS"]),
                "gated_z22_RMS": float(capture["gated_z22_RMS"]),
            }
            path = snapshot_directory / f"checkpoint={update}.pt"
            atomic_torch(
                path,
                {
                    "candidate_id": "L26_L22_FUSION",
                    "fold": fold.identity,
                    "seed": seed,
                    "checkpoint": update,
                    "optimizer_mode": "eval",
                    "fusion_state": model.export_state(),
                    "scores": scores,
                    "diagnostics": diagnostics[update],
                    "optimizer_required_for_inference": False,
                },
            )
            snapshots.append(
                {"checkpoint": update, "path": str(path.resolve()), "sha256": sha256_file(path)}
            )
        group = optimizer.param_groups[0]
        logs.append(
            {
                "identity_id": fold.identity,
                "seed": seed,
                "checkpoint": update,
                "d": _float_group_value(group, "d"),
                "effective_lr": _float_group_value(group, "effective_lr"),
                "d_times_effective_lr": _float_group_value(group, "d")
                * _float_group_value(group, "effective_lr"),
                "gradient_RMS": gradient_rms,
                "train_O_loss": float(o_loss.detach()),
                "train_W_loss": float(w_loss.detach()),
                "g1": diagnostics[update]["gates"][0],
                "g2": diagnostics[update]["gates"][1],
                "g3": diagnostics[update]["gates"][2],
                "g4": diagnostics[update]["gates"][3],
            }
        )
    elapsed = time.perf_counter() - started
    del optimizer, model, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return FusionTrajectoryResult(
        checkpoint_scores=score_panel,
        checkpoint_diagnostics=diagnostics,
        log_rows=logs,
        snapshot_rows=snapshots,
        elapsed_seconds=elapsed,
        all_finite=all_finite,
        sampler_schedule_sha256=schedule_digest.hexdigest(),
    )
