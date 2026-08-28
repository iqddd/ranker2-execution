"""Small typed specifications and analysis primitives for Step41 LR ranges.

Execution loops remain in their step scripts until their exact checkpoint and
metric contracts have been proven equal.  This module intentionally contains
only protocol-neutral data and deterministic range analysis.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ranker.attention_lora import (
    AttentionResidualHead,
    diversity_components,
    lora_parameters,
)
from ranker.attention_runtime import LastBlockReplay, RuntimeFold

@dataclass(frozen=True)
class HeadRangeSpec:
    learning_rates: tuple[float, ...]
    steps_or_epochs: int
    optimizer_parameters: Mapping[str, float]
    checkpoint_policy: str


@dataclass(frozen=True)
class LoRARangeSpec:
    learning_rates: tuple[float, ...]
    rank: int
    alpha: float
    lr_ratio_b_to_a: float | None
    steps_or_epochs: int
    checkpoint_policy: str


@dataclass(frozen=True)
class RangeRow:
    pilot_index: int
    seed: int
    step: int
    learning_rate: float
    finite: bool
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class RangeAnalysis:
    selected_steps: Mapping[tuple[int, int], int]
    selected_learning_rates: Mapping[tuple[int, int], float]
    nonfinite_points: int


def bias_corrected_ema(
    values: Sequence[float], *, beta: float = 0.8
) -> tuple[float, ...]:
    """Compute the stable, order-preserving EMA used by Step41 range reports."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("EMA beta must be in [0, 1).")
    moment = 0.0
    result: list[float] = []
    for index, value in enumerate(values):
        moment = beta * moment + (1.0 - beta) * value
        result.append(moment / (1.0 - beta ** (index + 1)))
    return tuple(result)


def analyze_range_rows(
    rows: Sequence[RangeRow],
    *,
    selection_policy: Callable[[Sequence[RangeRow]], RangeRow],
) -> RangeAnalysis:
    """Select one row per pilot/seed trajectory without changing row order."""
    trajectories: dict[tuple[int, int], list[RangeRow]] = {}
    for row in rows:
        trajectories.setdefault((row.pilot_index, row.seed), []).append(row)
    selected: dict[tuple[int, int], RangeRow] = {}
    for key, trajectory in trajectories.items():
        ordered = sorted(trajectory, key=lambda row: row.step)
        selected[key] = selection_policy(ordered)
    return RangeAnalysis(
        selected_steps={key: row.step for key, row in selected.items()},
        selected_learning_rates={
            key: row.learning_rate for key, row in selected.items()
        },
        nonfinite_points=sum(not row.finite for row in rows),
    )


def parameter_norm(parameters: Sequence[nn.Parameter]) -> float:
    """Return the joint FP32 parameter norm with the historical empty fallback."""
    values = [parameter.detach().float().square().sum() for parameter in parameters]
    return float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0


def head_optimizer(head: nn.Module, learning_rate: float) -> torch.optim.AdamW:
    """Construct the unchanged Step41A AdamW head optimizer groups."""
    linear_weights: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in head.named_parameters():
        if name in {"key.weight", "value.weight", "hidden.weight", "output.weight"}:
            linear_weights.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": linear_weights, "weight_decay": 0.05},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def hard_pair_losses(scores: torch.Tensor) -> torch.Tensor:
    """Compute 28 rank-ordered group losses in the existing pair order."""
    lower: list[int] = []
    higher: list[int] = []
    for first in range(8):
        for second in range(first + 1, 8):
            lower.append(first)
            higher.append(second)
    higher_tensor = torch.as_tensor(higher, device=scores.device)
    lower_tensor = torch.as_tensor(lower, device=scores.device)
    return torch.nn.functional.softplus(
        -(scores.index_select(0, higher_tensor) - scores.index_select(0, lower_tensor))
    )


def finite_mapping(values: Mapping[str, object]) -> bool:
    """Check metric mappings while treating bools as valid non-numeric flags."""
    return all(
        not isinstance(value, (float, int, np.floating, np.integer))
        or isinstance(value, (bool, np.bool_))
        or np.isfinite(float(value))
        for value in values.values()
    )


def training_state_finite(module: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
    """Verify parameter, gradient and optimizer-state finiteness."""
    for parameter in module.parameters():
        if not bool(torch.isfinite(parameter).all()):
            return False
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            return False
    return all(
        not torch.is_tensor(value) or bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
    )


def json_safe(value: object) -> object:
    """Convert range-report values to JSON while mapping non-finite floats to null."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def off_diagonal_cosines(rows: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return pairwise cosine values above the final two dimensions' diagonal."""
    normalized = rows / (torch.linalg.vector_norm(rows, dim=-1, keepdim=True) + eps)
    matrix = normalized @ normalized.transpose(-1, -2)
    indices = torch.triu_indices(
        matrix.shape[-2], matrix.shape[-1], offset=1, device=matrix.device
    )
    return matrix[..., indices[0], indices[1]]


def load_calibrated_head(
    record: Mapping[str, object],
    device: torch.device,
    *,
    signature: str,
    epoch: int = 9,
    gamma_lora_warmstart: float = 1.0,
    initial_query_norm: float,
) -> AttentionResidualHead:
    """Load and freeze a calibrated attention checkpoint after metadata checks."""
    path = Path(str(record["filename"]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        raise RuntimeError(f"Unexpected calibrated checkpoint signature: {path}.")
    if (
        int(payload["pilot_index"]) != int(record["pilot_index"])
        or int(payload["seed"]) != int(record["seed"])
        or int(payload["epoch"]) != epoch
        or float(payload["gamma_lora_warmstart"]) != gamma_lora_warmstart
    ):
        raise RuntimeError(f"Calibrated checkpoint metadata mismatch: {path}.")
    head = AttentionResidualHead(
        seed=int(record["seed"]), initial_query_norm=initial_query_norm
    )
    head.load_state_dict(payload["state_dict"], strict=True)
    head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def corrected_head(
    seed: int,
    device: torch.device,
    *,
    initial_query_norm: float,
) -> AttentionResidualHead:
    """Construct the corrected attention head used by R1 and R2."""
    return AttentionResidualHead(seed=seed, initial_query_norm=initial_query_norm).to(
        device
    )


def measure_head_train_loss(
    *,
    head: AttentionResidualHead,
    runtime: RuntimeFold,
    frozen_tokens: torch.Tensor,
    epoch_seed: int,
) -> dict[str, float]:
    """Measure the full deterministic head training objective without updates."""
    cuda_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_index]), torch.no_grad():
        torch.manual_seed(epoch_seed)
        torch.cuda.manual_seed_all(epoch_seed)
        head.train()
        order = list(range(len(runtime.group_aligned)))
        random.Random(epoch_seed).shuffle(order)
        aligned_total = counter_total = diversity_total = 0.0
        for group_index in order:
            start = 8 * group_index
            indices = runtime.train_cache_indices[start : start + 8]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                delta, diagnostics = head(
                    frozen_tokens.index_select(0, indices), diagnostics=True
                )
            assert diagnostics is not None
            scores = runtime.train_base[start : start + 8] + delta
            with torch.autocast(device_type="cuda", enabled=False):
                losses = hard_pair_losses(scores.float())
                aligned_sum = losses[runtime.group_aligned[group_index]].sum()
                counter_sum = losses[runtime.group_counter[group_index]].sum()
                dq, da = diversity_components(
                    diagnostics["queries"], diagnostics["logits"]
                )
                group_fraction = 8.0 / len(runtime.train_base)
            aligned_total += float(aligned_sum.cpu()) / runtime.n_aligned
            counter_total += float(counter_sum.cpu()) / runtime.n_counter
            diversity_total += group_fraction * float((0.5 * dq + 0.5 * da).cpu())
    return {
        "train_aligned_loss": aligned_total,
        "train_counter_loss": counter_total,
        "train_balanced_loss": (
            0.5 * aligned_total + 0.5 * counter_total + 0.02 * diversity_total
        ),
        "train_diversity_loss": diversity_total,
    }


def measure_lora_train_loss(
    *,
    replay: LastBlockReplay,
    head: AttentionResidualHead,
    runtime: RuntimeFold,
    pre_last: torch.Tensor,
    epoch_seed: int,
) -> dict[str, float]:
    """Measure the full deterministic LoRA training objective without updates."""
    cuda_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_index]), torch.no_grad():
        torch.manual_seed(epoch_seed)
        torch.cuda.manual_seed_all(epoch_seed)
        replay.train()
        head.eval()
        order = list(range(len(runtime.group_aligned)))
        random.Random(epoch_seed).shuffle(order)
        aligned_total = counter_total = 0.0
        for group_index in order:
            start = 8 * group_index
            indices = runtime.train_cache_indices[start : start + 8]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = replay(pre_last.index_select(0, indices))
                delta, _ = head(tokens)
            scores = runtime.train_base[start : start + 8] + delta
            with torch.autocast(device_type="cuda", enabled=False):
                losses = hard_pair_losses(scores.float())
                aligned_sum = losses[runtime.group_aligned[group_index]].sum()
                counter_sum = losses[runtime.group_counter[group_index]].sum()
            aligned_total += float(aligned_sum.cpu()) / runtime.n_aligned
            counter_total += float(counter_sum.cpu()) / runtime.n_counter
    return {
        "train_aligned_loss": aligned_total,
        "train_counter_loss": counter_total,
        "train_balanced_loss": 0.5 * aligned_total + 0.5 * counter_total,
        "train_diversity_loss": 0.0,
    }


def baseline_row(
    *,
    signature: str,
    arm: str,
    range_fold: int,
    pilot_index: int,
    seed: int,
    train: Mapping[str, float],
    metrics: Mapping[str, float | int],
    index_name: str,
    index_value: int,
) -> dict[str, object]:
    """Build the stable baseline row shared by R1 and R2 ranges."""
    row: dict[str, object] = {
        "signature": signature,
        "arm": arm,
        "range_fold": range_fold,
        "pilot_index": pilot_index,
        "seed": seed,
        index_name: index_value,
        "learning_rate": 0.0,
        "is_baseline": True,
        **train,
        **metrics,
        "gradient_norm": 0.0,
        "finite": finite_mapping({**train, **metrics}),
    }
    row["validation_aligned_loss_ratio_to_baseline"] = 1.0
    row["validation_counter_loss_ratio_to_baseline"] = 1.0
    row["validation_cross_all_WPA_delta_to_baseline"] = 0.0
    row["train_balanced_loss_ratio_to_baseline"] = 1.0
    return row


def add_relative_metrics(
    row: dict[str, object],
    *,
    baseline_train: Mapping[str, float],
    baseline_validation: Mapping[str, float | int],
) -> None:
    """Add the four baseline-relative metrics without changing arithmetic order."""
    row["train_balanced_loss_ratio_to_baseline"] = float(
        row["train_balanced_loss"]  # type: ignore[arg-type]
    ) / float(baseline_train["train_balanced_loss"])
    row["validation_aligned_loss_ratio_to_baseline"] = float(
        row["validation_aligned_loss"]  # type: ignore[arg-type]
    ) / float(baseline_validation["validation_aligned_loss"])
    row["validation_counter_loss_ratio_to_baseline"] = float(
        row["validation_counter_loss"]  # type: ignore[arg-type]
    ) / float(baseline_validation["validation_counter_loss"])
    row["validation_cross_all_WPA_delta_to_baseline"] = float(
        row["validation_cross_all_WPA"]  # type: ignore[arg-type]
    ) - float(baseline_validation["validation_cross_all_WPA"])


def train_head_pass(
    *,
    head: AttentionResidualHead,
    optimizer: torch.optim.Optimizer,
    runtime: RuntimeFold,
    frozen_tokens: torch.Tensor,
    learning_rate: float,
    epoch_seed: int,
    gradient_clip_norm: float | None = 1.0,
) -> tuple[dict[str, float], float]:
    """Run the original group-ordered frozen-head optimization pass."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    head.train()
    optimizer.zero_grad(set_to_none=True)
    order = list(range(len(runtime.group_aligned)))
    random.Random(epoch_seed).shuffle(order)
    aligned_total = counter_total = diversity_total = 0.0
    for group_index in order:
        start = 8 * group_index
        indices = runtime.train_cache_indices[start : start + 8]
        tokens = frozen_tokens.index_select(0, indices)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            delta, diagnostics = head(tokens, diagnostics=True)
        assert diagnostics is not None
        scores = runtime.train_base[start : start + 8] + delta
        with torch.autocast(device_type="cuda", enabled=False):
            losses = hard_pair_losses(scores.float())
            aligned_sum = losses[runtime.group_aligned[group_index]].sum()
            counter_sum = losses[runtime.group_counter[group_index]].sum()
            dq, da = diversity_components(diagnostics["queries"], diagnostics["logits"])
            group_fraction = 8.0 / len(runtime.train_base)
            loss = (
                0.5 * aligned_sum / runtime.n_aligned
                + 0.5 * counter_sum / runtime.n_counter
                + 0.02 * (0.5 * group_fraction * dq + 0.5 * group_fraction * da)
            )
        loss.backward()
        aligned_total += float(aligned_sum.detach().cpu()) / runtime.n_aligned
        counter_total += float(counter_sum.detach().cpu()) / runtime.n_counter
        diversity_total += group_fraction * float((0.5 * dq + 0.5 * da).detach().cpu())
    if gradient_clip_norm is None:
        gradient = torch.sqrt(
            torch.stack(
                [
                    parameter.grad.detach().float().square().sum()
                    for parameter in head.parameters()
                    if parameter.grad is not None
                ]
            ).sum()
        )
    else:
        gradient = torch.nn.utils.clip_grad_norm_(
            head.parameters(), gradient_clip_norm, foreach=False
        )
    optimizer.step()
    return {
        "train_aligned_loss": aligned_total,
        "train_counter_loss": counter_total,
        "train_balanced_loss": 0.5 * aligned_total
        + 0.5 * counter_total
        + 0.02 * diversity_total,
        "train_diversity_loss": diversity_total,
    }, float(gradient.detach().float().cpu())


def train_lora_pass(
    *,
    replay: LastBlockReplay,
    targets: Sequence[object],
    head: AttentionResidualHead,
    optimizer: torch.optim.Optimizer,
    runtime: RuntimeFold,
    pre_last: torch.Tensor,
    learning_rate: float,
    epoch_seed: int,
) -> tuple[dict[str, float], float]:
    """Run the original group-ordered replay LoRA optimization pass."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    replay.train()
    head.eval()
    optimizer.zero_grad(set_to_none=True)
    order = list(range(len(runtime.group_aligned)))
    random.Random(epoch_seed).shuffle(order)
    aligned_total = counter_total = 0.0
    for group_index in order:
        start = 8 * group_index
        indices = runtime.train_cache_indices[start : start + 8]
        inputs = pre_last.index_select(0, indices)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = replay(inputs)
            delta, _ = head(tokens)
        scores = runtime.train_base[start : start + 8] + delta
        with torch.autocast(device_type="cuda", enabled=False):
            losses = hard_pair_losses(scores.float())
            aligned_sum = losses[runtime.group_aligned[group_index]].sum()
            counter_sum = losses[runtime.group_counter[group_index]].sum()
            loss = (
                0.5 * aligned_sum / runtime.n_aligned
                + 0.5 * counter_sum / runtime.n_counter
            )
        loss.backward()
        aligned_total += float(aligned_sum.detach().cpu()) / runtime.n_aligned
        counter_total += float(counter_sum.detach().cpu()) / runtime.n_counter
    gradient = torch.nn.utils.clip_grad_norm_(
        lora_parameters(targets), 1.0, foreach=False
    )
    optimizer.step()
    return {
        "train_aligned_loss": aligned_total,
        "train_counter_loss": counter_total,
        "train_balanced_loss": 0.5 * aligned_total + 0.5 * counter_total,
        "train_diversity_loss": 0.0,
    }, float(gradient.detach().float().cpu())
