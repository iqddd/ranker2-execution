"""Reusable mini-batch attention-head search primitives."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ranker.attention_evaluation import forward_head_deltas
from ranker.attention_lora import AttentionResidualHead, diversity_components
from ranker.attention_training import training_state_finite
from ranker.data import RankedImage
from ranker.spatial import train_prior_pair_classes


@dataclass(frozen=True)
class GroupMiniBatchRuntime:
    groups: tuple[tuple[RankedImage, ...], ...]
    cache_indices: tuple[torch.Tensor, ...]
    base_scores: tuple[torch.Tensor, ...]
    lower_indices: tuple[torch.Tensor, ...]
    higher_indices: tuple[torch.Tensor, ...]
    aligned_masks: tuple[torch.Tensor, ...]
    counter_masks: tuple[torch.Tensor, ...]
    n_aligned: int
    n_counter: int

    @property
    def group_count(self) -> int:
        return len(self.groups)


def build_group_minibatch_runtime(
    *,
    groups: Sequence[Sequence[RankedImage]],
    base_scores: Mapping[str, float],
    name_to_cache: Mapping[str, int],
    device: torch.device,
) -> GroupMiniBatchRuntime:
    """Build variable-size group tensors with the historical A/C prior masks."""
    group_tuples = tuple(tuple(group) for group in groups)
    masks = train_prior_pair_classes(group_tuples)
    cache_indices: list[torch.Tensor] = []
    group_base: list[torch.Tensor] = []
    lower_indices: list[torch.Tensor] = []
    higher_indices: list[torch.Tensor] = []
    aligned_masks: list[torch.Tensor] = []
    counter_masks: list[torch.Tensor] = []
    for group, aligned, counter in zip(group_tuples, *masks, strict=True):
        ordered = sorted(group, key=lambda item: item.rank)
        if [item.rank for item in ordered] != list(range(len(ordered))):
            raise ValueError("Train O-group has non-contiguous ranks.")
        lower: list[int] = []
        higher: list[int] = []
        for first in range(len(ordered)):
            for second in range(first + 1, len(ordered)):
                lower.append(first)
                higher.append(second)
        cache_indices.append(
            torch.as_tensor(
                [name_to_cache[item.image_name] for item in ordered],
                dtype=torch.long,
                device=device,
            )
        )
        group_base.append(
            torch.as_tensor(
                [base_scores[item.image_name] for item in ordered],
                dtype=torch.float32,
                device=device,
            )
        )
        lower_indices.append(torch.as_tensor(lower, dtype=torch.long, device=device))
        higher_indices.append(torch.as_tensor(higher, dtype=torch.long, device=device))
        aligned_masks.append(torch.as_tensor(aligned, dtype=torch.bool, device=device))
        counter_masks.append(torch.as_tensor(counter, dtype=torch.bool, device=device))
    n_aligned = sum(int(mask.sum()) for mask in aligned_masks)
    n_counter = sum(int(mask.sum()) for mask in counter_masks)
    if not group_tuples or n_aligned <= 0 or n_counter <= 0:
        raise RuntimeError("Mini-batch head objective requires both global A/C strata.")
    return GroupMiniBatchRuntime(
        groups=group_tuples,
        cache_indices=tuple(cache_indices),
        base_scores=tuple(group_base),
        lower_indices=tuple(lower_indices),
        higher_indices=tuple(higher_indices),
        aligned_masks=tuple(aligned_masks),
        counter_masks=tuple(counter_masks),
        n_aligned=n_aligned,
        n_counter=n_counter,
    )


def deterministic_group_batch(
    *, group_count: int, requested_batch_size: int | None, batch_seed: int, step: int
) -> tuple[int, ...]:
    """Select the prescribed uniform no-replacement group batch for one step."""
    batch_size = group_count if requested_batch_size is None else min(requested_batch_size, group_count)
    return tuple(random.Random(batch_seed + step).sample(range(group_count), batch_size))


def train_group_minibatch_step(
    *,
    head: AttentionResidualHead,
    optimizer: torch.optim.Optimizer,
    runtime: GroupMiniBatchRuntime,
    frozen_tokens: torch.Tensor,
    learning_rate: float,
    requested_batch_size: int | None,
    batch_seed: int,
    step: int,
) -> dict[str, float | int | bool]:
    """Apply the globally normalized group-mini-batch objective exactly once."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    selected = deterministic_group_batch(
        group_count=runtime.group_count,
        requested_batch_size=requested_batch_size,
        batch_seed=batch_seed,
        step=step,
    )
    batch_size = len(selected)
    scale = runtime.group_count / batch_size
    torch.manual_seed(batch_seed + step)
    torch.cuda.manual_seed_all(batch_seed + step)
    head.train()
    optimizer.zero_grad(set_to_none=True)
    aligned_sum_total = 0.0
    counter_sum_total = 0.0
    diversity_total = 0.0
    for group_index in selected:
        tokens = frozen_tokens.index_select(0, runtime.cache_indices[group_index])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            delta, diagnostics = head(tokens, diagnostics=True)
        assert diagnostics is not None
        scores = runtime.base_scores[group_index] + delta
        with torch.autocast(device_type="cuda", enabled=False):
            pair_losses = torch.nn.functional.softplus(
                -(
                    scores.float().index_select(0, runtime.higher_indices[group_index])
                    - scores.float().index_select(0, runtime.lower_indices[group_index])
                )
            )
            aligned_sum = pair_losses[runtime.aligned_masks[group_index]].sum()
            counter_sum = pair_losses[runtime.counter_masks[group_index]].sum()
            query_diversity, attention_diversity = diversity_components(
                diagnostics["queries"], diagnostics["logits"]
            )
            group_diversity = 0.5 * (query_diversity + attention_diversity)
            loss = scale * (
                0.5 * aligned_sum / runtime.n_aligned
                + 0.5 * counter_sum / runtime.n_counter
            ) + 0.02 * group_diversity / batch_size
        loss.backward()
        aligned_sum_total += float(aligned_sum.detach().cpu())
        counter_sum_total += float(counter_sum.detach().cpu())
        diversity_total += float(group_diversity.detach().cpu())
    gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, foreach=False)
    optimizer.step()
    finite = bool(
        math.isfinite(float(gradient.detach().cpu()))
        and training_state_finite(head, optimizer)
    )
    return {
        "step": step,
        "effective_batch_size": batch_size,
        "aligned_pairs_in_batch": sum(
            int(runtime.aligned_masks[index].sum()) for index in selected
        ),
        "counter_pairs_in_batch": sum(
            int(runtime.counter_masks[index].sum()) for index in selected
        ),
        "objective_aligned_component": scale * 0.5 * aligned_sum_total / runtime.n_aligned,
        "objective_counter_component": scale * 0.5 * counter_sum_total / runtime.n_counter,
        "objective_diversity_component": 0.02 * diversity_total / batch_size,
        "gradient_l2_before_clip": float(gradient.detach().cpu()),
        "finite": finite,
    }


def calibrated_unit_residual(
    *,
    head: AttentionResidualHead,
    frozen_tokens: torch.Tensor,
    all_base_scores: np.ndarray,
    train_indices: np.ndarray,
    evaluation_batch_size: int = 2,
) -> tuple[np.ndarray | None, dict[str, float | bool]]:
    """Evaluate one snapshot and form its train-only unit-std residual."""
    device = frozen_tokens.device
    indices = torch.arange(len(all_base_scores), dtype=torch.long, device=device)
    delta, _ = forward_head_deltas(
        head=head,
        tokens=frozen_tokens,
        indices=indices,
        batch_size=evaluation_batch_size,
    )
    train_delta = delta[train_indices]
    train_base = np.asarray(all_base_scores, dtype=np.float64)[train_indices]
    mu = float(np.mean(train_delta))
    sigma_delta = float(np.std(train_delta, ddof=0))
    sigma_base = float(np.std(train_base, ddof=0))
    valid = bool(
        np.all(np.isfinite(delta))
        and math.isfinite(mu)
        and sigma_delta > 1e-8
        and sigma_base > 1e-8
    )
    normalized = None if not valid else (delta - mu) * (sigma_base / sigma_delta)
    return normalized, {
        "mu_delta": mu,
        "sigma_delta": sigma_delta,
        "sigma_base": sigma_base,
        "unit_beta": sigma_base / sigma_delta if sigma_delta > 0.0 else float("inf"),
        "calibration_valid": valid,
    }
