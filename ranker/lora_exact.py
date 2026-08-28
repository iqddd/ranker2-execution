"""Exact cache-partitioned LoRA replay kernels shared by R4 and Step41B."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ranker.attention_lora import AttentionResidualHead, LoRATarget
from ranker.attention_runtime import LastBlockReplay, RuntimeFold
from ranker.lora_protocol import (
    all_gradients_finite,
    grad_l2,
    hard_pair_indices,
    lora_groups,
    set_loraplus_lr,
)

REPLAY_MODE = "exact"


def replay_indices_cache_batched(
    *, replay: LastBlockReplay, pre_last: Tensor, indices: Tensor, batch_size: int = 2
) -> Tensor:
    """Replay cache indices in their original contiguous batch partition."""
    requested = [int(value) for value in indices.detach().cpu().tolist()]
    batch_ids = sorted({index // batch_size for index in requested})
    by_index: dict[int, Tensor] = {}
    for batch_id in batch_ids:
        start = batch_size * batch_id
        stop = min(start + batch_size, len(pre_last))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = replay(pre_last[start:stop]).float()
        for offset in range(stop - start):
            by_index[start + offset] = output[offset]
    return torch.stack([by_index[index] for index in requested])


def forward_group_scores(
    *,
    replay: LastBlockReplay,
    head: AttentionResidualHead,
    pre_last: Tensor,
    indices: Tensor,
    base_scores: Tensor,
    eval_batch_size: int = 2,
) -> tuple[Tensor, Tensor]:
    tokens = replay_indices_cache_batched(
        replay=replay, pre_last=pre_last, indices=indices, batch_size=eval_batch_size
    )
    residual_batches: list[Tensor] = []
    for start in range(0, len(tokens), eval_batch_size):
        with torch.autocast(device_type="cuda", enabled=False):
            residual, _ = head(tokens[start : start + eval_batch_size].float())
        residual_batches.append(residual.float())
    residuals = torch.cat(residual_batches)
    return base_scores.float() + residuals, tokens


def accumulate_exact_train_gradient(
    *,
    replay: LastBlockReplay,
    head: AttentionResidualHead,
    runtime: RuntimeFold,
    pre_last: Tensor,
    epoch_seed: int,
    eval_batch_size: int = 2,
) -> dict[str, float]:
    replay.eval()
    head.eval()
    replay.zero_grad(set_to_none=True)
    order = list(range(len(runtime.group_aligned)))
    random.Random(epoch_seed).shuffle(order)
    lower, higher, _ = hard_pair_indices(runtime.train_base.device)
    aligned_total = counter_total = 0.0
    for group_index in order:
        start = 8 * group_index
        scores, _ = forward_group_scores(
            replay=replay,
            head=head,
            pre_last=pre_last,
            indices=runtime.train_cache_indices[start : start + 8],
            base_scores=runtime.train_base[start : start + 8],
            eval_batch_size=eval_batch_size,
        )
        with torch.autocast(device_type="cuda", enabled=False):
            margins = scores.index_select(0, higher) - scores.index_select(0, lower)
            losses = F.softplus(-margins.float())
            aligned_sum = losses[runtime.group_aligned[group_index]].sum()
            counter_sum = losses[runtime.group_counter[group_index]].sum()
            loss = (
                0.5 * aligned_sum / runtime.n_aligned
                + 0.5 * counter_sum / runtime.n_counter
            )
        loss.backward()
        aligned_total += float(aligned_sum.detach().cpu()) / runtime.n_aligned
        counter_total += float(counter_sum.detach().cpu()) / runtime.n_counter
    return {
        "optimizer_pass_aligned_loss": aligned_total,
        "optimizer_pass_counter_loss": counter_total,
        "optimizer_pass_balanced_loss": 0.5 * aligned_total + 0.5 * counter_total,
    }


def train_exact_lora_epoch(
    *,
    replay: LastBlockReplay,
    targets: Sequence[LoRATarget],
    head: AttentionResidualHead,
    optimizer: torch.optim.Optimizer,
    runtime: RuntimeFold,
    pre_last: Tensor,
    lr_b: float,
    epoch_seed: int,
    lr_ratio_b_to_a: float = 16.0,
    eval_batch_size: int = 2,
) -> dict[str, float]:
    """Run one exact LoRA+ epoch with unchanged ordering and diagnostics."""
    set_loraplus_lr(optimizer, lr_b, lr_ratio_b_to_a=lr_ratio_b_to_a)
    optimizer.zero_grad(set_to_none=True)
    losses = accumulate_exact_train_gradient(
        replay=replay,
        head=head,
        runtime=runtime,
        pre_last=pre_last,
        epoch_seed=epoch_seed,
        eval_batch_size=eval_batch_size,
    )
    parameters_a, parameters_b = lora_groups(targets)
    parameters = [*parameters_a, *parameters_b]
    grad_a, grad_b = grad_l2(parameters_a), grad_l2(parameters_b)
    gradient_finite = all_gradients_finite(parameters)
    pre_clip = float(
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, foreach=False)
        .detach()
        .float()
        .cpu()
    )
    clipped = grad_l2(parameters)
    optimizer.step()
    return {
        **losses,
        "grad_A_l2": grad_a,
        "grad_B_l2": grad_b,
        "gradient_l2_before_clip": pre_clip,
        "clipped_gradient_l2": clipped,
        "gradients_finite": gradient_finite,
    }
