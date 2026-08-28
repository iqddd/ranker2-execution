"""Explicit fast-replay LoRA adapter used by Step41B and its audit."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ranker.attention_lora import AttentionResidualHead
from ranker.attention_runtime import LastBlockReplay, RuntimeFold
from ranker.lora_evaluation import train_metrics, validation_metrics
from ranker.lora_protocol import lora_groups

REPLAY_MODE = "fast_replay"


@dataclass
class FastReplayReference:
    """Frozen replay values in the exact batching used by the fast LoRA path."""

    train_plain: torch.Tensor
    train_evaluation_plain: torch.Tensor
    validation_plain: torch.Tensor


def build_fast_replay_reference(
    *,
    runtime: RuntimeFold,
    pre_last: torch.Tensor,
    frozen_template: LastBlockReplay,
    device: torch.device,
    batch_size: int = 8,
) -> FastReplayReference:
    plain = copy.deepcopy(frozen_template).to(device).eval()
    for parameter in plain.parameters():
        parameter.requires_grad_(False)

    def forward(
        indices: torch.Tensor,
        current_batch_size: int,
        replay_model: LastBlockReplay,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(indices), current_batch_size):
                current = indices[start : start + current_batch_size]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs.append(
                        replay_model(pre_last.index_select(0, current)).detach()
                    )
        return torch.cat(outputs).contiguous()

    reference = FastReplayReference(
        train_plain=forward(runtime.train_cache_indices, batch_size, plain),
        train_evaluation_plain=forward(runtime.train_cache_indices, 32, plain),
        validation_plain=forward(runtime.validation_cache_indices, 32, plain),
    )
    del plain
    torch.cuda.empty_cache()
    return reference


def fast_lora_train_pass(
    *,
    replay: LastBlockReplay,
    targets: Sequence[Any],
    head: AttentionResidualHead,
    optimizer: torch.optim.Optimizer,
    runtime: RuntimeFold,
    pre_last: torch.Tensor,
    frozen_tokens: torch.Tensor,
    reference: FastReplayReference,
    lr_b: float,
    epoch_seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    for group in optimizer.param_groups:
        group["lr"] = lr_b / 16.0 if group["lora_factor"] == "A" else lr_b
    optimizer.zero_grad(set_to_none=True)
    order = list(range(len(runtime.group_aligned)))
    random.Random(epoch_seed).shuffle(order)
    lower = torch.as_tensor(
        [first for first in range(8) for _ in range(first + 1, 8)],
        dtype=torch.long,
        device=pre_last.device,
    )
    higher = torch.as_tensor(
        [second for first in range(8) for second in range(first + 1, 8)],
        dtype=torch.long,
        device=pre_last.device,
    )
    aligned_total = torch.zeros((), device=pre_last.device)
    counter_total = torch.zeros((), device=pre_last.device)
    score_buffer = torch.empty(
        len(runtime.train_base), dtype=torch.float32, device=pre_last.device
    )
    replay.eval()
    head.eval()
    for group_index in order:
        start = 8 * group_index
        indices = runtime.train_cache_indices[start : start + 8]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            candidate = replay(pre_last.index_select(0, indices))
        # The correction keeps the frozen block exactly anchored to the
        # upstream batch-of-two cache while evaluating only the LoRA-induced
        # functional delta in the faster group-of-eight partition.
        tokens = (
            frozen_tokens.index_select(0, indices).float()
            + candidate.float()
            - reference.train_plain[start : start + 8].float()
        )
        with torch.autocast(device_type="cuda", enabled=False):
            residual, _ = head(tokens.float())
            scores = runtime.train_base[start : start + 8].float() + residual
            margins = scores.index_select(0, higher) - scores.index_select(0, lower)
            losses = torch.nn.functional.softplus(-margins)
            aligned_sum = losses[runtime.group_aligned[group_index]].sum()
            counter_sum = losses[runtime.group_counter[group_index]].sum()
            loss = (
                0.5 * aligned_sum / runtime.n_aligned
                + 0.5 * counter_sum / runtime.n_counter
            )
        score_buffer[start : start + 8] = scores.detach()
        loss.backward()
        aligned_total = aligned_total + aligned_sum.detach()
        counter_total = counter_total + counter_sum.detach()
    parameters_a, parameters_b = lora_groups(targets)
    parameters = [*parameters_a, *parameters_b]
    grad_a_tensor = torch.sqrt(
        torch.stack(
            [
                parameter.grad.detach().float().square().sum()
                for parameter in parameters_a
                if parameter.grad is not None
            ]
        ).sum()
    )
    grad_b_tensor = torch.sqrt(
        torch.stack(
            [
                parameter.grad.detach().float().square().sum()
                for parameter in parameters_b
                if parameter.grad is not None
            ]
        ).sum()
    )
    finite_tensors = [
        torch.isfinite(parameter.grad).all()
        for parameter in parameters
        if parameter.grad is not None
    ]
    gradients_finite = bool(torch.stack(finite_tensors).all().cpu())
    pre_clip = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, foreach=False)
    clipped_tensor = torch.sqrt(
        torch.stack(
            [
                parameter.grad.detach().float().square().sum()
                for parameter in parameters
                if parameter.grad is not None
            ]
        ).sum()
    )
    optimizer.step()
    values = (
        torch.stack(
            [
                aligned_total / runtime.n_aligned,
                counter_total / runtime.n_counter,
                grad_a_tensor,
                grad_b_tensor,
                pre_clip.detach().float(),
                clipped_tensor,
            ]
        )
        .detach()
        .cpu()
        .numpy()
    )
    report = {
        "optimizer_pass_aligned_loss": float(values[0]),
        "optimizer_pass_counter_loss": float(values[1]),
        "optimizer_pass_balanced_loss": float(0.5 * values[0] + 0.5 * values[1]),
        "grad_A_l2": float(values[2]),
        "grad_B_l2": float(values[3]),
        "gradient_l2_before_clip": float(values[4]),
        "clipped_gradient_l2": float(values[5]),
        "gradients_finite": gradients_finite,
    }
    return report, score_buffer.cpu().numpy().astype(np.float64)


def fast_forward_lora_dataset(
    *,
    replay: LastBlockReplay,
    head: AttentionResidualHead,
    indices: torch.Tensor,
    base: torch.Tensor,
    frozen_tokens: torch.Tensor,
    pre_last: torch.Tensor,
    plain_reference: torch.Tensor,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    scores: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    difference_sum = torch.zeros((), dtype=torch.float64, device=pre_last.device)
    difference_square_sum = torch.zeros_like(difference_sum)
    difference_count = 0
    cosine_sum = torch.zeros_like(difference_sum)
    cosine_count = 0
    token_l2_sum = torch.zeros_like(difference_sum)
    token_l2_count = 0
    replay.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            current = indices[start : start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                candidate = replay(pre_last.index_select(0, current))
            frozen = frozen_tokens.index_select(0, current).float()
            difference = (
                candidate.float()
                - plain_reference[start : start + len(current)].float()
            )
            tokens = frozen + difference
            with torch.autocast(device_type="cuda", enabled=False):
                residual, _ = head(tokens.float())
            residuals.append(residual)
            scores.append(base[start : start + len(current)].float() + residual)
            difference_double = difference.double()
            difference_sum += difference_double.sum()
            difference_square_sum += difference_double.square().sum()
            difference_count += difference.numel()
            cosine_sum += (
                torch.nn.functional.cosine_similarity(tokens.float(), frozen, dim=-1)
                .double()
                .sum()
            )
            cosine_count += tokens.shape[0] * tokens.shape[1]
            token_l2_sum += (
                torch.linalg.vector_norm(difference.float(), dim=-1).double().sum()
            )
            token_l2_count += tokens.shape[0] * tokens.shape[1]
    score_array = torch.cat(scores).float().cpu().numpy().astype(np.float64)
    residual_array = torch.cat(residuals).float().cpu().numpy().astype(np.float64)
    reductions = (
        torch.stack(
            [
                difference_sum,
                difference_square_sum,
                cosine_sum,
                token_l2_sum,
            ]
        )
        .cpu()
        .numpy()
    )
    mean = reductions[0] / difference_count
    variance = max(0.0, reductions[1] / difference_count - mean * mean)
    return (
        score_array,
        residual_array,
        {
            "patch_token_delta_std": math.sqrt(variance),
            "patch_token_cosine_vs_frozen": float(reductions[2] / cosine_count),
            "final_token_delta_l2": float(reductions[3] / token_l2_count),
        },
    )


def fast_evaluate_lora(
    *,
    replay: LastBlockReplay,
    head: AttentionResidualHead,
    runtime: RuntimeFold,
    frozen_tokens: torch.Tensor,
    pre_last: torch.Tensor,
    reference: FastReplayReference,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    train_scores, train_residuals, _ = fast_forward_lora_dataset(
        replay=replay,
        head=head,
        indices=runtime.train_cache_indices,
        base=runtime.train_base,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        plain_reference=reference.train_evaluation_plain,
    )
    validation_scores, validation_residuals, feature = fast_forward_lora_dataset(
        replay=replay,
        head=head,
        indices=runtime.validation_cache_indices,
        base=runtime.validation_base,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        plain_reference=reference.validation_plain,
    )
    metrics = train_metrics(runtime, train_scores)
    metrics.update(validation_metrics(runtime.fold, validation_scores))
    metrics.update(
        {
            "head_residual_std_train": float(np.std(train_residuals)),
            "head_residual_std_validation": float(np.std(validation_residuals)),
            "total_score_std_validation": float(np.std(validation_scores)),
            "corr_residual_base_validation": float(
                np.corrcoef(
                    validation_residuals,
                    runtime.fold.validation_base_scores,
                )[0, 1]
            )
            if np.std(validation_residuals) > 1e-12
            else 0.0,
            **feature,
        }
    )
    return metrics, {
        "train_scores": train_scores,
        "validation_scores": validation_scores,
        "train_residuals": train_residuals,
        "validation_residuals": validation_residuals,
    }


def fast_evaluate_lora_validation(
    *,
    replay: LastBlockReplay,
    head: AttentionResidualHead,
    runtime: RuntimeFold,
    frozen_tokens: torch.Tensor,
    pre_last: torch.Tensor,
    reference: FastReplayReference,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    validation_scores, validation_residuals, feature = fast_forward_lora_dataset(
        replay=replay,
        head=head,
        indices=runtime.validation_cache_indices,
        base=runtime.validation_base,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        plain_reference=reference.validation_plain,
    )
    metrics = validation_metrics(runtime.fold, validation_scores)
    metrics.update(
        {
            "head_residual_std_validation": float(np.std(validation_residuals)),
            "total_score_std_validation": float(np.std(validation_scores)),
            "corr_residual_base_validation": float(
                np.corrcoef(
                    validation_residuals,
                    runtime.fold.validation_base_scores,
                )[0, 1]
            )
            if np.std(validation_residuals) > 1e-12
            else 0.0,
            **feature,
        }
    )
    return metrics, {
        "validation_scores": validation_scores,
        "validation_residuals": validation_residuals,
    }


def fast_evaluate_lora_train_scores(
    *,
    replay: LastBlockReplay,
    head: AttentionResidualHead,
    runtime: RuntimeFold,
    frozen_tokens: torch.Tensor,
    pre_last: torch.Tensor,
    reference: FastReplayReference,
) -> np.ndarray:
    scores, _, _ = fast_forward_lora_dataset(
        replay=replay,
        head=head,
        indices=runtime.train_cache_indices,
        base=runtime.train_base,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        plain_reference=reference.train_plain,
        batch_size=8,
    )
    return scores
