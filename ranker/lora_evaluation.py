"""Shared exact LoRA evaluation metrics for Step41 range and LOPO runs."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ranker.attention_lora import AttentionResidualHead
from ranker.attention_runtime import LastBlockReplay, RuntimeFold
from ranker.lora_exact import replay_indices_cache_batched
from ranker.pilots import PilotFold


def numpy_pair_metrics(
    differences: np.ndarray,
    aligned: np.ndarray,
    counter: np.ndarray,
    distances: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    """Compute the fixed balanced pair metrics without changing pair order."""
    all_cross = aligned | counter
    losses = np.logaddexp(0.0, -differences)

    def accuracy(mask: np.ndarray) -> float:
        count = int(np.sum(mask))
        if count == 0:
            return float("nan")
        return float(np.sum(differences[mask] > 0.0) / count)

    def wpa(mask: np.ndarray) -> float:
        weights = distances[mask].astype(np.float64)
        denominator = float(np.sum(weights))
        if denominator == 0.0:
            return float("nan")
        return float(np.sum(weights * (differences[mask] > 0.0)) / denominator)

    aligned_loss = float(np.mean(losses[aligned]))
    counter_loss = float(np.mean(losses[counter]))
    result = {
        f"{prefix}_balanced_loss": 0.5 * aligned_loss + 0.5 * counter_loss,
        f"{prefix}_aligned_loss": aligned_loss,
        f"{prefix}_counter_loss": counter_loss,
        f"{prefix}_aligned_pair_acc": accuracy(aligned),
        f"{prefix}_aligned_WPA": wpa(aligned),
        f"{prefix}_counter_pair_acc": accuracy(counter),
        f"{prefix}_counter_WPA": wpa(counter),
        f"{prefix}_cross_all_pair_acc": accuracy(all_cross),
        f"{prefix}_cross_all_WPA": wpa(all_cross),
    }
    if prefix == "validation":
        result.update(
            {
                "validation_counter_d_ge_2_pair_acc": accuracy(
                    counter & (distances >= 2)
                ),
                "validation_counter_d_ge_3_pair_acc": accuracy(
                    counter & (distances >= 3)
                ),
            }
        )
    return result


def train_metrics(runtime: RuntimeFold, scores: np.ndarray) -> dict[str, float]:
    differences: list[np.ndarray] = []
    aligned: list[np.ndarray] = []
    counter: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    lower = np.asarray(
        [first for first in range(8) for _ in range(first + 1, 8)], dtype=np.int64
    )
    higher = np.asarray(
        [second for first in range(8) for second in range(first + 1, 8)],
        dtype=np.int64,
    )
    pair_distance = higher - lower
    for group_index in range(len(runtime.group_aligned)):
        group = scores[8 * group_index : 8 * group_index + 8]
        differences.append(group[higher] - group[lower])
        aligned.append(runtime.group_aligned[group_index].detach().cpu().numpy())
        counter.append(runtime.group_counter[group_index].detach().cpu().numpy())
        distances.append(pair_distance)
    return numpy_pair_metrics(
        np.concatenate(differences),
        np.concatenate(aligned).astype(bool),
        np.concatenate(counter).astype(bool),
        np.concatenate(distances),
        prefix="train",
    )


def validation_metrics(fold: PilotFold, scores: np.ndarray) -> dict[str, float]:
    differences = scores[fold.validation_higher] - scores[fold.validation_lower]
    return numpy_pair_metrics(
        differences,
        fold.validation_class == 0,
        fold.validation_class == 1,
        fold.validation_distance,
        prefix="validation",
    )


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def forward_dataset(
    *,
    replay: LastBlockReplay | None,
    head: AttentionResidualHead,
    indices: Tensor,
    base: Tensor,
    frozen_tokens: Tensor,
    pre_last: Tensor,
    eval_batch_size: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Evaluate one dataset through the exact cache-partitioned replay path."""
    scores: list[Tensor] = []
    residuals: list[Tensor] = []
    difference_sum = difference_square_sum = 0.0
    difference_count = 0
    token_cosine_sum = token_l2_sum = 0.0
    token_cosine_count = token_l2_count = 0
    head.eval()
    if replay is not None:
        replay.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), eval_batch_size):
            current = indices[start : start + eval_batch_size]
            frozen = frozen_tokens.index_select(0, current).float()
            if replay is None:
                tokens = frozen
            else:
                tokens = replay_indices_cache_batched(
                    replay=replay,
                    pre_last=pre_last,
                    indices=current,
                    batch_size=eval_batch_size,
                )
                difference = tokens - frozen
                difference_sum += float(difference.double().sum().cpu())
                difference_square_sum += float(difference.double().square().sum().cpu())
                difference_count += difference.numel()
                cosine = F.cosine_similarity(tokens.float(), frozen.float(), dim=-1)
                token_cosine_sum += float(cosine.double().sum().cpu())
                token_cosine_count += cosine.numel()
                per_token_l2 = torch.linalg.vector_norm(difference.float(), dim=-1)
                token_l2_sum += float(per_token_l2.double().sum().cpu())
                token_l2_count += per_token_l2.numel()
            with torch.autocast(device_type="cuda", enabled=False):
                delta, _ = head(tokens.float())
            current_base = base[start : start + len(current)].float()
            residuals.append(delta.float().cpu())
            scores.append((current_base + delta.float()).cpu())
    if replay is None or difference_square_sum == 0.0:
        feature = {
            "patch_token_delta_std": 0.0,
            "patch_token_cosine_vs_frozen": 1.0,
            "final_token_delta_l2": 0.0,
        }
    else:
        mean = difference_sum / difference_count
        variance = max(0.0, difference_square_sum / difference_count - mean * mean)
        feature = {
            "patch_token_delta_std": math.sqrt(variance),
            "patch_token_cosine_vs_frozen": token_cosine_sum / token_cosine_count,
            "final_token_delta_l2": token_l2_sum / token_l2_count,
        }
    return (
        torch.cat(scores).numpy().astype(np.float64),
        torch.cat(residuals).numpy().astype(np.float64),
        feature,
    )


def evaluate_point_r4(
    *,
    replay: LastBlockReplay | None,
    head: AttentionResidualHead,
    runtime: RuntimeFold,
    frozen_tokens: Tensor,
    pre_last: Tensor,
    eval_batch_size: int = 2,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate train and validation arrays with the preserved R4 contract."""
    train_scores, train_residuals, _ = forward_dataset(
        replay=replay,
        head=head,
        indices=runtime.train_cache_indices,
        base=runtime.train_base,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        eval_batch_size=eval_batch_size,
    )
    validation_scores, validation_residuals, feature = forward_dataset(
        replay=replay,
        head=head,
        indices=runtime.validation_cache_indices,
        base=runtime.validation_base,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        eval_batch_size=eval_batch_size,
    )
    metrics = train_metrics(runtime, train_scores)
    metrics.update(validation_metrics(runtime.fold, validation_scores))
    metrics.update(
        {
            "head_residual_std_train": float(np.std(train_residuals)),
            "head_residual_std_validation": float(np.std(validation_residuals)),
            "total_score_std_validation": float(np.std(validation_scores)),
            "corr_residual_base_validation": safe_correlation(
                validation_residuals,
                runtime.fold.validation_base_scores.astype(np.float64),
            ),
            **feature,
        }
    )
    return metrics, {
        "train_scores": train_scores,
        "validation_scores": validation_scores,
        "train_residuals": train_residuals,
        "validation_residuals": validation_residuals,
    }
