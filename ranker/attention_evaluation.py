"""Attention-head evaluation shared by the Step41A range protocols."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ranker.attention_lora import AttentionResidualHead, attention_diagnostics
from ranker.attention_runtime import LastBlockReplay, RuntimeFold
from ranker.pilots import PilotFold


def selected_lopo_validation_scores(
    *,
    ensemble: Mapping[str, np.ndarray],
    lopo: Mapping[str, Any],
    evaluation_fold_indices: Sequence[int],
    systems: Sequence[str],
) -> dict[str, dict[int, np.ndarray]]:
    """Extract direct and LOPO-selected validation scores from Step41B arrays."""
    frozen_selection = {
        int(row["heldout_pilot_index"]): row
        for row in lopo["Frozen_LOPO_selections"]
    }
    lora_selection = {
        int(row["heldout_pilot_index"]): row
        for row in lopo["LoRA_LOPO_selections"]
    }
    required = {
        "LINEAR_INNER_BAL",
        "HEAD_FIXED",
        "ATTN_FROZEN",
        "ATTN_LORA8_PLUS",
    }
    if set(systems) != required or len(systems) != len(required):
        raise ValueError(f"Expected each attention LOPO system exactly once: {required}.")
    result = {system: {} for system in systems}
    for pilot in evaluation_fold_indices:
        result["LINEAR_INNER_BAL"][pilot] = np.asarray(
            ensemble[f"linear_p{pilot:02d}"], dtype=np.float64
        )
        result["HEAD_FIXED"][pilot] = np.asarray(
            ensemble[f"head_fixed_p{pilot:02d}"], dtype=np.float64
        )
        frozen = frozen_selection[pilot]
        result["ATTN_FROZEN"][pilot] = np.asarray(
            ensemble[f"head_p{pilot:02d}"][
                int(frozen["lr_index"]), int(frozen["epoch"])
            ],
            dtype=np.float64,
        )
        lora = lora_selection[pilot]
        result["ATTN_LORA8_PLUS"][pilot] = np.asarray(
            ensemble[f"lora_p{pilot:02d}"][
                int(lora["lr_index"]), int(lora["epoch"])
            ],
            dtype=np.float64,
        )
    return result


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def forward_head_deltas(
    *,
    head: AttentionResidualHead,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int = 2,
    diagnostics: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate a frozen attention head in deterministic FP32 mini-batches."""
    deltas: list[torch.Tensor] = []
    effective: list[torch.Tensor] = []
    maximum: list[torch.Tensor] = []
    query_cosine: list[float] = []
    logit_cosine: list[float] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            current = indices[start : start + batch_size]
            with torch.autocast(device_type="cuda", enabled=False):
                delta, details = head(
                    tokens.index_select(0, current).float(),
                    diagnostics=diagnostics,
                )
            deltas.append(delta.float().cpu())
            if diagnostics and details is not None:
                values = attention_diagnostics(
                    details["queries"], details["logits"], details["attention"]
                )
                weights = details["attention"].float()
                effective.append(
                    torch.exp(
                        -(weights * torch.log(weights + 1e-12)).sum(dim=-1)
                    ).cpu()
                )
                maximum.append(weights.amax(dim=-1).cpu())
                query_cosine.append(
                    float(values["query_pairwise_cosine_max"].cpu())
                )
                logit_cosine.append(
                    float(values["attention_logit_cosine_max"].cpu())
                )
    report: dict[str, Any] = {}
    if diagnostics:
        report = {
            "effective_tokens_per_query": (
                torch.cat(effective).median(dim=0).values.tolist()
            ),
            "maximum_weight_per_query": (
                torch.cat(maximum).median(dim=0).values.tolist()
            ),
            "query_cosine_max": max(query_cosine),
            "attention_logit_cosine_max": float(np.median(logit_cosine)),
        }
    return torch.cat(deltas).numpy().astype(np.float64), report


def residual_calibration(
    train_delta: np.ndarray,
    train_base: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return train-only residual centering and unit-base-std scaling."""
    mu = float(np.mean(train_delta))
    sigma_delta = float(np.std(train_delta, ddof=0))
    sigma_base = float(np.std(train_base, ddof=0))
    if sigma_delta <= 1e-8 or sigma_base <= 1e-8:
        raise RuntimeError("Invalid train-only residual calibration.")
    return mu, sigma_delta, sigma_base, sigma_base / sigma_delta


def physically_calibrate_attention_head(
    *,
    head: AttentionResidualHead,
    runtime: RuntimeFold,
    frozen_tokens: torch.Tensor,
    batch_size: int = 2,
) -> dict[str, float | bool]:
    """Rescale the final scalar layer and certify score-level parity."""
    train_delta, _ = forward_head_deltas(
        head=head,
        tokens=frozen_tokens,
        indices=runtime.train_cache_indices,
        batch_size=batch_size,
    )
    validation_delta, _ = forward_head_deltas(
        head=head,
        tokens=frozen_tokens,
        indices=runtime.validation_cache_indices,
        batch_size=batch_size,
    )
    mu, sigma_delta, sigma_base, beta = residual_calibration(
        train_delta, runtime.fold.train_base_scores
    )
    target_train = runtime.fold.train_base_scores + (train_delta - mu) * beta
    target_validation = (
        runtime.fold.validation_base_scores + (validation_delta - mu) * beta
    )
    with torch.no_grad():
        head.output.weight.mul_(beta)
        head.output.bias.copy_(beta * (head.output.bias - mu))
    physical_train, _ = forward_head_deltas(
        head=head,
        tokens=frozen_tokens,
        indices=runtime.train_cache_indices,
        batch_size=batch_size,
    )
    physical_validation, _ = forward_head_deltas(
        head=head,
        tokens=frozen_tokens,
        indices=runtime.validation_cache_indices,
        batch_size=batch_size,
    )
    differences = np.concatenate(
        (
            runtime.fold.train_base_scores + physical_train - target_train,
            runtime.fold.validation_base_scores
            + physical_validation
            - target_validation,
        )
    )
    maximum = float(np.max(np.abs(differences)))
    mean_absolute = float(np.mean(np.abs(differences)))
    return {
        "mu_delta": mu,
        "sigma_delta": sigma_delta,
        "sigma_base": sigma_base,
        "beta": beta,
        "max_abs_score_difference": maximum,
        "mean_abs_score_difference": mean_absolute,
        "pass": maximum <= 1e-6 and mean_absolute <= 1e-7,
    }


def forward_scores(
    *,
    head: AttentionResidualHead,
    base: torch.Tensor,
    indices: torch.Tensor,
    frozen_tokens: torch.Tensor | None,
    pre_last: torch.Tensor | None,
    replay: LastBlockReplay | None,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    scores: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    diagnostics_rows: list[tuple[int, dict[str, float]]] = []
    token_delta_stds: list[float] = []
    token_cosines: list[float] = []
    head.eval()
    if replay is not None:
        replay.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            current_indices = indices[start : start + batch_size]
            frozen = (
                None
                if frozen_tokens is None
                else frozen_tokens.index_select(0, current_indices)
            )
            if replay is None:
                if frozen is None:
                    raise AssertionError("Frozen tokens are required without replay.")
                tokens = frozen
            else:
                if pre_last is None:
                    raise AssertionError("Pre-last tokens are required with replay.")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tokens = replay(pre_last.index_select(0, current_indices))
                if frozen is not None:
                    with torch.autocast(device_type="cuda", enabled=False):
                        difference = tokens.float() - frozen.float()
                        token_delta_stds.append(float(difference.std().cpu()))
                        cosine = F.cosine_similarity(
                            tokens.float().reshape(len(tokens), -1),
                            frozen.float().reshape(len(frozen), -1),
                            dim=1,
                        )
                        token_cosines.extend(cosine.cpu().tolist())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                delta, raw = head(tokens, diagnostics=True)
            assert raw is not None
            current_scores = base[start : start + len(current_indices)] + delta
            scores.append(current_scores.float().cpu())
            residuals.append(delta.float().cpu())
            diag = attention_diagnostics(
                raw["queries"], raw["logits"], raw["attention"]
            )
            diagnostics_rows.append(
                (
                    len(current_indices),
                    {name: float(value.cpu()) for name, value in diag.items()},
                )
            )
    total = sum(count for count, _ in diagnostics_rows)
    diagnostics = {
        name: (
            min(values)
            if name == "attention_effective_tokens_min"
            else sum(
                count * value
                for count, row in diagnostics_rows
                for value in [row[name]]
            )
            / total
        )
        for name in diagnostics_rows[0][1]
        for values in [[row[name] for _, row in diagnostics_rows]]
    }
    token_diagnostics = {
        "last_block_token_delta_std": float(np.mean(token_delta_stds))
        if token_delta_stds
        else 0.0,
        "last_block_token_cosine_vs_frozen": float(np.mean(token_cosines))
        if token_cosines
        else 1.0,
    }
    return (
        torch.cat(scores).numpy(),
        torch.cat(residuals).numpy(),
        diagnostics,
        token_diagnostics,
    )


def validation_metrics(fold: PilotFold, scores: np.ndarray) -> dict[str, float | int]:
    differences = scores[fold.validation_higher] - scores[fold.validation_lower]
    losses = np.logaddexp(0.0, -differences)
    aligned = fold.validation_class == 0
    counter = fold.validation_class == 1
    all_cross = aligned | counter

    def accuracy(mask: np.ndarray) -> float:
        return float(np.mean(differences[mask] > 0.0))

    def wpa(mask: np.ndarray) -> float:
        weights = fold.validation_distance[mask].astype(np.float64)
        return float(np.sum(weights * (differences[mask] > 0.0)) / np.sum(weights))

    counter_d2 = counter & (fold.validation_distance >= 2)
    counter_d3 = counter & (fold.validation_distance >= 3)
    aligned_loss = float(np.mean(losses[aligned]))
    counter_loss = float(np.mean(losses[counter]))
    return {
        "validation_aligned_loss": aligned_loss,
        "validation_counter_loss": counter_loss,
        "validation_balanced_loss": 0.5 * aligned_loss + 0.5 * counter_loss,
        "validation_counter_pair_acc": accuracy(counter),
        "validation_counter_WPA": wpa(counter),
        "validation_counter_d_ge_2_pair_acc": accuracy(counter_d2),
        "validation_counter_d_ge_3_pair_acc": accuracy(counter_d3),
        "validation_aligned_pair_acc": accuracy(aligned),
        "validation_aligned_WPA": wpa(aligned),
        "validation_cross_all_WPA": wpa(all_cross),
    }


def validation_metric_sufficient_statistics(
    fold: PilotFold,
    scores: np.ndarray,
) -> dict[str, tuple[int, int]]:
    """Return exact integer numerators/denominators for validation rank metrics."""
    differences = scores[fold.validation_higher] - scores[fold.validation_lower]
    correct = differences > 0.0
    distance = fold.validation_distance.astype(np.int64)
    aligned = fold.validation_class == 0
    counter = fold.validation_class == 1
    cross = aligned | counter

    def pair(mask: np.ndarray) -> tuple[int, int]:
        return int(np.sum(correct[mask])), int(np.sum(mask))

    def weighted(mask: np.ndarray) -> tuple[int, int]:
        return (
            int(np.sum(distance[mask] * correct[mask])),
            int(np.sum(distance[mask])),
        )

    return {
        "counter_pair_acc": pair(counter),
        "counter_WPA": weighted(counter),
        "counter_d_ge_2_pair_acc": pair(counter & (distance >= 2)),
        "counter_d_ge_3_pair_acc": pair(counter & (distance >= 3)),
        "aligned_pair_acc": pair(aligned),
        "aligned_WPA": weighted(aligned),
        "cross_all_pair_acc": pair(cross),
        "cross_all_WPA": weighted(cross),
    }


def evaluate_point(
    *,
    runtime: RuntimeFold,
    head: AttentionResidualHead,
    frozen_tokens: torch.Tensor,
    pre_last: torch.Tensor | None,
    replay: LastBlockReplay | None,
) -> dict[str, float | int]:
    validation_scores, validation_delta, attention, token = forward_scores(
        head=head,
        base=runtime.validation_base,
        indices=runtime.validation_cache_indices,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        replay=replay,
    )
    _, train_delta, _, _ = forward_scores(
        head=head,
        base=runtime.train_base,
        indices=runtime.train_cache_indices,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        replay=replay,
    )
    result: dict[str, float | int] = validation_metrics(runtime.fold, validation_scores)
    result.update(attention)
    result.update(token)
    result.update(
        {
            "residual_std_train": float(np.std(train_delta)),
            "residual_std_validation": float(np.std(validation_delta)),
            "corr_residual_base_train": safe_correlation(
                train_delta, runtime.fold.train_base_scores
            ),
            "corr_residual_base_validation": safe_correlation(
                validation_delta, runtime.fold.validation_base_scores
            ),
        }
    )
    return result
