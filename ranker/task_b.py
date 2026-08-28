from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import RankedImage


@dataclass(frozen=True)
class PairDataset:
    pair_features: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class TrainedTaskBModel:
    weights: np.ndarray
    mu: float
    sigma: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class RankNetStandardization:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class GlobalRankNetBundle:
    encoder_name: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    ranknet_weight: np.ndarray
    train_score_mean: float
    train_score_std: float
    sorted_train_scores: np.ndarray
    alpha: float
    target_gamma: float
    target_max: float


def pair_target(distance: int) -> float:
    return 0.5 + 0.45 * ((distance / 7.0) ** 0.7)


def compute_standardization(
    image_names: list[str],
    embeddings: dict[str, np.ndarray],
) -> RankNetStandardization:
    matrix = np.stack(
        [np.asarray(embeddings[name], dtype=np.float64) for name in image_names],
        axis=0,
    )
    feature_mean = matrix.mean(axis=0)
    feature_std = matrix.std(axis=0)
    safe_std = np.where(feature_std > 0.0, feature_std, 1.0)
    return RankNetStandardization(mean=feature_mean, std=safe_std)


def apply_standardization(
    image_names: list[str],
    embeddings: dict[str, np.ndarray],
    standardization: RankNetStandardization,
) -> dict[str, np.ndarray]:
    return {
        name: (
            np.asarray(embeddings[name], dtype=np.float64) - standardization.mean
        )
        / standardization.std
        for name in image_names
    }


def build_pair_dataset(
    groups: list[list[RankedImage]],
    embeddings: dict[str, np.ndarray],
) -> PairDataset:
    pair_features: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item_i = group[i]
                item_j = group[j]
                if item_j.rank > item_i.rank:
                    lower = item_i
                    upper = item_j
                else:
                    lower = item_j
                    upper = item_i

                distance = upper.rank - lower.rank
                pair_features.append(
                    embeddings[upper.image_name] - embeddings[lower.image_name]
                )
                targets.append(pair_target(distance))
                weights.append(float(distance))

    return PairDataset(
        pair_features=torch.tensor(np.stack(pair_features), dtype=torch.float64),
        targets=torch.tensor(targets, dtype=torch.float64),
        weights=torch.tensor(weights, dtype=torch.float64),
    )


def fit_pairwise_weight(
    pair_dataset: PairDataset,
    embedding_dim: int,
    l2: float,
    *,
    max_iter: int,
    tolerance: float,
    average_pairs: bool = False,
) -> torch.Tensor:
    weights = torch.zeros(embedding_dim, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights],
        max_iter=max_iter,
        tolerance_grad=tolerance,
        tolerance_change=tolerance,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = pair_dataset.pair_features.mv(weights)
        losses = F.binary_cross_entropy_with_logits(
            logits,
            pair_dataset.targets,
            reduction="none",
        )
        pair_loss = pair_dataset.weights * losses
        if average_pairs:
            pair_term = torch.mean(pair_loss)
        else:
            pair_term = torch.sum(pair_loss)
        loss = pair_term + l2 * torch.sum(weights * weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach()


def build_raw_scores(
    items: list[RankedImage],
    embeddings: dict[str, np.ndarray],
    weights: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    if isinstance(weights, torch.Tensor):
        weight_array = weights.detach().cpu().numpy().astype(np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)

    matrix = np.stack([embeddings[item.image_name] for item in items], axis=0).astype(np.float64)
    raw_scores = matrix @ weight_array
    return {
        item.image_name: float(score)
        for item, score in zip(items, raw_scores, strict=True)
    }


def percentile_score_from_sorted_train(
    sorted_train_scores: np.ndarray,
    raw_score: float,
) -> float:
    left = int(np.searchsorted(sorted_train_scores, raw_score, side="left"))
    right = int(np.searchsorted(sorted_train_scores, raw_score, side="right"))
    return 100.0 * (left + 0.5 * (right - left)) / len(sorted_train_scores)


def build_percentile_scores(
    raw_scores: dict[str, float],
    sorted_train_scores: np.ndarray,
) -> dict[str, float]:
    return {
        image_name: percentile_score_from_sorted_train(sorted_train_scores, raw_score)
        for image_name, raw_score in raw_scores.items()
    }


def build_z_scores(
    raw_scores: dict[str, float],
    mean_score: float,
    std_score: float,
) -> dict[str, float]:
    std_safe = 1.0 if math.isclose(std_score, 0.0) else std_score
    return {
        image_name: (raw_score - mean_score) / std_safe
        for image_name, raw_score in raw_scores.items()
    }


def fit_linear_calibration(
    items: list[RankedImage],
    standardized_scores: dict[str, float],
) -> tuple[float, float]:
    x = np.asarray([standardized_scores[item.image_name] for item in items], dtype=np.float64)
    y = np.asarray([item.z for item in items], dtype=np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(alpha), float(beta)


def build_score_b_from_raw(
    raw_scores: dict[str, float],
    mu: float,
    sigma: float,
    alpha: float,
    beta: float,
) -> tuple[dict[str, float], dict[str, float]]:
    sigma_safe = 1.0 if math.isclose(sigma, 0.0) else sigma
    standardized = {
        image_name: (score - mu) / sigma_safe
        for image_name, score in raw_scores.items()
    }
    z_hat = {
        image_name: alpha * standardized[image_name] + beta
        for image_name in raw_scores
    }
    score_b = {
        image_name: 50.0 * (min(1.0, max(-1.0, z_hat[image_name])) + 1.0)
        for image_name in raw_scores
    }
    return z_hat, score_b


def save_model_bundle(
    output_dir: Path,
    model: TrainedTaskBModel,
    summary_payload: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "task_b_model_weights.npz"
    np.savez(
        weights_path,
        weights=model.weights,
        mu=np.asarray([model.mu], dtype=np.float64),
        sigma=np.asarray([model.sigma], dtype=np.float64),
        alpha=np.asarray([model.alpha], dtype=np.float64),
        beta=np.asarray([model.beta], dtype=np.float64),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return weights_path, summary_path


def save_global_ranknet_bundle(
    output_dir: Path,
    *,
    encoder_name: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    ranknet_weight: np.ndarray,
    train_score_mean: float,
    train_score_std: float,
    sorted_train_scores: np.ndarray,
    alpha: float,
    target_gamma: float,
    target_max: float,
    summary_payload: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "final_B_ranknet_bundle.npz"
    np.savez(
        bundle_path,
        encoder_name=np.asarray([encoder_name]),
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_std=np.asarray(feature_std, dtype=np.float64),
        ranknet_weight=np.asarray(ranknet_weight, dtype=np.float64),
        train_score_mean=np.asarray([train_score_mean], dtype=np.float64),
        train_score_std=np.asarray([train_score_std], dtype=np.float64),
        sorted_train_scores=np.asarray(sorted_train_scores, dtype=np.float64),
        alpha=np.asarray([alpha], dtype=np.float64),
        target_gamma=np.asarray([target_gamma], dtype=np.float64),
        target_max=np.asarray([target_max], dtype=np.float64),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return bundle_path, summary_path


def load_global_ranknet_bundle(bundle_path: Path) -> GlobalRankNetBundle:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing global RankNet bundle: {bundle_path}")

    payload = np.load(bundle_path, allow_pickle=False)
    return GlobalRankNetBundle(
        encoder_name=str(payload["encoder_name"][0]),
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(payload["feature_std"], dtype=np.float64),
        ranknet_weight=np.asarray(payload["ranknet_weight"], dtype=np.float64),
        train_score_mean=float(payload["train_score_mean"][0]),
        train_score_std=float(payload["train_score_std"][0]),
        sorted_train_scores=np.asarray(payload["sorted_train_scores"], dtype=np.float64),
        alpha=float(payload["alpha"][0]),
        target_gamma=float(payload["target_gamma"][0]),
        target_max=float(payload["target_max"][0]),
    )


def load_model_bundle(model_dir: Path) -> TrainedTaskBModel:
    weights_path = model_dir / "task_b_model_weights.npz"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing model bundle in {model_dir}. Expected task_b_model_weights.npz."
        )
    payload = np.load(weights_path)
    return TrainedTaskBModel(
        weights=np.asarray(payload["weights"], dtype=np.float64),
        mu=float(payload["mu"][0]),
        sigma=float(payload["sigma"][0]),
        alpha=float(payload["alpha"][0]),
        beta=float(payload["beta"][0]),
    )
