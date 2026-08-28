"""Reusable attention RankNet primitives used by steps 24 and 25."""
from __future__ import annotations

import random
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import RankedImage
from .group_ranknet import ranknet_target


def set_torch_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def position_features(device: torch.device) -> torch.Tensor:
    axis = 2.0 * (torch.arange(7, dtype=torch.float32, device=device) + 0.5) / 7.0 - 1.0
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([x, y, x * y, x.square() - x.square().mean(), y.square() - y.square().mean(), torch.ones_like(x)], dim=-1).reshape(49, 6)


class ResidualAttention(nn.Module):
    def __init__(self, position: torch.Tensor) -> None:
        super().__init__(); self.register_buffer("position", position)
        self.attention = nn.Linear(1030, 8, dtype=torch.float32)
        self.a_att = nn.Parameter(torch.empty(8, dtype=torch.float32)); self.w_patch = nn.Parameter(torch.zeros(1024, dtype=torch.float32))

    def initialize(self) -> None:
        nn.init.xavier_uniform_(self.attention.weight); nn.init.zeros_(self.attention.bias)
        nn.init.normal_(self.a_att, mean=0.0, std=0.50); nn.init.zeros_(self.w_patch)

    def forward(self, patches: torch.Tensor, *, return_attention: bool = False):
        position = self.position.unsqueeze(0).expand(patches.shape[0], -1, -1)
        hidden = torch.tanh(self.attention(torch.cat([patches, position], dim=-1)))
        alpha = torch.softmax(hidden @ self.a_att, dim=1); delta = torch.sum(alpha.unsqueeze(-1) * patches, dim=1) @ self.w_patch
        return (delta, alpha) if return_attention else delta


def standardize_patch_tokens(raw: Mapping[str, np.ndarray], train_names: list[str], names: list[str], device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    train = np.stack([raw[name] for name in train_names]).astype(np.float32)
    mean, std = train.mean(axis=(0, 1)), np.maximum(train.std(axis=(0, 1), ddof=0), 1e-8)
    return torch.as_tensor(np.stack([(raw[name].astype(np.float32) - mean) / std for name in names]), dtype=torch.float32, device=device), {name: index for index, name in enumerate(names)}


def build_attention_pairs(groups: list[list[RankedImage]], indices: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lower, higher, targets, weights = [], [], [], []
    for group in groups:
        for offset, first in enumerate(group):
            for second in group[offset + 1:]:
                low, high = (first, second) if first.rank < second.rank else (second, first); distance = high.rank - low.rank
                lower.append(indices[low.image_name]); higher.append(indices[high.image_name]); targets.append(ranknet_target(distance)); weights.append(float(distance))
    return tuple(torch.as_tensor(values, dtype=dtype, device=device) for values, dtype in ((lower, torch.int64), (higher, torch.int64), (targets, torch.float32), (weights, torch.float32)))  # type: ignore[return-value]


def attention_rank_loss(scores: torch.Tensor, pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    lower, higher, targets, weights = pairs
    return torch.mean(weights * F.binary_cross_entropy_with_logits(scores[higher] - scores[lower], targets, reduction="none"))


def tensor_score_map(names: list[str], values: torch.Tensor) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values.detach().cpu().numpy(), strict=True)}


def summarize_attention(weights: Mapping[str, np.ndarray], names: list[str], coordinates: np.ndarray) -> dict[str, float]:
    effective, maximum, centers_x, centers_y = [], [], [], []
    for name in names:
        alpha = weights[name]; effective.append(float(np.exp(-np.sum(alpha * np.log(np.maximum(alpha, 1e-30)))))); maximum.append(float(alpha.max())); centers_x.append(float(alpha @ coordinates[:, 0])); centers_y.append(float(alpha @ coordinates[:, 1]))
    return {"effective_patch_count_mean": float(np.mean(effective)), "effective_patch_count_median": float(np.median(effective)), "max_attention_mean": float(np.mean(maximum)), "std_attention_center_x": float(np.std(centers_x, ddof=0)), "std_attention_center_y": float(np.std(centers_y, ddof=0))}
