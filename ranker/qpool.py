"""Reusable full-token query-pooling head and exact group-balanced RankNet helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import RankedImage
from .group_ranknet import ranknet_target


class QPoolBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(128)
        self.self_attention = nn.MultiheadAttention(128, 4, dropout=0.10, batch_first=True)
        self.cross_norm = nn.LayerNorm(128)
        self.cross_attention = nn.MultiheadAttention(128, 4, dropout=0.10, batch_first=True)
        self.mlp_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.Dropout(0.10),
        )

    def forward(
        self,
        queries: torch.Tensor,
        patches: torch.Tensor,
        *,
        return_cross_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.self_norm(queries)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + attended
        normalized = self.cross_norm(queries)
        attended, weights = self.cross_attention(
            normalized,
            patches,
            patches,
            need_weights=return_cross_attention,
            average_attn_weights=False,
        )
        queries = queries + attended
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries, weights if return_cross_attention else None


class FullTokenQPool(nn.Module):
    token_dimension = 1152
    internal_dimension = 128
    number_of_queries = 8

    def __init__(self, *, output_bias: bool = True, pre_normalized_input: bool = False) -> None:
        super().__init__()
        self.pre_normalized_input = pre_normalized_input
        self.input_projection = nn.Linear(1152, 128, bias=False)
        self.queries = nn.Parameter(torch.empty(8, 128))
        self.blocks = nn.ModuleList([QPoolBlock(), QPoolBlock()])
        self.output_norm = nn.LayerNorm(128)
        self.output = nn.Linear(128, 1, bias=output_bias)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        linear_scores: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if tokens.dtype != torch.float32:
            tokens = tokens.float()
        normalized_tokens = tokens if self.pre_normalized_input else F.layer_norm(tokens, (1152,))
        patches = self.input_projection(normalized_tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attention = None
        for index, block in enumerate(self.blocks):
            queries, attention = block(
                queries,
                patches,
                return_cross_attention=return_diagnostics and index == len(self.blocks) - 1,
            )
        pooled = queries.mean(dim=1)
        delta = self.output(self.output_norm(pooled)).squeeze(-1)
        diagnostics = None
        if return_diagnostics:
            if attention is None:
                raise AssertionError("Last-block cross-attention was not returned.")
            diagnostics = {"attention": attention, "query_outputs": queries, "delta": delta}
        return linear_scores + delta, diagnostics


@dataclass(frozen=True)
class PairIndexArrays:
    higher: torch.Tensor
    lower: torch.Tensor
    target: torch.Tensor
    distance: torch.Tensor
    denominator: float


def build_pair_index_arrays(
    groups: Sequence[Sequence[RankedImage]],
    name_to_index: Mapping[str, int],
    device: torch.device,
) -> PairIndexArrays:
    higher: list[int] = []
    lower: list[int] = []
    target: list[float] = []
    distance: list[float] = []
    for group in groups:
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lo, hi = (first, second) if first.rank < second.rank else (second, first)
                d = hi.rank - lo.rank
                higher.append(name_to_index[hi.image_name])
                lower.append(name_to_index[lo.image_name])
                target.append(ranknet_target(d))
                distance.append(float(d))
    if not higher:
        raise ValueError("Pairwise loss requires at least one group.")
    return PairIndexArrays(
        higher=torch.tensor(higher, dtype=torch.long, device=device),
        lower=torch.tensor(lower, dtype=torch.long, device=device),
        target=torch.tensor(target, dtype=torch.float32, device=device),
        distance=torch.tensor(distance, dtype=torch.float32, device=device),
        denominator=84.0 * len(groups),
    )


def balanced_ranknet_loss(
    scores: torch.Tensor,
    pairs_o: PairIndexArrays,
    pairs_w: PairIndexArrays,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for pairs in (pairs_o, pairs_w):
        logits = scores[pairs.higher] - scores[pairs.lower]
        losses.append(
            torch.sum(pairs.distance * F.binary_cross_entropy_with_logits(logits, pairs.target, reduction="none"))
            / pairs.denominator
        )
    return 0.5 * (losses[0] + losses[1])


def learning_rate_for_epoch(epoch: int) -> float:
    if not 1 <= epoch <= 600:
        raise ValueError("Epoch must be in [1, 600].")
    base = 2e-4
    final = 2e-5
    if epoch <= 30:
        return base * epoch / 30.0
    progress = (epoch - 30) / (600 - 30)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


def attention_diagnostics(attention: torch.Tensor, query_outputs: torch.Tensor) -> dict[str, np.ndarray]:
    """Return per-image diagnostics; attention is [B, heads, queries, patches]."""
    probabilities = attention.float().mean(dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12), dim=-1)
    effective = torch.exp(entropy).mean(dim=1)
    maximum = probabilities.max(dim=-1).values.mean(dim=1)
    normalized = F.normalize(query_outputs.float(), p=2, dim=-1)
    cosine = normalized @ normalized.transpose(1, 2)
    upper = torch.triu_indices(8, 8, offset=1, device=cosine.device)
    pairwise = cosine[:, upper[0], upper[1]].mean(dim=1)
    return {
        "effective_patch_count": effective.detach().cpu().numpy().astype(np.float64),
        "max_patch_attention": maximum.detach().cpu().numpy().astype(np.float64),
        "mean_pairwise_query_output_cosine": pairwise.detach().cpu().numpy().astype(np.float64),
    }
