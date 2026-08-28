from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class Float32LayerNorm(nn.LayerNorm):
    """LayerNorm with FP32 arithmetic and the caller's activation dtype."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        with torch.autocast(device_type=value.device.type, enabled=False):
            weight = None if self.weight is None else self.weight.float()
            bias = None if self.bias is None else self.bias.float()
            result = F.layer_norm(value.float(), self.normalized_shape, weight, bias, self.eps)
        return result.to(dtype=dtype)


class AttentionResidualHead(nn.Module):
    """Four-query attention pooling residual head for 1024 SigLIP2 patches."""

    def __init__(
        self,
        *,
        seed: int,
        initial_query_norm: float = 1.0,
        query_count: int = 4,
        value_dimension: int = 256,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(initial_query_norm) or initial_query_norm <= 0.0:
            raise ValueError("initial_query_norm must be finite and positive.")
        if query_count not in (4, 16, 64) or query_count * value_dimension != 1024:
            raise ValueError("Attention queries and values must concatenate to 1024D.")
        self.query_count = int(query_count)
        self.value_dimension = int(value_dimension)
        self.token_norm = Float32LayerNorm(1152)
        self.key = nn.Linear(1152, 128)
        self.value = nn.Linear(1152, value_dimension)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        gaussian = torch.randn(128, query_count, generator=generator)
        orthogonal, _ = torch.linalg.qr(gaussian, mode="reduced")
        self.queries = nn.Parameter(
            (float(initial_query_norm) * orthogonal.T).contiguous()
        )
        self.output_norm = Float32LayerNorm(1024)
        self.hidden = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.10)
        self.output = nn.Linear(256, 1)
        if zero_output:
            nn.init.zeros_(self.output.weight)
        else:
            nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / math.sqrt(256.0))
        nn.init.zeros_(self.output.bias)

    def forward(
        self, tokens: torch.Tensor, *, diagnostics: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        normalized = self.token_norm(tokens)
        keys = self.key(normalized)
        values = self.value(normalized)
        logits = torch.einsum("qd,btd->bqt", self.queries, keys) / math.sqrt(128.0)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            attention = torch.softmax(logits.float(), dim=-1)
            pooled = torch.einsum("bqt,btd->bqd", attention, values.float())
            concatenated = pooled.reshape(tokens.shape[0], 1024)
        hidden = self.dropout(F.gelu(self.hidden(self.output_norm(concatenated))))
        delta = self.output(hidden).squeeze(-1).float()
        if not diagnostics:
            return delta, None
        return delta, {
            "logits": logits.float(),
            "attention": attention,
            "queries": self.queries.float(),
        }


def _off_diagonal_cosines(rows: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    normalized = rows / (torch.linalg.vector_norm(rows, dim=-1, keepdim=True) + eps)
    matrix = normalized @ normalized.transpose(-1, -2)
    indices = torch.triu_indices(matrix.shape[-2], matrix.shape[-1], offset=1, device=matrix.device)
    return matrix[..., indices[0], indices[1]]


def diversity_components(
    queries: torch.Tensor, logits: torch.Tensor, *, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=logits.device.type, enabled=False):
        query_cosines = _off_diagonal_cosines(queries.float(), eps=eps)
        centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
        logit_cosines = _off_diagonal_cosines(centered, eps=eps)
        return query_cosines.square().mean(), logit_cosines.square().mean()


def attention_diagnostics(
    queries: torch.Tensor, logits: torch.Tensor, attention: torch.Tensor
) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=attention.device.type, enabled=False):
        query_cosines = _off_diagonal_cosines(queries.float())
        centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
        logit_cosines = _off_diagonal_cosines(centered)
        weights = attention.float()
        effective = torch.exp(-(weights * torch.log(weights + 1e-12)).sum(dim=-1))
        maximum = weights.amax(dim=-1)
        return {
            "query_pairwise_cosine_mean": query_cosines.mean(),
            "query_pairwise_cosine_max": query_cosines.amax(),
            "attention_logit_cosine_mean": logit_cosines.mean(),
            "attention_logit_cosine_max": logit_cosines.amax(),
            "attention_effective_tokens_mean": effective.mean(),
            "attention_effective_tokens_min": effective.amin(),
            "attention_max_weight_mean": maximum.mean(),
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        self.scaling = float(alpha / rank)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        update = F.linear(F.linear(value, self.lora_A), self.lora_B)
        return base + self.scaling * update


@dataclass(frozen=True)
class LoRATarget:
    name: str
    module: LoRALinear


def install_last_block_lora(last_block: nn.Module) -> tuple[LoRATarget, ...]:
    paths = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "mlp.fc1",
        "mlp.fc2",
    )
    targets: list[LoRATarget] = []
    for path in paths:
        parent = last_block
        components = path.split(".")
        for component in components[:-1]:
            parent = getattr(parent, component)
        leaf = components[-1]
        original = getattr(parent, leaf)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"LoRA target {path} is {type(original).__name__}, expected Linear.")
        wrapped = LoRALinear(original)
        setattr(parent, leaf, wrapped)
        targets.append(LoRATarget(path, wrapped))
    return tuple(targets)


def lora_parameters(targets: Iterable[LoRATarget]) -> list[nn.Parameter]:
    values: list[nn.Parameter] = []
    for target in targets:
        values.extend((target.module.lora_A, target.module.lora_B))
    return values


def lora_norms(targets: Iterable[LoRATarget]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_list = tuple(targets)
    first = torch.cat([target.module.lora_A.reshape(-1) for target in target_list])
    second = torch.cat([target.module.lora_B.reshape(-1) for target in target_list])
    delta_squares = [
        (target.module.scaling * (target.module.lora_B @ target.module.lora_A)).square().sum()
        for target in target_list
    ]
    return (
        torch.linalg.vector_norm(first.float()),
        torch.linalg.vector_norm(second.float()),
        torch.sqrt(torch.stack(delta_squares).sum().float()),
    )
