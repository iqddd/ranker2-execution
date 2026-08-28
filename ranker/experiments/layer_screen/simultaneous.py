"""Shared full-token execution for simultaneous intermediate-layer screens."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from ranker.attention_lora import AttentionResidualHead
from ranker.runtime_preflight import efficient_sdpa_only


def _require_exact(flags: Sequence[torch.Tensor], labels: Sequence[str]) -> None:
    """Synchronize once after a batch of device-side exact comparisons."""
    values = torch.stack(tuple(flags)).detach().cpu().tolist()
    for exact, label in zip(values, labels, strict=True):
        if not bool(exact):
            raise RuntimeError(f"All-efficient parity shadow mismatch at {label}.")


class LayerJointHead(nn.Module):
    """Historical JOINT4 readout whose token source is supplied by the caller."""

    def __init__(self, *, seed: int, device: torch.device) -> None:
        super().__init__()
        self.linear_weight = nn.Parameter(
            torch.randn(1152, dtype=torch.float32, device=device) / math.sqrt(1152.0)
        )
        self.linear_bias = nn.Parameter(
            torch.zeros((), dtype=torch.float32, device=device)
        )
        self.attention = AttentionResidualHead(
            seed=seed,
            initial_query_norm=math.sqrt(128.0),
            query_count=4,
            value_dimension=256,
            zero_output=True,
        ).to(device)

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            attention, _ = self.attention(tokens)
            linear = (
                pooled.to(torch.bfloat16)
                .matmul(self.linear_weight.to(torch.bfloat16))
                .add(self.linear_bias.to(torch.bfloat16))
                .float()
            )
        return linear + attention.float()


class MultiLayerScreen(nn.Module):
    """Seventeen independent heads over one frozen L10→L26 suffix stream."""

    def __init__(
        self,
        *,
        template: nn.Module,
        seed: int,
        device: torch.device,
        first_layer: int = 10,
        last_layer: int = 26,
    ) -> None:
        super().__init__()
        self.first_layer = int(first_layer)
        self.last_layer = int(last_layer)
        self.suffix = nn.ModuleList(
            copy.deepcopy(template.encoder.layers[index]).to(device).eval()
            for index in range(first_layer + 1, last_layer + 1)
        )
        for block in self.suffix:
            block.self_attn.config._attn_implementation = "sdpa"
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        self.post_layernorm = copy.deepcopy(template.post_layernorm).to(device).eval()
        for parameter in self.post_layernorm.parameters():
            parameter.requires_grad_(False)
        self.heads = nn.ModuleList(
            LayerJointHead(seed=seed, device=device)
            for _ in range(last_layer - first_layer + 1)
        )
        initial = self.heads[0].state_dict()
        for head in self.heads[1:]:
            head.load_state_dict(initial, strict=True)
        self.shadow_comparisons = 0

    def head_parameter_groups(self) -> list[dict[str, Any]]:
        return [{"params": list(head.parameters())} for head in self.heads]

    def train(self, mode: bool = True) -> MultiLayerScreen:
        """Train only readouts; the frozen encoder suffix always stays in eval mode."""
        self.training = bool(mode)
        self.heads.train(mode)
        self.suffix.eval()
        self.post_layernorm.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for head in self.heads for parameter in head.parameters()]

    @torch.no_grad()
    def capture_l26_tokens(self, residual_l10: torch.Tensor) -> torch.Tensor:
        """Return backend-matched L26 tokens for a parity-only group batch."""
        residual = residual_l10
        shadow = residual_l10.clone()
        exact_flags: list[torch.Tensor] = []
        exact_labels: list[str] = []
        for offset, block in enumerate(self.suffix):
            with (
                efficient_sdpa_only(),
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                residual = block(residual, attention_mask=None)
                shadow = block(shadow, attention_mask=None)
            exact_flags.append(torch.all(residual == shadow))
            exact_labels.append(f"raw layer {self.first_layer + offset + 1}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.post_layernorm(residual).to(torch.bfloat16)
            shadow_tokens = self.post_layernorm(shadow).to(torch.bfloat16)
        exact_flags.append(torch.all(tokens == shadow_tokens))
        exact_labels.append("L26 tokens")
        _require_exact(exact_flags, exact_labels)
        self.shadow_comparisons += len(self.suffix) + 1
        return tokens

    def parameter_count_per_head(self) -> tuple[int, ...]:
        return tuple(
            sum(parameter.numel() for parameter in head.parameters())
            for head in self.heads
        )

    def score_and_backward(
        self,
        residual_l10: torch.Tensor,
        pooled: torch.Tensor,
        loss_callbacks: Sequence[Any],
        source_weight: float,
        rng_states: list[torch.Tensor] | None = None,
    ) -> list[float]:
        """Stream one group and backpropagate each head before advancing suffix."""
        if len(loss_callbacks) != len(self.heads):
            raise ValueError("One loss callback is required per layer head.")
        residual = residual_l10
        shadow = residual_l10.clone()
        losses: list[float] = []
        exact_flags: list[torch.Tensor] = []
        exact_labels: list[str] = []
        for offset, head in enumerate(self.heads):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = self.post_layernorm(residual).to(torch.bfloat16)
                shadow_tokens = self.post_layernorm(shadow).to(torch.bfloat16)
            exact_flags.append(torch.all(tokens == shadow_tokens))
            exact_labels.append(f"head tokens layer {self.first_layer + offset}")
            self.shadow_comparisons += 1
            if rng_states is not None:
                torch.cuda.set_rng_state(rng_states[offset])
            score = head(tokens, pooled)
            if rng_states is not None:
                rng_states[offset] = torch.cuda.get_rng_state()
            loss = loss_callbacks[offset](score)
            (float(source_weight) * loss).backward()
            losses.append(float(loss.detach()))
            if offset < len(self.suffix):
                with (
                    torch.no_grad(),
                    efficient_sdpa_only(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    residual = self.suffix[offset](residual, attention_mask=None)
                    shadow = self.suffix[offset](shadow, attention_mask=None)
                exact_flags.append(torch.all(residual == shadow))
                exact_labels.append(f"raw layer {self.first_layer + offset + 1}")
        _require_exact(exact_flags, exact_labels)
        return losses

    @torch.no_grad()
    def capture_scores(
        self,
        residual_l10: torch.Tensor,
        pooled: torch.Tensor,
    ) -> torch.Tensor:
        modes = [bool(head.training) for head in self.heads]
        self.heads.eval()
        try:
            residual = residual_l10
            shadow = residual_l10.clone()
            rows: list[torch.Tensor] = []
            exact_flags: list[torch.Tensor] = []
            exact_labels: list[str] = []
            for offset, head in enumerate(self.heads):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = self.post_layernorm(residual).to(torch.bfloat16)
                    shadow_tokens = self.post_layernorm(shadow).to(torch.bfloat16)
                    exact_flags.append(torch.all(tokens == shadow_tokens))
                    exact_labels.append(
                        f"evaluation head tokens layer {self.first_layer + offset}"
                    )
                    self.shadow_comparisons += 1
                    rows.append(head(tokens, pooled).float())
                if offset < len(self.suffix):
                    with (
                        efficient_sdpa_only(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        residual = self.suffix[offset](residual, attention_mask=None)
                        shadow = self.suffix[offset](shadow, attention_mask=None)
                    exact_flags.append(torch.all(residual == shadow))
                    exact_labels.append(
                        f"evaluation raw layer {self.first_layer + offset + 1}"
                    )
            _require_exact(exact_flags, exact_labels)
            return torch.stack(rows)
        finally:
            for head, mode in zip(self.heads, modes, strict=True):
                head.train(mode)

    def load_shared_head_state(self, state: Mapping[str, torch.Tensor]) -> None:
        for head in self.heads:
            head.load_state_dict(state, strict=True)
