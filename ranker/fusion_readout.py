"""Shared-parameter L26+L22 readout used by fusion experiments and audits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ranker.attention_lora import AttentionResidualHead
from ranker.frozen_readouts import FrozenFeatureRuntime, linear_scores


class SharedLayerFusionReadout(nn.Module):
    """JOINT4 readout with shared attention parameters and four L22 gates."""

    def __init__(
        self,
        runtime: FrozenFeatureRuntime,
        tokens22: torch.Tensor,
        seed: int,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.tokens22 = tokens22
        self.linear_weight = nn.Parameter(
            torch.randn(1152, dtype=torch.float32, device=runtime.device)
            / math.sqrt(1152.0)
        )
        self.linear_bias = nn.Parameter(
            torch.zeros((), dtype=torch.float32, device=runtime.device)
        )
        self.head = AttentionResidualHead(
            seed=seed,
            initial_query_norm=math.sqrt(128.0),
            query_count=4,
            value_dimension=256,
            zero_output=True,
        ).to(runtime.device)
        self.gates = nn.Parameter(
            torch.zeros(4, dtype=torch.float32, device=runtime.device)
        )

    def common_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            ("linear.weight", self.linear_weight),
            ("linear.bias", self.linear_bias),
            *[
                (f"attention.{name}", parameter)
                for name, parameter in self.head.named_parameters()
            ],
        ]

    def all_parameters(self) -> list[nn.Parameter]:
        return [
            *[parameter for _, parameter in self.common_named_parameters()],
            self.gates,
        ]

    def common_state(self) -> dict[str, Any]:
        return {
            "candidate_id": "L26_FINAL",
            "mode": "joint",
            "queries": 4,
            "linear_weight": self.linear_weight.detach().float().cpu().clone(),
            "linear_bias": self.linear_bias.detach().float().cpu().clone(),
            "head_state_dict": {
                name: value.detach().float().cpu().clone()
                for name, value in self.head.state_dict().items()
            },
        }

    def load_common_state(self, state: dict[str, Any]) -> None:
        """Load a standard JOINT4 state while leaving fusion gates untouched."""
        if state.get("mode") != "joint" or int(state.get("queries", -1)) != 4:
            raise ValueError("Fusion common state must be a JOINT4 readout state.")
        with torch.no_grad():
            self.linear_weight.copy_(state["linear_weight"].to(self.runtime.device))
            self.linear_bias.copy_(state["linear_bias"].to(self.runtime.device))
        self.head.load_state_dict(state["head_state_dict"], strict=True)

    def export_state(self) -> dict[str, Any]:
        """Export a standalone inference state for the complete fusion readout."""
        return {
            **self.common_state(),
            "candidate_id": "L26_L22_FUSION",
            "gates": self.gates.detach().float().cpu().clone(),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Load a standalone fusion state without requiring an optimizer."""
        self.load_common_state(dict(state))
        gates = torch.as_tensor(state["gates"], dtype=torch.float32)
        if gates.shape != (4,):
            raise ValueError(f"Fusion gates must have shape (4,), got {tuple(gates.shape)}")
        with torch.no_grad():
            self.gates.copy_(gates.to(self.runtime.device))

    def _attention(
        self,
        tokens26: torch.Tensor,
        tokens22: torch.Tensor,
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized26 = self.head.token_norm(tokens26)
        keys26 = self.head.key(normalized26)
        values26 = self.head.value(normalized26)
        logits26 = (
            torch.einsum("qd,btd->bqt", self.head.queries, keys26)
            / math.sqrt(128.0)
        )
        normalized22 = self.head.token_norm(tokens22)
        keys22 = self.head.key(normalized22)
        values22 = self.head.value(normalized22)
        logits22 = (
            torch.einsum("qd,btd->bqt", self.head.queries, keys22)
            / math.sqrt(128.0)
        )
        with torch.autocast(device_type=tokens26.device.type, enabled=False):
            attention26 = torch.softmax(logits26.float(), dim=-1)
            pooled26 = torch.einsum("bqt,btd->bqd", attention26, values26.float())
            attention22 = torch.softmax(logits22.float(), dim=-1)
            pooled22 = torch.einsum("bqt,btd->bqd", attention22, values22.float())
            gated22 = gates.float()[None, :, None] * pooled22
            concatenated = (pooled26 + gated22).reshape(tokens26.shape[0], 1024)
        hidden = self.head.dropout(
            F.gelu(self.head.hidden(self.head.output_norm(concatenated)))
        )
        delta = self.head.output(hidden).squeeze(-1).float()
        return delta, pooled26, gated22

    def scores(
        self,
        indices: torch.Tensor,
        *,
        zero_gates: bool = False,
        full_fp32: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        linear = linear_scores(
            self.runtime.pooled,
            indices,
            self.linear_weight,
            self.linear_bias,
        )
        gates = torch.zeros_like(self.gates) if zero_gates else self.gates
        if full_fp32:
            delta, pooled26, gated22 = self._attention(
                self.runtime.tokens.index_select(0, indices).float(),
                self.tokens22.index_select(0, indices).float(),
                gates,
            )
            return linear + delta, pooled26, gated22
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            delta, pooled26, gated22 = self._attention(
                self.runtime.tokens.index_select(0, indices),
                self.tokens22.index_select(0, indices),
                gates,
            )
        return linear + delta, pooled26, gated22

    def capture(
        self,
        indices: torch.Tensor,
        batch_size: int = 64,
        *,
        full_fp32: bool = False,
    ) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        learned_scores: list[torch.Tensor] = []
        zero_scores: list[torch.Tensor] = []
        pooled26_squares = 0.0
        gated22_squares = 0.0
        element_count = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                learned, pooled26, gated22 = self.scores(
                    selected, full_fp32=full_fp32
                )
                zero, _, _ = self.scores(
                    selected, zero_gates=True, full_fp32=full_fp32
                )
                learned_scores.append(learned.detach().cpu())
                zero_scores.append(zero.detach().cpu())
                pooled26_squares += float(pooled26.float().square().sum())
                gated22_squares += float(gated22.float().square().sum())
                element_count += pooled26.numel()
        if was_training:
            self.train()
        return {
            "scores": torch.cat(learned_scores).numpy().astype(np.float64),
            "zero_scores": torch.cat(zero_scores).numpy().astype(np.float64),
            "z26_RMS": math.sqrt(pooled26_squares / element_count),
            "gated_z22_RMS": math.sqrt(gated22_squares / element_count),
            "gates": self.gates.detach().float().cpu().numpy().astype(np.float64),
        }

    def capture_partitions(
        self,
        partitions: Sequence[torch.Tensor],
        *,
        full_fp32: bool = False,
    ) -> dict[str, Any]:
        """Capture scores in stable historical partitions and merge diagnostics."""
        captures = [
            self.capture(partition, full_fp32=full_fp32) for partition in partitions
        ]
        if not captures:
            raise ValueError("At least one capture partition is required.")
        counts = np.asarray([len(partition) for partition in partitions], dtype=np.float64)
        total = float(counts.sum())
        return {
            "scores": np.concatenate([row["scores"] for row in captures]),
            "zero_scores": np.concatenate([row["zero_scores"] for row in captures]),
            "z26_RMS": math.sqrt(
                sum(row["z26_RMS"] ** 2 * count for row, count in zip(captures, counts, strict=True))
                / total
            ),
            "gated_z22_RMS": math.sqrt(
                sum(row["gated_z22_RMS"] ** 2 * count for row, count in zip(captures, counts, strict=True))
                / total
            ),
            "gates": captures[0]["gates"],
        }
