"""Producer-block cache and JOINT4 model for encoder-adaptation families."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ranker.attention_lora import AttentionResidualHead
from ranker.embeddings import resize_normalize_zero_pad_image
from ranker.io import atomic_torch, sha256_file
from ranker.runtime_preflight import efficient_sdpa_only


class ProducerBlockInputCache:
    """CPU cache with optional one-copy CUDA residency and stable indexing."""

    def __init__(self, names: Sequence[str], values: torch.Tensor) -> None:
        if values.ndim != 3 or tuple(values.shape[1:]) != (1024, 1152):
            raise ValueError("Producer input cache must have shape [N,1024,1152].")
        if len(names) != len(values) or values.dtype != torch.float32:
            raise ValueError("Producer input cache names/dtype mismatch.")
        self.names = list(names)
        self.values = values.contiguous()
        self.name_to_index = {name: index for index, name in enumerate(self.names)}
        self.device_values: torch.Tensor | None = None

    @classmethod
    def load(cls, path: Path) -> ProducerBlockInputCache:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(payload["names"], payload["values"])

    def preload(self, device: torch.device, *, batch_size: int = 8) -> None:
        target = torch.empty_like(self.values, device=device)
        for start in range(0, len(self.values), batch_size):
            target[start : start + batch_size].copy_(
                self.values[start : start + batch_size].to(device)
            )
        self.device_values = target

    def take(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.device_values is not None:
            return self.device_values.index_select(0, indices.to(device))
        return self.values.index_select(0, indices.cpu()).to(device)


def materialize_producer_input_cache(
    *,
    model: nn.Module,
    processor: Any,
    image_dir: Path,
    names: Sequence[str],
    producer_index: int,
    device: torch.device,
    output_path: Path,
    batch_size: int = 2,
    tick: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Capture the eager historical input to one transformer producer block."""
    values = torch.empty((len(names), 1024, 1152), dtype=torch.float32)
    mean, std = list(processor.image_mean), list(processor.image_std)
    started = time.perf_counter()
    model.eval()
    for start in range(0, len(names), batch_size):
        stop = min(start + batch_size, len(names))
        pixels = torch.stack(
            [
                resize_normalize_zero_pad_image(image_dir / name, mean, std)
                for name in names[start:stop]
            ]
        ).to(device)
        captured: list[torch.Tensor] = []
        handle = model.encoder.layers[producer_index].register_forward_pre_hook(
            lambda _module, args, sink=captured: sink.append(args[0].detach().clone())
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(pixel_values=pixels, return_dict=True)
        handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Producer block hook did not fire exactly once.")
        values[start:stop].copy_(captured[0].float().cpu())
        if tick is not None:
            tick()
    atomic_torch(
        output_path,
        {
            "names": list(names),
            "values": values,
            "producer_module": f"encoder.layers.{producer_index}",
            "extraction_backend": "transformers eager",
            "dtype": "float32",
        },
    )
    return {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "images": len(names),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "seconds": time.perf_counter() - started,
        "bytes": output_path.stat().st_size,
    }


class ProducerBlockJointReadout(nn.Module):
    """Train one producer block and JOINT4 while retaining fixed L26 pooling."""

    def __init__(
        self,
        *,
        producer_block: nn.Module,
        path_norm: nn.Module,
        readout_state: Mapping[str, Any],
        fixed_pooled: torch.Tensor,
        seed: int,
        device: torch.device,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.producer_block = copy.deepcopy(producer_block).to(device)
        self.path_norm = copy.deepcopy(path_norm).to(device)
        self.register_buffer("fixed_pooled", fixed_pooled.to(device), persistent=False)
        if attention_backend not in {"sdpa", "eager", "default"}:
            raise ValueError(f"Unsupported attention backend: {attention_backend}")
        self.attention_backend = attention_backend
        self.attention = AttentionResidualHead(
            seed=seed,
            initial_query_norm=math.sqrt(128.0),
            query_count=4,
            value_dimension=256,
            zero_output=True,
        ).to(device)
        self.attention.load_state_dict(readout_state["head_state_dict"], strict=True)
        self.linear_weight = nn.Parameter(
            torch.as_tensor(readout_state["linear_weight"], device=device)
            .detach()
            .float()
            .clone()
        )
        self.linear_bias = nn.Parameter(
            torch.as_tensor(readout_state["linear_bias"], device=device)
            .detach()
            .float()
            .clone()
        )

    def encode(self, producer_input: torch.Tensor) -> torch.Tensor:
        if self.attention_backend == "sdpa":
            self.producer_block.self_attn.config._attn_implementation = "sdpa"
            with efficient_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                return self.path_norm(
                    self.producer_block(producer_input, attention_mask=None)
                )
        if self.attention_backend == "eager":
            self.producer_block.self_attn.config._attn_implementation = "eager"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.path_norm(
                self.producer_block(producer_input, attention_mask=None)
            )

    def forward(
        self, producer_input: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.encode(producer_input)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            attention, _ = self.attention(tokens)
            linear = (
                self.fixed_pooled.index_select(0, indices)
                .to(torch.bfloat16)
                .matmul(self.linear_weight.to(torch.bfloat16))
                .add(self.linear_bias.to(torch.bfloat16))
                .float()
            )
        return linear + attention.float()

    def encoder_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            *[
                (f"encoder.layers.25.{name}", parameter)
                for name, parameter in self.producer_block.named_parameters()
            ],
            *[
                (f"post_layernorm.{name}", parameter)
                for name, parameter in self.path_norm.named_parameters()
            ],
        ]

    def readout_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            ("linear.weight", self.linear_weight),
            ("linear.bias", self.linear_bias),
            *[
                (f"attention.{name}", parameter)
                for name, parameter in self.attention.named_parameters()
            ],
        ]

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            *[parameter for _, parameter in self.encoder_named_parameters()],
            *[parameter for _, parameter in self.readout_named_parameters()],
        ]

    def export_state(self) -> dict[str, Any]:
        return {
            "producer_block_state_dict": {
                name: value.detach().float().cpu().clone()
                for name, value in self.producer_block.state_dict().items()
            },
            "post_layernorm_state_dict": {
                name: value.detach().float().cpu().clone()
                for name, value in self.path_norm.state_dict().items()
            },
            "head_state_dict": {
                name: value.detach().float().cpu().clone()
                for name, value in self.attention.state_dict().items()
            },
            "linear_weight": self.linear_weight.detach().float().cpu().clone(),
            "linear_bias": self.linear_bias.detach().float().cpu().clone(),
        }

    def load_exported_state(self, state: Mapping[str, Any]) -> None:
        self.producer_block.load_state_dict(
            state["producer_block_state_dict"], strict=True
        )
        self.path_norm.load_state_dict(state["post_layernorm_state_dict"], strict=True)
        self.attention.load_state_dict(state["head_state_dict"], strict=True)
        with torch.no_grad():
            self.linear_weight.copy_(
                torch.as_tensor(
                    state["linear_weight"], device=self.linear_weight.device
                )
            )
            self.linear_bias.copy_(
                torch.as_tensor(state["linear_bias"], device=self.linear_bias.device)
            )


def parameter_rows(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    module_role: str,
    optimizer_group: int | str,
) -> list[dict[str, object]]:
    return [
        {
            "parameter_name": name,
            "shape": list(parameter.shape),
            "parameter_count": int(parameter.numel()),
            "module_role": module_role,
            "optimizer_group": optimizer_group,
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in named_parameters
    ]
