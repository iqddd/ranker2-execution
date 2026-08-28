"""Reusable LoRA+ parameter grouping, optimization, and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ranker.attention_lora import LoRATarget

EXPECTED_LORA_PARAMETERS = 161_024


def vector_l2(values: Sequence[Tensor]) -> float:
    """Return the joint FP32 L2 norm, retaining the caller's parameter order."""
    squares = [
        value.detach().float().square().sum() for value in values if value is not None
    ]
    return float(torch.sqrt(torch.stack(squares).sum()).cpu()) if squares else 0.0


def grad_l2(parameters: Sequence[nn.Parameter]) -> float:
    return vector_l2(
        [parameter.grad for parameter in parameters if parameter.grad is not None]
    )


def all_gradients_finite(parameters: Sequence[nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def lora_groups(
    targets: Sequence[LoRATarget],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return LoRA A then B parameters in target-module order."""
    return [target.module.lora_A for target in targets], [
        target.module.lora_B for target in targets
    ]


def lora_diagnostics(
    targets: Sequence[LoRATarget], *, scaling: float = 2.0
) -> dict[str, float]:
    """Measure unscaled and effective BA updates without modifying the adapters."""
    parameters_a, parameters_b = lora_groups(targets)
    ba_squares: list[Tensor] = []
    effective_squares: list[Tensor] = []
    for target in targets:
        ba = target.module.lora_B.float() @ target.module.lora_A.float()
        ba_squares.append(ba.square().sum())
        effective_squares.append((scaling * ba).square().sum())
    return {
        "norm_A": vector_l2(parameters_a),
        "norm_B": vector_l2(parameters_b),
        "norm_effective_BA": float(
            torch.sqrt(torch.stack(ba_squares).sum()).detach().cpu()
        ),
        "effective_update_frobenius": float(
            torch.sqrt(torch.stack(effective_squares).sum()).detach().cpu()
        ),
    }


def module_update_ratios(targets: Sequence[LoRATarget]) -> dict[str, float]:
    """Return effective LoRA update/base-weight ratios in target order."""
    result: dict[str, float] = {}
    for target in targets:
        update = target.module.scaling * (
            target.module.lora_B.float() @ target.module.lora_A.float()
        )
        denominator = torch.linalg.vector_norm(
            target.module.base.weight.detach().float()
        )
        suffix = target.name.replace("self_attn.", "").replace("mlp.", "")
        result[f"effective_update_ratio_{suffix}"] = float(
            (torch.linalg.vector_norm(update) / denominator).detach().cpu()
        )
    return result


def make_loraplus_optimizer(
    targets: Sequence[LoRATarget], *, lr_b: float, lr_ratio_b_to_a: float = 16.0
) -> torch.optim.AdamW:
    """Build the fixed AdamW LoRA+ optimizer with explicit A/B LR ratio."""
    parameters_a, parameters_b = lora_groups(targets)
    return torch.optim.AdamW(
        [
            {
                "params": parameters_a,
                "lr": lr_b / lr_ratio_b_to_a,
                "weight_decay": 0.01,
                "lora_factor": "A",
            },
            {
                "params": parameters_b,
                "lr": lr_b,
                "weight_decay": 0.01,
                "lora_factor": "B",
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def set_loraplus_lr(
    optimizer: torch.optim.Optimizer, lr_b: float, *, lr_ratio_b_to_a: float = 16.0
) -> None:
    """Update the optimizer in place while preserving its A/B group order."""
    for group in optimizer.param_groups:
        group["lr"] = lr_b / lr_ratio_b_to_a if group["lora_factor"] == "A" else lr_b


def hard_pair_indices(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Return lower, higher and rank-distance tensors in stable 28-pair order."""
    lower: list[int] = []
    higher: list[int] = []
    distance: list[int] = []
    for first in range(8):
        for second in range(first + 1, 8):
            lower.append(first)
            higher.append(second)
            distance.append(second - first)
    return (
        torch.as_tensor(lower, dtype=torch.long, device=device),
        torch.as_tensor(higher, dtype=torch.long, device=device),
        torch.as_tensor(distance, dtype=torch.float32, device=device),
    )
