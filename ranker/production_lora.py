"""Final-only fixed LoRA+ training for production-compatible cross-fit folds."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ranker.attention_lora import install_last_block_lora
from ranker.attention_runtime import LastBlockReplay, RuntimeFold
from ranker.lora_evaluation import forward_dataset
from ranker.lora_exact import train_exact_lora_epoch
from ranker.lora_protocol import EXPECTED_LORA_PARAMETERS, make_loraplus_optimizer
from ranker.reproducibility import seed_everything


def _trainable_state_finite(
    parameters: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer
) -> bool:
    """Check LoRA parameters/gradients and optimizer state, not frozen base weights."""
    for parameter in parameters:
        if not bool(torch.isfinite(parameter).all()):
            return False
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            return False
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                return False
    return True


def train_fixed_production_lora(
    *,
    runtime: RuntimeFold,
    frozen_tokens: torch.Tensor,
    pre_last: torch.Tensor,
    frozen_template: LastBlockReplay,
    head: torch.nn.Module,
    head_scores: np.ndarray,
    seed: int,
    lr_b: float,
    epochs: int,
    device: torch.device,
    after_epoch: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train one exact batch-two LoRA trajectory and evaluate only epoch 0/final."""
    seed_everything(seed)
    replay = copy.deepcopy(frozen_template).to(device)
    for parameter in replay.parameters():
        parameter.requires_grad_(False)
    targets = install_last_block_lora(replay.last_block)
    n_parameters = sum(
        parameter.numel() for parameter in replay.parameters() if parameter.requires_grad
    )
    if n_parameters != EXPECTED_LORA_PARAMETERS:
        raise RuntimeError(
            f"Expected {EXPECTED_LORA_PARAMETERS} LoRA parameters, got {n_parameters}."
        )
    trainable_parameters = [
        parameter for parameter in replay.parameters() if parameter.requires_grad
    ]
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head.eval()
    optimizer = make_loraplus_optimizer(targets, lr_b=lr_b)
    epoch0_scores, _, _ = forward_dataset(
        replay=replay,
        head=head,
        indices=runtime.validation_cache_indices,
        base=runtime.validation_base,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        eval_batch_size=2,
    )
    epoch0_difference = epoch0_scores - head_scores
    epoch0_parity = {
        "max_abs_score_difference": float(np.max(np.abs(epoch0_difference))),
        "mean_abs_score_difference": float(np.mean(np.abs(epoch0_difference))),
        "pass": bool(np.max(np.abs(epoch0_difference)) <= 1e-6),
    }
    if not epoch0_parity["pass"]:
        raise RuntimeError("Fixed LoRA epoch-0 parity failed.")
    training_seconds = 0.0
    last_training: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = train_exact_lora_epoch(
            replay=replay,
            targets=targets,
            head=head,
            optimizer=optimizer,
            runtime=runtime,
            pre_last=pre_last,
            lr_b=lr_b,
            epoch_seed=seed + epoch,
            lr_ratio_b_to_a=16.0,
            eval_batch_size=2,
        )
        training_seconds += time.perf_counter() - started
        last_training = training
        if (
            not _trainable_state_finite(trainable_parameters, optimizer)
            or not all(
                not isinstance(value, (float, int)) or math.isfinite(float(value))
                for value in training.values()
            )
        ):
            raise RuntimeError("Fixed LoRA trajectory became non-finite.")
        if after_epoch is not None:
            after_epoch(epoch)
    final_scores, _, _ = forward_dataset(
        replay=replay,
        head=head,
        indices=runtime.validation_cache_indices,
        base=runtime.validation_base,
        frozen_tokens=frozen_tokens,
        pre_last=pre_last,
        eval_batch_size=2,
    )
    record = {
        "seed": seed,
        "lr_A": lr_b / 16.0,
        "lr_B": lr_b,
        "epochs": epochs,
        "n_lora_parameters": n_parameters,
        "epoch0_parity": epoch0_parity,
        "last_train_metrics": last_training,
        "finite": bool(np.all(np.isfinite(final_scores))),
        "training_seconds": training_seconds,
    }
    del replay, optimizer
    torch.cuda.empty_cache()
    return final_scores, record
