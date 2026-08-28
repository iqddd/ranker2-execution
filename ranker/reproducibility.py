"""Explicit deterministic-runtime setup shared by Step41 protocols."""

from __future__ import annotations

import os
import random
import hashlib
import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


def configure_determinism(seed: int | None = None) -> None:
    """Apply deterministic CUDA settings and optionally seed every RNG."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if seed is not None:
        seed_everything(seed)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch and all CUDA devices from one explicit seed."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_state_hash(state: Mapping[str, Any]) -> str:
    """Hash tensor leaves of a nested state mapping in stable key order.

    ``candidate_id`` is descriptive metadata and is deliberately excluded so
    bit-identical initializations can be compared across experimental arms.
    """
    digest = hashlib.sha256()

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(prefix.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value):
                if key != "candidate_id":
                    visit(f"{prefix}.{key}", value[key])

    visit("state", state)
    return digest.hexdigest()


def clone_state_with_candidate_id(
    state: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    """Deep-copy a model state while changing only descriptive candidate metadata."""
    result = copy.deepcopy(dict(state))
    result["candidate_id"] = candidate_id
    return result


def ordered_names_hash(names: list[str] | tuple[str, ...]) -> str:
    """Hash an ordered sequence of names using the historical newline encoding."""
    return hashlib.sha256("\n".join(names).encode()).hexdigest()
