"""Executable preflight gates for long CUDA training runs."""

from __future__ import annotations

import copy
import ctypes
import gc
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def efficient_sdpa_only() -> Any:
    """Force memory-efficient SDPA; unsupported kernels must hard-stop."""
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        yield


def available_windows_commit_bytes() -> int | None:
    """Return Windows available commit (ullAvailPageFile), if available."""
    if not hasattr(ctypes, "windll"):
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("GlobalMemoryStatusEx failed.")
    return int(status.ullAvailPageFile)


def require_available_commit(minimum_gib: float = 8.0) -> float | None:
    """Hard-stop a long run when Windows commit headroom is insufficient."""
    available = available_windows_commit_bytes()
    if available is None:
        return None
    available_gib = available / 2**30
    if available_gib < minimum_gib:
        raise RuntimeError(
            "Long-run commit preflight failed: "
            f"available={available_gib:.3f} GiB, required={minimum_gib:.3f} GiB."
        )
    return available_gib


def group_size_audit(
    o_groups: Sequence[Sequence[object]],
    w_groups: Sequence[Sequence[object]],
    *,
    require_six_image_o: bool = True,
) -> dict[str, object]:
    """Measure the real group domain and explicitly accept the six-image O group."""
    o_sizes = np.asarray([len(group) for group in o_groups], dtype=np.int64)
    w_sizes = np.asarray([len(group) for group in w_groups], dtype=np.int64)
    if len(o_sizes) == 0 or len(w_sizes) == 0:
        raise RuntimeError("Group-size preflight requires non-empty O and W populations.")
    if int(o_sizes.min()) < 2 or int(o_sizes.max()) > 8:
        raise RuntimeError("O group size is outside the supported [2, 8] domain.")
    if int(w_sizes.min()) < 2 or int(w_sizes.max()) > 8:
        raise RuntimeError("W group size is outside the supported [2, 8] domain.")
    six_image_o_observed = bool(np.any(o_sizes == 6))
    if require_six_image_o and not six_image_o_observed:
        raise RuntimeError("The canonical six-image O-group was not observed.")
    return {
        "O": {
            "minimum": int(o_sizes.min()),
            "median": float(np.median(o_sizes)),
            "maximum": int(o_sizes.max()),
        },
        "W": {
            "minimum": int(w_sizes.min()),
            "median": float(np.median(w_sizes)),
            "maximum": int(w_sizes.max()),
        },
        "six_image_O_group_observed": six_image_o_observed,
        "six_image_O_group_explicitly_accepted": bool(
            six_image_o_observed or not require_six_image_o
        ),
        "six_image_O_group_required": require_six_image_o,
    }


def require_efficient_attention_forward_backward(
    *,
    last_block: nn.Module,
    post_layernorm: nn.Module,
    worst_case_prefix: torch.Tensor,
) -> dict[str, float | bool | str]:
    """Execute one worst-case SDPA forward/backward before a long run."""
    if worst_case_prefix.device.type != "cuda" or len(worst_case_prefix) != 8:
        raise ValueError("VRAM preflight requires one eight-image CUDA prefix batch.")
    block = copy.deepcopy(last_block).to(worst_case_prefix.device).train()
    norm = copy.deepcopy(post_layernorm).to(worst_case_prefix.device).train()
    block.self_attn.config._attn_implementation = "sdpa"
    sample = worst_case_prefix.detach().requires_grad_(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with efficient_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = norm(block(sample, attention_mask=None))
        loss = output.float().square().mean()
    loss.backward()
    finite = bool(
        torch.isfinite(output).all()
        and torch.isfinite(loss)
        and sample.grad is not None
        and torch.isfinite(sample.grad).all()
        and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in [*block.parameters(), *norm.parameters()]
        )
    )
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    if not finite:
        raise RuntimeError("Efficient-attention VRAM preflight produced non-finite values.")
    del output, loss, sample, block, norm
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "pass": True,
        "backend": "SDPA EFFICIENT_ATTENTION",
        "worst_case_batch_size": 8,
        "peak_allocated_MiB": peak_mib,
        "verification_method": "measured_forward_backward",
    }


def require_trajectory_journal(run: object) -> None:
    """Reject long trajectory execution without a fingerprinted StepRun journal."""
    if getattr(run, "journal", None) is None or not getattr(
        run, "resume_fingerprint", None
    ):
        raise RuntimeError("Long trajectory run requires a fingerprinted Journal.")
