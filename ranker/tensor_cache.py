"""Typed tensor-cache transfer helpers shared by frozen-feature experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_exact_prefix_image_names(old_path: Path, extra_path: Path) -> list[str]:
    """Read and validate the ordered 520+190 image-name index without copying tensors."""
    old = torch.load(old_path, map_location="cpu", weights_only=False, mmap=True)
    extra = torch.load(extra_path, map_location="cpu", weights_only=False, mmap=True)
    old_names = list(map(str, old["image_names"]))
    extra_names = list(map(str, extra["image_names"]))
    if (
        len(old_names) != 520
        or len(extra_names) != 190
        or len(set(old_names)) != 520
        or len(set(extra_names)) != 190
        or set(old_names) & set(extra_names)
    ):
        raise RuntimeError("Exact prefix-cache image-name contract failed.")
    return [*old_names, *extra_names]


def load_numpy_cache_to_gpu_bf16(
    path: Path,
    device: torch.device,
    *,
    expected_rows: int | None = None,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Transfer a NumPy cache to one BF16 CUDA tensor without a full RAM copy."""
    source = np.load(path, mmap_mode="r")
    if expected_rows is not None and len(source) != expected_rows:
        raise RuntimeError(
            f"Cache row-count mismatch: expected {expected_rows}, got {len(source)}."
        )
    target = torch.empty(source.shape, dtype=torch.bfloat16, device=device)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        chunk = torch.from_numpy(np.asarray(source[start:stop]).copy())
        target[start:stop].copy_(chunk.to(device=device, dtype=torch.bfloat16))
    del source
    return target
