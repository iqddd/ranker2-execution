"""Frozen SigLIP2 intermediate-layer cache materialization and diagnostics."""

from __future__ import annotations

import copy
import gc
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ranker.attention_runtime import load_vision_model
from ranker.embeddings import resize_normalize_zero_pad_image
from ranker.last_block_finetune import ExactPrefixCache
from ranker.runtime_preflight import efficient_sdpa_only


def stable_rank_order(values: np.ndarray) -> np.ndarray:
    """Return deterministic zero-based ordinal ranks using stable mergesort."""
    return np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")


def write_layer_cache(path: Path, *, total: int = 710) -> np.memmap:
    """Create the canonical FP32 [image, token, channel] layer cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(total, 1024, 1152)
    )


def materialize_tail_layer_cache(
    layer: int,
    path: Path,
    prefix: ExactPrefixCache,
    template: Any,
    device: torch.device,
    tick: Callable[[], None],
) -> tuple[Path, torch.Tensor | None, float]:
    """Materialize block-25/26 tokens from the exact pre-last-block cache."""
    started = time.perf_counter()
    output = write_layer_cache(path)
    block = (
        copy.deepcopy(template.encoder.layers[-1]).to(device).eval()
        if layer == 26
        else None
    )
    norm = copy.deepcopy(template.post_layernorm).to(device).eval()
    pool = copy.deepcopy(template.head).to(device).eval() if layer == 26 else None
    if block is not None:
        block.self_attn.config._attn_implementation = "sdpa"
    pooled = torch.empty((710, 1152), dtype=torch.bfloat16) if layer == 26 else None
    with torch.inference_mode():
        for start in range(0, 710, 8):
            stop = min(start + 8, 710)
            indices = torch.arange(start, stop, dtype=torch.int64, device=device)
            with efficient_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                values = prefix.pre_last(indices, device)
                tokens = norm(
                    block(values, attention_mask=None) if block is not None else values
                )
                if pool is not None:
                    pooled[start:stop].copy_(
                        F.normalize(pool(tokens).float(), p=2, dim=1).to(
                            dtype=torch.bfloat16, device="cpu"
                        )
                    )
            output[start:stop] = tokens.float().cpu().numpy()
            tick()
    output.flush()
    del output, block, norm, pool, tokens, values
    gc.collect()
    torch.cuda.empty_cache()
    return path, pooled, time.perf_counter() - started


def materialize_early_layer_cache(
    layer: int,
    path: Path,
    names: Sequence[str],
    image_dir: Path,
    model_dir: Path,
    device: torch.device,
    tick: Callable[[], None],
) -> tuple[Path, float]:
    """Materialize an earlier block output followed by the final post-layernorm."""
    started = time.perf_counter()
    output = write_layer_cache(path)
    model, processor = load_vision_model(model_dir, device)
    for block in model.encoder.layers:
        block.self_attn.config._attn_implementation = "sdpa"
    model.config._attn_implementation = "sdpa"
    captured: list[torch.Tensor] = []

    def hook(
        _module: torch.nn.Module, _args: tuple[Any, ...], result: torch.Tensor
    ) -> None:
        captured.append(result)

    handle = model.encoder.layers[layer].register_forward_hook(hook)
    mean, std = list(processor.image_mean), list(processor.image_std)
    try:
        with torch.inference_mode():
            for start in range(0, 710, 8):
                stop = min(start + 8, 710)
                captured.clear()
                pixels = torch.stack(
                    [
                        resize_normalize_zero_pad_image(image_dir / name, mean, std)
                        for name in names[start:stop]
                    ]
                ).to(device)
                with efficient_sdpa_only(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    model(pixel_values=pixels, return_dict=True)
                    if len(captured) != 1:
                        raise RuntimeError("Layer hook did not capture exactly once.")
                    tokens = model.post_layernorm(captured[0])
                output[start:stop] = tokens.float().cpu().numpy()
                tick()
    finally:
        handle.remove()
    output.flush()
    del output, model, processor, pixels, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return path, time.perf_counter() - started


def classify_paired_seed_deltas(
    deltas: Sequence[float], *, practical_floor: float
) -> tuple[float, float, float, str]:
    """Classify a three-seed paired contrast with an explicit practical floor."""
    mean = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    half_width = max(practical_floor, 0.5 * sd)
    negative = sum(value < 0 for value in deltas)
    positive = sum(value > 0 for value in deltas)
    if mean < -half_width and negative >= 2:
        label = "resolved_improvement"
    elif mean > half_width and positive >= 2:
        label = "resolved_harm"
    elif (negative == 3 or positive == 3) and abs(mean) <= half_width:
        label = "below_target_uncertainty_floor"
    else:
        label = "statistically_unresolved"
    return mean, sd, half_width, label
