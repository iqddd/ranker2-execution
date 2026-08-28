"""Cache, replay and runtime-fold mechanics shared by Step41 scripts."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ranker.attention_lora import install_last_block_lora
from ranker.embeddings import resize_normalize_zero_pad_image
from ranker.io import atomic_torch
from ranker.pilots import PilotFold
from ranker.spatial import train_prior_pair_classes

FEATURE_CACHE_SIGNATURE = "step41a-pre-last-and-final-fp32-v2"


@dataclass
class FeatureCache:
    image_names: list[str]
    pre_last: torch.Tensor
    frozen_final: torch.Tensor
    parity_names: list[str]
    parity_full_final: torch.Tensor


def load_vision_model(model_dir: Path, device: torch.device) -> tuple[nn.Module, Any]:
    """Load the eager SigLIP vision model used by both exact and replay paths."""
    from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True).vision_config
    config._attn_implementation = "eager"
    model = (
        SiglipVisionModel.from_pretrained(
            model_dir, config=config, local_files_only=True, dtype=torch.float32
        )
        .eval()
        .to(device)
    )
    model.config._attn_implementation = "eager"
    for layer in model.encoder.layers:
        layer.self_attn.config._attn_implementation = "eager"
    return model, processor


def build_feature_cache(
    *,
    cache_path: Path,
    cache_signature: str,
    model: nn.Module,
    processor: Any,
    image_dir: Path,
    required_names: list[str],
    parity_names: list[str],
    batch_size: int,
    device: torch.device,
) -> FeatureCache:
    """Build or restore the cache while retaining stable image order and dtypes."""
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cached.get("signature") == cache_signature
            and cached.get("image_names") == required_names
            and cached.get("parity_names") == parity_names
            and tuple(cached["pre_last"].shape) == (len(required_names), 1024, 1152)
            and cached["pre_last"].dtype == torch.float32
            and tuple(cached["frozen_final"].shape) == (len(required_names), 1024, 1152)
            and cached["frozen_final"].dtype == torch.float32
        ):
            print("Restored exact pre-last-block cache.", flush=True)
            return FeatureCache(
                required_names,
                cached["pre_last"],
                cached["frozen_final"],
                parity_names,
                cached["parity_full_final"],
            )
    parity_set = set(parity_names)
    pre_batches: list[torch.Tensor] = []
    final_batches: list[torch.Tensor] = []
    parity_outputs: dict[str, torch.Tensor] = {}
    captured: list[torch.Tensor] = []

    def capture_input(_module: nn.Module, args: tuple[Any, ...]) -> None:
        captured.append(args[0].detach().float().cpu())

    hook = model.encoder.layers[-1].register_forward_pre_hook(capture_input)
    mean, std = list(processor.image_mean), list(processor.image_std)
    try:
        for start in range(0, len(required_names), batch_size):
            batch_names = required_names[start : start + batch_size]
            pixels = torch.stack(
                [
                    resize_normalize_zero_pad_image(image_dir / name, mean, std)
                    for name in batch_names
                ]
            ).to(device)
            captured.clear()
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                output = model(pixel_values=pixels, return_dict=True).last_hidden_state
            if len(captured) != 1:
                raise RuntimeError(
                    f"Expected one pre-last capture, got {len(captured)}."
                )
            pre_batches.append(captured[0])
            final_batches.append(output.detach().float().cpu())
            for local, name in enumerate(batch_names):
                if name in parity_set:
                    parity_outputs[name] = output[local].detach().float().cpu()
            print(
                f"Step41A token cache: {min(start + len(batch_names), len(required_names))}/{len(required_names)}",
                flush=True,
            )
    finally:
        hook.remove()
    if set(parity_outputs) != parity_set:
        raise RuntimeError("Replay parity image set was not fully cached.")
    feature_cache = FeatureCache(
        required_names,
        torch.cat(pre_batches).contiguous(),
        torch.cat(final_batches).contiguous(),
        parity_names,
        torch.stack([parity_outputs[name] for name in parity_names]),
    )
    atomic_torch(
        cache_path,
        {
            "signature": cache_signature,
            "image_names": feature_cache.image_names,
            "pre_last": feature_cache.pre_last,
            "frozen_final": feature_cache.frozen_final,
            "parity_names": feature_cache.parity_names,
            "parity_full_final": feature_cache.parity_full_final,
            "preprocessing": "resize-normalize-right-bottom-zero-pad-512",
            "mixed_precision": "BF16 linear/matmul/MLP; FP32 LayerNorm/softmax/reduction",
        },
    )
    return feature_cache


class LastBlockReplay(nn.Module):
    def __init__(self, last_block: nn.Module, final_norm: nn.Module) -> None:
        super().__init__()
        self.last_block = copy.deepcopy(last_block)
        self.final_norm = copy.deepcopy(final_norm)

    def forward(self, pre_last: torch.Tensor) -> torch.Tensor:
        return self.final_norm(self.last_block(pre_last, attention_mask=None))


def replay_tokens(
    model: LastBlockReplay, values: torch.Tensor, batch_size: int = 2
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs.append(model(values[start : start + batch_size]).float().cpu())
    return torch.cat(outputs)


def comparison_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    difference = (reference.float() - candidate.float()).abs()
    cosine = F.cosine_similarity(
        reference.float().reshape(len(reference), -1),
        candidate.float().reshape(len(candidate), -1),
        dim=1,
    )
    return {
        "max_abs_token_diff": float(difference.max()),
        "mean_abs_token_diff": float(difference.mean()),
        "min_per_image_cosine": min(1.0, float(cosine.min())),
    }


def certify_replay(
    *,
    feature_cache: FeatureCache,
    frozen_template: LastBlockReplay,
    parity_names: Sequence[str],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    name_to_cache = {
        name: index for index, name in enumerate(feature_cache.image_names)
    }
    indices = torch.as_tensor(
        [name_to_cache[name] for name in parity_names], dtype=torch.long
    )
    pre_last = feature_cache.pre_last.index_select(0, indices).to(device)
    frozen = copy.deepcopy(frozen_template).to(device).eval()
    replay_metrics = comparison_metrics(
        feature_cache.parity_full_final, replay_tokens(frozen, pre_last)
    )
    replay_metrics["pass"] = bool(
        replay_metrics["max_abs_token_diff"] <= 5e-4
        and replay_metrics["mean_abs_token_diff"] <= 5e-5
        and replay_metrics["min_per_image_cosine"] >= 0.99999
    )
    zero_lora = copy.deepcopy(frozen_template).to(device).eval()
    targets = install_last_block_lora(zero_lora.last_block)
    zero_metrics = comparison_metrics(
        feature_cache.parity_full_final, replay_tokens(zero_lora, pre_last)
    )
    zero_metrics["all_lora_B_zero"] = bool(
        all(int(torch.count_nonzero(target.module.lora_B)) == 0 for target in targets)
    )
    zero_metrics["pass"] = bool(
        zero_metrics["all_lora_B_zero"]
        and zero_metrics["max_abs_token_diff"] <= 5e-4
        and zero_metrics["mean_abs_token_diff"] <= 5e-5
        and zero_metrics["min_per_image_cosine"] >= 0.99999
    )
    del frozen, zero_lora, pre_last
    torch.cuda.empty_cache()
    return replay_metrics, zero_metrics


@dataclass(frozen=True)
class RuntimeFold:
    fold: PilotFold
    train_cache_indices: torch.Tensor
    validation_cache_indices: torch.Tensor
    train_base: torch.Tensor
    validation_base: torch.Tensor
    group_aligned: tuple[torch.Tensor, ...]
    group_counter: tuple[torch.Tensor, ...]
    n_aligned: int
    n_counter: int


def make_runtime_folds(
    folds: Sequence[PilotFold], feature_cache: FeatureCache, device: torch.device
) -> list[RuntimeFold]:
    name_to_cache = {
        name: index for index, name in enumerate(feature_cache.image_names)
    }
    runtime: list[RuntimeFold] = []
    for fold in folds:
        masks = train_prior_pair_classes(fold.spatial.train_o)
        aligned = tuple(
            torch.as_tensor(value, dtype=torch.bool, device=device)
            for value in masks[0]
        )
        counter = tuple(
            torch.as_tensor(value, dtype=torch.bool, device=device)
            for value in masks[1]
        )
        runtime.append(
            RuntimeFold(
                fold,
                torch.as_tensor(
                    [name_to_cache[name] for name in fold.spatial.train_names],
                    dtype=torch.long,
                    device=device,
                ),
                torch.as_tensor(
                    [name_to_cache[name] for name in fold.validation_names],
                    dtype=torch.long,
                    device=device,
                ),
                torch.as_tensor(
                    fold.train_base_scores, dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    fold.validation_base_scores, dtype=torch.float32, device=device
                ),
                aligned,
                counter,
                sum(int(value.sum()) for value in aligned),
                sum(int(value.sum()) for value in counter),
            )
        )
    return runtime


def split_violations(
    folds: Sequence[PilotFold], rows: Sequence[Mapping[str, Any]]
) -> tuple[int, list[dict[str, Any]]]:
    """Check the fixed inner-LOIO train/validation separation contract."""
    details: list[dict[str, Any]] = []
    violations = 0
    for fold, row in zip(folds, rows, strict=True):
        train_names = set(fold.spatial.train_names)
        validation_names = set(fold.validation_names)
        inner_identity = str(row["inner_heldout_identity"])
        inner_in_train = sum(
            item.identity == inner_identity
            for group in fold.spatial.train_o
            for item in group
        )
        overlap = len(train_names & validation_names)
        details.append(
            {
                "pilot_index": int(row["pilot_index"]),
                "train_validation_overlap": overlap,
                "inner_identity_images_in_train": inner_in_train,
            }
        )
        violations += int(overlap != 0) + int(inner_in_train != 0)
    return violations, details


def parity_image_names(source_groups: Sequence[Sequence[Any]]) -> list[str]:
    """Select the original deterministic two-image-per-identity replay panel."""
    by_identity: dict[str, set[str]] = {}
    for group in source_groups:
        for item in group:
            by_identity.setdefault(item.identity, set()).add(item.image_name)
    result = [
        name
        for identity in sorted(by_identity)
        for name in sorted(by_identity[identity])[:2]
    ]
    if len(result) != 28:
        raise AssertionError(f"Expected 28 replay images, got {len(result)}.")
    return result


def load_feature_cache(
    *,
    path: Path,
    signature: str,
    required_names: Sequence[str],
    parity_names: Sequence[str],
) -> FeatureCache:
    """Load a cached replay payload after validating its stable ordering contract."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        raise RuntimeError("Upstream feature cache signature mismatch.")
    if payload.get("image_names") != list(required_names):
        raise RuntimeError("Upstream feature-cache image contract mismatch.")
    if payload.get("parity_names") != list(parity_names):
        raise RuntimeError("Upstream feature-cache parity contract mismatch.")
    return FeatureCache(
        image_names=list(payload["image_names"]),
        pre_last=payload["pre_last"],
        frozen_final=payload["frozen_final"],
        parity_names=list(payload["parity_names"]),
        parity_full_final=payload["parity_full_final"],
    )


def load_validated_exact_feature_cache(
    *,
    path: Path,
    signature: str,
    required_names: Sequence[str],
    parity_names: Sequence[str],
) -> FeatureCache:
    """Load the complete exact cache and validate shapes, dtypes, and finiteness."""
    if not path.is_file():
        raise RuntimeError("The required exact feature cache does not exist.")
    cache = load_feature_cache(
        path=path,
        signature=signature,
        required_names=required_names,
        parity_names=parity_names,
    )
    if len(cache.image_names) != len(set(cache.image_names)):
        raise RuntimeError("The exact feature cache contains duplicate image IDs.")
    expected = (len(required_names), 1024, 1152)
    parity_expected = (len(parity_names), 1024, 1152)
    tensors = {
        "pre_last": (cache.pre_last, expected),
        "frozen_final": (cache.frozen_final, expected),
        "parity_full_final": (cache.parity_full_final, parity_expected),
    }
    for name, (tensor, shape) in tensors.items():
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise RuntimeError(f"Invalid exact-cache tensor contract for {name}.")
        # Avoid materialising a multi-gigabyte boolean tensor while validating the
        # two 520 x 1024 x 1152 cache arrays.  This is the same exact finite check,
        # performed in bounded image chunks.
        for start in range(0, tensor.shape[0], 8):
            if not bool(torch.isfinite(tensor[start : start + 8]).all()):
                raise RuntimeError(f"Non-finite values in exact-cache tensor {name}.")
    return cache


def load_pilot_score_matrix(
    *, path: Path, pilot_indices: tuple[int, ...], n_pilots: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Load finite score rows into their original full pilot-index matrix."""
    with np.load(path, allow_pickle=False) as payload:
        names = np.asarray(payload["image_names"])
        observed_indices = tuple(int(value) for value in payload["pilot_indices"])
        scores = np.asarray(payload["scores"], dtype=np.float64)
    if observed_indices != pilot_indices:
        raise RuntimeError("Upstream inner-linear pilot contract mismatch.")
    if scores.shape != (len(pilot_indices), len(names)) or not np.all(
        np.isfinite(scores)
    ):
        raise RuntimeError("Invalid upstream inner-linear scores.")
    matrix = np.full((n_pilots, len(names)), np.nan, dtype=np.float64)
    for pilot_index, row in zip(observed_indices, scores, strict=True):
        matrix[pilot_index] = row
    return names, matrix
