"""Frozen full-encoder extraction and streamed suffix provenance helpers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ranker.attention_runtime import load_vision_model
from ranker.embeddings import resize_normalize_zero_pad_image
from ranker.io import atomic_json, atomic_text, sha256_file
from ranker.last_block_finetune import ExactPrefixCache
from ranker.runtime_preflight import efficient_sdpa_only


def tensor_content_digest(values: torch.Tensor) -> str:
    cpu = values.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    array = cpu.numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def prepare_all_efficient_workspace(
    *,
    feature_dir: Path,
    image_dir: Path,
    model_dir: Path,
    old_prefix: Path,
    extra_prefix: Path,
    device: torch.device,
    reconstruct: bool,
    require_prepared: bool,
    seed_manifest: Path | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, Any, dict[str, Any]]:
    """Load or reconstruct the digest-bound all-efficient L10/L26 workspace."""
    l10_path = feature_dir / "L10_RESIDUAL.npy"
    pooled_path = feature_dir / "L26_POOLED.npy"
    manifest_path = feature_dir / "FEATURE_MANIFEST.json"
    if l10_path.is_file() and pooled_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extraction = manifest.get("extraction", {})
        if extraction.get("signature") == "STEP61A_ALL_EFFICIENT_FEATURES_V4":
            names = list(map(str, manifest["names"]))
            residual = torch.from_numpy(np.load(l10_path, mmap_mode="r"))
            pooled = (
                torch.from_numpy(np.load(pooled_path, allow_pickle=False).copy())
                .view(torch.bfloat16)
                .to(device)
            )
            if tensor_content_digest(residual) != extraction["L10_content_SHA256"]:
                raise RuntimeError("All-efficient L10 scratch digest mismatch.")
            if tensor_content_digest(pooled) != extraction["pooled_content_SHA256"]:
                raise RuntimeError("All-efficient pooled scratch digest mismatch.")
            model, _processor = load_vision_model(model_dir, device)
            model.cpu()
            torch.cuda.empty_cache()
            return names, residual, pooled, model, manifest
    if require_prepared and not reconstruct:
        raise FileNotFoundError("Execution requires prepared all-efficient scratch inputs.")
    feature_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file() or (seed_manifest is not None and seed_manifest.is_file()):
        source = manifest_path if manifest_path.is_file() else seed_manifest
        seed_payload = json.loads(source.read_text(encoding="utf-8"))
        names = list(map(str, seed_payload["names"]))
    else:
        prefix = ExactPrefixCache.load(old_prefix, extra_prefix)
        names = list(prefix.names)
        del prefix
    model, residual, pooled, extraction = extract_l10_workspace(
        names=names,
        image_dir=image_dir,
        model_dir=model_dir,
        workspace_file=l10_path,
        pooled_workspace_file=pooled_path,
        device=device,
    )
    manifest = {
        "names": names,
        "extraction": extraction,
        "source_digests": {
            "old_PRE_LAST": sha256_file(old_prefix) if old_prefix.is_file() else "not_used_remote_reconstruction",
            "extra_PRE_LAST": sha256_file(extra_prefix) if extra_prefix.is_file() else "not_used_remote_reconstruction",
            "model_weights": sha256_file(model_dir / "model.safetensors"),
        },
    }
    atomic_json(manifest_path, manifest)
    model.cpu()
    torch.cuda.empty_cache()
    return names, residual, pooled, model, manifest


@torch.no_grad()
def eager_lineage_diagnostic(
    *,
    old_prefix: Path,
    extra_prefix: Path,
    image_dir: Path,
    model_dir: Path,
    device: torch.device,
    count: int = 16,
) -> dict[str, Any]:
    """Explain historical eager PRE_LAST lineage without exporting its tensors."""
    started = time.perf_counter()
    prefix = ExactPrefixCache.load(old_prefix, extra_prefix)
    selected_names = prefix.names[:count]
    indices = torch.arange(count, dtype=torch.int64, device=device)
    historical = prefix.pre_last(indices, device).float().cpu()
    model, processor = load_vision_model(model_dir, device)
    captured: list[torch.Tensor] = []
    captured_l26: list[torch.Tensor] = []

    def hook_pre_last(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        captured.append(args[0].detach().float().cpu())

    def hook_l26(
        _module: torch.nn.Module, _args: tuple[Any, ...], result: torch.Tensor
    ) -> None:
        captured_l26.append(result.detach().clone())

    handle = model.encoder.layers[26].register_forward_pre_hook(hook_pre_last)
    handle_l26 = model.encoder.layers[26].register_forward_hook(hook_l26)
    mean, std = list(processor.image_mean), list(processor.image_std)

    def execute(
        runtime_model: torch.nn.Module,
        batch_size: int,
        implementation: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in runtime_model.encoder.layers:
            block.self_attn.config._attn_implementation = implementation
        runtime_model.config._attn_implementation = implementation
        rows: list[torch.Tensor] = []
        final_rows: list[torch.Tensor] = []
        for start in range(0, count, batch_size):
            pixels = torch.stack(
                [
                    resize_normalize_zero_pad_image(
                        image_dir / name, mean, std
                    )
                    for name in selected_names[start : start + batch_size]
                ]
            ).to(device)
            captured.clear()
            captured_l26.clear()
            if implementation == "sdpa":
                with efficient_sdpa_only(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    runtime_model(pixel_values=pixels, return_dict=True)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    runtime_model(pixel_values=pixels, return_dict=True)
            if len(captured) != 1 or len(captured_l26) != 1:
                raise RuntimeError("Lineage hooks did not fire exactly once.")
            rows.append(captured[0])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                final_rows.append(
                    runtime_model.post_layernorm(captured_l26[0])
                    .to(torch.bfloat16)
                    .cpu()
                )
        return torch.cat(rows), torch.cat(final_rows)

    try:
        eager, _eager_final = execute(model, 2, "eager")
        efficient, efficient_final = execute(model, 8, "sdpa")
        historical_final_rows: list[torch.Tensor] = []
        for start in range(0, count, 8):
            with efficient_sdpa_only(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                raw = model.encoder.layers[26](
                    historical[start : start + 8].to(device), attention_mask=None
                )
                historical_final_rows.append(
                    model.post_layernorm(raw).to(torch.bfloat16).cpu()
                )
        historical_final = torch.cat(historical_final_rows)
    finally:
        handle.remove()
        handle_l26.remove()
    payload = {
        "images": count,
        "optimizer_updates": 0,
        "tensors_entered_training_heads": False,
        "eager_batch_size": 2,
        "efficient_batch_size": 8,
        "eager_PRE_LAST_exact_historical": bool(torch.equal(eager, historical)),
        "eager_PRE_LAST_max_abs_difference": float(
            (eager - historical).abs().max()
        ),
        "efficient_vs_historical_PRE_LAST_max_abs_difference": float(
            (efficient - historical).abs().max()
        ),
        "all_efficient_vs_PRE_LAST_derived_L26_exact": bool(
            torch.equal(efficient_final, historical_final)
        ),
        "all_efficient_vs_PRE_LAST_derived_L26_max_abs_difference": float(
            (efficient_final.float() - historical_final.float()).abs().max()
        ),
        "runtime_seconds_excluded_from_training_projection": time.perf_counter()
        - started,
    }
    del prefix, historical, historical_final, eager, efficient, efficient_final, model
    torch.cuda.empty_cache()
    return payload


def extract_l10_workspace(
    *,
    names: Sequence[str],
    image_dir: Path,
    model_dir: Path,
    workspace_file: Path,
    pooled_workspace_file: Path,
    device: torch.device,
    tick: Callable[[], None] | None = None,
) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run one all-efficient full encoder pass and retain L10 plus pooled L26."""
    workspace_file.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        workspace_file,
        mode="w+",
        dtype=np.float32,
        shape=(len(names), 1024, 1152),
    )
    pooled_output = np.lib.format.open_memmap(
        pooled_workspace_file,
        mode="w+",
        dtype=np.uint16,
        shape=(len(names), 1152),
    )
    model, processor = load_vision_model(model_dir, device)
    model.config._attn_implementation = "sdpa"
    for block in model.encoder.layers:
        block.self_attn.config._attn_implementation = "sdpa"
    captured_l10: list[torch.Tensor] = []
    captured_l26: list[torch.Tensor] = []

    def hook_l10(_module: torch.nn.Module, _args: tuple[Any, ...], result: torch.Tensor) -> None:
        captured_l10.append(result.detach().clone())

    def hook_l26(
        _module: torch.nn.Module, _args: tuple[Any, ...], result: torch.Tensor
    ) -> None:
        captured_l26.append(result.detach().clone())

    handle_l10 = model.encoder.layers[10].register_forward_hook(hook_l10)
    handle_l26 = model.encoder.layers[26].register_forward_hook(hook_l26)
    mean, std = list(processor.image_mean), list(processor.image_std)
    l26_digest = hashlib.sha256()
    try:
        with torch.inference_mode():
            for start in range(0, len(names), 8):
                stop = min(start + 8, len(names))
                captured_l10.clear()
                captured_l26.clear()
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
                    if len(captured_l10) != 1 or len(captured_l26) != 1:
                        raise RuntimeError("L10/L26 hooks did not fire exactly once.")
                    l10 = captured_l10[0]
                    l26 = model.post_layernorm(captured_l26[0])
                    pooled = F.normalize(model.head(l26).float(), p=2, dim=1).to(
                        torch.bfloat16
                    )
                output[start:stop] = l10.detach().float().cpu().numpy()
                l26_u16 = (
                    l26.detach()
                    .to(torch.bfloat16)
                    .contiguous()
                    .cpu()
                    .view(torch.uint16)
                )
                l26_digest.update(memoryview(l26_u16.numpy()).cast("B"))
                pooled_output[start:stop] = (
                    pooled.detach().cpu().view(torch.uint16).numpy()
                )
                if tick is not None:
                    tick()
    finally:
        handle_l10.remove()
        handle_l26.remove()
    output.flush()
    pooled_output.flush()
    residual = torch.from_numpy(np.load(workspace_file, mmap_mode="r"))
    pooled = (
        torch.from_numpy(np.load(pooled_workspace_file, allow_pickle=False).copy())
        .view(torch.bfloat16)
        .to(device)
    )
    audit = {
        "signature": "STEP61A_ALL_EFFICIENT_FEATURES_V4",
        "source": "one frozen all-efficient full vision encoder forward",
        "layer": "vision_model.encoder.layers.10 output",
        "shape": list(residual.shape),
        "dtype": str(residual.dtype),
        "batch_size": 8,
        "autocast": "CUDA BF16",
        "encoder_layers_0_26": "SDPA EFFICIENT_ATTENTION only",
        "historical_STEP56A_numeric_parity_required": False,
        "L10_content_SHA256": tensor_content_digest(residual),
        "L26_content_SHA256": l26_digest.hexdigest(),
        "pooled_content_SHA256": tensor_content_digest(pooled),
    }
    atomic_text(
        workspace_file.with_suffix(".json"),
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
    )
    return model, residual, pooled, audit


@torch.no_grad()
def streamed_l26_digest(
    *,
    template: Any,
    residual_l10_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
) -> str:
    """Hash post-normalized L26 reconstructed from a raw L10 scratch stream."""
    suffix = torch.nn.ModuleList(
        template.encoder.layers[index].to(device).eval() for index in range(11, 27)
    )
    norm = template.post_layernorm.to(device).eval()
    digest = hashlib.sha256()
    for start in range(0, len(residual_l10_cpu), batch_size):
        stop = min(start + batch_size, len(residual_l10_cpu))
        residual = residual_l10_cpu[start:stop].to(device)
        with efficient_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
            for block in suffix:
                block.self_attn.config._attn_implementation = "sdpa"
                residual = block(residual, attention_mask=None)
            tokens = norm(residual)
        digest.update(
            memoryview(
                tokens.to(torch.bfloat16)
                .contiguous()
                .cpu()
                .view(torch.uint16)
                .numpy()
            ).cast("B")
        )
    suffix.cpu()
    norm.cpu()
    torch.cuda.empty_cache()
    return digest.hexdigest()


@torch.no_grad()
def materialize_streamed_l26(
    *,
    template: Any,
    residual_l10_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Materialize streamed L26 only in RAM for the one-update parity gate."""
    suffix = torch.nn.ModuleList(
        template.encoder.layers[index].to(device).eval() for index in range(11, 27)
    )
    norm = template.post_layernorm.to(device).eval()
    output = torch.empty(
        (len(residual_l10_cpu), 1024, 1152), dtype=torch.bfloat16, device=device
    )
    for start in range(0, len(residual_l10_cpu), batch_size):
        stop = min(start + batch_size, len(residual_l10_cpu))
        residual = residual_l10_cpu[start:stop].to(device)
        with efficient_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
            for block in suffix:
                block.self_attn.config._attn_implementation = "sdpa"
                residual = block(residual, attention_mask=None)
            output[start:stop].copy_(norm(residual).to(torch.bfloat16))
    suffix.cpu()
    norm.cpu()
    return output
