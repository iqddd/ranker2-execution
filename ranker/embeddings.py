from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from timm.data import create_transform, resolve_model_data_config

from .data import RankedImage
from .io import atomic_torch


def standardize_feature_map(
    raw_features: dict[str, np.ndarray],
    train_groups: list[list[RankedImage]],
    eval_groups: list[list[RankedImage]],
    *,
    min_std: float = 1e-8,
) -> dict[str, np.ndarray]:
    train_names = collect_unique_image_names(train_groups)
    names = sorted(set(train_names) | set(collect_unique_image_names(eval_groups)))
    train_matrix = np.stack([np.asarray(raw_features[name], dtype=np.float64) for name in train_names])
    mean = train_matrix.mean(axis=0)
    std = np.maximum(train_matrix.std(axis=0, ddof=0), min_std)
    return {name: (np.asarray(raw_features[name], dtype=np.float64) - mean) / std for name in names}


def standardize_array_from_train(
    train_values: np.ndarray, values: np.ndarray, *, min_std: float = 1e-8
) -> np.ndarray:
    train = np.asarray(train_values, dtype=np.float64)
    return (np.asarray(values, dtype=np.float64) - train.mean(axis=0)) / np.maximum(train.std(axis=0, ddof=0), min_std)


def extract_siglip_pooler_features(
    *,
    model: torch.nn.Module,
    processor: Any,
    image_dir: Path,
    image_names: list[str],
    cache_path: Path,
    cache_signature: str,
    device: torch.device,
    batch_size: int = 2,
) -> dict[str, np.ndarray]:
    """Extract or restore ordered FP32 SigLIP vision pooler features."""
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("signature") == cache_signature
            and payload.get("image_names") == image_names
            and isinstance(payload.get("features"), dict)
        ):
            restored = {
                name: np.asarray(payload["features"][name], dtype=np.float64)
                for name in image_names
            }
            if all(value.shape == (1152,) for value in restored.values()):
                return restored
    mean, std = list(processor.image_mean), list(processor.image_std)
    features: dict[str, np.ndarray] = {}
    model.eval()
    for start in range(0, len(image_names), batch_size):
        names = image_names[start : start + batch_size]
        pixels = torch.stack(
            [
                resize_normalize_zero_pad_image(image_dir / name, mean, std)
                for name in names
            ]
        ).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            values = model(pixel_values=pixels, return_dict=True).pooler_output
        values = F.normalize(values.float(), p=2, dim=1)
        for name, value in zip(names, values.cpu().numpy(), strict=True):
            features[name] = np.asarray(value, dtype=np.float64)
    if not all(value.shape == (1152,) and np.all(np.isfinite(value)) for value in features.values()):
        raise RuntimeError("Invalid SigLIP pooler feature extraction.")
    atomic_torch(
        cache_path,
        {
            "signature": cache_signature,
            "image_names": image_names,
            "features": features,
            "precision": "FP32",
            "batch_size": batch_size,
            "normalization": "rowwise L2, matching canonical Step26 S2 cache",
        },
    )
    return features


def standardize_feature_blocks(
    blocks: list[dict[str, np.ndarray]] | tuple[dict[str, np.ndarray], ...],
    train_groups: list[list[RankedImage]],
    eval_groups: list[list[RankedImage]],
    *, min_std: float = 1e-8,
) -> tuple[dict[str, np.ndarray], ...]:
    return tuple(standardize_feature_map(block, train_groups, eval_groups, min_std=min_std) for block in blocks)


def resize_normalize_zero_pad_image(path: Path, mean: list[float], std: list[float], *, size: int = 512) -> torch.Tensor:
    """EXIF-aware resize, normalize and right/bottom pad used by S2 experiments."""
    from PIL import ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        scale = min(size / width, size / height)
        new_width, new_height = round(width * scale), round(height * scale)
        image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - torch.tensor(mean, dtype=torch.float32).reshape(3, 1, 1)) / torch.tensor(std, dtype=torch.float32).reshape(3, 1, 1)
    return F.pad(tensor, (0, size - new_width, 0, size - new_height), value=0.0)


MODEL_REPO_ID = "timm/vit_large_patch14_dinov2.lvd142m"
MODEL_ARCH = "vit_large_patch14_dinov2"


def load_frozen_encoder(device: torch.device) -> tuple[torch.nn.Module, Any]:
    weights_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="model.safetensors",
        local_files_only=True,
    )
    state_dict = load_file(weights_path, device="cpu")
    model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=0, global_pool="token")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform


def collect_unique_image_names(ranked_groups: list[list[RankedImage]]) -> list[str]:
    return sorted({item.image_name for group in ranked_groups for item in group})


def extract_siglip2_full_patch_tokens(
    image_dir: Path,
    image_names: list[str],
    cache_path: Path,
    model_dir: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    """Cache production-preprocessed SigLIP2 final-layer patch tokens in bf16."""
    signature = "siglip2-so400m-patch16-512-full-patches-rbpad-v1"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_names = payload.get("image_names")
        tokens = payload.get("tokens")
        if (
            payload.get("signature") == signature
            and cached_names == image_names
            and isinstance(tokens, torch.Tensor)
            and tokens.dtype == torch.bfloat16
            and tuple(tokens.shape) == (len(image_names), 1024, 1152)
            and bool(torch.isfinite(tokens.float()).all())
        ):
            return list(cached_names), tokens.contiguous()

    if device.type != "cuda":
        raise RuntimeError("Full SigLIP2 token extraction requires CUDA.")
    from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True).vision_config
    model = SiglipVisionModel.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        dtype=torch.float32,
    ).eval().to(device)
    mean = list(processor.image_mean)
    std = list(processor.image_std)
    batches: list[torch.Tensor] = []
    for start in range(0, len(image_names), batch_size):
        batch_names = image_names[start : start + batch_size]
        pixels = torch.stack(
            [resize_normalize_zero_pad_image(image_dir / name, mean, std) for name in batch_names]
        ).to(device)
        with torch.inference_mode():
            output = model(pixel_values=pixels, return_dict=True)
            tokens = output.last_hidden_state
        if tuple(tokens.shape[1:]) != (1024, 1152):
            raise ValueError(f"Expected SigLIP2 patch tokens [1024, 1152], got {tuple(tokens.shape[1:])}.")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("SigLIP2 generated non-finite patch tokens.")
        batches.append(tokens.to(device="cpu", dtype=torch.bfloat16).contiguous())
        print(f"SigLIP2 full tokens: {min(start + len(batch_names), len(image_names))}/{len(image_names)}", flush=True)
    del model
    torch.cuda.empty_cache()
    result = torch.cat(batches, dim=0).contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(
        {
            "signature": signature,
            "encoder": "google/siglip2-so400m-patch16-512",
            "preprocessing": "production resize-normalize-right-bottom-zero-pad 512",
            "token_source": "vision_model_output.last_hidden_state",
            "image_names": image_names,
            "token_shape": [1024, 1152],
            "dtype": "bfloat16",
            "tokens": result,
        },
        temporary,
    )
    temporary.replace(cache_path)
    return image_names, result


def load_cached_embeddings(
    cache_path: Path,
    image_names: list[str],
) -> dict[str, np.ndarray] | None:
    if not cache_path.exists():
        return None

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("repo_id") != MODEL_REPO_ID:
        return None
    if payload.get("architecture") != MODEL_ARCH:
        return None

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, dict):
        return None
    if set(embeddings) != set(image_names):
        return None

    return {
        image_name: np.asarray(vector, dtype=np.float32)
        for image_name, vector in embeddings.items()
    }


def load_cached_embedding_subset(
    cache_path: Path,
    image_names: list[str],
) -> dict[str, np.ndarray]:
    """Load a requested subset from an existing frozen-encoder cache without encoding."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing embedding cache: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("repo_id") != MODEL_REPO_ID or payload.get("architecture") != MODEL_ARCH:
        raise ValueError("Embedding cache does not belong to the required frozen encoder.")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("Embedding cache has no embedding dictionary.")
    missing = sorted(set(image_names) - set(embeddings))
    if missing:
        raise ValueError(f"Embedding cache is missing {len(missing)} requested image(s): {missing[:5]}")
    return {name: np.asarray(embeddings[name], dtype=np.float32) for name in image_names}


def extract_embeddings(
    image_dir: Path,
    image_names: list[str],
    cache_path: Path,
    batch_size: int,
) -> dict[str, np.ndarray]:
    cached = load_cached_embeddings(cache_path, image_names)
    if cached is not None:
        return cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_frozen_encoder(device)

    embeddings: dict[str, np.ndarray] = {}
    for start in range(0, len(image_names), batch_size):
        batch_names = image_names[start : start + batch_size]
        batch_tensors = []
        for image_name in batch_names:
            image_path = image_dir / image_name
            with Image.open(image_path) as image:
                batch_tensors.append(transform(image.convert("RGB")))

        batch = torch.stack(batch_tensors, dim=0).to(device)
        with torch.inference_mode():
            features = model(batch)
            features = F.normalize(features, p=2, dim=1)

        feature_array = features.cpu().numpy().astype(np.float32)
        for image_name, feature in zip(batch_names, feature_array, strict=True):
            embeddings[image_name] = feature

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "repo_id": MODEL_REPO_ID,
            "architecture": MODEL_ARCH,
            "image_names": image_names,
            "embedding_dim": len(next(iter(embeddings.values()))),
            "embeddings": embeddings,
        },
        cache_path,
    )
    return embeddings


def encode_image_paths(image_paths: list[Path]) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_frozen_encoder(device)

    embeddings: dict[str, np.ndarray] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            feature = model(tensor)
            feature = F.normalize(feature, p=2, dim=1)
        embeddings[str(image_path)] = feature.squeeze(0).cpu().numpy().astype(np.float64)
    return embeddings


def encode_single_image(image_path: Path) -> np.ndarray:
    return encode_image_paths([image_path])[str(image_path)]


def extract_spatial_moment_features(
    image_dir: Path,
    image_names: list[str],
    cache_path: Path,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], int, int]:
    """Return normalized CLS-plus-six-moment features from frozen final-layer tokens.

    Raw patch tokens are reduced immediately, so the cache remains small while
    preserving the fixed spatial-moment construction used by the experiment.
    """
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("repo_id") == MODEL_REPO_ID
            and payload.get("architecture") == MODEL_ARCH
            and payload.get("feature_type") == "cls_plus_spatial_moments_v1"
            and set(payload.get("image_names", [])) == set(image_names)
        ):
            features = payload.get("features")
            if isinstance(features, dict):
                return (
                    {name: np.asarray(features[name], dtype=np.float32) for name in image_names},
                    int(payload["patch_grid_height"]),
                    int(payload["patch_grid_width"]),
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_frozen_encoder(device)
    grid_height, grid_width = model.patch_embed.grid_size
    patch_count = grid_height * grid_width
    v = torch.arange(grid_width, dtype=torch.float32, device=device)
    u = torch.arange(grid_height, dtype=torch.float32, device=device)
    x = 2.0 * (v + 0.5) / grid_width - 1.0
    y = 2.0 * (u + 0.5) / grid_height - 1.0
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    bases = torch.stack(
        [
            torch.ones_like(x_grid),
            x_grid,
            y_grid,
            x_grid * y_grid,
            x_grid.square() - x_grid.square().mean(),
            y_grid.square() - y_grid.square().mean(),
        ],
        dim=0,
    ).reshape(6, patch_count)
    bases = bases / torch.sqrt(torch.sum(bases.square(), dim=1, keepdim=True))

    features: dict[str, np.ndarray] = {}
    for start in range(0, len(image_names), batch_size):
        batch_names = image_names[start : start + batch_size]
        tensors = []
        for image_name in batch_names:
            with Image.open(image_dir / image_name) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.inference_mode():
            tokens = model.forward_features(batch)
            if tokens.ndim != 3 or tokens.shape[1] != patch_count + 1:
                raise ValueError(
                    "Unexpected token layout: expected one CLS token followed by "
                    f"{patch_count} patch tokens, got {tuple(tokens.shape)}."
                )
            cls = tokens[:, 0, :]
            patches = tokens[:, 1:, :]
            moments = torch.einsum("kp,bpd->bkd", bases, patches)
            combined = torch.cat([cls, moments.reshape(len(batch_names), -1)], dim=1)
            combined = F.normalize(combined, p=2, dim=1)
        for image_name, feature in zip(batch_names, combined.cpu().numpy(), strict=True):
            features[image_name] = feature.astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "repo_id": MODEL_REPO_ID,
            "architecture": MODEL_ARCH,
            "feature_type": "cls_plus_spatial_moments_v1",
            "image_names": image_names,
            "patch_grid_height": grid_height,
            "patch_grid_width": grid_width,
            "features": features,
        },
        cache_path,
    )
    return features, grid_height, grid_width


def extract_pooled_patch_tokens(
    image_dir: Path,
    image_names: list[str],
    cache_path: Path,
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int]:
    """Cache frozen final-layer CLS and 7×7 adaptive-average-pooled patch tokens."""
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("repo_id") == MODEL_REPO_ID
            and payload.get("architecture") == MODEL_ARCH
            and payload.get("feature_type") == "cls_plus_pooled_patch_tokens_v1"
            and set(payload.get("image_names", [])) == set(image_names)
        ):
            cls = payload.get("cls")
            patches = payload.get("patches")
            if isinstance(cls, dict) and isinstance(patches, dict):
                return (
                    {name: np.asarray(cls[name], dtype=np.float32) for name in image_names},
                    {name: np.asarray(patches[name], dtype=np.float32) for name in image_names},
                    int(payload["patch_grid_height"]),
                    int(payload["patch_grid_width"]),
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_frozen_encoder(device)
    source_height, source_width = model.patch_embed.grid_size
    source_patch_count = source_height * source_width
    cls_features: dict[str, np.ndarray] = {}
    patch_features: dict[str, np.ndarray] = {}
    for start in range(0, len(image_names), batch_size):
        batch_names = image_names[start : start + batch_size]
        tensors = []
        for image_name in batch_names:
            with Image.open(image_dir / image_name) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.inference_mode():
            tokens = model.forward_features(batch)
            if tokens.ndim != 3 or tokens.shape[1] != source_patch_count + 1:
                raise ValueError("Unexpected final-layer token layout.")
            cls = tokens[:, 0, :]
            patches = tokens[:, 1:, :].reshape(len(batch_names), source_height, source_width, -1)
            patches = patches.permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(patches, output_size=(7, 7))
            pooled = pooled.permute(0, 2, 3, 1).reshape(len(batch_names), 49, -1)
        for name, cls_value, patch_value in zip(
            batch_names, cls.cpu().numpy(), pooled.cpu().numpy(), strict=True
        ):
            cls_features[name] = cls_value.astype(np.float32)
            patch_features[name] = patch_value.astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "repo_id": MODEL_REPO_ID,
            "architecture": MODEL_ARCH,
            "feature_type": "cls_plus_pooled_patch_tokens_v1",
            "image_names": image_names,
            "patch_grid_height": 7,
            "patch_grid_width": 7,
            "cls": cls_features,
            "patches": patch_features,
        },
        cache_path,
    )
    return cls_features, patch_features, 7, 7
