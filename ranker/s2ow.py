"""Frozen S2_OW artifact loading and inference primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ranker.embeddings import resize_normalize_zero_pad_image
from ranker.io import atomic_torch


@dataclass(frozen=True)
class FrozenS2OW:
    mu: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray
    percentile: np.ndarray
    score_quantile: np.ndarray
    lambda_prod: float


def load_frozen_s2ow(path: Path) -> FrozenS2OW:
    """Load and strictly validate the frozen Step29 production artifact."""
    with np.load(path, allow_pickle=False) as payload:
        required = {"mu", "sigma", "w", "lambda_prod", "percentile", "score_quantile"}
        if set(payload.files) != required:
            raise RuntimeError("Incomplete or unexpected frozen S2_OW artifact fields.")
        model = FrozenS2OW(
            mu=np.asarray(payload["mu"], dtype=np.float64),
            sigma=np.asarray(payload["sigma"], dtype=np.float64),
            weights=np.asarray(payload["w"], dtype=np.float64),
            percentile=np.asarray(payload["percentile"], dtype=np.float64),
            score_quantile=np.asarray(payload["score_quantile"], dtype=np.float64),
            lambda_prod=float(payload["lambda_prod"]),
        )
    arrays = (model.mu, model.sigma, model.weights, model.percentile, model.score_quantile)
    if (
        model.mu.shape != (1152,)
        or model.sigma.shape != (1152,)
        or model.weights.shape != (1152,)
        or model.percentile.shape != (101,)
        or model.score_quantile.shape != (101,)
        or not all(np.all(np.isfinite(value)) for value in arrays)
        or np.any(model.sigma <= 0.0)
        or model.lambda_prod != 10.0
    ):
        raise RuntimeError("Frozen S2_OW numerical contract failed.")
    return model


def load_historical_s2_features(
    path: Path, *, expected_names: Sequence[str]
) -> tuple[list[str], dict[str, np.ndarray], dict[str, object]]:
    """Load the name-keyed Step26 cache that was consumed directly by Step29."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cached_names = payload.get("image_names")
    raw = payload.get("features")
    if (
        payload.get("repo") != "google/siglip2-so400m-patch16-512"
        or payload.get("patch_count") != 0
        or not isinstance(cached_names, list)
        or len(cached_names) != 520
        or len(set(cached_names)) != 520
        or not isinstance(raw, dict)
        or set(raw) != set(cached_names)
        or set(cached_names) != set(expected_names)
    ):
        raise RuntimeError("Historical S2 feature-cache provenance contract failed.")
    features = {
        name: np.asarray(raw[name], dtype=np.float64) for name in cached_names
    }
    matrix = np.stack([features[name] for name in cached_names])
    if (
        matrix.shape != (520, 1152)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise RuntimeError("Historical S2 feature-cache tensor contract failed.")
    metadata = {
        "repo": payload["repo"],
        "patch_count": int(payload["patch_count"]),
        "feature_dtype": str(matrix.dtype),
        "feature_shape": list(matrix.shape),
        "mapping": "features is a dictionary keyed by image_id; image_names is its saved order",
    }
    return list(cached_names), features, metadata


def load_step26_s2_feature_cache(
    path: Path, *, expected_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Load an exact Step26-compatible name-keyed pooled-feature cache."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cached_names = payload.get("image_names")
    raw = payload.get("features")
    expected = list(expected_names)
    if (
        not isinstance(cached_names, list)
        or len(cached_names) != len(expected)
        or set(map(str, cached_names)) != set(expected)
        or not isinstance(raw, dict)
        or set(map(str, raw)) != set(expected)
    ):
        raise RuntimeError("Step26-compatible feature-cache population failed.")
    result = {
        name: np.asarray(raw[name], dtype=np.float64)
        for name in expected
    }
    matrix = np.stack([result[name] for name in expected])
    if (
        matrix.shape != (len(expected), 1152)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise RuntimeError("Step26-compatible feature-cache tensor failed.")
    return result


def infer_s2ow_scores(
    model: FrozenS2OW,
    features: Mapping[str, np.ndarray],
    image_ids: Sequence[str],
) -> np.ndarray:
    """Apply the exact Step29 float64 standardize-then-dot inference order."""
    values: list[float] = []
    for image_id in image_ids:
        raw = np.asarray(features[image_id], dtype=np.float64)
        standardized = (raw - model.mu) / model.sigma
        values.append(float(np.asarray(standardized, dtype=np.float64) @ model.weights))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(image_ids),) or not np.all(np.isfinite(result)):
        raise RuntimeError("Frozen S2_OW inference produced invalid scores.")
    return result


def extract_step26_s2_features(
    *,
    model: torch.nn.Module,
    processor: object,
    image_dir: Path,
    image_ids: Sequence[str],
    cache_path: Path,
    device: torch.device,
    batch_size: int = 2,
) -> dict[str, np.ndarray]:
    """Extract pooled features with the historical Step26 float64 L2 normalization."""
    signature = "step43c0-r1-step26-s2-pooler-float64-l2-batch2-v1"
    ordered = list(image_ids)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("signature") == signature
            and payload.get("image_names") == ordered
            and isinstance(payload.get("features"), dict)
        ):
            restored = {
                name: np.asarray(payload["features"][name], dtype=np.float64)
                for name in ordered
            }
            matrix = np.stack([restored[name] for name in ordered])
            if matrix.shape == (len(ordered), 1152) and np.all(np.isfinite(matrix)):
                return restored
    mean = list(processor.image_mean)
    std = list(processor.image_std)
    result: dict[str, np.ndarray] = {}
    model.eval()
    for start in range(0, len(ordered), batch_size):
        names = ordered[start : start + batch_size]
        pixels = torch.stack(
            [resize_normalize_zero_pad_image(image_dir / name, mean, std) for name in names]
        ).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            values = model(pixel_values=pixels, return_dict=True).pooler_output
        for name, row in zip(names, values.float().cpu().numpy(), strict=True):
            vector = np.asarray(row, dtype=np.float64)
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm <= 0.0:
                raise RuntimeError("Invalid prospective pooled feature norm.")
            result[name] = vector / norm
    matrix = np.stack([result[name] for name in ordered])
    if (
        matrix.shape != (len(ordered), 1152)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("Invalid prospective Step26-compatible pooled features.")
    atomic_torch(
        cache_path,
        {
            "signature": signature,
            "image_names": ordered,
            "features": result,
            "source": "vision_model_output.pooler_output",
            "model_forward_precision": "FP32",
            "normalization": "NumPy float64 rowwise L2, matching Step26 l2_normalize",
            "batch_size": batch_size,
        },
    )
    return result
