"""Frozen Step-39 A_S2 artifact loading and score inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import extract_identity


FEATURE_DIMENSION = 1152
ARTIFACT_FIELDS = frozenset(
    {
        "image_names",
        "identity_names",
        "feature_mean",
        "feature_std",
        "identity_centroids",
        "visual_w",
        "visual_score_mean",
        "visual_score_std",
        "identity_prior_raw",
        "identity_prior_mean",
        "identity_prior_std",
        "alpha_prod",
        "identity_component",
        "visual_component",
        "A_S2_scores",
    }
)


@dataclass(frozen=True)
class FrozenAS2:
    identity_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    identity_centroids: np.ndarray
    visual_weights: np.ndarray
    visual_score_mean: float
    visual_score_std: float
    identity_prior_raw: np.ndarray
    identity_prior_mean: float
    identity_prior_std: float
    alpha_prod: float

    @property
    def identity_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.identity_names)}


def _scalar(payload: Mapping[str, np.ndarray], name: str) -> float:
    value = np.asarray(payload[name], dtype=np.float64)
    if value.shape != () or not np.isfinite(value):
        raise RuntimeError(f"A_S2 artifact has an invalid scalar field: {name}.")
    return float(value)


def load_frozen_a_s2(path: Path) -> FrozenAS2:
    """Load the exact, frozen Step-39 A_S2 production artifact."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != ARTIFACT_FIELDS:
                raise RuntimeError("A_S2 artifact has incomplete or unexpected fields.")
            payload = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unable to read A_S2 artifact: {error}") from error

    image_names = np.asarray(payload["image_names"])
    identity_names = tuple(str(value) for value in np.asarray(payload["identity_names"]).tolist())
    if (
        image_names.shape != (520,)
        or len({str(value) for value in image_names.tolist()}) != 520
        or len(identity_names) != 14
        or len(set(identity_names)) != 14
    ):
        raise RuntimeError("A_S2 artifact has an invalid image or identity manifest.")

    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float64)
    identity_centroids = np.asarray(payload["identity_centroids"], dtype=np.float64)
    visual_weights = np.asarray(payload["visual_w"], dtype=np.float64)
    identity_prior_raw = np.asarray(payload["identity_prior_raw"], dtype=np.float64)
    training_vectors = (
        np.asarray(payload["identity_component"], dtype=np.float64),
        np.asarray(payload["visual_component"], dtype=np.float64),
        np.asarray(payload["A_S2_scores"], dtype=np.float64),
    )
    if (
        feature_mean.shape != (FEATURE_DIMENSION,)
        or feature_std.shape != (FEATURE_DIMENSION,)
        or identity_centroids.shape != (len(identity_names), FEATURE_DIMENSION)
        or visual_weights.shape != (FEATURE_DIMENSION,)
        or identity_prior_raw.shape != (len(identity_names),)
        or any(vector.shape != (520,) for vector in training_vectors)
        or not all(
            np.all(np.isfinite(value))
            for value in (
                feature_mean,
                feature_std,
                identity_centroids,
                visual_weights,
                identity_prior_raw,
                *training_vectors,
            )
        )
        or np.any(feature_std <= 0.0)
    ):
        raise RuntimeError("A_S2 artifact has an invalid numerical contract.")

    visual_score_mean = _scalar(payload, "visual_score_mean")
    visual_score_std = _scalar(payload, "visual_score_std")
    identity_prior_mean = _scalar(payload, "identity_prior_mean")
    identity_prior_std = _scalar(payload, "identity_prior_std")
    alpha_prod = _scalar(payload, "alpha_prod")
    if visual_score_std <= 0.0 or identity_prior_std <= 0.0:
        raise RuntimeError("A_S2 artifact has a non-positive score standard deviation.")

    return FrozenAS2(
        identity_names=identity_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        identity_centroids=identity_centroids,
        visual_weights=visual_weights,
        visual_score_mean=visual_score_mean,
        visual_score_std=visual_score_std,
        identity_prior_raw=identity_prior_raw,
        identity_prior_mean=identity_prior_mean,
        identity_prior_std=identity_prior_std,
        alpha_prod=alpha_prod,
    )


def score_a_s2_features(
    model: FrozenAS2,
    image_names: Sequence[str],
    features: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Score known-identity images from Step-26-compatible pooled features."""
    if len(set(image_names)) != len(image_names):
        raise ValueError("A_S2 score input contains duplicate image names.")
    identities = [extract_identity(name) for name in image_names]
    identity_to_index = model.identity_to_index
    unknown = sorted(set(identities) - set(identity_to_index))
    if unknown:
        raise ValueError(f"A_S2 does not support unknown identity: {', '.join(unknown)}.")

    try:
        raw = np.stack([np.asarray(features[name], dtype=np.float64) for name in image_names])
    except KeyError as error:
        raise ValueError(f"A_S2 feature is missing for image: {error.args[0]}.") from error
    if raw.shape != (len(image_names), FEATURE_DIMENSION) or not np.all(np.isfinite(raw)):
        raise ValueError("A_S2 received invalid pooled image features.")

    identity_indices = np.asarray([identity_to_index[identity] for identity in identities], dtype=np.int64)
    standardized_features = (raw - model.feature_mean) / model.feature_std
    residual = standardized_features - model.identity_centroids[identity_indices]
    visual_raw = residual @ model.visual_weights
    visual = (visual_raw - model.visual_score_mean) / model.visual_score_std
    priors = model.identity_prior_raw[identity_indices]
    identity = (priors - model.identity_prior_mean) / model.identity_prior_std
    scores = identity + model.alpha_prod * visual
    if not np.all(np.isfinite(scores)):
        raise ValueError("A_S2 produced non-finite scores.")
    return {name: float(score) for name, score in zip(image_names, scores, strict=True)}
