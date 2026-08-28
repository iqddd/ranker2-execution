"""Fold-local connected human targets for leakage-safe level supervision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from ranker.comparison_graph import connected_components
from ranker.data import RankedImage
from ranker.frozen_readouts import TrainingFold
from ranker.latent_scale import comparison_arrays, fit_theta


@dataclass(frozen=True)
class LCCTarget:
    """One immutable training-only target on a fold's unique largest component."""

    heldout_identity: str
    image_names: tuple[str, ...]
    identity_by_image: dict[str, str]
    theta_by_image: dict[str, float]
    retained_groups: tuple[tuple[RankedImage, ...], ...]
    audit: dict[str, Any]

    @property
    def image_set(self) -> frozenset[str]:
        return frozenset(self.image_names)


def _fit(
    groups: list[list[RankedImage]], names: tuple[str, ...]
) -> tuple[np.ndarray, float, float]:
    image_index = {name: index for index, name in enumerate(names)}
    lower, higher, targets, weights, group_ids = comparison_arrays(groups, image_index)
    return fit_theta(lower, higher, targets, weights, group_ids, len(names))


def build_lcc_target(fold: TrainingFold) -> LCCTarget:
    """Build and independently repeat the exact STEP44B target on the fold LCC."""
    all_groups = [list(group) for group in [*fold.retained_o, *fold.retained_w]]
    if any(item.identity == fold.identity for group in all_groups for item in group):
        raise RuntimeError("Held-out identity leaked into the LCC source groups.")
    components = connected_components(all_groups)
    if not components or (
        len(components) > 1 and len(components[0]) == len(components[1])
    ):
        raise RuntimeError("Training graph does not have a unique largest component.")
    largest = set(components[0])
    groups = [
        group
        for group in all_groups
        if all(item.image_name in largest for item in group)
    ]
    if not groups or any(
        any(item.image_name not in largest for item in group) for group in groups
    ):
        raise RuntimeError("LCC group restriction is invalid.")
    names = tuple(sorted(largest, key=str.encode))
    identity_by_image: dict[str, str] = {}
    for group in groups:
        for item in group:
            previous = identity_by_image.setdefault(item.image_name, item.identity)
            if previous != item.identity:
                raise RuntimeError("One image has inconsistent identity metadata.")
    identities = sorted(set(identity_by_image.values()), key=str.encode)
    if len(identities) != 13:
        raise RuntimeError(f"LCC covers {len(identities)} identities rather than 13.")
    theta, objective, gradient_norm = _fit(groups, names)
    repeat, repeat_objective, repeat_gradient_norm = _fit(groups, names)
    repeat_max = float(np.max(np.abs(theta - repeat)))
    if repeat_max > 1e-12:
        raise RuntimeError(
            "Fold-local theta solver is not independently deterministic."
        )
    theta_by_image = {name: float(theta[index]) for index, name in enumerate(names)}
    o_groups = sum(group in fold.retained_o for group in groups)
    w_groups = len(groups) - o_groups
    pair_occurrences = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    payload = {
        "heldout_identity": fold.identity,
        "image_names": names,
        "theta": theta.tolist(),
        "groups": [[item.image_name for item in group] for group in groups],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    audit = {
        "heldout_identity": fold.identity,
        "component_count": len(components),
        "component_sizes": [len(value) for value in components],
        "largest_component_unique": True,
        "lcc_images": len(names),
        "training_identities": len(identities),
        "identities": identities,
        "O_groups": int(o_groups),
        "W_groups": int(w_groups),
        "pair_occurrences": int(pair_occurrences),
        "objective": objective,
        "gradient_norm": gradient_norm,
        "repeat_objective": repeat_objective,
        "repeat_gradient_norm": repeat_gradient_norm,
        "repeat_maximum_difference": repeat_max,
        "target_digest": digest,
        "heldout_absent": True,
    }
    return LCCTarget(
        heldout_identity=fold.identity,
        image_names=names,
        identity_by_image=identity_by_image,
        theta_by_image=theta_by_image,
        retained_groups=tuple(tuple(group) for group in groups),
        audit=audit,
    )
