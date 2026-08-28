"""Pure construction, evaluation, and decision rules for STEP60A."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ranker.data import RankedImage
from ranker.experiments.layer_screen.analysis import (
    aggregate_exposure_rows,
    exposure_loss_rows,
    purged_groups,
)
from ranker.external_validation import percentile_against
from ranker.frozen_readouts import FrozenFeatureRuntime, TrainingFold
from ranker.trainer_contract import heldout_fold_metrics
from ranker.validation_resolution import (
    direct_ordinal_exposures,
    filter_direct_ordinal_exposures,
)


def mapping_digest(names: Sequence[str], assignment: Sequence[int]) -> str:
    """Hash the stable image-to-fold mapping, independent of array storage."""
    payload = [
        {"image_id": str(name), "fold": int(fold)}
        for name, fold in sorted(zip(names, assignment, strict=True), key=lambda row: row[0].encode())
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_image_heldout_fold(
    runtime: FrozenFeatureRuntime,
    fold: int,
    names: Sequence[str],
    assignment: np.ndarray,
    o_groups: Sequence[Sequence[RankedImage]],
    w_groups: Sequence[Sequence[RankedImage]],
    canonical_count: int,
) -> TrainingFold:
    """Build a whole-image-purged fold while preserving group order and ranks."""
    heldout = {str(names[index]) for index in np.flatnonzero(assignment == fold)}
    retained_o = purged_groups(o_groups, heldout)
    retained_w = purged_groups(w_groups, heldout)
    group_indices = {
        (source, index): np.asarray(
            [runtime.name_to_index[item.image_name] for item in group], dtype=np.int64
        )
        for source, groups in (("O", retained_o), ("W", retained_w))
        for index, group in enumerate(groups)
    }
    import torch

    cuda_indices = {
        key: torch.as_tensor(value, dtype=torch.int64, device=runtime.device)
        for key, value in group_indices.items()
    }
    canonical_assignment = np.asarray(assignment[:canonical_count], dtype=np.int64)
    return TrainingFold(
        identity=f"IMAGE_FOLD_{fold}",
        retained_o=retained_o,
        retained_w=retained_w,
        group_indices=cuda_indices,
        validation_mask=canonical_assignment == fold,
        reference_mask=canonical_assignment != fold,
    )


def strict_heldout_exposures(
    panels: Mapping[str, Sequence[Sequence[RankedImage]]],
    image_fold: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Keep only target-anchor occurrences held out together in one image fold."""
    output: list[dict[str, Any]] = []
    for panel, groups in panels.items():
        for group_position, group in enumerate(groups):
            rows, _ = direct_ordinal_exposures([group])
            for source in rows:
                target = str(source["image_key"])
                fold = int(image_fold[target])
                keep = [
                    index
                    for index, anchor in enumerate(source["anchor_keys"])
                    if int(image_fold[str(anchor)]) == fold
                ]
                if not keep:
                    continue
                row = dict(source)
                for field in (
                    "anchor_keys",
                    "anchor_identities",
                    "anchor_ranks",
                    "distances",
                ):
                    row[field] = tuple(source[field][index] for index in keep)
                row.update(
                    {
                        "panel": panel,
                        "panel_group_position": group_position,
                        "fold": fold,
                    }
                )
                output.append(row)
    return output


def fold_percentile_maps(
    raw_scores: np.ndarray,
    seed_indices: Sequence[int],
    assignment: np.ndarray,
    names: Sequence[str],
    *,
    canonical_count: int = 520,
) -> list[dict[str, float]]:
    """Apply historical fold-reference percentile-first ensemble semantics."""
    values = np.asarray(raw_scores, dtype=np.float64)
    canonical_assignment = np.asarray(assignment[:canonical_count], dtype=np.int64)
    output: list[dict[str, float]] = []
    for fold in range(5):
        reference = canonical_assignment != fold
        transformed = [
            percentile_against(values[index, fold, :canonical_count], values[index, fold, :canonical_count][reference])
            for index in seed_indices
        ]
        ensemble = np.mean(np.stack(transformed), axis=0, dtype=np.float64)
        output.append(
            {
                str(name): float(score)
                for name, score in zip(names[:canonical_count], ensemble, strict=True)
            }
        )
    return output


def _direct_vector(
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    fold_maps: Sequence[Mapping[str, float]],
    identities: Sequence[str],
) -> np.ndarray:
    evaluated = exposure_loss_rows(exposures, image_fold, fold_maps)
    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_identity[str(row["identity_id"])].append(row)
    if set(by_identity) != set(identities):
        raise RuntimeError("Strict endpoint identity coverage mismatch.")
    return np.asarray(
        [aggregate_exposure_rows(by_identity[identity])[0] for identity in identities],
        dtype=np.float64,
    )


def _offset_vector(
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    fold_maps: Sequence[Mapping[str, float]],
    identities: Sequence[str],
) -> np.ndarray:
    values: list[float] = []
    for identity in identities:
        high = 0.0
        low = 0.0
        for row in exposures:
            if str(row["identity_id"]) != identity:
                continue
            panel = fold_maps[int(image_fold[str(row["image_key"])])]
            target = float(panel[str(row["image_key"])])
            for anchor, anchor_rank in zip(row["anchor_keys"], row["anchor_ranks"], strict=True):
                model_sign = np.sign(target - float(panel[str(anchor)]))
                human_sign = np.sign(int(row["human_rank"]) - int(anchor_rank))
                high += float(human_sign < 0 and model_sign > 0) - 0.5 * float(model_sign == 0)
                low += float(human_sign > 0 and model_sign < 0) - 0.5 * float(model_sign == 0)
        error = high + low
        values.append(abs((high - low) / error) if error > 0 else 0.0)
    return np.asarray(values, dtype=np.float64)


def strict_endpoint_vectors(
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    fold_maps: Sequence[Mapping[str, float]],
    identities: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return direct, G, and d>=3 identity vectors on one strict population."""
    return {
        "direct": _direct_vector(exposures, image_fold, fold_maps, identities),
        "G": _offset_vector(exposures, image_fold, fold_maps, identities),
        "gross": _direct_vector(
            filter_direct_ordinal_exposures(exposures, minimum_distance=3),
            image_fold,
            fold_maps,
            identities,
        ),
    }


def level_metric_vectors(
    fold_maps: Sequence[Mapping[str, float]],
    names: Sequence[str],
    identities_by_image: Sequence[str],
    assignment: np.ndarray,
    theta: np.ndarray,
    identity_order: Sequence[str],
    *,
    eligible_images: set[str] | None,
) -> dict[str, np.ndarray]:
    """Evaluate fold-local IMA and FrameLTE10 as equal-fold identity vectors."""
    image_names = np.asarray(names).astype(str)
    image_identities = np.asarray(identities_by_image).astype(str)
    fold_assignment = np.asarray(assignment, dtype=np.int64)
    human_maps: list[np.ndarray] = []
    for fold in range(5):
        reference = fold_assignment != fold
        human_maps.append(percentile_against(theta, theta[reference]))
    ima: list[float] = []
    frame: list[float] = []
    for identity in identity_order:
        fold_ima: list[float] = []
        fold_frame: list[float] = []
        for fold in range(5):
            selected = (fold_assignment == fold) & (image_identities == identity)
            if eligible_images is not None:
                selected &= np.asarray([name in eligible_images for name in image_names])
            if not np.any(selected):
                raise RuntimeError(f"Empty fold/identity level cell: {fold}/{identity}")
            ids = image_names[selected]
            model = np.asarray([fold_maps[fold][name] for name in ids], dtype=np.float64)
            metrics = heldout_fold_metrics(
                image_ids=ids,
                model_percentiles=model,
                human_percentiles=human_maps[fold][selected],
            )
            fold_ima.append(float(metrics["IdentityOffsetMAE"]))
            fold_frame.append(float(metrics["IB_FrameLTE10"]))
        ima.append(float(np.mean(fold_ima)))
        frame.append(float(np.mean(fold_frame)))
    return {
        "IdentityOffsetMAE": np.asarray(ima, dtype=np.float64),
        "IB_FrameLTE10": np.asarray(frame, dtype=np.float64),
    }


def classify_image_transfer(
    direct: Mapping[str, Any],
    offset: Mapping[str, Any],
    gross: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered STEP60A precedence table."""
    point = float(direct["point_delta"])
    lower = float(direct["CI90_lower"])
    upper = float(direct["CI90_upper"])
    g_harm = float(offset["point_delta"]) > 0.0 and float(offset["CI90_lower"]) > 0.0
    gross_harm = float(gross["point_delta"]) >= 0.005 and float(gross["CI90_lower"]) > 0.0
    sign = "resolved_improvement" if upper < 0.0 else "resolved_harm" if lower > 0.0 else "sign_unresolved"
    if g_harm or gross_harm:
        result = "image_transfer_guard_blocked"
    elif lower >= -0.005 and upper <= 0.005:
        result = "practical_equivalence"
    elif point <= -0.005 and upper < 0.0:
        result = "image_heldout_horizon_supported"
    elif -0.005 < point < 0.0 and upper < 0.0:
        result = "resolved_submargin_improvement"
    elif point >= 0.005 and lower > 0.0:
        result = "image_heldout_horizon_harm"
    elif 0.0 < point < 0.005 and lower > 0.0:
        result = "resolved_submargin_harm"
    else:
        result = "unresolved"
    return {
        "STEP60A_result": result,
        "direct_sign_annotation": sign,
        "resolved_G_harm": g_harm,
        "resolved_gross_harm": gross_harm,
    }
