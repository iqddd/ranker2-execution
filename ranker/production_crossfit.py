"""Production-compatible multi-group cross-fit fold construction and metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ranker.data import RankedImage
from ranker.group_ranknet import ranknet_target
from ranker.identity import compute_identity_priors
from ranker.pairs import enumerate_group_pairs
from ranker.pilots import PilotFold
from ranker.spatial import (
    SpatialFoldData,
    linear_pair_coefficients,
    train_prior_pair_classes,
)

def build_full_production_head_fold(
    *,
    original_groups: Sequence[Sequence[RankedImage]],
    base_scores: Mapping[str, float],
    validation_names: Sequence[str],
    name_to_global: Mapping[str, int],
    device: torch.device,
) -> PilotFold:
    """Build the no-heldout full-data training runtime for a frozen head recipe."""
    train_o = tuple(tuple(group) for group in original_groups)
    train_names = tuple(item.image_name for group in train_o for item in group)
    train_name_to_local = {name: index for index, name in enumerate(train_names)}
    train_higher: list[int] = []
    train_lower: list[int] = []
    train_targets: list[float] = []
    for group in train_o:
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (
                    (first, second) if first.rank < second.rank else (second, first)
                )
                train_higher.append(train_name_to_local[higher.image_name])
                train_lower.append(train_name_to_local[lower.image_name])
                train_targets.append(ranknet_target(higher.rank - lower.rank))
    masks = train_prior_pair_classes(train_o)
    coefficients = linear_pair_coefficients(train_o, masks, balanced=True)
    aligned_flat = np.concatenate(masks[0])
    counter_flat = np.concatenate(masks[1])
    aligned_coefficients = np.where(aligned_flat, 2.0 * coefficients, 0.0)
    counter_coefficients = np.where(counter_flat, 2.0 * coefficients, 0.0)
    if not np.isclose(aligned_coefficients.sum(), 1.0) or not np.isclose(
        counter_coefficients.sum(), 1.0
    ):
        raise RuntimeError("Full production A/C coefficient normalization failed.")
    train_values = np.asarray([base_scores[name] for name in train_names], dtype=np.float64)
    validation = tuple(validation_names)
    validation_values = np.asarray(
        [base_scores[name] for name in validation], dtype=np.float64
    )
    train_global = np.asarray([name_to_global[name] for name in train_names], dtype=np.int64)
    validation_global = np.asarray(
        [name_to_global[name] for name in validation], dtype=np.int64
    )
    spatial = SpatialFoldData(
        fold_index=0,
        heldout=(),
        train_o=train_o,
        train_names=train_names,
        global_indices=torch.as_tensor(train_global, dtype=torch.long, device=device),
        base_scores=torch.as_tensor(train_values, dtype=torch.float32, device=device),
        higher=torch.as_tensor(train_higher, dtype=torch.long, device=device),
        lower=torch.as_tensor(train_lower, dtype=torch.long, device=device),
        targets=torch.as_tensor(train_targets, dtype=torch.float32, device=device),
        coefficients=torch.as_tensor(coefficients, dtype=torch.float32, device=device),
        n_validation_counter_pairs=0,
    )
    empty_int = np.asarray([], dtype=np.int64)
    return PilotFold(
        row={"pilot_index": "full_65", "heldout_group_ids": []},
        spatial=spatial,
        validation_names=validation,
        validation_global_indices=validation_global,
        validation_base_scores=validation_values,
        validation_higher=empty_int,
        validation_lower=empty_int,
        validation_distance=empty_int,
        validation_class=np.asarray([], dtype=np.int8),
        train_base_scores=train_values,
        train_higher=np.asarray(train_higher, dtype=np.int64),
        train_lower=np.asarray(train_lower, dtype=np.int64),
        train_targets=np.asarray(train_targets, dtype=np.float64),
        train_balanced_coefficients=np.asarray(coefficients, dtype=np.float64),
        train_aligned_coefficients_np=np.asarray(aligned_coefficients, dtype=np.float64),
        train_counter_coefficients_np=np.asarray(counter_coefficients, dtype=np.float64),
        train_aligned_coefficients=torch.as_tensor(
            aligned_coefficients, dtype=torch.float32, device=device
        ),
        train_counter_coefficients=torch.as_tensor(
            counter_coefficients, dtype=torch.float32, device=device
        ),
    )


def build_production_crossfit_fold(
    *,
    fold_index: int,
    heldout_group_ids: Sequence[int],
    original_groups: Sequence[Sequence[RankedImage]],
    base_scores: Mapping[str, float],
    name_to_global: Mapping[str, int],
    device: torch.device,
) -> PilotFold:
    heldout_ids = set(map(int, heldout_group_ids))
    train_o = tuple(
        tuple(group) for index, group in enumerate(original_groups) if index not in heldout_ids
    )
    heldout_o = tuple(tuple(original_groups[index]) for index in heldout_group_ids)
    train_names = tuple(item.image_name for group in train_o for item in group)
    train_name_to_local = {name: index for index, name in enumerate(train_names)}
    train_higher: list[int] = []
    train_lower: list[int] = []
    train_targets: list[float] = []
    for group in train_o:
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (
                    (first, second) if first.rank < second.rank else (second, first)
                )
                train_higher.append(train_name_to_local[higher.image_name])
                train_lower.append(train_name_to_local[lower.image_name])
                train_targets.append(ranknet_target(higher.rank - lower.rank))
    masks = train_prior_pair_classes(train_o)
    coefficients = linear_pair_coefficients(train_o, masks, balanced=True)
    aligned_flat = np.concatenate(masks[0])
    counter_flat = np.concatenate(masks[1])
    aligned_coefficients = np.where(aligned_flat, 2.0 * coefficients, 0.0)
    counter_coefficients = np.where(counter_flat, 2.0 * coefficients, 0.0)
    if not np.isclose(aligned_coefficients.sum(), 1.0) or not np.isclose(
        counter_coefficients.sum(), 1.0
    ):
        raise RuntimeError("Production A/C coefficient normalization failed.")
    train_global = np.asarray([name_to_global[name] for name in train_names], dtype=np.int64)
    train_values = np.asarray([base_scores[name] for name in train_names], dtype=np.float64)
    validation_names = tuple(item.image_name for group in heldout_o for item in group)
    validation_values = np.asarray(
        [base_scores[name] for name in validation_names], dtype=np.float64
    )
    validation_global = np.asarray(
        [name_to_global[name] for name in validation_names], dtype=np.int64
    )
    priors, _, _ = compute_identity_priors(list(map(list, train_o)), 8.0)
    validation_higher: list[int] = []
    validation_lower: list[int] = []
    validation_distance: list[int] = []
    validation_class: list[int] = []
    for group_offset, group in enumerate(heldout_o):
        local = {item.image_name: group_offset * 8 + item.rank for item in group}
        for first_rank in range(8):
            for second_rank in range(first_rank + 1, 8):
                lower = group[first_rank]
                higher = group[second_rank]
                validation_lower.append(local[lower.image_name])
                validation_higher.append(local[higher.image_name])
                validation_distance.append(second_rank - first_rank)
                if lower.identity == higher.identity:
                    validation_class.append(2)
                else:
                    delta = priors[higher.identity] - priors[lower.identity]
                    validation_class.append(0 if delta > 0.0 else 1 if delta < 0.0 else 3)
    spatial = SpatialFoldData(
        fold_index=fold_index,
        heldout=tuple(map(str, heldout_group_ids)),
        train_o=train_o,
        train_names=train_names,
        global_indices=torch.as_tensor(train_global, dtype=torch.long, device=device),
        base_scores=torch.as_tensor(train_values, dtype=torch.float32, device=device),
        higher=torch.as_tensor(train_higher, dtype=torch.long, device=device),
        lower=torch.as_tensor(train_lower, dtype=torch.long, device=device),
        targets=torch.as_tensor(train_targets, dtype=torch.float32, device=device),
        coefficients=torch.as_tensor(coefficients, dtype=torch.float32, device=device),
        n_validation_counter_pairs=int(np.sum(np.asarray(validation_class) == 1)),
    )
    return PilotFold(
        row={"pilot_index": fold_index, "heldout_group_ids": list(heldout_group_ids)},
        spatial=spatial,
        validation_names=validation_names,
        validation_global_indices=validation_global,
        validation_base_scores=validation_values,
        validation_higher=np.asarray(validation_higher, dtype=np.int64),
        validation_lower=np.asarray(validation_lower, dtype=np.int64),
        validation_distance=np.asarray(validation_distance, dtype=np.int64),
        validation_class=np.asarray(validation_class, dtype=np.int8),
        train_base_scores=train_values,
        train_higher=np.asarray(train_higher, dtype=np.int64),
        train_lower=np.asarray(train_lower, dtype=np.int64),
        train_targets=np.asarray(train_targets, dtype=np.float64),
        train_balanced_coefficients=np.asarray(coefficients, dtype=np.float64),
        train_aligned_coefficients_np=np.asarray(aligned_coefficients, dtype=np.float64),
        train_counter_coefficients_np=np.asarray(counter_coefficients, dtype=np.float64),
        train_aligned_coefficients=torch.as_tensor(
            aligned_coefficients, dtype=torch.float32, device=device
        ),
        train_counter_coefficients=torch.as_tensor(
            counter_coefficients, dtype=torch.float32, device=device
        ),
    )


def pair_masks(fold: PilotFold) -> dict[str, np.ndarray]:
    distance = fold.validation_distance
    same = fold.validation_class == 2
    aligned = fold.validation_class == 0
    counter = fold.validation_class == 1
    all_pairs = np.ones(len(distance), dtype=bool)
    lower_rank = np.tile(
        np.asarray([first for first in range(8) for _ in range(first + 1, 8)]),
        len(fold.validation_names) // 8,
    )
    higher_rank = np.tile(
        np.asarray([second for first in range(8) for second in range(first + 1, 8)]),
        len(fold.validation_names) // 8,
    )
    return {
        "all_pairs": all_pairs,
        "same_identity": same,
        "aligned": aligned,
        "counter": counter,
        "all_d_ge_3": distance >= 3,
        "aligned_d_ge_3": aligned & (distance >= 3),
        "counter_d_ge_3": counter & (distance >= 3),
        "aligned_d_1": aligned & (distance == 1),
        "aligned_d_2": aligned & (distance == 2),
        "counter_d_1": counter & (distance == 1),
        "counter_d_2": counter & (distance == 2),
        "outer_quartile": (lower_rank <= 1) & (higher_rank >= 6),
        "rank_0_vs_7": (lower_rank == 0) & (higher_rank == 7),
        "cross_all": aligned | counter,
    }


def score_statistics(
    fold: PilotFold, scores: np.ndarray
) -> dict[str, tuple[int, int]]:
    differences = scores[fold.validation_higher] - scores[fold.validation_lower]
    correct = differences > 0.0
    masks = pair_masks(fold)
    distance = fold.validation_distance.astype(np.int64)
    result: dict[str, tuple[int, int]] = {}
    for name, mask in masks.items():
        result[name] = (int(np.sum(correct[mask])), int(np.sum(mask)))
    for name in ("all_pairs", "same_identity", "aligned", "counter", "cross_all"):
        mask = masks[name]
        result[f"{name}_WPA"] = (
            int(np.sum(distance[mask] * correct[mask])),
            int(np.sum(distance[mask])),
        )
    top_correct = 0
    for start in range(0, len(scores), 8):
        top_correct += int(np.argmax(scores[start : start + 8]) == 7)
    result["top1_hard"] = (top_correct, len(scores) // 8)
    return result


def crossfit_group_pair_records(
    *,
    groups: Sequence[Sequence[RankedImage]],
    image_folds: Mapping[str, int],
    reference_scores: Mapping[str, float],
    candidate_scores: Mapping[str, float],
    prior_shrinkage: float = 8.0,
) -> list[dict[str, Any]]:
    """Reconstruct fold-local pair records from completed group-OOF scores.

    Each held-out group's identity prior is fitted only on groups assigned to
    other cross-fit folds.  The helper deliberately consumes mappings rather
    than experiment artifacts so it can be reused by descriptive audits.
    """
    group_folds: list[int] = []
    for group in groups:
        observed = {int(image_folds[item.image_name]) for item in group}
        if len(observed) != 1:
            raise RuntimeError("An O-group spans multiple cross-fit folds.")
        group_folds.append(next(iter(observed)))
    priors_by_fold: dict[int, dict[str, float]] = {}
    for fold_index in sorted(set(group_folds)):
        train_groups = [
            list(group)
            for group, assigned in zip(groups, group_folds, strict=True)
            if assigned != fold_index
        ]
        priors_by_fold[fold_index] = compute_identity_priors(
            train_groups, prior_shrinkage
        )[0]

    rows: list[dict[str, Any]] = []
    for group, fold_index in zip(groups, group_folds, strict=True):
        ordered = sorted(group, key=lambda item: item.rank)
        if [item.rank for item in ordered] != list(range(len(ordered))):
            raise RuntimeError("An O-group does not have contiguous true ranks.")
        priors = priors_by_fold[fold_index]
        for lower_rank, lower in enumerate(ordered):
            for higher_rank in range(lower_rank + 1, len(ordered)):
                higher = ordered[higher_rank]
                if lower.identity == higher.identity:
                    relation = "same_identity"
                else:
                    delta = priors[higher.identity] - priors[lower.identity]
                    relation = (
                        "aligned" if delta > 0.0 else "counter" if delta < 0.0 else "tied_cross"
                    )
                reference_margin = (
                    reference_scores[higher.image_name]
                    - reference_scores[lower.image_name]
                )
                candidate_margin = (
                    candidate_scores[higher.image_name]
                    - candidate_scores[lower.image_name]
                )
                rows.append(
                    {
                        "group_id": int(lower.group_index),
                        "fold_index": fold_index,
                        "lower_rank": lower_rank,
                        "higher_rank": higher_rank,
                        "distance": higher_rank - lower_rank,
                        "prior_relation": relation,
                        "reference_correct": bool(reference_margin > 0.0),
                        "candidate_correct": bool(candidate_margin > 0.0),
                    }
                )
    return rows


def fixed_prior_group_pair_records(
    *,
    groups: Sequence[Sequence[RankedImage]],
    identity_priors: Mapping[str, float],
    reference_scores: Mapping[str, float],
    candidate_scores: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Build paired correctness records using one externally fixed identity prior.

    Missing identities receive the formula-consistent zero prior (``n_a = 0``).
    This is useful for prospective audits where the prior must be fitted once on
    historical groups and then held fixed for every new group.
    """
    rows: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda item: item.rank)
        if [item.rank for item in ordered] != list(range(len(ordered))):
            raise RuntimeError("An O-group does not have contiguous true ranks.")
        group_size = len(ordered)
        for lower_rank, lower in enumerate(ordered):
            for higher_rank in range(lower_rank + 1, group_size):
                higher = ordered[higher_rank]
                if lower.identity == higher.identity:
                    relation = "same_identity"
                else:
                    delta = float(identity_priors.get(higher.identity, 0.0)) - float(
                        identity_priors.get(lower.identity, 0.0)
                    )
                    relation = (
                        "aligned" if delta > 0.0 else "counter" if delta < 0.0 else "tied_cross"
                    )
                reference_margin = float(reference_scores[higher.image_name]) - float(
                    reference_scores[lower.image_name]
                )
                candidate_margin = float(candidate_scores[higher.image_name]) - float(
                    candidate_scores[lower.image_name]
                )
                rows.append(
                    {
                        "group_id": int(lower.group_index),
                        "group_size": group_size,
                        "lower_rank": lower_rank,
                        "higher_rank": higher_rank,
                        "distance": higher_rank - lower_rank,
                        "prior_relation": relation,
                        "reference_correct": bool(reference_margin > 0.0),
                        "candidate_correct": bool(candidate_margin > 0.0),
                    }
                )
    return rows


def fixed_prior_system_metrics(
    *,
    groups: Sequence[Sequence[RankedImage]],
    identity_priors: Mapping[str, float],
    score_maps: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Evaluate several frozen systems under one externally fixed identity prior.

    The metric contract matches prospective ranking audits: strict score ordering,
    distance-weighted pair accuracy, variable group sizes, outer-quartile and
    rank-extreme strata, and deterministic image-name tie breaking for top-1.
    """
    totals: dict[str, dict[str, list[float]]] = {
        system: {
            name: [0.0, 0.0]
            for name in (
                "all_pairs",
                "same_identity",
                "aligned",
                "counter",
                "all_d_ge_3",
                "aligned_d_ge_3",
                "counter_d_ge_3",
                "outer_quartile",
                "rank_extreme",
            )
        }
        for system in score_maps
    }
    weighted: dict[str, dict[str, list[float]]] = {
        system: {
            name: [0.0, 0.0]
            for name in ("all_pairs", "same_identity", "aligned", "counter", "all_d_ge_3")
        }
        for system in score_maps
    }
    top_hits = {system: 0 for system in score_maps}

    names = [item.image_name for group in groups for item in group]
    try:
        pairs = enumerate_group_pairs(
            groups, index_of={name: index for index, name in enumerate(names)}
        )
    except ValueError as error:
        raise RuntimeError("A ranked group does not have contiguous true ranks.") from error
    same_identity = pairs.mask("same_identity")
    prior_delta = np.asarray(
        [
            float(identity_priors.get(higher, 0.0))
            - float(identity_priors.get(lower, 0.0))
            for lower, higher in zip(
                pairs.lower_identity, pairs.higher_identity, strict=True
            )
        ],
        dtype=np.float64,
    )
    masks = {
        "all_pairs": pairs.mask("all"),
        "same_identity": same_identity,
        "aligned": ~same_identity & (prior_delta > 0.0),
        "counter": ~same_identity & (prior_delta < 0.0),
        "all_d_ge_3": pairs.mask("distance_ge_3"),
        "outer_quartile": pairs.mask("outer_quartile"),
        "rank_extreme": pairs.mask("rank_extreme"),
    }
    masks["aligned_d_ge_3"] = masks["aligned"] & masks["all_d_ge_3"]
    masks["counter_d_ge_3"] = masks["counter"] & masks["all_d_ge_3"]
    for system, scores in score_maps.items():
        score_values = np.asarray([float(scores[name]) for name in names], dtype=np.float64)
        correct = score_values[pairs.higher] > score_values[pairs.lower]
        for stratum, mask in masks.items():
            totals[system][stratum][0] = float(np.count_nonzero(correct[mask]))
            totals[system][stratum][1] = float(np.count_nonzero(mask))
        for stratum in weighted[system]:
            mask = masks[stratum]
            weighted[system][stratum][0] = float(
                np.sum(pairs.distance[mask] * correct[mask])
            )
            weighted[system][stratum][1] = float(np.sum(pairs.distance[mask]))

    for group in groups:
        ordered = sorted(group, key=lambda item: item.rank)
        true_top = ordered[-1].image_name
        for system, scores in score_maps.items():
            predicted_top = min(
                ordered,
                key=lambda item: (-float(scores[item.image_name]), item.image_name),
            ).image_name
            top_hits[system] += int(predicted_top == true_top)

    result: dict[str, dict[str, float]] = {}
    for system in score_maps:
        metrics = {
            f"{name}_pair_accuracy": numerator / denominator
            for name, (numerator, denominator) in totals[system].items()
            if denominator
        }
        metrics.update(
            {
                f"{name}_WPA": numerator / denominator
                for name, (numerator, denominator) in weighted[system].items()
                if denominator
            }
        )
        metrics["top1_hard"] = top_hits[system] / len(groups)
        result[system] = metrics
    return result
