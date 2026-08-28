"""Stable reconstruction and fold building for the Step40P pilot protocol."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ranker.data import RankedImage, load_ranked_groups
from ranker.group_ranknet import ranknet_target
from ranker.heldout_ranknet import validate_dataset_identities
from ranker.io import atomic_json, atomic_npz, sha256_file
from ranker.spatial import (
    CudaLinearObjective,
    SpatialFoldData,
    fit_cuda_linear,
    linear_pair_coefficients,
    pair_arrays,
    train_prior_pair_classes,
)

def select_attention_range_rows(
    packet_path: Path,
    *,
    pilot_indices: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and return the three fixed small/medium/large attention folds."""
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if not packet.get("pilot_packet_pass"):
        raise RuntimeError("The Step-40P-R1 pilot packet did not pass.")
    selected = [dict(row) for row in packet["selected_candidates"]]
    by_index = {int(row["pilot_index"]): row for row in selected}
    rows = [by_index[int(index)] for index in pilot_indices]
    expected = (
        (1, "AbiRat", "TifTow", "DiaMel", 64, "small"),
        (10, "Sabina", "ashleytervort", "DmRs", 72, "medium"),
        (9, "JorCar", "ashleytervort", "sharishanya", 88, "large"),
    )
    observed = tuple(
        (
            int(row["pilot_index"]),
            str(row["outer_identity_1"]),
            str(row["outer_identity_2"]),
            str(row["inner_heldout_identity"]),
            int(row["inner_train_images"]),
            str(row["size_class"]),
        )
        for row in rows
    )
    if observed != expected:
        raise RuntimeError(f"Range-fold contract mismatch: {observed!r}.")
    for row in rows:
        same_class = [
            candidate
            for candidate in selected
            if candidate["size_class"] == row["size_class"]
        ]
        if int(row["validation_counter_d_ge_2"]) != max(
            int(candidate["validation_counter_d_ge_2"]) for candidate in same_class
        ):
            raise RuntimeError(
                "A range fold is not the maximum-d>=2 candidate in its size class."
            )
    return selected, rows


@dataclass(frozen=True)
class PilotFold:
    """One selected inner-LOIO fold, retaining all original pair ordering."""

    row: Mapping[str, Any]
    spatial: SpatialFoldData
    validation_names: tuple[str, ...]
    validation_global_indices: np.ndarray
    validation_base_scores: np.ndarray
    validation_higher: np.ndarray
    validation_lower: np.ndarray
    validation_distance: np.ndarray
    validation_class: np.ndarray
    train_base_scores: np.ndarray
    train_higher: np.ndarray
    train_lower: np.ndarray
    train_targets: np.ndarray
    train_balanced_coefficients: np.ndarray
    train_aligned_coefficients_np: np.ndarray
    train_counter_coefficients_np: np.ndarray
    train_aligned_coefficients: torch.Tensor
    train_counter_coefficients: torch.Tensor


def reconstruct_groups(
    source_groups: Sequence[Sequence[RankedImage]], row: Mapping[str, Any]
) -> tuple[
    tuple[tuple[RankedImage, ...], ...],
    tuple[tuple[RankedImage, ...], ...],
    tuple[tuple[RankedImage, ...], ...],
]:
    """Rebuild R0's outer/train/validation partitions without re-sorting them."""
    outer_heldout = {str(row["outer_identity_1"]), str(row["outer_identity_2"])}
    inner_identity = str(row["inner_heldout_identity"])
    outer_train = tuple(
        tuple(group)
        for group in source_groups
        if all(item.identity not in outer_heldout for item in group)
    )
    inner_train = tuple(
        group
        for group in outer_train
        if all(item.identity != inner_identity for item in group)
    )
    inner_validation = tuple(
        group
        for group in outer_train
        if any(item.identity == inner_identity for item in group)
    )
    if len(inner_train) != int(row["inner_train_groups"]):
        raise AssertionError("Reconstructed inner-train group count differs from R0.")
    return outer_train, inner_train, inner_validation


def fit_inner_linear_models(
    *,
    selected: Sequence[Mapping[str, Any]],
    source_groups: Sequence[Sequence[RankedImage]],
    raw_features: np.ndarray,
    names: np.ndarray,
    output_dir: Path,
    device: torch.device,
    linear_l2: float,
    linear_options: Mapping[str, float | int],
    success_gradient_l2: float,
    signature: str = "step40p-r1-spatial-inner-loio-v1",
    standardize: Callable[
        [np.ndarray, np.ndarray, Sequence[Sequence[RankedImage]]],
        tuple[np.ndarray, Mapping[str, int]],
    ],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Fit and cache the 12 certified inner-balanced linear models."""
    score_rows = np.full((12, len(names)), np.nan, dtype=np.float64)
    weights = np.full((12, raw_features.shape[1]), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    artifact_path = output_dir / "STEP40P_R1_INNER_LINEAR_PREDICTIONS.npz"
    record_path = output_dir / "inner_linear_training.json"
    if artifact_path.exists() and record_path.exists():
        with np.load(artifact_path, allow_pickle=False) as cached:
            cached_scores = np.asarray(
                cached["linear_INNER_BAL_scores"], dtype=np.float64
            )
            cached_weights = np.asarray(
                cached["linear_INNER_BAL_weights"], dtype=np.float64
            )
            cached_names = np.asarray(cached["image_names"])
        cached_records = json.loads(record_path.read_text(encoding="utf-8"))["models"]
        if (
            cached_scores.shape == score_rows.shape
            and cached_weights.shape == weights.shape
            and np.array_equal(cached_names, names)
            and np.all(np.isfinite(cached_scores))
            and np.all(np.isfinite(cached_weights))
            and len(cached_records) == 12
            and all(bool(record["certificate_success"]) for record in cached_records)
        ):
            print("Restored 12 certified LINEAR_INNER_BAL models.", flush=True)
            return cached_scores, cached_weights, cached_records
    for row in selected:
        pilot_index = int(row["pilot_index"])
        started = time.perf_counter()
        _, inner_train, _ = reconstruct_groups(source_groups, row)
        masks = train_prior_pair_classes(inner_train)
        standardized, name_to_row = standardize(raw_features, names, inner_train)
        differences, targets, _ = pair_arrays(inner_train, standardized, name_to_row)
        coefficients = linear_pair_coefficients(inner_train, masks, balanced=True)
        weight, diagnostic = fit_cuda_linear(
            CudaLinearObjective(
                differences, targets, coefficients, device, l2=linear_l2
            ),
            options=linear_options,
            success_gradient_l2=success_gradient_l2,
        )
        scores = standardized @ weight
        diagnostic["finite_scores"] = bool(np.all(np.isfinite(scores)))
        diagnostic["certificate_success"] = bool(
            diagnostic["certificate_success"] and diagnostic["finite_scores"]
        )
        diagnostic.update(
            {
                "pilot_index": pilot_index,
                "outer_pair_index": int(row["outer_pair_index"]),
                "inner_heldout_identity": str(row["inner_heldout_identity"]),
                "train_groups": len(inner_train),
                "train_images": len(inner_train) * 8,
                "timing_seconds": float(time.perf_counter() - started),
            }
        )
        score_rows[pilot_index] = scores
        weights[pilot_index] = weight
        records.append(diagnostic)
        print(
            f"linear [{pilot_index + 1}/12] inner={row['inner_heldout_identity']} groups={len(inner_train)} "
            f"grad={diagnostic['gradient_l2']:.3e} pass={diagnostic['certificate_success']}",
            flush=True,
        )
    atomic_npz(
        artifact_path,
        {
            "image_names": names,
            "linear_INNER_BAL_scores": score_rows,
            "linear_INNER_BAL_weights": weights,
        },
    )
    atomic_json(
        record_path,
        {
            "signature": signature,
            "models": records,
            "n_inner_linear_models": 12,
            "n_successful_inner_linear_models": sum(
                bool(record["certificate_success"]) for record in records
            ),
            "artifact": {
                "filename": str(artifact_path),
                "sha256": sha256_file(artifact_path),
            },
        },
    )
    return score_rows, weights, records


def build_pilot_folds(
    *,
    selected: Sequence[Mapping[str, Any]],
    source_groups: Sequence[Sequence[RankedImage]],
    names: np.ndarray,
    linear_scores: np.ndarray,
    device: torch.device,
) -> list[PilotFold]:
    """Create pilot folds with the exact original train and validation pair order."""
    name_to_global = {str(name): index for index, name in enumerate(names.tolist())}
    folds: list[PilotFold] = []
    for row in selected:
        pilot_index = int(row["pilot_index"])
        inner_identity = str(row["inner_heldout_identity"])
        outer_train, inner_train, inner_validation = reconstruct_groups(
            source_groups, row
        )
        outer_aligned, outer_counter = train_prior_pair_classes(outer_train)
        outer_mask_by_group = {
            group[0].group_index: (aligned, counter)
            for group, aligned, counter in zip(
                outer_train, outer_aligned, outer_counter, strict=True
            )
        }
        validation_names = tuple(
            item.image_name for group in inner_validation for item in group
        )
        validation_global_indices = np.asarray(
            [name_to_global[name] for name in validation_names], dtype=np.int64
        )
        validation_name_to_local = {
            name: index for index, name in enumerate(validation_names)
        }
        higher: list[int] = []
        lower: list[int] = []
        distance: list[int] = []
        classes: list[int] = []
        for group in inner_validation:
            aligned, counter = outer_mask_by_group[group[0].group_index]
            pair_index = 0
            for offset, first in enumerate(group):
                for second in group[offset + 1 :]:
                    is_star = (first.identity == inner_identity) != (
                        second.identity == inner_identity
                    )
                    if is_star:
                        lo, hi = (
                            (first, second)
                            if first.rank < second.rank
                            else (second, first)
                        )
                        higher.append(validation_name_to_local[hi.image_name])
                        lower.append(validation_name_to_local[lo.image_name])
                        distance.append(hi.rank - lo.rank)
                        classes.append(
                            0
                            if bool(aligned[pair_index])
                            else 1
                            if bool(counter[pair_index])
                            else 2
                        )
                    pair_index += 1
        if sum(value == 1 for value in classes) != int(row["validation_counter_pairs"]):
            raise AssertionError(
                "Reconstructed validation counter count differs from R0."
            )
        if sum(value == 0 for value in classes) != int(row["validation_aligned_pairs"]):
            raise AssertionError(
                "Reconstructed validation aligned count differs from R0."
            )
        train_names = tuple(item.image_name for group in inner_train for item in group)
        name_to_local = {name: index for index, name in enumerate(train_names)}
        train_higher: list[int] = []
        train_lower: list[int] = []
        train_targets: list[float] = []
        for group in inner_train:
            for offset, first in enumerate(group):
                for second in group[offset + 1 :]:
                    lo, hi = (
                        (first, second) if first.rank < second.rank else (second, first)
                    )
                    train_higher.append(name_to_local[hi.image_name])
                    train_lower.append(name_to_local[lo.image_name])
                    train_targets.append(ranknet_target(hi.rank - lo.rank))
        train_masks = train_prior_pair_classes(inner_train)
        coefficients = linear_pair_coefficients(
            inner_train, train_masks, balanced=True
        ).astype(np.float32)
        aligned_flat, counter_flat = (
            np.concatenate(train_masks[0]),
            np.concatenate(train_masks[1]),
        )
        aligned_coefficients = np.where(aligned_flat, 2.0 * coefficients, 0.0).astype(
            np.float32
        )
        counter_coefficients = np.where(counter_flat, 2.0 * coefficients, 0.0).astype(
            np.float32
        )
        if not np.isclose(aligned_coefficients.sum(), 1.0) or not np.isclose(
            counter_coefficients.sum(), 1.0
        ):
            raise AssertionError("Train class coefficients must each sum to one.")
        global_indices_np = np.asarray(
            [name_to_global[name] for name in train_names], dtype=np.int64
        )
        spatial = SpatialFoldData(
            fold_index=pilot_index,
            heldout=(inner_identity, inner_identity),
            train_o=inner_train,
            train_names=train_names,
            global_indices=torch.as_tensor(
                global_indices_np, dtype=torch.long, device=device
            ),
            base_scores=torch.as_tensor(
                linear_scores[pilot_index, global_indices_np],
                dtype=torch.float32,
                device=device,
            ),
            higher=torch.as_tensor(train_higher, dtype=torch.long, device=device),
            lower=torch.as_tensor(train_lower, dtype=torch.long, device=device),
            targets=torch.as_tensor(train_targets, dtype=torch.float32, device=device),
            coefficients=torch.as_tensor(
                coefficients, dtype=torch.float32, device=device
            ),
            n_validation_counter_pairs=int(row["validation_counter_pairs"]),
        )
        folds.append(
            PilotFold(
                row=row,
                spatial=spatial,
                validation_names=validation_names,
                validation_global_indices=validation_global_indices,
                validation_base_scores=np.asarray(
                    linear_scores[pilot_index, validation_global_indices],
                    dtype=np.float64,
                ),
                validation_higher=np.asarray(higher, dtype=np.int64),
                validation_lower=np.asarray(lower, dtype=np.int64),
                validation_distance=np.asarray(distance, dtype=np.int64),
                validation_class=np.asarray(classes, dtype=np.int8),
                train_base_scores=np.asarray(
                    linear_scores[pilot_index, global_indices_np], dtype=np.float64
                ),
                train_higher=np.asarray(train_higher, dtype=np.int64),
                train_lower=np.asarray(train_lower, dtype=np.int64),
                train_targets=np.asarray(train_targets, dtype=np.float64),
                train_balanced_coefficients=np.asarray(coefficients, dtype=np.float64),
                train_aligned_coefficients_np=np.asarray(
                    aligned_coefficients, dtype=np.float64
                ),
                train_counter_coefficients_np=np.asarray(
                    counter_coefficients, dtype=np.float64
                ),
                train_aligned_coefficients=torch.as_tensor(
                    aligned_coefficients, dtype=torch.float32, device=device
                ),
                train_counter_coefficients=torch.as_tensor(
                    counter_coefficients, dtype=torch.float32, device=device
                ),
            )
        )
    return folds


def load_validated_pilot_rows(
    packet_path: Path,
    *,
    evaluation_fold_indices: tuple[int, ...],
    development_fold_indices: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load selected pilot rows while retaining the caller's fixed fold order."""
    packet_hash = sha256_file(packet_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if not bool(packet.get("pilot_packet_pass")):
        raise RuntimeError("Pinned Step-40P-R1 pilot packet did not pass.")
    selected = [dict(row) for row in packet["selected_candidates"]]
    if tuple(sorted(int(row["pilot_index"]) for row in selected)) != tuple(range(12)):
        raise RuntimeError("Pilot packet does not contain exactly pilot_index 0..11.")
    by_index = {int(row["pilot_index"]): row for row in selected}
    rows = [by_index[index] for index in evaluation_fold_indices]
    if tuple(int(row["pilot_index"]) for row in rows) != evaluation_fold_indices:
        raise AssertionError("Evaluation fold order changed.")
    return rows, {
        "packet_path": str(packet_path),
        "packet_sha256": packet_hash,
        "development_fold_indices": list(development_fold_indices),
        "evaluation_fold_indices": list(evaluation_fold_indices),
        "n_evaluation_folds": len(rows),
    }


def reconstruct_saved_pilot_folds(
    *,
    packet_path: Path,
    source_state: Path,
    saved_validation_names: Mapping[int, np.ndarray],
    evaluation_fold_indices: tuple[int, ...],
    development_fold_indices: tuple[int, ...],
    n_source_groups: int = 65,
    n_pilots: int = 12,
) -> tuple[list[PilotFold], list[list[RankedImage]]]:
    """Rebuild pilot folds and certify their saved validation-image ordering."""
    rows, _ = load_validated_pilot_rows(
        packet_path,
        evaluation_fold_indices=evaluation_fold_indices,
        development_fold_indices=development_fold_indices,
    )
    source_groups = load_ranked_groups(source_state)[:n_source_groups]
    validate_dataset_identities(source_groups)
    names = np.asarray(
        [item.image_name for group in source_groups for item in group]
    )
    if len(set(names.tolist())) != len(names):
        raise RuntimeError("Global image names are not unique.")
    folds = build_pilot_folds(
        selected=rows,
        source_groups=source_groups,
        names=names,
        linear_scores=np.zeros((n_pilots, len(names)), dtype=np.float64),
        device=torch.device("cpu"),
    )
    for fold in folds:
        pilot = int(fold.row["pilot_index"])
        expected = np.asarray(saved_validation_names[pilot])
        if not np.array_equal(expected, np.asarray(fold.validation_names)):
            raise RuntimeError(
                f"Saved validation-name order differs for pilot {pilot}."
            )
    return folds, source_groups
