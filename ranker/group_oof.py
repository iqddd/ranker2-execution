"""Leakage-safe group-OOF splits and attention runtime construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ranker.data import RankedImage
from ranker.group_ranknet import ranknet_target
from ranker.pilots import PilotFold
from ranker.spatial import (
    SpatialFoldData,
    linear_pair_coefficients,
    train_prior_pair_classes,
)


@dataclass(frozen=True)
class GroupOOFSplit:
    fold_index: int
    group_id: int
    heldout_o: tuple[RankedImage, ...]
    train_o: tuple[tuple[RankedImage, ...], ...]
    train_w: tuple[tuple[RankedImage, ...], ...]
    purged_w: tuple[tuple[RankedImage, ...], ...]
    train_aligned: tuple[np.ndarray, ...]
    train_counter: tuple[np.ndarray, ...]
    heldout_aligned: np.ndarray
    heldout_counter: np.ndarray

    @property
    def heldout_names(self) -> tuple[str, ...]:
        return tuple(item.image_name for item in self.heldout_o)

    @property
    def train_names(self) -> tuple[str, ...]:
        return tuple(item.image_name for group in self.train_o for item in group)

    def manifest_row(self) -> dict[str, Any]:
        train_unique = {
            item.image_name for group in (*self.train_o, *self.train_w) for item in group
        }
        heldout = set(self.heldout_names)
        counter_d2 = 0
        for group, mask in zip(self.train_o, self.train_counter, strict=True):
            distances = np.asarray(
                [
                    abs(first.rank - second.rank)
                    for offset, first in enumerate(group)
                    for second in group[offset + 1 :]
                ],
                dtype=np.int64,
            )
            counter_d2 += int(np.sum(mask & (distances >= 2)))
        train_images = {
            item.image_name for group in (*self.train_o, *self.train_w) for item in group
        }
        return {
            "fold_index": self.fold_index,
            "group_id": self.group_id,
            "heldout_images": "|".join(self.heldout_names),
            "n_train_O_groups": len(self.train_o),
            "n_train_W_groups": len(self.train_w),
            "n_train_unique_images": len(train_unique),
            "n_purged_W_groups": len(self.purged_w),
            "n_purged_W_images": sum(len(group) for group in self.purged_w),
            "n_aligned_train_pairs": sum(int(mask.sum()) for mask in self.train_aligned),
            "n_counter_train_pairs": sum(int(mask.sum()) for mask in self.train_counter),
            "n_counter_d_ge_2_train_pairs": counter_d2,
            "heldout_overlap_train_O": sum(
                item.image_name in heldout for group in self.train_o for item in group
            ),
            "heldout_overlap_train_W": sum(
                item.image_name in heldout for group in self.train_w for item in group
            ),
            "train_image_union_matches": train_images == train_unique,
        }


def build_group_oof_splits(
    original_groups: Sequence[Sequence[RankedImage]],
    within_groups: Sequence[Sequence[RankedImage]],
) -> list[GroupOOFSplit]:
    """Hold out one complete O-group and purge every overlapping W-group."""
    ordered_o = sorted(
        (tuple(group) for group in original_groups),
        key=lambda group: int(group[0].group_index),
    )
    ordered_w = tuple(
        sorted(
            (tuple(group) for group in within_groups),
            key=lambda group: int(group[0].group_index),
        )
    )
    group_ids = [int(group[0].group_index) for group in ordered_o]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Original O group IDs must be unique.")
    full_aligned, full_counter = train_prior_pair_classes(ordered_o)
    splits: list[GroupOOFSplit] = []
    for fold_index, heldout_o in enumerate(ordered_o):
        heldout_names = {item.image_name for item in heldout_o}
        train_o = tuple(
            group for index, group in enumerate(ordered_o) if index != fold_index
        )
        train_w = tuple(
            group
            for group in ordered_w
            if not any(item.image_name in heldout_names for item in group)
        )
        purged_w = tuple(
            group
            for group in ordered_w
            if any(item.image_name in heldout_names for item in group)
        )
        train_aligned, train_counter = train_prior_pair_classes(train_o)
        splits.append(
            GroupOOFSplit(
                fold_index=fold_index,
                group_id=int(heldout_o[0].group_index),
                heldout_o=heldout_o,
                train_o=train_o,
                train_w=train_w,
                purged_w=purged_w,
                train_aligned=train_aligned,
                train_counter=train_counter,
                heldout_aligned=np.asarray(full_aligned[fold_index], dtype=bool),
                heldout_counter=np.asarray(full_counter[fold_index], dtype=bool),
            )
        )
    return splits


def select_group_oof_profile_splits(
    splits: Sequence[GroupOOFSplit],
) -> dict[str, GroupOOFSplit]:
    """Select min/median/max retained-W profiles with the fixed fold tie-break."""
    if not splits:
        raise ValueError("At least one group-OOF split is required.")
    values = sorted(len(split.train_w) for split in splits)
    median_value = values[len(values) // 2]
    return {
        "profile_low": min(splits, key=lambda split: (len(split.train_w), split.fold_index)),
        "profile_mid": min(
            (split for split in splits if len(split.train_w) == median_value),
            key=lambda split: split.fold_index,
        ),
        "profile_high": min(
            splits, key=lambda split: (-len(split.train_w), split.fold_index)
        ),
    }


def build_group_oof_attention_fold(
    *,
    split: GroupOOFSplit,
    base_scores: Mapping[str, float],
    name_to_global: Mapping[str, int],
    device: torch.device,
) -> PilotFold:
    """Adapt one group-OOF split to the shared frozen-head/LoRA runtime contract."""
    train_names = split.train_names
    train_name_to_local = {name: index for index, name in enumerate(train_names)}
    train_higher: list[int] = []
    train_lower: list[int] = []
    train_targets: list[float] = []
    for group in split.train_o:
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (
                    (first, second) if first.rank < second.rank else (second, first)
                )
                train_higher.append(train_name_to_local[higher.image_name])
                train_lower.append(train_name_to_local[lower.image_name])
                train_targets.append(ranknet_target(higher.rank - lower.rank))
    masks = (split.train_aligned, split.train_counter)
    coefficients = linear_pair_coefficients(split.train_o, masks, balanced=True)
    aligned_flat = np.concatenate(split.train_aligned)
    counter_flat = np.concatenate(split.train_counter)
    aligned_coefficients = np.where(aligned_flat, 2.0 * coefficients, 0.0)
    counter_coefficients = np.where(counter_flat, 2.0 * coefficients, 0.0)
    if not np.isclose(aligned_coefficients.sum(), 1.0) or not np.isclose(
        counter_coefficients.sum(), 1.0
    ):
        raise AssertionError("Group-OOF A/C coefficients must each sum to one.")
    train_global = np.asarray(
        [name_to_global[name] for name in train_names], dtype=np.int64
    )
    validation_names = split.heldout_names
    validation_global = np.asarray(
        [name_to_global[name] for name in validation_names], dtype=np.int64
    )
    validation_higher: list[int] = []
    validation_lower: list[int] = []
    validation_distance: list[int] = []
    for lower in range(8):
        for higher in range(lower + 1, 8):
            validation_lower.append(lower)
            validation_higher.append(higher)
            validation_distance.append(higher - lower)
    validation_class = np.full(28, 2, dtype=np.int8)
    validation_class[split.heldout_aligned] = 0
    validation_class[split.heldout_counter] = 1
    train_values = np.asarray([base_scores[name] for name in train_names], dtype=np.float64)
    validation_values = np.asarray(
        [base_scores[name] for name in validation_names], dtype=np.float64
    )
    spatial = SpatialFoldData(
        fold_index=split.fold_index,
        heldout=(str(split.group_id), str(split.group_id)),
        train_o=split.train_o,
        train_names=train_names,
        global_indices=torch.as_tensor(train_global, dtype=torch.long, device=device),
        base_scores=torch.as_tensor(train_values, dtype=torch.float32, device=device),
        higher=torch.as_tensor(train_higher, dtype=torch.long, device=device),
        lower=torch.as_tensor(train_lower, dtype=torch.long, device=device),
        targets=torch.as_tensor(train_targets, dtype=torch.float32, device=device),
        coefficients=torch.as_tensor(coefficients, dtype=torch.float32, device=device),
        n_validation_counter_pairs=int(split.heldout_counter.sum()),
    )
    return PilotFold(
        row={"pilot_index": split.fold_index, "group_id": split.group_id},
        spatial=spatial,
        validation_names=validation_names,
        validation_global_indices=validation_global,
        validation_base_scores=validation_values,
        validation_higher=np.asarray(validation_higher, dtype=np.int64),
        validation_lower=np.asarray(validation_lower, dtype=np.int64),
        validation_distance=np.asarray(validation_distance, dtype=np.int64),
        validation_class=validation_class,
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
