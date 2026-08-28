"""Empirical validation-resolution and direct ordinal endpoint helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ranker.anchored_ordinal import comparison_credit
from ranker.data import RankedImage


def empirical_sign_mde(distribution: np.ndarray, direction: int) -> float:
    """Smallest injected magnitude reaching 90% empirical sign probability."""
    values = np.asarray(distribution, dtype=np.float64)
    residual = values - np.mean(values, dtype=np.float64)
    predicate = (
        (lambda shift: bool(np.mean(residual + shift < 0.0) >= 0.90))
        if direction < 0
        else (lambda shift: bool(np.mean(residual + shift > 0.0) >= 0.90))
    )
    result = bisect_gate_mde(predicate, direction, tolerance=1e-6)
    if result is None:
        raise RuntimeError("empirical sign MDE was not bracketed")
    return result


def bisect_gate_mde(
    predicate: Callable[[float], bool],
    direction: int,
    *,
    tolerance: float = 1e-6,
    maximum: float = 1e6,
) -> float | None:
    """Find the smallest signed effect magnitude satisfying a monotone gate."""
    low, high = 0.0, 1.0
    while high <= maximum and not predicate(direction * high):
        high *= 2.0
    if high > maximum:
        return None
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if predicate(direction * middle):
            high = middle
        else:
            low = middle
    return high


def direct_ordinal_exposures(
    groups: Sequence[Sequence[RankedImage]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate cross-identity anchored exposures and coverage records."""
    exposures: list[dict[str, Any]] = []
    for group in groups:
        by_identity: dict[str, list[RankedImage]] = defaultdict(list)
        for item in group:
            by_identity[item.identity].append(item)
        for identity, new_items in by_identity.items():
            anchors = [item for item in group if item.identity != identity]
            if not new_items or not anchors:
                continue
            anchor_ranks = np.asarray([item.rank for item in anchors], dtype=np.int64)
            for item in new_items:
                distances = 1 + np.asarray(
                    [
                        np.count_nonzero(
                            (anchor_ranks > min(item.rank, anchor.rank))
                            & (anchor_ranks < max(item.rank, anchor.rank))
                        )
                        for anchor in anchors
                    ],
                    dtype=np.int64,
                )
                exposures.append(
                    {
                        "group_index": item.group_index,
                        "image_key": item.image_name,
                        "identity_id": identity,
                        "human_rank": item.rank,
                        "anchor_keys": tuple(anchor.image_name for anchor in anchors),
                        "anchor_identities": tuple(anchor.identity for anchor in anchors),
                        "anchor_ranks": tuple(anchor.rank for anchor in anchors),
                        "distances": tuple(map(int, distances)),
                    }
                )
    coverage: list[dict[str, Any]] = []
    identities = sorted({row["identity_id"] for row in exposures})
    for identity in identities:
        rows = [row for row in exposures if row["identity_id"] == identity]
        anchor_counts = np.asarray([len(row["anchor_keys"]) for row in rows])
        distance = [value for row in rows for value in row["distances"]]
        coverage.append(
            {
                "identity_id": identity,
                "pseudo_new_images": len({row["image_key"] for row in rows}),
                "human_group_exposures": len(rows),
                "unique_cross_identity_anchors": len(
                    {key for row in rows for key in row["anchor_keys"]}
                ),
                "new_anchor_comparisons": int(anchor_counts.sum()),
                "mean_anchors_per_exposure": float(anchor_counts.mean()),
                "min_anchors_per_exposure": int(anchor_counts.min()),
                "max_anchors_per_exposure": int(anchor_counts.max()),
                "human_rank_distance_distribution": str(
                    {value: distance.count(value) for value in sorted(set(distance))}
                ),
                "outer_position_comparison_count": sum(
                    row["human_rank"] in {0, 7} for row in rows for _ in row["anchor_keys"]
                ),
            }
        )
    return exposures, coverage


def evaluate_direct_ordinal(
    exposures: Sequence[Mapping[str, Any]],
    scores_by_identity: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Evaluate image-, identity-, and globally balanced direct ordinal loss."""
    exposure_rows: list[dict[str, Any]] = []
    for exposure in exposures:
        identity = str(exposure["identity_id"])
        if identity not in scores_by_identity:
            continue
        panel = scores_by_identity[identity]
        image_key = str(exposure["image_key"])
        if image_key not in panel or any(
            key not in panel for key in exposure["anchor_keys"]
        ):
            continue
        credits = np.asarray(
            [
                comparison_credit(
                    panel[image_key] - panel[anchor_key],
                    int(exposure["human_rank"]) - int(anchor_rank),
                )
                for anchor_key, anchor_rank in zip(
                    exposure["anchor_keys"], exposure["anchor_ranks"], strict=True
                )
            ],
            dtype=np.float64,
        )
        distances = np.asarray(exposure["distances"], dtype=np.float64)
        exposure_rows.append(
            {
                **dict(exposure),
                "loss": float(np.sum(distances * (1.0 - credits)) / distances.sum()),
            }
        )
    image_rows: list[dict[str, Any]] = []
    for identity in sorted({row["identity_id"] for row in exposure_rows}):
        keys = sorted(
            {row["image_key"] for row in exposure_rows if row["identity_id"] == identity}
        )
        for key in keys:
            values = [
                row["loss"]
                for row in exposure_rows
                if row["identity_id"] == identity and row["image_key"] == key
            ]
            image_rows.append(
                {"identity_id": identity, "image_key": key, "loss": float(np.mean(values))}
            )
    identity_rows = [
        {
            "identity_id": identity,
            "loss": float(
                np.mean(
                    [row["loss"] for row in image_rows if row["identity_id"] == identity]
                )
            ),
        }
        for identity in sorted({row["identity_id"] for row in image_rows})
    ]
    return {
        "loss": (
            float(np.mean([row["loss"] for row in identity_rows]))
            if identity_rows
            else None
        ),
        "exposures": exposure_rows,
        "images": image_rows,
        "identities": identity_rows,
    }


def direct_identity_loss_vector(
    exposures: Sequence[Mapping[str, Any]],
    scores_by_identity: Mapping[str, Mapping[str, float]],
    identities: Sequence[str],
) -> np.ndarray:
    """Return a direct-ordinal identity vector in an explicit stable order."""
    result = evaluate_direct_ordinal(exposures, scores_by_identity)
    rows = {
        str(row["identity_id"]): float(row["loss"])
        for row in result["identities"]
    }
    if set(rows) != set(identities):
        raise RuntimeError(f"direct endpoint coverage mismatch: {sorted(rows)}")
    return np.asarray([rows[identity] for identity in identities], dtype=np.float64)


def filter_direct_ordinal_exposures(
    exposures: Sequence[Mapping[str, Any]],
    *,
    minimum_distance: int,
) -> list[dict[str, Any]]:
    """Restrict each exposure to anchors meeting a distance threshold.

    Exposures without an eligible anchor are omitted.  Downstream evaluation
    therefore retains the canonical exposure -> image -> identity balancing
    while renormalizing the weighted loss over eligible anchors only.
    """
    if minimum_distance < 1:
        raise ValueError("minimum_distance must be positive")
    output: list[dict[str, Any]] = []
    for source in exposures:
        keep = [
            index
            for index, distance in enumerate(source["distances"])
            if int(distance) >= minimum_distance
        ]
        if not keep:
            continue
        row = dict(source)
        for field in ("anchor_keys", "anchor_identities", "anchor_ranks", "distances"):
            row[field] = tuple(source[field][index] for index in keep)
        output.append(row)
    return output


def contiguous_checkpoint_plateau(
    checkpoints: Sequence[int],
    losses: Sequence[float],
    selected_index: int,
    tolerance: float,
) -> list[int]:
    """Return the contiguous checkpoint-grid plateau around a selected point."""
    if len(checkpoints) != len(losses):
        raise ValueError("Checkpoint and loss counts differ.")
    qualified = np.asarray(losses, dtype=np.float64) <= float(np.min(losses)) + tolerance
    left = selected_index
    right = selected_index
    while left > 0 and qualified[left - 1]:
        left -= 1
    while right + 1 < len(checkpoints) and qualified[right + 1]:
        right += 1
    return list(checkpoints[left : right + 1])


def direct_retest_swap_contributions(
    original_groups: Sequence[Sequence[RankedImage]],
    retest_groups: Sequence[Sequence[RankedImage]],
    source_indices: Sequence[int],
    scores_by_identity: Mapping[str, Mapping[str, float]],
    identities: Sequence[str],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Measure exact additive endpoint changes for independent group replacements.

    The direct endpoint first averages repeated exposures of an image, then images
    within identity, then identities.  Contributions therefore retain the exact
    exposure/image/identity denominators of the complete original population.
    """
    full_exposures, _ = direct_ordinal_exposures(original_groups)
    evaluated = evaluate_direct_ordinal(full_exposures, scores_by_identity)
    baseline = float(evaluated["loss"])
    exposure_count: dict[tuple[str, str], int] = defaultdict(int)
    image_count: dict[str, int] = defaultdict(int)
    for row in evaluated["exposures"]:
        exposure_count[(str(row["identity_id"]), str(row["image_key"]))] += 1
    for row in evaluated["images"]:
        image_count[str(row["identity_id"])] += 1
    if set(image_count) != set(identities):
        raise RuntimeError("Retest endpoint identity coverage mismatch.")

    def exposure_losses(group: Sequence[RankedImage]) -> dict[tuple[str, str], float]:
        exposures, _ = direct_ordinal_exposures([group])
        rows = evaluate_direct_ordinal(exposures, scores_by_identity)["exposures"]
        return {
            (str(row["identity_id"]), str(row["image_key"])): float(row["loss"])
            for row in rows
        }

    contributions = np.zeros(len(source_indices), dtype=np.float64)
    identity_contributions = np.zeros((len(source_indices), len(identities)), dtype=np.float64)
    identity_index = {identity: index for index, identity in enumerate(identities)}
    for swap_index, (source_index, retest_group) in enumerate(
        zip(source_indices, retest_groups, strict=True)
    ):
        source = exposure_losses(original_groups[int(source_index)])
        replacement = exposure_losses(retest_group)
        if set(source) != set(replacement):
            raise RuntimeError("Retest replacement changes the exposed image set.")
        for key in source:
            identity, image_key = key
            change = replacement[key] - source[key]
            weighted = change / exposure_count[(identity, image_key)] / image_count[identity]
            identity_contributions[swap_index, identity_index[identity]] += weighted
        contributions[swap_index] = identity_contributions[swap_index].mean()
    return baseline, contributions, identity_contributions
