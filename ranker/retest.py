"""Reusable primitives for paired forced-ranking test--retest audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SUFFICIENT_FIELDS = (
    "pair_correct",
    "pair_count",
    "old_weighted_correct",
    "old_weight_total",
    "new_weighted_correct",
    "new_weight_total",
    "symmetric_weighted_correct",
    "symmetric_weight_total",
    "d_min_ge_3_correct",
    "d_min_ge_3_count",
    "rank_abs_error_sum",
    "rank_image_count",
    "exact_rank_correct",
    "top1_correct",
    "top1_count",
    "top2_set_correct",
    "top2_set_count",
)


def ranking_positions(
    membership: Sequence[str], state: Mapping[str, Any], *, label: str
) -> dict[str, int]:
    """Validate one complete tie-free forced ranking and return 0-based ranks."""
    ranked = state.get("ranked")
    unranked = state.get("unranked")
    if (
        not isinstance(ranked, list)
        or unranked != []
        or len(ranked) != len(membership)
        or len(set(ranked)) != len(ranked)
        or set(ranked) != set(membership)
    ):
        raise ValueError(f"{label} is not a complete tie-free ranking permutation.")
    return {str(image): rank for rank, image in enumerate(ranked)}


def paired_group_retest(
    *,
    source_group_id: int,
    retest_group_id: int,
    identity: str,
    membership: Sequence[str],
    old_ranks: Mapping[str, int],
    new_ranks: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build pair records and sufficient statistics for one paired ranking."""
    names = sorted(map(str, membership))
    if set(old_ranks) != set(names) or set(new_ranks) != set(names):
        raise ValueError("Old and retest rank maps do not match the group membership.")
    expected = list(range(len(names)))
    if sorted(old_ranks.values()) != expected or sorted(new_ranks.values()) != expected:
        raise ValueError("Old and retest ranks must both be contiguous permutations.")

    pair_rows: list[dict[str, Any]] = []
    for offset, image_1 in enumerate(names):
        for image_2 in names[offset + 1 :]:
            old_difference = int(old_ranks[image_2] - old_ranks[image_1])
            new_difference = int(new_ranks[image_2] - new_ranks[image_1])
            d_old = abs(old_difference)
            d_new = abs(new_difference)
            agreement = int((old_difference > 0) == (new_difference > 0))
            pair_rows.append(
                {
                    "source_group_id": source_group_id,
                    "retest_group_id": retest_group_id,
                    "identity": identity,
                    "image_id_1": image_1,
                    "image_id_2": image_2,
                    "old_rank_1": int(old_ranks[image_1]),
                    "old_rank_2": int(old_ranks[image_2]),
                    "retest_rank_1": int(new_ranks[image_1]),
                    "retest_rank_2": int(new_ranks[image_2]),
                    "old_order_sign": 1 if old_difference > 0 else -1,
                    "retest_order_sign": 1 if new_difference > 0 else -1,
                    "d_old": d_old,
                    "d_new": d_new,
                    "d_min": min(d_old, d_new),
                    "agreement": agreement,
                    "discordant": 1 - agreement,
                }
            )

    agreement = np.asarray([row["agreement"] for row in pair_rows], dtype=np.float64)
    d_old = np.asarray([row["d_old"] for row in pair_rows], dtype=np.float64)
    d_new = np.asarray([row["d_new"] for row in pair_rows], dtype=np.float64)
    d_min = np.minimum(d_old, d_new)
    rank_errors = np.asarray(
        [abs(int(new_ranks[name]) - int(old_ranks[name])) for name in names],
        dtype=np.float64,
    )
    old_top = max(names, key=old_ranks.__getitem__)
    new_top = max(names, key=new_ranks.__getitem__)
    old_top2 = set(sorted(names, key=old_ranks.__getitem__, reverse=True)[:2])
    new_top2 = set(sorted(names, key=new_ranks.__getitem__, reverse=True)[:2])
    d3 = d_min >= 3
    group = {
        "source_group_id": source_group_id,
        "retest_group_id": retest_group_id,
        "identity": identity,
        "pair_correct": float(agreement.sum()),
        "pair_count": int(len(agreement)),
        "pair_agreement": float(agreement.mean()),
        "kendall_tau": float(2.0 * agreement.mean() - 1.0),
        "old_weighted_correct": float(np.sum(d_old * agreement)),
        "old_weight_total": float(d_old.sum()),
        "new_weighted_correct": float(np.sum(d_new * agreement)),
        "new_weight_total": float(d_new.sum()),
        "symmetric_weighted_correct": float(np.sum(d_min * agreement)),
        "symmetric_weight_total": float(d_min.sum()),
        "old_weighted_WPA": float(np.sum(d_old * agreement) / d_old.sum()),
        "new_weighted_WPA": float(np.sum(d_new * agreement) / d_new.sum()),
        "symmetric_WPA": float(np.sum(d_min * agreement) / d_min.sum()),
        "d_min_ge_3_correct": float(agreement[d3].sum()),
        "d_min_ge_3_count": int(d3.sum()),
        "d_min_ge_3_agreement": float(agreement[d3].mean()),
        "rank_abs_error_sum": float(rank_errors.sum()),
        "rank_image_count": int(len(rank_errors)),
        "rank_MAE": float(rank_errors.mean()),
        "exact_rank_correct": int(np.sum(rank_errors == 0.0)),
        "exact_rank_agreement": float(np.mean(rank_errors == 0.0)),
        "top1_correct": int(old_top == new_top),
        "top1_count": 1,
        "top1_agreement": float(old_top == new_top),
        "top2_set_correct": int(old_top2 == new_top2),
        "top2_set_count": 1,
        "top2_set_agreement": float(old_top2 == new_top2),
        "discordant_pairs": int(np.sum(agreement == 0.0)),
    }
    return pair_rows, group


def aggregate_retest_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Pool group/identity sufficient statistics before forming ratios."""
    totals = {field: float(sum(float(row[field]) for row in rows)) for field in SUFFICIENT_FIELDS}
    return {
        **totals,
        "pair_agreement": totals["pair_correct"] / totals["pair_count"],
        "old_weighted_WPA": totals["old_weighted_correct"] / totals["old_weight_total"],
        "new_weighted_WPA": totals["new_weighted_correct"] / totals["new_weight_total"],
        "symmetric_WPA": totals["symmetric_weighted_correct"] / totals["symmetric_weight_total"],
        "d_min_ge_3_agreement": totals["d_min_ge_3_correct"] / totals["d_min_ge_3_count"],
        "rank_MAE": totals["rank_abs_error_sum"] / totals["rank_image_count"],
        "exact_rank_agreement": totals["exact_rank_correct"] / totals["rank_image_count"],
        "top1_agreement": totals["top1_correct"] / totals["top1_count"],
        "top2_set_agreement": totals["top2_set_correct"] / totals["top2_set_count"],
    }


def bootstrap_retest_metrics(
    rows: Sequence[Mapping[str, Any]], *, seed: int, replicates: int
) -> dict[str, tuple[float, float]]:
    """Percentile cluster bootstrap for the five preregistered retest metrics."""
    if not rows or replicates <= 0:
        raise ValueError("Bootstrap requires clusters and a positive replicate count.")
    arrays = {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in SUFFICIENT_FIELDS
    }
    output = {
        metric: np.empty(replicates, dtype=np.float64)
        for metric in (
            "pair_agreement",
            "symmetric_WPA",
            "d_min_ge_3_agreement",
            "rank_MAE",
            "top1_agreement",
        )
    }
    rng = np.random.default_rng(seed)
    chunk_size = 10_000
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, len(rows), size=(stop - start, len(rows)))

        def pooled(numerator: str, denominator: str) -> np.ndarray:
            return arrays[numerator][indices].sum(axis=1) / arrays[denominator][indices].sum(axis=1)

        output["pair_agreement"][start:stop] = pooled("pair_correct", "pair_count")
        output["symmetric_WPA"][start:stop] = pooled(
            "symmetric_weighted_correct", "symmetric_weight_total"
        )
        output["d_min_ge_3_agreement"][start:stop] = pooled(
            "d_min_ge_3_correct", "d_min_ge_3_count"
        )
        output["rank_MAE"][start:stop] = pooled("rank_abs_error_sum", "rank_image_count")
        output["top1_agreement"][start:stop] = pooled("top1_correct", "top1_count")
    return {
        metric: tuple(map(float, np.percentile(values, [2.5, 97.5])))
        for metric, values in output.items()
    }
