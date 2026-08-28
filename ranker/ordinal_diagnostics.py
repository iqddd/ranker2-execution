"""Reusable diagnostics for direct-ordinal score panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


def five_system_concordance(
    internal: Mapping[str, float], external: Mapping[str, float]
) -> dict[str, Any]:
    """Compare two lower-is-better system orderings with explicit tie credit."""
    systems = sorted(internal)
    if set(systems) != set(external):
        raise ValueError("Internal and external system panels differ.")
    x = np.asarray([internal[name] for name in systems], dtype=np.float64)
    y = np.asarray([external[name] for name in systems], dtype=np.float64)
    agreement = 0.0
    for first, second in combinations(systems, 2):
        internal_sign = np.sign(internal[first] - internal[second])
        external_sign = np.sign(external[first] - external[second])
        agreement += 0.5 if internal_sign == 0 else float(internal_sign == external_sign)
    order = sorted(systems, key=lambda name: (internal[name], name.encode("utf-8")))
    external_order = sorted(systems, key=lambda name: (external[name], name.encode("utf-8")))
    return {
        "order": order,
        "external_order": external_order,
        "spearman_rho": float(spearmanr(x, y).statistic),
        "kendall_tau_b": float(kendalltau(x, y, variant="b").statistic),
        "pairwise_order_agreement": float(agreement),
        "external_best_reproduced": order[0] == external_order[0],
        "external_worst_reproduced": order[-1] == external_order[-1],
    }


def within_identity_rank_change(
    reference: np.ndarray,
    candidate: np.ndarray,
    identity_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Measure within-identity rank changes and unordered pair flips."""
    rows: list[dict[str, Any]] = []
    for identity in sorted(set(identity_ids.tolist()), key=lambda value: value.encode("utf-8")):
        mask = identity_ids == identity
        first = np.asarray(reference[mask], dtype=np.float64)
        second = np.asarray(candidate[mask], dtype=np.float64)
        ranks_first = rankdata(first, method="average")
        ranks_second = rankdata(second, method="average")
        flips = []
        for left, right in combinations(range(len(first)), 2):
            a = np.sign(first[left] - first[right])
            b = np.sign(second[left] - second[right])
            flips.append(0.5 if a == 0 or b == 0 else float(a != b))
        shift = np.abs(ranks_second - ranks_first)
        tau = float(kendalltau(first, second, variant="b").statistic)
        rows.append({
            "identity_id": identity,
            "images": int(mask.sum()),
            "fraction_rank_changed": float(np.mean(shift != 0)),
            "mean_absolute_rank_shift": float(np.mean(shift)),
            "maximum_absolute_rank_shift": float(np.max(shift)),
            "within_identity_Kendall_tau_b": tau,
            "within_identity_pair_flip_rate": float(np.mean(flips)),
        })
    weights = np.asarray([row["images"] for row in rows], dtype=np.float64)
    summary = {
        "image_weighted_fraction_changed": float(np.average([row["fraction_rank_changed"] for row in rows], weights=weights)),
        "identity_balanced_fraction_changed": float(np.mean([row["fraction_rank_changed"] for row in rows])),
        "identity_balanced_mean_absolute_rank_shift": float(np.mean([row["mean_absolute_rank_shift"] for row in rows])),
        "identity_balanced_pair_flip_rate": float(np.mean([row["within_identity_pair_flip_rate"] for row in rows])),
    }
    return rows, summary


def offset_guard(
    exposures: Sequence[Mapping[str, Any]],
    scores_by_identity: Mapping[str, Mapping[str, float]],
    identities: Sequence[str],
) -> tuple[list[dict[str, Any]], float]:
    """Compute the theta-free directional over/under-estimation guard."""
    output: list[dict[str, Any]] = []
    for identity in identities:
        high = 0.0
        low = 0.0
        panel = scores_by_identity[identity]
        for row in exposures:
            if str(row["identity_id"]) != identity:
                continue
            x_score = float(panel[str(row["image_key"])])
            for anchor, anchor_rank in zip(row["anchor_keys"], row["anchor_ranks"], strict=True):
                model_sign = np.sign(x_score - float(panel[str(anchor)]))
                human_sign = np.sign(int(row["human_rank"]) - int(anchor_rank))
                high += float(human_sign < 0 and model_sign > 0) - 0.5 * float(model_sign == 0)
                low += float(human_sign > 0 and model_sign < 0) - 0.5 * float(model_sign == 0)
        error = high + low
        signed = (high - low) / error if error > 0 else 0.0
        output.append({
            "identity_id": identity,
            "N_high": high,
            "N_low": low,
            "N_error": error,
            "S_b": signed,
            "abs_S_b": abs(signed),
        })
    return output, float(np.mean([row["abs_S_b"] for row in output]))
