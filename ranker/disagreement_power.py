"""Historical transport simulation for frozen same-identity disagreement edges."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import binom


def confidence_bin(value: float) -> str:
    if value < 0.05:
        return "[0,0.05)"
    if value < 0.10:
        return "[0.05,0.10)"
    if value < 0.25:
        return "[0.10,0.25)"
    if value < 0.50:
        return "[0.25,0.50)"
    return "[0.50,+inf)"


def parse_boolean(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def historical_same_identity_records(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Extract the Step43D same-identity D710 historical transport records."""
    result: list[dict[str, Any]] = []
    for row in rows:
        if row["source"] != "43D" or not parse_boolean(row["same_identity"]):
            continue
        result.append(
            {
                "identity": str(row["lower_identity"]),
                "distance": int(row["distance"]),
                "sign": 1 if parse_boolean(row["HEAD_win"]) else -1,
                "confidence_bin": confidence_bin(
                    float(row["disagreement_confidence"])
                ),
            }
        )
    if not result:
        raise RuntimeError("No Step43D same-identity D710 history was found.")
    return result


def exact_sign_test_table(maximum_groups: int) -> np.ndarray:
    """Precompute P[X >= k], X~Binomial(m, 0.5), for all group counts."""
    table = np.ones((maximum_groups + 1, maximum_groups + 1), dtype=np.float64)
    for m in range(1, maximum_groups + 1):
        for k in range(m + 1):
            table[m, k] = float(binom.sf(k - 1, m, 0.5))
    return table


def simulate_sign_test_power(
    *,
    edges: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    group_count: int,
    replicates: int,
    seed: int,
    minimum_qualifying_pairs: int,
    minimum_nonzero_denominator_groups: int,
    effect_multipliers: Sequence[float],
    chunk_size: int = 10_000,
) -> list[dict[str, Any]]:
    """Simulate optimistic/conservative transport under the frozen sign test."""
    identities = sorted({str(row["identity"]) for row in history})
    pooled_by_bin: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_identity_bin: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in history:
        identity = str(row["identity"])
        bin_name = str(row["confidence_bin"])
        pooled_by_bin[bin_name].append(row)
        by_identity[identity].append(row)
        by_identity_bin[(identity, bin_name)].append(row)
    if len(identities) != 14:
        raise RuntimeError("Historical transport requires exactly fourteen identities.")

    edge_group = np.asarray([int(edge["group"]) for edge in edges], dtype=np.int16)
    edge_reserve = np.asarray(
        [int(edge["reserve_index"]) for edge in edges], dtype=np.int8
    )
    edge_stable = np.asarray([bool(edge["stable"]) for edge in edges], dtype=bool)
    edge_bin = [str(edge["confidence_bin"]) for edge in edges]
    sign_p_values = exact_sign_test_table(group_count)
    rng = np.random.default_rng(seed)
    effects = tuple(float(value) for value in effect_multipliers)
    if not effects:
        raise ValueError("effect_multipliers must not be empty.")
    totals = {
        (transport, effect): {
            "qualifying": 0.0,
            "nonzero_denominator": 0.0,
            "nonzero_net": 0.0,
            "positive_net": 0.0,
            "p_sign": 0,
            "pooled_direction": 0,
            "full_pass": 0,
        }
        for transport in ("optimistic", "conservative")
        for effect in effects
    }

    completed = 0
    while completed < replicates:
        size = min(chunk_size, replicates - completed)
        chosen_identity = rng.integers(0, len(identities), size=(size, 2))
        distances = np.empty((size, len(edges)), dtype=np.int8)
        historical_signs = np.empty((size, len(edges)), dtype=np.int8)
        for reserve_index in range(2):
            edge_indices = np.flatnonzero(edge_reserve == reserve_index)
            for identity_index, identity in enumerate(identities):
                mask = chosen_identity[:, reserve_index] == identity_index
                count = int(mask.sum())
                if not count:
                    continue
                for edge_index in edge_indices:
                    bin_name = edge_bin[int(edge_index)]
                    pool = by_identity_bin[(identity, bin_name)]
                    if not pool:
                        pool = by_identity[identity]
                    if not pool:
                        pool = pooled_by_bin[bin_name]
                    if not pool:
                        pool = list(history)
                    draws = rng.integers(0, len(pool), size=count)
                    distances[mask, edge_index] = np.asarray(
                        [pool[index]["distance"] for index in draws],
                        dtype=np.int8,
                    )
                    historical_signs[mask, edge_index] = np.asarray(
                        [pool[index]["sign"] for index in draws],
                        dtype=np.int8,
                    )
        qualifying = distances >= 2
        for transport in ("optimistic", "conservative"):
            for effect in effects:
                reflected = np.where(
                    rng.random(historical_signs.shape)
                    < ((1.0 + effect) / 2.0),
                    historical_signs,
                    -historical_signs,
                )
                if transport == "conservative":
                    null_sign = np.where(
                        rng.random(historical_signs.shape) < 0.5, 1, -1
                    )
                    effective_sign = np.where(
                        edge_stable.reshape(1, -1), reflected, null_sign
                    )
                else:
                    effective_sign = reflected

                group_n = np.zeros((size, group_count), dtype=np.int16)
                group_net = np.zeros((size, group_count), dtype=np.int16)
                for group_index in range(group_count):
                    mask = edge_group == group_index
                    group_n[:, group_index] = qualifying[:, mask].sum(axis=1)
                    group_net[:, group_index] = (
                        effective_sign[:, mask] * qualifying[:, mask]
                    ).sum(axis=1)
                total_n = group_n.sum(axis=1)
                denominator_groups = np.count_nonzero(group_n, axis=1)
                nonzero_net = np.count_nonzero(group_net, axis=1)
                positive_net = np.count_nonzero(group_net > 0, axis=1)
                p_sign = sign_p_values[nonzero_net, positive_net]
                total_net = group_net.sum(axis=1)
                sign_pass = p_sign <= 0.05
                pooled_direction = total_net > 0
                full_pass = (
                    (total_n >= minimum_qualifying_pairs)
                    & (
                        denominator_groups
                        >= minimum_nonzero_denominator_groups
                    )
                    & sign_pass
                    & pooled_direction
                )
                bucket = totals[(transport, effect)]
                bucket["qualifying"] += float(total_n.sum())
                bucket["nonzero_denominator"] += float(denominator_groups.sum())
                bucket["nonzero_net"] += float(nonzero_net.sum())
                bucket["positive_net"] += float(positive_net.sum())
                bucket["p_sign"] += int(np.count_nonzero(sign_pass))
                bucket["pooled_direction"] += int(
                    np.count_nonzero(pooled_direction)
                )
                bucket["full_pass"] += int(np.count_nonzero(full_pass))
        completed += size

    result: list[dict[str, Any]] = []
    for transport in ("optimistic", "conservative"):
        for effect in effects:
            bucket = totals[(transport, effect)]
            result.append(
                {
                    "transport": transport,
                    "effect_multiplier": effect,
                    "raw_D710_edges": len(edges),
                    "expected_qualifying_pairs": (
                        bucket["qualifying"] / replicates
                    ),
                    "expected_groups_with_N_gt_0": (
                        bucket["nonzero_denominator"] / replicates
                    ),
                    "expected_groups_with_nonzero_net": (
                        bucket["nonzero_net"] / replicates
                    ),
                    "expected_groups_with_positive_net": (
                        bucket["positive_net"] / replicates
                    ),
                    "P_p_sign_le_0_05": bucket["p_sign"] / replicates,
                    "P_pooled_HEAD_wins_gt_BASE_wins": (
                        bucket["pooled_direction"] / replicates
                    ),
                    "P_full_pass": bucket["full_pass"] / replicates,
                    "simulation_replicates": replicates,
                    "seed": seed,
                }
            )
    return result
