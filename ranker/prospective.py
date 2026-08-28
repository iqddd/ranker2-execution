"""Reusable statistics for frozen prospective ranking campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta, binom


def ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio for a strictly positive denominator."""
    if denominator <= 0:
        raise ZeroDivisionError("A prospective metric denominator must be positive.")
    return float(numerator) / float(denominator)


def exact_positive_group_sign_test(nets: Sequence[int | float]) -> dict[str, Any]:
    """One-sided exact sign test after removing zero-net groups."""
    values = np.asarray(nets, dtype=np.float64)
    nonzero = values != 0.0
    m = int(np.count_nonzero(nonzero))
    k = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    p_value = 1.0 if m == 0 else float(binom.sf(k - 1, m, 0.5))
    return {
        "m_nonzero_net_groups": m,
        "k_positive_net_groups": k,
        "zero_net_groups": int(len(values) - m),
        "negative_net_groups": negative,
        "exact_one_sided_p": p_value,
    }


def percentile_interval(values: np.ndarray) -> list[float]:
    """Canonical two-sided 95% percentile interval with linear quantiles."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("No finite bootstrap values were available.")
    return [
        float(value)
        for value in np.quantile(finite, [0.025, 0.975], method="linear")
    ]


def bootstrap_primary_groups(
    *,
    head_wins: Sequence[int],
    denominators: Sequence[int],
    group_q: Sequence[float],
    replicates: int,
    seed: int,
    chunk_size: int = 20_000,
) -> dict[str, Any]:
    """Bootstrap complete groups for the preregistered primary summaries."""
    head = np.asarray(head_wins, dtype=np.int64)
    total = np.asarray(denominators, dtype=np.int64)
    q = np.asarray(group_q, dtype=np.float64)
    if not (len(head) == len(total) == len(q)) or not len(q):
        raise ValueError("Primary group arrays must have one common positive length.")
    rng = np.random.Generator(np.random.PCG64(seed))
    mean_q = np.empty(replicates, dtype=np.float64)
    pooled_head_win_rate = np.empty(replicates, dtype=np.float64)
    pooled_net = np.empty(replicates, dtype=np.int64)
    zero_denominator_replicates = 0
    cursor = 0
    while cursor < replicates:
        stop = min(cursor + chunk_size, replicates)
        indexes = rng.integers(0, len(q), size=(stop - cursor, len(q)))
        sampled_head = head[indexes].sum(axis=1)
        sampled_total = total[indexes].sum(axis=1)
        mean_q[cursor:stop] = q[indexes].mean(axis=1)
        pooled_net[cursor:stop] = 2 * sampled_head - sampled_total
        valid = sampled_total > 0
        zero_denominator_replicates += int(np.count_nonzero(~valid))
        pooled_head_win_rate[cursor:stop] = np.divide(
            sampled_head,
            sampled_total,
            out=np.full(stop - cursor, np.nan, dtype=np.float64),
            where=valid,
        )
        cursor = stop
    return {
        "mean_group_q": mean_q,
        "pooled_head_win_rate": pooled_head_win_rate,
        "pooled_net": pooled_net,
        "zero_denominator_replicates": zero_denominator_replicates,
        "mean_group_q_ci": percentile_interval(mean_q),
        "pooled_head_win_rate_ci": percentile_interval(pooled_head_win_rate),
    }


def paired_group_mean_difference_bootstrap(
    *,
    first: Sequence[float],
    second: Sequence[float],
    replicates: int,
    seed: int,
    chunk_size: int = 20_000,
) -> dict[str, Any]:
    """Paired complete-group bootstrap for two identity contribution vectors."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape or not len(first_values):
        raise ValueError("Paired identity vectors must have one common positive length.")
    rng = np.random.Generator(np.random.PCG64(seed))
    differences = np.empty(replicates, dtype=np.float64)
    cursor = 0
    while cursor < replicates:
        stop = min(cursor + chunk_size, replicates)
        indexes = rng.integers(
            0, len(first_values), size=(stop - cursor, len(first_values))
        )
        differences[cursor:stop] = (
            first_values[indexes].mean(axis=1)
            - second_values[indexes].mean(axis=1)
        )
        cursor = stop
    return {
        "differences": differences,
        "ci": percentile_interval(differences),
    }


def bootstrap_pooled_binary_rate(
    *,
    numerators: Sequence[int],
    denominators: Sequence[int],
    replicates: int,
    seed: int,
    chunk_size: int = 20_000,
) -> dict[str, Any]:
    """Bootstrap complete groups and pool binary numerators before division."""
    numerator = np.asarray(numerators, dtype=np.int64)
    denominator = np.asarray(denominators, dtype=np.int64)
    if numerator.shape != denominator.shape or not len(numerator):
        raise ValueError("Binary group vectors must have one common positive length.")
    rng = np.random.Generator(np.random.PCG64(seed))
    rates = np.empty(replicates, dtype=np.float64)
    zero_denominator_replicates = 0
    cursor = 0
    while cursor < replicates:
        stop = min(cursor + chunk_size, replicates)
        indexes = rng.integers(
            0, len(numerator), size=(stop - cursor, len(numerator))
        )
        sampled_numerator = numerator[indexes].sum(axis=1)
        sampled_denominator = denominator[indexes].sum(axis=1)
        valid = sampled_denominator > 0
        zero_denominator_replicates += int(np.count_nonzero(~valid))
        rates[cursor:stop] = np.divide(
            sampled_numerator,
            sampled_denominator,
            out=np.full(stop - cursor, np.nan, dtype=np.float64),
            where=valid,
        )
        cursor = stop
    return {
        "rates": rates,
        "ci": percentile_interval(rates),
        "zero_denominator_replicates": zero_denominator_replicates,
    }


def clopper_pearson_upper(
    successes: int, total: int, *, confidence: float = 0.95
) -> float:
    """Exact one-sided binomial upper confidence bound."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Expected 0 <= successes <= total with total > 0.")
    if successes == total:
        return 1.0
    if successes == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / total))
    return float(beta.ppf(confidence, successes + 1, total - successes))


def grouped_binary_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    selected_key: str,
    event_key: str,
    group_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect event and selected counts in a fixed complete-group order."""
    position = {group: index for index, group in enumerate(group_order)}
    events = np.zeros(len(group_order), dtype=np.int64)
    totals = np.zeros(len(group_order), dtype=np.int64)
    for row in rows:
        if not bool(row[selected_key]):
            continue
        index = position[str(row[group_key])]
        totals[index] += 1
        events[index] += int(bool(row[event_key]))
    return events, totals


def bootstrap_identity_pair_weighted_differences(
    *,
    numerators: Mapping[str, Sequence[int | float]],
    denominators: Mapping[str, Sequence[int | float]],
    identity_pairs: Sequence[tuple[str, str]],
    identity_order: Sequence[str],
    replicates: int,
    seed: int,
    chunk_size: int = 10_000,
) -> dict[str, dict[str, Any]]:
    """Bootstrap identities and weight each held-out identity pair multiplicatively."""
    if set(numerators) != set(denominators) or not numerators:
        raise ValueError("Bootstrap numerator and denominator keys must match.")
    if replicates <= 0 or chunk_size <= 0 or not identity_order:
        raise ValueError("Bootstrap sizes and identity order must be positive.")
    identity_index = {identity: index for index, identity in enumerate(identity_order)}
    if len(identity_index) != len(identity_order):
        raise ValueError("Identity order must be unique.")
    try:
        pair_indices = np.asarray(
            [(identity_index[a], identity_index[b]) for a, b in identity_pairs],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"Unknown identity in held-out pair: {error.args[0]}") from error
    expected_length = len(identity_pairs)
    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in numerators:
        numerator = np.asarray(numerators[name], dtype=np.float64)
        denominator = np.asarray(denominators[name], dtype=np.float64)
        if numerator.shape != (expected_length,) or denominator.shape != (expected_length,):
            raise ValueError("Every bootstrap vector must match identity_pairs.")
        prepared[name] = numerator, denominator
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(
        len(identity_order),
        np.full(len(identity_order), 1.0 / len(identity_order)),
        size=replicates,
    ).astype(np.int16)
    distributions = {
        name: np.empty(replicates, dtype=np.float64) for name in prepared
    }
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        current = multiplicities[start:stop]
        weights = current[:, pair_indices[:, 0]] * current[:, pair_indices[:, 1]]
        for name, (numerator, denominator) in prepared.items():
            weighted_denominator = weights @ denominator
            distributions[name][start:stop] = np.divide(
                weights @ numerator,
                weighted_denominator,
                out=np.full(stop - start, np.nan),
                where=weighted_denominator > 0,
            )
    result: dict[str, dict[str, Any]] = {}
    for name, distribution in distributions.items():
        finite = np.isfinite(distribution)
        if not np.any(finite):
            raise ValueError(f"Bootstrap {name!r} has no finite replicates.")
        result[name] = {
            "distribution": distribution,
            "ci": tuple(
                float(value)
                for value in np.percentile(distribution[finite], [2.5, 97.5])
            ),
            "replicates_valid": int(np.count_nonzero(finite)),
        }
    return result
