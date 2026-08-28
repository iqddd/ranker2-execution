from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import mean, pstdev
from typing import Any, Callable, Mapping

import numpy as np

from .data import RankedImage


@dataclass
class PairMetricAccumulator:
    """Accumulate the canonical score-based pair metrics used by step scripts."""

    distance_thresholds: tuple[int, ...] = (2, 3, 4)
    pair_correct: float = 0.0
    pair_count: float = 0.0
    weighted_correct: float = 0.0
    weight: float = 0.0
    _thresholds: dict[int, list[float]] | None = None

    def __post_init__(self) -> None:
        if self._thresholds is None:
            self._thresholds = {threshold: [0.0, 0.0] for threshold in self.distance_thresholds}

    def add(self, *, correct: float, distance: float) -> None:
        self.pair_correct += correct
        self.pair_count += 1.0
        self.weighted_correct += distance * correct
        self.weight += distance
        assert self._thresholds is not None
        for threshold, totals in self._thresholds.items():
            if distance >= threshold:
                totals[0] += distance * correct
                totals[1] += distance

    def as_metrics(self, *, top1_hard: float | None = None) -> dict[str, float]:
        ratio = lambda numerator, denominator: float("nan") if denominator == 0 else numerator / denominator
        result = {
            "pair_acc": ratio(self.pair_correct, self.pair_count),
            "WPA": ratio(self.weighted_correct, self.weight),
        }
        assert self._thresholds is not None
        for threshold, (numerator, denominator) in self._thresholds.items():
            result[f"d_ge_{threshold}_WPA"] = ratio(numerator, denominator)
        if top1_hard is not None:
            result["top1_hard"] = float(top1_hard)
        return result


def accumulate_ranked_pair(
    accumulator: PairMetricAccumulator,
    higher: RankedImage,
    lower: RankedImage,
    scores: Mapping[str, float],
) -> None:
    """Apply the shared rank direction and score-tie rule to one pair."""
    distance = float(higher.rank - lower.rank)
    accumulator.add(
        correct=pair_correct_from_scores(
            float(scores[higher.image_name]), float(scores[lower.image_name])
        ),
        distance=distance,
    )


def evaluate_score_map(
    groups: list[list[RankedImage]],
    scores: Mapping[str, float],
    *,
    distance_thresholds: tuple[int, ...] = (2, 3, 4),
) -> dict[str, float]:
    accumulator = PairMetricAccumulator(distance_thresholds)
    top_hits = 0
    for group in groups:
        true_top = max(group, key=lambda item: item.rank).image_name
        predicted_top = min(group, key=lambda item: (-float(scores[item.image_name]), item.image_name)).image_name
        top_hits += int(predicted_top == true_top)
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (first, second) if first.rank < second.rank else (second, first)
                accumulate_ranked_pair(accumulator, higher, lower, scores)
    return accumulator.as_metrics(top1_hard=top_hits / len(groups))


def r_squared(y: np.ndarray, prediction: np.ndarray, *, centered: bool) -> float:
    denominator = float(np.sum((y - y.mean()) ** 2)) if centered else float(np.sum(y ** 2))
    return float(1.0 - np.sum((y - prediction) ** 2) / denominator)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    value = float(spearmanr(x, y).statistic)
    return 0.0 if not np.isfinite(value) else value


def benjamini_hochberg(p_values: list[float] | tuple[float, ...]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return [float(value) for value in result]


@dataclass(frozen=True)
class MetricTriple:
    pair_acc: float
    weighted_pair_acc: float
    top1: float


@dataclass(frozen=True)
class SystemMetrics:
    pair_acc: float
    weighted_pair_acc: float
    top1_hard: float
    top1_tie_aware: float


@dataclass(frozen=True)
class GroupPairTotals:
    """Pairwise numerator and denominator totals for one ranked group."""

    correct: float
    count: int
    weighted_correct: float
    weight: float


def pair_correct_from_scores(
    higher_score: float,
    lower_score: float,
    *,
    allow_ties: bool = True,
) -> float:
    if higher_score > lower_score:
        return 1.0
    if allow_ties and math.isclose(higher_score, lower_score):
        return 0.5
    return 0.0


def evaluate_group_scores(
    groups: list[list[RankedImage]],
    scores_by_image: Mapping[str, float],
) -> tuple[float, float, float]:
    pair_score_total = 0.0
    pair_count = 0
    weighted_score_total = 0.0
    weight_total = 0.0
    top1_total = 0.0

    for group in groups:
        scores = {
            item.image_name: float(scores_by_image.get(item.image_name, 0.0))
            for item in group
        }

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item_i = group[i]
                item_j = group[j]

                if item_j.rank > item_i.rank:
                    lower = item_i
                    upper = item_j
                else:
                    lower = item_j
                    upper = item_i

                lower_score = scores[lower.image_name]
                upper_score = scores[upper.image_name]
                correct = pair_correct_from_scores(
                    upper_score,
                    lower_score,
                    allow_ties=True,
                )

                distance = upper.rank - lower.rank
                pair_score_total += correct
                pair_count += 1
                weighted_score_total += distance * correct
                weight_total += distance

        max_score = max(scores.values())
        tied_max = [item for item in group if math.isclose(scores[item.image_name], max_score)]
        true_top = max(group, key=lambda item: item.rank)
        if any(item.image_name == true_top.image_name for item in tied_max):
            top1_total += 1.0 / len(tied_max)

    return (
        pair_score_total / pair_count,
        weighted_score_total / weight_total,
        top1_total / len(groups),
    )


def metric_summary(metrics: list[MetricTriple]) -> dict[str, float]:
    return {
        "pair_acc_mean": mean(metric.pair_acc for metric in metrics),
        "pair_acc_std": pstdev(metric.pair_acc for metric in metrics),
        "weighted_pair_acc_mean": mean(metric.weighted_pair_acc for metric in metrics),
        "weighted_pair_acc_std": pstdev(
            metric.weighted_pair_acc for metric in metrics
        ),
        "top1_mean": mean(metric.top1 for metric in metrics),
        "top1_std": pstdev(metric.top1 for metric in metrics),
    }


def weighted_mean(values: list[float], weights: list[int]) -> float:
    denominator = sum(weights)
    if denominator == 0:
        raise ValueError("weighted_mean requires a non-zero total weight")
    numerator = sum(value * weight for value, weight in zip(values, weights, strict=True))
    return numerator / denominator


def safe_corr(
    fn: Callable[[np.ndarray, np.ndarray], Any],
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    result = fn(x, y)
    if hasattr(result, "statistic"):
        value = result.statistic
    elif isinstance(result, tuple):
        value = result[0]
    else:
        value = result
    if np.isnan(value):
        return 0.0
    return float(value)


def choose_top_name(
    group: list[RankedImage],
    *score_maps: Mapping[str, float],
) -> str:
    if not score_maps:
        raise ValueError("choose_top_name requires at least one score map")

    def ranking_key(item: RankedImage) -> tuple[float | str, ...]:
        score_key = tuple(-float(score_map[item.image_name]) for score_map in score_maps)
        return score_key + (item.image_name,)

    return min(group, key=ranking_key).image_name


def rank_group_by_scores(
    group: list[RankedImage],
    primary_scores: Mapping[str, float],
    *tie_break_scores: Mapping[str, float],
) -> dict[str, int]:
    """Produce deterministic ascending ranks from one or more score maps."""
    ordered = sorted(
        group,
        key=lambda item: tuple(
            float(scores[item.image_name])
            for scores in (primary_scores, *tie_break_scores)
        )
        + (item.image_name,),
    )
    return {item.image_name: rank for rank, item in enumerate(ordered)}


def evaluate_predicted_ranks(
    groups: list[list[RankedImage]],
    ranks_by_group: list[Mapping[str, int]],
    *,
    distance_thresholds: tuple[int, ...] = (2, 3, 4),
) -> tuple[dict[str, float], list[GroupPairTotals]]:
    """Evaluate deterministic predicted ranks against human ranks in each group."""
    pair_correct = 0.0
    pair_count = 0
    weighted_correct = 0.0
    total_weight = 0.0
    filtered = {distance: [0.0, 0.0] for distance in distance_thresholds}
    top_hits = 0
    totals: list[GroupPairTotals] = []
    for group, predicted in zip(groups, ranks_by_group, strict=True):
        local_correct = 0.0
        local_weighted_correct = 0.0
        local_weight = 0.0
        for i, item_i in enumerate(group):
            for item_j in group[i + 1 :]:
                lower, higher = (
                    (item_i, item_j)
                    if item_i.rank < item_j.rank
                    else (item_j, item_i)
                )
                correct = float(predicted[higher.image_name] > predicted[lower.image_name])
                weight = float(higher.rank - lower.rank)
                pair_correct += correct
                pair_count += 1
                weighted_correct += weight * correct
                total_weight += weight
                local_correct += correct
                local_weighted_correct += weight * correct
                local_weight += weight
                for minimum, values in filtered.items():
                    if weight >= minimum:
                        values[0] += weight * correct
                        values[1] += weight
        true_top = max(group, key=lambda item: item.rank).image_name
        top_hits += int(predicted[true_top] == len(group) - 1)
        totals.append(
            GroupPairTotals(
                local_correct,
                len(group) * (len(group) - 1) // 2,
                local_weighted_correct,
                local_weight,
            )
        )
    result = {
        "pair_acc": pair_correct / pair_count,
        "WPA": weighted_correct / total_weight,
        "top1_hard": top_hits / len(groups),
    }
    for minimum, (numerator, denominator) in filtered.items():
        result[f"d_ge_{minimum}_WPA"] = numerator / denominator
    return result, totals


def cluster_bootstrap_ratio_ci(
    group_values: list[tuple[float, float]],
    *,
    seed: int,
    n_bootstrap: int = 10000,
) -> list[float]:
    """Percentile CI for a ratio after resampling whole groups with replacement."""
    numerators = np.asarray([pair[0] for pair in group_values], dtype=np.float64)
    denominators = np.asarray([pair[1] for pair in group_values], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(group_values), size=(n_bootstrap, len(group_values)))
    estimates = numerators[samples].sum(axis=1) / denominators[samples].sum(axis=1)
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def pooled_ratio_metrics(
    sufficient_statistics: Mapping[str, tuple[np.ndarray, np.ndarray]],
    cluster_counts: np.ndarray,
) -> dict[str, float]:
    """Pool ratio numerators and denominators under cluster multiplicities."""
    counts = np.asarray(cluster_counts, dtype=np.int64)
    result: dict[str, float] = {}
    for metric, (numerator, denominator) in sufficient_statistics.items():
        pooled_denominator = float(counts @ denominator)
        if pooled_denominator == 0.0:
            raise ZeroDivisionError(f"Metric {metric!r} has zero pooled denominator.")
        result[metric] = float(counts @ numerator) / pooled_denominator
    return result


def pair_transition_metrics(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, int | float]:
    """Count paired correctness transitions and expose their accuracy identity."""
    reference = np.asarray(reference_correct, dtype=bool)
    candidate = np.asarray(candidate_correct, dtype=bool)
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate correctness arrays must have equal shape.")
    selected = np.ones(reference.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != reference.shape:
        raise ValueError("Transition mask must match correctness-array shape.")
    reference = reference[selected]
    candidate = candidate[selected]
    n_pairs = int(len(reference))
    correct_correct = int(np.sum(reference & candidate))
    recovered = int(np.sum(~reference & candidate))
    broken = int(np.sum(reference & ~candidate))
    wrong_wrong = int(np.sum(~reference & ~candidate))
    reference_correct_count = correct_correct + broken
    candidate_correct_count = correct_correct + recovered
    reference_wrong = recovered + wrong_wrong
    candidate_wrong = broken + wrong_wrong
    return {
        "n_pairs": n_pairs,
        "linear_correct_candidate_correct": correct_correct,
        "linear_wrong_candidate_correct": recovered,
        "linear_correct_candidate_wrong": broken,
        "linear_wrong_candidate_wrong": wrong_wrong,
        "recovered": recovered,
        "broken": broken,
        "net_corrected": recovered - broken,
        "LINEAR_correct": reference_correct_count,
        "CANDIDATE_correct": candidate_correct_count,
        "LINEAR_wrong": reference_wrong,
        "CANDIDATE_wrong": candidate_wrong,
        "LINEAR_pair_acc": (
            float(reference_correct_count) / n_pairs if n_pairs else float("nan")
        ),
        "CANDIDATE_pair_acc": (
            float(candidate_correct_count) / n_pairs if n_pairs else float("nan")
        ),
        "delta_pair_acc": (
            float(candidate_correct_count - reference_correct_count) / n_pairs
            if n_pairs
            else float("nan")
        ),
    }


def exact_multinomial_count_vectors(
    n_clusters: int,
    *,
    n_draws: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate multinomial count vectors and their integer multiplicities."""
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    draws = n_clusters if n_draws is None else n_draws
    if draws < 0:
        raise ValueError("n_draws must be non-negative.")
    values: list[tuple[int, ...]] = []
    multiplicities: list[int] = []
    factorial_draws = math.factorial(draws)
    for bars in combinations(range(draws + n_clusters - 1), n_clusters - 1):
        previous = -1
        counts: list[int] = []
        for bar in bars:
            counts.append(bar - previous - 1)
            previous = bar
        counts.append(draws + n_clusters - previous - 2)
        values.append(tuple(counts))
        denominator = math.prod(math.factorial(count) for count in counts)
        multiplicities.append(factorial_draws // denominator)
    return (
        np.asarray(values, dtype=np.int64),
        np.asarray(multiplicities, dtype=np.int64),
    )


def _exact_weighted_quantile(
    values: np.ndarray,
    multiplicities: np.ndarray,
    *,
    numerator: int,
    denominator: int,
    total_mass: int,
) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(multiplicities[order], dtype=np.int64)
    target = numerator * total_mass
    index = int(np.flatnonzero(denominator * cumulative >= target)[0])
    return float(values[order[index]])


def exact_ratio_difference_distribution(
    *,
    count_vectors: np.ndarray,
    multiplicities: np.ndarray,
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    threshold: tuple[int, int] | None = None,
    lower_quantile: tuple[int, int] = (1, 40),
    upper_quantile: tuple[int, int] = (39, 40),
) -> dict[str, Any]:
    """Exact weighted distribution of a difference between two pooled ratios."""
    counts = np.asarray(count_vectors, dtype=np.int64)
    weights = np.asarray(multiplicities, dtype=np.int64)
    first_num = counts @ first[0]
    first_den = counts @ first[1]
    second_num = counts @ second[0]
    second_den = counts @ second[1]
    if np.any(first_den == 0) or np.any(second_den == 0):
        raise ZeroDivisionError("An exact-bootstrap count vector has a zero denominator.")
    signed_num = first_num * second_den - second_num * first_den
    common_den = first_den * second_den
    values = (
        first_num.astype(np.float64) / first_den
        - second_num.astype(np.float64) / second_den
    )
    total_mass = int(np.sum(weights))

    def probability(mask: np.ndarray) -> float:
        return float(np.sum(weights[mask])) / total_mass

    result: dict[str, Any] = {
        "lower_95_CI": _exact_weighted_quantile(
            values,
            weights,
            numerator=lower_quantile[0],
            denominator=lower_quantile[1],
            total_mass=total_mass,
        ),
        "upper_95_CI": _exact_weighted_quantile(
            values,
            weights,
            numerator=upper_quantile[0],
            denominator=upper_quantile[1],
            total_mass=total_mass,
        ),
        "P_delta_lt_0": probability(signed_num < 0),
        "P_delta_eq_0": probability(signed_num == 0),
        "P_delta_gt_0": probability(signed_num > 0),
        "negative_mass_integer": int(np.sum(weights[signed_num < 0])),
        "zero_mass_integer": int(np.sum(weights[signed_num == 0])),
        "positive_mass_integer": int(np.sum(weights[signed_num > 0])),
        "probability_denominator": total_mass,
    }
    if threshold is not None:
        threshold_num, threshold_den = threshold
        below = threshold_den * signed_num < threshold_num * common_den
        result.update(
            {
                "required_threshold": float(threshold_num) / threshold_den,
                "P_delta_lt_required_threshold": probability(below),
                "P_delta_ge_required_threshold": probability(~below),
                "below_threshold_mass_integer": int(np.sum(weights[below])),
                "at_or_above_threshold_mass_integer": int(np.sum(weights[~below])),
            }
        )
    return result


def tie_aware_top1_contribution(
    group: list[RankedImage],
    primary_scores: Mapping[str, float],
) -> float:
    max_score = max(float(primary_scores[item.image_name]) for item in group)
    tied = [
        item.image_name
        for item in group
        if math.isclose(float(primary_scores[item.image_name]), max_score)
    ]
    true_top = max(group, key=lambda item: item.rank).image_name
    return 1.0 / len(tied) if true_top in tied else 0.0


def score_map_from_array(
    items: list[RankedImage],
    values: np.ndarray,
) -> dict[str, float]:
    return {
        item.image_name: float(value)
        for item, value in zip(items, values, strict=True)
    }


def compute_system_metrics(
    groups: list[list[RankedImage]],
    primary_scores: Mapping[str, float],
    tiebreak_scores: Mapping[str, float],
    *,
    allow_ties: bool = True,
) -> SystemMetrics:
    pair_correct_sum = 0.0
    pair_count = 0
    weighted_correct_sum = 0.0
    weight_total = 0.0
    hard_hits = 0
    tie_aware_total = 0.0

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item_i = group[i]
                item_j = group[j]
                if item_j.rank > item_i.rank:
                    lower = item_i
                    higher = item_j
                else:
                    lower = item_j
                    higher = item_i

                correct = pair_correct_from_scores(
                    primary_scores[higher.image_name],
                    primary_scores[lower.image_name],
                    allow_ties=allow_ties,
                )
                weight = float(higher.rank - lower.rank)
                pair_correct_sum += correct
                pair_count += 1
                weighted_correct_sum += weight * correct
                weight_total += weight

        pred_top = choose_top_name(group, primary_scores, tiebreak_scores)
        true_top = max(group, key=lambda item: item.rank).image_name
        hard_hits += int(pred_top == true_top)
        tie_aware_total += tie_aware_top1_contribution(group, primary_scores)

    return SystemMetrics(
        pair_acc=pair_correct_sum / pair_count,
        weighted_pair_acc=weighted_correct_sum / weight_total,
        top1_hard=hard_hits / len(groups),
        top1_tie_aware=tie_aware_total / len(groups),
    )


def compute_same_id_weighted_pair_acc(
    groups: list[list[RankedImage]],
    score_map: Mapping[str, float],
    *,
    allow_ties: bool = True,
) -> float:
    weighted_correct_sum = 0.0
    weight_total = 0.0

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item_i = group[i]
                item_j = group[j]
                if item_i.identity != item_j.identity:
                    continue
                if item_j.rank > item_i.rank:
                    lower = item_i
                    higher = item_j
                else:
                    lower = item_j
                    higher = item_i

                weight = float(higher.rank - lower.rank)
                correct = pair_correct_from_scores(
                    float(score_map[higher.image_name]),
                    float(score_map[lower.image_name]),
                    allow_ties=allow_ties,
                )
                weighted_correct_sum += weight * correct
                weight_total += weight

    if math.isclose(weight_total, 0.0):
        return float("nan")
    return weighted_correct_sum / weight_total


def compute_group_weighted_pair_acc(
    group: list[RankedImage],
    score_map: Mapping[str, float],
    *,
    allow_ties: bool = True,
) -> float:
    weighted_correct_sum = 0.0
    weight_total = 0.0

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            item_i = group[i]
            item_j = group[j]
            if item_j.rank > item_i.rank:
                lower = item_i
                higher = item_j
            else:
                lower = item_j
                higher = item_i

            weight = float(higher.rank - lower.rank)
            correct = pair_correct_from_scores(
                float(score_map[higher.image_name]),
                float(score_map[lower.image_name]),
                allow_ties=allow_ties,
            )
            weighted_correct_sum += weight * correct
            weight_total += weight

    if math.isclose(weight_total, 0.0):
        return float("nan")
    return weighted_correct_sum / weight_total


def compute_weighted_pair_acc_filtered(
    groups: list[list[RankedImage]],
    score_map: Mapping[str, float],
    *,
    allow_ties: bool = True,
    pair_filter: Callable[[RankedImage, RankedImage, float], bool] | None = None,
) -> float:
    weighted_correct_sum = 0.0
    weight_total = 0.0

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item_i = group[i]
                item_j = group[j]
                if item_j.rank > item_i.rank:
                    lower = item_i
                    higher = item_j
                else:
                    lower = item_j
                    higher = item_i

                weight = float(higher.rank - lower.rank)
                if pair_filter is not None and not pair_filter(higher, lower, weight):
                    continue

                correct = pair_correct_from_scores(
                    float(score_map[higher.image_name]),
                    float(score_map[lower.image_name]),
                    allow_ties=allow_ties,
                )
                weighted_correct_sum += weight * correct
                weight_total += weight

    if math.isclose(weight_total, 0.0):
        return float("nan")
    return weighted_correct_sum / weight_total
