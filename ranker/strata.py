from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .data import RankedImage
from .metrics import pair_correct_from_scores


DEFAULT_EPS = 1e-12


@dataclass
class StratumAccumulator:
    n_pairs: int = 0
    sum_weight: float = 0.0
    pair_correct_sum: float = 0.0
    weighted_correct_sum: float = 0.0

    def add(self, *, correct: float, weight: float) -> None:
        self.n_pairs += 1
        self.sum_weight += weight
        self.pair_correct_sum += correct
        self.weighted_correct_sum += weight * correct

    def as_metrics(self) -> dict[str, float]:
        if self.n_pairs == 0 or math.isclose(self.sum_weight, 0.0):
            return {
                "pair_acc": float("nan"),
                "weighted_pair_acc": float("nan"),
            }
        return {
            "pair_acc": self.pair_correct_sum / self.n_pairs,
            "weighted_pair_acc": self.weighted_correct_sum / self.sum_weight,
        }


@dataclass(frozen=True)
class PairScoreSpec:
    score_map: Mapping[str, float]
    allow_ties: bool = True


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def initialize_strata_accumulators(
    strata_order: Sequence[str],
) -> dict[str, StratumAccumulator]:
    return {stratum: StratumAccumulator() for stratum in strata_order}


def initialize_system_strata_accumulators(
    system_names: Sequence[str],
    strata_order: Sequence[str],
) -> dict[str, dict[str, StratumAccumulator]]:
    return {
        system_name: initialize_strata_accumulators(strata_order)
        for system_name in system_names
    }


def derive_pair_strata(
    higher_item: RankedImage,
    lower_item: RankedImage,
    identity_scores: Mapping[str, float],
    *,
    eps: float = DEFAULT_EPS,
) -> list[str]:
    if higher_item.identity == lower_item.identity:
        return ["same_id"]

    higher_name = higher_item.image_name
    lower_name = lower_item.image_name
    delta_b = float(identity_scores[higher_name]) - float(identity_scores[lower_name])
    abs_delta_b = abs(delta_b)

    if math.isclose(delta_b, 0.0, abs_tol=eps):
        directional_stratum = "cross_tie"
    elif delta_b > 0.0:
        directional_stratum = "cross_aligned"
    else:
        directional_stratum = "cross_counter"

    if abs_delta_b <= 0.10 + eps:
        gap_stratum = "weak_prior"
    elif abs_delta_b <= 0.30 + eps:
        gap_stratum = "mid_prior"
    else:
        gap_stratum = "strong_prior"

    return ["cross_id", directional_stratum, gap_stratum]


def add_pair_result(
    accumulators: dict[str, StratumAccumulator],
    stratum: str,
    *,
    weight: float,
    score_map: Mapping[str, float],
    higher_name: str,
    lower_name: str,
    allow_ties: bool = True,
) -> None:
    accumulators[stratum].add(
        correct=pair_correct_from_scores(
            float(score_map[higher_name]),
            float(score_map[lower_name]),
            allow_ties=allow_ties,
        ),
        weight=weight,
    )


def add_pair_result_many(
    accumulators: dict[str, StratumAccumulator],
    strata: Sequence[str],
    *,
    weight: float,
    score_map: Mapping[str, float],
    higher_name: str,
    lower_name: str,
    allow_ties: bool = True,
) -> None:
    for stratum in strata:
        add_pair_result(
            accumulators,
            stratum,
            weight=weight,
            score_map=score_map,
            higher_name=higher_name,
            lower_name=lower_name,
            allow_ties=allow_ties,
        )


def add_multi_system_pair_results(
    accumulators_by_system: dict[str, dict[str, StratumAccumulator]],
    strata: Sequence[str],
    *,
    system_specs: Mapping[str, PairScoreSpec],
    weight: float,
    higher_name: str,
    lower_name: str,
) -> None:
    for system_name, spec in system_specs.items():
        add_pair_result_many(
            accumulators_by_system[system_name],
            strata,
            weight=weight,
            score_map=spec.score_map,
            higher_name=higher_name,
            lower_name=lower_name,
            allow_ties=spec.allow_ties,
        )


def accumulator_sizes(
    accumulators: Mapping[str, StratumAccumulator],
) -> dict[str, dict[str, float]]:
    return {
        stratum: {
            "n_pairs": accumulator.n_pairs,
            "sum_weight": accumulator.sum_weight,
        }
        for stratum, accumulator in accumulators.items()
    }


def accumulator_metrics(
    accumulators: Mapping[str, StratumAccumulator],
) -> dict[str, dict[str, float]]:
    return {
        stratum: accumulator.as_metrics()
        for stratum, accumulator in accumulators.items()
    }
