from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .data import RankedImage
from .embeddings import collect_unique_image_names
from .metrics import PairMetricAccumulator, accumulate_ranked_pair, pair_correct_from_scores
from .strata import StratumAccumulator


IDENTITY_ORDER = (
    "AbiRat",
    "AlinaRose",
    "Cynthia",
    "DenMil",
    "DiaMel",
    "DmRs",
    "JaiOje",
    "JorCar",
    "LynPer",
    "Sabina",
    "TifTow",
    "WenFio",
    "ashleytervort",
    "sharishanya",
)

def validate_dataset_identities(groups: list[list[RankedImage]]) -> None:
    discovered_identities = sorted(
        {item.identity for group in groups for item in group}
    )
    expected_identities = sorted(IDENTITY_ORDER)
    if discovered_identities != expected_identities:
        raise ValueError(
            "Dataset identities changed. "
            f"Expected {expected_identities}, got {discovered_identities}."
        )


def split_groups_by_heldout_identities(
    groups: list[list[RankedImage]],
    heldout_identities: Iterable[str],
) -> tuple[list[list[RankedImage]], list[list[RankedImage]]]:
    heldout_set = set(heldout_identities)
    train_groups: list[list[RankedImage]] = []
    val_groups: list[list[RankedImage]] = []
    for group in groups:
        identities = {item.identity for item in group}
        if identities & heldout_set:
            val_groups.append(group)
        else:
            train_groups.append(group)
    return train_groups, val_groups


def build_standardized_embeddings(
    train_groups: list[list[RankedImage]],
    val_groups: list[list[RankedImage]],
    embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    from .embeddings import standardize_feature_map
    return standardize_feature_map(embeddings, train_groups, val_groups)


def validate_within_identity_groups(
    groups: list[list[RankedImage]], *, expected_identities: Iterable[str] = IDENTITY_ORDER,
    groups_per_identity: int = 4, group_size: int = 8,
) -> dict[str, list[list[RankedImage]]]:
    by_identity: dict[str, list[list[RankedImage]]] = {}
    for group in groups:
        identities = {item.identity for item in group}
        if len(group) != group_size or len(identities) != 1:
            raise ValueError("Within-identity validation requires single-identity octets.")
        by_identity.setdefault(group[0].identity, []).append(group)
    if set(by_identity) != set(expected_identities) or any(len(value) != groups_per_identity for value in by_identity.values()):
        raise ValueError("Expected exactly four within-identity groups for every identity.")
    return by_identity


def evaluate_heldout_score_map(
    groups: list[list[RankedImage]], scores: dict[str, float], heldout_identities: Iterable[str] | str
) -> dict[str, dict[str, float]]:
    heldout = {heldout_identities} if isinstance(heldout_identities, str) else set(heldout_identities)
    names = ("all_val", "heldout_involved", "heldout_vs_seen", "both_heldout", "heldout_cross", "heldout_same", "no_heldout")
    accumulators = {name: PairMetricAccumulator() for name in names}
    top_hits = 0
    for group in groups:
        true_top = max(group, key=lambda item: item.rank).image_name
        predicted_top = min(group, key=lambda item: (-float(scores[item.image_name]), item.image_name)).image_name
        top_hits += int(true_top == predicted_top)
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                lower, higher = (first, second) if first.rank < second.rank else (second, first)
                endpoints = {lower.identity, higher.identity}
                strata = ["all_val"]
                if endpoints & heldout:
                    strata.append("heldout_involved")
                    if endpoints <= heldout:
                        strata.extend(("both_heldout", "heldout_cross" if len(endpoints) == 2 else "heldout_same"))
                    else:
                        strata.append("heldout_vs_seen")
                else:
                    strata.append("no_heldout")
                for name in strata:
                    accumulate_ranked_pair(accumulators[name], higher, lower, scores)
    result = {name: {"n_pairs": float(accumulator.pair_count), **accumulator.as_metrics()} for name, accumulator in accumulators.items()}
    if len(heldout) == 1:
        # Legacy LOIO artifacts used this spelling; retain it while exposing
        # the canonical heldout_same stratum as well.
        result["heldout_same_id"] = dict(result["heldout_same"])
    result["all_val"]["top1_hard"] = top_hits / len(groups)
    return result


def assert_disjoint_group_images(train_groups: list[list[RankedImage]], validation_groups: list[list[RankedImage]]) -> None:
    train_names = {item.image_name for group in train_groups for item in group}
    validation_names = {item.image_name for group in validation_groups for item in group}
    if train_names & validation_names:
        raise ValueError("Train and validation images overlap.")


def split_nested_heldout_groups(
    groups: list[list[RankedImage]], outer_heldout: Iterable[str], inner_heldout: str,
) -> tuple[list[list[RankedImage]], list[list[RankedImage]]]:
    outer_train, _ = split_groups_by_heldout_identities(groups, outer_heldout)
    inner_train, inner_validation = split_groups_by_heldout_identities(outer_train, [inner_heldout])
    assert_disjoint_group_images(inner_train, inner_validation)
    if any(item.identity in set(outer_heldout) for group in [*inner_train, *inner_validation] for item in group):
        raise ValueError("Outer held-out identity leaked into inner split.")
    return inner_train, inner_validation


def build_score_map(
    items: list[RankedImage],
    standardized_embeddings: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, float]:
    matrix = np.stack(
        [standardized_embeddings[item.image_name] for item in items],
        axis=0,
    )
    raw_scores = matrix @ weights
    return {
        item.image_name: float(score)
        for item, score in zip(items, raw_scores, strict=True)
    }


def compute_top1_hard(
    groups: list[list[RankedImage]],
    scores: dict[str, float],
) -> float:
    hits = 0
    for group in groups:
        predicted_order = sorted(
            group,
            key=lambda item: (float(scores[item.image_name]), item.image_name),
        )
        predicted_top = predicted_order[-1].image_name
        true_top = max(group, key=lambda item: item.rank).image_name
        hits += int(predicted_top == true_top)
    return hits / len(groups)


def initialize_accumulators(
    strata_order: list[str],
) -> dict[str, StratumAccumulator]:
    return {name: StratumAccumulator() for name in strata_order}


def accumulate_validation_pairs(
    groups: list[list[RankedImage]],
    scores: dict[str, float],
    strata_order: list[str],
    assign_strata: Callable[[RankedImage, RankedImage], list[str]],
) -> dict[str, StratumAccumulator]:
    accumulators = initialize_accumulators(strata_order)

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
                correct = pair_correct_from_scores(
                    float(scores[higher.image_name]),
                    float(scores[lower.image_name]),
                    allow_ties=True,
                )
                for stratum in assign_strata(higher, lower):
                    accumulators[stratum].add(correct=correct, weight=weight)

    return accumulators


def metrics_from_accumulator(accumulator: StratumAccumulator) -> dict[str, Any]:
    metrics = accumulator.as_metrics()
    return {
        "n_pairs": int(accumulator.n_pairs),
        "sum_weight": float(accumulator.sum_weight),
        "pair_acc": float(metrics["pair_acc"]),
        "weighted_pair_acc": float(metrics["weighted_pair_acc"]),
    }
