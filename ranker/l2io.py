"""Strict L2IO split and evaluation primitives preserving the Step-28 contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ranker.data import RankedImage
from ranker.heldout_ranknet import (
    evaluate_heldout_score_map,
    split_groups_by_heldout_identities,
)


@dataclass(frozen=True)
class L2IOSplit:
    outer_index: int
    heldout: tuple[str, str]
    train_o: tuple[tuple[RankedImage, ...], ...]
    evaluation_o: tuple[tuple[RankedImage, ...], ...]
    train_w: tuple[tuple[RankedImage, ...], ...]
    evaluation_w: tuple[tuple[RankedImage, ...], ...]


def build_strict_l2io_split(
    *,
    outer_index: int,
    heldout: tuple[str, str],
    original_groups: Sequence[Sequence[RankedImage]],
    within_by_identity: Mapping[str, Sequence[Sequence[RankedImage]]],
) -> L2IOSplit:
    """Apply the historical Step-28 full-group O/W identity removal."""
    train_o, evaluation_o = split_groups_by_heldout_identities(
        [list(group) for group in original_groups], heldout
    )
    heldout_set = set(heldout)
    train_w = [
        list(group)
        for identity, groups in within_by_identity.items()
        if identity not in heldout_set
        for group in groups
    ]
    evaluation_w = [
        list(group) for identity in heldout for group in within_by_identity[identity]
    ]
    return L2IOSplit(
        outer_index=outer_index,
        heldout=heldout,
        train_o=tuple(map(tuple, train_o)),
        evaluation_o=tuple(map(tuple, evaluation_o)),
        train_w=tuple(map(tuple, train_w)),
        evaluation_w=tuple(map(tuple, evaluation_w)),
    )


def l2io_pair_records(
    *,
    split: L2IOSplit,
    base_scores: Mapping[str, float],
    head_scores: Mapping[str, float],
    diagnostic_identity_prior: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Build O-evaluation records with exact historical heldout predicates."""
    heldout = set(split.heldout)
    records: list[dict[str, Any]] = []
    for group in split.evaluation_o:
        ordered = sorted(group, key=lambda item: item.rank)
        for low_index, lower in enumerate(ordered):
            for high_index in range(low_index + 1, len(ordered)):
                higher = ordered[high_index]
                endpoints = {lower.identity, higher.identity}
                both = endpoints <= heldout
                heldout_cross = both and len(endpoints) == 2
                heldout_same = both and len(endpoints) == 1
                involved = bool(endpoints & heldout)
                if lower.identity == higher.identity:
                    relation = "same_identity"
                else:
                    prior_delta = float(
                        diagnostic_identity_prior.get(higher.identity, 0.0)
                    ) - float(diagnostic_identity_prior.get(lower.identity, 0.0))
                    relation = (
                        "aligned"
                        if prior_delta > 0.0
                        else "counter"
                        if prior_delta < 0.0
                        else "tied_cross"
                    )
                base_margin = float(base_scores[higher.image_name]) - float(
                    base_scores[lower.image_name]
                )
                head_margin = float(head_scores[higher.image_name]) - float(
                    head_scores[lower.image_name]
                )
                if both:
                    pair_type = (
                        "both_heldout_same" if heldout_same else "both_heldout_cross"
                    )
                else:
                    pair_type = "one_heldout" if involved else "neither_heldout"
                records.append(
                    {
                        "outer_index": split.outer_index,
                        "group_id": int(lower.group_index),
                        "group_size": len(ordered),
                        "lower_rank": int(lower.rank),
                        "higher_rank": int(higher.rank),
                        "distance": int(higher.rank - lower.rank),
                        "heldout_involved": involved,
                        "both_heldout": both,
                        "heldout_cross": heldout_cross,
                        "heldout_same": heldout_same,
                        "pair_type": pair_type,
                        "prior_relation": relation,
                        "lower_image_id": lower.image_name,
                        "higher_image_id": higher.image_name,
                        "lower_identity_id": lower.identity,
                        "higher_identity_id": higher.identity,
                        "base_signed_margin": base_margin,
                        "head_signed_margin": head_margin,
                        "residual_margin": head_margin - base_margin,
                        "base_correct": bool(base_margin > 0.0),
                        "head_correct": bool(head_margin > 0.0),
                    }
                )
    return records


def step28_predicate_count_parity(split: L2IOSplit) -> dict[str, Any]:
    """Compare record predicates to the historical Step-28 evaluation helper."""
    names = {
        item.image_name
        for group in split.evaluation_o
        for item in group
    }
    dummy = {name: float(index) for index, name in enumerate(sorted(names))}
    historical = evaluate_heldout_score_map(
        [list(group) for group in split.evaluation_o], dummy, split.heldout
    )
    heldout = set(split.heldout)
    observed = {
        "all_val": 0,
        "heldout_involved": 0,
        "both_heldout": 0,
        "heldout_cross": 0,
        "heldout_same": 0,
    }
    for group in split.evaluation_o:
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                endpoints = {first.identity, second.identity}
                observed["all_val"] += 1
                if endpoints & heldout:
                    observed["heldout_involved"] += 1
                    if endpoints <= heldout:
                        observed["both_heldout"] += 1
                        observed[
                            "heldout_cross" if len(endpoints) == 2 else "heldout_same"
                        ] += 1
    differences = {
        name: observed[name] - int(historical[name]["n_pairs"])
        for name in observed
    }
    maximum = max(map(abs, differences.values()), default=0)
    return {
        "observed": observed,
        "historical": {name: int(historical[name]["n_pairs"]) for name in observed},
        "differences": differences,
        "max_abs_predicate_count_difference": maximum,
        "evaluation_predicates_equal_historical_step28": maximum == 0,
    }


def pair_record_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Convert L2IO pair records to stable arrays for metric aggregation."""
    fields = {
        "outer_index": np.int16,
        "group_id": np.int16,
        "group_size": np.int8,
        "lower_rank": np.int8,
        "higher_rank": np.int8,
        "distance": np.int8,
        "heldout_involved": bool,
        "both_heldout": bool,
        "heldout_cross": bool,
        "heldout_same": bool,
        "prior_relation": str,
        "base_correct": bool,
        "head_correct": bool,
    }
    return {
        name: np.asarray([row[name] for row in records], dtype=dtype)
        for name, dtype in fields.items()
    }
