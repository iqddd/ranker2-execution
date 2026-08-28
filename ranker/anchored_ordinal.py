"""Direct anchored ordinal metrics for external rank-group evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ranker.data import load_completed_ranked_groups_range, load_ranked_groups
from ranker.io import read_json


def comparison_credit(model_difference: float, human_difference: float) -> float:
    """Return 1/0 correctness, assigning half credit to an exact model tie."""
    if model_difference == 0.0:
        return 0.5
    return float(np.sign(model_difference) == np.sign(human_difference))


def anchored_image_metrics(
    *,
    new_human_value: float,
    anchor_human_values: Sequence[float],
    new_model_score: float,
    anchor_model_scores: Sequence[float],
) -> dict[str, float | int | bool]:
    """Evaluate one image against fixed anchors with anchor-distance weights."""
    human = np.asarray(anchor_human_values, dtype=np.float64)
    model = np.asarray(anchor_model_scores, dtype=np.float64)
    if human.ndim != 1 or len(human) == 0 or model.shape != human.shape:
        raise ValueError("Anchor human/model arrays must be equal nonempty vectors.")
    if not np.all(np.isfinite(human)) or not np.all(np.isfinite(model)):
        raise ValueError("Anchored ordinal metrics require finite inputs.")
    distance = 1 + np.asarray(
        [
            np.count_nonzero(
                (human > min(new_human_value, value))
                & (human < max(new_human_value, value))
            )
            for value in human
        ],
        dtype=np.int64,
    )
    credit = np.asarray(
        [
            comparison_credit(
                float(new_model_score - model[index]),
                float(new_human_value - human[index]),
            )
            for index in range(len(human))
        ],
        dtype=np.float64,
    )
    wrong = 1.0 - credit
    count = float(np.sum(wrong))
    return {
        "anchor_loss": float(np.sum(distance * wrong) / np.sum(distance)),
        "anchor_pair_accuracy": float(np.mean(credit)),
        "anchor_misplacement_count": count,
        "gross_anchor_misplacement": count >= 3,
        "weighted_denominator": int(np.sum(distance)),
    }


def evaluate_strict_anchor_campaign(
    rows: Iterable[Mapping[str, Any]],
    scores_by_system: Mapping[str, Mapping[str, float]],
    *,
    systems: tuple[str, ...],
    new_slot_types: frozenset[str] = frozenset({"new", "pseudo_new"}),
    image_order: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen strict 2-new/6-anchor identity-balanced endpoint."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        grouped[str(row["group_id"])].append(row)
    per_image: list[dict[str, Any]] = []
    for group_id in sorted(grouped, key=lambda value: (len(value), value)):
        group = grouped[group_id]
        if len(group) != 8:
            raise ValueError(f"Group {group_id} does not contain eight rows.")
        ranks = [int(row["rank"]) for row in group]
        if sorted(ranks) != list(range(8)):
            raise ValueError(
                f"Group {group_id} human ranks are not permutation 0...7."
            )
        new_rows = [
            row for row in group if str(row["slot_type"]) in new_slot_types
        ]
        anchors = [row for row in group if str(row["slot_type"]) == "anchor"]
        if len(new_rows) != 2 or len(anchors) != 6:
            raise ValueError(f"Group {group_id} composition is not 2 new + 6 anchors.")
        anchor_ranks = np.asarray(
            [int(row["rank"]) for row in anchors], dtype=np.int64
        )
        for new in new_rows:
            result: dict[str, Any] = {
                "group_id": group_id,
                "image_key": str(new["image_key"]),
                "identity_id": str(new["identity_id"]),
                "human_rank": int(new["rank"]),
            }
            for system in systems:
                panel = scores_by_system[system]
                metrics = anchored_image_metrics(
                    new_human_value=float(new["rank"]),
                    anchor_human_values=anchor_ranks,
                    new_model_score=float(panel[str(new["image_key"])]),
                    anchor_model_scores=np.asarray(
                        [panel[str(anchor["image_key"])] for anchor in anchors],
                        dtype=np.float64,
                    ),
                )
                result[f"{system}_anchor_loss"] = metrics["anchor_loss"]
                result[f"{system}_anchor_pair_accuracy"] = metrics[
                    "anchor_pair_accuracy"
                ]
            per_image.append(result)
    if image_order is not None:
        position = {key: index for index, key in enumerate(image_order)}
        per_image.sort(key=lambda row: position[row["image_key"]])
    identities = sorted(
        {str(row["identity_id"]) for row in per_image},
        key=lambda value: value.encode("utf-8"),
    )
    per_identity: list[dict[str, Any]] = []
    for identity in identities:
        images = [row for row in per_image if row["identity_id"] == identity]
        record: dict[str, Any] = {"identity_id": identity, "image_count": len(images)}
        for system in systems:
            record[f"{system}_anchor_loss"] = float(
                np.mean(
                    [row[f"{system}_anchor_loss"] for row in images],
                    dtype=np.float64,
                )
            )
            record[f"{system}_anchor_pair_accuracy"] = float(
                np.mean(
                    [row[f"{system}_anchor_pair_accuracy"] for row in images],
                    dtype=np.float64,
                )
            )
        per_identity.append(record)
    overall: dict[str, float] = {}
    for system in systems:
        overall[f"{system}_IB_ANCHOR_LOSS"] = float(
            np.mean(
                [row[f"{system}_anchor_loss"] for row in per_identity],
                dtype=np.float64,
            )
        )
        overall[f"{system}_IB_ANCHOR_PAIR_ACC"] = float(
            np.mean(
                [row[f"{system}_anchor_pair_accuracy"] for row in per_identity],
                dtype=np.float64,
            )
        )
    return {"overall": overall, "per_image": per_image, "per_identity": per_identity}


def percentile_identity_cluster_bootstrap(
    identity_deltas: np.ndarray,
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> tuple[np.ndarray, float, float]:
    """Compute the frozen percentile identity-cluster bootstrap interval."""
    values = np.asarray(identity_deltas, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Identity deltas must be a nonempty finite vector.")
    rng = np.random.Generator(np.random.PCG64(seed))
    draw = rng.integers(
        0,
        len(values),
        size=(resamples, len(values)),
        endpoint=False,
        dtype=np.int64,
    )
    distribution = np.mean(values[draw], axis=1, dtype=np.float64)
    tail = (1.0 - confidence) / 2.0
    lower = float(np.quantile(distribution, tail, method="linear"))
    upper = float(np.quantile(distribution, 1.0 - tail, method="linear"))
    return distribution, lower, upper


def partition_distinct_identity_images(
    identities: np.ndarray,
    image_ids: np.ndarray,
    width: int,
    *,
    salt: str,
    forbidden_pairs: set[frozenset[int]] | None = None,
) -> list[list[int]]:
    """Deterministically partition images into distinct-identity groups."""
    if len(image_ids) % width:
        raise ValueError("Pseudo-new population is not divisible by topology width.")
    forbidden = forbidden_pairs or set()
    for attempt in range(100):
        queues: dict[str, list[int]] = {}
        for identity in sorted(set(identities.tolist())):
            values = np.flatnonzero(identities == identity).tolist()
            values.sort(
                key=lambda index: hashlib.sha256(
                    f"{salt}|{attempt}|{image_ids[index]}".encode()
                ).hexdigest()
            )
            queues[identity] = values
        groups: list[list[int]] = []
        failed = False
        while any(queues.values()):
            selected: list[int] = []
            selected_identities: set[str] = set()
            candidates = [index for values in queues.values() for index in values]
            candidates.sort(
                key=lambda index: (
                    -len(queues[str(identities[index])]),
                    hashlib.sha256(
                        f"{salt}|{attempt}|{len(groups)}|{image_ids[index]}".encode()
                    ).hexdigest(),
                )
            )
            for index in candidates:
                identity = str(identities[index])
                if identity in selected_identities:
                    continue
                if any(frozenset((index, other)) in forbidden for other in selected):
                    continue
                selected.append(index)
                selected_identities.add(identity)
                if len(selected) == width:
                    break
            if len(selected) != width:
                failed = True
                break
            for index in selected:
                queues[str(identities[index])].remove(index)
            groups.append(selected)
        if not failed and sorted(
            index for group in groups for index in group
        ) == list(range(len(image_ids))):
            return groups
    raise RuntimeError(f"Could not construct topology partition width={width}.")


def load_retest_permutations(
    source_state: Path, retest_state: Path
) -> list[list[int]]:
    """Load the 15 empirical O-group test/retest rank permutations."""
    original = load_completed_ranked_groups_range(source_state, start=0, stop=65)
    retest = load_ranked_groups(retest_state)
    metadata = read_json(retest_state)["metadata"]["source_group_indices_zero_based"]
    permutations: list[list[int]] = []
    for retest_index, group in enumerate(retest):
        source = original[int(metadata[retest_index])]
        new_rank = {item.image_name: int(item.rank) for item in group}
        permutation = [
            new_rank[item.image_name]
            for item in sorted(source, key=lambda item: item.rank)
        ]
        if sorted(permutation) != list(range(8)):
            raise RuntimeError("An O retest row is not a rank permutation.")
        permutations.append(permutation)
    if len(permutations) != 15:
        raise RuntimeError("Expected exactly 15 O retest permutations.")
    return permutations


def synthetic_ordinal_groups(
    image_ids: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    percentile: np.ndarray,
    support: np.ndarray,
    *,
    anchor_targets: Sequence[float],
    salt: str,
) -> list[dict[str, Any]]:
    """Construct deterministic 2-new/6-anchor pseudo-groups."""
    pairs = partition_distinct_identity_images(
        identities, image_ids, 2, salt=salt
    )
    banks = {
        target: sorted(
            range(len(image_ids)),
            key=lambda index: (
                abs(float(percentile[index]) - target),
                -int(support[index]),
                str(image_ids[index]),
            ),
        )
        for target in anchor_targets
    }
    groups: list[dict[str, Any]] = []
    for group_index, new in enumerate(pairs):
        excluded = {str(identities[index]) for index in new}
        anchors: list[int] = []
        anchor_ids: set[str] = set()
        for target in anchor_targets:
            bank = banks[target]
            rotated = bank[group_index % len(image_ids) :] + bank[
                : group_index % len(image_ids)
            ]
            chosen = next(
                index
                for index in rotated
                if str(identities[index]) not in excluded | anchor_ids
                and index not in anchors
            )
            anchors.append(chosen)
            anchor_ids.add(str(identities[chosen]))
        members = [*new, *anchors]
        if len({str(identities[index]) for index in members}) != len(members):
            raise RuntimeError("Synthetic group identity distinctness failed.")
        order = sorted(
            range(len(members)),
            key=lambda local: (
                float(theta[members[local]]),
                str(image_ids[members[local]]),
            ),
        )
        ranks = np.empty(len(members), dtype=np.int64)
        for rank, local in enumerate(order):
            ranks[local] = rank
        groups.append(
            {"new": new, "anchors": anchors, "members": members, "ranks": ranks}
        )
    if sorted(index for group in groups for index in group["new"]) != list(
        range(len(image_ids))
    ):
        raise RuntimeError("Synthetic pseudo-new coverage failed.")
    return groups


def evaluate_projected_ordinal_groups(
    groups: list[dict[str, Any]],
    theta: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    identities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Evaluate baseline/challenger on deterministic pseudo-groups."""
    values: dict[str, dict[str, np.ndarray]] = {
        name: {
            key: np.empty(len(identities), dtype=np.float64)
            for key in ("loss", "accuracy", "gross", "outer")
        }
        for name in ("baseline", "challenger")
    }
    new_pair_correct = {"baseline": [], "challenger": []}
    for group in groups:
        anchors = np.asarray(group["anchors"], dtype=np.int64)
        for new_index in group["new"]:
            for name, score in (("baseline", baseline), ("challenger", challenger)):
                result = anchored_image_metrics(
                    new_human_value=float(theta[new_index]),
                    anchor_human_values=theta[anchors],
                    new_model_score=float(score[new_index]),
                    anchor_model_scores=score[anchors],
                )
                values[name]["loss"][new_index] = float(result["anchor_loss"])
                values[name]["accuracy"][new_index] = float(
                    result["anchor_pair_accuracy"]
                )
                values[name]["gross"][new_index] = float(
                    result["gross_anchor_misplacement"]
                )
                outer_credit = [
                    comparison_credit(
                        float(score[new_index] - score[anchors[position]]),
                        float(theta[new_index] - theta[anchors[position]]),
                    )
                    for position in (0, 5)
                ]
                values[name]["outer"][new_index] = float(
                    np.mean([1.0 - credit for credit in outer_credit])
                )
        left, right = group["new"]
        for name, score in (("baseline", baseline), ("challenger", challenger)):
            new_pair_correct[name].append(
                comparison_credit(
                    float(score[left] - score[right]),
                    float(theta[left] - theta[right]),
                )
            )
    identity_order = sorted(set(identities.tolist()))
    identity_rows: list[dict[str, Any]] = []
    for identity in identity_order:
        mask = identities == identity
        baseline_loss = float(np.mean(values["baseline"]["loss"][mask]))
        challenger_loss = float(np.mean(values["challenger"]["loss"][mask]))
        identity_rows.append(
            {
                "identity_id": identity,
                "image_count": int(np.count_nonzero(mask)),
                "baseline_anchor_loss": baseline_loss,
                "challenger_anchor_loss": challenger_loss,
                "delta_anchor": challenger_loss - baseline_loss,
                "winner": (
                    "challenger"
                    if challenger_loss < baseline_loss
                    else "baseline" if challenger_loss > baseline_loss else "tie"
                ),
                "noise_sign_retention_probability": "pending",
            }
        )

    def identity_balanced(name: str, key: str) -> float:
        return float(
            np.mean(
                [
                    np.mean(values[name][key][identities == identity])
                    for identity in identity_order
                ]
            )
        )

    baseline_loss = identity_balanced("baseline", "loss")
    challenger_loss = identity_balanced("challenger", "loss")
    projected = {
        "baseline_IB_ANCHOR_LOSS": baseline_loss,
        "challenger_IB_ANCHOR_LOSS": challenger_loss,
        "projected_delta_anchor": challenger_loss - baseline_loss,
        "improved_identities": sum(row["delta_anchor"] < 0 for row in identity_rows),
        "tied_identities": sum(row["delta_anchor"] == 0 for row in identity_rows),
        "worsened_identities": sum(row["delta_anchor"] > 0 for row in identity_rows),
        "baseline_IB_ANCHOR_PAIR_ACC": identity_balanced("baseline", "accuracy"),
        "challenger_IB_ANCHOR_PAIR_ACC": identity_balanced(
            "challenger", "accuracy"
        ),
        "baseline_IB_GROSS_ANCHOR_RATE": identity_balanced("baseline", "gross"),
        "challenger_IB_GROSS_ANCHOR_RATE": identity_balanced(
            "challenger", "gross"
        ),
        "baseline_IB_OUTER_ANCHOR_ERROR": identity_balanced("baseline", "outer"),
        "challenger_IB_OUTER_ANCHOR_ERROR": identity_balanced(
            "challenger", "outer"
        ),
        "baseline_new_versus_new_pair_accuracy_descriptive": float(
            np.mean(new_pair_correct["baseline"])
        ),
        "challenger_new_versus_new_pair_accuracy_descriptive": float(
            np.mean(new_pair_correct["challenger"])
        ),
        "development_projection_only": True,
    }
    delta = values["challenger"]["loss"] - values["baseline"]["loss"]
    return (
        projected,
        identity_rows,
        {
            "delta": delta,
            **{
                f"{name}_{key}": value
                for name, fields in values.items()
                for key, value in fields.items()
            },
        },
    )


def noisy_group_losses(
    group: dict[str, Any],
    permutation: list[int],
    theta: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one empirical rank permutation and recompute per-new-image losses."""
    ranks = np.asarray(
        [permutation[int(rank)] for rank in group["ranks"]], dtype=np.int64
    )
    anchors = group["anchors"]
    outputs = []
    for score in (baseline, challenger):
        losses = []
        for local_new, new_index in enumerate(group["new"]):
            anchor_ranks = ranks[2:]
            new_rank = int(ranks[local_new])
            distances = 1 + np.asarray(
                [
                    np.count_nonzero(
                        (anchor_ranks > min(new_rank, rank))
                        & (anchor_ranks < max(new_rank, rank))
                    )
                    for rank in anchor_ranks
                ]
            )
            credits = np.asarray(
                [
                    comparison_credit(
                        float(score[new_index] - score[anchor]),
                        float(new_rank - int(anchor_ranks[position])),
                    )
                    for position, anchor in enumerate(anchors)
                ]
            )
            losses.append(
                float(np.sum(distances * (1.0 - credits)) / np.sum(distances))
            )
        outputs.append(np.asarray(losses))
    return outputs[0], outputs[1]
