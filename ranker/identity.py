from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from .data import RankedImage


def compute_identity_priors(
    train_groups: list[list[RankedImage]],
    lambda_shrinkage: float,
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    zs_by_identity: dict[str, list[float]] = defaultdict(list)
    for group in train_groups:
        for item in group:
            zs_by_identity[item.identity].append(item.z)

    priors: dict[str, float] = {}
    counts: dict[str, int] = {}
    means: dict[str, float] = {}
    for identity, zs in zs_by_identity.items():
        n_train = len(zs)
        mean_z = mean(zs)
        shrink = n_train / (n_train + lambda_shrinkage)
        priors[identity] = shrink * mean_z
        counts[identity] = n_train
        means[identity] = mean_z
    return priors, counts, means


def summarize_identity_priors(
    all_fold_priors: list[dict[str, float]],
    all_fold_counts: list[dict[str, int]],
) -> list[dict[str, Any]]:
    identities = sorted({identity for priors in all_fold_priors for identity in priors})
    rows: list[dict[str, Any]] = []
    for identity in identities:
        prior_values = [priors.get(identity, 0.0) for priors in all_fold_priors]
        count_values = [counts.get(identity, 0) for counts in all_fold_counts]
        rows.append(
            {
                "identity": identity,
                "n_train_avg": mean(count_values),
                "b_id_mean": mean(prior_values),
                "b_id_std_across_folds": pstdev(prior_values),
            }
        )
    return rows

