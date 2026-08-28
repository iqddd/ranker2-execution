"""Pure numerical routines for physical counter-prior response audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ranker.frozen_readouts import TrainingFold, midrank_percentile
from ranker.level_evaluation import ensemble_percentile_fold_result


def _finite_float(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Non-finite diagnostic value.")
    return result


def build_response_bins(
    image_ids: np.ndarray, human_deviation: np.ndarray, bins: int = 5
) -> np.ndarray:
    """Assign deterministic equal-count directional bins before model inspection."""
    ids = np.asarray(image_ids).astype(str)
    x = np.asarray(human_deviation, dtype=np.float64)
    labels = np.full(len(ids), "zero", dtype="U32")
    for direction, selected in (("negative", x < 0.0), ("positive", x > 0.0)):
        indices = np.flatnonzero(selected)
        order = indices[np.lexsort((ids[indices], x[indices]))]
        for number, chunk in enumerate(np.array_split(order, bins), start=1):
            labels[chunk] = f"{direction}_q{number}"
    return labels


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": _finite_float(np.mean(values)),
        "median": _finite_float(np.median(values)),
        "SD": _finite_float(np.std(values, ddof=0)),
        "P10": _finite_float(np.percentile(values, 10)),
        "P25": _finite_float(np.percentile(values, 25)),
        "P75": _finite_float(np.percentile(values, 75)),
        "P90": _finite_float(np.percentile(values, 90)),
    }


def _finite_nested_mean(values: np.ndarray) -> float:
    row_means = [
        float(np.mean(row[np.isfinite(row)]))
        for row in values
        if np.any(np.isfinite(row))
    ]
    return float(np.mean(row_means)) if row_means else np.nan


def response_curve_rows(
    *,
    scale: str,
    image_ids: np.ndarray,
    identities: np.ndarray,
    human_deviation: np.ndarray,
    model_deviation: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Describe the model response in fixed bins and overlapping strong tails."""
    ids = np.asarray(image_ids).astype(str)
    identity = np.asarray(identities).astype(str)
    x = np.asarray(human_deviation, dtype=np.float64)
    y = np.asarray(model_deviation, dtype=np.float64)
    strata: list[tuple[str, np.ndarray]] = [
        (name, labels == name)
        for name in [
            *(f"negative_q{i}" for i in range(1, 6)),
            *(f"positive_q{i}" for i in range(1, 6)),
        ]
    ]
    strata.extend((("negative_tail20", x <= -20.0), ("positive_tail20", x >= 20.0)))
    rows: list[dict[str, Any]] = []
    for name, selected in strata:
        if not np.any(selected):
            continue
        ys = y[selected]
        xs = x[selected]
        stats = _summary(ys)
        rows.append(
            {
                "scale": scale,
                "stratum": name,
                "count": int(np.count_nonzero(selected)),
                "identity_coverage": len(set(identity[selected].tolist())),
                "x_min": float(np.min(xs)),
                "x_max": float(np.max(xs)),
                "x_mean": float(np.mean(xs)),
                "x_median": float(np.median(xs)),
                **{f"y_{key}": value for key, value in stats.items()},
                "mean_y_minus_x": float(np.mean(ys - xs)),
                "median_y_minus_x": float(np.median(ys - xs)),
                "image_id_first": str(ids[selected][0]),
            }
        )
    return rows


def _level_row(identity: str, human: np.ndarray, model: np.ndarray) -> dict[str, Any]:
    human = np.asarray(human, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    x = human - human.mean()
    y = model - model.mean()
    denominator = float(np.sum(x * x))
    strong = np.abs(x) >= 20.0
    correct = np.sign(y[strong]) == np.sign(x[strong])
    sigma_human = float(np.std(x, ddof=0))
    sigma_model = float(np.std(y, ddof=0))
    frame_error = np.abs(y - x)
    tail_count = int(np.ceil(0.10 * len(frame_error)))
    tail = np.sort(frame_error, kind="mergesort")[-tail_count:]
    return {
        "aggregation": identity,
        "image_count": len(human),
        "beta": float(np.sum(x * y) / denominator),
        "Pearson": float(pearsonr(x, y).statistic),
        "Spearman": float(spearmanr(x, y).statistic),
        "sigma_model": sigma_model,
        "sigma_human": sigma_human,
        "sigma_ratio": sigma_model / sigma_human,
        "strong_count": int(np.count_nonzero(strong)),
        "strong_deviation_sign_accuracy": (
            float(np.mean(correct)) if np.any(strong) else 1.0
        ),
        "wrong_side_rate": float(np.mean(~correct)) if np.any(strong) else 0.0,
        "IB_FrameLTE10": float(np.mean(tail)),
        "IdentityOffsetMAE": float(abs(model.mean() - human.mean())),
    }


def level_decomposition_rows(
    identities: np.ndarray, human: np.ndarray, model: np.ndarray
) -> list[dict[str, Any]]:
    """Return per-identity, equal-identity macro, and pooled descriptions."""
    identity = np.asarray(identities).astype(str)
    per_identity = [
        _level_row(name, human[identity == name], model[identity == name])
        for name in sorted(set(identity.tolist()))
    ]
    numeric = [
        "beta",
        "Pearson",
        "Spearman",
        "sigma_model",
        "sigma_human",
        "sigma_ratio",
        "strong_deviation_sign_accuracy",
        "wrong_side_rate",
        "IB_FrameLTE10",
        "IdentityOffsetMAE",
    ]
    macro = {
        "aggregation": "equal_identity_macro",
        "image_count": len(identity),
        "strong_count": int(sum(row["strong_count"] for row in per_identity)),
        **{
            name: float(np.mean([row[name] for row in per_identity]))
            for name in numeric
        },
    }
    pooled = _level_row("pooled_descriptive", human, model)
    return [*per_identity, macro, pooled]


def crossing_observations(
    *,
    target_identity: str,
    identities: np.ndarray,
    human: np.ndarray,
    model: np.ndarray,
    human_centers: Mapping[str, float],
    response_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct one target-fold directed deployment universe in RAM."""
    identity = np.asarray(identities).astype(str)
    targets = np.flatnonzero(identity == target_identity)
    anchors = np.flatnonzero(identity != target_identity)
    ti = np.repeat(targets, len(anchors))
    aj = np.tile(anchors, len(targets))
    anchor_identity = identity[aj]
    human_sign = np.sign(human[ti] - human[aj]).astype(np.int8)
    prior_sign = np.asarray(
        [
            np.sign(human_centers[target_identity] - human_centers[name])
            for name in anchor_identity
        ],
        dtype=np.int8,
    )
    model_sign = np.sign(model[ti] - model[aj]).astype(np.int8)
    if np.any(human_sign == 0) or np.any(prior_sign == 0):
        raise ValueError("Human/prior tie in directed crossing universe.")
    return {
        "target_identity": np.full(len(ti), target_identity, dtype="U64"),
        "target_index": ti,
        "anchor_index": aj,
        "anchor_identity": anchor_identity,
        "human_sign": human_sign,
        "prior_sign": prior_sign,
        "model_sign": model_sign,
        "prior_wrong": human_sign != prior_sign,
        "model_correct_credit": np.where(
            model_sign == human_sign, 1.0, np.where(model_sign == 0, 0.5, 0.0)
        ),
        "model_tie": model_sign == 0,
        "response_label": response_labels[ti],
        "target_deviation": np.asarray(
            human - np.asarray([human_centers[name] for name in identity]),
            dtype=np.float64,
        )[ti],
        "theta_gap20": np.abs(human[ti] - human[aj]) >= 20.0,
    }


def _stratum_mask(obs: Mapping[str, np.ndarray], stratum: str) -> np.ndarray:
    base, _, coarse = stratum.partition("|")
    selected = np.ones(len(obs["target_index"]), dtype=bool)
    if base != "all":
        if base == "negative_tail20":
            selected &= obs["target_deviation"] <= -20.0
        elif base == "positive_tail20":
            selected &= obs["target_deviation"] >= 20.0
        else:
            selected &= obs["response_label"] == base
    if coarse == "THETA_GAP20":
        selected &= obs["theta_gap20"]
    return selected


def crossing_cell_rows(
    *,
    scale: str,
    target_identity: str,
    obs: Mapping[str, np.ndarray],
    strata: Sequence[str],
) -> list[dict[str, Any]]:
    """Compact directed-identity cell totals; never materializes a disk pair table."""
    rows: list[dict[str, Any]] = []
    for stratum in strata:
        stratum_mask = _stratum_mask(obs, stratum)
        for anchor in sorted(set(obs["anchor_identity"].astype(str).tolist())):
            selected = stratum_mask & (obs["anchor_identity"] == anchor)
            if not np.any(selected):
                continue
            wrong = selected & obs["prior_wrong"]
            correct = selected & ~obs["prior_wrong"]
            credit = obs["model_correct_credit"]
            recovered = float(np.sum(credit[wrong]))
            preserved = float(np.sum(credit[correct]))
            wrong_n = int(np.count_nonzero(wrong))
            correct_n = int(np.count_nonzero(correct))
            rows.append(
                {
                    "scale": scale,
                    "stratum": stratum,
                    "target_identity": target_identity,
                    "anchor_identity": anchor,
                    "eligible_pairs": int(np.count_nonzero(selected)),
                    "eligible_target_images": len(
                        set(obs["target_index"][selected].tolist())
                    ),
                    "prior_wrong": wrong_n,
                    "prior_correct": correct_n,
                    "recovered_credit": recovered,
                    "missed_credit": wrong_n - recovered,
                    "preserved_credit": preserved,
                    "broken_credit": correct_n - preserved,
                    "recovered_rate": recovered / wrong_n if wrong_n else np.nan,
                    "preserved_rate": preserved / correct_n if correct_n else np.nan,
                    "net_normalized": (recovered - (correct_n - preserved))
                    / int(np.count_nonzero(selected)),
                    "model_ties": int(np.count_nonzero(selected & obs["model_tie"])),
                }
            )
    return rows


def aggregate_crossing_observations(
    observations: Sequence[Mapping[str, np.ndarray]],
    *,
    stratum: str,
    aggregation: str,
) -> dict[str, Any]:
    """Aggregate raw in-memory observations, including equal-target-image macro."""
    if aggregation != "equal_target_image_macro":
        rows = [
            row
            for obs in observations
            for row in crossing_cell_rows(
                scale="internal",
                target_identity=str(obs["target_identity"][0]),
                obs=obs,
                strata=(stratum,),
            )
        ]
        return aggregate_crossings(rows, aggregation=aggregation)
    target_metrics: list[dict[str, float]] = []
    totals = {
        name: 0.0
        for name in ("eligible", "wrong", "correct", "recovered", "preserved", "ties")
    }
    for obs in observations:
        selected = _stratum_mask(obs, stratum)
        if not np.any(selected):
            continue
        target_indices = np.unique(obs["target_index"][selected])
        anchor_names = sorted(
            set(obs["anchor_identity"][selected].astype(str).tolist())
        )
        target_code = np.searchsorted(target_indices, obs["target_index"])
        anchor_code = np.searchsorted(
            np.asarray(anchor_names), obs["anchor_identity"].astype(str)
        )
        group = target_code * len(anchor_names) + anchor_code
        group_count = len(target_indices) * len(anchor_names)
        wrong_mask = selected & obs["prior_wrong"]
        correct_mask = selected & ~obs["prior_wrong"]
        eligible_count = np.bincount(group[selected], minlength=group_count).astype(
            float
        )
        wrong_count = np.bincount(group[wrong_mask], minlength=group_count).astype(
            float
        )
        correct_count = np.bincount(group[correct_mask], minlength=group_count).astype(
            float
        )
        recovered = np.bincount(
            group[wrong_mask],
            weights=obs["model_correct_credit"][wrong_mask],
            minlength=group_count,
        )
        preserved = np.bincount(
            group[correct_mask],
            weights=obs["model_correct_credit"][correct_mask],
            minlength=group_count,
        )
        recovery = np.divide(
            recovered,
            wrong_count,
            out=np.full(group_count, np.nan),
            where=wrong_count > 0,
        ).reshape(len(target_indices), -1)
        preservation = np.divide(
            preserved,
            correct_count,
            out=np.full(group_count, np.nan),
            where=correct_count > 0,
        ).reshape(len(target_indices), -1)
        net = np.divide(
            recovered - (correct_count - preserved),
            eligible_count,
            out=np.full(group_count, np.nan),
            where=eligible_count > 0,
        ).reshape(len(target_indices), -1)
        target_metrics.append(
            {
                "recovery": _finite_nested_mean(recovery),
                "preservation": _finite_nested_mean(preservation),
                "net": _finite_nested_mean(net),
            }
        )
        totals["eligible"] += float(np.sum(selected))
        totals["wrong"] += float(np.sum(wrong_mask))
        totals["correct"] += float(np.sum(correct_mask))
        totals["recovered"] += float(np.sum(obs["model_correct_credit"][wrong_mask]))
        totals["preserved"] += float(np.sum(obs["model_correct_credit"][correct_mask]))
        totals["ties"] += float(np.sum(selected & obs["model_tie"]))
    broken = totals["correct"] - totals["preserved"]
    return {
        "aggregation": aggregation,
        "eligible_pairs": int(totals["eligible"]),
        "eligible_images": int(
            sum(
                len(set(obs["target_index"][_stratum_mask(obs, stratum)].tolist()))
                for obs in observations
            )
        ),
        "directed_identity_cells": int(
            sum(
                len(set(obs["anchor_identity"][_stratum_mask(obs, stratum)].tolist()))
                for obs in observations
            )
        ),
        "prior_wrong": int(totals["wrong"]),
        "prior_correct": int(totals["correct"]),
        "recovered_credit": totals["recovered"],
        "missed_credit": totals["wrong"] - totals["recovered"],
        "recovered_rate": float(
            np.nanmean([row["recovery"] for row in target_metrics])
        ),
        "preserved_credit": totals["preserved"],
        "broken_credit": broken,
        "preserved_rate": float(
            np.nanmean([row["preservation"] for row in target_metrics])
        ),
        "net": float(np.nanmean([row["net"] for row in target_metrics])),
        "model_ties": int(totals["ties"]),
    }


def _macro(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    target_values: list[float] = []
    for target in sorted({str(row["target_identity"]) for row in rows}):
        values = [
            float(row[field])
            for row in rows
            if row["target_identity"] == target and np.isfinite(float(row[field]))
        ]
        if values:
            target_values.append(float(np.mean(values)))
    return float(np.mean(target_values))


def _eligible_image_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count each represented target image once rather than once per anchor cell."""
    return sum(
        max(
            int(row["eligible_target_images"])
            for row in rows
            if row["target_identity"] == target
        )
        for target in sorted({str(row["target_identity"]) for row in rows})
    )


def aggregate_crossings(
    rows: Sequence[Mapping[str, Any]], *, aggregation: str
) -> dict[str, Any]:
    """Aggregate compact cells using macro or pair-micro semantics."""
    if not rows:
        raise ValueError("Cannot aggregate empty crossing rows.")
    if aggregation == "pair_micro":
        wrong = sum(int(row["prior_wrong"]) for row in rows)
        correct = sum(int(row["prior_correct"]) for row in rows)
        eligible = wrong + correct
        recovered = sum(float(row["recovered_credit"]) for row in rows)
        preserved = sum(float(row["preserved_credit"]) for row in rows)
        broken = correct - preserved
        recovery = recovered / wrong
        preservation = preserved / correct
        net = recovered - broken
    else:
        recovery = _macro(rows, "recovered_rate")
        preservation = _macro(rows, "preserved_rate")
        net = _macro(rows, "net_normalized")
        eligible = sum(int(row["eligible_pairs"]) for row in rows)
        wrong = sum(int(row["prior_wrong"]) for row in rows)
        correct = sum(int(row["prior_correct"]) for row in rows)
        recovered = sum(float(row["recovered_credit"]) for row in rows)
        preserved = sum(float(row["preserved_credit"]) for row in rows)
        broken = correct - preserved
    return {
        "aggregation": aggregation,
        "eligible_pairs": eligible,
        "eligible_images": _eligible_image_count(rows),
        "directed_identity_cells": len(rows),
        "prior_wrong": wrong,
        "prior_correct": correct,
        "recovered_credit": recovered,
        "missed_credit": wrong - recovered,
        "recovered_rate": recovery,
        "preserved_credit": preserved,
        "broken_credit": broken,
        "preserved_rate": preservation,
        "net": net,
        "model_ties": sum(int(row["model_ties"]) for row in rows),
    }


def bootstrap_counts(replicates: int, identities: int, seed: int) -> np.ndarray:
    """Deterministic paired identity-bootstrap multiplicities."""
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, identities, size=(replicates, identities))
    return np.stack(
        [(draws == index).sum(axis=1) for index in range(identities)], axis=1
    ).astype(np.int16)


def hierarchical_bootstrap(matrix: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Recompute target->anchor equal-weight macro under paired multiplicities."""
    values = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    n = values.shape[0]
    target_values = np.full((len(weights), n), np.nan, dtype=np.float64)
    for target in range(n):
        valid = np.isfinite(values[target])
        denominator = weights[:, valid].sum(axis=1)
        numerator = weights[:, valid] @ values[target, valid]
        target_values[:, target] = np.divide(
            numerator,
            denominator,
            out=np.full(len(weights), np.nan),
            where=denominator > 0,
        )
    valid = np.isfinite(target_values)
    numerator = np.nansum(target_values * weights, axis=1)
    denominator = np.sum(weights * valid, axis=1)
    return np.divide(
        numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0
    )


def scale_panels(
    *,
    image_ids: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    theta_global: np.ndarray,
    raw: Mapping[tuple[str, int], np.ndarray],
    seeds: Sequence[int],
    folds: Mapping[str, TrainingFold],
    triple: Sequence[int] = (0, 1, 2),
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    """Build fold-local percentile panels and historical level diagnostics."""
    identity_order = sorted(set(np.asarray(identities).astype(str).tolist()))
    output: dict[str, dict[str, np.ndarray]] = {}
    fold_results: list[dict[str, Any]] = []
    for identity in identity_order:
        reference = identities != identity
        seed_percentiles = [
            midrank_percentile(
                raw[(identity, int(seeds[index]))],
                raw[(identity, int(seeds[index]))][reference],
            )
            for index in triple
        ]
        model = np.mean(np.stack(seed_percentiles), axis=0)
        human_fold = midrank_percentile(theta, theta[reference])
        output[identity] = {
            "model": model,
            "fold_local": human_fold,
            "global": theta_global,
        }
        if tuple(triple) == (0, 1, 2):
            fold_results.append(
                ensemble_percentile_fold_result(
                    image_ids,
                    identities,
                    theta,
                    folds[identity],
                    [raw[(identity, seed)] for seed in seeds],
                )
            )
    return output, fold_results


def stitched_panel(
    scale_panels: Mapping[str, Mapping[str, np.ndarray]],
    identities: np.ndarray,
    scale: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch each held-out fold's target values into one canonical vector."""
    model = np.empty(len(identities), dtype=np.float64)
    human = np.empty(len(identities), dtype=np.float64)
    for identity, values in scale_panels.items():
        selected = identities == identity
        model[selected] = values["model"][selected]
        human[selected] = values[scale][selected]
    return human, model


def identity_centers(values: np.ndarray, identities: np.ndarray) -> dict[str, float]:
    """Compute one descriptive human-score center per identity."""
    return {
        identity: float(np.mean(values[identities == identity]))
        for identity in sorted(set(np.asarray(identities).astype(str).tolist()))
    }


def panel_crossing_observations(
    scale_panels: Mapping[str, Mapping[str, np.ndarray]],
    identities: np.ndarray,
    scale: str,
    labels: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    """Construct all target-fold crossing observations for a stitched scale."""
    observations = []
    for identity in sorted(scale_panels):
        human = scale_panels[identity][scale]
        observations.append(
            crossing_observations(
                target_identity=identity,
                identities=identities,
                human=human,
                model=scale_panels[identity]["model"],
                human_centers=identity_centers(human, identities),
                response_labels=labels,
            )
        )
    return observations


def crossing_metric_matrix(
    rows: Sequence[Mapping[str, Any]], identities: Sequence[str], field: str
) -> np.ndarray:
    """Lay compact directed crossing rows out as a target-by-anchor matrix."""
    index = {name: number for number, name in enumerate(identities)}
    values = np.full((len(identities), len(identities)), np.nan)
    for row in rows:
        values[index[str(row["target_identity"])], index[str(row["anchor_identity"])]] = float(
            row[field]
        )
    return values


def uq_summary(values: np.ndarray, point: float) -> dict[str, float]:
    """Summarize a finite bootstrap distribution with its fixed point estimate."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "point": float(point),
        "bootstrap_mean": float(np.mean(finite)),
        "bootstrap_SD": float(np.std(finite, ddof=0)),
        "CI90_lower": float(np.percentile(finite, 5)),
        "CI90_upper": float(np.percentile(finite, 95)),
        "P_lt_0": float(np.mean(finite < 0.0)),
        "P_gt_0": float(np.mean(finite > 0.0)),
        "combined_evaluations": len(finite),
    }


def crossing_uq_rows(
    *,
    raw: Mapping[tuple[str, int], np.ndarray],
    image_ids: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    theta_global: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    triples: Sequence[tuple[int, int, int]],
    identity_order: Sequence[str],
    seeds: Sequence[int],
    folds: Mapping[str, TrainingFold],
) -> list[dict[str, Any]]:
    """Run hierarchical UQ for the primary all and THETA_GAP20 crossings."""
    distributions: dict[tuple[str, str], list[np.ndarray]] = {
        (stratum, metric): []
        for stratum in ("all", "all|THETA_GAP20")
        for metric in ("recovered_rate", "preserved_rate", "net_normalized")
    }
    points: dict[tuple[str, str], float] = {}
    for triple in triples:
        panels, _ = scale_panels(
            image_ids=image_ids,
            identities=identities,
            theta=theta,
            theta_global=theta_global,
            raw=raw,
            seeds=seeds,
            folds=folds,
            triple=triple,
        )
        observations = panel_crossing_observations(
            panels, identities, "fold_local", labels
        )
        for stratum in ("all", "all|THETA_GAP20"):
            rows = [
                row
                for observation in observations
                for row in crossing_cell_rows(
                    scale="fold_local",
                    target_identity=str(observation["target_identity"][0]),
                    obs=observation,
                    strata=(stratum,),
                )
            ]
            for metric in ("recovered_rate", "preserved_rate", "net_normalized"):
                matrix = crossing_metric_matrix(rows, identity_order, metric)
                distributions[(stratum, metric)].append(
                    hierarchical_bootstrap(matrix, counts)
                )
                if tuple(triple) == (0, 1, 2):
                    points[(stratum, metric)] = float(
                        aggregate_crossings(
                            rows, aggregation="primary_hierarchical"
                        )[
                            {
                                "recovered_rate": "recovered_rate",
                                "preserved_rate": "preserved_rate",
                                "net_normalized": "net",
                            }[metric]
                        ]
                    )
    return [
        {
            "estimand": f"crossing_{metric}",
            "stratum": stratum,
            **uq_summary(np.concatenate(parts), points[(stratum, metric)]),
        }
        for (stratum, metric), parts in distributions.items()
    ]


def response_uq_rows(
    *,
    raw: Mapping[tuple[str, int], np.ndarray],
    image_ids: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    theta_global: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    triples: Sequence[tuple[int, int, int]],
    identity_order: Sequence[str],
    base_strata: Sequence[str],
    seeds: Sequence[int],
    folds: Mapping[str, TrainingFold],
) -> list[dict[str, Any]]:
    """Run hierarchical UQ for identity-macro response-curve means."""
    index = {name: number for number, name in enumerate(identity_order)}
    distributions: dict[str, list[np.ndarray]] = {
        name: [] for name in base_strata[1:]
    }
    points: dict[str, float] = {}
    for triple in triples:
        panels, _ = scale_panels(
            image_ids=image_ids,
            identities=identities,
            theta=theta,
            theta_global=theta_global,
            raw=raw,
            seeds=seeds,
            folds=folds,
            triple=triple,
        )
        human, model = stitched_panel(panels, identities, "fold_local")
        model_deviation = model - np.asarray(
            [np.mean(model[identities == name]) for name in identities]
        )
        human_deviation = human - np.asarray(
            [np.mean(human[identities == name]) for name in identities]
        )
        masks = {name: labels == name for name in base_strata[1:-2]}
        masks["negative_tail20"] = human_deviation <= -20.0
        masks["positive_tail20"] = human_deviation >= 20.0
        for name, selected in masks.items():
            vector = np.full(len(identity_order), np.nan)
            for identity in identity_order:
                chosen = selected & (identities == identity)
                if np.any(chosen):
                    vector[index[identity]] = np.mean(model_deviation[chosen])
            valid = np.isfinite(vector)
            numerator = counts[:, valid] @ vector[valid]
            denominator = counts[:, valid].sum(axis=1)
            distribution = np.divide(
                numerator,
                denominator,
                out=np.full(len(counts), np.nan),
                where=denominator > 0,
            )
            distributions[name].append(distribution)
            if tuple(triple) == (0, 1, 2):
                points[name] = float(np.mean(vector[valid]))
    return [
        {
            "estimand": "response_mean_y",
            "stratum": name,
            **uq_summary(np.concatenate(parts), points[name]),
        }
        for name, parts in distributions.items()
    ]
