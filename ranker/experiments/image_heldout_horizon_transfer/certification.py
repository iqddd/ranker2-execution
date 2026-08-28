"""Pure endpoint-certification analytics shared by STEP60-family audits."""

from __future__ import annotations

import hashlib
import itertools
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

from ranker.experiments.layer_screen.analysis import exposure_loss_rows
from ranker.validation_resolution import empirical_sign_mde


def partition_anchor_exposures(
    exposures: Sequence[Mapping[str, Any]], image_fold: Mapping[str, int]
) -> dict[str, list[dict[str, Any]]]:
    """Split every target exposure into seen- and unseen-anchor partitions."""
    output = {"legacy_mixed": [], "seen_anchor": [], "strict_unseen_anchor": []}
    for source in exposures:
        target = str(source["image_key"])
        target_fold = int(image_fold[target])
        output["legacy_mixed"].append(dict(source))
        for endpoint, same in (("seen_anchor", False), ("strict_unseen_anchor", True)):
            keep = [
                index
                for index, anchor in enumerate(source["anchor_keys"])
                if (int(image_fold[str(anchor)]) == target_fold) is same
            ]
            if not keep:
                continue
            row = dict(source)
            for field in ("anchor_keys", "anchor_identities", "anchor_ranks", "distances"):
                row[field] = tuple(source[field][index] for index in keep)
            row["fold"] = target_fold
            output[endpoint].append(row)
    return output


def endpoint_coverage(
    rows: Sequence[Mapping[str, Any]], image_fold: Mapping[str, int]
) -> dict[str, Any]:
    images = {str(row["image_key"]) for row in rows}
    anchors = {str(anchor) for row in rows for anchor in row["anchor_keys"]}
    cells = {
        (str(row["identity_id"]), str(identity))
        for row in rows
        for identity in row["anchor_identities"]
    }
    return {
        "exposures": len(rows),
        "comparisons": int(sum(len(row["anchor_keys"]) for row in rows)),
        "target_images": len(images),
        "anchor_images": len(anchors),
        "identities": len({str(row["identity_id"]) for row in rows}),
        "directed_cells": len(cells),
        "fold_comparisons": {
            str(fold): int(
                sum(
                    len(row["anchor_keys"])
                    for row in rows
                    if int(image_fold[str(row["image_key"])]) == fold
                )
            )
            for fold in range(5)
        },
    }


def per_target_direct(
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    fold_maps: Sequence[Mapping[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exposure-balanced direct contributions, one row per target image."""
    evaluated = exposure_loss_rows(exposures, image_fold, fold_maps)
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in evaluated:
        key = (
            str(row["identity_id"]),
            str(row["image_key"]),
            int(image_fold[str(row["image_key"])]),
        )
        grouped[key].append(float(row["loss"]))
    keys = sorted(grouped, key=lambda key: (key[0].encode(), key[1].encode()))
    return (
        np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64),
        np.asarray([key[0] for key in keys]),
        np.asarray([key[1] for key in keys]),
        np.asarray([key[2] for key in keys], dtype=np.int64),
    )


def equal_identity_mean(values: np.ndarray, identities: np.ndarray) -> float:
    return float(
        np.mean(
            [np.mean(values[identities == identity]) for identity in sorted(set(identities), key=str.encode)]
        )
    )


def bootstrap_weight_matrix(
    identities: np.ndarray,
    *,
    resamples: int,
    seed: int,
    random_identity: bool,
) -> np.ndarray:
    """Build exact clustered target weights for fixed or random identity strata."""
    identity_order = sorted(set(identities.astype(str)), key=str.encode)
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = np.zeros((resamples, len(identities)), dtype=np.float32)
    if random_identity:
        identity_counts = rng.multinomial(
            len(identity_order), np.full(len(identity_order), 1.0 / len(identity_order)), size=resamples
        )
    else:
        identity_counts = np.ones((resamples, len(identity_order)), dtype=np.int64)
    for identity_index, identity in enumerate(identity_order):
        positions = np.flatnonzero(identities == identity)
        n_images = len(positions)
        multiplicity = identity_counts[:, identity_index]
        for count in np.unique(multiplicity):
            selected = np.flatnonzero(multiplicity == count)
            if count == 0:
                continue
            draws = rng.multinomial(
                int(count) * n_images,
                np.full(n_images, 1.0 / n_images),
                size=len(selected),
            )
            weights[np.ix_(selected, positions)] = draws / (len(identity_order) * n_images)
    return weights


def uq_from_image_contrasts(
    contrasts: np.ndarray,
    identities: np.ndarray,
    folds: np.ndarray,
    *,
    resamples: int = 100_000,
    seed: int = 20260817,
    fixed_weights: np.ndarray | None = None,
    random_weights: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Four preregistered UQ views for [27 seed triples, target images]."""
    if contrasts.shape[0] != 27 or contrasts.shape[1] != len(identities):
        raise ValueError("Expected [27, target_images] contrasts.")
    point_index = list(itertools.product(range(3), repeat=3)).index((0, 1, 2))
    point = equal_identity_mean(contrasts[point_index], identities)
    rows: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for name, random_identity in (
        ("fixed_known_identity_target_bootstrap", False),
        ("random_identity_nested_target", True),
    ):
        supplied = random_weights if random_identity else fixed_weights
        weights = (
            bootstrap_weight_matrix(
                identities,
                resamples=resamples,
                seed=seed,
                random_identity=random_identity,
            )
            if supplied is None
            else np.asarray(supplied)
        )
        distribution = (contrasts @ weights.T).reshape(-1)
        digests[name] = hashlib.sha256(distribution.tobytes()).hexdigest()
        rows.append(_uq_row(name, point, distribution, combined=len(distribution)))
        del distribution
    seed_only = np.asarray(
        [equal_identity_mean(row, identities) for row in contrasts], dtype=np.float64
    )
    digests["seed_only"] = hashlib.sha256(seed_only.tobytes()).hexdigest()
    rows.append(_uq_row("seed_only", point, seed_only, combined=27))
    lofo = np.asarray(
        [
            equal_identity_mean(contrasts[point_index][folds != fold], identities[folds != fold])
            for fold in range(5)
        ],
        dtype=np.float64,
    )
    digests["five_fold_leave_one_fold_out"] = hashlib.sha256(lofo.tobytes()).hexdigest()
    rows.append(_uq_row("five_fold_leave_one_fold_out", point, lofo, combined=5))
    return rows, digests


def _uq_row(name: str, point: float, distribution: np.ndarray, *, combined: int) -> dict[str, Any]:
    values = np.asarray(distribution, dtype=np.float64)
    return {
        "uq_semantics": name,
        "point_delta": point,
        "bootstrap_mean": float(np.mean(values)),
        "SD": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "CI90_lower": float(np.quantile(values, 0.05, method="linear")),
        "CI90_upper": float(np.quantile(values, 0.95, method="linear")),
        "P_delta_lt_0": float(np.mean(values < 0)),
        "negative_sign90_MDE": float(empirical_sign_mde(values, -1)),
        "combined_evaluations": combined,
    }


def fit_constrained_reml(
    values: np.ndarray,
    identities: np.ndarray,
    images: np.ndarray,
    folds: np.ndarray,
    seeds: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit the preregistered crossed constrained-REML variance model."""
    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=np.float64),
            "identity": identities.astype(str),
            "image": images.astype(str),
            "fold": folds.astype(str),
            "seed": seeds.astype(str),
        }
    )
    frame["fold_seed"] = frame["fold"] + ":" + frame["seed"]
    frame["all"] = "all"
    vc = {
        "identity": "0 + C(identity)",
        "target_image": "0 + C(image)",
        "fold": "0 + C(fold)",
        "seed": "0 + C(seed)",
        "fold_seed": "0 + C(fold_seed)",
    }
    caught: list[str] = []
    result = None
    method_used = "lbfgs"
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        model = MixedLM.from_formula(
            "value ~ 1", groups="all", re_formula="0", vc_formula=vc, data=frame, use_sparse=True
        )
        result = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
        caught.extend(str(item.message) for item in observed)
    if not bool(result.converged):
        method_used = "powell"
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            result = model.fit(reml=True, method="powell", maxiter=2000, disp=False)
            caught.extend(str(item.message) for item in observed)
    names = list(model.exog_vc.names)
    estimates = dict(zip(names, np.asarray(result.vcomp, dtype=np.float64), strict=True))
    estimates["residual"] = float(result.scale)
    total = float(sum(estimates.values()))
    threshold = max(1e-15, total * 1e-8)
    rows = [
        {
            "component": name,
            "variance": 0.0 if value <= threshold else float(value),
            "raw_variance": float(value),
            "share": float(max(value, 0.0) / total) if total > 0 else np.nan,
            "boundary": bool(value <= threshold),
        }
        for name, value in estimates.items()
    ]
    covariance_available = True
    hessian_available = True
    try:
        covariance_available = bool(np.all(np.isfinite(np.asarray(result.cov_params()))))
    except Exception:  # noqa: BLE001 - availability is itself the diagnostic
        covariance_available = False
    try:
        hessian = model.hessian(result.params)[0]
        hessian_available = bool(np.all(np.isfinite(np.asarray(hessian))))
    except Exception:  # noqa: BLE001 - availability is itself the diagnostic
        hessian_available = False
    audit = {
        "converged": bool(result.converged),
        "optimizer": method_used,
        "log_likelihood": float(result.llf),
        "warnings": caught,
        "singular_or_boundary_components": [row["component"] for row in rows if row["boundary"]],
        "hessian_available": hessian_available,
        "covariance_available": covariance_available,
    }
    return rows, audit


def variance_routing_and_projections(
    components: Mapping[str, float],
    identities: np.ndarray,
    folds: np.ndarray,
    *,
    current_mde: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = max(float(components.get("identity", 0.0)), 0.0)
    image = max(float(components.get("target_image", 0.0)), 0.0)
    denominator = identity + image
    ratio = float(identity / denominator) if denominator > 0 else np.nan
    route = (
        "new_identities_prioritized"
        if ratio >= 0.50
        else "new_images_of_known_identities_prioritized"
        if ratio <= 0.20
        else "no_data_source_priority"
    )
    identity_order = sorted(set(identities.astype(str)), key=str.encode)
    image_weights = np.zeros(len(identities), dtype=np.float64)
    for identity_name in identity_order:
        selected = identities == identity_name
        image_weights[selected] = 1.0 / (len(identity_order) * int(np.sum(selected)))
    fold_weights = np.asarray(
        [np.sum(image_weights[folds == fold]) for fold in sorted(set(folds))],
        dtype=np.float64,
    )
    contributions = {
        "identity": identity / len(identity_order),
        "target_image": image * float(np.sum(image_weights**2)),
        "fold": max(float(components.get("fold", 0.0)), 0.0) * float(np.sum(fold_weights**2)),
        "seed": max(float(components.get("seed", 0.0)), 0.0) / 3.0,
        "fold_seed": max(float(components.get("fold_seed", 0.0)), 0.0) * float(np.sum(fold_weights**2)) / 3.0,
        "residual": max(float(components.get("residual", 0.0)), 0.0) * float(np.sum(image_weights**2)) / 3.0,
    }
    total = sum(contributions.values())
    scenarios = {
        "double_target_images": {"target_image": 0.5, "residual": 0.5},
        "28_identities": {"identity": 0.5, "target_image": 0.5, "residual": 0.5},
        "E6": {"seed": 0.5, "fold_seed": 0.5, "residual": 0.5},
        "two_independent_image_partitions": {"fold": 0.5, "fold_seed": 0.5, "residual": 0.5},
    }
    projections = []
    for scenario, scaling in scenarios.items():
        projected = sum(value * scaling.get(name, 1.0) for name, value in contributions.items())
        projections.append(
            {
                "scenario": scenario,
                "projected_sign90_MDE": float(current_mde * np.sqrt(projected / total)) if total > 0 else np.nan,
                "model_extrapolation": scenario == "two_independent_image_partitions",
            }
        )
    return {"R_id": ratio, "routing": route, "estimator_variance_contributions": contributions}, projections
