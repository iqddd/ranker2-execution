"""Pure multi-system analytics for counter-prior transfer diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ranker.experiments.counter_prior_response.analysis import (
    aggregate_crossings,
    crossing_cell_rows,
    crossing_metric_matrix,
    panel_crossing_observations,
    scale_panels,
    uq_summary,
)
from ranker.frozen_readouts import TrainingFold


def full_data_raw_panel(
    raw_scores: np.ndarray,
    checkpoints: Sequence[int],
    seeds: Sequence[int],
    identities: Sequence[str],
    canonical_indices: np.ndarray,
) -> dict[int, dict[tuple[str, int], np.ndarray]]:
    """Replicate one full-data raw vector across target-fold roles."""
    values = np.asarray(raw_scores, dtype=np.float64)
    checkpoint_values = np.asarray(checkpoints, dtype=np.int64)
    indices = np.asarray(canonical_indices, dtype=np.int64)
    if values.shape[:2] != (len(seeds), len(checkpoints)):
        raise ValueError("Unexpected full-data score panel grid.")
    result: dict[int, dict[tuple[str, int], np.ndarray]] = {}
    for checkpoint in (112, 192, 256):
        matches = np.flatnonzero(checkpoint_values == checkpoint)
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate checkpoint {checkpoint}.")
        position = int(matches[0])
        result[checkpoint] = {
            (identity, int(seed)): values[seed_index, position, indices].copy()
            for identity in identities
            for seed_index, seed in enumerate(seeds)
        }
    return result


def tail_identity_vectors(
    human: np.ndarray,
    model: np.ndarray,
    identities: np.ndarray,
    identity_order: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return per-identity positive/negative inward shortfall and asymmetry."""
    human_values = np.asarray(human, dtype=np.float64)
    model_values = np.asarray(model, dtype=np.float64)
    identity_values = np.asarray(identities).astype(str)
    human_centers = np.asarray([
        np.mean(human_values[identity_values == identity]) for identity in identity_values
    ])
    model_centers = np.asarray([
        np.mean(model_values[identity_values == identity]) for identity in identity_values
    ])
    x = human_values - human_centers
    y = model_values - model_centers
    plus = np.full(len(identity_order), np.nan, dtype=np.float64)
    minus = np.full(len(identity_order), np.nan, dtype=np.float64)
    for index, identity in enumerate(identity_order):
        own = identity_values == identity
        positive = own & (x >= 20.0)
        negative = own & (x <= -20.0)
        if np.any(positive):
            plus[index] = float(np.mean(x[positive] - y[positive]))
        if np.any(negative):
            minus[index] = float(np.mean(y[negative] - x[negative]))
    return {"C_plus": plus, "C_minus": minus, "A": plus - minus}


def weighted_identity_macro(vector: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Bootstrap an equal-identity vector while preserving empty-cell coverage."""
    values = np.asarray(vector, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    valid = np.isfinite(values)
    numerator = weights[:, valid] @ values[valid]
    denominator = weights[:, valid].sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(weights), np.nan),
        where=denominator > 0,
    )


def system_panels(
    *,
    raw: Mapping[tuple[str, int], np.ndarray],
    image_ids: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    theta_global: np.ndarray,
    seeds: Sequence[int],
    folds: Mapping[str, TrainingFold],
    triple: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    """Build one system's exact fold-local percentile-first panels."""
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
    return panels


def crossing_rows_for_panels(
    *,
    system: str,
    scale: str,
    panels: Mapping[str, Mapping[str, np.ndarray]],
    identities: np.ndarray,
    labels: np.ndarray,
    strata: Sequence[str],
) -> tuple[list[dict[str, Any]], list[Mapping[str, np.ndarray]]]:
    """Return compact cells and raw in-memory observations for one system."""
    observations = panel_crossing_observations(panels, identities, scale, labels)
    rows = [
        {"system": system, **row}
        for observation in observations
        for row in crossing_cell_rows(
            scale=scale,
            target_identity=str(observation["target_identity"][0]),
            obs=observation,
            strata=strata,
        )
    ]
    return rows, observations


def crossing_metric_matrices(
    rows: Sequence[Mapping[str, Any]],
    identity_order: Sequence[str],
) -> dict[str, np.ndarray]:
    """Lay the three primary crossing estimands out as compact matrices."""
    return {
        field: crossing_metric_matrix(rows, identity_order, field)
        for field in ("recovered_rate", "preserved_rate", "net_normalized")
    }


def hierarchical_bootstrap_stack(
    matrices: np.ndarray,
    counts: np.ndarray,
    *,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Bootstrap many target-by-anchor matrices with one shared draw panel."""
    values = np.asarray(matrices, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (weights.shape[1], weights.shape[1]):
        raise ValueError("Expected Q x identity x identity matrices.")
    finite = np.isfinite(values)
    if not np.all(finite == finite[:1]):
        raise ValueError("Stacked matrices must share human-defined finite-cell coverage.")
    result = np.empty((len(weights), len(values)), dtype=np.float64)
    identity_count = weights.shape[1]
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights))
        chunk = weights[start:stop]
        target_values = np.full((len(chunk), len(values), identity_count), np.nan)
        for target in range(identity_count):
            valid = finite[0, target]
            denominator = chunk[:, valid].sum(axis=1)
            numerator = chunk[:, valid] @ values[:, target, valid].T
            target_values[:, :, target] = np.divide(
                numerator,
                denominator[:, None],
                out=np.full_like(numerator, np.nan),
                where=denominator[:, None] > 0,
            )
        valid_targets = np.isfinite(target_values)
        numerator = np.nansum(target_values * chunk[:, None, :], axis=2)
        denominator = np.sum(chunk[:, None, :] * valid_targets, axis=2)
        result[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
    return result


def crossing_point(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return primary crossing point metrics from compact directed cells."""
    aggregate = aggregate_crossings(rows, aggregation="primary_hierarchical")
    return {
        "recovered_rate": float(aggregate["recovered_rate"]),
        "preserved_rate": float(aggregate["preserved_rate"]),
        "net_normalized": float(aggregate["net"]),
    }


def paired_uq_row(
    *,
    estimand: str,
    scale: str,
    stratum: str,
    system_or_contrast: str,
    direction: str,
    point: float,
    distribution: np.ndarray,
) -> dict[str, Any]:
    """Format one common paired-UQ record."""
    return {
        "estimand": estimand,
        "scale": scale,
        "stratum": stratum,
        "system_or_contrast": system_or_contrast,
        "direction": direction,
        **uq_summary(distribution, point),
    }
