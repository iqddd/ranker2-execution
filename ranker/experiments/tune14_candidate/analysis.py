"""Pure common analysis for matched TUNE14 candidate steps."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ranker.experiments.branch_mixing.analysis import endpoint_vectors
from ranker.level_evaluation import (
    aggregate_level_fold_results,
    ensemble_percentile_fold_result,
)
from ranker.tune14 import paired_uq, percentile_maps
from ranker.validation_resolution import direct_identity_loss_vector


def checkpoint_surface(
    panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    *,
    system: str,
    checkpoints: Sequence[int],
    identities: Sequence[str],
    names: np.ndarray,
    identity_ids: np.ndarray,
    panels: Mapping[str, Sequence[Mapping[str, Any]]],
    diagnostics: Callable[[Mapping[str, int]], float],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Build the common fixed-checkpoint TUNE14 surface and selection vectors."""
    combined = panels["COMBINED"]
    checkpoint_identity: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_map = {identity: int(checkpoint) for identity in identities}
        maps = percentile_maps(
            panel,
            system,
            checkpoint_map,
            (0, 1, 2),
            identities,
            names,
            identity_ids,
        )
        for panel_name, exposures in panels.items():
            values = direct_identity_loss_vector(exposures, maps, identities)
            rows.append(
                {
                    "record_type": "aggregate",
                    "checkpoint": checkpoint,
                    "panel": panel_name,
                    "direct_loss": float(values.mean()),
                    "G": "",
                    "gross_d3": "",
                    "IdentityOffsetMAE": "",
                }
            )
            if panel_name == "COMBINED":
                checkpoint_identity[int(checkpoint)] = values
        endpoint = endpoint_vectors(maps, combined, identities)
        rows.append(
            {
                "record_type": "combined_diagnostics",
                "checkpoint": checkpoint,
                "panel": "COMBINED",
                "direct_loss": endpoint["direct_loss"],
                "G": endpoint["G"],
                "gross_d3": endpoint["gross_loss"],
                "IdentityOffsetMAE": float(diagnostics(checkpoint_map)),
            }
        )
    return rows, checkpoint_identity


def evaluate_candidate(
    panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    *,
    system: str,
    checkpoint_map: Mapping[str, int],
    triple: Sequence[int],
    identities: Sequence[str],
    names: np.ndarray,
    identity_ids: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate a checkpoint policy on direct, offset, and gross endpoints."""
    maps = percentile_maps(
        panel, system, checkpoint_map, triple, identities, names, identity_ids
    )
    return endpoint_vectors(maps, exposures, identities)


def level_metrics(
    panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    *,
    system: str,
    checkpoint_map: Mapping[str, int],
    identities: Sequence[str],
    folds: Mapping[str, Any],
    image_ids: np.ndarray,
    canonical_identities: np.ndarray,
    theta: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, float]:
    """Aggregate historical level metrics for a selected checkpoint policy."""
    rows = [
        ensemble_percentile_fold_result(
            image_ids,
            canonical_identities,
            theta,
            folds[identity],
            [
                panel[(system, identity, seed)][int(checkpoint_map[identity])][:520]
                for seed in seeds
            ],
        )
        for identity in identities
    ]
    return {
        name: float(value)
        for name, value in aggregate_level_fold_results(rows).items()
    }


def level_metric_vector(
    panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    *,
    system: str,
    checkpoint_map: Mapping[str, int],
    triple: Sequence[int],
    identities: Sequence[str],
    folds: Mapping[str, Any],
    image_ids: np.ndarray,
    canonical_identities: np.ndarray,
    theta: np.ndarray,
    seeds: Sequence[int],
    metric: str,
) -> np.ndarray:
    """Return one held-out level metric per identity for a seed triple."""
    return np.asarray(
        [
            float(
                ensemble_percentile_fold_result(
                    image_ids,
                    canonical_identities,
                    theta,
                    folds[identity],
                    [
                        panel[(system, identity, int(seeds[index]))][
                            int(checkpoint_map[identity])
                        ][:520]
                        for index in triple
                    ],
                )["metrics"][metric]
            )
            for identity in identities
        ],
        dtype=np.float64,
    )


def paired_candidate_uq(
    *,
    candidate_panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    candidate_system: str,
    candidate_checkpoint_map: Mapping[str, int],
    reference_panel: Mapping[tuple[str, str, int], Mapping[int, np.ndarray]],
    reference_system: str,
    reference_checkpoint_map: Mapping[str, int],
    identities: Sequence[str],
    names: np.ndarray,
    identity_ids: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
    extra_candidate_vectors: Mapping[str, Sequence[np.ndarray]] | None = None,
    extra_reference_vectors: Mapping[str, Sequence[np.ndarray]] | None = None,
    guard_metrics: frozenset[str] = frozenset({"offset", "gross"}),
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return shared 27-triple UQ and guard rows for a matched LOIO contrast."""
    triples = tuple(itertools.product(range(3), repeat=3))
    standard = (0, 1, 2)
    metrics = ("direct", "offset", "gross")
    candidate_vectors = {metric: [] for metric in metrics}
    reference_vectors = {metric: [] for metric in metrics}
    for triple in triples:
        candidate = evaluate_candidate(
            candidate_panel,
            system=candidate_system,
            checkpoint_map=candidate_checkpoint_map,
            triple=triple,
            identities=identities,
            names=names,
            identity_ids=identity_ids,
            exposures=exposures,
        )
        reference = evaluate_candidate(
            reference_panel,
            system=reference_system,
            checkpoint_map=reference_checkpoint_map,
            triple=triple,
            identities=identities,
            names=names,
            identity_ids=identity_ids,
            exposures=exposures,
        )
        for metric in metrics:
            candidate_vectors[metric].append(candidate[metric])
            reference_vectors[metric].append(reference[metric])
    candidate_extras = extra_candidate_vectors or {}
    reference_extras = extra_reference_vectors or {}
    if set(candidate_extras) != set(reference_extras):
        raise ValueError("Candidate and reference extra-metric keys must agree.")
    for metric, values in candidate_extras.items():
        if len(values) != len(triples) or len(reference_extras[metric]) != len(triples):
            raise ValueError("Extra metric vectors must cover all 27 seed triples.")
        candidate_vectors[metric] = list(values)
        reference_vectors[metric] = list(reference_extras[metric])
    payloads: dict[str, dict[str, Any]] = {}
    uq_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    standard_index = triples.index(standard)
    for metric, candidate_values in candidate_vectors.items():
        reference_values = reference_vectors[metric]
        payload, _ = paired_uq(
            np.stack(candidate_values), np.stack(reference_values)
        )
        payloads[metric] = payload
        uq_rows.append(
            {
                "record_type": "aggregate",
                "metric": metric,
                **payload,
                "identity_id": "TUNE14",
            }
        )
        point_candidate = np.asarray(candidate_values)[standard_index]
        point_reference = np.asarray(reference_values)[standard_index]
        for identity, reference_value, candidate_value in zip(
            identities, point_reference, point_candidate, strict=True
        ):
            delta = float(candidate_value - reference_value)
            uq_rows.append(
                {
                    "record_type": "identity",
                    "metric": metric,
                    "point_delta": delta,
                    "CI90_lower": "",
                    "CI90_upper": "",
                    "identity_id": identity,
                }
            )
            if metric in guard_metrics:
                guard_rows.append(
                    {
                        "record_type": "identity",
                        "metric": metric,
                        "identity_id": identity,
                        "reference": float(reference_value),
                        "candidate": float(candidate_value),
                        "delta": delta,
                        "CI90_lower": "",
                        "CI90_upper": "",
                        "P_delta_lt_0": "",
                    }
                )
        if metric in guard_metrics:
            guard_rows.append(
                {
                    "record_type": "aggregate",
                    "metric": metric,
                    "identity_id": "TUNE14",
                    "reference": float(point_reference.mean()),
                    "candidate": float(point_candidate.mean()),
                    "delta": payload["point_delta"],
                    "CI90_lower": payload["CI90_lower"],
                    "CI90_upper": payload["CI90_upper"],
                    "P_delta_lt_0": payload["P_delta_lt_0"],
                }
            )
    return payloads, uq_rows, guard_rows
