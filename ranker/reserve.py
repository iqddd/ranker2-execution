"""Reusable primitives for frozen reserve scoring and disagreement audits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np

from ranker.task_b import percentile_score_from_sorted_train


def score_bundle(
    train_scores: np.ndarray, evaluation_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return train-referenced z-scores and deterministic mid-rank percentiles."""
    train = np.asarray(train_scores, dtype=np.float64)
    values = np.asarray(evaluation_scores, dtype=np.float64)
    mean = float(np.mean(train))
    standard_deviation = float(np.std(train, ddof=0))
    if standard_deviation <= 0.0 or not np.isfinite(standard_deviation):
        raise RuntimeError("A train-score distribution has invalid scale.")
    ordered = np.sort(train)
    percentiles = np.asarray(
        [
            percentile_score_from_sorted_train(ordered, float(value))
            for value in values
        ],
        dtype=np.float64,
    )
    return (values - mean) / standard_deviation, percentiles


def maximum_matching_size(
    nodes: Sequence[str], edges: Sequence[tuple[str, str]]
) -> int:
    """Return the cardinality of a deterministic undirected maximum matching."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return len(nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True))


def disagreement_diagnostics(
    *,
    graph_rows: Sequence[Mapping[str, Any]],
    identity_nodes: Mapping[str, Sequence[str]],
    graph_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Summarize one frozen disagreement graph by reserve identity."""
    flag = f"{graph_name}_disagreement"
    confidence = f"{graph_name}_confidence"
    rows: list[dict[str, Any]] = []
    global_confidence: dict[str, int] = {}
    for identity, nodes in identity_nodes.items():
        selected = [
            row for row in graph_rows if str(row["reserve_identity_id"]) == identity
        ]
        disagreements = [row for row in selected if bool(row[flag])]
        degree = Counter({node: 0 for node in nodes})
        edges: list[tuple[str, str]] = []
        for row in disagreements:
            first = str(row["image_id_1"])
            second = str(row["image_id_2"])
            degree[first] += 1
            degree[second] += 1
            edges.append((first, second))
        values = np.asarray(list(degree.values()), dtype=np.int64)
        rows.append(
            {
                "graph": graph_name,
                "reserve_identity_id": identity,
                "total_same_identity_pairs": len(selected),
                "disagreement_pair_count": len(disagreements),
                "disagreement_rate": len(disagreements) / len(selected),
                "incident_images": int(np.count_nonzero(values)),
                "degree_zero_images": int(np.count_nonzero(values == 0)),
                "minimum_degree": int(values.min()),
                "median_degree": float(np.median(values)),
                "maximum_degree": int(values.max()),
                "maximum_cardinality_matching_size": maximum_matching_size(
                    nodes, edges
                ),
            }
        )
    all_disagreements = [row for row in graph_rows if bool(row[flag])]
    for threshold in (0.0, 0.05, 0.10, 0.25, 0.50):
        label = "0" if threshold == 0.0 else f"{threshold:.2f}"
        global_confidence[label] = sum(
            float(row[confidence]) >= threshold for row in all_disagreements
        )
    return rows, global_confidence


def residual_scale_metrics(
    *, base_scores: np.ndarray, head_scores: np.ndarray
) -> dict[str, float]:
    """Audit an ensemble residual and its projection orthogonal to the base."""
    base = np.asarray(base_scores, dtype=np.float64)
    head = np.asarray(head_scores, dtype=np.float64)
    residual = head - base
    if (
        base.shape != head.shape
        or base.ndim != 1
        or len(base) < 2
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(head))
    ):
        raise RuntimeError("Residual-scale population is invalid.")
    variance = float(np.var(base, ddof=0))
    if variance <= 0.0:
        raise RuntimeError("Residual-scale base has zero variance.")
    covariance = float(
        np.mean((residual - np.mean(residual)) * (base - np.mean(base)))
    )
    beta = covariance / variance
    orthogonal = residual - beta * base
    residual_sd = float(np.std(residual, ddof=0))
    orthogonal_sd = float(np.std(orthogonal, ddof=0))

    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        if np.std(first, ddof=0) == 0.0 or np.std(second, ddof=0) == 0.0:
            return 0.0
        return float(np.corrcoef(first, second)[0, 1])

    return {
        "SD_BASE": float(np.std(base, ddof=0)),
        "SD_residual": residual_sd,
        "SD_HEAD": float(np.std(head, ddof=0)),
        "corr_residual_BASE": correlation(residual, base),
        "beta": beta,
        "SD_orthogonal_residual": orthogonal_sd,
        "SD_orthogonal_over_SD_residual": (
            orthogonal_sd / residual_sd if residual_sd > 0.0 else 0.0
        ),
        "corr_orthogonal_residual_BASE": correlation(orthogonal, base),
    }
