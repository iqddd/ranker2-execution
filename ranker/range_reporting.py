"""Reusable CSV and diagnostic helpers for LoRA learning-rate trajectories."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def read_range_csv(
    path: Path, *, expected_trajectories: int = 6, expected_points: int = 36
) -> list[dict[str, Any]]:
    """Read typed rows and enforce the expected trajectory rectangle."""
    integer_fields = {"range_fold", "pilot_index", "seed", "k"}
    boolean_fields = {"is_baseline", "finite", "gradients_finite"}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for name, value in raw.items():
                if value == "":
                    continue
                if name == "signature":
                    row[name] = value
                elif name in integer_fields:
                    row[name] = int(value)
                elif name in boolean_fields:
                    row[name] = value.lower() == "true"
                else:
                    row[name] = float(value)
            rows.append(row)
    expected_rows = expected_trajectories * (expected_points + 1)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} saved range rows, got {len(rows)}."
        )
    expected_lr_rows = expected_trajectories * expected_points
    if sum(int(row["k"] >= 0) for row in rows) != expected_lr_rows:
        raise RuntimeError("Saved range CSV does not contain the expected LR points.")
    return rows


def trajectory_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    """Group rows by pilot and seed, sorted by the original k field."""
    result: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        result.setdefault((int(row["pilot_index"]), int(row["seed"])), []).append(row)
    for values in result.values():
        values.sort(key=lambda row: int(row["k"]))
    return result


def counter_diagnostics(
    *, rows: Sequence[Mapping[str, Any]], selected_k: int | None
) -> dict[str, Any] | None:
    """Aggregate the fixed counter metrics for a selected range point."""
    if selected_k is None:
        return None
    grouped = trajectory_rows(rows)
    fields = (
        "validation_counter_pair_acc",
        "validation_counter_WPA",
        "validation_counter_d_ge_2_pair_acc",
        "validation_counter_d_ge_3_pair_acc",
    )
    baseline: dict[str, float] = {}
    selected: dict[str, float] = {}
    deltas: dict[str, float] = {}
    for field in fields:
        base_values: list[float] = []
        selected_values: list[float] = []
        delta_values: list[float] = []
        for values in grouped.values():
            base = next(row for row in values if int(row["k"]) == -1)
            point = next(row for row in values if int(row["k"]) == selected_k)
            base_values.append(float(base[field]))
            selected_values.append(float(point[field]))
            delta_values.append(float(point[field]) - float(base[field]))
        baseline[field] = float(np.median(base_values))
        selected[field] = float(np.median(selected_values))
        deltas[field] = float(np.median(delta_values))
    fold_rows: list[dict[str, Any]] = []
    for pilot in sorted({key[0] for key in grouped}):
        keys = [key for key in grouped if key[0] == pilot]

        def mean_delta(
            field: str, trajectory_keys: list[tuple[int, int]] = keys
        ) -> float:
            return float(
                np.mean(
                    [
                        float(
                            next(
                                row
                                for row in grouped[key]
                                if int(row["k"]) == selected_k
                            )[field]
                        )
                        - float(
                            next(row for row in grouped[key] if int(row["k"]) == -1)[
                                field
                            ]
                        )
                        for key in trajectory_keys
                    ]
                )
            )

        fold_rows.append(
            {
                "pilot_index": pilot,
                "mean_seed_counter_pair_acc_delta": mean_delta(
                    "validation_counter_pair_acc"
                ),
                "mean_seed_counter_d_ge_2_pair_acc_delta": mean_delta(
                    "validation_counter_d_ge_2_pair_acc"
                ),
            }
        )
    return {
        "aggregation": (
            "pooled values are trajectory medians; fold signs use the mean "
            "delta of the two seed trajectories"
        ),
        "baseline": baseline,
        "selected_range_point": selected,
        "deltas": deltas,
        "n_positive_counter_folds": sum(
            int(row["mean_seed_counter_pair_acc_delta"] > 0.0) for row in fold_rows
        ),
        "n_positive_d_ge_2_folds": sum(
            int(row["mean_seed_counter_d_ge_2_pair_acc_delta"] > 0.0)
            for row in fold_rows
        ),
        "foldwise": fold_rows,
    }


def build_lora_checkpoint_manifest(
    *,
    signature: str,
    marker: str,
    source: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    rank: int,
    alpha: float,
    target_modules: Sequence[str],
    expected_parameters: int,
    lr_ratio_b_to_a: float,
) -> dict[str, Any]:
    """Assemble the stable LoRA checkpoint lineage and optimizer contract."""
    return {
        "signature": signature,
        "marker": marker,
        "input_contract": dict(source),
        "calibrated_attention_checkpoints": [
            {
                "pilot_index": int(row["pilot_index"]),
                "seed": int(row["seed"]),
                "filename": str(row["filename"]),
                "sha256": str(row["sha256"]),
            }
            for row in records
        ],
        "model_state": {
            "base_LINEAR_INNER_BAL": "frozen",
            "calibrated_attention_head": "frozen",
            "original_layer_26": "frozen",
            "lora": "trainable",
        },
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "scaling": alpha / rank,
            "dropout": 0.0,
            "A_initialization": "Kaiming uniform",
            "B_initialization": "zero",
            "target_modules": list(target_modules),
            "expected_trainable_parameters": expected_parameters,
            "lr_ratio_B_to_A": lr_ratio_b_to_a,
        },
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay_A": 0.01,
            "weight_decay_B": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "range_checkpoint_retained": False,
        "range_checkpoint_reuse_authorized": False,
    }
