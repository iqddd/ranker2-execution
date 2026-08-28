"""Slope-based LR range analysis shared by Step41A-R1 and R2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ranker.attention_training import bias_corrected_ema

def analyze_slope_range(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    initial_attention_scale_pass: bool,
    learning_rates: Sequence[float],
    ema_beta: float,
) -> dict[str, Any]:
    range_lrs = tuple(float(value) for value in learning_rates)
    range_steps = len(range_lrs)
    if range_steps < 3 or any(value <= 0.0 for value in range_lrs):
        raise ValueError("Expected at least three positive learning rates.")
    trajectories: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        trajectories.setdefault((int(row["pilot_index"]), int(row["seed"])), []).append(
            row
        )
    if len(trajectories) != 6:
        raise AssertionError(f"Expected six {arm} trajectories.")
    curves: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    ordered_trajectories: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    x = np.log(np.asarray(range_lrs, dtype=np.float64))
    for key, trajectory in trajectories.items():
        baseline = [row for row in trajectory if int(row["step"]) == -1]
        points = sorted(
            (row for row in trajectory if int(row["step"]) >= 0),
            key=lambda row: int(row["step"]),
        )
        if len(baseline) != 1 or len(points) != range_steps:
            raise AssertionError(f"Invalid {arm} trajectory shape for {key}.")
        ordered_trajectories[key] = points
        ratios = np.asarray(
            [
                float(row.get("train_balanced_loss_ratio_to_baseline", np.nan))
                if bool(row.get("finite", False))
                else np.nan
                for row in points
            ],
            dtype=np.float64,
        )
        smoothed = np.full(range_steps, np.nan, dtype=np.float64)
        finite_prefix = 0
        for value in ratios:
            if not np.isfinite(value) or value <= 0.0:
                break
            finite_prefix += 1
        if finite_prefix:
            smoothed[:finite_prefix] = bias_corrected_ema(
                ratios[:finite_prefix], beta=ema_beta
            )
        slope = np.full(range_steps, np.nan, dtype=np.float64)
        for step in range(1, range_steps - 1):
            if (
                np.isfinite(smoothed[step - 1])
                and np.isfinite(smoothed[step])
                and np.isfinite(smoothed[step + 1])
            ):
                slope[step] = (
                    math.log(smoothed[step + 1]) - math.log(smoothed[step - 1])
                ) / (x[step + 1] - x[step - 1])
        curves[key] = {
            "raw_train_ratio": ratios,
            "smoothed_train_ratio": smoothed,
            "slope": slope,
        }

    pooled: list[dict[str, Any]] = []
    for step, learning_rate in enumerate(range_lrs):
        point_rows = [ordered_trajectories[key][step] for key in ordered_trajectories]

        def finite_median(values: Sequence[float]) -> float:
            array = np.asarray(values, dtype=np.float64)
            return (
                float(np.median(array)) if np.all(np.isfinite(array)) else float("nan")
            )

        pooled.append(
            {
                "step": step,
                "learning_rate": learning_rate,
                "finite_trajectories": int(
                    sum(bool(row.get("finite", False)) for row in point_rows)
                ),
                "median_smoothed_train_balanced_ratio": finite_median(
                    [curves[key]["smoothed_train_ratio"][step] for key in curves]
                ),
                "median_train_balanced_ratio": finite_median(
                    [curves[key]["raw_train_ratio"][step] for key in curves]
                ),
                "median_train_balanced_slope": finite_median(
                    [curves[key]["slope"][step] for key in curves]
                ),
                "median_validation_aligned_loss_ratio": finite_median(
                    [
                        float(
                            row.get(
                                "validation_aligned_loss_ratio_to_baseline",
                                np.nan,
                            )
                        )
                        for row in point_rows
                    ]
                ),
                "median_validation_counter_loss_ratio": finite_median(
                    [
                        float(
                            row.get(
                                "validation_counter_loss_ratio_to_baseline",
                                np.nan,
                            )
                        )
                        for row in point_rows
                    ]
                ),
                "median_validation_cross_all_WPA_delta": finite_median(
                    [
                        float(
                            row.get(
                                "validation_cross_all_WPA_delta_to_baseline",
                                np.nan,
                            )
                        )
                        for row in point_rows
                    ]
                ),
                "median_train_residual_std": finite_median(
                    [float(row.get("residual_std_train", np.nan)) for row in point_rows]
                ),
                "median_last_block_token_delta_std": finite_median(
                    [
                        float(row.get("last_block_token_delta_std", np.nan))
                        for row in point_rows
                    ]
                ),
            }
        )

    eligible: list[dict[str, Any]] = []
    for point in pooled[1:-1]:
        common = bool(
            point["finite_trajectories"] == 6
            and np.isfinite(point["median_train_balanced_slope"])
            and point["median_validation_aligned_loss_ratio"] <= 1.050
            and point["median_validation_cross_all_WPA_delta"] >= -0.030
        )
        if arm == "head":
            common = bool(common and point["median_train_residual_std"] >= 0.01)
        else:
            common = bool(
                common and point["median_last_block_token_delta_std"] >= 0.001
            )
        if common:
            eligible.append(point)
    selected = (
        min(
            eligible,
            key=lambda point: (
                point["median_train_balanced_slope"],
                point["median_smoothed_train_balanced_ratio"],
                point["step"],
            ),
        )
        if eligible
        else None
    )
    star = -1 if selected is None else int(selected["step"])
    finite_smoothed_curve = [
        float(point["median_smoothed_train_balanced_ratio"])
        for point in pooled
        if np.isfinite(point["median_smoothed_train_balanced_ratio"])
    ]
    finite_raw_curve = [
        float(point["median_train_balanced_ratio"])
        for point in pooled
        if np.isfinite(point["median_train_balanced_ratio"])
    ]
    last_four_raw_values = [
        float(point["median_train_balanced_ratio"]) for point in pooled[-4:]
    ]
    last_four_smoothed_values = [
        float(point["median_smoothed_train_balanced_ratio"]) for point in pooled[-4:]
    ]
    last_four_raw = (
        float(np.median(last_four_raw_values))
        if all(np.isfinite(last_four_raw_values))
        else float("nan")
    )
    last_four_smoothed = (
        float(np.median(last_four_smoothed_values))
        if all(np.isfinite(last_four_smoothed_values))
        else float("nan")
    )
    minimum_raw = min(finite_raw_curve) if finite_raw_curve else float("nan")
    minimum_smoothed = (
        min(finite_smoothed_curve) if finite_smoothed_curve else float("nan")
    )
    last_nonfinite = 6 - int(pooled[-1]["finite_trajectories"])
    final_behavior_pass = bool(
        (
            np.isfinite(last_four_raw)
            and np.isfinite(minimum_raw)
            and last_four_raw >= 1.20 * minimum_raw
        )
        or last_nonfinite >= 3
    )
    selected_common_pass = bool(
        selected is not None
        and 6 <= star <= 29
        and selected["median_validation_aligned_loss_ratio"] <= 1.050
        and selected["median_validation_cross_all_WPA_delta"] >= -0.030
        and selected["finite_trajectories"] == 6
        and final_behavior_pass
    )
    if arm == "head":
        range_pass = bool(
            initial_attention_scale_pass
            and selected_common_pass
            and selected is not None
            and selected["median_smoothed_train_balanced_ratio"] <= 0.990
        )
    else:
        range_pass = bool(
            selected_common_pass
            and selected is not None
            and selected["median_smoothed_train_balanced_ratio"] <= 0.995
            and selected["median_last_block_token_delta_std"] >= 0.001
        )
    candidate = (
        None
        if selected is None
        else [
            float(selected["learning_rate"]) / 3.0,
            float(selected["learning_rate"]),
            3.0 * float(selected["learning_rate"]),
        ]
    )
    result = {
        "arm": arm,
        "train_loss_semantics": {
            "timing": "post_optimizer_step",
            "measurement": (
                "full inner_train metric pass with fixed per-trajectory "
                "dropout seed; optimizer objective and mixed precision unchanged"
            ),
            "head_balanced_loss_includes_lambda_div": arm == "head",
            "ema_beta": ema_beta,
            "tail_gate_curve": (
                "raw per-trajectory post-step train ratio, median across "
                "six at each k; EMA is used only at-star and slope"
            ),
        },
        "k_star_R1": star,
        "lr_star_R1": None if selected is None else selected["learning_rate"],
        "lr_grid_full": list(range_lrs),
        "median_train_loss_curve": [
            point["median_smoothed_train_balanced_ratio"] for point in pooled
        ],
        "median_train_loss_slope_curve": [
            point["median_train_balanced_slope"] for point in pooled
        ],
        "pooled_curve": pooled,
        "selected": selected,
        "validation_safety_at_star": (
            None
            if selected is None
            else {
                "median_validation_aligned_loss_ratio": selected[
                    "median_validation_aligned_loss_ratio"
                ],
                "median_validation_cross_all_WPA_delta": selected[
                    "median_validation_cross_all_WPA_delta"
                ],
                "median_validation_counter_loss_ratio_non_authorizing": selected[
                    "median_validation_counter_loss_ratio"
                ],
                "all_6_trajectories_finite": (selected["finite_trajectories"] == 6),
            }
        ),
        "final_range_behavior": {
            "median_last_4_train_balanced_ratio": last_four_raw,
            "minimum_median_train_balanced_ratio": minimum_raw,
            "tail_to_minimum_ratio": (
                last_four_raw / minimum_raw
                if np.isfinite(last_four_raw)
                and np.isfinite(minimum_raw)
                and minimum_raw > 0.0
                else float("nan")
            ),
            "diagnostic_smoothed_median_last_4_ratio": (last_four_smoothed),
            "diagnostic_smoothed_minimum_median_ratio": (minimum_smoothed),
            "nonfinite_trajectories_at_last_point": last_nonfinite,
            "pass": final_behavior_pass,
        },
        "proposed_lr_grid": candidate,
        "authorized_lr_grid": candidate if range_pass else None,
        "range_pass": range_pass,
    }
    if arm == "head":
        result.update(
            {
                "k_head_star_R1": star,
                "head_lr_star_R1": (
                    None if selected is None else selected["learning_rate"]
                ),
                "head_lr_grid": candidate if range_pass else None,
                "head_lr_range_R1_pass": range_pass,
            }
        )
    else:
        result.update(
            {
                "k_lora_star_R1": star,
                "lora_lr_star_R1": (
                    None if selected is None else selected["learning_rate"]
                ),
                "lora_lr_grid": candidate if range_pass else None,
                "train_loss_curve": [
                    point["median_smoothed_train_balanced_ratio"] for point in pooled
                ],
                "train_loss_slope_curve": [
                    point["median_train_balanced_slope"] for point in pooled
                ],
                "token_delta": (
                    None
                    if selected is None
                    else selected["median_last_block_token_delta_std"]
                ),
                "lora_lr_range_R1_pass": range_pass,
            }
        )
    return result
