"""Replay and selection helpers for simultaneous layer-screen score panels."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ranker.data import RankedImage
from ranker.external_validation import percentile_against
from ranker.validation_resolution import (
    contiguous_checkpoint_plateau,
    evaluate_direct_ordinal,
    filter_direct_ordinal_exposures,
)


def fivefold_assignment(names: Sequence[str], identities: np.ndarray) -> np.ndarray:
    """Deterministic identity-stratified fivefold image assignment."""
    assignment = np.empty(len(names), dtype=np.int64)
    for identity in sorted(
        set(identities.tolist()), key=lambda value: value.encode("utf-8")
    ):
        positions = [
            index for index, value in enumerate(identities) if value == identity
        ]
        positions.sort(key=lambda index: names[index].encode("utf-8"))
        for offset, index in enumerate(positions):
            assignment[index] = offset % 5
    return assignment


def purged_groups(
    groups: Sequence[Sequence[RankedImage]], validation_names: set[str]
) -> list[list[RankedImage]]:
    """Remove held-out images while retaining groups with at least one pair."""
    output: list[list[RankedImage]] = []
    for group in groups:
        retained = [item for item in group if item.image_name not in validation_names]
        if len(retained) >= 2:
            output.append(retained)
    return output


def fold_inventory(
    names: Sequence[str],
    identities: np.ndarray,
    assignment: np.ndarray,
    o_groups: Sequence[Sequence[RankedImage]],
    w_groups: Sequence[Sequence[RankedImage]],
) -> list[dict[str, Any]]:
    """Audit the known-identity image-heldout fold construction."""
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation = {names[index] for index in np.flatnonzero(assignment == fold)}
        retained_o = purged_groups(o_groups, validation)
        retained_w = purged_groups(w_groups, validation)
        rows.append(
            {
                "fold": fold,
                "validation_images": len(validation),
                "training_images": len(names) - len(validation),
                "represented_identities": len(set(identities[assignment == fold])),
                "remaining_O_groups": len(retained_o),
                "remaining_W_groups": len(retained_w),
                "remaining_O_pairs": sum(
                    len(group) * (len(group) - 1) // 2 for group in retained_o
                ),
                "remaining_W_pairs": sum(
                    len(group) * (len(group) - 1) // 2 for group in retained_w
                ),
                "empty_O_groups": sum(
                    len([item for item in group if item.image_name not in validation])
                    < 2
                    for group in o_groups
                ),
                "empty_W_groups": sum(
                    len([item for item in group if item.image_name not in validation])
                    < 2
                    for group in w_groups
                ),
                "partial_O_groups": sum(
                    2
                    <= len(
                        [item for item in group if item.image_name not in validation]
                    )
                    < len(group)
                    for group in o_groups
                ),
                "partial_W_groups": sum(
                    2
                    <= len(
                        [item for item in group if item.image_name not in validation]
                    )
                    < len(group)
                    for group in w_groups
                ),
                "image_leakage": 0,
            }
        )
    return rows


def exposure_loss_rows(
    exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    fold_maps: Sequence[Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Evaluate exposures using the score map of each image's held-out fold."""
    rows: list[dict[str, Any]] = []
    for source in exposures:
        key = str(source["image_key"])
        fold = int(image_fold[key])
        panel = fold_maps[fold]
        x_score = float(panel[key])
        distances = np.asarray(source["distances"], dtype=np.float64)
        credits = []
        for anchor, anchor_rank in zip(
            source["anchor_keys"], source["anchor_ranks"], strict=True
        ):
            model_delta = x_score - float(panel[str(anchor)])
            human_delta = int(source["human_rank"]) - int(anchor_rank)
            credits.append(
                0.5 if model_delta == 0.0 else float(model_delta * human_delta > 0)
            )
        row = dict(source)
        row["fold"] = fold
        row["loss"] = float(
            np.sum(distances * (1.0 - np.asarray(credits))) / distances.sum()
        )
        rows.append(row)
    return rows


def aggregate_exposure_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, float]]:
    """Apply exposure→image→identity→global endpoint balancing."""
    images: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        images[(str(row["identity_id"]), str(row["image_key"]))].append(
            float(row["loss"])
        )
    identities: dict[str, list[float]] = defaultdict(list)
    for (identity, _key), values in images.items():
        identities[identity].append(float(np.mean(values)))
    identity_losses = {
        identity: float(np.mean(values)) for identity, values in identities.items()
    }
    return float(np.mean(list(identity_losses.values()))), identity_losses


def endpoint_metrics(
    exposures: Sequence[Mapping[str, Any]],
    gross_exposures: Sequence[Mapping[str, Any]],
    image_fold: Mapping[str, int],
    fold_maps: Sequence[Mapping[str, float]],
) -> tuple[float, float, float, dict[str, float]]:
    """Compute direct loss, theta-free offset G, and conditional gross loss."""
    direct_rows = exposure_loss_rows(exposures, image_fold, fold_maps)
    direct, identity_losses = aggregate_exposure_rows(direct_rows)
    gross, _ = aggregate_exposure_rows(
        exposure_loss_rows(gross_exposures, image_fold, fold_maps)
    )
    high: dict[str, float] = defaultdict(float)
    low: dict[str, float] = defaultdict(float)
    for row in exposures:
        identity = str(row["identity_id"])
        panel = fold_maps[int(image_fold[str(row["image_key"])])]
        x_score = float(panel[str(row["image_key"])])
        for anchor, anchor_rank in zip(
            row["anchor_keys"], row["anchor_ranks"], strict=True
        ):
            model_sign = np.sign(x_score - float(panel[str(anchor)]))
            human_sign = np.sign(int(row["human_rank"]) - int(anchor_rank))
            high[identity] += float(human_sign < 0 and model_sign > 0) - 0.5 * float(
                model_sign == 0
            )
            low[identity] += float(human_sign > 0 and model_sign < 0) - 0.5 * float(
                model_sign == 0
            )
    signed = []
    for identity in sorted(identity_losses, key=lambda value: value.encode("utf-8")):
        error = high[identity] + low[identity]
        signed.append(0.0 if error <= 0.0 else (high[identity] - low[identity]) / error)
    return direct, float(np.mean(np.abs(signed))), gross, identity_losses


def percentile_score_maps(
    scores: np.ndarray,
    seed_indices: Sequence[int],
    assignment: np.ndarray,
    names: Sequence[str],
) -> list[dict[str, float]]:
    """Transform each seed against its fold-training canonical reference."""
    output: list[dict[str, float]] = []
    canonical = np.arange(520)
    for fold in range(5):
        reference = canonical[assignment[:520] != fold]
        transformed = [
            percentile_against(scores[index, fold], scores[index, fold, reference])
            for index in seed_indices
        ]
        ensemble = np.mean(np.stack(transformed), axis=0)
        output.append(
            {
                name: float(value)
                for name, value in zip(names[:520], ensemble, strict=True)
            }
        )
    return output


def analyze_panel(
    panel: Mapping[str, np.ndarray],
    names: Sequence[str],
    assignment: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay checkpoint surfaces, FULL/LOFO selection, guards, and d diagnostics."""
    layers = np.asarray(panel["layers"], dtype=np.int64)
    checkpoints = np.asarray(panel["checkpoints"], dtype=np.int64)
    scores = np.asarray(panel["scores"], dtype=np.float64)
    seed_count = int(np.sum(np.isfinite(np.asarray(panel["runtimes"])).any(axis=1)))
    seed_indices = tuple(range(seed_count))
    image_fold = {name: int(fold) for name, fold in zip(names, assignment, strict=True)}
    gross_exposures = filter_direct_ordinal_exposures(exposures, minimum_distance=3)
    shape = (len(layers), len(checkpoints))
    direct = np.empty(shape, dtype=np.float64)
    offset = np.empty(shape, dtype=np.float64)
    gross = np.empty(shape, dtype=np.float64)
    per_fold = np.empty((*shape, 5), dtype=np.float64)
    identity_surface: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            maps = percentile_score_maps(
                scores[:, :, checkpoint_index, layer_index],
                seed_indices,
                assignment,
                names,
            )
            (
                direct[layer_index, checkpoint_index],
                offset[layer_index, checkpoint_index],
                gross[layer_index, checkpoint_index],
                identity_losses,
            ) = endpoint_metrics(exposures, gross_exposures, image_fold, maps)
            evaluated = exposure_loss_rows(exposures, image_fold, maps)
            for fold in range(5):
                selected = [row for row in evaluated if int(row["fold"]) == fold]
                per_fold[layer_index, checkpoint_index, fold] = aggregate_exposure_rows(
                    selected
                )[0]
            identity_surface.extend(
                {
                    "layer": int(layer),
                    "checkpoint": int(checkpoint),
                    "identity_id": identity,
                    "loss": loss,
                }
                for identity, loss in identity_losses.items()
            )

    full_indices = np.argmin(direct, axis=1)
    lofo_maps = np.empty((len(layers), 5), dtype=np.int64)
    lofo_loss = np.empty(len(layers), dtype=np.float64)
    lofo_fold_loss = np.empty((len(layers), 5), dtype=np.float64)
    lofo_rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        selected_rows: list[dict[str, Any]] = []
        for heldout in range(5):
            train = [fold for fold in range(5) if fold != heldout]
            checkpoint_index = int(
                np.argmin(per_fold[layer_index][:, train].mean(axis=1))
            )
            checkpoint = int(checkpoints[checkpoint_index])
            lofo_maps[layer_index, heldout] = checkpoint
            maps = percentile_score_maps(
                scores[:, :, checkpoint_index, layer_index],
                seed_indices,
                assignment,
                names,
            )
            rows = [
                row
                for row in exposure_loss_rows(exposures, image_fold, maps)
                if int(row["fold"]) == heldout
            ]
            lofo_fold_loss[layer_index, heldout] = aggregate_exposure_rows(rows)[0]
            selected_rows.extend(rows)
            lofo_rows.append(
                {
                    "record_type": "heldout_fold",
                    "layer": int(layer),
                    "fold": heldout,
                    "checkpoint": checkpoint,
                    "heldout_loss": lofo_fold_loss[layer_index, heldout],
                }
            )
        lofo_loss[layer_index] = aggregate_exposure_rows(selected_rows)[0]
        lofo_rows.append(
            {
                "record_type": "aggregate",
                "layer": int(layer),
                "fold": "ALL",
                "checkpoint": "mixed",
                "heldout_loss": lofo_loss[layer_index],
            }
        )

    surface_rows = []
    full_rows = []
    guard_rows = []
    d_rows = []
    d_values = np.asarray(panel["d"], dtype=np.float64)
    reference_index = int(np.flatnonzero(layers == 26)[0])
    for layer_index, layer in enumerate(layers):
        selected_index = int(full_indices[layer_index])
        qualified = np.flatnonzero(
            direct[layer_index] <= direct[layer_index].min() + 0.005
        )
        guard_index = int(
            min(
                qualified,
                key=lambda index: (
                    offset[layer_index, index],
                    direct[layer_index, index],
                    checkpoints[index],
                ),
            )
        )
        full_rows.append(
            {
                "layer": int(layer),
                "checkpoint": int(checkpoints[selected_index]),
                "direct_loss": direct[layer_index, selected_index],
                "LOFO_loss": lofo_loss[layer_index],
                "LOFO_delta_vs_L26": lofo_loss[layer_index]
                - lofo_loss[reference_index],
                "offset_G": offset[layer_index, selected_index],
                "gross_d3_loss": gross[layer_index, selected_index],
                "plateau_count": len(qualified),
                "contiguous_plateau": str(
                    contiguous_checkpoint_plateau(
                        checkpoints,
                        direct[layer_index],
                        selected_index,
                        0.005,
                    )
                ),
                "FULL_boundary": bool(checkpoints[selected_index] == checkpoints[-1]),
                "any_LOFO_boundary": bool(checkpoints[-1] in lofo_maps[layer_index]),
            }
        )
        guard_rows.append(
            {
                "layer": int(layer),
                "checkpoint": int(checkpoints[guard_index]),
                "direct_loss": direct[layer_index, guard_index],
                "offset_G": offset[layer_index, guard_index],
                "eligible_within_best_plus_0_005": True,
            }
        )
        mean_d = np.nanmean(d_values[:seed_count, :, :, layer_index], axis=(0, 1))
        final = float(mean_d[-1])
        tolerance = max(1e-12, abs(final) * 0.01)
        stabilized: int | None = None
        for checkpoint_index in range(1, len(checkpoints)):
            if np.all(np.abs(mean_d[checkpoint_index:] - final) <= tolerance):
                stabilized = int(checkpoints[checkpoint_index])
                break
        d_rows.extend(
            {
                "record_type": "checkpoint_mean",
                "layer": int(layer),
                "checkpoint": int(checkpoint),
                "d": float(mean_d[index]),
                "gradient_norm": float(
                    np.nanmean(
                        np.asarray(panel["gradient_norm"])[
                            :seed_count, :, index, layer_index
                        ]
                    )
                ),
                "training_O_loss": float(
                    np.nanmean(
                        np.asarray(panel["training_O_loss"])[
                            :seed_count, :, index, layer_index
                        ]
                    )
                ),
                "training_W_loss": float(
                    np.nanmean(
                        np.asarray(panel["training_W_loss"])[
                            :seed_count, :, index, layer_index
                        ]
                    )
                ),
                "d_stabilization_checkpoint": stabilized,
                "validation_minimum_checkpoint": int(checkpoints[selected_index]),
            }
            for index, checkpoint in enumerate(checkpoints)
        )
        surface_rows.extend(
            {
                "layer": int(layer),
                "checkpoint": int(checkpoint),
                "direct_loss": direct[layer_index, index],
                "offset_G": offset[layer_index, index],
                "gross_d3_loss": gross[layer_index, index],
                **{
                    f"fold_{fold}_loss": per_fold[layer_index, index, fold]
                    for fold in range(5)
                },
            }
            for index, checkpoint in enumerate(checkpoints)
        )
    return {
        "seed_count": seed_count,
        "direct": direct,
        "offset": offset,
        "gross": gross,
        "per_fold": per_fold,
        "full_indices": full_indices,
        "lofo_maps": lofo_maps,
        "lofo_loss": lofo_loss,
        "surface_rows": surface_rows,
        "identity_surface_rows": identity_surface,
        "full_rows": full_rows,
        "lofo_rows": lofo_rows,
        "guard_rows": guard_rows,
        "optimizer_rows": d_rows,
    }


def replay_step47a_selected(
    state_paths: Mapping[tuple[str, int], Path],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Replay the four selected STEP47A systems from their saved scores."""
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    systems = sorted({system for system, _fold in state_paths})
    results: dict[str, dict[str, Any]] = {}
    for system in systems:
        maps: dict[str, dict[str, float]] = {}
        checkpoint = None
        for fold in range(4):
            payload = torch.load(
                state_paths[(system, fold)], map_location="cpu", weights_only=False
            )
            identity = str(payload["fold"])
            checkpoint = int(payload["checkpoint"])
            raw = np.asarray(payload["scores"], dtype=np.float64)
            percentiles = percentile_against(raw, raw[canonical_identities != identity])
            maps[identity] = {
                name: float(value)
                for name, value in zip(canonical_names, percentiles, strict=True)
            }
        result = evaluate_direct_ordinal(exposures, maps)
        results[system] = result
        identity_losses = {
            str(row["identity_id"]): float(row["loss"]) for row in result["identities"]
        }
        rows.extend(
            {
                "record_type": "identity",
                "system": system,
                "checkpoint": checkpoint,
                "identity_id": identity,
                "direct_loss": loss,
                "contrast": "",
                "delta": "",
            }
            for identity, loss in identity_losses.items()
        )
        rows.append(
            {
                "record_type": "system",
                "system": system,
                "checkpoint": checkpoint,
                "identity_id": "ALL",
                "direct_loss": result["loss"],
                "contrast": "",
                "delta": "",
            }
        )
    for candidate, reference, label in (
        ("J3_JOINT4_L1FT", "P3_JOINT4_FROZEN", "J3-P3"),
        ("J2_MAIN4_L1FT", "P2_MAIN4_FROZEN", "J2-P2"),
    ):
        candidate_rows = {
            str(row["identity_id"]): float(row["loss"])
            for row in results[candidate]["identities"]
        }
        reference_rows = {
            str(row["identity_id"]): float(row["loss"])
            for row in results[reference]["identities"]
        }
        rows.extend(
            {
                "record_type": "contrast_identity",
                "system": candidate,
                "checkpoint": "",
                "identity_id": identity,
                "direct_loss": "",
                "contrast": label,
                "delta": candidate_rows[identity] - reference_rows[identity],
            }
            for identity in sorted(candidate_rows)
        )
        rows.append(
            {
                "record_type": "contrast",
                "system": candidate,
                "checkpoint": "",
                "identity_id": "ALL",
                "direct_loss": "",
                "contrast": label,
                "delta": float(results[candidate]["loss"])
                - float(results[reference]["loss"]),
            }
        )
    return rows, time.perf_counter() - started
