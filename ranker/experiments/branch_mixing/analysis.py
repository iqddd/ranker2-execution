"""Pure retained-state analysis for additive JOINT4 score branches."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from ranker.attention_lora import AttentionResidualHead
from ranker.external_validation import percentile_against
from ranker.frozen_readouts import attention_scores, linear_scores
from ranker.level_evaluation import (
    aggregate_level_fold_results,
    ensemble_percentile_fold_result,
)
from ranker.ordinal_diagnostics import offset_guard
from ranker.tune14 import paired_uq
from ranker.validation_resolution import (
    direct_identity_loss_vector,
    filter_direct_ordinal_exposures,
)

ALPHAS = np.arange(17, dtype=np.float64) / 8.0
SEED_TRIPLES = tuple(itertools.product(range(3), repeat=3))

ComponentKey = tuple[str, int, int]
Components = dict[ComponentKey, dict[str, np.ndarray]]


def _stable_rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.asarray(values), kind="stable")


def _state_structure_rows(
    identity: str,
    checkpoint: int,
    seed: int,
    pooled: np.ndarray,
    attention: np.ndarray,
    combined: np.ndarray,
    canonical_identities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    branch_values = {"pooled": pooled, "attention": attention, "combined": combined}
    row: dict[str, Any] = {
        "identity_id": identity,
        "checkpoint": checkpoint,
        "seed": seed,
        "pooled_source": "authoritative_combined_minus_exact_attention",
    }
    profiles: list[dict[str, Any]] = []
    for branch, values in branch_values.items():
        means = []
        within_sd = []
        fitted = np.empty_like(values)
        residual = np.empty_like(values)
        for value in sorted(
            set(canonical_identities.tolist()), key=lambda x: x.encode("utf-8")
        ):
            mask = canonical_identities == value
            mean = float(values[mask].mean())
            means.append(mean)
            within_sd.append(float(values[mask].std(ddof=0)))
            fitted[mask] = mean
            residual[mask] = values[mask] - mean
        total_variance = float(values.var(ddof=0))
        between_variance = float(fitted.var(ddof=0))
        within_variance = float(residual.var(ddof=0))
        row[f"{branch}_overall_SD"] = float(values.std(ddof=0))
        row[f"{branch}_mean_within_identity_SD"] = float(np.mean(within_sd))
        row[f"{branch}_between_identity_SD"] = float(np.std(means, ddof=0))
        row[f"{branch}_between_variance_fraction"] = (
            between_variance / total_variance if total_variance else 0.0
        )
        row[f"{branch}_within_variance_fraction"] = (
            within_variance / total_variance if total_variance else 0.0
        )
    row["Spearman_pooled_attention"] = float(spearmanr(pooled, attention).statistic)
    row["Spearman_pooled_combined"] = float(spearmanr(pooled, combined).statistic)
    row["Spearman_attention_combined"] = float(spearmanr(attention, combined).statistic)
    for value in sorted(
        set(canonical_identities.tolist()), key=lambda x: x.encode("utf-8")
    ):
        mask = canonical_identities == value
        profiles.append(
            {
                "fold_identity": identity,
                "checkpoint": checkpoint,
                "seed": seed,
                "profile_identity": value,
                "images": int(mask.sum()),
                "mean_pooled": float(pooled[mask].mean()),
                "mean_attention": float(attention[mask].mean()),
                "SD_pooled": float(pooled[mask].std(ddof=0)),
                "SD_attention": float(attention[mask].std(ddof=0)),
                "Pearson_pooled_attention": float(
                    pearsonr(pooled[mask], attention[mask]).statistic
                ),
            }
        )
    return row, profiles


def load_branch_components(
    state_paths: Mapping[ComponentKey, Path],
    tokens: torch.Tensor,
    canonical_identities: np.ndarray,
    device: torch.device,
) -> tuple[
    Components, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Load retained states and split authoritative scores into additive branches."""
    components: Components = {}
    parity_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    indices = torch.arange(tokens.shape[0], dtype=torch.int64, device=device)
    for identity, checkpoint, seed in sorted(
        state_paths, key=lambda key: (key[0].encode(), key[1], key[2])
    ):
        payload = torch.load(
            state_paths[(identity, checkpoint, seed)],
            map_location="cpu",
            weights_only=False,
        )
        state = payload["readout_state"]
        head = (
            AttentionResidualHead(
                seed=seed,
                initial_query_norm=128.0**0.5,
                query_count=4,
                value_dimension=256,
                zero_output=True,
            )
            .to(device)
            .eval()
        )
        head.load_state_dict(state["head_state_dict"], strict=True)
        attention = (
            attention_scores(head, tokens, indices, batch_size=128)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        authoritative = np.asarray(payload["raw_scores"], dtype=np.float64)
        pooled = authoritative - attention
        reconstructed = pooled + attention
        maximum = float(np.max(np.abs(reconstructed - authoritative)))
        raw_exact = bool(np.array_equal(reconstructed, authoritative))
        rank_exact = bool(
            _stable_rank(reconstructed).tolist() == _stable_rank(authoritative).tolist()
        )
        reference = canonical_identities != identity
        reconstructed_percentile = percentile_against(
            reconstructed[:520], reconstructed[:520][reference]
        )
        authoritative_percentile = percentile_against(
            authoritative[:520], authoritative[:520][reference]
        )
        percentile_exact = bool(
            np.array_equal(reconstructed_percentile, authoritative_percentile)
        )
        components[(identity, checkpoint, seed)] = {
            "pooled": pooled,
            "attention": attention,
            "combined": authoritative,
        }
        parity_rows.append(
            {
                "identity_id": identity,
                "checkpoint": checkpoint,
                "seed": seed,
                "images": len(authoritative),
                "maximum_raw_score_difference": maximum,
                "raw_score_parity": raw_exact,
                "rank_parity": rank_exact,
                "canonical520_percentile_parity": percentile_exact,
                "state_identity_match": str(payload["identity_id"]) == identity,
                "state_checkpoint_match": int(payload["checkpoint"]) == checkpoint,
                "state_seed_match": int(payload["seed"]) == seed,
            }
        )
        structure, profiles = _state_structure_rows(
            identity,
            checkpoint,
            seed,
            pooled[:520],
            attention[:520],
            authoritative[:520],
            canonical_identities,
        )
        structure_rows.append(structure)
        profile_rows.extend(profiles)
        del head
    return components, parity_rows, structure_rows, profile_rows


def audit_direct_pooled(
    components: Components,
    state_paths: Mapping[ComponentKey, Path],
    pooled_features: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare an independent pooled cache with the algebraic score residual."""
    indices = torch.arange(pooled_features.shape[0], dtype=torch.int64, device=device)
    rows: list[dict[str, Any]] = []
    all_differences: list[np.ndarray] = []
    all_rounding: list[np.ndarray] = []
    for key in sorted(
        state_paths, key=lambda item: (item[0].encode(), item[1], item[2])
    ):
        identity, checkpoint, seed = key
        payload = torch.load(state_paths[key], map_location="cpu", weights_only=False)
        state = payload["readout_state"]
        direct = (
            linear_scores(
                pooled_features,
                indices,
                torch.as_tensor(state["linear_weight"], device=device),
                torch.as_tensor(state["linear_bias"], device=device),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        residual = components[key]["pooled"]
        attention = components[key]["attention"]
        authoritative = components[key]["combined"]
        difference = np.abs(direct - residual)
        execution_sum = (
            torch.as_tensor(direct, dtype=torch.float32)
            .add(torch.as_tensor(attention, dtype=torch.float32))
            .numpy()
            .astype(np.float64)
        )
        high_precision_sum = direct + attention
        rounding = np.abs(high_precision_sum - execution_sum)
        positive_rounding = rounding[rounding > 0]
        rows.append(
            {
                "identity_id": identity,
                "checkpoint": checkpoint,
                "seed": seed,
                "images": len(direct),
                "maximum_abs_b_direct_minus_b_residual": float(difference.max()),
                "median_abs_b_direct_minus_b_residual": float(np.median(difference)),
                "P90_abs_b_direct_minus_b_residual": float(
                    np.quantile(difference, 0.90)
                ),
                "P99_abs_b_direct_minus_b_residual": float(
                    np.quantile(difference, 0.99)
                ),
                "Spearman_b_direct_b_residual": float(
                    spearmanr(direct, residual).statistic
                ),
                "fraction_nonexact": float(np.mean(direct != residual)),
                "execution_dtype": "BF16 pooled/features and weights; FP32 output",
                "addition_dtype": "FP32",
                "observed_rounding_quantum": (
                    float(positive_rounding.min()) if len(positive_rounding) else 0.0
                ),
                "historical_execution_raw_exact": bool(
                    np.array_equal(execution_sum, authoritative)
                ),
                "historical_execution_maximum_difference": float(
                    np.max(np.abs(execution_sum - authoritative))
                ),
                "float64_reconstruction_raw_exact": bool(
                    np.array_equal(high_precision_sum, authoritative)
                ),
                "float64_reconstruction_maximum_difference": float(
                    np.max(np.abs(high_precision_sum - authoritative))
                ),
            }
        )
        all_differences.append(difference)
        all_rounding.append(rounding)
    differences = np.concatenate(all_differences)
    rounding = np.concatenate(all_rounding)
    positive_rounding = rounding[rounding > 0]
    aggregate = {
        "states": len(rows),
        "values": int(differences.size),
        "maximum_abs_b_direct_minus_b_residual": float(differences.max()),
        "median_abs_b_direct_minus_b_residual": float(np.median(differences)),
        "P90_abs_b_direct_minus_b_residual": float(np.quantile(differences, 0.90)),
        "P99_abs_b_direct_minus_b_residual": float(np.quantile(differences, 0.99)),
        "fraction_nonexact": float(np.mean(differences != 0)),
        "execution_dtype": "BF16 pooled/features and weights; FP32 output",
        "addition_dtype": "FP32",
        "observed_rounding_quantum": (
            float(positive_rounding.min()) if len(positive_rounding) else 0.0
        ),
        "all_historical_execution_raw_exact": all(
            bool(row["historical_execution_raw_exact"]) for row in rows
        ),
        "all_float64_reconstruction_raw_exact": all(
            bool(row["float64_reconstruction_raw_exact"]) for row in rows
        ),
    }
    return rows, aggregate


def score_maps(
    components: Components,
    checkpoint_map: Mapping[str, int],
    alpha_map: Mapping[str, float],
    seed_indices: Sequence[int],
    identities: Sequence[str],
    seeds: Sequence[int],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Build held-out-fold E3 percentile maps without averaging raw seed scores."""
    maps: dict[str, dict[str, float]] = {}
    for identity in identities:
        checkpoint = int(checkpoint_map[identity])
        alpha = float(alpha_map[identity])
        reference = canonical_identities != identity
        percentiles = []
        for seed_index in seed_indices:
            component = components[(identity, checkpoint, int(seeds[seed_index]))]
            raw = component["pooled"][:520] + alpha * component["attention"][:520]
            percentiles.append(percentile_against(raw, raw[reference]))
        ensemble = np.mean(np.stack(percentiles), axis=0, dtype=np.float64)
        maps[identity] = {
            str(name): float(score)
            for name, score in zip(canonical_names, ensemble, strict=True)
        }
    return maps


def endpoint_vectors(
    maps: Mapping[str, Mapping[str, float]],
    exposures: Sequence[Mapping[str, Any]],
    identities: Sequence[str],
) -> dict[str, Any]:
    """Return direct, offset and conditional gross identity vectors."""
    direct = direct_identity_loss_vector(exposures, maps, identities)
    offset_rows, offset = offset_guard(exposures, maps, identities)
    offset_vector = np.asarray([float(row["abs_S_b"]) for row in offset_rows])
    gross_exposures = filter_direct_ordinal_exposures(exposures, minimum_distance=3)
    gross = direct_identity_loss_vector(gross_exposures, maps, identities)
    return {
        "direct": direct,
        "offset": offset_vector,
        "gross": gross,
        "offset_rows": offset_rows,
        "direct_loss": float(direct.mean()),
        "G": float(offset),
        "gross_loss": float(gross.mean()),
    }


def _level_metrics(
    components: Components,
    checkpoint_map: Mapping[str, int],
    alpha: float,
    identities: Sequence[str],
    seeds: Sequence[int],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    theta: np.ndarray,
    folds: Mapping[str, Any],
) -> dict[str, float]:
    rows = []
    for identity in identities:
        seed_scores = []
        checkpoint = int(checkpoint_map[identity])
        for seed in seeds:
            item = components[(identity, checkpoint, int(seed))]
            seed_scores.append(
                item["pooled"][:520] + float(alpha) * item["attention"][:520]
            )
        rows.append(
            ensemble_percentile_fold_result(
                canonical_names,
                canonical_identities,
                theta,
                folds[identity],
                seed_scores,
            )
        )
    aggregate = aggregate_level_fold_results(rows)
    return {
        "IB_LTE10": float(aggregate["IB_LTE10"]),
        "IB_FrameLTE10": float(aggregate["IB_FrameLTE10"]),
        "IdentityOffsetMAE": float(aggregate["IdentityOffsetMAE"]),
    }


def evaluate_alpha_surface(
    components: Components,
    checkpoint_map: Mapping[str, int],
    identities: Sequence[str],
    seeds: Sequence[int],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    theta: np.ndarray,
    folds: Mapping[str, Any],
    panels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[float, dict[str, Any]]]]:
    """Evaluate the preregistered alpha grid on FULL checkpoint states."""
    rows: list[dict[str, Any]] = []
    results: dict[str, dict[float, dict[str, Any]]] = {name: {} for name in panels}
    level_cache: dict[float, dict[str, float]] = {}
    for alpha in ALPHAS:
        alpha_map = {identity: float(alpha) for identity in identities}
        maps = score_maps(
            components,
            checkpoint_map,
            alpha_map,
            (0, 1, 2),
            identities,
            seeds,
            canonical_names,
            canonical_identities,
        )
        level_cache[float(alpha)] = _level_metrics(
            components,
            checkpoint_map,
            float(alpha),
            identities,
            seeds,
            canonical_names,
            canonical_identities,
            theta,
            folds,
        )
        for panel_name, exposures in panels.items():
            evaluated = endpoint_vectors(maps, exposures, identities)
            results[panel_name][float(alpha)] = evaluated
            rows.append(
                {
                    "panel": panel_name,
                    "alpha": float(alpha),
                    "direct_loss": evaluated["direct_loss"],
                    "offset_G": evaluated["G"],
                    "gross_d3_loss": evaluated["gross_loss"],
                    **level_cache[float(alpha)],
                }
            )
    for panel_name in panels:
        reference = results[panel_name][1.0]
        for row in rows:
            if row["panel"] == panel_name:
                row["delta_direct_vs_alpha1"] = (
                    row["direct_loss"] - reference["direct_loss"]
                )
                row["delta_G_vs_alpha1"] = row["offset_G"] - reference["G"]
                row["delta_gross_vs_alpha1"] = (
                    row["gross_d3_loss"] - reference["gross_loss"]
                )
    return rows, results


def _select_d(direct: np.ndarray) -> int:
    return min(
        range(len(ALPHAS)),
        key=lambda index: (direct[index], abs(ALPHAS[index] - 1.0), ALPHAS[index]),
    )


def _select_g(direct: np.ndarray, offset: np.ndarray) -> int:
    eligible = np.flatnonzero(direct <= direct[8] + 0.005)
    return min(
        map(int, eligible),
        key=lambda index: (
            offset[index],
            direct[index],
            abs(ALPHAS[index] - 1.0),
            ALPHAS[index],
        ),
    )


def select_policies(
    loio_results: Mapping[float, Mapping[str, Any]], identities: Sequence[str]
) -> dict[str, Any]:
    """Freeze FULL and honest leave-one-identity-out alpha maps."""
    direct = np.stack([loio_results[float(alpha)]["direct"] for alpha in ALPHAS])
    offset = np.stack([loio_results[float(alpha)]["offset"] for alpha in ALPHAS])
    full_d = _select_d(direct.mean(axis=1))
    full_g = _select_g(direct.mean(axis=1), offset.mean(axis=1))
    maps = {"D": {}, "G": {}}
    rows: list[dict[str, Any]] = []
    for heldout, identity in enumerate(identities):
        retained = [index for index in range(len(identities)) if index != heldout]
        d_index = _select_d(direct[:, retained].mean(axis=1))
        g_index = _select_g(
            direct[:, retained].mean(axis=1), offset[:, retained].mean(axis=1)
        )
        maps["D"][identity] = float(ALPHAS[d_index])
        maps["G"][identity] = float(ALPHAS[g_index])
        for policy, index in (("D", d_index), ("G", g_index)):
            rows.append(
                {
                    "record_type": "identity_selection",
                    "policy": policy,
                    "identity_id": identity,
                    "alpha": float(ALPHAS[index]),
                    "heldout_direct_loss": float(direct[index, heldout]),
                    "heldout_offset_abs_S": float(offset[index, heldout]),
                    "lower_boundary": index == 0,
                    "upper_boundary": index == len(ALPHAS) - 1,
                }
            )
    return {
        "full_alpha": {"D": float(ALPHAS[full_d]), "G": float(ALPHAS[full_g])},
        "loio_alpha_maps": maps,
        "rows": rows,
    }


def evaluate_policy_uq(
    components: Components,
    checkpoint_map: Mapping[str, int],
    policy_maps: Mapping[str, Mapping[str, float]],
    identities: Sequence[str],
    seeds: Sequence[int],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Evaluate fixed policy maps for all paired seed triples and bootstrap them."""
    reference_values = {metric: [] for metric in ("direct", "offset", "gross")}
    candidate_values = {
        policy: {metric: [] for metric in reference_values} for policy in policy_maps
    }
    standard_details: dict[str, dict[str, Any]] = {}
    reference_map = {identity: 1.0 for identity in identities}
    for triple in SEED_TRIPLES:
        reference_maps = score_maps(
            components,
            checkpoint_map,
            reference_map,
            triple,
            identities,
            seeds,
            canonical_names,
            canonical_identities,
        )
        reference = endpoint_vectors(reference_maps, exposures, identities)
        for metric, values in reference_values.items():
            values.append(reference[metric])
        for policy, alpha_map in policy_maps.items():
            maps = score_maps(
                components,
                checkpoint_map,
                alpha_map,
                triple,
                identities,
                seeds,
                canonical_names,
                canonical_identities,
            )
            evaluated = endpoint_vectors(maps, exposures, identities)
            for metric in reference_values:
                candidate_values[policy][metric].append(evaluated[metric])
            if triple == (0, 1, 2):
                standard_details[policy] = {
                    "reference": reference,
                    "candidate": evaluated,
                }
    uq: dict[str, dict[str, Any]] = {}
    for policy in policy_maps:
        uq[policy] = {}
        for metric, reference_rows in reference_values.items():
            candidate = np.stack(candidate_values[policy][metric])
            reference = np.stack(reference_rows)
            payload, _distribution = paired_uq(candidate, reference)
            uq[policy][metric] = payload
    return uq, standard_details
