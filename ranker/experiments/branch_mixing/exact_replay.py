"""Exact historical-token reconstruction and generic branch-scale replay."""

from __future__ import annotations

import copy
import itertools
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

from ranker.attention_lora import AttentionResidualHead
from ranker.experiments.branch_mixing.analysis import (
    Components,
    endpoint_vectors,
    score_maps,
)
from ranker.external_validation import percentile_against
from ranker.frozen_readouts import attention_scores, linear_scores
from ranker.last_block_finetune import ExactPrefixCache, load_last_block_template
from ranker.runtime_preflight import efficient_sdpa_only
from ranker.tune14 import paired_uq

SEED_TRIPLES = tuple(itertools.product(range(3), repeat=3))


@dataclass(frozen=True)
class MixingSystem:
    """One fixed inference-time branch mixing system."""

    name: str
    alpha: float | None
    lambda_value: float | None
    mode: str = "mixed"


def reconstruct_historical_l26(
    old_prefix: Path,
    extra_prefix: Path,
    model_dir: Path,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Reconstruct STEP48D-R1 L26 tokens and pooled features once in RAM."""
    started = time.perf_counter()
    prefix = ExactPrefixCache.load(old_prefix, extra_prefix)
    template = load_last_block_template(model_dir)
    block = copy.deepcopy(template.encoder.layers[-1]).to(device).eval()
    norm = copy.deepcopy(template.post_layernorm).to(device).eval()
    pool = copy.deepcopy(template.head).to(device).eval()
    block.self_attn.config._attn_implementation = "sdpa"
    tokens = torch.empty((710, 1024, 1152), dtype=torch.bfloat16, device=device)
    pooled = torch.empty((710, 1152), dtype=torch.bfloat16, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, 710, 8):
            stop = min(start + 8, 710)
            indices = torch.arange(start, stop, dtype=torch.int64, device=device)
            with efficient_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                pre_last = prefix.pre_last(indices, device)
                final = norm(block(pre_last, attention_mask=None))
                pooled_fp32 = F.normalize(pool(final).float(), p=2, dim=1)
            tokens[start:stop].copy_(final.to(torch.bfloat16))
            pooled[start:stop].copy_(pooled_fp32.to(torch.bfloat16))
    metadata = {
        "runtime_seconds": time.perf_counter() - started,
        "block_batch_size": 8,
        "only_transformer_block": "vision_model.encoder.layers.26",
        "attn_implementation": "sdpa",
        "sdpa_backend": "EFFICIENT_ATTENTION only",
        "autocast": "CUDA BF16",
        "post_layernorm": f"{type(norm).__module__}.{type(norm).__name__}",
        "pooling_head": f"{type(pool).__module__}.{type(pool).__name__}",
        "tokens_shape": list(tokens.shape),
        "tokens_dtype": str(tokens.dtype),
        "tokens_bytes": tokens.numel() * tokens.element_size(),
        "pooled_shape": list(pooled.shape),
        "pooled_dtype": str(pooled.dtype),
        "pooled_bytes": pooled.numel() * pooled.element_size(),
        "peak_GPU_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_GPU_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    del block, norm, pool, template
    return prefix.names, tokens, pooled, metadata


def load_true_components(
    state_paths: Mapping[tuple[str, int, int], Path],
    tokens: torch.Tensor,
    pooled: torch.Tensor,
    canonical_identities: np.ndarray,
    device: torch.device,
) -> tuple[Components, list[dict[str, Any]], dict[str, Any]]:
    """Evaluate both real JOINT4 branches with historical execution semantics."""
    components: Components = {}
    rows: list[dict[str, Any]] = []
    residual_errors: list[np.ndarray] = []
    indices = torch.arange(710, dtype=torch.int64, device=device)
    exact_scores = 0
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
                initial_query_norm=math.sqrt(128.0),
                query_count=4,
                value_dimension=256,
                zero_output=True,
            )
            .to(device)
            .eval()
        )
        head.load_state_dict(state["head_state_dict"], strict=True)
        attention = torch.cat(
            (
                attention_scores(head, tokens, indices[:520], batch_size=128),
                attention_scores(head, tokens, indices[520:], batch_size=128),
            )
        )
        pooled_score = linear_scores(
            pooled,
            indices,
            torch.as_tensor(state["linear_weight"], device=device),
            torch.as_tensor(state["linear_bias"], device=device),
        )
        combined = pooled_score + attention
        pooled_np = pooled_score.detach().cpu().numpy().astype(np.float64)
        attention_np = attention.detach().cpu().numpy().astype(np.float64)
        combined_np = combined.detach().cpu().numpy().astype(np.float64)
        authoritative = np.asarray(payload["raw_scores"], dtype=np.float64)
        difference = np.abs(combined_np - authoritative)
        exact_scores += int(np.count_nonzero(difference == 0.0))
        rank_exact = bool(
            np.array_equal(
                np.argsort(combined_np, kind="stable"),
                np.argsort(authoritative, kind="stable"),
            )
        )
        reference = canonical_identities != identity
        percentile_exact = bool(
            np.array_equal(
                percentile_against(combined_np[:520], combined_np[:520][reference]),
                percentile_against(authoritative[:520], authoritative[:520][reference]),
            )
        )
        residual = np.abs(pooled_np - (authoritative - attention_np))
        residual_errors.append(residual)
        components[(identity, checkpoint, seed)] = {
            "pooled": pooled_np,
            "attention": attention_np,
            "combined": authoritative,
        }
        rows.append(
            {
                "identity_id": identity,
                "checkpoint": checkpoint,
                "seed": seed,
                "score_count": len(authoritative),
                "exact_score_count": int(np.count_nonzero(difference == 0.0)),
                "maximum_raw_score_difference": float(difference.max()),
                "rank_parity": rank_exact,
                "percentile_parity": percentile_exact,
                "maximum_pooled_residual_difference": float(residual.max()),
            }
        )
        del head
    residual = np.concatenate(residual_errors)
    summary = {
        "states": len(rows),
        "score_comparisons": len(rows) * 710,
        "exact_score_count": exact_scores,
        "raw_exact_state_count": sum(
            row["maximum_raw_score_difference"] == 0.0 for row in rows
        ),
        "rank_exact_count": sum(bool(row["rank_parity"]) for row in rows),
        "percentile_exact_count": sum(bool(row["percentile_parity"]) for row in rows),
        "maximum_raw_score_difference": max(
            float(row["maximum_raw_score_difference"]) for row in rows
        ),
        "pooled_residual": {
            "maximum": float(residual.max()),
            "median": float(np.median(residual)),
            "P90": float(np.quantile(residual, 0.90)),
            "P99": float(np.quantile(residual, 0.99)),
            "nonexact_count": int(np.count_nonzero(residual)),
        },
    }
    return components, rows, summary


def branch_structure(
    components: Components,
    selected_keys: Sequence[tuple[str, int, int]],
    canonical_identities: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure true branch scale and within/between allocation on selected states."""
    rows: list[dict[str, Any]] = []
    identity_values = sorted(set(canonical_identities.tolist()), key=str.encode)
    for key in selected_keys:
        identity, checkpoint, seed = key
        item = components[key]
        row: dict[str, Any] = {
            "identity_id": identity,
            "checkpoint": checkpoint,
            "seed": seed,
        }
        for branch in ("pooled", "attention", "combined"):
            values = item[branch][:520]
            fitted = np.empty_like(values)
            for value in identity_values:
                mask = canonical_identities == value
                fitted[mask] = values[mask].mean()
            total = float(values.var(ddof=0))
            between = float(fitted.var(ddof=0))
            within = float((values - fitted).var(ddof=0))
            row[f"SD_{branch}"] = float(values.std(ddof=0))
            row[f"between_fraction_{branch}"] = between / total if total else 0.0
            row[f"within_fraction_{branch}"] = within / total if total else 0.0
        row["Spearman_pooled_attention"] = float(
            spearmanr(item["pooled"][:520], item["attention"][:520]).statistic
        )
        row["Spearman_attention_combined"] = float(
            spearmanr(item["attention"][:520], item["combined"][:520]).statistic
        )
        rows.append(row)
    aggregate = {
        "states": len(rows),
        **{
            field: float(np.mean([float(row[field]) for row in rows]))
            for field in rows[0]
            if field
            not in {
                "identity_id",
                "checkpoint",
                "seed",
            }
        },
    }
    aggregate["kappa_true"] = aggregate["SD_pooled"] / aggregate["SD_attention"]
    return rows, aggregate


def system_raw(component: Mapping[str, np.ndarray], system: MixingSystem) -> np.ndarray:
    """Return one raw score array for a fixed branch-mixing system."""
    if system.mode == "pooled_only":
        return component["pooled"]
    if system.mode == "attention_only":
        return component["attention"]
    assert system.alpha is not None
    return component["pooled"] + float(system.alpha) * component["attention"]


def system_score_maps(
    components: Components,
    checkpoint_map: Mapping[str, int],
    system_map: Mapping[str, MixingSystem],
    seed_indices: Sequence[int],
    identities: Sequence[str],
    seeds: Sequence[int],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Build E3 maps for arbitrary fixed branch systems."""
    maps: dict[str, dict[str, float]] = {}
    for identity in identities:
        checkpoint = int(checkpoint_map[identity])
        reference = canonical_identities != identity
        percentiles = []
        for seed_index in seed_indices:
            item = components[(identity, checkpoint, int(seeds[seed_index]))]
            raw = system_raw(item, system_map[identity])[:520]
            percentiles.append(percentile_against(raw, raw[reference]))
        ensemble = np.mean(np.stack(percentiles), axis=0, dtype=np.float64)
        maps[identity] = {
            str(name): float(score)
            for name, score in zip(canonical_names, ensemble, strict=True)
        }
    return maps


def evaluate_systems(
    components: Components,
    checkpoint_map: Mapping[str, int],
    systems: Sequence[MixingSystem],
    identities: Sequence[str],
    seeds: Sequence[int],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    panels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Evaluate arbitrary branch systems on direct, offset and gross endpoints."""
    rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, dict[str, Any]]] = {panel: {} for panel in panels}
    for system in systems:
        maps = system_score_maps(
            components,
            checkpoint_map,
            {identity: system for identity in identities},
            (0, 1, 2),
            identities,
            seeds,
            canonical_names,
            canonical_identities,
        )
        for panel, exposures in panels.items():
            evaluated = endpoint_vectors(maps, exposures, identities)
            results[panel][system.name] = evaluated
            rows.append(
                {
                    "panel": panel,
                    "system": system.name,
                    "lambda": system.lambda_value
                    if system.lambda_value is not None
                    else "",
                    "alpha": system.alpha if system.alpha is not None else "",
                    "mode": system.mode,
                    "direct_loss": evaluated["direct_loss"],
                    "offset_G": evaluated["G"],
                    "gross_d3_loss": evaluated["gross_loss"],
                }
            )
    return rows, results


def _tie_key(system: MixingSystem) -> tuple[float, float, str]:
    if system.lambda_value is None or system.lambda_value <= 0:
        return math.inf, math.inf, system.name
    return abs(math.log2(system.lambda_value)), system.lambda_value, system.name


def select_system_policies(
    results: Mapping[str, Mapping[str, Any]],
    systems: Sequence[MixingSystem],
    identities: Sequence[str],
    reference_name: str,
) -> dict[str, Any]:
    """Select scale systems honestly across held-out identities."""
    direct = np.stack([results[system.name]["direct"] for system in systems])
    offset = np.stack([results[system.name]["offset"] for system in systems])
    reference_index = next(
        index for index, system in enumerate(systems) if system.name == reference_name
    )

    def choose_d(values: np.ndarray) -> int:
        return min(
            range(len(systems)),
            key=lambda index: (values[index], *_tie_key(systems[index])),
        )

    def choose_g(d_values: np.ndarray, g_values: np.ndarray) -> int:
        eligible = np.flatnonzero(d_values <= d_values[reference_index] + 0.005)
        return min(
            map(int, eligible),
            key=lambda index: (
                g_values[index],
                d_values[index],
                *_tie_key(systems[index]),
            ),
        )

    full_d = choose_d(direct.mean(axis=1))
    full_g = choose_g(direct.mean(axis=1), offset.mean(axis=1))
    maps: dict[str, dict[str, MixingSystem]] = {"D": {}, "G": {}}
    rows: list[dict[str, Any]] = []
    for heldout, identity in enumerate(identities):
        retained = [index for index in range(len(identities)) if index != heldout]
        d_index = choose_d(direct[:, retained].mean(axis=1))
        g_index = choose_g(
            direct[:, retained].mean(axis=1), offset[:, retained].mean(axis=1)
        )
        for policy, index in (("D", d_index), ("G", g_index)):
            system = systems[index]
            maps[policy][identity] = system
            rows.append(
                {
                    "policy": policy,
                    "identity_id": identity,
                    "system": system.name,
                    "lambda": system.lambda_value
                    if system.lambda_value is not None
                    else "",
                    "alpha": system.alpha if system.alpha is not None else "",
                    "heldout_direct_loss": float(direct[index, heldout]),
                    "heldout_offset_abs_S": float(offset[index, heldout]),
                }
            )
    return {
        "full": {"D": systems[full_d], "G": systems[full_g]},
        "loio_maps": maps,
        "rows": rows,
    }


def evaluate_system_policy_uq(
    components: Components,
    checkpoint_map: Mapping[str, int],
    policy_maps: Mapping[str, Mapping[str, MixingSystem]],
    reference: MixingSystem,
    identities: Sequence[str],
    seeds: Sequence[int],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Paired 27-seed-triple UQ for fixed scale-policy maps."""
    metrics = ("direct", "offset", "gross")
    reference_values = {metric: [] for metric in metrics}
    candidate_values = {
        policy: {metric: [] for metric in metrics} for policy in policy_maps
    }
    details: dict[str, dict[str, Any]] = {}
    reference_map = {identity: reference for identity in identities}
    for triple in SEED_TRIPLES:
        ref_maps = system_score_maps(
            components,
            checkpoint_map,
            reference_map,
            triple,
            identities,
            seeds,
            canonical_names,
            canonical_identities,
        )
        ref = endpoint_vectors(ref_maps, exposures, identities)
        for metric in metrics:
            reference_values[metric].append(ref[metric])
        for policy, system_map in policy_maps.items():
            maps = system_score_maps(
                components,
                checkpoint_map,
                system_map,
                triple,
                identities,
                seeds,
                canonical_names,
                canonical_identities,
            )
            evaluated = endpoint_vectors(maps, exposures, identities)
            for metric in metrics:
                candidate_values[policy][metric].append(evaluated[metric])
            if triple == (0, 1, 2):
                details[policy] = {"reference": ref, "candidate": evaluated}
    uq: dict[str, dict[str, Any]] = {}
    for policy in policy_maps:
        uq[policy] = {}
        for metric in metrics:
            payload, _ = paired_uq(
                np.stack(candidate_values[policy][metric]),
                np.stack(reference_values[metric]),
            )
            uq[policy][metric] = payload
    return uq, details


def changed_pair_orders(
    first: Mapping[str, Mapping[str, float]],
    second: Mapping[str, Mapping[str, float]],
    exposures: Sequence[Mapping[str, Any]],
) -> int:
    """Count directed exposure orders that change between two percentile maps."""
    changed = 0
    for exposure in exposures:
        identity = str(exposure["identity_id"])
        image = str(exposure["image_key"])
        for anchor in exposure["anchor_keys"]:
            anchor = str(anchor)
            first_sign = np.sign(first[identity][image] - first[identity][anchor])
            second_sign = np.sign(second[identity][image] - second[identity][anchor])
            changed += int(first_sign != second_sign)
    return changed


def old_grid_rows(
    *,
    published_path: Path,
    true_components: Components,
    wrong_components: Components,
    full_map: Mapping[str, int],
    identities: Sequence[str],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
    corrected: Mapping[float, Mapping[str, Any]],
    alphas: Sequence[float],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Replay published alpha rows against true and historical-wrong branches."""
    import csv

    with published_path.open("r", encoding="utf-8", newline="") as handle:
        published = {
            float(row["alpha"]): row
            for row in csv.DictReader(handle)
            if row["panel"] == "COMBINED"
        }
    rows = []
    for alpha in alphas:
        alpha_map = {identity: alpha for identity in identities}
        true_maps = score_maps(
            true_components,
            full_map,
            alpha_map,
            (0, 1, 2),
            identities,
            seeds,
            canonical_names,
            canonical_identities,
        )
        wrong_maps = score_maps(
            wrong_components,
            full_map,
            alpha_map,
            (0, 1, 2),
            identities,
            seeds,
            canonical_names,
            canonical_identities,
        )
        old = published[alpha]
        new = corrected[alpha]
        wrong = endpoint_vectors(wrong_maps, exposures, identities)
        rows.append(
            {
                "record_type": "surface",
                "alpha": alpha,
                "published_direct": float(old["direct_loss"]),
                "recomputed_published_direct": wrong["direct_loss"],
                "published_replay_direct_difference": (
                    wrong["direct_loss"] - float(old["direct_loss"])
                ),
                "corrected_direct": new["direct_loss"],
                "direct_difference": new["direct_loss"] - float(old["direct_loss"]),
                "published_G": float(old["offset_G"]),
                "recomputed_published_G": wrong["G"],
                "published_replay_G_difference": wrong["G"] - float(old["offset_G"]),
                "corrected_G": new["G"],
                "G_difference": new["G"] - float(old["offset_G"]),
                "changed_pair_orders": changed_pair_orders(
                    wrong_maps, true_maps, exposures
                ),
            }
        )
    return rows


def scale_systems(
    kappa: float, lambda_values: Sequence[float]
) -> list[MixingSystem]:
    """Construct the scale-calibrated pooled-to-attention system grid."""
    systems = [MixingSystem("pooled_only", 0.0, None, "pooled_only")]
    systems.extend(
        MixingSystem(f"lambda_{value:g}", value * kappa, value)
        for value in lambda_values
    )
    systems.extend(
        (
            MixingSystem("historical_reference", 1.0, 1.0 / kappa),
            MixingSystem("attention_only", None, None, "attention_only"),
        )
    )
    return systems


def classify_scale_result(
    selection: Mapping[str, Any], uq: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Apply fixed direct, offset, and gross classification gates to scale UQ."""
    direct = uq["D"]["direct"]
    offset_d = uq["D"]["offset"]
    gross_d = uq["D"]["gross"]
    improved = sum(float(value) < 0 for value in direct["identity_deltas"])
    offset_harm = offset_d["point_delta"] > 0 and offset_d["CI90_lower"] > 0
    gross_harm = gross_d["point_delta"] >= 0.005 and gross_d["CI90_lower"] > 0
    winner = (
        direct["point_delta"] <= -0.005
        and direct["CI90_upper"] < 0
        and direct["P_delta_lt_0"] >= 0.90
        and improved >= 10
        and not offset_harm
        and not gross_harm
    )
    equivalent = (
        direct["CI90_lower"] > -0.005
        and direct["CI90_upper"] < 0.005
        and not offset_harm
        and not gross_harm
    )
    policy_d = {
        "FULL_system": selection["full"]["D"].name,
        "classification": (
            "development_winner"
            if winner
            else "practical_equivalence"
            if equivalent
            else "not_promoted"
        ),
        "development_winner": winner,
        "practical_equivalence": equivalent,
        "improved_identities": improved,
        "resolved_offset_harm": offset_harm,
        "resolved_gross_harm": gross_harm,
    }
    offset = uq["G"]["offset"]
    direct_g = uq["G"]["direct"]
    gross_g = uq["G"]["gross"]
    improvement = (
        offset["point_delta"] < 0
        and offset["CI90_upper"] < 0
        and offset["P_delta_lt_0"] >= 0.90
    )
    direct_safe = direct_g["CI90_upper"] <= 0.005
    gross_safe = not (gross_g["point_delta"] >= 0.005 and gross_g["CI90_lower"] > 0)
    policy_g = {
        "FULL_system": selection["full"]["G"].name,
        "classification": (
            "resolved_offset_improvement"
            if improvement and direct_safe and gross_safe
            else "offset_improvement_guard_failed"
            if improvement
            else "no_resolved_offset_improvement"
        ),
        "resolved_offset_improvement": improvement,
        "direct_safety": direct_safe,
        "gross_safety": gross_safe,
    }
    result = (
        "scale_calibrated_development_winner"
        if winner
        else "scale_calibrated_offset_improvement"
        if improvement and direct_safe and gross_safe
        else "scale_calibrated_practical_equivalence"
        if equivalent
        else "scale_calibrated_not_promoted"
    )
    return policy_d, policy_g, result


def main4_context(
    *,
    paths: Mapping[tuple[str, int], Path],
    main4_checkpoints: Mapping[str, int],
    true_components: Components,
    l26_checkpoints: Mapping[str, int],
    identities: Sequence[str],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    seed_triples: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    """Compare retained MAIN4 training context with exact JOINT4 branch paths."""
    raw = {
        key: np.asarray(
            torch.load(path, map_location="cpu", weights_only=False)["raw_scores"],
            dtype=np.float64,
        )
        for key, path in paths.items()
    }
    attention = MixingSystem("attention_only", None, None, "attention_only")
    reference = MixingSystem("historical_reference", 1.0, 1.0)
    candidate_rows: list[np.ndarray] = []
    attention_rows: list[np.ndarray] = []
    total_rows: list[np.ndarray] = []
    standard: dict[str, Any] = {}
    for triple in seed_triples:
        main_maps: dict[str, dict[str, float]] = {}
        for identity in identities:
            selected = canonical_identities != identity
            pct = [
                percentile_against(
                    raw[(identity, int(seeds[index]))][:520],
                    raw[(identity, int(seeds[index]))][:520][selected],
                )
                for index in triple
            ]
            ensemble = np.mean(np.stack(pct), axis=0, dtype=np.float64)
            main_maps[identity] = {
                str(name): float(value)
                for name, value in zip(canonical_names, ensemble, strict=True)
            }
        attention_maps = system_score_maps(
            true_components,
            l26_checkpoints,
            {identity: attention for identity in identities},
            triple,
            identities,
            seeds,
            canonical_names,
            canonical_identities,
        )
        total_maps = system_score_maps(
            true_components,
            l26_checkpoints,
            {identity: reference for identity in identities},
            triple,
            identities,
            seeds,
            canonical_names,
            canonical_identities,
        )
        main_endpoint = endpoint_vectors(main_maps, exposures, identities)
        attention_endpoint = endpoint_vectors(attention_maps, exposures, identities)
        total_endpoint = endpoint_vectors(total_maps, exposures, identities)
        candidate_rows.append(main_endpoint["direct"])
        attention_rows.append(attention_endpoint["direct"])
        total_rows.append(total_endpoint["direct"])
        if tuple(triple) == (0, 1, 2):
            standard = {
                "MAIN4": main_endpoint,
                "JOINT4_ATTENTION_ONLY": attention_endpoint,
                "JOINT4_TOTAL": total_endpoint,
            }
    inference_uq, _ = paired_uq(np.stack(attention_rows), np.stack(total_rows))
    context_uq, _ = paired_uq(np.stack(candidate_rows), np.stack(attention_rows))
    return {
        "inference_contrast": {
            "name": "JOINT4_ATTENTION_ONLY-JOINT4_TOTAL",
            **inference_uq,
        },
        "training_context_contrast": {
            "name": "MAIN4_TRAINED-JOINT4_ATTENTION_ONLY",
            **context_uq,
        },
        "point_losses": {
            "JOINT4_TOTAL": standard["JOINT4_TOTAL"]["direct_loss"],
            "JOINT4_ATTENTION_ONLY": standard["JOINT4_ATTENTION_ONLY"]["direct_loss"],
            "MAIN4_TRAINED": standard["MAIN4"]["direct_loss"],
        },
        "compatibility": {
            "identity_folds": "exact",
            "seeds": "exact",
            "token_source": "L26 exact",
            "objective": "HARD_REL_M75 exact",
            "optimizer": "ProdigyPlusScheduleFree 2.0.1 exact",
            "parameter_group_topology": "one readout group exact",
            "checkpoint_maps": "different",
            "checkpoint_confounding": True,
            "training_context_interpretation": "descriptive",
            "MAIN4_checkpoint_map": dict(main4_checkpoints),
            "JOINT4_checkpoint_map": dict(l26_checkpoints),
        },
    }
