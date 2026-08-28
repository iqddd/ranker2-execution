"""STEP61A: provenance-valid full L10-L26 block-skip screen."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ranker.attention_runtime import load_vision_model
from ranker.data import load_completed_ranked_groups_range, load_ranked_groups
from ranker.experiments.image_heldout_horizon_transfer import (
    build_image_heldout_fold,
    mapping_digest,
)
from ranker.experiments.image_heldout_horizon_transfer.certification import (
    endpoint_coverage,
    partition_anchor_exposures,
)
from ranker.experiments.layer_screen.analysis import fivefold_assignment, fold_inventory
from ranker.experiments.layer_screen.execution import (
    eager_lineage_diagnostic,
    extract_l10_workspace,
    materialize_streamed_l26,
    streamed_l26_digest,
    tensor_content_digest,
)
from ranker.experiments.layer_screen.step61_analysis import (
    all_seed_triple_vectors,
    endpoint_fold_loss,
    level_diagnostics,
    select_lofo_maps,
    selected_fold_maps,
)
from ranker.experiments.layer_screen.training import (
    backend_matched_l26_first_update_parity,
    build_screen,
    capture_all_scores,
    run_multihead_trajectory,
    trajectory_from_payload,
    trajectory_payload,
)
from ranker.frozen_readouts import FrozenFeatureRuntime
from ranker.io import atomic_json, atomic_text, read_csv, sha256_file
from ranker.last_block_finetune import ExactPrefixCache
from ranker.level_evaluation import load_canonical_level_targets, load_tune_rank_identity_split
from ranker.project_ledger import append_project_event, read_project_events, verify_project_ledger
from ranker.reproducibility import tensor_state_hash
from ranker.step import Artifact, StepRun, TimeBudget, Workspace
from ranker.step_registry import STEPS
from ranker.tune14 import paired_uq
from ranker.validation_resolution import direct_ordinal_exposures

STEP = STEPS["61a"]
SIGNATURE = "STEP61A_FULL_L10_L25_BLOCK_SKIP_SCAN_ALL_EFFICIENT_V9"
SEEDS = (20260803, 20260804, 20260805)
LAYERS = tuple(range(10, 27))
CHECKPOINTS = (0, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512)


def _artifact(key: str, kind: str, fields: set[str], *, terminal: bool = False) -> Artifact:
    return Artifact(STEP.artifact_name(key), kind, frozenset(fields), terminal=terminal)


ARTIFACTS = (
    _artifact("contract", "json", {"signature", "training", "layer_semantics", "endpoints"}),
    _artifact("driver_source", "text", set()),
    _artifact("input_provenance", "json", {"features", "fold_mapping", "prohibited_inputs"}),
    _artifact("architecture_audit", "csv", {"layer", "tensor_source", "post_layernorm", "token_count", "width"}),
    _artifact("initialization_parity", "csv_union", {"record_type", "seed", "pass"}),
    _artifact("evaluation_parity", "csv_union", {"record_type", "fold", "seed", "pass"}),
    _artifact("runtime_hardware", "json", {"local_pilot_seconds", "projected_local_total_seconds", "analytical_hardware"}),
    _artifact("trajectory_journal_audit", "json", {"completed_units", "expected_units", "journal"}),
    _artifact("scalar_scores", "npz", {"raw_scores", "seeds", "folds", "checkpoints", "layers", "image_ids"}),
    _artifact("endpoint_closure", "json", {"seen_anchor", "strict_unseen_anchor", "closure_pass"}),
    _artifact("fixed_surface", "csv", {"layer", "checkpoint", "endpoint", "direct", "G", "gross"}),
    _artifact("layerwise_lofo", "csv_union", {"record_type", "layer", "fold", "checkpoint", "direct"}),
    _artifact("joint_lofo", "csv", {"fold", "layer", "checkpoint", "heldout_direct"}),
    _artifact("selected_scores", "npz", {"raw_scores", "systems", "checkpoint_maps", "layers", "seeds"}),
    _artifact("contrasts", "csv_union", {"record_type", "endpoint", "metric", "delta"}),
    _artifact("uq", "csv_union", {"record_type", "endpoint", "metric", "point_delta", "CI90_lower", "CI90_upper"}),
    _artifact("guards", "csv", {"metric", "point_delta", "CI90_lower", "CI90_upper", "resolved_harm"}),
    _artifact("level_diagnostics", "csv", {"system", "IdentityOffsetMAE", "IB_FrameLTE10", "beta", "Pearson"}),
    _artifact("boundary_audit", "csv", {"layer", "aggregate_best_checkpoint", "cp512_minus_cp448", "LOFO_cp512_count"}),
    _artifact("policy_decisions", "json", {"STEP61A_result", "screen_lead", "layer_promoted", "historical_margin_role"}),
    _artifact("gates", "json", {"STEP61A_result", "gates"}),
    _artifact("ledger_update_audit", "json", {"head_before", "head_after", "chain_integrity_pass"}),
    _artifact("summary", "json", {"signature", "STEP61A_result", "trajectories_completed", "training_performed"}, terminal=True),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=STEP.output_dir)
    result.add_argument("--workspace", type=Path, default=Path("_work/step61a_full_l10_l25_block_skip_scan_v9"))
    result.add_argument("--feature-workspace", type=Path, default=Path("_work/step61a_full_l10_l25_features"))
    result.add_argument("--restart-workspace", action="store_true")
    result.add_argument("--hardware", choices=("auto", "remote"), default="auto")
    result.add_argument("--local-runtime-evidence", type=Path)
    result.add_argument("--reconstruct-features", action="store_true")
    result.add_argument("--image-dir", type=Path, default=Path("images"))
    result.add_argument("--source-state", type=Path, default=Path("ranking_state.json"))
    result.add_argument("--within-state", type=Path, default=Path("ranking_state_within_identity.json"))
    result.add_argument("--overlap-state", type=Path, default=Path("ranking_state_overlap_anchor.json"))
    result.add_argument("--theta", type=Path, default=STEPS["44b"].artifact("theta"))
    result.add_argument("--split", type=Path, default=STEPS["45"].artifact("split"))
    result.add_argument("--old-prefix", type=Path, default=STEPS["43a"].artifact("token_cache"))
    result.add_argument("--extra-prefix", type=Path, default=STEPS["43c0_r1"].artifact("tokens"))
    result.add_argument("--model-dir", type=Path, default=Path("artifacts/step26_frozen_encoder_bakeoff/official_model_files/siglip2-so400m-patch16-512"))
    result.add_argument("--step56a-scores", type=Path, default=STEPS["56a"].artifact("scalar_scores"))
    result.add_argument("--step56a-fold-inventory", type=Path, default=STEPS["56a"].artifact("fold_inventory"))
    result.add_argument("--step60a-scores", type=Path, default=STEPS["60a"].artifact("scalar_scores"))
    result.add_argument("--step60a-init", type=Path, default=STEPS["60a"].artifact("initialization_parity"))
    result.add_argument("--inventory", type=Path, default=Path("PROJECT_INVENTORY.md"))
    return result


def _prepare_features(args: argparse.Namespace, device: torch.device) -> tuple[list[str], torch.Tensor, torch.Tensor, Any, dict[str, Any]]:
    feature_dir = args.feature_workspace
    l10_path = feature_dir / "L10_RESIDUAL.npy"
    pooled_path = feature_dir / "L26_POOLED.npy"
    manifest_path = feature_dir / "FEATURE_MANIFEST.json"
    if l10_path.is_file() and pooled_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extraction = manifest.get("extraction", {})
        if extraction.get("signature") == "STEP61A_ALL_EFFICIENT_FEATURES_V4":
            names = list(map(str, manifest["names"]))
            residual = torch.from_numpy(np.load(l10_path, mmap_mode="r"))
            pooled = (
                torch.from_numpy(np.load(pooled_path, allow_pickle=False).copy())
                .view(torch.bfloat16)
                .to(device)
            )
            if tensor_content_digest(residual) != extraction["L10_content_SHA256"]:
                raise RuntimeError("All-efficient L10 scratch digest mismatch.")
            if tensor_content_digest(pooled) != extraction["pooled_content_SHA256"]:
                raise RuntimeError("All-efficient pooled scratch digest mismatch.")
            model, _processor = load_vision_model(args.model_dir, device)
            model.cpu()
            torch.cuda.empty_cache()
            return names, residual, pooled, model, manifest
    if args.hardware == "remote" and not args.reconstruct_features:
        raise FileNotFoundError("Remote execution requires prepared L10/pooled scratch inputs.")
    feature_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        seed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = list(map(str, seed_manifest["names"]))
    else:
        prefix = ExactPrefixCache.load(args.old_prefix, args.extra_prefix)
        names = list(prefix.names)
        del prefix
    model, residual, pooled, extraction = extract_l10_workspace(
        names=names,
        image_dir=args.image_dir,
        model_dir=args.model_dir,
        workspace_file=l10_path,
        pooled_workspace_file=pooled_path,
        device=device,
    )
    manifest = {
        "names": names,
        "extraction": extraction,
        "source_digests": {
            "old_PRE_LAST": sha256_file(args.old_prefix) if args.old_prefix.is_file() else "not_used_remote_reconstruction",
            "extra_PRE_LAST": sha256_file(args.extra_prefix) if args.extra_prefix.is_file() else "not_used_remote_reconstruction",
            "model_weights": sha256_file(args.model_dir / "model.safetensors"),
        },
    }
    atomic_json(manifest_path, manifest)
    gc.collect()
    torch.cuda.empty_cache()
    model.cpu()
    return names, residual, pooled, model, manifest


def _state_hash(screen: torch.nn.Module) -> str:
    return tensor_state_hash({name: value.detach().cpu() for name, value in screen.state_dict().items() if name.startswith("heads.")})


def _architecture_rows() -> list[dict[str, Any]]:
    return [
        {
            "layer": layer,
            "tensor_source": f"vision_model.encoder.layers.{layer} output",
            "post_layernorm": "vision_model.post_layernorm applied to head input only",
            "residual_to_next_block": "unnormalized",
            "token_count": 1024,
            "width": 1152,
            "head_input_dtype": "torch.bfloat16",
            "pooled_source": "shared all-efficient L26 pooled BF16",
            "historical_STEP56A_index": layer,
            "training_backend": "SDPA EFFICIENT_ATTENTION only",
        }
        for layer in LAYERS
    ]


def _inventory_update(path: Path, summary: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    fact = (
        f"- STEP61A provenance-valid L10…L26 E3 screen: `{summary['STEP61A_result']}`; "
        f"primary endpoint `SEEN_ANCHOR`, lead `{summary.get('screen_lead', 'none')}`. "
        "The result is exploratory and does not promote a layer or change FULL192/external baseline."
    )
    if "STEP61A provenance-valid L10…L26 E3 screen:" not in text:
        anchor = "### Диагностически"
        position = text.index("\n", text.index(anchor)) + 1
        text = text[:position] + "\n" + fact + text[position:]
    source = "- `artifacts/step61a_full_l10_l25_block_skip_scan/STEP61A_SUMMARY.json` — provenance-valid full layer/block-skip screen."
    if source not in text:
        text = text.rstrip() + "\n" + source + "\n"
    if "_STATUS_START" in text or "_STATUS_END" in text:
        raise RuntimeError("Inventory marker blocks are forbidden.")
    atomic_text(path, text)


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("STEP61A requires a BF16 CUDA GPU.")
    if version("prodigy-plus-schedule-free") != "2.0.1":
        raise RuntimeError("STEP61A requires PPSF 2.0.1.")
    started = time.perf_counter()
    device = torch.device("cuda")
    names, residual_l10, pooled, template, feature_manifest = _prepare_features(args, device)
    try:
        eager_diagnostic = eager_lineage_diagnostic(
            old_prefix=args.old_prefix,
            extra_prefix=args.extra_prefix,
            image_dir=args.image_dir,
            model_dir=args.model_dir,
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as error:
        eager_diagnostic = {
            "status": "not_verifiable",
            "error": f"{type(error).__name__}: {error}",
            "optimizer_updates": 0,
            "tensors_entered_training_heads": False,
            "runtime_seconds_excluded_from_training_projection": 0.0,
        }
    residual_l10 = residual_l10.to(device)
    image_ids, canonical_identities, theta = load_canonical_level_targets(args.theta)
    tune4, rank10 = load_tune_rank_identity_split(args.split)
    identity_order = sorted([*tune4, *rank10], key=str.encode)
    training_o = load_completed_ranked_groups_range(args.source_state, start=0, stop=89)
    original_o = training_o[:65]
    training_w = load_ranked_groups(args.within_state)
    overlap = load_ranked_groups(args.overlap_state)
    identity_lookup = {str(name): str(identity) for name, identity in zip(image_ids, canonical_identities, strict=True)}
    for group in [*training_o, *training_w]:
        for item in group:
            identity_lookup.setdefault(item.image_name, item.identity)
    identities710 = np.asarray([identity_lookup[name] for name in names], dtype=str)
    assignment = fivefold_assignment(names, identities710)
    image_fold = dict(zip(names, map(int, assignment), strict=True))
    exposures, _ = direct_ordinal_exposures([*original_o, *overlap])
    partitions = partition_anchor_exposures(exposures, image_fold)
    seen = partitions["seen_anchor"]
    strict = partitions["strict_unseen_anchor"]
    closure = {
        "seen_anchor": endpoint_coverage(seen, image_fold),
        "strict_unseen_anchor": endpoint_coverage(strict, image_fold),
    }
    closure["closure_pass"] = closure["seen_anchor"]["comparisons"] == 5842 and closure["strict_unseen_anchor"]["comparisons"] == 1472 and 5842 + 1472 == 7314
    canonical_indices = torch.arange(520, dtype=torch.int64, device=device)
    runtime = FrozenFeatureRuntime(names, pooled, torch.empty(0, device=device), {name: index for index, name in enumerate(names)}, canonical_indices, device)
    folds = [build_image_heldout_fold(runtime, fold, names, assignment, training_o, training_w, 520) for fold in range(5)]
    feature_manifest_path = args.feature_workspace / "FEATURE_MANIFEST.json"
    inputs = {
        "driver": Path(__file__),
        "feature_manifest": feature_manifest_path,
        "L10_scratch": args.feature_workspace / "L10_RESIDUAL.npy",
        "pooled_scratch": args.feature_workspace / "L26_POOLED.npy",
        "O89": args.source_state,
        "W56": args.within_state,
        "OVERLAP70_evaluation_only": args.overlap_state,
        "theta": args.theta,
        "split": args.split,
        "model_config": args.model_dir / "config.json",
        "model_weights": args.model_dir / "model.safetensors",
        "processor": args.model_dir / "preprocessor_config.json",
        "step56a_scores": args.step56a_scores,
        "step56a_fold_inventory": args.step56a_fold_inventory,
        "step60a_scores": args.step60a_scores,
        "step60a_init": args.step60a_init,
        "multihead_implementation": Path("ranker/experiments/layer_screen/simultaneous.py"),
        "training_implementation": Path("ranker/experiments/layer_screen/training.py"),
        "analysis_implementation": Path("ranker/experiments/layer_screen/step61_analysis.py"),
    }
    if args.local_runtime_evidence is not None:
        inputs["local_runtime_evidence"] = args.local_runtime_evidence
    resume_contract = {
        "compute_signature": SIGNATURE,
        "layers": LAYERS,
        "seeds": SEEDS,
        "folds": 5,
        "checkpoints": CHECKPOINTS,
        "updates": 512,
        "endpoint_counts": [5842, 1472, 7314],
    }
    with StepRun(
        SIGNATURE,
        args.output_dir,
        ARTIFACTS,
        manifest_name=STEP.artifact_name("artifact_manifest"),
        inputs=inputs,
        time_budget=TimeBudget(3600.0 if args.hardware == "remote" else 5400.0, 3420.0 if args.hardware == "remote" else 5250.0, 180.0 if args.hardware == "remote" else 120.0),
        workspace=Workspace(args.workspace, lifecycle="keep_always"),
        restart_workspace=args.restart_workspace,
        resume_contract=resume_contract,
        on_existing="replace" if args.restart_workspace else "resume",
    ) as run:
        assert run.journal is not None
        source_text = Path(__file__).read_text(encoding="utf-8")
        run.emit(STEP.artifact_name("driver_source"), source_text)
        run.emit(STEP.artifact_name("contract"), {
            "signature": SIGNATURE,
            "training": {"folds": 5, "seeds": list(SEEDS), "multihead_trajectories": 15, "layers": list(LAYERS), "updates": 512, "objective": "HARD_REL_M75", "normalization": "group_weight", "optimizer": "ProdigyPlusScheduleFree 2.0.1", "optimizer_instances": 17, "groups_per_optimizer": 1},
            "layer_semantics": _architecture_rows(),
            "checkpoints": list(CHECKPOINTS),
            "endpoints": {"primary": "SEEN_ANCHOR", "primary_comparisons": 5842, "secondary": "STRICT_UNSEEN_ANCHOR", "secondary_comparisons": 1472, "seen_seen_layer_surface": "not_identifiable_omitted"},
            "classification_scope": "exploratory screen; no promotion or baseline replacement",
            "parity_preserving_optimizations": {
                "L10_residency": "one immutable full tensor on CUDA",
                "indexing": "device-side without per-group CPU round trip",
                "shadow_checks": "device-side exact flags; one synchronization per batch",
                "excluded": ["torch.compile", "CUDA Graphs", "custom kernels"],
            },
            "feature_lineage": {
                "training_and_evaluation": "SDPA EFFICIENT_ATTENTION only",
                "reference": "L26_EFFICIENT_SCAN_REFERENCE",
                "historical_STEP60A_surface_equivalence": False,
                "eager": "read-only lineage diagnostic only",
            },
        })
        run.emit(STEP.artifact_name("architecture_audit"), _architecture_rows())
        run.require("endpoint_closure", bool(closure["closure_pass"]))
        with np.load(args.step56a_scores, allow_pickle=False) as saved56:
            partition_exact = np.array_equal(assignment, saved56["fold_assignment"].astype(np.int64))
        run.require("STEP56A_partition_exact", partition_exact)
        computed_inventory = fold_inventory(names, identities710, assignment, training_o, training_w)
        stored_inventory = read_csv(args.step56a_fold_inventory)
        inventory_exact = all(str(a[key]) == str(b[key]) for a, b in zip(computed_inventory, stored_inventory, strict=True) for key in ("fold", "validation_images", "training_images", "remaining_O_groups", "remaining_W_groups", "remaining_O_pairs", "remaining_W_pairs"))
        run.require("fold_inventory_exact", inventory_exact)
        streamed_digest = streamed_l26_digest(template=template, residual_l10_cpu=residual_l10, device=device)
        authoritative_digest = str(feature_manifest["extraction"]["L26_content_SHA256"])
        run.require("streamed_L26_all_efficient_digest", streamed_digest == authoritative_digest)
        run.emit(STEP.artifact_name("input_provenance"), {
            "features": {**feature_manifest["extraction"], "streamed_L26_SHA256": streamed_digest, "scratch_only": True},
            "eager_lineage_diagnostic": eager_diagnostic,
            "fold_mapping": {"SHA256": mapping_digest(names, assignment), "partition_exact": partition_exact, "inventory_exact": inventory_exact, "fold_sizes": np.bincount(assignment, minlength=5).tolist()},
            "prohibited_inputs": {"STEP49": False, "test": False, "retests": False, "RESERVE200": False},
        })
        with np.load(args.step60a_scores, allow_pickle=False) as saved60:
            raw60 = saved60["raw_scores"].astype(np.float64)
            cp0_index = saved60["checkpoints"].astype(int).tolist().index(0)
            saved_seeds = saved60["seeds"].astype(int).tolist()
        initialization_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            screen_a, _, _ = build_screen(template=template, seed=seed, device=device)
            screen_b, _, _ = build_screen(template=template, seed=seed, device=device)
            count = screen_a.parameter_count_per_head()
            shared = all(tensor_state_hash({name: value.detach().cpu() for name, value in head.state_dict().items()}) == tensor_state_hash({name: value.detach().cpu() for name, value in screen_a.heads[0].state_dict().items()}) for head in screen_a.heads)
            repeat = _state_hash(screen_a) == _state_hash(screen_b)
            scores = capture_all_scores(screen_a, residual_l10, pooled, canonical_indices)
            expected = raw60[saved_seeds.index(seed), 0, cp0_index, :520]
            historical_l26_difference = np.abs(
                scores[-1].astype(np.float64) - expected
            )
            l26_finite = bool(np.isfinite(scores[-1]).all())
            topology = len(screen_a.head_parameter_groups()) == 17 and len(set(map(id, screen_a.trainable_parameters()))) == len(screen_a.trainable_parameters())
            passed = repeat and shared and l26_finite and topology and set(count) == {711426}
            initialization_rows.append({"record_type": "seed_gate", "seed": seed, "fold": "ALL", "pass": passed, "deterministic_repeat": repeat, "all_heads_shared": shared, "L26_reference": "L26_EFFICIENT_SCAN_REFERENCE", "L26_cp0_finite": l26_finite, "historical_STEP60A_cp0_max_abs_difference_diagnostic": float(historical_l26_difference.max()), "historical_STEP60A_cp0_exact_diagnostic": bool(np.count_nonzero(historical_l26_difference) == 0), "parameter_count_per_head": count[0], "optimizer_instances": 17, "groups_per_optimizer": 1, "no_shared_storage": topology})
            del screen_a, screen_b
            gc.collect()
            torch.cuda.empty_cache()
        run.require("initialization_all_pass", all(row["pass"] for row in initialization_rows))
        streamed_tokens = materialize_streamed_l26(template=template, residual_l10_cpu=residual_l10, device=device)
        parity_runtime = FrozenFeatureRuntime(names, pooled, streamed_tokens, runtime.name_to_index, canonical_indices, device)
        for seed in SEEDS:
            parity = backend_matched_l26_first_update_parity(
                template=template,
                residual_l10_cpu=residual_l10,
                runtime=parity_runtime,
                fold=folds[0],
                seed=seed,
                capture_indices=canonical_indices,
            )
            initialization_rows.append(
                {
                    "record_type": "backend_matched_one_update_shadow",
                    "seed": seed,
                    "fold": 0,
                    **parity,
                }
            )
        del streamed_tokens, parity_runtime
        gc.collect()
        torch.cuda.empty_cache()
        run.require("backend_matched_L26_parity", all(row["pass"] for row in initialization_rows if row["record_type"] == "backend_matched_one_update_shadow"), "backend_matched_L26_parity_failed")
        run.emit(STEP.artifact_name("initialization_parity"), initialization_rows)

        pilot = None
        if args.hardware == "remote":
            if args.local_runtime_evidence is None:
                raise ValueError("Remote execution requires --local-runtime-evidence.")
            local_runtime = json.loads(args.local_runtime_evidence.read_text(encoding="utf-8"))
            pilot_seconds = float(local_runtime["local_pilot_seconds"])
            setup_elapsed = float(local_runtime["setup_elapsed_seconds"])
            projected_local = float(local_runtime["projected_local_total_seconds"])
            if projected_local <= 5400.0:
                raise ValueError("Remote execution is forbidden when the recorded local projection is within budget.")
        else:
            pilot_payload = run.journal.run("local_pilot", lambda: trajectory_payload(run_multihead_trajectory(template=template, residual_l10_cpu=residual_l10, pooled=pooled, fold=folds[0], seed=SEEDS[0], capture_indices=canonical_indices, checkpoints=CHECKPOINTS, max_updates=512, tick=run.tick)), fold=0, seed=SEEDS[0])
            pilot = trajectory_from_payload(pilot_payload)
            pilot_seconds = pilot.elapsed_seconds
            eager_seconds = float(eager_diagnostic.get("runtime_seconds_excluded_from_training_projection", 0.0))
            setup_elapsed = time.perf_counter() - started - pilot_seconds - eager_seconds
            projected_local = setup_elapsed + 15.0 * pilot_seconds + 300.0
            if projected_local > 5400.0:
                run.emit(STEP.artifact_name("runtime_hardware"), {"local_pilot_seconds": pilot_seconds, "setup_elapsed_seconds": setup_elapsed, "projected_local_total_seconds": projected_local, "local_limit_seconds": 5400.0, "analytical_hardware": "remote_required", "pilot_in_analytical_panel": False})
                raise RuntimeError(f"REMOTE_ROUTING_REQUIRED projected_local_total={projected_local:.3f}")
        analytical_hardware = "remote_RTX5090" if args.hardware == "remote" else "local"
        results: dict[tuple[int, int], Any] = {}
        if analytical_hardware == "local":
            assert pilot is not None
            results[(0, SEEDS[0])] = pilot
        ordered_units = [(fold, seed) for fold in range(5) for seed in SEEDS]
        completed_times: list[float] = []
        for fold, seed in ordered_units:
            if (fold, seed) in results:
                continue
            if run.journal.is_done("analytical_trajectory", fold=fold, seed=seed):
                result = trajectory_from_payload(run.journal.load("analytical_trajectory", fold=fold, seed=seed))
                results[(fold, seed)] = result
                completed_times.append(result.elapsed_seconds)
                continue
            if args.hardware == "remote" and completed_times:
                remaining = 3600.0 - run.runtime.elapsed_seconds
                estimate = max(completed_times)
                if remaining < estimate + 180.0:
                    raise RuntimeError(f"REMOTE_CONTINUATION_REQUIRED completed={len(results)} remaining={remaining:.3f}")
            payload = run.journal.run("analytical_trajectory", lambda fold=fold, seed=seed: trajectory_payload(run_multihead_trajectory(template=template, residual_l10_cpu=residual_l10, pooled=pooled, fold=folds[fold], seed=seed, capture_indices=canonical_indices, checkpoints=CHECKPOINTS, max_updates=512, tick=None if args.hardware == "remote" else run.tick)), fold=fold, seed=seed)
            result = trajectory_from_payload(payload)
            results[(fold, seed)] = result
            completed_times.append(result.elapsed_seconds)
            run.progress(stage="analytical_trajectory", completed=len(results), total=15, fold=fold, seed=seed)
        run.count("trajectories_completed", len(results), expected=15)
        raw_scores = np.stack([[[results[(fold, seed)].checkpoint_scores[checkpoint] for checkpoint in CHECKPOINTS] for fold in range(5)] for seed in SEEDS])
        run.emit(STEP.artifact_name("scalar_scores"), {"raw_scores": raw_scores, "seeds": np.asarray(SEEDS), "folds": np.arange(5), "checkpoints": np.asarray(CHECKPOINTS), "layers": np.asarray(LAYERS), "image_ids": np.asarray(image_ids), "fold_assignment": assignment})
        run.emit(STEP.artifact_name("runtime_hardware"), {"local_pilot_seconds": pilot_seconds, "setup_elapsed_seconds": setup_elapsed, "projected_local_total_seconds": projected_local, "local_limit_seconds": 5400.0, "analytical_hardware": analytical_hardware, "pilot_in_analytical_panel": analytical_hardware == "local", "trajectory_seconds": [results[key].elapsed_seconds for key in ordered_units]})
        run.emit(STEP.artifact_name("trajectory_journal_audit"), {"completed_units": len(results), "expected_units": 15, "journal": run.journal.audit(), "local_pilot_role": "analytical trajectory 0" if analytical_hardware == "local" else "runtime/parity evidence only"})
        run.emit(STEP.artifact_name("endpoint_closure"), closure)
        run.emit(STEP.artifact_name("evaluation_parity"), [{"record_type": "checkpoint", "fold": fold, "seed": seed, "checkpoint": checkpoint, "pass": bool(np.all(np.isfinite(results[(fold, seed)].checkpoint_scores[checkpoint]))), "lossless_helper": "neutral_schedulefree_evaluation", "eval_copy": "independent numpy copy"} for fold, seed in ordered_units for checkpoint in CHECKPOINTS])

        standard_triple = (0, 1, 2)
        fixed_rows: list[dict[str, Any]] = []
        per_fold_direct = np.empty((17, len(CHECKPOINTS), 5), dtype=np.float64)
        fixed_vectors: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
        for layer_index, layer in enumerate(LAYERS):
            for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
                maps = selected_fold_maps(raw_scores, assignment, image_ids, layer_index=layer_index, checkpoint_indices=[checkpoint_index] * 5, seed_indices=standard_triple)
                for endpoint_name, endpoint_rows in (("SEEN_ANCHOR", seen), ("STRICT_UNSEEN_ANCHOR", strict)):
                    from ranker.experiments.image_heldout_horizon_transfer.analysis import strict_endpoint_vectors
                    vectors = strict_endpoint_vectors(endpoint_rows, image_fold, maps, identity_order)
                    fixed_vectors[(endpoint_name, layer_index, checkpoint_index)] = vectors
                    fixed_rows.append({"layer": layer, "checkpoint": checkpoint, "endpoint": endpoint_name, "direct": float(vectors["direct"].mean()), "G": float(vectors["G"].mean()), "gross": float(vectors["gross"].mean()), "target_images": closure["seen_anchor" if endpoint_name == "SEEN_ANCHOR" else "strict_unseen_anchor"]["target_images"], "identities": 14})
                for fold in range(5):
                    per_fold_direct[layer_index, checkpoint_index, fold] = endpoint_fold_loss(seen, image_fold, maps, fold)
        run.emit(STEP.artifact_name("fixed_surface"), fixed_rows)
        layerwise, joint = select_lofo_maps(per_fold_direct, LAYERS, CHECKPOINTS)
        lofo_rows: list[dict[str, Any]] = []
        layerwise_direct = np.empty(17, dtype=np.float64)
        selected_maps_by_layer: list[list[dict[str, float]]] = []
        from ranker.experiments.image_heldout_horizon_transfer.analysis import strict_endpoint_vectors
        for layer_index, layer in enumerate(LAYERS):
            maps = selected_fold_maps(raw_scores, assignment, image_ids, layer_index=layer_index, checkpoint_indices=layerwise[layer_index], seed_indices=standard_triple)
            selected_maps_by_layer.append(maps)
            vector = strict_endpoint_vectors(seen, image_fold, maps, identity_order)["direct"]
            layerwise_direct[layer_index] = vector.mean()
            for fold in range(5):
                heldout = endpoint_fold_loss(seen, image_fold, maps, fold)
                lofo_rows.append({"record_type": "heldout_fold", "layer": layer, "fold": fold, "checkpoint": CHECKPOINTS[layerwise[layer_index, fold]], "direct": heldout})
            lofo_rows.append({"record_type": "aggregate", "layer": layer, "fold": "ALL", "checkpoint": "mixed", "direct": layerwise_direct[layer_index]})
        run.emit(STEP.artifact_name("layerwise_lofo"), lofo_rows)
        joint_maps = []
        joint_rows = []
        for fold in range(5):
            layer_index, checkpoint_index = map(int, joint[fold])
            maps = selected_fold_maps(raw_scores, assignment, image_ids, layer_index=layer_index, checkpoint_indices=[checkpoint_index] * 5, seed_indices=standard_triple)
            joint_maps.append(maps[fold])
            joint_rows.append({"fold": fold, "layer": LAYERS[layer_index], "checkpoint": CHECKPOINTS[checkpoint_index], "heldout_direct": endpoint_fold_loss(seen, image_fold, maps, fold)})
        run.emit(STEP.artifact_name("joint_lofo"), joint_rows)
        lead_index = min(range(16), key=lambda index: (layerwise_direct[index], -LAYERS[index]))
        reference_index = 16
        selected_raw = np.stack([[raw_scores[seed_index, fold, layerwise[layer_index, fold], layer_index] for seed_index in range(3) for fold in range(5)] for layer_index in (lead_index, reference_index)])
        run.emit(STEP.artifact_name("selected_scores"), {"raw_scores": selected_raw, "systems": np.asarray([f"L{LAYERS[lead_index]}", "L26_EFFICIENT_SCAN_REFERENCE"]), "checkpoint_maps": np.stack([layerwise[lead_index], layerwise[reference_index]]), "layers": np.asarray([LAYERS[lead_index], 26]), "seeds": np.asarray(SEEDS)})
        rng = np.random.Generator(np.random.PCG64(20260817))
        identity_draws = rng.integers(0, 14, size=(100000, 14), dtype=np.int64)
        uq_rows: list[dict[str, Any]] = []
        contrast_rows: list[dict[str, Any]] = []
        guard_rows: list[dict[str, Any]] = []
        uq_by_endpoint: dict[str, dict[str, dict[str, Any]]] = {}
        for endpoint_name, endpoint_rows in (("SEEN_ANCHOR", seen), ("STRICT_UNSEEN_ANCHOR", strict)):
            candidate = all_seed_triple_vectors(raw_scores, assignment, image_ids, endpoint_rows, image_fold, identity_order, layer_index=lead_index, checkpoint_indices=layerwise[lead_index])
            reference = all_seed_triple_vectors(raw_scores, assignment, image_ids, endpoint_rows, image_fold, identity_order, layer_index=reference_index, checkpoint_indices=layerwise[reference_index])
            uq_by_endpoint[endpoint_name] = {}
            for metric in ("direct", "G", "gross"):
                payload, _distribution = paired_uq(candidate[metric], reference[metric], identity_draws=identity_draws)
                uq_by_endpoint[endpoint_name][metric] = payload
                uq_rows.append({"record_type": "frozen_selection_map", "endpoint": endpoint_name, "metric": metric, **payload})
                contrast_rows.extend({"record_type": "identity", "endpoint": endpoint_name, "metric": metric, "identity_id": identity, "delta": delta} for identity, delta in zip(identity_order, payload["identity_deltas"], strict=True))
            selection_candidate: dict[str, list[np.ndarray]] = {
                metric: [] for metric in ("direct", "G", "gross")
            }
            selection_reference: dict[str, list[np.ndarray]] = {
                metric: [] for metric in ("direct", "G", "gross")
            }
            for triple in itertools.product(range(3), repeat=3):
                triple_fold_direct = np.empty(
                    (17, len(CHECKPOINTS), 5), dtype=np.float64
                )
                for current_layer in (lead_index, reference_index):
                    for checkpoint_index in range(len(CHECKPOINTS)):
                        trial_maps = selected_fold_maps(
                            raw_scores,
                            assignment,
                            image_ids,
                            layer_index=current_layer,
                            checkpoint_indices=[checkpoint_index] * 5,
                            seed_indices=triple,
                        )
                        for fold in range(5):
                            triple_fold_direct[
                                current_layer, checkpoint_index, fold
                            ] = endpoint_fold_loss(
                                endpoint_rows, image_fold, trial_maps, fold
                            )
                for current_layer, destination in (
                    (lead_index, selection_candidate),
                    (reference_index, selection_reference),
                ):
                    checkpoint_map = np.empty(5, dtype=np.int64)
                    for fold in range(5):
                        train_folds = [item for item in range(5) if item != fold]
                        means = triple_fold_direct[current_layer][
                            :, train_folds
                        ].mean(axis=1)
                        checkpoint_map[fold] = int(np.argmin(means))
                    trial_maps = selected_fold_maps(
                        raw_scores,
                        assignment,
                        image_ids,
                        layer_index=current_layer,
                        checkpoint_indices=checkpoint_map,
                        seed_indices=triple,
                    )
                    vectors = strict_endpoint_vectors(
                        endpoint_rows, image_fold, trial_maps, identity_order
                    )
                    for metric in destination:
                        destination[metric].append(vectors[metric])
            for metric, candidate_rows in selection_candidate.items():
                payload, _distribution = paired_uq(
                    np.stack(candidate_rows),
                    np.stack(selection_reference[metric]),
                    identity_draws=identity_draws,
                )
                uq_rows.append(
                    {
                        "record_type": "selection_aware_seed_triples",
                        "endpoint": endpoint_name,
                        "metric": metric,
                        **payload,
                    }
                )
        run.emit(STEP.artifact_name("uq"), uq_rows)
        run.emit(STEP.artifact_name("contrasts"), contrast_rows)
        for metric in ("G", "gross"):
            payload = uq_by_endpoint["SEEN_ANCHOR"][metric]
            harm = payload["point_delta"] > 0 and payload["CI90_lower"] > 0 if metric == "G" else payload["point_delta"] >= 0.005 and payload["CI90_lower"] > 0
            guard_rows.append({"metric": metric, "point_delta": payload["point_delta"], "CI90_lower": payload["CI90_lower"], "CI90_upper": payload["CI90_upper"], "resolved_harm": harm})
        run.emit(STEP.artifact_name("guards"), guard_rows)
        level_rows = []
        for system, layer_index in ((f"L{LAYERS[lead_index]}", lead_index), ("L26_EFFICIENT_SCAN_REFERENCE", reference_index)):
            level_rows.append({"system": system, **level_diagnostics(selected_maps_by_layer[layer_index], image_ids, canonical_identities, theta, assignment[:520], identity_order)})
        run.emit(STEP.artifact_name("level_diagnostics"), level_rows)
        boundary_rows = []
        for layer_index, layer in enumerate(LAYERS):
            fixed_seen = np.asarray([fixed_vectors[("SEEN_ANCHOR", layer_index, checkpoint_index)]["direct"].mean() for checkpoint_index in range(len(CHECKPOINTS))])
            best_index = int(np.argmin(fixed_seen))
            boundary_rows.append({"layer": layer, "aggregate_best_checkpoint": CHECKPOINTS[best_index], "LOFO_checkpoints": json.dumps([CHECKPOINTS[index] for index in layerwise[layer_index]]), "cp512_minus_cp448": float(fixed_seen[-1] - fixed_seen[-2]), "LOFO_cp512_count": int(np.sum(layerwise[layer_index] == len(CHECKPOINTS) - 1))})
        run.emit(STEP.artifact_name("boundary_audit"), boundary_rows)
        lead_boundary = boundary_rows[lead_index]["aggregate_best_checkpoint"] == 512 or boundary_rows[lead_index]["LOFO_cp512_count"] >= 3
        guard_harm = any(row["resolved_harm"] for row in guard_rows)
        if lead_boundary:
            result = "right_boundary_limited"
        elif layerwise_direct[lead_index] < layerwise_direct[reference_index] and not guard_harm:
            result = "screen_lead_found"
        else:
            result = "screen_complete_no_lead"
        decision = {"STEP61A_result": result, "screen_lead": f"L{LAYERS[lead_index]}", "lead_primary_direct": layerwise_direct[lead_index], "L26_primary_direct": layerwise_direct[reference_index], "resolved_guard_harm": guard_harm, "right_boundary_limited": lead_boundary, "layer_promoted": False, "historical_margin_role": "0.005 policy reference only; not scan resolution", "joint_policy_role": "selection-policy diagnostic only", "seen_seen_layer_concordance": "not_identifiable_omitted"}
        run.emit(STEP.artifact_name("policy_decisions"), decision)
        gates = {"architecture_semantics": True, "initialization": True, "one_update_L26": True, "lossless_evaluation": True, "endpoint_closure": True, "trajectories_15": len(results) == 15, "all_finite": all(result_item.all_finite for result_item in results.values()), "no_prohibited_access": True, "driver_hash_recorded_before_update": True}
        run.emit(STEP.artifact_name("gates"), {"STEP61A_result": result, "gates": gates})
        head_before = str(read_project_events()[-1]["event_sha256"])
        _, head_after, appended = append_project_event("STEP61A", {"event_type": "diagnostic_result", "subject": "provenance-valid full L10-L26 block-skip scan", "statement": f"STEP61A completed 15/15 E3 multihead trajectories; result {result}; screen lead L{LAYERS[lead_index]}; no promotion or baseline change", "status_before": "full_layer_scan_open_deferred_provenance_unverified", "status_after": result, "evidence_class": "prospective seen-anchor exploratory screen", "evidence": [{"artifact_path": str(run.path(STEP.artifact_name("gates")).resolve()), "artifact_sha256": sha256_file(run.path(STEP.artifact_name("gates"))), "locator": {"kind": "json_pointer", "assertions": {"/STEP61A_result": result}}}], "supersedes_event_ids": [], "limitations": ["known identities", "single fivefold image partition", "16 layer alternatives", "no seen/seen layer surface", "no external evidence"], "protected_conclusion": f"STEP61A result is {result}; no layer promoted", "explicitly_not_protected": ["external promotion", "baseline replacement", "layer closure when boundary-limited"], "author_role": "Executor"})
        ledger = verify_project_ledger()
        run.require("ledger_integrity", ledger.all_pass)
        run.emit(STEP.artifact_name("ledger_update_audit"), {"head_before": head_before, "head_after": head_after, "event_appended": appended, **ledger.as_dict()})
        summary = {"signature": SIGNATURE, "STEP61A_result": result, "screen_lead": f"L{LAYERS[lead_index]}", "trajectories_completed": len(results), "updates_per_trajectory": 512, "analytical_hardware": analytical_hardware, "runtime_seconds": time.perf_counter() - started, "primary": {"lead_direct": layerwise_direct[lead_index], "L26_direct": layerwise_direct[reference_index], "direct_UQ": uq_by_endpoint["SEEN_ANCHOR"]["direct"]}, "secondary": {"direct_UQ": uq_by_endpoint["STRICT_UNSEEN_ANCHOR"]["direct"]}, "boundary": boundary_rows[lead_index], "guards": guard_rows, "level_diagnostics": level_rows, "training_performed": True, "new_labels_used": 0, "STEP49_test_retests_RESERVE200_accessed": False, "FULL192_changed": False, "current_baseline_changed": False}
        _inventory_update(args.inventory, summary)
        run.emit(STEP.artifact_name("summary"), summary)
        manifest = run.finalize()
        run.result(STEP61A_result=result, screen_lead=f"L{LAYERS[lead_index]}", trajectories_completed=len(results), analytical_hardware=analytical_hardware, training_performed=True, current_baseline_changed=False, report_completeness_pass=manifest["report_completeness_pass"], artifact_set_equality_pass=manifest["artifact_set_equality_pass"])


if __name__ == "__main__":
    main()
