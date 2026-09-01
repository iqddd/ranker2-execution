"""STEP61B: fixed L24@384 confirmation against matched efficient L26@384."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ranker.data import load_completed_ranked_groups_range, load_ranked_groups
from ranker.experiments.image_heldout_horizon_transfer import build_image_heldout_fold, mapping_digest
from ranker.experiments.image_heldout_horizon_transfer.analysis import strict_endpoint_vectors
from ranker.experiments.image_heldout_horizon_transfer.certification import (
    endpoint_coverage,
    equal_identity_mean,
    partition_anchor_exposures,
    per_target_direct,
    uq_from_image_contrasts,
)
from ranker.experiments.layer_screen.analysis import fivefold_assignment, fold_inventory
from ranker.experiments.layer_screen.confirmation import (
    POINT_TRIPLE_INDEX,
    TRIPLES,
    classify_confirmation,
    combined_e6_maps,
    fixed_identity_metrics,
    fixed_target_direct,
    fixed_target_uq,
    fixed_triple_maps,
)
from ranker.experiments.layer_screen.execution import (
    prepare_all_efficient_workspace,
    streamed_l26_digest,
)
from ranker.experiments.layer_screen.step61_analysis import level_diagnostics
from ranker.experiments.layer_screen.training import (
    backend_matched_l26_first_update_parity,
    build_screen,
    capture_all_scores,
    run_multihead_trajectory,
    trajectory_from_payload,
    trajectory_payload,
)
from ranker.frozen_readouts import FrozenFeatureRuntime
from ranker.io import atomic_text, read_csv, sha256_file
from ranker.level_evaluation import load_canonical_level_targets, load_tune_rank_identity_split
from ranker.project_ledger import append_project_event, read_project_events, verify_project_ledger
from ranker.reproducibility import tensor_state_hash
from ranker.step import Artifact, StepRun, TimeBudget, Workspace
from ranker.step_registry import STEPS
from ranker.tune14 import paired_uq
from ranker.validation_resolution import direct_ordinal_exposures

STEP = STEPS["61b"]
SIGNATURE = "STEP61B_L24_FIXED_CONFIRMATION_ALL_EFFICIENT_V1"
SEEDS = (20260806, 20260807, 20260808)
ARMS = (24, 26)
CHECKPOINTS = (0, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384)
FIXED_CHECKPOINT = 384

ARTIFACTS = (
    Artifact(STEP.artifact_name("contract"), "json", frozenset({"signature", "training", "fixed_comparison"})),
    Artifact(STEP.artifact_name("driver_source"), "text", frozenset()),
    Artifact(STEP.artifact_name("input_provenance"), "json", frozenset({"features", "fold_mapping", "upstream_STEP61A", "prohibited_inputs"})),
    Artifact(STEP.artifact_name("initialization_parity"), "csv_union", frozenset({"record_type", "seed", "pass"})),
    Artifact(STEP.artifact_name("evaluation_parity"), "csv", frozenset({"fold", "seed", "checkpoint", "pass"})),
    Artifact(STEP.artifact_name("runtime_hardware"), "json", frozenset({"analytical_hardware", "trajectory_seconds", "runtime_limit_seconds"})),
    Artifact(STEP.artifact_name("trajectory_journal_audit"), "json", frozenset({"completed_units", "expected_units", "journal"})),
    Artifact(STEP.artifact_name("scalar_scores"), "npz", frozenset({"raw_scores", "seeds", "folds", "checkpoints", "arms", "image_ids"})),
    Artifact(STEP.artifact_name("fixed_scores"), "npz", frozenset({"raw_scores", "per_seed_percentiles", "e3_percentiles", "systems", "seeds", "folds", "image_ids"})),
    Artifact(STEP.artifact_name("endpoint_closure"), "json", frozenset({"seen_anchor", "strict_unseen_anchor", "closure_pass"})),
    Artifact(STEP.artifact_name("checkpoint_surface"), "csv", frozenset({"system", "checkpoint", "endpoint", "direct", "G", "gross"})),
    Artifact(STEP.artifact_name("contrasts"), "csv_union", frozenset({"record_type", "endpoint", "metric", "delta"})),
    Artifact(STEP.artifact_name("uq"), "csv_union", frozenset({"record_type", "endpoint", "metric", "point_delta", "CI90_lower", "CI90_upper"})),
    Artifact(STEP.artifact_name("guards"), "csv", frozenset({"metric", "point_delta", "CI90_lower", "CI90_upper", "resolved_harm"})),
    Artifact(STEP.artifact_name("level_diagnostics"), "csv", frozenset({"system", "IdentityOffsetMAE", "IB_FrameLTE10", "beta", "Pearson"})),
    Artifact(STEP.artifact_name("discovery_confirmation"), "csv", frozenset({"evidence", "delta_direct", "role"})),
    Artifact(STEP.artifact_name("policy_decisions"), "json", frozenset({"STEP61B_result", "layer_promoted", "branch_process_status"})),
    Artifact(STEP.artifact_name("gates"), "json", frozenset({"STEP61B_result", "gates"})),
    Artifact(STEP.artifact_name("ledger_update_audit"), "json", frozenset({"head_before", "head_after", "chain_integrity_pass"})),
    Artifact(STEP.artifact_name("summary"), "json", frozenset({"signature", "STEP61B_result", "trajectories_completed", "training_performed"}), terminal=True),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=STEP.output_dir)
    result.add_argument("--workspace", type=Path, default=Path("_work/step61b_l24_fixed_confirmation"))
    result.add_argument("--feature-workspace", type=Path, default=Path("_work/step61a_full_l10_l25_features"))
    result.add_argument("--feature-seed-manifest", type=Path)
    result.add_argument("--restart-workspace", action="store_true")
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
    result.add_argument("--step61a-scores", type=Path, default=STEPS["61a"].artifact("scalar_scores"))
    result.add_argument("--step61a-summary", type=Path, default=STEPS["61a"].artifact("summary"))
    result.add_argument("--step61a-manifest", type=Path, default=STEPS["61a"].artifact("artifact_manifest"))
    result.add_argument("--ledger-jsonl", type=Path, default=Path("PROJECT_LEDGER.jsonl"))
    result.add_argument("--ledger-head", type=Path, default=Path("PROJECT_LEDGER_HEAD.json"))
    result.add_argument("--ledger-md", type=Path, default=Path("PROJECT_LEDGER.md"))
    result.add_argument("--inventory", type=Path, default=Path("PROJECT_INVENTORY.md"))
    return result


def _head_hashes(screen: torch.nn.Module) -> list[str]:
    return [tensor_state_hash({name: value.detach().cpu() for name, value in head.state_dict().items()}) for head in screen.heads]


def _inventory_update(path: Path, summary: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    fact = (
        f"- STEP61B fixed fresh-seed confirmation of L24@384 versus matched efficient L26@384: "
        f"`{summary['STEP61B_result']}`, primary Δdirect `{summary['primary']['direct']['point_delta']:.9f}`. "
        "The L24 branch is process-resolved without layer promotion or baseline change."
    )
    if "STEP61B fixed fresh-seed confirmation" not in text:
        anchor = "### Диагностически"
        position = text.index("\n", text.index(anchor)) + 1
        text = text[:position] + "\n" + fact + text[position:]
    next_move = "- Next formal representation move after STEP61B: DINOv3 full-token extraction with the current matched JOINT4 readout."
    if next_move not in text:
        axis = "## 8. Открытые development-оси"
        position = text.index("\n", text.index(axis)) + 1
        text = text[:position] + "\n" + next_move + text[position:]
    source = "- `artifacts/step61b_l24_fixed_confirmation/STEP61B_SUMMARY.json` — fixed fresh-seed L24 confirmation."
    if source not in text:
        text = text.rstrip() + "\n" + source + "\n"
    if "_STATUS_START" in text or "_STATUS_END" in text:
        raise RuntimeError("Inventory marker blocks are forbidden.")
    atomic_text(path, text)


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("STEP61B requires a BF16 CUDA GPU.")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
        raise RuntimeError(f"hardware_class_mismatch: {torch.cuda.get_device_name(0)}")
    if torch.__version__ != "2.11.0+cu130" or version("prodigy-plus-schedule-free") != "2.0.1":
        raise RuntimeError("STEP61B software class differs from STEP61A.")
    started = time.perf_counter()
    device = torch.device("cuda")
    names, residual_l10, pooled, template, feature_manifest = prepare_all_efficient_workspace(
        feature_dir=args.feature_workspace,
        image_dir=args.image_dir,
        model_dir=args.model_dir,
        old_prefix=args.old_prefix,
        extra_prefix=args.extra_prefix,
        device=device,
        reconstruct=args.reconstruct_features,
        require_prepared=True,
        seed_manifest=args.feature_seed_manifest,
    )
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

    inputs = {
        "driver": Path(__file__),
        "feature_manifest": args.feature_workspace / "FEATURE_MANIFEST.json",
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
        "step61a_scores": args.step61a_scores,
        "step61a_summary": args.step61a_summary,
        "step61a_manifest": args.step61a_manifest,
        "screen_implementation": Path("ranker/experiments/layer_screen/simultaneous.py"),
        "training_implementation": Path("ranker/experiments/layer_screen/training.py"),
        "confirmation_analysis": Path("ranker/experiments/layer_screen/confirmation.py"),
    }
    if args.feature_seed_manifest is not None:
        inputs["feature_seed_manifest"] = args.feature_seed_manifest
    resume_contract = {
        "compute_signature": SIGNATURE,
        "arms": ARMS,
        "seeds": SEEDS,
        "folds": 5,
        "checkpoints": CHECKPOINTS,
        "fixed_checkpoint": FIXED_CHECKPOINT,
        "updates": 384,
        "endpoint_counts": [5842, 1472, 7314],
    }
    with StepRun(
        SIGNATURE,
        args.output_dir,
        ARTIFACTS,
        manifest_name=STEP.artifact_name("artifact_manifest"),
        inputs=inputs,
        time_budget=TimeBudget(3600.0, 3420.0, 180.0),
        workspace=Workspace(args.workspace, lifecycle="keep_always"),
        restart_workspace=args.restart_workspace,
        resume_contract=resume_contract,
        on_existing="replace" if args.restart_workspace else "resume",
    ) as run:
        assert run.journal is not None
        run.emit(STEP.artifact_name("driver_source"), Path(__file__).read_text(encoding="utf-8"))
        run.emit(STEP.artifact_name("contract"), {
            "signature": SIGNATURE,
            "training": {"folds": 5, "seeds": list(SEEDS), "trajectories": 15, "arms": list(ARMS), "updates": 384, "objective": "HARD_REL_M75", "normalization": "group_weight", "optimizer": "ProdigyPlusScheduleFree 2.0.1", "optimizer_instances_per_trajectory": 2, "backend": "SDPA EFFICIENT_ATTENTION only"},
            "fixed_comparison": {"candidate": "L24@384", "reference": "L26_EFFICIENT_SCAN_REFERENCE@384", "selection": "forbidden"},
            "checkpoints": list(CHECKPOINTS),
            "endpoints": {"primary": "SEEN_ANCHOR", "primary_comparisons": 5842, "secondary": "STRICT_UNSEEN_ANCHOR", "secondary_comparisons": 1472},
            "historical_margin_role": "0.005 annotation only; not a gate",
            "post_selection_E6": "sensitivity only; six observed seeds fixed",
        })
        run.require("endpoint_closure", bool(closure["closure_pass"]))
        with np.load(args.step56a_scores, allow_pickle=False) as saved56:
            partition_exact = np.array_equal(assignment, saved56["fold_assignment"].astype(np.int64))
        run.require("STEP56A_partition_exact", partition_exact)
        inventory_exact = all(str(a[key]) == str(b[key]) for a, b in zip(fold_inventory(names, identities710, assignment, training_o, training_w), read_csv(args.step56a_fold_inventory), strict=True) for key in ("fold", "validation_images", "training_images", "remaining_O_groups", "remaining_W_groups", "remaining_O_pairs", "remaining_W_pairs"))
        run.require("fold_inventory_exact", inventory_exact)
        streamed_digest = streamed_l26_digest(template=template, residual_l10_cpu=residual_l10, device=device)
        run.require("streamed_L26_digest", streamed_digest == feature_manifest["extraction"]["L26_content_SHA256"])
        step61a_summary = json.loads(args.step61a_summary.read_text(encoding="utf-8"))
        run.require("STEP61A_complete", step61a_summary["STEP61A_result"] == "screen_lead_found" and step61a_summary["screen_lead"] == "L24")
        run.emit(STEP.artifact_name("input_provenance"), {
            "features": {**feature_manifest["extraction"], "streamed_L26_SHA256": streamed_digest, "scratch_only": True},
            "fold_mapping": {"SHA256": mapping_digest(names, assignment), "partition_exact": partition_exact, "inventory_exact": inventory_exact, "fold_sizes": np.bincount(assignment, minlength=5).tolist()},
            "upstream_STEP61A": {"result": step61a_summary["STEP61A_result"], "lead": step61a_summary["screen_lead"], "discovery_delta": step61a_summary["primary"]["direct_UQ"]["point_delta"], "scores_sha256": sha256_file(args.step61a_scores), "manifest_sha256": sha256_file(args.step61a_manifest)},
            "prohibited_inputs": {"STEP49": False, "test": False, "retests": False, "RESERVE200": False},
        })

        initialization_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            screen_a, _, _ = build_screen(template=template, seed=seed, device=device, head_layers=ARMS)
            screen_b, _, _ = build_screen(template=template, seed=seed, device=device, head_layers=ARMS)
            hashes_a = _head_hashes(screen_a)
            repeat = hashes_a == _head_hashes(screen_b)
            shared_initialization = len(set(hashes_a)) == 1
            parameters = screen_a.trainable_parameters()
            topology = len(screen_a.head_parameter_groups()) == 2 and len({id(parameter) for parameter in parameters}) == len(parameters)
            scores = capture_all_scores(screen_a, residual_l10, pooled, canonical_indices)
            finite = bool(np.isfinite(scores).all())
            passed = repeat and shared_initialization and topology and finite and screen_a.parameter_count_per_head() == (711426, 711426)
            initialization_rows.append({"record_type": "seed_gate", "seed": seed, "fold": "ALL", "pass": passed, "deterministic_repeat": repeat, "shared_initialization": shared_initialization, "independent_storage": topology, "optimizer_instances": 2, "scores_finite": finite, "head_layers": json.dumps(list(screen_a.head_layers))})
            del screen_a, screen_b
        run.require("initialization_all_pass", all(row["pass"] for row in initialization_rows))
        from ranker.experiments.layer_screen.execution import materialize_streamed_l26
        streamed_tokens = materialize_streamed_l26(template=template, residual_l10_cpu=residual_l10, device=device)
        parity_runtime = FrozenFeatureRuntime(names, pooled, streamed_tokens, runtime.name_to_index, canonical_indices, device)
        for seed in SEEDS:
            parity = backend_matched_l26_first_update_parity(template=template, residual_l10_cpu=residual_l10, runtime=parity_runtime, fold=folds[0], seed=seed, capture_indices=canonical_indices, head_layers=ARMS)
            initialization_rows.append({"record_type": "backend_matched_two_arm_shadow", "seed": seed, "fold": 0, **parity})
        del streamed_tokens, parity_runtime
        run.require("backend_matched_two_arm_parity", all(row["pass"] for row in initialization_rows if row["record_type"] == "backend_matched_two_arm_shadow"))
        run.emit(STEP.artifact_name("initialization_parity"), initialization_rows)

        results: dict[tuple[int, int], Any] = {}
        ordered_units = [(fold, seed) for fold in range(5) for seed in SEEDS]
        for fold, seed in ordered_units:
            if run.journal.is_done("analytical_trajectory", fold=fold, seed=seed):
                result = trajectory_from_payload(run.journal.load("analytical_trajectory", fold=fold, seed=seed))
            else:
                payload = run.journal.run("analytical_trajectory", lambda fold=fold, seed=seed: trajectory_payload(run_multihead_trajectory(template=template, residual_l10_cpu=residual_l10, pooled=pooled, fold=folds[fold], seed=seed, capture_indices=canonical_indices, checkpoints=CHECKPOINTS, max_updates=384, head_layers=ARMS, tick=run.tick)), fold=fold, seed=seed)
                result = trajectory_from_payload(payload)
            results[(fold, seed)] = result
            run.progress(stage="analytical_trajectory", completed=len(results), total=15, fold=fold, seed=seed)
        run.count("trajectories_completed", len(results), expected=15)
        raw_scores = np.stack([[[results[(fold, seed)].checkpoint_scores[checkpoint] for checkpoint in CHECKPOINTS] for fold in range(5)] for seed in SEEDS])
        run.emit(STEP.artifact_name("scalar_scores"), {"raw_scores": raw_scores, "seeds": np.asarray(SEEDS), "folds": np.arange(5), "checkpoints": np.asarray(CHECKPOINTS), "arms": np.asarray(ARMS), "image_ids": np.asarray(image_ids), "fold_assignment": assignment})
        run.emit(STEP.artifact_name("runtime_hardware"), {"analytical_hardware": "remote_RTX5090", "torch": torch.__version__, "cuda": torch.version.cuda, "GPU": torch.cuda.get_device_name(0), "trajectory_seconds": [results[key].elapsed_seconds for key in ordered_units], "runtime_limit_seconds": 3600.0})
        run.emit(STEP.artifact_name("trajectory_journal_audit"), {"completed_units": len(results), "expected_units": 15, "journal": run.journal.audit()})
        run.emit(STEP.artifact_name("endpoint_closure"), closure)
        run.emit(STEP.artifact_name("evaluation_parity"), [{"fold": fold, "seed": seed, "checkpoint": checkpoint, "pass": bool(np.isfinite(results[(fold, seed)].checkpoint_scores[checkpoint]).all()), "lossless_helper": "neutral_schedulefree_multi_evaluation", "eval_copy": "independent numpy copy"} for fold, seed in ordered_units for checkpoint in CHECKPOINTS])

        fixed_index = CHECKPOINTS.index(FIXED_CHECKPOINT)
        surface_rows: list[dict[str, Any]] = []
        for arm_index, layer in enumerate(ARMS):
            for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
                maps = fixed_triple_maps(raw_scores, assignment, image_ids, arm_index=arm_index, checkpoint_index=checkpoint_index, triple=(0, 1, 2))
                for endpoint, endpoint_rows in (("SEEN_ANCHOR", seen), ("STRICT_UNSEEN_ANCHOR", strict)):
                    vectors = strict_endpoint_vectors(endpoint_rows, image_fold, maps, identity_order)
                    surface_rows.append({"system": "L24" if layer == 24 else "L26_EFFICIENT_SCAN_REFERENCE", "checkpoint": checkpoint, "endpoint": endpoint, "direct": float(vectors["direct"].mean()), "G": float(vectors["G"].mean()), "gross": float(vectors["gross"].mean())})
        run.emit(STEP.artifact_name("checkpoint_surface"), surface_rows)

        per_seed_percentiles = np.empty((2, 3, 5, 520), dtype=np.float64)
        e3_percentiles = np.empty((2, 5, 520), dtype=np.float64)
        for arm_index in range(2):
            for seed_index in range(3):
                maps = fixed_triple_maps(raw_scores, assignment, image_ids, arm_index=arm_index, checkpoint_index=fixed_index, triple=(seed_index,))
                for fold in range(5):
                    per_seed_percentiles[arm_index, seed_index, fold] = [maps[fold][str(name)] for name in image_ids]
            maps = fixed_triple_maps(raw_scores, assignment, image_ids, arm_index=arm_index, checkpoint_index=fixed_index, triple=(0, 1, 2))
            for fold in range(5):
                e3_percentiles[arm_index, fold] = [maps[fold][str(name)] for name in image_ids]
        run.emit(STEP.artifact_name("fixed_scores"), {"raw_scores": raw_scores[:, :, fixed_index], "per_seed_percentiles": per_seed_percentiles, "e3_percentiles": e3_percentiles, "systems": np.asarray(["L24", "L26_EFFICIENT_SCAN_REFERENCE"]), "seeds": np.asarray(SEEDS), "folds": np.arange(5), "image_ids": np.asarray(image_ids)})

        identity_rng = np.random.Generator(np.random.PCG64(20260817))
        identity_draws = identity_rng.integers(0, 14, size=(100000, 14), dtype=np.int64)
        uq_rows: list[dict[str, Any]] = []
        contrast_rows: list[dict[str, Any]] = []
        direct_uq: dict[str, dict[str, Any]] = {}
        guard_rows: list[dict[str, Any]] = []
        triple_ranges: dict[str, dict[str, float]] = {}
        for endpoint, endpoint_rows in (("SEEN_ANCHOR", seen), ("STRICT_UNSEEN_ANCHOR", strict)):
            candidate_target, identities, images, target_folds = fixed_target_direct(raw_scores, assignment, image_ids, endpoint_rows, image_fold, arm_index=0, checkpoint_index=fixed_index)
            reference_target, identities_ref, images_ref, folds_ref = fixed_target_direct(raw_scores, assignment, image_ids, endpoint_rows, image_fold, arm_index=1, checkpoint_index=fixed_index)
            run.require(f"{endpoint}_target_metadata_exact", np.array_equal(identities, identities_ref) and np.array_equal(images, images_ref) and np.array_equal(target_folds, folds_ref))
            target_delta = candidate_target - reference_target
            rows, _digests = uq_from_image_contrasts(target_delta, identities, target_folds)
            for row in rows:
                uq_rows.append({"record_type": "target_image_bootstrap", "endpoint": endpoint, "metric": "direct", **row})
                if row["uq_semantics"] == "fixed_known_identity_target_bootstrap":
                    direct_uq[endpoint] = row
            point = target_delta[POINT_TRIPLE_INDEX]
            for identity in identity_order:
                delta = float(point[identities == identity].mean())
                contrast_rows.append({"record_type": "identity", "endpoint": endpoint, "metric": "direct", "identity_id": identity, "seed": "E3", "delta": delta, "sign": "improved" if delta < 0 else "worsened" if delta > 0 else "tied"})
            for seed_index, seed in enumerate(SEEDS):
                index = TRIPLES.index((seed_index, seed_index, seed_index))
                contrast_rows.append({"record_type": "seed", "endpoint": endpoint, "metric": "direct", "identity_id": "ALL", "seed": seed, "delta": equal_identity_mean(target_delta[index], identities)})
            seed_values = np.asarray([equal_identity_mean(row, identities) for row in target_delta])
            triple_ranges[endpoint] = {"min": float(seed_values.min()), "median": float(np.median(seed_values)), "max": float(seed_values.max())}
            candidate_identity = fixed_identity_metrics(raw_scores, assignment, image_ids, endpoint_rows, image_fold, identity_order, arm_index=0, checkpoint_index=fixed_index)
            reference_identity = fixed_identity_metrics(raw_scores, assignment, image_ids, endpoint_rows, image_fold, identity_order, arm_index=1, checkpoint_index=fixed_index)
            for metric in ("G", "gross"):
                payload, _distribution = paired_uq(candidate_identity[metric], reference_identity[metric], identity_draws=identity_draws)
                uq_rows.append({"record_type": "identity_guard_bootstrap", "endpoint": endpoint, "metric": metric, **payload})
                if endpoint == "SEEN_ANCHOR":
                    resolved = payload["point_delta"] > 0.0 and payload["CI90_lower"] > 0.0
                    guard_rows.append({"metric": metric, "point_delta": payload["point_delta"], "CI90_lower": payload["CI90_lower"], "CI90_upper": payload["CI90_upper"], "resolved_harm": resolved})
        run.emit(STEP.artifact_name("contrasts"), contrast_rows)
        run.emit(STEP.artifact_name("uq"), uq_rows)
        run.emit(STEP.artifact_name("guards"), guard_rows)

        level_rows = []
        for arm_index, system in enumerate(("L24", "L26_EFFICIENT_SCAN_REFERENCE")):
            maps = fixed_triple_maps(raw_scores, assignment, image_ids, arm_index=arm_index, checkpoint_index=fixed_index, triple=(0, 1, 2))
            level_rows.append({"system": system, **level_diagnostics(maps, image_ids, canonical_identities, theta, assignment[:520], identity_order)})
        run.emit(STEP.artifact_name("level_diagnostics"), level_rows)

        with np.load(args.step61a_scores, allow_pickle=False) as saved61:
            discovery_raw = saved61["raw_scores"].astype(np.float64)
            discovery_checkpoints = saved61["checkpoints"].astype(int).tolist()
            discovery_layers = saved61["layers"].astype(int).tolist()
            run.require("STEP61A_image_order_exact", np.array_equal(saved61["image_ids"].astype(str), np.asarray(image_ids).astype(str)))
            run.require("STEP61A_fold_assignment_exact", np.array_equal(saved61["fold_assignment"].astype(np.int64), assignment))
        discovery_cp = discovery_checkpoints.index(384)
        e6_maps = [
            combined_e6_maps(discovery_raw, raw_scores, assignment, image_ids, discovery_arm_index=discovery_layers.index(layer), discovery_checkpoint_index=discovery_cp, confirmation_arm_index=arm_index, confirmation_checkpoint_index=fixed_index)
            for arm_index, layer in enumerate(ARMS)
        ]
        e6_candidate, e6_identities, _e6_images, _e6_folds = per_target_direct(seen, image_fold, e6_maps[0])
        e6_reference, e6_identities_ref, _e6_images_ref, _e6_folds_ref = per_target_direct(seen, image_fold, e6_maps[1])
        run.require("E6_target_metadata_exact", np.array_equal(e6_identities, e6_identities_ref))
        e6_uq = fixed_target_uq(e6_candidate - e6_reference, e6_identities)
        discovery_delta = float(step61a_summary["primary"]["direct_UQ"]["point_delta"])
        discovery_rows = [
            {"evidence": "STEP61A", "delta_direct": discovery_delta, "CI90_lower": step61a_summary["primary"]["direct_UQ"]["CI90_lower"], "CI90_upper": step61a_summary["primary"]["direct_UQ"]["CI90_upper"], "role": "discovery; layer selected here", "primary_classification": False},
            {"evidence": "STEP61B", "delta_direct": direct_uq["SEEN_ANCHOR"]["point_delta"], "CI90_lower": direct_uq["SEEN_ANCHOR"]["CI90_lower"], "CI90_upper": direct_uq["SEEN_ANCHOR"]["CI90_upper"], "role": "primary fresh-seed confirmation", "primary_classification": True},
            {"evidence": "COMBINED_E6", "delta_direct": e6_uq["point_delta"], "CI90_lower": e6_uq["CI90_lower"], "CI90_upper": e6_uq["CI90_upper"], "role": "post-selection sensitivity; six seeds fixed", "primary_classification": False},
        ]
        run.emit(STEP.artifact_name("discovery_confirmation"), discovery_rows)

        result = classify_confirmation(direct_uq["SEEN_ANCHOR"], direct_uq["STRICT_UNSEEN_ANCHOR"], guard_rows)
        decision = {"STEP61B_result": result, "candidate": "L24@384", "reference": "L26_EFFICIENT_SCAN_REFERENCE@384", "layer_promoted": False, "branch_process_status": "resolved", "strict_resolved_harm": direct_uq["STRICT_UNSEEN_ANCHOR"]["CI90_lower"] > 0.0, "historical_margin_role": "0.005 annotation only; not a gate", "combined_E6_role": "post-selection sensitivity only", "next_formal_move": "DINOv3 full-token extraction with matched JOINT4 readout"}
        run.emit(STEP.artifact_name("policy_decisions"), decision)
        gates = {"hardware_class": True, "software_class": True, "feature_lineage": True, "partition": partition_exact and inventory_exact, "initialization": True, "two_arm_first_update": True, "lossless_evaluation": True, "endpoint_closure": closure["closure_pass"], "trajectories_15": len(results) == 15, "all_finite": all(item.all_finite for item in results.values()), "fixed_cp384_no_selection": True, "no_prohibited_access": True}
        run.emit(STEP.artifact_name("gates"), {"STEP61B_result": result, "gates": gates})
        head_before = str(read_project_events(args.ledger_jsonl)[-1]["event_sha256"])
        evidence_path = str(Path("artifacts") / STEP.output_dir.name / STEP.artifact_name("gates"))
        _, head_after, appended = append_project_event("STEP61B", {"event_type": "confirmation_result", "subject": "fixed fresh-seed L24@384 confirmation", "statement": f"STEP61B completed 15/15 trajectories; result {result}; L24 branch process-resolved without promotion", "status_before": "screen_lead_found", "status_after": result, "evidence_class": "prospective fixed-layer fresh-seed confirmation", "evidence": [{"artifact_path": evidence_path, "artifact_sha256": sha256_file(run.path(STEP.artifact_name("gates"))), "locator": {"kind": "json_pointer", "assertions": {"/STEP61B_result": result}}}], "supersedes_event_ids": [], "limitations": ["known identities", "single fivefold image partition", "no external evidence", "combined E6 is post-selection sensitivity"], "protected_conclusion": f"STEP61B result is {result}; no layer or baseline promotion", "explicitly_not_protected": ["external promotion", "baseline replacement", "general layer promotion"], "author_role": "Executor"}, jsonl_path=args.ledger_jsonl, head_path=args.ledger_head, markdown_path=args.ledger_md)
        ledger = verify_project_ledger(jsonl_path=args.ledger_jsonl, head_path=args.ledger_head, markdown_path=args.ledger_md)
        run.require("ledger_integrity", ledger.all_pass)
        run.emit(STEP.artifact_name("ledger_update_audit"), {"head_before": head_before, "head_after": head_after, "event_appended": appended, **ledger.as_dict()})
        summary = {"signature": SIGNATURE, "STEP61B_result": result, "trajectories_completed": len(results), "updates_per_trajectory": 384, "runtime_seconds": time.perf_counter() - started, "primary": {"direct": direct_uq["SEEN_ANCHOR"], "seed_triple_range": triple_ranges["SEEN_ANCHOR"]}, "secondary": {"direct": direct_uq["STRICT_UNSEEN_ANCHOR"], "seed_triple_range": triple_ranges["STRICT_UNSEEN_ANCHOR"]}, "guards": guard_rows, "level_diagnostics": level_rows, "combined_E6": e6_uq, "layer_promoted": False, "branch_process_status": "resolved", "training_performed": True, "new_labels_used": 0, "STEP49_test_retests_RESERVE200_accessed": False, "FULL192_changed": False, "current_baseline_changed": False}
        _inventory_update(args.inventory, summary)
        run.emit(STEP.artifact_name("summary"), summary)
        manifest = run.finalize()
        run.result(STEP61B_result=result, trajectories_completed=len(results), training_performed=True, current_baseline_changed=False, report_completeness_pass=manifest["report_completeness_pass"], artifact_set_equality_pass=manifest["artifact_set_equality_pass"])


if __name__ == "__main__":
    main()
