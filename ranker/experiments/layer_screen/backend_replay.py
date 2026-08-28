"""Token-backend sensitivity replay for retained frozen-layer readouts."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ranker.attention_runtime import load_vision_model
from ranker.experiments.encoder_adaptation import ProducerBlockInputCache
from ranker.experiments.layer_screen.simultaneous import LayerJointHead
from ranker.external_validation import percentile_against
from ranker.last_block_finetune import ExactPrefixCache
from ranker.runtime_preflight import efficient_sdpa_only
from ranker.validation_resolution import evaluate_direct_ordinal


def replay_l25_backend_sensitivity(
    *,
    state_paths: Mapping[tuple[str, int], Path],
    selected_checkpoints: Mapping[str, int],
    identities: Sequence[str],
    seeds: Sequence[int],
    canonical_names: np.ndarray,
    canonical_identities: np.ndarray,
    exposures: Sequence[Mapping[str, Any]],
    model_dir: Path,
    old_prefix_path: Path,
    extra_prefix_path: Path,
    pre25_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Compare historical and efficient-SDPA L25 tokens on fixed trained heads."""
    started = time.perf_counter()
    prefix = ExactPrefixCache.load(old_prefix_path, extra_prefix_path)
    pre25 = ProducerBlockInputCache.load(pre25_path)
    if prefix.names != pre25.names:
        raise RuntimeError("PRE25 and exact-prefix image orders differ.")
    template, _processor = load_vision_model(model_dir, device)
    block = copy.deepcopy(template.encoder.layers[25]).to(device).eval()
    block.self_attn.config._attn_implementation = "sdpa"
    norm = copy.deepcopy(template.post_layernorm).to(device).eval()
    ordered = [(identity, int(seed)) for identity in identities for seed in seeds]
    heads: list[LayerJointHead] = []
    historical = np.empty((len(ordered), 520), dtype=np.float32)
    for identity, seed in ordered:
        checkpoint = int(selected_checkpoints[identity])
        payload = torch.load(
            state_paths[(identity, checkpoint, seed)],
            map_location="cpu",
            weights_only=False,
        )
        state = payload["readout_state"]
        historical[len(heads)] = np.asarray(payload["raw_scores"], dtype=np.float32)[
            :520
        ]
        head = LayerJointHead(seed=seed, device=device).eval()
        head.attention.load_state_dict(state["head_state_dict"], strict=True)
        with torch.no_grad():
            head.linear_weight.copy_(
                torch.as_tensor(state["linear_weight"], device=device)
            )
            head.linear_bias.copy_(torch.as_tensor(state["linear_bias"], device=device))
        heads.append(head)

    runtime = np.empty_like(historical)
    maximum_token_difference = 0.0
    with torch.inference_mode():
        for start in range(0, 710, 32):
            stop = min(start + 32, 710)
            indices = torch.arange(start, stop, dtype=torch.int64)
            historical_pre = prefix.pre_last(indices, device)
            reconstructed_pre = pre25.take(indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                historical_tokens = norm(historical_pre)
            with efficient_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                runtime_tokens = norm(block(reconstructed_pre, attention_mask=None))
            maximum_token_difference = max(
                maximum_token_difference,
                float((runtime_tokens - historical_tokens).abs().max()),
            )
            if start >= 520:
                continue
            canonical_stop = min(stop, 520)
            size = canonical_stop - start
            for head_index, head in enumerate(heads):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    historical_attention, _ = head.attention(
                        historical_tokens[:size]
                    )
                    runtime_attention, _ = head.attention(runtime_tokens[:size])
                delta = (runtime_attention - historical_attention).cpu().numpy()
                runtime[head_index, start:canonical_stop] = (
                    historical[head_index, start:canonical_stop] + delta
                )

    def ensemble_maps(scores: np.ndarray) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for identity_index, identity in enumerate(identities):
            seed_rows = scores[
                identity_index * len(seeds) : (identity_index + 1) * len(seeds)
            ]
            transformed = [
                percentile_against(row, row[canonical_identities != identity])
                for row in seed_rows
            ]
            ensemble = np.mean(np.stack(transformed), axis=0)
            output[identity] = {
                name: float(value)
                for name, value in zip(canonical_names, ensemble, strict=True)
            }
        return output

    historical_maps = ensemble_maps(historical)
    runtime_maps = ensemble_maps(runtime)
    historical_result = evaluate_direct_ordinal(exposures, historical_maps)
    runtime_result = evaluate_direct_ordinal(exposures, runtime_maps)
    changed = 0
    comparisons = 0
    for exposure in exposures:
        identity = str(exposure["identity_id"])
        if identity not in historical_maps:
            continue
        key = str(exposure["image_key"])
        for anchor in exposure["anchor_keys"]:
            historical_order = np.sign(
                historical_maps[identity][key] - historical_maps[identity][str(anchor)]
            )
            runtime_order = np.sign(
                runtime_maps[identity][key] - runtime_maps[identity][str(anchor)]
            )
            changed += int(historical_order != runtime_order)
            comparisons += 1
    return {
        "status": "computed",
        "population_images": 710,
        "trained_states": len(heads),
        "maximum_token_difference": maximum_token_difference,
        "maximum_trained_score_difference": float(np.max(np.abs(runtime - historical))),
        "changed_pair_orders": changed,
        "pair_order_comparisons": comparisons,
        "fraction_changed_pair_orders": float(changed / comparisons),
        "historical_direct_loss": float(historical_result["loss"]),
        "runtime_backend_direct_loss": float(runtime_result["loss"]),
        "direct_loss_difference": float(runtime_result["loss"])
        - float(historical_result["loss"]),
        "elapsed_seconds": time.perf_counter() - started,
        "new_cache_written": False,
    }
