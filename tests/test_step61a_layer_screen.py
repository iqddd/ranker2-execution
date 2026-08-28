from __future__ import annotations

from pathlib import Path

import numpy as np

from ranker.artifacts import verify_artifact
from ranker.experiments.layer_screen.step61_analysis import select_lofo_maps
from ranker.io import atomic_text
from ranker.step_registry import STEPS


def test_step61a_checkpoint_and_artifact_contract() -> None:
    step = STEPS["61a"]
    assert step.artifact_name("summary") == "STEP61A_SUMMARY.json"
    assert step.artifact_name("driver_source") == "step61a_full_l10_l25_block_skip_scan.py"


def test_layerwise_and_joint_lofo_ties() -> None:
    layers = (10, 11, 12)
    checkpoints = (0, 32, 64)
    values = np.ones((3, 3, 5), dtype=np.float64)
    values[0, 2, :] = 0.8
    values[1, 1, :] = 0.7
    values[2, 0, :] = 0.7
    layerwise, joint = select_lofo_maps(values, layers, checkpoints)
    assert np.all(layerwise[0] == 2)
    assert np.all(layerwise[1] == 1)
    assert np.all(layerwise[2] == 0)
    # Exact direct tie prefers deeper layer, then earlier checkpoint.
    assert np.all(joint == np.asarray((2, 0)))


def test_text_artifact_verification(tmp_path: Path) -> None:
    source = tmp_path / "driver.py"
    atomic_text(source, "print('ok')\n")
    result = verify_artifact(source, set(), kind="text")
    assert result["verified"] is True


def test_step61a_training_suffix_is_all_efficient() -> None:
    source = Path("ranker/experiments/layer_screen/simultaneous.py").read_text(
        encoding="utf-8"
    )
    assert '_attn_implementation = "eager"' not in source
    assert '_attn_implementation = "sdpa"' in source
    assert "All-efficient shadow mismatch" in source


def test_step61a_reference_is_not_historical_step60a() -> None:
    source = Path("step61a_full_l10_l25_block_skip_scan.py").read_text(
        encoding="utf-8"
    )
    assert "L26_EFFICIENT_SCAN_REFERENCE" in source
    assert '"historical_STEP60A_surface_equivalence": False' in source
    assert "backend_matched_L26_parity_failed" in source


def test_step61a_uses_one_optimizer_per_head() -> None:
    source = Path("ranker/experiments/layer_screen/training.py").read_text(
        encoding="utf-8"
    )
    assert "optimizers = [ProdigyPlusScheduleFree([group]" in source
    assert "neutral_schedulefree_multi_evaluation" in source
