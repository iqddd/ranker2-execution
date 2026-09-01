from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from ranker.experiments.layer_screen.confirmation import classify_confirmation
from ranker.experiments.layer_screen.simultaneous import MultiLayerScreen
from ranker.step_registry import STEPS


def _direct(point: float, lower: float, upper: float) -> dict[str, float]:
    return {"point_delta": point, "CI90_lower": lower, "CI90_upper": upper}


def test_step61b_registry_and_fixed_contract() -> None:
    step = STEPS["61b"]
    assert step.artifact_name("summary") == "STEP61B_SUMMARY.json"
    source = Path("step61b_l24_fixed_confirmation.py").read_text(encoding="utf-8")
    assert "ARMS = (24, 26)" in source
    assert "FIXED_CHECKPOINT = 384" in source
    assert "select_lofo_maps" not in source


def test_selected_heads_do_not_change_stream_source() -> None:
    source = Path("ranker/experiments/layer_screen/simultaneous.py").read_text(
        encoding="utf-8"
    )
    assert "head_layers must lie inside the streamed layer range" in source
    assert "for layer in range(self.first_layer, self.last_layer + 1)" in source
    assert "self.suffix[layer - self.first_layer]" in source

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="sdpa")
            )

        def forward(
            self, values: torch.Tensor, attention_mask: None = None
        ) -> torch.Tensor:
            return values

    template = nn.Module()
    template.encoder = SimpleNamespace(layers=[Block() for _ in range(27)])
    template.post_layernorm = nn.Identity()
    screen = MultiLayerScreen(
        template=template,
        seed=20260806,
        device=torch.device("cpu"),
        head_layers=(24, 26),
    )
    assert screen.first_layer == 10
    assert screen.last_layer == 26
    assert screen.head_layers == (24, 26)
    assert len(screen.suffix) == 16
    assert len(screen.heads) == 2


def test_confirmation_decision_precedence() -> None:
    safe = [{"resolved_harm": False}, {"resolved_harm": False}]
    assert classify_confirmation(_direct(-0.001, -0.002, -0.0001), _direct(0.001, 0.0001, 0.002), safe) == "confirmed_seen_anchor_only_strict_harm"
    assert classify_confirmation(_direct(-0.001, -0.002, -0.0001), _direct(0.0, -0.001, 0.001), safe) == "confirmed_layer_improvement"
    assert classify_confirmation(_direct(-0.001, -0.002, 0.001), _direct(0.0, -0.001, 0.001), safe) == "direction_replicated_but_unresolved"
    assert classify_confirmation(_direct(0.001, -0.001, 0.002), _direct(0.0, -0.001, 0.001), safe) == "screen_lead_not_replicated"
    assert classify_confirmation(_direct(-0.001, -0.002, -0.0001), _direct(0.0, -0.001, 0.001), [{"resolved_harm": True}]) == "guard_failed"


def test_e6_is_not_a_primary_decision_input() -> None:
    source = Path("step61b_l24_fixed_confirmation.py").read_text(encoding="utf-8")
    decision_call = "classify_confirmation(direct_uq[\"SEEN_ANCHOR\"], direct_uq[\"STRICT_UNSEEN_ANCHOR\"], guard_rows)"
    assert decision_call in source
    assert "e6_uq" not in decision_call
    assert "post-selection sensitivity" in source
