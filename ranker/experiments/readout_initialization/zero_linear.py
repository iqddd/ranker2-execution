"""Pure helpers for a paired zero pooled-linear JOINT4 initialization."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch

from ranker.frozen_readouts import (
    FrozenFeatureRuntime,
    ReadoutSpec,
    StandardReadout,
    configure_determinism,
    export_readout_state,
    make_model,
)
from ranker.reproducibility import tensor_state_hash


def paired_zero_linear_states(
    runtime: FrozenFeatureRuntime,
    spec: ReadoutSpec,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Construct the historical state, then zero only its copied linear branch."""
    configure_determinism(seed)
    reference = make_model(runtime, spec, seed)
    if not isinstance(reference, StandardReadout):
        raise TypeError("Zero-linear pairing requires StandardReadout.")
    reference_state = export_readout_state(reference)
    candidate_state = copy.deepcopy(reference_state)
    with torch.no_grad():
        torch.as_tensor(candidate_state["linear_weight"]).zero_()
        torch.as_tensor(candidate_state["linear_bias"]).zero_()

    unrelated_exact = True
    for key in reference_state:
        if key == "linear_weight":
            continue
        reference_value = reference_state[key]
        candidate_value = candidate_state[key]
        if isinstance(reference_value, torch.Tensor):
            unrelated_exact &= torch.equal(reference_value, candidate_value)
        elif isinstance(reference_value, Mapping):
            unrelated_exact &= set(reference_value) == set(candidate_value)
            unrelated_exact &= all(
                torch.equal(
                    torch.as_tensor(reference_value[name]),
                    torch.as_tensor(candidate_value[name]),
                )
                for name in reference_value
            )
        else:
            unrelated_exact &= reference_value == candidate_value

    reference_names = [name for name, _ in reference.named_parameters()]
    audit = {
        "seed": seed,
        "reference_state_SHA256": tensor_state_hash(reference_state),
        "candidate_state_SHA256": tensor_state_hash(candidate_state),
        "all_non_linear_weight_parameters_bit_exact": bool(unrelated_exact),
        "linear_weight_exactly_zero": bool(
            torch.count_nonzero(candidate_state["linear_weight"]) == 0
        ),
        "linear_bias_exactly_zero": bool(
            torch.count_nonzero(candidate_state["linear_bias"]) == 0
        ),
        "attention_output_weight_exactly_zero": bool(
            torch.count_nonzero(candidate_state["head_state_dict"]["output.weight"])
            == 0
        ),
        "attention_output_bias_exactly_zero": bool(
            torch.count_nonzero(candidate_state["head_state_dict"]["output.bias"]) == 0
        ),
        "parameter_names": reference_names,
        "parameter_count": sum(
            parameter.numel() for _, parameter in reference.named_parameters()
        ),
    }
    del reference
    return reference_state, candidate_state, audit
