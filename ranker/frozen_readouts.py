"""Reusable frozen-feature readout candidates and V2 trajectory training."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from ranker.attention_lora import AttentionResidualHead
from ranker.data import RankedImage
from ranker.trainer_contract import (
    CyclingGroupSampler,
    heldout_fold_metrics,
    reliability_weight,
    reliability_weighted_group_loss,
    scheduled_learning_rate,
)


@dataclass(frozen=True)
class ReadoutSpec:
    candidate_id: str
    mode: str
    queries: int = 0

    @property
    def value_dimension(self) -> int:
        return 0 if self.queries == 0 else 1024 // self.queries

    @property
    def requires_pooled(self) -> bool:
        return self.mode in {"linear", "residual", "joint", "level_deviation"}

    @property
    def requires_tokens(self) -> bool:
        return self.mode != "linear"


@dataclass
class FrozenFeatureRuntime:
    names: list[str]
    pooled: torch.Tensor
    tokens: torch.Tensor
    name_to_index: dict[str, int]
    canonical_indices: torch.Tensor
    device: torch.device


@dataclass
class TrainingFold:
    identity: str
    retained_o: list[list[RankedImage]]
    retained_w: list[list[RankedImage]]
    group_indices: dict[tuple[str, int], torch.Tensor]
    validation_mask: np.ndarray
    reference_mask: np.ndarray


def training_fold_pair_work(fold: TrainingFold) -> float:
    """Estimate per-update work from mean O/W pair counts in a training fold."""
    return float(
        np.mean([len(group) * (len(group) - 1) / 2 for group in fold.retained_o])
        + np.mean([len(group) * (len(group) - 1) / 2 for group in fold.retained_w])
    )


class ReadoutModel(Protocol):
    spec: ReadoutSpec

    def named_parameters(self) -> Sequence[tuple[str, torch.nn.Parameter]]: ...

    def trainable_parameters(self) -> list[torch.nn.Parameter]: ...

    def train_scores(self, indices: torch.Tensor) -> torch.Tensor: ...

    def group_loss(
        self,
        indices: torch.Tensor,
        group: Sequence[RankedImage],
        source: str,
        pair_weighting: str,
    ) -> torch.Tensor: ...

    def capture_scores(
        self, indices: torch.Tensor, reference_positions: torch.Tensor
    ) -> torch.Tensor: ...

    def train(self) -> None: ...


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pooled_cache(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    names = payload.get("image_names")
    raw = payload.get("features")
    if (
        not isinstance(names, list)
        or not isinstance(raw, dict)
        or set(names) != set(raw)
    ):
        raise RuntimeError(f"Invalid pooled feature cache {path}.")
    features = {name: np.asarray(raw[name], dtype=np.float32) for name in names}
    if any(
        value.shape != (1152,) or not np.all(np.isfinite(value))
        for value in features.values()
    ):
        raise RuntimeError(f"Invalid pooled feature vector in {path}.")
    return names, features


def load_token_caches(
    old_path: Path, extra_path: Path, device: torch.device
) -> tuple[list[str], torch.Tensor]:
    old = torch.load(old_path, map_location="cpu", weights_only=False, mmap=True)
    extra = torch.load(extra_path, map_location="cpu", weights_only=False, mmap=True)
    old_names = list(map(str, old["image_names"]))
    extra_names = list(map(str, extra["image_names"]))
    if (
        len(old_names) != 520
        or len(extra_names) != 190
        or set(old_names) & set(extra_names)
    ):
        raise RuntimeError("Frozen token populations are not disjoint 520+190.")
    names = [*old_names, *extra_names]
    tokens = torch.empty((710, 1024, 1152), dtype=torch.bfloat16, device=device)
    for start in range(0, 520, 8):
        stop = min(start + 8, 520)
        tokens[start:stop].copy_(
            old["frozen_final"][start:stop].to(device=device, dtype=torch.bfloat16)
        )
    for start in range(0, 190, 8):
        stop = min(start + 8, 190)
        tokens[520 + start : 520 + stop].copy_(
            extra["frozen_final"][start:stop].to(device=device, dtype=torch.bfloat16)
        )
    if not bool(torch.isfinite(tokens).all()):
        raise RuntimeError("Frozen patch-token cache contains non-finite values.")
    return names, tokens


def optimizer_for_parameters(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in named_parameters:
        lower = name.lower()
        if (
            lower.endswith(("bias", "queries"))
            or "norm" in lower
            or "temperature" in lower
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def linear_scores(
    pooled: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return (
        pooled.index_select(0, indices)
        .matmul(weight.to(torch.bfloat16))
        .add(bias.to(torch.bfloat16))
        .float()
    )


def attention_scores(
    head: AttentionResidualHead,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int = 128,
) -> torch.Tensor:
    was_training = head.training
    head.eval()
    output: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                score, _ = head(tokens.index_select(0, selected))
            output.append(score)
    if was_training:
        head.train()
    return torch.cat(output)


class StandardReadout:
    def __init__(
        self,
        runtime: FrozenFeatureRuntime,
        spec: ReadoutSpec,
        seed: int,
        base_scores: np.ndarray | None,
        *,
        attention_zero_output: bool | None = None,
    ) -> None:
        self.runtime = runtime
        self.spec = spec
        self.linear_weight: torch.nn.Parameter | None = None
        self.linear_bias: torch.nn.Parameter | None = None
        self.head: AttentionResidualHead | None = None
        self.base_scores: torch.Tensor | None = None
        if spec.mode in {"linear", "joint"}:
            self.linear_weight = torch.nn.Parameter(
                torch.randn(1152, dtype=torch.float32, device=runtime.device)
                / math.sqrt(1152.0)
            )
            self.linear_bias = torch.nn.Parameter(
                torch.zeros((), dtype=torch.float32, device=runtime.device)
            )
        if spec.requires_tokens:
            zero_output = (
                spec.mode in {"residual", "joint"}
                if attention_zero_output is None
                else bool(attention_zero_output)
            )
            self.head = AttentionResidualHead(
                seed=seed,
                initial_query_norm=math.sqrt(128.0),
                query_count=spec.queries,
                value_dimension=spec.value_dimension,
                zero_output=zero_output,
            ).to(runtime.device)
        if spec.mode == "residual":
            if base_scores is None or len(base_scores) != len(runtime.names):
                raise ValueError(
                    "Residual readout requires one frozen base score per image."
                )
            self.base_scores = torch.as_tensor(
                base_scores, dtype=torch.float32, device=runtime.device
            )

    def named_parameters(self) -> Sequence[tuple[str, torch.nn.Parameter]]:
        values: list[tuple[str, torch.nn.Parameter]] = []
        if self.linear_weight is not None and self.linear_bias is not None:
            values.extend(
                (
                    ("linear.weight", self.linear_weight),
                    ("linear.bias", self.linear_bias),
                )
            )
        if self.head is not None:
            values.extend(
                (f"attention.{name}", value)
                for name, value in self.head.named_parameters()
            )
        return values

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for _, parameter in self.named_parameters()]

    def _linear(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.linear_weight is not None and self.linear_bias is not None
        return linear_scores(
            self.runtime.pooled, indices, self.linear_weight, self.linear_bias
        )

    def _attention_train(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.head is not None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            score, _ = self.head(self.runtime.tokens.index_select(0, indices))
        return score

    def train_scores(self, indices: torch.Tensor) -> torch.Tensor:
        if self.spec.mode == "linear":
            return self._linear(indices)
        attention = self._attention_train(indices)
        if self.spec.mode == "main":
            return attention
        if self.spec.mode == "joint":
            return self._linear(indices) + attention
        assert self.base_scores is not None
        return self.base_scores.index_select(0, indices) + attention

    def group_loss(
        self,
        indices: torch.Tensor,
        group: Sequence[RankedImage],
        source: str,
        pair_weighting: str,
    ) -> torch.Tensor:
        return reliability_weighted_group_loss(
            self.train_scores(indices),
            group,
            source,
            pair_weighting=pair_weighting,
        )

    def capture_scores(
        self, indices: torch.Tensor, reference_positions: torch.Tensor
    ) -> torch.Tensor:
        del reference_positions
        if self.spec.mode == "linear":
            return self._linear(indices)
        assert self.head is not None
        attention = attention_scores(self.head, self.runtime.tokens, indices)
        if self.spec.mode == "main":
            return attention
        if self.spec.mode == "joint":
            return self._linear(indices) + attention
        assert self.base_scores is not None
        return self.base_scores.index_select(0, indices) + attention

    def train(self) -> None:
        if self.head is not None:
            self.head.train()


class LevelDeviationReadout:
    def __init__(
        self, runtime: FrozenFeatureRuntime, spec: ReadoutSpec, seed: int
    ) -> None:
        self.runtime = runtime
        self.spec = spec
        self.linear_weight = torch.nn.Parameter(
            torch.randn(1152, dtype=torch.float32, device=runtime.device)
            / math.sqrt(1152.0)
        )
        self.linear_bias = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float32, device=runtime.device)
        )
        self.head = AttentionResidualHead(
            seed=seed,
            initial_query_norm=math.sqrt(128.0),
            query_count=4,
            value_dimension=256,
            zero_output=False,
        ).to(runtime.device)

    def named_parameters(self) -> Sequence[tuple[str, torch.nn.Parameter]]:
        return [
            ("level.weight", self.linear_weight),
            ("level.bias", self.linear_bias),
            *[
                (f"deviation.{name}", parameter)
                for name, parameter in self.head.named_parameters()
            ],
        ]

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for _, parameter in self.named_parameters()]

    def _level(self, indices: torch.Tensor) -> torch.Tensor:
        return linear_scores(
            self.runtime.pooled,
            indices,
            self.linear_weight,
            self.linear_bias,
        )

    def _deviation_train(self, indices: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            score, _ = self.head(self.runtime.tokens.index_select(0, indices))
        return score

    def train_scores(self, indices: torch.Tensor) -> torch.Tensor:
        return self._level(indices) + self._deviation_train(indices)

    def group_loss(
        self,
        indices: torch.Tensor,
        group: Sequence[RankedImage],
        source: str,
        pair_weighting: str,
    ) -> torch.Tensor:
        level = self._level(indices)
        deviation = self._deviation_train(indices)
        losses: list[torch.Tensor] = []
        weights: list[float] = []
        for first_index, first in enumerate(group):
            for second_index in range(first_index + 1, len(group)):
                second = group[second_index]
                if first.rank < second.rank:
                    lower, higher = first_index, second_index
                    distance = second.rank - first.rank
                else:
                    lower, higher = second_index, first_index
                    distance = first.rank - second.rank
                weight = (
                    reliability_weight(source, int(distance))
                    if pair_weighting == "REL"
                    else 1.0
                )
                branch = (
                    deviation
                    if source == "W" or first.identity == second.identity
                    else level
                )
                losses.append(
                    torch.nn.functional.softplus(-(branch[higher] - branch[lower]))
                    * weight
                )
                weights.append(weight)
        return torch.stack(losses).sum() / float(sum(weights))

    def capture_scores(
        self, indices: torch.Tensor, reference_positions: torch.Tensor
    ) -> torch.Tensor:
        level = self._level(indices)
        deviation = attention_scores(self.head, self.runtime.tokens, indices)
        level_reference = level.index_select(0, reference_positions)
        deviation_reference = deviation.index_select(0, reference_positions)
        level_sd = level_reference.std(unbiased=False)
        deviation_sd = deviation_reference.std(unbiased=False)
        if not bool(level_sd > 0.0 and deviation_sd > 0.0):
            raise RuntimeError("LEVEL_DEVIATION reference SD is zero.")
        return (level - level_reference.mean()) / level_sd + (
            deviation - deviation_reference.mean()
        ) / deviation_sd

    def train(self) -> None:
        self.head.train()


def make_model(
    runtime: FrozenFeatureRuntime,
    spec: ReadoutSpec,
    seed: int,
    base_scores: np.ndarray | None = None,
    *,
    attention_zero_output: bool = True,
) -> ReadoutModel:
    if spec.mode == "level_deviation":
        return LevelDeviationReadout(runtime, spec, seed)
    override = attention_zero_output if spec.mode == "joint" else None
    return StandardReadout(
        runtime,
        spec,
        seed,
        base_scores,
        attention_zero_output=override,
    )


def export_readout_state(model: ReadoutModel) -> dict[str, Any]:
    """Return an optimizer-independent CPU checkpoint for a frozen readout."""
    if not isinstance(model, StandardReadout):
        raise TypeError("Only standard readouts can seed last-block fine-tuning.")
    payload: dict[str, Any] = {
        "candidate_id": model.spec.candidate_id,
        "mode": model.spec.mode,
        "queries": model.spec.queries,
    }
    if model.linear_weight is not None and model.linear_bias is not None:
        payload["linear_weight"] = model.linear_weight.detach().float().cpu().clone()
        payload["linear_bias"] = model.linear_bias.detach().float().cpu().clone()
    if model.head is not None:
        payload["head_state_dict"] = {
            name: value.detach().float().cpu().clone()
            for name, value in model.head.state_dict().items()
        }
    return payload


def load_readout_state(model: ReadoutModel, payload: Mapping[str, Any]) -> None:
    """Load a checkpoint emitted by :func:`export_readout_state`."""
    if not isinstance(model, StandardReadout):
        raise TypeError("Only standard readouts support materialized checkpoints.")
    expected = (model.spec.candidate_id, model.spec.mode, model.spec.queries)
    observed = (
        str(payload.get("candidate_id")),
        str(payload.get("mode")),
        int(payload.get("queries", -1)),
    )
    if observed != expected:
        raise RuntimeError(
            f"Readout checkpoint mismatch: {observed!r} != {expected!r}."
        )
    with torch.no_grad():
        if model.linear_weight is not None and model.linear_bias is not None:
            model.linear_weight.copy_(
                torch.as_tensor(payload["linear_weight"], device=model.runtime.device)
            )
            model.linear_bias.copy_(
                torch.as_tensor(payload["linear_bias"], device=model.runtime.device)
            )
        if model.head is not None:
            model.head.load_state_dict(payload["head_state_dict"], strict=True)


def attention_only_state_from_joint(
    joint_state: Mapping[str, Any], *, candidate_id: str
) -> dict[str, Any]:
    """Create a MAIN4 state with the exact attention tensors from JOINT4."""
    if (
        str(joint_state.get("mode")) != "joint"
        or int(joint_state.get("queries", -1)) != 4
        or "head_state_dict" not in joint_state
    ):
        raise ValueError("Expected an exported JOINT4 state.")
    return {
        "candidate_id": candidate_id,
        "mode": "main",
        "queries": 4,
        "head_state_dict": {
            name: torch.as_tensor(value).detach().float().cpu().clone()
            for name, value in joint_state["head_state_dict"].items()
        },
    }


def pooled_branch_ablation_parity(
    runtime: FrozenFeatureRuntime,
    *,
    seed: int,
    indices: torch.Tensor,
    joint_candidate_id: str = "L26_JOINT4",
    main_candidate_id: str = "L26_MAIN4",
) -> dict[str, Any]:
    """Audit paired JOINT4/MAIN4 initialization and score decomposition."""
    configure_determinism(seed)
    joint = make_model(runtime, ReadoutSpec(joint_candidate_id, "joint", 4), seed)
    if not isinstance(joint, StandardReadout):
        raise TypeError("JOINT4 parity requires StandardReadout.")
    joint_state = export_readout_state(joint)
    main_state = attention_only_state_from_joint(
        joint_state, candidate_id=main_candidate_id
    )
    configure_determinism(seed)
    main = make_model(runtime, ReadoutSpec(main_candidate_id, "main", 4), seed)
    if not isinstance(main, StandardReadout):
        raise TypeError("MAIN4 parity requires StandardReadout.")
    load_readout_state(main, main_state)

    joint_attention = dict(joint.named_parameters())
    main_attention = dict(main.named_parameters())
    joint_names = [name for name in joint_attention if name.startswith("attention.")]
    shared_exact = set(main_attention) == set(joint_names) and all(
        joint_attention[name].shape == main_attention[name].shape
        and torch.equal(joint_attention[name], main_attention[name])
        for name in joint_names
    )
    with torch.no_grad():
        joint_scores = joint.capture_scores(indices, indices)
        main_scores = main.capture_scores(indices, indices)
        linear_scores_only = joint._linear(indices)
    decomposition = float(
        torch.max(torch.abs((joint_scores - main_scores) - linear_scores_only))
        .detach()
        .cpu()
    )
    joint_count = sum(parameter.numel() for parameter in joint.trainable_parameters())
    main_count = sum(parameter.numel() for parameter in main.trainable_parameters())
    pooled_count = int(joint.linear_weight.numel() + joint.linear_bias.numel())
    result = {
        "seed": seed,
        "diagnostic_images": int(indices.numel()),
        "shared_attention_parameter_names": joint_names,
        "shared_attention_initialization_exact": shared_exact,
        "maximum_forward_decomposition_difference": decomposition,
        "JOINT4_trainable_parameters": joint_count,
        "MAIN4_trainable_parameters": main_count,
        "pooled_linear_parameters": pooled_count,
        "parameter_reduction": joint_count - main_count,
        "parameter_set_exact": joint_count - main_count == pooled_count,
    }
    del joint, main
    return result


def materialize_trajectory(
    *,
    runtime: FrozenFeatureRuntime,
    fold: TrainingFold,
    spec: ReadoutSpec,
    learning_rate: float,
    selected_checkpoint: int,
    seed: int,
    pair_weighting: str,
    o_coefficient: float,
    w_coefficient: float,
    weight_decay: float,
    target_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    base_scores: np.ndarray | None = None,
    tick: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay a selected frozen-readout trajectory and retain its exact weights."""
    if selected_checkpoint not in (0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        raise ValueError("Selected checkpoint is outside TRAINER_CONTRACT_V2.")
    configure_determinism(seed)
    model = make_model(runtime, spec, seed, base_scores)
    optimizer = optimizer_for_parameters(
        model.named_parameters(), learning_rate, weight_decay
    )
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)
    model.train()
    for update in range(1, selected_checkpoint + 1):
        if tick is not None:
            tick()
        lr = scheduled_learning_rate(learning_rate, update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        o_index = o_sampler.next()
        w_index = w_sampler.next()
        (
            o_coefficient
            * model.group_loss(
                fold.group_indices[("O", o_index)],
                fold.retained_o[o_index],
                "O",
                pair_weighting,
            )
        ).backward()
        (
            w_coefficient
            * model.group_loss(
                fold.group_indices[("W", w_index)],
                fold.retained_w[w_index],
                "W",
                pair_weighting,
            )
        ).backward()
        parameters = model.trainable_parameters()
        if not all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise RuntimeError(f"{spec.candidate_id} gradient became non-finite.")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise RuntimeError(f"{spec.candidate_id} parameter became non-finite.")
    with torch.no_grad():
        scores = (
            model.capture_scores(target_indices, reference_positions)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    state = export_readout_state(model)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"{spec.candidate_id} materialized scores are non-finite.")
    del optimizer, model
    return scores, state


def build_training_fold(
    runtime: FrozenFeatureRuntime,
    identity: str,
    o_groups: Sequence[Sequence[RankedImage]],
    w_groups: Sequence[Sequence[RankedImage]],
    canonical_identities: np.ndarray,
) -> TrainingFold:
    retained_o = [
        list(group)
        for group in o_groups
        if all(item.identity != identity for item in group)
    ]
    retained_w = [
        list(group)
        for group in w_groups
        if all(item.identity != identity for item in group)
    ]
    group_indices = {
        (source, index): torch.as_tensor(
            [runtime.name_to_index[item.image_name] for item in group],
            dtype=torch.int64,
            device=runtime.device,
        )
        for source, groups in (("O", retained_o), ("W", retained_w))
        for index, group in enumerate(groups)
    }
    return TrainingFold(
        identity=identity,
        retained_o=retained_o,
        retained_w=retained_w,
        group_indices=group_indices,
        validation_mask=canonical_identities == identity,
        reference_mask=canonical_identities != identity,
    )


def build_full_training_fold(
    runtime: FrozenFeatureRuntime,
    o_groups: Sequence[Sequence[RankedImage]],
    w_groups: Sequence[Sequence[RankedImage]],
    canonical_count: int,
) -> TrainingFold:
    """Build the no-purge full-data analogue of an identity training fold."""
    retained_o = [list(group) for group in o_groups]
    retained_w = [list(group) for group in w_groups]
    group_indices = {
        (source, index): torch.as_tensor(
            [runtime.name_to_index[item.image_name] for item in group],
            dtype=torch.int64,
            device=runtime.device,
        )
        for source, groups in (("O", retained_o), ("W", retained_w))
        for index, group in enumerate(groups)
    }
    return TrainingFold(
        identity="FULL_DEVELOPMENT_DATA",
        retained_o=retained_o,
        retained_w=retained_w,
        group_indices=group_indices,
        validation_mask=np.zeros(canonical_count, dtype=bool),
        reference_mask=np.ones(canonical_count, dtype=bool),
    )


def train_trajectory(
    *,
    runtime: FrozenFeatureRuntime,
    fold: TrainingFold,
    spec: ReadoutSpec,
    learning_rate: float,
    checkpoint: Sequence[int],
    seed: int,
    pair_weighting: str,
    o_coefficient: float,
    w_coefficient: float,
    weight_decay: float,
    target_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    base_scores: np.ndarray | None = None,
    tick: Callable[[], None] | None = None,
) -> dict[int, np.ndarray]:
    configure_determinism(seed)
    model = make_model(runtime, spec, seed, base_scores)
    optimizer = optimizer_for_parameters(
        model.named_parameters(), learning_rate, weight_decay
    )
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)
    capture = set(map(int, checkpoint))
    output: dict[int, np.ndarray] = {}
    if 0 in capture:
        with torch.no_grad():
            output[0] = (
                model.capture_scores(target_indices, reference_positions)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    model.train()
    for update in range(1, 257):
        if tick is not None:
            tick()
        lr = scheduled_learning_rate(learning_rate, update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        o_index = o_sampler.next()
        w_index = w_sampler.next()
        (
            o_coefficient
            * model.group_loss(
                fold.group_indices[("O", o_index)],
                fold.retained_o[o_index],
                "O",
                pair_weighting,
            )
        ).backward()
        (
            w_coefficient
            * model.group_loss(
                fold.group_indices[("W", w_index)],
                fold.retained_w[w_index],
                "W",
                pair_weighting,
            )
        ).backward()
        parameters = model.trainable_parameters()
        if not all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise RuntimeError(f"{spec.candidate_id} gradient became non-finite.")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise RuntimeError(f"{spec.candidate_id} parameter became non-finite.")
        if update in capture:
            with torch.no_grad():
                output[update] = (
                    model.capture_scores(target_indices, reference_positions)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
        model.train()
    if set(output) != capture or not all(
        np.all(np.isfinite(value)) for value in output.values()
    ):
        raise RuntimeError(f"{spec.candidate_id} checkpoint capture failed.")
    del optimizer, model
    return output


def trainable_parameter_count(spec: ReadoutSpec) -> int:
    device = torch.device("cpu")
    runtime = FrozenFeatureRuntime(
        names=["image"],
        pooled=torch.zeros((1, 1152), dtype=torch.bfloat16),
        tokens=torch.zeros((1, 1, 1152), dtype=torch.bfloat16),
        name_to_index={"image": 0},
        canonical_indices=torch.as_tensor([0]),
        device=device,
    )
    model = make_model(
        runtime,
        spec,
        seed=0,
        base_scores=np.zeros(1, dtype=np.float64) if spec.mode == "residual" else None,
    )
    return sum(parameter.numel() for parameter in model.trainable_parameters())


def midrank_percentile(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64), kind="mergesort")
    values = np.asarray(query, dtype=np.float64)
    lower = np.searchsorted(ordered, values, side="left")
    upper = np.searchsorted(ordered, values, side="right")
    return 100.0 * (lower + 0.5 * (upper - lower)) / len(ordered)


def heldout_level_result(
    *,
    image_ids: np.ndarray,
    identities: np.ndarray,
    theta: np.ndarray,
    fold: TrainingFold,
    canonical_scores: np.ndarray,
) -> dict[str, object]:
    model = midrank_percentile(
        canonical_scores[fold.validation_mask], canonical_scores[fold.reference_mask]
    )
    human = midrank_percentile(theta[fold.validation_mask], theta[fold.reference_mask])
    ids = image_ids[fold.validation_mask]
    metrics = heldout_fold_metrics(
        image_ids=ids,
        model_percentiles=model,
        human_percentiles=human,
    )
    return {
        "identity": fold.identity,
        "image_ids": ids,
        "identities": identities[fold.validation_mask],
        "human_percentile": human,
        "model_percentile": model,
        "metrics": metrics,
        "maximum_percentile_error": float(np.max(np.abs(model - human))),
    }


def xmr_metrics(
    human: np.ndarray, model: np.ndarray, identities: np.ndarray
) -> dict[str, float | int]:
    definitions = {
        "XMR_10_to_40": (human <= 10.0, model >= 40.0),
        "XMR_10_to_50": (human <= 10.0, model >= 50.0),
        "XMR_90_to_60": (human >= 90.0, model <= 60.0),
        "XMR_90_to_50": (human >= 90.0, model <= 50.0),
    }
    output: dict[str, float | int] = {}
    for name, (eligible, event) in definitions.items():
        denominator = int(np.count_nonzero(eligible))
        if denominator == 0:
            raise RuntimeError(f"{name} denominator is zero.")
        output[name] = float(np.mean(event[eligible]))
        rates = [
            float(np.mean(event[selected]))
            for identity in sorted(set(identities.tolist()))
            if np.any(selected := (eligible & (identities == identity)))
        ]
        output[f"{name}_identity_balanced"] = float(np.mean(rates))
        output[f"{name}_denominator"] = denominator
    return output


def aggregate_level_results(
    results: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    metric_names = (
        "IB_LTE10",
        "IB_FrameLTE10",
        "IdentityOffsetMAE",
        "strong_deviation_sign_accuracy",
        "mean_deviation_beta",
    )
    output: dict[str, float | int] = {
        name: float(
            np.mean([float(result["metrics"][name]) for result in results])  # type: ignore[index]
        )
        for name in metric_names
    }
    human = np.concatenate(
        [np.asarray(result["human_percentile"], dtype=np.float64) for result in results]
    )
    model = np.concatenate(
        [np.asarray(result["model_percentile"], dtype=np.float64) for result in results]
    )
    identities = np.concatenate(
        [np.asarray(result["identities"]).astype(str) for result in results]
    )
    error = np.abs(model - human)
    for threshold in (20, 30, 40, 60):
        output[f"GLE{threshold}"] = float(np.mean(error >= threshold))
    output["maximum_percentile_error"] = float(np.max(error))
    output.update(xmr_metrics(human, model, identities))
    return output
