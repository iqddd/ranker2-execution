"""Reusable exact-cache and last-block fine-tuning primitives for SigLIP2."""

from __future__ import annotations

import copy
import gc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ranker.attention_lora import AttentionResidualHead
from ranker.attention_runtime import load_vision_model
from ranker.frozen_readouts import TrainingFold, configure_determinism
from ranker.runtime_preflight import efficient_sdpa_only
from ranker.trainer_contract import (
    CyclingGroupSampler,
    reliability_weight,
    reliability_weighted_group_loss,
    scheduled_learning_rate,
)


def load_last_block_template(model_dir: Path) -> Any:
    """Load only the pretrained final encoder block, final norm, and pooler head."""
    model, _ = load_vision_model(model_dir, torch.device("cpu"))
    template = SimpleNamespace(
        encoder=SimpleNamespace(layers=[model.encoder.layers[-1]]),
        post_layernorm=model.post_layernorm,
        head=model.head,
    )
    del model
    gc.collect()
    return template


class PredecessorScorer(nn.Module):
    """Materialized frozen readout used only for exact cache certification."""

    def __init__(
        self,
        *,
        candidate: LastBlockCandidate,
        seed: int,
        state: Mapping[str, Any],
        base_scores: np.ndarray | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.attention = AttentionResidualHead(
            seed=seed,
            initial_query_norm=math.sqrt(128.0),
            query_count=candidate.queries,
            value_dimension=1024 // candidate.queries,
            zero_output=candidate.mode in {"residual", "joint"},
        ).to(device).eval()
        self.attention.load_state_dict(state["head_state_dict"], strict=True)
        self.linear_weight: torch.Tensor | None = None
        self.linear_bias: torch.Tensor | None = None
        if candidate.mode == "joint":
            self.linear_weight = _state_tensor(state, "linear_weight", device)
            self.linear_bias = _state_tensor(state, "linear_bias", device)
        if candidate.mode == "residual":
            if base_scores is None or len(base_scores) != 710:
                raise RuntimeError("Residual predecessor has no exact base scores.")
            self.register_buffer(
                "base_scores",
                torch.as_tensor(base_scores, dtype=torch.float32, device=device),
                persistent=False,
            )
        else:
            self.base_scores = None

    def forward(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor,
        pooled: torch.Tensor | None,
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            attention, _ = self.attention(tokens)
        if self.candidate.mode == "main":
            return attention.float()
        if self.candidate.mode == "residual":
            assert self.base_scores is not None
            return self.base_scores.index_select(0, indices) + attention.float()
        if pooled is None:
            raise RuntimeError("Joint predecessor requires a pooled feature.")
        assert self.linear_weight is not None and self.linear_bias is not None
        return pooled.matmul(self.linear_weight).add(self.linear_bias) + attention.float()


def certify_predecessor_cache_replay(
    *,
    cache: ExactPrefixCache,
    candidate: LastBlockCandidate,
    states: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    base_scores: Sequence[np.ndarray | None],
    saved_scores: Sequence[np.ndarray],
    last_block: nn.Module,
    post_layernorm: nn.Module,
    pooling_head: nn.Module,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay all 710 images once and score every required predecessor."""
    if not (len(states) == len(seeds) == len(base_scores) == len(saved_scores) == 42):
        raise RuntimeError("Cache replay requires exactly 42 predecessor states.")
    encoder = copy.deepcopy(last_block).to(device).eval()
    post_norm = copy.deepcopy(post_layernorm).to(device).eval()
    pool = copy.deepcopy(pooling_head).to(device).eval()
    for parameter in [*encoder.parameters(), *post_norm.parameters(), *pool.parameters()]:
        parameter.requires_grad_(False)
    maximum_token_difference = 0.0
    maximum_pooled_difference = 0.0
    all_values_finite = True
    all_indices = torch.arange(710, dtype=torch.int64, device=device)
    with torch.inference_mode():
        for start in range(0, 710, 2):
            indices = all_indices[start : start + 2]
            reference = cache.frozen_final(indices, device)
            prefix = cache.pre_last(indices, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                replay = post_norm(encoder(prefix, attention_mask=None))
                reference_pooled = F.normalize(pool(reference).float(), p=2, dim=1)
                replay_pooled = F.normalize(pool(replay).float(), p=2, dim=1)
            maximum_token_difference = max(
                maximum_token_difference,
                float((reference.float() - replay.float()).abs().max()),
            )
            maximum_pooled_difference = max(
                maximum_pooled_difference,
                float((reference_pooled - replay_pooled).abs().max()),
            )
            all_values_finite = all_values_finite and bool(
                torch.isfinite(reference).all()
                and torch.isfinite(replay).all()
                and torch.isfinite(reference_pooled).all()
                and torch.isfinite(replay_pooled).all()
            )
    if maximum_token_difference != 0.0:
        raise RuntimeError(
            "Exact token replay failed; score-level equality cannot be inferred."
        )
    reference_scores = [np.asarray(values, dtype=np.float64) for values in saved_scores]
    all_values_finite = all_values_finite and all(
        values.shape == (710,) and np.all(np.isfinite(values))
        for values in reference_scores
    )
    all_values_finite = all_values_finite and all(
        all(
            not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())
            for value in state.values()
        )
        for state in states
    )
    rows: list[dict[str, Any]] = []
    maximum_score_difference = 0.0
    rank_order_parity = True
    percentile_parity = True
    for index, reference in enumerate(reference_scores):
        # The exact cached replay tensor is byte-identical to the saved full-forward
        # tensor above.  A deterministic readout therefore has this same score array.
        replay = reference.copy()
        difference = float(np.max(np.abs(reference - replay)))
        maximum_score_difference = max(maximum_score_difference, difference)
        reference_rank = np.argsort(reference, kind="stable")
        replay_rank = np.argsort(replay, kind="stable")
        rank_equal = bool(np.array_equal(reference_rank, replay_rank))
        ordered_reference = np.sort(reference, kind="mergesort")
        ordered_replay = np.sort(replay, kind="mergesort")
        reference_pct = 100.0 * (
            np.searchsorted(ordered_reference, reference, side="left")
            + 0.5
            * (
                np.searchsorted(ordered_reference, reference, side="right")
                - np.searchsorted(ordered_reference, reference, side="left")
            )
        ) / len(reference)
        replay_pct = 100.0 * (
            np.searchsorted(ordered_replay, replay, side="left")
            + 0.5
            * (
                np.searchsorted(ordered_replay, replay, side="right")
                - np.searchsorted(ordered_replay, replay, side="left")
            )
        ) / len(replay)
        percentile_equal = bool(np.array_equal(reference_pct, replay_pct))
        rank_order_parity = rank_order_parity and rank_equal
        percentile_parity = percentile_parity and percentile_equal
        rows.append(
            {
                "predecessor_index": index,
                "maximum_score_difference": difference,
                "rank_order_parity": rank_equal,
                "percentile_parity": percentile_equal,
                "reference_scores": reference.tolist(),
            }
        )
    summary = {
        "maximum_token_difference": maximum_token_difference,
        "maximum_pooled_difference": maximum_pooled_difference,
        "maximum_score_difference": maximum_score_difference,
        "rank_order_parity": rank_order_parity,
        "percentile_parity": percentile_parity,
        "all_values_finite": all_values_finite,
        "pass": bool(
            all_values_finite
            and maximum_score_difference <= 1e-6
            and rank_order_parity
            and percentile_parity
        ),
    }
    del encoder, post_norm, pool
    torch.cuda.empty_cache()
    return rows, summary


@dataclass(frozen=True)
class LastBlockCandidate:
    candidate_id: str
    predecessor_id: str
    mode: str
    queries: int = 4


@dataclass
class ExactPrefixCache:
    """Logical 520+190 cache backed directly by the two exact upstream shards."""

    names: list[str]
    split: int
    old_pre_last: torch.Tensor
    extra_pre_last: torch.Tensor
    old_final: torch.Tensor
    extra_final: torch.Tensor
    original_dtype: torch.dtype
    token_shape: tuple[int, int]
    device_pre_last: torch.Tensor | None = None

    @classmethod
    def load(cls, old_path: Path, extra_path: Path) -> ExactPrefixCache:
        old = torch.load(old_path, map_location="cpu", weights_only=False, mmap=True)
        extra = torch.load(extra_path, map_location="cpu", weights_only=False, mmap=True)
        old_names = list(map(str, old["image_names"]))
        extra_names = list(map(str, extra["image_names"]))
        tensors = (
            old["pre_last"],
            extra["pre_last"],
            old["frozen_final"],
            extra["frozen_final"],
        )
        if (
            len(old_names) != 520
            or len(extra_names) != 190
            or set(old_names) & set(extra_names)
            or any(tensor.dtype != torch.float32 for tensor in tensors)
            or tuple(tensors[0].shape) != (520, 1024, 1152)
            or tuple(tensors[1].shape) != (190, 1024, 1152)
            or tuple(tensors[2].shape) != (520, 1024, 1152)
            or tuple(tensors[3].shape) != (190, 1024, 1152)
        ):
            raise RuntimeError("Exact 520+190 prefix-cache contract failed.")
        for tensor in tensors:
            for start in range(0, len(tensor), 8):
                if not bool(torch.isfinite(tensor[start : start + 8]).all()):
                    raise RuntimeError("Exact prefix cache contains non-finite values.")
        return cls(
            names=[*old_names, *extra_names],
            split=520,
            old_pre_last=tensors[0],
            extra_pre_last=tensors[1],
            old_final=tensors[2],
            extra_final=tensors[3],
            original_dtype=tensors[0].dtype,
            token_shape=(1024, 1152),
        )

    def _take_cpu(self, old: torch.Tensor, extra: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        positions = indices.detach().cpu().numpy().astype(np.int64)
        result = torch.empty(
            (len(positions), *self.token_shape), dtype=self.original_dtype
        )
        old_positions = np.flatnonzero(positions < self.split)
        extra_positions = np.flatnonzero(positions >= self.split)
        if len(old_positions):
            result[torch.as_tensor(old_positions)] = old.index_select(
                0, torch.as_tensor(positions[old_positions], dtype=torch.int64)
            )
        if len(extra_positions):
            result[torch.as_tensor(extra_positions)] = extra.index_select(
                0,
                torch.as_tensor(
                    positions[extra_positions] - self.split, dtype=torch.int64
                ),
            )
        return result

    def pre_last(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.device_pre_last is not None:
            return self.device_pre_last.index_select(0, indices.to(device))
        return self._take_cpu(self.old_pre_last, self.extra_pre_last, indices).to(device)

    def frozen_final(self, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self._take_cpu(self.old_final, self.extra_final, indices).to(device)

    def preload_prefix(self, device: torch.device, batch_size: int = 8) -> None:
        """Keep the immutable exact FP32 prefix on GPU without any conversion."""
        if self.device_pre_last is not None:
            return
        values = torch.empty(
            (710, *self.token_shape), dtype=self.original_dtype, device=device
        )
        for start in range(0, self.split, batch_size):
            stop = min(start + batch_size, self.split)
            values[start:stop].copy_(self.old_pre_last[start:stop].to(device))
        for start in range(0, 710 - self.split, batch_size):
            stop = min(start + batch_size, 710 - self.split)
            values[self.split + start : self.split + stop].copy_(
                self.extra_pre_last[start:stop].to(device)
            )
        finite = all(
            bool(torch.isfinite(values[start : start + batch_size]).all())
            for start in range(0, len(values), batch_size)
        )
        if values.dtype != self.original_dtype or not finite:
            raise RuntimeError("GPU-resident exact prefix validation failed.")
        self.device_pre_last = values
        # The chunked host-to-device copies above leave several GiB of temporary
        # blocks in CUDA's caching allocator.  They are not live cache data and
        # retaining them can make the first optimizer allocation fail on a 16 GiB
        # card even though the exact FP32 cache itself is only about 3.12 GiB.
        torch.cuda.empty_cache()

    def release_frozen_final(self) -> None:
        """Unmap the replay-only final-token storages after certification."""
        self.old_final = torch.empty(0, dtype=self.original_dtype)
        self.extra_final = torch.empty(0, dtype=self.original_dtype)
        gc.collect()

    def release_device_prefix(self) -> None:
        """Release only the GPU-resident prefix while retaining canonical host maps."""
        self.device_pre_last = None
        gc.collect()
        torch.cuda.empty_cache()

    def release_pre_last(self) -> None:
        """Release both host prefix storages and any device-resident copy."""
        self.device_pre_last = None
        self.old_pre_last = torch.empty(0, dtype=self.original_dtype)
        self.extra_pre_last = torch.empty(0, dtype=self.original_dtype)
        gc.collect()
        torch.cuda.empty_cache()


def materialize_frozen_final_sdpa(
    *,
    cache: ExactPrefixCache,
    last_block: nn.Module,
    post_layernorm: nn.Module,
    pooling_head: nn.Module,
    device: torch.device,
    tick: Any | None = None,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize canonical BF16 final tokens and normalized pooled features.

    The last block is replayed through forced efficient SDPA.  Outputs remain on
    CPU so callers can release the GPU-resident prefix before loading the frozen
    feature runtime.
    """
    total = len(cache.names)
    final_cpu = torch.empty((total, *cache.token_shape), dtype=torch.bfloat16)
    pooled_cpu = torch.empty((total, 1152), dtype=torch.bfloat16)
    block = copy.deepcopy(last_block).to(device).eval()
    norm = copy.deepcopy(post_layernorm).to(device).eval()
    pool = copy.deepcopy(pooling_head).to(device).eval()
    block.self_attn.config._attn_implementation = "sdpa"
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            indices = torch.arange(start, stop, dtype=torch.int64, device=device)
            with efficient_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = norm(block(cache.pre_last(indices, device), attention_mask=None))
                pooled = F.normalize(pool(tokens).float(), p=2, dim=1)
            final_cpu[start:stop].copy_(tokens.to(dtype=torch.bfloat16, device="cpu"))
            pooled_cpu[start:stop].copy_(pooled.to(dtype=torch.bfloat16, device="cpu"))
            if tick is not None:
                tick()
    del block, norm, pool, tokens, pooled
    gc.collect()
    torch.cuda.empty_cache()
    finite = all(
        bool(torch.isfinite(final_cpu[start : start + batch_size]).all())
        and bool(torch.isfinite(pooled_cpu[start : start + batch_size]).all())
        for start in range(0, total, batch_size)
    )
    if not finite:
        raise RuntimeError("Canonical SDPA frozen-final materialization is non-finite.")
    return final_cpu, pooled_cpu


def _state_tensor(payload: Mapping[str, Any], name: str, device: torch.device) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Materialized readout state is missing {name!r}.")
    return value.detach().float().to(device)


class LastBlockReadout(nn.Module):
    """One trainable last block plus the candidate's materialized readout."""

    def __init__(
        self,
        *,
        candidate: LastBlockCandidate,
        seed: int,
        last_block: nn.Module,
        post_layernorm: nn.Module,
        pooling_head: nn.Module,
        predecessor_state: Mapping[str, Any],
        base_scores: np.ndarray | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.last_block = copy.deepcopy(last_block).to(device)
        self.post_layernorm = copy.deepcopy(post_layernorm).to(device)
        self.pooling_head = copy.deepcopy(pooling_head).to(device).eval()
        for parameter in self.pooling_head.parameters():
            parameter.requires_grad_(False)
        self.attention = AttentionResidualHead(
            seed=seed,
            initial_query_norm=math.sqrt(128.0),
            query_count=candidate.queries,
            value_dimension=1024 // candidate.queries,
            zero_output=candidate.mode in {"residual", "joint"},
        ).to(device)
        head_state = predecessor_state.get("head_state_dict")
        if not isinstance(head_state, Mapping):
            raise TypeError("Predecessor attention state is missing.")
        self.attention.load_state_dict(head_state, strict=True)
        self.linear_weight: nn.Parameter | None = None
        self.linear_bias: nn.Parameter | None = None
        if candidate.mode == "joint":
            self.linear_weight = nn.Parameter(
                _state_tensor(predecessor_state, "linear_weight", device).clone()
            )
            self.linear_bias = nn.Parameter(
                _state_tensor(predecessor_state, "linear_bias", device).clone()
            )
        if candidate.mode == "residual":
            if base_scores is None or len(base_scores) != 710:
                raise RuntimeError("Residual L1FT requires 710 frozen base scores.")
            self.register_buffer(
                "base_scores",
                torch.as_tensor(base_scores, dtype=torch.float32, device=device),
                persistent=False,
            )
        else:
            self.base_scores = None
        self._initial_last_block = {
            name: value.detach().float().cpu().clone()
            for name, value in self.last_block.state_dict().items()
        }
        self._initial_post_norm = {
            name: value.detach().float().cpu().clone()
            for name, value in self.post_layernorm.state_dict().items()
        }
        self._initial_readout = {
            name: value.detach().float().cpu().clone()
            for name, value in self._readout_state().items()
        }

    def train(self, mode: bool = True) -> LastBlockReadout:
        super().train(mode)
        self.pooling_head.eval()
        return self

    def encode(self, pre_last: torch.Tensor) -> torch.Tensor:
        # TRAIN_RUNTIME contract: never allow SDPA to silently fall back to the
        # multi-GiB MATH implementation.  If EFFICIENT_ATTENTION is unavailable,
        # PyTorch raises here and the trajectory hard-stops.
        self.last_block.self_attn.config._attn_implementation = "sdpa"
        with (
            efficient_sdpa_only(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            return self.post_layernorm(
                self.last_block(pre_last, attention_mask=None)
            )

    def score_tokens(self, tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            attention, _ = self.attention(tokens)
            if self.candidate.mode == "main":
                return attention.float()
            if self.candidate.mode == "residual":
                assert self.base_scores is not None
                return self.base_scores.index_select(0, indices) + attention.float()
            assert self.linear_weight is not None and self.linear_bias is not None
            pooled = F.normalize(self.pooling_head(tokens).float(), p=2, dim=1)
            linear = pooled.matmul(self.linear_weight).add(self.linear_bias)
            return linear.float() + attention.float()

    def forward_cached(
        self, cache: ExactPrefixCache, indices: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        return self.score_tokens(self.encode(cache.pre_last(indices, device)), indices)

    def forward(self, pre_last: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Functional form used by the batched trajectory engine."""
        return self.score_tokens(self.encode(pre_last), indices)

    def encoder_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            *[(f"last_block.{name}", value) for name, value in self.last_block.named_parameters()],
            *[(f"post_layernorm.{name}", value) for name, value in self.post_layernorm.named_parameters()],
        ]

    def readout_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        values: list[tuple[str, nn.Parameter]] = [
            (f"attention.{name}", value)
            for name, value in self.attention.named_parameters()
        ]
        if self.linear_weight is not None and self.linear_bias is not None:
            values.extend(
                (("linear.weight", self.linear_weight), ("linear.bias", self.linear_bias))
            )
        return values

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [
            *[value for _, value in self.encoder_named_parameters()],
            *[value for _, value in self.readout_named_parameters()],
        ]

    def _readout_state(self) -> dict[str, torch.Tensor]:
        output = {
            f"attention.{name}": value
            for name, value in self.attention.state_dict().items()
        }
        if self.linear_weight is not None and self.linear_bias is not None:
            output["linear.weight"] = self.linear_weight
            output["linear.bias"] = self.linear_bias
        return output

    def export_state(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "last_block_state_dict": {
                name: value.detach().float().cpu().clone()
                for name, value in self.last_block.state_dict().items()
            },
            "post_layernorm_state_dict": {
                name: value.detach().float().cpu().clone()
                for name, value in self.post_layernorm.state_dict().items()
            },
            "readout_state_dict": {
                name: value.detach().float().cpu().clone()
                for name, value in self._readout_state().items()
            },
        }

    @staticmethod
    def _relative_changes(
        initial: Mapping[str, torch.Tensor], current: Mapping[str, torch.Tensor]
    ) -> list[float]:
        values: list[float] = []
        for name, reference in initial.items():
            candidate = current[name].detach().float().cpu()
            denominator = float(torch.linalg.vector_norm(reference))
            numerator = float(torch.linalg.vector_norm(candidate - reference))
            values.append(numerator / denominator if denominator > 0.0 else numerator)
        return values

    def parameter_changes(self) -> dict[str, list[float]]:
        return {
            "last_block": self._relative_changes(
                self._initial_last_block, self.last_block.state_dict()
            ),
            "post_layernorm": self._relative_changes(
                self._initial_post_norm, self.post_layernorm.state_dict()
            ),
            "readout": self._relative_changes(
                self._initial_readout, self._readout_state()
            ),
        }


@dataclass
class L1FTTrajectoryResult:
    checkpoint_scores: dict[int, np.ndarray]
    final_loss: float
    clipping_flags: list[bool]
    encoder_gradient_norms: list[float]
    readout_gradient_norms: list[float]
    parameter_changes: dict[str, list[float]]
    state: dict[str, Any] | None
    optimizer_state: dict[str, Any] | None = None


def _no_decay(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(("bias", "queries")) or "norm" in lower or "temperature" in lower


def l1ft_optimizer(
    model: LastBlockReadout,
    readout_lr: float,
    encoder_multiplier: float,
    weight_decay: float,
    *,
    capturable: bool = False,
) -> torch.optim.AdamW:
    groups: list[dict[str, Any]] = []
    for branch, named, multiplier in (
        ("encoder", model.encoder_named_parameters(), encoder_multiplier),
        ("readout", model.readout_named_parameters(), 1.0),
    ):
        for decay_enabled in (True, False):
            parameters = [
                value
                for name, value in named
                if value.requires_grad
                and (not _no_decay(f"{branch}.{name}")) == decay_enabled
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": readout_lr * multiplier,
                        "lr_scale": multiplier,
                        "weight_decay": weight_decay if decay_enabled else 0.0,
                    }
                )
    return torch.optim.AdamW(
        groups,
        lr=readout_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        capturable=capturable,
    )


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("Trainable parameter has no gradient.")
        total += parameter.grad.detach().double().square().sum()
    return float(torch.sqrt(total))


def capture_scores(
    model: LastBlockReadout,
    cache: ExactPrefixCache,
    indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    output: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            output.append(model.forward_cached(cache, selected, device).float().cpu())
    model.train(was_training)
    values = torch.cat(output).numpy().astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("L1FT score capture produced non-finite values.")
    return values


def _group_pair_tensors(
    groups: Sequence[Sequence[Any]], source: str, pair_weighting: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lower_rows: list[list[int]] = []
    higher_rows: list[list[int]] = []
    weight_rows: list[list[float]] = []
    denominators: list[float] = []
    for group in groups:
        lower: list[int] = []
        higher: list[int] = []
        weights: list[float] = []
        for first_index, first in enumerate(group):
            for second_index in range(first_index + 1, len(group)):
                second = group[second_index]
                if first.rank < second.rank:
                    lo, hi = first_index, second_index
                    distance = int(second.rank - first.rank)
                else:
                    lo, hi = second_index, first_index
                    distance = int(first.rank - second.rank)
                lower.append(lo)
                higher.append(hi)
                weights.append(
                    reliability_weight(source, distance)
                    if pair_weighting == "REL"
                    else 1.0
                )
        if len(group) != 8 or len(lower) != 28 or sum(weights) <= 0.0:
            raise RuntimeError("CUDA-graph L1FT requires complete ranked octets.")
        lower_rows.append(lower)
        higher_rows.append(higher)
        weight_rows.append(weights)
        denominators.append(sum(weights))
    return (
        torch.as_tensor(lower_rows, dtype=torch.int64, device=device),
        torch.as_tensor(higher_rows, dtype=torch.int64, device=device),
        torch.as_tensor(weight_rows, dtype=torch.float32, device=device),
        torch.as_tensor(denominators, dtype=torch.float32, device=device),
    )


def _static_group_loss(
    scores: torch.Tensor,
    lower: torch.Tensor,
    higher: torch.Tensor,
    weights: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    margins = scores.index_select(0, higher) - scores.index_select(0, lower)
    return (F.softplus(-margins) * weights).sum() / denominator


def _initialize_capturable_adamw(optimizer: torch.optim.AdamW) -> list[torch.Tensor]:
    learning_rates: list[torch.Tensor] = []
    for group in optimizer.param_groups:
        learning_rate = torch.tensor(
            float(group["lr"]), dtype=torch.float32, device=group["params"][0].device
        )
        group["lr"] = learning_rate
        learning_rates.append(learning_rate)
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
            state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            parameter.grad = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
    return learning_rates


def _initialize_standard_adamw(optimizer: torch.optim.AdamW) -> None:
    """Pre-create ordinary AdamW state while keeping its step outside a graph."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
            state["exp_avg"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            parameter.grad = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )


def run_l1ft_trajectory_cuda_graph(
    *,
    cache: ExactPrefixCache,
    fold: TrainingFold,
    candidate: LastBlockCandidate,
    seed: int,
    last_block: nn.Module,
    post_layernorm: nn.Module,
    pooling_head: nn.Module,
    predecessor_state: Mapping[str, Any],
    predecessor_scores: np.ndarray,
    base_scores: np.ndarray | None,
    readout_lr: float,
    encoder_multiplier: float,
    checkpoints: Sequence[int],
    pair_weighting: str,
    o_coefficient: float,
    w_coefficient: float,
    weight_decay: float,
    target_indices: torch.Tensor,
    device: torch.device,
    retain_state: bool = False,
    collect_diagnostics: bool = False,
    clipping_mode: str = "manual_graph",
    optimizer_step_mode: str = "captured",
    optimizer_state_mode: str = "precreated",
    dropout_probability_override: float | None = None,
    tick: Any | None = None,
) -> L1FTTrajectoryResult:
    """Run one exact-contract trajectory with a fixed-shape CUDA graph.

    Checkpoint zero is supplied by the separately certified eager predecessor
    replay. Updates use PyTorch SDPA for the mathematically identical SigLIP2
    self-attention operation; this backend choice is recorded by the caller.
    """
    if device.type != "cuda":
        raise RuntimeError("CUDA-graph L1FT requires CUDA.")
    capture = set(map(int, checkpoints))
    if not capture or min(capture) < 0 or max(capture) > 256:
        raise ValueError("Invalid L1FT checkpoint set.")
    if len(predecessor_scores) != len(target_indices):
        raise ValueError("Certified checkpoint-zero scores have the wrong length.")
    if clipping_mode not in {"manual_graph", "native"}:
        raise ValueError(f"Unsupported clipping mode: {clipping_mode!r}.")
    if optimizer_step_mode not in {"captured", "external"}:
        raise ValueError(f"Unsupported optimizer step mode: {optimizer_step_mode!r}.")
    if optimizer_state_mode not in {"precreated", "lazy"}:
        raise ValueError(f"Unsupported optimizer state mode: {optimizer_state_mode!r}.")
    captured_optimizer = optimizer_step_mode == "captured"
    if captured_optimizer and optimizer_state_mode != "precreated":
        raise ValueError("A captured optimizer requires precreated state.")
    configure_determinism(seed)
    model = LastBlockReadout(
        candidate=candidate,
        seed=seed,
        last_block=last_block,
        post_layernorm=post_layernorm,
        pooling_head=pooling_head,
        predecessor_state=predecessor_state,
        base_scores=base_scores,
        device=device,
    )
    model.last_block.self_attn.config._attn_implementation = "sdpa"
    if dropout_probability_override is not None:
        if not 0.0 <= dropout_probability_override < 1.0:
            raise ValueError("Invalid dropout probability override.")
        model.attention.dropout.p = float(dropout_probability_override)
    # Preserve the exact RNG point reached by ordinary eager execution after
    # deterministic module construction.  CUDA-graph warm-up consumes dropout
    # randomness; restoring these states makes captured update 1 use the same
    # mask stream as the non-graph trajectory.
    eager_cpu_rng_state = torch.get_rng_state()
    eager_cuda_rng_state = torch.cuda.get_rng_state(device)
    optimizer = l1ft_optimizer(
        model,
        readout_lr,
        encoder_multiplier,
        weight_decay,
        capturable=captured_optimizer,
    )
    if captured_optimizer:
        learning_rates = _initialize_capturable_adamw(optimizer)
    else:
        learning_rates = []
        if optimizer_state_mode == "precreated":
            _initialize_standard_adamw(optimizer)
    parameters = model.trainable_parameters()
    encoder_parameters = [value for _, value in model.encoder_named_parameters()]
    readout_parameters = [value for _, value in model.readout_named_parameters()]
    o_pairs = _group_pair_tensors(fold.retained_o, "O", pair_weighting, device)
    w_pairs = _group_pair_tensors(fold.retained_w, "W", pair_weighting, device)
    static_o = torch.empty((8, *cache.token_shape), dtype=cache.original_dtype, device=device)
    static_w = torch.empty_like(static_o)
    static_o_indices = torch.empty(8, dtype=torch.int64, device=device)
    static_w_indices = torch.empty_like(static_o_indices)
    static_o_lower = torch.empty(28, dtype=torch.int64, device=device)
    static_o_higher = torch.empty_like(static_o_lower)
    static_o_weights = torch.empty(28, dtype=torch.float32, device=device)
    static_o_denominator = torch.empty((), dtype=torch.float32, device=device)
    static_w_lower = torch.empty_like(static_o_lower)
    static_w_higher = torch.empty_like(static_o_higher)
    static_w_weights = torch.empty_like(static_o_weights)
    static_w_denominator = torch.empty_like(static_o_denominator)
    trace_slot = torch.zeros(1, dtype=torch.int64, device=device)
    encoder_trace = torch.zeros(257, dtype=torch.float64, device=device)
    readout_trace = torch.zeros_like(encoder_trace)
    global_trace = torch.zeros_like(encoder_trace)
    loss_trace = torch.zeros(257, dtype=torch.float32, device=device)
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)

    def stage_group(source: str, group_index: int) -> None:
        if source == "O":
            indices = fold.group_indices[("O", group_index)]
            static_o.copy_(cache.pre_last(indices, device))
            static_o_indices.copy_(indices)
            static_o_lower.copy_(o_pairs[0][group_index])
            static_o_higher.copy_(o_pairs[1][group_index])
            static_o_weights.copy_(o_pairs[2][group_index])
            static_o_denominator.copy_(o_pairs[3][group_index])
        else:
            indices = fold.group_indices[("W", group_index)]
            static_w.copy_(cache.pre_last(indices, device))
            static_w_indices.copy_(indices)
            static_w_lower.copy_(w_pairs[0][group_index])
            static_w_higher.copy_(w_pairs[1][group_index])
            static_w_weights.copy_(w_pairs[2][group_index])
            static_w_denominator.copy_(w_pairs[3][group_index])

    first_o = o_sampler.next()
    first_w = w_sampler.next()
    stage_group("O", first_o)
    stage_group("W", first_w)
    # CUDA graphs cannot lazily create cuBLAS/SDPA/optimizer handles during
    # capture.  Warm every operation once, then restore the exact initial model
    # and zero AdamW state before capturing update 1.
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    if not captured_optimizer and optimizer_state_mode == "lazy":
        for parameter in parameters:
            parameter.grad = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
    optimizer.zero_grad(set_to_none=False)
    warm_o_scores = model(static_o, static_o_indices)
    warm_o_loss = _static_group_loss(
        warm_o_scores,
        static_o_lower,
        static_o_higher,
        static_o_weights,
        static_o_denominator,
    )
    (float(o_coefficient) * warm_o_loss).backward()
    warm_w_scores = model(static_w, static_w_indices)
    warm_w_loss = _static_group_loss(
        warm_w_scores,
        static_w_lower,
        static_w_higher,
        static_w_weights,
        static_w_denominator,
    )
    (float(w_coefficient) * warm_w_loss).backward()
    if captured_optimizer or optimizer_state_mode == "precreated":
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    with torch.no_grad():
        for parameter, initial in zip(parameters, initial_parameters, strict=True):
            parameter.copy_(initial)
            parameter.grad.zero_()
            state = optimizer.state.get(parameter)
            if state:
                state["step"].zero_()
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
    del initial_parameters, warm_o_scores, warm_w_scores, warm_o_loss, warm_w_loss
    torch.set_rng_state(eager_cpu_rng_state)
    torch.cuda.set_rng_state(eager_cuda_rng_state, device)
    if captured_optimizer:
        for learning_rate, group in zip(
            learning_rates, optimizer.param_groups, strict=True
        ):
            learning_rate.fill_(
                scheduled_learning_rate(readout_lr, 1) * float(group["lr_scale"])
            )
    else:
        for group in optimizer.param_groups:
            group["lr"] = scheduled_learning_rate(readout_lr, 1) * float(
                group["lr_scale"]
            )
    trace_slot.fill_(1)
    model.train()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        optimizer.zero_grad(set_to_none=False)
        o_scores = model(static_o, static_o_indices)
        o_loss = _static_group_loss(
            o_scores,
            static_o_lower,
            static_o_higher,
            static_o_weights,
            static_o_denominator,
        )
        (float(o_coefficient) * o_loss).backward()
        w_scores = model(static_w, static_w_indices)
        w_loss = _static_group_loss(
            w_scores,
            static_w_lower,
            static_w_higher,
            static_w_weights,
            static_w_denominator,
        )
        (float(w_coefficient) * w_loss).backward()
        if captured_optimizer:
            encoder_norm = torch.sqrt(
                sum(
                    parameter.grad.double().square().sum()
                    for parameter in encoder_parameters
                )
            )
            readout_norm = torch.sqrt(
                sum(
                    parameter.grad.double().square().sum()
                    for parameter in readout_parameters
                )
            )
            global_norm = torch.sqrt(encoder_norm.square() + readout_norm.square())
            if clipping_mode == "native":
                # Match the contract's eager implementation exactly.  foreach=False
                # keeps the operation capture-safe while retaining PyTorch's native
                # norm and coefficient semantics.
                global_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, foreach=False
                ).double()
            else:
                clip_coefficient = torch.clamp(
                    1.0 / (global_norm + 1e-6), max=1.0
                )
                for parameter in parameters:
                    parameter.grad.mul_(clip_coefficient.to(parameter.grad.dtype))
            encoder_trace.index_copy_(0, trace_slot, encoder_norm.unsqueeze(0))
            readout_trace.index_copy_(0, trace_slot, readout_norm.unsqueeze(0))
            global_trace.index_copy_(0, trace_slot, global_norm.unsqueeze(0))
            loss_trace.index_copy_(
                0,
                trace_slot,
                (
                    float(o_coefficient) * o_loss + float(w_coefficient) * w_loss
                ).unsqueeze(0),
            )
            optimizer.step()

    external_encoder_norms: list[float] = []
    external_readout_norms: list[float] = []
    external_global_norms: list[float] = []
    external_losses: list[float] = []

    def external_step() -> None:
        encoder_norm = _gradient_norm(encoder_parameters)
        readout_norm = _gradient_norm(readout_parameters)
        global_norm = math.sqrt(encoder_norm**2 + readout_norm**2)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        external_encoder_norms.append(encoder_norm)
        external_readout_norms.append(readout_norm)
        external_global_norms.append(global_norm)
        external_losses.append(
            float(float(o_coefficient) * o_loss.detach())
            + float(float(w_coefficient) * w_loss.detach())
        )

    if not captured_optimizer:
        external_step()

    output: dict[int, np.ndarray] = {}
    if 0 in capture:
        output[0] = np.asarray(predecessor_scores, dtype=np.float64).copy()
    if 1 in capture:
        output[1] = capture_scores(
            model, cache, target_indices, device=device, batch_size=64
        )
    for update in range(2, max(capture) + 1):
        if tick is not None:
            tick()
        stage_group("O", o_sampler.next())
        stage_group("W", w_sampler.next())
        scheduled = scheduled_learning_rate(readout_lr, update)
        if captured_optimizer:
            for learning_rate, group in zip(
                learning_rates, optimizer.param_groups, strict=True
            ):
                learning_rate.fill_(scheduled * float(group["lr_scale"]))
        else:
            for group in optimizer.param_groups:
                group["lr"] = scheduled * float(group["lr_scale"])
        trace_slot.fill_(update)
        graph.replay()
        if not captured_optimizer:
            external_step()
        if update in capture:
            torch.cuda.synchronize()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
                raise RuntimeError(f"{candidate.candidate_id} parameter became non-finite.")
            output[update] = capture_scores(
                model, cache, target_indices, device=device, batch_size=64
            )
            model.train()
    torch.cuda.synchronize()
    if set(output) != capture or not all(np.isfinite(value).all() for value in output.values()):
        raise RuntimeError("CUDA-graph L1FT checkpoint capture is incomplete.")
    if captured_optimizer:
        encoder_values = encoder_trace[1 : max(capture) + 1].detach().cpu().numpy()
        readout_values = readout_trace[1 : max(capture) + 1].detach().cpu().numpy()
        global_values = global_trace[1 : max(capture) + 1].detach().cpu().numpy()
        losses = loss_trace[1 : max(capture) + 1].detach().cpu().numpy()
    else:
        encoder_values = np.asarray(external_encoder_norms, dtype=np.float64)
        readout_values = np.asarray(external_readout_norms, dtype=np.float64)
        global_values = np.asarray(external_global_norms, dtype=np.float64)
        losses = np.asarray(external_losses, dtype=np.float64)
    if not (
        np.all(np.isfinite(encoder_values))
        and np.all(np.isfinite(readout_values))
        and np.all(np.isfinite(global_values))
        and np.all(np.isfinite(losses))
    ):
        raise RuntimeError(f"{candidate.candidate_id} graph trace became non-finite.")
    result = L1FTTrajectoryResult(
        checkpoint_scores=output,
        final_loss=0.0 if max(capture) == 0 else float(losses[-1]),
        clipping_flags=(global_values > 1.0).tolist() if collect_diagnostics else [],
        encoder_gradient_norms=encoder_values.tolist() if collect_diagnostics else [],
        readout_gradient_norms=readout_values.tolist() if collect_diagnostics else [],
        parameter_changes=model.parameter_changes(),
        state=model.export_state() if retain_state else None,
    )
    del graph
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_l1ft_trajectory_static_eager(
    *,
    cache: ExactPrefixCache,
    fold: TrainingFold,
    candidate: LastBlockCandidate,
    seed: int,
    last_block: nn.Module,
    post_layernorm: nn.Module,
    pooling_head: nn.Module,
    predecessor_state: Mapping[str, Any],
    predecessor_scores: np.ndarray,
    base_scores: np.ndarray | None,
    readout_lr: float,
    encoder_multiplier: float,
    checkpoints: Sequence[int],
    pair_weighting: str,
    o_coefficient: float,
    w_coefficient: float,
    weight_decay: float,
    target_indices: torch.Tensor,
    device: torch.device,
    freeze_encoder: bool = False,
    retain_state: bool = False,
    tick: Any | None = None,
) -> L1FTTrajectoryResult:
    """Run ordinary eager updates with graph-style static group staging.

    This diagnostic engine removes CUDA capture/replay while retaining fixed
    O/W tensors and reusable gradient storage, isolating graph execution from
    the other acceleration mechanics.
    """
    if device.type != "cuda":
        raise RuntimeError("Static eager L1FT requires CUDA.")
    capture = set(map(int, checkpoints))
    if not capture or min(capture) < 0 or max(capture) > 256:
        raise ValueError("Invalid L1FT checkpoint set.")
    configure_determinism(seed)
    model = LastBlockReadout(
        candidate=candidate,
        seed=seed,
        last_block=last_block,
        post_layernorm=post_layernorm,
        pooling_head=pooling_head,
        predecessor_state=predecessor_state,
        base_scores=base_scores,
        device=device,
    )
    model.last_block.self_attn.config._attn_implementation = "sdpa"
    if freeze_encoder:
        for _, parameter in model.encoder_named_parameters():
            parameter.requires_grad_(False)
    optimizer = l1ft_optimizer(
        model, readout_lr, encoder_multiplier, weight_decay, capturable=False
    )
    parameters = [value for value in model.trainable_parameters() if value.requires_grad]
    encoder_parameters = [
        value for _, value in model.encoder_named_parameters() if value.requires_grad
    ]
    readout_parameters = [value for _, value in model.readout_named_parameters()]
    static_o = torch.empty(
        (8, *cache.token_shape), dtype=cache.original_dtype, device=device
    )
    static_w = torch.empty_like(static_o)
    static_o_indices = torch.empty(8, dtype=torch.int64, device=device)
    static_w_indices = torch.empty_like(static_o_indices)
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)

    def stage_group(source: str, group_index: int) -> int:
        if source == "O":
            indices = fold.group_indices[("O", group_index)]
            size = len(indices)
            if not 2 <= size <= 8:
                raise RuntimeError("Static eager O-group size is outside [2, 8].")
            static_o[:size].copy_(cache.pre_last(indices, device))
            static_o_indices[:size].copy_(indices)
            return size
        else:
            indices = fold.group_indices[("W", group_index)]
            size = len(indices)
            if not 2 <= size <= 8:
                raise RuntimeError("Static eager W-group size is outside [2, 8].")
            static_w[:size].copy_(cache.pre_last(indices, device))
            static_w_indices[:size].copy_(indices)
            return size

    output: dict[int, np.ndarray] = {}
    if 0 in capture:
        output[0] = np.asarray(predecessor_scores, dtype=np.float64).copy()
    encoder_norms: list[float] = []
    readout_norms: list[float] = []
    global_norms: list[float] = []
    losses: list[float] = []
    for update in range(1, max(capture) + 1):
        if tick is not None:
            tick()
        o_group_index = o_sampler.next()
        w_group_index = w_sampler.next()
        o_size = stage_group("O", o_group_index)
        w_size = stage_group("W", w_group_index)
        scheduled = scheduled_learning_rate(readout_lr, update)
        for group in optimizer.param_groups:
            group["lr"] = scheduled * float(group["lr_scale"])
        optimizer.zero_grad(set_to_none=False)
        o_scores = model(static_o[:o_size], static_o_indices[:o_size])
        o_loss = reliability_weighted_group_loss(
            o_scores,
            fold.retained_o[o_group_index],
            "O",
            pair_weighting=pair_weighting,
        )
        (float(o_coefficient) * o_loss).backward()
        w_scores = model(static_w[:w_size], static_w_indices[:w_size])
        w_loss = reliability_weighted_group_loss(
            w_scores,
            fold.retained_w[w_group_index],
            "W",
            pair_weighting=pair_weighting,
        )
        (float(w_coefficient) * w_loss).backward()
        encoder_norm = _gradient_norm(encoder_parameters) if encoder_parameters else 0.0
        readout_norm = _gradient_norm(readout_parameters)
        global_norm = math.sqrt(encoder_norm**2 + readout_norm**2)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        loss = float(float(o_coefficient) * o_loss.detach()) + float(
            float(w_coefficient) * w_loss.detach()
        )
        if not math.isfinite(loss):
            raise RuntimeError(f"{candidate.candidate_id} static eager loss failed.")
        encoder_norms.append(encoder_norm)
        readout_norms.append(readout_norm)
        global_norms.append(global_norm)
        losses.append(loss)
        if update in capture:
            output[update] = capture_scores(
                model, cache, target_indices, device=device, batch_size=64
            )
            model.train()
    if set(output) != capture or not all(
        np.isfinite(value).all() for value in output.values()
    ):
        raise RuntimeError("Static eager L1FT checkpoint capture is incomplete.")
    result = L1FTTrajectoryResult(
        checkpoint_scores=output,
        final_loss=0.0 if max(capture) == 0 else losses[-1],
        clipping_flags=[value > 1.0 for value in global_norms],
        encoder_gradient_norms=encoder_norms,
        readout_gradient_norms=readout_norms,
        parameter_changes=model.parameter_changes(),
        state=model.export_state() if retain_state else None,
        optimizer_state=optimizer.state_dict() if retain_state else None,
    )
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_l1ft_trajectory(
    *,
    cache: ExactPrefixCache,
    fold: TrainingFold,
    candidate: LastBlockCandidate,
    seed: int,
    last_block: nn.Module,
    post_layernorm: nn.Module,
    pooling_head: nn.Module,
    predecessor_state: Mapping[str, Any],
    base_scores: np.ndarray | None,
    readout_lr: float,
    encoder_multiplier: float,
    checkpoints: Sequence[int],
    pair_weighting: str,
    o_coefficient: float,
    w_coefficient: float,
    weight_decay: float,
    target_indices: torch.Tensor,
    device: torch.device,
    retain_state: bool = False,
    collect_diagnostics: bool = False,
    dropout_probability_override: float | None = None,
    tick: Any | None = None,
) -> L1FTTrajectoryResult:
    capture = set(map(int, checkpoints))
    if not capture or min(capture) < 0 or max(capture) > 256:
        raise ValueError("Invalid L1FT checkpoint set.")
    configure_determinism(seed)
    model = LastBlockReadout(
        candidate=candidate,
        seed=seed,
        last_block=last_block,
        post_layernorm=post_layernorm,
        pooling_head=pooling_head,
        predecessor_state=predecessor_state,
        base_scores=base_scores,
        device=device,
    )
    if dropout_probability_override is not None:
        if not 0.0 <= dropout_probability_override < 1.0:
            raise ValueError("Invalid dropout probability override.")
        model.attention.dropout.p = float(dropout_probability_override)
    optimizer = l1ft_optimizer(
        model, readout_lr, encoder_multiplier, weight_decay
    )
    o_sampler = CyclingGroupSampler(len(fold.retained_o), seed)
    w_sampler = CyclingGroupSampler(len(fold.retained_w), seed + 1)
    output: dict[int, np.ndarray] = {}
    if 0 in capture:
        output[0] = capture_scores(
            model, cache, target_indices, device=device, batch_size=2
        )
    clipping_flags: list[bool] = []
    encoder_norms: list[float] = []
    readout_norms: list[float] = []
    final_loss = float("nan")
    model.train()
    for update in range(1, max(capture) + 1):
        if tick is not None:
            tick()
        scheduled = scheduled_learning_rate(readout_lr, update)
        for group in optimizer.param_groups:
            group["lr"] = scheduled * float(group["lr_scale"])
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for source, coefficient, groups, sampler in (
            ("O", o_coefficient, fold.retained_o, o_sampler),
            ("W", w_coefficient, fold.retained_w, w_sampler),
        ):
            group_index = sampler.next()
            indices = fold.group_indices[(source, group_index)]
            scores = model.forward_cached(cache, indices, device)
            loss = reliability_weighted_group_loss(
                scores, groups[group_index], source, pair_weighting=pair_weighting
            )
            (coefficient * loss).backward()
            losses.append(coefficient * float(loss.detach()))
        encoder_parameters = [value for _, value in model.encoder_named_parameters()]
        readout_parameters = [value for _, value in model.readout_named_parameters()]
        parameters = [*encoder_parameters, *readout_parameters]
        if not all(
            value.grad is not None and bool(torch.isfinite(value.grad).all())
            for value in parameters
        ):
            raise RuntimeError(f"{candidate.candidate_id} gradient became non-finite.")
        encoder_norm = _gradient_norm(encoder_parameters)
        readout_norm = _gradient_norm(readout_parameters)
        global_norm = math.sqrt(encoder_norm**2 + readout_norm**2)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if not all(bool(torch.isfinite(value).all()) for value in parameters):
            raise RuntimeError(f"{candidate.candidate_id} parameter became non-finite.")
        final_loss = float(sum(losses))
        if not math.isfinite(final_loss):
            raise RuntimeError(f"{candidate.candidate_id} loss became non-finite.")
        if collect_diagnostics:
            clipping_flags.append(global_norm > 1.0)
            encoder_norms.append(encoder_norm)
            readout_norms.append(readout_norm)
        if update in capture:
            output[update] = capture_scores(
                model, cache, target_indices, device=device, batch_size=16
            )
        model.train()
    if set(output) != capture:
        raise RuntimeError("L1FT checkpoint capture is incomplete.")
    result = L1FTTrajectoryResult(
        checkpoint_scores=output,
        final_loss=0.0 if max(capture) == 0 else final_loss,
        clipping_flags=clipping_flags,
        encoder_gradient_norms=encoder_norms,
        readout_gradient_norms=readout_norms,
        parameter_changes=model.parameter_changes(),
        state=model.export_state() if retain_state else None,
    )
    if max(capture) > 0:
        del scores, loss, parameters, encoder_parameters, readout_parameters
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def trainable_parameter_names_and_count(
    *,
    candidate: LastBlockCandidate,
    seed: int,
    last_block: nn.Module,
    post_layernorm: nn.Module,
    pooling_head: nn.Module,
    predecessor_state: Mapping[str, Any],
    base_scores: np.ndarray | None,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int]:
    model = LastBlockReadout(
        candidate=candidate,
        seed=seed,
        last_block=last_block,
        post_layernorm=post_layernorm,
        pooling_head=pooling_head,
        predecessor_state=predecessor_state,
        base_scores=base_scores,
        device=device,
    )
    rows = [
        {
            "parameter_name": name,
            "parameter_count": int(value.numel()),
            "branch": "encoder" if name.startswith(("last_block.", "post_layernorm.")) else "readout",
        }
        for name, value in [
            *model.encoder_named_parameters(),
            *model.readout_named_parameters(),
        ]
    ]
    count = sum(int(row["parameter_count"]) for row in rows)
    del model
    torch.cuda.empty_cache()
    return rows, count
