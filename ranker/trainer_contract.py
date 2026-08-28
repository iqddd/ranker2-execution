"""Reusable primitives frozen by TRAINER_CONTRACT_V1."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import RankedImage
from .latent_scale import midrank_percentiles


O_RELIABILITY: Mapping[int, float] = MappingProxyType({1: 0.77635})
W_RELIABILITY: Mapping[int, float] = MappingProxyType({1: 0.50048, 2: 0.79408})


def reliability_weight(source: str, distance: int) -> float:
    """Return the frozen LEVEL_METRICS_V1 pair reliability weight."""
    if source == "O":
        return O_RELIABILITY.get(int(distance), 1.0)
    if source == "W":
        return W_RELIABILITY.get(int(distance), 1.0)
    raise ValueError(f"Unknown group source {source!r}.")


def mean_pair_total_coefficient_mass(
    o_groups: Sequence[Sequence[RankedImage]],
    w_groups: Sequence[Sequence[RankedImage]],
    *,
    o_coefficient: float = 0.75,
    w_coefficient: float = 0.25,
    pair_weighting: str = "REL",
) -> float:
    """Return the frozen full-fold coefficient mass of raw mean-pair loss."""

    def source_mass(
        groups: Sequence[Sequence[RankedImage]], source: str, coefficient: float
    ) -> float:
        if not groups:
            raise ValueError(f"{source} training groups must not be empty.")
        group_masses: list[float] = []
        for group in groups:
            weights: list[float] = []
            for first_index, first in enumerate(group):
                for second in group[first_index + 1 :]:
                    distance = abs(int(first.rank) - int(second.rank))
                    if pair_weighting == "REL":
                        weights.append(reliability_weight(source, distance))
                    elif pair_weighting == "UNIF":
                        weights.append(1.0)
                    else:
                        raise ValueError(
                            f"Unknown pair-weighting scheme {pair_weighting!r}."
                        )
            if not weights:
                raise ValueError("A ranked group must produce comparisons.")
            group_masses.append(float(sum(weights)) / len(weights))
        return float(coefficient) * math.fsum(group_masses) / len(groups)

    mass = source_mass(o_groups, "O", o_coefficient) + source_mass(
        w_groups, "W", w_coefficient
    )
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("Mean-pair coefficient mass must be finite and positive.")
    return mass


def training_group_manifest_digest(
    o_groups: Sequence[Sequence[RankedImage]],
    w_groups: Sequence[Sequence[RankedImage]],
) -> str:
    """Hash the exact ordered training groups used by a fold."""
    payload = []
    for source, groups in (("O", o_groups), ("W", w_groups)):
        for group_index, group in enumerate(groups):
            payload.append(
                {
                    "source": source,
                    "group_index": group_index,
                    "items": [
                        {
                            "image_name": item.image_name,
                            "identity": item.identity,
                            "rank": int(item.rank),
                        }
                        for item in group
                    ],
                }
            )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mean_pair_coefficient_vector(
    o_groups: Sequence[Sequence[RankedImage]],
    w_groups: Sequence[Sequence[RankedImage]],
    *,
    o_coefficient: float = 0.75,
    w_coefficient: float = 0.25,
    pair_weighting: str = "REL",
) -> np.ndarray:
    """Build the ordered Float64 full-fold raw mean-pair coefficient vector."""
    coefficients: list[float] = []
    for source, groups, source_coefficient in (
        ("O", o_groups, o_coefficient),
        ("W", w_groups, w_coefficient),
    ):
        if not groups:
            raise ValueError(f"{source} training groups must not be empty.")
        for group in groups:
            weights: list[float] = []
            for first_index, first in enumerate(group):
                for second in group[first_index + 1 :]:
                    distance = abs(int(first.rank) - int(second.rank))
                    if pair_weighting == "REL":
                        weights.append(reliability_weight(source, distance))
                    elif pair_weighting == "UNIF":
                        weights.append(1.0)
                    else:
                        raise ValueError(
                            f"Unknown pair-weighting scheme {pair_weighting!r}."
                        )
            denominator = len(weights)
            coefficients.extend(
                float(source_coefficient) * weight / (len(groups) * denominator)
                for weight in weights
            )
    return np.asarray(coefficients, dtype=np.float64)


def coefficient_vector_digest(coefficients: np.ndarray) -> str:
    """Hash a coefficient vector with the project's tensor-header convention."""
    array = np.ascontiguousarray(coefficients)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def reliability_weighted_group_loss(
    scores: torch.Tensor,
    group: Sequence[RankedImage],
    source: str,
    *,
    pair_weighting: str = "REL",
    normalization: str = "group_weight",
) -> torch.Tensor:
    """Compute the normalized positive-sign RankNet loss for one ranked group."""
    if scores.ndim != 1 or len(scores) != len(group):
        raise ValueError("Group scores and ranked items must have matching lengths.")
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for first_index, first in enumerate(group):
        for second_index in range(first_index + 1, len(group)):
            second = group[second_index]
            if first.rank < second.rank:
                lower_index, higher_index = first_index, second_index
                distance = second.rank - first.rank
            else:
                lower_index, higher_index = second_index, first_index
                distance = first.rank - second.rank
            if pair_weighting == "REL":
                weight = reliability_weight(source, int(distance))
            elif pair_weighting == "UNIF":
                weight = 1.0
            else:
                raise ValueError(f"Unknown pair-weighting scheme {pair_weighting!r}.")
            losses.append(F.softplus(-(scores[higher_index] - scores[lower_index])) * weight)
            weights.append(weight)
    if not losses or sum(weights) <= 0.0:
        raise ValueError("A ranked group must produce positive-weight comparisons.")
    if normalization == "group_weight":
        denominator = float(sum(weights))
    elif normalization == "mean_pair":
        denominator = float(len(losses))
    else:
        raise ValueError(
            "normalization must be 'group_weight' or 'mean_pair'."
        )
    return torch.stack(losses).sum() / denominator


def factorial_group_loss(
    scores: torch.Tensor,
    group: Sequence[RankedImage],
    source: str,
    *,
    target_kind: str,
    pair_weighting: str,
) -> torch.Tensor:
    """Generalized per-group loss used by the STEP51A frozen-linear screen.

    Normalization is deliberately local to the sampled group.  ``DIST`` is
    the historical D2 distance weight, while ``SOFT`` is its historical
    distance-dependent RankNet target.
    """
    if scores.ndim != 1 or len(scores) != len(group):
        raise ValueError("Group scores and ranked items must have matching lengths.")
    if target_kind not in {"HARD", "SOFT"}:
        raise ValueError(f"Unknown target kind {target_kind!r}.")
    if pair_weighting not in {"REL", "DIST"}:
        raise ValueError(f"Unknown pair-weighting scheme {pair_weighting!r}.")
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for first_index, first in enumerate(group):
        for second_index in range(first_index + 1, len(group)):
            second = group[second_index]
            if first.rank < second.rank:
                lower_index, higher_index = first_index, second_index
                distance = second.rank - first.rank
            else:
                lower_index, higher_index = second_index, first_index
                distance = first.rank - second.rank
            weight = (
                reliability_weight(source, int(distance))
                if pair_weighting == "REL"
                else float(distance)
            )
            logit = scores[higher_index] - scores[lower_index]
            if target_kind == "HARD":
                loss = F.softplus(-logit)
            else:
                target = 0.5 + 0.45 * ((float(distance) / 7.0) ** 0.7)
                loss = F.binary_cross_entropy_with_logits(
                    logit, torch.as_tensor(target, dtype=logit.dtype, device=logit.device)
                )
            losses.append(loss * weight)
            weights.append(weight)
    if not losses or sum(weights) <= 0.0:
        raise ValueError("A ranked group must produce positive-weight comparisons.")
    return torch.stack(losses).sum() / float(sum(weights))


@dataclass
class CyclingGroupSampler:
    """Independent shuffled cycles with no replacement inside each cycle."""

    size: int
    seed: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("A group sampler cannot be empty.")
        self._rng = np.random.default_rng(self.seed)
        self._order = np.empty(0, dtype=np.int64)
        self._position = 0

    def next(self) -> int:
        if self._position >= len(self._order):
            self._order = self._rng.permutation(self.size)
            self._position = 0
        value = int(self._order[self._position])
        self._position += 1
        return value


def scheduled_learning_rate(base_lr: float, update: int) -> float:
    """Eight-update linear warmup followed by cosine decay through update 256."""
    if not 0 <= update <= 256:
        raise ValueError("TRAINER_CONTRACT_V1 schedule is defined on updates 0..256.")
    if update == 0:
        return 0.0
    if update <= 8:
        return float(base_lr) * update / 8.0
    progress = (update - 8.0) / (256.0 - 8.0)
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _tail_indices(
    values: np.ndarray, image_ids: np.ndarray, selected: np.ndarray, fraction: float
) -> np.ndarray:
    candidates = np.flatnonzero(selected)
    count = int(math.ceil(fraction * len(candidates)))
    order = np.lexsort((image_ids[candidates], -values[candidates]))
    return candidates[order[:count]]


def evaluate_level_metrics(
    *,
    image_ids: np.ndarray,
    identities: np.ndarray,
    outer_indices: np.ndarray,
    human_folds: np.ndarray,
    model_folds: np.ndarray,
) -> dict[str, float]:
    """Evaluate the seven replay metrics frozen in LEVEL_METRICS_V1."""
    image_ids = np.asarray(image_ids).astype(str)
    identities = np.asarray(identities).astype(str)
    outer_indices = np.asarray(outer_indices)
    human_folds = np.asarray(human_folds, dtype=np.float64)
    model_folds = np.asarray(model_folds, dtype=np.float64)
    if (
        outer_indices.shape != human_folds.shape
        or human_folds.shape != model_folds.shape
        or human_folds.shape[0] != len(image_ids)
        or not np.all(np.isfinite(human_folds))
        or not np.all(np.isfinite(model_folds))
    ):
        raise ValueError("Level evaluator received invalid fold-percentile arrays.")
    identity_order = sorted(set(identities.tolist()))
    human_mean = human_folds.mean(axis=1)
    model_mean = model_folds.mean(axis=1)
    absolute_level = np.abs(model_mean - human_mean)
    identity_tails = []
    for identity in identity_order:
        selected = identities == identity
        tail = _tail_indices(absolute_level, image_ids, selected, 0.10)
        identity_tails.append(float(np.mean(absolute_level[tail])))

    offsets: list[float] = []
    frame_absolute: list[float] = []
    frame_tails: dict[str, list[float]] = {identity: [] for identity in identity_order}
    beta_values: list[float] = []
    strong_correct = 0
    strong_count = 0
    for outer_index in sorted(set(outer_indices.ravel().tolist())):
        rows, _ = np.where(outer_indices == outer_index)
        for identity in sorted(set(identities[rows].tolist())):
            selected_rows = np.asarray(
                [row for row in rows if identities[row] == identity], dtype=np.int64
            )
            columns = np.asarray(
                [int(np.flatnonzero(outer_indices[row] == outer_index)[0]) for row in selected_rows],
                dtype=np.int64,
            )
            human = human_folds[selected_rows, columns]
            model = model_folds[selected_rows, columns]
            human_identity_mean = float(np.mean(human))
            model_identity_mean = float(np.mean(model))
            offsets.append(model_identity_mean - human_identity_mean)
            delta_human = human - human_identity_mean
            delta_model = model - model_identity_mean
            absolute_frame = np.abs(delta_model - delta_human)
            frame_absolute.extend(absolute_frame.tolist())
            order = np.lexsort((image_ids[selected_rows], -absolute_frame))
            count = int(math.ceil(0.10 * len(selected_rows)))
            frame_tails[identity].append(float(np.mean(absolute_frame[order[:count]])))
            denominator = float(np.sum(delta_human**2))
            if denominator <= 0.0:
                raise ValueError("Human deviation amplitude denominator is zero.")
            beta_values.append(float(np.sum(delta_human * delta_model) / denominator))
            negative = delta_human <= -20.0
            positive = delta_human >= 20.0
            strong_correct += int(np.count_nonzero(negative & (delta_model < 0.0)))
            strong_correct += int(np.count_nonzero(positive & (delta_model > 0.0)))
            strong_count += int(np.count_nonzero(negative | positive))
    if not strong_count:
        raise ValueError("Strong-deviation replay stratum is empty.")
    offsets_array = np.asarray(offsets, dtype=np.float64)
    return {
        "IB_LTE10": float(np.mean(identity_tails)),
        "IB_FrameLTE10": float(
            np.mean([np.mean(frame_tails[identity]) for identity in identity_order])
        ),
        "IdentityOffsetMAE": float(np.mean(np.abs(offsets_array))),
        "strong_deviation_sign_accuracy": strong_correct / strong_count,
        "mean_deviation_beta": float(np.mean(beta_values)),
        "GLE40": float(np.mean(absolute_level >= 40.0)),
        "GLE60": float(np.mean(absolute_level >= 60.0)),
    }


def heldout_fold_metrics(
    *,
    image_ids: np.ndarray,
    model_percentiles: np.ndarray,
    human_percentiles: np.ndarray,
) -> dict[str, float]:
    """Finite one-identity metric subset used by the sentinel smoke test."""
    image_ids = np.asarray(image_ids).astype(str)
    model = np.asarray(model_percentiles, dtype=np.float64)
    human = np.asarray(human_percentiles, dtype=np.float64)
    absolute = np.abs(model - human)
    tail = _tail_indices(absolute, image_ids, np.ones(len(image_ids), dtype=bool), 0.10)
    model_delta = model - model.mean()
    human_delta = human - human.mean()
    frame_absolute = np.abs(model_delta - human_delta)
    frame_tail = _tail_indices(
        frame_absolute, image_ids, np.ones(len(image_ids), dtype=bool), 0.10
    )
    strong = np.abs(human_delta) >= 20.0
    sign_correct = np.sign(model_delta[strong]) == np.sign(human_delta[strong])
    denominator = float(np.sum(human_delta**2))
    return {
        "IB_LTE10": float(np.mean(absolute[tail])),
        "IB_FrameLTE10": float(np.mean(frame_absolute[frame_tail])),
        "IdentityOffsetMAE": float(abs(model.mean() - human.mean())),
        "strong_deviation_sign_accuracy": float(np.mean(sign_correct)) if np.any(strong) else 1.0,
        "mean_deviation_beta": float(np.sum(human_delta * model_delta) / denominator),
        "GLE40": float(np.mean(absolute >= 40.0)),
        "GLE60": float(np.mean(absolute >= 60.0)),
    }
