"""Canonical ranked-pair enumeration and paired correctness transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ranker.data import RankedImage


def canonical_edge(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def signum(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


@dataclass(frozen=True)
class PairTable:
    lower: np.ndarray
    higher: np.ndarray
    distance: np.ndarray
    low_rank: np.ndarray
    high_rank: np.ndarray
    group_size: np.ndarray
    group_position: np.ndarray
    lower_identity: np.ndarray
    higher_identity: np.ndarray
    lower_name: np.ndarray
    higher_name: np.ndarray

    def __len__(self) -> int:
        return len(self.lower)

    def mask(
        self, name: str, *, heldout: Sequence[str] = ()
    ) -> np.ndarray:
        all_pairs = np.ones(len(self), dtype=bool)
        same_identity = self.lower_identity == self.higher_identity
        heldout_set = {str(value) for value in heldout}
        lower_heldout = np.isin(self.lower_identity, list(heldout_set))
        higher_heldout = np.isin(self.higher_identity, list(heldout_set))
        involved = lower_heldout | higher_heldout
        both = lower_heldout & higher_heldout
        cross = both & ~same_identity
        same_heldout = both & same_identity
        masks = {
            "all": all_pairs,
            "all_val": all_pairs,
            "W": all_pairs,
            "distance_ge_3": self.distance >= 3,
            "same_identity": same_identity,
            "cross_identity": ~same_identity,
            "outer_quartile": (self.low_rank <= 1)
            & (self.high_rank >= self.group_size - 2),
            "rank_extreme": (self.low_rank == 0)
            & (self.high_rank == self.group_size - 1),
            "heldout_involved": involved,
            "both_heldout_all": both,
            "both_heldout_d_ge_3": both & (self.distance >= 3),
            "heldout_cross_d_ge_3": cross & (self.distance >= 3),
            "heldout_same_d_ge_3": same_heldout & (self.distance >= 3),
            "heldout_involved_d_ge_3": involved & (self.distance >= 3),
            "heldout_involved_outer_quartile": involved
            & (self.low_rank <= 1)
            & (self.high_rank >= self.group_size - 2),
            "heldout_involved_rank_extreme": involved
            & (self.low_rank == 0)
            & (self.high_rank == self.group_size - 1),
        }
        if name not in masks:
            raise KeyError(f"Unknown pair stratum {name!r}.")
        return masks[name]


def enumerate_group_pairs(
    groups: Sequence[Sequence[RankedImage]],
    *,
    index_of: Mapping[str, int] | Callable[[str], int],
) -> PairTable:
    """Enumerate every lower/higher ranked pair in deterministic group order."""

    lookup = index_of.__getitem__ if isinstance(index_of, Mapping) else index_of
    values: dict[str, list[Any]] = {
        name: []
        for name in (
            "lower",
            "higher",
            "distance",
            "low_rank",
            "high_rank",
            "group_size",
            "group_position",
            "lower_identity",
            "higher_identity",
            "lower_name",
            "higher_name",
        )
    }
    for group_position, group in enumerate(groups):
        ordered = sorted(group, key=lambda item: item.rank)
        if [item.rank for item in ordered] != list(range(len(ordered))):
            raise ValueError("Ranked groups must have contiguous zero-based ranks.")
        for low_rank, lower in enumerate(ordered):
            for high_rank in range(low_rank + 1, len(ordered)):
                higher = ordered[high_rank]
                values["lower"].append(int(lookup(lower.image_name)))
                values["higher"].append(int(lookup(higher.image_name)))
                values["distance"].append(high_rank - low_rank)
                values["low_rank"].append(low_rank)
                values["high_rank"].append(high_rank)
                values["group_size"].append(len(ordered))
                values["group_position"].append(group_position)
                values["lower_identity"].append(str(lower.identity))
                values["higher_identity"].append(str(higher.identity))
                values["lower_name"].append(str(lower.image_name))
                values["higher_name"].append(str(higher.image_name))
    return PairTable(
        lower=np.asarray(values["lower"], dtype=np.int64),
        higher=np.asarray(values["higher"], dtype=np.int64),
        distance=np.asarray(values["distance"], dtype=np.int16),
        low_rank=np.asarray(values["low_rank"], dtype=np.int16),
        high_rank=np.asarray(values["high_rank"], dtype=np.int16),
        group_size=np.asarray(values["group_size"], dtype=np.int16),
        group_position=np.asarray(values["group_position"], dtype=np.int64),
        lower_identity=np.asarray(values["lower_identity"], dtype=str),
        higher_identity=np.asarray(values["higher_identity"], dtype=str),
        lower_name=np.asarray(values["lower_name"], dtype=str),
        higher_name=np.asarray(values["higher_name"], dtype=str),
    )


@dataclass
class TransitionTally:
    n_pairs: int = 0
    baseline_correct: int = 0
    candidate_correct: int = 0
    both_correct: int = 0
    recovered: int = 0
    broken: int = 0
    both_wrong: int = 0

    def add(
        self,
        baseline_correct: np.ndarray,
        candidate_correct: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> TransitionTally:
        baseline = np.asarray(baseline_correct, dtype=bool)
        candidate = np.asarray(candidate_correct, dtype=bool)
        if baseline.shape != candidate.shape:
            raise ValueError("Correctness arrays must have equal shape.")
        selected = np.ones(baseline.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if selected.shape != baseline.shape:
            raise ValueError("Transition mask must match correctness arrays.")
        baseline = baseline[selected]
        candidate = candidate[selected]
        self.n_pairs += len(baseline)
        self.baseline_correct += int(np.count_nonzero(baseline))
        self.candidate_correct += int(np.count_nonzero(candidate))
        self.both_correct += int(np.count_nonzero(baseline & candidate))
        self.recovered += int(np.count_nonzero(~baseline & candidate))
        self.broken += int(np.count_nonzero(baseline & ~candidate))
        self.both_wrong += int(np.count_nonzero(~baseline & ~candidate))
        return self

    def merge(self, other: TransitionTally) -> TransitionTally:
        for name in (
            "n_pairs",
            "baseline_correct",
            "candidate_correct",
            "both_correct",
            "recovered",
            "broken",
            "both_wrong",
        ):
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))
        return self

    def finish(self) -> dict[str, int | float | None]:
        if self.both_correct + self.recovered + self.broken + self.both_wrong != self.n_pairs:
            raise RuntimeError("Transition cells do not partition the selected pairs.")
        if self.candidate_correct - self.baseline_correct != self.recovered - self.broken:
            raise RuntimeError("Transition net identity failed.")
        return {
            "n_pairs": self.n_pairs,
            "baseline_correct": self.baseline_correct,
            "candidate_correct": self.candidate_correct,
            "both_correct": self.both_correct,
            "recovered": self.recovered,
            "broken": self.broken,
            "both_wrong": self.both_wrong,
            "net": self.recovered - self.broken,
            "delta": (
                (self.recovered - self.broken) / self.n_pairs
                if self.n_pairs
                else None
            ),
        }

    def as_step43e(self) -> dict[str, int | float | str]:
        values = self.finish()
        return {
            "n_pairs": self.n_pairs,
            "BASE_correct": self.baseline_correct,
            "CANDIDATE_correct": self.candidate_correct,
            "recovered": self.recovered,
            "broken": self.broken,
            "both_wrong": self.both_wrong,
            "baseline_correct_candidate_correct": self.both_correct,
            "net_corrected": int(values["net"]),
            "delta_pair_accuracy": values["delta"] if values["delta"] is not None else "",
        }

    def as_head_base(self) -> dict[str, int | float]:
        return {
            "n": self.n_pairs,
            "HEAD_wins": self.candidate_correct,
            "BASE_wins": self.baseline_correct,
            "HEAD_win_rate": (
                self.candidate_correct / self.n_pairs if self.n_pairs else 0.0
            ),
        }


def tally_record_transitions(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_key: str,
    candidate_key: str,
    selected: Callable[[Mapping[str, Any]], bool] | None = None,
) -> TransitionTally:
    """Build a transition tally from heterogeneous analysis records."""
    chosen = list(rows) if selected is None else [row for row in rows if selected(row)]
    return TransitionTally().add(
        np.asarray([bool(row[baseline_key]) for row in chosen], dtype=bool),
        np.asarray([bool(row[candidate_key]) for row in chosen], dtype=bool),
    )


def grouped_transition_signs(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    group_order: Sequence[str],
    baseline_key: str,
    candidate_key: str,
    selected: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, int]:
    """Count positive/zero/negative candidate-minus-baseline group nets."""
    nets = {str(group): 0 for group in group_order}
    for row in rows:
        if selected is not None and not selected(row):
            continue
        nets[str(row[group_key])] += int(bool(row[candidate_key])) - int(
            bool(row[baseline_key])
        )
    return {
        "positive_groups": sum(value > 0 for value in nets.values()),
        "zero_groups": sum(value == 0 for value in nets.values()),
        "negative_groups": sum(value < 0 for value in nets.values()),
    }
