"""Structural-only balanced group partitions with complete W-group purge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from pyscipopt import Model, quicksum

from ranker.data import extract_identity, load_group_memberships


@dataclass(frozen=True)
class StructuralPartitionData:
    o_groups: tuple[tuple[str, ...], ...]
    w_groups: tuple[tuple[str, ...], ...]
    identities: tuple[str, ...]
    identity_counts: np.ndarray
    identity_totals: np.ndarray
    o_w_masks: tuple[int, ...]
    w_to_o: tuple[tuple[int, ...], ...]


def load_structural_partition_data(
    source_state: Path,
    within_state: Path,
    *,
    n_o_groups: int,
) -> StructuralPartitionData:
    """Load old-O/W membership only; annotation/rank state is not deserialized."""
    o_groups = tuple(load_group_memberships(source_state, limit=n_o_groups))
    w_groups = tuple(load_group_memberships(within_state))
    if len(o_groups) != n_o_groups or any(len(group) != 8 for group in o_groups):
        raise ValueError(f"Expected {n_o_groups} structural O groups of size eight.")
    o_names = [name for group in o_groups for name in group]
    if len(o_names) != len(set(o_names)):
        raise ValueError("Structural O images must occur exactly once.")
    if any(len(group) != 8 for group in w_groups):
        raise ValueError("Structural W groups must have size eight.")
    if not {name for group in w_groups for name in group}.issubset(set(o_names)):
        raise ValueError("Every W image must belong to the selected old-O universe.")
    identities = tuple(sorted({extract_identity(name) for name in o_names}))
    identity_index = {identity: index for index, identity in enumerate(identities)}
    identity_counts = np.zeros((n_o_groups, len(identities)), dtype=np.int64)
    for group_index, group in enumerate(o_groups):
        for name in group:
            identity_counts[group_index, identity_index[extract_identity(name)]] += 1
    o_sets = tuple(set(group) for group in o_groups)
    w_sets = tuple(set(group) for group in w_groups)
    w_to_o = tuple(
        tuple(group for group, images in enumerate(o_sets) if images & w_group)
        for w_group in w_sets
    )
    if any(not links for links in w_to_o):
        raise ValueError("Every W group must intersect the old-O universe.")
    o_w_masks = []
    for images in o_sets:
        mask = 0
        for w_group, w_images in enumerate(w_sets):
            if images & w_images:
                mask |= 1 << w_group
        o_w_masks.append(mask)
    return StructuralPartitionData(
        o_groups=o_groups,
        w_groups=w_groups,
        identities=identities,
        identity_counts=identity_counts,
        identity_totals=identity_counts.sum(axis=0),
        o_w_masks=tuple(o_w_masks),
        w_to_o=w_to_o,
    )


def canonicalize_folds(
    folds: Sequence[Sequence[int]], *, descending_size: bool = True
) -> tuple[tuple[int, ...], ...]:
    normalized = [tuple(sorted(map(int, fold))) for fold in folds]
    if descending_size:
        return tuple(sorted(normalized, key=lambda fold: (-len(fold), fold)))
    return tuple(sorted(normalized))


def fold_w_mask(groups: Sequence[int], o_w_masks: Sequence[int]) -> int:
    mask = 0
    for group in groups:
        mask |= int(o_w_masks[group])
    return mask


def solve_first_feasible_partition(
    data: StructuralPartitionData,
    *,
    fold_sizes: Sequence[int],
    max_purge_per_fold: int,
    max_total_purge: int,
    seed: int,
    time_limit_seconds: float,
) -> tuple[tuple[tuple[int, ...], ...] | None, dict[str, object]]:
    """Return SCIP's first feasible structural assignment, without an objective."""
    n_folds = len(fold_sizes)
    model = Model("structural_partition_feasibility")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", seed)
    model.setIntParam("randomization/permutationseed", seed)
    model.setIntParam("randomization/lpseed", seed)
    model.setRealParam("limits/time", float(time_limit_seconds))
    model.setIntParam("limits/solutions", 1)
    x = {
        (group, fold): model.addVar(vtype="B", name=f"x_{group}_{fold}")
        for group in range(len(data.o_groups))
        for fold in range(n_folds)
    }
    y = {
        (w_group, fold): model.addVar(vtype="B", name=f"y_{w_group}_{fold}")
        for w_group in range(len(data.w_groups))
        for fold in range(n_folds)
    }
    for group in range(len(data.o_groups)):
        model.addCons(quicksum(x[group, fold] for fold in range(n_folds)) == 1)
    for fold, size in enumerate(fold_sizes):
        model.addCons(
            quicksum(x[group, fold] for group in range(len(data.o_groups))) == size
        )
    for w_group, links in enumerate(data.w_to_o):
        for fold in range(n_folds):
            for group in links:
                model.addCons(y[w_group, fold] >= x[group, fold])
            model.addCons(
                y[w_group, fold] <= quicksum(x[group, fold] for group in links)
            )
    purge_sums = [
        quicksum(y[w_group, fold] for w_group in range(len(data.w_groups)))
        for fold in range(n_folds)
    ]
    for purge in purge_sums:
        model.addCons(purge <= max_purge_per_fold)
    model.addCons(quicksum(purge_sums) <= max_total_purge)
    model.setObjective(0.0, "minimize")
    started = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - started
    n_solutions = int(model.getNSols())
    record: dict[str, object] = {
        "status": str(model.getStatus()),
        "feasible": n_solutions > 0,
        "n_solutions": n_solutions,
        "seconds": elapsed,
        "nodes": int(model.getNNodes()),
        "threads": 1,
        "randomization_seed": seed,
        "permutation_seed": seed,
        "solution_limit": 1,
        "time_limit_seconds": time_limit_seconds,
        "objective": "none",
    }
    if not n_solutions:
        return None, record
    solution = model.getBestSol()
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for group in range(len(data.o_groups)):
        values = [float(model.getSolVal(solution, x[group, fold])) for fold in range(n_folds)]
        if any(value not in (0.0, 1.0) for value in values) or sum(values) != 1.0:
            raise RuntimeError("SCIP feasibility assignment failed exact binary acceptance.")
        folds[values.index(1.0)].append(group)
    canonical = canonicalize_folds(folds)
    purge_counts = [fold_w_mask(fold, data.o_w_masks).bit_count() for fold in canonical]
    record.update(
        {
            "canonical_assignment": [list(fold) for fold in canonical],
            "P_max": max(purge_counts),
            "P_total": sum(purge_counts),
        }
    )
    return canonical, record
