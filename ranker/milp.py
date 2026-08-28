"""Shared deterministic lexicographic optimization for SCIP-based steps."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class LexicographicStage:
    name: str
    expression: Any
    sense: Literal["maximize", "minimize"]
    tolerance: float
    value_of: Callable[[Any, Sequence[Any]], float]
    fix: bool = True


def optimize_lexicographic(
    model: Any,
    stages: Sequence[LexicographicStage],
    *,
    select: Callable[[Any], Sequence[Any]],
    time_limit: float,
    require_optimal: bool = False,
    tick: Callable[[], None] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Optimize and fix each incumbent before advancing to the next objective."""
    if time_limit <= 0.0 or not stages:
        raise ValueError("Lexicographic optimization requires stages and a time limit.")
    started = time.perf_counter()
    selected: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        if tick is not None:
            tick()
        remaining = max(0.1, time_limit - (time.perf_counter() - started))
        model.setParam("limits/time", remaining)
        model.setObjective(stage.expression, stage.sense)
        stage_started = time.perf_counter()
        model.optimize()
        solution = model.getBestSol()
        if solution is None:
            raise RuntimeError(f"No feasible incumbent for objective {stage.name!r}.")
        status = str(model.getStatus())
        if require_optimal and status != "optimal":
            raise RuntimeError(
                f"Objective {stage.name!r} did not reach optimal status: {status}."
            )
        selected = list(select(solution))
        value = float(stage.value_of(solution, selected))
        diagnostics.append(
            {
                "stage_index": index,
                "objective": stage.name,
                "value": value,
                "sense": stage.sense,
                "status": status,
                "relative_gap": float(model.getGap()),
                "solver_seconds": time.perf_counter() - stage_started,
                "remaining_seconds_before_stage": remaining,
                "fixed_for_next_stage": stage.fix,
            }
        )
        if stage.fix:
            model.freeTransform()
            if stage.sense == "maximize":
                model.addCons(stage.expression >= value - stage.tolerance)
            else:
                model.addCons(stage.expression <= value + stage.tolerance)
    return selected, diagnostics
