"""Pure metrics and decision rules for full-data horizon transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def global_ima_vector(
    maps: Mapping[str, Mapping[str, float]],
    names: Sequence[str],
    identity_ids: Sequence[str],
    theta_percentile: Sequence[float],
    identities: Sequence[str],
) -> np.ndarray:
    """Return one absolute global theta-percentile mean offset per identity."""
    image_names = np.asarray(names).astype(str)
    image_identities = np.asarray(identity_ids).astype(str)
    human = np.asarray(theta_percentile, dtype=np.float64)
    rows: list[float] = []
    for identity in identities:
        selected = image_identities == identity
        model = np.asarray(
            [float(maps[identity][name]) for name in image_names[selected]],
            dtype=np.float64,
        )
        rows.append(abs(float(model.mean()) - float(human[selected].mean())))
    return np.asarray(rows, dtype=np.float64)


def classify_horizon_transfer(
    direct: Mapping[str, Any],
    offset: Mapping[str, Any],
    gross: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered STEP59A D-contrast decision table."""
    point = float(direct["point_delta"])
    lower = float(direct["CI90_lower"])
    upper = float(direct["CI90_upper"])
    offset_harm = float(offset["point_delta"]) > 0.0 and float(offset["CI90_lower"]) > 0.0
    gross_harm = float(gross["point_delta"]) >= 0.005 and float(gross["CI90_lower"]) > 0.0
    candidate_signal = point <= -0.005 and upper < 0.0
    sign = (
        "resolved_improvement"
        if point <= -0.005 and upper < 0.0
        else "resolved_submargin_improvement"
        if -0.005 < point < 0.0 and upper < 0.0
        else "resolved_submargin_harm"
        if 0.0 < point < 0.005 and lower > 0.0
        else "sign_unresolved"
    )
    if candidate_signal:
        result = (
            "development_candidate_guard_blocked"
            if offset_harm or gross_harm
            else "development_candidate_horizon_supported"
        )
    elif lower >= -0.005 and upper <= 0.005:
        result = "practical_equivalence"
    elif point >= 0.005 and lower > 0.0:
        result = "resolved_harm"
    elif -0.005 < point < 0.0 and upper < 0.0:
        result = "resolved_submargin_improvement"
    elif 0.0 < point < 0.005 and lower > 0.0:
        result = "resolved_submargin_harm"
    else:
        result = "unresolved"
    return {
        "STEP59A_result": result,
        "direct_sign_annotation": sign,
        "resolved_G_harm": offset_harm,
        "resolved_gross_harm": gross_harm,
        "candidate_signal": candidate_signal,
        "cp112_retained_by_economy": result == "practical_equivalence",
    }
