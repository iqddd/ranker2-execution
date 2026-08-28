"""Pure analytics shared by transductive lookup replay and later sensitivity work."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ranker.comparison_graph import connected_components as _connected_components
from ranker.data import RankedImage


def connected_components(image_ids: Sequence[str], groups: Sequence[Sequence[RankedImage]]) -> list[list[str]]:
    """Return deterministic components of the image-comparison graph."""
    allowed = {str(name) for name in image_ids}
    restricted = [
        [item for item in group if item.image_name in allowed] for group in groups
    ]
    return _connected_components(restricted, nodes=image_ids)


def canonical_group_audit(
    canonical_ids: Sequence[str], o_groups: Sequence[Sequence[RankedImage]], w_groups: Sequence[Sequence[RankedImage]]
) -> tuple[list[list[RankedImage]], list[dict[str, Any]]]:
    """Select whole groups contained in canonical520 and expose every exclusion."""
    canonical = set(map(str, canonical_ids))
    retained: list[list[RankedImage]] = []
    rows: list[dict[str, Any]] = []
    for source, groups in (("O", o_groups), ("W", w_groups)):
        for ordinal, group in enumerate(groups):
            members = {item.image_name for item in group}
            canonical_count = len(members & canonical)
            status = "retained" if members <= canonical else "excluded_outside_canonical_domain"
            if status == "retained":
                retained.append(list(group))
            rows.append(
                {
                    "source": source,
                    "source_ordinal": ordinal,
                    "group_index": int(group[0].group_index),
                    "group_size": len(group),
                    "canonical_images": canonical_count,
                    "outside_canonical_images": len(members) - canonical_count,
                    "status": status,
                }
            )
    return retained, rows
