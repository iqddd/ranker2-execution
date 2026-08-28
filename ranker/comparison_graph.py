"""Deterministic connected components for ranked comparison graphs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ranker.data import RankedImage


def connected_components(
    groups: Iterable[Sequence[RankedImage]], *, nodes: Iterable[str] = ()
) -> list[list[str]]:
    """Return components with optional isolated nodes in deterministic order."""
    names = {str(name) for name in nodes}
    adjacency: dict[str, set[str]] = {name: set() for name in names}
    for group in groups:
        members = [str(item.image_name) for item in group]
        names.update(members)
        for name in members:
            adjacency.setdefault(name, set()).update(
                other for other in members if other != name
            )
    unseen = set(names)
    output: list[list[str]] = []
    while unseen:
        seed = min(unseen, key=str.encode)
        unseen.remove(seed)
        component = {seed}
        stack = [seed]
        while stack:
            neighbours = adjacency[stack.pop()] & unseen
            unseen.difference_update(neighbours)
            component.update(neighbours)
            stack.extend(neighbours)
        output.append(sorted(component, key=str.encode))
    return sorted(output, key=lambda value: (-len(value), value))
