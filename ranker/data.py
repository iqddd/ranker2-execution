from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


GROUP_SIZE = 8


def _load_top_level_array_prefix(
    state_file: Path, key: str, *, limit: int | None = None
) -> list[object]:
    """Incrementally deserialize a prefix of one top-level JSON array."""
    decoder = json.JSONDecoder()
    buffer = ""
    groups_start: int | None = None
    cursor: int | None = None
    array_opened = False
    items: list[object] = []
    with state_file.open(encoding="utf-8") as stream:
        while True:
            chunk = stream.read(4096)
            if chunk:
                buffer += chunk
            if groups_start is None:
                match = re.search(rf'"{re.escape(key)}"\s*:\s*', buffer)
                if match is not None:
                    groups_start = match.end()
                    cursor = groups_start
            if cursor is not None:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    if not chunk:
                        break
                    continue
                if not array_opened and buffer[cursor] == "[":
                    cursor += 1
                    array_opened = True
                while True:
                    while cursor < len(buffer) and (
                        buffer[cursor].isspace() or buffer[cursor] == ","
                    ):
                        cursor += 1
                    if cursor >= len(buffer):
                        break
                    if buffer[cursor] == "]":
                        return items
                    try:
                        payload, end = decoder.raw_decode(buffer, cursor)
                    except json.JSONDecodeError:
                        break
                    else:
                        items.append(payload)
                        cursor = end
                        if limit is not None and len(items) == limit:
                            return items
            if not chunk:
                break
    raise ValueError(f"Could not decode top-level array {key!r} from {state_file}.")


def load_group_memberships(
    state_file: Path, *, limit: int | None = None
) -> list[tuple[str, ...]]:
    """Load only structural group membership, without parsing rank state."""
    payload = _load_top_level_array_prefix(state_file, "groups", limit=limit)
    groups: list[tuple[str, ...]] = []
    for group in payload:
        if not isinstance(group, list) or not all(
            isinstance(name, str) for name in group
        ):
            raise ValueError("Every structural group must be a list of image names.")
        groups.append(tuple(group))
    return groups


@dataclass(frozen=True)
class RankedImage:
    group_index: int
    image_name: str
    identity: str
    rank: int
    z: float


def load_ranked_groups_prefix(
    state_file: Path, *, limit: int
) -> list[list[RankedImage]]:
    """Load exactly the first completed rank groups without reading later states."""
    memberships = load_group_memberships(state_file, limit=limit)
    states = _load_top_level_array_prefix(state_file, "group_states", limit=limit)
    if len(memberships) != limit or len(states) != limit:
        raise ValueError(f"Expected {limit} group memberships and rank states.")
    result: list[list[RankedImage]] = []
    for group_index, (membership, state) in enumerate(
        zip(memberships, states, strict=True)
    ):
        if not isinstance(state, dict):
            raise ValueError("Every group state must be an object.")
        ranked = state.get("ranked")
        unranked = state.get("unranked")
        if (
            not isinstance(ranked, list)
            or not all(isinstance(name, str) for name in ranked)
            or unranked != []
            or len(ranked) != GROUP_SIZE
            or set(ranked) != set(membership)
        ):
            raise ValueError(f"Group {group_index} is not a completed rank octet.")
        result.append(
            [
                RankedImage(
                    group_index=group_index,
                    image_name=image_name,
                    identity=extract_identity(image_name),
                    rank=rank,
                    z=normalize_rank(rank),
                )
                for rank, image_name in enumerate(ranked)
            ]
        )
    return result


def load_completed_ranked_groups_range(
    state_file: Path,
    *,
    start: int,
    stop: int,
    expected_sizes: tuple[int, ...] | None = None,
) -> list[list[RankedImage]]:
    """Load a completed contiguous group range, allowing variable group sizes.

    Unlike :func:`load_ranked_groups_prefix`, this helper is suitable for
    prospective manifests whose final ranking instance is not an octet.  Group
    indices remain the original top-level indices in ``state_file``.
    """
    if start < 0 or stop <= start:
        raise ValueError("Expected 0 <= start < stop.")
    memberships = load_group_memberships(state_file, limit=stop)
    states = _load_top_level_array_prefix(state_file, "group_states", limit=stop)
    if len(memberships) != stop or len(states) != stop:
        raise ValueError(f"Expected at least {stop} group memberships and rank states.")
    selected_memberships = memberships[start:stop]
    selected_states = states[start:stop]
    observed_sizes = tuple(len(group) for group in selected_memberships)
    if expected_sizes is not None and observed_sizes != expected_sizes:
        raise ValueError(
            f"Group-size contract mismatch: observed {observed_sizes}, "
            f"expected {expected_sizes}."
        )
    result: list[list[RankedImage]] = []
    for group_index, (membership, state) in enumerate(
        zip(selected_memberships, selected_states, strict=True), start=start
    ):
        if not isinstance(state, dict):
            raise ValueError("Every group state must be an object.")
        ranked = state.get("ranked")
        unranked = state.get("unranked")
        if (
            not isinstance(ranked, list)
            or not all(isinstance(name, str) for name in ranked)
            or unranked != []
            or len(ranked) != len(membership)
            or set(ranked) != set(membership)
        ):
            raise ValueError(f"Group {group_index} is not a completed ranking instance.")
        midpoint = (len(ranked) - 1.0) / 2.0
        if midpoint <= 0.0:
            raise ValueError("A completed ranking group must contain at least two images.")
        result.append(
            [
                RankedImage(
                    group_index=group_index,
                    image_name=image_name,
                    identity=extract_identity(image_name),
                    rank=rank,
                    z=(rank - midpoint) / midpoint,
                )
                for rank, image_name in enumerate(ranked)
            ]
        )
    return result


def extract_identity(image_name: str) -> str:
    stem = Path(image_name).stem
    if "_" not in stem:
        raise ValueError(f"Cannot infer identity from image name: {image_name}")
    return stem.rsplit("_", 1)[0]


def normalize_rank(rank: int) -> float:
    return (rank - 3.5) / 3.5


def load_ranked_groups(state_file: Path) -> list[list[RankedImage]]:
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    groups = payload["groups"]
    group_states = payload["group_states"]

    ranked_groups: list[list[RankedImage]] = []
    for group_index, (group, group_state) in enumerate(zip(groups, group_states, strict=True)):
        ranked = group_state["ranked"]
        unranked = group_state["unranked"]
        if ranked and not unranked:
            if len(group) != GROUP_SIZE or len(ranked) != GROUP_SIZE:
                raise ValueError(
                    f"Expected completed group {group_index} to have {GROUP_SIZE} items."
                )

            ranked_groups.append(
                [
                    RankedImage(
                        group_index=group_index,
                        image_name=image_name,
                        identity=extract_identity(image_name),
                        rank=rank,
                        z=normalize_rank(rank),
                    )
                    for rank, image_name in enumerate(ranked)
                ]
            )

    return ranked_groups


def build_folds(num_groups: int, folds: int, seed: int) -> list[list[int]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if num_groups < folds:
        raise ValueError("number of ranked groups must be >= folds")

    indices = list(range(num_groups))
    rng = random.Random(seed)
    rng.shuffle(indices)

    base_size, remainder = divmod(num_groups, folds)
    fold_sizes = [base_size + (1 if index < remainder else 0) for index in range(folds)]

    result: list[list[int]] = []
    start = 0
    for fold_size in fold_sizes:
        result.append(indices[start : start + fold_size])
        start += fold_size
    return result


def split_train_val_groups(
    ranked_groups: list[list[RankedImage]],
    val_indices: list[int],
) -> tuple[list[list[RankedImage]], list[list[RankedImage]]]:
    val_set = set(val_indices)
    train_groups = [
        group for index, group in enumerate(ranked_groups) if index not in val_set
    ]
    val_groups = [group for index, group in enumerate(ranked_groups) if index in val_set]
    return train_groups, val_groups


def flatten_groups(groups: list[list[RankedImage]]) -> list[RankedImage]:
    return [item for group in groups for item in group]


def count_pairs(groups: list[list[RankedImage]]) -> int:
    return sum(len(group) * (len(group) - 1) // 2 for group in groups)
