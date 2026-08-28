from __future__ import annotations

"""Create the fixed follow-up annotation batches from SUMMARY.md."""

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ranker.data import extract_identity
from ranker.io import atomic_json, sha256_file


@dataclass(frozen=True)
class FollowupGroupProtocol:
    """Selection parameters supplied by the step that constructs a batch."""

    source_group_count: int
    expected_identity_count: int
    group_size: int
    test_retest_group_indices: tuple[int, ...]
    anchors: tuple[str, ...]
    anchor_contexts: int
    within_identity_octets_per_identity: int
    within_identity_retest_octets_per_identity: int
    selection_seed: int
    extension_seed: int
    within_identity_retest_seed: int
    within_identity_group_label_plural: str


def load_source_groups(
    state_file: Path, *, protocol: FollowupGroupProtocol
) -> list[list[str]]:
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    groups = payload["groups"][:protocol.source_group_count]
    states = payload["group_states"][:protocol.source_group_count]
    if len(groups) != protocol.source_group_count:
        raise ValueError(f"Expected at least {protocol.source_group_count} source groups.")
    if any(
        state["unranked"] or len(state["ranked"]) != protocol.group_size
        for state in states
    ):
        raise ValueError("Every source group must be fully ranked before planning follow-up.")
    if any(len(group) != protocol.group_size for group in groups):
        raise ValueError(
            f"Every source group must contain exactly {protocol.group_size} images."
        )
    return [list(group) for group in groups]


def build_anchor_groups(
    source_groups: list[list[str]], *, protocol: FollowupGroupProtocol
) -> list[dict[str, Any]]:
    all_images = [image for group in source_groups for image in group]
    if len(all_images) != len(set(all_images)):
        raise ValueError("Source images must be unique across the configured groups.")
    missing = set(protocol.anchors) - set(all_images)
    if missing:
        raise ValueError(f"Anchors are missing from the source groups: {sorted(missing)}")

    rng = random.Random(protocol.selection_seed)
    images_by_identity: dict[str, list[str]] = defaultdict(list)
    for image in all_images:
        if image not in protocol.anchors:
            images_by_identity[extract_identity(image)].append(image)
    for images in images_by_identity.values():
        rng.shuffle(images)

    anchor_schedule = [
        anchor for _ in range(protocol.anchor_contexts) for anchor in protocol.anchors
    ]
    rng.shuffle(anchor_schedule)
    remaining = Counter({identity: len(images) for identity, images in images_by_identity.items()})
    groups: list[dict[str, Any]] = []
    for batch_index, anchor in enumerate(anchor_schedule, start=1):
        anchor_identity = extract_identity(anchor)
        chosen_identities: set[str] = {anchor_identity}
        fillers: list[str] = []
        for _ in range(protocol.group_size - 1):
            candidates = [
                identity
                for identity, count in remaining.items()
                if count and identity not in chosen_identities
            ]
            if not candidates:
                raise RuntimeError("Could not allocate enough distinct filler identities.")
            highest_count = max(remaining[identity] for identity in candidates)
            tied = sorted(identity for identity in candidates if remaining[identity] == highest_count)
            identity = rng.choice(tied)
            fillers.append(images_by_identity[identity].pop())
            remaining[identity] -= 1
            chosen_identities.add(identity)
        images = [anchor, *fillers]
        rng.shuffle(images)
        groups.append(
            {
                "batch_group": batch_index,
                "anchor": anchor,
                "images": images,
            }
        )

    if any(
        len({extract_identity(image) for image in group["images"]})
        != protocol.group_size
        for group in groups
    ):
        raise AssertionError("Anchor groups must contain distinct identities.")
    filler_images = [
        image for group in groups for image in group["images"] if image != group["anchor"]
    ]
    if len(filler_images) != len(set(filler_images)):
        raise AssertionError("A filler image was assigned more than once.")
    if Counter(group["anchor"] for group in groups) != Counter(
        {anchor: protocol.anchor_contexts for anchor in protocol.anchors}
    ):
        raise AssertionError("Each anchor must occur in every planned context.")
    return groups


def build_within_identity_groups(
    source_groups: list[list[str]], *, protocol: FollowupGroupProtocol
) -> list[dict[str, Any]]:
    """Create four disjoint single-identity octets for every source identity."""
    all_images = [image for group in source_groups for image in group]
    if len(all_images) != len(set(all_images)):
        raise ValueError("Source images must be unique across the configured groups.")

    images_by_identity: dict[str, list[str]] = defaultdict(list)
    for image in all_images:
        images_by_identity[extract_identity(image)].append(image)

    required_images_per_identity = (
        protocol.within_identity_octets_per_identity * protocol.group_size
    )
    insufficient_identities = {
        identity: len(images)
        for identity, images in images_by_identity.items()
        if len(images) < required_images_per_identity
    }
    if insufficient_identities:
        raise ValueError(
            "Each identity needs at least "
            f"{required_images_per_identity} source images: {insufficient_identities}"
        )

    rng = random.Random(protocol.selection_seed)
    groups: list[dict[str, Any]] = []
    for identity in sorted(images_by_identity):
        images = images_by_identity[identity][:]
        rng.shuffle(images)
        selected_images = images[:required_images_per_identity]
        for octet_index in range(protocol.within_identity_octets_per_identity):
            start = octet_index * protocol.group_size
            group_images = selected_images[start : start + protocol.group_size]
            rng.shuffle(group_images)
            groups.append(
                {
                    "identity": identity,
                    "octet_index_zero_based": octet_index,
                    "images": group_images,
                }
            )
    rng.shuffle(groups)

    expected_group_count = (
        len(images_by_identity) * protocol.within_identity_octets_per_identity
    )
    selected_images = [image for group in groups for image in group["images"]]
    if len(groups) != expected_group_count:
        raise AssertionError("Unexpected within-identity group count.")
    if any(len(group["images"]) != protocol.group_size for group in groups):
        raise AssertionError("Every within-identity group must have the configured size.")
    if any(
        len({extract_identity(image) for image in group["images"]}) != 1
        for group in groups
    ):
        raise AssertionError("Every within-identity group must contain one identity.")
    if len(selected_images) != len(set(selected_images)):
        raise AssertionError("A within-identity image was assigned more than once.")
    if Counter(group["identity"] for group in groups) != Counter(
        {
            identity: protocol.within_identity_octets_per_identity
            for identity in images_by_identity
        }
    ):
        raise AssertionError("Every identity must have the configured octet count.")
    return groups


def build_plan(state_file: Path, *, protocol: FollowupGroupProtocol) -> dict[str, Any]:
    source_groups = load_source_groups(state_file, protocol=protocol)
    rng = random.Random(protocol.selection_seed)
    retests = []
    for source_index in protocol.test_retest_group_indices:
        images = source_groups[source_index][:]
        rng.shuffle(images)
        retests.append({"source_group_index": source_index, "images": images})
    anchor_groups = build_anchor_groups(source_groups, protocol=protocol)
    return {
        "protocol": "test_retest_and_overlap_anchor_v1",
        "seed": protocol.selection_seed,
        "test_retest": retests,
        "anchors": list(protocol.anchors),
        "anchor_contexts_per_image": protocol.anchor_contexts,
        "anchor_groups": anchor_groups,
        "checks": {
            "test_retest_groups": len(retests),
            "anchor_groups": len(anchor_groups),
            "distinct_anchor_images": len(protocol.anchors),
            "unique_filler_images": len({
                image
                for group in anchor_groups
                for image in group["images"]
                if image != group["anchor"]
            }),
        },
    }


def build_annotation_state(
    source_payload: dict[str, Any],
    plan: dict[str, Any],
    task: str,
    *,
    source_state_file: Path,
    protocol: FollowupGroupProtocol,
) -> dict[str, Any]:
    if task == "test_retest":
        groups = [entry["images"] for entry in plan["test_retest"]]
        metadata = {
            "experiment": "test_retest",
            "description": (
                "Повторное ранжирование "
                f"{len(protocol.test_retest_group_indices)} исходных групп "
                "для оценки intra-annotator reliability."
            ),
            "source_state_file": source_state_file.name,
            "source_group_indices_zero_based": list(protocol.test_retest_group_indices),
            "allow_image_subset": True,
            "allow_image_reuse": False,
            "exclude_from_model_training": True,
            "selection_seed": protocol.selection_seed,
        }
    elif task == "overlap_anchor":
        groups = [entry["images"] for entry in plan["anchor_groups"]]
        metadata = {
            "experiment": "overlap_anchor",
            "description": (
                f"Контекстный тест: {len(protocol.anchors)} anchor-изображений "
                f"в {protocol.anchor_contexts} разных группах каждое."
            ),
            "source_state_file": source_state_file.name,
            "anchors": list(protocol.anchors),
            "contexts_per_anchor": protocol.anchor_contexts,
            "anchor_by_group_zero_based": {
                str(index): entry["anchor"]
                for index, entry in enumerate(plan["anchor_groups"])
            },
            "allow_image_subset": True,
            "allow_image_reuse": True,
            "exclude_from_model_training": True,
            "selection_seed": protocol.selection_seed,
        }
    else:
        raise ValueError(f"Unknown annotation task: {task}")

    return {
        "version": source_payload["version"],
        "image_root": source_payload["image_root"],
        "groups": groups,
        "group_states": [{"unranked": group[:], "ranked": []} for group in groups],
        "cycle_group_counts": [len(groups)],
        "current_group_index": 0,
        "metadata": metadata,
    }


def build_within_identity_state(
    source_payload: dict[str, Any],
    source_groups: list[list[str]],
    *,
    source_state_file: Path,
    protocol: FollowupGroupProtocol,
) -> dict[str, Any]:
    group_entries = build_within_identity_groups(source_groups, protocol=protocol)
    groups = [entry["images"] for entry in group_entries]
    identities = sorted({entry["identity"] for entry in group_entries})
    selected_images = [image for group in groups for image in group]
    expected_pair_count = (
        len(groups) * protocol.group_size * (protocol.group_size - 1) // 2
    )
    expected_group_count = (
        protocol.expected_identity_count * protocol.within_identity_octets_per_identity
    )
    expected_image_count = expected_group_count * protocol.group_size

    if len(identities) != protocol.expected_identity_count:
        raise AssertionError("Within-identity batch has an unexpected identity count.")
    if len(groups) != expected_group_count or len(selected_images) != expected_image_count:
        raise AssertionError("Within-identity batch dimensions do not match the protocol.")

    return {
        "version": source_payload["version"],
        "image_root": source_payload["image_root"],
        "groups": groups,
        "group_states": [{"unranked": group[:], "ranked": []} for group in groups],
        "cycle_group_counts": [len(groups)],
        "current_group_index": 0,
        "metadata": {
            "experiment": "within_identity",
            "description": (
                "Чистый visual-тест: "
                f"{protocol.within_identity_octets_per_identity} непересекающихся "
                f"{protocol.within_identity_group_label_plural} для каждой из "
                f"{protocol.expected_identity_count} identity."
            ),
            "source_state_file": source_state_file.name,
            "source_group_count": protocol.source_group_count,
            "identities": len(identities),
            "octets_per_identity": protocol.within_identity_octets_per_identity,
            "selected_images": len(selected_images),
            "same_identity_pairs": expected_pair_count,
            "identity_by_group_zero_based": {
                str(index): entry["identity"] for index, entry in enumerate(group_entries)
            },
            "allow_image_subset": True,
            "allow_image_reuse": False,
            "exclude_from_model_training": True,
            "selection_seed": protocol.selection_seed,
        },
    }


def build_anchor_extension_groups(
    source_groups: list[list[str]],
    existing_state: dict[str, Any],
    *,
    protocol: FollowupGroupProtocol,
) -> list[dict[str, Any]]:
    """Build one additional context without changing the completed anchor batch."""
    metadata = existing_state.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("experiment") != "overlap_anchor":
        raise ValueError("The target state must be an overlap_anchor experiment.")
    anchor_by_group = metadata.get("anchor_by_group_zero_based")
    if not isinstance(anchor_by_group, dict):
        raise ValueError("The target state has no anchor_by_group_zero_based metadata.")

    existing_groups = existing_state.get("groups")
    expected_existing_groups = len(protocol.anchors) * protocol.anchor_contexts
    if not isinstance(existing_groups, list) or len(existing_groups) != expected_existing_groups:
        raise ValueError("The extension has an unexpected existing anchor-group count.")
    existing_anchors = [
        anchor_by_group.get(str(index)) for index in range(len(existing_groups))
    ]
    if Counter(existing_anchors) != Counter(
        {anchor: protocol.anchor_contexts for anchor in protocol.anchors}
    ):
        raise ValueError("The existing batch has an unexpected context count per anchor.")

    all_images = [image for group in source_groups for image in group]
    used_fillers: set[str] = set()
    for index, (group, anchor) in enumerate(zip(existing_groups, existing_anchors, strict=True)):
        if (
            not isinstance(group, list)
            or anchor not in protocol.anchors
            or group.count(anchor) != 1
        ):
            raise ValueError(f"Invalid anchor assignment in existing group {index}.")
        if (
            len(group) != protocol.group_size
            or len(set(group)) != protocol.group_size
            or set(group).intersection(protocol.anchors) != {anchor}
        ):
            raise ValueError(f"Existing group {index} is not a valid anchor group.")
        fillers = set(group) - {anchor}
        if used_fillers.intersection(fillers):
            raise ValueError("Existing filler images must not repeat.")
        used_fillers.update(fillers)

    remaining_images = set(all_images) - set(protocol.anchors) - used_fillers
    expected_remaining_images = (
        protocol.source_group_count * protocol.group_size
        - len(protocol.anchors)
        - expected_existing_groups * (protocol.group_size - 1)
    )
    if len(remaining_images) != expected_remaining_images:
        raise ValueError("Unexpected count of unused fillers after the anchor batch.")
    rng = random.Random(protocol.extension_seed)
    images_by_identity: dict[str, list[str]] = defaultdict(list)
    for image in sorted(remaining_images):
        images_by_identity[extract_identity(image)].append(image)
    for images in images_by_identity.values():
        rng.shuffle(images)
    remaining = Counter({identity: len(images) for identity, images in images_by_identity.items()})

    schedule = list(protocol.anchors)
    rng.shuffle(schedule)
    extension_groups: list[dict[str, Any]] = []
    for anchor in schedule:
        chosen_identities: set[str] = {extract_identity(anchor)}
        fillers: list[str] = []
        for _ in range(protocol.group_size - 1):
            candidates = [
                identity
                for identity, count in remaining.items()
                if count and identity not in chosen_identities
            ]
            if not candidates:
                raise RuntimeError("Could not allocate enough distinct filler identities.")
            highest_count = max(remaining[identity] for identity in candidates)
            tied = sorted(identity for identity in candidates if remaining[identity] == highest_count)
            identity = rng.choice(tied)
            fillers.append(images_by_identity[identity].pop())
            remaining[identity] -= 1
            chosen_identities.add(identity)
        images = [anchor, *fillers]
        rng.shuffle(images)
        extension_groups.append({"anchor": anchor, "images": images})

    if any(
        len({extract_identity(image) for image in group["images"]})
        != protocol.group_size
        for group in extension_groups
    ):
        raise AssertionError("Extension groups must contain distinct identities.")
    return extension_groups


def extend_completed_anchor_state(
    source_state_file: Path,
    anchor_state_file: Path,
    *,
    protocol: FollowupGroupProtocol,
) -> None:
    source_groups = load_source_groups(source_state_file, protocol=protocol)
    state = json.loads(anchor_state_file.read_text(encoding="utf-8"))
    extension_groups = build_anchor_extension_groups(
        source_groups, state, protocol=protocol
    )
    first_new_index = len(state["groups"])
    state["groups"].extend(entry["images"] for entry in extension_groups)
    state["group_states"].extend(
        {"unranked": entry["images"][:], "ranked": []} for entry in extension_groups
    )
    state["cycle_group_counts"].append(len(extension_groups))
    state["current_group_index"] = first_new_index
    metadata = state["metadata"]
    metadata["contexts_per_anchor"] = protocol.anchor_contexts + 1
    metadata["anchor_by_group_zero_based"].update(
        {
            str(first_new_index + index): entry["anchor"]
            for index, entry in enumerate(extension_groups)
        }
    )
    metadata["description"] = "Контекстный тест с дополнительным контекстом для каждого anchor-изображения."
    metadata["extension"] = {
        "added_contexts_per_anchor": 1,
        "algorithm": "unused fillers; highest remaining identity count; deterministic tie-break",
        "seed": protocol.extension_seed,
    }
    anchor_state_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_annotation_states(
    source_state_file: Path,
    output_dir: Path,
    *,
    protocol: FollowupGroupProtocol,
) -> tuple[Path, Path]:
    source_payload = json.loads(source_state_file.read_text(encoding="utf-8"))
    plan = build_plan(source_state_file, protocol=protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_retest_file = output_dir / "ranking_state_test_retest.json"
    overlap_anchor_file = output_dir / "ranking_state_overlap_anchor.json"
    test_retest_file.write_text(
        json.dumps(
            build_annotation_state(
                source_payload,
                plan,
                "test_retest",
                source_state_file=source_state_file,
                protocol=protocol,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    overlap_anchor_file.write_text(
        json.dumps(
            build_annotation_state(
                source_payload,
                plan,
                "overlap_anchor",
                source_state_file=source_state_file,
                protocol=protocol,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return test_retest_file, overlap_anchor_file


def write_within_identity_state(
    source_state_file: Path,
    output_file: Path,
    *,
    protocol: FollowupGroupProtocol,
) -> Path:
    """Write the configured within-identity batch without overwriting a state."""
    if output_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing within-identity state file: {output_file}"
        )

    source_payload = json.loads(source_state_file.read_text(encoding="utf-8"))
    state = build_within_identity_state(
        source_payload,
        load_source_groups(source_state_file, protocol=protocol),
        source_state_file=source_state_file,
        protocol=protocol,
    )
    output_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def build_within_identity_retest_state(
    source_state_file: Path,
    *,
    protocol: FollowupGroupProtocol,
) -> dict[str, Any]:
    """Select configured completed W-groups per identity without copying ranks."""
    source = json.loads(source_state_file.read_text(encoding="utf-8"))
    groups = source.get("groups")
    states = source.get("group_states")
    expected_group_count = (
        protocol.expected_identity_count * protocol.within_identity_octets_per_identity
    )
    if (
        not isinstance(groups, list)
        or not isinstance(states, list)
        or len(groups) != expected_group_count
    ):
        raise ValueError("Within-identity state has an unexpected group count.")
    if len(states) != len(groups):
        raise ValueError("Within-identity group/state counts differ.")

    source_groups_by_identity: dict[str, list[int]] = defaultdict(list)
    for group_index, (group, state) in enumerate(zip(groups, states, strict=True)):
        if not isinstance(group, list) or len(group) != protocol.group_size:
            raise ValueError(f"Source W-group {group_index} has an unexpected size.")
        identities = {extract_identity(image) for image in group}
        if len(identities) != 1:
            raise ValueError(f"Source W-group {group_index} mixes identities.")
        if state.get("unranked") != [] or len(state.get("ranked", [])) != len(group):
            raise ValueError(f"Source W-group {group_index} is not fully ranked.")
        source_groups_by_identity[next(iter(identities))].append(group_index)
    if len(source_groups_by_identity) != protocol.expected_identity_count or any(
        len(indices) != protocol.within_identity_octets_per_identity
        for indices in source_groups_by_identity.values()
    ):
        raise ValueError("Within-identity source state has an unexpected identity layout.")

    rng = random.Random(protocol.within_identity_retest_seed)
    entries: list[dict[str, Any]] = []
    for identity in sorted(source_groups_by_identity):
        selected = sorted(
            rng.sample(
                source_groups_by_identity[identity],
                protocol.within_identity_retest_octets_per_identity,
            )
        )
        for source_group_index in selected:
            images = list(groups[source_group_index])
            rng.shuffle(images)
            entries.append(
                {
                    "identity": identity,
                    "source_group_index": source_group_index,
                    "images": images,
                }
            )
    rng.shuffle(entries)

    retest_groups = [entry["images"] for entry in entries]
    selected_images = [image for group in retest_groups for image in group]
    if (
        len(retest_groups)
        != protocol.expected_identity_count
        * protocol.within_identity_retest_octets_per_identity
        or len(selected_images)
        != protocol.expected_identity_count
        * protocol.within_identity_retest_octets_per_identity
        * protocol.group_size
        or len(set(selected_images)) != len(selected_images)
        or Counter(entry["identity"] for entry in entries)
        != Counter(
            {
                identity: protocol.within_identity_retest_octets_per_identity
                for identity in source_groups_by_identity
            }
        )
    ):
        raise AssertionError("Within-identity retest dimensions are invalid.")

    return {
        "version": source["version"],
        "image_root": source["image_root"],
        "groups": retest_groups,
        "group_states": [
            {"unranked": list(group), "ranked": []} for group in retest_groups
        ],
        "cycle_group_counts": [len(retest_groups)],
        "current_group_index": 0,
        "metadata": {
            "experiment": "within_identity_test_retest",
            "description": (
                "Blind forced-choice retest of two original W-octets per identity."
            ),
            "source_state_file": source_state_file.name,
            "source_state_sha256": sha256_file(source_state_file),
            "identities": protocol.expected_identity_count,
            "octets_per_identity": protocol.within_identity_retest_octets_per_identity,
            "selected_groups": len(retest_groups),
            "selected_images": len(selected_images),
            "same_identity_pairs": (
                len(retest_groups) * protocol.group_size * (protocol.group_size - 1) // 2
            ),
            "source_group_index_by_retest_group_zero_based": {
                str(index): entry["source_group_index"]
                for index, entry in enumerate(entries)
            },
            "identity_by_group_zero_based": {
                str(index): entry["identity"] for index, entry in enumerate(entries)
            },
            "old_ranks_copied": False,
            "old_ranks_visible_in_state": False,
            "forced_choice": True,
            "ties_allowed": False,
            "allow_image_subset": True,
            "allow_image_reuse": False,
            "exclude_from_model_training": True,
            "selection_and_display_seed": protocol.within_identity_retest_seed,
        },
    }


def write_within_identity_retest_state(
    source_state_file: Path,
    output_file: Path,
    *,
    protocol: FollowupGroupProtocol,
) -> Path:
    """Write a blind W-retest state without overwriting an existing file."""
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite W-retest state file: {output_file}")
    first = build_within_identity_retest_state(source_state_file, protocol=protocol)
    second = build_within_identity_retest_state(source_state_file, protocol=protocol)
    if first != second:
        raise RuntimeError("Within-identity retest construction is not deterministic.")
    atomic_json(output_file, first)
    return output_file
