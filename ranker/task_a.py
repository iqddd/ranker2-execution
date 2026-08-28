from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge

from .data import RankedImage, extract_identity
from .embeddings import encode_image_paths
from .metrics import choose_top_name, evaluate_group_scores


@dataclass(frozen=True)
class TrainedTaskAModel:
    identity_priors: dict[str, float]
    identity_counts: dict[str, int]
    ridge_weights: np.ndarray
    ridge_intercept: float
    tau: float


@dataclass(frozen=True)
class FinalAProductionModel:
    encoder_name: str
    identity_list: list[str]
    identity_priors: dict[str, float]
    identity_counts: dict[str, int]
    mu_b: float
    sigma_b: float
    mu_e: np.ndarray
    sigma_e: np.ndarray
    ridge_weights: np.ndarray
    ridge_intercept: float
    q: float
    m: float
    mu_u: float
    sigma_u: float
    mu_u_id_shrunk: dict[str, float]
    mu_p: float
    sigma_p: float
    lambda_a: float
    sorted_a_train_scores: np.ndarray
    sorted_b_train_scores: np.ndarray


def fit_ridge_residual_full(
    items: list[RankedImage],
    embeddings: dict[str, np.ndarray],
    priors: dict[str, float],
    alpha: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    x = np.stack([embeddings[item.image_name] for item in items], axis=0)
    y = np.asarray(
        [item.z - priors.get(item.identity, 0.0) for item in items],
        dtype=np.float64,
    )
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x, y)
    predictions = model.predict(x).astype(np.float64)
    return model.coef_.astype(np.float64), float(model.intercept_), predictions


def fit_ridge_residual_full_exact(
    items: list[RankedImage],
    embeddings: dict[str, np.ndarray],
    priors: dict[str, float],
    alpha: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    x = np.stack([embeddings[item.image_name] for item in items], axis=0).astype(np.float64)
    y = np.asarray(
        [item.z - priors.get(item.identity, 0.0) for item in items],
        dtype=np.float64,
    )
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean

    gram = (x_centered.T @ x_centered) / len(items)
    rhs = (x_centered.T @ y_centered) / len(items)
    weights = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        rhs,
    )
    intercept = y_mean - float(x_mean @ weights)
    predictions = x @ weights + intercept
    return weights.astype(np.float64), float(intercept), predictions.astype(np.float64)


def build_order_scores(
    items: list[RankedImage],
    priors: dict[str, float],
    visual_scores: dict[str, float],
    lambda_order: float,
) -> dict[str, float]:
    return {
        item.image_name: float(
            priors.get(item.identity, 0.0) + lambda_order * visual_scores[item.image_name]
        )
        for item in items
    }


def choose_identity_top(
    group: list[RankedImage],
    identity_scores: dict[str, float],
    order_scores: dict[str, float],
) -> str:
    return choose_top_name(group, identity_scores, order_scores)


def choose_order_top(
    group: list[RankedImage],
    order_scores: dict[str, float],
    identity_scores: dict[str, float],
) -> str:
    return choose_top_name(group, order_scores, identity_scores)


def build_gate_ranking_scores(
    groups: list[list[RankedImage]],
    identity_scores: dict[str, float],
    order_scores: dict[str, float],
    tau: float,
) -> tuple[dict[str, float], float, float, float, float]:
    final_scores: dict[str, float] = {}
    gate_top_correct = 0
    identity_top_correct = 0
    order_top_correct = 0
    replaced = 0

    for group in groups:
        top_id = choose_identity_top(group, identity_scores, order_scores)
        top_ord = choose_order_top(group, order_scores, identity_scores)
        margin = math.inf if top_ord == top_id else order_scores[top_ord] - order_scores[top_id]
        top_gate = top_ord if margin >= tau else top_id
        if top_gate != top_id:
            replaced += 1

        rest = [item for item in group if item.image_name != top_gate]
        rest_sorted = sorted(
            rest,
            key=lambda item: (
                order_scores[item.image_name],
                identity_scores[item.image_name],
                item.image_name,
            ),
        )
        final_ranking = rest_sorted + [
            next(item for item in group if item.image_name == top_gate)
        ]
        for rank_position, item in enumerate(final_ranking):
            final_scores[item.image_name] = float(rank_position)

        true_top = max(group, key=lambda item: item.rank).image_name
        gate_top_correct += int(top_gate == true_top)
        identity_top_correct += int(top_id == true_top)
        order_top_correct += int(top_ord == true_top)

    group_count = len(groups)
    return (
        final_scores,
        gate_top_correct / group_count,
        identity_top_correct / group_count,
        order_top_correct / group_count,
        replaced / group_count,
    )


def build_gate_final_scores(
    groups: list[list[RankedImage]],
    identity_scores: dict[str, float],
    order_scores: dict[str, float],
    tau: float,
) -> tuple[dict[str, float], float, float]:
    final_scores, gate_top1, identity_top1, _order_top1, _replace_rate = build_gate_ranking_scores(
        groups,
        identity_scores,
        order_scores,
        tau,
    )
    return final_scores, identity_top1, gate_top1


def select_tau_full(
    groups: list[list[RankedImage]],
    identity_scores: dict[str, float],
    order_scores: dict[str, float],
    tau_grid: Sequence[float],
) -> tuple[float, dict[str, Any]]:
    tau_values = tuple(float(value) for value in tau_grid)
    if not tau_values:
        raise ValueError("tau_grid must not be empty.")
    identity_only_scores, top1_identity_train, _ = build_gate_final_scores(
        groups,
        identity_scores,
        order_scores,
        tau=math.inf,
    )
    identity_pair_acc, identity_weighted_pair_acc, _ = evaluate_group_scores(
        groups, identity_only_scores
    )

    best_key: tuple[float, float, float] | None = None
    best_tau: float | None = None
    tau_rows: list[dict[str, float | bool]] = []

    for tau in tau_values:
        final_scores, _identity_top1_unused, gate_top1 = build_gate_final_scores(
            groups,
            identity_scores,
            order_scores,
            tau=tau,
        )
        pair_acc, weighted_pair_acc, _ = evaluate_group_scores(groups, final_scores)
        admissible = gate_top1 >= top1_identity_train
        tau_rows.append(
            {
                "tau": tau,
                "pair_acc_train": pair_acc,
                "weighted_pair_acc_train": weighted_pair_acc,
                "top1_hard_train": gate_top1,
                "admissible": admissible,
            }
        )
        if not admissible:
            continue

        candidate_key = (-weighted_pair_acc, -gate_top1, tau)
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_tau = tau

    if best_tau is None:
        raise RuntimeError("No admissible tau found for the full-data task A model.")

    diagnostics = {
        "identity_pair_acc_train": identity_pair_acc,
        "identity_weighted_pair_acc_train": identity_weighted_pair_acc,
        "top1_identity_train": top1_identity_train,
        "tau_grid": tau_rows,
    }
    return best_tau, diagnostics


def save_identity_priors_csv(
    output_dir: Path,
    priors: dict[str, float],
    counts: dict[str, int],
) -> Path:
    path = output_dir / "identity_priors.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["identity", "b_id", "n_id"])
        writer.writeheader()
        for identity in sorted(priors):
            writer.writerow(
                {
                    "identity": identity,
                    "b_id": priors[identity],
                    "n_id": counts[identity],
                }
            )
    return path


def save_mu_u_id_shrunk_csv(
    output_dir: Path,
    mu_u_id_shrunk: dict[str, float],
    counts: dict[str, int],
) -> Path:
    path = output_dir / "mu_u_id_shrunk.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["identity", "mu_u_id_shrunk", "n_id"],
        )
        writer.writeheader()
        for identity in sorted(mu_u_id_shrunk):
            writer.writerow(
                {
                    "identity": identity,
                    "mu_u_id_shrunk": mu_u_id_shrunk[identity],
                    "n_id": counts[identity],
                }
            )
    return path


def save_model_bundle(
    output_dir: Path,
    model: TrainedTaskAModel,
    summary_payload: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    priors_csv = save_identity_priors_csv(output_dir, model.identity_priors, model.identity_counts)
    weights_path = output_dir / "task_a_model_weights.npz"
    np.savez(
        weights_path,
        ridge_weights=model.ridge_weights,
        ridge_intercept=np.asarray([model.ridge_intercept], dtype=np.float64),
        tau=np.asarray([model.tau], dtype=np.float64),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return priors_csv, weights_path, summary_path


def save_final_a_bundle(
    output_dir: Path,
    model: FinalAProductionModel,
    summary_payload: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    priors_csv = save_identity_priors_csv(
        output_dir,
        model.identity_priors,
        model.identity_counts,
    )
    mu_u_id_csv = save_mu_u_id_shrunk_csv(
        output_dir,
        model.mu_u_id_shrunk,
        model.identity_counts,
    )
    bundle_path = output_dir / "final_A_production_bundle.npz"
    np.savez(
        bundle_path,
        encoder_name=np.asarray([model.encoder_name]),
        identity_list=np.asarray(model.identity_list),
        mu_b=np.asarray([model.mu_b], dtype=np.float64),
        sigma_b=np.asarray([model.sigma_b], dtype=np.float64),
        mu_e=np.asarray(model.mu_e, dtype=np.float64),
        sigma_e=np.asarray(model.sigma_e, dtype=np.float64),
        ridge_weights=np.asarray(model.ridge_weights, dtype=np.float64),
        ridge_intercept=np.asarray([model.ridge_intercept], dtype=np.float64),
        q=np.asarray([model.q], dtype=np.float64),
        m=np.asarray([model.m], dtype=np.float64),
        mu_u=np.asarray([model.mu_u], dtype=np.float64),
        sigma_u=np.asarray([model.sigma_u], dtype=np.float64),
        mu_p=np.asarray([model.mu_p], dtype=np.float64),
        sigma_p=np.asarray([model.sigma_p], dtype=np.float64),
        lambda_a=np.asarray([model.lambda_a], dtype=np.float64),
        sorted_a_train_scores=np.asarray(model.sorted_a_train_scores, dtype=np.float64),
        sorted_b_train_scores=np.asarray(model.sorted_b_train_scores, dtype=np.float64),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return priors_csv, mu_u_id_csv, bundle_path, summary_path


def load_model_bundle(model_dir: Path) -> TrainedTaskAModel:
    priors_path = model_dir / "identity_priors.csv"
    weights_path = model_dir / "task_a_model_weights.npz"
    if not priors_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing model bundle in {model_dir}. Expected identity_priors.csv and task_a_model_weights.npz."
        )

    priors: dict[str, float] = {}
    counts: dict[str, int] = {}
    with priors_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            priors[row["identity"]] = float(row["b_id"])
            counts[row["identity"]] = int(row["n_id"])

    payload = np.load(weights_path)
    return TrainedTaskAModel(
        identity_priors=priors,
        identity_counts=counts,
        ridge_weights=np.asarray(payload["ridge_weights"], dtype=np.float64),
        ridge_intercept=float(payload["ridge_intercept"][0]),
        tau=float(payload["tau"][0]),
    )


def load_final_a_identity_priors(priors_path: Path) -> dict[str, float]:
    """Load the frozen final-A identity prior table."""
    with priors_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["identity"]: float(row["b_id"]) for row in csv.DictReader(handle)}


def load_final_a_purged_means(purged_means_path: Path) -> dict[str, float]:
    """Load the frozen final-A per-identity purged residual means."""
    with purged_means_path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["identity"]: float(row["mu_u_id_shrunk"])
            for row in csv.DictReader(handle)
        }


def score_final_a_production(
    items: list[RankedImage],
    embeddings: Mapping[str, np.ndarray],
    *,
    bundle_path: Path,
    priors_path: Path,
    purged_means_path: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return frozen identity and production-A scores with no fitted parameters."""
    bundle = np.load(bundle_path, allow_pickle=False)
    priors = load_final_a_identity_priors(priors_path)
    purged_means = load_final_a_purged_means(purged_means_path)
    names = [item.image_name for item in items]
    identities = [item.identity for item in items]
    x = np.stack([np.asarray(embeddings[name], dtype=np.float64) for name in names])
    b_values = np.asarray([priors[identity] for identity in identities], dtype=np.float64)
    b_norm = (b_values - float(bundle["mu_b"][0])) / float(bundle["sigma_b"][0])
    x_a = (x - bundle["mu_e"]) / bundle["sigma_e"]
    visual = x_a @ bundle["ridge_weights"] + float(bundle["ridge_intercept"][0])
    u = visual - float(bundle["q"][0]) * b_values - float(bundle["m"][0])
    u_norm = (u - float(bundle["mu_u"][0])) / float(bundle["sigma_u"][0])
    purged = u_norm - np.asarray([purged_means[identity] for identity in identities])
    p = (purged - float(bundle["mu_p"][0])) / float(bundle["sigma_p"][0])
    a_values = b_norm + float(bundle["lambda_a"][0]) * p
    return (
        {name: float(value) for name, value in zip(names, b_values, strict=True)},
        {name: float(value) for name, value in zip(names, a_values, strict=True)},
    )


def predict_group(
    model: TrainedTaskAModel,
    image_paths: list[Path],
    lambda_order: float,
) -> dict[str, Any]:
    if len(image_paths) != 8:
        raise ValueError("Task A prediction expects exactly 8 images in a group.")

    embeddings = encode_image_paths(image_paths)
    items: list[dict[str, Any]] = []
    for image_path in image_paths:
        identity = extract_identity(image_path.name)
        if identity not in model.identity_priors:
            raise ValueError(
                f"Unknown identity '{identity}' in {image_path.name}. Task A expects one of: "
                f"{', '.join(sorted(model.identity_priors))}"
            )
        embedding = embeddings[str(image_path)]
        visual_score = float(np.dot(model.ridge_weights, embedding) + model.ridge_intercept)
        identity_score = float(model.identity_priors[identity])
        order_score = float(identity_score + lambda_order * visual_score)
        items.append(
            {
                "path": str(image_path),
                "filename": image_path.name,
                "identity": identity,
                "b": identity_score,
                "v": visual_score,
                "score_order": order_score,
            }
        )

    identity_scores = {item["filename"]: item["b"] for item in items}
    order_scores = {item["filename"]: item["score_order"] for item in items}
    ranked_group = [
        RankedImage(
            group_index=0,
            image_name=item["filename"],
            identity=item["identity"],
            rank=0,
            z=0.0,
        )
        for item in items
    ]

    top_id = choose_identity_top(ranked_group, identity_scores, order_scores)
    top_ord = choose_order_top(ranked_group, order_scores, identity_scores)
    margin = math.inf if top_id == top_ord else order_scores[top_ord] - order_scores[top_id]
    top = top_ord if margin >= model.tau else top_id

    rest = [item for item in items if item["filename"] != top]
    rest_sorted = sorted(
        rest,
        key=lambda item: (item["score_order"], item["b"], item["filename"]),
    )
    top_item = next(item for item in items if item["filename"] == top)
    predicted_ranking = rest_sorted + [top_item]

    return {
        "tau_A": model.tau,
        "top_id": top_id,
        "top_ord": top_ord,
        "margin": margin,
        "selected_top": top,
        "predicted_ranking": predicted_ranking,
    }
