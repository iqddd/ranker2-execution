from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.linear_model import Ridge

from .data import RankedImage


def write_contact_sheet(
    image_dir: Path, names: list[str], destination: Path, *, columns: int = 5,
    rows: int = 5, box_size: int = 256, label_height: int = 24,
) -> Path:
    font = ImageFont.load_default()
    page = Image.new("RGB", (columns * box_size, rows * (box_size + label_height)), (128, 128, 128))
    draw = ImageDraw.Draw(page)
    for slot, image_id in enumerate(names[:columns * rows]):
        row, column = divmod(slot, columns); x, y = column * box_size, row * (box_size + label_height)
        with Image.open(image_dir / image_id) as source:
            thumb = ImageOps.contain(source.convert("RGB"), (box_size, box_size), method=Image.Resampling.LANCZOS)
        page.paste(thumb, (x + (box_size - thumb.width) // 2, y + (box_size - thumb.height) // 2))
        text_box = draw.textbbox((0, 0), image_id, font=font)
        draw.text((x + (box_size - (text_box[2] - text_box[0])) // 2, y + box_size + 5), image_id, fill=(20, 20, 20), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True); page.save(destination)
    return destination


def write_contact_sheet_pages(
    image_dir: Path, names: list[str], output_dir: Path, *, columns: int = 5,
    rows: int = 5, box_size: int = 256, label_height: int = 24, filename_prefix: str = "contact_sheet",
) -> list[Path]:
    return [write_contact_sheet(image_dir, names[start:start + columns * rows], output_dir / f"{filename_prefix}_{number:02d}.png", columns=columns, rows=rows, box_size=box_size, label_height=label_height) for number, start in enumerate(range(0, len(names), columns * rows), 1)]


@dataclass(frozen=True)
class ResidualPredictions:
    train_visual: np.ndarray
    val_visual: np.ndarray


def build_visual_baseline_parser(
    *,
    description: str,
    default_output_dir: Path,
    default_folds: int,
    default_lambda_shrinkage: float,
    default_alpha: float,
    default_seed: int,
    default_batch_size: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("ranking_state.json"),
        help="Path to ranking_state.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("images"),
        help="Directory containing image files referenced by ranking_state.json",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=default_folds,
        help="Number of group-CV folds",
    )
    parser.add_argument(
        "--lambda-shrinkage",
        type=float,
        default=default_lambda_shrinkage,
        help="Shrinkage coefficient lambda for identity priors",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=default_alpha,
        help="Ridge alpha",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="Random seed for shuffled fold assignment",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts") / "visual_residual_baseline" / "dinov2_vitl14_embeddings.pt",
        help="Path to cached image embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where JSON outputs are written",
    )
    return parser


def build_identity_scores(
    items: list[RankedImage],
    priors: dict[str, float],
) -> dict[str, float]:
    return {
        item.image_name: float(priors.get(item.identity, 0.0))
        for item in items
    }


def build_blended_scores(
    items: list[RankedImage],
    priors: dict[str, float],
    visual_predictions: np.ndarray,
    blend_lambda: float = 1.0,
) -> dict[str, float]:
    return {
        item.image_name: float(
            priors.get(item.identity, 0.0) + blend_lambda * visual_score
        )
        for item, visual_score in zip(items, visual_predictions, strict=True)
    }


def fit_ridge_residuals(
    *,
    train_items: list[RankedImage],
    val_items: list[RankedImage],
    embeddings: dict[str, np.ndarray],
    priors: dict[str, float],
    alpha: float,
) -> ResidualPredictions:
    x_train = np.stack([embeddings[item.image_name] for item in train_items], axis=0)
    y_train = np.asarray(
        [item.z - priors.get(item.identity, 0.0) for item in train_items],
        dtype=np.float32,
    )
    x_val = np.stack([embeddings[item.image_name] for item in val_items], axis=0)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train, y_train)

    return ResidualPredictions(
        train_visual=model.predict(x_train).astype(np.float32),
        val_visual=model.predict(x_val).astype(np.float32),
    )
