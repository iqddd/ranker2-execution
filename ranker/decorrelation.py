from __future__ import annotations

import math

import numpy as np


def safe_std(values: np.ndarray) -> float:
    sigma = float(values.std())
    return 1.0 if math.isclose(sigma, 0.0) else sigma


def normalize_from_train(
    train_values: np.ndarray,
    eval_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mu = float(train_values.mean())
    sigma = safe_std(train_values)
    train_norm = (train_values - mu) / sigma
    eval_norm = (eval_values - mu) / sigma
    return train_norm, eval_norm, {"mu": mu, "sigma": sigma}


def project_out_linear_component(
    train_anchor: np.ndarray,
    train_values: np.ndarray,
    eval_anchor: np.ndarray,
    eval_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean_anchor = float(train_anchor.mean())
    mean_values = float(train_values.mean())
    centered_anchor = train_anchor - mean_anchor
    centered_values = train_values - mean_values
    var_anchor = float(np.mean(centered_anchor * centered_anchor))
    if math.isclose(var_anchor, 0.0):
        slope = 0.0
    else:
        cov = float(np.mean(centered_values * centered_anchor))
        slope = cov / var_anchor
    intercept = mean_values - slope * mean_anchor

    train_residual = train_values - slope * train_anchor - intercept
    eval_residual = eval_values - slope * eval_anchor - intercept
    return train_residual, eval_residual, {"q": slope, "m": intercept}
