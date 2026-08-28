"""Identity-balanced ridge primitives for concept readout experiments."""
from __future__ import annotations
import numpy as np
from .embeddings import standardize_array_from_train

def fit_weighted_global_ridge(x, y, identities, ridge):
    weights = np.empty(len(y), dtype=np.float64); unique = np.unique(identities)
    for identity in unique:
        mask = identities == identity; weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    x_mean = np.average(x, axis=0, weights=weights); y_mean = float(np.average(y, weights=weights)); centered = x - x_mean
    beta = np.linalg.solve((centered * weights[:, None]).T @ centered + ridge * np.eye(x.shape[1]), (centered * weights[:, None]).T @ (y - y_mean))
    return float(y_mean - x_mean @ beta), beta

def build_within_identity_pair_arrays(x, y, identities, *, identity_order):
    dx=[]; dy=[]; pair_identity=[]; left=[]; right=[]
    for identity in identity_order:
        indices=np.flatnonzero(identities == identity)
        for position, first in enumerate(indices):
            for second in indices[position + 1:]: dx.append(x[second]-x[first]); dy.append(y[second]-y[first]); pair_identity.append(identity); left.append(first); right.append(second)
    return np.asarray(dx), np.asarray(dy), np.asarray(pair_identity), np.asarray(left), np.asarray(right)

def fit_weighted_within_ridge(dx, dy, pair_identity, ridge):
    weights=np.empty(len(dy),dtype=np.float64); unique=np.unique(pair_identity)
    for identity in unique:
        mask=pair_identity == identity; weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    return np.linalg.solve((dx * weights[:, None]).T @ dx + ridge * np.eye(dx.shape[1]), (dx * weights[:, None]).T @ dy)

def choose_global_ridge(x, y, identities, outer_train, *, identity_order, lambda_grid):
    results=[]
    for ridge in lambda_grid:
        mse=[]
        for held in identity_order:
            validation=outer_train & (identities == held)
            if not validation.any(): continue
            train=outer_train & ~validation; z_train=standardize_array_from_train(x[train],x[train]); intercept,beta=fit_weighted_global_ridge(z_train,y[train],identities[train],ridge); z_val=standardize_array_from_train(x[train],x[validation]); mse.append(float(np.mean((y[validation]-(intercept+z_val@beta))**2)))
        results.append((float(np.mean(mse)),float(np.median(mse)),-ridge))
    return -min(results)[2]

def choose_within_ridge(x, y, identities, outer_train, *, identity_order, lambda_grid):
    results=[]
    for ridge in lambda_grid:
        mse=[]
        for held in identity_order:
            validation=outer_train & (identities == held)
            if not validation.any(): continue
            train=outer_train & ~validation; z_train=standardize_array_from_train(x[train],x[train]); dx,dy,pair_id,*_=build_within_identity_pair_arrays(z_train,y[train],identities[train],identity_order=identity_order); gamma=fit_weighted_within_ridge(dx,dy,pair_id,ridge); z_val=standardize_array_from_train(x[train],x[validation]); dx,dy,*_=build_within_identity_pair_arrays(z_val,y[validation],identities[validation],identity_order=identity_order); mse.append(float(np.mean((dy-dx@gamma)**2)))
        results.append((float(np.mean(mse)),float(np.median(mse)),-ridge))
    return -min(results)[2]
