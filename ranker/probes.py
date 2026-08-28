"""Contrastive logistic probe utilities."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def build_same_identity_contrastive_pairs(identities, labels, features, allowed_identities):
    differences=[]; weights=[]
    for identity in allowed_identities:
        positive=np.flatnonzero((identities == identity) & (labels == 1)); negative=np.flatnonzero((identities == identity) & (labels == -1))
        for first in positive:
            for second in negative: differences.append(features[first]-features[second]); weights.append(1/(len(allowed_identities)*len(positive)*len(negative)))
    return np.asarray(differences,float),np.asarray(weights,float)

def fit_rowspace_logistic_probe(differences, weights, ridge):
    if len(differences)==0: raise ValueError("no pairs")
    _, singular, vectors=np.linalg.svd(differences,full_matrices=False); rank=np.count_nonzero(singular > singular[0]*1e-12); vectors=vectors[:rank]; projected=differences@vectors.T
    def objective(alpha):
        logits=projected@alpha; return np.sum(weights*np.logaddexp(0,-logits))+ridge*alpha@alpha,projected.T@(-weights*expit(-logits))+2*ridge*alpha
    result=minimize(objective,np.zeros(rank),jac=True,method="L-BFGS-B",options={"maxiter":100000,"maxfun":250000,"maxcor":100,"ftol":1e-15,"gtol":1e-12,"maxls":200})
    return vectors.T@result.x,result

def choose_probe_lambda_loio(features, identities, labels, heldout_identity, *, identity_order, lambda_grid):
    """Select ridge by the original nested identity-LOIO logistic loss."""
    eligible = [
        identity for identity in identity_order
        if identity != heldout_identity
        and np.any((identities == identity) & (labels == 1))
        and np.any((identities == identity) & (labels == -1))
    ]
    candidates = []
    for ridge in lambda_grid:
        losses = []
        for validation_identity in eligible:
            train = (identities != heldout_identity) & (identities != validation_identity)
            mean = features[train].mean(axis=0)
            std = np.maximum(features[train].std(axis=0, ddof=0), 1e-8)
            standardized = (features - mean) / std
            train_identities = [identity for identity in eligible if identity != validation_identity]
            differences, weights = build_same_identity_contrastive_pairs(
                identities, labels, standardized, train_identities
            )
            vector, _ = fit_rowspace_logistic_probe(differences, weights, ridge)
            validation_differences, _ = build_same_identity_contrastive_pairs(
                identities, labels, standardized, [validation_identity]
            )
            losses.append(float(np.logaddexp(0.0, -validation_differences @ vector).mean()))
        candidates.append((float(np.mean(losses)), float(np.median(losses)), -ridge))
    return -min(candidates)[2]
