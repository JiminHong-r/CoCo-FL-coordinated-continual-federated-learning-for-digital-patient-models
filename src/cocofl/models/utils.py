"""
TFN Utility Functions
====================

Utility functions for representation analysis and evaluation,
adapted from the original TFN paper.

Functions:
- compute_mutual_information_appr: MI approximation between representations
- modality_mi_loss: Modality specialization loss
- compute_mutual_information: Full MI computation
- decorrelation_loss: Encourage independent dimensions
- linear_probe_evaluation: Linear probe evaluation of embeddings
"""

# TFN (Temporal Fusion Nexus) Utilities
# Adapted from original TFN for MIMIC-IV

import torch
import torch.nn as nn
import numpy as np

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def compute_mutual_information_appr(x, y):
    """
    Compute approximation of mutual information between two representations.
    Args:
        x (torch.Tensor): First representation of shape (B, D1)
        y (torch.Tensor): Second representation of shape (B, D2)
    Returns:
        torch.Tensor: MI estimate of shape (D1, D2)
    """
    # Normalize inputs
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # Compute normalized correlation as MI approximation
    # (B, D1) and (B, D2) -> (D1, D2)
    mi = torch.abs((x.T @ y) / (x.shape[0] - 1))
    return mi


def modality_mi_loss(lstm_states, static_emb, notes_emb):
    """Compute modality specialization loss using mutual information."""
    B, D = lstm_states.shape

    # Compute MI between each lstm dimension and each modality
    mi_static = compute_mutual_information_appr(lstm_states, static_emb)
    mi_notes = compute_mutual_information_appr(lstm_states, notes_emb)

    eps = 1e-8
    total_mi = mi_static.sum(dim=1) + mi_notes.sum(dim=1) + eps

    # Compute concentration scores
    static_concentration = (mi_static.sum(dim=1) / total_mi) ** 2
    notes_concentration = (mi_notes.sum(dim=1) / total_mi) ** 2

    # Remove negative sign - we want to maximize concentration
    concentration_loss = (static_concentration + notes_concentration).mean()

    # Subtract from 1 to convert to a loss (1 = max loss, 0 = perfect specialization)
    return 1.0 - concentration_loss


def compute_mutual_information(X, y):
    """
    Compute mutual information between features X and target y.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    mi_scores = []
    for i in range(X.shape[1]):
        mi = mutual_info_regression(X[:, i:i+1], y.ravel())
        mi_scores.append(mi[0])

    return np.array(mi_scores)


def decorrelation_loss(embeddings):
    """
    Compute decorrelation loss to encourage independent dimensions.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(embeddings_scaled.T)

    # Remove diagonal
    corr_matrix = corr_matrix - np.eye(corr_matrix.shape[0])

    # Return mean absolute correlation
    return np.abs(corr_matrix).mean()


def linear_probe_evaluation(embeddings, labels, test_ratio=0.2):
    """
    Evaluate embeddings using linear probe.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Simple train/test split
    n_samples = len(embeddings)
    n_test = int(n_samples * test_ratio)
    indices = np.random.permutation(n_samples)

    train_embeddings = embeddings[indices[:-n_test]]
    test_embeddings = embeddings[indices[-n_test:]]
    train_labels = labels[indices[:-n_test]]
    test_labels = labels[indices[-n_test:]]

    # Train linear probe
    probe = LinearRegression()
    probe.fit(train_embeddings, train_labels)

    # Evaluate
    train_pred = probe.predict(train_embeddings)
    test_pred = probe.predict(test_embeddings)

    train_r2 = r2_score(train_labels, train_pred)
    test_r2 = r2_score(test_labels, test_pred)

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_predictions': train_pred,
        'test_predictions': test_pred
    }