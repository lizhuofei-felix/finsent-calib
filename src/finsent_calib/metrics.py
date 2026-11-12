"""Classification and calibration metrics shared by every model."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins: sum over bins of (bin share) * |bin accuracy - bin confidence|."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean of -log(probability given to the true class)."""
    p_true = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean over sentences of the squared distance between probs and the one-hot true label."""
    one_hot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """All metrics for one set of predicted probabilities."""
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "ece": expected_calibration_error(probs, y_true),
        "nll": negative_log_likelihood(probs, y_true),
        "brier": brier_score(probs, y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
