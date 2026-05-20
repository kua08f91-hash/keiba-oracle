"""Probability calibration for horse racing predictions (Phase 3).

Ensures that when the model says "30% chance to win", the horse actually
wins ~30% of the time. This is critical for EV-driven bet selection.

Methods:
- Isotonic regression (non-parametric, preserves ordering)
- Temperature scaling (single parameter, simpler)

Usage:
    calibrator = IsotonicCalibrator()
    calibrator.fit(predicted_probs, actual_wins)
    calibrated = calibrator.transform(new_probs)
"""
from __future__ import annotations

import logging
import math
import os
import pickle
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATOR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "calibrator.pkl"
)


class IsotonicCalibrator:
    """Isotonic regression calibrator for win probabilities."""

    def __init__(self):
        self._model = None

    def fit(self, predicted: np.ndarray, actual: np.ndarray):
        """Fit calibrator on predicted probabilities vs actual outcomes.

        Args:
            predicted: Model's predicted win probabilities [0,1]
            actual: Binary outcomes (1=won, 0=lost)
        """
        from sklearn.isotonic import IsotonicRegression
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(predicted, actual)
        logger.info("Isotonic calibrator fitted on %d samples", len(predicted))

    def transform(self, predicted: np.ndarray) -> np.ndarray:
        """Calibrate predicted probabilities."""
        if self._model is None:
            return predicted
        return self._model.transform(predicted)

    def save(self, path: str = CALIBRATOR_PATH):
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "type": "isotonic"}, f)

    @classmethod
    def load(cls, path: str = CALIBRATOR_PATH) -> Optional["IsotonicCalibrator"]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            c = cls()
            c._model = data["model"]
            return c
        except Exception:
            return None


class TemperatureCalibrator:
    """Temperature scaling calibrator (simpler, single parameter)."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, predicted: np.ndarray, actual: np.ndarray):
        """Find optimal temperature by minimizing log-loss."""
        best_t = 1.0
        best_loss = float("inf")

        for t in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]:
            scaled = self._scale(predicted, t)
            loss = -np.mean(actual * np.log(scaled + 1e-10) + (1 - actual) * np.log(1 - scaled + 1e-10))
            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = best_t
        logger.info("Temperature calibrator: T=%.2f (log-loss=%.4f)", best_t, best_loss)

    def transform(self, predicted: np.ndarray) -> np.ndarray:
        return self._scale(predicted, self.temperature)

    @staticmethod
    def _scale(probs: np.ndarray, t: float) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        logits = np.log(probs / (1 - probs + 1e-10) + 1e-10)
        scaled_logits = logits / t
        return 1 / (1 + np.exp(-scaled_logits))


def compute_ece(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Lower is better. 0.0 = perfectly calibrated.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (predicted >= bin_boundaries[i]) & (predicted < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = actual[mask].mean()
        bin_conf = predicted[mask].mean()
        ece += mask.sum() / len(predicted) * abs(bin_acc - bin_conf)
    return ece


def compute_log_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute negative log-likelihood (lower is better)."""
    eps = 1e-10
    return -np.mean(actual * np.log(predicted + eps) + (1 - actual) * np.log(1 - predicted + eps))
