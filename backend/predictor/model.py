"""Prediction model interface for future ML upgrade."""
from __future__ import annotations

from abc import ABC, abstractmethod


class PredictionModel(ABC):
    """Abstract base class for prediction models."""

    @abstractmethod
    def predict(self, race_info: dict, entries: list[dict]) -> list[dict]:
        """Generate predictions for a race.

        Returns list of {horseNumber, score, mark, factors}
        """
        pass
