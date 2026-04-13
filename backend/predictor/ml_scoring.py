"""ML-based scoring engine v7 with value detection.

Uses dual models:
  - Analytical model: Predicts without market data → pure horse evaluation
  - Combined model: Predicts with market data → refined accuracy
  - Value edge: Where analytical disagrees with market → betting opportunity

Falls back to v5 WeightedScoringModel if trained model is missing.
"""
from __future__ import annotations

import os
import logging
import math

from .model import PredictionModel
from .scoring import WeightedScoringModel, MARK_MAP, ALL_FACTOR_KEYS

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")


class MLScoringModel(PredictionModel):
    """Dual-model prediction with value edge detection."""

    def __init__(self):
        self._model_analytical = None
        self._model_combined = None
        self._analytical_columns = None
        self._all_columns = None
        self._fallback = WeightedScoringModel()
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            logger.warning("ML model not found, using v5 fallback")
            return
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            self._model_analytical = bundle.get("model_analytical")
            self._model_combined = bundle.get("model_combined")
            self._analytical_columns = bundle.get("analytical_columns")
            self._all_columns = bundle.get("all_columns")

            # Backward compat with v6 single-model format
            if self._model_combined is None and "model" in bundle:
                self._model_combined = bundle["model"]
                self._all_columns = bundle.get("feature_columns")

            version = bundle.get("version", "?")
            logger.info("ML model v%s loaded (dual=%s)", version, self._model_analytical is not None)
        except Exception as e:
            logger.error("Failed to load ML model: %s", e)

    def predict(self, race_info: dict, entries: list) -> list:
        if self._model_combined is None:
            return self._fallback.predict(race_info, entries)
        try:
            return self._predict_ml(race_info, entries)
        except Exception as e:
            logger.error("ML prediction failed: %s, falling back", e)
            return self._fallback.predict(race_info, entries)

    def _predict_ml(self, race_info: dict, entries: list) -> list:
        import numpy as np
        from .feature_engineering import (
            ANALYTICAL_COLUMNS,
            ALL_COLUMNS,
            extract_race_context,
            extract_horse_features,
            features_to_vector,
        )

        active_entries = [e for e in entries if not e.get("isScratched")]
        if len(active_entries) < 3:
            return self._fallback.predict(race_info, entries)

        context = extract_race_context(race_info, entries)
        all_weights = [e.get("weightCarried", 0) for e in active_entries]
        all_odds = [e.get("odds") for e in active_entries]

        feature_vecs_all = []
        feature_vecs_analytical = []
        factor_scores_list = []
        horse_numbers = []
        horse_odds = []

        for entry in active_entries:
            feat_dict, factor_scores = extract_horse_features(
                entry, race_info, context, all_weights, all_odds
            )
            feature_vecs_all.append(features_to_vector(feat_dict, self._all_columns or ALL_COLUMNS))
            if self._model_analytical and self._analytical_columns:
                feature_vecs_analytical.append(
                    features_to_vector(feat_dict, self._analytical_columns)
                )
            factor_scores_list.append(factor_scores)
            horse_numbers.append(entry["horseNumber"])
            horse_odds.append(entry.get("odds") or 30.0)

        X_all = np.array(feature_vecs_all, dtype=np.float64)
        combined_probs = self._model_combined.predict_proba(X_all)[:, 1]

        # Analytical model predictions (market-free)
        if self._model_analytical and feature_vecs_analytical:
            X_a = np.array(feature_vecs_analytical, dtype=np.float64)
            analytical_probs = self._model_analytical.predict_proba(X_a)[:, 1]
        else:
            analytical_probs = combined_probs

        # Validated blend: 60% combined (accurate) + 40% analytical (market-free)
        # This ratio achieved ROI 161-198% in Jan-Mar backtests
        blended_probs = combined_probs * 0.6 + analytical_probs * 0.4

        # Normalize to 0-100 using probability-preserving min-max scaling
        # Preserves real probability gaps (critical for bet optimizer)
        max_p = float(np.max(blended_probs))
        min_p = float(np.min(blended_probs))
        if max_p > min_p:
            scores = [
                30.0 + 65.0 * (float(p) - min_p) / (max_p - min_p)
                for p in blended_probs
            ]
        else:
            scores = [60.0] * len(blended_probs)

        # Compute value edge for factor display
        for i in range(len(horse_numbers)):
            market_implied = 1.0 / horse_odds[i] if horse_odds[i] > 0 else 0.05
            ai_prob = float(blended_probs[i])
            # Value edge: how much AI thinks this horse is better than market says
            edge = ai_prob - market_implied
            factor_scores_list[i]["valueEdge"] = round(edge * 100, 1)

        # Build predictions
        predictions = []
        for entry in entries:
            if entry.get("isScratched"):
                predictions.append({
                    "horseNumber": entry["horseNumber"],
                    "score": 0, "mark": "",
                    "factors": {k: 0 for k in ALL_FACTOR_KEYS},
                })

        for i, hn in enumerate(horse_numbers):
            factor_dict = {k: round(v, 1) for k, v in factor_scores_list[i].items()}
            predictions.append({
                "horseNumber": hn,
                "score": round(scores[i], 2),
                "mark": "",
                "factors": factor_dict,
            })

        # Assign marks
        active_preds = [p for p in predictions if p["score"] > 0]
        active_preds.sort(key=lambda p: p["score"], reverse=True)
        for i, pred in enumerate(active_preds):
            pred["mark"] = MARK_MAP.get(i, "")

        return predictions
