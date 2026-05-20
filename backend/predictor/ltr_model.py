"""Learning-to-Rank model for horse racing prediction (Phase 1).

Uses LightGBM with conditional logit / listwise ranking objective.
Each race is a "query group" — the model learns to rank horses within a race.

Key advantages over v5 linear model:
- Automatic interaction learning (脚質×馬場×ペース, 距離×血統, etc.)
- Non-linear relationships
- Feature importance for interpretability

Usage:
    # Training
    /usr/bin/python3 -m backend.predictor.ltr_model --train --data backtest/cache/

    # Inference (used by MLScoringModel as optional predictor)
    model = LTRModel.load("data/ltr_model.pkl")
    scores = model.predict(race_info, entries)
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

from .factors import (
    calc_market_score, calc_past_performance, calc_jockey_ability,
    calc_course_affinity, calc_distance_aptitude, calc_trainer_ability,
    calc_track_condition_affinity, calc_track_direction, calc_track_specific,
    calc_age_and_sex, calc_weight_carried, calc_horse_weight_change,
    calc_form_trend, calc_same_distance_performance, calc_same_surface_performance,
    calc_same_condition_performance, calc_speed_figure,
    calc_running_style_consistency, calc_days_since_last_race,
    calc_weight_carried_trend, calc_agari3f_score, calc_margin_score,
    calc_draw_bias,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          "data", "ltr_model.pkl")


def extract_features(race_info: dict, entries: list) -> tuple:
    """Extract feature matrix for one race.

    Returns:
        (feature_matrix: np.ndarray [n_horses x n_features],
         horse_numbers: list[int],
         feature_names: list[str])
    """
    surface = race_info.get("surface", "芝")
    distance = race_info.get("distance", 2000)
    head_count = len([e for e in entries if not e.get("isScratched")])
    all_weights = [e.get("weightCarried", 0) for e in entries if not e.get("isScratched")]
    track_condition = race_info.get("trackCondition", "")
    course_detail = race_info.get("courseDetail", "")
    racecourse_code = race_info.get("racecourseCode", "")
    race_date = race_info.get("date", "")
    race_date_norm = ""
    if race_date and len(race_date) == 8 and race_date.isdigit():
        race_date_norm = f"{race_date[:4]}.{race_date[4:6]}.{race_date[6:]}"

    rows = []
    horse_numbers = []

    for entry in entries:
        if entry.get("isScratched"):
            continue

        past_races = entry.get("pastRaces", [])
        sire = entry.get("sireName", "")
        bms = entry.get("broodmareSire", "")
        jockey = entry.get("jockeyName", "")
        trainer = entry.get("trainerName", "")
        age_str = entry.get("age", "")
        weight = entry.get("weightCarried", 0)
        odds = entry.get("odds")
        popularity = entry.get("popularity")
        horse_weight = entry.get("horseWeight", "")
        frame = entry.get("frameNumber", entry.get("horseNumber", 0))

        # v5 factor scores (22 factors)
        factors = {
            "marketScore": calc_market_score(odds, popularity, head_count),
            "pastPerformance": calc_past_performance(past_races),
            "jockeyAbility": calc_jockey_ability(jockey),
            "courseAffinity": calc_course_affinity(sire, surface),
            "distanceAptitude": calc_distance_aptitude(sire, distance),
            "trainerAbility": calc_trainer_ability(trainer),
            "trackCondition": calc_track_condition_affinity(sire, track_condition, bms),
            "trackDirection": calc_track_direction(past_races, course_detail, distance),
            "trackSpecific": calc_track_specific(past_races, racecourse_code),
            "ageAndSex": calc_age_and_sex(age_str),
            "weightCarried": calc_weight_carried(weight, all_weights),
            "horseWeightChange": calc_horse_weight_change(horse_weight),
            "formTrend": calc_form_trend(past_races),
            "sameDistance": calc_same_distance_performance(past_races, distance),
            "sameSurface": calc_same_surface_performance(past_races, surface),
            "sameCondition": calc_same_condition_performance(past_races, track_condition),
            "speedFigure": calc_speed_figure(past_races, distance),
            "runningStyle": calc_running_style_consistency(past_races),
            "daysSinceLast": calc_days_since_last_race(past_races, race_date_norm),
            "weightCarriedTrend": calc_weight_carried_trend(past_races, weight),
            "agari3f": calc_agari3f_score(past_races, surface),
            "marginScore": calc_margin_score(past_races),
            "drawBias": calc_draw_bias(frame, head_count, surface, distance, course_detail),
        }

        # Relative features (within-race rank)
        row = list(factors.values())

        # Race context features
        row.extend([
            head_count,
            1 if surface == "芝" else 0,
            distance,
            1 if track_condition in ("重", "不良") else 0,
            popularity if popularity else head_count // 2,
            odds if odds else 10.0,
        ])

        rows.append(row)
        horse_numbers.append(entry["horseNumber"])

    if not rows:
        return np.array([]), [], []

    feature_names = list(factors.keys()) + [
        "head_count", "is_turf", "distance", "is_heavy_track",
        "popularity", "odds",
    ]

    X = np.array(rows, dtype=np.float32)

    # Add relative features (rank within race for key factors)
    if len(rows) >= 2:
        for i, fname in enumerate(feature_names[:len(factors)]):
            col = X[:, i]
            ranks = np.argsort(np.argsort(-col)) + 1  # 1-based rank (highest = 1)
            diff_from_top = col - col.max()
            X = np.column_stack([X, ranks, diff_from_top])
            feature_names.extend([f"{fname}_rank", f"{fname}_diff_top"])

    return X, horse_numbers, feature_names


class LTRModel:
    """LightGBM Learning-to-Rank model."""

    def __init__(self):
        self._model = None
        self._feature_names = None

    def train(self, races: list, val_races: list = None):
        """Train on a list of race dicts (from backtest cache).

        Each race dict must have: info, entries, positions.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. Run: pip install lightgbm")
            return

        X_all, y_all, groups = [], [], []

        for rd in races:
            X, horse_numbers, fnames = extract_features(rd["info"], rd["entries"])
            if len(X) < 3:
                continue
            self._feature_names = fnames

            # Labels: relevance = inverse of finish position
            # 1st = highest relevance, last = 0
            positions = rd["positions"]
            labels = []
            for hn in horse_numbers:
                pos = positions.get(hn, len(horse_numbers))
                labels.append(max(0, len(horse_numbers) - pos))

            X_all.append(X)
            y_all.extend(labels)
            groups.append(len(horse_numbers))

        if not X_all:
            logger.error("No valid training data")
            return

        X_train = np.vstack(X_all)
        y_train = np.array(y_all, dtype=np.float32)

        train_data = lgb.Dataset(X_train, label=y_train, group=groups,
                                 feature_name=self._feature_names)

        # Validation
        val_data = None
        if val_races:
            X_val_all, y_val_all, val_groups = [], [], []
            for rd in val_races:
                X, hns, _ = extract_features(rd["info"], rd["entries"])
                if len(X) < 3:
                    continue
                positions = rd["positions"]
                labels = [max(0, len(hns) - positions.get(hn, len(hns))) for hn in hns]
                X_val_all.append(X)
                y_val_all.extend(labels)
                val_groups.append(len(hns))
            if X_val_all:
                X_val = np.vstack(X_val_all)
                y_val = np.array(y_val_all, dtype=np.float32)
                val_data = lgb.Dataset(X_val, label=y_val, group=val_groups,
                                       reference=train_data)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        callbacks = [lgb.log_evaluation(50)]
        if val_data:
            callbacks.append(lgb.early_stopping(50))

        self._model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data] if val_data else [],
            callbacks=callbacks,
        )

        logger.info("LTR model trained: %d rounds, %d features",
                     self._model.best_iteration or 500, len(self._feature_names))

    def predict(self, race_info: dict, entries: list) -> list:
        """Predict scores for each horse in a race.

        Returns list of {horseNumber, score, mark, factors} compatible with v5.
        """
        if self._model is None:
            return []

        X, horse_numbers, _ = extract_features(race_info, entries)
        if len(X) == 0:
            return []

        raw_scores = self._model.predict(X)

        # Normalize to 0-100 scale
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            scores = (raw_scores - min_s) / (max_s - min_s) * 80 + 20
        else:
            scores = np.full_like(raw_scores, 50.0)

        MARK_MAP = {0: "◎", 1: "◯", 2: "▲", 3: "▲", 4: "△", 5: "△"}

        results = []
        ranked = sorted(zip(horse_numbers, scores), key=lambda x: -x[1])
        for rank, (hn, score) in enumerate(ranked):
            results.append({
                "horseNumber": hn,
                "score": round(float(score), 1),
                "mark": MARK_MAP.get(rank, ""),
                "factors": {},
            })

        return results

    def save(self, path: str = MODEL_PATH):
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "feature_names": self._feature_names,
                "version": "ltr_v1",
            }, f)
        logger.info("LTR model saved to %s", path)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> Optional["LTRModel"]:
        """Load model from disk."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            m = cls()
            m._model = data["model"]
            m._feature_names = data.get("feature_names")
            logger.info("LTR model loaded (version=%s)", data.get("version", "?"))
            return m
        except Exception as e:
            logger.warning("Failed to load LTR model: %s", e)
            return None

    def feature_importance(self, top_n: int = 20) -> list:
        """Get top feature importances."""
        if self._model is None:
            return []
        imp = self._model.feature_importance(importance_type="gain")
        names = self._feature_names or [f"f{i}" for i in range(len(imp))]
        pairs = sorted(zip(names, imp), key=lambda x: -x[1])
        return pairs[:top_n]
