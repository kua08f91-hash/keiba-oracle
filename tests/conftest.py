"""Shared test fixtures for KEIBA ORACLE test suite."""
from __future__ import annotations

import sys
import os

# Ensure backend is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def sample_race_info():
    return {
        "raceId": "202606030111",
        "raceName": "日経賞",
        "raceNumber": 11,
        "grade": "GII",
        "distance": 2500,
        "surface": "芝",
        "courseDetail": "右回り",
        "startTime": "15:45",
        "racecourseCode": "06",
        "date": "20260328",
        "headCount": 14,
        "trackCondition": "重",
    }


@pytest.fixture
def sample_entries():
    return [
        {
            "horseNumber": 1, "frameNumber": 1, "horseName": "テストホースA",
            "horseId": "2020100001", "sireName": "ゴールドシップ",
            "damName": "テストダム", "broodmareSire": "サンデーサイレンス",
            "age": "牡4", "weightCarried": 58.0, "jockeyName": "ルメール",
            "trainerName": "矢作", "horseWeight": "486(+2)",
            "odds": 2.5, "popularity": 1, "isScratched": False,
            "pastRaces": [
                {"pos": 1, "track": "中山"},
                {"pos": 3, "track": "東京"},
                {"pos": 2, "track": "中山"},
            ],
        },
        {
            "horseNumber": 2, "frameNumber": 1, "horseName": "テストホースB",
            "horseId": "2020100002", "sireName": "ディープインパクト",
            "damName": "テストダム2", "broodmareSire": "",
            "age": "牝3", "weightCarried": 54.0, "jockeyName": "川田",
            "trainerName": "友道", "horseWeight": "468(-8)",
            "odds": 5.0, "popularity": 2, "isScratched": False,
            "pastRaces": [
                {"pos": 8, "track": "阪神"},
                {"pos": 5, "track": "中山"},
                {"pos": 3, "track": "東京"},
            ],
        },
        {
            "horseNumber": 3, "frameNumber": 2, "horseName": "テストホースC",
            "horseId": "2020100003", "sireName": "エピファネイア",
            "damName": "テストダム3", "broodmareSire": "ブライアンズタイム",
            "age": "牡5", "weightCarried": 58.0, "jockeyName": "横山武",
            "trainerName": "国枝", "horseWeight": "490(+4)",
            "odds": 10.0, "popularity": 3, "isScratched": False,
            "pastRaces": [
                {"pos": 2, "track": "中山"},
                {"pos": 1, "track": "中山"},
            ],
        },
        {
            "horseNumber": 4, "frameNumber": 2, "horseName": "テストホースD",
            "horseId": "2020100004", "sireName": "ロードカナロア",
            "damName": "テストダム4", "age": "牡6", "weightCarried": 58.0,
            "jockeyName": "三浦", "trainerName": "田村",
            "horseWeight": "504(+12)", "odds": 50.0, "popularity": 8,
            "isScratched": False, "pastRaces": [{"pos": 10, "track": "東京"}],
        },
        {
            "horseNumber": 5, "frameNumber": 3, "horseName": "取消馬",
            "horseId": "2020100005", "sireName": "", "damName": "",
            "age": "牡4", "weightCarried": 57.0, "jockeyName": "",
            "trainerName": "", "horseWeight": "", "odds": None,
            "popularity": None, "isScratched": True, "pastRaces": [],
        },
    ]


@pytest.fixture
def sample_predictions(sample_entries, sample_race_info):
    """Generate predictions using v5 fallback (no ML model needed)."""
    from backend.predictor.scoring import WeightedScoringModel
    model = WeightedScoringModel()
    return model.predict(sample_race_info, sample_entries)


@pytest.fixture
def sample_odds_data():
    return {
        "tansho": [
            {"horses": [1], "odds": 2.5, "payout": 250},
            {"horses": [2], "odds": 5.0, "payout": 500},
            {"horses": [3], "odds": 10.0, "payout": 1000},
        ],
        "fukusho": [
            {"horses": [1], "odds": 1.3, "payout": 130},
            {"horses": [2], "odds": 1.8, "payout": 180},
            {"horses": [3], "odds": 2.5, "payout": 250},
        ],
        "umaren": [
            {"horses": [1, 2], "odds": 8.5, "payout": 850},
            {"horses": [1, 3], "odds": 15.0, "payout": 1500},
            {"horses": [2, 3], "odds": 25.0, "payout": 2500},
        ],
        "wide": [
            {"horses": [1, 2], "odds": 2.8, "payout": 280},
            {"horses": [1, 3], "odds": 4.2, "payout": 420},
            {"horses": [2, 3], "odds": 6.5, "payout": 650},
        ],
        "sanrenpuku": [
            {"horses": [1, 2, 3], "odds": 22.0, "payout": 2200},
        ],
        "sanrentan": [
            {"horses": [1, 2, 3], "odds": 120.0, "payout": 12000},
        ],
    }
