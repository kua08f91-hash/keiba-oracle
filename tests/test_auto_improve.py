"""TDD tests for auto_improve pipeline.

Covers: result collection, historical data update, model evaluation,
retrain trigger logic, performance logging.
"""
from __future__ import annotations

import json
import os
import tempfile
import pytest


class TestUpdateHistoricalData:
    def test_adds_new_races(self, tmp_path):
        from backend.auto_improve import update_historical_data, HIST_FILE
        # Override HIST_FILE for test
        import backend.auto_improve as module
        original = module.HIST_FILE
        module.HIST_FILE = str(tmp_path / "hist.json")

        try:
            # Write initial data
            with open(module.HIST_FILE, "w") as f:
                json.dump([{"race_id": "AAA", "date": "20260101"}], f)

            new_races = [
                {"race_id": "BBB", "date": "20260102", "race_info": {}, "entries": [], "results": {}},
            ]
            added = update_historical_data(new_races)
            assert added == 1

            with open(module.HIST_FILE) as f:
                data = json.load(f)
            assert len(data) == 2
        finally:
            module.HIST_FILE = original

    def test_no_duplicates(self, tmp_path):
        from backend.auto_improve import update_historical_data
        import backend.auto_improve as module
        original = module.HIST_FILE
        module.HIST_FILE = str(tmp_path / "hist.json")

        try:
            with open(module.HIST_FILE, "w") as f:
                json.dump([{"race_id": "AAA"}], f)

            new_races = [{"race_id": "AAA", "race_info": {}, "entries": [], "results": {}}]
            added = update_historical_data(new_races)
            assert added == 0
        finally:
            module.HIST_FILE = original

    def test_creates_file_if_missing(self, tmp_path):
        from backend.auto_improve import update_historical_data
        import backend.auto_improve as module
        original = module.HIST_FILE
        module.HIST_FILE = str(tmp_path / "new_hist.json")

        try:
            new_races = [{"race_id": "CCC", "date": "20260401", "race_info": {}, "entries": [
                {"horseNumber": 1, "horseName": "Test"}
            ], "results": {"1": 1}}]
            added = update_historical_data(new_races)
            assert added == 1
            assert os.path.exists(module.HIST_FILE)
        finally:
            module.HIST_FILE = original


class TestShouldRetrain:
    def test_low_tansho_triggers(self):
        from backend.auto_improve import should_retrain
        assert should_retrain({"tansho": 20, "wide": 50}) is True

    def test_low_wide_triggers(self):
        from backend.auto_improve import should_retrain
        assert should_retrain({"tansho": 30, "wide": 35}) is True

    def test_good_metrics_no_retrain(self, tmp_path):
        from backend.auto_improve import should_retrain
        import backend.auto_improve as module
        original = module.PERF_LOG
        module.PERF_LOG = str(tmp_path / "perf.json")
        try:
            assert should_retrain({"tansho": 35, "wide": 55}) is False
        finally:
            module.PERF_LOG = original

    def test_empty_metrics(self):
        from backend.auto_improve import should_retrain
        assert should_retrain({}) is False

    def test_declining_trend_triggers(self, tmp_path):
        from backend.auto_improve import should_retrain
        import backend.auto_improve as module
        original = module.PERF_LOG
        module.PERF_LOG = str(tmp_path / "perf.json")

        try:
            # 3 declining entries
            log = [
                {"tansho": 40, "wide": 60},
                {"tansho": 35, "wide": 55},
                {"tansho": 30, "wide": 50},
            ]
            with open(module.PERF_LOG, "w") as f:
                json.dump(log, f)

            result = should_retrain({"tansho": 28, "wide": 48})
            assert result is True
        finally:
            module.PERF_LOG = original


class TestLogPerformance:
    def test_creates_log(self, tmp_path):
        from backend.auto_improve import log_performance
        import backend.auto_improve as module
        original = module.PERF_LOG
        module.PERF_LOG = str(tmp_path / "perf.json")

        try:
            log_performance({"tansho": 35, "wide": 55, "date": "2026-04-01"})
            with open(module.PERF_LOG) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["tansho"] == 35
        finally:
            module.PERF_LOG = original

    def test_appends_to_existing(self, tmp_path):
        from backend.auto_improve import log_performance
        import backend.auto_improve as module
        original = module.PERF_LOG
        module.PERF_LOG = str(tmp_path / "perf.json")

        try:
            with open(module.PERF_LOG, "w") as f:
                json.dump([{"tansho": 30}], f)

            log_performance({"tansho": 35})
            with open(module.PERF_LOG) as f:
                data = json.load(f)
            assert len(data) == 2
        finally:
            module.PERF_LOG = original


class TestEvaluateCurrentModel:
    def test_returns_metrics(self, sample_race_info, sample_entries):
        from backend.auto_improve import evaluate_current_model
        races = [{
            "entries": sample_entries,
            "race_info": sample_race_info,
            "results": {"1": 1, "2": 2, "3": 3, "4": 4},
        }]
        metrics = evaluate_current_model(races)
        assert "tansho" in metrics
        assert "wide" in metrics
        assert "races" in metrics
        assert metrics["races"] == 1

    def test_empty_races(self):
        from backend.auto_improve import evaluate_current_model
        metrics = evaluate_current_model([])
        assert metrics == {}
