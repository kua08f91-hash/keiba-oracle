"""TDD tests for DB fallback logic, grade cross-validation,
WeightedScoringModel constructor injection, and range odds parsing.

RED → GREEN cycle applied to:
1. _race_list_from_db  — empty DB, single course, multi-course, grade, sort, COURSE_MAP
2. _is_known_graded    — exact match, startsWith, false positives, unknown names
3. WeightedScoringModel.__init__ — custom weights, market_weight injection, None handling
4. fetch_live_combination_odds — range odds (wide/fukusho), oddsMin/oddsMax, midpoint
"""
from __future__ import annotations

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# 1. _race_list_from_db
# =====================================================================
class TestRaceListFromDb:
    """Tests for main._race_list_from_db — DB fallback when netkeiba rate-limits."""

    def _make_race(self, race_id, race_number, race_name, racecourse_code, date,
                   grade=None, start_time="10:00"):
        """Create a minimal mock Race ORM object."""
        r = MagicMock()
        r.race_id = race_id
        r.race_number = race_number
        r.race_name = race_name
        r.racecourse_code = racecourse_code
        r.date = date
        r.grade = grade
        r.start_time = start_time
        return r

    def _mock_session(self, races):
        """Return a mock session whose query().filter().order_by().all() returns `races`."""
        session = MagicMock()
        query_chain = MagicMock()
        query_chain.filter.return_value = query_chain
        query_chain.order_by.return_value = query_chain
        query_chain.all.return_value = races
        session.query.return_value = query_chain
        return session

    def test_empty_db_returns_empty_list(self):
        """When no races exist for a date, return []."""
        from backend.main import _race_list_from_db
        session = self._mock_session([])
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260412")
        assert result == []

    def test_single_course_returns_one_entry(self):
        """Single course with 3 races returns exactly one course entry."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060601", 1, "未勝利", "06", "20260406"),
            self._make_race("202604060602", 2, "1勝クラス", "06", "20260406"),
            self._make_race("202604060603", 3, "2勝クラス", "06", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        assert len(result) == 1
        assert result[0]["code"] == "06"
        assert result[0]["name"] == "中山"
        assert len(result[0]["races"]) == 3

    def test_multi_course_returns_multiple_entries(self):
        """Races from two courses produce two grouped entries."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060601", 1, "未勝利", "06", "20260406"),
            self._make_race("202604060901", 1, "未勝利", "09", "20260406"),
            self._make_race("202604060902", 2, "1勝クラス", "09", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        codes = {c["code"] for c in result}
        assert codes == {"06", "09"}
        nakayama = next(c for c in result if c["code"] == "06")
        hanshin = next(c for c in result if c["code"] == "09")
        assert nakayama["name"] == "中山"
        assert hanshin["name"] == "阪神"
        assert len(nakayama["races"]) == 1
        assert len(hanshin["races"]) == 2

    def test_races_sorted_by_race_number_within_course(self):
        """Races must be sorted ascending by race_number inside each course."""
        from backend.main import _race_list_from_db
        # Provide deliberately out-of-order DB results
        races = [
            self._make_race("202604060603", 3, "2勝クラス", "06", "20260406"),
            self._make_race("202604060601", 1, "未勝利", "06", "20260406"),
            self._make_race("202604060602", 2, "1勝クラス", "06", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        numbers = [r["race_number"] for r in result[0]["races"]]
        assert numbers == [1, 2, 3]

    def test_grade_field_preserved(self):
        """grade value from Race model is included in each race dict."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060611", 11, "日経賞", "06", "20260406", grade="GII"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        race_entry = result[0]["races"][0]
        assert race_entry["grade"] == "GII"

    def test_grade_none_propagated(self):
        """grade=None is propagated as-is (not coerced to empty string)."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060601", 1, "未勝利", "06", "20260406", grade=None),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        assert result[0]["races"][0]["grade"] is None

    def test_unknown_course_code_uses_code_as_name(self):
        """A course code not in COURSE_MAP uses the raw code as name."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604069901", 1, "テスト", "99", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        assert result[0]["name"] == "99"

    def test_all_known_course_codes_resolved(self):
        """Verify COURSE_MAP covers all 10 JRA venues."""
        from backend.main import COURSE_MAP
        expected = {
            "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
        }
        for code, name in expected.items():
            assert COURSE_MAP.get(code) == name, f"COURSE_MAP missing {code}→{name}"

    def test_race_dict_has_dual_keys(self):
        """Each race dict exposes both snake_case and camelCase keys for compatibility."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060601", 1, "未勝利", "06", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        r = result[0]["races"][0]
        assert "race_id" in r and "raceId" in r
        assert "race_number" in r and "raceNumber" in r
        assert "race_name" in r and "raceName" in r
        assert r["race_id"] == r["raceId"]
        assert r["race_number"] == r["raceNumber"]

    def test_start_time_empty_string_when_none(self):
        """start_time=None on the ORM model becomes '' in the output dict."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060601", 1, "未勝利", "06", "20260406", start_time=None),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        assert result[0]["races"][0]["start_time"] == ""

    def test_courses_sorted_by_jra_standard_order(self):
        """Courses sorted by JRA standard order (main venues first), not by code string.

        COURSE_ORDER: 東京(05)=0, 中山(06)=1, 京都(08)=2, 阪神(09)=3, 札幌(01)=4, ...
        This differs from alphabetical/code-string order where 01<05<06<08<09.
        """
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060901", 1, "未勝利", "09", "20260406"),
            self._make_race("202604060501", 1, "未勝利", "05", "20260406"),
            self._make_race("202604060601", 1, "未勝利", "06", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        codes = [c["code"] for c in result]
        # JRA order: 東京(05)→中山(06)→阪神(09), NOT string order 05→06→09 (coincides here)
        assert codes == ["05", "06", "09"]

    def test_sort_order_nakayama_hanshin_fukushima(self):
        """中山(06), 阪神(09), 福島(03) → JRA order is 中山, 阪神, 福島.

        String sort would produce 03<06<09 (福島→中山→阪神).
        JRA COURSE_ORDER: 06=1, 09=3, 03=6 → 中山→阪神→福島.
        """
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060301", 1, "未勝利", "03", "20260406"),
            self._make_race("202604060901", 1, "未勝利", "09", "20260406"),
            self._make_race("202604060601", 1, "未勝利", "06", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        codes = [c["code"] for c in result]
        assert codes == ["06", "09", "03"], f"Expected 中山→阪神→福島, got {codes}"

    def test_sort_order_tokyo_first_among_main_venues(self):
        """東京(05) must come before 中山(06) — both are main venues."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060601", 1, "未勝利", "06", "20260406"),
            self._make_race("202604060501", 1, "未勝利", "05", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        codes = [c["code"] for c in result]
        assert codes == ["05", "06"], f"Expected 東京→中山, got {codes}"

    def test_sort_order_single_course_unchanged(self):
        """Single course returns as-is regardless of sort order."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604060301", 1, "未勝利", "03", "20260406"),
            self._make_race("202604060302", 2, "1勝クラス", "03", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        assert len(result) == 1
        assert result[0]["code"] == "03"
        assert result[0]["name"] == "福島"

    def test_sort_order_unknown_code_sorted_last(self):
        """Unknown course code (not in COURSE_ORDER) is placed after all known venues."""
        from backend.main import _race_list_from_db
        races = [
            self._make_race("202604069901", 1, "テスト", "99", "20260406"),
            self._make_race("202604061001", 1, "未勝利", "10", "20260406"),
            self._make_race("202604060501", 1, "未勝利", "05", "20260406"),
        ]
        session = self._mock_session(races)
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        codes = [c["code"] for c in result]
        # 東京(05)=0, 小倉(10)=9 are known; 99 is unknown → sorted last
        assert codes.index("05") < codes.index("99")
        assert codes.index("10") < codes.index("99")
        assert codes[-1] == "99"

    def test_racecourse_code_fallback_from_race_id(self):
        """When racecourse_code is None, fall back to characters 4-6 of race_id.

        JRA race ID format: YYYY CC MM DD RR
          positions 0-3: year (e.g. 2026)
          positions 4-5: course code (e.g. 06 = 中山)
          positions 6-7: meeting number
          positions 8-9: day within meeting
          positions 10-11: race number within day
        Example: '202606030201' → course_code='06' (中山)
        """
        from backend.main import _race_list_from_db
        r = MagicMock()
        r.race_id = "202606030201"  # positions 4-5 = "06" → 中山
        r.race_number = 1
        r.race_name = "未勝利"
        r.racecourse_code = None  # trigger fallback: race_id[4:6] == "06"
        r.date = "20260406"
        r.grade = None
        r.start_time = "10:00"
        session = self._mock_session([r])
        with patch("backend.main.get_session", return_value=session):
            result = _race_list_from_db("20260406")
        assert result[0]["code"] == "06"
        assert result[0]["name"] == "中山"

    def test_session_is_always_closed(self):
        """get_session().close() must be called even when an exception is raised."""
        from backend.main import _race_list_from_db
        session = MagicMock()
        session.query.side_effect = RuntimeError("DB error")
        with patch("backend.main.get_session", return_value=session):
            with pytest.raises(RuntimeError):
                _race_list_from_db("20260406")
        session.close.assert_called_once()


# =====================================================================
# 2. get_race_list endpoint — fallback integration
# =====================================================================
class TestGetRaceListFallback:
    """Integration: endpoint uses _race_list_from_db when netkeiba returns empty."""

    def test_falls_back_to_db_when_netkeiba_empty(self):
        """If fetch_race_list returns [], _race_list_from_db result is returned."""
        from backend.main import get_race_list
        db_result = [{"code": "06", "name": "中山", "races": [{"raceId": "202604060601"}]}]
        with patch("backend.main.fetch_race_list", return_value=[]), \
             patch("backend.main._race_list_from_db", return_value=db_result) as mock_db:
            result = get_race_list("20260406")
        mock_db.assert_called_once_with("20260406")
        assert result == db_result

    def test_netkeiba_result_used_when_available(self):
        """If fetch_race_list returns data, _race_list_from_db is never called."""
        from backend.main import get_race_list
        netkeiba_result = [{"code": "05", "name": "東京", "races": []}]
        with patch("backend.main.fetch_race_list", return_value=netkeiba_result), \
             patch("backend.main._race_list_from_db") as mock_db:
            result = get_race_list("20260406")
        mock_db.assert_not_called()
        assert result == netkeiba_result

    def test_invalid_date_raises_400(self):
        """Non-digit or wrong-length date string raises HTTPException 400."""
        from backend.main import get_race_list
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            get_race_list("invalid")
        assert exc_info.value.status_code == 400

    def test_seven_digit_date_raises_400(self):
        from backend.main import get_race_list
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            get_race_list("2026040")  # 7 digits, not 8


# =====================================================================
# 3. _is_known_graded
# =====================================================================
class TestIsKnownGraded:
    """Tests for parser._is_known_graded — prevents false-positive grade badges."""

    def test_exact_gi_match(self):
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("有馬記念", "GI") is True

    def test_exact_gii_match(self):
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("日経賞", "GII") is True

    def test_exact_giii_match(self):
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("中山金杯", "GIII") is True

    def test_starts_with_match_gi(self):
        """'天皇賞(春)' starts with '天皇賞', a known GI — should match."""
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("天皇賞(春)", "GI") is True

    def test_starts_with_match_gii(self):
        """'弥生賞ディープインパクト記念' starts with known GII key '弥生賞'."""
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("弥生賞ディープインパクト記念", "GII") is True

    def test_starts_with_match_giii(self):
        """'ダービー卿CT' starts with known GIII key 'ダービー卿CT'."""
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("ダービー卿CT", "GIII") is True

    def test_false_positive_sode_ga_ura_special(self):
        """袖ケ浦特別 must NOT match GIII (the bug that motivated this function)."""
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("袖ケ浦特別", "GIII") is False

    def test_false_positive_common_listed_race(self):
        """An ordinary Listed race name does not match any grade."""
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("大阪城S", "GII") is False
        assert _is_known_graded("大阪城S", "GIII") is False

    def test_wrong_grade_returns_false(self):
        """A GI race name returns False when queried as GII."""
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("有馬記念", "GII") is False

    def test_unknown_grade_key_returns_false(self):
        """Querying with an unknown grade string (e.g. 'Listed') returns False."""
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("日本ダービー", "Listed") is False

    def test_empty_race_name_returns_false(self):
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("", "GI") is False

    def test_empty_grade_returns_false(self):
        from backend.scraper.parser import _is_known_graded
        assert _is_known_graded("有馬記念", "") is False

    def test_partial_substring_not_at_start_returns_false(self):
        """A substring match that is NOT at the start must not trigger."""
        from backend.scraper.parser import _is_known_graded
        # "記念有馬" contains "有馬" but does not startWith a key "有馬記念"
        assert _is_known_graded("記念有馬記念", "GI") is False

    def test_nhk_mile_cup_variations(self):
        """Both 'NHKマイルC' and 'NHKマイルカップ' resolve to GI."""
        from backend.scraper.parser import _is_known_graded
        # "NHKマイルC" is in the list directly
        assert _is_known_graded("NHKマイルC", "GI") is True
        # "NHKマイルカップ2026" starts with "NHKマイルC" (6 chars match)
        assert _is_known_graded("NHKマイルカップ2026", "GI") is True

    def test_all_gi_names_resolve(self):
        """Spot-check that every entry in _KNOWN_GRADED['GI'] self-resolves."""
        from backend.scraper.parser import _is_known_graded, _KNOWN_GRADED
        for name in _KNOWN_GRADED["GI"]:
            assert _is_known_graded(name, "GI") is True, f"GI lookup failed for: {name}"

    def test_all_gii_names_resolve(self):
        from backend.scraper.parser import _is_known_graded, _KNOWN_GRADED
        for name in _KNOWN_GRADED["GII"]:
            assert _is_known_graded(name, "GII") is True, f"GII lookup failed for: {name}"

    def test_all_giii_names_resolve(self):
        from backend.scraper.parser import _is_known_graded, _KNOWN_GRADED
        for name in _KNOWN_GRADED["GIII"]:
            assert _is_known_graded(name, "GIII") is True, f"GIII lookup failed for: {name}"


# =====================================================================
# 4. WeightedScoringModel constructor injection
# =====================================================================
class TestWeightedScoringModelInit:
    """Tests for scoring.WeightedScoringModel.__init__ — weight injection."""

    def test_default_weights_used_when_none(self):
        """Passing no args uses module-level ANALYTICAL_WEIGHTS and MARKET_WEIGHT."""
        from backend.predictor.scoring import WeightedScoringModel, ANALYTICAL_WEIGHTS, MARKET_WEIGHT
        model = WeightedScoringModel()
        assert model._weights == ANALYTICAL_WEIGHTS
        assert model._market_weight == pytest.approx(MARKET_WEIGHT)

    def test_custom_analytical_weights_injected(self):
        """Custom analytical_weights dict is stored without mutation."""
        from backend.predictor.scoring import WeightedScoringModel
        custom = {"trackDirection": 0.5, "jockeyAbility": 0.5}
        model = WeightedScoringModel(analytical_weights=custom)
        assert model._weights is custom

    def test_custom_market_weight_injected(self):
        from backend.predictor.scoring import WeightedScoringModel
        model = WeightedScoringModel(market_weight=0.30)
        assert model._market_weight == pytest.approx(0.30)
        assert model._analytical_weight == pytest.approx(0.70)

    def test_market_weight_zero_accepted(self):
        """market_weight=0.0 is a valid explicit value (not falsy-treated as None)."""
        from backend.predictor.scoring import WeightedScoringModel
        model = WeightedScoringModel(market_weight=0.0)
        assert model._market_weight == pytest.approx(0.0)
        assert model._analytical_weight == pytest.approx(1.0)

    def test_market_weight_none_falls_back_to_default(self):
        from backend.predictor.scoring import WeightedScoringModel, MARKET_WEIGHT
        model = WeightedScoringModel(market_weight=None)
        assert model._market_weight == pytest.approx(MARKET_WEIGHT)

    def test_analytical_weight_is_complement_of_market_weight(self):
        """_analytical_weight == 1.0 - _market_weight for any injected value."""
        from backend.predictor.scoring import WeightedScoringModel
        for mw in (0.0, 0.10, 0.15, 0.25, 0.50, 1.0):
            model = WeightedScoringModel(market_weight=mw)
            assert model._analytical_weight == pytest.approx(1.0 - mw), \
                f"Complement check failed for market_weight={mw}"

    def test_default_weights_not_mutated_between_instances(self):
        """Two default instances share the same module-level dict reference values
        but injecting into one does NOT affect the other."""
        from backend.predictor.scoring import WeightedScoringModel, ANALYTICAL_WEIGHTS
        model_a = WeightedScoringModel()
        model_b = WeightedScoringModel(analytical_weights={"trackDirection": 1.0})
        # model_a should still use the full weight dict
        assert model_a._weights == ANALYTICAL_WEIGHTS
        assert "jockeyAbility" in model_a._weights

    def test_custom_weights_affect_prediction_scores(self, sample_race_info, sample_entries):
        """Injecting custom weights that heavily favour jockeyAbility changes output scores."""
        from backend.predictor.scoring import WeightedScoringModel, ANALYTICAL_WEIGHTS
        # All weight on jockey so only jockey ability matters
        jockey_only = {k: (1.0 if k == "jockeyAbility" else 0.0) for k in ANALYTICAL_WEIGHTS}
        model_default = WeightedScoringModel()
        model_jockey = WeightedScoringModel(analytical_weights=jockey_only, market_weight=0.0)
        preds_default = model_default.predict(sample_race_info, sample_entries)
        preds_jockey = model_jockey.predict(sample_race_info, sample_entries)
        scores_default = [p["score"] for p in preds_default if p["score"] > 0]
        scores_jockey = [p["score"] for p in preds_jockey if p["score"] > 0]
        # The two models must produce different rankings (not identical)
        assert scores_default != scores_jockey

    def test_predict_still_works_with_injected_weights(self, sample_race_info, sample_entries):
        """Injected custom weights don't break predict() output structure."""
        from backend.predictor.scoring import WeightedScoringModel, ANALYTICAL_WEIGHTS
        custom = {k: 1.0 / len(ANALYTICAL_WEIGHTS) for k in ANALYTICAL_WEIGHTS}
        model = WeightedScoringModel(analytical_weights=custom, market_weight=0.10)
        preds = model.predict(sample_race_info, sample_entries)
        assert isinstance(preds, list)
        assert len(preds) == len(sample_entries)
        for p in preds:
            assert "score" in p
            assert "mark" in p
            assert "factors" in p


# =====================================================================
# 5. fetch_live_combination_odds — range odds (ワイド/複勝)
# =====================================================================
class TestFetchLiveCombinationOddsRangeOdds:
    """Tests for odds.fetch_live_combination_odds — range odds parsing."""

    def _mock_requests_get(self, responses_by_type: dict):
        """Build a requests.get side-effect that returns different payloads per URL."""
        def _get(url, headers=None, timeout=None):
            for api_type, payload in responses_by_type.items():
                if f"type={api_type}" in url:
                    resp = MagicMock()
                    resp.text = __import__("json").dumps(payload)
                    return resp
            resp = MagicMock()
            resp.text = "{}"
            return resp
        return _get

    def test_wide_uses_midpoint_of_range(self):
        """ワイド vals=[min, max, pop] → odds=(min+max)/2 stored, oddsMin/oddsMax attached."""
        from backend.scraper.odds import fetch_live_combination_odds
        payload = {
            "data": {
                "odds": {
                    "5": {
                        "0102": ["3.0", "5.0", "2"],
                    }
                }
            }
        }
        mock_get = self._mock_requests_get({5: payload})
        with patch("backend.scraper.odds.requests.get", side_effect=mock_get):
            result = fetch_live_combination_odds("202604060601", {}, include_win_place=False)
        assert "wide" in result
        wide_entries = result["wide"]
        assert len(wide_entries) == 1
        entry = wide_entries[0]
        assert entry["odds"] == pytest.approx((3.0 + 5.0) / 2, abs=0.05)
        assert entry["oddsMin"] == pytest.approx(3.0)
        assert entry["oddsMax"] == pytest.approx(5.0)

    def test_fukusho_uses_midpoint_of_range(self):
        """複勝 vals=[min, max, pop] → midpoint used for odds."""
        from backend.scraper.odds import fetch_live_combination_odds
        payload = {
            "data": {
                "odds": {
                    "2": {
                        "01": ["2.0", "4.0", "1"],
                    }
                }
            }
        }
        mock_get = self._mock_requests_get({2: payload})
        with patch("backend.scraper.odds.requests.get", side_effect=mock_get):
            result = fetch_live_combination_odds("202604060601", {}, include_win_place=True)
        assert "fukusho" in result
        entry = result["fukusho"][0]
        assert entry["odds"] == pytest.approx((2.0 + 4.0) / 2, abs=0.05)
        assert entry["oddsMin"] == pytest.approx(2.0)
        assert entry["oddsMax"] == pytest.approx(4.0)

    def test_payout_reflects_midpoint(self):
        """payout field = int(midpoint * 100) for range types."""
        from backend.scraper.odds import fetch_live_combination_odds
        payload = {
            "data": {
                "odds": {
                    "5": {
                        "0304": ["4.0", "8.0", "3"],
                    }
                }
            }
        }
        mock_get = self._mock_requests_get({5: payload})
        with patch("backend.scraper.odds.requests.get", side_effect=mock_get):
            result = fetch_live_combination_odds("202604060601", {})
        entry = result["wide"][0]
        expected_midpoint = (4.0 + 8.0) / 2  # 6.0
        assert entry["payout"] == int(expected_midpoint * 100)  # 600

    def test_wide_equal_min_max_uses_exact_value(self):
        """When oddsMax parsing fails / gives 0, oddsMax stays == oddsMin (no artificial spread)."""
        from backend.scraper.odds import fetch_live_combination_odds
        payload = {
            "data": {
                "odds": {
                    "5": {
                        "0102": ["5.0", "5.0", "2"],
                    }
                }
            }
        }
        mock_get = self._mock_requests_get({5: payload})
        with patch("backend.scraper.odds.requests.get", side_effect=mock_get):
            result = fetch_live_combination_odds("202604060601", {})
        entry = result["wide"][0]
        assert entry["odds"] == pytest.approx(5.0)
        assert entry["oddsMin"] == pytest.approx(5.0)
        assert entry["oddsMax"] == pytest.approx(5.0)

    def test_non_range_type_umaren_has_no_oddsmin_max(self):
        """馬連 (non-range type) does NOT attach oddsMin/oddsMax fields."""
        from backend.scraper.odds import fetch_live_combination_odds
        payload = {
            "data": {
                "odds": {
                    "4": {
                        "0102": ["8.5", "0", "1"],
                    }
                }
            }
        }
        mock_get = self._mock_requests_get({4: payload})
        with patch("backend.scraper.odds.requests.get", side_effect=mock_get):
            result = fetch_live_combination_odds("202604060601", {})
        assert "umaren" in result
        entry = result["umaren"][0]
        assert "oddsMin" not in entry
        assert "oddsMax" not in entry

    def test_tansho_single_value_no_range_fields(self):
        """単勝 with include_win_place=True uses single odds, no range fields."""
        from backend.scraper.odds import fetch_live_combination_odds
        payload = {
            "data": {
                "odds": {
                    "1": {
                        "01": ["3.5", "0", "1"],
                    }
                }
            }
        }
        mock_get = self._mock_requests_get({1: payload})
        with patch("backend.scraper.odds.requests.get", side_effect=mock_get):
            result = fetch_live_combination_odds("202604060601", {}, include_win_place=True)
        assert "tansho" in result
        entry = result["tansho"][0]
        assert entry["odds"] == pytest.approx(3.5)
        assert "oddsMin" not in entry
        assert "oddsMax" not in entry

    def test_zero_odds_skipped(self):
        """Entries with odds_min <= 0 are skipped entirely."""
        from backend.scraper.odds import fetch_live_combination_odds
        payload = {
            "data": {
                "odds": {
                    "5": {
                        "0102": ["0", "0", "0"],
                        "0103": ["3.0", "5.0", "2"],
                    }
                }
            }
        }
        mock_get = self._mock_requests_get({5: payload})
        with patch("backend.scraper.odds.requests.get", side_effect=mock_get):
            result = fetch_live_combination_odds("202604060601", {})
        assert len(result.get("wide", [])) == 1

    def test_comma_formatted_odds_parsed_correctly(self):
        """Odds strings with commas like '1,234.5' are parsed as floats."""
        from backend.scraper.odds import fetch_live_combination_odds
        payload = {
            "data": {
                "odds": {
                    "5": {
                        "0102": ["1,000.0", "2,000.0", "5"],
                    }
                }
            }
        }
        mock_get = self._mock_requests_get({5: payload})
        with patch("backend.scraper.odds.requests.get", side_effect=mock_get):
            result = fetch_live_combination_odds("202604060601", {})
        entry = result["wide"][0]
        assert entry["oddsMin"] == pytest.approx(1000.0)
        assert entry["oddsMax"] == pytest.approx(2000.0)
        assert entry["odds"] == pytest.approx(1500.0)

    def test_fallback_odds_preserved_when_api_fails(self):
        """When all API calls fail, the fallback_odds dict is returned unchanged."""
        from backend.scraper.odds import fetch_live_combination_odds
        fallback = {"tansho": [{"horses": [1], "odds": 2.5, "payout": 250}]}
        with patch("backend.scraper.odds.requests.get", side_effect=Exception("timeout")):
            result = fetch_live_combination_odds("202604060601", fallback)
        assert result["tansho"] == fallback["tansho"]

    def test_include_win_place_false_excludes_tansho_fukusho(self):
        """When include_win_place=False, tansho/fukusho are not fetched."""
        from backend.scraper.odds import fetch_live_combination_odds
        with patch("backend.scraper.odds.requests.get", side_effect=Exception("should not call for 1/2")) as mock_get:
            # We expect calls only for type 4,5,6,7,8, not 1 or 2
            result = fetch_live_combination_odds("202604060601", {}, include_win_place=False)
        # No crash; tansho/fukusho should be absent (or from fallback only)
        assert "tansho" not in result
        assert "fukusho" not in result

    def test_horses_parsed_from_combo_key(self):
        """Combo key '0508' → horses=[5,8], '050812' → horses=[5,8,12]."""
        from backend.scraper.odds import parse_combo_key
        assert parse_combo_key("0508") == [5, 8]
        assert parse_combo_key("050812") == [5, 8, 12]
        assert parse_combo_key("01") == [1]
