"""TDD tests for _parse_race_info() grade detection in parser.py.

Covers:
- Flat-race icon classes: Icon_GradeType1/2/3
- Jumps-race icon classes: Icon_GradeType15/12 (GI), 16/13 (GII), 17/14 (GIII)
- Name-based fallback for jumps races (icons absent)
- Priority: icon detection wins over name-based fallback
- Edge cases: no RaceName element, unknown icons, conflicting signals
"""
from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from backend.scraper.parser import parse_race_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_html(race_name: str, icon_class: str = "", extra_name_text: str = "") -> str:
    """Build a minimal netkeiba-style shutuba page snippet."""
    icon_html = f'<span class="{icon_class}"></span>' if icon_class else ""
    return f"""
    <html><body>
      <div class="RaceName">
        {icon_html}
        {race_name}{extra_name_text}
      </div>
    </body></html>
    """


def parse_grade(race_name: str, icon_class: str = "") -> str | None:
    """Helper: return grade from parse_race_card for a minimal page."""
    html = _make_html(race_name, icon_class)
    result = parse_race_card(html)
    return result["race_info"]["grade"]


# ---------------------------------------------------------------------------
# Class 1: Flat-race icon detection (existing behaviour, regression guard)
# ---------------------------------------------------------------------------

class TestFlatRaceIconDetection:
    """Icon_GradeType1/2/3 — original flat-race icons must still work."""

    def test_icon_grade_type1_returns_gi(self):
        assert parse_grade("大阪杯", "Icon_GradeType1") == "GI"

    def test_icon_grade_type2_returns_gii(self):
        assert parse_grade("日経賞", "Icon_GradeType2") == "GII"

    def test_icon_grade_type3_returns_giii(self):
        assert parse_grade("中山金杯", "Icon_GradeType3") == "GIII"

    def test_no_icon_no_known_name_returns_none(self):
        assert parse_grade("未知の特別戦") is None


# ---------------------------------------------------------------------------
# Class 2: Jumps-race icon detection (new icon classes)
# ---------------------------------------------------------------------------

class TestJumpsRaceIconDetection:
    """Icon_GradeType15/12 → GI, 16/13 → GII, 17/14 → GIII."""

    # GI jumps icons
    def test_icon_grade_type15_returns_gi(self):
        assert parse_grade("中山グランドジャンプ", "Icon_GradeType15") == "GI"

    def test_icon_grade_type12_returns_gi(self):
        assert parse_grade("中山グランドジャンプ", "Icon_GradeType12") == "GI"

    # GII jumps icons
    def test_icon_grade_type16_returns_gii(self):
        assert parse_grade("東京ハイジャンプ", "Icon_GradeType16") == "GII"

    def test_icon_grade_type13_returns_gii(self):
        assert parse_grade("阪神スプリングジャンプ", "Icon_GradeType13") == "GII"

    # GIII jumps icons
    def test_icon_grade_type17_returns_giii(self):
        assert parse_grade("京都ジャンプS", "Icon_GradeType17") == "GIII"

    def test_icon_grade_type14_returns_giii(self):
        assert parse_grade("阪神ジャンプS", "Icon_GradeType14") == "GIII"

    # Icon takes priority regardless of name content
    def test_icon12_on_unknown_name_no_grade(self):
        """GI icons also require cross-validation (netkeiba uses them for Open races too)."""
        assert parse_grade("特別レース", "Icon_GradeType12") is None

    def test_icon13_on_unknown_name_no_grade(self):
        """GII/GIII icons on unknown names are rejected (cross-validation)."""
        assert parse_grade("障害特別", "Icon_GradeType13") is None

    def test_icon14_on_unknown_name_no_grade(self):
        assert parse_grade("小障害", "Icon_GradeType14") is None


# ---------------------------------------------------------------------------
# Class 3: Name-based fallback (no icon present)
# ---------------------------------------------------------------------------

class TestNameBasedFallback:
    """When no icon is present, use name-based fallback for jumps races."""

    # GI by name
    def test_nakayama_grand_jump_by_name(self):
        assert parse_grade("中山グランドジャンプ") == "GI"

    def test_nakayama_daishogai_by_name(self):
        assert parse_grade("中山大障害") == "GI"

    def test_nakayama_grand_jump_with_year_prefix(self):
        """Name contains keyword but doesn't startWith — uses 'in' check."""
        # _is_known_graded uses startsWith, so prefix won't match
        # This is expected: grade detected via icon or exact name only
        assert parse_grade("第39回中山グランドジャンプ") is None

    def test_nakayama_daishogai_with_year_prefix(self):
        assert parse_grade("第58回中山大障害") is None

    # GII by name
    def test_tokyo_high_jump_by_name(self):
        assert parse_grade("東京ハイジャンプ") == "GII"

    def test_hanshin_spring_jump_by_name(self):
        assert parse_grade("阪神スプリングジャンプ") == "GII"

    def test_tokyo_autumn_jump_by_name(self):
        assert parse_grade("東京オータムジャンプ") == "GII"

    def test_kyoto_high_jump_by_name(self):
        assert parse_grade("京都ハイジャンプ") == "GII"

    # GIII by name
    def test_kyoto_jump_by_name(self):
        assert parse_grade("京都ジャンプS") == "GIII"

    def test_hanshin_jump_by_name(self):
        assert parse_grade("阪神ジャンプS") == "GIII"

    def test_niigata_jump_by_name(self):
        assert parse_grade("新潟ジャンプS") == "GIII"

    # Flat races without icons should NOT get a grade via name fallback
    def test_known_flat_race_without_icon_returns_grade(self):
        """Known graded race names trigger grade even without icon."""
        assert parse_grade("アンタレスS") == "GIII"
        assert parse_grade("マイラーズC") == "GII"

    def test_mainichi_hai_is_known_gii(self):
        """毎日杯 is in _KNOWN_GRADED GII list, so name fallback gives GII."""
        assert parse_grade("毎日杯") == "GII"

    def test_ordinary_race_without_icon_returns_none(self):
        assert parse_grade("3歳未勝利") is None

    def test_partial_keyword_no_false_positive(self):
        """'グランドジャンプ' without '中山' must not trigger GI."""
        # The fallback checks for '中山グランドジャンプ' as a whole substring,
        # so a name containing only 'グランドジャンプ' must not match.
        assert parse_grade("グランドジャンプ特別") is None

    def test_similar_name_no_false_positive(self):
        """'大障害' without '中山' must not trigger GI."""
        assert parse_grade("大障害特別") is None


# ---------------------------------------------------------------------------
# Class 4: Icon priority over name-based fallback
# ---------------------------------------------------------------------------

class TestIconPriorityOverNameFallback:
    """Icons must win when both icon and name-based signals are present."""

    def test_gi_icon_on_gii_name_cross_validates(self):
        """GI icon on GII-named race → name is GII not GI, so falls back to GII."""
        assert parse_grade("東京ハイジャンプ", "Icon_GradeType1") == "GII"

    def test_gii_icon_on_gi_jumps_name_cross_validates(self):
        """GII icon on GI-named race → cross-validation: name is GI, not GII, so falls back to name."""
        # 中山グランドジャンプ is known GI; GII icon won't match GII list, so name fallback gives GI
        assert parse_grade("中山グランドジャンプ", "Icon_GradeType2") == "GI"

    def test_giii_icon_on_gi_name_cross_validates(self):
        """GIII icon on GI-named race → name fallback gives GI."""
        assert parse_grade("中山大障害", "Icon_GradeType3") == "GI"

    def test_jumps_gi_icon_on_gii_name_cross_validates(self):
        """GI icon on GII-named race → cross-validation gives GII."""
        assert parse_grade("阪神スプリングジャンプ", "Icon_GradeType15") == "GII"

    def test_jumps_gii_icon_on_giii_name_cross_validates(self):
        """GII icon on GIII-named race → cross-validation against GII list fails, name fallback gives GIII."""
        assert parse_grade("京都ジャンプS", "Icon_GradeType16") == "GIII"

    def test_name_fallback_not_used_when_icon_already_set_gi(self):
        """When Icon_GradeType1 sets GI, the name-fallback block is skipped."""
        # Both signals agree on GI; grade must be exactly "GI" (not set twice)
        result = parse_grade("中山グランドジャンプ", "Icon_GradeType1")
        assert result == "GI"

    def test_name_fallback_not_used_when_icon_already_set_gii(self):
        result = parse_grade("東京ハイジャンプ", "Icon_GradeType2")
        assert result == "GII"


# ---------------------------------------------------------------------------
# Class 5: Edge cases and robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Null/absent elements, empty strings, multiple icon classes."""

    def test_no_race_name_element_grade_is_none(self):
        """Page without .RaceName should return grade=None."""
        html = "<html><body><div class='Other'>無関係</div></body></html>"
        result = parse_race_card(html)
        assert result["race_info"]["grade"] is None

    def test_empty_race_name_grade_is_none(self):
        html = "<html><body><div class='RaceName'></div></body></html>"
        result = parse_race_card(html)
        assert result["race_info"]["grade"] is None

    def test_unknown_icon_class_returns_none(self):
        """Icon_GradeType99 or similar unknown class must not assign a grade."""
        assert parse_grade("大レース", "Icon_GradeType99") is None

    def test_icon_grade_type4_not_recognised(self):
        """Type4 is not a valid grade icon; should fall through to name or None."""
        assert parse_grade("謎のレース", "Icon_GradeType4") is None

    def test_race_name_preserved_with_icon(self):
        """race_name text is captured even when icon span is present."""
        html = _make_html("大阪杯", "Icon_GradeType1")
        result = parse_race_card(html)
        assert "大阪杯" in result["race_info"]["raceName"]

    def test_multiple_icon_classes_unknown_name_none(self):
        """Unknown name with GI+GII icons → cross-validation fails, grade is None."""
        html = """
        <html><body>
          <div class="RaceName">
            <span class="Icon_GradeType1 Icon_GradeType2"></span>
            テストレース
          </div>
        </body></html>
        """
        result = parse_race_card(html)
        assert result["race_info"]["grade"] is None

    def test_unicode_race_name_does_not_crash(self):
        """Emoji and unusual Unicode in race names must not raise."""
        result = parse_race_card(_make_html("🐎レース🏆"))
        assert result["race_info"]["grade"] is None

    def test_very_long_race_name_handled(self):
        """Very long name with keyword NOT at start → no match (startsWith)."""
        name = "A" * 500 + "中山グランドジャンプ"
        assert parse_grade(name) is None

    def test_grade_defaults_to_none_in_info_dict(self):
        """Baseline: newly constructed info dict has grade=None."""
        html = "<html><body></body></html>"
        result = parse_race_card(html)
        assert result["race_info"]["grade"] is None

    def test_jumps_gii_all_four_keywords_covered(self):
        """All four GII name-fallback keywords must each independently work."""
        keywords = ["東京ハイジャンプ", "阪神スプリングジャンプ", "東京オータムジャンプ", "京都ハイジャンプ"]
        for kw in keywords:
            assert parse_grade(kw) == "GII", f"Expected GII for name '{kw}'"

    def test_jumps_giii_all_three_keywords_covered(self):
        """All three GIII name-fallback keywords must each independently work."""
        keywords = ["京都ジャンプS", "阪神ジャンプS", "新潟ジャンプS"]
        for kw in keywords:
            assert parse_grade(kw) == "GIII", f"Expected GIII for name '{kw}'"

    def test_jumps_gi_both_keywords_covered(self):
        """Both GI name-fallback keywords must each independently work."""
        keywords = ["中山グランドジャンプ", "中山大障害"]
        for kw in keywords:
            assert parse_grade(kw) == "GI", f"Expected GI for name '{kw}'"

    def test_all_new_jumps_icon_classes_gi_with_known_name(self):
        """GI icons require cross-validation with known GI names."""
        for cls in ("Icon_GradeType15", "Icon_GradeType12"):
            assert parse_grade("中山グランドジャンプ", cls) == "GI", f"Failed for {cls}"
            assert parse_grade("障害テスト", cls) is None  # Unknown name → rejected

    def test_all_new_jumps_icon_classes_gii_with_known_name(self):
        """GII icons require cross-validation with known race name."""
        for cls in ("Icon_GradeType16", "Icon_GradeType13"):
            assert parse_grade("東京ハイジャンプ", cls) == "GII", f"Failed for {cls}"
            assert parse_grade("障害テスト", cls) is None  # Unknown name → rejected

    def test_all_new_jumps_icon_classes_giii_with_known_name(self):
        for cls in ("Icon_GradeType17", "Icon_GradeType14"):
            assert parse_grade("京都ジャンプS", cls) == "GIII", f"Failed for {cls}"
            assert parse_grade("障害テスト", cls) is None  # Unknown name → rejected


# ---------------------------------------------------------------------------
# Class 6: Date extraction in _parse_race_info() (Change 2)
# ---------------------------------------------------------------------------


def _make_html_with_date(date_text: str, extra_body: str = "") -> str:
    """Build a minimal page HTML that embeds *date_text* in the body."""
    return f"""
    <html><body>
      <div class="RaceName">テストレース</div>
      <p>{date_text}</p>
      {extra_body}
    </body></html>
    """


def _extract_date(html: str) -> str:
    """Helper: return race_info['date'] from parse_race_card for the given HTML."""
    result = parse_race_card(html)
    return result["race_info"]["date"]


class TestDateExtraction:
    """_parse_race_info() extracts race date from page text.

    The regex r"(\\d{4})年(\\d{1,2})月(\\d{1,2})日" is applied to the full
    page text.  The extracted date is stored as "YYYYMMDD" (zero-padded).
    """

    # ------------------------------------------------------------------
    # Test 1: standard full date → YYYYMMDD
    # ------------------------------------------------------------------

    def test_full_date_extracted_and_formatted(self):
        """'2026年5月10日' in the page body → date = '20260510'."""
        html = _make_html_with_date("2026年5月10日")
        assert _extract_date(html) == "20260510"

    # ------------------------------------------------------------------
    # Test 2: single-digit month and day are zero-padded
    # ------------------------------------------------------------------

    def test_single_digit_month_and_day_zero_padded(self):
        """'2026年1月3日' → '20260103' (month 1 → 01, day 3 → 03)."""
        html = _make_html_with_date("2026年1月3日")
        assert _extract_date(html) == "20260103"

    def test_single_digit_month_two_digit_day(self):
        """'2025年3月15日' → '20250315'."""
        html = _make_html_with_date("2025年3月15日")
        assert _extract_date(html) == "20250315"

    def test_two_digit_month_single_digit_day(self):
        """'2026年12月5日' → '20261205'."""
        html = _make_html_with_date("2026年12月5日")
        assert _extract_date(html) == "20261205"

    def test_two_digit_month_and_day(self):
        """'2026年11月22日' → '20261122'."""
        html = _make_html_with_date("2026年11月22日")
        assert _extract_date(html) == "20261122"

    # ------------------------------------------------------------------
    # Test 3: no date in page → date = "" (empty string)
    # ------------------------------------------------------------------

    def test_no_date_in_page_returns_empty_string(self):
        """When the page contains no Japanese date pattern, date must be ''."""
        html = "<html><body><div class='RaceName'>テストレース</div></body></html>"
        assert _extract_date(html) == ""

    def test_partial_date_pattern_not_matched(self):
        """'2026年5月' (missing 日) must not produce a date."""
        html = _make_html_with_date("2026年5月")
        assert _extract_date(html) == ""

    def test_western_date_format_not_matched(self):
        """'2026/05/10' (slash-separated) must not be matched by the 年月日 regex."""
        html = _make_html_with_date("2026/05/10")
        assert _extract_date(html) == ""

    # ------------------------------------------------------------------
    # Test 4: multiple dates → first match used
    # ------------------------------------------------------------------

    def test_multiple_dates_first_match_used(self):
        """When the page contains two dates, the first one (in document order)
        must be the value stored in race_info['date']."""
        html = _make_html_with_date(
            "2026年4月12日",
            extra_body="<p>次回は2026年4月19日です</p>",
        )
        # Both are present; the regex re.search() finds the earliest match.
        assert _extract_date(html) == "20260412"

    def test_multiple_dates_different_years_first_wins(self):
        """First date in document order wins, even if a later date has a
        different year."""
        html = _make_html_with_date(
            "2025年12月28日",
            extra_body="<p>翌年は2026年1月4日</p>",
        )
        assert _extract_date(html) == "20251228"

    # ------------------------------------------------------------------
    # Robustness / edge cases
    # ------------------------------------------------------------------

    def test_date_in_race_data_section(self):
        """Date embedded inside a .RaceData01 block (realistic page structure)
        should still be found because we search soup.get_text()."""
        html = """
        <html><body>
          <div class="RaceName">テストレース</div>
          <div class="RaceData01">15:45発走 / 芝2000m(右) / 馬場:良</div>
          <div class="RaceData02">2026年6月7日 東京 11R</div>
        </body></html>
        """
        assert _extract_date(html) == "20260607"

    def test_empty_page_does_not_crash(self):
        """Completely empty HTML must not raise an exception."""
        assert _extract_date("") == ""

    def test_date_field_type_is_str(self):
        """race_info['date'] must always be a str (never None or int)."""
        html = _make_html_with_date("2026年8月1日")
        result = parse_race_card(html)
        assert isinstance(result["race_info"]["date"], str)
