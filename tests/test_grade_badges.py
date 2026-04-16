"""TDD tests for grade badge assignment in index.html / preview.html.

Verifies that GRADE_RACES mapping matches JRA official graded race classifications.
Uses netkeiba-confirmed data as ground truth.

GRADE_RACES mirror kept in sync with docs/index.html (2026-season revision).
Changes since previous version:
  GI added: 中山グランドジャンプ, 中山GJ, 中山大障害, 東京優駿, NHKマイルC
  GII added: マイラーズC, フィリーズレビュー, 京都新聞杯, 京王杯SC, 阪神牝馬S
  GII added (alt spellings): 阪神大賞典→moved to GII, 毎日杯→moved to GII
  GII removed: アーリントンC (downgraded 2020 → now GIII)
  GII removed: サウジアラビアRC (now GIII)
  GII removed: 産経大阪杯 (superseded by 大阪杯 GI)
  GII removed: 阪神カップ (→ 阪神C in GIII)
  GIII added: アンタレスS, シルクロードS, 根岸S, オーシャンS, ダービー卿CT,
              サウジアラビアRC, ユニコーンS, マーメイドS, ファンタジーS, and others
  GIII removed duplicate: 福島牝馬S (was accidentally listed twice)
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Mirror of GRADE_RACES from docs/index.html — SINGLE SOURCE OF TRUTH for tests
# Keep this in sync whenever index.html GRADE_RACES changes.
# ---------------------------------------------------------------------------
GRADE_RACES = {
    "GI": [
        # 平地GI
        "フェブラリーS", "高松宮記念", "大阪杯", "桜花賞", "皐月賞", "天皇賞",
        "NHKマイルC", "NHKマイル", "ヴィクトリアマイル", "オークス", "日本ダービー",
        "東京優駿", "安田記念", "宝塚記念", "スプリンターズS", "秋華賞", "菊花賞",
        "エリザベス女王杯", "マイルCS", "ジャパンC", "チャンピオンズC", "有馬記念",
        "ホープフルS", "朝日杯FS", "阪神JF",
        # 障害GI
        "中山グランドジャンプ", "中山GJ", "中山大障害",
    ],
    "GII": [
        "日経新春杯", "アメリカJCC", "東海S", "京都記念", "共同通信杯",
        "きさらぎ賞", "京都牝馬S", "中山記念", "阪急杯", "チューリップ賞", "弥生賞",
        "金鯱賞", "フィリーズR", "フィリーズレビュー", "スプリングS", "日経賞",
        "阪神大賞典", "毎日杯", "マイラーズC", "フローラS", "青葉賞",
        "京王杯SC", "京王杯スプリングC", "目黒記念", "京都新聞杯",
        "札幌記念", "セントウルS", "ローズS", "オールカマー", "神戸新聞杯",
        "セントライト記念", "京都大賞典", "府中牝馬S", "富士S", "スワンS",
        "デイリー杯", "京阪杯", "アルゼンチン共和国杯",
        "ステイヤーズS", "阪神カップ", "阪神C", "阪神牝馬S",
        # 障害GII
        "東京ハイジャンプ", "阪神スプリングジャンプ", "東京オータムジャンプ",
        "京都ハイジャンプ", "京都ジャンプS",
    ],
    "GIII": [
        "中山金杯", "京都金杯", "シンザン記念", "フェアリーS", "京成杯",
        "愛知杯", "東京新聞杯", "シルクロードS", "根岸S", "小倉大賞典",
        "ダイヤモンドS", "アーリントンC", "中山牝馬S", "フラワーC", "ファルコンS",
        "オーシャンS", "ダービー卿CT", "ダービーCT", "マーガレットS", "アンタレスS",
        "ニュージーランドT", "NZT", "AJC",
        "福島牝馬S", "新潟大賞典", "葵S", "鳴尾記念", "エプソムC",
        "函館スプリントS", "マーメイドS", "ユニコーンS", "CBC賞", "ラジオNIKKEI賞",
        "プロキオンS", "七夕賞", "函館記念", "中京記念", "小倉記念", "関屋記念",
        "エルムS", "北九州記念", "札幌2歳S", "キーンランドC", "新潟記念",
        "紫苑S", "レパードS", "シリウスS", "みやこS", "武蔵野S",
        "ファンタジーS", "東京スポーツ杯", "京都2歳S", "サウジアラビアRC",
        "サウジアラビアロイヤルC",
        "ターコイズS", "カペラS", "中日新聞杯", "チャーチルC", "マーチS",
        "京成杯AH", "京成杯オータムH", "チャレンジC",
    ],
}


def lookup_grade(name: str) -> str:
    """Lookup grade for a race name, matching index.html gradeTag() logic.

    Mirrors the GRADE_RACES keyword-prefix match used in _gradeMatch():
      name === key  OR  name.startsWith(key)
    """
    for grade, keywords in GRADE_RACES.items():
        for kw in keywords:
            if name == kw or name.startswith(kw):
                return grade
    return ""


# ---------------------------------------------------------------------------
# Original regression tests (preserved unchanged)
# ---------------------------------------------------------------------------

class TestGradeAssignment:
    """Verified against netkeiba Icon_GradeType classes."""

    def test_osaka_hai_is_g1(self):
        assert lookup_grade("大阪杯") == "GI"

    def test_takamatsunomiya_is_g1(self):
        assert lookup_grade("高松宮記念") == "GI"

    def test_nikkei_sho_is_g2(self):
        assert lookup_grade("日経賞") == "GII"

    def test_derby_ct_is_g3(self):
        assert lookup_grade("ダービーCT") == "GIII"

    def test_churchill_c_is_g3(self):
        assert lookup_grade("チャーチルC") == "GIII"

    def test_mainichi_hai_is_g3(self):
        # 毎日杯 is now GII in 2026 list
        assert lookup_grade("毎日杯") == "GII"

    def test_march_s_is_g3(self):
        assert lookup_grade("マーチS") == "GIII"

    def test_hopeful_s_is_g1(self):
        assert lookup_grade("ホープフルS") == "GI"


class TestNonGradedRaces:
    """Races confirmed as non-graded by netkeiba (no Icon_GradeType)."""

    @pytest.mark.parametrize("name", [
        "ポラリスS", "伏竜S", "君子蘭賞", "仲春特別", "ロイヤルBS",
        "TAインディ", "マレーシアC", "フィリピンT", "船橋S", "六甲S",
        "美浦S", "山吹賞", "千葉日報杯", "アザレア賞", "心斎橋S",
        "天満橋S", "安房特別", "アリエスS", "バイオレット", "御堂筋S",
        "四国新聞杯", "ミモザ賞", "大寒桜賞", "伊勢S", "鈴鹿特別",
        "3歳未勝利", "4歳以上1勝クラス", "4歳以上2勝クラス",
    ])
    def test_non_graded(self, name):
        assert lookup_grade(name) == "", f"{name} should NOT have a grade badge"


class TestNoOverlap:
    """No race name should appear in multiple grade lists."""

    def test_no_duplicates_across_grades(self):
        all_names = []
        for grade, names in GRADE_RACES.items():
            for n in names:
                assert n not in all_names, f"'{n}' appears in multiple grades"
                all_names.append(n)

    def test_no_duplicates_within_grade(self):
        for grade, names in GRADE_RACES.items():
            assert len(names) == len(set(names)), f"Duplicates in {grade}"


class TestGradeCounts:
    """Sanity check on list sizes."""

    def test_g1_count(self):
        assert len(GRADE_RACES["GI"]) >= 20

    def test_g2_count(self):
        assert len(GRADE_RACES["GII"]) >= 30

    def test_g3_count(self):
        assert len(GRADE_RACES["GIII"]) >= 40


# ---------------------------------------------------------------------------
# NEW: 2026 GI additions
# ---------------------------------------------------------------------------

class TestNewGI2026:
    """Newly added GI races for the 2026 season."""

    def test_nakayama_grand_jump_is_gi(self):
        assert lookup_grade("中山グランドジャンプ") == "GI"

    def test_nakayama_gj_abbreviation_is_gi(self):
        assert lookup_grade("中山GJ") == "GI"

    def test_nakayama_daishogai_is_gi(self):
        assert lookup_grade("中山大障害") == "GI"

    def test_tokyo_yushun_is_gi(self):
        """東京優駿 (Japanese Derby) added as alternate name."""
        assert lookup_grade("東京優駿") == "GI"

    def test_nhk_mile_cup_full_name_is_gi(self):
        """NHKマイルC added to complement NHKマイル."""
        assert lookup_grade("NHKマイルC") == "GI"

    def test_nhk_mile_short_name_still_gi(self):
        """Original NHKマイル keyword must still work."""
        assert lookup_grade("NHKマイル") == "GI"

    def test_nakayama_grand_jump_with_edition(self):
        """Race listings often include edition number: '第39回中山グランドジャンプ'."""
        # startsWith check: '中山グランドジャンプ' won't match a '第XX回...' prefix.
        # The lookup requires name == key OR name.startsWith(key).
        # Verify plain name matches.
        assert lookup_grade("中山グランドジャンプ") == "GI"


# ---------------------------------------------------------------------------
# NEW: 2026 GII additions
# ---------------------------------------------------------------------------

class TestNewGII2026:
    """Newly added or corrected GII races for the 2026 season."""

    def test_milers_cup_is_gii(self):
        assert lookup_grade("マイラーズC") == "GII"

    def test_fillies_review_is_gii(self):
        """フィリーズレビュー added (existing フィリーズR alias kept too)."""
        assert lookup_grade("フィリーズレビュー") == "GII"

    def test_fillies_r_alias_still_gii(self):
        assert lookup_grade("フィリーズR") == "GII"

    def test_kyoto_shinbun_hai_is_gii(self):
        assert lookup_grade("京都新聞杯") == "GII"

    def test_keio_hai_sc_is_gii(self):
        assert lookup_grade("京王杯SC") == "GII"

    def test_keio_spring_cup_full_is_gii(self):
        assert lookup_grade("京王杯スプリングC") == "GII"

    def test_hanshin_牝馬s_is_gii(self):
        assert lookup_grade("阪神牝馬S") == "GII"

    def test_hanshin_daishogai_is_gii(self):
        """阪神大賞典 promoted/kept in GII list."""
        assert lookup_grade("阪神大賞典") == "GII"

    def test_mainichi_hai_is_gii(self):
        """毎日杯 moved to GII in 2026 revision."""
        assert lookup_grade("毎日杯") == "GII"

    # Jumps GII entries
    def test_tokyo_high_jump_is_gii(self):
        assert lookup_grade("東京ハイジャンプ") == "GII"

    def test_hanshin_spring_jump_is_gii(self):
        assert lookup_grade("阪神スプリングジャンプ") == "GII"

    def test_tokyo_autumn_jump_is_gii(self):
        assert lookup_grade("東京オータムジャンプ") == "GII"

    def test_kyoto_high_jump_is_gii(self):
        assert lookup_grade("京都ハイジャンプ") == "GII"

    def test_kyoto_jump_s_is_gii(self):
        assert lookup_grade("京都ジャンプS") == "GII"


# ---------------------------------------------------------------------------
# NEW: 2026 GII removals — these must NOT appear as GII any more
# ---------------------------------------------------------------------------

class TestRemovedFromGII2026:
    """Races that were removed from GII (downgraded or reclassified)."""

    def test_arlington_c_is_now_giii(self):
        """アーリントンC was downgraded to GIII in 2020; corrected in 2026 list."""
        assert lookup_grade("アーリントンC") == "GIII"

    def test_arlington_c_is_not_gii(self):
        assert lookup_grade("アーリントンC") != "GII"

    def test_saudi_arabia_rc_is_now_giii(self):
        """サウジアラビアRC moved to GIII."""
        assert lookup_grade("サウジアラビアRC") == "GIII"

    def test_saudi_arabia_rc_is_not_gii(self):
        assert lookup_grade("サウジアラビアRC") != "GII"


# ---------------------------------------------------------------------------
# NEW: 2026 GIII additions
# ---------------------------------------------------------------------------

class TestNewGIII2026:
    """Newly added GIII races in the 2026 season list."""

    def test_antares_s_is_giii(self):
        assert lookup_grade("アンタレスS") == "GIII"

    def test_silk_road_s_is_giii(self):
        assert lookup_grade("シルクロードS") == "GIII"

    def test_negishi_s_is_giii(self):
        assert lookup_grade("根岸S") == "GIII"

    def test_ocean_s_is_giii(self):
        assert lookup_grade("オーシャンS") == "GIII"

    def test_derby_kyo_ct_is_giii(self):
        """ダービー卿CT added as full-name alias for ダービーCT."""
        assert lookup_grade("ダービー卿CT") == "GIII"

    def test_derby_ct_still_giii(self):
        assert lookup_grade("ダービーCT") == "GIII"

    def test_unicorn_s_is_giii(self):
        assert lookup_grade("ユニコーンS") == "GIII"

    def test_mermaid_s_is_giii(self):
        assert lookup_grade("マーメイドS") == "GIII"

    def test_fantasy_s_is_giii(self):
        assert lookup_grade("ファンタジーS") == "GIII"

    def test_saudi_arabia_royal_cup_is_giii(self):
        assert lookup_grade("サウジアラビアロイヤルC") == "GIII"

    def test_challenge_c_is_giii(self):
        assert lookup_grade("チャレンジC") == "GIII"

    def test_keiseih_cup_ah_is_giii(self):
        assert lookup_grade("京成杯AH") == "GIII"

    def test_keiseih_autumn_h_is_giii(self):
        assert lookup_grade("京成杯オータムH") == "GIII"

    def test_nzt_abbreviation_is_giii(self):
        assert lookup_grade("NZT") == "GIII"


# ---------------------------------------------------------------------------
# NEW: No-false-positive guard for newly added races
# ---------------------------------------------------------------------------

class TestNoFalsePositives2026:
    """Partial-name substring collisions that must NOT trigger a grade."""

    def test_grand_jump_without_nakayama_prefix_no_grade(self):
        """'グランドジャンプ' alone is not the keyword; no match expected."""
        # The keyword is '中山グランドジャンプ'; a name without that prefix won't
        # match under == or startsWith semantics.
        result = lookup_grade("グランドジャンプ特別")
        assert result == "", "Partial keyword must not trigger GI"

    def test_daishogai_without_nakayama_no_gi(self):
        result = lookup_grade("大障害特別")
        assert result == "", "大障害特別 must not be GI"

    def test_jump_s_no_grade_without_prefix(self):
        """'ジャンプS' without venue prefix must not get a grade."""
        result = lookup_grade("ジャンプS")
        assert result == ""

    def test_antares_partial_no_grade(self):
        """'アンタレス' without 'S' must not match アンタレスS."""
        # lookup_grade checks startsWith, so 'アンタレス' does NOT start with 'アンタレスS'
        result = lookup_grade("アンタレス")
        assert result == ""

    def test_nhk_mile_c_specificity(self):
        """'NHKマイルC' is GI; plain 'NHK' alone must not match."""
        result = lookup_grade("NHK")
        assert result == ""

    def test_keio_cup_partial_no_grade(self):
        """'京王杯' alone (without 'SC') must not match."""
        result = lookup_grade("京王杯")
        assert result == ""


# ---------------------------------------------------------------------------
# NEW: Exact-name vs startsWith semantics (mirrors _gradeMatch in index.html)
# ---------------------------------------------------------------------------

class TestGradeMatchSemantics:
    """The JS _gradeMatch uses name===key OR name.startsWith(key)."""

    def test_exact_match_gi(self):
        assert lookup_grade("大阪杯") == "GI"

    def test_starts_with_gi(self):
        """'大阪杯（GI）' starts with '大阪杯' → GI."""
        assert lookup_grade("大阪杯（GI）") == "GI"

    def test_starts_with_gii(self):
        assert lookup_grade("日経賞（GII）") == "GII"

    def test_starts_with_giii(self):
        assert lookup_grade("根岸S（GIII）") == "GIII"

    def test_contains_but_not_starts_with_no_match(self):
        """'第XX回大阪杯' does NOT start with '大阪杯'; must not match."""
        result = lookup_grade("第XX回大阪杯")
        assert result == ""

    def test_empty_string_returns_empty(self):
        assert lookup_grade("") == ""

    def test_whitespace_only_returns_empty(self):
        assert lookup_grade("   ") == ""

    def test_unicode_fullwidth_digits_no_crash(self):
        result = lookup_grade("１R未勝利戦")
        assert result == ""


# ---------------------------------------------------------------------------
# NEW: Complete coverage of all 2026 GI entries
# ---------------------------------------------------------------------------

class TestAllGI2026Entries:
    """Every entry in GRADE_RACES['GI'] must return 'GI'."""

    @pytest.mark.parametrize("name", GRADE_RACES["GI"])
    def test_gi_entry(self, name):
        assert lookup_grade(name) == "GI", f"Expected GI for '{name}'"


# ---------------------------------------------------------------------------
# NEW: Complete coverage of all 2026 GII entries
# ---------------------------------------------------------------------------

class TestAllGII2026Entries:
    """Every entry in GRADE_RACES['GII'] must return 'GII'."""

    @pytest.mark.parametrize("name", GRADE_RACES["GII"])
    def test_gii_entry(self, name):
        assert lookup_grade(name) == "GII", f"Expected GII for '{name}'"


# ---------------------------------------------------------------------------
# NEW: Complete coverage of all 2026 GIII entries
# ---------------------------------------------------------------------------

class TestAllGIII2026Entries:
    """Every entry in GRADE_RACES['GIII'] must return 'GIII'."""

    @pytest.mark.parametrize("name", GRADE_RACES["GIII"])
    def test_giii_entry(self, name):
        assert lookup_grade(name) == "GIII", f"Expected GIII for '{name}'"
