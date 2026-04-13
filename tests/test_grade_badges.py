"""TDD tests for grade badge assignment in preview.html.

Verifies that GRADE_RACES mapping matches JRA official graded race classifications.
Uses netkeiba-confirmed data as ground truth.
"""
from __future__ import annotations

import pytest

# Mirror the GRADE_RACES from preview.html for testability
GRADE_RACES = {
    "GI": [
        "フェブラリーS","高松宮記念","大阪杯","桜花賞","皐月賞","天皇賞","NHKマイル",
        "ヴィクトリアマイル","オークス","日本ダービー","安田記念","宝塚記念",
        "スプリンターズS","秋華賞","菊花賞","エリザベス女王杯","マイルCS",
        "ジャパンC","チャンピオンズC","有馬記念","ホープフルS",
        "朝日杯FS","阪神JF",
    ],
    "GII": [
        "日経新春杯","アメリカJCC","東海S","京都記念","共同通信杯",
        "京都牝馬S","中山記念","阪急杯","チューリップ賞","弥生賞",
        "金鯱賞","フィリーズR","スプリングS","日経賞",
        "ニュージーランドT","アーリントンC","サウジアラビアRC",
        "フローラS","青葉賞","京王杯SC","目黒記念",
        "セントウルS","ローズS","オールカマー","京都大賞典",
        "府中牝馬S","アルゼンチン共和国杯","ステイヤーズS",
        "札幌記念","セントライト記念","神戸新聞杯","富士S",
        "スワンS","デイリー杯","京阪杯","阪神カップ","産経大阪杯",
    ],
    "GIII": [
        "中山金杯","京都金杯","シンザン記念","フェアリーS","京成杯",
        "愛知杯","東京新聞杯","きさらぎ賞","小倉大賞典","ダイヤモンドS",
        "中山牝馬S","フラワーC","ファルコンS","阪神大賞典",
        "ダービーCT","マーガレットS","福島牝馬S","新潟大賞典",
        "京都新聞杯","葵S","鳴尾記念","エプソムC","函館スプリントS",
        "CBC賞","ラジオNIKKEI賞","プロキオンS","七夕賞","函館記念",
        "中京記念","小倉記念","関屋記念","エルムS","北九州記念",
        "札幌2歳S","キーンランドC","新潟記念","紫苑S","レパードS",
        "シリウスS","みやこS","武蔵野S","東京スポーツ杯",
        "京都2歳S","ターコイズS","カペラS","中日新聞杯",
        "阪神C","チャーチルC","毎日杯","マーチS",
    ],
}


def lookup_grade(name: str) -> str:
    """Lookup grade for a race name, matching preview.html logic."""
    for grade, keywords in GRADE_RACES.items():
        for kw in keywords:
            if kw in name:
                return grade
    return ""


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
        assert lookup_grade("毎日杯") == "GIII"

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
