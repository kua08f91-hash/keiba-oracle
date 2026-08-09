"""Individual prediction factor calculators.

Each function returns a score from 0 to 100.

Updated: Incorporates market odds, horse weight change, trainer ratings,
and removes redundant bloodline factor.
"""
from __future__ import annotations

import re
from datetime import date as _date_type
from .sire_data import get_sire_profile

# Top jockeys with their overall ability ratings
JOCKEY_RATINGS = {
    "ルメール": 96, "川田": 94, "横山武": 89, "戸崎": 88, "福永": 88,
    "Ｃ.ルメール": 96, "Ｍ.デムーロ": 86, "松山": 85, "吉田隼": 82,
    "横山和": 84, "岩田望": 82, "武豊": 85, "田辺": 80, "石橋脩": 79,
    "三浦": 78, "津村": 77, "丸山": 78, "池添": 80, "浜中": 82,
    "藤岡佑": 81, "鮫島駿": 79, "坂井": 83, "菅原明": 78, "団野": 78,
    "西村淳": 77, "横山典": 80, "内田博": 78, "石川": 77, "大野": 76,
    "荻野極": 76, "丹内": 75, "柴田善": 76, "松岡": 75, "木幡巧": 74,
    "原田和": 74, "角田和": 75, "舟山": 73, "吉田豊": 76, "横山琉": 75,
    "菱田": 77, "岩田康": 79, "幸": 77, "和田竜": 78, "酒井": 76,
    "秋山真": 76, "北村友": 78, "藤岡康": 77, "斎藤": 76, "永野": 74,
    "Ｄ.レーン": 92, "Ｒ.ムーア": 96, "ムーア": 96,
    "デムーロ": 86, "レーン": 92, "モレイラ": 96,
    "長浜": 70, "石田": 72, "杉原": 74, "西村太": 74,
    "上里": 70, "小崎": 74, "小林凌": 72,
    "柴田裕": 72, "田口": 72, "国分恭": 75,
    "藤懸": 74, "古川吉": 75, "泉谷": 76, "今村": 75,
    "西塚": 70, "松本大": 71, "富田": 74, "水口": 70,
    "小沢": 70, "笹川": 71, "亀田": 70, "佐々木": 74,
    "ディー": 86,
}

# Top trainer ratings
TRAINER_RATINGS = {
    "矢作": 92, "友道": 90, "国枝": 90, "堀": 90, "藤原英": 88,
    "中内田": 88, "手塚": 87, "木村": 86, "萩原": 86, "須貝": 85,
    "池江": 85, "音無": 84, "安田隆": 86, "田中博": 83, "高野": 84,
    "杉山晴": 83, "高橋亮": 83, "石橋": 82, "尾形": 82, "鹿戸": 82,
    "宮田": 84, "武幸": 82, "松永幹": 82, "大竹": 81, "藤岡健": 81,
    "昆": 82, "栗田": 80, "上村": 80, "小林真": 80, "古賀慎": 80,
    "田村": 80, "安田翔": 81, "嘉藤": 78, "金成": 78, "渡辺": 78,
    "堀内": 78, "中舘": 78, "高木": 82, "池上": 79, "竹内": 79,
    "西田": 78, "上原博": 78, "高柳瑞": 78, "林": 79, "西園": 80,
    "松下": 78, "鈴木伸": 78, "宮本": 79, "清水": 78,
    "小崎": 78, "安田": 80,
}


def _get_distance_category(distance: int) -> str:
    if distance <= 1400:
        return "sprint"
    elif distance <= 1800:
        return "mile"
    elif distance <= 2200:
        return "intermediate"
    else:
        return "stayer"


def calc_market_score(odds: float | None, popularity: int | None, head_count: int) -> float:
    """Score based on market odds/popularity.

    This is the strongest predictor in horse racing.
    Uses a non-linear curve that gives strong separation for top favorites.

    JRA win rates by popularity:
      1人気 ~33%, 2人気 ~19%, 3人気 ~13%, 4人気 ~10%, 5人気 ~8%
    So scoring should reflect this steep dropoff.
    """
    import math

    if popularity is not None and popularity > 0:
        if head_count <= 0:
            head_count = 16

        # Non-linear scoring based on actual JRA win probabilities
        # 1人気=97, 2人気=88, 3人気=80, 4人気=72, 5人気=65, ...
        # Using exponential decay: score = 100 - k * ln(popularity)
        score = 100 - 22 * math.log(max(popularity, 1))
        # Ensure floor based on field size
        floor = max(15, 30 - head_count)
        return max(floor, min(98, score))

    if odds is not None and odds > 0:
        # Odds-based scoring with steeper curve
        # 1.5x → ~97, 3x → ~85, 10x → ~65, 50x → ~40, 100x → ~30
        score = 105 - 18 * math.log(max(odds, 1.1))
        return max(20, min(98, score))

    return 50.0  # Unknown - neutral (no penalty when odds unavailable)


def calc_course_affinity(sire_name: str, surface: str) -> float:
    """Score based on sire's affinity for the surface type."""
    profile = get_sire_profile(sire_name)
    surface_key = "芝" if surface == "芝" else "ダート"
    return float(profile.get(surface_key, 50))


def calc_distance_aptitude(sire_name: str, distance: int) -> float:
    """Score based on sire's distance aptitude."""
    profile = get_sire_profile(sire_name)
    category = _get_distance_category(distance)
    return float(profile.get(category, 50))


def calc_age_and_sex(age_str: str) -> float:
    """Score based on horse's age and sex."""
    match = re.search(r"[牡牝セ騸](\d+)", age_str)
    if not match:
        return 50.0

    age = int(match.group(1))
    age_scores = {2: 62, 3: 80, 4: 92, 5: 88, 6: 75, 7: 62, 8: 50, 9: 40}
    score = age_scores.get(age, max(30, 92 - (age - 4) * 12))

    if "牝" in age_str:
        score -= 3
    elif "セ" in age_str or "騸" in age_str:
        score -= 2

    return float(max(0, min(100, score)))


def calc_weight_carried(weight: float, all_weights: list) -> float:
    """Score based on weight carried relative to the field.

    Lower weight = advantage.
    """
    if not all_weights or weight <= 0:
        return 50.0

    valid_weights = [w for w in all_weights if w > 0]
    if not valid_weights:
        return 50.0

    min_weight = min(valid_weights)
    max_weight = max(valid_weights)

    if max_weight == min_weight:
        return 50.0

    # Inverted: lighter weight = higher score
    normalized = 1 - (weight - min_weight) / (max_weight - min_weight)
    return float(35 + normalized * 45)


def calc_jockey_ability(jockey_name: str) -> float:
    """Score based on jockey's overall ability rating."""
    if jockey_name in JOCKEY_RATINGS:
        return float(JOCKEY_RATINGS[jockey_name])

    for name, rating in JOCKEY_RATINGS.items():
        if len(name) >= 2 and (name in jockey_name or jockey_name in name):
            return float(rating)

    return 55.0


def calc_trainer_ability(trainer_name: str) -> float:
    """Score based on trainer's ability rating."""
    if trainer_name in TRAINER_RATINGS:
        return float(TRAINER_RATINGS[trainer_name])

    for name, rating in TRAINER_RATINGS.items():
        if len(name) >= 2 and (name in trainer_name or trainer_name in name):
            return float(rating)

    return 55.0


def calc_horse_weight_change(horse_weight_str: str) -> float:
    """Score based on horse weight change.

    Ideal: small change (-4 to +4).
    Bad: large decrease (<-10) or large increase (>+10).
    """
    if not horse_weight_str:
        return 50.0

    match = re.search(r"\(([+-]?\d+)\)", horse_weight_str)
    if not match:
        return 50.0

    change = int(match.group(1))
    abs_change = abs(change)

    if abs_change <= 2:
        return 80.0  # Very stable
    elif abs_change <= 4:
        return 75.0  # Good
    elif abs_change <= 6:
        return 65.0  # Acceptable
    elif abs_change <= 8:
        return 55.0  # Concerning
    elif abs_change <= 12:
        return 40.0  # Bad sign
    else:
        return 25.0  # Very bad sign


# ---------------------------------------------------------------------------
# Sire heavy track affinity ratings (higher = better on heavy/soft ground)
# ---------------------------------------------------------------------------
SIRE_HEAVY_TRACK = {
    # Good on heavy ground
    "ゴールドシップ": 80,
    "ステイゴールド": 78,
    "オルフェーヴル": 77,
    "ハーツクライ": 72,
    "スクリーンヒーロー": 73,
    "エピファネイア": 70,
    "ルーラーシップ": 72,
    "マンハッタンカフェ": 70,
    "キングカメハメハ": 65,
    "シンボリクリスエス": 68,
    "ドゥラメンテ": 65,
    "キタサンブラック": 68,
    "モーリス": 66,
    "ジャスタウェイ": 64,
    "リアルスティール": 63,
    # Bad on heavy ground (prefer good ground)
    "ロードカナロア": 38,
    "ダイワメジャー": 40,
    "ディープインパクト": 42,
    "サトノダイヤモンド": 45,
    "ヘニーヒューズ": 55,
    "パイロ": 55,
    "ドレフォン": 50,
    "マインドユアビスケッツ": 52,
    "コパノリッキー": 55,
    "ホッコータルマエ": 55,
    "サウスヴィグラス": 50,
    # Common BMS (broodmare sires)
    "サンデーサイレンス": 55,
    "ブライアンズタイム": 72,
    "トニービン": 68,
    "ノーザンテースト": 65,
    "フジキセキ": 48,
    "クロフネ": 58,
    "スペシャルウィーク": 60,
    "アグネスタキオン": 45,
    "タニノギムレット": 62,
    "ネオユニヴァース": 65,
    "ゼンノロブロイ": 58,
    "ウォーエンブレム": 60,
    "グラスワンダー": 70,
    "エルコンドルパサー": 62,
    "ストームキャット": 55,
    "デピュティミニスター": 60,
    "フレンチデピュティ": 58,
}

# Racecourse code to track name mapping
RACECOURSE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


def calc_past_performance(past_races: list) -> float:
    """Score 0-100 based on recent race results.

    Weights recent races more heavily:
      race 1 = 40%, race 2 = 25%, race 3 = 18%, race 4 = 10%, race 5 = 7%
    Finish position scoring (finer granularity):
      1st=100, 2nd=88, 3rd=78, 4th=68, 5th=58, 6th=48, 7th=42, 8th=38,
      9th=34, 10th+=25-30 (gradual decline)
    Also considers: consistency (all top-5 finishes = bonus)
    """
    if not past_races:
        return 45.0  # No data = slightly below average

    weights = [0.40, 0.25, 0.18, 0.10, 0.07]

    def pos_to_score(pos):
        if pos <= 0:
            return 38.0  # Unknown
        elif pos == 1:
            return 100.0
        elif pos == 2:
            return 88.0
        elif pos == 3:
            return 78.0
        elif pos == 4:
            return 68.0
        elif pos == 5:
            return 58.0
        elif pos <= 7:
            return 45.0
        elif pos <= 9:
            return 35.0
        elif pos <= 12:
            return 28.0
        else:
            return 20.0

    total_weight = 0.0
    total_score = 0.0
    positions = []

    for i, race in enumerate(past_races[:5]):
        w = weights[i] if i < len(weights) else 0.0
        pos = race.get("pos", 0)
        score = pos_to_score(pos)

        total_score += score * w
        total_weight += w
        if pos > 0:
            positions.append(pos)

    if total_weight <= 0:
        return 45.0

    base_score = total_score / total_weight

    # Consistency bonus: if all recent races are top-5, add bonus
    if len(positions) >= 3:
        top5_count = sum(1 for p in positions if p <= 5)
        if top5_count == len(positions):
            base_score = min(100, base_score + 8)
        elif top5_count >= len(positions) * 0.8:
            base_score = min(100, base_score + 4)

    # Winning streak bonus
    if len(positions) >= 2 and positions[0] == 1 and positions[1] <= 2:
        base_score = min(100, base_score + 5)

    return base_score


def calc_running_style_consistency(past_races: list) -> float:
    """Score based on running style consistency.

    A consistent running style (always 逃げ or always 差し) indicates
    the horse knows its role. Inconsistent style (changing each race)
    indicates struggle to find identity.

    Gracefully degrades: if no runningStyle data, returns neutral 50.0.
    """
    if not past_races:
        return 50.0

    styles = [r.get("runningStyle", "") for r in past_races[:5] if r.get("runningStyle")]
    if len(styles) < 2:
        return 50.0

    from collections import Counter
    counts = Counter(styles)
    most_common_count = counts.most_common(1)[0][1]
    consistency_ratio = most_common_count / len(styles)

    # 100% consistent = 80, 50% = 50 (baseline), less = penalty
    return 30.0 + 50.0 * consistency_ratio


def calc_speed_figure(past_races: list, target_distance: int) -> float:
    """Score based on speed figures from finish times at similar distances.

    Converts M:SS.f times to seconds, compares to distance benchmarks.
    Lower times = higher score. Gracefully degrades without time data.
    """
    if not past_races or target_distance <= 0:
        return 50.0

    # Benchmark times for distances (seconds per meter, rough estimates)
    # Turf: 2400m ≈ 144s → 0.060 s/m; 1600m ≈ 96s → 0.060 s/m
    # Dirt: slightly slower
    def time_to_sec(t):
        try:
            parts = t.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except (ValueError, IndexError):
            pass
        return None

    matching_speeds = []
    for race in past_races[:5]:
        dist = race.get("distance", 0)
        time_str = race.get("finishTime", "")
        if dist <= 0 or not time_str or abs(dist - target_distance) > 400:
            continue
        secs = time_to_sec(time_str)
        if secs is None or secs <= 0:
            continue
        # Speed in m/s
        speed = dist / secs
        matching_speeds.append(speed)

    if not matching_speeds:
        return 50.0

    avg_speed = sum(matching_speeds) / len(matching_speeds)
    # JRA horses typically run 16-17 m/s. Score linearly.
    # 15 m/s = 30, 17 m/s = 80
    score = 30.0 + (avg_speed - 15.0) * 25.0
    return max(20.0, min(95.0, score))


def calc_weight_carried_trend(past_races: list, current_weight: float) -> float:
    """Score based on weight carried trend.

    If current weight is significantly higher than recent past, it's a handicap.
    If lower, it's favorable. Gracefully degrades without data.
    """
    if not past_races or current_weight <= 0:
        return 50.0

    past_weights = [r.get("weightCarried", 0) for r in past_races[:4] if r.get("weightCarried", 0) > 0]
    if not past_weights:
        return 50.0

    avg_past = sum(past_weights) / len(past_weights)
    delta = current_weight - avg_past

    # More weight = harder. -2kg = +15, +0kg = 50, +2kg = -15
    # Typical range: -3 to +3 kg
    score = 50.0 - delta * 7.5
    return max(25.0, min(80.0, score))


def calc_days_since_last_race(past_races: list, current_date: str = "") -> float:
    """Score based on days since last race (休養明け判定).

    Scoring:
      1-14 days (中○週): 50 (normal)
      15-28 days: 55 (slightly fresh)
      29-60 days: 60 (freshness peak for many horses)
      61-120 days: 55 (returning from layoff)
      121-180 days: 45 (rust risk)
      180+ days: 35 (significant layoff)
      First career race: 45 (unknown)
    """
    if not past_races or not current_date:
        return 50.0

    last_race = past_races[0]
    last_date = last_race.get("date", "")
    if not last_date:
        return 50.0

    # Parse "YYYY.MM.DD" format
    def parse_date(s):
        try:
            parts = s.replace("/", ".").split(".")
            if len(parts) == 3:
                return _date_type(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            pass
        return None

    d1 = parse_date(current_date)
    d2 = parse_date(last_date)
    if not d1 or not d2:
        return 50.0

    days = (d1 - d2).days
    if days < 0:
        return 50.0

    if days <= 14:
        return 50.0
    elif days <= 28:
        return 55.0
    elif days <= 60:
        return 60.0  # Peak freshness
    elif days <= 120:
        return 55.0
    elif days <= 180:
        return 45.0
    else:
        return 35.0


def calc_same_distance_performance(past_races: list, target_distance: int) -> float:
    """Score based on past performance at similar distance (±200m).

    Addresses gap: horses that ran well at 1200m but are now running 2000m
    shouldn't get full credit for past performance.
    """
    if not past_races or target_distance <= 0:
        return 50.0

    matching_positions = []
    for race in past_races[:6]:
        dist = race.get("distance", 0)
        pos = race.get("pos", 0)
        if dist > 0 and pos > 0 and abs(dist - target_distance) <= 200:
            matching_positions.append(pos)

    if not matching_positions:
        return 48.0  # No distance match = slight penalty

    avg_pos = sum(matching_positions) / len(matching_positions)
    # Convert avg position to 0-100 score
    if avg_pos <= 1.5:
        return 95.0
    elif avg_pos <= 3:
        return 80.0
    elif avg_pos <= 5:
        return 65.0
    elif avg_pos <= 8:
        return 50.0
    else:
        return 35.0


def calc_same_surface_performance(past_races: list, target_surface: str) -> float:
    """Score based on past performance on same surface (芝/ダ).

    Addresses gap: dirt specialists on turf and vice versa get wrong scores.
    """
    if not past_races or not target_surface:
        return 50.0

    matching_positions = []
    for race in past_races[:6]:
        surf = race.get("surface", "")
        pos = race.get("pos", 0)
        if surf and pos > 0 and surf == target_surface:
            matching_positions.append(pos)

    if not matching_positions:
        return 40.0  # No surface match = penalty (may be first time on this surface)

    avg_pos = sum(matching_positions) / len(matching_positions)
    if avg_pos <= 1.5:
        return 95.0
    elif avg_pos <= 3:
        return 80.0
    elif avg_pos <= 5:
        return 65.0
    elif avg_pos <= 8:
        return 50.0
    else:
        return 35.0


def calc_same_condition_performance(past_races: list, target_condition: str) -> float:
    """Score based on past performance on similar track condition.

    Groups: {良} vs {稍重, 重, 不良}. Horses that only ran on firm ground
    shouldn't be highly rated on heavy ground.
    """
    if not past_races or not target_condition:
        return 50.0

    # Normalize condition into firm/soft categories
    def is_firm(c):
        return c == "良"

    target_firm = is_firm(target_condition)

    matching_positions = []
    for race in past_races[:6]:
        cond = race.get("condition", "")
        pos = race.get("pos", 0)
        if cond and pos > 0 and is_firm(cond) == target_firm:
            matching_positions.append(pos)

    if not matching_positions:
        return 42.0  # No condition match = slight penalty

    avg_pos = sum(matching_positions) / len(matching_positions)
    if avg_pos <= 1.5:
        return 90.0
    elif avg_pos <= 3:
        return 75.0
    elif avg_pos <= 5:
        return 60.0
    elif avg_pos <= 8:
        return 48.0
    else:
        return 35.0


def _lookup_heavy_track(name: str) -> int | None:
    """Look up a sire/BMS name in SIRE_HEAVY_TRACK with partial matching."""
    if name in SIRE_HEAVY_TRACK:
        return SIRE_HEAVY_TRACK[name]
    for key, rating in SIRE_HEAVY_TRACK.items():
        if key in name or name in key:
            return rating
    return None


def calc_track_condition_affinity(
    sire_name: str, track_condition: str, bms_name: str = ""
) -> float:
    """Score based on how well sire's offspring perform on heavy/soft ground.

    On 良 (good) ground, returns 50 (neutral - no advantage).
    On 稍重/重/不良, uses sire and BMS heavy track affinity data.
    When both sire and BMS are known, blends 70% sire + 30% BMS.
    """
    if not track_condition or track_condition == "良":
        return 50.0

    sire_aff = _lookup_heavy_track(sire_name) if sire_name else None
    bms_aff = _lookup_heavy_track(bms_name) if bms_name else None

    if sire_aff is not None and bms_aff is not None:
        affinity = sire_aff * 0.7 + bms_aff * 0.3
    elif sire_aff is not None:
        affinity = sire_aff
    elif bms_aff is not None:
        affinity = bms_aff
    else:
        affinity = 50

    # Scale effect by severity: 稍重 < 重 < 不良
    severity = {"稍重": 0.5, "重": 0.8, "不良": 1.0}.get(track_condition, 0.5)

    # Blend toward the affinity based on severity
    # At severity 1.0, return full affinity; at 0.5, blend with neutral
    return 50.0 + (affinity - 50.0) * severity


def calc_track_direction(
    past_races: list, direction: str, target_distance: int = 0
) -> float:
    """Score based on horse's performance on same-direction tracks.

    direction is '右' or '左' from courseDetail.
    When target_distance > 0, weights past races by distance similarity.
    """
    if not past_races or not direction:
        return 50.0

    # Filter for '右' or '左'
    dir_char = ""
    if "右" in direction:
        dir_char = "右"
    elif "左" in direction:
        dir_char = "左"

    if not dir_char:
        return 50.0

    same_dir_races = [r for r in past_races if r.get("direction") == dir_char]
    if not same_dir_races:
        return 50.0

    if target_distance > 0:
        # Distance-weighted scoring: closer distance = higher relevance
        weighted_score = 0.0
        total_relevance = 0.0
        for r in same_dir_races:
            pos = r.get("pos", 0)
            if pos <= 0:
                continue
            past_dist = r.get("distance", 0)
            if past_dist > 0:
                relevance = 1.0 / (1.0 + abs(past_dist - target_distance) / 400.0)
            else:
                relevance = 0.5  # Unknown distance: half relevance
            # Position to score
            if pos == 1:
                pos_score = 85.0
            elif pos <= 3:
                pos_score = 70.0
            elif pos <= 5:
                pos_score = 58.0
            else:
                pos_score = 45.0
            weighted_score += pos_score * relevance
            total_relevance += relevance

        if total_relevance > 0:
            return weighted_score / total_relevance
        return 50.0

    # Fallback: original logic when no target_distance
    wins = sum(1 for r in same_dir_races if r.get("pos") == 1)
    places = sum(1 for r in same_dir_races if 1 <= r.get("pos", 0) <= 3)

    if wins >= 2:
        return 85.0
    elif wins == 1:
        return 75.0
    elif places >= 2:
        return 70.0
    elif places == 1:
        return 60.0
    else:
        return 50.0


def calc_form_trend(past_races: list) -> float:
    """Score based on form trend direction (improving vs declining).

    Uses linear regression slope of finish positions across recent races.
    Improving form (positions getting lower/better) = higher score.
    """
    if not past_races:
        return 50.0

    # Extract valid positions (most recent first: index 0 = most recent)
    positions = []
    for race in past_races[:5]:
        pos = race.get("pos", 0)
        if pos > 0:
            positions.append(pos)

    if len(positions) < 2:
        return 50.0

    # Linear regression: x = time index (0=oldest, n-1=most recent)
    # Negative slope = improving (positions getting smaller = better)
    n = len(positions)
    # Reverse so oldest is x=0, newest is x=n-1
    pos_reversed = list(reversed(positions))
    x_mean = (n - 1) / 2.0
    y_mean = sum(pos_reversed) / n

    numerator = sum((i - x_mean) * (pos_reversed[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        slope = 0.0
    else:
        slope = numerator / denominator

    # Map slope to score: negative slope = improving = high score
    if slope < -2.0:
        score = 85.0   # Strong improvement
    elif slope < -0.5:
        score = 70.0   # Mild improvement
    elif slope <= 0.5:
        score = 55.0   # Stable
    elif slope <= 2.0:
        score = 40.0   # Mild decline
    else:
        score = 30.0   # Strong decline

    # Bonus: latest race was a win after non-wins
    if positions[0] == 1 and len(positions) >= 2 and positions[1] > 1:
        score = min(100.0, score + 10.0)

    return score


def calc_track_specific(past_races: list, racecourse_code: str) -> float:
    """Score based on horse's results at this specific track.

    racecourse_code maps to track names (06=中山, 09=阪神, etc.)
    """
    if not past_races or not racecourse_code:
        return 50.0

    track_name = RACECOURSE_CODE_MAP.get(racecourse_code, "")
    if not track_name:
        return 50.0

    track_races = [r for r in past_races if r.get("track") == track_name]
    if not track_races:
        return 50.0

    best_pos = min((r.get("pos", 99) for r in track_races if r.get("pos", 0) > 0), default=99)

    if best_pos == 1:
        return 85.0
    elif best_pos <= 3:
        return 70.0
    else:
        return 55.0


def calc_agari3f_score(past_races: list, surface: str = "") -> float:
    """Score based on average last-3-furlong time (上がり3F).

    Faster agari = stronger finishing kick = higher score.
    Surface-specific: turf agari ~33-36s, dirt ~36-40s.
    """
    agari_times = []
    for pr in past_races:
        a = pr.get("agari3f", 0)
        if a > 0:
            # Filter by same surface if specified
            if surface and pr.get("surface"):
                pr_surface = "芝" if pr["surface"] == "芝" else "ダート" if pr["surface"] in ("ダ", "ダート") else ""
                if surface and pr_surface and surface != pr_surface:
                    continue
            agari_times.append(a)

    if not agari_times:
        return 50.0

    avg_agari = sum(agari_times) / len(agari_times)

    # Score: faster = better. Turf benchmark ~34.0, dirt ~37.0
    if surface == "芝":
        score = max(20, min(95, 130 - avg_agari * 2.5))
    else:
        score = max(20, min(95, 145 - avg_agari * 2.5))

    return round(score, 1)


def calc_margin_score(past_races: list) -> float:
    """Score based on average margin from winner (着差).

    Smaller margin = closer to winning = higher score.
    Margin 0.0 = won the race.
    """
    margins = []
    for pr in past_races:
        m = pr.get("margin", 0)
        pos = pr.get("pos", 0)
        if pos > 0 and pos <= 18:
            if pos == 1:
                margins.append(0.0)
            elif m > 0:
                margins.append(m)

    if not margins:
        return 50.0

    avg_margin = sum(margins) / len(margins)

    # Score: 0.0 margin = 95, 0.5 = 80, 1.0 = 65, 2.0 = 40, 3.0+ = 25
    score = max(20, min(95, 95 - avg_margin * 25))
    return round(score, 1)


def calc_pace_predict(entries: list) -> float:
    """Predict pace from running styles of all entered horses.

    More front-runners (逃げ/先行) = faster pace = favors closers.
    Returns score relative to each horse's running style.
    """
    if not entries:
        return 50.0

    styles = []
    for e in entries:
        if e.get("isScratched"):
            continue
        prs = e.get("pastRaces", [])
        for pr in prs[:3]:  # Recent 3 races
            rs = pr.get("runningStyle", "")
            if rs:
                styles.append(rs)

    if not styles:
        return 50.0

    front_count = sum(1 for s in styles if s in ("逃げ", "先行"))
    total = len(styles)
    front_ratio = front_count / total if total > 0 else 0.5

    # High front ratio = fast pace expected
    # Score: 0.5 (neutral) = 50, high ratio = favor closers
    return round(50 + (front_ratio - 0.5) * 60, 1)


def calc_draw_bias(post_position: int, head_count: int, surface: str = "",
                   distance: int = 0, course_detail: str = "",
                   course_code: str = "", track_condition: str = "") -> float:
    """Score based on post position (枠順) advantage with track bias.

    Incorporates course-specific and condition-specific biases:
    - 新潟(04): 芝は内枠有利、ダートは外枠有利（直線1000mは大外有利）
    - 中京(07): 芝は内枠やや有利、ダートはフラット
    - 札幌(01): 芝は内枠有利（洋芝で内ラチ沿いが有利）、ダートは内枠有利
    - 重/不良馬場: 外枠有利にシフト（内が荒れるため）
    """
    if post_position <= 0 or head_count <= 0:
        return 50.0

    # Normalize position: 0.0 (innermost) to 1.0 (outermost)
    norm_pos = (post_position - 1) / max(head_count - 1, 1)

    is_turf = "芝" in surface if surface else True
    is_heavy = track_condition in ("重", "不良") if track_condition else False

    # Course-specific bias (positive = inner advantage)
    # Based on JRA course characteristics
    COURSE_BIAS = {
        # course_code: (turf_bias, dirt_bias)
        "01": (12.0, 8.0),    # 札幌: 洋芝・内有利、ダートも内有利
        "02": (8.0, 5.0),     # 函館: 洋芝・内有利
        "03": (6.0, 3.0),     # 福島: 小回り・内有利
        "04": (8.0, -3.0),    # 新潟: 芝内有利、ダート外やや有利
        "05": (5.0, 3.0),     # 東京: 大箱・バイアス小
        "06": (7.0, 5.0),     # 中山: 内有利
        "07": (6.0, 0.0),     # 中京: 芝は内やや有利、ダートはフラット
        "08": (5.0, 3.0),     # 京都: 外回り含むためバイアス小
        "09": (7.0, 5.0),     # 阪神: 内有利
        "10": (10.0, 6.0),    # 小倉: 小回り・内有利
    }

    turf_bias, dirt_bias = COURSE_BIAS.get(course_code, (5.0, 3.0))
    bias = turf_bias if is_turf else dirt_bias

    # Distance adjustment
    if distance > 0 and distance <= 1200:
        bias *= 1.3  # スプリントは枠順影響大
    elif distance > 0 and distance >= 2400:
        bias *= 0.5  # 長距離は枠順影響小

    # 新潟直線1000m: 大外有利（特殊コース）
    if course_code == "04" and distance == 1000 and is_turf:
        bias = -15.0  # 外枠有利

    # 重/不良馬場: 内が荒れるので外枠有利にシフト
    if is_heavy:
        bias -= 5.0

    # Small fields: position matters less
    if head_count <= 8:
        bias *= 0.5

    score = 50.0 + bias * (0.5 - norm_pos)
    return round(max(30, min(70, score)), 1)


# ---------------------------------------------------------------------------
# D6: 4 new analytical factors
# ---------------------------------------------------------------------------

def calc_jockey_course_distance(
    jockey: str, course_code: str, distance: int, past_races: list
) -> float:
    """Jockey x Course x Distance 3-way affinity score.

    Looks through past_races for entries matching the same course_code AND
    the same distance category (sprint/mile/intermediate/stayer).

    Scoring:
      2+ wins  → 85
      1 win    → 75
      2+ places (no win) → 70
      1 place  (no win)  → 60
      no match → 50
    """
    if not past_races:
        return 50.0

    target_cat = _get_distance_category(distance)

    wins = 0
    places = 0
    for race in past_races:
        rc = race.get("course_code", "")
        if rc != course_code:
            continue
        dist = race.get("distance", 0)
        if not dist or _get_distance_category(dist) != target_cat:
            continue
        pos = race.get("pos", 0)
        if pos == 1:
            wins += 1
        if 1 <= pos <= 3:
            places += 1

    if wins >= 2:
        return 85.0
    elif wins == 1:
        return 75.0
    elif places >= 2:
        return 70.0
    elif places == 1:
        return 60.0
    return 50.0


def calc_pace_position_advantage(
    past_races: list, entries: list, head_count: int
) -> float:
    """Race pace / position advantage based on running style mix.

    Counts front-runners (逃げ/先行) across all entries (using most recent
    past race for each).  Then scores this horse based on its own style
    relative to the field mix.

    Returns:
      75  — <= 1 front-runner in field AND this horse is a front-runner
      35  — >= 3 front-runners in field AND this horse is a front-runner
      70  — >= 3 front-runners in field AND this horse is a closer
      50  — otherwise (neutral)
    """
    FRONT_STYLES = {"逃げ", "先行"}
    CLOSER_STYLES = {"差し", "追込"}

    def _most_recent_style(prs: list) -> str:
        for r in prs:
            s = r.get("runningStyle", "")
            if s:
                return s
        return ""

    # Determine this horse's running style
    this_style = _most_recent_style(past_races)

    # Count front-runners in the full entries list
    front_count = 0
    for entry in entries:
        if entry.get("isScratched"):
            continue
        style = _most_recent_style(entry.get("pastRaces", []))
        if style in FRONT_STYLES:
            front_count += 1

    # No style data → neutral
    if not this_style:
        return 50.0

    this_is_front = this_style in FRONT_STYLES
    this_is_closer = this_style in CLOSER_STYLES

    if front_count <= 1 and this_is_front:
        return 75.0
    if front_count >= 3 and this_is_front:
        return 35.0
    if front_count >= 3 and this_is_closer:
        return 70.0
    return 50.0


def calc_rotation_fitness(
    past_races: list, race_date: str, class_change: int
) -> float:
    """Rotation / interval fitness score.

    Calculates days since last race, applies class change modifier and
    winning form carry-over bonus.

    Interval base scores:
      < 14 days  → 55  (too soon)
      14-35 days → 70  (optimal)
      36-60 days → 65
      61-90 days → 55
      91+ days   → 45

    Modifiers (additive):
      class_change = +1 (class up)   → -5
      class_change = -1 (class down) → +5
      most recent race pos == 1       → +10
    """
    if not past_races:
        return 50.0

    last_race = past_races[0]
    last_date_str = last_race.get("date", "")
    if not last_date_str:
        return 50.0

    def _parse(s: str):
        try:
            parts = s.replace("/", ".").split(".")
            if len(parts) == 3:
                return _date_type(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            pass
        return None

    d_race = _parse(race_date)
    d_last = _parse(last_date_str)
    if d_race is None or d_last is None:
        return 50.0

    days = (d_race - d_last).days
    if days < 0:
        return 50.0

    if days < 14:
        base = 55.0
    elif days <= 35:
        base = 70.0
    elif days <= 60:
        base = 65.0
    elif days <= 90:
        base = 55.0
    else:
        base = 45.0

    # Class change modifier
    if class_change > 0:
        base -= 5.0
    elif class_change < 0:
        base += 5.0

    # Winning form carry-over
    if last_race.get("pos") == 1:
        base += 10.0

    return float(base)


# Known heavy-track sires: included here for the bloodline factor
# (same sires are already in SIRE_HEAVY_TRACK with numeric ratings)
_KNOWN_MUD_SIRES: set[str] = {
    "ゴールドシップ",
    "ハーツクライ",
    "キングカメハメハ",
    "ステイゴールド",
    "オルフェーヴル",
    "スクリーンヒーロー",
    "エピファネイア",
    "ルーラーシップ",
}


def calc_bloodline_track_condition(
    sire: str, bms: str, track_condition: str, past_races: list
) -> float:
    """Bloodline x Track condition affinity.

    On 良 / 稍重 → 50 (neutral).
    On 重 / 不良:
      1. Scan past_races for races on 重/不良.
         - 1+ wins  → 80
         - 1+ places (no win) → 70
      2. If no past heavy data, check sire (then bms) against known mud sires
         and return 60 (bonus) or 50 (neutral).
    """
    HEAVY_CONDITIONS = {"重", "不良"}

    if track_condition not in HEAVY_CONDITIONS:
        return 50.0

    # 1. Scan past race history for heavy-track results
    heavy_wins = 0
    heavy_places = 0
    for race in past_races:
        cond = race.get("condition", "")
        if cond not in HEAVY_CONDITIONS:
            continue
        pos = race.get("pos", 0)
        if pos == 1:
            heavy_wins += 1
        if 1 <= pos <= 3:
            heavy_places += 1

    if heavy_wins >= 1:
        return 80.0
    if heavy_places >= 1:
        return 70.0

    # 2. Fall back to sire/bms bloodline profile
    def _is_mud_sire(name: str) -> bool:
        if not name:
            return False
        for ms in _KNOWN_MUD_SIRES:
            if ms in name or name in ms:
                return True
        return False

    if _is_mud_sire(sire) or _is_mud_sire(bms):
        return 60.0

    return 50.0
