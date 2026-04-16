"""Individual prediction factor calculators.

Each function returns a score from 0 to 100.

Updated: Incorporates market odds, horse weight change, trainer ratings,
and removes redundant bloodline factor.
"""
from __future__ import annotations

import re
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

    return 45.0  # Unknown - slightly below average


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
                from datetime import date
                return date(int(parts[0]), int(parts[1]), int(parts[2]))
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
