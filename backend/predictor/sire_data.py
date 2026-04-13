"""Comprehensive sire profile database for JRA races.

Each sire profile contains affinity scores (0-100) for:
- surface: 芝 (turf) and ダート (dirt)
- distance categories: sprint (1000-1400m), mile (1401-1800m),
  intermediate (1801-2200m), stayer (2201m+)

Data compiled from publicly available JRA race statistics,
Wikipedia articles on Japanese thoroughbred stallions,
and JRA official stallion records.
"""

# Format: {sire_name: {turf, dirt, sprint, mile, intermediate, stayer}}
SIRE_PROFILES: dict = {
    # ===== Major Turf Sires =====
    "ディープインパクト": {"芝": 92, "ダート": 35, "sprint": 55, "mile": 85, "intermediate": 92, "stayer": 82},
    "キングカメハメハ": {"芝": 80, "ダート": 75, "sprint": 65, "mile": 82, "intermediate": 85, "stayer": 72},
    "ハーツクライ": {"芝": 90, "ダート": 40, "sprint": 35, "mile": 70, "intermediate": 92, "stayer": 92},
    "ステイゴールド": {"芝": 88, "ダート": 40, "sprint": 40, "mile": 68, "intermediate": 88, "stayer": 88},
    "ロードカナロア": {"芝": 88, "ダート": 50, "sprint": 95, "mile": 88, "intermediate": 58, "stayer": 35},
    "ダイワメジャー": {"芝": 85, "ダート": 55, "sprint": 78, "mile": 92, "intermediate": 62, "stayer": 38},
    "エピファネイア": {"芝": 88, "ダート": 48, "sprint": 48, "mile": 75, "intermediate": 92, "stayer": 82},
    "キタサンブラック": {"芝": 90, "ダート": 42, "sprint": 42, "mile": 75, "intermediate": 88, "stayer": 92},
    "ドゥラメンテ": {"芝": 88, "ダート": 58, "sprint": 52, "mile": 82, "intermediate": 92, "stayer": 75},
    "モーリス": {"芝": 88, "ダート": 52, "sprint": 58, "mile": 92, "intermediate": 82, "stayer": 58},
    "サトノダイヤモンド": {"芝": 85, "ダート": 42, "sprint": 38, "mile": 65, "intermediate": 88, "stayer": 90},
    "オルフェーヴル": {"芝": 88, "ダート": 62, "sprint": 48, "mile": 72, "intermediate": 90, "stayer": 88},
    "ジャスタウェイ": {"芝": 88, "ダート": 48, "sprint": 55, "mile": 88, "intermediate": 85, "stayer": 62},
    "ルーラーシップ": {"芝": 85, "ダート": 52, "sprint": 48, "mile": 75, "intermediate": 88, "stayer": 82},
    "スクリーンヒーロー": {"芝": 80, "ダート": 58, "sprint": 48, "mile": 75, "intermediate": 88, "stayer": 78},
    "サトノクラウン": {"芝": 82, "ダート": 52, "sprint": 48, "mile": 75, "intermediate": 88, "stayer": 82},
    "ゴールドシップ": {"芝": 82, "ダート": 52, "sprint": 38, "mile": 58, "intermediate": 82, "stayer": 92},
    "コントレイル": {"芝": 90, "ダート": 45, "sprint": 52, "mile": 82, "intermediate": 92, "stayer": 85},
    "イクイノックス": {"芝": 92, "ダート": 45, "sprint": 52, "mile": 82, "intermediate": 94, "stayer": 82},
    "スワーヴリチャード": {"芝": 82, "ダート": 68, "sprint": 55, "mile": 82, "intermediate": 88, "stayer": 75},
    "キズナ": {"芝": 88, "ダート": 55, "sprint": 48, "mile": 80, "intermediate": 90, "stayer": 82},
    "リアルスティール": {"芝": 85, "ダート": 52, "sprint": 52, "mile": 80, "intermediate": 88, "stayer": 72},
    "マカヒキ": {"芝": 82, "ダート": 48, "sprint": 48, "mile": 78, "intermediate": 88, "stayer": 75},
    "レイデオロ": {"芝": 85, "ダート": 55, "sprint": 50, "mile": 78, "intermediate": 90, "stayer": 78},
    "ワールドプレミア": {"芝": 85, "ダート": 45, "sprint": 42, "mile": 70, "intermediate": 85, "stayer": 90},
    "シルバーステート": {"芝": 85, "ダート": 50, "sprint": 55, "mile": 82, "intermediate": 85, "stayer": 68},
    "ブラックタイド": {"芝": 85, "ダート": 48, "sprint": 52, "mile": 78, "intermediate": 88, "stayer": 78},
    "ハービンジャー": {"芝": 85, "ダート": 42, "sprint": 42, "mile": 72, "intermediate": 88, "stayer": 85},
    "マンハッタンカフェ": {"芝": 85, "ダート": 55, "sprint": 42, "mile": 68, "intermediate": 85, "stayer": 88},
    "ネオユニヴァース": {"芝": 82, "ダート": 55, "sprint": 48, "mile": 75, "intermediate": 85, "stayer": 78},
    "ディープブリランテ": {"芝": 82, "ダート": 52, "sprint": 52, "mile": 80, "intermediate": 85, "stayer": 72},
    "ロゴタイプ": {"芝": 78, "ダート": 60, "sprint": 58, "mile": 82, "intermediate": 78, "stayer": 58},
    "リオンディーズ": {"芝": 78, "ダート": 65, "sprint": 58, "mile": 82, "intermediate": 82, "stayer": 62},
    "サトノアラジン": {"芝": 82, "ダート": 48, "sprint": 65, "mile": 85, "intermediate": 72, "stayer": 52},
    "ミッキーアイル": {"芝": 85, "ダート": 45, "sprint": 85, "mile": 85, "intermediate": 55, "stayer": 35},
    "イスラボニータ": {"芝": 82, "ダート": 52, "sprint": 60, "mile": 85, "intermediate": 78, "stayer": 58},
    "ドレフォン": {"芝": 65, "ダート": 75, "sprint": 82, "mile": 82, "intermediate": 65, "stayer": 42},
    "マクフィ": {"芝": 80, "ダート": 55, "sprint": 65, "mile": 85, "intermediate": 75, "stayer": 55},
    "ナダル": {"芝": 55, "ダート": 82, "sprint": 68, "mile": 85, "intermediate": 78, "stayer": 55},
    "ニューイヤーズデイ": {"芝": 55, "ダート": 82, "sprint": 72, "mile": 85, "intermediate": 72, "stayer": 48},
    "フィエールマン": {"芝": 88, "ダート": 42, "sprint": 38, "mile": 68, "intermediate": 85, "stayer": 92},
    "ブリックスアンドモルタル": {"芝": 88, "ダート": 55, "sprint": 52, "mile": 82, "intermediate": 88, "stayer": 72},
    "サリオス": {"芝": 88, "ダート": 52, "sprint": 58, "mile": 88, "intermediate": 85, "stayer": 62},

    # ===== Major Dirt Sires =====
    "ヘニーヒューズ": {"芝": 30, "ダート": 94, "sprint": 92, "mile": 82, "intermediate": 58, "stayer": 32},
    "シニスターミニスター": {"芝": 28, "ダート": 92, "sprint": 72, "mile": 88, "intermediate": 78, "stayer": 48},
    "パイロ": {"芝": 28, "ダート": 90, "sprint": 82, "mile": 88, "intermediate": 68, "stayer": 42},
    "ゴールドアリュール": {"芝": 28, "ダート": 94, "sprint": 62, "mile": 88, "intermediate": 82, "stayer": 58},
    "マジェスティックウォリアー": {"芝": 32, "ダート": 90, "sprint": 88, "mile": 85, "intermediate": 62, "stayer": 38},
    "コパノリッキー": {"芝": 28, "ダート": 90, "sprint": 68, "mile": 85, "intermediate": 82, "stayer": 52},
    "ルヴァンスレーヴ": {"芝": 32, "ダート": 90, "sprint": 68, "mile": 88, "intermediate": 82, "stayer": 52},
    "ゴールドドリーム": {"芝": 30, "ダート": 90, "sprint": 72, "mile": 90, "intermediate": 78, "stayer": 48},
    "ホッコータルマエ": {"芝": 28, "ダート": 92, "sprint": 65, "mile": 85, "intermediate": 85, "stayer": 58},
    "エスポワールシチー": {"芝": 25, "ダート": 90, "sprint": 72, "mile": 88, "intermediate": 78, "stayer": 48},
    "サウスヴィグラス": {"芝": 25, "ダート": 92, "sprint": 94, "mile": 78, "intermediate": 52, "stayer": 30},
    "カジノドライヴ": {"芝": 30, "ダート": 88, "sprint": 72, "mile": 85, "intermediate": 75, "stayer": 48},
    "カネヒキリ": {"芝": 28, "ダート": 90, "sprint": 65, "mile": 85, "intermediate": 82, "stayer": 55},
    "クロフネ": {"芝": 55, "ダート": 82, "sprint": 78, "mile": 85, "intermediate": 72, "stayer": 48},
    "キンシャサノキセキ": {"芝": 72, "ダート": 65, "sprint": 92, "mile": 78, "intermediate": 48, "stayer": 30},
    "ダノンレジェンド": {"芝": 28, "ダート": 88, "sprint": 90, "mile": 78, "intermediate": 55, "stayer": 32},
    "アジアエクスプレス": {"芝": 42, "ダート": 82, "sprint": 72, "mile": 82, "intermediate": 72, "stayer": 48},
    "スズカコーズウェイ": {"芝": 35, "ダート": 80, "sprint": 78, "mile": 80, "intermediate": 62, "stayer": 42},
    "ダンカーク": {"芝": 35, "ダート": 82, "sprint": 68, "mile": 82, "intermediate": 75, "stayer": 52},
    "ミッキーロケット": {"芝": 82, "ダート": 55, "sprint": 48, "mile": 75, "intermediate": 88, "stayer": 78},

    # ===== Overseas/International Sires active in Japan =====
    "フランケル": {"芝": 92, "ダート": 38, "sprint": 55, "mile": 90, "intermediate": 88, "stayer": 72},
    "ダークエンジェル": {"芝": 82, "ダート": 48, "sprint": 88, "mile": 82, "intermediate": 58, "stayer": 35},
    "サクソンウォリアー": {"芝": 85, "ダート": 48, "sprint": 55, "mile": 85, "intermediate": 82, "stayer": 68},
    "アメリカンファラオ": {"芝": 60, "ダート": 82, "sprint": 68, "mile": 85, "intermediate": 82, "stayer": 62},
    "Tiznow": {"芝": 42, "ダート": 88, "sprint": 62, "mile": 82, "intermediate": 85, "stayer": 68},
    "エスケンデレヤ": {"芝": 35, "ダート": 85, "sprint": 68, "mile": 82, "intermediate": 78, "stayer": 52},

    # ===== Classic/Older Sires (still appearing in pedigrees) =====
    "サンデーサイレンス": {"芝": 90, "ダート": 55, "sprint": 55, "mile": 82, "intermediate": 92, "stayer": 85},
    "スペシャルウィーク": {"芝": 88, "ダート": 48, "sprint": 45, "mile": 75, "intermediate": 90, "stayer": 85},
    "フジキセキ": {"芝": 78, "ダート": 68, "sprint": 75, "mile": 85, "intermediate": 72, "stayer": 48},
    "ブライアンズタイム": {"芝": 75, "ダート": 70, "sprint": 55, "mile": 78, "intermediate": 85, "stayer": 75},
    "ワイルドラッシュ": {"芝": 35, "ダート": 85, "sprint": 78, "mile": 82, "intermediate": 72, "stayer": 48},
    "エンパイアメーカー": {"芝": 40, "ダート": 88, "sprint": 65, "mile": 82, "intermediate": 82, "stayer": 62},
    "タートルボウル": {"芝": 78, "ダート": 55, "sprint": 62, "mile": 82, "intermediate": 78, "stayer": 58},
    "スキャターザゴールド": {"芝": 42, "ダート": 85, "sprint": 78, "mile": 82, "intermediate": 68, "stayer": 42},
    "キングカメハメハ": {"芝": 80, "ダート": 75, "sprint": 65, "mile": 82, "intermediate": 85, "stayer": 72},
    "コロナドズクエスト": {"芝": 45, "ダート": 82, "sprint": 72, "mile": 82, "intermediate": 72, "stayer": 48},
    "タイキシャトル": {"芝": 82, "ダート": 58, "sprint": 88, "mile": 85, "intermediate": 55, "stayer": 32},
    "シンボリクリスエス": {"芝": 82, "ダート": 62, "sprint": 52, "mile": 78, "intermediate": 88, "stayer": 78},
    "クリスエス": {"芝": 82, "ダート": 62, "sprint": 52, "mile": 78, "intermediate": 88, "stayer": 78},
    "マンハッタンカフェ": {"芝": 85, "ダート": 55, "sprint": 42, "mile": 68, "intermediate": 85, "stayer": 88},
    "タニノギムレット": {"芝": 82, "ダート": 55, "sprint": 52, "mile": 78, "intermediate": 88, "stayer": 72},
    "アグネスタキオン": {"芝": 85, "ダート": 52, "sprint": 62, "mile": 85, "intermediate": 82, "stayer": 62},
    "ゼンノロブロイ": {"芝": 82, "ダート": 55, "sprint": 48, "mile": 75, "intermediate": 88, "stayer": 78},
    "メイショウサムソン": {"芝": 78, "ダート": 58, "sprint": 48, "mile": 72, "intermediate": 85, "stayer": 78},
    "ダンスインザダーク": {"芝": 85, "ダート": 48, "sprint": 38, "mile": 65, "intermediate": 85, "stayer": 90},
    "ディープスカイ": {"芝": 80, "ダート": 58, "sprint": 55, "mile": 80, "intermediate": 82, "stayer": 68},
    "トーセンホマレボシ": {"芝": 78, "ダート": 58, "sprint": 52, "mile": 78, "intermediate": 82, "stayer": 72},
}

# Default profile for unknown sires
DEFAULT_PROFILE = {"芝": 50, "ダート": 50, "sprint": 50, "mile": 50, "intermediate": 50, "stayer": 50}


def get_sire_profile(sire_name: str) -> dict:
    """Get sire profile, returning default for unknown sires."""
    if sire_name in SIRE_PROFILES:
        return SIRE_PROFILES[sire_name]

    # Try partial match for international sires with different name formats
    for name, profile in SIRE_PROFILES.items():
        if name in sire_name or sire_name in name:
            return profile

    return DEFAULT_PROFILE
