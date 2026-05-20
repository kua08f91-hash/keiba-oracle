# データ監査レポート (Phase 0-E/G)

## リーク監査結果

### ファクター別 (22項目)

| ファクター | 状態 | データソース |
|-----------|------|------------|
| marketScore | ✓ OK | 凍結時オッズ/人気 |
| pastPerformance | ✓ OK | 過去レース結果（確定済み） |
| jockeyAbility | ✓ OK | 騎手の過去成績 |
| courseAffinity | ✓ OK | 血統データ（静的） |
| distanceAptitude | ✓ OK | 血統データ（静的） |
| trainerAbility | ✓ OK | 調教師の過去成績 |
| trackCondition | ✓ OK | 当日馬場発表（レース前確定） |
| trackDirection | ✓ OK | 過去走のコース方向 |
| trackSpecific | ✓ OK | 過去走の競馬場成績 |
| ageAndSex | ✓ OK | 馬齢・性別（静的） |
| weightCarried | ✓ OK | 斤量（出馬表確定時） |
| horseWeightChange | ✓ OK | 馬体重（パドック後、発走前に確定） |
| formTrend | ✓ OK | 過去着順推移 |
| sameDistance | ✓ OK | 過去走の同距離成績 |
| sameSurface | ✓ OK | 過去走の同芝/ダ成績 |
| sameCondition | ✓ OK | 過去走の同馬場状態成績 |
| speedFigure | ✓ OK | 過去走のタイム/上がり |
| runningStyle | ✓ OK | 過去走の通過順/脚質 |
| daysSinceLast | ✓ OK | 前走日付から計算 |
| weightCarriedTrend | ✓ OK | 過去走の斤量変化 |
| agari3f | ✓ OK | 過去走の上がり3F |
| marginScore | ✓ OK | 過去走の着差 |
| drawBias | ✓ OK | 枠番（出馬表確定時） |

**結果: 22/22 ファクターがリークなし**

### バックテスト時の注意

- `fetch_race_card()` → `_fetch_result_data()` は過去レースで確定オッズを返す
- バックテストハーネスでは `estimate_from_entries()` + live API で取得したオッズを使用
- 確定後のオッズと凍結時オッズには差がある（特に直前のオッズ変動）

## データ量

| 期間 | レース数 | 状態 |
|------|---------|------|
| 2023 | ~1,099R | historical_races.json (pastRaces情報が貧弱) |
| 2024 | ~3,400R | 収集中 |
| 2025 | ~1,600R (〜5月) | 収集中 |
| 2026 | ~500R (〜5月) | backtest/cache + DB |

目標: 3年分 ~10,000R

## データ品質リスク

- 2023年データは過去走情報が貧弱（pos/trackのみ） → 新ファクター(agari3f/margin等)が機能しない
- 対策: 2024年以降のデータを主力として使用
