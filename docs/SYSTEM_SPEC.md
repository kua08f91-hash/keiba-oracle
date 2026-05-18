# KEIBA ORACLE システム仕様一覧

## 1. アーキテクチャ

| 層 | 技術 | URL |
|---|---|---|
| Frontend | Vercel (静的HTML + 60秒ポーリング) | https://jra-prediction-app-ochre.vercel.app |
| Backend | Railway (FastAPI + SQLite/PostgreSQL) | https://keiba-oracle-api-production.up.railway.app |
| Data | netkeiba.com (スクレイピング + API) | |
| CI/CD | GitHub Actions (予想自動export) | https://github.com/kua08f91-hash/keiba-oracle |

---

## 2. スコアリングエンジン (v5 WeightedScoringModel)

```
score = analytical × 60% + marketScore(オッズ) × 40%
```

### ファクター重み (22ファクター)

| ファクター | 分析内重み | 最終重み | 説明 |
|-----------|----------|---------|------|
| marketScore | - | **40.00%** | 市場オッズ |
| daysSinceLast | 9.30% | 5.58% | 休養明け |
| speedFigure | 8.54% | 5.12% | スピード指数 |
| weightCarriedTrend | 7.01% | 4.21% | 斤量変化 |
| trackDirection | 6.81% | 4.09% | コース方向 |
| ageAndSex | 6.59% | 3.95% | 年齢・性別 |
| sameCondition | 6.04% | 3.62% | 同馬場状態実績 |
| distanceAptitude | 5.93% | 3.56% | 距離適性 |
| pastPerformance | 5.88% | 3.53% | 過去成績 |
| marginScore | 5.88% | 3.53% | 着差スコア (NEW) |
| drawBias | 4.90% | 2.94% | 枠順バイアス (NEW) |
| sameSurface | 4.85% | 2.91% | 同芝/ダ実績 |
| runningStyle | 4.76% | 2.86% | 脚質一貫性 |
| jockeyAbility | 4.30% | 2.58% | 騎手能力 |
| trackCondition | 3.74% | 2.24% | 馬場状態適性 |
| sameDistance | 3.44% | 2.06% | 同距離実績 |
| trackSpecific | 3.35% | 2.01% | コース別実績 |
| agari3f | 2.94% | 1.76% | 上がり3F (NEW) |
| trainerAbility | 2.27% | 1.36% | 調教師能力 |
| formTrend | 1.92% | 1.15% | 調子推移 |
| horseWeightChange | 1.18% | 0.71% | 馬体重変動 |
| courseAffinity | 0.35% | 0.21% | コース適性(血統) |
| weightCarried | 0.00% | 0.00% | 斤量 (除外: HURTS) |

### 基本設定

| パラメータ | 値 |
|-----------|-----|
| TEMPERATURE | 9.5 (softmax) |
| MC_SAMPLES | 5,000 |
| JRA_TAKEOUT | 25% |

---

## 3. 買い目戦略 (S8 Value-Range)

### ルール
AI top5の馬が絡む × 券種別オッズ帯 × 高オッズ順

### 券種別オッズ帯

| 券種 | オッズ下限 | オッズ上限 | 最大点数/R |
|------|----------|----------|-----------|
| 馬単 (umatan) | 20倍 | 300倍 | 2点 |
| 馬連 (umaren) | 20倍 | 100倍 | 2点 |
| ワイド (wide) | 10倍 | 50倍 | 2点 |

### 制約

- 全体 最大5点/レース
- AI top5に含まれる馬が1頭以上必要
- リアルオッズ必須 (推定オッズは使わない)
- 高オッズ順にソート
- 除外券種: 単勝, 複勝, 枠連, 3連複, 3連単

### 検証結果 (314R, Tune/Validate分割)

| セット | ROI |
|--------|-----|
| Tune (121R) | 131.6% |
| Validate (193R) | 126.8% |

---

## 4. データパイプライン

### スクレイピング

| ソース | データ |
|--------|-------|
| shutuba.html | 出馬表 (枠番, 馬番, 騎手, 斤量, 馬体重) |
| shutuba_past.html | 過去成績 (16フィールド): pos, condition, surface, distance, track, direction, date, finishTime, fieldSize, postPosition, popularity, weightCarried, corners, runningStyle, agari3f, margin |
| result.html | 結果+払戻 (着順, 配当) |
| API type=1 | 単勝オッズ (リアルタイム) |
| API type=4,5,7,8 | 組合せオッズ (リアルタイム) |

### キャッシュ

| 項目 | 設定 |
|------|------|
| DB TTL | 30日 |
| 枠番=0自動無効化 | 50%超でframe=0なら再スクレイピング |
| force_refresh | exportから強制再取得可能 |

### race_info抽出

| フィールド | 抽出方法 |
|-----------|---------|
| date | ページ内のYYYY年M月D日 → YYYYMMDD |
| trackCondition | 馬場:良/稍/重/不 → DBに保存+返却 |
| headCount | len(entries) で設定 |

### オッズ取得

| 項目 | 設定 |
|------|------|
| リトライ | 最大3回 (バックオフ 3s, 6s, 9s) |
| タイムアウト | 15秒 |
| odds=None時 | marketScore = 50.0 (中立) |
| 推定オッズ閾値 | 2頭以上oddsあれば推定実行 |

---

## 5. 自動パイプライン

### GitHub Actions (PCオフでも動作)

| 曜日 | スケジュール (JST) |
|------|-------------------|
| 土曜 | 06:00, 09:00, 11:00, 13:00, 15:00 |
| 日曜 | 06:00, 09:00, 11:00, 13:00, 15:00 |

タイムアウト: 45分。predictions.json + archive自動commit+push → Vercelデプロイ。

### ローカルcron (PCオン時のみ)

| スケジュール | 処理 |
|------------|------|
| 木曜 21:00 | prefetch_weekly (次週末データ事前取得) |
| 金-土 14-23時毎時 | prefetch_weekly (オッズ更新) |
| 土日 07:00 | export + git push |
| 土日 12:00, 15:00 | export + git push |
| 土日 9-16時 5分毎 | refresh_raceday (ライブオッズ) |
| 月曜 06:00 | auto_improve (結果収集+精度評価) |

---

## 6. リアルタイム更新

### realtime_worker

| 項目 | 設定 |
|------|------|
| 更新開始 | 発走30分前 |
| 更新間隔 | 60秒 (最寄りレース20分以内), 300秒 (それ以外) |
| 凍結 | 発走10分前 (最終オッズ→予想更新→frozen=True) |
| 終了 | 全レース発走10分後 |

### フロントエンド

| 項目 | 設定 |
|------|------|
| 詳細画面 | 60秒ポーリング (APIからオッズ+スコア+買い目を自動更新) |
| 一覧戻り時 | ポーリング停止 |
| stale判定 | 48時間 (predictions.json) |

---

## 7. アーカイブ・振り返り

| ファイル | 用途 |
|---------|------|
| predictions.json | メインデータ (上書きされる) |
| archive/predictions_YYYYMMDD.json | 日付別バックアップ (上書きされない) |

---

## 8. テストカバレッジ

| 項目 | 数値 |
|------|------|
| 総テスト数 | 832 passed + 3 skipped |

### 主要テストファイル

| ファイル | テスト数 | 内容 |
|---------|---------|------|
| test_bet_optimizer.py | ~500 | S8戦略, 旧戦略, _diversify |
| test_new_factors.py | ~130 | 22ファクター, パーサー |
| test_scoring.py | ~50 | 重みブレンド, market_weight |
| test_parser_grade_detection.py | ~50 | グレードバッジ |
| test_cache_invalidation.py | ~45 | キャッシュ無効化 |
| test_odds_retry.py | ~7 | オッズリトライ |

---

## 9. グレードバッジ

### 判定優先順位

1. raceInfo.grade (netkeiba Icon_GradeType × _KNOWN_GRADED交差検証)
2. レース名内 (GI)/(GII)/(GIII) パターン
3. GRADE_RACES名前リスト (フロント側フォールバック)

全アイコンタイプ (GI/GII/GIII) で_KNOWN_GRADEDリスト照合必須。
障害レース用アイコン (Type12-17) も対応。

---

## 10. デプロイ

| サービス | URL |
|---------|-----|
| GitHub | https://github.com/kua08f91-hash/keiba-oracle |
| Vercel | https://jra-prediction-app-ochre.vercel.app |
| Railway | https://keiba-oracle-api-production.up.railway.app |
