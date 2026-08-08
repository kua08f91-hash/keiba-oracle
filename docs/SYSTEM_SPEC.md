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

## 3. 買い目戦略 (D5 ◎軸厚張り)

### コンセプト
◎(AIスコア最高馬)を軸に厚張り × 勝負レース絞り込み × ケリー基準金額配分

### 買い目構成

| 券種 | 相手 | 点数/R | 最低オッズ | 備考 |
|------|------|--------|-----------|------|
| 馬単 (umatan) | ◎→AI 2位 | 1点 | 5倍 | ◎1着固定 |
| 馬連 (umaren) | ◎-AI 2位 | 1点 | 3倍 | |
| ワイド (wide) | ◎-AI 2~4位 | 3点 | 2.5倍 | 的中安定源 |
| 単勝 (tansho) | ◎ | 0~1点 | 6倍 | ◎オッズ6倍以上時 |
| 複勝 (fukusho) | ◎ | 0~3点 | 2倍 | 1:3リスクヘッジ |

※ コンボ買い目: 最大5点/レース + 単複ヘッジ(別枠)

### 勝負レース判定

- **A (勝負)**: ◎score >= 79 → 全買い目購入
- **C (SKIP)**: ◎score < 79 → 購入見送り

### 金額配分: ハーフケリー基準

- f = ((b+1)×p - 1) / b × 0.5
- betSize: 1~10単位 (1単位=¥500)
- 単複: 単勝1単位:複勝3単位の固定比率

### トラックバイアス

コース別枠順バイアスを10場分データ化:
- 札幌: 芝内有利(+12), ダート内有利(+8)
- 新潟: 芝内有利(+8), ダート外やや有利(-3), 直線1000m大外有利
- 中京: 芝内やや有利(+6), ダートフラット(0)
- 重/不良馬場: 外枠有利に-5シフト
- drawBias重み: 4.0% (trackDirectionから移行)

### 検証結果 (8/8 36R)

| 条件 | ROI | 投資 | 収支 |
|------|-----|------|------|
| D5 ◎>=79, 馬単1+馬連1+ワイド3 | **202%** | ¥20,000 | +¥20,300 |
| D5 ◎>=78, 馬連2+ワイド3 | 163% | ¥25,000 | +¥15,750 |
| 旧D4 全レース3点 | 0% | ¥54,000 | -¥54,000 |

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
| DB TTL (通常) | 30日 |
| DB TTL (レース当日) | 30分 |
| DB TTL (オッズなし) | 5分 |
| 枠番=0自動無効化 | 50%超でframe=0なら再スクレイピング |
| オッズなし自動無効化 | 50%超でodds=Noneなら再取得 |
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
| 土曜 | 06:00, 09:00〜16:00 30分間隔 + リトライ3回 |
| 日曜 | 06:00, 09:00〜16:00 30分間隔 + リトライ3回 |

タイムアウト: 45分。concurrency制御あり。predictions.json + archive自動commit+push → GitHub Pages + Vercel自動デプロイ。

### ローカルcron (PCオン時のみ)

| スケジュール | 処理 |
|------------|------|
| 木曜 21:00 | prefetch_weekly (次週末データ事前取得) |
| 金-土 14-23時毎時 | prefetch_weekly (オッズ更新) |
| 土日 07:00 | export + git push |
| 土日 8:50 + 9:00 | realtime_worker (常駐、PIDロック付き) |
| 土日 9-16時毎時 | export + git push (GitHub Pages最新化) |
| 月曜 06:00 | auto_improve (結果収集+精度評価) |

---

## 6. リアルタイム更新

### realtime_worker

| 項目 | 設定 |
|------|------|
| 起動 | cron 8:50 (PIDロック付き、重複起動なし) |
| 更新開始 | 発走30分前 |
| 更新間隔 | 30秒 (≤7分), 60秒 (≤20分), 300秒 (それ以外) |
| 凍結 | 発走7分前 (最終オッズ→予想更新→frozen=True) |
| 終了 | 全レース発走10分後 |
| API自動凍結 | Workerなしでも7分前にAPIリクエスト時に自動凍結 |

### フロントエンド

| 項目 | 設定 |
|------|------|
| データ取得 | API優先 (Railway経由)、静的データはフォールバック |
| 詳細画面 | 30秒ポーリング (APIからオッズ+スコア+買い目を自動更新) |
| 凍結済みレース | ポーリング停止、凍結時点のデータ固定 |
| 一覧戻り時 | ポーリング停止 |
| オッズ欠落時 | /api/live-odds エンドポイントで直接netkeiba取得 |

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

---

## 11. 予想ロジック・ルール詳細

### スコア計算

```
score = analytical(22ファクター加重平均) × 60% + marketScore(オッズ) × 40%
```

- スコアが高い順に ◎◯▲▲△△ のマーク付与
- ◎ = AI 1位, ◯ = AI 2位, ▲ = AI 3-4位, △ = AI 5-6位
- 各ファクターは0〜100のスコアを返す (50=中立)
- softmax(temperature=9.5)でスコアを勝率に変換

### レースパターン別温度調整

| パターン | 温度倍率 | 効果 |
|---------|---------|------|
| 本命堅軸 | ×0.85 | 差が広がる=本命有利 |
| 2強対決 | ×0.92 | やや差が広がる |
| 標準配置 | ×1.0 | 変更なし |
| 混戦模様 | ×1.15 | 差が縮まる=波乱含み |

### 買い目選出 (D5 ◎軸厚張り戦略)

**コンセプト**: ◎(AIスコア最高馬)を軸にした厚張り。勝負レース(◎>=79)のみ購入

**買い目構成**: ◎軸展開 + 単複リスクヘッジ
- 馬単: ◎→AI 2位 (1点, 5倍以上)
- 馬連: ◎-AI 2位 (1点, 3倍以上)
- ワイド: ◎-AI 2~4位 (3点, 2.5倍以上)
- 単勝: ◎ (◎オッズ6倍以上の場合, 1単位)
- 複勝: ◎ (◎オッズ2倍以上の場合, 3単位)

**金額配分**: ハーフケリー基準 (betSize 1~10単位)

**勝負判定**: ◎score >= 79 → 購入, < 79 → SKIP

### 穴馬券 (longshot)

- 買い目とは別に1点だけ提示
- オッズ20〜100倍, hitProb>0.5%, EV>-0.3の候補から選出

### market_weight = 40% の構造的意味

- オッズ(人気)の影響が40% → 人気馬ほどスコアが高くなりやすい
- しかし買い目はオッズ20倍以上の中穴を狙う
- つまり「◎◯は人気馬が来やすいが、買い目はその人気馬と穴馬の組み合わせ」

### ファクター22個の内訳

**市場データ (40%)**

| ファクター | 最終重み | 説明 |
|-----------|---------|------|
| marketScore | 40.00% | オッズ/人気から算出 |

**レース適性 (23.5%)**

| ファクター | 最終重み | 説明 |
|-----------|---------|------|
| daysSinceLast | 5.58% | 休養期間(前走からの日数) |
| trackDirection | 4.09% | 同コース方向(右/左)での成績 |
| sameCondition | 3.62% | 同馬場状態での成績 |
| drawBias | 2.94% | 枠順有利不利(内枠有利etc) |
| sameSurface | 2.91% | 同芝/ダートでの成績 |
| trackCondition | 2.24% | 馬場状態適性 |
| sameDistance | 2.06% | 同距離での成績 |
| trackSpecific | 2.01% | 同競馬場での成績 |

**馬の実力 (22.2%)**

| ファクター | 最終重み | 説明 |
|-----------|---------|------|
| speedFigure | 5.12% | タイム/上がりからのスピード指数 |
| weightCarriedTrend | 4.21% | 斤量変化の影響 |
| pastPerformance | 3.53% | 過去着順の加重平均 |
| marginScore | 3.53% | 1着馬との着差(小さい=強い) |
| runningStyle | 2.86% | 脚質の一貫性 |
| agari3f | 1.76% | 上がり3Fタイム(末脚の速さ) |
| formTrend | 1.15% | 近走の着順推移 |

**関係者 (3.9%)**

| ファクター | 最終重み | 説明 |
|-----------|---------|------|
| jockeyAbility | 2.58% | 騎手の勝率 |
| trainerAbility | 1.36% | 調教師の勝率 |

**馬の属性 (8.4%)**

| ファクター | 最終重み | 説明 |
|-----------|---------|------|
| ageAndSex | 3.95% | 年齢・性別による有利不利 |
| distanceAptitude | 3.56% | 血統からの距離適性 |
| horseWeightChange | 0.71% | 馬体重の増減 |
| courseAffinity | 0.21% | 血統からの芝/ダ適性 |

**除外**

| ファクター | 理由 |
|-----------|------|
| weightCarried | 着順予測を悪化させる(HURTS判定) |

### オッズ未取得時のルール

- odds=Noneの馬: marketScore = 50.0 (中立, ペナルティなし)
- オッズ取得: 最大3回リトライ (バックオフ 3s→6s→9s)
- 2頭以上のoddsがあれば組合せオッズ推定実行

### レース当日の運用ルール

| タイミング | 動作 |
|-----------|------|
| 発走30分前 | オッズ取得開始 |
| 発走20〜10分前 | 60秒間隔で更新 |
| **発走7分前** | **最終オッズ取得 → 予想確定 → 凍結(frozen)** |
| 凍結後 | スコア・買い目を変更しない |
| フロント | 60秒ポーリングでAPIからスコア・オッズ・買い目を自動更新 (凍結前まで) |

### 予想データの更新タイミング

| 方法 | スケジュール |
|------|------------|
| GitHub Actions | 土日 06:00, 09:00〜16:00 毎時 (1時間間隔, PCオフでも動作) |
| フロント | レース詳細画面で60秒ごとにAPIリアルタイム更新 |
| 凍結 | 発走10分前に確定 → 以降変更なし |

### 予想アーカイブ

- export時に `docs/data/archive/predictions_YYYYMMDD.json` に日別保存
- 翌週exportで上書きされても振り返り可能
