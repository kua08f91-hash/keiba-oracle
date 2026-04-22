# KEIBA ORACLE — AI競馬予想アプリ

JRA（日本中央競馬会）の全レースをAIが分析し、最適な買い目を提案する競馬予想アプリケーションです。

## 概要

- **AIスコアリング**: 19の分析ファクターで各馬を0-100でスコアリング
- **買い目最適化**: モンテカルロシミュレーション + EV最適化で全8券種から最適な5点を選出
- **リアルタイム対応**: 発走前にオッズ・予想を自動更新、10分前に確定凍結
- **マルチプラットフォーム**: PC・スマートフォン対応のレスポンシブUI

## デモ・アクセス

- **パブリック版**: [https://kua08f91-hash.github.io/keiba-oracle/](https://kua08f91-hash.github.io/keiba-oracle/)
- **ローカル版**: `http://localhost:8080/preview.html`（バックエンド起動後）

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| フロントエンド | HTML + Tailwind CSS (CDN) + Vanilla JS |
| バックエンド | FastAPI + Python 3.9 |
| データソース | netkeiba.com, keibabook.co.jp (フォールバック) |
| データベース | SQLite (SQLAlchemy ORM) |
| ホスティング | GitHub Pages (静的) |
| CI/CD | cron自動パイプライン |

## セットアップ

### 必要環境

- Python 3.9+
- pip (Python パッケージマネージャー)

### インストール

```bash
git clone https://github.com/kua08f91-hash/keiba-oracle.git
cd keiba-oracle
pip install -r backend/requirements.txt
```

### バックエンド起動

```bash
python3 -m uvicorn backend.main:app --port 8000 --host 0.0.0.0
```

### ローカルプレビュー

```bash
python3 -m http.server 8080
# ブラウザで http://localhost:8080/preview.html を開く
```

### 予想データのエクスポート（GitHub Pages用）

```bash
python3 -m backend.export_predictions
git add docs/data/predictions.json
git commit -m "Update predictions"
git push
```

## アーキテクチャ

### スコアリングエンジン (v5 — 19ファクター)

ルールベースの重み付き線形モデル。MLモデル（v7）より安定したROIを実現。

#### 19の分析ファクター

| カテゴリ | ファクター | 重み |
|---------|-----------|------|
| **コア** | trackDirection（回り適性）| 13.0% |
| | trackCondition（馬場適性）| 13.0% |
| | jockeyAbility（騎手能力）| 10.0% |
| **条件マッチング** | sameDistance（同距離実績）| 7.0% |
| | sameSurface（同馬場種実績）| 7.0% |
| | sameCondition（同馬場状態）| 5.0% |
| | pastPerformance（近走成績）| 5.0% |
| **拡張スクレイピング** | speedFigure（速度指数）| 5.0% |
| | runningStyle（脚質一貫性）| 4.0% |
| | daysSinceLast（休養明け）| 2.0% |
| | weightCarriedTrend（斤量変化）| 2.0% |
| **補助** | trackSpecific（コース別実績）| 5.0% |
| | formTrend（調子トレンド）| 4.0% |
| | ageAndSex（年齢・性別）| 4.0% |
| | weightCarried（斤量）| 3.0% |
| | horseWeightChange（馬体重変動）| 3.0% |
| | trainerAbility（調教師能力）| 3.0% |
| **血統** | courseAffinity（コース適性）| 3.0% |
| | distanceAptitude（距離適性）| 2.0% |

最終スコア: `score = analytical × 85% + market × 15%`

### 買い目オプティマイザ (v8 — 全8券種対応)

| 対応券種 |
|---------|
| 単勝・複勝・枠連・馬連・馬単・ワイド・3連複・3連単 |

**戦略**:
1. **ROIアンカー**: 3連単を1枠確保（ROI実績250%+）
2. **ヒットアンカー**: ワイドまたは複勝を1枠確保（的中率20%+）
3. **EV順フィル**: 残り枠をEV順に全券種から選出

**信頼度ゲート**: 最良EVが低いレースは自動的に投資額を削減（最大5点→3点）

### データフロー

```
[netkeiba / keibabook] → [スクレイパー] → [SQLite DB]
                                              ↓
                               [v5 スコアリング (19ファクター)]
                                              ↓
                               [買い目オプティマイザ (MC 5000回)]
                                              ↓
                          [predictions.json] → [GitHub Pages / ローカル]
```

## UI機能

### レース一覧

- **GI/GII/GIIIバッジ**: raceInfo.gradeフィールド優先 + 名前リスト照合
- **自信度マーク**: ◎と◯のスコア差 ≥10pt → 🔴自信 / ≥5pt → 🟡注目
- **レスポンシブ**: モバイル1列、タブレット2列、PC3列
- **ヘッダータップ**: 最新データにリロード

### レース詳細

| デバイス | 表示カラム |
|---------|----------|
| **モバイル** | 予想・番号・馬名（騎手/性齢/斤量サブ行）・オッズ・人気・Score |
| **PC** | 予想・枠・番号・馬名・性齢・斤量・騎手・調教師・オッズ・人気・Score |

- **買い目**: 最大5点、EV・オッズ・的中率表示
- **ワイド/複勝**: レンジオッズ表示（例: 11.9-12.9倍）
- **穴場券**: 20-100倍の高配当1点（オプション）

## 自動パイプライン

| スケジュール | スクリプト | 内容 |
|------------|----------|------|
| 木曜 21:00 | `prefetch_weekly.py` | 次週末レースデータ事前取得 |
| 土曜 07:03 | `export_predictions.py` | 朝イチ予想エクスポート + GitHub Pages更新 |
| 日曜 07:07 | `export_predictions.py` | 同上（日曜分） |
| 土日 12:03/15:03 | `export_predictions.py` | 最新オッズで再エクスポート + push |
| 土日 9:00-16:55 | `refresh_raceday.py` | 5分毎ライブオッズ更新 |
| 月曜 06:03 | `auto_improve.py` | 結果収集 + 精度評価 + v5重み再最適化 |

## ルール（ロック済み）

- ✅ 発走10分前に予想凍結（以降変更なし）
- ✅ MIN_ODDS 2.0（1倍台は推奨しない）
- ✅ 穴場券 20-100倍
- ✅ レース当日のモデル変更禁止
- ✅ 買い目最大5点 / レース
- ✅ Monte Carlo固定シード（再現性保証）

## データソース冗長化

| データ | プライマリ | フォールバック |
|--------|----------|-------------|
| レース一覧 | netkeiba | DBキャッシュ |
| 出馬表 | netkeiba | DBキャッシュ |
| リアルタイムオッズ | netkeiba API | 推定値 |
| レース結果 | db.netkeiba | keibabook |
| 払戻データ | db.netkeiba | keibabook |

## 実績（2026年）

### シミュレーション（3月 300レース・実払戻）

| 指標 | 値 |
|------|------|
| ROI | 134.4% |
| 総投資額 | ¥142,100 |
| 総払戻額 | ¥190,960 |
| 収支 | +¥48,860 |

### ライブ実績（4/18-19 週末）

ユーザー馬券選択: **購入¥32,500 → 払戻¥73,850 (ROI 227%)** 🎯

## テスト

```bash
python3 -m pytest tests/ -q
# 638 tests passing
```

| テストファイル | テスト数 | 対象 |
|-------------|---------|------|
| test_bet_optimizer.py | 89 | 買い目最適化ロジック |
| test_new_factors.py | 119 | 新7ファクター計算 |
| test_grade_badges.py | 231 | 重賞バッジ判定 |
| test_parser_grade_detection.py | 50 | スクレイパーgrade検出 |
| test_db_fallback_and_new_features.py | 58 | DBフォールバック |
| test_scoring.py | 43 | スコアリングエンジン |
| test_feature_engineering.py | 24 | 特徴量エンジニアリング |
| test_auto_improve.py | 12 | 自動改善パイプライン |
| test_refresh_raceday.py | 11 | レース当日更新 |

## ディレクトリ構成

```
keiba-oracle/
├── docs/                          # GitHub Pages
│   ├── index.html                 # パブリック版UI
│   └── data/predictions.json      # 静的予想データ
├── preview.html                   # ローカル版UI
├── backend/
│   ├── main.py                    # FastAPI エンドポイント
│   ├── predictor/
│   │   ├── scoring.py             # v5 スコアリングエンジン (19ファクター)
│   │   ├── ml_scoring.py          # エンジン選択 + 重みロード
│   │   ├── bet_optimizer.py       # 買い目最適化 (全8券種)
│   │   ├── factors.py             # ファクター計算関数
│   │   └── feature_engineering.py # ML特徴量 (バリデーション用)
│   ├── scraper/
│   │   ├── netkeiba.py            # netkeiba スクレイパー
│   │   ├── keibabook.py           # keibabook バックアップスクレイパー
│   │   ├── odds.py                # オッズ取得 (リアルタイム + 推定)
│   │   └── parser.py              # HTML パーサー
│   ├── database/
│   │   ├── db.py                  # SQLite接続
│   │   └── models.py              # SQLAlchemy モデル (7テーブル)
│   ├── export_predictions.py      # 静的JSON エクスポート
│   ├── realtime_worker.py         # レース当日バックグラウンドワーカー
│   ├── auto_improve.py            # 月曜自動改善パイプライン
│   ├── optimize_weights_real.py   # v5重み最適化 (実パイプライン)
│   ├── cross_validate.py          # 複数月クロス検証
│   └── analyze_drift.py           # 時系列ドリフト分析
├── data/
│   ├── optimized_weights.json     # 最適化済みv5重み
│   ├── historical_races.json      # 訓練データ (1,227レース)
│   └── jra_races.db               # SQLite DB
├── tests/                         # 638テスト
├── CLAUDE.md                      # 開発ガイド
├── render.yaml                    # Render.com デプロイ設定
└── README.md                      # このファイル
```

## ライセンス

Private repository. All rights reserved.

## 開発

詳細な開発ガイド・設定は [CLAUDE.md](CLAUDE.md) を参照してください。
