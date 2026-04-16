# JRA Horse Racing AI Prediction App

## Project Overview
JRA (Japan Racing Association) horse racing prediction web app.
AI scores each horse (0-100) using 11 analytical factors, then a dynamic bet optimizer selects the top 5 bets per race by expected value (EV).

## Tech Stack
- **Frontend**: Next.js 14 + React 18 + TypeScript + Tailwind CSS (port 3000)
- **Backend**: FastAPI + Python 3.9 (port 8000)
- **Data Source**: netkeiba.com (scraping)
- **DB**: SQLite via SQLAlchemy

## Development Setup

### Start backend
```bash
cd "/Users/atsushi.furutani/Claude Code/jra-prediction-app"
/usr/bin/python3 -m uvicorn backend.main:app --port 8000 --host 0.0.0.0
```

### Start frontend
```bash
cd "/Users/atsushi.furutani/Claude Code/jra-prediction-app"
npx next dev
```

### Python 3.9 Compatibility
Always use `from __future__ import annotations` at the top of Python files.
Do NOT use `dict | None` syntax — it causes TypeError on 3.9.

## Architecture

### Scoring Engine (v5 Rule-Based, 19 Factors — LOCKED with Real-Pipeline Optimized Weights)
- **Architecture**: v5 WeightedScoringModel with **19 analytical factors** (linear model)
- **Validated performance** (real bet_optimizer pipeline, MC=5000):
  - **2026-04 (最新月): ROI 64.9% → 90.0% (+25.1pt)** — 本番環境に近い最新データ
  - 2023-07 Hold-out: 163.1% → 126.7% (-36.4pt) — 古い有利条件への特化は犠牲
  - Hold-out min ROI: 64.9% → **90.0% (+25.1pt)** — ワーストケース改善
  - Train mean: 139.1% → 204.5% (訓練: 2023-04,05,06)
- **Weights**: auto-optimized via `optimize_weights_real.py` using real bet_optimizer
  - 4回目の試行で成功（過去3回はsimulation function過学習で失敗）
  - 時系列ドリフト対策として最新月優先の判定基準を採用
- **Market weight**: 14.7% market + 85.3% analytical
- **Final blend**: `score = analytical * 0.853 + market * 0.147`

### 採用された最適化済み重み（latest-month-priority criteria）

**Top 5 (20%+ of total weight):**
| Factor | Weight | Original | Δ |
|--------|--------|----------|---|
| daysSinceLast | 10.48% | 2.00% | **+8.48pt** ▲ |
| speedFigure | 8.71% | 5.00% | +3.71pt ▲ |
| sameCondition | 8.16% | 5.00% | +3.16pt ▲ |
| runningStyle | 7.85% | 4.00% | +3.85pt ▲ |
| weightCarriedTrend | 7.15% | 2.00% | +5.15pt ▲ |

**Reduced:**
| Factor | Weight | Original | Δ |
|--------|--------|----------|---|
| trackCondition | 3.81% | 13.00% | -9.19pt ▼ |
| trackDirection | 6.94% | 13.00% | -6.06pt ▼ |
| jockeyAbility | 4.38% | 10.00% | -5.62pt ▼ |
| courseAffinity | 0.36% | 3.00% | -2.64pt ▼ |
| horseWeightChange | 1.20% | 3.00% | -1.80pt ▼ |

**Key insight**: 新ファクター（daysSinceLast, speedFigure, runningStyle, weightCarriedTrend）が上位を占める。
従来のtrack系重み（direction/condition/jockey）から、**条件一致性・休養周期・脚質**重視にシフト。
これは2026年の波乱傾向（1人気勝率25%）への適応。

### v5 19ファクター重み（LOCKED）

**Core track/condition (36%)**:
| Factor | Weight |
|--------|--------|
| trackDirection | 13.0% |
| trackCondition | 13.0% |
| jockeyAbility | 10.0% |

**Condition matching (24%)**:
| Factor | Weight |
|--------|--------|
| sameDistance | 7.0% (new) |
| sameSurface | 7.0% (new) |
| sameCondition | 5.0% (new) |
| pastPerformance | 5.0% |

**Enhanced from scraping (13%)**:
| Factor | Weight |
|--------|--------|
| speedFigure | 5.0% (new, 上がり/タイム) |
| runningStyle | 4.0% (new, 脚質一貫性) |
| daysSinceLast | 2.0% (new, 休養明け) |
| weightCarriedTrend | 2.0% (new, 斤量変化) |

**Supporting (17%)**:
| Factor | Weight |
|--------|--------|
| trackSpecific | 5.0% |
| formTrend | 4.0% |
| ageAndSex | 4.0% |
| weightCarried | 3.0% |
| horseWeightChange | 3.0% |
| trainerAbility | 3.0% |

**Pedigree (5%)**:
| Factor | Weight |
|--------|--------|
| courseAffinity | 3.0% |
| distanceAptitude | 2.0% |

### 最適化試行の教訓（4つのスクリプト、3失敗→1成功）

| スクリプト | アプローチ | 結果 |
|----------|-----------|------|
| `optimize_weights.py` | top-k accuracy最大化 | **失敗**: 的中≠利益 |
| `optimize_weights_roi.py` | 簡易simulation ROI | **失敗**: March 541%, CV 59.8%（過学習） |
| `optimize_weights_robust.py` | minimax + 簡易simulation | **失敗**: CV 86.4%（simulation関数への過学習） |
| **`optimize_weights_real.py`** | **実bet_optimizer + minimax + hold-out validation** | **成功**: 最新月 +25.1pt |

**成功の鍵**:
1. 評価関数として**実bet_optimizer**を使用（simulation関数への過学習を根絶）
2. MC_SAMPLES可変化（最適化時500→3分、本番5000→高精度）
3. 時系列 train/hold-out 分割（汎化性能を検証）
4. **最新月優先の判定基準**（時系列ドリフトに対応）

**重要**: 重みは本番運用中に定期的に更新することを推奨（毎月 or auto_improve時）

### テストカバレッジ（TDD実施済み）

- **全339テスト PASS**（119 tests追加）
- **factors.py 新7関数**: 97%+ カバレッジ
- **bet_optimizer.py**: 99% カバレッジ
- **netkeiba.py `_parse_past_race_td`**: 100% カバレッジ
- テストファイル: `tests/test_new_factors.py`（338行）

### TDDセッションで発見・修正された本番バグ

**場所**: `backend/scraper/netkeiba.py` `_parse_past_race_td`

**バグ**: サンプルテキスト `"58.514-14-14-"` でweight（58.5）抽出後のcorner正規表現がweightの末尾桁 "5" を取り込み、`[514, 14, 14]` を生成していた。

**影響範囲**: ライブスクレイピング時のみ（historical_races.json のpastRacesにはcorners/runningStyle情報なしで最適化への影響はゼロ）。

**修正** (lines 398-413): weight match が見つかった場合、その部分を text から除去してから corner 正規表現を実行。結果: `corners=[14,14,14]`, `runningStyle=追込` が正しく算出される。

### 時系列ドリフト観察
- 1番人気勝率: 2023年 32.8% → 2026年 25.2% (波乱傾向強まる)
- 勝ち馬オッズ中央値: 4.60x → 6.40x
- 2023年データはpastRaces情報が貧弱（pos/trackのみ）でsameDistance等の新ファクターが機能しない
- 2026年以降のライブデータでは新ファクターが完全に機能

### ML Models (Validation/Future Use)
- `trained_model.pkl` — v7 dual ML (Analytical 39f + Combined 45f) retained for validation via `MLScoringModel.predict_ml()`
- Not used in production predictions; auto_improve retrains weekly for drift detection
- Retrain: `python -m backend.train_model`

### Dynamic Bet Optimizer
- Converts AI scores to win probabilities via softmax (temperature=9.5)
- Race-pattern-based temperature adjustment (本命堅軸×0.85, 混戦模様×1.15)
- Monte Carlo simulation (5000 samples) for hit probability estimation
- Calculates EV = P(hit) * odds - 1 for each candidate bet
- Selects top 5 bets per race by EV with diversification (max 2 of same type)
- Adapts bet types to each race's characteristics (no fixed strategy)

### Key API Endpoints
- `GET /api/racecard/{race_id}` — Race card with AI predictions (DB cache → live fallback)
- `GET /api/optimized-bets/{race_id}` — Top 5 bets by EV (DB cache → live fallback)
- `GET /api/odds/{race_id}` — Combination odds (real or estimated)
- `GET /api/odds-history/{race_id}` — Odds time-series from realtime_worker
- `GET /api/race-list?date=YYYYMMDD` — Available races
- `GET /api/race-dates?weeks=3` — Upcoming race dates
- `GET /api/race-status?date=YYYYMMDD` — Race lifecycle statuses

### Frontend Proxy
`next.config.js` rewrites `/backend/:path*` → `http://localhost:8000/api/:path*`

## Key Files

### Backend
- `backend/main.py` — FastAPI app, all endpoints
- `backend/predictor/ml_scoring.py` — v6 ML scoring engine (HistGradientBoosting)
- `backend/predictor/feature_engineering.py` — 28 feature extraction (shared train/inference)
- `backend/predictor/scoring.py` — v5 scoring engine (fallback)
- `backend/predictor/factors.py` — 12 analytical factor calculations
- `backend/predictor/bet_optimizer.py` — Dynamic EV optimizer (MC simulation)
- `backend/train_model.py` — ML model training pipeline
- `backend/scraper/netkeiba.py` — Race card + race list scraping
- `backend/scraper/odds.py` — Odds fetching + estimation
- `backend/scraper/parser.py` — HTML parsing helpers
- `backend/simulate_betting.py` — Backtesting simulation script
- `backend/optimize_from_history.py` — Weight optimization (3-phase)
- `backend/collect_fast.py` — Historical data collection

### Frontend
- `app/page.tsx` — Main page (date picker, race selector, predictions)
- `components/BettingPredictions.tsx` — Dynamic bet display with EV badges
- `components/PredictionTable.tsx` — Horse ranking table
- `components/FrameColorBox.tsx` — JRA frame color indicator
- `lib/types.ts` — TypeScript interfaces

## Simulation
```bash
PYTHONUNBUFFERED=1 /usr/bin/python3 -m backend.simulate_march_fast
```
Runs all March 2026 races, ¥100 per bet x 5 bets/race, outputs per-race detail and ROI summary.

## Auto Pipeline (Cron)
Fully automated weekly cycle:

| Schedule | Script | Purpose |
|----------|--------|---------|
| **木曜 21:00** | `prefetch_weekly.py` | 次週末レースデータを事前取得+キャッシュ |
| **土曜 07:00** | `prefetch_weekly.py` | 最新オッズで予想を再計算 |
| **土日 9:00-17:00** | `realtime_worker.py` (常駐) | リアルタイムオッズ取得+予想更新+10分前凍結 |
| **月曜 06:00** | `auto_improve.py` | 結果収集→精度評価→自動再訓練 |

### Auto-Improve Pipeline (月曜自動実行)
1. 直近2週の全レース結果をnetkeibaから収集
2. `data/historical_races.json` に追加 (重複除外)
3. 現モデルの精度を評価 (単勝/馬連/ワイド/3連複)
4. 精度低下を検知したらMLモデル自動再訓練 (validation用):
   - 単勝精度 < 25% → 再訓練トリガー
   - ワイド精度 < 40% → 再訓練トリガー
   - 3回連続精度低下 → 再訓練トリガー
5. **v5重み自動最適化**: `optimize_weights.py` を実行 → data/optimized_weights.json更新
6. パフォーマンスログを `data/performance_log.json` に記録

```bash
# Manual run:
/usr/bin/python3 -m backend.auto_improve
# Logs: /tmp/keiba-improve.log, /tmp/keiba-prefetch.log
```

## Data
- `data/historical_races.json` — 1,107 historical races (2023-2024) for optimization
- Race IDs: 12-digit format, e.g., `202606030201` (year+meeting+course+day+race)
