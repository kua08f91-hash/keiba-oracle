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

### Scoring Engine (v7 — Dual ML with Value Detection)
- **Architecture**: Dual model — Analytical (39 features, market-free) + Combined (45 features)
- **Score blend**: 60% Combined + 40% Analytical → reduces market dependency
- **Value edge**: AI probability vs market-implied probability → detects undervalued horses
- **Key new features**: horse weight kg/change, carry vs field, win/place rate, same-track results, frame bias, debut detection
- **Training data**: 1,107 races (15,044 entries), time-based split
- **Market independence**: AI独自選択 != 1番人気 in 33% of races (was 18% in v6)
- **Fallback**: v5 WeightedScoringModel if trained_model.pkl missing
- **Retrain**: `python -m backend.train_model`

### Dynamic Bet Optimizer
- Converts AI scores to win probabilities via softmax (temperature=9.5)
- Race-pattern-based temperature adjustment (本命堅軸×0.85, 混戦模様×1.15)
- Monte Carlo simulation (5000 samples) for hit probability estimation
- Calculates EV = P(hit) * odds - 1 for each candidate bet
- Selects top 5 bets per race by EV with diversification (max 2 of same type)
- Adapts bet types to each race's characteristics (no fixed strategy)

### Key API Endpoints
- `GET /api/racecard/{race_id}` — Race card with AI predictions
- `GET /api/optimized-bets/{race_id}` — Dynamic top 5 bets by EV
- `GET /api/odds/{race_id}` — Combination odds (real or estimated)
- `GET /api/race-list?date=YYYYMMDD` — Available races
- `GET /api/race-dates?weeks=3` — Upcoming race dates

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
| **土日 9:30-16:30** | `refresh_raceday.py` (5分毎) | 各レース30分前に馬場状態+最新オッズで予想再計算 |
| **月曜 06:00** | `auto_improve.py` | 結果収集→精度評価→自動再訓練 |

### Auto-Improve Pipeline (月曜自動実行)
1. 直近2週の全レース結果をnetkeibaから収集
2. `data/historical_races.json` に追加 (重複除外)
3. 現モデルの精度を評価 (単勝/馬連/ワイド/3連複)
4. 精度低下を検知したら自動再訓練:
   - 単勝精度 < 25% → 再訓練トリガー
   - ワイド精度 < 40% → 再訓練トリガー
   - 3回連続精度低下 → 再訓練トリガー
5. パフォーマンスログを `data/performance_log.json` に記録

```bash
# Manual run:
/usr/bin/python3 -m backend.auto_improve
# Logs: /tmp/keiba-improve.log, /tmp/keiba-prefetch.log
```

## Data
- `data/historical_races.json` — 1,107 historical races (2023-2024) for optimization
- Race IDs: 12-digit format, e.g., `202606030201` (year+meeting+course+day+race)
