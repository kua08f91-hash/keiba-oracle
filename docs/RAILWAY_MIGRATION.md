# Railway + Vercel 移行計画

> **目的**: 3時間同期遅延の解消、家庭PC依存の排除、運用負荷の削減
> **推奨構成**: Railway (Postgres + worker + API) + Vercel (frontend)
> **月額**: $10-15
> **工数**: 1-2日（バグ修正込み）

---

## 1. TL;DR

現状は **家庭PC + cron + git push** 経由で `docs/data/predictions.json` を土日 12:03/15:03 の2回だけ更新する構成。GitHub Pages 経由の閲覧者（スマホ/PC）には **最大3時間の遅延** が発生する。

本ドキュメントは:

1. **同期遅延の原因7箇所**を特定
2. **コード側バグ4件**を指摘
3. **Railway + Vercel への移行計画**を提示
4. **移行前の検証手順**と**オーナー向けデプロイ手順**を提供

移行後は:

- データ同期遅延 **3時間 → 数秒〜10秒**
- 家庭PC起動不要（24/7 クラウド稼働）
- `git push` パイプライン撤廃（リポジトリ汚染解消）
- SQLite → Postgres で並行書き込み解消
- 運用画面（ログ/メトリクス）をRailwayダッシュボードで一元化

---

## 2. 現状のデータフローと同期遅延の原因

### 2.1 現状アーキテクチャ

```
[netkeiba]
   ↓ scrape (家庭PCから requests 同期、5秒sleep)
[realtime_worker.py] (家庭PC常駐、Sat/Sun 9-17時)
   ↓ 書き込み
[SQLite: data/jra_races.db]
   ↓ cron 土日 12:03/15:03
[export_predictions.py] (40分かけて再スクレイプ)
   ↓ JSON生成
[docs/data/predictions.json]
   ↓ git add/commit/push
[GitHub Pages CDN]
   ↓
[ブラウザ (スマホ/PC)]  ← 最大3時間古いデータ
```

### 2.2 同期遅延を生む7つの問題

| # | 場所 | 症状 | 原因 |
|---|---|---|---|
| 1 | `backend/refresh_raceday.py:216` | 標準単発パスがNameErrorで毎回死ぬ | `window_start` / `window_end` が未定義（`wide_start` の typo） |
| 2 | `backend/realtime_worker.py:84-118, 139` | オッズ取得失敗時に `CombinationOdds` が全削除される | `except Exception: pass` で空dict返却 → `delete()` だけ走る |
| 3 | `backend/realtime_worker.py:315-321` | 10分前凍結がTZずれで狂う | `datetime.utcnow()` (DB保存) と `datetime.now()` (スケジュール) が混在 |
| 4 | `backend/realtime_worker.py:354-406` | 発走間際に全レース直列処理が間に合わない | 1プロセス順次ループ、`asyncio` 未使用 |
| 5 | `backend/main.py:170-193` | APIハンドラが詰まる | DBキャッシュミス時にリクエスト内で同期 `requests.get` |
| 6 | `docs/index.html:179-192, 521-535` | staticData優先時に `optimized-bets` API が呼ばれない | フォールバックロジックが `racecard` しか叩かない |
| 7 | `export_predictions.py` + git push | 予想更新が土日 12:03/15:03 の2回だけ | cron頻度が低い + `git push` 経由 |

### 2.3 修正範囲マッピング

| 問題 | バグ修正で解決 | 移行で解決 |
|---|---|---|
| #1 NameError | ✅ | — |
| #2 空dict delete | ✅ | — |
| #3 TZ混在 | ✅ | — |
| #4 順次ループ | ⚠️ 一部（asyncio化） | ✅ Railway常駐なら sleep interval 短縮可 |
| #5 API同期HTTP | ✅ | ✅ Postgres常駐で完全に不要化 |
| #6 フロント | ✅ | ✅ Vercel化で static分岐削除 |
| #7 git push | ❌ | ✅ Postgres直読みで撤廃 |

**#7 だけは載せ替えでしか解決しない** → Railway 移行を推奨。

---

## 3. 代替案の評価

| 案 | 月額 | 工数 | PC不要 | 遅延 | 評価 |
|---|---|---|---|---|---|
| A: cron頻度UP + 軽量化 | $0 | 30分 | ❌ | 5-10分 | 家庭PC依存残る |
| B: Cloudflare Tunnel で家庭PC公開 | $0 | 1時間 | ❌ | リアルタイム | 家庭PC常時起動必須 |
| C: 自前VPS に移設 | ¥880/月〜 | 半日 | ✅ | リアルタイム | オーナー側のVPS契約必要 |
| D: GitHub Actions cron | $0 | 1時間 | ✅ | 5-10分 | Git履歴汚染、Actions IP の block リスク |
| E: Oracle Cloud Free Tier | $0 | 半日 | ✅ | リアルタイム | Oracle アカウント必要、稀に停止報告 |
| **F: Railway + Vercel** ★ | **$10-15** | **1-2日** | **✅** | **リアルタイム** | **運用負荷最小** |

### Railway推奨の根拠

**コスト = 運用工数の対価**。月$10-15は以下の運用負担を吸収する:

- `git push` パイプライン死活監視不要
- SQLite concurrent access 問題が消える
- realtime_worker の障害時 auto-restart
- ログ/メトリクス統合UI
- スキーマ変更を Alembic 標準化
- 将来の機能追加（LINE通知、Discord bot、課金化）が乗せやすい

年間 $120-180 ≈ **運用対応 6-9時間分**。GitHub Actions で failed run / netkeiba HTML変更 / git履歴整理に費やす時間を考慮すれば、2〜3ヶ月で元が取れる。

---

## 4. 移行前の検証（必須・30分）

Railway IP から netkeiba が叩けるかを **事前検証**。ブロックされる場合は計画全体を見直す必要あり。

### 4.1 ローカルbaseline取得（済）

```bash
python3 scripts/verify_netkeiba_access.py
```

**確認済み結果（2026-04-22）**: 家庭IPから HTML全エンドポイント + オッズJSON API 疎通OK。

### 4.2 Railway IP 検証

```bash
# Railway CLI インストール (要 Node.js)
npm i -g @railway/cli
railway login

# 検証プロジェクト作成（無料枠内）
cd /path/to/keiba-oracle
railway init --name keiba-oracle-verify
railway run -- pip install requests
railway run -- python3 scripts/verify_netkeiba_access.py

# 検証終了後は削除
railway down
```

### 4.3 判定基準

| 結果 | 判定 | 次アクション |
|---|---|---|
| `All endpoints OK` | ✅ Railway採用 | 本計画どおり進行 |
| `BOT DETECTION DETECTED` | ❌ Railway不可 | 案E (Oracle Cloud Free) または案C (VPS) に切替 |
| 一部 timeout | ⚠️ 要調査 | リトライ戦略 + User-Agent 見直し |

### 4.4 GitHub Actions からの検証（代替手段）

Railway CLIが使えない環境では、以下のworkflowでも同等の検証が可能:

```yaml
# .github/workflows/verify-netkeiba.yml
on: workflow_dispatch
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install requests
      - run: python3 scripts/verify_netkeiba_access.py
```

※ GitHub Actions IP と Railway IP は別range のため、**Railway採用判断の代替にはならない**。参考情報として。

---

## 5. Phase別 実装計画

### Phase 0: バグ修正（必須、約2時間）

**移行前に必ず修正**。バグを抱えたままクラウドに載せると、同じ障害がクラウドで再現するだけ。

#### 0.1 `backend/refresh_raceday.py:216` NameError

```diff
- if not (window_start <= start_dt <= window_end):
+ if not (wide_start <= start_dt <= wide_end):
```

または（意図次第で）標準単発パス自体を削除し rapid window のみ残す。

#### 0.2 `backend/realtime_worker.py:84-118` 空dict防御

```diff
def fetch_combination_odds(self, race_id: str) -> dict:
    TYPE_MAP = {4: "umaren", 5: "wide", 7: "sanrenpuku", 8: "sanrentan"}
    result = {}
    for api_type, bet_type in TYPE_MAP.items():
        try:
            # ... fetch ...
            if entries:
                result[bet_type] = entries
        except Exception:
-            pass
+            logger.warning("fetch_combination_odds type=%d failed, preserving existing", api_type)
+            return None  # 失敗フラグ
    return result
```

呼び出し側 `save_odds_to_db` で None チェックし、None なら `CombinationOdds.delete` をスキップ:

```diff
def save_odds_to_db(self, race_id, win_odds, combo_odds):
+    if combo_odds is None:
+        logger.warning("Skipping CombinationOdds update for %s (fetch failed)", race_id)
+        return
    # ... existing logic ...
```

#### 0.3 `backend/realtime_worker.py:315-321` TZ統一

```diff
+from datetime import timezone, timedelta
+JST = timezone(timedelta(hours=9))

def get_minutes_to_post(self, race_id: str) -> float:
    # ...
-    now = datetime.now()
+    now = datetime.now(JST)
-    start = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
+    start = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0, tzinfo=JST)
    return (start - now).total_seconds() / 60
```

全 `datetime.utcnow()` / `datetime.now()` を検索し、以下の原則で統一:
- **DB保存**: `datetime.now(JST)` または全UTC
- **ユーザー表示**: JST
- **外部API送信**: 仕様に従う

#### 0.4 `docs/index.html:180, 521` staticData stale判定

```diff
if(staticData){
    const dayData=staticData.find(d=>d.date===ds);
-   if(dayData){
+   const staleThreshold = 15 * 60 * 1000; // 15分
+   const isStale = dayData && window._perf?.exportedAt
+       && (Date.now() - new Date(window._perf.exportedAt).getTime()) > staleThreshold;
+   if(dayData && !isStale){
        // static data 使用
    }
+   // isStale または未見 → API フォールバック
}
```

また `setInterval(() => racecard)` ルーチンで `/api/optimized-bets/` も並列で呼ぶ:

```diff
 setInterval(() => {
    if (_currentDetailId && ...) {
-     fetch(`${API}/racecard/${_currentDetailId}?_t=${Date.now()}`)
+     Promise.all([
+       fetch(`${API}/racecard/${_currentDetailId}?_t=${Date.now()}`).then(r => r.json()),
+       fetch(`${API}/optimized-bets/${_currentDetailId}?_t=${Date.now()}`).then(r => r.json()),
+     ]).then(([card, bets]) => {
         // ...update both entries and bets...
+     });
    }
 }, 60 * 1000);
```

---

### Phase 1: Railway対応コード改修（約4時間）

#### 1.1 DB抽象化

`backend/database/db.py` を `DATABASE_URL` 環境変数駆動に:

```python
import os
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/jra_races.db")
# Railway は postgres:// を返すが SQLAlchemy は postgresql:// が正
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
```

#### 1.2 依存関係追加

`backend/requirements.txt`:

```
+psycopg2-binary>=2.9.9
+alembic>=1.13.0
```

#### 1.3 Alembic 初期化

```bash
alembic init backend/alembic
# alembic.ini の sqlalchemy.url を env var 参照に
# backend/alembic/env.py で DATABASE_URL を読み込む
alembic revision --autogenerate -m "initial schema"
```

既存SQLiteのスキーマを初期 migration 化。

#### 1.4 Worker / API のエントリポイント整備

```
backend/
├── main.py              # FastAPI (既存、Railway service: api)
├── realtime_worker.py   # daemon (既存、Railway service: worker)
├── prefetch_weekly.py   # Railway cron (木曜21:00)
├── auto_improve.py      # Railway cron (月曜06:00)
```

それぞれ `Procfile` or Railway config に記述。

#### 1.5 Export Predictions の軽量化

`export_predictions.py` を **DB読み出しモード** 追加:

```python
def main(from_db: bool = False):
    if from_db:
        # SQLite/Postgres から PredictionsCache を読んでJSON化
        # 40分 → 10秒
    else:
        # 既存: netkeibaから再スクレイプ
```

移行後は `from_db=True` で呼び出し、cron不要（常駐workerがDBを常に更新しているため）。

---

### Phase 2: Railway設定ファイル（約2時間）

#### 2.1 `railway.json`

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": { "builder": "NIXPACKS" },
  "deploy": {
    "numReplicas": 1,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### 2.2 サービス別 Dockerfile（または Procfile）

```
# Dockerfile.api
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ backend/
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
```

```
# Dockerfile.worker
FROM python:3.11-slim
# ... 同様 ...
CMD ["python3", "-m", "backend.realtime_worker"]
```

#### 2.3 `.env.example`

```
DATABASE_URL=postgresql://user:pass@host:5432/keiba
TZ=Asia/Tokyo
LOG_LEVEL=INFO
```

---

### Phase 3: Vercel対応（約1時間）

#### 3.1 `vercel.json`

```json
{
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://<railway-api-url>/api/:path*" }
  ],
  "headers": [
    {
      "source": "/data/predictions.json",
      "headers": [{ "key": "Cache-Control", "value": "public, max-age=30" }]
    }
  ]
}
```

#### 3.2 `docs/index.html` API 定数の環境駆動化

```diff
-const _CLOUD_API = "https://keiba-oracle-api.onrender.com/api";
+// Vercel rewrites が /api を Railway に転送するので相対パスでOK
+const _CLOUD_API = "/api";
```

#### 3.3 静的JSON export 廃止

`docs/data/predictions.json` の git 追跡を停止:

```
# .gitignore
docs/data/predictions.json
```

フロントは常に `/api/race-list` / `/api/racecard` を叩く。

---

### Phase 4: 段階カットオーバー（約1時間）

1. **Railway先行稼働** — 本番トラフィックを流さず、realtime_worker だけ動かす
2. **データ整合性確認** — 旧SQLite と 新Postgres で予想が一致することを確認
3. **Vercel dev deploy** — 新URLで動作確認
4. **DNS 切り替え** — 既存GitHub Pages URL (`https://kua08f91-hash.github.io/keiba-oracle/`) からVercel URLへ301 redirect
5. **旧cron停止** — 家庭PCのcron entry 削除
6. **`export_predictions.py` の git push 停止**

---

## 6. オーナー向けデプロイ手順

前提: 本PR がマージ済み。Railway / Vercel アカウント作成済み。

### 6.1 Railway セットアップ（15分）

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → `kua08f91-hash/keiba-oracle`
2. **Add Service** → **Database** → **PostgreSQL** を追加
3. 元のサービスを複製して worker 用にし、**Settings** → **Custom Start Command** を `python3 -m backend.realtime_worker` に
4. **Variables** タブで以下を追加:
   - `TZ=Asia/Tokyo`
   - `DATABASE_URL` は自動注入される
5. **Deploy** → ログで `INFO Starting main loop...` を確認
6. API 側のサービスで **Settings** → **Networking** → **Generate Domain** → URL をメモ

### 6.2 Vercel セットアップ（5分）

1. [vercel.com](https://vercel.com) → **Import Git Repository** → `keiba-oracle`
2. **Root Directory** を `docs` に設定
3. **Build Command** は空欄、**Output Directory** も空欄（静的HTML）
4. **Environment Variables** は不要（`vercel.json` の rewrites で自動設定）
5. **Deploy**
6. カスタムドメイン不要なら `https://keiba-oracle-xxx.vercel.app` を共有すればOK

### 6.3 cron 設定

Railway の **Cron** 機能で以下2つを登録:

| Schedule | Command | 目的 |
|---|---|---|
| `0 21 * * 4` | `python3 -m backend.prefetch_weekly` | 木曜21時: 次週末データ事前取得 |
| `0 6 * * 1` | `python3 -m backend.auto_improve` | 月曜6時: 結果収集+重み再最適化 |

### 6.4 動作確認

```bash
# API疎通
curl https://<vercel-url>/api/health
# → {"status":"ok"}

# 予想取得
curl https://<vercel-url>/api/race-dates?weeks=3
# → 開催日一覧
```

---

## 7. 月額コスト内訳

### 推奨構成: Worker weekend-only + API常駐

| サービス | スペック | 月額 |
|---|---|---|
| Railway Postgres | 共有, ~500MB | $1〜3 |
| Railway API (FastAPI) | 256MB, 24/7 | $5〜8 |
| Railway Worker | 512MB, 土日9-17時のみ (約68h/月) | $2 |
| Railway Cron (prefetch/improve) | 256MB, 月数時間 | $0.5 |
| Railway Hobby plan 基本料 | - | $5 (credit充当) |
| **Railway 小計** | | **$10〜15** |
| Vercel Hobby | 個人利用 | **$0** |
| **合計** | | **$10〜15/月** |

### スケールアップ閾値

| 状況 | 対応 |
|---|---|
| Railway使用量 > $5 credit | Pro プラン $20/mo |
| Postgres容量 > 1GB | `odds_snapshot` の週次aggregate cron追加 |
| Railway IP が netkeiba にブロック | Static Outbound IP アドオン $5/mo |

---

## 8. ロールバック手順

何かあれば即座に戻せるように:

1. **DNS** を GitHub Pages URL へ戻す（Vercel で設定変更だけ）
2. **家庭PC のcron** を再有効化
3. **旧 Render API** は **停止せずに残す**（一定期間並行稼働）
4. **SQLite バックアップ** は `data/jra_races.db.backup.YYYYMMDD` として移行前に保存

### 移行判定基準

| 期間 | 指標 | 許容値 |
|---|---|---|
| 移行後24時間 | API 5xx 発生率 | < 1% |
| 移行後1週間 | ROI（バックテスト） | 旧構成と±5ptが目安 |
| 移行後2週間 | オッズ取得成功率 | > 95% |

基準未達なら ロールバック → 原因調査。

---

## 9. 検証・受け入れチェックリスト

### 移行前（必須）

- [ ] `python3 scripts/verify_netkeiba_access.py` がローカルでOK
- [ ] Railway IPから同スクリプトを実行しOK
- [ ] Phase 0 のバグ4件すべて修正済み
- [ ] テスト638件 pass
- [ ] DB migration dry-run 成功

### 移行中

- [ ] Railway Postgres にテーブル作成完了
- [ ] 旧SQLiteからデータ移行（`historical_races.json` 再ロード or `pg_dump` 風インポート）
- [ ] realtime_worker が JSTで動作
- [ ] API が Postgres から予想返却
- [ ] Vercel デプロイ成功

### 移行後

- [ ] スマホで Vercel URL 開いて予想が見える
- [ ] 発走10分前に予想が凍結される
- [ ] 土日 9-17時のオッズ更新がリアルタイム
- [ ] 月曜 auto_improve が走る
- [ ] ログでエラー率 < 1%

---

## 10. 関連ファイル

| パス | 役割 |
|---|---|
| `scripts/verify_netkeiba_access.py` | 本番IP疎通確認スクリプト |
| `.github/workflows/verify-netkeiba.yml` | GitHub Actions 経由の疎通確認 |
| `docs/RAILWAY_MIGRATION.md` | 本ドキュメント |
| `docs/DEPLOY.md` | (Phase 1完了後に追加予定) デプロイ手順の最終版 |

---

## 11. 意思決定フロー

```
┌─────────────────────────────┐
│ Phase 0: バグ修正 (2h)       │  ← まずここから
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ netkeiba 疎通検証 (30min)   │
└──────────────┬──────────────┘
               ↓
         ┌─────┴─────┐
         │ Railway可?│
         └─────┬─────┘
          Yes  │  No
     ┌────────┴────────┐
     ↓                 ↓
┌──────────┐    ┌──────────────┐
│ Railway  │    │ Oracle Cloud │
│ 移行     │    │ Free or VPS  │
└──────────┘    └──────────────┘
```

---

## 付録A: 現在のcron構成

```
# crontab (家庭PC, 移行後は削除)
0 21 * * 4     cd /path/to/keiba-oracle && python3 -m backend.prefetch_weekly
3 7 * * 6      cd /path/to/keiba-oracle && python3 -m backend.export_predictions && git push
3 7 * * 0      cd /path/to/keiba-oracle && python3 -m backend.export_predictions && git push
3 12 * * 6,0   cd /path/to/keiba-oracle && python3 -m backend.export_predictions && git push
3 15 * * 6,0   cd /path/to/keiba-oracle && python3 -m backend.export_predictions && git push
*/5 9-16 * * 6,0  cd /path/to/keiba-oracle && python3 -m backend.refresh_raceday
3 6 * * 1      cd /path/to/keiba-oracle && python3 -m backend.auto_improve
```

## 付録B: netkeiba IP block時のフォールバック

Railway IP が block された場合:

### B.1 Oracle Cloud Free Tier (推奨)

- ARM Ampere A1 (4 OCPU / 24GB RAM) 永久無料
- 日本リージョン可（IP範囲が家庭ISPに近いと判定される可能性高）
- セットアップ半日、以後無料

### B.2 Static Outbound IP (Railway アドオン)

- 月$5追加でRailway に固定IP付与
- netkeiba側でそのIPをallowlistする手段がないため、**根本解決にはならない**可能性
- Railwayだけがblock対象の場合に有効

### B.3 家庭PC + Cloudflare Tunnel

- 家庭PCを常駐化、Cloudflare Tunnel (無料) でドメイン公開
- オーナーが家庭PC常駐を許容できる場合のみ

---

## 付録C: 参考リンク

- Railway Docs: https://docs.railway.app
- Vercel Docs: https://vercel.com/docs
- FastAPI on Railway: https://docs.railway.app/guides/fastapi
- SQLAlchemy + Alembic: https://alembic.sqlalchemy.org/

---

**更新履歴**

| 日付 | 内容 | 筆者 |
|---|---|---|
| 2026-04-22 | 初版作成 | — |
