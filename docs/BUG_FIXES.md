# バグ修正一覧（Phase 0）

> Railway 移行の**前**に適用すべき修正。これらを抱えたままクラウドに載せても、
> 同じ障害がクラウド上で再現するだけで解決にならない。

## 修正対象一覧

| # | ファイル | 行 | 重大度 | 工数 |
|---|---|---|---|---|
| 1 | `backend/refresh_raceday.py` | 216 | **Critical** (毎回死ぬ) | 1分 |
| 2 | `backend/realtime_worker.py` | 84-158 | **High** (オッズ一時消失) | 30分 |
| 3 | `backend/realtime_worker.py` | 全体 | **High** (TZずれ) | 1時間 |
| 4 | `docs/index.html` | 179-192, 521-535 | **Medium** (UIに古いデータ) | 30分 |

---

## バグ #1: `refresh_raceday.py:216` NameError

### 症状

`rapid_races` が空の時に実行される「標準単発パス」で未定義変数を参照。**実行のたびに NameError で停止**。

### 原因

L117-118 で `wide_start` / `wide_end` を定義しているが、L216 で存在しない `window_start` / `window_end` を参照している（typo）。

```python
# backend/refresh_raceday.py:117-118
wide_start = now + timedelta(minutes=10)
wide_end = now + timedelta(minutes=20)

# 同ファイル:216
if not (window_start <= start_dt <= window_end):  # ← NameError
    continue
```

### 修正パッチ

**Option A: typo 修正（最小変更）**

```diff
-            if not (window_start <= start_dt <= window_end):
+            if not (wide_start <= start_dt <= wide_end):
```

**Option B: 標準単発パス自体を削除（推奨）**

rapid_races ブロックで既に同等処理を rapid refresh mode で実行しているため、L200-300 の standard 単発パスは dead code に近い。削除して単純化する:

```diff
-    # Standard single-pass for races outside rapid window
-    for schedule in schedules:
-        ...  # L200-300 を全削除
-    if refreshed == 0:
-        print("  No races in refresh window (25-35 min before post)")
-    else:
-        print(f"\n  Refreshed {refreshed} race(s)")
-    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done")
+    print(f"[{datetime.now().strftime('%H:%M:%S')}] No rapid window races. Done.")
```

### 受け入れテスト

```bash
# rapid window 外の時刻で実行してもNameErrorが出ないことを確認
python3 -m backend.refresh_raceday
```

---

## バグ #2: `realtime_worker.py` オッズ取得失敗時の全削除

### 症状

netkeiba API が一時的にエラーを返した時、`CombinationOdds` テーブルの既存データを **全削除したまま再挿入されない**。結果、予想が「オッズ未確定」状態に戻る。

### 原因

```python
# backend/realtime_worker.py:84-118
def fetch_combination_odds(self, race_id: str) -> dict:
    result = {}
    for api_type, bet_type in TYPE_MAP.items():
        try:
            # ... fetch ...
            if entries:
                result[bet_type] = entries
        except Exception:
            pass  # ← エラーを無視、result は空のまま

    return result  # 空 dict が返る
```

```python
# backend/realtime_worker.py:139
db.query(CombinationOdds).filter(CombinationOdds.race_id == race_id).delete()
for bet_type, entries in combo_odds.items():  # combo_odds が空なら何も insert されない
    for e in entries:
        db.add(...)
```

### 修正パッチ

```diff
+# backend/realtime_worker.py:84
 def fetch_combination_odds(self, race_id: str) -> dict:
     TYPE_MAP = {4: "umaren", 5: "wide", 7: "sanrenpuku", 8: "sanrentan"}
     result = {}
+    failures = 0
     for api_type, bet_type in TYPE_MAP.items():
         try:
             # ... fetch ...
             if entries:
                 result[bet_type] = entries
         except Exception as e:
-            pass
+            logger.warning("fetch_combination_odds type=%d failed: %s", api_type, e)
+            failures += 1
+    # 半数以上失敗したら取得失敗とみなす
+    if failures > len(TYPE_MAP) // 2:
+        return None
     return result
```

```diff
+# backend/realtime_worker.py:120
 def save_odds_to_db(self, race_id: str, win_odds: dict, combo_odds: dict):
+    if combo_odds is None:
+        logger.warning("Skipping CombinationOdds update for %s (fetch failed)", race_id)
+        combo_odds = {}  # win_odds は保存、combo_odds のみスキップ
     now = datetime.utcnow()
     db = get_session()
     try:
-        # Combination odds (replace old, keep latest)
-        db.query(CombinationOdds).filter(CombinationOdds.race_id == race_id).delete()
-        for bet_type, entries in combo_odds.items():
-            ...
+        # Combination odds: 新データが取得できた場合のみ replace
+        if combo_odds:
+            db.query(CombinationOdds).filter(CombinationOdds.race_id == race_id).delete()
+            for bet_type, entries in combo_odds.items():
+                for e in entries:
+                    key = "-".join(f"{h:02d}" for h in e["horses"])
+                    db.add(CombinationOdds(
+                        race_id=race_id, bet_type=bet_type, horses_key=key,
+                        odds=e["odds"], captured_at=now,
+                    ))
```

### 受け入れテスト

```python
# tests/test_realtime_worker.py に追加
def test_save_odds_preserves_on_fetch_failure(monkeypatch):
    """netkeiba API失敗時にCombinationOddsが消されないこと"""
    worker = RealtimeWorker()
    # DB にダミー odds を入れる
    db = get_session()
    db.add(CombinationOdds(race_id="TEST", bet_type="umaren", horses_key="01-02", odds=5.0, captured_at=datetime.utcnow()))
    db.commit()
    db.close()

    # combo_odds=None で save
    worker.save_odds_to_db("TEST", win_odds={}, combo_odds=None)

    # 既存データが残っていることを確認
    db = get_session()
    count = db.query(CombinationOdds).filter(CombinationOdds.race_id == "TEST").count()
    assert count == 1
    db.close()
```

---

## バグ #3: TZ混在

### 症状

`datetime.now()` (ローカルTZ) と `datetime.utcnow()` (UTC) が混在。VPS/コンテナのTZが JST でない場合、**発走10分前凍結ルールが9時間ズレる**。

### 原因箇所

```bash
# 検索:
grep -rn "datetime.now()\|datetime.utcnow()" backend/
```

主要な混在箇所:
- `realtime_worker.py:61` `self.today = datetime.now().strftime("%Y%m%d")` （日付判定）
- `realtime_worker.py:122` `now = datetime.utcnow()` （DB保存）
- `realtime_worker.py:317` `now = datetime.now()` （発走時刻比較）
- `main.py:309` `today = datetime.now()` （レース日一覧）
- `main.py:429` `date = datetime.now().strftime("%Y%m%d")`

### 修正方針

**原則**:
- **DB保存は全てUTC** (`datetime.now(timezone.utc)` を使用、`utcnow()` はdeprecated)
- **ユーザー表示・スケジュール判定は JST**
- **境界で必ずconversion**

### 修正パッチ

```python
# backend/_tz.py (新規作成)
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9), name="JST")

def now_jst() -> datetime:
    """現在時刻（JST aware）"""
    return datetime.now(JST)

def now_utc() -> datetime:
    """現在時刻（UTC aware）。DB保存用"""
    return datetime.now(timezone.utc)

def to_jst(dt: datetime) -> datetime:
    """naive または UTC の datetime を JST に変換"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(JST)
```

```diff
 # backend/realtime_worker.py
+from backend._tz import now_jst, now_utc, JST

 class RealtimeWorker:
     def __init__(self):
         init_db()
         self.predictor = MLScoringModel()
-        self.today = datetime.now().strftime("%Y%m%d")
+        self.today = now_jst().strftime("%Y%m%d")
         ...

     def save_odds_to_db(self, race_id, win_odds, combo_odds):
-        now = datetime.utcnow()
+        now = now_utc()
         ...

     def get_minutes_to_post(self, race_id: str) -> float:
         ...
         try:
             h, m = st.split(":")
-            now = datetime.now()
-            start = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
+            now = now_jst()
+            start = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
             return (start - now).total_seconds() / 60
```

### 受け入れテスト

```python
def test_tz_aware_datetime():
    """全エントリポイントで tzinfo が付いていること"""
    from backend._tz import now_jst, now_utc
    assert now_jst().tzinfo is not None
    assert now_utc().tzinfo is not None

def test_get_minutes_to_post_across_tz():
    """UTC環境とJST環境で結果が一致すること"""
    # os.environ["TZ"] を変えて同じ race_id で呼び出し、結果が一致
    ...
```

### Railway設定

Railway の環境変数に `TZ=Asia/Tokyo` を追加することで、OSレベルのTZも JST に揃える（2重保険）。

---

## バグ #4: フロントエンド staticData フォールバック

### 症状

GitHub Pages 経由で `docs/data/predictions.json` を読んだ場合、その JSON が古くても **新しいAPIを叩かない**。また詳細画面の自動更新タイマーが `racecard` しか叩かず **買い目が更新されない**。

### 原因

```javascript
// docs/index.html:179-192
if(staticData){
    const dayData=staticData.find(d=>d.date===ds);
    if(dayData){
        // 時刻チェックなしで static を使う
        scheduleCache[ds]=sc; renderDay(sc,ds); return;
    }
}
```

```javascript
// docs/index.html:521-535 (自動更新タイマー)
setInterval(() => {
    fetch(`${API}/racecard/${_currentDetailId}?_t=${Date.now()}`)
        .then(...)
    // ↑ optimized-bets を叩いていない
}, 60 * 1000);
```

### 修正パッチ

```diff
 // docs/index.html:179
 if(staticData){
     const dayData=staticData.find(d=>d.date===ds);
-    if(dayData){
+    // exportedAt から15分以上経過していたらAPIを優先
+    const STALE_MS = 15 * 60 * 1000;
+    const exportedAt = window._perf?.exportedAt
+        ? new Date(window._perf.exportedAt).getTime()
+        : 0;
+    const isStale = Date.now() - exportedAt > STALE_MS;
+    if(dayData && !isStale){
         // ... static 使用 ...
     }
 }
```

```diff
 // docs/index.html:521
 setInterval(() => {
     if (_currentDetailId && document.getElementById("page-detail").classList.contains("active")) {
-        fetch(`${API}/racecard/${_currentDetailId}?_t=${Date.now()}`)
-            .then(r => r.ok ? r.json() : null)
-            .then(data => {
-                if (!data) return;
-                dE = data.entries;
-                renderTable();
-            })
-            .catch(() => {});
+        Promise.all([
+            fetch(`${API}/racecard/${_currentDetailId}?_t=${Date.now()}`).then(r => r.ok ? r.json() : null),
+            fetch(`${API}/optimized-bets/${_currentDetailId}?_t=${Date.now()}`).then(r => r.ok ? r.json() : null),
+        ]).then(([card, bets]) => {
+            if (card) { dE = card.entries; }
+            if (bets) { dB = bets.bets || dB; dLongshot = bets.longshot || dLongshot; }
+            renderTable();
+            renderBets();
+        }).catch(() => {});
     }
 }, 60 * 1000);
```

### 受け入れテスト

手動確認:
1. `docs/data/predictions.json` を古いタイムスタンプで上書き
2. GitHub Pages URL を開く
3. Network tab で `/api/race-list` が呼ばれていることを確認

---

## 適用順序

1. **#1 を先に** (1分、crash防止)
2. **#3 TZ統一** (1時間、#2修正時の時刻扱いに影響するため先行)
3. **#2 空dict防御** (30分)
4. **#4 フロント** (30分、後方互換のみ)

全修正後、テスト `python3 -m pytest tests/ -q` が pass することを確認。

---

## 関連

- 全体計画: [RAILWAY_MIGRATION.md](./RAILWAY_MIGRATION.md)
- netkeiba 検証: [../scripts/verify_netkeiba_access.py](../scripts/verify_netkeiba_access.py)
