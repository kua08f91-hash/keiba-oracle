"""Verify netkeiba accessibility from the current environment.

Tests all endpoints used by backend/scraper/netkeiba.py and reports:
- Source IP (so we know where requests originate)
- HTTP status, latency, response size
- Content validity (detects Cloudflare challenges, empty responses, block pages)
- Bot-detection signals (challenge pages, captcha)

Usage (local baseline):
    python3 scripts/verify_netkeiba_access.py

Usage (Railway):
    railway init
    railway run python3 scripts/verify_netkeiba_access.py

Or deploy as a one-shot service:
    railway up
    (then check logs)

Exit codes:
    0 = all endpoints reachable with valid content
    1 = at least one endpoint blocked / returns challenge / invalid content
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime

import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HTML_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}
API_HEADERS = {
    "User-Agent": UA,
    "X-Requested-With": "XMLHttpRequest",
}

# Pick an upcoming race (odds API only serves current/future races).
# 2026-04-25 (土) 東京 1R = 202605020101
SAMPLE_RACE_ID = "202605020101"
SAMPLE_DATE = "20260425"


def detect_block(text: str) -> str | None:
    """Return reason string if response looks like a block/challenge, else None."""
    if not text:
        return "empty body"
    lower = text.lower()
    markers = {
        "cloudflare challenge": ["cf-chl", "cloudflare", "challenge-platform"],
        "captcha": ["captcha", "recaptcha", "hcaptcha"],
        "403 block page": ["access denied", "you have been blocked", "forbidden"],
        "akamai": ["akamai", "reference #"],
        "bot detection": ["are you a robot", "unusual traffic"],
    }
    for reason, tokens in markers.items():
        hits = sum(1 for t in tokens if t in lower)
        if hits >= 2 or (reason == "cloudflare challenge" and "cf-chl" in lower):
            return reason
    return None


def fetch(url: str, headers: dict, referer: str | None = None, timeout: int = 15) -> dict:
    """Fetch URL and return structured report."""
    h = dict(headers)
    if referer:
        h["Referer"] = referer
    t0 = time.time()
    try:
        r = requests.get(url, headers=h, timeout=timeout)
        elapsed = time.time() - t0
        text = r.text
        block_reason = detect_block(text) if r.status_code == 200 else None
        return {
            "url": url,
            "status": r.status_code,
            "elapsed_sec": round(elapsed, 2),
            "size_kb": round(len(text) / 1024, 1),
            "content_type": r.headers.get("Content-Type", ""),
            "cf_ray": r.headers.get("CF-RAY", ""),
            "server": r.headers.get("Server", ""),
            "block_reason": block_reason,
            "ok": r.status_code == 200 and not block_reason,
            "body_head": text[:200] if not r.ok or block_reason else None,
        }
    except requests.RequestException as e:
        return {
            "url": url,
            "status": 0,
            "elapsed_sec": round(time.time() - t0, 2),
            "ok": False,
            "error": str(e),
        }


def fetch_json(url: str, headers: dict, referer: str | None = None) -> dict:
    """Fetch JSON API and validate structure."""
    report = fetch(url, headers, referer, timeout=10)
    if not report["ok"]:
        return report
    try:
        r = requests.get(url, headers={**headers, "Referer": referer or ""}, timeout=10)
        d = json.loads(r.text)
        data = d.get("data")
        if not isinstance(data, dict):
            report["json_valid"] = False
            report["json_note"] = f"no 'data' dict (got {type(data).__name__})"
            report["ok"] = False
        else:
            odds = data.get("odds", {})
            report["json_valid"] = True
            report["odds_types"] = list(odds.keys()) if isinstance(odds, dict) else []
    except json.JSONDecodeError as e:
        report["json_valid"] = False
        report["json_note"] = f"JSON parse failed: {e}"
        report["ok"] = False
    return report


def get_source_ip() -> str:
    """Report outgoing IP so we know which network we're on."""
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=5)
        return r.json().get("ip", "unknown")
    except Exception as e:
        return f"unknown ({e})"


def print_report(name: str, rep: dict):
    ok = "OK " if rep.get("ok") else "FAIL"
    status = rep.get("status", "-")
    elapsed = rep.get("elapsed_sec", "-")
    size = rep.get("size_kb", "-")
    print(f"  [{ok}] {name:20s} status={status} time={elapsed}s size={size}KB", end="")
    if rep.get("cf_ray"):
        print(f" cf-ray={rep['cf_ray']}", end="")
    if rep.get("block_reason"):
        print(f" BLOCK={rep['block_reason']}", end="")
    if rep.get("error"):
        print(f" ERROR={rep['error']}", end="")
    if rep.get("json_note"):
        print(f" JSON={rep['json_note']}", end="")
    if rep.get("odds_types"):
        print(f" odds_types={rep['odds_types']}", end="")
    print()
    if rep.get("body_head"):
        print(f"       body[:200]: {rep['body_head']!r}")


def main():
    print("=" * 70)
    print(f"netkeiba access verification  [{datetime.now().isoformat()}]")
    print("=" * 70)

    ip = get_source_ip()
    print(f"Source IP:  {ip}")
    print(f"Race ID:    {SAMPLE_RACE_ID}")
    print(f"Date:       {SAMPLE_DATE}")
    print()

    # 1. Race list (HTML)
    print("[1] Race list (HTML, 開催一覧)")
    rep1 = fetch(
        f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={SAMPLE_DATE}",
        HTML_HEADERS,
    )
    print_report("race_list_sub", rep1)

    # 2. Race card (HTML, 出馬表)
    print("\n[2] Race card (HTML, 出馬表)")
    rep2 = fetch(
        f"https://race.netkeiba.com/race/shutuba.html?race_id={SAMPLE_RACE_ID}",
        HTML_HEADERS,
    )
    print_report("shutuba", rep2)

    # 3. Race card past (HTML, 血統+過去5走)
    print("\n[3] Race card past (HTML, 血統+過去5走)")
    rep3 = fetch(
        f"https://race.netkeiba.com/race/shutuba_past.html?race_id={SAMPLE_RACE_ID}",
        HTML_HEADERS,
    )
    print_report("shutuba_past", rep3)

    # 4. Result page (HTML)
    print("\n[4] Result page (HTML, 結果)")
    rep4 = fetch(
        f"https://race.netkeiba.com/race/result.html?race_id={SAMPLE_RACE_ID}",
        HTML_HEADERS,
    )
    print_report("result", rep4)

    # 5. Live odds API (JSON)
    print("\n[5] Live odds API (JSON)")
    referer = f"https://race.netkeiba.com/odds/index.html?race_id={SAMPLE_RACE_ID}"
    odds_reports = []
    for api_type, label in [(1, "tansho"), (4, "umaren"), (5, "wide"), (7, "sanrenpuku"), (8, "sanrentan")]:
        url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={SAMPLE_RACE_ID}&type={api_type}&action=init"
        rep = fetch_json(url, API_HEADERS, referer=referer)
        print_report(f"odds_type={api_type}({label})", rep)
        odds_reports.append(rep)

    # 6. db.netkeiba (payouts fallback)
    print("\n[6] db.netkeiba (payouts)")
    rep6 = fetch(
        f"https://db.netkeiba.com/race/{SAMPLE_RACE_ID}/",
        HTML_HEADERS,
    )
    print_report("db_netkeiba", rep6)

    # Summary
    all_reports = [rep1, rep2, rep3, rep4, rep6] + odds_reports
    ok_count = sum(1 for r in all_reports if r.get("ok"))
    total = len(all_reports)

    print()
    print("=" * 70)
    print(f"Summary: {ok_count}/{total} endpoints reachable")
    print("=" * 70)

    # Bot detection signals
    blocks = [r for r in all_reports if r.get("block_reason")]
    if blocks:
        print("BOT DETECTION DETECTED:")
        for r in blocks:
            print(f"  {r['url']}")
            print(f"    reason: {r['block_reason']}")
        print("\n-> Railway/cloud IP is likely blocked. Stay on local/home IP.")
        return 1

    # All endpoints 403/5xx?
    bad = [r for r in all_reports if not r.get("ok")]
    if bad:
        print("\nFailures:")
        for r in bad:
            print(f"  {r.get('url', '?')}: status={r.get('status')} error={r.get('error', '-')}")
        return 1

    print("All endpoints OK. netkeiba is accessible from this IP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
