"""Simulate remaining Jan-Feb races (2/8-2/28 only)."""
from __future__ import annotations
import re, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from backend.scraper.netkeiba import fetch_race_list, fetch_race_card
from backend.scraper.odds import estimate_from_entries, fetch_combination_odds
from backend.predictor.ml_scoring import MLScoringModel
from backend.predictor.bet_optimizer import optimize_bets, detect_race_pattern, scores_to_probabilities
from backend.database.db import init_db

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
CM = {"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
predictor = MLScoringModel()

def fetch_payouts(rid):
    try:
        r = requests.get(f"https://db.netkeiba.com/race/{rid}/", headers=HEADERS, timeout=15)
        r.encoding = r.apparent_encoding or "UTF-8"
        soup = BeautifulSoup(r.text, "html.parser")
        payouts = {}
        for t in soup.select("table.pay_table_01"):
            for row in t.select("tr"):
                th = row.select_one("th"); tds = row.select("td")
                if not th or len(tds)<2: continue
                label = th.get_text(strip=True)
                combos = tds[0].get_text("|",strip=True).split("|")
                amounts = tds[1].get_text("|",strip=True).split("|")
                entries = []
                for c,a in zip(combos, amounts):
                    try: amt=int(a.replace(",",""))
                    except: continue
                    nums=[int(n) for n in re.findall(r"\d+",c)]
                    if nums: entries.append({"nums":nums,"amount":amt})
                if entries: payouts[label]=entries
        return payouts
    except: return {}

def check_hit(bet, payouts):
    tm={"tansho":"単勝","fukusho":"複勝","umaren":"馬連","wide":"ワイド","sanrenpuku":"三連複","sanrentan":"三連単"}
    label=tm.get(bet["type"]); horses=bet["horses"]
    if not label or label not in payouts: return False,0
    for e in payouts[label]:
        pn=e["nums"]; pa=e["amount"]
        if bet["type"]=="tansho" and len(horses)==1 and horses[0] in pn: return True,pa
        elif bet["type"]=="fukusho" and len(horses)==1 and horses[0]==pn[0]: return True,pa
        elif bet["type"] in ("umaren","wide","sanrenpuku") and set(horses)==set(pn): return True,pa
        elif bet["type"]=="sanrentan" and horses==pn: return True,pa
    return False,0

def main():
    init_db()
    # Only process missing dates
    target_dates = ["20260208","20260214","20260215","20260221","20260222","20260228"]

    print("="*70)
    print("  KEIBA ORACLE v7 - 残りレース (2/8-2/28)")
    print("="*70)

    all_ids = {}
    for ds in target_dates:
        time.sleep(3)
        schedules = fetch_race_list(ds)
        if schedules:
            ids = []
            for s in schedules:
                for r in s.get("races",[]):
                    rid=r.get("race_id","")
                    if rid and rid not in ids: ids.append(rid)
            if ids:
                all_ids[ds]=ids
                print(f"  {int(ds[4:6])}/{int(ds[6:8])}: {len(ids)} races")

    total = sum(len(v) for v in all_ids.values())
    print(f"\nTotal: {total} races\n")

    results = []; type_stats = {}

    for ds in sorted(all_ids.keys()):
        dl = f"{int(ds[4:6])}/{int(ds[6:8])}"
        print(f"{'─'*70}\n  {dl} ({len(all_ids[ds])} races)\n{'─'*70}")

        for rid in all_ids[ds]:
            course=CM.get(rid[4:6],"??"); rnum=int(rid[10:12])
            data=None
            for att in range(3):
                try:
                    data=fetch_race_card(rid)
                    if data: break
                    time.sleep(8)
                except:
                    time.sleep(10)
            if not data: continue

            entries=data.get("entries",[]); info=data.get("race_info",{})
            if len(entries)<3: continue
            try: preds=predictor.predict(info,entries)
            except: continue
            if len(preds)<3: continue

            od=estimate_from_entries(entries) or {}
            try:
                real=fetch_combination_odds(rid)
                if real:
                    for k,el in real.items():
                        if k in od:
                            rhs=[frozenset(e["horses"]) for e in el]
                            od[k]=el+[e for e in od[k] if frozenset(e["horses"]) not in rhs]
                        else: od[k]=el
            except: pass

            try: opt=optimize_bets(preds,od,info)
            except: continue
            if not opt: continue

            payouts=None
            for att in range(3):
                payouts=fetch_payouts(rid)
                if payouts: break
                time.sleep(8)
            if not payouts: continue

            rb=len(opt)*100; rp=0; rh=[]
            for b in opt:
                hit,amt=check_hit(b,payouts)
                bt=b["type"]
                if bt not in type_stats: type_stats[bt]={"bets":0,"hits":0,"inv":0,"ret":0,"label":b.get("typeLabel",bt)}
                type_stats[bt]["bets"]+=1; type_stats[bt]["inv"]+=100
                if hit:
                    rp+=amt; rh.append(b.get("typeLabel",bt))
                    type_stats[bt]["hits"]+=1; type_stats[bt]["ret"]+=amt

            profit=rp-rb
            probs=scores_to_probabilities(preds,info.get("headCount",16))
            pat=detect_race_pattern(probs)
            mark="+" if profit>0 else (" " if profit==0 else "-")
            hs=",".join(rh) if rh else "---"
            print(f"  {mark} {course}{rnum:2d}R [{pat:4s}] ¥{rb}→¥{rp:>6,} {profit:>+7,} ({hs})")
            results.append({"date":ds,"course":course,"rnum":rnum,"bet":rb,"payout":rp,"profit":profit,"hits":rh,"pattern":pat})

    # Summary
    n=len(results)
    if n==0: print("\nNo results."); return
    tb=sum(r["bet"] for r in results); tp=sum(r["payout"] for r in results)
    tpr=tp-tb; roi=tp/tb*100 if tb>0 else 0
    th=sum(len(r["hits"]) for r in results)
    w=sum(1 for r in results if r["profit"]>0)
    l=sum(1 for r in results if r["profit"]<0)

    print(f"\n{'='*70}")
    print(f"  2/8-2/28 検証結果")
    print(f"{'='*70}")
    print(f"  レース数: {n}")
    print(f"  投資: ¥{tb:,}  払戻: ¥{tp:,}  収支: {'+'if tpr>=0 else ''}¥{tpr:,}")
    print(f"  ROI: {roi:.1f}%  的中: {th}/{n*5} ({th/(n*5)*100:.1f}%)  勝率: {w}勝{l}敗 ({w/n*100:.1f}%)")

    print(f"\n  --- 券種別 ---")
    for bt in ["tansho","fukusho","umaren","wide","sanrenpuku","sanrentan"]:
        if bt not in type_stats: continue
        s=type_stats[bt]; sr=s["ret"]/s["inv"]*100 if s["inv"]>0 else 0
        print(f"  {s['label']:4s}: {s['bets']:3d}点 的中{s['hits']:2d} 投資¥{s['inv']:>6,} 払戻¥{s['ret']:>7,} ROI {sr:5.1f}%")

    print(f"\n  --- Top3 ---")
    for r in sorted(results,key=lambda x:-x["payout"])[:3]:
        if r["payout"]>0:
            print(f"  {int(r['date'][4:6])}/{int(r['date'][6:8])} {r['course']}{r['rnum']:2d}R: ¥{r['payout']:>7,} ({','.join(r['hits'])})")
    print("="*70)

if __name__=="__main__": main()
