"""S8 strategy optimization with 3-way split validation.

Usage:
    PYTHONUNBUFFERED=1 /usr/bin/python3 -m backtest.optimize_s8
"""
from __future__ import annotations
import pickle, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predictor.scoring import WeightedScoringModel
from backend.predictor.bet_optimizer import (
    scores_to_probabilities, monte_carlo_finish, estimate_hit_probabilities,
    generate_candidates, find_odds_for_bet, detect_race_pattern,
    MC_SAMPLES, HITPROB_DEFLATION,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
TYPE_JP = {'tansho':['単勝'],'fukusho':['複勝'],'umaren':['馬連'],'umatan':['馬単'],
           'wide':['ワイド'],'sanrenpuku':['三連複','3連複'],'sanrentan':['三連単','3連単']}

def load_data():
    all_races = []
    for f in ['hist_2024.pkl','hist_2025.pkl','cached_april_races.pkl','cached_516_517_v2.pkl']:
        path = os.path.join(CACHE_DIR, f)
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                all_races.extend(pickle.load(fh))
    all_races.sort(key=lambda r: r.get('date',''))
    return all_races

def check_hit(bt, horses, positions, payouts):
    hit = False
    if bt=='tansho': hit=positions.get(horses[0],99)==1
    elif bt=='fukusho': hit=positions.get(horses[0],99)<=3
    elif bt=='umaren': hit=all(positions.get(h,99)<=2 for h in horses)
    elif bt=='wide': hit=all(positions.get(h,99)<=3 for h in horses)
    elif bt=='sanrenpuku': hit=all(positions.get(h,99)<=3 for h in horses)
    elif bt=='sanrentan': hit=len(horses)==3 and [positions.get(h,99) for h in horses]==[1,2,3]
    elif bt=='umatan': hit=len(horses)==2 and positions.get(horses[0],99)==1 and positions.get(horses[1],99)==2
    payout=0
    if hit:
        for jp in TYPE_JP.get(bt,[]):
            for e in payouts.get(jp,[]):
                if bt in ('umatan','sanrentan'):
                    if e['nums']==horses: payout=e['amount']
                elif bt in ('tansho','fukusho'):
                    if horses[0] in e['nums']: payout=e['amount']
                else:
                    if set(e['nums'])==set(horses): payout=e['amount']
                if payout: break
            if payout: break
    return hit, payout

def simulate(races, model, config):
    odds_ranges = config.get('odds_ranges', {'umatan':(20,300),'umaren':(20,100),'wide':(10,50)})
    min_gap = config.get('min_gap', 0)
    exclude_patterns = config.get('exclude_patterns', set())
    max_per_type = config.get('max_per_type', 2)
    max_total = config.get('max_total', 5)
    # Use reduced MC for speed
    mc = config.get('mc_samples', 500)

    total_inv=0; total_ret=0
    for rd in races:
        preds = model.predict(rd['info'], rd['entries'])
        if len(preds) < 3: continue
        sorted_p = sorted(preds, key=lambda p:-p['score'])
        gap = sorted_p[0]['score'] - sorted_p[1]['score'] if len(sorted_p)>=2 else 0
        if gap < min_gap: continue
        head_count = rd['info'].get('headCount',16)
        probs = scores_to_probabilities(preds, head_count)
        if len(probs) < 3: continue
        pattern = detect_race_pattern(probs)
        if pattern in exclude_patterns: continue
        temp = {'本命堅軸':0.85,'混戦模様':1.15,'2強対決':0.92}.get(pattern,1.0)
        if temp != 1.0: probs = scores_to_probabilities(preds, head_count, temp_adjust=temp)
        cands = generate_candidates(probs, top_n=min(7,len(probs)), entries=rd['entries'])
        rng = random.Random(42)
        fins = monte_carlo_finish(probs, mc, rng=rng)
        cands = estimate_hit_probabilities(fins, cands)
        ai_top5 = set(p['horseNumber'] for p in sorted_p[:5])
        viable = []
        for c in cands:
            bt = c['type']
            if bt not in odds_ranges: continue
            lo, hi = odds_ranges[bt]
            oi = find_odds_for_bet(c, rd['odds_data'])
            if not oi: continue
            if oi['odds'] < lo or oi['odds'] > hi: continue
            if not any(h in ai_top5 for h in c['horses']): continue
            c['_odds'] = oi['odds']
            viable.append(c)
        viable.sort(key=lambda x: -x['_odds'])
        type_counts = {}
        for c in viable:
            bt = c['type']
            if type_counts.get(bt,0) >= max_per_type: continue
            total_inv += 500
            hit, payout = check_hit(bt, c['horses'], rd['positions'], rd['payouts'])
            if hit: total_ret += payout * 5
            type_counts[bt] = type_counts.get(bt,0) + 1
            if sum(type_counts.values()) >= max_total: break

    return total_ret/total_inv*100 if total_inv > 0 else 0

def main():
    print("Loading data...", flush=True)
    all_races = load_data()
    n = len(all_races)
    tune = all_races[:int(n*0.4)]
    validate = all_races[int(n*0.4):int(n*0.7)]
    holdout = all_races[int(n*0.7):]
    print(f"Total: {n}R | Tune: {len(tune)}R | Validate: {len(validate)}R | Holdout: {len(holdout)}R", flush=True)

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'optimized_weights.json')) as f:
        wd = json.load(f)
    model = WeightedScoringModel(analytical_weights=wd['analytical_weights'], market_weight=wd['market_weight'])

    configs = {
        'S8 current': {'odds_ranges': {'umatan':(20,300),'umaren':(20,100),'wide':(10,50)}},
        'A1: 50x+ only': {'odds_ranges': {'umatan':(50,300),'umaren':(50,100),'wide':(50,100)}},
        'A2: umatan50+ umaren30+ wide20+': {'odds_ranges': {'umatan':(50,300),'umaren':(30,100),'wide':(20,50)}},
        'A3: umatan50+ umaren50+ no wide': {'odds_ranges': {'umatan':(50,300),'umaren':(50,300)}},
        'B1: S8 + gap>=5': {'odds_ranges': {'umatan':(20,300),'umaren':(20,100),'wide':(10,50)}, 'min_gap': 5},
        'B2: S8 + gap>=8': {'odds_ranges': {'umatan':(20,300),'umaren':(20,100),'wide':(10,50)}, 'min_gap': 8},
        'B3: S8 + gap>=10': {'odds_ranges': {'umatan':(20,300),'umaren':(20,100),'wide':(10,50)}, 'min_gap': 10},
        'C1: 50x+ gap>=5': {'odds_ranges': {'umatan':(50,300),'umaren':(50,300),'wide':(50,100)}, 'min_gap': 5},
        'C2: 50x+ gap>=8': {'odds_ranges': {'umatan':(50,300),'umaren':(50,300),'wide':(50,100)}, 'min_gap': 8},
        'C3: umatan50+ umaren30+ gap>=5': {'odds_ranges': {'umatan':(50,300),'umaren':(30,100),'wide':(20,50)}, 'min_gap': 5},
        'C4: umatan50+ umaren30+ gap>=8': {'odds_ranges': {'umatan':(50,300),'umaren':(30,100),'wide':(20,50)}, 'min_gap': 8},
        'D1: 10-30x + 50-300x': {'odds_ranges': {'umatan':(50,300),'umaren':(20,30),'wide':(10,30)}},
        'D2: D1 + gap>=5': {'odds_ranges': {'umatan':(50,300),'umaren':(20,30),'wide':(10,30)}, 'min_gap': 5},
        'E1: exclude 標準配置': {'odds_ranges': {'umatan':(20,300),'umaren':(20,100),'wide':(10,50)}, 'exclude_patterns': {'標準配置'}},
        'E2: excl標準 + gap>=5': {'odds_ranges': {'umatan':(20,300),'umaren':(20,100),'wide':(10,50)}, 'exclude_patterns': {'標準配置'}, 'min_gap': 5},
        'F1: umatan only 50-300x': {'odds_ranges': {'umatan':(50,300)}},
        'F2: umatan only 100-300x': {'odds_ranges': {'umatan':(100,300)}},
        'F3: umatan50+umaren50+ gap>=5 max3': {'odds_ranges': {'umatan':(50,300),'umaren':(50,300)}, 'min_gap': 5, 'max_total': 3},
        'G1: excl標準 + gap>=3': {'odds_ranges': {'umatan':(20,300),'umaren':(20,100),'wide':(10,50)}, 'exclude_patterns': {'標準配置'}, 'min_gap': 3},
        'G2: umatan30+ wide10-30 gap>=3': {'odds_ranges': {'umatan':(30,300),'umaren':(20,100),'wide':(10,30)}, 'min_gap': 3},
        'G3: umatan30+ umaren20-50 wide10-30 gap>=5': {'odds_ranges': {'umatan':(30,300),'umaren':(20,50),'wide':(10,30)}, 'min_gap': 5},
    }

    print(flush=True)
    print("=" * 95, flush=True)
    print("  Strategy Grid (MC=500 for speed, 3-way split)", flush=True)
    print("=" * 95, flush=True)
    print("%-45s %7s %7s %7s %5s %5s" % ('Strategy','Tune','Valid','Hold','Min','Avg'), flush=True)
    print("-" * 95, flush=True)

    results = []
    for i, (name, cfg) in enumerate(configs.items()):
        cfg['mc_samples'] = 500
        t = simulate(tune, model, cfg)
        v = simulate(validate, model, cfg)
        h = simulate(holdout, model, cfg)
        mn = min(t,v,h); avg = (t+v+h)/3
        results.append((name,t,v,h,mn,avg))
        print("%-45s %6.1f%% %6.1f%% %6.1f%% %4.1f %4.1f" % (name,t,v,h,mn,avg), flush=True)

    print(flush=True)
    print("=" * 95, flush=True)
    print("  Top 5 by Min (most stable)", flush=True)
    print("=" * 95, flush=True)
    for name,t,v,h,mn,avg in sorted(results, key=lambda x:-x[4])[:5]:
        print("  %-45s T:%5.1f%% V:%5.1f%% H:%5.1f%% Min:%5.1f Avg:%5.1f" % (name,t,v,h,mn,avg), flush=True)

    print(flush=True)
    print("  Top 5 by Avg ROI", flush=True)
    for name,t,v,h,mn,avg in sorted(results, key=lambda x:-x[5])[:5]:
        print("  %-45s T:%5.1f%% V:%5.1f%% H:%5.1f%% Min:%5.1f Avg:%5.1f" % (name,t,v,h,mn,avg), flush=True)

if __name__ == "__main__":
    main()
