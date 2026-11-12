#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01 agent v4.1 — Monte Carlo trajectory forecaster.

Reads /input/history.ndjson + task.json, writes /output/forecast.json.
Stdlib only, deterministic (fixed seed), no network, no model calls.

Model: each game has a latent strength s. Rank at any hour = order of s.
  s0 = -ln(rank) at the last read (+ TT-top20 boost, + velocity drift).
  Per hour: s += -KAPPA*(s - mean_s) + drift_i*exp(-t/36) + SIGMA*gauss()
  Age-out hazard per hour (created date when present in details, else
  first-seen lower bound): 0 below 24d, rising to 0.003/h above 30d.
  New games arrive ~P_NEW/hour at the bottom and push the weakest out.
The forecast is the MEAN over the 72 hourly reads: P(game, place) = frequency
over sims x hours. Unnamed share = 1 - sum(named) per place.
"""
import json
import math
import os
import random

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tunable parameters (v4.1: v4 + sigma 0.18, kappa 0.004) ----
PARAMS = {
    "n_sims": 1000,
    "kappa": 0.004,        # mean-reversion rate per hour
    "sigma": 0.18,         # hourly latent noise sd
    "p_new": 0.5,          # new games arriving per hour
    "w_vel": 0.0,          # weight on rank-velocity drift
    "b_tt": 0.20,          # latent boost for Top Trending top-20 games
    "unnamed_min": 0.08,   # minimum unnamed share left at each place
    "p_threshold": 0.003,  # drop probability rows below this
    "age_scale": 1.0,     # multiplier on age-out hazard
    "gone_window": 3,     # include games seen in last N reads but not last chart
    "seed": 12345,
}


def parse_dt(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history():
    reads = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chart = sorted(d["up_and_coming"], key=lambda r: r["rank"])
            tt = {}
            for r in (d.get("top_trending") or []):
                tt[r["universe_id"]] = r["rank"]
            det = d.get("details") or {}
            reads.append({
                "at": parse_dt(d["at"]),
                "chart": [(r["universe_id"], r.get("playing")) for r in chart],
                "tt": tt,
                "det": det,
            })
    task = json.load(open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8"))
    return reads, task


def build_forecast(reads, task, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    rng = random.Random(p["seed"])
    places = task.get("places", 10)
    last = reads[-1]
    last_chart = last["chart"]  # [(uid, playing)] in rank order
    n_chart = len(last_chart)
    # first seen + created age
    first_seen = {}
    created = {}
    for r in reads:
        for uid, _ in r["chart"]:
            first_seen.setdefault(uid, r["at"])
        for uid, det in (r.get("det") or {}).items():
            if det and det.get("created"):
                created[int(uid)] = parse_dt(det["created"])
    # rank velocity (log-rank per hour) from last two reads
    vel = {}
    if len(reads) >= 2:
        prev = {uid: i + 1 for i, (uid, _) in enumerate(reads[-2]["chart"])}
        dt_h = max(1.0, (last["at"] - reads[-2]["at"]).total_seconds() / 3600.0)
        for i, (uid, _) in enumerate(last_chart):
            r_now = i + 1
            r_prev = prev.get(uid)
            if r_prev:
                vel[uid] = (math.log(r_prev) - math.log(r_now)) / dt_h
    tt_top = set(uid for uid, rk in last.get("tt", {}).items() if rk <= 20)
    # initial latent strengths
    games = []  # dicts: uid, s, drift, age_days
    for i, (uid, _) in enumerate(last_chart):
        s = -math.log(i + 1)
        if uid in tt_top:
            s += p["b_tt"]
        age = None
        if uid in created:
            age = (last["at"] - created[uid]).total_seconds() / 86400.0
        else:
            age = (last["at"] - first_seen[uid]).total_seconds() / 86400.0
        games.append({"uid": uid, "s": s, "drift": p["w_vel"] * vel.get(uid, 0.0),
                      "age": age})
    # recently-gone games: chart churn is high, they often re-enter
    if p.get("gone_window", 0) and len(reads) >= 2:
        last_uids = {uid for uid, _ in last_chart}
        seen = {}
        for r in reads[-p["gone_window"]:-1]:
            for uid, _ in r["chart"]:
                seen.setdefault(uid, r["at"])
        for uid in seen:
            if uid not in last_uids:
                age = (last["at"] - first_seen[uid]).total_seconds() / 86400.0
                games.append({"uid": uid, "s": -math.log(n_chart + 10),
                              "drift": 0.0, "age": age})
    # Monte Carlo
    counts = {}  # (uid, place) -> count
    sims = p["n_sims"]
    hours = task.get("window", {}).get("slots", 72)
    for _ in range(sims):
        state = [dict(g) for g in games]
        for h in range(hours):
            if not state:
                break
            mu = sum(g["s"] for g in state) / len(state)
            decay = math.exp(-h / 36.0)
            for g in state:
                g["s"] += -p["kappa"] * (g["s"] - mu) + g["drift"] * decay + p["sigma"] * rng.gauss(0, 1)
            # age-out exits
            state = [g for g in state if rng.random() > age_hazard(g["age"], h) * p.get("age_scale", 1.0)]
            # new arrivals push the weakest out
            if rng.random() < p["p_new"] and state:
                state.sort(key=lambda g: g["s"])
                bottom = state[0]["s"]
                state.pop(0)
                state.append({"uid": -1, "s": bottom + rng.gauss(0, 0.3),
                              "drift": 0.0, "age": 0.0})
            state.sort(key=lambda g: -g["s"])
            for k in range(min(places, len(state))):
                uid = state[k]["uid"]
                if uid > 0:
                    counts[(uid, k + 1)] = counts.get((uid, k + 1), 0) + 1
    total = sims * hours
    probs = {}
    for (uid, k), c in counts.items():
        probs[(uid, k)] = c / total
    # per-place cap: leave at least unnamed_min unnamed
    rows = []
    for k in range(1, places + 1):
        named = sorted([(u, pr) for (u, kk), pr in probs.items() if kk == k],
                       key=lambda x: -x[1])
        ssum = sum(pr for _, pr in named)
        cap = 1.0 - p["unnamed_min"]
        if ssum > cap:
            scale = cap / ssum
            named = [(u, pr * scale) for u, pr in named]
        for u, pr in named:
            if pr >= p["p_threshold"]:
                rows.append({"universe_id": u, "place": k, "p": round(pr, 6)})
    # top10 by total probability of being in top 10
    tot10 = {}
    for (u, k), pr in probs.items():
        tot10[u] = tot10.get(u, 0.0) + pr
    top10 = [u for u, _ in sorted(tot10.items(), key=lambda x: -x[1])[:places]]
    return {"probabilities": rows, "top10": top10}


def age_hazard(age_days, h):
    if age_days is None or age_days < 24:
        return 0.0
    if age_days >= 30:
        return 0.003
    return 0.0005 * (age_days - 24)


def main():
    reads, task = load_history()
    fc = build_forecast(reads, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(fc, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
