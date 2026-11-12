#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01: forecast Roblox's Up-and-Coming chart.

Reads /input/history.ndjson (chart reads, oldest first) and /input/task.json,
writes /output/forecast.json: {"probabilities": [...], "top10": [...]}.

Method: rank-diffusion Monte Carlo. For each game on the last chart, simulate
its rank over the scored window as a mean-reverting diffusion:
  r(t+1) = r(t) + v(t) + d0 + pull * (r_eq(t) - r(t)) + sigma * N(0,1)
  - v(t): the game's recent rank velocity (shrunk, decaying with time
    constant tau_v) -- risers keep rising for a while, then fade.
  - d0: global downward drift (games fall in rank over time as new games
    enter above them and the chart churns).
  - pull toward r_eq, the game's recent mean rank, which itself drifts down
    slowly (eq_drift).
Averaging place occupancy over the window x many seeded paths gives
P(game at place k). Each place's probabilities are scaled to sum PLACE_SUM,
leaving 1 - PLACE_SUM unnamed: the unnamed share is spread by the scorer over
every game on a read's chart that was not named at that place, which covers
new entrants and deep-chart games the simulation did not name.

Deterministic: fixed RNG seed, no network, stdlib only. Runs twice on the
same history and writes byte-identical output.
"""

import json
import math
import os
import random
import statistics
from datetime import datetime, timedelta

# --- tuned parameters (practice log score -31.5 vs house baseline -43.5) ---
PARAMS = {
    "n_paths": 1500,      # Monte-Carlo paths per game
    "seed": 12345,        # fixed seed: deterministic output
    "horizon": 72,        # hours to simulate (scored window length)
    "vel_hours": 72,      # window for recent rank velocity
    "vel_shrink": 0.5,    # shrink velocity toward 0 (it is noisy)
    "tau_v": 24.0,        # velocity decay time constant (hours)
    "drift_down": 0.2,    # global downward drift, ranks/hour
    "sigma": 0.5,         # hourly rank noise
    "pull": 0.01,         # mean reversion per hour toward r_eq
    "eq_drift": 0.05,     # r_eq rises (falls in rank) per hour
    "chart_n": 100,       # simulated chart size (rank > this = off-chart)
    "place_sum": 0.75,    # each place's named probabilities sum to this
    "min_p": 0.002,       # drop rows below this probability
}


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recent_velocity(hist, t_end, hours):
    """Per-game rank velocity (ranks/hour, negative = rising) over the last
    `hours` hours of history."""
    series = {}
    for rec in hist:
        t = parse_time(rec["at"])
        for g in rec["up_and_coming"]:
            series.setdefault(g["universe_id"], []).append((t, g["rank"]))
    out = {}
    for uid, s in series.items():
        s = [(t, rk) for t, rk in s
             if t <= t_end and t >= t_end - timedelta(hours=hours)]
        if len(s) >= 2:
            dt = (s[-1][0] - s[0][0]).total_seconds() / 3600.0
            if dt > 0:
                out[uid] = (s[-1][1] - s[0][1]) / dt
    return out


def recent_mean_rank(hist, uid, t_end, hours=72):
    rs = []
    for rec in hist:
        t = parse_time(rec["at"])
        if t > t_end or t < t_end - timedelta(hours=hours):
            continue
        for g in rec["up_and_coming"]:
            if g["universe_id"] == uid:
                rs.append(g["rank"])
    return statistics.mean(rs) if rs else None


def build_forecast(hist, task, p):
    t_end = parse_time(hist[-1]["at"])
    last = sorted(hist[-1]["up_and_coming"], key=lambda g: g["rank"])
    chart = [(g["universe_id"], g["rank"]) for g in last]
    vel = recent_velocity(hist, t_end, hours=p["vel_hours"])
    rng = random.Random(p["seed"])
    n_paths = p["n_paths"]
    horizon = p["horizon"]
    places = task.get("places", 10)

    counts = {}
    for uid, r0 in chart:
        v0 = p["vel_shrink"] * vel.get(uid, 0.0)
        r_eq0 = recent_mean_rank(hist, uid, t_end) or float(r0)
        for _ in range(n_paths):
            r = float(r0)
            v = v0
            r_eq = r_eq0
            for _h in range(1, horizon + 1):
                v *= math.exp(-1.0 / p["tau_v"])
                r_eq += p["eq_drift"]
                r += v + p["drift_down"] + p["pull"] * (r_eq - r) \
                    + p["sigma"] * rng.gauss(0, 1)
                if r < 1.0:
                    r = 1.0 + abs(r - 1.0) * 0.3
                if r > p["chart_n"]:
                    break
                k = int(round(r))
                if 1 <= k <= places:
                    counts[(uid, k)] = counts.get((uid, k), 0) + 1

    total = n_paths * horizon
    place_sum = {}
    rows = []
    for (uid, k), c in counts.items():
        pr = c / total
        rows.append({"universe_id": uid, "place": k, "p": pr})
        place_sum[k] = place_sum.get(k, 0.0) + pr

    out = []
    for row in rows:
        s = place_sum[row["place"]]
        pr = row["p"] / s * p["place_sum"] if s > 0 else 0.0
        if pr >= p["min_p"]:
            out.append({"universe_id": row["universe_id"],
                        "place": row["place"], "p": round(pr, 6)})

    tot = {}
    for row in out:
        tot[row["universe_id"]] = tot.get(row["universe_id"], 0.0) + row["p"]
    last_rank = dict(chart)
    top10 = sorted(tot, key=lambda u: (-tot[u], last_rank.get(u, 999)))[:places]
    return {"probabilities": out, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    hist = load_history(os.path.join(input_dir, "history.ndjson"))
    forecast = build_forecast(hist, task, PARAMS)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
