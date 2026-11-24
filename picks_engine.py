"""
CBB pick engine - the model's memory and its daily picks.

Every time it runs (after the records code, before the site cell) it:
  1. GRADES  - fills in final scores and closing lines for games it logged before.
  2. LEARNS  - re-fits the model from its own graded history:
                 * home-court points, and how much to trust each input:
                     - the workbook's adjusted ratings (lean on preseason early)
                     - its own schedule-adjusted ratings built from every game
                       result (credits opponent strength and venue, caps blowouts)
                     - the schedule-strength gap between the two teams
                 * a scoring adjustment for totals
                 * how much of the model's disagreement with the betting line
                   actually came true (the key number - if it's near 0, the
                   market already knows what the model knows, and it stops picking)
  3. LOGS    - records the projection and line for every unplayed game from today
               through LOOKAHEAD_DAYS ahead, using the ratings as they are BEFORE
               the game is played (no peeking). Unplayed games are re-logged on
               every run, so an early look at tomorrow gets refreshed tomorrow.
  4. PICKS   - picks only games where the learned edge clears a confidence bar
               (the per-day caps apply to each day separately).
  5. WRITES  - adds Picks, Pick Log and Model tabs to the workbook (the site shows them).

Files it keeps in this folder - don't delete them, they ARE the learning:
    games_log.csv       every game it has seen: projection, line, result
    model_params.json   the latest learned numbers

Use in JupyterLab: put this in its own notebook (picks_engine.ipynb) in
PythonFun, and in your records notebook add a cell between the records code
and the site cell:
    %run picks_engine.ipynb
"""
import json
import math
import os
import shutil
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    FOLDER = Path(__file__).resolve().parent
except NameError:                      # pasted into a notebook cell
    FOLDER = Path.cwd()

WORKBOOK = FOLDER / "college_basketball_by_conference.xlsx"
LOG_FILE = FOLDER / "games_log.csv"
PARAMS_FILE = FOLDER / "model_params.json"

# ----------------------------------------------------------------------
# Pick rules (edit to taste)
# ----------------------------------------------------------------------
STRONG = 0.570          # model chance to win the bet for a "Strong" pick
LEAN = 0.545            # ... for a "Lean" pick (52.4% breaks even at -110)
MAX_SPREAD_PICKS = 5    # per day, best first
MAX_TOTAL_PICKS = 3
PAYOUT = 100 / 110      # units won per unit risked at -110
# Also pick games this many days ahead (0 = today only, 1 = today + tomorrow).
# A game only gets a pick once a line is posted, so tomorrow's slate can be thin
# if the update runs early in the morning.
LOOKAHEAD_DAYS = 1

# Starting beliefs, used until the log has enough graded games to overrule them.
# PRIOR_GAMES = how many graded games each belief is "worth".
PRIORS = {
    "rating_weight": 0.90,  # weight on the workbook's adjusted ratings
    "results_weight": 0.10, # weight on the engine's own schedule-adjusted results ratings
    "sos_weight": 0.0,      # points per 1.0 of schedule-strength gap (+ = tough schedules underrated)
    "hca": 3.0,             # home-court points
    "total_bias": 0.0,      # points added to every projected total
    "market_a": 0.0,        # learned home/away lean vs the spread
    "market_b": 0.25,       # share of the model-vs-line gap that comes true
    "total_a": 0.0,
    "total_c": 0.15,        # same, for totals (ratings models know less about totals)
    "spread_xbar": 0.0, "total_xbar": 0.0,   # average model-vs-line gap (fits are centered on it)
    # How the market's numbers line up with the model's: market ~ alpha + beta * model.
    # Re-measured every run, so only disagreements in WHO is better count as edges,
    # not the model's numbers being too compressed or too spread out overall.
    "spread_alpha": 0.0, "spread_beta": 1.0, "total_alpha": 0.0, "total_beta": 1.0,
    "sd_margin": 11.0, "sd_vs_spread": 11.0, "sd_vs_total": 15.0,
}
PRIOR_GAMES = {"margin": 150, "total": 40, "market": 400}
# Results ratings: ridge regression on game margins (tested out of sample on
# Nov 8-18 games: these settings predicted best). Blowouts capped at 30.
RESULTS_LAMBDA, RESULTS_CAP = 2.0, 30.0
# A gap bigger than this is usually news the ratings don't know (injury, suspension,
# rest) rather than an edge. In learning it counts no more than the cap. Such games
# can still be picks, but they're listed separately as FLAGGED and graded separately.
GAP_CAP = {"spread": 10.0, "total": 15.0}
MAX_FLAGGED = {"spread": 3, "total": 2}      # flagged picks shown per day
# "Strong" picks only once this share of the model is learned from results.
STRONG_NEEDS_LEARNED = 0.30


def today_et():
    forced = os.environ.get("CBB_TODAY")       # for testing
    if forced:
        return date.fromisoformat(forced)
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return date.today()


def slate_days(today):
    """Today plus the look-ahead days, in order."""
    return [today + timedelta(days=i) for i in range(max(0, int(LOOKAHEAD_DAYS)) + 1)]


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ----------------------------------------------------------------------
# Reading the workbook
# ----------------------------------------------------------------------
def to_minutes(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, dtime):
        return v.hour * 60 + v.minute
    if isinstance(v, datetime):
        return v.hour * 60 + v.minute
    if isinstance(v, (int, float)):
        return int(round((v % 1) * 1440))
    return None


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def load_workbook_data():
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"No workbook at {WORKBOOK} - run the records code first.")
    teams = pd.read_excel(WORKBOOK, sheet_name="All Teams")
    rows = pd.read_excel(WORKBOOK, sheet_name="Games")
    teams = teams[teams["Team"].notna()]
    by_team = {r["Team"]: r for r in teams.to_dict("records")}

    games = {}
    for r in rows.to_dict("records"):
        gid = clean(r.get("GameID"))
        if gid is None or clean(r.get("Team")) is None:
            continue
        gid = str(int(gid)) if isinstance(gid, float) else str(gid)
        away = r.get("Site") == "Away"
        flip = (lambda v: None if clean(v) is None else (-v if away else v))
        d = r.get("Date")
        d = pd.to_datetime(d).date() if clean(d) is not None else None
        g = {
            "game_id": gid, "date": d.isoformat() if d else None, "minutes": to_minutes(r.get("Time (ET)")),
            "home": r["Opponent"] if away else r["Team"], "away": r["Team"] if away else r["Opponent"],
            "neutral": r.get("Site") == "Neutral", "conf_game": r.get("Conf Game") == "Yes",
            "spread": flip(r.get("Spread")), "open_spread": flip(r.get("Open Spread")),
            "total_line": clean(r.get("O/U")), "open_total": clean(r.get("Open O/U")),
            "home_ml": clean(r.get("Opp Moneyline") if away else r.get("Moneyline")),
            "away_ml": clean(r.get("Moneyline") if away else r.get("Opp Moneyline")),
            "final": clean(r.get("Result")) is not None,
            "home_pts": clean(r.get("Opp Pts") if away else r.get("Pts")),
            "away_pts": clean(r.get("Pts") if away else r.get("Opp Pts")),
            "from_home": r.get("Site") == "Home",
        }
        prev = games.get(gid)
        if prev is None or (not prev["from_home"] and g["from_home"]):
            games[gid] = g
    return teams, by_team, games, rows


# ----------------------------------------------------------------------
# Schedule-adjusted results ratings (no extra API calls - uses the Games tab)
# ----------------------------------------------------------------------
def results_ratings(games, d1, before):
    """
    margin = rating_home - rating_away + home_court, fit by ridge regression on
    every final game before `before`. Each team is credited for who it played and
    where. Non-D1 opponents share one rating. Returns ({team: rating}, home_court).
    """
    played = [g for g in games.values() if g["final"] and g["home_pts"] is not None
              and g["date"] and g["date"] < before.isoformat()]
    if len(played) < 50:
        return {}, PRIORS["hca"]
    name = lambda t: t if t in d1 else "non-D1"
    teams = sorted({name(g["home"]) for g in played} | {name(g["away"]) for g in played})
    ix = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    X = np.zeros((len(played), k + 1))
    y = np.zeros(len(played))
    for j, g in enumerate(played):
        X[j, ix[name(g["home"])]] += 1
        X[j, ix[name(g["away"])]] -= 1
        X[j, k] = 0 if g["neutral"] else 1
        y[j] = max(-RESULTS_CAP, min(RESULTS_CAP, float(g["home_pts"]) - float(g["away_pts"])))
    pen = np.eye(k + 1) * RESULTS_LAMBDA
    pen[k, k] = 0.01
    beta = np.linalg.solve(X.T @ X + pen, X.T @ y)
    return {t: float(beta[i]) for t, i in ix.items()}, float(beta[k])


def schedule_strength(rows):
    """Average opponent rating over each team's played games (from the Games tab)."""
    played = rows[rows["Result"].notna() & rows["Opp Rating"].notna()]
    return played.groupby("Team")["Opp Rating"].mean().to_dict()


# ----------------------------------------------------------------------
# Projection (same math as the site)
# ----------------------------------------------------------------------
def league(teams):
    return {"o": teams["Adj O"].mean(), "d": teams["Adj D"].mean(), "pace": teams["Pace"].mean()}


def raw_projection(h, a, lg):
    """Neutral-floor projection with ratings at face value. None if ratings are missing."""
    vals = [h.get("Adj O"), h.get("Adj D"), a.get("Adj O"), a.get("Adj D")]
    if any(clean(v) is None for v in vals):
        return None
    hp_pace = clean(h.get("Pace")) or lg["pace"]
    ap_pace = clean(a.get("Pace")) or lg["pace"]
    poss = hp_pace * ap_pace / lg["pace"]
    hp = h["Adj O"] * a["Adj D"] / lg["d"] * poss / 100
    ap = a["Adj O"] * h["Adj D"] / lg["d"] * poss / 100
    return hp - ap, hp + ap, poss


def model_numbers(raw_margin, raw_total, neutral, p, results_margin=None, sos_diff=None):
    rm = raw_margin if results_margin is None else results_margin
    margin = (p["rating_weight"] * raw_margin + p["results_weight"] * rm
              + p["sos_weight"] * (sos_diff or 0.0) + (0 if neutral else p["hca"]))
    total = raw_total + p["total_bias"]
    return margin, total


def row_inputs(r):
    """The model inputs saved for a logged game (older rows lack the newer ones)."""
    return num(r["raw_margin"]), num(r["raw_total"]), num(r.get("results_margin")), num(r.get("sos_diff"))


# ----------------------------------------------------------------------
# Log
# ----------------------------------------------------------------------
LOG_COLS = [
    "game_id", "date", "minutes", "away", "home", "neutral", "conf_game", "logged_at",
    "home_adj_o", "home_adj_d", "home_pace", "away_adj_o", "away_adj_d", "away_pace",
    "raw_margin", "raw_total", "poss", "results_margin", "sos_diff", "model_margin", "model_total",
    "line_spread", "open_spread", "line_total", "open_total", "home_ml", "away_ml",
    "spread_pick", "spread_edge", "spread_prob", "spread_tier", "spread_flag", "spread_note",
    "total_pick", "total_edge", "total_prob", "total_tier", "total_flag", "total_note",
    "final", "home_pts", "away_pts", "close_spread", "close_total",
    "spread_result", "total_result", "spread_units", "total_units", "spread_clv", "total_clv",
]


def load_log():
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE, dtype={"game_id": str})
        for c in LOG_COLS:
            if c not in df.columns:
                df[c] = None
        return df[LOG_COLS].astype(object)
    return pd.DataFrame(columns=LOG_COLS).astype(object)


def save_log(df):
    if LOG_FILE.exists():
        shutil.copy2(LOG_FILE, LOG_FILE.with_suffix(".csv.bak"))
    tmp = LOG_FILE.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, LOG_FILE)


def is_true(v):
    return v is True or v == 1 or str(v).lower() in ("true", "1", "1.0")


def num(v):
    try:
        v = float(v)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# 1. Grade
# ----------------------------------------------------------------------
def bet_result(x):
    return "P" if abs(x) < 1e-9 else ("W" if x > 0 else "L")


def units(res):
    return {"W": PAYOUT, "L": -1.0, "P": 0.0}.get(res)


def grade(log, games):
    n = 0
    for i, row in log.iterrows():
        if is_true(row["final"]):
            continue
        g = games.get(str(row["game_id"]))
        if not g or not g["final"] or g["home_pts"] is None:
            continue
        hp, ap = float(g["home_pts"]), float(g["away_pts"])
        margin, total = hp - ap, hp + ap
        log.at[i, "final"] = True
        log.at[i, "home_pts"], log.at[i, "away_pts"] = hp, ap
        log.at[i, "close_spread"] = g["spread"]
        log.at[i, "close_total"] = g["total_line"]
        line, tl = num(row["line_spread"]), num(row["line_total"])
        if isinstance(row["spread_pick"], str) and row["spread_pick"] in ("home", "away") and line is not None:
            side = 1 if row["spread_pick"] == "home" else -1
            res = bet_result(side * (margin + line))
            log.at[i, "spread_result"], log.at[i, "spread_units"] = res, units(res)
            if g["spread"] is not None:
                log.at[i, "spread_clv"] = side * (line - g["spread"])
        if isinstance(row["total_pick"], str) and row["total_pick"] in ("over", "under") and tl is not None:
            side = 1 if row["total_pick"] == "over" else -1
            res = bet_result(side * (total - tl))
            log.at[i, "total_result"], log.at[i, "total_units"] = res, units(res)
            if g["total_line"] is not None:
                log.at[i, "total_clv"] = side * (g["total_line"] - tl)
        n += 1
    return n


# ----------------------------------------------------------------------
# 2. Learn
# ----------------------------------------------------------------------
def ridge2(x1, x2, y, prior, strength):
    """Least squares for y ~ b1*x1 + b2*x2, pulled toward prior by `strength` pseudo-games each."""
    x1, x2, y = map(np.asarray, (x1, x2, y))
    l1 = strength * max(np.mean(x1 ** 2), 1e-6) if len(x1) else strength
    l2 = strength * max(np.mean(x2 ** 2), 1e-6) if len(x2) else strength
    s11, s22, s12 = x1 @ x1 + l1, x2 @ x2 + l2, x1 @ x2
    t1, t2 = x1 @ y + l1 * prior[0], x2 @ y + l2 * prior[1]
    det = s11 * s22 - s12 * s12
    return (t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det


def ridge_n(X, y, prior, strength):
    """Least squares for y ~ X @ b, each coefficient pulled toward its prior by `strength` pseudo-games."""
    X, y, prior = np.asarray(X, float), np.asarray(y, float), np.asarray(prior, float)
    lam = strength * np.maximum(np.mean(X ** 2, axis=0), 1e-6)
    A = X.T @ X + np.diag(lam)
    return np.linalg.solve(A, X.T @ y + lam * prior)


def shrunk_sd(resid, prior, strength):
    n = len(resid)
    if n == 0:
        return prior
    return math.sqrt((np.sum(np.asarray(resid) ** 2) + strength * prior ** 2) / (n + strength))


def market_fit(model_vals, market_vals, prior_beta=1.0, min_n=20):
    """market ~ alpha + beta * model by least squares (needs min_n games with lines)."""
    m, k = np.asarray(model_vals, float), np.asarray(market_vals, float)
    ok = ~(np.isnan(m) | np.isnan(k))
    m, k = m[ok], k[ok]
    if len(m) < min_n or np.var(m) < 1e-6:
        return 0.0, prior_beta
    beta = float(np.cov(m, k, bias=True)[0, 1] / np.var(m))
    beta = float(np.clip(beta, 0.5, 3.0))
    return float(np.mean(k) - beta * np.mean(m)), beta


def aligned_gap(model_val, market_val, alpha, beta):
    """Model vs line after putting the model on the market's scale (+ = model likes home / over)."""
    return (alpha + beta * model_val) - market_val


def learn(log):
    p = dict(PRIORS)
    g = log[log["final"].map(is_true).astype(bool)].copy()
    for c in ["raw_margin", "raw_total", "home_pts", "away_pts", "line_spread", "line_total"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g[g["raw_margin"].notna() & g["home_pts"].notna()]
    stats = {"graded_games": int(len(g))}
    if len(g):
        actual_m = (g["home_pts"] - g["away_pts"]).to_numpy()
        actual_t = (g["home_pts"] + g["away_pts"]).to_numpy()
        home = (~g["neutral"].map(is_true).astype(bool)).astype(float).to_numpy()
        # margins: actual ~ a*adjusted ratings + b*results ratings + c*schedule gap + hca*home
        raw = g["raw_margin"].to_numpy(float)
        res = pd.to_numeric(g["results_margin"], errors="coerce").to_numpy(float)
        res = np.where(np.isnan(res), raw, res)          # older rows: assume the same as the ratings
        sos = pd.to_numeric(g["sos_diff"], errors="coerce").fillna(0).to_numpy(float)
        coef = ridge_n(np.c_[raw, res, sos, home], actual_m,
                       [PRIORS["rating_weight"], PRIORS["results_weight"], PRIORS["sos_weight"], PRIORS["hca"]],
                       PRIOR_GAMES["margin"])
        p["rating_weight"] = float(np.clip(coef[0], 0.0, 1.3))
        p["results_weight"] = float(np.clip(coef[1], 0.0, 1.0))
        p["sos_weight"] = float(np.clip(coef[2], -1.0, 1.0))
        p["hca"] = float(np.clip(coef[3], 0.0, 6.0))
        model_m = p["rating_weight"] * raw + p["results_weight"] * res + p["sos_weight"] * sos + p["hca"] * home
        p["sd_margin"] = shrunk_sd(actual_m - model_m, PRIORS["sd_margin"], 50)
        # totals: bias = how far totals land from the raw projection on average (shrunk)
        diff = actual_t - g["raw_total"].to_numpy()
        n0 = PRIOR_GAMES["total"]
        p["total_bias"] = float(np.clip(np.sum(diff) / (len(diff) + n0), -12, 12))
        model_t = g["raw_total"].to_numpy() + p["total_bias"]

        # line the model up with the market's scale, using every logged game with a line
        allr = log.copy()
        a_raw = pd.to_numeric(allr["raw_margin"], errors="coerce").to_numpy(float)
        a_res = pd.to_numeric(allr["results_margin"], errors="coerce").to_numpy(float)
        a_res = np.where(np.isnan(a_res), a_raw, a_res)
        a_sos = pd.to_numeric(allr["sos_diff"], errors="coerce").fillna(0).to_numpy(float)
        a_home = (~allr["neutral"].map(is_true).astype(bool)).astype(float).to_numpy()
        a_mm = p["rating_weight"] * a_raw + p["results_weight"] * a_res + p["sos_weight"] * a_sos + p["hca"] * a_home
        a_mt = pd.to_numeric(allr["raw_total"], errors="coerce").to_numpy(float) + p["total_bias"]
        p["spread_alpha"], p["spread_beta"] = market_fit(a_mm, -pd.to_numeric(allr["line_spread"], errors="coerce").to_numpy(float))
        p["total_alpha"], p["total_beta"] = market_fit(a_mt, pd.to_numeric(allr["line_total"], errors="coerce").to_numpy(float))

        # vs the market: how much of the (aligned) model-vs-line gap comes true?
        ls = g["line_spread"].to_numpy()
        ok = ~np.isnan(ls)
        stats["games_with_spread"] = int(ok.sum())
        if ok.any():
            mkt = -ls[ok]                                  # market's expected home margin
            x = aligned_gap(model_m[ok], mkt, p["spread_alpha"], p["spread_beta"])
            y = actual_m[ok] - mkt
            p["spread_xbar"] = float(np.mean(x))
            xc = np.clip(x - p["spread_xbar"], -GAP_CAP["spread"], GAP_CAP["spread"])  # centered + capped
            a, b = ridge2(np.ones(ok.sum()), xc, y, (PRIORS["market_a"], PRIORS["market_b"]), PRIOR_GAMES["market"])
            p["market_a"], p["market_b"] = float(np.clip(a, -2, 2)), float(np.clip(b, 0.0, 1.0))
            p["sd_vs_spread"] = shrunk_sd(y - (p["market_a"] + p["market_b"] * xc), PRIORS["sd_vs_spread"], 50)
            stats["model_mae_vs_result"] = float(np.mean(np.abs(actual_m[ok] - model_m[ok])))
            stats["market_mae_vs_result"] = float(np.mean(np.abs(actual_m[ok] - mkt)))
        lt = g["line_total"].to_numpy()
        ok = ~np.isnan(lt)
        stats["games_with_total"] = int(ok.sum())
        if ok.any():
            x = aligned_gap(model_t[ok], lt[ok], p["total_alpha"], p["total_beta"])
            y = actual_t[ok] - lt[ok]
            p["total_xbar"] = float(np.mean(x))
            xc = np.clip(x - p["total_xbar"], -GAP_CAP["total"], GAP_CAP["total"])
            a, c = ridge2(np.ones(ok.sum()), xc, y, (PRIORS["total_a"], PRIORS["total_c"]), PRIOR_GAMES["market"])
            p["total_a"], p["total_c"] = float(np.clip(a, -4, 4)), float(np.clip(c, 0.0, 1.0))
            p["sd_vs_total"] = shrunk_sd(y - (p["total_a"] + p["total_c"] * xc), PRIORS["sd_vs_total"], 50)
            stats["model_total_mae"] = float(np.mean(np.abs(actual_t[ok] - model_t[ok])))
            stats["market_total_mae"] = float(np.mean(np.abs(actual_t[ok] - lt[ok])))
    trust = stats["graded_games"] / (stats["graded_games"] + PRIOR_GAMES["market"])
    stats["learned_share"] = trust
    return p, stats


# ----------------------------------------------------------------------
# 3 + 4. Log the slate and pick
# ----------------------------------------------------------------------
def tier(prob):
    return "Strong" if prob >= STRONG else ("Lean" if prob >= LEAN else "")


def log_slate(log, games, by_team, lg, p, days, results=None, sos=None):
    """
    Add or refresh every unplayed game on `days` (ISO date strings). Ratings are
    frozen once a game is final; until then each run overwrites the row, so the
    numbers graded are the last ones logged before tip-off.
    """
    now = datetime.now().isoformat(timespec="minutes")
    slate = [g for g in games.values() if g["date"] in days and not g["final"]]
    index = {str(gid): i for i, gid in log["game_id"].items()}
    new_rows = []
    for g in slate:
        h, a = by_team.get(g["home"]), by_team.get(g["away"])
        if h is None or a is None:
            continue
        proj = raw_projection(h, a, lg)
        if proj is None:
            continue
        rm, rt, poss = proj
        ratings_r = (results or {})
        res_m = (ratings_r.get(g["home"], 0.0) - ratings_r.get(g["away"], 0.0)) if ratings_r else None
        sd = None
        if sos and g["home"] in sos and g["away"] in sos:
            sd = float(sos[g["home"]] - sos[g["away"]])
        mm, mt = model_numbers(rm, rt, g["neutral"], p, res_m, sd)
        row = {
            "game_id": g["game_id"], "date": g["date"], "minutes": g["minutes"], "away": g["away"], "home": g["home"],
            "neutral": g["neutral"], "conf_game": g["conf_game"], "logged_at": now,
            "home_adj_o": h["Adj O"], "home_adj_d": h["Adj D"], "home_pace": h.get("Pace"),
            "away_adj_o": a["Adj O"], "away_adj_d": a["Adj D"], "away_pace": a.get("Pace"),
            "raw_margin": rm, "raw_total": rt, "poss": poss, "results_margin": res_m, "sos_diff": sd,
            "model_margin": mm, "model_total": mt,
            "line_spread": g["spread"] if g["spread"] is not None else g["open_spread"],
            "open_spread": g["open_spread"],
            "line_total": g["total_line"] if g["total_line"] is not None else g["open_total"],
            "open_total": g["open_total"], "home_ml": g["home_ml"], "away_ml": g["away_ml"], "final": False,
        }
        if row["game_id"] in index:
            i = index[row["game_id"]]
            for k, v in row.items():
                log.at[i, k] = v
        else:
            new_rows.append(row)
    if new_rows:
        add = pd.DataFrame(new_rows, columns=LOG_COLS).astype(object)
        log = add if log.empty else pd.concat([log, add], ignore_index=True)
    return log


def slate_offsets(log, mask, p, stats):
    """
    Typical model-vs-line gap to center on. With little history, lean on the day's
    slate (its median gap), so a slate-wide offset can't masquerade as edges.
    """
    gs, gt = [], []
    for i in log.index[mask]:
        r = log.loc[i]
        rm, rt, res_m, sd = row_inputs(r)
        if rm is None:
            continue
        mm, mt = model_numbers(rm, rt, is_true(r["neutral"]), p, res_m, sd)
        if num(r["line_spread"]) is not None:
            gs.append(aligned_gap(mm, -num(r["line_spread"]), p["spread_alpha"], p["spread_beta"]))
        if num(r["line_total"]) is not None:
            gt.append(aligned_gap(mt, num(r["line_total"]), p["total_alpha"], p["total_beta"]))
    out = {}
    for key, gaps, n_hist in (("spread", gs, stats.get("games_with_spread", 0)), ("total", gt, stats.get("games_with_total", 0))):
        hist = p[f"{key}_xbar"]
        med = float(np.median(gaps)) if len(gaps) >= 5 else hist
        out[key] = (n_hist * hist + 100 * med) / (n_hist + 100)
    return out


def align_to_market(log, p):
    """Re-measure the model-to-market scale with every logged game that has a line, the current slate included."""
    raw = pd.to_numeric(log["raw_margin"], errors="coerce").to_numpy(float)
    res = pd.to_numeric(log["results_margin"], errors="coerce").to_numpy(float)
    res = np.where(np.isnan(res), raw, res)
    sos = pd.to_numeric(log["sos_diff"], errors="coerce").fillna(0).to_numpy(float)
    home = (~log["neutral"].map(is_true).astype(bool)).astype(float).to_numpy()
    mm = p["rating_weight"] * raw + p["results_weight"] * res + p["sos_weight"] * sos + p["hca"] * home
    mt = pd.to_numeric(log["raw_total"], errors="coerce").to_numpy(float) + p["total_bias"]
    p = dict(p)
    p["spread_alpha"], p["spread_beta"] = market_fit(mm, -pd.to_numeric(log["line_spread"], errors="coerce").to_numpy(float))
    p["total_alpha"], p["total_beta"] = market_fit(mt, pd.to_numeric(log["line_total"], errors="coerce").to_numpy(float))
    return p


def pick(log, p, day, stats):
    """Score one day's unplayed games and mark the best ones as picks (caps are per day)."""
    allow_strong = stats.get("learned_share", 0) >= STRONG_NEEDS_LEARNED
    mask = (log["date"] == day.isoformat()) & ~log["final"].map(is_true).astype(bool)
    xbar = slate_offsets(log, mask, p, stats)
    cands_s, cands_t = [], []
    for i in log.index[mask]:
        r = log.loc[i]
        rm, rt, res_m, sd = row_inputs(r)
        if rm is None:
            continue
        mm, mt = model_numbers(rm, rt, is_true(r["neutral"]), p, res_m, sd)
        log.at[i, "model_margin"], log.at[i, "model_total"] = mm, mt
        for c in ["spread_pick", "spread_edge", "spread_prob", "spread_tier", "spread_flag", "spread_note",
                  "total_pick", "total_edge", "total_prob", "total_tier", "total_flag", "total_note"]:
            log.at[i, c] = None
        line = num(r["line_spread"])
        if line is not None:
            gap = aligned_gap(mm, -line, p["spread_alpha"], p["spread_beta"])   # model vs market, home view
            gap_c = gap - xbar["spread"]
            over = abs(gap_c) > GAP_CAP["spread"]
            gap_used = max(-GAP_CAP["spread"], min(GAP_CAP["spread"], gap_c))
            edge = p["market_a"] + p["market_b"] * gap_used                  # learned expected beat of the line
            prob = phi(abs(edge) / p["sd_vs_spread"])
            flag = (f"Model and line disagree by {abs(gap_c):.1f} pts (guardrail: {GAP_CAP['spread']:.0f}). "
                    "Gaps this big usually mean news the model can't see - check injuries first.") if over else None
            cands_s.append((prob, i, "home" if edge > 0 else "away", edge, flag))
        tl = num(r["line_total"])
        if tl is not None and mt is not None:
            gap_c = aligned_gap(mt, tl, p["total_alpha"], p["total_beta"]) - xbar["total"]
            over = abs(gap_c) > GAP_CAP["total"]
            gap_used = max(-GAP_CAP["total"], min(GAP_CAP["total"], gap_c))
            edge = p["total_a"] + p["total_c"] * gap_used
            prob = phi(abs(edge) / p["sd_vs_total"])
            flag = (f"Model and line disagree by {abs(gap_c):.1f} pts on the total (guardrail: {GAP_CAP['total']:.0f}). "
                    "Check injuries and lineup news first.") if over else None
            cands_t.append((prob, i, "over" if edge > 0 else "under", edge, flag))
    for cands, kind, cap in ((cands_s, "spread", MAX_SPREAD_PICKS), (cands_t, "total", MAX_TOTAL_PICKS)):
        cands.sort(key=lambda c: c[0], reverse=True)
        n_clean = n_flagged = 0
        for prob, i, side, edge, flag in cands:
            log.at[i, f"{kind}_edge"], log.at[i, f"{kind}_prob"] = edge, prob
            t = tier(prob)
            if not t:
                continue
            if t == "Strong" and not allow_strong:
                t = "Lean"
                log.at[i, f"{kind}_note"] = (f"Model rates this Strong ({prob:.0%}), but Strong picks are held back "
                                             f"until {STRONG_NEEDS_LEARNED:.0%} of the model is learned from results.")
            if flag:
                if n_flagged >= MAX_FLAGGED[kind]:
                    continue
                n_flagged += 1
                log.at[i, f"{kind}_flag"] = flag
            else:
                if n_clean >= cap:
                    continue
                n_clean += 1
            log.at[i, f"{kind}_pick"], log.at[i, f"{kind}_tier"] = side, t
    return log


# ----------------------------------------------------------------------
# 5. Write tabs
# ----------------------------------------------------------------------
def pick_rows(log):
    out = []
    for _, r in log.iterrows():
        for kind in ("spread", "total"):
            side = r[f"{kind}_pick"]
            if not isinstance(side, str) or side not in ("home", "away", "over", "under"):
                continue
            line = num(r["line_spread"] if kind == "spread" else r["line_total"])
            if kind == "spread":
                team = r["home"] if side == "home" else r["away"]
                team_line = line if side == "home" else -line
                text = f"{team} {'PK' if team_line == 0 else f'{team_line:+.1f}'}"
                model_line = -num(r["model_margin"]) if side == "home" else num(r["model_margin"])
            else:
                text = f"{side.title()} {line:.1f}"
                model_line = num(r["model_total"])
            out.append({
                "Date": r["date"], "Time (ET)": (num(r["minutes"]) / 1440 if num(r["minutes"]) is not None else None), "GameID": r["game_id"],
                "Away": r["away"], "Home": r["home"], "Neutral": "Yes" if is_true(r["neutral"]) else "No",
                "Type": kind.title(), "Pick": text, "Side": side, "Line": line,
                "Model Line": model_line, "Edge": num(r[f"{kind}_edge"]), "Confidence": num(r[f"{kind}_prob"]),
                "Tier": r[f"{kind}_tier"],
                "Flag": r[f"{kind}_flag"] if isinstance(r[f"{kind}_flag"], str) else None,
                "Note": r[f"{kind}_note"] if isinstance(r[f"{kind}_note"], str) else None,
                "Model Margin (Home)": num(r["model_margin"]),
                "Model Total": num(r["model_total"]), "Open Line": num(r["open_spread"] if kind == "spread" else r["open_total"]),
                "Result": r[f"{kind}_result"] if isinstance(r[f"{kind}_result"], str) else ("Pending" if not is_true(r["final"]) else "No line"),
                "Units": num(r[f"{kind}_units"]), "Close Line": num(r["close_spread"] if kind == "spread" else r["close_total"]),
                "CLV": num(r[f"{kind}_clv"]), "Home Pts": num(r["home_pts"]), "Away Pts": num(r["away_pts"]),
            })
    return pd.DataFrame(out)


def write_tabs(log, p, stats, today, days):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill

    picks = pick_rows(log)
    window = [d.isoformat() for d in days]
    upcoming = (picks[picks["Date"].isin(window)].sort_values(["Date", "Time (ET)"], na_position="last")
                if len(picks) else picks)
    history = picks.sort_values(["Date", "Time (ET)"], ascending=[False, True]) if len(picks) else picks
    graded = history[history["Result"].isin(["W", "L", "P"])] if len(history) else history
    graded_clean = graded[graded["Flag"].isna()] if len(graded) else graded

    model_rows = [
        ("As of", today.isoformat(), "Date the engine last ran"),
        ("Look-ahead days", int(LOOKAHEAD_DAYS), "Picks tab covers today plus this many days ahead"),
        ("Graded games", stats.get("graded_games", 0), "Games the model has logged and seen finish"),
        ("Games with a spread", stats.get("games_with_spread", 0), ""),
        ("Learned share", round(stats.get("learned_share", 0), 3), "0 = all starting beliefs, 1 = all learned from results"),
        ("Home court (pts)", round(p["hca"], 2), "Learned from margins"),
        ("Rating weight", round(p["rating_weight"], 3), "Weight on the workbook's adjusted ratings"),
        ("Results rating weight", round(p["results_weight"], 3), "Weight on the engine's own schedule-adjusted ratings from game results"),
        ("Schedule gap weight", round(p["sos_weight"], 3), "Points per point of schedule-strength gap; + means tough schedules were underrated"),
        ("Total adjustment (pts)", round(p["total_bias"], 2), "Added to every projected total"),
        ("Spread: share of gap that comes true", round(p["market_b"], 3), "Near 0 = no edge over the line; higher = model adds information"),
        ("Spread: home lean vs line (pts)", round(p["market_a"], 2), ""),
        ("Total: share of gap that comes true", round(p["total_c"], 3), ""),
        ("Total: lean vs line (pts)", round(p["total_a"], 2), "+ = games tend to go over the line"),
        ("Spread scale vs market", round(p["spread_beta"], 3), "Market margin per model point (above 1 = the model's numbers are too compressed)"),
        ("Total scale vs market", round(p["total_beta"], 3), ""),
        ("Margin error SD", round(p["sd_margin"], 2), ""),
        ("Error vs spread SD", round(p["sd_vs_spread"], 2), ""),
        ("Error vs total SD", round(p["sd_vs_total"], 2), ""),
        ("Model avg miss (margin)", round(stats["model_mae_vs_result"], 2) if "model_mae_vs_result" in stats else None, "Lower is better"),
        ("Line avg miss (margin)", round(stats["market_mae_vs_result"], 2) if "market_mae_vs_result" in stats else None, "The bar to beat"),
        ("Model avg miss (total)", round(stats["model_total_mae"], 2) if "model_total_mae" in stats else None, ""),
        ("Line avg miss (total)", round(stats["market_total_mae"], 2) if "market_total_mae" in stats else None, ""),
        ("Strong pick bar", STRONG, "Model chance to win the bet"),
        ("Lean pick bar", LEAN, "52.4% breaks even at -110"),
        ("Picks graded", int(len(graded_clean)), "Not counting flagged picks"),
        ("Units", round(float(graded_clean["Units"].sum()), 2) if len(graded_clean) else 0, "1 unit risked per pick at -110, flagged picks excluded"),
        ("Avg CLV (pts)", round(float(graded_clean["CLV"].dropna().mean()), 2) if len(graded_clean) and graded_clean["CLV"].notna().any() else None,
         "Beat the closing line by this much on average - the best early sign of a real edge"),
    ]
    model_df = pd.DataFrame(model_rows, columns=["Parameter", "Value", "Note"])

    wb = load_workbook(WORKBOOK)
    for name in ("Picks", "Pick Log", "Model"):
        if name in wb.sheetnames:
            del wb[name]
    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864")

    def write(name, df, index):
        ws = wb.create_sheet(name, index)
        cols = list(df.columns)
        ws.append(cols)
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and math.isnan(v)) else v for v in row])
        for c in ws[1]:
            c.font, c.fill = head_font, head_fill
        for i, col in enumerate(cols, 1):
            width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)]) if len(df) else len(str(col))
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(width + 2, 8), 40)
        ws.freeze_panes = "A2"
        for i, col in enumerate(cols, 1):
            fmt = {"Time (ET)": "h:mm AM/PM", "Confidence": "0.0%", "Edge": "+0.0;-0.0;0.0", "CLV": "+0.0;-0.0;0.0",
                   "Units": "+0.00;-0.00;0.00"}.get(col)
            if fmt:
                for row in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                    row[0].number_format = fmt

    cols = ["Date", "Time (ET)", "GameID", "Away", "Home", "Neutral", "Type", "Pick", "Side", "Line", "Model Line",
            "Edge", "Confidence", "Tier", "Flag", "Note", "Model Margin (Home)", "Model Total", "Open Line", "Result", "Units",
            "Close Line", "CLV", "Home Pts", "Away Pts"]
    empty = pd.DataFrame(columns=cols)
    write("Picks", upcoming if len(upcoming) else empty, 0)
    write("Pick Log", history if len(history) else empty, 1)
    write("Model", model_df, 2)
    wb.save(WORKBOOK)
    return upcoming, graded


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
def run_engine():
    global LG
    today = today_et()
    days = slate_days(today)
    teams, by_team, games, rows = load_workbook_data()
    LG = league(teams)
    results, results_hca = results_ratings(games, set(by_team), today)
    sos = schedule_strength(rows)
    log = load_log()
    n_graded = grade(log, games)
    params, stats = learn(log)
    log = log_slate(log, games, by_team, LG, params, {d.isoformat() for d in days}, results, sos)
    params = align_to_market(log, params)
    for d in days:
        log = pick(log, params, d, stats)
    save_log(log)
    PARAMS_FILE.write_text(json.dumps({"as_of": today.isoformat(), "params": params, "stats": stats}, indent=2))
    upcoming, graded = write_tabs(log, params, stats, today, days)

    print(f"Pick engine - {today:%a %b %d}" + (f" (+{len(days) - 1} day look-ahead)" if len(days) > 1 else ""))
    print(f"  graded {n_graded} newly finished games; {stats['graded_games']} graded in total "
          f"(learned share {stats['learned_share']:.0%})")
    print(f"  home court {params['hca']:.2f} pts, spread gap that comes true {params['market_b']:.2f}, "
          f"total gap {params['total_c']:.2f}")
    print(f"  weights: adjusted ratings {params['rating_weight']:.2f}, results ratings {params['results_weight']:.2f}, "
          f"schedule gap {params['sos_weight']:+.2f}  ({len(results)} teams in results ratings)")
    if len(graded):
        clean_g = graded[graded["Flag"].isna()]
        w, l = (clean_g['Result'] == 'W').sum(), (clean_g['Result'] == 'L').sum()
        print(f"  pick record {w}-{l}, units {clean_g['Units'].sum():+.2f} (flagged picks tracked separately)")
    for d in days:
        iso = d.isoformat()
        on_day = log[(log["date"] == iso) & ~log["final"].map(is_true).astype(bool)]
        with_line = int(pd.to_numeric(on_day["line_spread"], errors="coerce").notna().sum())
        day_picks = upcoming[upcoming["Date"] == iso] if len(upcoming) else upcoming
        label = "today" if d == today else f"{d:%a %b %d}"
        print(f"  {label}: {len(on_day)} games logged, {with_line} with a spread posted, {len(day_picks)} picks")
        for _, r in day_picks.iterrows():
            mark = "  FLAGGED" if isinstance(r["Flag"], str) else ""
            print(f"    {r['Tier']:6} {r['Pick']:28} ({r['Away']} @ {r['Home']})  {r['Confidence']:.1%}{mark}")


LG = None
if __name__ == "__main__":
    run_engine()
