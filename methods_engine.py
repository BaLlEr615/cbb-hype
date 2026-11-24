"""
CBB Hype - pick methods engine.

Runs five ways of picking every game, day by day, using ONLY games played before each
day (no peeking), and writes the results to the workbook for the site's Picks tab:

  Model          the team-ratings model vs the line
  Model + Trends the model, nudged by learned trend factors (underdog size, home court,
                 line movement; for totals: high/low totals, total movement, overall lean)
  Trends         the trend factors alone
  Torvik         Torvik's projected score vs the line (needs torvik_pull.py to have run)
  Consensus      only games where two or more of Model, Trends and Torvik agree

Confidence: each method is a small logistic model whose weights are re-learned every
day from earlier games and pulled toward "no edge" (50%) until results prove otherwise.
A trend only raises confidence after it has kept working.

Manual trend weights: put trend_settings.json in this folder (the site's Export button
makes it). With "mode": "manual", Model + Trends uses your weights instead of learned ones.

Workbook tabs written:
  Method Picks    one row per game, bet type and method: side, confidence, tier, result
  Method Records  each method's record (picks that cleared the bar), spreads and totals
  Method Factors  the inputs the site needs to redo Model + Trends with your own weights
  Method Weights  what each method has learned, as of the latest day

Use in JupyterLab (PythonFun), after the pick engine and Torvik cells:
    %run methods_engine.py
"""
import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    FOLDER = Path(__file__).resolve().parent
except NameError:                      # pasted into a notebook cell
    FOLDER = Path.cwd()

WORKBOOK = Path(os.environ.get("CBB_WORKBOOK", FOLDER / "college_basketball_by_conference.xlsx"))
LOG_FILE = FOLDER / "games_log.csv"
SETTINGS_FILE = FOLDER / "trend_settings.json"

LEAN, STRONG = 0.545, 0.570           # same bars as the pick engine (52.4% breaks even at -110)
PAYOUT = 100 / 110
MIN_TRAIN = 150                       # finished games needed before a method starts picking
PRIOR_GAMES = 300                     # how hard weights are pulled toward "no edge"
GAP_CLIP = {"spread": 12.0, "total": 15.0}
RESULTS_LAMBDA, RESULTS_CAP = 2.0, 30.0
METHODS = ["Model", "Model + Trends", "Trends", "Torvik", "Consensus"]

# Trend factors. Each is +1/-1/0 (or a number between) from the HOME side for spreads,
# from the OVER side for totals. "pp" = percentage points of confidence at a coin flip.
FACTORS = {
    "spread": [
        ("dog_small", "Underdog of 0.5-3.5"),
        ("dog_mid", "Underdog of 4-9.5"),
        ("dog_big", "Underdog of 10+"),
        ("home", "Home team (not neutral)"),
        ("move", "Line moved toward this side (per 3 pts)"),
    ],
    "total": [
        ("lean", "Every game (overall over/under lean)"),
        ("high", "Total of 150+"),
        ("low", "Total under 135"),
        ("tmove", "Total moved this way (per 3 pts)"),
    ],
}
PP_PER_LOGIT = 25.0                   # 1 logit unit = 25 pp at 50%


def today_et():
    forced = os.environ.get("CBB_TODAY")
    if forced:
        return date.fromisoformat(forced)
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return date.today()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_games():
    g = pd.read_excel(WORKBOOK, sheet_name="Games")
    d1 = set(pd.read_excel(WORKBOOK, sheet_name="All Teams")["Team"].dropna())
    g = g[g["GameID"].notna() & g["Team"].notna() & g["Opponent"].notna()].copy()
    g["GameID"] = g["GameID"].map(lambda v: str(int(v)) if isinstance(v, float) else str(v))
    g["pref"] = g["Site"].map({"Home": 0, "Neutral": 1, "Away": 2}).fillna(3)
    g = g.sort_values(["GameID", "pref"]).drop_duplicates("GameID")
    away = (g["Site"] == "Away").to_numpy()
    flip = lambda col: np.where(away, -g[col], g[col]).astype(float)
    G = pd.DataFrame({
        "id": g["GameID"].to_numpy(),
        "date": pd.to_datetime(g["Date"]).dt.date.to_numpy(),
        "time": g["Time (ET)"].to_numpy(),
        "home": np.where(away, g["Opponent"], g["Team"]),
        "away": np.where(away, g["Team"], g["Opponent"]),
        "neutral": (g["Site"] == "Neutral").to_numpy(),
        "final": g["Result"].notna().to_numpy() & g["Pts"].notna().to_numpy(),
        "hp": np.where(away, g["Opp Pts"], g["Pts"]).astype(float),
        "ap": np.where(away, g["Pts"], g["Opp Pts"]).astype(float),
        "close": flip("Spread"), "open": flip("Open Spread"),
        "tclose": g["O/U"].astype(float).to_numpy(), "topen": g["Open O/U"].astype(float).to_numpy(),
    })
    G["open"] = G["open"].fillna(G["close"])
    G["topen"] = G["topen"].fillna(G["tclose"])
    G = G[G["date"].notna()].sort_values(["date", "id"]).reset_index(drop=True)
    return G, d1


def load_logged_model(G):
    """The pick engine's pre-game model numbers, re-oriented to this table's home team."""
    if not LOG_FILE.exists():
        return {}
    log = pd.read_csv(LOG_FILE, dtype={"game_id": str})
    if "model_margin" not in log:
        return {}
    homes = dict(zip(G["id"], G["home"]))
    out = {}
    for r in log.itertuples(index=False):
        gid = str(r.game_id)
        if gid not in homes or pd.isna(r.model_margin):
            continue
        m = float(r.model_margin)
        if r.home != homes[gid]:
            m = -m
        t = None if pd.isna(getattr(r, "model_total", np.nan)) else float(r.model_total)
        out[gid] = (m, t)
    return out


def load_torvik(G):
    try:
        t = pd.read_excel(WORKBOOK, sheet_name="Torvik Games")
    except Exception:
        return {}
    if not len(t) or "Torvik Home Margin" not in t:
        return {}
    homes = dict(zip(G["id"], G["home"]))
    out = {}
    for r in t.itertuples(index=False):
        gid = str(int(r.GameID)) if isinstance(r.GameID, float) else str(r.GameID)
        if gid not in homes or pd.isna(r[t.columns.get_loc("Torvik Home Margin")]):
            continue
        m = float(r[t.columns.get_loc("Torvik Home Margin")])
        tot = r[t.columns.get_loc("Torvik Total")]
        if r.Home != homes[gid]:
            m = -m
        out[gid] = (m, None if pd.isna(tot) else float(tot))
    return out


def load_settings():
    if SETTINGS_FILE.exists():
        try:
            s = json.loads(SETTINGS_FILE.read_text())
            return {"mode": s.get("mode", "learned"), "spread": s.get("spread", {}), "total": s.get("total", {})}
        except Exception as e:
            print(f"  trend_settings.json couldn't be read ({e}); using learned weights")
    return {"mode": "learned", "spread": {}, "total": {}}


# ----------------------------------------------------------------------
# Walk-forward team ratings (margins and points), from games before a date
# ----------------------------------------------------------------------
def fit_ratings(prior, d1):
    name = np.vectorize(lambda t: t if t in d1 else "non-D1")
    h, a = name(prior["home"].to_numpy()), name(prior["away"].to_numpy())
    teams = sorted(set(h) | set(a))
    ix = {t: i for i, t in enumerate(teams)}
    k, n = len(teams), len(prior)
    hi, ai = np.array([ix[t] for t in h]), np.array([ix[t] for t in a])
    home = (~prior["neutral"].to_numpy()).astype(float)
    rows = np.arange(n)
    X = np.zeros((n, k + 1)); X[rows, hi] += 1; X[rows, ai] -= 1; X[:, k] = home
    y = np.clip((prior["hp"] - prior["ap"]).to_numpy(), -RESULTS_CAP, RESULTS_CAP)
    P = np.eye(k + 1) * RESULTS_LAMBDA; P[k, k] = 0.01
    b = np.linalg.solve(X.T @ X + P, X.T @ y)
    # points: home pts = mu + O_h + D_a + h/2 ; away pts = mu + O_a + D_h - h/2
    XT = np.zeros((2 * n, 2 * k + 2))
    XT[rows, hi] = 1; XT[rows, k + ai] = 1; XT[rows, 2 * k] = 1; XT[rows, 2 * k + 1] = 0.5 * home
    XT[n + rows, ai] = 1; XT[n + rows, k + hi] = 1; XT[n + rows, 2 * k] = 1; XT[n + rows, 2 * k + 1] = -0.5 * home
    yT = np.r_[prior["hp"].to_numpy(), prior["ap"].to_numpy()]
    PT = np.eye(2 * k + 2) * RESULTS_LAMBDA * 4; PT[2 * k, 2 * k] = PT[2 * k + 1, 2 * k + 1] = 1e-4
    bT = np.linalg.solve(XT.T @ XT + PT, XT.T @ yT)
    return ix, b, bT, k


def wf_projection(fit, d1, home, away, neutral):
    ix, b, bT, k = fit
    nm = lambda t: t if t in d1 else "non-D1"
    h, a = ix.get(nm(home)), ix.get(nm(away))
    if h is None or a is None:
        return None, None
    margin = b[h] - b[a] + (0 if neutral else b[k])
    total = 2 * bT[2 * k] + bT[h] + bT[k + a] + bT[a] + bT[k + h]
    return float(margin), float(total)


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------
def add_factor_columns(G):
    L, T = G["close"], G["tclose"]
    for f, (lo, hi) in {"dog_small": (0.5, 3.5), "dog_mid": (4.0, 9.5), "dog_big": (10.0, 99.0)}.items():
        G[f"fs_{f}"] = np.where(L.between(lo, hi), 1.0, np.where((-L).between(lo, hi), -1.0, 0.0))
    G["fs_home"] = np.where(G["neutral"], 0.0, 1.0)
    G["fs_move"] = np.clip(G["open"] - G["close"], -3, 3).fillna(0) / 3
    G["ft_lean"] = 1.0
    G["ft_high"] = (T >= 150).astype(float)
    G["ft_low"] = (T < 135).astype(float)
    G["ft_tmove"] = np.clip(G["tclose"] - G["topen"], -3, 3).fillna(0) / 3


def fit_logistic(X, y, strength=PRIOR_GAMES, prior=None):
    """Logistic regression with no intercept, each weight pulled toward its prior (default 0)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    k = X.shape[1]
    prior = np.zeros(k) if prior is None else np.asarray(prior, float)
    lam = strength * 0.25 * np.maximum((X ** 2).mean(axis=0) if len(X) else np.ones(k), 1e-6)
    w = prior.copy()
    for _ in range(25):
        p = sigmoid(X @ w)
        grad = X.T @ (y - p) - lam * (w - prior)
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(lam)
        step = np.linalg.solve(H, grad)
        w += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def tier(p):
    return "Strong" if p >= STRONG else ("Lean" if p >= LEAN else "")


def run_methods():
    today = today_et()
    G, d1 = load_games()
    logged = load_logged_model(G)
    torvik = load_torvik(G)
    settings = load_settings()
    print(f"Methods engine - {today:%a %b %d}: {len(G)} games, {G['final'].sum()} final; "
          f"{len(logged)} with logged model numbers, {len(torvik)} with Torvik numbers; "
          f"trend weights: {settings['mode']}")

    add_factor_columns(G)
    G["x_spread"] = G["hp"] - G["ap"] + G["close"]          # + = home covered
    G["x_total"] = G["hp"] + G["ap"] - G["tclose"]          # + = over
    for c in ["model_m", "model_t", "torvik_m", "torvik_t"]:
        G[c] = np.nan
    G["model_src"] = ""

    last_day = today + timedelta(days=1)
    dates = sorted(d for d in G["date"].unique() if d <= last_day)
    rows, factor_rows = [], []
    weights_latest = {}

    for d in dates:
        prior = G[(G["date"] < d) & G["final"]]
        day = G.index[G["date"] == d]
        fit = fit_ratings(prior, d1) if len(prior) >= 50 else None
        for i in day:
            r = G.loc[i]
            if r["id"] in logged:
                G.at[i, "model_m"], mt = logged[r["id"]]
                G.at[i, "model_t"] = mt if mt is not None else np.nan
                G.at[i, "model_src"] = "pick engine"
            elif fit is not None:
                m, t = wf_projection(fit, d1, r["home"], r["away"], r["neutral"])
                if m is not None:
                    G.at[i, "model_m"], G.at[i, "model_t"] = m, t
                    G.at[i, "model_src"] = "results ratings"
            if r["id"] in torvik:
                G.at[i, "torvik_m"], tt = torvik[r["id"]]
                G.at[i, "torvik_t"] = tt if tt is not None else np.nan
        if len(prior) < MIN_TRAIN:
            continue

        for kind in ("spread", "total"):
            line_col, x_col = ("close", "x_spread") if kind == "spread" else ("tclose", "x_total")
            mcol, tcol = ("model_m", "torvik_m") if kind == "spread" else ("model_t", "torvik_t")
            sign = 1.0 if kind == "spread" else -1.0     # spread gap = model margin + line; total gap = model total - line
            fkeys = [f for f, _ in FACTORS[kind]]
            fcols = [("fs_" if kind == "spread" else "ft_") + f for f in fkeys]
            clip = GAP_CLIP[kind]

            def gap_of(df, col):
                v = df[col] + (df[line_col] if kind == "spread" else -df[line_col])
                return np.clip(v.to_numpy(float), -clip, clip) / clip      # scaled to [-1, 1]

            tr = prior[prior[line_col].notna() & prior[x_col].notna() & (prior[x_col] != 0)]
            y = (tr[x_col] > 0).astype(float).to_numpy()
            F_tr = tr[fcols].to_numpy(float)
            learned = {}
            okm = tr[mcol].notna().to_numpy()
            if okm.sum() >= 50:
                learned["Model"] = fit_logistic(gap_of(tr[okm], mcol)[:, None], y[okm])
                learned["Model + Trends"] = fit_logistic(np.c_[gap_of(tr[okm], mcol), F_tr[okm]], y[okm])
            if len(tr) >= 50:
                learned["Trends"] = fit_logistic(F_tr, y)
            okt = tr[tcol].notna().to_numpy()
            if okt.sum() >= 50:
                learned["Torvik"] = fit_logistic(gap_of(tr[okt], tcol)[:, None], y[okt])
            weights_latest[kind] = {"date": d.isoformat(), "n": len(tr), **{m: w.tolist() for m, w in learned.items()}}

            manual = settings["mode"] == "manual"
            man_w = np.array([float(settings[kind].get(f, 0.0)) / PP_PER_LOGIT for f in fkeys])

            for i in day:
                r = G.loc[i]
                if pd.isna(r[line_col]):
                    continue
                fv = r[fcols].to_numpy(float)
                f = dict(zip(fkeys, fv))
                gm = gap_of(G.loc[[i]], mcol)[0] if pd.notna(r[mcol]) else None
                gt = gap_of(G.loc[[i]], tcol)[0] if pd.notna(r[tcol]) else None
                probs = {}
                if "Model" in learned and gm is not None:
                    probs["Model"] = float(sigmoid(learned["Model"][0] * gm))
                    if manual:
                        probs["Model + Trends"] = float(sigmoid(learned["Model"][0] * gm + man_w @ fv))
                    else:
                        probs["Model + Trends"] = float(sigmoid(learned["Model + Trends"] @ np.r_[gm, fv]))
                if "Trends" in learned:
                    probs["Trends"] = float(sigmoid((man_w if manual else learned["Trends"]) @ fv))
                if "Torvik" in learned and gt is not None:
                    probs["Torvik"] = float(sigmoid(learned["Torvik"][0] * gt))
                probs["Consensus"] = consensus(probs)

                fin = bool(r["final"])
                xval = r[x_col] if fin else np.nan
                factor_rows.append({
                    "Date": d.isoformat(), "GameID": r["id"], "Kind": kind.title(), "Away": r["away"], "Home": r["home"],
                    "Line": r[line_col], "Model Gap": gm, "Model Slope": float(learned["Model"][0]) if "Model" in learned else None,
                    **{f"F {k}": f[k] for k in fkeys}, "Outcome": xval, "Final": fin,
                })
                for meth in METHODS:
                    p_home = probs.get(meth)
                    if p_home is None:
                        continue
                    side_home = p_home >= 0.5
                    conf = p_home if side_home else 1 - p_home
                    if kind == "spread":
                        side = "home" if side_home else "away"
                        team = r["home"] if side_home else r["away"]
                        tl = r["close"] if side_home else -r["close"]
                        text = f"{team} {'PK' if tl == 0 else f'{tl:+.1f}'}"
                    else:
                        side = "over" if side_home else "under"
                        text = f"{side.title()} {r['tclose']:.1f}"
                    t = tier(conf)
                    res, units = "Pending", None
                    if fin and not pd.isna(xval):
                        s = 1 if side_home else -1
                        res = "P" if xval == 0 else ("W" if s * xval > 0 else "L")
                        units = {"W": PAYOUT, "L": -1.0, "P": 0.0}[res] if t else None
                    rows.append({
                        "Date": d.isoformat(), "Time (ET)": r["time"], "GameID": r["id"], "Away": r["away"], "Home": r["home"],
                        "Neutral": "Yes" if r["neutral"] else "No", "Type": kind.title(), "Method": meth,
                        "Pick": text, "Side": side, "Line": r[line_col], "Open": r["open"] if kind == "spread" else r["topen"],
                        "Confidence": round(conf, 4), "Tier": t, "Result": res, "Units": units,
                        "Model Value": r[mcol] if meth in ("Model", "Model + Trends") else (r[tcol] if meth == "Torvik" else None),
                        "Model Source": r["model_src"] if meth in ("Model", "Model + Trends") else None,
                        "Home Pts": r["hp"] if fin else None, "Away Pts": r["ap"] if fin else None,
                    })

    picks = pd.DataFrame(rows)
    factors = pd.DataFrame(factor_rows)
    records = method_records(picks)
    weights = weight_table(weights_latest, settings)
    write_tabs(picks, records, factors, weights)

    print("  records (picks that cleared the bar, graded at the closing line):")
    for _, r in records[records["Scope"] == "Season"].iterrows():
        print(f"    {r['Method']:15} {r['Type']:6} {r['Record']:>9}  {r['Win %'] if pd.notna(r['Win %']) else '':>6}  units {r['Units']:+.1f}")
    return picks, records


INDEPENDENT = ("Model", "Trends", "Torvik")   # Model + Trends overlaps both, so it doesn't vote
AGREE = 0.52


def consensus(probs):
    """
    A pick when 2+ of the independent methods (Model, Trends, Torvik) lean the same side
    at 52%+ and none leans the other way at 52%+. Confidence = their average.
    Otherwise shows the average lean, held just under the pick bar.
    """
    votes = [(p >= 0.5, max(p, 1 - p)) for m, p in probs.items() if m in INDEPENDENT and p is not None]
    if len(votes) < 2:
        return None
    home = [c for h, c in votes if h and c >= AGREE]
    away = [c for h, c in votes if not h and c >= AGREE]
    if len(home) >= 2 and not away:
        return float(np.mean(home))
    if len(away) >= 2 and not home:
        return 1 - float(np.mean(away))
    lean = float(np.mean([(c if h else 1 - c) for h, c in votes]))
    return float(np.clip(lean, 1 - LEAN + 0.001, LEAN - 0.001))


def method_records(picks):
    out = []
    if not len(picks):
        return pd.DataFrame(columns=["Scope", "Method", "Type", "Record", "Win %", "Units", "Picks"])
    graded = picks[(picks["Tier"] != "") & picks["Result"].isin(["W", "L", "P"])]
    for meth in METHODS:
        for kind in ("Spread", "Total", "All"):
            g = graded[(graded["Method"] == meth) & ((graded["Type"] == kind) | (kind == "All"))]
            w, l, p = (g["Result"] == "W").sum(), (g["Result"] == "L").sum(), (g["Result"] == "P").sum()
            out.append({"Scope": "Season", "Method": meth, "Type": kind, "Record": f"{w}-{l}" + (f"-{p}" if p else ""),
                        "Win %": round(w / (w + l), 3) if w + l else None, "Units": round(float(g["Units"].sum()), 2), "Picks": int(len(g))})
    return pd.DataFrame(out)


def weight_table(wl, settings):
    rows = []
    for kind in ("spread", "total"):
        w = wl.get(kind)
        if not w:
            continue
        fk = FACTORS[kind]
        mt = w.get("Model + Trends"); tr = w.get("Trends")
        rows.append({"Kind": kind.title(), "Factor": "model_gap", "Label": "Model disagreement with the line (full gap)",
                     "Model": w["Model"][0] * PP_PER_LOGIT if "Model" in w else None,
                     "Model + Trends": mt[0] * PP_PER_LOGIT if mt else None, "Trends": None,
                     "Torvik": w["Torvik"][0] * PP_PER_LOGIT if "Torvik" in w else None,
                     "Manual": None, "As Of": w["date"], "Games": w["n"]})
        for j, (f, label) in enumerate(fk):
            rows.append({"Kind": kind.title(), "Factor": f, "Label": label, "Model": None,
                         "Model + Trends": mt[1 + j] * PP_PER_LOGIT if mt else None,
                         "Trends": tr[j] * PP_PER_LOGIT if tr else None, "Torvik": None,
                         "Manual": settings[kind].get(f), "As Of": w["date"], "Games": w["n"]})
    df = pd.DataFrame(rows)
    for c in ["Model", "Model + Trends", "Trends", "Torvik"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    df.attrs["mode"] = settings["mode"]
    return df


def write_tabs(picks, records, factors, weights):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    wb = load_workbook(WORKBOOK)
    for name in ("Method Picks", "Method Records", "Method Factors", "Method Weights"):
        if name in wb.sheetnames:
            del wb[name]
    hf, hfill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864")
    pos = wb.sheetnames.index("Model") + 1 if "Model" in wb.sheetnames else 0
    weights = weights.copy()
    weights["Mode"] = weights.attrs.get("mode", "learned")
    for off, (name, df) in enumerate((("Method Picks", picks), ("Method Records", records),
                                      ("Method Weights", weights), ("Method Factors", factors))):
        ws = wb.create_sheet(name, pos + off)
        cols = list(df.columns)
        ws.append(cols)
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and math.isnan(v)) else v for v in row])
        for c in ws[1]:
            c.font, c.fill = hf, hfill
        ws.freeze_panes = "A2"
        for i, col in enumerate(cols, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(len(str(col)) + 2, 9), 30)
            if col == "Confidence":
                for cell in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                    cell[0].number_format = "0.0%"
            if col == "Time (ET)":
                for cell in ws.iter_rows(min_row=2, min_col=i, max_col=i):
                    cell[0].number_format = "h:mm AM/PM"
    wb.save(WORKBOOK)


if __name__ == "__main__":
    METHOD_PICKS, METHOD_RECORDS = run_methods()
