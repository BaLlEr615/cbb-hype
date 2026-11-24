"""
CBB Hype - everything the model uses from Bart Torvik (barttorvik.com), in one file.
(No API credits - this is Torvik's own site.)

What it pulls
  1. TEAM RATINGS   {YEAR}_team_results.csv
       Adj O, Adj D, tempo, Barthag, rank for every team.
  2. GAME PREDICTIONS  {YEAR}_super_sked.json
       Torvik's own projected score and win chance for every game this season,
       matched to the workbook's GameIDs (projected HOME margin and total).
  3. SNAPSHOTS
       A dated copy of both files every run, in torvik_snapshots/ - a record of what
       Torvik thought before each game, which is what an honest backtest needs.
  4. PAST SEASONS (optional) - pull_past_seasons([2024, 2025, 2026]).

Changes for the GitHub version (everything else is the same as torvik_pull.ipynb):
  * If the download fails (barttorvik.com sometimes blocks cloud servers), it falls back
    to a copy saved in this folder, then to the most recent snapshot, and says so.
  * Snapshots are only saved for fresh downloads, so a fallback never looks like new data.
  * Any error is printed and the rest of the daily update keeps going.

Season naming: Torvik uses the year a season ENDS, so 2026-27 is 2027.
"""
import io
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    FOLDER = Path(__file__).resolve().parent
except NameError:                       # pasted into a notebook cell
    FOLDER = Path.cwd()

YEAR = 2027
BASE = "https://barttorvik.com"
WORKBOOK = FOLDER / "college_basketball_by_conference.xlsx"
SNAP_DIR = FOLDER / "torvik_snapshots"
HIST_DIR = FOLDER / "torvik_history"
HEADERS = {"User-Agent": "Mozilla/5.0 (CBB Hype personal project)"}

# Torvik name -> workbook name. Add whatever the "no match" list prints.
NAME_FIX = {
    # "Connecticut": "UConn",
    # "Miami FL": "Miami",
}

# Column order of the super schedule (each game is a list, not a labeled object).
SKED_COLS = [
    "muid", "date", "conmatch", "matchup", "prediction", "ttq", "conf", "venue",
    "team1", "t1oe", "t1de", "t1py", "t1wp", "t1propt",
    "team2", "t2oe", "t2de", "t2py", "t2wp", "t2propt", "tpro",
    "t1qual", "t2qual", "gp", "result", "tempo", "possessions", "t1pts", "t2pts",
    "winner", "loser", "t1adjt", "t2adjt", "t1adjo", "t1adjd", "t2adjo", "t2adjd",
    "gamevalue", "mismatch", "blowout", "t1elite", "t2elite", "ord_date",
    "t1ppp", "t2ppp", "gameppp", "t1rk", "t2rk", "t1gs", "t2gs", "gamestats",
    "overtimes", "t1fun", "t2fun", "results",
]


# ----------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------
def _is_html(text):
    head = text[:400].lower()
    return "<html" in head or "<!doctype" in head


def _read_maybe_gz(path):
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
    return path.read_text(encoding="utf-8", errors="replace")


def fetch(filename, kind):
    """Text of barttorvik.com/<filename>, else a copy in this folder, else the latest snapshot."""
    url = f"{BASE}/{filename}"
    try:
        r = requests.get(url, timeout=60, headers=HEADERS)
        if r.ok and not _is_html(r.text) and len(r.text) > 1000:
            return r.text, "web"
        print(f"  {filename}: download didn't return a {kind} file (status {r.status_code}).")
    except requests.RequestException as e:
        print(f"  {filename}: download failed ({e}).")
    local = FOLDER / filename
    if local.exists():
        print(f"  {filename}: using the copy saved in this folder.")
        return local.read_text(encoding="utf-8", errors="replace"), "file"
    snaps = sorted(SNAP_DIR.glob(f"*_{filename}*")) if SNAP_DIR.exists() else []
    if snaps:
        print(f"  {filename}: using the most recent snapshot, {snaps[-1].name} (Torvik numbers may be a day or more old).")
        return _read_maybe_gz(snaps[-1]), "snapshot " + snaps[-1].name[:8]
    raise RuntimeError(f"No Torvik data for {filename}: the download failed and there's no saved copy.")


def snapshot(filename, text, src):
    if src != "web":
        return
    SNAP_DIR.mkdir(exist_ok=True)
    path = SNAP_DIR / f"{date.today():%Y%m%d}_{filename}"
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------
# Names: Torvik -> workbook
# ----------------------------------------------------------------------
def workbook_teams():
    if not WORKBOOK.exists():
        return set()
    return set(pd.read_excel(WORKBOOK, sheet_name="All Teams")["Team"].dropna().astype(str))


def make_namer(ours):
    swaps = ((" St.", " State"), (" State", " St."), ("Saint ", "St. "), ("St. ", "Saint "),
             ("&amp;", "&"), ("Cal St.", "Cal State"))
    cache = {}

    def name(t):
        t = str(t).strip()
        if t in cache:
            return cache[t]
        out = NAME_FIX.get(t, t)
        if ours and out not in ours:
            for a, b in swaps:
                alt = out.replace(a, b)
                if alt in ours:
                    out = alt
                    break
        cache[t] = out
        return out
    return name


# ----------------------------------------------------------------------
# 1. Team ratings
# ----------------------------------------------------------------------
def parse_ratings(text):
    """
    The header row has one fewer name than each data row (its last cell, "Fun Rk, adjt",
    covers two values), so reading it with the header would shift every column.
    Read by position: 0 rank, 1 team, 2 conf, 3 record, 4 adjoe, 6 adjde, 8 barthag,
    last = adjusted tempo.
    """
    raw = pd.read_csv(io.StringIO(text), header=None, skiprows=1)
    df = pd.DataFrame({
        "Torvik Team": raw[1].astype(str).str.strip(),
        "T Rank": pd.to_numeric(raw[0], errors="coerce"),
        "T Conf": raw[2],
        "T Record": raw[3],
        "T Adj O": pd.to_numeric(raw[4], errors="coerce"),
        "T Adj D": pd.to_numeric(raw[6], errors="coerce"),
        "T Barthag": pd.to_numeric(raw[8], errors="coerce"),
        "T Tempo": pd.to_numeric(raw[raw.columns[-1]], errors="coerce"),
    })
    for col, (lo, hi) in {"T Adj O": (75, 140), "T Adj D": (75, 140),
                          "T Tempo": (55, 85), "T Barthag": (0, 1)}.items():
        med = df[col].median()
        if not (lo <= med <= hi):
            raise ValueError(f"Torvik ratings: {col} median is {med:.2f}, expected {lo}-{hi}. "
                             "The file layout may have changed - check the positions in parse_ratings().")
    return df.dropna(subset=["T Adj O", "T Adj D"])


def pull_ratings(namer, ours):
    filename = f"{YEAR}_team_results.csv"
    text, src = fetch(filename, "CSV")
    snapshot(filename, text, src)
    df = parse_ratings(text)
    df["Team"] = df["Torvik Team"].map(namer)
    if ours:
        missing = sorted(ours - set(df["Team"]))
        print(f"  ratings: {len(df)} teams ({src}); matched {len(ours) - len(missing)} of {len(ours)} workbook teams")
        if missing:
            leftover = sorted(set(df["Team"]) - ours)
            print("  workbook teams with no Torvik match - add pairs to NAME_FIX:")
            print("     workbook:", ", ".join(missing))
            print("     Torvik leftovers:", ", ".join(leftover))
    return df


# ----------------------------------------------------------------------
# 2. Game predictions (super schedule)
# ----------------------------------------------------------------------
PRED_RE = re.compile(r"^\s*(?P<fav>.+?)\s+(?:\(?-?\d+(?:\.\d+)?\)?,?\s*)?(?P<a>\d{2,3})-(?P<b>\d{2,3})\s*\((?P<wp>\d{1,3})%\)")


def parse_prediction(s):
    """'Duke -7.5, 78-70 (75%)' -> ('Duke', 78, 70, 0.75). None if it doesn't parse."""
    if not isinstance(s, str):
        return None
    m = PRED_RE.match(s)
    if not m:
        return None
    a, b = int(m["a"]), int(m["b"])
    return m["fav"].strip(), max(a, b), min(a, b), int(m["wp"]) / 100


def parse_sked(text, year):
    raw = json.loads(text)
    rows = [[";".join(map(str, x)) if isinstance(x, list) else x for x in g] for g in raw]
    width = max(len(r) for r in rows)
    cols = SKED_COLS if width == len(SKED_COLS) else [f"c{i}" for i in range(width)]
    if width != len(SKED_COLS):
        print(f"  super schedule has {width} fields per game (expected {len(SKED_COLS)}); "
              "Torvik changed the layout - numbers below may be wrong.")
    df = pd.DataFrame([r + [None] * (width - len(r)) for r in rows], columns=cols)
    df["year"] = year
    df["game_date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce").dt.date
    for c in ["t1pts", "t2pts", "t1propt", "t2propt", "t1wp", "gp", "venue"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def torvik_projection(row):
    """
    Torvik's projected points for team1 and team2 and team1's win chance.
    Uses the numeric projected-points fields when they look like real scores,
    otherwise reads them out of the prediction text.
    """
    p1, p2, wp = row.get("t1propt"), row.get("t2propt"), row.get("t1wp")
    if pd.notna(p1) and pd.notna(p2) and 40 <= p1 <= 130 and 40 <= p2 <= 130:
        return float(p1), float(p2), (float(wp) if pd.notna(wp) and 0 <= wp <= 1 else None)
    parsed = parse_prediction(row.get("prediction"))
    if not parsed:
        return None
    fav, hi, lo, fwp = parsed
    if fav == str(row["team1"]).strip():
        return float(hi), float(lo), fwp
    if fav == str(row["team2"]).strip():
        return float(lo), float(hi), 1 - fwp
    return None


def workbook_games():
    """GameID, date and home/away teams from the workbook's Games tab."""
    if not WORKBOOK.exists():
        return pd.DataFrame(columns=["GameID", "game_date", "Home", "Away", "Neutral"])
    g = pd.read_excel(WORKBOOK, sheet_name="Games")
    g = g[g["GameID"].notna() & g["Team"].notna()]
    away = g["Site"] == "Away"
    out = pd.DataFrame({
        "GameID": g["GameID"].map(lambda v: str(int(v)) if isinstance(v, float) else str(v)),
        "game_date": pd.to_datetime(g["Date"], errors="coerce").dt.date,
        "Home": np.where(away, g["Opponent"], g["Team"]),
        "Away": np.where(away, g["Team"], g["Opponent"]),
        "Neutral": g["Site"] == "Neutral",
    })
    return out.drop_duplicates("GameID")


def match_games(sked, namer, wb):
    """Line Torvik's games up with the workbook's GameIDs and orient them home/away."""
    lookup = {}
    for r in wb.itertuples(index=False):
        lookup[(r.game_date, frozenset((r.Home, r.Away)))] = r
    out, unmatched = [], 0
    for _, row in sked.iterrows():
        t1, t2 = namer(row["team1"]), namer(row["team2"])
        proj = torvik_projection(row)
        hit = lookup.get((row["game_date"], frozenset((t1, t2))))
        if hit is None:
            unmatched += 1
            continue
        if proj is None:
            continue
        p1, p2, wp1 = proj
        home_is_1 = (t1 == hit.Home)
        hp, ap = (p1, p2) if home_is_1 else (p2, p1)
        hwp = None if wp1 is None else (wp1 if home_is_1 else 1 - wp1)
        out.append({
            "GameID": hit.GameID, "Date": hit.game_date, "Away": hit.Away, "Home": hit.Home,
            "Neutral": "Yes" if hit.Neutral else "No",
            "Torvik Home Pts": hp, "Torvik Away Pts": ap,
            "Torvik Home Margin": hp - ap, "Torvik Total": hp + ap, "Torvik Home Win %": hwp,
            "Torvik Prediction": row.get("prediction"),
        })
    return pd.DataFrame(out), unmatched


def pull_games(namer):
    filename = f"{YEAR}_super_sked.json"
    text, src = fetch(filename, "JSON")
    snapshot(filename, text, src)
    sked = parse_sked(text, YEAR)
    wb = workbook_games()
    games, unmatched = match_games(sked, namer, wb)
    print(f"  game predictions: {len(sked)} Torvik games ({src}); {len(games)} matched to workbook GameIDs"
          + (f", {unmatched} not in the workbook (mostly non-D1 or games it doesn't list)" if unmatched else ""))
    if len(games):
        n_today = (games["Date"] == date.today()).sum()
        print(f"  today: {n_today} games with a Torvik projection")
    return games


# ----------------------------------------------------------------------
# 3. Past seasons, for the backtest
# ----------------------------------------------------------------------
def pull_past_seasons(years):
    """
    Save Torvik's game predictions and results for earlier seasons to torvik_history/.
    The prediction text is Torvik's pre-game call. Don't use the t1adjo/t1adjd columns
    from past seasons as pre-game inputs - they may be end-of-season numbers.
    """
    HIST_DIR.mkdir(exist_ok=True)
    for y in years:
        try:
            text, src = fetch(f"{y}_super_sked.json", "JSON")
        except RuntimeError as e:
            print(f"  {y}: {e}")
            continue
        sked = parse_sked(text, y)
        proj = sked.apply(torvik_projection, axis=1)
        sked["torvik_t1_pts"] = [p[0] if p else None for p in proj]
        sked["torvik_t2_pts"] = [p[1] if p else None for p in proj]
        sked["torvik_t1_wp"] = [p[2] if p else None for p in proj]
        keep = ["year", "game_date", "muid", "team1", "team2", "venue", "conf", "prediction",
                "torvik_t1_pts", "torvik_t2_pts", "torvik_t1_wp", "gp", "t1pts", "t2pts", "possessions"]
        path = HIST_DIR / f"{y}_torvik_games.csv"
        sked[[c for c in keep if c in sked]].to_csv(path, index=False)
        ok = sked["torvik_t1_pts"].notna().mean()
        print(f"  {y}: {len(sked)} games saved to {path.name} ({ok:.0%} with a readable Torvik projection)")


# ----------------------------------------------------------------------
# Writing the Torvik tab and the helper the engine calls
# ----------------------------------------------------------------------
def write_tab(ratings, games):
    if not WORKBOOK.exists():
        return
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    wb = load_workbook(WORKBOOK)
    tabs = [("Torvik", ratings.sort_values("T Rank"))]
    if games is not None and len(games):
        tabs.append(("Torvik Games", games))
    elif "Torvik Games" in wb.sheetnames:
        print("  keeping the Torvik Games tab already in the workbook (no new game predictions this run)")
    for name, _ in tabs:
        if name in wb.sheetnames:
            del wb[name]
    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864")
    for name, df in tabs:
        ws = wb.create_sheet(name)
        cols = list(df.columns)
        ws.append(cols)
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in row])
        for c in ws[1]:
            c.font, c.fill = head_font, head_fill
        ws.freeze_panes = "A2"
        for i, col in enumerate(cols, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(len(col) + 2, 9), 34)
    wb.save(WORKBOOK)


def torvik_for_game(game_id):
    """Torvik's numbers for one workbook GameID, or None."""
    if TORVIK_GAMES is None or not len(TORVIK_GAMES):
        return None
    hit = TORVIK_GAMES[TORVIK_GAMES["GameID"] == str(game_id)]
    if not len(hit):
        return None
    r = hit.iloc[0]
    return {"margin": float(r["Torvik Home Margin"]), "total": float(r["Torvik Total"]),
            "home_win": None if pd.isna(r["Torvik Home Win %"]) else float(r["Torvik Home Win %"])}


def run():
    print(f"Torvik pull - {YEAR} season, {datetime.now():%a %b %d %I:%M %p}")
    ours = workbook_teams()
    namer = make_namer(ours)
    ratings = pull_ratings(namer, ours)
    try:
        games = pull_games(namer)
    except Exception as e:                  # ratings still useful if the schedule fails
        print(f"  game predictions skipped: {e}")
        games = pd.DataFrame()
    write_tab(ratings, games)
    return ratings, games


TORVIK, TORVIK_GAMES = None, None
if __name__ == "__main__":
    try:
        TORVIK, TORVIK_GAMES = run()
    except Exception as e:      # never stop the daily update over Torvik; the rest still runs
        print(f"Torvik pull skipped: {e}")

# For the backtest, run once (takes a minute):
# pull_past_seasons([2024, 2025, 2026])
