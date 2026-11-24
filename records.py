"""
College basketball records by conference -> Excel
Data source: CollegeBasketballData.com API (https://api.collegebasketballdata.com)

This is the records code from Updated_CBB_1.ipynb, unchanged except:
  * the API key comes ONLY from the CBBD_API_KEY environment variable (on GitHub
    that's the repository secret), so the key never sits in the code.
  * the %run lines at the bottom are gone - run_daily.py runs the other steps.

Creates one workbook with:
  - "All Teams" sheet: every team ranked by adjusted net rating
  - One sheet per conference, sorted like a standings table

Each team gets: overall + conference record, adjusted net / offense / defense
(with national ranks), pace, and four-factor stats for the team and its
opponents (eFG%, TO%, ORB%), SRS, and best win / worst loss.
Stat columns are color-scaled green (good) to red.

A "Games" tab lists every game of the season (played and upcoming) with
date, time, site, opponent, result, and score - one row per team per game,
as a filterable Excel table. It also has betting lines (spread, over/under,
moneyline, opening lines) and, once played, cover (ATS) and over/under results,
plus Spread Difference / O/U Difference per game.

A "Today" tab (first tab) lists today's games, one row per game: home and
away teams, their conferences and ratings, and the opening spread and total.

An "ATS & Totals" tab averages those per team in four side-by-side blocks
(Overall, Home, Neutral, Away) with national ranks.
"""

import os
import re
import sys
from datetime import date, timedelta

import pandas as pd
import requests

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
API_KEY = os.getenv("CBBD_API_KEY", "").strip()
BASE_URL = "https://api.collegebasketballdata.com"

# Season = the year the season ENDS (2026-27 season -> 2027).
# Leave as None to pick the current season automatically.
SEASON = None

# Pull the season's games for the Games tab, conference W-L, and
# best win / worst loss (~7+ extra calls).
INCLUDE_GAME_RESULTS = True

# Rating used to judge best win / worst loss: "srs", "adj_net", or "auto"
# ("auto" = SRS once it's published, Adj Net until then).
QUALITY_RATING = "auto"

# Betting lines on the Games tab (/lines endpoint, ~7 extra calls).
INCLUDE_BETTING_LINES = True
# When several sportsbooks have a line, use the first one found in this order
# (case-insensitive); otherwise whichever book has a spread.
PREFERRED_PROVIDERS = ["DraftKings", "ESPN BET", "consensus", "Bovada"]

# Line used for Spread Difference / O/U Difference and the ATS & Totals tab:
# "open" (opening line) or "close" (closing line). If the chosen line wasn't
# posted for a game, the other one is used.
ANALYSIS_LINE = "open"

OUTPUT_FILE = "college_basketball_by_conference.xlsx"


# ----------------------------------------------------------------------
# API helpers
# ----------------------------------------------------------------------
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
})


def api_get(endpoint, **params):
    """GET an endpoint and return parsed JSON (drops None params)."""
    params = {k: v for k, v in params.items() if v is not None}
    resp = session.get(f"{BASE_URL}{endpoint}", params=params, timeout=60)
    if resp.status_code == 401:
        sys.exit("401 Unauthorized - check your API key.")
    if resp.status_code == 429:
        sys.exit("429 Too Many Requests - you've hit your monthly/rate limit.")
    resp.raise_for_status()
    return resp.json()


def current_season():
    """Seasons are labeled by ending year; Oct-Dec belongs to next year's season."""
    today = date.today()
    return today.year + 1 if today.month >= 10 else today.year


# ----------------------------------------------------------------------
# Data pulls
# ----------------------------------------------------------------------
def get_conferences():
    """Map conference abbreviation -> full name."""
    data = api_get("/conferences")
    return {c.get("abbreviation"): c.get("name") for c in data if c.get("abbreviation")}


def get_teams(season):
    """All teams with their conference for the season."""
    data = api_get("/teams", season=season)
    rows = []
    for t in data:
        rows.append({
            "TeamID": t.get("id"),
            "Team": t.get("school"),
            "Mascot": t.get("mascot"),
            "Conference": t.get("conference"),
        })
    return pd.DataFrame(rows)


def get_season_stats(season):
    """Record, pace, and four factors (team + opponent) from /stats/team/season."""
    data = api_get("/stats/team/season", season=season)
    rows = []
    for s in data:
        tff = (s.get("teamStats") or {}).get("fourFactors") or {}
        off = (s.get("opponentStats") or {}).get("fourFactors") or {}
        rows.append({
            "TeamID": s.get("teamId"),
            "Games": s.get("games") or 0,
            "W": s.get("wins") or 0,
            "L": s.get("losses") or 0,
            "StatsConference": s.get("conference"),
            "Pace": s.get("pace"),
            "eFG%": tff.get("effectiveFieldGoalPct"),
            "Opp eFG%": off.get("effectiveFieldGoalPct"),
            "TO%": tff.get("turnoverRatio"),
            "Opp TO%": off.get("turnoverRatio"),
            "ORB%": tff.get("offensiveReboundPct"),
            "Opp ORB%": off.get("offensiveReboundPct"),
        })
    return pd.DataFrame(rows)


def get_adjusted_ratings(season):
    """Adjusted offense / defense / net ratings from /ratings/adjusted."""
    data = api_get("/ratings/adjusted", season=season)
    rows = [{
        "TeamID": r.get("teamId"),
        "Adj Net": r.get("netRating"),
        "Adj O": r.get("offensiveRating"),
        "Adj D": r.get("defensiveRating"),
    } for r in data]
    return pd.DataFrame(rows, columns=["TeamID", "Adj Net", "Adj O", "Adj D"])


def get_srs(season):
    """Simple Rating System from /ratings/srs (empty early in the season)."""
    data = api_get("/ratings/srs", season=season)
    rows = [{"TeamID": r.get("teamId"), "SRS": r.get("rating")} for r in data]
    return pd.DataFrame(rows, columns=["TeamID", "SRS"])


GAMES_PER_CALL_CAP = 3000  # /games returns at most this many per request


def fetch_range(endpoint, season, start, end):
    """Fetch [start, end]; split the range in half if the per-call cap was hit."""
    chunk = api_get(endpoint, season=season,
                    startDateRange=start.isoformat(), endDateRange=end.isoformat())
    if len(chunk) >= GAMES_PER_CALL_CAP and start < end:
        mid = start + (end - start) / 2
        return (fetch_range(endpoint, season, start, mid)
                + fetch_range(endpoint, season, mid + timedelta(days=1), end))
    return chunk


def fetch_season(endpoint, season, id_field):
    """Pull a game-level endpoint for the whole season, month by month, de-duplicated."""
    start, end = date(season - 1, 10, 1), date(season, 4, 30)
    items, seen = [], set()
    cursor = start
    while cursor <= end:
        nxt = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        for item in fetch_range(endpoint, season, cursor, min(nxt - timedelta(days=1), end)):
            if item.get(id_field) not in seen:
                seen.add(item.get(id_field))
                items.append(item)
        cursor = nxt
    return items


def get_all_games(season):
    """Every game of the season - played, upcoming, postponed."""
    return fetch_season("/games", season, "id")


def get_betting_lines(season):
    """
    One line per game: {gameId: {...}} with the spread from the HOME team's view
    (negative = home favored), total, moneylines, opening numbers, and book.
    """
    preferred = [p.lower() for p in PREFERRED_PROVIDERS]

    def rank(line):
        name = str(line.get("provider") or "").lower()
        has_spread = line.get("spread") is not None
        return (not has_spread, preferred.index(name) if name in preferred else len(preferred))

    picked = {}
    for game in fetch_season("/lines", season, "gameId"):
        lines = [l for l in (game.get("lines") or [])
                 if l.get("spread") is not None or l.get("overUnder") is not None]
        if lines:
            picked[game.get("gameId")] = min(lines, key=rank)

    # Sanity-check the sign convention with moneylines: the favorite's moneyline is
    # negative. If home favorites mostly carry a POSITIVE spread, the API is giving
    # the away team's view, so flip everything to home view.
    agree = disagree = 0
    for l in picked.values():
        hml, aml, sp = l.get("homeMoneyline"), l.get("awayMoneyline"), l.get("spread")
        if None in (hml, aml, sp) or sp == 0 or hml == aml:
            continue
        home_fav = hml < aml
        if home_fav == (sp < 0):
            agree += 1
        else:
            disagree += 1
    flip = -1 if disagree > agree else 1

    out = {}
    for gid, l in picked.items():
        sp, spo = l.get("spread"), l.get("spreadOpen")
        out[gid] = {
            "home_spread": None if sp is None else flip * sp,
            "home_spread_open": None if spo is None else flip * spo,
            "total": l.get("overUnder"),
            "total_open": l.get("overUnderOpen"),
            "home_ml": l.get("homeMoneyline"),
            "away_ml": l.get("awayMoneyline"),
            "provider": l.get("provider"),
        }
    return out


def is_completed(g):
    hp, ap = g.get("homePoints"), g.get("awayPoints")
    if hp is None or ap is None or hp == ap:
        return False
    return str(g.get("status", "final")).lower() in ("final", "completed", "complete")


def schedule_table(games, teams, ratings, d1_ids, lines=None):
    """
    One row per D-I team per game (each D-I vs D-I game appears twice, once from
    each side), so filtering the Team column gives that team's full schedule.
    """
    conf_of = dict(zip(teams["TeamID"], teams["Conference"]))
    rating_of = dict(zip(ratings["TeamID"], ratings["Rating"]))
    lines = lines or {}

    def team_view(value, side):
        """Home-view spread -> this team's spread (e.g. -5.5 = this team favored by 5.5)."""
        if value is None:
            return None
        return value if side == "home" else -value

    rows = []
    for g in games:
        start = pd.to_datetime(g.get("startDate"), utc=True, errors="coerce")
        local = start.tz_convert("America/New_York") if pd.notna(start) else None
        done = is_completed(g)
        neutral = bool(g.get("neutralSite"))
        for side, opp in (("home", "away"), ("away", "home")):
            tid, oid = g.get(f"{side}TeamId"), g.get(f"{opp}TeamId")
            if tid not in d1_ids:
                continue
            pts, opp_pts = g.get(f"{side}Points"), g.get(f"{opp}Points")
            orating = rating_of.get(oid)
            line = lines.get(g.get("id"), {})
            spread = team_view(line.get("home_spread"), side)
            total = line.get("total")
            if done and spread is not None:
                ats = pts - opp_pts + spread       # >0 covered, 0 push, <0 didn't
                ats = "Cover" if ats > 0 else ("Push" if ats == 0 else "Miss")
            else:
                ats = None
            # Spread Difference = margin + spread (+ = covered by that much)
            # O/U Difference    = line - game total (+ = went under by that much)
            open_spread = team_view(line.get("home_spread_open"), side)
            open_total = line.get("total_open")
            if ANALYSIS_LINE == "open":
                a_spread = open_spread if open_spread is not None else spread
                a_total = open_total if open_total is not None else total
            else:
                a_spread = spread if spread is not None else open_spread
                a_total = total if total is not None else open_total
            spread_diff = (pts - opp_pts + a_spread) if done and a_spread is not None else None
            ou_diff = (a_total - (pts + opp_pts)) if done and a_total is not None else None
            if done and total is not None:
                game_total = pts + opp_pts
                ou = "Over" if game_total > total else ("Push" if game_total == total else "Under")
            else:
                ou = None
            rows.append({
                "Date": local.date() if local is not None else None,
                # Excel stores a time as a fraction of a day; formatted h:mm AM/PM below
                "Time (ET)": None if (local is None or g.get("startTimeTbd"))
                             else (local.hour * 60 + local.minute) / 1440,
                "Team": g.get(f"{side}Team"),
                "Conference": conf_of.get(tid),
                "Site": "Neutral" if neutral else side.title(),
                "Opponent": g.get(f"{opp}Team"),
                "Opp Conference": conf_of.get(oid) or ("non-D1" if oid not in d1_ids else None),
                "Opp Rating": None if orating is None or pd.isna(orating) else round(float(orating), 1),
                "Result": ("W" if pts > opp_pts else "L") if done else None,
                "Pts": pts if done else None,
                "Opp Pts": opp_pts if done else None,
                "Margin": (pts - opp_pts) if done else None,
                "Score": f"{pts}-{opp_pts}" if done else None,
                "Spread": spread,
                "Open Spread": team_view(line.get("home_spread_open"), side),
                "ATS": ats,
                "O/U": total,
                "Open O/U": line.get("total_open"),
                "Game Total": (pts + opp_pts) if done else None,
                "Over/Under": ou,
                "Spread Difference": spread_diff,
                "O/U Difference": ou_diff,
                "Moneyline": line.get(f"{side}_ml"),
                "Opp Moneyline": line.get(f"{opp}_ml"),
                "Sportsbook": line.get("provider"),
                "Conf Game": "Yes" if g.get("conferenceGame") else "No",
                "Status": str(g.get("status") or "").replace("_", " ").title(),
                "Game Type": g.get("tournament") or g.get("gameType"),
                "Venue": g.get("venue"),
                "City": g.get("city"),
                "State": g.get("state"),
                "GameID": g.get("id"),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Date", "Time (ET)", "Team"], na_position="last")
    return out


def team_results(games):
    """Flatten games into one row per team per game: who they played, where, W/L."""
    rows = []
    for g in (g for g in games if is_completed(g)):
        home_won = g["homePoints"] > g["awayPoints"]
        neutral = bool(g.get("neutralSite"))
        for side, opp, won in (("home", "away", home_won), ("away", "home", not home_won)):
            if neutral:
                site = "vs"   # neutral site
            else:
                site = "vs" if side == "home" else "@"
            rows.append({
                "TeamID": g.get(f"{side}TeamId"),
                "OppID": g.get(f"{opp}TeamId"),
                "OppName": g.get(f"{opp}Team"),
                "Site": site,
                "Neutral": neutral,
                "Won": won,
                "Score": f"{g[side + 'Points']}-{g[opp + 'Points']}",
                "Loc": "Neutral" if neutral else side.title(),
                "Margin": g[side + "Points"] - g[opp + "Points"],
                "ConfGame": bool(g.get("conferenceGame")),
            })
    return pd.DataFrame(rows)


def conference_records(results):
    conf = results[results["ConfGame"]]
    out = conf.groupby("TeamID")["Won"].agg(["sum", "count"]).reset_index()
    out["Conf W"] = out["sum"].astype(int)
    out["Conf L"] = (out["count"] - out["sum"]).astype(int)
    return out[["TeamID", "Conf W", "Conf L"]]


def location_splits(results):
    """
    Record and average point differential per game at Home / Away / Neutral.
    Won at home by 5 -> Home +/- of +5.0; lost away by 10 -> Away +/- of -10.0.
    With several games it's the per-game average, so teams with more home
    games aren't inflated.
    """
    out = pd.DataFrame({"TeamID": results["TeamID"].unique()})
    for loc in ("Home", "Away", "Neutral"):
        g = results[results["Loc"] == loc].groupby("TeamID").agg(
            W=("Won", "sum"), G=("Won", "count"), Margin=("Margin", "mean")).reset_index()
        g[f"{loc} Rec"] = g["W"].astype(int).astype(str) + "-" + (g["G"] - g["W"]).astype(int).astype(str)
        g[f"{loc} +/-"] = g["Margin"].round(1)
        out = out.merge(g[["TeamID", f"{loc} Rec", f"{loc} +/-"]], on="TeamID", how="left")
    return out


def best_win_worst_loss(results, ratings, d1_ids):
    """
    Best win  = beaten D-I opponent with the HIGHEST rating.
    Worst loss = opponent lost to with the LOWEST rating; a loss to a
                 non-D-I team (not in the /teams list) is always the worst.
    A D-I opponent missing a rating is labeled [unrated], never non-D1.
    """
    r = results.merge(ratings.rename(columns={"TeamID": "OppID", "Rating": "OppRating"}),
                      on="OppID", how="left")
    r["OppD1"] = r["OppID"].isin(d1_ids)

    def label(row):
        if pd.notna(row["OppRating"]):
            tag = f"{row['OppRating']:+.1f}"
        else:
            tag = "unrated" if row["OppD1"] else "non-D1"
        neutral = " (N)" if row["Neutral"] else ""
        return f"{row['Site']} {row['OppName']}{neutral}, {row['Score']} [{tag}]"

    # Wins: rated D-I opponents rank by rating; unrated D-I sit below them; skip non-D-I
    wins = r[r["Won"] & r["OppD1"]].copy()
    wins["Sort"] = wins["OppRating"].fillna(float("-inf"))
    best = wins.loc[wins.groupby("TeamID")["Sort"].idxmax()]
    best = best.assign(**{"Best Win": best.apply(label, axis=1)})[["TeamID", "Best Win"]]

    # Losses: non-D-I is worst, then lowest rating, unrated D-I last
    losses = r[~r["Won"]].copy()
    losses["Sort"] = losses["OppRating"].fillna(float("inf"))
    losses.loc[~losses["OppD1"], "Sort"] = float("-inf")
    worst = losses.loc[losses.groupby("TeamID")["Sort"].idxmin()]
    worst = worst.assign(**{"Worst Loss": worst.apply(label, axis=1)})[["TeamID", "Worst Loss"]]

    return best.merge(worst, on="TeamID", how="outer")


def quality_ratings(df, season):
    """
    Rating used to judge wins/losses, with a fallback chain:
    this season's SRS -> this season's Adj Net -> LAST season's final ratings.
    Returns (DataFrame[TeamID, Rating], description).
    """
    order = {"srs": ["SRS"], "adj_net": ["Adj Net"]}.get(QUALITY_RATING, ["SRS", "Adj Net"])
    for col in order:
        if df[col].notna().any():
            return df[["TeamID", col]].rename(columns={col: "Rating"}), col

    # Nothing published for this season yet - use last season as a preseason proxy
    prev = season - 1
    getters = {"SRS": get_srs, "Adj Net": get_adjusted_ratings}
    for col in order:
        prior = getters[col](prev)
        if not prior.empty and prior[col].notna().any():
            prior = pd.to_numeric(prior.set_index("TeamID")[col], errors="coerce")
            return prior.rename("Rating").reset_index(), f"last season's {col} (current not out yet)"
    return pd.DataFrame(columns=["TeamID", "Rating"]), "no ratings available yet"


# ----------------------------------------------------------------------
# Excel output
# ----------------------------------------------------------------------
def win_pct(wins, losses):
    """Wins / (wins + losses), 0 when no games played."""
    total = wins + losses
    return (wins / total.where(total > 0)).fillna(0).round(3)


def as_percent(series):
    """Show rates as percents (52.3). The API may send 0.523 or 52.3; handle both."""
    series = pd.to_numeric(series, errors="coerce")
    if series.notna().any() and series.abs().max() <= 1.5:
        series = series * 100
    return series.round(1)


# Which direction is "good" for each stat column (drives the color scale)
HIGHER_IS_BETTER = ["Win %", "Conf Win %", "Home +/-", "Away +/-", "Neutral +/-",
                    "Adj Net", "SRS", "Adj O",
                    "eFG%", "Opp TO%", "ORB%"]
LOWER_IS_BETTER = ["Adj D", "Opp eFG%", "TO%", "Opp ORB%"]
NUMBER_FORMATS = {
    "Win %": "0.000", "Conf Win %": "0.000",
    "Home +/-": "+0.0;-0.0;0.0", "Away +/-": "+0.0;-0.0;0.0", "Neutral +/-": "+0.0;-0.0;0.0",
    "Adj Net": "+0.0;-0.0;0.0", "SRS": "+0.0;-0.0;0.0", "Adj O": "0.0", "Adj D": "0.0", "Pace": "0.0",
    "eFG%": "0.0", "Opp eFG%": "0.0", "TO%": "0.0", "Opp TO%": "0.0",
    "ORB%": "0.0", "Opp ORB%": "0.0",
}


def color_and_format(ws, header_row, n_rows):
    """Green = good, red = bad, per column; apply number formats."""
    from openpyxl.formatting.rule import ColorScaleRule

    if n_rows == 0:
        return
    good, mid, bad = "63BE7B", "FFFFFF", "F8696B"
    first, last = header_row + 1, header_row + n_rows
    for cell in ws[header_row]:
        name, col = cell.value, cell.column_letter
        rng = f"{col}{first}:{col}{last}"
        if name in NUMBER_FORMATS:
            for row in ws[rng]:
                row[0].number_format = NUMBER_FORMATS[name]
        if name in HIGHER_IS_BETTER or name in LOWER_IS_BETTER:
            lo, hi = (bad, good) if name in HIGHER_IS_BETTER else (good, bad)
            ws.conditional_formatting.add(rng, ColorScaleRule(
                start_type="min", start_color=lo,
                mid_type="percentile", mid_value=50, mid_color=mid,
                end_type="max", end_color=hi))


def safe_sheet_name(name, used):
    """Excel sheet names: max 31 chars, no []:*?/\\ and must be unique."""
    clean = re.sub(r"[\[\]\:\*\?\/\\]", "-", str(name or "Unknown"))[:31] or "Unknown"
    base, i = clean, 2
    while clean.lower() in used:
        suffix = f" ({i})"
        clean = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(clean.lower())
    return clean


def style_header(ws, header_row):
    """Bold navy header, freeze panes below it, auto-fit columns."""
    from openpyxl.styles import Font, PatternFill

    for cell in ws[header_row]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F3864")
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    for col in ws.iter_cols(min_row=header_row):
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(width + 3, 45)


def format_games_sheet(ws, n_rows):
    """Excel table with filters, date/time formats, W/L coloring."""
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import PatternFill, Font
    from openpyxl.worksheet.table import Table, TableStyleInfo

    last_col = ws.cell(row=1, column=ws.max_column).column_letter
    table = Table(displayName="Games", ref=f"A1:{last_col}{n_rows + 1}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    ws.add_table(table)
    ws.freeze_panes = "A2"

    headers = {c.value: c.column_letter for c in ws[1]}
    for header, fmt in (("Date", "ddd mm/dd/yyyy"), ("Time (ET)", "h:mm AM/PM"),
                        ("Opp Rating", "+0.0;-0.0;0.0"), ("Margin", "+0;-0;0"),
                        ("Spread", "+0.0;-0.0;PK"), ("Open Spread", "+0.0;-0.0;PK"),
                        ("O/U", "0.0"), ("Open O/U", "0.0"),
                        ("Moneyline", "+0;-0;0"), ("Opp Moneyline", "+0;-0;0"),
                        ("Spread Difference", "+0.0;-0.0;0.0"), ("O/U Difference", "+0.0;-0.0;0.0")):
        col = headers.get(header)
        if col:
            for (cell,) in ws[f"{col}2:{col}{n_rows + 1}"]:
                cell.number_format = fmt

    green = dict(fill=PatternFill("solid", fgColor="C6EFCE"), font=Font(color="006100", bold=True))
    red = dict(fill=PatternFill("solid", fgColor="FFC7CE"), font=Font(color="9C0006", bold=True))
    for header, good, bad in (("Result", "W", "L"), ("ATS", "Cover", "Miss")):
        col = headers.get(header)
        if col:
            rng = f"{col}2:{col}{n_rows + 1}"
            ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{good}"'], **green))
            ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{bad}"'], **red))
    for header in ("Spread Difference", "O/U Difference"):
        col = headers.get(header)
        if col:
            plus_minus_colors(ws, f"{col}2:{col}{n_rows + 1}")

    for col in ws.iter_cols(min_row=1, max_row=min(n_rows + 1, 500)):
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(width, 8) + 3, 40)
    if headers.get("Date"):
        ws.column_dimensions[headers["Date"]].width = 16


def todays_games(games, teams, ratings, lines, d1_ids, day=None):
    """
    One row per game played today (Eastern time). Spread is from the HOME
    team's view (-5.5 = home favored by 5.5). Each team's rating is the same
    rating used for Opp Rating on the Games tab.
    """
    day = day or pd.Timestamp.now(tz="America/New_York").date()
    conf_of = dict(zip(teams["TeamID"], teams["Conference"]))
    rating_of = dict(zip(ratings["TeamID"], ratings["Rating"]))
    lines = lines or {}

    def conf(team_id, api_conf):
        if team_id in d1_ids:
            return conf_of.get(team_id) or api_conf
        return "non-D1"

    def rating(team_id):
        r = rating_of.get(team_id)
        return None if r is None or pd.isna(r) else round(float(r), 1)

    rows = []
    for g in games:
        start = pd.to_datetime(g.get("startDate"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        local = start.tz_convert("America/New_York")
        if local.date() != day:
            continue
        line = lines.get(g.get("id"), {})
        hid, aid = g.get("homeTeamId"), g.get("awayTeamId")
        rows.append({
            "Time (ET)": None if g.get("startTimeTbd") else (local.hour * 60 + local.minute) / 1440,
            "Away Team": g.get("awayTeam"),
            "Away Conference": conf(aid, g.get("awayConference")),
            "Away Rating": rating(aid),
            "Home Team": g.get("homeTeam"),
            "Home Conference": conf(hid, g.get("homeConference")),
            "Home Rating": rating(hid),
            "Neutral Site": "Yes" if g.get("neutralSite") else "No",
            "Open Spread (Home)": line.get("home_spread_open"),
            "Open O/U": line.get("total_open"),
            "GameID": g.get("id"),
        })
    out = pd.DataFrame(rows, columns=["Time (ET)", "Away Team", "Away Conference", "Away Rating",
                                      "Home Team", "Home Conference", "Home Rating", "Neutral Site",
                                      "Open Spread (Home)", "Open O/U", "GameID"])
    return out.sort_values(["Time (ET)", "Home Team"], na_position="last") if not out.empty else out


def write_today_sheet(writer, sheet_name, today_df, day, rating_desc):
    """Today's games as a filterable table, with a title row."""
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.table import Table, TableStyleInfo

    header_row = 3
    today_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=header_row - 1)
    ws = writer.sheets[sheet_name]
    ws["A1"] = f"Games today - {day:%A %m/%d/%Y}  ({len(today_df)} games)"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = (f"Open Spread is from the home team's view (-5.5 = home favored by 5.5).  "
                f"Ratings: {rating_desc}.  Blank line = not posted yet.")
    ws["A2"].font = Font(italic=True, color="595959")

    n = len(today_df)
    if n == 0:
        ws[f"A{header_row + 1}"] = "No games scheduled today."
        return
    last_col = ws.cell(row=header_row, column=today_df.shape[1]).column_letter
    table = Table(displayName="Today", ref=f"A{header_row}:{last_col}{header_row + n}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    ws.add_table(table)
    ws.freeze_panes = f"A{header_row + 1}"

    formats = {"Time (ET)": "h:mm AM/PM", "Away Rating": "+0.0;-0.0;0.0",
               "Home Rating": "+0.0;-0.0;0.0", "Open Spread (Home)": "+0.0;-0.0;PK",
               "Open O/U": "0.0"}
    widths = {"Time (ET)": 10, "Away Team": 24, "Away Conference": 15, "Away Rating": 9,
              "Home Team": 24, "Home Conference": 15, "Home Rating": 9, "Neutral Site": 9,
              "Open Spread (Home)": 11, "Open O/U": 9, "GameID": 9}
    for c in range(1, today_df.shape[1] + 1):
        head = ws.cell(row=header_row, column=c)
        letter = head.column_letter
        head.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[letter].width = widths.get(head.value, 12)
        if head.value in formats:
            for (cell,) in ws[f"{letter}{header_row + 1}:{letter}{header_row + n}"]:
                cell.number_format = formats[head.value]
    ws.row_dimensions[header_row].height = 30


def plus_minus_colors(ws, rng):
    """Green above 0, red below 0 (same look as the W/L and Cover/Miss cells)."""
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import PatternFill, Font

    ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0"],
        fill=PatternFill("solid", fgColor="C6EFCE"), font=Font(color="006100")))
    ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0"],
        fill=PatternFill("solid", fgColor="FFC7CE"), font=Font(color="9C0006")))


ATS_LOCATIONS = ("Overall", "Home", "Neutral", "Away")


def ats_tables(schedule, teams):
    """
    Per-team averages of Spread Difference and O/U Difference, overall and by site.
    Each average uses only games that had that line, and teams with no games at a
    site are left blank and unranked. Rank 1 = biggest number (ties share a rank).
    All four blocks use the same team order (best overall ATS first) so a row
    reads straight across.
    """
    base = teams[["Team", "Conference"]].dropna(subset=["Team"]).drop_duplicates("Team")
    played = schedule[schedule["Result"].notna()] if not schedule.empty else schedule
    blocks = {}
    for loc in ATS_LOCATIONS:
        g = played if loc == "Overall" else played[played["Site"] == loc]
        agg = g.groupby("Team").agg(
            Games=("Result", "size"),
            **{"Games w/ Spread": ("Spread Difference", "count"),
               "Games w/ O/U": ("O/U Difference", "count"),
               "Spread Difference": ("Spread Difference", "mean"),
               "O/U Difference": ("O/U Difference", "mean")}).reset_index()
        out = base.merge(agg, on="Team", how="left")
        for c in ("Games", "Games w/ Spread", "Games w/ O/U"):
            out[c] = out[c].fillna(0).astype(int)
        for c in ("Spread Difference", "O/U Difference"):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
        out.insert(3, "Location", loc)
        out["Spread Rank"] = out["Spread Difference"].rank(ascending=False, method="min").astype("Int64")
        out["O/U Rank"] = out["O/U Difference"].rank(ascending=False, method="min").astype("Int64")
        blocks[loc] = out

    order = (blocks["Overall"].sort_values(["Spread Rank", "Team"], na_position="last")["Team"])
    return {loc: b.set_index("Team").loc[order].reset_index() for loc, b in blocks.items()}


def write_ats_sheet(writer, sheet_name, blocks, season_label):
    """Four blocks side by side (Overall | Home | Neutral | Away), each its own Excel table."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    title_row, block_row, header_row = 1, 3, 4
    line = "opening" if ANALYSIS_LINE == "open" else "closing"
    col = 1
    for loc in ATS_LOCATIONS:
        block = blocks[loc]
        block.to_excel(writer, sheet_name=sheet_name, index=False,
                       startrow=header_row - 1, startcol=col - 1)
        ws = writer.sheets[sheet_name]
        first, last = col, col + block.shape[1] - 1
        first_l, last_l = get_column_letter(first), get_column_letter(last)
        n = len(block)

        ws.merge_cells(f"{first_l}{block_row}:{last_l}{block_row}")
        cell = ws[f"{first_l}{block_row}"]
        cell.value = loc
        cell.font = Font(bold=True, size=12)
        cell.fill = PatternFill("solid", fgColor="D4E1EB")
        cell.alignment = Alignment(horizontal="center")

        table = Table(displayName=f"ATS_{loc}", ref=f"{first_l}{header_row}:{last_l}{header_row + n}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        ws.add_table(table)

        headers = {ws.cell(header_row, c).value: get_column_letter(c) for c in range(first, last + 1)}
        for h in ("Spread Difference", "O/U Difference"):
            letter = headers[h]
            rng = f"{letter}{header_row + 1}:{letter}{header_row + n}"
            for (c,) in ws[rng]:
                c.number_format = "+0.0;-0.0;0.0"
            plus_minus_colors(ws, rng)

        widths = {"Team": 24, "Conference": 14, "Location": 10, "Games": 8,
                  "Games w/ Spread": 9, "Games w/ O/U": 9,
                  "Spread Difference": 11, "O/U Difference": 11, "Spread Rank": 8, "O/U Rank": 8}
        for h, letter in headers.items():
            ws.column_dimensions[letter].width = widths.get(h, 10)
            ws[f"{letter}{header_row}"].alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[header_row].height = 30
        col = last + 2  # one blank column between blocks

    ws["A1"] = (f"Against the spread & totals - {season_label} season, {line} lines.  "
                "Spread Difference = margin + spread (+ = covered by that much).  "
                "O/U Difference = line - game total (+ = under, - = over).  Rank 1 = highest.")
    ws["A1"].font = Font(bold=True)
    ws.freeze_panes = f"B{header_row + 1}"


def build_workbook(df, conf_names, season, rating_note="", schedule=None, ats=None,
                   today=None, rating_desc=""):
    from openpyxl.styles import Font

    sort_cols = ["Win %", "W", "Adj Net"]
    if "Conf Win %" in df.columns:
        sort_cols = ["Conf Win %", "Conf W", "Win %", "Adj Net"]
    season_label = f"{season - 1}-{str(season)[-2:]}"

    used = set()
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # Today's games first, so the workbook opens on them
        if today is not None:
            day = pd.Timestamp.now(tz="America/New_York").date()
            write_today_sheet(writer, safe_sheet_name("Today", used), today, day, rating_desc)

        # Overview sheet: whole country ranked by adjusted net rating
        name = safe_sheet_name("All Teams", used)
        overview = df.sort_values("Adj Net", ascending=False, na_position="last")
        overview.to_excel(writer, sheet_name=name, index=False)
        style_header(writer.sheets[name], 1)
        color_and_format(writer.sheets[name], 1, len(overview))

        # Every game, one row per team per game, as a filterable Excel table
        if schedule is not None and not schedule.empty:
            name = safe_sheet_name("Games", used)
            schedule.to_excel(writer, sheet_name=name, index=False)
            format_games_sheet(writer.sheets[name], len(schedule))

        # Per-team spread / total averages, overall and by site
        if ats:
            write_ats_sheet(writer, safe_sheet_name("ATS & Totals", used), ats, season_label)

        # One sheet per conference, ranked like a standings table
        for conf, group in sorted(df.groupby("Conference"), key=lambda x: str(x[0])):
            group = group.sort_values(sort_cols, ascending=False).drop(columns=["Conference"])
            group.insert(0, "Rank", range(1, len(group) + 1))
            name = safe_sheet_name(conf, used)
            group.to_excel(writer, sheet_name=name, index=False, startrow=1)  # header on row 2
            ws = writer.sheets[name]
            ws["A1"] = f"{conf_names.get(conf, conf)} - {season_label} season{rating_note}"
            ws["A1"].font = Font(bold=True, size=14)
            style_header(ws, 2)
            color_and_format(ws, 2, len(group))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if not API_KEY:
        sys.exit("No API key. Set the CBBD_API_KEY environment variable "
                 "(on GitHub: Settings > Secrets and variables > Actions > CBBD_API_KEY).")

    season = SEASON or current_season()
    print(f"Pulling season {season - 1}-{str(season)[-2:]} ...")

    conf_names = get_conferences()
    teams = get_teams(season)
    records = get_season_stats(season)

    if records.empty and SEASON is None:
        print("No stats yet for this season - falling back to last season.")
        season -= 1
        teams = get_teams(season)
        records = get_season_stats(season)

    df = teams.merge(records, on="TeamID", how="left")

    print("Pulling adjusted efficiency ratings ...")
    df = df.merge(get_adjusted_ratings(season), on="TeamID", how="left")
    df = df.merge(get_srs(season), on="TeamID", how="left")
    print(f"Ratings found: Adj Net for {df['Adj Net'].notna().sum()} teams, "
          f"SRS for {df['SRS'].notna().sum()} teams.")
    # Fill in conference from stats if /teams had none
    df["Conference"] = df["Conference"].fillna(df.get("StatsConference"))
    df = df.drop(columns=["StatsConference"], errors="ignore")
    df = df[df["Conference"].notna()]

    for col in ["Games", "W", "L"]:
        df[col] = df[col].fillna(0).astype(int)
    df["Record"] = df["W"].astype(str) + "-" + df["L"].astype(str)
    df["Win %"] = win_pct(df["W"], df["L"])

    # Rates as percents; national ranks for the adjusted ratings (1 = best)
    for col in ["eFG%", "Opp eFG%", "TO%", "Opp TO%", "ORB%", "Opp ORB%"]:
        df[col] = as_percent(df[col])
    for col in ["Adj Net", "SRS", "Adj O", "Adj D", "Pace"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(1)
    df["Net Rk"] = df["Adj Net"].rank(ascending=False, method="min").astype("Int64")
    df["O Rk"] = df["Adj O"].rank(ascending=False, method="min").astype("Int64")
    df["D Rk"] = df["Adj D"].rank(ascending=True, method="min").astype("Int64")  # fewer pts allowed = better

    rating_note, schedule, ats, today, rating_desc = "", None, None, None, ""
    if INCLUDE_GAME_RESULTS:
        print("Pulling game results ...")
        try:
            all_games = get_all_games(season)
            results = team_results(all_games)
        except requests.HTTPError as e:
            print(f"Skipping game-based columns ({e}).")
            all_games, results = [], pd.DataFrame()

        # Rating used for opponent quality (Games tab + best win / worst loss)
        ratings, rating_desc = quality_ratings(df, season)
        d1_ids = set(teams["TeamID"])
        lines = {}
        if INCLUDE_BETTING_LINES:
            print("Pulling betting lines ...")
            try:
                lines = get_betting_lines(season)
                print(f"Betting lines found for {len(lines)} games.")
            except requests.HTTPError as e:
                print(f"Skipping betting lines ({e}).")
        schedule = schedule_table(all_games, teams, ratings, d1_ids, lines)
        ats = ats_tables(schedule, teams)
        today = todays_games(all_games, teams, ratings, lines, d1_ids)
        print(f"Today tab: {len(today)} games.")
        played = int(schedule["Result"].notna().sum()) if not schedule.empty else 0
        print(f"Games tab: {len(schedule)} team-game rows ({played} with results).")

        if not results.empty:
            df = df.merge(conference_records(results), on="TeamID", how="left")
            df = df.merge(location_splits(results), on="TeamID", how="left")

            print(f"Best win / worst loss judged by {rating_desc}.")
            rating_note = f"  |  Best win / worst loss by {rating_desc}"
            df = df.merge(best_win_worst_loss(results, ratings, d1_ids),
                          on="TeamID", how="left")

        for col in ["Conf W", "Conf L"]:
            df[col] = df[col].fillna(0).astype(int) if col in df else 0
        df["Conf Record"] = df["Conf W"].astype(str) + "-" + df["Conf L"].astype(str)
        df["Conf Win %"] = win_pct(df["Conf W"], df["Conf L"])
        for col in ["Best Win", "Worst Loss", "Home Rec", "Away Rec", "Neutral Rec"]:
            df[col] = df[col].fillna("0-0") if col in df and "Rec" in col else (
                      df[col].fillna("-") if col in df else ("0-0" if "Rec" in col else "-"))

    cols = ["Team", "Conference", "Record", "W", "L", "Win %",
            "Conf Record", "Conf W", "Conf L", "Conf Win %",
            "Home Rec", "Home +/-", "Away Rec", "Away +/-", "Neutral Rec", "Neutral +/-",
            "Best Win", "Worst Loss",
            "Adj Net", "Net Rk", "SRS", "Adj O", "O Rk", "Adj D", "D Rk", "Pace",
            "eFG%", "Opp eFG%", "TO%", "Opp TO%", "ORB%", "Opp ORB%",
            "Games", "Mascot", "TeamID"]
    df = df[[c for c in cols if c in df.columns]]

    build_workbook(df, conf_names, season, rating_note, schedule, ats, today, rating_desc)
    print(f"Saved {OUTPUT_FILE}: {len(df)} teams across {df['Conference'].nunique()} conferences.")


if __name__ == "__main__":
    main()
