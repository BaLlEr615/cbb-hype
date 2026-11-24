"""
Builds the CBB Hype site from the latest workbook.

GitHub version: writes site/index.html (what GitHub Pages serves) plus a copy of the
workbook (site/college_basketball_by_conference.xlsx) so you can download it from the
site. It also embeds a small "run info" block - when it last updated, API calls used
this month, and a link to run it now - that shows in the top bar of the page.

Run on your own computer too:  python build_site.py   (writes site/index.html)
"""
import base64
import json
import os
import shutil
from html import escape
from pathlib import Path

try:
    FOLDER = Path(__file__).resolve().parent
except NameError:            # pasted into a notebook cell
    FOLDER = Path.cwd()

TEMPLATE = FOLDER / "cbb_hype.html"
WORKBOOK = FOLDER / "college_basketball_by_conference.xlsx"
ROSTERS = FOLDER / "college_basketball_rosters_player_stats.xlsx"
SITE_DIR = FOLDER / "site"
OUTPUT = SITE_DIR / "index.html"
RUN_INFO = FOLDER / "run_info.json"          # written by run_daily.py just before this runs
ROSTER_COLS = {"Team": "team", "Player": "name", "#": "num", "Pos": "pos", "Class": "cls", "Ht": "ht"}


def roster_json():
    """Name / number / position / class / height for every player (no stats)."""
    if not ROSTERS.exists():
        return None
    import pandas as pd
    try:
        df = pd.read_excel(ROSTERS, sheet_name="All Players")
    except Exception as e:                       # wrong file or missing tab: build without rosters
        print(f"Skipping rosters ({e})")
        return None
    df = df[[c for c in ROSTER_COLS if c in df.columns]].rename(columns=ROSTER_COLS)
    df = df[df["team"].notna() & df["name"].notna()]
    if "num" in df:
        df["num"] = df["num"].map(lambda v: None if pd.isna(v) else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)))
    df = df.astype(object).where(df.notna(), None)
    modified = int(ROSTERS.stat().st_mtime * 1000)
    return json.dumps({"modified": modified, "players": df.to_dict("records")}, ensure_ascii=False)


def build_site():
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Put cbb_hype.html in {FOLDER}")
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"No workbook yet in {FOLDER} - run records.py first")
    page = TEMPLATE.read_text(encoding="utf-8")
    data = base64.b64encode(WORKBOOK.read_bytes()).decode("ascii")
    modified = int(WORKBOOK.stat().st_mtime * 1000)
    tag = (f'<script id="workbook-data" type="text/plain" data-name="{escape(WORKBOOK.name)}" '
           f'data-modified="{modified}">{data}</script>')
    rj = roster_json()
    if rj:
        tag += '\n<script id="roster-data" type="application/json">' + rj.replace("</", "<\\/") + "</script>"
    if RUN_INFO.exists():
        tag += '\n<script id="run-info" type="application/json">' + RUN_INFO.read_text(encoding="utf-8").replace("</", "<\\/") + "</script>"
    page = page.replace("<!--WORKBOOK-->", tag, 1)
    SITE_DIR.mkdir(exist_ok=True)
    OUTPUT.write_text(page, encoding="utf-8")
    shutil.copy2(WORKBOOK, SITE_DIR / WORKBOOK.name)
    (SITE_DIR / ".nojekyll").write_text("")
    print(f"Built {OUTPUT.relative_to(FOLDER)}" + (" (with rosters)" if rj else ""))


if __name__ == "__main__":
    build_site()
