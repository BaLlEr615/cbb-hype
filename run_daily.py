"""
CBB Hype daily runner - what GitHub runs every morning (you can also run it yourself).

    python run_daily.py                 # full update: API pull + picks + Torvik + methods + site
    python run_daily.py --rebuild       # no API calls: re-run picks/methods/site on saved data
    python run_daily.py --force         # run even if today's update already finished

Order (same as your notebook's %run lines):
    records.py  ->  picks_engine.py  ->  torvik_pull.py  ->  methods_engine.py  ->  build_site.py

Rules it follows so it never wastes API credits:
  * Scheduled runs wait until 4 AM Eastern, then run once per day. GitHub wakes it up a few
    times each morning (that's how it stays at 4 AM through daylight saving time); every
    wake-up after a good run just stops.
  * If the API pull fails, later wake-ups that morning try again, up to 3 tries a day.
  * Every call to CollegeBasketballData.com is counted (api_usage.json). Scheduled runs skip
    the API pull once the month is within one update of your limit, so a manual run is
    always left over. Change the limit with the API_MONTHLY_LIMIT setting (see README).
  * A manual full update on a day that already updated stops and tells you, unless you
    tick "force".
"""
import argparse
import gzip
import json
import os
import runpy
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
SEED = HERE / "seed"
STATE_OUT = HERE / "_state"
USAGE_FILE = HERE / "api_usage.json"
RUN_STATE = HERE / "run_state.json"
RUN_INFO = HERE / "run_info.json"
WORKBOOK = HERE / "college_basketball_by_conference.xlsx"

START_HOUR_ET = 4                    # the daily update runs at/after 4 AM Eastern
MAX_TRIES_PER_DAY = 3
CALLS_PER_UPDATE = 25                # a full update uses ~20; keep a little margin
MONTHLY_LIMIT = int(os.environ.get("API_MONTHLY_LIMIT") or 1000)

# Files that carry over from one run to the next (saved on the "data" branch).
# (The rosters file and trend_settings.json live with the code instead, so you can replace
#  them by uploading a new copy.)
STATE_FILES = [
    "college_basketball_by_conference.xlsx",
    "games_log.csv", "games_log.csv.bak", "model_params.json",
    "api_usage.json", "run_state.json",
]
STATE_DIRS = ["torvik_snapshots", "torvik_history"]

STEPS = [
    ("records", "records.py", "Pull from the API and build the workbook"),
    ("picks", "picks_engine.py", "Pick engine: grade, learn, pick"),
    ("torvik", "torvik_pull.py", "Torvik ratings and game projections"),
    ("methods", "methods_engine.py", "Pick methods engine"),
    ("site", "build_site.py", "Build the site"),
]


# ----------------------------------------------------------------------
def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def gh_output(**kv):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            for k, v in kv.items():
                f.write(f"{k}={v}\n")


def summary(md):
    print(md)
    out = os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(md + "\n")


def seed_missing():
    """First run: copy your starting files from seed/ for anything not saved yet."""
    if not SEED.exists():
        return
    for src in SEED.rglob("*"):
        if src.is_file() and src.name != "README.md":
            dst = HERE / src.relative_to(SEED)
            if not dst.exists() and not Path(str(dst) + ".gz").exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  seeded {dst.relative_to(HERE)}")


# ----------------------------------------------------------------------
# API call counter (wraps requests so every CollegeBasketballData call is counted)
# ----------------------------------------------------------------------
class Counter:
    def __init__(self):
        self.calls = 0
        self.headers = {}

    def install(self):
        import requests
        orig = requests.Session.send
        counter = self

        def send(sess, request, **kw):
            resp = orig(sess, request, **kw)
            if "collegebasketballdata.com" in (request.url or ""):
                counter.calls += 1
                for k, v in resp.headers.items():
                    if any(w in k.lower() for w in ("limit", "remaining", "quota", "calls")):
                        counter.headers[k] = v
            return resp
        requests.Session.send = send


def month_key(now):
    return now.strftime("%Y-%m")


def usage_this_month(usage, now):
    return int(usage.get("months", {}).get(month_key(now), {}).get("calls", 0))


def add_usage(usage, now, calls, headers):
    m = usage.setdefault("months", {}).setdefault(month_key(now), {"calls": 0, "runs": 0})
    m["calls"] = int(m.get("calls", 0)) + calls
    m["runs"] = int(m.get("runs", 0)) + 1
    if headers:
        usage["last_headers"] = headers
    usage["updated"] = now.isoformat(timespec="minutes")


# ----------------------------------------------------------------------
def run_step(name, script):
    """Run one script as if from the command line. Returns (ok, message)."""
    path = HERE / script
    print(f"\n========== {name}: {script} ==========", flush=True)
    t0 = time.time()
    try:
        runpy.run_path(str(path), run_name="__main__")
        ok, msg = True, ""
    except SystemExit as e:
        ok = e.code in (None, 0)
        msg = "" if ok else str(e.code)
    except Exception as e:                      # keep going; report it
        traceback.print_exc()
        ok, msg = False, f"{type(e).__name__}: {e}"
    print(f"---------- {name}: {'ok' if ok else 'FAILED'} in {time.time() - t0:.0f}s ----------", flush=True)
    return ok, msg


def compress_snapshots():
    """Torvik snapshots add up over a season; gzip them (the Torvik pull can read .gz)."""
    d = HERE / "torvik_snapshots"
    if not d.exists():
        return
    for f in d.iterdir():
        if f.is_file() and f.suffix != ".gz":
            with open(f, "rb") as src, gzip.open(str(f) + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            f.unlink()


def export_state():
    """Copy everything that must carry over to _state/ (the workflow saves it to the data branch)."""
    if STATE_OUT.exists():
        shutil.rmtree(STATE_OUT)
    STATE_OUT.mkdir()
    for name in STATE_FILES:
        if (HERE / name).exists():
            shutil.copy2(HERE / name, STATE_OUT / name)
    for d in STATE_DIRS:
        if (HERE / d).exists():
            shutil.copytree(HERE / d, STATE_OUT / d)
    (STATE_OUT / "ABOUT_THIS_BRANCH.md").write_text(
        "Saved data for CBB Hype, replaced by every daily run. Don't edit this branch by hand;\n"
        "to change a file, put the new copy in seed/ on the main branch and delete it here.\n")


def picks_summary():
    """A few lines about today's picks for the run summary (read from the workbook)."""
    try:
        import pandas as pd
        today = datetime.now(ET).date().isoformat()
        lines = []
        try:
            mp = pd.read_excel(WORKBOOK, sheet_name="Method Picks")
            mp = mp[(mp["Date"].astype(str).str[:10] == today) & mp["Tier"].fillna("").astype(str).ne("")]
            if len(mp):
                counts = mp.groupby("Method").size().to_dict()
                lines.append("Picks today by method: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        except Exception:
            pass
        try:
            p = pd.read_excel(WORKBOOK, sheet_name="Picks")
            p = p[p["Date"].astype(str).str[:10] == today]
            for _, r in p.head(10).iterrows():
                lines.append(f"- {r.get('Tier', '')} **{r['Pick']}** ({r['Away']} @ {r['Home']}) {float(r['Confidence']):.0%}")
        except Exception:
            pass
        return "\n".join(lines)
    except Exception:
        return ""


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="no API calls; re-run picks/methods/site on saved data")
    ap.add_argument("--force", action="store_true", help="run even if today's update already finished")
    args = ap.parse_args()

    os.chdir(HERE)
    sys.path.insert(0, str(HERE))
    now = datetime.now(ET)
    today = now.date().isoformat()
    scheduled = os.environ.get("GITHUB_EVENT_NAME") == "schedule"
    seed_missing()

    state = load_json(RUN_STATE, {})
    if state.get("date") != today:
        state = {"date": today, "status": "pending", "attempts": 0}
    usage = load_json(USAGE_FILE, {})
    used = usage_this_month(usage, now)
    mode = "rebuild" if args.rebuild else "full"
    problem = ""

    # ---- should this wake-up do anything?
    if scheduled:
        if now.hour < START_HOUR_ET:
            summary(f"Woke up at {now:%I:%M %p} ET - too early, the update runs at {START_HOUR_ET} AM ET.")
            gh_output(did_run="false", deploy="false", failed="false")
            return 0
        if state["status"] == "ok":
            summary(f"Already updated today ({state.get('finished', '')}). Nothing to do.")
            gh_output(did_run="false", deploy="false", failed="false")
            return 0
        if state["attempts"] >= MAX_TRIES_PER_DAY:
            summary(f"The update failed {state['attempts']} times today; not trying again until tomorrow. Check earlier runs' logs.")
            gh_output(did_run="false", deploy="false", failed="false")
            return 0
    elif mode == "full" and state["status"] == "ok" and not args.force:
        summary(f"**Today's update already finished** ({state.get('finished', '')}), so this run stopped "
                f"to save API calls ({used} used this month).\n\nTo pull again anyway, run it with "
                "**Force** ticked. To refresh the site without using any calls, choose **Rebuild site only**.")
        gh_output(did_run="false", deploy="false", failed="false")
        return 0

    if mode == "full" and used + CALLS_PER_UPDATE > MONTHLY_LIMIT:
        if scheduled or not args.force:
            mode = "rebuild"
            problem = (f"API pull skipped: {used} of {MONTHLY_LIMIT} calls used this month "
                       f"(an update needs ~{CALLS_PER_UPDATE}).")
            if not scheduled:
                problem += " Tick Force to pull anyway."
    elif mode == "full" and scheduled and used + 2 * CALLS_PER_UPDATE > MONTHLY_LIMIT:
        mode = "rebuild"
        problem = (f"Scheduled API pull skipped to keep one manual update in reserve "
                   f"({used} of {MONTHLY_LIMIT} calls used this month).")

    if mode == "full" and not os.environ.get("CBBD_API_KEY"):
        mode = "rebuild"
        problem = "No API key found - add the CBBD_API_KEY secret (see README). Rebuilt from saved data."

    if not WORKBOOK.exists() and mode == "rebuild":
        summary("No workbook saved yet and no API pull this run, so there's nothing to build. " + problem)
        gh_output(did_run="false", deploy="false", failed="true")
        return 1

    # ---- run
    print(f"CBB Hype update - {now:%a %b %d %I:%M %p} ET - mode: {mode}" + (f" ({problem})" if problem else ""))
    counter = Counter()
    counter.install()
    results = {}
    if mode == "full":
        state["attempts"] += 1
        before = WORKBOOK.stat().st_mtime if WORKBOOK.exists() else 0
        ok, msg = run_step(*STEPS[0][:2])
        pulled = WORKBOOK.exists() and WORKBOOK.stat().st_mtime > before
        results["records"] = (ok and pulled, msg or ("" if pulled else "workbook wasn't written"))
        if not results["records"][0]:
            problem = f"API pull failed ({results['records'][1] or 'see log'}); showing the last saved data."
            if not WORKBOOK.exists():
                add_usage(usage, now, counter.calls, counter.headers)
                save_json(USAGE_FILE, usage)
                state["status"] = "failed"
                save_json(RUN_STATE, state)
                export_state()
                summary(f"**Update failed** and there's no saved workbook to fall back on. {problem}")
                gh_output(did_run="true", deploy="false", failed="true")
                return 1

    for name, script, _ in STEPS[1:4]:
        results[name] = run_step(name, script)
    compress_snapshots()

    # ---- bookkeeping, then the site (so the page shows this run's numbers)
    add_usage(usage, now, counter.calls, counter.headers)
    save_json(USAGE_FILE, usage)
    full_ok = mode == "full" and results["records"][0]
    if full_ok:
        state.update(status="ok", finished=datetime.now(ET).strftime("%I:%M %p ET").lstrip("0"))
    elif mode == "full":
        state["status"] = "failed"
    elif scheduled:      # limit reached: rebuilt without a pull; don't repeat on the next wake-up
        state.update(status="ok", finished=datetime.now(ET).strftime("%I:%M %p ET").lstrip("0") + " (no API pull)")
    save_json(RUN_STATE, state)

    failed_steps = [n for n, (ok, _) in results.items() if not ok]
    if failed_steps and not problem:
        problem = "These steps had errors: " + ", ".join(failed_steps) + " (see the run log)."
    server, repo, run_id = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    save_json(RUN_INFO, {
        "finished": datetime.now(ET).isoformat(timespec="minutes"), "mode": mode,
        "api_month": usage_this_month(usage, now), "api_limit": MONTHLY_LIMIT, "api_this_run": counter.calls,
        "problem": problem,
        "run_url": f"{server}/{repo}/actions/workflows/daily.yml" if server and repo else None,
        "log_url": f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else None,
    })
    results["site"] = run_step("site", "build_site.py")
    export_state()

    # ---- report
    icon = lambda ok: "✅" if ok else "❌"
    md = [f"## CBB Hype update - {now:%a %b %d}, {now:%I:%M %p} ET",
          f"Mode: **{'full update' if mode == 'full' else 'rebuild only (no API pull)'}**"
          + (f" · try {state['attempts']} of {MAX_TRIES_PER_DAY}" if mode == "full" else ""),
          f"API calls: **{counter.calls}** this run · **{usage_this_month(usage, now)} / {MONTHLY_LIMIT}** this month",
          ""]
    md += [f"- {icon(results[n][0])} {label}" + (f" - {results[n][1]}" if results[n][1] else "")
           for n, _, label in STEPS if n in results]
    if problem:
        md += ["", f"⚠ {problem}"]
    ps = picks_summary()
    if ps:
        md += ["", ps]
    summary("\n".join(md))

    site_ok = results["site"][0]
    failed = (mode == "full" and not full_ok) or not site_ok
    gh_output(did_run="true", deploy="true" if site_ok else "false", failed="true" if failed else "false")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
