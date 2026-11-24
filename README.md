# CBB Hype (cloud version)

Every morning at about 4 AM Eastern, GitHub runs the whole CBB Hype pipeline and publishes the
site. Your computer doesn't need to be on, and you can open the site on your phone.

```
records.py  ->  picks_engine.py  ->  torvik_pull.py  ->  methods_engine.py  ->  build_site.py
(API pull)      (grade/learn/pick)   (Torvik, no API)    (5 pick methods)       (the site)
```
`run_daily.py` runs these in order, the same way your notebook's `%run` lines do.

## One-time setup (about 15 minutes)

**1. Make the repository.** Sign in at github.com (create a free account if you need one).
Click **+** (top right), then **New repository**. Name it `cbb-hype` and choose **Public**.
Leave everything else unticked and click **Create repository**.

> Public is required for the free version of GitHub Pages. Anyone with the link can see the site
> and the code, but your API key stays hidden. A private repo with a site needs GitHub Pro
> (about $4 a month).

**2. Put in your latest data.** Before uploading, refresh the files in this folder's `seed/` with
the newest copies from PythonFun, so the cloud picks up exactly where your Mac left off:
- `games_log.csv` and `model_params.json`. These are what the model has learned. **Most important.**
- `college_basketball_by_conference.xlsx`
- the whole `torvik_snapshots` folder (its contents go into `seed/torvik_snapshots/`)

If you've used manual trend weights, also put `trend_settings.json` next to `README.md` (not in seed).

**3. Upload the files.** On the new repository's page, click **uploading an existing file**.
Drag in everything from this folder: all the `.py` files, `cbb_hype.html`, `requirements.txt`,
`README.md`, `daily.yml`, the rosters workbook, and the `seed` folder. Then click **Commit changes**.

**4. Add the schedule.** Click **Add file**, then **Create new file**. For the name, type exactly
`.github/workflows/daily.yml`. Open `daily.yml` from this folder in TextEdit, copy all of it, and
paste it into the big box. Click **Commit changes**.
(This step exists because Mac Finder hides folders whose names start with a dot.)

**5. Add your API key as a secret.** Go to **Settings**, then **Secrets and variables**, then **Actions**,
then **New repository secret**. Name: `CBBD_API_KEY`. Value: your CollegeBasketballData key. Click
**Add secret**. GitHub keeps it hidden, and it never appears in the code or the logs.

**6. Turn on the site.** Go to **Settings**, then **Pages**. Under **Source**, choose **GitHub Actions**.

**7. First run.** Go to **Actions**, then **CBB Hype daily update**, then **Run workflow**, keep
**Full update**, and click **Run workflow**. It takes about 3 to 5 minutes. When it's green,
your site is at:

    https://YOUR-GITHUB-USERNAME.github.io/cbb-hype/

Bookmark it, and add it to your phone's home screen.

**8. Turn off the Mac's daily update**, or both will pull every day and double your API use.
In a JupyterLab cell: `!python daily_update.py uninstall`

## Every day

- **4 AM ET:** it updates by itself. GitHub's timer is sometimes 10 to 30 minutes late.
- **Run it now** (from your phone too): use **Run now** in the site's top bar, or **Actions**,
  then **CBB Hype daily update**, then **Run workflow**. You need to be signed in to GitHub (the
  GitHub app works).
  - **Full update** pulls fresh data (about 20 API calls). If today's update already finished,
    it stops and tells you, unless you tick **Force**.
  - **Rebuild site only** re-runs picks, methods and the site on the saved data. It uses no API calls.
- The top bar shows when the site last updated and how many API calls you've used this month.
  The workbook can be downloaded from there too.
- Each run's **Summary** page lists today's picks and anything that went wrong. If a run fails,
  GitHub emails you.

## API credits

A full update uses about 20 calls. Once a day for a month is about 600 calls.
- Every CollegeBasketballData call is counted, by month.
- Scheduled runs skip the API pull once the month is within two updates of the limit, so a manual
  run is always left over. When that happens, the site is rebuilt from saved data and says so.
- The limit is 1,000 by default. To change it: **Settings**, then **Secrets and variables**, then
  **Actions**, then the **Variables** tab, then **New variable**. Name: `API_MONTHLY_LIMIT`.
- The Torvik pull uses Torvik's own site, not your API credits.

## Where things live

- **main branch:** the code, `cbb_hype.html`, the rosters workbook, `seed/` (starting data only).
- **data branch:** the saved data (workbook, `games_log.csv`, `model_params.json`, Torvik
  snapshots, API usage). Every run loads it, updates it and saves it back. Don't edit it by hand.
  Each run also keeps a 14-day backup under **Actions**, then the run, then **Artifacts**.
- **Changing code:** open a `.py` file on GitHub, click the pencil, edit, and commit. Or upload
  a replacement file. The next run uses it.
- **New rosters file or trend weights:** upload the new file over the old one on the main branch.

## Good to know

- **Torvik on cloud servers:** barttorvik.com sometimes blocks cloud computers. If it does, the
  run uses the latest saved snapshot and says so in the log, and the rest of the run carries on.
  Uploading your `torvik_snapshots` folder in step 2 gives it something to fall back on from day one.
- **Inactive repositories:** GitHub pauses scheduled runs on repositories with no activity for
  60 days. It emails you first, and one click under **Actions** turns it back on.
