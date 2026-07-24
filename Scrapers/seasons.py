"""Single source of truth for which MLB seasons the project covers.

Scrapers import YEARS / CURRENT_SEASON from here instead of hardcoding a
range, so the annual rollover (2026 -> 2027 -> ...) needs no code edits:
from March 1 the new calendar year becomes the current season and every
scraper starts fetching it alongside the stored history.

Also provides the stored-season cache helper the per-year scrapers share:
a completed season's stats never change once scraped, so a scraper only
needs to hit the network for the current season. That both speeds the
daily job and removes the failure mode where an upstream hiccup on a
HISTORICAL fetch (e.g. Savant throttling a years-old leaderboard page)
kills the whole job.
"""

import csv
import os
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

# 2015 = first Statcast season. Sources that start later (OAA 2016,
# arsenals ~2017, bat tracking 2023) are simply empty for the early years.
FIRST_SEASON = 2015


def current_season(today=None):
    """The season scrapers treat as in progress.

    January/February belong to the PREVIOUS season — no games have been
    played and the new year's leaderboards are empty. From March 1 the
    new season exists for scraping purposes (Opening Day is late March;
    empty early fetches are cheap and harmless)."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def years(today=None):
    """Every season the project covers, oldest first."""
    return range(FIRST_SEASON, current_season(today) + 1)


YEARS = years()
CURRENT_SEASON = current_season()


@contextmanager
def atomic_write(path, mode="w", **open_kwargs):
    """Crash-safe file rewrite: write to <name>.tmp, then os.replace over
    the target on clean exit — a crash, kill, or power loss mid-write can
    never leave a truncated warehouse CSV (the pattern the odds store and
    pitch archive already used; this makes it the shared default for
    every scraper's in-place rewrite). On an exception the tmp file is
    removed and the original is untouched. The final replace retries
    through transient PermissionError because Data/ lives under OneDrive,
    whose sync client briefly locks files it is uploading."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    f = open(tmp, mode, **open_kwargs)
    try:
        yield f
    except BaseException:
        f.close()
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    f.close()
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.3 * (attempt + 1))


def stored_rows_by_season(csv_path, season_col):
    """Rows of an existing combined CSV grouped {season: [dict, ...]}, or
    {} if the file doesn't exist. Completed seasons are served straight
    from this cache; the current season's rows are the stale fallback
    when its fetch fails (better a day-old snapshot than a dead job)."""
    path = Path(csv_path)
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                yr = int(row[season_col])
            except (KeyError, TypeError, ValueError):
                continue
            out.setdefault(yr, []).append(row)
    return out
