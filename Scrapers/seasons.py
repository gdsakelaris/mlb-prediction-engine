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
import shutil
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


def _replace_retry(tmp, path):
    """os.replace that retries through transient PermissionError because
    Data/ lives under OneDrive, whose sync client briefly locks files it
    is uploading. Shared by atomic_write and atomic_append."""
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.3 * (attempt + 1))


@contextmanager
def atomic_write(path, mode="w", **open_kwargs):
    """Crash-safe file rewrite: write to <name>.tmp, then os.replace over
    the target on clean exit — a crash, kill, or power loss mid-write can
    never leave a truncated warehouse CSV (the pattern the odds store and
    pitch archive already used; this makes it the shared default for
    every scraper's in-place rewrite). On an exception the tmp file is
    removed and the original is untouched."""
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
    _replace_retry(tmp, path)


def atomic_append(path, text, encoding="utf-8"):
    """Crash-safe append for the append-only archives (mlb_pbp.csv, the
    MiLB game logs): copy the file to <name>.tmp, append the batch there,
    then os.replace over the original. A kill mid-write leaves the
    published file exactly as it was, so it always ends on a complete
    batch boundary and a resume-by-GamePk cache can never mark a
    half-written game as done — plain append mode could be cut mid-batch,
    leaving a torn final line plus a partially-written game the resume
    logic then skipped forever. Same OneDrive replace retry as
    atomic_write."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        if path.exists():
            shutil.copyfile(path, tmp)
        with open(tmp, "a", newline="", encoding=encoding) as f:
            f.write(text)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    _replace_retry(tmp, path)


def trim_torn_tail(path):
    """Repair an append-only CSV left torn by a mid-append kill under the
    old plain-append code: a file that does not end with a newline holds
    a torn final line, and the physically-last game block is a partial
    write (batches append each game's rows contiguously, so only the
    final block can be incomplete). Truncates the torn fragment PLUS the
    whole final contiguous block of the last GamePk (first CSV field), so
    the caller's resume cache no longer sees those games and refetches
    them complete. Cheap (tail read + truncate); a no-op on a clean file,
    and permanently a no-op once every append goes through
    atomic_append."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "r+b") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return                          # clean tail: nothing torn
        win = 1 << 20
        while True:
            read_from = max(0, size - win)
            f.seek(read_from)
            tail = f.read(size - read_from)
            head = 0 if read_from == 0 else tail.find(b"\n") + 1
            # (absolute offset, first CSV field) of each line in view
            recs, pos = [], head
            for ln in tail[head:].split(b"\n"):
                if ln:
                    fld = ln.split(b",", 1)[0]
                    # utf-8-sig files BOM-prefix the header's first
                    # field; normalize it or the header guard below
                    # never matches and a BOM-header + torn-fragment
                    # file is truncated to 0 bytes (and the BOM char
                    # crashes the cp1252 log print)
                    if read_from + pos == 0 and fld[:3] == b"\xef\xbb\xbf":
                        fld = fld[3:]
                    recs.append((read_from + pos, fld))
                pos += len(ln) + 1
            if not recs:
                return
            # strip the torn fragment and the last complete line's whole
            # game block (the fragment may be cut mid-GamePk, so the two
            # first-fields can differ; both games get refetched)
            strip = {recs[-1][1]}
            if len(recs) >= 2:
                strip.add(recs[-2][1])
            strip.discard(b"GamePk")        # never eat the header line
            cut = None
            for off, fld in reversed(recs):
                if fld not in strip:
                    break
                cut = off
            if cut is not None and cut == recs[0][0] and read_from > 0:
                win *= 2                    # block may start before view
                continue
            break
        if cut is None:
            return
        f.truncate(cut)
    pks = b" / ".join(sorted(strip)).decode("utf-8", errors="replace")
    print(f"    {path.name}: trimmed torn tail from a killed append "
          f"(game {pks} will be refetched)", flush=True)


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
