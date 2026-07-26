"""Scrape Statcast batted-ball data (every ball in play) from Baseball Savant.

The HR log covers only home runs — a sample censored to each batter's best
contact. This pulls EVERY tracked ball in play (2015 on, regular season +
postseason — one engine, postseason rows are ordinary evidence) from the
public statcast_search CSV endpoint (the same service behind
baseballsavant.mlb.com/statcast_search): exit velocity, launch angle, barrel
classification, expected stats on contact (xBA/xwOBA), batted-ball type, and
hit distance, one row per batted ball.

Why it matters for the model: contact-quality ("process") stats stabilize
far faster than outcome stats — exit velo is reliable in ~40 batted balls vs
hundreds of AB for batting average — so they detect real skill and real
in-season change earlier than anything in the box scores. The pitcher side
(contact quality ALLOWED) is equally informative.

Relational keys: BatterId/PitcherId are MLBAM ids matching PlayerId in every
other CSV; GamePk matches the game logs. (GamePk, AtBat, PitchNum) uniquely
identifies a batted ball.

Default run is incremental — the output CSV doubles as the cache: stored
seasons are reused, and only dates after the newest stored row (minus a
2-day refetch window for Statcast's own corrections) plus any season missing
from the file are downloaded. Seconds in the daily job. --backfill ignores
the cache and rescrapes all seasons (~10 minutes, ~730k rows); only needed
when the file itself is suspect. --postseason-backfill is the one-time
repair for seasons stored before this scraper fetched postseason rounds.

Usage:
    python scrape_statcast.py [-o output.csv] [--backfill]
                              [--postseason-backfill]
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from seasons import YEARS, atomic_write

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_statcast_bip.csv"

API_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# output column -> Savant CSV field
COLUMNS = {
    "GamePk": "game_pk",
    "Season": None,               # derived from Date
    "Date": "game_date",
    "BatterId": "batter",
    "PitcherId": "pitcher",
    "Stand": "stand",
    "PThrows": "p_throws",
    "Events": "events",
    "BBType": "bb_type",
    "ExitVelo": "launch_speed",
    "LaunchAngle": "launch_angle",
    "LSA": "launch_speed_angle",  # Savant contact code; 6 = barrel
    "xBA": "estimated_ba_using_speedangle",
    "xwOBA": "estimated_woba_using_speedangle",
    "HitDistance": "hit_distance_sc",
    "HcX": "hc_x",              # hit coordinates -> spray direction
    "HcY": "hc_y",
    "AtBat": "at_bat_number",
    "PitchNum": "pitch_number",
}
KEY = ["GamePk", "AtBat", "PitchNum"]

CHUNK_DAYS = 14        # ~10k rows in-season; well under Savant's result cap
CAP_ROWS = 24000       # a chunk this big was probably truncated -> split it
SLEEP = 2.0            # politeness between requests
REFETCH_DAYS = 2       # re-pull the newest days: Savant back-corrects data

# suspended-game completeness re-check (find_stub_dates, shared with
# scrape_pitches): Savant stamps EVERY pitch of a suspended game with the
# ORIGINAL game_date, so a completion played more than REFETCH_DAYS after
# the suspension sits behind the incremental watermark and is never
# fetched without a targeted re-check.
GAMES_CSV = DATA_DIR / "mlb_games.csv"   # finals only (scrape_gamelogs)
STUB_WINDOW_DAYS = 30  # re-check horizon: cross-series resumptions can
#                        land weeks after the original date; the memo
#                        below makes the wide window cost one refetch
#                        per short final, not one per day
STUB_FLOOR_BIP = 25    # final game with fewer stored BIP = partial ingest
# memo of already-refetched low-count dates (per scraper scope): a
# flagged date whose per-game counts are UNCHANGED since its last
# successful refetch is settled — a rain-shortened 5-inning final sits
# below the floor legitimately and must not trigger a full-day Savant
# pull every run for the whole window. Written ONLY after a successful
# refetch+merge (record_stub_state), so a crashed refetch retries.
STUB_MEMO = DATA_DIR / ".stub_checked.json"

# postseason rows are ordinary evidence (one engine, no October fork) —
# same game-type universe as scrape_pitches / the gamelogs
GT_ALL = "R|F|D|L|W|"        # regular + the four postseason rounds
GT_POST = "F|D|L|W|"         # postseason only (targeted backfill)


def fetch_range(d0, d1, tries=3, gt=GT_ALL):
    """One CSV request for batted balls in [d0, d1] (regular + postseason).
    Returns a raw Savant DataFrame; recursively splits ranges that hit the
    result cap so nothing is silently truncated."""
    params = {
        "all": "true", "type": "details", "player_type": "batter",
        "hfBBT": "fly_ball|ground_ball|line_drive|popup|",
        "hfGT": gt, "minors": "false",
        "game_date_gt": str(d0), "game_date_lt": str(d1),
    }
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=180)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            break
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    retry {d0}..{d1} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    if len(df) >= CAP_ROWS and d0 != d1:
        mid = d0 + (d1 - d0) / 2
        print(f"    {d0}..{d1}: {len(df):,} rows (cap?) -> splitting",
              flush=True)
        time.sleep(SLEEP)
        left = fetch_range(d0, mid, gt=gt)
        time.sleep(SLEEP)
        right = fetch_range(mid + timedelta(days=1), d1, gt=gt)
        return pd.concat([left, right], ignore_index=True)
    return df


def require_source_columns(raw, cols, context):
    """Fail CLOSED on upstream schema drift: a non-empty Savant payload
    missing a mapped source column means an upstream rename/drop, and
    silently filling with NA would poison the store for months before
    validate_data's cumulative NaN tripwires fire. Raising marks the
    scraper FAILED in the 6AM job and the watchdog alerts.
    SAVANT_SCHEMA_LAX=1 degrades to a loud stderr warning (missing
    values become NA) for deliberate operator runs. Shared with
    scrape_pitches."""
    missing = [c for c in cols if c not in raw.columns]
    if not missing:
        return
    msg = (f"{context}: Savant payload is missing mapped source "
           f"column(s) {missing} — upstream schema drift? Set "
           f"SAVANT_SCHEMA_LAX=1 to run anyway (missing values "
           f"become NA).")
    if os.environ.get("SAVANT_SCHEMA_LAX") == "1":
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)
        return
    raise RuntimeError(msg)


def to_schema(raw):
    """Savant frame -> our column schema (empty frame if no rows).
    Fails closed when a mapped source column is missing (schema drift);
    SAVANT_SCHEMA_LAX=1 degrades to a warning + NA fill."""
    if raw.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    require_source_columns(raw,
                           [s for s in COLUMNS.values() if s is not None],
                           "scrape_statcast.to_schema")
    out = pd.DataFrame()
    for col, src in COLUMNS.items():
        if src is None:
            continue
        out[col] = raw[src] if src in raw.columns else pd.NA
    d = pd.to_datetime(out["Date"])
    out["Date"] = d.dt.date
    out["Season"] = d.dt.year
    return out[list(COLUMNS)]


def _stub_counts(events, games_path, window_days, until):
    """Shared prep for find_stub_dates/record_stub_state: (finals frame
    clipped to the window, per-(Date, GamePk) stored-row counts, lo).
    finals is None when there is nothing to check."""
    games_path = Path(games_path) if games_path is not None else GAMES_CSV
    if not games_path.exists():
        return None, None, None
    finals = pd.read_csv(games_path, usecols=["GamePk", "Date"],
                         encoding="utf-8-sig")
    finals["Date"] = pd.to_datetime(finals["Date"]).dt.date
    finals["GamePk"] = pd.to_numeric(finals["GamePk"], errors="coerce")
    finals = finals.dropna().astype({"GamePk": "int64"})
    lo = date.today() - timedelta(days=window_days)
    hi = until if until is not None else date.today() + timedelta(days=1)
    finals = finals[(finals["Date"] >= lo) & (finals["Date"] < hi)]
    if finals.empty:
        return None, None, lo
    ev = pd.DataFrame({
        "Date": events["Date"],
        "GamePk": pd.to_numeric(events["GamePk"], errors="coerce")})
    ev = ev.dropna().astype({"GamePk": "int64"})
    ev = ev[ev["Date"] >= lo]
    counts = ev.groupby(["Date", "GamePk"]).size() if len(ev) else None
    return finals, counts, lo


def _date_counts(counts, finals, d):
    """{str(GamePk): stored rows} for date d's FINAL games (0-filled) —
    the memo signature find_stub_dates compares against."""
    pks = finals.loc[finals["Date"] == d, "GamePk"]
    if counts is None:
        return {str(int(pk)): 0 for pk in pks}
    return {str(int(pk)): int(counts.get((d, int(pk)), 0)) for pk in pks}


def _load_stub_memo(memo_path=None):
    p = Path(memo_path) if memo_path is not None else STUB_MEMO
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_RESUMED_CACHE = {}


def _resumed_finals(lo, hi):
    """ORIGINAL dates of suspended games whose completion is now FINAL,
    from one keyless StatsAPI schedule call. The count floor cannot see
    a late suspension (a 7th-inning stub holds ~220 pitches, well above
    the floor), and Savant stamps the completion's pitches with the
    original date — so those dates must be flagged by schedule status,
    not by counts. Failure degrades to count-floor-only, loudly."""
    key = (lo, hi)
    if key in _RESUMED_CACHE:
        return _RESUMED_CACHE[key]
    dates = set()
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "startDate": lo.isoformat(),
                    "endDate": hi.isoformat()},
            timeout=30)
        r.raise_for_status()
        for day in r.json().get("dates", []):
            for g in day.get("games", []):
                if (g.get("status") or {}).get("codedGameState") != "F":
                    continue
                # the completion record carries resumedFrom (points at
                # the original date); the original-date record carries
                # resumeDate — either way the ORIGINAL date is the one
                # whose Savant rows are incomplete
                rf = str(g.get("resumedFrom") or "")
                if rf:
                    try:
                        dates.add(date.fromisoformat(rf[:10]))
                    except ValueError:
                        pass
                elif g.get("resumeDate") or g.get("resumeGameDate"):
                    try:
                        dates.add(date.fromisoformat(
                            str(day.get("date"))[:10]))
                    except ValueError:
                        pass
    except Exception as ex:                          # noqa: BLE001
        print(f"  resumed-game probe failed ({ex}); count-floor check "
              f"only this run", file=sys.stderr)
    _RESUMED_CACHE[key] = dates
    return dates


def find_stub_dates(events, floor, until=None, games_path=None,
                    window_days=STUB_WINDOW_DAYS, scope="bip",
                    memo_path=None, probe=None):
    """Dates in the last `window_days` days whose stored rows undercount
    a FINAL game (suspended-game signature: the completion's pitches
    carry the ORIGINAL game_date, already behind the watermark) — plus
    dates the StatsAPI schedule marks as resumed-and-completed, which a
    count floor alone cannot see. Dates whose per-game counts are
    unchanged since their last successful refetch (memo, see
    record_stub_state) are settled and skipped, so a legitimately short
    rain-shortened final costs ONE refetch, not one per day.

    events: frame with Date (datetime.date) + GamePk columns, one row
    per stored pitch/batted ball. until: the incremental watermark
    start — dates at/after it were just fully refetched. probe: tests
    inject a set of resumed dates; None = live schedule probe, run only
    with the production games file (tests pass games_path). Returns
    sorted stub dates ([] when clean). Shared with scrape_pitches."""
    finals, counts, lo = _stub_counts(events, games_path, window_days,
                                      until)
    if finals is None:
        return []
    got = counts.reindex(
        pd.MultiIndex.from_frame(finals[["Date", "GamePk"]]),
        fill_value=0).to_numpy() if counts is not None else \
        pd.Series(0, index=range(len(finals))).to_numpy()
    low = set(finals.loc[got < floor, "Date"])
    if probe is None:
        probe = (_resumed_finals(lo, date.today())
                 if games_path is None else set())
    low |= {d for d in probe if (finals["Date"] == d).any()}
    if not low:
        return []
    memo = _load_stub_memo(memo_path).get(scope, {})
    return sorted(d for d in low
                  if memo.get(str(d)) != _date_counts(counts, finals, d))


def record_stub_state(events, floor, scope, games_path=None,
                      window_days=STUB_WINDOW_DAYS, memo_path=None,
                      probe=None):
    """Write the memo AFTER a successful refetch+merge: per-game counts
    of every still-low or resumed-flagged date in the window. Tomorrow's
    find_stub_dates skips dates whose counts did not change; a crashed
    run never reaches this call, so its dates retry. Replacing the
    scope's dict wholesale prunes dates that aged out or healed."""
    finals, counts, lo = _stub_counts(events, games_path, window_days,
                                      until=None)
    memo = _load_stub_memo(memo_path)
    if finals is None:
        memo[scope] = {}
    else:
        got = counts.reindex(
            pd.MultiIndex.from_frame(finals[["Date", "GamePk"]]),
            fill_value=0).to_numpy() if counts is not None else \
            pd.Series(0, index=range(len(finals))).to_numpy()
        low = set(finals.loc[got < floor, "Date"])
        if probe is None:
            probe = (_resumed_finals(lo, date.today())
                     if games_path is None else set())
        low |= {d for d in probe if (finals["Date"] == d).any()}
        memo[scope] = {str(d): _date_counts(counts, finals, d)
                       for d in low}
    p = Path(memo_path) if memo_path is not None else STUB_MEMO
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(memo, indent=1, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, p)


def season_windows(year, start=None):
    """[(d0, d1), ...] CHUNK_DAYS-sized windows covering the season's
    possible dates (Mar 1 - Nov 30; empty off-season chunks are cheap),
    optionally clipped to begin at `start`."""
    d0 = date(year, 3, 1) if start is None else max(start, date(year, 3, 1))
    end = min(date(year, 11, 30), date.today())
    windows = []
    while d0 <= end:
        d1 = min(d0 + timedelta(days=CHUNK_DAYS - 1), end)
        windows.append((d0, d1))
        d0 = d1 + timedelta(days=1)
    return windows


def postseason_backfill(out_path):
    """One-time targeted pull of postseason batted balls (hfGT F|D|L|W)
    for every season, merged into the output CSV (mirrors
    scrape_pitches.postseason_backfill). The incremental daily run now
    fetches all game types, but seasons already stored were scraped
    regular-season only and are never re-downloaded — this fills their
    October/November rows. Regular-season rows are untouched; safe to
    re-run."""
    existing = None
    if out_path.exists():
        existing = pd.read_csv(out_path, encoding="utf-8-sig",
                               low_memory=False)
        existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
    frames = []
    for year in YEARS:
        got = 0
        d0 = date(year, 9, 20)
        end = min(date(year, 11, 30), date.today())
        while d0 <= end:
            d1 = min(d0 + timedelta(days=CHUNK_DAYS - 1), end)
            df = to_schema(fetch_range(d0, d1, gt=GT_POST))
            got += len(df)
            if len(df):
                frames.append(df)
            time.sleep(SLEEP)
            d0 = d1 + timedelta(days=1)
        print(f"{year}: {got:,} postseason batted balls", flush=True)
    new = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=list(COLUMNS))
    if existing is not None:
        new = pd.concat([existing, new], ignore_index=True)
    new = (new.drop_duplicates(KEY, keep="last")
           .sort_values(["Date", "GamePk", "AtBat", "PitchNum"]))
    out_path.parent.mkdir(exist_ok=True)
    with atomic_write(out_path, "w", newline="", encoding="utf-8-sig") as f:
        new.to_csv(f, index=False)
    print(f"wrote {len(new):,} rows -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="rescrape all seasons from scratch (default run is "
                         "incremental from the newest stored date)")
    ap.add_argument("--postseason-backfill", action="store_true",
                    dest="post_backfill",
                    help="one-time pull of postseason batted balls for every "
                         "season already stored; safe to re-run")
    args = ap.parse_args()
    out_path = Path(args.output)
    if args.post_backfill:
        postseason_backfill(out_path)
        return

    # The output CSV is the cache: stored rows are reused, so the default
    # run only refetches (a) the newest REFETCH_DAYS (Savant back-corrects
    # recent games) and (b) any whole season somehow missing from the file.
    # Completed seasons already stored are never re-downloaded; --backfill
    # is the escape hatch when the file itself is suspect.
    existing = None
    start = None
    have = set()
    if out_path.exists() and not args.backfill:
        existing = pd.read_csv(out_path, encoding="utf-8-sig",
                               low_memory=False)
        existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
        have = set(pd.to_numeric(existing["Season"],
                                 errors="coerce").dropna().astype(int))
        newest = existing["Date"].max()
        start = newest - timedelta(days=REFETCH_DAYS)
        existing = existing[existing["Date"] < start]
        print(f"incremental: {len(existing):,} rows kept, refetching from "
              f"{start}", flush=True)

    frames = []
    for year in YEARS:
        if start is not None and year < start.year and year in have:
            continue                    # completed season already stored
        clip = start if start is not None and year >= start.year else None
        windows = season_windows(year, clip)
        got = 0
        for d0, d1 in windows:
            raw = fetch_range(d0, d1)
            df = to_schema(raw)
            got += len(df)
            if len(df):
                frames.append(df)
            time.sleep(SLEEP)
        print(f"{year}: {got:,} batted balls", flush=True)

    new = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=list(COLUMNS))
    if existing is not None:
        new = pd.concat([existing, new], ignore_index=True)
    n0 = len(new)
    new = new.drop_duplicates(KEY, keep="last")
    if len(new) < n0:
        print(f"dropped {n0 - len(new):,} duplicate rows on {KEY}", flush=True)
    # suspended-game completeness re-check (incremental runs only — a
    # backfill refetches every date anyway): refetching a flagged date
    # is idempotent because the KEY dedupe keeps the refetched rows
    if start is not None:
        for d in find_stub_dates(new, floor=STUB_FLOOR_BIP, until=start,
                                 scope="bip"):
            print(f"    completeness: refetching {d} (final game below "
                  f"{STUB_FLOOR_BIP} stored batted balls, or resumed "
                  f"per schedule)", flush=True)
            df = to_schema(fetch_range(d, d))
            if len(df):
                new = pd.concat([new, df], ignore_index=True)
            time.sleep(SLEEP)
        new = new.drop_duplicates(KEY, keep="last")
        # memo the post-merge counts so unchanged short finals settle
        record_stub_state(new, floor=STUB_FLOOR_BIP, scope="bip")
    new = new.sort_values(["Date", "GamePk", "AtBat", "PitchNum"])
    out_path.parent.mkdir(exist_ok=True)
    with atomic_write(out_path, "w", newline="", encoding="utf-8-sig") as f:
        new.to_csv(f, index=False)
    print(f"wrote {len(new):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
