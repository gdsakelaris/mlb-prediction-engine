"""Scrape pitch-level MINOR-LEAGUE Statcast data as daily per-player
aggregates — the tracked-minors mirror of scrape_pitches.py.

Savant's pitch feed covers the minors where Hawk-Eye is installed: every
AAA park since 2023 and the Florida State League (Single-A) since 2021.
That is exactly the population where sharper priors pay: a called-up
rookie's MLB sample is thin, but his AAA swing-and-miss, chase and velo
profile is measured by the same system that measures the majors, so the
same aggregate columns translate directly.

Reuses scrape_pitches.aggregate — one row per player per day of the same
sufficient statistics (see that module's docstring for the full column
story) — plus a Level column (AAA / A, from the StatsAPI schedule for
each tracked league) so consumers can level-adjust. PlayerId is the MLBAM
id used everywhere, so these rows join the MiLB game logs and the MLB
files directly.

--backfill also archives the raw pitches to Data/raw_pitches_milb/
pitches_{year}.parquet (every Savant detail column, zstd), so future
schema changes re-aggregate from disk via --from-raw instead of another
long download.

Default run is incremental — the output CSVs double as the cache: stored
rows are reused; only the newest REFETCH_DAYS plus any missing season are
downloaded. Seconds in the daily job.

Usage:
    python scrape_pitches_milb.py [--outdir DIR] [--backfill] [--from-raw]
"""

import argparse
import io
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

import scrape_pitches as SP
from seasons import YEARS, atomic_write

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
RAW_DIR = DATA_DIR / "raw_pitches_milb"
OUT_PITCHERS = "milb_pitch_daily_pitchers.csv"
OUT_BATTERS = "milb_pitch_daily_batters.csv"

FIRST_TRACKED = 2021             # FSL 2021; every AAA park from 2023
CHUNK_DAYS = 3                   # smaller than MLB: AAA+FSL slates are big
SLEEP = 2.0

STATSAPI = "https://statsapi.mlb.com/api/v1"
LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A"}


def fetch_range(d0, d1, tries=3):
    """All tracked-minors pitches in [d0, d1] (regular season). Splits on
    the result cap, same guard as the MLB fetch."""
    params = {
        "all": "true", "type": "details", "player_type": "batter",
        "hfGT": "R|", "minors": "true",
        "game_date_gt": str(d0), "game_date_lt": str(d1),
    }
    for attempt in range(tries):
        try:
            r = requests.get(SP.API_URL, params=params, headers=SP.HEADERS,
                             timeout=240)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            break
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    retry {d0}..{d1} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    if len(df) >= SP.CAP_ROWS and d0 != d1:
        mid = d0 + (d1 - d0) / 2
        print(f"    {d0}..{d1}: {len(df):,} rows (cap?) -> splitting",
              flush=True)
        time.sleep(SLEEP)
        left = fetch_range(d0, mid)
        time.sleep(SLEEP)
        right = fetch_range(mid + timedelta(days=1), d1)
        return pd.concat([left, right], ignore_index=True)
    return df


def level_map(year, tries=3):
    """gamePk -> level name for every affiliated game that season, from
    one StatsAPI schedule call per league. Labels each aggregate row."""
    out = {}
    for sport_id, level in LEVELS.items():
        for attempt in range(tries):
            try:
                r = requests.get(f"{STATSAPI}/schedule",
                                 params={"sportId": sport_id, "season": year,
                                         "gameType": "R"},
                                 headers=SP.HEADERS, timeout=60)
                r.raise_for_status()
                for day in r.json().get("dates", []):
                    for g in day.get("games", []):
                        out[g["gamePk"]] = level
                break
            except Exception as e:                  # noqa: BLE001
                if attempt == tries - 1:
                    raise
                time.sleep(5 * (attempt + 1))
    return out


def aggregate_leveled(raw, lmap):
    """One Savant chunk -> (pitcher day rows, batter day rows), each with a
    Level column. A player plays one level on a given day, so splitting by
    level before aggregating keeps every daily row level-pure."""
    if raw.empty:
        return None, None
    lvl = pd.to_numeric(raw["game_pk"], errors="coerce").map(lmap).fillna("")
    pit_frames, bat_frames = [], []
    for level, sub in raw.groupby(lvl):
        pit, bat = SP.aggregate(sub)
        if pit is None:
            continue
        pit.insert(2, "Level", level)
        bat.insert(2, "Level", level)
        pit_frames.append(pit)
        bat_frames.append(bat)
    if not pit_frames:
        return None, None
    return (pd.concat(pit_frames, ignore_index=True),
            pd.concat(bat_frames, ignore_index=True))


def season_windows(year, start=None):
    d0 = date(year, 3, 1) if start is None else max(start, date(year, 3, 1))
    end = min(date(year, 11, 30), date.today())
    windows = []
    while d0 <= end:
        d1 = min(d0 + timedelta(days=CHUNK_DAYS - 1), end)
        windows.append((d0, d1))
        d0 = d1 + timedelta(days=1)
    return windows


def write_raw(year, frames):
    """Archive one season of raw minors pitches (schema changes never need
    a re-download). Object columns are stringified (mixed-type chunks)."""
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).where(df[c].notna())
    path = RAW_DIR / f"pitches_{year}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False, compression="zstd")
    except Exception as e:                              # noqa: BLE001
        try:
            os.remove(tmp)
        except OSError:
            pass
        path = RAW_DIR / f"pitches_{year}.csv.gz"
        tmp = path.with_name(path.name + ".tmp")
        print(f"    parquet failed ({e}); falling back to csv.gz", flush=True)
        df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    print(f"    raw archive: {len(df):,} pitches -> {path.name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(DATA_DIR))
    ap.add_argument("--backfill", action="store_true",
                    help="re-download all seasons (default is incremental)")
    ap.add_argument("--from-raw", action="store_true", dest="from_raw",
                    help="re-aggregate seasons from Data/raw_pitches_milb "
                         "archives where present")
    args = ap.parse_args()
    if args.from_raw:
        args.backfill = True
    outdir = Path(args.outdir)
    p_path, b_path = outdir / OUT_PITCHERS, outdir / OUT_BATTERS

    kept_p, start, have = SP.load_existing(p_path, args.backfill)
    kept_b, _, _ = SP.load_existing(b_path, args.backfill)
    if start is not None:
        print(f"incremental: {len(kept_p):,} pitcher-day rows kept, "
              f"refetching from {start}", flush=True)

    pit_frames, bat_frames = [], []
    for year in YEARS:
        if year < FIRST_TRACKED:
            continue
        if start is not None and year < start.year and year in have:
            continue
        clip = start if start is not None and year >= start.year else None
        lmap = level_map(year)
        raw_path = RAW_DIR / f"pitches_{year}.parquet"
        if args.from_raw and clip is None and raw_path.exists():
            pit, bat = aggregate_leveled(pd.read_parquet(raw_path), lmap)
            if pit is not None:
                pit_frames.append(pit)
                bat_frames.append(bat)
            print(f"{year}: {int(pit['n'].sum()) if pit is not None else 0:,}"
                  f" pitches (raw archive)", flush=True)
            continue
        got = 0
        raw_chunks = [] if args.backfill else None
        for d0, d1 in season_windows(year, clip):
            raw = fetch_range(d0, d1)
            if raw_chunks is not None and not raw.empty:
                raw_chunks.append(raw)
            pit, bat = aggregate_leveled(raw, lmap)
            if pit is not None:
                got += int(pit["n"].sum())
                pit_frames.append(pit)
                bat_frames.append(bat)
            time.sleep(SLEEP)
        if raw_chunks is not None:
            write_raw(year, raw_chunks)
            raw_chunks.clear()
        print(f"{year}: {got:,} pitches", flush=True)

    def finish(kept, frames, path):
        new = pd.concat(frames, ignore_index=True) if frames else None
        if new is not None and kept is not None:
            new = pd.concat([kept, new], ignore_index=True)
        elif new is None:
            new = kept
        if new is None:
            print(f"nothing to write for {path}", flush=True)
            return
        if "Level" not in new.columns:      # rows kept from an older file
            new["Level"] = ""
        new = (new.drop_duplicates(["PlayerId", "Date", "Level"],
                                   keep="last")
               .sort_values(["PlayerId", "Date"]))
        path.parent.mkdir(exist_ok=True)
        with atomic_write(path, "w", newline="", encoding="utf-8-sig") as f:
            new.to_csv(f, index=False)
        print(f"wrote {len(new):,} rows -> {path}", flush=True)

    finish(kept_p, pit_frames, p_path)
    finish(kept_b, bat_frames, b_path)


if __name__ == "__main__":
    main()
