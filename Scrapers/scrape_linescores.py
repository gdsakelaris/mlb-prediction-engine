"""Scrape per-inning linescores for every game from the MLB StatsAPI.

The First-5-innings (F5) markets — F5 moneyline and F5 totals — grade on
runs through five innings, which `mlb_games.csv` (final scores only) cannot
provide. This accumulates that grading history from day one, so
partial-game markets can be graded whenever they ship.

Source: statsapi `/api/v1/game/{GamePk}/linescore`, whose `innings` list
carries away/home runs per inning. Stored LONG — one row per (GamePk,
Inning) — so any partial-game market (F3, F5, F7) can be graded from the
same file.

The game universe is `mlb_games.csv` (the authoritative list of played
games). Games already in the output CSV are cached — only new gamePks hit
the network, so completed seasons never refetch. --backfill forces a full
refetch. A final game at least EMPTY_MIN_AGE_DAYS old whose payload
parses to zero rows on EMPTY_MAX_TRIES total attempts lands in a
negative cache (Data/.empty_finals.json, shared with scrape_pbp — see
EmptyFinalsCache) and is skipped with a per-run count; without it such a
game never enters the CSV resume cache and is refetched every run
forever, invisibly inside "0 fetch failures". The entry is dropped the
moment the game returns data (e.g. a --backfill run after a late
statsapi fill), so the cache self-heals.

Usage:
    python scrape_linescores.py [-o output.csv] [--backfill] [--limit N]
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from seasons import atomic_write

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_linescores.csv"
GAMES_CSV = DATA_DIR / "mlb_games.csv"

LS_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/linescore"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
COLS = ["GamePk", "Date", "Season", "Inning", "AwayRuns", "HomeRuns"]

# Negative cache for finals whose endpoint payload is permanently empty
# (very old data gaps): a dotfile, NOT a .csv, so validate_data (specs
# keyed by *.csv name) never sees it. Shared with scrape_pbp, which
# imports EmptyFinalsCache; each scraper owns one scope inside the file.
EMPTY_CACHE = DATA_DIR / ".empty_finals.json"
EMPTY_MIN_AGE_DAYS = 3       # younger finals may legitimately still fill
EMPTY_MAX_TRIES = 3


class EmptyFinalsCache:
    """{GamePk: empty-attempt count} for one scraper scope.

    A final at least EMPTY_MIN_AGE_DAYS old whose payload parses to zero
    rows on EMPTY_MAX_TRIES total attempts is skipped thereafter
    (known_empty); record_data drops the entry the moment the game
    returns rows (e.g. a --backfill run after a late statsapi fill), so
    a healed upstream gap heals here too. save() re-merges the file so
    one scraper's save never clobbers the other scraper's scope."""

    def __init__(self, scope, path):
        self.scope = scope
        self.path = Path(path)
        self.counts = {int(k): int(v)
                       for k, v in (self._load().get(scope) or {}).items()}

    def _load(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def known_empty(self, pk):
        return self.counts.get(int(pk), 0) >= EMPTY_MAX_TRIES

    def record_empty(self, pk):
        self.counts[int(pk)] = self.counts.get(int(pk), 0) + 1

    def record_data(self, pk):
        self.counts.pop(int(pk), None)

    def save(self):
        data = self._load()
        data[self.scope] = {str(k): v
                            for k, v in sorted(self.counts.items())}
        self.path.parent.mkdir(exist_ok=True)
        with atomic_write(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)


def game_age_days(d):
    """Days since a game's Date (unparseable -> -1: never negative-cached)."""
    ts = pd.to_datetime(d, errors="coerce")
    return -1 if pd.isna(ts) else (date.today() - ts.date()).days


def linescore(pk, tries=3):
    """[(inning, away runs, home runs), ...] for one game; [] when the
    endpoint has no innings (very old data gaps)."""
    for attempt in range(tries):
        try:
            r = requests.get(LS_URL.format(pk=pk), headers=HEADERS,
                             timeout=30)
            r.raise_for_status()
            out = []
            for inn in r.json().get("innings", []):
                out.append((inn.get("num"),
                            inn.get("away", {}).get("runs"),
                            inn.get("home", {}).get("runs")))
            return out
        except Exception as e:           # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 5 * 2 ** attempt      # 5s, 10s
            print(f"    retry {pk} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every game, ignoring the cache")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N network fetches (smoke testing)")
    ap.add_argument("--sleep", type=float, default=0.12,
                    help="politeness delay between fetches (seconds)")
    args = ap.parse_args()

    if not GAMES_CSV.exists():
        sys.exit(f"{GAMES_CSV} not found — run scrape_gamelogs.py first")
    games = pd.read_csv(GAMES_CSV, encoding="utf-8-sig",
                        usecols=["GamePk", "Date", "Season"])
    games = games.drop_duplicates("GamePk").sort_values("GamePk")

    out_path = Path(args.output)
    stored = None
    if out_path.exists() and not args.backfill:
        stored = pd.read_csv(out_path, encoding="utf-8-sig")
    have = set() if stored is None else set(
        pd.to_numeric(stored["GamePk"], errors="coerce").dropna()
        .astype("int64"))

    todo = games[~games["GamePk"].isin(have)]
    cache = EmptyFinalsCache("linescores", EMPTY_CACHE)
    n_skip = 0
    if not args.backfill:                    # --backfill retries them all
        skip = todo["GamePk"].map(cache.known_empty).astype(bool)
        n_skip = int(skip.sum())
        todo = todo[~skip]
    print(f"{len(games):,} games in universe; {len(have):,} cached; "
          f"{n_skip:,} known-empty finals skipped; "
          f"{len(todo):,} to fetch", flush=True)

    fetched, fail = [], 0
    for i, g in enumerate(todo.itertuples(index=False)):
        if args.limit and i >= args.limit:
            print(f"--limit {args.limit} reached; stopping", flush=True)
            break
        try:
            innings = linescore(g.GamePk)
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  WARNING: {g.GamePk} failed ({e}); skipping (retried "
                  f"next run)", flush=True)
            continue
        if innings:
            cache.record_data(g.GamePk)
        elif game_age_days(g.Date) >= EMPTY_MIN_AGE_DAYS:
            cache.record_empty(g.GamePk)
        for num, away, home in innings:
            fetched.append({"GamePk": g.GamePk, "Date": g.Date,
                            "Season": g.Season, "Inning": num,
                            "AwayRuns": away, "HomeRuns": home})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1:,}/{len(todo):,} fetched", flush=True)
        time.sleep(args.sleep)
    cache.save()

    parts = [df for df in (stored, pd.DataFrame(fetched, columns=COLS))
             if df is not None and len(df)]
    if not parts:
        print("nothing to write", flush=True)
        return
    allrows = pd.concat(parts, ignore_index=True)[COLS]
    for c in ("GamePk", "Inning"):
        allrows[c] = pd.to_numeric(allrows[c], errors="coerce")
    allrows = (allrows.dropna(subset=["GamePk", "Inning"])
               .drop_duplicates(["GamePk", "Inning"], keep="last")
               .sort_values(["GamePk", "Inning"]))
    allrows["GamePk"] = allrows["GamePk"].astype("int64")
    allrows["Inning"] = allrows["Inning"].astype("int64")
    out_path.parent.mkdir(exist_ok=True)
    with atomic_write(out_path, "w", newline="", encoding="utf-8-sig") as f:
        allrows.to_csv(f, index=False)
    print(f"wrote {len(allrows):,} rows ({allrows['GamePk'].nunique():,} "
          f"games) -> {out_path} ({fail:,} fetch failures this run)",
          flush=True)


if __name__ == "__main__":
    main()
