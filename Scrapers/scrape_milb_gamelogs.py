"""Scrape per-game MiLB boxscore logs (AAA + AA) from the MLB StatsAPI.

milb_batting.csv / milb_pitching.csv hold season AGGREGATES — fine for
level translation, useless for recency: "hot month at Triple-A right
before the call-up" needs game grain. This file adds it for the two
call-up feeder levels, from 2014 — one season of lookback before the MLB
files' 2015 start, so every call-up in the training window carries the
same as-of form features tonight's serve computes (2020 was canceled;
deeper history stays the season files' job).

  milb_game_batting.csv    one row per batter-game: lineup slot, position,
                           PA/AB/H/HR/BB/SO/SB/... (mirrors
                           mlb_game_batting.csv), plus Level (AAA/AA),
                           League (park/environment context) and Org — the
                           MLB parent club, i.e. where a call-up lands
  milb_game_pitching.csv   one row per pitcher-game: GS/GF, IP, BF,
                           pitches, H/R/ER/HR/BB/SO (mirrors
                           mlb_game_pitching.csv), same Level/League/Org

PlayerId is the MLBAM id used everywhere, so these rows join the MLB files
and the minors pitch aggregates directly.

The game universe is the StatsAPI schedule per level-season (final
regular-season games). Games already in the output are cached — only new
GamePks hit the network, concurrently, and completed rows are APPENDED
batch-by-batch, so an interrupted backfill resumes where it stopped.
--backfill starts the files over.

Usage:
    python scrape_milb_gamelogs.py [--backfill] [--limit N] [--workers N]
"""

import argparse
import concurrent.futures
import csv
import io
import threading
import time
from pathlib import Path

import pandas as pd
import requests

from seasons import CURRENT_SEASON

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
OUT_BAT = DATA_DIR / "milb_game_batting.csv"
OUT_PIT = DATA_DIR / "milb_game_pitching.csv"

API = "https://statsapi.mlb.com/api/v1"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

FIRST_SEASON = 2014              # one lookback year before the MLB files'
                                 # 2015 start; 2020 canceled (no schedule)
LEVELS = {11: "AAA", 12: "AA"}   # the call-up feeder levels
BATCH_GAMES = 500                # completed games buffered between appends

BAT_STATS = {  # CSV column -> boxscore batting field (mirrors MLB file)
    "PA": "plateAppearances", "AB": "atBats", "R": "runs", "H": "hits",
    "2B": "doubles", "3B": "triples", "HR": "homeRuns", "RBI": "rbi",
    "BB": "baseOnBalls", "IBB": "intentionalWalks", "SO": "strikeOuts",
    "HBP": "hitByPitch", "SB": "stolenBases", "CS": "caughtStealing",
    "SAC": "sacBunts", "SF": "sacFlies", "GIDP": "groundIntoDoublePlay",
    "TB": "totalBases", "LOB": "leftOnBase",
}
PIT_STATS = {  # CSV column -> boxscore pitching field (mirrors MLB file)
    "GS": "gamesStarted", "GF": "gamesFinished", "IP": "inningsPitched",
    "BF": "battersFaced", "NP": "numberOfPitches", "Strikes": "strikes",
    "H": "hits", "R": "runs", "ER": "earnedRuns", "HR": "homeRuns",
    "BB": "baseOnBalls", "IBB": "intentionalWalks", "SO": "strikeOuts",
    "HBP": "hitBatsmen", "WP": "wildPitches", "BK": "balks",
    "W": "wins", "L": "losses", "SV": "saves", "HLD": "holds",
}
META = ["GamePk", "Season", "Date", "Level", "League", "PlayerId", "Name",
        "Team", "Opponent", "Org", "Home"]
BAT_COLS = META + ["BattingOrder", "Position"] + list(BAT_STATS)
PIT_COLS = META + list(PIT_STATS)

_local = threading.local()


def get_session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update(HEADERS)
    return _local.session


def get_json(url, params=None, tries=4):
    for attempt in range(tries):
        try:
            resp = get_session().get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:                            # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def team_meta(season):
    """teamId -> (abbrev, league, parent-org MLB abbrev) for both levels,
    with the parent resolved through the MLB team list for that season."""
    mlb = get_json(f"{API}/teams", {"sportId": 1, "season": season})
    org_ab = {t["id"]: t.get("abbreviation", "")
              for t in mlb.get("teams", [])}
    meta = {}
    for sport_id in LEVELS:
        data = get_json(f"{API}/teams", {"sportId": sport_id,
                                         "season": season})
        for t in data.get("teams", []):
            meta[t["id"]] = (
                t.get("abbreviation", ""),
                (t.get("league") or {}).get("name", ""),
                org_ab.get(t.get("parentOrgId"), ""),
            )
    return meta


def season_universe(season):
    """[(GamePk, Date, Level, away id, home id)] of final regular-season
    games at both levels."""
    games, seen = [], set()
    for sport_id, level in LEVELS.items():
        data = get_json(f"{API}/schedule", {"sportId": sport_id,
                                            "season": season,
                                            "gameType": "R"})
        for day in data.get("dates", []):
            for g in day.get("games", []):
                if g["status"].get("codedGameState") != "F":
                    continue
                pk = g["gamePk"]
                if pk in seen:  # resumed/suspended games list twice
                    continue
                seen.add(pk)
                games.append({
                    "GamePk": pk,
                    "Date": g["officialDate"],
                    "Level": level,
                    "away_id": g["teams"]["away"]["team"]["id"],
                    "home_id": g["teams"]["home"]["team"]["id"],
                })
    return games


def parse_boxscore(game, box, meta, season):
    """(batting rows, pitching rows) for one game."""
    a_ab, a_lg, a_org = meta.get(game["away_id"], ("", "", ""))
    h_ab, h_lg, h_org = meta.get(game["home_id"], ("", "", ""))
    bat_rows, pit_rows = [], []
    for side, team, lg, org, opp, is_home in (
            ("away", a_ab, a_lg, a_org, h_ab, 0),
            ("home", h_ab, h_lg, h_org, a_ab, 1)):
        for p in box["teams"][side]["players"].values():
            common = {
                "GamePk": game["GamePk"], "Season": season,
                "Date": game["Date"], "Level": game["Level"], "League": lg,
                "PlayerId": p["person"]["id"],
                "Name": p["person"]["fullName"],
                "Team": team, "Opponent": opp, "Org": org, "Home": is_home,
            }
            bat = p.get("stats", {}).get("batting", {})
            if bat.get("gamesPlayed") or p.get("battingOrder"):
                row = dict(common)
                row["BattingOrder"] = p.get("battingOrder", "")
                row["Position"] = p.get("position", {}).get("abbreviation",
                                                            "")
                for col, field in BAT_STATS.items():
                    row[col] = bat.get(field, 0)
                bat_rows.append(row)
            pit = p.get("stats", {}).get("pitching", {})
            if pit.get("gamesPlayed"):
                row = dict(common)
                for col, field in PIT_STATS.items():
                    row[col] = pit.get(field, 0)
                pit_rows.append(row)
    return bat_rows, pit_rows


def append_rows(path, rows, cols, write_header):
    """Serialize a batch once and append it in a single write (a killed run
    can only lose the in-flight batch)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    if write_header:
        w.writeheader()
    w.writerows(rows)
    with open(path, "a", newline="", encoding="utf-8-sig" if write_header
              else "utf-8") as f:
        f.write(buf.getvalue())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backfill", action="store_true",
                    help="start the files over and refetch every game")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N network fetches (smoke testing)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    if args.backfill:
        for p in (OUT_BAT, OUT_PIT):
            if p.exists():
                p.unlink()
    have = set()
    if OUT_BAT.exists():
        have = set(pd.to_numeric(
            pd.read_csv(OUT_BAT, encoding="utf-8-sig",
                        usecols=["GamePk"])["GamePk"],
            errors="coerce").dropna().astype("int64"))

    header_due = not OUT_BAT.exists()
    n_bat = n_pit = n_fail = 0
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        universe = season_universe(season)
        todo = [g for g in universe if g["GamePk"] not in have]
        print(f"{season}: {len(universe):,} final games, "
              f"{len(todo):,} to fetch", flush=True)
        if not todo:
            continue
        if args.limit:
            todo = todo[:args.limit]
        meta = team_meta(season)

        def work(g):
            box = get_json(f"{API}/game/{g['GamePk']}/boxscore")
            return parse_boxscore(g, box, meta, season)

        bat_batch, pit_batch, n_done = [], [], 0
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers) as pool:
            futures = {pool.submit(work, g): g for g in todo}
            for fut in concurrent.futures.as_completed(futures):
                g = futures[fut]
                try:
                    bats, pits = fut.result()
                    bat_batch.extend(bats)
                    pit_batch.extend(pits)
                except Exception as e:               # noqa: BLE001
                    n_fail += 1
                    print(f"  WARNING: {g['GamePk']} failed ({e}); skipping "
                          f"(retried next run)", flush=True)
                n_done += 1
                if len(bat_batch) >= BATCH_GAMES * 22 or \
                        n_done == len(futures):
                    if bat_batch:
                        append_rows(OUT_BAT, bat_batch, BAT_COLS, header_due)
                        append_rows(OUT_PIT, pit_batch, PIT_COLS, header_due)
                        header_due = False
                        n_bat += len(bat_batch)
                        n_pit += len(pit_batch)
                        bat_batch, pit_batch = [], []
                if n_done % 500 == 0:
                    print(f"  {season}: {n_done:,}/{len(futures):,}",
                          flush=True)
        if args.limit:
            break

    print(f"appended {n_bat:,} batting + {n_pit:,} pitching rows "
          f"({n_fail:,} fetch failures this run)", flush=True)


if __name__ == "__main__":
    main()
