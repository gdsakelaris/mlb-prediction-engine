"""Scrape labeled play-by-play runner movements for every game.

The raw pitch archive shows the base-out STATE around every pitch, but not
what happened between the states: which runner took the extra base, who was
thrown out where, whether a run was earned, who got the error. This file
captures that layer — the training data for runner advancement, stolen-base
attempt/success, double plays, sac flies, and the earned/unearned run split
that ER props grade on.

Source: statsapi `/api/v1/game/{GamePk}/playByPlay`. One row per RUNNER
MOVEMENT within a play, batter included (a strikeout is the batter's out
movement; a homer is the batter's own trip to H). Each row carries:

  the play        AtBatIndex, Inning/Half, the PA's final result
                  (PlayEvent/PlayEventType), batter and pitcher
  the movement    RunnerId, StartBase -> EndBase (H = scored), IsOut with
                  OutBase/OutNumber, and PlayIndex — which event within
                  the PA caused it, so mid-PA actions (stolen bases, wild
                  pitches, passed balls, pickoffs, balks) are their own
                  labeled rows via Event/EventType (e.g. stolen_base_2b),
                  distinct from the PA's final result
  attribution     RBI / Earned / TeamUnearned flags per movement (runs and
                  the ER split), RespPitcherId (the pitcher charged with an
                  inherited runner), and Credits — the fielders on the play
                  as credit:position:playerId triplets separated by ';'
                  (putout/assist/fielding_error/...), which is where errors,
                  ROE and outfield-arm outcomes live

Base-out state BEFORE each play comes from joining the raw pitch archive on
(GamePk, AtBatIndex + 1 = at_bat_number) — statsapi indexes plays from 0,
Savant from 1.

The game universe is mlb_games.csv (the authoritative list of played
games). Games already in the output CSV are cached — only new GamePks hit
the network, concurrently like scrape_gamelogs.py, and completed rows are
APPENDED batch-by-batch (never a full-file rewrite), so an interrupted
backfill resumes where it stopped. --backfill starts the file over.

Usage:
    python scrape_pbp.py [-o output.csv] [--backfill] [--limit N]
                         [--workers N]
"""

import argparse
import concurrent.futures
import csv
import io
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_pbp.csv"
GAMES_CSV = DATA_DIR / "mlb_games.csv"

PBP_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

COLS = ["GamePk", "Season", "Date", "AtBatIndex", "PlayIndex", "Inning",
        "Half", "PlayEvent", "PlayEventType", "BatterId", "PitcherId",
        "RunnerId", "StartBase", "EndBase", "IsOut", "OutBase", "OutNumber",
        "RBI", "Earned", "TeamUnearned", "RespPitcherId", "Event",
        "EventType", "MovementReason", "Credits"]

BATCH_GAMES = 400        # completed games buffered between CSV appends

_local = threading.local()


def get_session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update(HEADERS)
    return _local.session


def get_json(url, tries=4):
    for attempt in range(tries):
        try:
            resp = get_session().get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:                            # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def parse_game(pk, season, date, pbp):
    """One playByPlay JSON -> movement rows (see module docstring)."""
    rows = []
    for play in pbp.get("allPlays", []):
        res = play.get("result") or {}
        about = play.get("about") or {}
        matchup = play.get("matchup") or {}
        base = {
            "GamePk": pk, "Season": season, "Date": date,
            "AtBatIndex": play.get("atBatIndex", ""),
            "Inning": about.get("inning", ""),
            "Half": about.get("halfInning", ""),
            "PlayEvent": res.get("event", ""),
            "PlayEventType": res.get("eventType", ""),
            "BatterId": (matchup.get("batter") or {}).get("id", ""),
            "PitcherId": (matchup.get("pitcher") or {}).get("id", ""),
        }
        for r in play.get("runners") or []:
            mv = r.get("movement") or {}
            det = r.get("details") or {}
            end = mv.get("end")
            out_b = mv.get("outBase")
            out_n = mv.get("outNumber")
            credits = ";".join(
                "{}:{}:{}".format(
                    str(c.get("credit") or "").removeprefix("f_"),
                    (c.get("position") or {}).get("abbreviation", ""),
                    (c.get("player") or {}).get("id", ""))
                for c in r.get("credits") or [])
            rows.append({
                **base,
                "PlayIndex": det.get("playIndex", ""),
                "RunnerId": (det.get("runner") or {}).get("id", ""),
                "StartBase": mv.get("start") or "",
                "EndBase": "H" if end in ("score", "4B") else (end or ""),
                "IsOut": int(bool(mv.get("isOut"))),
                "OutBase": "H" if out_b in ("score", "4B") else (out_b or ""),
                "OutNumber": "" if out_n is None else out_n,
                "RBI": int(bool(det.get("rbi"))),
                "Earned": int(bool(det.get("earned"))),
                "TeamUnearned": int(bool(det.get("teamUnearned"))),
                "RespPitcherId":
                    (det.get("responsiblePitcher") or {}).get("id", ""),
                "Event": det.get("event", ""),
                "EventType": det.get("eventType", ""),
                "MovementReason": det.get("movementReason") or "",
                "Credits": credits,
            })
    return rows


def append_rows(out_path, rows, write_header):
    """Serialize a batch once and append it in a single write, so a killed
    run can only lose the in-flight batch, not corrupt finished ones."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS)
    if write_header:
        w.writeheader()
    w.writerows(rows)
    with open(out_path, "a", newline="", encoding="utf-8-sig" if write_header
              else "utf-8") as f:
        f.write(buf.getvalue())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="start the file over and refetch every game")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N network fetches (smoke testing)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not GAMES_CSV.exists():
        sys.exit(f"{GAMES_CSV} not found — run scrape_gamelogs.py first")
    games = pd.read_csv(GAMES_CSV, encoding="utf-8-sig",
                        usecols=["GamePk", "Date", "Season"])
    games = games.drop_duplicates("GamePk").sort_values(["Date", "GamePk"])

    out_path = Path(args.output)
    if args.backfill and out_path.exists():
        out_path.unlink()
    have = set()
    if out_path.exists():
        have = set(pd.to_numeric(
            pd.read_csv(out_path, encoding="utf-8-sig",
                        usecols=["GamePk"])["GamePk"],
            errors="coerce").dropna().astype("int64"))

    todo = games[~games["GamePk"].isin(have)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(games):,} games in universe; {len(have):,} cached; "
          f"{len(todo):,} to fetch", flush=True)
    if todo.empty:
        print("nothing to fetch", flush=True)
        return

    def work(g):
        pbp = get_json(PBP_URL.format(pk=g.GamePk))
        return parse_game(g.GamePk, g.Season, g.Date, pbp)

    header_due = not out_path.exists()
    n_rows = n_done = n_fail = 0
    batch = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers) as pool:
        futures = {pool.submit(work, g): g
                   for g in todo.itertuples(index=False)}
        for fut in concurrent.futures.as_completed(futures):
            g = futures[fut]
            try:
                batch.extend(fut.result())
            except Exception as e:                   # noqa: BLE001
                n_fail += 1
                print(f"  WARNING: {g.GamePk} failed ({e}); skipping "
                      f"(retried next run)", flush=True)
            n_done += 1
            if len(batch) >= BATCH_GAMES * 100 or \
                    (batch and n_done == len(futures)):
                append_rows(out_path, batch, header_due)
                header_due = False
                n_rows += len(batch)
                batch = []
            if n_done % 500 == 0:
                print(f"  {n_done:,}/{len(futures):,} games "
                      f"({n_rows:,} rows so far)", flush=True)
    if batch:
        append_rows(out_path, batch, header_due)
        n_rows += len(batch)

    print(f"appended {n_rows:,} rows from {n_done - n_fail:,} games "
          f"-> {out_path} ({n_fail:,} fetch failures this run)", flush=True)


if __name__ == "__main__":
    main()
