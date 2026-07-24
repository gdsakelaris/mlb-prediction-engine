"""Scrape Statcast fielder arm strength per player per season from Savant.

Arm strength (average mph on competitive throws, tracked 2020+) is the
direct skill behind cutting down advancing runners — the outfield-arm
input a runner-advancement model needs when deciding whether the runner on
second scores on a single to right. The catcher file covers throwing to
bases on steals; this covers every fielder, split by the position the
throws came from (a CF's arm plays differently than the same arm in LF).

One row per (Year, PlayerId): total tracked throws, the max, the overall
average, and per-position averages (Arm1B..ArmRF plus the ArmInf/ArmOf
rollups) — NaN where the player logged no throws at that spot.

Designed for PRIOR-season consumption (leakage-free): a game sees the
previous season's measurement. Arm strength is among the most stable
year-to-year skills, so the lag costs little.

Completed seasons are served from the existing output CSV (they never
change); only the current season hits the network, and a failed
current-season fetch falls back to the previous run's rows. --backfill
forces a full refetch.

Usage:
    python scrape_arm_strength.py [-o output.csv] [--backfill]
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from seasons import CURRENT_SEASON, YEARS, atomic_write

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_arm_strength.csv"

API_URL = "https://baseballsavant.mlb.com/leaderboard/arm-strength"
FIRST_TRACKED = 2020             # arm-strength tracking starts here
MIN_THROWS = 20                  # low floor to catch bench/utility fielders
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# output column -> Savant CSV column
FIELDS = {
    "Throws": "total_throws",
    "MaxArm": "max_arm_strength",
    "ArmOverall": "arm_overall",
    "ArmInf": "arm_inf",
    "ArmOf": "arm_of",
    "Arm1B": "arm_1b",
    "Arm2B": "arm_2b",
    "Arm3B": "arm_3b",
    "ArmSS": "arm_ss",
    "ArmLF": "arm_lf",
    "ArmCF": "arm_cf",
    "ArmRF": "arm_rf",
}
COLS = ["Year", "PlayerId", "Name", "Pos"] + list(FIELDS)


def fetch_year(year, tries=4):
    params = {"type": "player", "year": year, "minThrows": MIN_THROWS,
              "csv": "true"}
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=120)
            r.raise_for_status()
            # Savant throttling returns HTTP 200 with an HTML page; make
            # that a retryable error instead of a pandas parse crash
            if "arm_overall" not in r.text[:2000]:
                raise ValueError("response is not the arm-strength CSV "
                                 "(throttled?)")
            return pd.read_csv(io.StringIO(r.text.lstrip("﻿")))
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 15 * 2 ** attempt                # 15s, 30s, 60s
            print(f"    retry {year} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every season, ignoring stored rows")
    args = ap.parse_args()

    out_path = Path(args.output)
    stored = None
    if out_path.exists() and not args.backfill:
        stored = pd.read_csv(out_path, encoding="utf-8-sig")
    have = set() if stored is None else \
        set(pd.to_numeric(stored["Year"], errors="coerce").dropna().astype(int))

    frames = []
    for year in YEARS:
        if year < FIRST_TRACKED:
            continue
        if year != CURRENT_SEASON and year in have:
            rows = stored[stored["Year"] == year]
            frames.append(rows[COLS])
            print(f"{year}: {len(rows):,} fielders (stored)", flush=True)
            continue
        try:
            df = fetch_year(year)
        except Exception as e:                      # noqa: BLE001
            if year in have:
                rows = stored[stored["Year"] == year]
                frames.append(rows[COLS])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(rows):,} rows (model uses "
                      f"prior-season arm strength, so this costs nothing)",
                      flush=True)
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?", flush=True)
                continue
            sys.exit(f"{year}: FAILED ({e}) and no stored rows to fall "
                     f"back on — run --backfill once the source recovers")
        out = pd.DataFrame({
            "Year": year,
            "PlayerId": pd.to_numeric(df["player_id"], errors="coerce"),
            "Name": df["fielder_name"],
            "Pos": pd.to_numeric(df["primary_position"], errors="coerce"),
            **{col: pd.to_numeric(df[src], errors="coerce")
               for col, src in FIELDS.items()},
        }).dropna(subset=["PlayerId"])
        out["PlayerId"] = out["PlayerId"].astype("int64")
        frames.append(out[COLS])
        print(f"{year}: {len(out):,} fielders", flush=True)
        time.sleep(1.0)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.drop_duplicates(["Year", "PlayerId"], keep="last")
    all_rows = all_rows.sort_values(["Year", "PlayerId"])
    out_path.parent.mkdir(exist_ok=True)
    with atomic_write(out_path, "w", newline="", encoding="utf-8-sig") as f:
        all_rows.to_csv(f, index=False)
    print(f"wrote {len(all_rows):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
