"""Run every scraper to bring all CSVs in Data/ up to date.

Discovers scrape_*.py in this directory and runs each one with its default
output (all default to Data/). The pitch-arsenal scraper runs twice (pitcher
and batter views). build_ballparks.py is intentionally excluded: park
dimensions and elevations don't change daily. "Tools/2) Scrape Odds.py" lives
outside this directory on purpose: betting lines must be captured near game
time (closing lines), not in this morning data job — run it alongside
"Tools/1) Get Todays Games.py" near first pitch.

Each scraper is fault-isolated: one failing doesn't stop the rest, and the
exit code is non-zero if anything failed.

Safety net (validate_data.py): before each scraper runs, its current
known-good CSVs are copied to Data/backups/; after it runs, the fresh files
are schema-validated (required columns, keys, row counts vs the backup, date
sanity). A file that fails validation is REPLACED by its backup and the job
is marked FAILED — a silent upstream format change cannot quietly poison
everything downstream. The last log line is always "RESULT: OK" or
"RESULT: FAILED" for easy scanning of Logs/update_*.log.

Usage:
    python Scrapers/update_all.py [--retrain]

    --retrain    also retrain the models (Model/train.py --rebuild) after a
                 fully successful update. Skipped harmlessly while no
                 Model/train.py exists yet.
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import validate_data as V

SCRIPTS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN = SCRIPTS_DIR.parent / "Model" / "train.py"
DATA_DIR = V.DATA_DIR
BACKUP_DIR = V.BACKUP_DIR
# machine-readable outcome of the last run; the GUI reads this at startup
# and warns when the morning job failed (otherwise the only signal is a
# log line nobody looks at until predictions have quietly gone stale)
STATUS_FILE = SCRIPTS_DIR.parent / "Logs" / "last_run_status.json"


# names that must NOT run in the 6 AM data job even if a copy ever lands in
# this directory (the odds scraper lives at "Tools/2) Scrape Odds.py" and is
# run by hand near first pitch): betting lines are captured near game time
# (closing lines), and a morning run would grab opening/empty markets and
# waste the odds-API quota.
EXCLUDE = {"scrape_odds.py", "2_scrape_odds.py", "2) Scrape Odds.py"}

# which Data/ files each job owns (backed up before the run, validated after)
JOB_FILES = {
    "scrape_arm_strength.py": ["mlb_arm_strength.csv"],
    "scrape_batting_stats.py": ["mlb_batting_stats.csv"],
    "scrape_gamelogs.py": ["mlb_games.csv",
                              "mlb_game_batting.csv",
                              "mlb_game_pitching.csv"],
    "scrape_handedness.py": ["mlb_handedness.csv"],
    "scrape_homeruns.py": ["mlb_homeruns.csv"],
    "scrape_pitch_arsenals.py (pitchers)":
        ["mlb_pitch_arsenals.csv"],
    "scrape_pitch_arsenals.py (batters)":
        ["mlb_pitch_arsenals_batters.csv"],
    "scrape_pitching_stats.py": ["mlb_pitching_stats.csv"],
    "scrape_rosters.py": ["mlb_rosters.csv"],
    "scrape_milb.py": ["milb_batting.csv", "milb_pitching.csv"],
    "scrape_milb_gamelogs.py": ["milb_game_batting.csv",
                                "milb_game_pitching.csv"],
    "scrape_pbp.py": ["mlb_pbp.csv"],
    "scrape_statcast.py": ["mlb_statcast_bip.csv"],
    "scrape_pitches.py": ["mlb_pitch_daily_pitchers.csv",
                          "mlb_pitch_daily_batters.csv"],
    "scrape_pitches_milb.py": ["milb_pitch_daily_pitchers.csv",
                               "milb_pitch_daily_batters.csv"],
    "scrape_sprint_speed.py": ["mlb_sprint_speed.csv"],
    "scrape_oaa.py": ["mlb_oaa.csv", "mlb_oaa_players.csv"],
    "scrape_baserunning.py": ["mlb_baserunning.csv"],
    "scrape_weather.py": ["mlb_weather.csv"],
    "scrape_umpires.py": ["mlb_umpires.csv"],
    "scrape_bat_tracking.py": ["mlb_bat_tracking.csv"],
    "scrape_linescores.py": ["mlb_linescores.csv"],
    "scrape_catchers.py": ["mlb_catchers.csv", "mlb_catchers_team.csv"],
    "scrape_transactions.py": ["mlb_il_events.csv", "mlb_il.csv"],
}


def discover_jobs():
    jobs = []  # (label, [args])
    for script in sorted(SCRIPTS_DIR.glob("scrape_*.py")):
        if script.name in EXCLUDE:
            continue
        if "pitch_arsenals" in script.name:
            jobs.append((f"{script.name} (pitchers)", [str(script)]))
            jobs.append((f"{script.name} (batters)", [str(script), "--type", "batter"]))
        else:
            jobs.append((script.name, [str(script)]))
    return jobs


def run(label, args):
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, *args])
    took = time.time() - t0
    ok = proc.returncode == 0
    print(f">>> {label}: {'OK' if ok else f'FAILED (exit {proc.returncode})'} "
          f"in {took:.0f}s", flush=True)
    return ok, took


def rotate_backups(n=3):
    """Keep n generations of the backup set: backups.1 (newest snapshot)
    ... backups.{n-1}, rotated at the start of each run. The LIVE
    backups/ dir is COPIED, never renamed away — backup_known_good
    deliberately preserves the last-good backup when a live file is
    already invalid, and renaming the dir would destroy that guarantee.
    One bad-but-schema-valid morning can no longer poison the only
    restore point."""
    if not BACKUP_DIR.exists():
        return
    gens = [BACKUP_DIR.with_name(f"{BACKUP_DIR.name}.{i}")
            for i in range(1, n)]
    if gens and gens[-1].exists():
        shutil.rmtree(gens[-1])
    for older, newer in zip(reversed(gens[1:]), reversed(gens[:-1])):
        if newer.exists():
            newer.rename(older)
    if gens:
        shutil.copytree(BACKUP_DIR, gens[0])


def backup_known_good(files):
    """Copy each currently-valid file to Data/backups/ before its scraper
    rewrites it. A file that is ALREADY invalid is not backed up — that would
    clobber the last good backup with a bad copy."""
    BACKUP_DIR.mkdir(exist_ok=True)
    for name in files:
        src = DATA_DIR / name
        if not src.exists():
            continue
        if V.validate_file(src):        # current copy itself fails validation
            print(f"    (not backing up {name}: current copy already fails "
                  f"validation; keeping the existing backup)", flush=True)
            continue
        shutil.copy2(src, BACKUP_DIR / name)


def validate_and_restore(files):
    """Validate a job's fresh output against the backups. On failure, restore
    the backup so downstream consumers keep working. Returns True if every
    file passed."""
    all_ok = True
    for name in files:
        prev = BACKUP_DIR / name
        problems = V.validate_file(DATA_DIR / name,
                                   prev if prev.exists() else None)
        if not problems:
            continue
        all_ok = False
        for p in problems:
            print(f"    VALIDATION FAIL: {p}", flush=True)
        if prev.exists():
            shutil.copy2(prev, DATA_DIR / name)
            print(f"    restored {name} from backup", flush=True)
        else:
            print(f"    no backup available for {name}; the bad file was "
                  f"left in place for inspection", flush=True)
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain the models after a successful update")
    args = ap.parse_args()

    results = []
    rotate_backups()
    for label, cmd in discover_jobs():
        files = JOB_FILES.get(label, [])
        if files:
            backup_known_good(files)
        ok, took = run(label, cmd)
        # validate even when the scraper exited nonzero: a crash or kill
        # MID-WRITE used to leave a truncated file in place because the
        # restore only fired on exit 0. (The scrapers now write
        # atomically too — this is the second layer of the same defense,
        # and restoring over an untouched-but-valid file is a no-op.)
        if files:
            valid = validate_and_restore(files)
            if not valid:
                print(f">>> {label}: data FAILED validation", flush=True)
            ok = ok and valid
        results.append((label, ok, took))

    all_ok = all(ok for _, ok, _ in results)
    if args.retrain:
        if not all_ok:
            print("\nskipping retrain: at least one scraper failed or "
                  "produced invalid data", file=sys.stderr)
        elif not MODEL_TRAIN.exists():
            print(f"\nskipping retrain: {MODEL_TRAIN} does not exist yet",
                  flush=True)
            results.append(("retrain models (skipped: no Model/train.py)",
                            True, 0.0))
        else:
            ok, took = run("Model/train.py --rebuild",
                           [str(MODEL_TRAIN), "--rebuild"])
            results.append(("retrain models", ok, took))
            all_ok = all_ok and ok

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for label, ok, took in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {took:6.0f}s  {label}")

    STATUS_FILE.parent.mkdir(exist_ok=True)
    STATUS_FILE.write_text(json.dumps({
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "ok": all_ok,
        "failed_jobs": [label for label, ok, _ in results if not ok],
    }, indent=1))

    if not all_ok:
        print("\nRESULT: FAILED", flush=True)
        sys.exit(1)
    print("\nRESULT: OK")


if __name__ == "__main__":
    main()
