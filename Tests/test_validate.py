"""Data-layer safety net (Scrapers/validate_data.py + update_all's
backup/restore loop) on synthetic corrupted CSVs in tmp_path. Registered
contracts are exercised by NAME (validate_file looks specs up by
path.name), so the live Data/ tree is never touched. Date-dependent checks
pin the module clock to 2026-07-23 (mid-season) so the freshness tripwire
is deterministic year-round.
"""
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scrapers"))

import update_all as U       # noqa: E402
import validate_data as V    # noqa: E402


class _July23(date):
    @classmethod
    def today(cls):
        return date(2026, 7, 23)


def _pin_clock(monkeypatch):
    monkeypatch.setattr(V, "date", _July23)


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


# ------------------------------------------------- mlb_ballparks contract
# (small static file: key=Ballpark dup_frac 0.0, numeric tol 0.0,
#  min_rows 30, shrink_tol 1.0 — the tightest registered contract)

BALLPARK_COLS = ["Ballpark", "Team", "LF", "CF", "RF", "Elevation_ft",
                 "Lat", "Lon", "Roof"]


def _ballpark_rows(n):
    return [[f"Park {i}", f"T{i % 30}", 330 + i % 10, 400, 328, 500 + i,
             40.0 + i * 0.01, -75.0 - i * 0.01, "open"] for i in range(n)]


def _parks(tmp_path, rows, cols=BALLPARK_COLS, name="mlb_ballparks.csv"):
    p = tmp_path / name
    _write(p, cols, rows)
    return p


def test_conforming_registered_contract_passes(tmp_path):
    p = _parks(tmp_path, _ballpark_rows(35))
    assert V.validate_file(p) == []


def test_unregistered_filename_fails_closed(tmp_path):
    p = _parks(tmp_path, _ballpark_rows(35), name="mystery.csv")
    assert V.validate_file(p) == ["no schema spec for mystery.csv"]


def test_missing_required_column_fails(tmp_path):
    rows = [r[:-1] for r in _ballpark_rows(35)]        # Roof dropped
    p = _parks(tmp_path, rows, cols=BALLPARK_COLS[:-1])
    problems = V.validate_file(p)
    assert len(problems) == 1                          # early return
    assert "missing required columns" in problems[0]
    assert "'Roof'" in problems[0]


def test_duplicate_keys_beyond_tolerance_fail(tmp_path):
    rows = _ballpark_rows(35)
    rows.append(rows[0][:])                            # dup Ballpark key
    p = _parks(tmp_path, rows)
    problems = V.validate_file(p)
    assert len(problems) == 1
    assert "duplicate rows on key" in problems[0]
    assert "Ballpark" in problems[0]


def test_excess_nan_fails(tmp_path):
    rows = _ballpark_rows(35)
    rows[7][2] = ""                                    # blank LF, tol 0.0
    p = _parks(tmp_path, rows)
    problems = V.validate_file(p)
    assert len(problems) == 1
    assert "non-numeric/blank" in problems[0] and "'LF'" in problems[0]


def test_shrink_beyond_limit_fails(tmp_path):
    # shrink is new-vs-PREVIOUS-copy row count: prev 40 -> new 34 trips
    # shrink_tol=1.0 (still above min_rows=30, so only the shrink fires)
    (tmp_path / "b").mkdir()
    (tmp_path / "c").mkdir()
    prev = _parks(tmp_path / "b", _ballpark_rows(40))
    new = _parks(tmp_path, _ballpark_rows(34))
    problems = V.validate_file(new, prev)
    assert len(problems) == 1
    assert "shrank 40 -> 34 rows" in problems[0]
    # equal size passes, and no-prev skips the check entirely
    same = _parks(tmp_path / "c", _ballpark_rows(40))
    assert V.validate_file(same, prev) == []
    assert V.validate_file(new) == []


def _bat_rows(n):
    return [[2024 + i % 2, 1000 + i, 71.5, 7.3] for i in range(n)]


def test_shrink_abs_grace_then_fail(tmp_path):
    # mlb_bat_tracking.csv: shrink_tol .999 PLUS shrink_abs=15 rows of
    # grace (qualifier churn); an 8-row drop passes, a 20-row drop fails
    cols = ["Year", "PlayerId", "BatSpeed", "SwingLength"]
    (tmp_path / "b").mkdir()
    prev = tmp_path / "b" / "mlb_bat_tracking.csv"
    _write(prev, cols, _bat_rows(460))
    ok = tmp_path / "mlb_bat_tracking.csv"
    _write(ok, cols, _bat_rows(452))                   # -8 <= 15: fine
    assert V.validate_file(ok, prev) == []
    _write(ok, cols, _bat_rows(440))                   # -20 > 15: fails
    problems = V.validate_file(ok, prev)
    assert len(problems) == 1 and "shrank" in problems[0]


def test_vanished_season_fails(tmp_path):
    cols = ["Year", "PlayerId", "BatSpeed", "SwingLength"]
    (tmp_path / "b").mkdir()
    prev = tmp_path / "b" / "mlb_bat_tracking.csv"
    _write(prev, cols, _bat_rows(460))
    new = tmp_path / "mlb_bat_tracking.csv"
    _write(new, cols, [[2025, 1000 + i, 71.5, 7.3]     # 2024 history gone,
                       for i in range(460)])           # same row count
    problems = V.validate_file(new, prev)
    assert len(problems) == 1
    assert "seasons vanished" in problems[0] and "2024" in problems[0]


# ------------------------------------------------------ odds-store dates

ODDS_COLS = V.SPECS["mlb_odds.csv"]["required_cols"]


def _odds_rows(dates):
    return [[d, "776001", "592450", "batter_hits", 1.5, -110, -110,
             "draftkings", f"{d}T12:00:00", -110, -110, f"{d}T09:00:00"]
            for d in dates]


def test_odds_store_conforming_passes(tmp_path, monkeypatch):
    _pin_clock(monkeypatch)
    p = tmp_path / "mlb_odds.csv"
    _write(p, ODDS_COLS, _odds_rows(["2026-07-21", "2026-07-22",
                                     "2026-07-23"]))
    assert V.validate_file(p) == []


def test_odds_store_future_date_fails(tmp_path, monkeypatch):
    _pin_clock(monkeypatch)
    p = tmp_path / "mlb_odds.csv"
    _write(p, ODDS_COLS, _odds_rows(["2026-07-22", "2026-08-30"]))
    problems = V.validate_file(p)
    assert len(problems) == 1
    assert "is in the future" in problems[0]


def test_odds_store_unparseable_dates_fail(tmp_path, monkeypatch):
    _pin_clock(monkeypatch)
    p = tmp_path / "mlb_odds.csv"
    _write(p, ODDS_COLS, _odds_rows(["2026-07-22"] * 4 + ["not-a-date"]))
    problems = V.validate_file(p)
    assert len(problems) == 1
    assert "fails to parse as a date" in problems[0]


# -------------------------------------------------- freshness tripwire

def _linescores(tmp_path, last_date):
    p = tmp_path / "mlb_linescores.csv"
    cols = ["GamePk", "Date", "Season", "Inning", "AwayRuns", "HomeRuns"]
    rows = []
    for gp in range(140):                              # 1260 rows >= 1000
        d = f"2026-06-{gp % 28 + 1:02d}" if gp else last_date
        for inn in range(1, 10):
            rows.append([700000 + gp, d, 2026, inn, inn % 3, inn % 2])
    _write(p, cols, rows)
    return p


def test_freshness_stale_midseason_fails(tmp_path, monkeypatch):
    _pin_clock(monkeypatch)                            # 2026-07-23, July
    p = _linescores(tmp_path, "2026-06-28")            # newest 25 days old
    problems = V.validate_file(p)
    assert len(problems) == 1
    assert "days old mid-season" in problems[0]


def test_freshness_recent_passes(tmp_path, monkeypatch):
    _pin_clock(monkeypatch)
    p = _linescores(tmp_path, "2026-07-22")            # within fresh_days=6
    assert V.validate_file(p) == []


# --------------------------------------- update_all validate_and_restore

def _restore_env(tmp_path, monkeypatch):
    data = tmp_path / "Data"
    bak = data / "backups"
    bak.mkdir(parents=True)
    monkeypatch.setattr(U, "DATA_DIR", data)
    monkeypatch.setattr(U, "BACKUP_DIR", bak)
    return data, bak


def test_validate_and_restore_replaces_corrupt_live_with_backup(
        tmp_path, monkeypatch):
    data, bak = _restore_env(tmp_path, monkeypatch)
    _write(bak / "mlb_ballparks.csv", BALLPARK_COLS, _ballpark_rows(35))
    _write(data / "mlb_ballparks.csv", BALLPARK_COLS[:-1],
           [r[:-1] for r in _ballpark_rows(35)])       # column vanished
    assert U.validate_and_restore(["mlb_ballparks.csv"]) is False
    live = (data / "mlb_ballparks.csv").read_bytes()
    assert live == (bak / "mlb_ballparks.csv").read_bytes()
    assert V.validate_file(data / "mlb_ballparks.csv") == []


def test_validate_and_restore_leaves_good_file_alone(tmp_path, monkeypatch):
    data, bak = _restore_env(tmp_path, monkeypatch)
    _write(bak / "mlb_ballparks.csv", BALLPARK_COLS, _ballpark_rows(40))
    _write(data / "mlb_ballparks.csv", BALLPARK_COLS, _ballpark_rows(41))
    before = (data / "mlb_ballparks.csv").read_bytes()
    assert U.validate_and_restore(["mlb_ballparks.csv"]) is True
    assert (data / "mlb_ballparks.csv").read_bytes() == before


def test_validate_and_restore_no_backup_leaves_bad_file(
        tmp_path, monkeypatch):
    data, _ = _restore_env(tmp_path, monkeypatch)
    bad_rows = [r[:-1] for r in _ballpark_rows(35)]
    _write(data / "mlb_ballparks.csv", BALLPARK_COLS[:-1], bad_rows)
    before = (data / "mlb_ballparks.csv").read_bytes()
    assert U.validate_and_restore(["mlb_ballparks.csv"]) is False
    # bad file deliberately left in place for inspection
    assert (data / "mlb_ballparks.csv").read_bytes() == before
