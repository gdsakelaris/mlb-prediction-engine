"""Savant scraper guards (Scrapers/scrape_statcast.py + scrape_pitches.py)
on their pure seams — no network:

- schema drift fails CLOSED: a non-empty payload missing a mapped source
  column raises (SAVANT_SCHEMA_LAX=1 degrades to a stderr warning + NA)
- suspended-game completeness re-check: find_stub_dates flags FINAL games
  (mlb_games.csv) whose stored per-(Date, GamePk) rows sit below the
  plausibility floor — the signature of a completion stamped with the
  original game_date behind the incremental watermark
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scrapers"))

import scrape_pitches as SP   # noqa: E402
import scrape_statcast as SS  # noqa: E402


# ------------------------------------------------- schema drift: statcast

def _savant_bip(n=3, rename=None):
    cols = [s for s in SS.COLUMNS.values() if s is not None]
    raw = pd.DataFrame({c: [1] * n for c in cols})
    raw["game_date"] = "2026-07-20"
    if rename:
        raw = raw.rename(columns=rename)
    return raw


def test_to_schema_missing_source_column_raises(monkeypatch):
    monkeypatch.delenv("SAVANT_SCHEMA_LAX", raising=False)
    raw = _savant_bip(rename={"launch_speed": "ls_renamed"})
    with pytest.raises(RuntimeError) as ei:
        SS.to_schema(raw)
    assert "launch_speed" in str(ei.value)


def test_to_schema_lax_env_warns_and_fills_na(monkeypatch, capsys):
    monkeypatch.setenv("SAVANT_SCHEMA_LAX", "1")
    raw = _savant_bip(rename={"launch_speed": "ls_renamed"})
    out = SS.to_schema(raw)
    assert len(out) == 3
    assert out["ExitVelo"].isna().all()
    err = capsys.readouterr().err
    assert "WARNING" in err and "launch_speed" in err


def test_to_schema_empty_payload_never_raises(monkeypatch):
    monkeypatch.delenv("SAVANT_SCHEMA_LAX", raising=False)
    assert SS.to_schema(pd.DataFrame()).empty


# --------------------------------------------- schema drift: pitch daily

def _savant_pitches(n=4, drop=None):
    raw = pd.DataFrame({c: [None] * n for c in SP.REQUIRED_SRC})
    raw["game_pk"] = 777001
    raw["game_date"] = "2026-07-20"
    raw["at_bat_number"] = range(1, n + 1)
    raw["pitch_number"] = 1
    raw["pitcher"] = 605483
    raw["batter"] = 592450
    raw["description"] = "called_strike"
    raw["pitch_type"] = "FF"
    raw["release_speed"] = 94.0
    raw["balls"] = 0
    raw["strikes"] = 0
    raw["zone"] = 5
    if drop:
        raw = raw.drop(columns=[drop])
    return raw


def test_aggregate_missing_source_column_raises(monkeypatch):
    monkeypatch.delenv("SAVANT_SCHEMA_LAX", raising=False)
    with pytest.raises(RuntimeError) as ei:
        SP.aggregate(_savant_pitches(drop="release_speed"))
    assert "release_speed" in str(ei.value)


def test_aggregate_lax_env_fills_na_and_warns(monkeypatch, capsys):
    monkeypatch.setenv("SAVANT_SCHEMA_LAX", "1")
    pit, bat = SP.aggregate(_savant_pitches(drop="release_speed"))
    assert pit is not None and pit["n"].sum() == 4
    err = capsys.readouterr().err
    assert "WARNING" in err and "release_speed" in err


def test_aggregate_intact_payload_passes(monkeypatch):
    monkeypatch.delenv("SAVANT_SCHEMA_LAX", raising=False)
    pit, bat = SP.aggregate(_savant_pitches())
    assert pit["cs_n"].sum() == 4 and bat["n"].sum() == 4


# --------------------------------- suspended-game completeness re-check

def _games_csv(tmp_path, rows):
    p = tmp_path / "mlb_games.csv"
    pd.DataFrame(rows, columns=["GamePk", "Date"]).to_csv(p, index=False)
    return p


def _events(gpk, d, n):
    return pd.DataFrame({"Date": [d] * n, "GamePk": [gpk] * n})


def test_stub_floor_flags_80_pitch_final_passes_280(tmp_path):
    d_stub = date.today() - timedelta(days=6)
    d_ok = date.today() - timedelta(days=5)
    games = _games_csv(tmp_path, [[777001, str(d_stub)],
                                  [777002, str(d_ok)]])
    ev = pd.concat([_events(777001, d_stub, 80),
                    _events(777002, d_ok, 280)], ignore_index=True)
    stubs = SS.find_stub_dates(ev, floor=SP.STUB_FLOOR_PITCHES,
                               until=date.today() - timedelta(days=2),
                               games_path=games)
    assert stubs == [d_stub]


def test_bip_floor_flags_stub_passes_full_game(tmp_path):
    d_stub = date.today() - timedelta(days=6)
    d_ok = date.today() - timedelta(days=5)
    games = _games_csv(tmp_path, [[777001, str(d_stub)],
                                  [777002, str(d_ok)]])
    ev = pd.concat([_events(777001, d_stub, 10),
                    _events(777002, d_ok, 48)], ignore_index=True)
    stubs = SS.find_stub_dates(ev, floor=SS.STUB_FLOOR_BIP,
                               until=date.today() - timedelta(days=2),
                               games_path=games)
    assert stubs == [d_stub]


def test_final_missing_entirely_from_store_flagged(tmp_path):
    d_gone = date.today() - timedelta(days=7)
    d_ok = date.today() - timedelta(days=4)
    games = _games_csv(tmp_path, [[777001, str(d_gone)],
                                  [777002, str(d_ok)]])
    ev = _events(777002.0, d_ok, 280)      # float GamePk still matches
    stubs = SS.find_stub_dates(ev, floor=150,
                               until=date.today() - timedelta(days=2),
                               games_path=games)
    assert stubs == [d_gone]


def test_watermark_dates_and_old_dates_excluded(tmp_path):
    d_fresh = date.today() - timedelta(days=1)     # >= until: just fetched
    d_old = (date.today()                          # outside the window
             - timedelta(days=SS.STUB_WINDOW_DAYS + 5))
    games = _games_csv(tmp_path, [[777001, str(d_fresh)],
                                  [777002, str(d_old)]])
    ev = _events(777003, date.today() - timedelta(days=3), 280)
    assert SS.find_stub_dates(ev, floor=150,
                              until=date.today() - timedelta(days=2),
                              games_path=games) == []


def test_no_games_file_returns_empty(tmp_path):
    ev = _events(777001, date.today() - timedelta(days=3), 80)
    assert SS.find_stub_dates(ev, floor=150, until=date.today(),
                              games_path=tmp_path / "missing.csv") == []


def test_stub_memo_settles_unchanged_short_final(tmp_path):
    """A legitimately short (rain-shortened) final is flagged ONCE:
    record_stub_state stores its post-refetch counts, unchanged counts
    stop flagging, and a count change (completion arrived) re-flags."""
    d = date.today() - timedelta(days=6)
    games = _games_csv(tmp_path, [[777001, str(d)]])
    memo = tmp_path / "memo.json"
    until = date.today() - timedelta(days=2)
    ev = _events(777001, d, 130)
    assert SS.find_stub_dates(ev, floor=150, until=until,
                              games_path=games, memo_path=memo,
                              probe=set()) == [d]
    SS.record_stub_state(ev, floor=150, scope="bip", games_path=games,
                         memo_path=memo, probe=set())
    assert SS.find_stub_dates(ev, floor=150, until=until,
                              games_path=games, memo_path=memo,
                              probe=set()) == []
    ev2 = _events(777001, d, 145)          # counts changed -> re-flag
    assert SS.find_stub_dates(ev2, floor=150, until=until,
                              games_path=games, memo_path=memo,
                              probe=set()) == [d]


def test_resumed_probe_flags_above_floor_game(tmp_path):
    """A LATE suspension (~220 stored pitches) passes the count floor —
    only the schedule probe can flag it; the memo settles the date once
    the post-refetch record matches. Probe dates without a final in the
    games file are ignored."""
    d = date.today() - timedelta(days=6)
    games = _games_csv(tmp_path, [[777001, str(d)]])
    memo = tmp_path / "memo.json"
    until = date.today() - timedelta(days=2)
    ev = _events(777001, d, 220)
    assert SS.find_stub_dates(ev, floor=150, until=until,
                              games_path=games, memo_path=memo,
                              probe={d}) == [d]
    SS.record_stub_state(ev, floor=150, scope="bip", games_path=games,
                         memo_path=memo, probe={d})
    assert SS.find_stub_dates(ev, floor=150, until=until,
                              games_path=games, memo_path=memo,
                              probe={d}) == []
    stray = date.today() - timedelta(days=8)       # no final that day
    assert SS.find_stub_dates(ev, floor=150, until=until,
                              games_path=games, memo_path=memo,
                              probe={stray}) == []


def test_refetch_merge_is_idempotent_on_key():
    # the guard's refetch relies on the KEY dedupe (keep="last"):
    # refetched rows must REPLACE the stub's rows, never duplicate them
    d = date.today() - timedelta(days=6)
    stub = pd.DataFrame({"GamePk": 777001, "AtBat": range(1, 31),
                         "PitchNum": 1, "Date": d, "ExitVelo": 80.0})
    full = pd.DataFrame({"GamePk": 777001, "AtBat": range(1, 91),
                         "PitchNum": 1, "Date": d, "ExitVelo": 95.0})
    merged = (pd.concat([stub, full], ignore_index=True)
              .drop_duplicates(SS.KEY, keep="last"))
    assert len(merged) == 90
    assert (merged["ExitVelo"] == 95.0).all()
