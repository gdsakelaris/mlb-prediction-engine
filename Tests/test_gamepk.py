"""GamePk identity chain (2026-07-23): the odds store keys on GamePk so a
doubleheader's two games stop overwriting each other, Bets rows carry
G#/GamePk, and Tools/4 settles a DH bet against exactly its own game's
final. These tests load the Tools scripts by file (digit-and-space names
are not importable) and run entirely on tmp_path — the live store and
workbooks are never touched.
"""
import csv
import datetime as dt
import sys
from pathlib import Path

import importlib

_TOOLS = Path(__file__).resolve().parents[1] / "Tools"


def _tool(name):
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    return importlib.import_module(name)


def _row(date="2026-07-23", gamepk="", team="BOS", pid="123",
         market="batter_hits", line=1.5, over=-110, under=-110,
         book="draftkings", captured="2026-07-23T12:00:00"):
    return {"Date": date, "GamePk": gamepk, "Team": team, "PlayerId": pid,
            "PlayerName": "Some Guy", "Market": market, "Line": line,
            "OverPrice": over, "UnderPrice": under, "Book": book,
            "CapturedAt": captured}


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------- write_store

def test_write_store_dh_rows_survive_separately(tmp_path):
    """The 07-23 data-loss bug: game 2's recapture used to overwrite
    game 1's close. With GamePk in the key both games keep their rows."""
    t2 = _tool("2) Scrape Odds")
    out = tmp_path / "odds.csv"
    t2.write_store([_row(gamepk="111", captured="2026-07-23T12:00:00"),
                    _row(gamepk="222", captured="2026-07-23T18:00:00",
                         over=-125)], out)
    rows = _read(out)
    assert len(rows) == 2
    assert {r["GamePk"] for r in rows} == {"111", "222"}


def test_write_store_legacy_open_close_merge_unchanged(tmp_path):
    """Blank-pk rows keep the old key: a recapture merges into one row
    holding the earliest open and the latest close."""
    t2 = _tool("2) Scrape Odds")
    out = tmp_path / "odds.csv"
    t2.write_store([_row(over=-110, captured="2026-07-23T09:00:00")], out)
    t2.write_store([_row(over=-130, captured="2026-07-23T18:00:00")], out)
    rows = _read(out)
    assert len(rows) == 1
    assert rows[0]["OpenCapturedAt"] == "2026-07-23T09:00:00"
    assert rows[0]["OpenOverPrice"] == "-110"
    assert rows[0]["CapturedAt"] == "2026-07-23T18:00:00"
    assert rows[0]["OverPrice"] == "-130"


def test_write_store_transition_folds_legacy_into_lone_pk_sibling(tmp_path):
    """Format transition: a legacy blank-pk open + a pk'd close for the
    same identity merge into ONE pk'd row with open->close continuity."""
    t2 = _tool("2) Scrape Odds")
    out = tmp_path / "odds.csv"
    t2.write_store([_row(over=-110, captured="2026-07-23T09:00:00")], out)
    t2.write_store([_row(gamepk="111", over=-130,
                         captured="2026-07-23T18:00:00")], out)
    rows = _read(out)
    assert len(rows) == 1
    assert rows[0]["GamePk"] == "111"
    assert rows[0]["OpenCapturedAt"] == "2026-07-23T09:00:00"
    assert rows[0]["OverPrice"] == "-130"


def test_write_store_newer_blank_never_folds(tmp_path):
    """A blank captured alongside or AFTER its pk sibling can't be
    attributed (it may be the other DH game under a schedule outage) —
    it survives as its own legacy row and the pk row is untouched."""
    t2 = _tool("2) Scrape Odds")
    out = tmp_path / "odds.csv"
    t2.write_store([_row(gamepk="111", over=-110,
                         captured="2026-07-23T09:00:00")], out)
    t2.write_store([_row(over=-140, captured="2026-07-23T18:00:00")], out)
    rows = _read(out)
    assert len(rows) == 2
    pk_row = next(r for r in rows if r["GamePk"] == "111")
    assert pk_row["OverPrice"] == "-110"           # close NOT overwritten
    blank = next(r for r in rows if r["GamePk"] == "")
    assert blank["OverPrice"] == "-140"


def test_write_store_same_run_blank_never_folds(tmp_path):
    """Same-CapturedAt blank + pk rows (half-resolved DH run) must never
    merge — the strict predates-the-sibling rule blocks equal stamps."""
    t2 = _tool("2) Scrape Odds")
    out = tmp_path / "odds.csv"
    t2.write_store([_row(gamepk="111", over=-110,
                         captured="2026-07-23T12:00:00"),
                    _row(over=-140, captured="2026-07-23T12:00:00")], out)
    rows = _read(out)
    assert len(rows) == 2
    assert sorted(r["GamePk"] for r in rows) == ["", "111"]


def test_write_store_ambiguous_legacy_left_alone_on_dh(tmp_path):
    """A blank-pk row that could belong to either DH game is never
    guessed into one of them — it stays its own legacy row."""
    t2 = _tool("2) Scrape Odds")
    out = tmp_path / "odds.csv"
    t2.write_store([_row(over=-105, captured="2026-07-23T09:00:00")], out)
    t2.write_store([_row(gamepk="111", captured="2026-07-23T12:30:00"),
                    _row(gamepk="222", captured="2026-07-23T18:00:00")], out)
    rows = _read(out)
    assert len(rows) == 3
    assert sorted(r["GamePk"] for r in rows) == ["", "111", "222"]


# -------------------------------------------------------- match_game_pk

def test_match_game_pk():
    t2 = _tool("2) Scrape Odds")
    lone = {("BOS", "NYY"): [(111, "2026-07-23T17:05:00Z")]}
    dh = {("BOS", "NYY"): [(111, "2026-07-23T17:05:00Z"),
                           (222, "2026-07-23T23:10:00Z")]}
    noon = dt.datetime(2026, 7, 23, 17, 0, tzinfo=dt.timezone.utc)
    night = dt.datetime(2026, 7, 23, 23, 0, tzinfo=dt.timezone.utc)
    assert t2.match_game_pk(lone, "BOS", "NYY", None) == "111"
    assert t2.match_game_pk(lone, "BOS", "NYY", noon) == "111"
    assert t2.match_game_pk(dh, "BOS", "NYY", noon) == "111"
    assert t2.match_game_pk(dh, "BOS", "NYY", night) == "222"
    assert t2.match_game_pk(dh, "BOS", "NYY", None) == ""      # ambiguous
    assert t2.match_game_pk(dh, "SEA", "NYY", night) == ""     # unknown


def test_match_game_pk_never_guesses():
    """A stale event for a postponed game (commence far from the lone
    survivor) and an effectively-tied DH must both resolve blank."""
    t2 = _tool("2) Scrape Odds")
    lone = {("BOS", "NYY"): [(111, "2026-07-23T23:10:00Z")]}
    stale = dt.datetime(2026, 7, 23, 13, 0, tzinfo=dt.timezone.utc)
    assert t2.match_game_pk(lone, "BOS", "NYY", stale) == ""   # >6h away
    tied = {("BOS", "NYY"): [(111, "2026-07-23T17:05:00Z"),
                             (222, "2026-07-23T17:05:00Z")]}
    noon = dt.datetime(2026, 7, 23, 17, 0, tzinfo=dt.timezone.utc)
    assert t2.match_game_pk(tied, "BOS", "NYY", noon) == ""    # placeholder
    unparse = {("BOS", "NYY"): [(111, ""), (222, "")]}
    assert t2.match_game_pk(unparse, "BOS", "NYY", noon) == ""


# ----------------------------------------------------- Tools/4 settling

def _settle(row, finals):
    t4 = _tool("4) Grade Results")
    games = {"BOS@NYY": finals}
    return t4._settle_bet(row, {}, {}, games,
                          lambda g, n: None, lambda g, n: None, {}, {})


def _tot(side="Over", line=8.5, **ids):
    return {"Game": "BOS@NYY", "Prop": "total runs", "Side": side,
            "Line": line, **ids}


_F1 = {"total": 9.0, "away": 4.0, "home": 5.0, "gamepk": 111,
       "winner": "NYY"}
_F2 = {"total": 7.0, "away": 6.0, "home": 1.0, "gamepk": 222,
       "winner": "BOS"}


def test_settle_bet_gamepk_pins_the_right_final():
    assert _settle(_tot(GamePk=111), [_F1, _F2]) is True    # 9 > 8.5
    assert _settle(_tot(GamePk=222), [_F1, _F2]) is False   # 7 < 8.5
    assert _settle(_tot(side="Under", GamePk=222), [_F1, _F2]) is True


def test_settle_bet_pk_waits_for_its_own_final():
    # game 2's bet must NOT grade against game 1's lone final
    assert _settle(_tot(GamePk=222), [_F1]) is None
    assert _settle(_tot(GamePk=999), [_F1, _F2]) is None


def test_settle_bet_gnum_fallback_and_legacy_rules():
    # G# ordinal fallback (no pk) on a multi-final day
    assert _settle(_tot(**{"G#": 1}), [_F1, _F2]) is True
    assert _settle(_tot(**{"G#": 2}), [_F1, _F2]) is False
    # G#=2 with only one final is ambiguous -> unsettled
    assert _settle(_tot(**{"G#": 2}), [_F1]) is None
    # legacy row (neither id): one final grades, two stay unsettled
    assert _settle(_tot(), [_F1]) is True
    assert _settle(_tot(), [_F1, _F2]) is None


def test_settle_bet_moneyline_with_pk():
    row = {"Game": "BOS@NYY", "Prop": "moneyline", "Side": "BOS",
           "Line": "", "GamePk": 222}
    assert _settle(row, [_F1, _F2]) is True
    row["GamePk"] = 111
    assert _settle(row, [_F1, _F2]) is False
