"""Scraper gap-visibility fixes (2026-07-25 audit): games at venues with
no coordinates must write null-weather rows (visible + retried daily)
instead of being skipped forever, and FINAL games whose endpoint payload
is permanently empty must land in a negative cache
(Data/.empty_finals.json) instead of being refetched every run inside
"0 fetch failures". Pure seams plus main() wiring, all offline — network
functions are monkeypatched and every CSV lives in tmp_path.
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scrapers"))

import scrape_linescores as L    # noqa: E402
import scrape_pbp as P           # noqa: E402
import scrape_weather as W       # noqa: E402


# ------------------------------------------- weather: unmatched venues

def test_field_of_dreams_has_coordinates():
    lat, lon = W.EXTRA_VENUES["Field of Dreams"]
    assert abs(lat - 42.4958) < 0.01
    assert abs(lon + 91.0545) < 0.01


def test_null_weather_rows_cover_unmatched_and_nan_venues():
    games = pd.DataFrame({
        "GamePk": [111, 222],
        "Date": pd.to_datetime(["2021-08-12", "2021-08-12"]),
        "DayNight": ["night", "day"],
        "Venue": ["Nowhere Park", float("nan")],
    })
    rows = W.null_weather_rows(games)
    assert [r["GamePk"] for r in rows] == [111, 222]
    assert all(r["Date"] == "2021-08-12" for r in rows)
    assert all(r["Humidity"] is None and r["Pressure"] is None
               and r["Precip"] is None for r in rows)


def test_weather_main_writes_null_rows_for_unmatched_venue(
        tmp_path, monkeypatch, capsys):
    pd.DataFrame({
        "GamePk": [660271, 662024],
        "Date": ["2021-08-12", "2022-08-11"],
        "DayNight": ["night", "night"],
        "Venue": ["Bermuda Triangle Field", "Bermuda Triangle Field"],
    }).to_csv(tmp_path / "mlb_games.csv", index=False)
    pd.DataFrame({"Ballpark": ["Real Park"], "Lat": [40.0],
                  "Lon": [-75.0]}).to_csv(
        tmp_path / "mlb_ballparks.csv", index=False)
    out = tmp_path / "weather.csv"

    def no_network(*a, **k):
        raise AssertionError("unmatched-venue run must not hit Open-Meteo")

    monkeypatch.setattr(W, "GAMES_CSV", tmp_path / "mlb_games.csv")
    monkeypatch.setattr(W, "PARKS_CSV", tmp_path / "mlb_ballparks.csv")
    monkeypatch.setattr(W, "fetch_hourly", no_network)
    monkeypatch.setattr(sys, "argv", ["scrape_weather.py", "-o", str(out)])
    W.main()

    got = pd.read_csv(out, encoding="utf-8-sig")
    assert sorted(got["GamePk"]) == [660271, 662024]   # visible, not skipped
    assert got["Humidity"].isna().all()
    assert got["Pressure"].isna().all()
    assert got["Precip"].isna().all()
    warned = capsys.readouterr().out
    assert "Bermuda Triangle Field" in warned          # names the venue
    assert "EXTRA_VENUES" in warned                    # says how to fix it


# --------------------------------- negative cache: pure transitions

def test_cache_add_reaches_skip_threshold(tmp_path):
    c = L.EmptyFinalsCache("linescores", tmp_path / ".empty_finals.json")
    for _ in range(L.EMPTY_MAX_TRIES - 1):
        c.record_empty(746402)
        assert not c.known_empty(746402)
    c.record_empty(746402)
    assert c.known_empty(746402)


def test_cache_persists_across_instances(tmp_path):
    p = tmp_path / ".empty_finals.json"
    c = L.EmptyFinalsCache("linescores", p)
    for _ in range(L.EMPTY_MAX_TRIES):
        c.record_empty(746402)
    c.save()
    again = L.EmptyFinalsCache("linescores", p)
    assert again.known_empty(746402)
    assert not again.known_empty(999999)


def test_cache_self_heals_when_data_returns(tmp_path):
    p = tmp_path / ".empty_finals.json"
    c = L.EmptyFinalsCache("linescores", p)
    for _ in range(L.EMPTY_MAX_TRIES):
        c.record_empty(746402)
    c.record_data(746402)
    assert not c.known_empty(746402)
    c.save()
    assert L.EmptyFinalsCache("linescores", p).counts == {}


def test_cache_scopes_are_independent_and_save_remerges(tmp_path):
    p = tmp_path / ".empty_finals.json"
    a = L.EmptyFinalsCache("linescores", p)
    for _ in range(L.EMPTY_MAX_TRIES):
        a.record_empty(746402)
    a.save()
    b = L.EmptyFinalsCache("pbp", p)
    assert not b.known_empty(746402)         # other scope's verdicts unseen
    b.record_empty(745873)
    b.save()                                 # must re-merge, not overwrite
    assert L.EmptyFinalsCache("linescores", p).known_empty(746402)


def test_cache_tolerates_missing_and_corrupt_file(tmp_path):
    p = tmp_path / ".empty_finals.json"
    assert L.EmptyFinalsCache("linescores", p).counts == {}
    p.write_text("{not json", encoding="utf-8")
    assert L.EmptyFinalsCache("linescores", p).counts == {}


def test_game_age_days():
    old = (date.today() - timedelta(days=10)).isoformat()
    assert L.game_age_days(old) == 10
    assert L.game_age_days("garbage") == -1


# ------------------------------------- negative cache: main() wiring

def _games_csv(tmp_path, pks, day):
    pd.DataFrame({"GamePk": pks, "Date": [day] * len(pks),
                  "Season": [int(day[:4])] * len(pks)}).to_csv(
        tmp_path / "mlb_games.csv", index=False)


def _saved(tmp_path):
    return json.loads(
        (tmp_path / ".empty_finals.json").read_text(encoding="utf-8"))


def _run_linescores(tmp_path, monkeypatch, fetch, backfill=False):
    monkeypatch.setattr(L, "GAMES_CSV", tmp_path / "mlb_games.csv")
    monkeypatch.setattr(L, "EMPTY_CACHE", tmp_path / ".empty_finals.json")
    monkeypatch.setattr(L, "linescore", fetch)
    argv = ["scrape_linescores.py", "-o", str(tmp_path / "ls.csv"),
            "--sleep", "0"]
    if backfill:
        argv.append("--backfill")
    monkeypatch.setattr(sys, "argv", argv)
    L.main()


def test_linescores_empty_final_cached_then_skipped(
        tmp_path, monkeypatch, capsys):
    _games_csv(tmp_path, [746402], "2024-06-01")
    for _ in range(L.EMPTY_MAX_TRIES):
        _run_linescores(tmp_path, monkeypatch, lambda pk: [])
    assert _saved(tmp_path)["linescores"]["746402"] == L.EMPTY_MAX_TRIES

    calls = []
    _run_linescores(tmp_path, monkeypatch,
                    lambda pk: calls.append(pk) or [])
    assert calls == []                       # never refetched
    assert "1 known-empty finals skipped" in capsys.readouterr().out


def test_linescores_backfill_refetches_and_heals_cache(
        tmp_path, monkeypatch):
    _games_csv(tmp_path, [746402], "2024-06-01")
    for _ in range(L.EMPTY_MAX_TRIES):
        _run_linescores(tmp_path, monkeypatch, lambda pk: [])
    _run_linescores(tmp_path, monkeypatch, lambda pk: [(1, 0, 1)],
                    backfill=True)
    assert _saved(tmp_path)["linescores"] == {}          # entry dropped
    got = pd.read_csv(tmp_path / "ls.csv", encoding="utf-8-sig")
    assert list(got["GamePk"]) == [746402]


def test_linescores_young_empty_final_never_cached(tmp_path, monkeypatch):
    _games_csv(tmp_path, [746402], date.today().isoformat())
    calls = []
    for _ in range(L.EMPTY_MAX_TRIES + 1):
        _run_linescores(tmp_path, monkeypatch,
                        lambda pk: calls.append(pk) or [])
    assert _saved(tmp_path).get("linescores", {}) == {}
    assert len(calls) == L.EMPTY_MAX_TRIES + 1           # still retried


def _run_pbp(tmp_path, monkeypatch, fetch, backfill=False):
    monkeypatch.setattr(P, "GAMES_CSV", tmp_path / "mlb_games.csv")
    monkeypatch.setattr(P, "EMPTY_CACHE", tmp_path / ".empty_finals.json")
    monkeypatch.setattr(P, "get_json", fetch)
    argv = ["scrape_pbp.py", "-o", str(tmp_path / "pbp.csv"),
            "--workers", "1"]
    if backfill:
        argv.append("--backfill")
    monkeypatch.setattr(sys, "argv", argv)
    P.main()


def test_pbp_empty_final_cached_then_skipped(tmp_path, monkeypatch, capsys):
    _games_csv(tmp_path, [745873], "2024-06-01")
    for _ in range(L.EMPTY_MAX_TRIES):
        _run_pbp(tmp_path, monkeypatch, lambda url: {"allPlays": []})
    assert _saved(tmp_path)["pbp"]["745873"] == L.EMPTY_MAX_TRIES

    calls = []
    _run_pbp(tmp_path, monkeypatch,
             lambda url: calls.append(url) or {"allPlays": []})
    assert calls == []                       # never refetched
    assert "1 known-empty finals skipped" in capsys.readouterr().out


def test_pbp_backfill_refetches_and_heals_cache(tmp_path, monkeypatch):
    _games_csv(tmp_path, [745873], "2024-06-01")
    for _ in range(L.EMPTY_MAX_TRIES):
        _run_pbp(tmp_path, monkeypatch, lambda url: {"allPlays": []})
    payload = {"allPlays": [{
        "result": {"event": "Strikeout", "eventType": "strikeout"},
        "about": {"inning": 1, "halfInning": "top"},
        "matchup": {"batter": {"id": 1}, "pitcher": {"id": 2}},
        "atBatIndex": 0,
        "runners": [{
            "movement": {"start": None, "end": None, "isOut": True,
                         "outBase": "1B", "outNumber": 1},
            "details": {"playIndex": 0, "runner": {"id": 1},
                        "event": "Strikeout", "eventType": "strikeout"},
        }],
    }]}
    _run_pbp(tmp_path, monkeypatch, lambda url: payload, backfill=True)
    assert _saved(tmp_path)["pbp"] == {}                 # entry dropped
    got = pd.read_csv(tmp_path / "pbp.csv", encoding="utf-8-sig")
    assert list(got["GamePk"].unique()) == [745873]
