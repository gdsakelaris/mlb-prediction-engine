"""Noon capture path: Tools/1 slate helpers (lineup provenance, team maps)
and Tools/2 Odds API event parsing into canonical store rows. Everything
runs on hand-built structures — no network, no Data/ reads or writes.
write_store / match_game_pk are covered in test_gamepk.py, not here.
"""
from conftest import load_tool

T1 = load_tool("1) Get Todays Games")
T2 = load_tool("2) Scrape Odds")


# -------------------------------------------------- Tools/1 classify_side

def test_classify_side_truth_table():
    """Reliance is judged by an actual player change, NOT by reaching 9."""
    cs = T1.classify_side
    full9 = list(range(1, 10))
    # mlb.com posted it, fallbacks changed nothing (full or partial)
    assert cs(full9, full9) == ("mlb", 0, 0, 0, 0)
    assert cs([1, 2, 3], [1, 2, 3]) == ("mlb", 0, 0, 0, 0)
    # nothing from anywhere
    assert cs([], []) == ("none", 0, 0, 0, 0)
    # whole side fallback-sourced; completed only when it reached 9
    assert cs([], full9) == ("full", 1, 1, 0, 1)
    assert cs([], [1, 2, 3, 4, 5]) == ("full", 1, 1, 0, 0)
    # fallback topped up a partial mlb.com lineup
    assert cs([1, 2, 3], full9) == ("top", 1, 0, 1, 1)
    assert cs([1, 2, 3], [1, 2, 3, 4]) == ("top", 1, 0, 1, 0)


def test_classify_side_edges():
    cs = T1.classify_side
    # same size but a player CHANGED still counts as fallback reliance
    assert cs([1, 2, 3], [1, 2, 4]) == ("top", 1, 0, 1, 0)
    # order-insensitive: a reshuffle of the same pids is not a change
    assert cs([3, 2, 1], [1, 2, 3]) == ("mlb", 0, 0, 0, 0)
    # an emptied-out side reads 'none' regardless of what came before
    assert cs([1, 2], []) == ("none", 0, 0, 0, 0)


# -------------------------------------------------- Tools/1 lineup_report

def _game(away, home, n_away, n_home):
    return {"away_team": away, "home_team": home,
            "away_lineup": [[100 + i, i] for i in range(1, n_away + 1)],
            "home_lineup": [[200 + i, i] for i in range(1, n_home + 1)]}


def test_lineup_report_counts_and_incomplete_sides():
    games = [_game("BOS", "NYY", 9, 9), _game("SEA", "TEX", 7, 0)]
    sources = [{"away": "mlb", "home": "mlb"},
               {"away": "top", "home": "none"}]
    lineup_src = {"contributed": 1, "fully": 0, "topped": 1, "completed": 0}
    lines = T1.lineup_report(games, sources, lineup_src)
    assert lines[0] == ("  fallback: contributed to 1/4 side(s) "
                        "(0 fully sourced, 1 topped up); of those, "
                        "0 reached a full 9, 1 still under 9")
    # both under-9 sides listed, tagged with their provenance
    assert len(lines) == 4
    assert "2 side(s) under" in lines[1]
    assert lines[2].split() == ["SEA", "@", "TEX", "away", "SEA",
                                "7/9", "(top)"]
    assert lines[3].split() == ["SEA", "@", "TEX", "home", "TEX",
                                "0/9", "(none)"]


def test_lineup_report_clean_slate_is_one_line():
    games = [_game("BOS", "NYY", 9, 9)]
    sources = [{"away": "mlb", "home": "mlb"}]
    zero = {"contributed": 0, "fully": 0, "topped": 0, "completed": 0}
    lines = T1.lineup_report(games, sources, zero)
    assert len(lines) == 1
    assert "contributed to 0/2 side(s)" in lines[0]


# ------------------------------------------------------ Tools/1 team maps

def test_team_maps_thirty_clubs_and_aliases():
    targets = set(T1.NICKNAME_TO_ABBREV.values())
    assert len(targets) == 30
    # aliases map INTO the canonical set and never collide with it
    assert set(T1.ABBREV_ALIASES.values()) <= targets
    assert not set(T1.ABBREV_ALIASES) & targets
    # both Athletics spellings land on the same club
    assert T1.NICKNAME_TO_ABBREV["A's"] == "ATH"
    assert T1.NICKNAME_TO_ABBREV["Athletics"] == "ATH"


# ------------------------------------------------ Tools/2 parse_event_props

def _resolver(name):
    return (592450, "NYY") if name == "Aaron Judge" else (None, None)


def _wrap(markets, book="draftkings"):
    return {"bookmakers": [{"key": book, "markets": markets}]}


def test_parse_event_props_pairs_over_under_by_player_and_line():
    ev = _wrap([{"key": "batter_hits", "outcomes": [
        {"name": "Over", "description": "Aaron Judge",
         "point": 1.5, "price": -120},
        {"name": "Under", "description": "Aaron Judge",
         "point": 1.5, "price": 100},
        {"name": "Over", "description": "Aaron Judge",
         "point": 0.5, "price": -300},
        {"name": "Under", "description": "Aaron Judge",
         "point": 0.5, "price": 240},
        # resolver returns (None, None): the row must be skipped
        {"name": "Over", "description": "Nobody Nemo",
         "point": 0.5, "price": -110},
    ]}])
    rows = T2.parse_event_props(ev, _resolver, "2026-07-23", "776001",
                                "2026-07-23T12:00:00")
    assert len(rows) == 2      # one row per (player, line, book)
    by_line = {r["Line"]: r for r in rows}
    assert by_line[1.5]["OverPrice"] == -120
    assert by_line[1.5]["UnderPrice"] == 100
    assert by_line[0.5]["OverPrice"] == -300
    assert by_line[0.5]["UnderPrice"] == 240
    for r in rows:
        assert r["GamePk"] == "776001"       # stamped through
        assert r["PlayerId"] == 592450 and r["Team"] == "NYY"
        assert r["PlayerName"] == "Aaron Judge"
        assert r["Market"] == "batter_hits" and r["Book"] == "draftkings"
        assert r["Date"] == "2026-07-23"
        assert r["CapturedAt"] == "2026-07-23T12:00:00"


def test_parse_event_props_alternate_market_stored_under_base():
    ev = _wrap([{"key": "batter_hits_alternate", "outcomes": [
        {"name": "Over", "description": "Aaron Judge",
         "point": 2.5, "price": 320},
    ]}])
    rows = T2.parse_event_props(
        ev, _resolver, "2026-07-23", "", "ts",
        prop_apis=("batter_hits", "batter_hits_alternate"))
    assert len(rows) == 1
    assert rows[0]["Market"] == "batter_hits"    # O.ALT_MARKET normalized
    assert rows[0]["Line"] == 2.5
    assert rows[0]["OverPrice"] == 320 and rows[0]["UnderPrice"] is None
    # PINNED: the DEFAULT prop_apis is the BASE api set only — alternates
    # are captured because main() passes the resolve_markets list. A
    # refactor that starts leaning on the default silently drops every
    # alternate-line capture; this makes that loud.
    assert T2.parse_event_props(ev, _resolver, "2026-07-23", "", "ts") == []


def test_parse_event_props_yes_no_hr_gets_default_line():
    ev = _wrap([{"key": "batter_home_runs", "outcomes": [
        {"name": "Yes", "description": "Aaron Judge", "price": 450},
        {"name": "No", "description": "Aaron Judge", "price": -700},
    ]}])
    rows = T2.parse_event_props(ev, _resolver, "2026-07-23", "776001", "ts")
    assert len(rows) == 1
    assert rows[0]["Line"] == 0.5                # O.API_DEFAULT_LINE
    assert rows[0]["OverPrice"] == 450 and rows[0]["UnderPrice"] == -700
    assert rows[0]["Market"] == "batter_home_runs"


# ------------------------------------------------ Tools/2 parse_event_games

def test_parse_event_games_totals_require_both_sides():
    ev = _wrap([{"key": "totals", "outcomes": [
        {"name": "Over", "point": 8.5, "price": -110},
        {"name": "Under", "point": 8.5, "price": -105},
        {"name": "Over", "point": 9.5, "price": 120},   # no under: dropped
    ]}], book="fanduel")
    rows = T2.parse_event_games(ev, "NYY", "BOS", "2026-07-23", "776001",
                                "ts")
    assert len(rows) == 1
    r = rows[0]
    assert r["Market"] == "totals" and r["Line"] == 8.5
    assert r["OverPrice"] == -110 and r["UnderPrice"] == -105
    assert r["Team"] == "NYY"                   # totals key to the home club
    assert r["PlayerId"] == "" and r["GamePk"] == "776001"
    assert r["Book"] == "fanduel"


def test_parse_event_games_h2h_home_away_prices():
    ev = _wrap([{"key": "h2h", "outcomes": [
        {"name": "Boston Red Sox", "price": 130},
        {"name": "New York Yankees", "price": -150},
    ]}])
    rows = T2.parse_event_games(ev, "NYY", "BOS", "2026-07-23", "", "ts")
    assert len(rows) == 1
    assert rows[0]["OverPrice"] == -150         # home price
    assert rows[0]["UnderPrice"] == 130         # away price
    assert rows[0]["Team"] == "NYY" and rows[0]["Line"] is None


def test_parse_event_games_team_totals_and_alternates():
    ev = _wrap([
        {"key": "team_totals", "outcomes": [
            {"name": "Over", "description": "New York Yankees",
             "point": 4.5, "price": -115},
            {"name": "Under", "description": "New York Yankees",
             "point": 4.5, "price": -108},
            # BOS has only an Over posted: half-pair, dropped
            {"name": "Over", "description": "Boston Red Sox",
             "point": 3.5, "price": -120},
        ]},
        {"key": "alternate_totals", "outcomes": [
            {"name": "Over", "point": 10.5, "price": 200},
            {"name": "Under", "point": 10.5, "price": -250},
        ]},
    ])
    rows = T2.parse_event_games(ev, "NYY", "BOS", "2026-07-23", "776001",
                                "ts", keys=T2.EVENT_GAME_APIS)
    tt = [r for r in rows if r["Market"] == "team_totals"]
    assert len(tt) == 1
    assert tt[0]["Team"] == "NYY" and tt[0]["Line"] == 4.5  # the named club
    assert tt[0]["OverPrice"] == -115 and tt[0]["UnderPrice"] == -108
    alt = [r for r in rows if r["Market"] == "totals"]
    assert len(alt) == 1                # alternate_totals -> 'totals'
    assert alt[0]["Line"] == 10.5
    assert alt[0]["OverPrice"] == 200 and alt[0]["UnderPrice"] == -250
    assert len(rows) == 2


# --------------------------------------------- Tools/2 full_name_to_abbrev

def test_full_name_to_abbrev_known_and_unknown():
    f = T2.full_name_to_abbrev
    assert f("New York Yankees") == "NYY"
    assert f("Boston Red Sox") == "BOS"
    assert f("Chicago White Sox") == "CWS"
    assert f("Arizona Diamondbacks") == "AZ"
    assert f("St. Louis Cardinals") == "STL"
    assert f("Athletics") == "ATH"              # no city prefix since 2025
    assert f("Springfield Isotopes") is None
    assert f("") is None
