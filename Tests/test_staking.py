"""Paper-trading staking ledger (Model/staking.py) against the REAL
Data/ schema: §3-§5 math on a marked-to-market bankroll, the
append-only supersede lifecycle (originals stamped Outcome='superseded'
and dead to settlement — PnL counted once), settlement over CSVs that
mirror the actual mlb_games / mlb_game_batting / mlb_game_pitching
headers (IP, no OUTS column — outs are derived), the string-key dtype
round trip that once silently broke game-market supersedes, DNP voiding
on the next-day sweep, and the CLV backfill joins (totals via Game
label + GamePk-exact, away-side moneyline via the home-keyed store).

Every test monkeypatches ART, LEDGER, DATA and O.DEFAULT_STORE — the
old suite left LEDGER live in two enrich tests, so _bankroll() read the
REAL project ledger and the stake assertions would break the day real
PnL accrued.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate as EV
import odds as O
import staking as SK

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

MKT_FAM = {"batter_hits": "h", "pitcher_strikeouts": "pk",
           "pitcher_outs": "pout", "pitcher_earned_runs": "per",
           "totals": "tot", "h2h": "ml", "team_totals": "tt"}

DATE = "2026-07-24"

# ------------------------------------------------------------ fixtures
# column lists mirror the REAL Data/ headers exactly (guarded by
# test_fixture_headers_mirror_real_schema below): pitching has IP and
# no OUTS column — settle() must derive outs, not crash on r.OUTS.

GAMES_COLS = ["GamePk", "Season", "Date", "DayNight", "AwayTeam",
              "HomeTeam", "AwayScore", "HomeScore", "Venue", "Temp",
              "Condition", "WindSpeed", "WindDir", "GameType"]
BAT_COLS = ["GamePk", "Season", "Date", "PlayerId", "Name", "Team",
            "Opponent", "Home", "BattingOrder", "Position", "PA", "AB",
            "R", "H", "2B", "3B", "HR", "RBI", "BB", "IBB", "SO", "HBP",
            "SB", "CS", "SAC", "SF", "GIDP", "TB", "LOB"]
PIT_COLS = ["GamePk", "Season", "Date", "PlayerId", "Name", "Team",
            "Opponent", "Home", "GS", "GF", "IP", "BF", "NP", "Strikes",
            "H", "R", "ER", "HR", "BB", "IBB", "SO", "HBP", "WP", "BK",
            "W", "L", "SV", "HLD"]


def _game_row(pk, away="BOS", home="NYY", asc=4, hsc=5, date=DATE):
    return dict(GamePk=pk, Season=2026, Date=date, DayNight="night",
                AwayTeam=away, HomeTeam=home, AwayScore=asc,
                HomeScore=hsc, Venue="Yankee Stadium", Temp=78,
                Condition="Clear", WindSpeed=8, WindDir="Out To CF",
                GameType="R")


def _bat_row(pk, pid, h=1, pa=4, date=DATE, **kw):
    r = dict(GamePk=pk, Season=2026, Date=date, PlayerId=pid,
             Name=f"Bat {pid}", Team="BOS", Opponent="NYY", Home=0,
             BattingOrder=100, Position="LF", PA=pa, AB=pa, R=0, H=h,
             RBI=0, BB=0, IBB=0, SO=1, HBP=0, SB=0, CS=0, SAC=0, SF=0,
             GIDP=0, TB=h, LOB=2, HR=0, **{"2B": 0, "3B": 0})
    r.update(kw)
    return r


def _pit_row(pk, pid, ip="5.2", so=7, er=2, date=DATE, **kw):
    r = dict(GamePk=pk, Season=2026, Date=date, PlayerId=pid,
             Name=f"Arm {pid}", Team="NYY", Opponent="BOS", Home=1,
             GS=1, GF=0, IP=ip, BF=24, NP=95, Strikes=62, H=5, R=2,
             ER=er, HR=1, BB=2, IBB=0, SO=so, HBP=0, WP=0, BK=0, W=1,
             L=0, SV=0, HLD=0)
    r.update(kw)
    return r


def _write_data(tmp, games=(), bats=(), pits=()):
    pd.DataFrame(list(games), columns=GAMES_COLS).to_csv(
        tmp / "mlb_games.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(list(bats), columns=BAT_COLS).to_csv(
        tmp / "mlb_game_batting.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(list(pits), columns=PIT_COLS).to_csv(
        tmp / "mlb_game_pitching.csv", index=False,
        encoding="utf-8-sig")


def _store_row(market, line, team="", pid="", over="", under="",
               book="pinnacle", pk="", date=DATE):
    return dict(Date=date, GamePk=pk, Team=team, PlayerId=pid,
                PlayerName="", Market=market, Line=line,
                OverPrice=over, UnderPrice=under, Book=book,
                CapturedAt=f"{date}T18:55:00", OpenOverPrice=over,
                OpenUnderPrice=under, OpenCapturedAt=f"{date}T10:00:00")


def _write_store(tmp, rows):
    pd.DataFrame(rows, columns=O.ODDS_COLUMNS).to_csv(
        tmp / "mlb_odds.csv", index=False, encoding="utf-8-sig")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fully hermetic staking environment: no test may ever touch the
    real ledger, gate report, Data/ CSVs or odds store."""
    monkeypatch.setattr(SK, "ART", tmp_path)
    monkeypatch.setattr(SK, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(SK, "DATA", tmp_path)
    monkeypatch.setattr(SK.O, "DEFAULT_STORE", tmp_path / "mlb_odds.csv")
    return tmp_path


def _bet(market="batter_hits", pid=123, side="Over", line=1.5,
         odds=120, pm=0.55, pc=0.50, gpk=777, team="BOS", player="Some Guy",
         game="BOS@NYY"):
    return {"Game": game, "G#": 1, "GamePk": gpk, "PlayerId": pid,
            "Player": player, "Team": team, "Prop": "hits o1.5",
            "Side": side, "Line": line, "Model %": pm, "Mkt %": pc,
            "Best Odds": odds, "Book": "draftkings", "EV%": 0.1,
            "Books": 3, "_market": market}


def _tot_bet(side="Over", line=8.5, odds=100, pm=0.60, pc=0.55, gpk=777,
             game="BOS@NYY"):
    return {"Game": game, "G#": 1, "GamePk": gpk, "PlayerId": None,
            "Player": "", "Team": "", "Prop": "total runs", "Side": side,
            "Line": line, "Model %": pm, "Mkt %": pc, "Best Odds": odds,
            "Book": "pinnacle", "EV%": 0.1, "Books": 3,
            "_market": "totals"}


def _ml_bet(team="NYY", odds=110, pm=0.58, pc=0.55, gpk=777,
            game="BOS@NYY"):
    return {"Game": game, "G#": 1, "GamePk": gpk, "PlayerId": None,
            "Player": "", "Team": team, "Prop": "moneyline", "Side": team,
            "Line": "", "Model %": pm, "Mkt %": pc, "Best Odds": odds,
            "Book": "pinnacle", "EV%": 0.1, "Books": 3,
            "_market": "h2h"}


# ------------------------------------------------- real-schema tripwire

@pytest.mark.parametrize("fname,cols", [
    ("mlb_games.csv", GAMES_COLS),
    ("mlb_game_batting.csv", BAT_COLS),
    ("mlb_game_pitching.csv", PIT_COLS),
])
def test_fixture_headers_mirror_real_schema(fname, cols):
    """The fixtures ARE the real schema — if a scraper migration renames
    a column, this fails before a green suite can certify settle() logic
    against a shape production never produces."""
    p = DATA_DIR / fname
    if not p.exists():
        pytest.skip(f"no {fname} on this machine")
    with p.open(encoding="utf-8-sig") as fh:
        header = fh.readline().strip().split(",")
    assert header == cols


def test_pitching_schema_has_ip_not_outs():
    # the defect class this file exists to pin: settlement must derive
    # outs from IP because the real CSV has no OUTS column
    assert "OUTS" not in PIT_COLS and "IP" in PIT_COLS


def test_with_outs_derivation():
    gp = pd.DataFrame({"IP": ["5.2", "6.0", "0.1", ""]})
    out = SK._with_outs(gp)
    assert list(out.OUTS) == [17, 18, 1, 0]
    # a frame already carrying OUTS (evaluate's) passes through as-is
    gp2 = pd.DataFrame({"IP": ["5.2"], "OUTS": [3]})
    assert list(SK._with_outs(gp2).OUTS) == [3]


# ------------------------------------------------------------ §3-§5

def test_enrich_math_and_track_status(env):
    rows = SK.enrich([_bet()], DATE, MKT_FAM)      # no gate report
    assert len(rows) == 1
    r = rows[0]
    # +120 -> implied 100/220, dec 2.2; p_bet = .5*.55 + .5*.50 = .525
    assert r["implied"] == pytest.approx(100 / 220, abs=1e-6)
    assert r["p_bet"] == pytest.approx(0.525, abs=1e-6)
    assert r["edge"] == pytest.approx(0.525 - 100 / 220, abs=1e-6)
    assert r["EV"] == pytest.approx(0.525 * 2.2 - 1, abs=1e-6)
    assert r["f_kelly"] == pytest.approx(
        0.25 * (0.525 * 2.2 - 1) / 1.2, abs=1e-6)
    # no PASS family anywhere -> tracked, zero stake, reason recorded
    assert r["Status"] == "track"
    assert r["stake_units"] == 0.0
    assert r["stake_capped_by"] == "family-not-PASS"


def test_enrich_pass_family_stakes_with_per_bet_cap(env):
    (env / "market_gate_report.csv").write_text(
        "family,verdict\nh,PASS\npk,NO-EDGE\n")
    rows = SK.enrich([_bet(pm=0.62, pc=0.58),          # big edge -> cap
                      _bet(market="pitcher_strikeouts", pid=99,
                           pm=0.62, pc=0.58)], DATE, MKT_FAM)
    by_mkt = {r["Market"]: r for r in rows}
    h = by_mkt["batter_hits"]
    assert h["Status"] == "paper"
    assert h["stake_capped_by"] == "per-bet-1%"
    assert h["stake_units"] == pytest.approx(0.01 * SK.START_BANKROLL)
    assert by_mkt["pitcher_strikeouts"]["Status"] == "track"


def test_enrich_stake_band_longshot_and_single_book(env):
    """§3.5-§3.6 (2026-07-26 amendment): a stake additionally requires
    implied probability >= 0.20 and a two-book quote. A +1000-class
    price clears every EV screen on tail miscalibration alone, and a
    single stale quote is line-shopping winner's curse, not a market —
    both still RECORD as track rows so tail evidence accrues."""
    (env / "market_gate_report.csv").write_text(
        "family,verdict\nh,PASS\n")
    long = _bet(odds=1000, pm=0.15, pc=0.12)   # edge .044, EV +.49
    single = _bet(pid=124, pm=0.55, pc=0.50)   # in-band, one book
    single["Books"] = 1
    rows = SK.enrich([long, single], DATE, MKT_FAM)
    by = {r["PlayerId"]: r for r in rows}
    assert by[123]["Status"] == "track"
    assert by[123]["stake_capped_by"] == "longshot-band"
    assert float(by[123]["stake_units"]) == 0.0
    assert by[124]["Status"] == "track"
    assert by[124]["stake_capped_by"] == "single-book"
    assert float(by[124]["stake_units"]) == 0.0


def test_enrich_stakes_mark_to_market_bankroll(env):
    """§5: f applies to the CURRENT bankroll (START + settled PnL), not
    the START constant — in drawdown stakes must shrink."""
    (env / "market_gate_report.csv").write_text("family,verdict\nh,PASS\n")
    led = pd.DataFrame([dict.fromkeys(SK.LEDGER_COLS, "")])
    led["Status"], led["Outcome"], led["PnL_units"] = \
        "paper", "lose", "-30.0"
    led[SK.LEDGER_COLS].to_csv(SK.LEDGER, index=False,
                               encoding="utf-8-sig")
    rows = SK.enrich([_bet(pm=0.62, pc=0.58)], DATE, MKT_FAM)
    assert rows[0]["stake_units"] == pytest.approx(
        0.01 * (SK.START_BANKROLL - 30.0))


def test_enrich_never_both_sides(env):
    rows = SK.enrich([_bet(side="Over", odds=110),
                      _bet(side="Under", odds=250, pm=0.45, pc=0.50)],
                     DATE, MKT_FAM)
    assert len(rows) == 1                     # higher-EV side survives
    assert rows[0]["Side"] == "Under"


def test_enrich_moneyline_sides_collide(env):
    """§3: h2h Team encodes the SIDE, so the two moneyline sides of one
    game are ONE bet identity — enrich may never record both."""
    rows = SK.enrich([_ml_bet(team="NYY", odds=105, pm=0.55, pc=0.52),
                      _ml_bet(team="BOS", odds=115, pm=0.48, pc=0.50)],
                     DATE, MKT_FAM)
    assert len(rows) == 1
    # NYY EV .535*2.05-1 = .097 beats BOS .49*2.15-1 = .054
    assert rows[0]["Team"] == "NYY"           # the higher-EV side


# --------------------------------------------------- key normalization

def test_key_norm_canonicalizes_csv_round_trip_values():
    assert SK._key_norm(777) == SK._key_norm(777.0) \
        == SK._key_norm("777.0") == SK._key_norm("777") == "777"
    assert SK._key_norm(1.5) == SK._key_norm("1.50") == "1.5"
    for blank in (None, "", "nan", "None", "<NA>", float("nan")):
        assert SK._key_norm(blank) == ""


def test_bet_key_is_side_agnostic_and_team_blind_for_h2h():
    a = {"Date": DATE, "GamePk": 777, "PlayerId": 5, "Team": "BOS",
         "Market": "batter_hits", "Line": 1.5, "Side": "Over"}
    b = dict(a, Side="Under")
    assert SK._bet_key(a) == SK._bet_key(b)
    hm = {"Date": DATE, "GamePk": 777, "PlayerId": None, "Team": "NYY",
          "Market": "h2h", "Line": "", "Side": "NYY"}
    aw = dict(hm, Team="BOS", Side="BOS")
    assert SK._bet_key(hm) == SK._bet_key(aw)
    # team totals: Team IS the identity and must stay in the key
    t1 = dict(hm, Market="team_totals", Line=3.5, Side="Over")
    t2 = dict(t1, Team="BOS")
    assert SK._bet_key(t1) != SK._bet_key(t2)


def test_bet_key_survives_disk_round_trip_for_game_markets(env):
    """The 2026-07-25 critical: pd.read_csv turned blank Team/Line into
    NaN while fresh rows carried ''/None strings, so totals and h2h
    supersede keys never matched. The key must be identical computed
    from the fresh dict and from the reloaded CSV row — totals (blank
    Team), h2h (blank Line), and float-artifact PlayerIds alike."""
    bets = [_bet(pid=42), _tot_bet(), _ml_bet()]
    rows = SK.enrich(bets, DATE, MKT_FAM)
    SK.append(rows)
    led = SK._load()
    fresh = {SK._bet_key(r) for r in rows}
    loaded = {SK._bet_key(led.loc[i]) for i in led.index}
    assert fresh == loaded
    # and PlayerId '42' vs '42.0' can never split the key
    assert SK._bet_key(dict(rows[0], PlayerId="42.0")) \
        == SK._bet_key(rows[0])


# ------------------------------------------------- supersede lifecycle

def test_append_supersede_is_append_only(env):
    r1 = SK.enrich([_bet(odds=110)], DATE, MKT_FAM)
    SK.append(r1)
    r2 = SK.enrich([_bet(odds=130)], DATE, MKT_FAM)
    SK.append(r2)
    led = pd.read_csv(SK.LEDGER, dtype=str, keep_default_na=False)
    # 3 rows: original, its void marker, the re-serve — never an edit
    # of money fields; the ORIGINAL is stamped Outcome='superseded'
    assert len(led) == 3
    assert list(led.Status) == ["track", "void", "track"]
    assert led.iloc[0]["Outcome"] == "superseded"
    assert led.iloc[1]["stake_capped_by"] == "superseded-by-re-serve"
    assert led.iloc[2]["PriceAmerican"] == "130"
    assert led.iloc[2]["Outcome"] == ""


def test_supersede_matches_game_market_rows_after_reload(env):
    """Totals/moneyline rows re-served across a disk round trip must
    supersede too (the NaN-vs-'' key bug left them stacking as live
    duplicates with no marker at all)."""
    SK.append(SK.enrich([_tot_bet(odds=100), _ml_bet(odds=105)],
                        DATE, MKT_FAM))
    SK.append(SK.enrich([_tot_bet(odds=110), _ml_bet(odds=115)],
                        DATE, MKT_FAM))
    led = SK._load()
    assert int((led.Status == "void").sum()) == 2      # both markers
    live = led[led.Status.isin(("paper", "track")) & (led.Outcome == "")]
    assert len(live) == 2                              # only gen 2
    assert set(live.PriceAmerican) == {"110", "115"}
    sup = led[led.Outcome == "superseded"]
    assert len(sup) == 2
    assert set(sup.PriceAmerican) == {"100", "105"}


def test_supersede_retires_flipped_side(env):
    """§3 side-agnostic identity: when the EV side flips between serves
    (Over -> Under), the abandoned Over must retire — otherwise both
    sides of one market settle as a hedged pair."""
    SK.append(SK.enrich([_bet(side="Over", odds=105)], DATE, MKT_FAM))
    SK.append(SK.enrich([_bet(side="Under", odds=115, pm=0.45,
                              pc=0.48)], DATE, MKT_FAM))
    led = SK._load()
    over = led[(led.Side == "Over") & (led.Status == "track")]
    assert list(over.Outcome) == ["superseded"]
    live = led[led.Status.isin(("paper", "track")) & (led.Outcome == "")]
    assert list(live.Side) == ["Under"]


def test_superseded_original_never_settles_pnl_counted_once(env):
    """The critical double-count: settle() must grade ONLY the newest
    generation. One real-world bet, re-served once, wins -> exactly one
    settled row, one stake's PnL."""
    (env / "market_gate_report.csv").write_text("family,verdict\nh,PASS\n")
    SK.append(SK.enrich([_bet(pid=1, odds=100, pm=0.62, pc=0.58)],
                        DATE, MKT_FAM))
    SK.append(SK.enrich([_bet(pid=1, odds=120, pm=0.62, pc=0.58)],
                        DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777)],
                bats=[_bat_row(777, 1, h=2)])
    SK.settle(DATE)
    led = SK._load()
    settled = led[led.Outcome.isin(("win", "lose", "push"))]
    assert len(settled) == 1                      # ONE bet, once
    assert settled.PriceAmerican.iloc[0] == "120"
    stake = float(settled.stake_units.iloc[0])
    assert stake == pytest.approx(0.01 * SK.START_BANKROLL)
    pnl = pd.to_numeric(led.PnL_units, errors="coerce").sum()
    assert pnl == pytest.approx(stake * 1.2)      # +120 win pays 1.2x
    # the superseded original keeps its stamp and never re-enters
    assert (led.Outcome == "superseded").sum() == 1
    SK.settle(DATE)                               # idempotent
    led2 = SK._load()
    assert pd.to_numeric(led2.PnL_units,
                         errors="coerce").sum() == pytest.approx(pnl)
    assert len(led2) == len(led)


def test_reserve_after_settlement_never_double_settles(env):
    """The other double-count: re-serving a date whose rows already
    SETTLED. The live-only supersede scan can't see settled rows, so
    the old append() stacked a fresh LIVE duplicate of a graded bet and
    the next settle counted its PnL twice into the bankroll. A settled
    §3 identity must skip outright; genuinely new bets still append."""
    (env / "market_gate_report.csv").write_text("family,verdict\nh,PASS\n")
    _write_data(env, games=[_game_row(777)],
                bats=[_bat_row(777, 1, h=2), _bat_row(777, 2, h=0)])
    SK.append(SK.enrich([_bet(pid=1, odds=120, pm=0.62, pc=0.58)],
                        DATE, MKT_FAM))
    SK.settle(DATE)                               # pid 1 wins, +1.2u
    n = SK.append(SK.enrich([_bet(pid=1, odds=130, pm=0.62, pc=0.58),
                             _bet(pid=2, odds=100, pm=0.62, pc=0.58)],
                            DATE, MKT_FAM))
    assert n == 1                     # pid 1 skipped, pid 2 appended
    SK.settle(DATE)
    led = SK._load()
    settled = led[led.Outcome.isin(("win", "lose", "push"))]
    assert sorted(SK._key_norm(p) for p in settled.PlayerId) == ["1", "2"]
    one = settled[settled.PlayerId.map(SK._key_norm) == "1"]
    assert list(one.PriceAmerican) == ["120"]     # gen 1, once
    stake1 = float(one.stake_units.iloc[0])
    stake2 = float(settled[settled.PlayerId.map(SK._key_norm) == "2"]
                   .stake_units.iloc[0])
    pnl = pd.to_numeric(led.PnL_units, errors="coerce").sum()
    assert pnl == pytest.approx(stake1 * 1.2 - stake2)
    assert (led.Outcome == "superseded").sum() == 0


def test_settle_retires_live_duplicate_of_settled_identity(env):
    """A pre-fix ledger may already hold a live duplicate of a settled
    identity (appended before append() learned to skip them): settle()
    must retire it as superseded, never grade it a second time."""
    (env / "market_gate_report.csv").write_text("family,verdict\nh,PASS\n")
    _write_data(env, games=[_game_row(777)], bats=[_bat_row(777, 1, h=2)])
    SK.append(SK.enrich([_bet(pid=1, odds=120, pm=0.62, pc=0.58)],
                        DATE, MKT_FAM))
    SK.settle(DATE)
    led = SK._load()
    dup = led.iloc[0].copy()          # the poisoned live duplicate
    for c in ("Outcome", "PnL_units", "SettledAt", "CloseAmerican",
              "p_close_final", "CLV"):
        dup[c] = ""
    led = pd.concat([led, dup.to_frame().T], ignore_index=True)
    SK._write(led[SK.LEDGER_COLS])
    SK.settle(DATE)
    led2 = SK._load()
    assert int(led2.Outcome.isin(("win", "lose", "push")).sum()) == 1
    assert int((led2.Outcome == "superseded").sum()) == 1
    stake = float(led2[led2.Outcome == "win"].stake_units.iloc[0])
    pnl = pd.to_numeric(led2.PnL_units, errors="coerce").sum()
    assert pnl == pytest.approx(stake * 1.2)      # counted once


# ------------------------------------------------- settlement (real CSVs)

def test_settle_pitcher_rows_from_real_pitching_schema(env):
    """Pitcher ladders settle off the IP-only real schema: IP 5.2 ->
    17 outs (evaluate's exact derivation). A mixed batch (batter +
    pitcher + push + open) proves per-row isolation — the old eager
    stat dict crashed the WHOLE batch on r.OUTS for any pitcher row."""
    bets = [
        _bet(pid=1, line=1.5, side="Over", odds=100),          # 2 hits: win
        _bet(market="pitcher_strikeouts", pid=9, line=6.5,
             side="Over", odds=100, player="Arm 9", team="NYY"),  # 7 K: win
        _bet(market="pitcher_outs", pid=9, line=17.5,
             side="Over", odds=100, player="Arm 9", team="NYY"),  # 17: lose
        _bet(market="pitcher_earned_runs", pid=9, line=2.0,
             side="Over", odds=100, player="Arm 9", team="NYY"),  # 2: push
        _bet(pid=4, line=1.5, side="Over", odds=100),          # no box: open
    ]
    SK.append(SK.enrich(bets, DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777)],
                bats=[_bat_row(777, 1, h=2)],
                pits=[_pit_row(777, 9, ip="5.2", so=7, er=2)])
    SK.settle(DATE)
    led = SK._load()
    out = {(r.Market, SK._key_norm(r.PlayerId)): r.Outcome
           for _, r in led.iterrows()}
    assert out[("batter_hits", "1")] == "win"
    assert out[("pitcher_strikeouts", "9")] == "win"
    assert out[("pitcher_outs", "9")] == "lose"       # 17 outs < 17.5
    assert out[("pitcher_earned_runs", "9")] == "push"
    assert out[("batter_hits", "4")] == ""            # stays open


def test_settle_game_markets_win_lose(env):
    SK.append(SK.enrich([_ml_bet(team="NYY", odds=110),
                         _tot_bet(side="Over", line=8.5, odds=100)],
                        DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777, asc=4, hsc=5)])   # NYY 5-4
    SK.settle(DATE)
    led = SK._load()
    out = {r.Market: r.Outcome for _, r in led.iterrows()}
    assert out["h2h"] == "win"                 # home NYY won
    assert out["totals"] == "win"              # 9 > 8.5


def test_settle_totals_push_voids(env):
    SK.append(SK.enrich([_tot_bet(side="Over", line=9.0, odds=100)],
                        DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777, asc=4, hsc=5)])   # total 9
    SK.settle(DATE)
    led = SK._load()
    assert list(led.Outcome) == ["push"]
    assert float(led.PnL_units.iloc[0]) == 0.0


def test_dnp_rows_void_on_next_day_sweep(env):
    """A scratched batter (PA=0) and an unused pitcher stay OPEN on
    same-date settle (box may lag), then void on the next day's sweep
    — the book's did-not-play rule; open rows can't accrete forever."""
    bets = [_bet(pid=1, line=1.5),                       # played: grades
            _bet(pid=2, line=1.5),                       # PA=0: DNP
            _bet(market="pitcher_strikeouts", pid=9, line=5.5,
                 player="Arm 9", team="NYY")]            # never pitched
    SK.append(SK.enrich(bets, DATE, MKT_FAM))
    day2 = "2026-07-25"
    _write_data(env,
                games=[_game_row(777)],
                bats=[_bat_row(777, 1, h=2), _bat_row(777, 2, h=0, pa=0)],
                pits=[_pit_row(777, 8)])   # box exists, pid 9 absent
    SK.settle(DATE)
    led = SK._load()
    by = {SK._key_norm(r.PlayerId): r for _, r in led.iterrows()}
    assert by["1"].Outcome == "win"
    assert by["2"].Outcome == "" and by["9"].Outcome == ""   # same-date: open
    SK.settle(day2)                        # next-day sweep
    led = SK._load()
    by = {SK._key_norm(r.PlayerId): r for _, r in led.iterrows()}
    for pid in ("2", "9"):
        assert by[pid].Outcome == "void" and by[pid].Status == "void"
        assert float(by[pid].PnL_units) == 0.0


def test_relief_only_appearance_voids_starter_prop(env):
    """A scheduled starter scratched into same-game RELIEF (GS=0): US
    books VOID the starter prop. The old settle graded the ladder
    against the relief line (a 2-K mop-up settled 'strikeouts o4.5' as
    lose, disagreeing with Tools/4's own GS==1 workbook grade); it must
    stay open same-date, then VOID on the sweep — stake returned, PnL
    0, never a loss."""
    SK.append(SK.enrich([_bet(market="pitcher_strikeouts", pid=9,
                              line=4.5, side="Over", player="Arm 9",
                              team="NYY")], DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777)],
                pits=[_pit_row(777, 9, ip="2.0", so=2, GS=0, GF=1)])
    SK.settle(DATE)
    led = SK._load()
    assert led.Outcome.iloc[0] == ""       # relief line never grades
    SK.settle("2026-07-25")                # next-day sweep
    led = SK._load()
    assert led.Outcome.iloc[0] == "void"
    assert led.Status.iloc[0] == "void"
    assert float(led.PnL_units.iloc[0]) == 0.0


# ------------------------------------------------------- CLV backfill

def test_clv_totals_fills_via_gamepk_not_team(env):
    """Ledger totals rows carry Team='' — the join must go through the
    Game label + GamePk. Two games' totals at the same line sit in the
    store; only THIS GamePk's close may price the CLV."""
    SK.append(SK.enrich([_tot_bet(side="Over", line=8.5, odds=100)],
                        DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777)])
    _write_store(env, [
        _store_row("totals", 8.5, team="NYY", over=-120, under=100,
                   pk=777),
        # the OTHER game's total at the same line — a wrong join would
        # blend it into the fair
        _store_row("totals", 8.5, team="LAD", over=200, under=-300,
                   pk=888),
    ])
    SK.settle(DATE)
    led = SK._load()
    r = led.iloc[0]
    fair = O.no_vig(-120, 100)[0]
    assert r.Outcome == "win"
    assert float(r.p_close_final) == pytest.approx(fair, abs=1e-6)
    assert float(r.CLV) == pytest.approx(fair - float(r.implied),
                                         abs=1e-6)
    assert float(r.CloseAmerican) == -120          # Over side, own book


def test_clv_ambiguous_multi_pk_group_stays_blank(env):
    """No CLV beats wrong CLV: a pk-less ledger row meeting a multi-pk
    store group (DH day) must stay blank, never blend two games'
    prices into one fair."""
    SK.append(SK.enrich([_tot_bet(side="Over", line=8.5, odds=100,
                                  gpk=None)], DATE, MKT_FAM))
    led = SK._load()
    store = pd.DataFrame([
        _store_row("totals", 8.5, team="NYY", over=-120, under=100,
                   pk=777),
        _store_row("totals", 8.5, team="NYY", over=110, under=-130,
                   pk=888),
    ], columns=O.ODDS_COLUMNS)
    SK._fill_clv(led, 0, led.loc[0], store, None, "totals", "Over")
    assert led.loc[0, "p_close_final"] == ""
    assert led.loc[0, "CLV"] == ""
    # a single-pk group WITH a matching ledger pk resolves normally
    SK._fill_clv(led, 0, led.loc[0], store, 777, "totals", "Over")
    fair = O.no_vig(-120, 100)[0]
    assert float(led.loc[0, "p_close_final"]) == pytest.approx(
        fair, abs=1e-6)


def test_clv_totals_blank_store_pks_filter_by_home_club(env):
    """Store degradation day (blank GamePks, a coded Tools/2 fallback):
    totals candidates must still narrow to THIS game via Team == home
    club — the old join had no game filter at all for totals, so the
    blank-pk fallback averaged EVERY same-line game's fair into the
    permanent p_close_final/CLV."""
    SK.append(SK.enrich([_tot_bet(side="Over", line=8.5, odds=100)],
                        DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777)])
    _write_store(env, [
        _store_row("totals", 8.5, team="NYY", over=-120, under=100,
                   pk=""),
        # another game's total at the same line, also pk-less — pooling
        # it in shifts the fair far off this game's close
        _store_row("totals", 8.5, team="LAD", over=200, under=-300,
                   pk=""),
    ])
    SK.settle(DATE)
    r = SK._load().iloc[0]
    fair = O.no_vig(-120, 100)[0]                  # NYY game only
    assert r.Outcome == "win"
    assert float(r.p_close_final) == pytest.approx(fair, abs=1e-6)
    assert float(r.CLV) == pytest.approx(fair - float(r.implied),
                                         abs=1e-6)


def test_clv_blank_pk_fallback_stays_blank_on_dh_day(env):
    """Doubleheader + blank store pks: the same-club fallback rows can
    mix (or BE) the other game's prices, so the money fields must stay
    BLANK (§9 makes a wrong write permanent) while settlement itself
    still grades off the row's own GamePk."""
    SK.append(SK.enrich([_tot_bet(side="Over", line=8.5, odds=100,
                                  gpk=777)], DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777), _game_row(888)])  # DH: same clubs
    _write_store(env, [
        _store_row("totals", 8.5, team="NYY", over=-120, under=100,
                   pk=""),
    ])
    SK.settle(DATE)
    r = SK._load().iloc[0]
    assert r.Outcome == "win"                      # game 777: 9 > 8.5
    assert r.p_close_final == ""
    assert r.CLV == ""
    assert r.CloseAmerican == ""


def test_clv_away_moneyline_takes_complement_of_home_fair(env):
    """The store keys h2h under the HOME club with OverPrice = home
    side. An away-side ledger row must join through the Game label and
    take 1 - fair(home) and the UnderPrice close — the old r.Team
    equality matched nothing (and would have taken the home fair)."""
    SK.append(SK.enrich([_ml_bet(team="BOS", odds=130)], DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777, asc=4, hsc=5)])   # BOS loses
    _write_store(env, [
        _store_row("h2h", "", team="NYY", over=-150, under=130, pk=777),
    ])
    SK.settle(DATE)
    led = SK._load()
    r = led.iloc[0]
    fair_home = O.no_vig(-150, 130)[0]
    assert r.Outcome == "lose"
    assert float(r.p_close_final) == pytest.approx(1 - fair_home,
                                                   abs=1e-6)
    assert float(r.CLV) == pytest.approx((1 - fair_home)
                                         - float(r.implied), abs=1e-6)
    assert float(r.CloseAmerican) == 130           # the away-side price


def test_clv_player_prop_gamepk_exact_on_dh(env):
    """DH day: the same player's same prop is priced in BOTH games.
    settle() must de-vig only the ledger row's own GamePk."""
    SK.append(SK.enrich([_bet(pid=5, line=0.5, side="Over", odds=100,
                              gpk=777)], DATE, MKT_FAM))
    _write_data(env, games=[_game_row(777), _game_row(888)],
                bats=[_bat_row(777, 5, h=1)])
    _write_store(env, [
        _store_row("batter_hits", 0.5, pid=5, over=-150, under=120,
                   pk=777),
        _store_row("batter_hits", 0.5, pid=5, over=-110, under=-110,
                   pk=888),
    ])
    SK.settle(DATE)
    led = SK._load()
    r = led.iloc[0]
    fair = O.no_vig(-150, 120)[0]                  # game 777 only
    assert float(r.p_close_final) == pytest.approx(fair, abs=1e-6)


# ------------------------------------------------------ known-gap pins

def test_outcome_y_team_totals_is_safe_open():
    """DOCUMENTED GAP (2026-07-25 handoff): outcome_y has no
    team_totals branch yet, so tt ledger rows must stay safely OPEN
    (None) — never crash, never mis-grade. When the branch lands this
    pin flips and the test must be rewritten to grade the side score."""
    games = pd.DataFrame([dict(GamePk=1, AwayScore=4, HomeScore=5)])
    gb = pd.DataFrame(columns=["GamePk", "PlayerId", "PA"])
    gp = pd.DataFrame(columns=["GamePk", "PlayerId", "SO", "OUTS", "H",
                               "BB", "ER"])
    assert SK.outcome_y(gb, gp, games, 1, None, "team_totals", 3.5) \
        is None


# ------------------------------------------------------ gate delegation

def test_odds_y_delegates_to_staking():
    games = pd.DataFrame([dict(GamePk=1, AwayScore=4, HomeScore=5)])
    gb = pd.DataFrame([dict(GamePk=1, PlayerId=9, PA=4, H=2, HR=0, TB=2,
                            R=0, RBI=0, BB=0, SB=0,
                            **{"2B": 0, "3B": 0})])
    # GS mirrors the real pitching schema: pid 8 started, pid 7 is a
    # relief-only line — both the gate and the ledger must void it
    gp = pd.DataFrame([dict(GamePk=1, PlayerId=8, GS=1, SO=7, OUTS=17,
                            H=5, BB=2, ER=2),
                       dict(GamePk=1, PlayerId=7, GS=0, SO=2, OUTS=6,
                            H=1, BB=0, ER=0)])
    for args in ((gb, gp, games, 1, None, "totals", 9.0),
                 (gb, gp, games, 1, 9, "batter_hits", 2.0),
                 (gb, gp, games, 1, 9, "batter_hits", 1.5),
                 (gb, gp, games, 1, 8, "pitcher_outs", 16.5),
                 (gb, gp, games, 1, 8, "pitcher_strikeouts", 7.0),
                 (gb, gp, games, 1, 7, "pitcher_strikeouts", 1.5),
                 (gb, gp, games, 1, None, "h2h", None)):
        assert EV._odds_y(*args) == SK.outcome_y(*args)
    # the relief-only line is a VOID (None) on both sides, never 1/0
    assert SK.outcome_y(gb, gp, games, 1, 7, "pitcher_strikeouts", 1.5) \
        is None


def test_outcome_y_missing_scores_void_not_zero():
    games = pd.DataFrame([dict(GamePk=1, AwayScore=np.nan,
                               HomeScore=np.nan)])
    gb = pd.DataFrame(columns=["GamePk", "PlayerId", "PA"])
    gp = pd.DataFrame(columns=["GamePk", "PlayerId", "SO", "OUTS", "H",
                               "BB", "ER"])
    assert SK.outcome_y(gb, gp, games, 1, None, "h2h", None) is None
    assert SK.outcome_y(gb, gp, games, 1, None, "totals", 8.5) is None


def test_settled_keys_matches_bet_key_row_by_row():
    """_settled_keys is the vectorized twin of the per-row _bet_key walk
    (append/settle dedup). Any drift between them silently reopens the
    double-settle path, so pin exact set equality over every _bet_key
    branch: h2h (Team dropped from the key), team-scoped markets (Team
    kept), pk'd rows (Game/GNum blanked), pk-less rows (Game/GNum stand
    in), float-artifact keys ('777.0' == 777), and non-settled Outcomes
    (superseded / open) excluded."""
    led = pd.DataFrame([
        dict(Date=DATE, GamePk="777.0", PlayerId="601", Team="NYY",
             Market="batter_hits", Line="1.5", Game="", GNum="",
             Status="paper", Outcome="win"),
        dict(Date=DATE, GamePk="778", PlayerId="", Team="BOS",
             Market="h2h", Line="", Game="BOS@NYY", GNum="1",
             Status="track", Outcome="lose"),
        dict(Date=DATE, GamePk="", PlayerId="", Team="TOR",
             Market="team_totals", Line="4.5", Game="TOR@TB", GNum="2",
             Status="paper", Outcome="void"),
        dict(Date=DATE, GamePk="779", PlayerId="602.0", Team="TB",
             Market="pitcher_strikeouts", Line="5.5", Game="", GNum="",
             Status="paper", Outcome="push"),
        dict(Date=DATE, GamePk="780", PlayerId="603", Team="TB",
             Market="batter_hits", Line="0.5", Game="", GNum="",
             Status="paper", Outcome="superseded"),
        dict(Date=DATE, GamePk="781", PlayerId="604", Team="TB",
             Market="batter_hits", Line="0.5", Game="", GNum="",
             Status="track", Outcome=""),
    ]).astype(str)
    expect = {SK._bet_key(led.loc[i])
              for i in led.index[led.Outcome.isin(SK._SETTLED)]}
    assert SK._settled_keys(led) == expect
    assert len(SK._settled_keys(led)) == 4   # superseded + open excluded
