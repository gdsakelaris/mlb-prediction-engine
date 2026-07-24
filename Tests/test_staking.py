"""Paper-trading staking ledger (Model/staking.py): §3-§5 math, the
append-only supersede contract, settlement (win/lose/push/open), and
the evaluate._odds_y delegation that keeps gate and ledger settlement
identical.
"""
import pandas as pd
import pytest

import evaluate as EV
import staking as SK

MKT_FAM = {"batter_hits": "h", "pitcher_strikeouts": "pk",
           "totals": "tot", "h2h": "ml"}


def _bet(market="batter_hits", pid=123, side="Over", line=1.5,
         odds=120, pm=0.55, pc=0.50, gpk=777):
    return {"Game": "BOS@NYY", "G#": 1, "GamePk": gpk, "PlayerId": pid,
            "Player": "Some Guy", "Team": "BOS", "Prop": "hits o1.5",
            "Side": side, "Line": line, "Model %": pm, "Mkt %": pc,
            "Best Odds": odds, "Book": "draftkings", "EV%": 0.1,
            "_market": market}


def test_enrich_math_and_track_status(tmp_path, monkeypatch):
    monkeypatch.setattr(SK, "ART", tmp_path)      # no gate report ->
    rows = SK.enrich([_bet()], "2026-07-24", MKT_FAM)
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


def test_enrich_pass_family_stakes_with_per_bet_cap(tmp_path,
                                                    monkeypatch):
    (tmp_path / "market_gate_report.csv").write_text(
        "family,verdict\nh,PASS\npk,NO-EDGE\n")
    monkeypatch.setattr(SK, "ART", tmp_path)
    rows = SK.enrich([_bet(pm=0.62, pc=0.58),          # big edge -> cap
                      _bet(market="pitcher_strikeouts", pid=99,
                           pm=0.62, pc=0.58)], "2026-07-24", MKT_FAM)
    by_mkt = {r["Market"]: r for r in rows}
    h = by_mkt["batter_hits"]
    assert h["Status"] == "paper"
    assert h["stake_capped_by"] == "per-bet-1%"
    assert h["stake_units"] == pytest.approx(0.01 * SK.START_BANKROLL)
    assert by_mkt["pitcher_strikeouts"]["Status"] == "track"


def test_enrich_never_both_sides(tmp_path, monkeypatch):
    monkeypatch.setattr(SK, "ART", tmp_path)
    rows = SK.enrich([_bet(side="Over", odds=110),
                      _bet(side="Under", odds=250, pm=0.45, pc=0.50)],
                     "2026-07-24", MKT_FAM)
    assert len(rows) == 1                     # higher-EV side survives
    assert rows[0]["Side"] == "Under"


def test_append_supersede_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(SK, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(SK, "ART", tmp_path)
    r1 = SK.enrich([_bet(odds=110)], "2026-07-24", MKT_FAM)
    SK.append(r1)
    r2 = SK.enrich([_bet(odds=130)], "2026-07-24", MKT_FAM)
    SK.append(r2)
    led = pd.read_csv(tmp_path / "ledger.csv", dtype=str)
    # 3 rows: original, its void marker, the re-serve — never an edit
    assert len(led) == 3
    assert list(led.Status) == ["track", "void", "track"]
    assert led.iloc[1]["stake_capped_by"] == "superseded-by-re-serve"
    assert led.iloc[2]["PriceAmerican"] == "130"


def test_settle_win_lose_push_and_open(tmp_path, monkeypatch):
    monkeypatch.setattr(SK, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(SK, "ART", tmp_path)
    monkeypatch.setattr(SK, "DATA", tmp_path)
    monkeypatch.setattr(SK.O, "DEFAULT_STORE", tmp_path / "odds.csv")
    date = "2026-07-24"
    bets = [_bet(pid=1, line=1.5, side="Over", odds=100),   # 2 hits: win
            _bet(pid=2, line=1.5, side="Over", odds=100),   # 1 hit: lose
            _bet(pid=3, line=2.0, side="Over", odds=100),   # 2 hits: push
            _bet(pid=4, line=1.5, side="Over", odds=100)]   # no box: open
    SK.append(SK.enrich(bets, date, MKT_FAM))
    pd.DataFrame([
        dict(Date=date, GamePk=777, AwayTeam="BOS", HomeTeam="NYY",
             AwayScore=4, HomeScore=5)]).to_csv(
        tmp_path / "mlb_games.csv", index=False)
    gb = [dict(Date=date, GamePk=777, PlayerId=p, PA=4, H=h, HR=0, TB=h,
               R=0, RBI=0, BB=0, SB=0, **{"2B": 0, "3B": 0})
          for p, h in ((1, 2), (2, 1), (3, 2))]
    pd.DataFrame(gb).to_csv(tmp_path / "mlb_game_batting.csv",
                            index=False)
    pd.DataFrame(columns=["Date", "GamePk", "PlayerId", "SO", "OUTS",
                          "H", "BB", "ER"]).to_csv(
        tmp_path / "mlb_game_pitching.csv", index=False)
    SK.settle(date)
    led = pd.read_csv(tmp_path / "ledger.csv", dtype=str)
    out = {int(float(r.PlayerId)): r.Outcome
           for _, r in led.iterrows() if r.Status == "track"}
    assert out[1] == "win" and out[2] == "lose" and out[3] == "push"
    assert pd.isna(led[led.PlayerId == "4"].Outcome.iloc[0])
    # tracked rows settle with zero PnL — evidence, not money
    assert float(led[led.PlayerId == "1"].PnL_units.iloc[0]) == 0.0
    # settling twice is a no-op (rows already carry an Outcome)
    SK.settle(date)
    assert len(pd.read_csv(tmp_path / "ledger.csv")) == len(led)


def test_odds_y_delegates_to_staking():
    games = pd.DataFrame([dict(GamePk=1, AwayScore=4, HomeScore=5)])
    gb = pd.DataFrame([dict(GamePk=1, PlayerId=9, PA=4, H=2, HR=0, TB=2,
                            R=0, RBI=0, BB=0, SB=0,
                            **{"2B": 0, "3B": 0})])
    gp = pd.DataFrame(columns=["GamePk", "PlayerId", "SO", "OUTS", "H",
                               "BB", "ER"])
    for args in ((gb, gp, games, 1, None, "totals", 9.0),
                 (gb, gp, games, 1, 9, "batter_hits", 2.0),
                 (gb, gp, games, 1, 9, "batter_hits", 1.5),
                 (gb, gp, games, 1, None, "h2h", None)):
        assert EV._odds_y(*args) == SK.outcome_y(*args)
