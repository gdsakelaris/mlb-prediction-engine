"""Characterization tests for Model/evaluate.py helpers on synthetic
ledgers (no artifacts, no replays): logloss/brier, _odds_y settlement
(pushes void, relief-only pitcher rows void), ab_compare verdicts,
fit_calibrators recovery/identity fallbacks, goal_metrics,
reliability_bands, the artifact-free calibration plumbing (_cal,
PlattCal, _tail_prob), and the market_gate doubleheader replay contract
(one predict_slate call per date, faked Predictor — no sims)."""
import math
import types

import joblib
import numpy as np
import pandas as pd

import evaluate as EV
import features as F
import predict as PR
import sim


# ------------------------------------------------------- logloss / brier

def test_logloss_hand_computed():
    y, p = [1, 0, 1], [0.8, 0.3, 0.9]
    want = -(math.log(0.8) + math.log(0.7) + math.log(0.9)) / 3
    assert math.isclose(EV.logloss(y, p), want, rel_tol=1e-12)


def test_logloss_clips_at_1e6():
    # p exactly 1 with y=0 clips to 1-1e-6, not inf
    assert math.isclose(EV.logloss([0], [1.0]), -math.log(1e-6))


def test_brier_hand_computed():
    want = (0.2 ** 2 + 0.3 ** 2 + 0.1 ** 2) / 3
    assert math.isclose(EV.brier([1, 0, 1], [0.8, 0.3, 0.9], ), want)


# --------------------------------------------- _odds_y settlement table

def _frames():
    games = pd.DataFrame([
        {"GamePk": 1, "AwayScore": 2, "HomeScore": 5},
        {"GamePk": 2, "AwayScore": 7, "HomeScore": 3},
        {"GamePk": 3, "AwayScore": np.nan, "HomeScore": np.nan},
    ])
    gp = pd.DataFrame([
        {"GamePk": 1, "PlayerId": 10, "GS": 1, "SO": 7,
         "OUTS": 18, "H": 5, "BB": 2, "ER": 3},
        # relief-only appearance (GS=0): a book void for starter props
        {"GamePk": 1, "PlayerId": 11, "GS": 0, "SO": 3,
         "OUTS": 6, "H": 1, "BB": 0, "ER": 0},
    ])
    gb = pd.DataFrame([
        {"GamePk": 1, "PlayerId": 20, "PA": 4, "H": 2, "2B": 1,
         "3B": 0, "HR": 0, "TB": 3, "R": 1, "RBI": 2, "BB": 0, "SB": 0},
        {"GamePk": 1, "PlayerId": 21, "PA": 0, "H": 0, "2B": 0,
         "3B": 0, "HR": 0, "TB": 0, "R": 0, "RBI": 0, "BB": 0, "SB": 0},
    ])
    return gb, gp, games


def test_odds_y_game_markets():
    gb, gp, games = _frames()
    assert EV._odds_y(gb, gp, games, 1, -1, "h2h", None) == 1
    assert EV._odds_y(gb, gp, games, 2, -1, "h2h", None) == 0
    assert EV._odds_y(gb, gp, games, 1, -1, "totals", 6.5) == 1
    assert EV._odds_y(gb, gp, games, 1, -1, "totals", 8.5) == 0
    assert EV._odds_y(gb, gp, games, 1, -1, "totals", 7.0) is None  # push
    assert EV._odds_y(gb, gp, games, 3, -1, "h2h", None) is None    # no score
    assert EV._odds_y(gb, gp, games, 3, -1, "totals", 6.5) is None
    assert EV._odds_y(gb, gp, games, 99, -1, "h2h", None) is None   # unplayed


def test_odds_y_pitcher_markets():
    gb, gp, games = _frames()
    f = EV._odds_y
    assert f(gb, gp, games, 1, 10, "pitcher_strikeouts", 6.5) == 1
    assert f(gb, gp, games, 1, 10, "pitcher_strikeouts", 7.5) == 0
    assert f(gb, gp, games, 1, 10, "pitcher_strikeouts", 7.0) is None  # push
    assert f(gb, gp, games, 1, 10, "pitcher_outs", 17.5) == 1
    assert f(gb, gp, games, 1, 10, "pitcher_earned_runs", 3.0) is None
    assert f(gb, gp, games, 1, 99, "pitcher_strikeouts", 6.5) is None


def test_odds_y_pitcher_relief_only_voids():
    # GS==1 settlement (the Tools/4 convention): a relief-only
    # appearance is a book void, never a graded start — SO=3 over a
    # 2.5 line must NOT settle as a win
    gb, gp, games = _frames()
    f = EV._odds_y
    assert f(gb, gp, games, 1, 11, "pitcher_strikeouts", 2.5) is None
    assert f(gb, gp, games, 1, 11, "pitcher_outs", 4.5) is None
    # the true start in the same game still settles alongside it
    assert f(gb, gp, games, 1, 10, "pitcher_strikeouts", 6.5) == 1


def test_odds_y_batter_markets():
    gb, gp, games = _frames()
    f = EV._odds_y
    assert f(gb, gp, games, 1, 20, "batter_hits", 1.5) == 1
    assert f(gb, gp, games, 1, 20, "batter_hits", 2.5) == 0
    assert f(gb, gp, games, 1, 20, "batter_hits", 2.0) is None      # push
    assert f(gb, gp, games, 1, 20, "batter_singles", 0.5) == 1      # 2-1-0-0
    assert f(gb, gp, games, 1, 20, "batter_singles", 1.0) is None   # push
    assert f(gb, gp, games, 1, 20, "batter_hits_runs_rbis", 4.5) == 1  # 5
    assert f(gb, gp, games, 1, 21, "batter_hits", 0.5) is None      # PA == 0
    assert f(gb, gp, games, 1, 20, "batter_triples", 0.5) is None   # unknown
    assert f(gb, gp, games, 1, 999, "batter_hits", 0.5) is None


# ------------------------------------------------------------ ab_compare

def _ab_ledger():
    """40 slates; families h/pk have n=1000 (>= min_n 800), sb n=160."""
    rng = np.random.default_rng(11)
    dates = ([f"2025-06-{d:02d}" for d in range(1, 31)]
             + [f"2025-07-{d:02d}" for d in range(1, 11)])
    rows = []
    for i, date in enumerate(dates):
        for fam, mkt, k in (("h", "Hit", 25), ("pk", "K > 4.5", 25),
                            ("sb", "SB", 4)):
            for j in range(k):
                p = float(rng.uniform(0.05, 0.95))
                y = int(rng.random() < p)
                rows.append((1000 + i, date, fam, mkt, p, y,
                             10000 + j, "NYY", 1))
    return pd.DataFrame(rows, columns=EV.ROW_COLS)


def test_ab_compare_identical_ledgers_tie(tmp_path):
    df = _ab_ledger()
    pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
    df.to_parquet(pa)
    df.to_parquet(pb)
    rep = EV.ab_compare(pa, pb).set_index("family")
    fams = rep.drop(index="ALL")
    assert (fams.delta == 0).all()
    assert rep.loc["h", "verdict"] == "TIE"
    assert rep.loc["pk", "verdict"] == "TIE"
    assert rep.loc["sb", "verdict"] == "INSUFFICIENT n"
    assert rep.loc["ALL", "delta"] == 0


def test_ab_compare_shaded_b_wins(tmp_path):
    da = _ab_ledger()
    db = da.copy()
    db["p"] = da.p + 0.2 * (da.y - da.p)   # toward y: better every row
    pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
    da.to_parquet(pa)
    db.to_parquet(pb)
    rep = EV.ab_compare(pa, pb).set_index("family")
    assert rep.loc["h", "verdict"] == "B BETTER"
    assert rep.loc["pk", "verdict"] == "B BETTER"
    assert rep.loc["sb", "verdict"] == "INSUFFICIENT n"   # despite delta>0
    assert (rep.delta > 0).all()
    assert rep.loc["h", "ci_lo"] > 0


# ------------------------------------------------------- fit_calibrators

def test_fit_calibrators_reuse_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(EV, "ART", tmp_path)
    rng = np.random.default_rng(3)
    a_true, b_true = 0.5, 1.3
    n = 6000
    p = rng.uniform(0.05, 0.9, n)
    z = np.log(p) - np.log1p(-p)
    y = (rng.random(n) < 1 / (1 + np.exp(-(a_true + b_true * z)))
         ).astype(int)
    dates = [f"2025-06-{(i % 30) + 1:02d}" for i in range(n)]
    df = pd.DataFrame({
        "GamePk": np.arange(n) % 400, "Date": dates,
        "family": "h", "market": "Hit", "p": p, "y": y,
        "PlayerId": np.arange(n), "Team": "NYY", "Home": 1})
    tiny = df.head(60).assign(family="pk", market="K > 4.5")
    onecls = df.head(800).assign(family="sb", market="SB", y=0)
    allrows = pd.concat([df, tiny, onecls], ignore_index=True)
    allrows.to_parquet(tmp_path / "calib_rows.parquet")

    out = EV.fit_calibrators(None, None, reuse_rows=True)
    # tiny-n and single-class families are ABSENT (identity by absence)
    assert set(out) == {"h", "_meta"}
    cal = out["h"]
    assert abs(cal.a - a_true) < 0.25
    assert abs(cal.b - b_true) < 0.25
    meta = out["_meta"]
    assert meta["fit_start"] == "2025-06-01"
    assert meta["fit_end"] == "2025-06-30"
    assert meta["n_rows"] == len(allrows)
    saved = joblib.load(tmp_path / "output_calibrators.joblib")
    assert set(saved) == set(out)


# ---------------------------------------- goal_metrics / reliability

GOAL_COLS = ["market", "n", "base", "auc", "t1_hit", "t3_hit",
             "t10_stated", "t10_hit", "t10_gap", "hi_n", "hi_stated",
             "hi_hit", "hi_gap", "trust_depth"]


def _goal_ledger(perfect):
    rng = np.random.default_rng(5)
    rows = []
    for d in range(1, 21):                     # 20 slates x 20 rows
        for j in range(20):
            y = int(j < 6)                     # base rate 0.3
            if perfect:
                p = (0.7 if y else 0.05) + 0.01 * j
            else:
                p = float(rng.uniform(0.05, 0.95))
                y = int(rng.random() < 0.3)
            rows.append((f"2025-06-{d:02d}", "Hit", p, y))
    return pd.DataFrame(rows, columns=["Date", "market", "p", "y"])


def test_goal_metrics_perfect_prediction():
    g = EV.goal_metrics(_goal_ledger(perfect=True))
    assert list(g.columns) == GOAL_COLS
    r = g.iloc[0]
    assert r.t1_hit == 1.0 and r.t3_hit == 1.0
    assert r.auc == 1.0
    assert r.t10_hit == 0.6                    # 6 hits per 20-row slate
    assert r.trust_depth >= 6


def test_goal_metrics_random_p_no_trust_depth():
    g = EV.goal_metrics(_goal_ledger(perfect=False))
    assert g.iloc[0].trust_depth == 0


def test_reliability_bands():
    df = pd.DataFrame({"p": [0.52, 0.53, 0.72, 0.40],
                       "y": [1, 0, 1, 1]})
    bands = EV.reliability_bands(df)           # lo=0.5 -> 0.40 excluded
    assert list(bands.band) == ["[0.50,0.55)", "[0.70,0.75)"]
    b0 = bands.iloc[0]
    assert b0.n == 2 and b0.stated == 0.525 and b0.hit == 0.5


# ------------------------- artifact-free calibration plumbing

def test_cal_identity_fallbacks():
    assert PR._cal(None, "h", 0.37) == 0.37
    assert PR._cal({}, "h", 0.37) == 0.37
    assert PR._cal(None, "h", None) is None


def test_cal_line_map_wins_over_family_map():
    calib = {"pout": F.PlattCal(0.0, 1.0),
             "_lines": {"Outs > 14.5": F.PlattCal(1.0, 1.0)}}
    got = PR._cal(calib, "pout", 0.5, market="Outs > 14.5")
    assert math.isclose(got, 1 / (1 + math.exp(-1.0)), abs_tol=1e-9)
    # no line map for this market string -> family map (identity here)
    assert math.isclose(PR._cal(calib, "pout", 0.5, market="Outs > 15.5"),
                        0.5, abs_tol=1e-6)


def test_plattcal_monotone_and_open_interval():
    cal = F.PlattCal(0.3, 1.2)
    out = cal.predict(np.linspace(0.0, 1.0, 101))   # includes hard 0/1
    assert (out > 0).all() and (out < 1).all()
    assert (np.diff(out) >= 0).all()
    assert (np.diff(out[1:-1]) > 0).all()          # strict in the interior


def test_tail_prob_smooth_nonzero_tail():
    counts = np.ones(200)
    beyond = PR._tail_prob(counts, 5.5)            # beyond support
    assert 0 < beyond < 0.01                        # parametric, not hard 0
    assert PR._tail_prob(counts, 0.5) > beyond      # decreasing in thr
    assert PR._tail_prob(np.zeros(500), 0.5) == 0.0


# ------------------------------- market_gate doubleheader replay contract

def test_gate_fingerprint_carries_replay_version(tmp_path, monkeypatch):
    # r2 = whole-date replay + GS==1 settlement; the token must sit in
    # the cache key so rows graded under r1 (per-game DH context,
    # relief rows settled) can never be served from cache again
    monkeypatch.setattr(EV, "ART", tmp_path)
    assert EV.GATE_REPLAY_VERSION >= 2
    fp = EV._gate_fingerprint(4000)
    assert f"|r{EV.GATE_REPLAY_VERSION}|" in fp


def _dh_counts(pk):
    """Deterministic per-game hit columns: 30/40 sims for game 101,
    10/40 for game 102 — distinct p_model per DH game."""
    h = {101: 30, 102: 10}[pk]
    return np.concatenate([np.ones(h), np.zeros(40 - h)])


def test_market_gate_dh_date_one_slate_call(tmp_path, monkeypatch):
    """The gate must price ALL of a date's specs in ONE predict_slate
    call — the serve's slate-level context, where DH game 2 sees game 1
    as its same-day previous game — and map each result back to its own
    GamePk. Per-game calls (the r1 replay) graded context the engine
    never served. Also pins the per-date cache round trip under the
    r-token key."""
    date = "2025-06-10"
    games = pd.DataFrame([
        {"GamePk": 101, "Date": pd.Timestamp(date), "AwayTeam": "NYY",
         "HomeTeam": "BOS", "AwayScore": 2, "HomeScore": 5},
        {"GamePk": 102, "Date": pd.Timestamp(date), "AwayTeam": "NYY",
         "HomeTeam": "BOS", "AwayScore": 7, "HomeScore": 3},
    ])
    calls = []

    class FakeP:
        def __init__(self):
            self.calib = {}
            self.mkt_blend = True
            self.stores = types.SimpleNamespace(raw={"games": games})

        def predict_slate(self, specs, n_sims=None):
            calls.append([sp["game_pk"] for sp in specs])
            out = []
            for sp in specs:
                t = np.zeros((40, 20, len(sim.STATS)))
                t[:, 0, sim.SIDX["H"]] = _dh_counts(sp["game_pk"])
                out.append({"spec": sp, "calib": self.calib,
                            "meta": {"players": [500] + [-1] * 19},
                            "tensor": t})
            return out

    def fake_spec(P, g, lineups, starters, umps, wx):
        return dict(date=date, game_pk=int(g.GamePk),
                    away_lineup=[(500 + i, i + 1) for i in range(9)],
                    away_starter=600, home_starter=601)

    # batter 500 plays both DH games: 2 hits in game 101, 0 in game 102
    gb = pd.DataFrame([
        {"GamePk": 101, "PlayerId": 500, "PA": 4, "H": 2, "2B": 0,
         "3B": 0, "HR": 0, "TB": 2, "R": 0, "RBI": 0, "BB": 0, "SB": 0},
        {"GamePk": 102, "PlayerId": 500, "PA": 4, "H": 0, "2B": 0,
         "3B": 0, "HR": 0, "TB": 0, "R": 0, "RBI": 0, "BB": 0, "SB": 0},
    ])
    gp = pd.DataFrame(columns=["GamePk", "PlayerId", "GS", "SO",
                               "OUTS", "H", "BB", "ER"])
    odds = pd.DataFrame([dict(
        Date=date, GamePk=pk, Team="NYY", PlayerId=500,
        PlayerName="DH Batter", Market="batter_hits", Line=0.5,
        OverPrice=-110, UnderPrice=-110, Book="pinnacle",
        CapturedAt=f"{date}T16:00:00", OpenOverPrice=-105,
        OpenUnderPrice=-115, OpenCapturedAt=f"{date}T12:00:00")
        for pk in (101, 102)])
    store = tmp_path / "odds.csv"
    odds.to_csv(store, index=False)

    monkeypatch.setattr(EV, "ART", tmp_path)
    monkeypatch.setattr(EV, "_load_actuals", lambda: (gb, gp))
    monkeypatch.setattr(EV.O, "DEFAULT_STORE", store)
    monkeypatch.setattr(EV.PR, "Predictor", FakeP)
    monkeypatch.setattr(EV.B, "_spec_frames",
                        lambda P: (None, None, None, None))
    monkeypatch.setattr(EV.B, "build_spec", fake_spec)

    df, _ = EV.market_gate(date, date, n_sims=40, min_n=1, boot=5)
    assert calls == [[101, 102]]         # ONE call carrying both specs
    assert len(df) == 2
    # each priced row graded against its OWN game's tensor: y separates
    # the DH games (2 hits -> 1, 0 hits -> 0)
    by_y = df.set_index("y").p_model
    assert math.isclose(by_y[1], PR._tail_prob(_dh_counts(101), 0.5),
                        rel_tol=1e-12)
    assert math.isclose(by_y[0], PR._tail_prob(_dh_counts(102), 0.5),
                        rel_tol=1e-12)
    cache = pd.read_parquet(tmp_path / "gate_rows_cache.parquet")
    assert cache.key.str.contains(f"|r{EV.GATE_REPLAY_VERSION}|",
                                  regex=False).all()
    # unchanged stack + odds: the second run serves from cache, no re-sim
    df2, _ = EV.market_gate(date, date, n_sims=40, min_n=1, boot=5)
    assert calls == [[101, 102]]
    assert sorted(df2.p_model) == sorted(df.p_model)
