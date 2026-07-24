"""Characterization tests for Model/evaluate.py helpers on synthetic
ledgers (no artifacts, no replays): logloss/brier, _odds_y settlement
(pushes void), ab_compare verdicts, fit_calibrators recovery/identity
fallbacks, goal_metrics, reliability_bands, plus the artifact-free
calibration plumbing (_cal, PlattCal, _tail_prob)."""
import math

import joblib
import numpy as np
import pandas as pd

import evaluate as EV
import features as F
import predict as PR


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
    gp = pd.DataFrame([{"GamePk": 1, "PlayerId": 10, "SO": 7,
                        "OUTS": 18, "H": 5, "BB": 2, "ER": 3}])
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
