"""Characterization tests for Tools/5 Prop Rankings' pure stats helpers:
_slope_irls against sklearn's unpenalized MLE, _auc against
roc_auc_score, the weighted-bootstrap path (_prep_binlike/_wdiags) with a
single all-ones weight row reproducing the plain _binary_diags point
values, _crossfit_pcal's per-fold monotonicity + identity fallbacks, and
the hand-checkable lift helpers. All synthetic, seeded, no artifacts.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from conftest import load_tool

t5 = load_tool("5) Prop Rankings")


def _line(n_days, per_day, seed, y_const=None):
    """One synthetic priced line: day ids, roughly-calibrated p, y."""
    rng = np.random.default_rng(seed)
    day_id = np.repeat(np.arange(n_days), per_day)
    p = rng.uniform(0.05, 0.95, n_days * per_day)
    if y_const is None:
        y = (rng.random(n_days * per_day) < p).astype(float)
    else:
        y = np.full(n_days * per_day, float(y_const))
    dates = pd.Timestamp("2026-06-01") + pd.to_timedelta(day_id, unit="D")
    df = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"),
                       "p": p, "p_cal": p, "y": y})
    return day_id, p, y, df


# ------------------------------------------------------------ _slope_irls

def test_slope_irls_recovers_logistic():
    rng = np.random.default_rng(20260723)
    z = rng.normal(0.0, 1.2, 2000)
    a_true, b_true = 0.3, 1.4
    y = (rng.random(2000)
         < 1.0 / (1.0 + np.exp(-(a_true + b_true * z)))).astype(float)
    a, b = t5._slope_irls(z, y, np.ones_like(y))
    lr = LogisticRegression(penalty=None, tol=1e-12, max_iter=10000)
    lr.fit(z.reshape(-1, 1), y)
    # Newton/IRLS lands on the same MLE as sklearn's lbfgs
    assert np.isclose(a, lr.intercept_[0], atol=1e-3)
    assert np.isclose(b, lr.coef_[0][0], atol=1e-3)
    # and near the generating params at this n
    assert np.isclose(a, a_true, atol=0.15)
    assert np.isclose(b, b_true, atol=0.15)


# ------------------------------------------------------------------ _auc

def test_auc_matches_sklearn():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, 500)
    y = (rng.random(500) < p).astype(float)
    assert np.isclose(t5._auc(y, p), roc_auc_score(y, p), atol=1e-12)
    # heavy ties: average ranks == sklearn's tie handling
    pt = np.round(p, 1)
    assert np.isclose(t5._auc(y, pt), roc_auc_score(y, pt), atol=1e-12)
    # single class -> nan, not a crash
    assert np.isnan(t5._auc(np.ones(5), np.linspace(0.1, 0.9, 5)))


# ------------------------------------------- bootstrap identity (all-ones)

def test_weighted_diags_all_ones_reproduce_plain_diags():
    """The day-block bootstrap's reweighting path with a single all-ones
    weight row must give back the plain point diagnostics."""
    n_days = 30
    day_id, p, y, df = _line(n_days, 40, seed=20260723)
    pp = t5._prep_binlike(day_id, n_days, p, y)
    wd = t5._wdiags(pp, np.ones((1, n_days)))[0]

    plain = t5._binary_diags(p, y, t5.top10_lift(df))
    plain["orl"] = t5.or_lift(plain["lift"], plain["base"])

    # exact identities (fp-level): per-day sums and the fixed sort orders
    # reduce to the unweighted formulas under w == 1
    for k in ("rel", "auc", "brier_rel", "acc", "acc_base", "mae_rel",
              "top10", "orl"):
        assert np.isclose(wd[k], plain[k], rtol=0, atol=1e-9), k
    # calibration line: _wslope warm-starts AT the full-data fit, so the
    # all-ones refit stays on it
    assert np.isclose(wd["slope"], plain["slope"], atol=1e-9)
    assert np.isclose(wd["int"], plain["int"], atol=1e-9)
    # PINNED: ece_rel is close but NOT identical — _wece cuts bins by
    # cumulative-weight searchsorted (first bin one row short of qcut's
    # equal-count deciles), observed diff ~3e-3 on this seed
    assert np.isclose(wd["ece_rel"], plain["ece_rel"], atol=1e-2)
    assert abs(wd["ece_rel"] - plain["ece_rel"]) > 1e-6


# --------------------------------------------------------- _crossfit_pcal

def test_crossfit_pcal_monotone_per_fold():
    # 25 days x 40 rows: every fold's complement is big and two-class,
    # so all five folds get a real Platt refit
    _, p, _, df = _line(25, 40, seed=5)
    out = t5._crossfit_pcal(df)
    clipped = np.clip(p, 1e-4, 1 - 1e-4)
    assert (out >= 1e-4).all() and (out <= 1 - 1e-4).all()
    assert not np.allclose(out, clipped)          # fits actually applied
    day_id, _ = pd.factorize(df["Date"], sort=True)
    fold = day_id % t5.CV_FOLDS
    for k in range(t5.CV_FOLDS):
        m = fold == k
        order = np.argsort(clipped[m])
        # one increasing logistic map per fold -> monotone within it
        assert (np.diff(out[m][order]) >= -1e-12).all()


def test_crossfit_pcal_identity_fallbacks():
    # single-class training folds (y all 0) -> raw clipped p, exactly
    _, p, _, df = _line(25, 40, seed=6, y_const=0)
    assert np.array_equal(t5._crossfit_pcal(df), np.clip(p, 1e-4, 1 - 1e-4))
    # thin family (< CV_MIN_TRAIN rows in any complement) -> identity too
    _, p2, _, df2 = _line(5, 10, seed=7)
    assert np.array_equal(t5._crossfit_pcal(df2),
                          np.clip(p2, 1e-4, 1 - 1e-4))


# ------------------------------------------------------------ lift helpers

def test_or_lift():
    # lift 2x at base .25: top hit rate .5 -> odds 1 vs base odds 1/3
    assert np.isclose(t5.or_lift(2.0, 0.25), 3.0, atol=1e-12)
    assert np.isclose(t5.or_lift(1.0, 0.4), 1.0, atol=1e-12)   # no lift
    assert t5.or_lift(2.5, 0.25) > t5.or_lift(2.0, 0.25)
    assert np.isnan(t5.or_lift(1.5, 0.0))      # degenerate base
    assert np.isnan(t5.or_lift(1.5, 1.0))
    assert np.isnan(t5.or_lift(np.nan, 0.3))


def test_top10_lift_hand_frame():
    # day 1: 12 rows, top-10 by p hold 5 of the 7 hits; day 2: 5 rows
    # (fewer than 10 -> all picked), all hits
    d1_p = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
            0.45, 0.40]
    d1_y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    d2_p = [0.99, 0.98, 0.97, 0.96, 0.30]
    df = pd.DataFrame({
        "Date": ["2026-07-01"] * 12 + ["2026-07-02"] * 5,
        "p_cal": d1_p + d2_p,
        "y": d1_y + [1] * 5,
    })
    # picks mean = (5+5)/15 = 2/3, base = 12/17 -> lift = 17/18
    assert np.isclose(t5.top10_lift(df), 17 / 18, atol=1e-12)
    # zero base -> nan
    z = pd.DataFrame({"Date": ["2026-07-01"] * 3, "p_cal": [0.5, 0.4, 0.3],
                      "y": [0, 0, 0]})
    assert np.isnan(t5.top10_lift(z))


def test_top1_lift_hand_frame():
    # per day the single most CONFIDENT game (conf = max(p, 1-p)) is
    # picked and scored on side-correctness, not on y directly
    df = pd.DataFrame({
        "Date": ["2026-07-01", "2026-07-01",
                 "2026-07-02", "2026-07-02",
                 "2026-07-03", "2026-07-03"],
        "p_cal": [0.90, 0.60, 0.20, 0.55, 0.70, 0.45],
        "y":     [1,    0,    0,    1,    0,    0],
    })
    # picks: d1 p=.9 y=1 hit; d2 p=.2 y=0 hit (confident UNDER counts);
    # d3 p=.7 y=0 miss -> hit rate 2/3 over base 1/3 = 2.0
    assert np.isclose(t5.top1_lift(df), 2.0, atol=1e-12)
