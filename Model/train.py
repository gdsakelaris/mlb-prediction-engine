"""Fit every component model from the feature warehouse.

Models fitted (artifacts in Model/artifacts/):
  a1_model.joblib     8-class PA outcome: XGBoost multiclass
                      + vector-scaling calibrator fit on the calibration
                      year. Feature list and class order ride along.
  a2_model.joblib     4-class batted-ball type given in-play, same X.
  hazard_model.joblib starter-removal hazard per batter faced (binary
                      XGBoost + isotonic calibration).
  sb_models.joblib    steal-of-2B attempt and success logistic models
                      (with imputation pipelines).
  latent.json         game-level latent-variance sigmas (fit by
                      --fit-latent via sim moment matching; written with
                      fitted=false defaults until that runs).
  manifest.json       data horizon, split years, holdout metrics.

Split protocol: the SERVE artifact trains on seasons <= 2024 and
calibrates on 2025 (the newest completed season); 2026 is forward
monitoring only. --design-eval instead trains <= 2023 and evaluates on
2024, leaving 2025 untouched for the frozen design holdout.

Usage:
    python Model/train.py                # fit serve artifacts
    python Model/train.py --rebuild      # rebuild stores first, then fit
    python Model/train.py --design-eval  # 2024 evaluation report
    python Model/train.py --fit-latent   # sim moment-match (needs sim.py)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ART = F.ART
STORES = F.STORES
CLASSES = F.CLASSES
SERVE_CAL_YEAR = 2025
DESIGN_CAL_YEAR = 2024
BBTYPES = F.BBTYPES


def _xgb_overrides():
    """Tuned T1/T3 hyperparameters from artifacts/xgb_params.json
    (written by --tune). Absent or unreadable -> {} and the hand-set
    defaults stand."""
    p = ART / "xgb_params.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("params", {})
    except Exception:
        return {}


def make_clf(max_iter=300, eval_metric="mlogloss", tuned=False,
             params=None, seed=7):
    """XGBoost estimator factory (CUDA histogram trees). `params` wins
    outright (Optuna trials); else `tuned=True` applies the persisted
    xgb_params.json overrides (the PA tree only — hazard/A2 keep the
    hand-set defaults by scope)."""
    kw = dict(n_estimators=max_iter, learning_rate=0.08,
              grow_policy="lossguide", max_leaves=63, max_depth=0,
              min_child_weight=100, reg_lambda=1.0,
              subsample=1.0, colsample_bytree=1.0)
    if params is not None:
        kw.update(params)
    elif tuned:
        kw.update(_xgb_overrides())
    return XGBClassifier(
        **kw, tree_method="hist", device="cuda",
        early_stopping_rounds=15, eval_metric=eval_metric,
        n_jobs=-1, random_state=seed, verbosity=0)


def _fit_es(clf, Xtr, ytr, w=None, seed=7):
    """Fit with 5% held out as the early-stopping eval_set. `w` =
    optional training sample weights (the eval slice stays unweighted
    so early stopping keeps judging current-fit quality evenly)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    cut = int(len(idx) * 0.95)
    take = ((lambda a, i: a.iloc[i]) if hasattr(Xtr, "iloc")
            else (lambda a, i: a[i]))
    kw = {}
    if w is not None:
        kw["sample_weight"] = np.asarray(w)[idx[:cut]]
    clf.fit(take(Xtr, idx[:cut]), ytr[idx[:cut]],
            eval_set=[(take(Xtr, idx[cut:]), ytr[idx[cut:]])],
            verbose=False, **kw)
    return clf


def _decay_weights(pa, mask, hl_years):
    """Exponential recency weights for training rows: a PA hl_years
    old counts half of one played yesterday. None -> unweighted."""
    if hl_years is None:
        return None
    d = pd.to_datetime(pa["Date"])
    age = (d[mask].max() - d[mask]).dt.days / 365.25
    return np.power(0.5, age.values / float(hl_years))


def _fit_bag(make, Xtr, ytr, w, seeds):
    """One fit per seed, averaged via F.BaggedClf (single seed ->
    the bare estimator, artifact shape unchanged)."""
    members = [
        _fit_es(make(s), Xtr, ytr, w=w, seed=s) for s in seeds]
    return members[0] if len(members) == 1 else F.BaggedClf(members)


# A1 PA-tree training config (design-eval gated, sweep 2026-07-20:
# recency decay LOST at every half-life — hl5 +0.00009, hl3 +0.00034,
# hl2 +0.00054 design logloss, old PAs carry signal the era features
# don't; 3-seed bagging WON −0.00022)
A1_DECAY_HL_YEARS = None       # recency half-life in years, None = off
A1_BAG_SEEDS = (7, 17, 27)     # seed-bagged PA tree (W4.18)


VectorScaler = F.VectorScaler


def _banner(title):
    print(f"\n{'=' * 66}\n  {title}\n{'=' * 66}", flush=True)


def _kv(label, value):
    print(f"  {label:<28} {value}", flush=True)


def _report_a1(name, y, p, classes):
    ll = log_loss(y, p, labels=list(range(len(classes))))
    _kv(f"logloss ({name})", f"{ll:.5f}")
    print(f"\n    {'class':<5} {'rate':>8} {'mean_p':>8} {'bias':>9}",
          flush=True)
    for i, c in enumerate(classes):
        yi = (y == i).astype(float)
        r, mp = float(yi.mean()), float(p[:, i].mean())
        print(f"    {c:<5} {r:>8.4f} {mp:>8.4f} {mp - r:>+9.4f}",
              flush=True)
    print("", flush=True)
    return ll


def _a1_baselines(pa, X, y, tr, ca):
    lg = pd.Series(y[tr]).value_counts(normalize=True).sort_index().values
    ll_league = log_loss(y[ca], np.tile(lg, (ca.sum(), 1)),
                         labels=list(range(len(CLASSES))))
    marg = np.zeros((ca.sum(), len(CLASSES)))
    for j, c in enumerate(CLASSES):
        col = f"b_{c}_rate"
        marg[:, j] = (X.loc[ca, col].to_numpy(dtype=float)
                      if col in X.columns else np.nan)
    rate_cols = [f"b_{c}_rate" for c in F.BAT_RATES]
    have = [c for c in rate_cols if c in X.columns]
    bat_only = X.loc[ca, have].to_numpy(dtype=float)
    rest = np.clip(1 - np.nansum(bat_only, axis=1), 1e-6, 1)
    hbp = np.full(ca.sum(), lg[CLASSES.index("HBP")])
    marg[:, CLASSES.index("HBP")] = hbp
    marg[:, CLASSES.index("IPO")] = rest - hbp
    marg = np.nan_to_num(marg, nan=1e-6)
    marg = np.clip(marg, 1e-6, 1)
    marg /= marg.sum(axis=1, keepdims=True)
    ll_marg = log_loss(y[ca], marg, labels=list(range(len(CLASSES))))
    return ll_league, ll_marg


def fit_a2(pa, X, cal_year):
    tr = (pa["Season"] < cal_year).values
    ca = (pa["Season"] == cal_year).values
    Xf = X.astype(np.float32)
    inplay = pa["bb_type"].isin(BBTYPES).values
    y2 = pa["bb_type"].map({b: i for i, b in enumerate(BBTYPES)}).values
    tr2, ca2 = tr & inplay, ca & inplay
    _banner("A2 — batted-ball type model (4 classes)")
    _kv("train rows", f"{tr2.sum():,}  (in-play)")
    a2 = make_clf(max_iter=200)
    _fit_es(a2, Xf[tr2], y2[tr2])
    p2_raw = a2.predict_proba(Xf[ca2])
    scaler2 = VectorScaler().fit(p2_raw, y2[ca2])
    ll2 = log_loss(y2[ca2], scaler2.transform(p2_raw),
                   labels=list(range(len(BBTYPES))))
    _kv("logloss (calibrated)", f"{ll2:.5f}")
    return dict(model=a2, scaler=scaler2, classes=BBTYPES,
                features=list(X.columns), cal_year=cal_year,
                metrics=dict(logloss=ll2))


def fit_a1_flat(pa, X, cal_year):
    y = pa["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    tr = (pa["Season"] < cal_year).values
    ca = (pa["Season"] == cal_year).values
    Xf = X.astype(np.float32)
    _banner(f"A1 — flat PA outcome model ({len(CLASSES)} classes, "
            f"{X.shape[1]} features)")
    _kv("train rows", f"{tr.sum():,}  (seasons <= {cal_year - 1})")
    _kv("calibrate rows", f"{ca.sum():,}  ({cal_year})")
    t0 = time.time()
    a1 = make_clf()
    _fit_es(a1, Xf[tr], y[tr])
    _kv("fit time", f"{time.time() - t0:.0f}s")
    p_cal_raw = a1.predict_proba(Xf[ca])
    scaler = VectorScaler().fit(p_cal_raw, y[ca])
    p_cal = scaler.transform(p_cal_raw)
    ll_model = _report_a1("A1 calibrated", y[ca], p_cal, CLASSES)
    ll_league, ll_marg = _a1_baselines(pa, X, y, tr, ca)
    _kv("baseline: league", f"{ll_league:.5f}   (model gain "
        f"{ll_league - ll_model:+.5f})")
    _kv("baseline: batter-marginal", f"{ll_marg:.5f}   (model gain "
        f"{ll_marg - ll_model:+.5f})")
    return dict(model=a1, scaler=scaler, classes=CLASSES,
                features=list(X.columns), cal_year=cal_year,
                metrics=dict(logloss=ll_model, league=ll_league,
                             batter_marginal=ll_marg))


def _t3_matrix(Xf, bb_idx):
    """Append the bb-type dummy block (GB baseline) as constants."""
    n = len(Xf)
    d = np.zeros((n, 3), dtype=np.float32)
    if bb_idx > 0:
        d[:, bb_idx - 1] = 1.0
    return np.column_stack([Xf, d])


def fit_a1_tree(pa, X, cal_year, a2d, hl_years=None, seeds=None):
    """Hierarchical contact tree: T1 {K,BB,HBP,in-play} and T3
    outcome|bb-type {out,1B,2B,3B,HR}, composed with A2 (=T2) into the
    flat 8-class vector + the class-conditional bb mix. Evaluated as
    the COMPOSITE against the flat baselines. hl_years/seeds default
    to the module config (A1_DECAY_HL_YEARS / A1_BAG_SEEDS)."""
    hl_years = A1_DECAY_HL_YEARS if hl_years is None else hl_years
    hl_years = None if hl_years in (0, "none") else hl_years
    seeds = tuple(seeds or A1_BAG_SEEDS)
    y8 = pa["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    tr = (pa["Season"] < cal_year).values
    ca = (pa["Season"] == cal_year).values
    Xf = X.astype(np.float32).to_numpy()
    feats = list(X.columns)

    t1_map = {"K": 0, "BB": 1, "HBP": 2}
    y1 = pa["label"].map(lambda c: t1_map.get(c, 3)).values
    _banner(f"A1 TREE / T1 — PA gate ({len(F.T1_CLASSES)} classes, "
            f"{len(feats)} features)")
    _kv("train rows", f"{tr.sum():,}")
    if hl_years or len(seeds) > 1:
        _kv("config", f"decay hl={hl_years or 'off'}y, "
            f"seeds={list(seeds)}")
    w1 = _decay_weights(pa, tr, hl_years)
    t0 = time.time()
    t1 = _fit_bag(lambda s: make_clf(tuned=True, seed=s),
                  Xf[tr], y1[tr], w1, seeds)
    s1 = VectorScaler().fit(t1.predict_proba(Xf[ca]), y1[ca])
    _kv("fit time", f"{time.time() - t0:.0f}s")

    inplay = pa["bb_type"].isin(BBTYPES).values
    bb_idx = pa["bb_type"].map(
        {b: i for i, b in enumerate(BBTYPES)}).fillna(0).astype(int)
    y3 = pa["label"].map(
        {c: i for i, c in enumerate(F.T3_CLASSES)}).values
    tr3, ca3 = tr & inplay, ca & inplay
    _banner(f"A1 TREE / T3 — outcome | bb-type "
            f"({len(F.T3_CLASSES)} classes)")
    _kv("train rows", f"{tr3.sum():,}  (in-play)")
    d3 = np.zeros((len(pa), 3), dtype=np.float32)
    for j in range(1, 4):
        d3[bb_idx.values == j, j - 1] = 1.0
    X3 = np.column_stack([Xf, d3])
    w3 = _decay_weights(pa, tr3, hl_years)
    t0 = time.time()
    t3 = _fit_bag(lambda s: make_clf(tuned=True, seed=s),
                  X3[tr3], y3[tr3], w3, seeds)
    s3 = VectorScaler().fit(t3.predict_proba(X3[ca3]), y3[ca3])
    _kv("fit time", f"{time.time() - t0:.0f}s")

    # composite 8-class evaluation on the cal year
    p1c = s1.transform(t1.predict_proba(Xf[ca]))
    p2c = a2d["scaler"].transform(a2d["model"].predict_proba(Xf[ca]))
    p3c = np.stack(
        [s3.transform(t3.predict_proba(_t3_matrix(Xf[ca], bi)))
         for bi in range(4)], axis=1)
    p8, _ = F.tree_compose(p1c, p2c, p3c)
    _banner("A1 TREE — composite 8-class evaluation")
    ll_model = _report_a1("tree composite", y8[ca], p8, CLASSES)
    ll_league, ll_marg = _a1_baselines(pa, X, y8, tr, ca)
    _kv("baseline: league", f"{ll_league:.5f}   (model gain "
        f"{ll_league - ll_model:+.5f})")
    _kv("baseline: batter-marginal", f"{ll_marg:.5f}   (model gain "
        f"{ll_marg - ll_model:+.5f})")

    return dict(kind="tree", classes=CLASSES, features=feats,
                cal_year=cal_year,
                t1=dict(model=t1, scaler=s1, classes=F.T1_CLASSES),
                t3=dict(model=t3, scaler=s3, classes=F.T3_CLASSES,
                        features=feats + F.BB_DUMMIES),
                metrics=dict(logloss=ll_model, league=ll_league,
                             batter_marginal=ll_marg))


def tune_a1(pa, X, a2d, n_trials):
    """Optuna TPE sweep over the T1/T3 XGBoost hyperparameters.
    Objective: composite 8-class logloss on the DESIGN cal year (2024)
    — the serve calibration year stays untouched by design decisions.
    Scope guard: tunes ONLY the PA-tree estimators. Latent knobs belong
    to backtest.moment_match_latent (moment targets, not logloss) and
    anything fit on calib_rows is gate-adjacent and off limits.
    Winner -> artifacts/xgb_params.json (trial 0 is the hand-set
    defaults, so the persisted best can never be worse than them)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y8 = pa["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    tr = (pa["Season"] < DESIGN_CAL_YEAR).values
    ca = (pa["Season"] == DESIGN_CAL_YEAR).values
    Xf = X.astype(np.float32).to_numpy()
    t1_map = {"K": 0, "BB": 1, "HBP": 2}
    y1 = pa["label"].map(lambda c: t1_map.get(c, 3)).values
    inplay = pa["bb_type"].isin(BBTYPES).values
    bb_idx = pa["bb_type"].map(
        {b: i for i, b in enumerate(BBTYPES)}).fillna(0).astype(int)
    y3 = pa["label"].map(
        {c: i for i, c in enumerate(F.T3_CLASSES)}).values
    tr3, ca3 = tr & inplay, ca & inplay
    d3 = np.zeros((len(pa), 3), dtype=np.float32)
    for j in range(1, 4):
        d3[bb_idx.values == j, j - 1] = 1.0
    X3 = np.column_stack([Xf, d3])
    p2c = a2d["scaler"].transform(a2d["model"].predict_proba(Xf[ca]))

    def objective(trial):
        prm = dict(
            learning_rate=trial.suggest_float(
                "learning_rate", 0.03, 0.16, log=True),
            max_leaves=trial.suggest_int("max_leaves", 31, 255,
                                         log=True),
            min_child_weight=trial.suggest_float(
                "min_child_weight", 20.0, 600.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.3, 20.0,
                                           log=True),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            n_estimators=trial.suggest_int("n_estimators", 250, 700,
                                           log=True))
        t0 = time.time()
        t1 = make_clf(params=prm)
        _fit_es(t1, Xf[tr], y1[tr])
        s1 = VectorScaler().fit(t1.predict_proba(Xf[ca]), y1[ca])
        t3 = make_clf(params=prm)
        _fit_es(t3, X3[tr3], y3[tr3])
        s3 = VectorScaler().fit(t3.predict_proba(X3[ca3]), y3[ca3])
        p1c = s1.transform(t1.predict_proba(Xf[ca]))
        p3c = np.stack(
            [s3.transform(t3.predict_proba(_t3_matrix(Xf[ca], bi)))
             for bi in range(4)], axis=1)
        p8, _ = F.tree_compose(p1c, p2c, p3c)
        ll = log_loss(y8[ca], p8, labels=list(range(len(CLASSES))))
        print(f"  trial {trial.number:>3}  ll {ll:.5f}  "
              f"({time.time() - t0:.0f}s)  {trial.params}", flush=True)
        return ll

    _banner(f"A1 TREE — Optuna sweep ({n_trials} trials, "
            f"design cal {DESIGN_CAL_YEAR})")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=7))
    study.enqueue_trial(dict(
        learning_rate=0.08, max_leaves=63, min_child_weight=100.0,
        reg_lambda=1.0, subsample=1.0, colsample_bytree=1.0,
        n_estimators=300))
    study.optimize(objective, n_trials=n_trials)
    d0 = float(study.trials[0].value)
    best = study.best_trial
    _kv("defaults logloss", f"{d0:.5f}")
    _kv("best logloss", f"{float(best.value):.5f}   (gain "
        f"{d0 - float(best.value):+.5f})")
    out = dict(params=best.params, logloss=float(best.value),
               defaults_logloss=d0, gain=d0 - float(best.value),
               cal_year=DESIGN_CAL_YEAR, n_trials=len(study.trials),
               tuned=date.today().isoformat())
    F.write_artifact(ART / "xgb_params.json",
                     lambda p: p.write_text(json.dumps(out, indent=2)))
    print(f"  -> {ART / 'xgb_params.json'}", flush=True)


HAZ_FEATS = ["bf", "cum_pitches", "tto", "inning", "outs", "score_diff",
             "k_so_far", "br_so_far", "runs_so_far", "rest_p",
             "leash_np", "leash_bf", "leash_starts", "season_idx",
             "gap_days", "ramp", "prev_short", "il_ret30", "outs_sd",
             "pen_np3", "post", "team_hook"]


def fit_hazard(cal_year):
    hz = pd.read_parquet(STORES / "hazard_table.parquet")
    leash = pd.read_parquet(STORES / "panel_leash.parquet")
    m = F.merge_asof_panel(
        hz.rename(columns={"PitcherId": "PlayerId"}), leash, ["PlayerId"],
        ["starts", "np_sum", "bf_sum", "outs_sum", "outs2_sum"], "lz_")
    hz["leash_starts"] = m["lz_starts"]
    hz["leash_np"] = m["lz_np_sum"] / m["lz_starts"].clip(lower=1e-9)
    hz["leash_bf"] = m["lz_bf_sum"] / m["lz_starts"].clip(lower=1e-9)
    mu_o = m["lz_outs_sum"] / m["lz_starts"].clip(lower=1e-9)
    var_o = (m["lz_outs2_sum"] / m["lz_starts"].clip(lower=1e-9)
             - mu_o ** 2)
    hz["outs_sd"] = np.where(m["lz_starts"] >= 5,
                             np.sqrt(np.clip(var_o, 0, None)), np.nan)
    hz["season_idx"] = hz["Season"] - 2015
    tr = hz["Season"] < cal_year
    ca = hz["Season"] == cal_year
    Xh = hz[HAZ_FEATS].astype(np.float32)
    yh = hz["removed"].values
    _banner(f"B — starter-removal hazard ({len(HAZ_FEATS)} features)")
    _kv("train rows", f"{tr.sum():,}")
    _kv("removal rate", f"{yh[tr.values].mean():.3%}")
    if "cause" in hz.columns:   # competing-risks label (diagnostic only;
        mix = (hz.loc[hz.removed == 1, "cause"]  # the served hazard stays
               .value_counts(normalize=True))    # all-cause by design)
        _kv("removal cause mix", "  ".join(
            f"{k}:{v:.1%}" for k, v in mix.items() if k))
    mdl = make_clf(max_iter=250, eval_metric="logloss")
    _fit_es(mdl, Xh[tr.values], yh[tr.values])
    p_ca = mdl.predict_proba(Xh[ca.values])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca, yh[ca.values])
    ll = log_loss(yh[ca.values], np.clip(iso.predict(p_ca), 1e-6, 1 - 1e-6))
    base = log_loss(yh[ca.values],
                    np.full(ca.sum(), yh[tr.values].mean()))
    _kv("logloss (calibrated)", f"{ll:.5f}")
    _kv("baseline: constant rate", f"{base:.5f}   (model gain "
        f"{base - ll:+.5f})")
    return dict(model=mdl, iso=iso, features=HAZ_FEATS,
                cal_year=cal_year, metrics=dict(logloss=ll, base=base))


SB_ATT_FEATS = ["SprintSpeed", "speed_miss", "speed2", "sb_allowed_rate",
                "cs_rate", "PopTime", "CSAA", "outs", "outs1",
                "score_close", "era_new", "lhp", "post"]
SB_SUC_FEATS = ["SprintSpeed", "cs_rate", "PopTime", "CSAA", "lhp",
                "era_new", "post"]
SB_SPEED_C = 27.3        # sprint-speed center for the quadratic term


def fit_sb(cal_year):
    sb = pd.read_parquet(STORES / "sb_table.parquet")
    sb["score_close"] = (sb["score_diff"].abs() <= 2).astype(float)
    sb["era_new"] = (sb["Season"] >= 2023).astype(float)
    sb["lhp"] = (sb["p_throws"].astype(str) == "L").astype(float)
    # missing sprint speed (rookies/callups) attempts 23% MORE than the
    # median-imputed rate; both speed tails over-forecast on a pure
    # linear term; attempt rate peaks at 1 out (non-monotone)
    sb["speed_miss"] = sb["SprintSpeed"].isna().astype(float)
    sb["speed2"] = (sb["SprintSpeed"] - SB_SPEED_C) ** 2
    sb["outs1"] = (sb["outs"] == 1).astype(float)
    if "post" not in sb.columns:           # pre-rebuild sb_table
        sb["post"] = 0
    sb["post"] = sb["post"].astype(float)
    tr = sb["Season"] <= cal_year          # small models: use through cal
    pipe = lambda feats: Pipeline([        # noqa: E731
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    att = pipe(SB_ATT_FEATS).fit(sb.loc[tr, SB_ATT_FEATS],
                                 sb.loc[tr, "attempt"])
    on_att = tr & (sb["attempt"] == 1)
    suc = pipe(SB_SUC_FEATS).fit(sb.loc[on_att, SB_SUC_FEATS],
                                 sb.loc[on_att, "success"])
    p_att = att.predict_proba(sb.loc[tr, SB_ATT_FEATS])[:, 1]
    # truth anchoring: the table misses inning-ending caught steals
    # (voided PAs), so scale attempts up and shift success down to the
    # PBP-true era means computed by the warehouse
    scale = json.loads((STORES / "sb_scale.json").read_text())
    logit = lambda p: np.log(p / (1 - p))            # noqa: E731
    shifts = {}
    for era, mask in (("pre2023", tr & (sb.Season < 2023)),
                      ("post2023", tr & (sb.Season >= 2023))):
        rows = mask & (sb["attempt"] == 1)
        if rows.sum() == 0:
            shifts[era] = 0.0
            continue
        mean_model = suc.predict_proba(
            sb.loc[rows, SB_SUC_FEATS])[:, 1].mean()
        shifts[era] = float(logit(scale[era]["success_true"])
                            - logit(mean_model))
    # sim-time state conditioning: exact raw-scale logit deltas off the
    # serve baseline (outs=1, score_close=1) so the engines can stop
    # blowout innings stealing like one-run games
    sc_, lr_ = att.named_steps["sc"], att.named_steps["lr"]
    braw = {f: float(lr_.coef_[0][i] / sc_.scale_[i])
            for i, f in enumerate(SB_ATT_FEATS)}
    att_state = dict(outs0=-braw["outs"] - braw["outs1"],
                     outs2=braw["outs"] - braw["outs1"],
                     sc_far=-braw["score_close"])
    _banner("D — stolen-base attempt/success")
    _kv("opportunities", f"{tr.sum():,}")
    _kv("attempt rate", f"{sb.loc[tr, 'attempt'].mean():.3%}  "
        f"(mean pred {p_att.mean():.3%})")
    _kv("success shifts", str({k: round(v, 3)
                               for k, v in shifts.items()}))
    _kv("state logit", str({k: round(v, 3)
                            for k, v in att_state.items()}))
    ps_i = SB_SUC_FEATS.index("post")
    b_suc_post = float(suc.named_steps["lr"].coef_[0][ps_i]
                       / suc.named_steps["sc"].scale_[ps_i])
    _kv("post coef (att/suc)", f"{braw['post']:+.3f} / {b_suc_post:+.3f}")
    return dict(attempt=att, success=suc, att_features=SB_ATT_FEATS,
                suc_features=SB_SUC_FEATS, scale=scale,
                success_logit_shift=shifts, att_state_logit=att_state,
                speed_center=SB_SPEED_C)


def fit_latent():
    """Moment-match the game-level latent sigmas against history using
    the simulator: grid over (pitcher-form, offense-day, HR-env) logit
    sigmas, replay a sample of games, and pick the triple whose team-run
    and starter-K dispersion best matches the actual distributions."""
    import backtest as B  # local import: sim/backtest exist by P2
    res = B.moment_match_latent()
    F.write_artifact(ART / "latent.json",
                     lambda p: p.write_text(json.dumps(res, indent=1)))
    print(f"latent fitted: {res}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the feature stores first")
    ap.add_argument("--design-eval", action="store_true",
                    help="train <=2023, evaluate 2024 (no serve artifacts)")
    ap.add_argument("--fit-latent", action="store_true",
                    help="moment-match latent sigmas (requires sim.py)")
    ap.add_argument("--no-write", action="store_true",
                    help="fit + report only; leave artifacts untouched "
                         "(trial runs)")
    ap.add_argument("--tune", type=int, default=0, metavar="N",
                    help="Optuna sweep of the T1/T3 XGB hyperparams "
                         "(N trials on the design split), write "
                         "artifacts/xgb_params.json, exit — no serve "
                         "artifacts touched")
    ap.add_argument("--flat", action="store_true",
                    help="train the flat 8-class A1 instead of the "
                         "contact tree (A/B baseline; the 2026-07-19 "
                         "full-2025 ledger A/B was a statistical tie "
                         "0.06538 vs 0.06528 — tree SHIPPED by "
                         "decision: correct generative process, "
                         "class-conditional bb for the sim/SGP)")
    args = ap.parse_args()

    if args.fit_latent:
        fit_latent()
        return
    if args.rebuild:
        rc = subprocess.run([sys.executable,
                             str(Path(__file__).parent / "features.py"),
                             "--build"]).returncode
        if rc != 0:
            sys.exit("store rebuild failed; not fitting on stale stores")

    cal_year = (DESIGN_CAL_YEAR if (args.design_eval or args.tune)
                else SERVE_CAL_YEAR)
    stores = F.load_stores()
    pa = pd.read_parquet(STORES / "pa_table.parquet")
    _banner("FEATURE ASSEMBLY")
    _kv("PA rows", f"{len(pa):,}")
    t0 = time.time()
    X, cols = F.assemble_features(pa, stores)
    _kv("features", f"{len(cols)}")
    _kv("assembly time", f"{time.time() - t0:.0f}s")

    a2 = fit_a2(pa, X, cal_year)
    if args.tune:
        tune_a1(pa, X, a2, args.tune)
        return
    a1 = (fit_a1_flat(pa, X, cal_year) if args.flat
          else fit_a1_tree(pa, X, cal_year, a2))
    hz = fit_hazard(cal_year)
    sb = fit_sb(cal_year)

    if args.design_eval or args.no_write:
        print("\nreport-only run: artifacts NOT written", flush=True)
        return
    ART.mkdir(parents=True, exist_ok=True)
    for name, obj in (("a1_model", a1), ("a2_model", a2),
                      ("hazard_model", hz), ("sb_models", sb)):
        F.write_artifact(ART / f"{name}.joblib",
                         lambda p, o=obj: joblib.dump(o, p))
    if not (ART / "latent.json").exists():
        (ART / "latent.json").write_text(json.dumps(
            dict(fitted=False, mu_env=0.0, sigma_env=0.0,
                 sigma_pitcher=0.0, sigma_hr=0.0), indent=1))
        print("latent.json: defaults written (run --fit-latent after "
              "sim.py exists)", flush=True)
    manifest = dict(
        trained=date.today().isoformat(),
        cal_year=cal_year,
        data_max_date=str(pa.Date.max().date()),
        n_pa=int(len(pa)),
        a1=a1["metrics"], a2=a2["metrics"], hazard=hz["metrics"])
    F.write_artifact(ART / "manifest.json",
                     lambda p: p.write_text(json.dumps(manifest, indent=1)))
    _banner("ARTIFACTS WRITTEN")
    _kv("models", "a1 / a2 / hazard / sb  -> Model/artifacts/")
    _kv("cal_year", f"{cal_year}")
    _kv("data through", str(pa.Date.max().date()))


if __name__ == "__main__":
    main()
