"""Fit every component model from the feature warehouse.

Models fitted (artifacts in Model/artifacts/):
  a1_model.joblib     8-class PA outcome: HistGradientBoosting multiclass
                      + vector-scaling calibrator fit on the calibration
                      year. Feature list and class order ride along.
  a2_model.joblib     4-class batted-ball type given in-play, same X.
  hazard_model.joblib starter-removal hazard per batter faced (binary
                      HGBT + isotonic calibration).
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
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ART = F.ART
STORES = F.STORES
CLASSES = F.CLASSES
SERVE_CAL_YEAR = 2025
DESIGN_CAL_YEAR = 2024
BBTYPES = F.BBTYPES


def hgbt(**kw):
    base = dict(learning_rate=0.08, max_iter=300, max_leaf_nodes=63,
                min_samples_leaf=100, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.05,
                n_iter_no_change=15, random_state=7)
    base.update(kw)
    return HistGradientBoostingClassifier(**base)


VectorScaler = F.VectorScaler


def _report_a1(name, y, p, classes):
    ll = log_loss(y, p, labels=list(range(len(classes))))
    per = {}
    for i, c in enumerate(classes):
        yi = (y == i).astype(int)
        per[c] = dict(rate=float(yi.mean()), mean_p=float(p[:, i].mean()))
    print(f"  {name}: logloss {ll:.5f}; per-class rate vs mean-p:",
          {c: (round(v['rate'], 4), round(v['mean_p'], 4))
           for c, v in per.items()}, flush=True)
    return ll


def fit_a_models(pa, X, cal_year):
    y = pa["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    tr = (pa["Season"] < cal_year).values
    ca = (pa["Season"] == cal_year).values
    Xf = X.astype(np.float32)

    print(f"A1: train {tr.sum():,} rows (<= {cal_year - 1}), calibrate "
          f"{ca.sum():,} ({cal_year})", flush=True)
    a1 = hgbt()
    a1.fit(Xf[tr], y[tr])
    p_cal_raw = a1.predict_proba(Xf[ca])
    scaler = VectorScaler().fit(p_cal_raw, y[ca])
    p_cal = scaler.transform(p_cal_raw)

    ll_model = _report_a1("A1 calibrated", y[ca], p_cal, CLASSES)
    lg = pd.Series(y[tr]).value_counts(normalize=True).sort_index().values
    ll_league = log_loss(y[ca], np.tile(lg, (ca.sum(), 1)),
                         labels=list(range(len(CLASSES))))
    rate_cols = [f"b_{c}_rate" for c in F.BAT_RATES]
    have = [c for c in rate_cols if c in X.columns]
    bat_only = X.loc[ca, have].to_numpy(dtype=float)
    rest = np.clip(1 - np.nansum(bat_only, axis=1), 1e-6, 1)
    marg = np.zeros((ca.sum(), len(CLASSES)))
    for j, c in enumerate(CLASSES):
        col = f"b_{c}_rate"
        marg[:, j] = (X.loc[ca, col].to_numpy(dtype=float)
                      if col in X.columns else np.nan)
    hbp = np.full(ca.sum(), lg[CLASSES.index("HBP")])
    marg[:, CLASSES.index("HBP")] = hbp
    marg[:, CLASSES.index("IPO")] = rest - hbp
    marg = np.nan_to_num(marg, nan=1e-6)
    marg = np.clip(marg, 1e-6, 1)
    marg /= marg.sum(axis=1, keepdims=True)
    ll_marg = log_loss(y[ca], marg, labels=list(range(len(CLASSES))))
    print(f"  baselines: league {ll_league:.5f}, batter-marginal "
          f"{ll_marg:.5f} -> model gain vs league "
          f"{ll_league - ll_model:+.5f}", flush=True)

    inplay = pa["bb_type"].isin(BBTYPES).values
    y2 = pa["bb_type"].map({b: i for i, b in enumerate(BBTYPES)}).values
    tr2, ca2 = tr & inplay, ca & inplay
    print(f"A2: train {tr2.sum():,} in-play rows", flush=True)
    a2 = hgbt(max_iter=200)
    a2.fit(Xf[tr2], y2[tr2])
    p2_raw = a2.predict_proba(Xf[ca2])
    scaler2 = VectorScaler().fit(p2_raw, y2[ca2])
    ll2 = log_loss(y2[ca2], scaler2.transform(p2_raw),
                   labels=list(range(len(BBTYPES))))
    print(f"  A2 calibrated logloss {ll2:.5f}", flush=True)

    return (dict(model=a1, scaler=scaler, classes=CLASSES,
                 features=list(X.columns), cal_year=cal_year,
                 metrics=dict(logloss=ll_model, league=ll_league,
                              batter_marginal=ll_marg)),
            dict(model=a2, scaler=scaler2, classes=BBTYPES,
                 features=list(X.columns), cal_year=cal_year,
                 metrics=dict(logloss=ll2)))


HAZ_FEATS = ["bf", "cum_pitches", "tto", "inning", "outs", "score_diff",
             "k_so_far", "br_so_far", "runs_so_far", "rest_p",
             "leash_np", "leash_bf", "leash_starts", "season_idx"]


def fit_hazard(cal_year):
    hz = pd.read_parquet(STORES / "hazard_table.parquet")
    leash = pd.read_parquet(STORES / "panel_leash.parquet")
    m = F.merge_asof_panel(
        hz.rename(columns={"PitcherId": "PlayerId"}), leash, ["PlayerId"],
        ["starts", "np_sum", "bf_sum", "outs_sum"], "lz_")
    hz["leash_starts"] = m["lz_starts"]
    hz["leash_np"] = m["lz_np_sum"] / m["lz_starts"].clip(lower=1e-9)
    hz["leash_bf"] = m["lz_bf_sum"] / m["lz_starts"].clip(lower=1e-9)
    hz["season_idx"] = hz["Season"] - 2015
    tr = hz["Season"] < cal_year
    ca = hz["Season"] == cal_year
    Xh = hz[HAZ_FEATS].astype(np.float32)
    yh = hz["removed"].values
    print(f"B hazard: train {tr.sum():,} rows, removal rate "
          f"{yh[tr.values].mean():.3%}", flush=True)
    mdl = hgbt(max_iter=250)
    mdl.fit(Xh[tr.values], yh[tr.values])
    p_ca = mdl.predict_proba(Xh[ca.values])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca, yh[ca.values])
    ll = log_loss(yh[ca.values], np.clip(iso.predict(p_ca), 1e-6, 1 - 1e-6))
    base = log_loss(yh[ca.values],
                    np.full(ca.sum(), yh[tr.values].mean()))
    print(f"  hazard logloss {ll:.5f} (constant-rate baseline {base:.5f})",
          flush=True)
    return dict(model=mdl, iso=iso, features=HAZ_FEATS,
                cal_year=cal_year, metrics=dict(logloss=ll, base=base))


SB_ATT_FEATS = ["SprintSpeed", "sb_allowed_rate", "cs_rate", "PopTime",
                "CSAA", "outs", "score_close", "era_new", "lhp"]
SB_SUC_FEATS = ["SprintSpeed", "cs_rate", "PopTime", "CSAA", "lhp",
                "era_new"]


def fit_sb(cal_year):
    sb = pd.read_parquet(STORES / "sb_table.parquet")
    sb["score_close"] = (sb["score_diff"].abs() <= 2).astype(float)
    sb["era_new"] = (sb["Season"] >= 2023).astype(float)
    sb["lhp"] = (sb["p_throws"].astype(str) == "L").astype(float)
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
    print(f"D SB: {tr.sum():,} opps, attempt rate "
          f"{sb.loc[tr, 'attempt'].mean():.3%} (mean pred "
          f"{p_att.mean():.3%}); success shifts {shifts}; "
          f"scale {scale}", flush=True)
    return dict(attempt=att, success=suc, att_features=SB_ATT_FEATS,
                suc_features=SB_SUC_FEATS, scale=scale,
                success_logit_shift=shifts)


def fit_latent():
    """Moment-match the game-level latent sigmas against history using
    the simulator: grid over (pitcher-form, offense-day, HR-env) logit
    sigmas, replay a sample of games, and pick the triple whose team-run
    and starter-K dispersion best matches the actual distributions."""
    import backtest as B  # local import: sim/backtest exist by P2
    res = B.moment_match_latent()
    (ART / "latent.json").write_text(json.dumps(res, indent=1))
    print(f"latent fitted: {res}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the feature stores first")
    ap.add_argument("--design-eval", action="store_true",
                    help="train <=2023, evaluate 2024 (no serve artifacts)")
    ap.add_argument("--fit-latent", action="store_true",
                    help="moment-match latent sigmas (requires sim.py)")
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

    cal_year = DESIGN_CAL_YEAR if args.design_eval else SERVE_CAL_YEAR
    stores = F.load_stores()
    pa = pd.read_parquet(STORES / "pa_table.parquet")
    print(f"assembling features for {len(pa):,} PAs ...", flush=True)
    X, cols = F.assemble_features(pa, stores)

    a1, a2 = fit_a_models(pa, X, cal_year)
    hz = fit_hazard(cal_year)
    sb = fit_sb(cal_year)

    if args.design_eval:
        print("design-eval run: artifacts NOT written", flush=True)
        return
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(a1, ART / "a1_model.joblib")
    joblib.dump(a2, ART / "a2_model.joblib")
    joblib.dump(hz, ART / "hazard_model.joblib")
    joblib.dump(sb, ART / "sb_models.joblib")
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
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("artifacts written", flush=True)


if __name__ == "__main__":
    main()
