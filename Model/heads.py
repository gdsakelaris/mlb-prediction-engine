"""Residual heads: one light GBM per market family, for EVERY family —
uniform process, evidence-scaled effect.

Construction (the adopted design): the CALIBRATED sim probability
enters as the BASE MARGIN (init_score = logit of the Platt output),
NEVER as a feature; the trees see context features only; shallow depth,
heavy minimum-leaf, early stopping against a time-ordered held-out
fold. A family with no residual structure early-stops to zero trees and
the head IS the identity by construction. Post-hoc family selection
("build only where the ledger shows bias") is winner's-curse selection;
this trains everywhere and lets held-out evidence set each head's size.

This module TRAINS and EVALUATES (artifacts/residual_heads.joblib +
heads_report.csv). Serving applies the artifact in
predict._apply_heads (after the family calibrators, with a cross-line
ladder guard); replays never load heads, so calib_rows stays raw and
heads always retrain against an unadjusted base. Retrain here after
every calibration replay (features.py --only teamctx first if game
results have advanced).

Requires: calib_rows.parquet WITH identity columns (PlayerId/Team/Home
— replays run after 2026-07-19) and stores/team_game_context.parquet
(python Model/features.py --only teamctx).

Usage:
    python Model/heads.py --train
    python Model/heads.py --train --min-rows 4000 --holdout 0.25
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F   # noqa: E402
import predict as PR   # noqa: E402

ART = PR.ART

# canonical definitions live in predict.py (the serve side) — aliased
# here so train-time and serve-time feature construction cannot drift
CTX_NUM = PR.HEAD_CTX
_line_of = PR._line_of


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _load_rows():
    df = pd.read_parquet(ART / "calib_rows.parquet")
    if not {"PlayerId", "Team", "Home"}.issubset(df.columns):
        raise SystemExit(
            "calib_rows.parquet predates the identity columns "
            "(PlayerId/Team/Home) — run a fresh replay first")
    ctx = pd.read_parquet(F.STORES / "team_game_context.parquet")
    df = df.merge(ctx.drop(columns=["Date"]), on="GamePk", how="left",
                  suffixes=("", "_ctx"))
    seas = df["Season"] if "Season" in df.columns else df["Season_ctx"]
    own_home = df["Home"].fillna(1).astype(int).values == 1
    for c in CTX_NUM:
        h, a = df[f"home_{c}"].values, df[f"away_{c}"].values
        df[f"own_{c}"] = np.where(own_home, h, a)
        df[f"opp_{c}"] = np.where(own_home, a, h)
    df["own_team"] = np.where(own_home, df["home_team"],
                              df["away_team"])
    df["opp_team"] = np.where(own_home, df["away_team"],
                              df["home_team"])
    df["_seas"] = seas
    deft = pd.read_parquet(F.STORES / "defense_team.parquet")[
        ["Year", "Team", "def_oaa", "FrameRV_pt"]]
    for side in ("own", "opp"):
        j = df[[f"{side}_team", "_seas"]].merge(
            deft, left_on=[f"{side}_team", "_seas"],
            right_on=["Team", "Year"], how="left")
        df[f"{side}_def_oaa"] = j["def_oaa"].values
        df[f"{side}_frame"] = j["FrameRV_pt"].values
    # game-grain effects: venue HR factor + ump run factor (predict
    # serves the same values via _game_effects; 1.0 when unknown)
    games = pd.read_csv(F.DATA / "mlb_games.csv", encoding="utf-8-sig",
                        usecols=["GamePk", "Venue"], low_memory=False)
    umps = pd.read_csv(F.DATA / "mlb_umpires.csv", encoding="utf-8-sig",
                       usecols=["GamePk", "HpUmpId"], low_memory=False)
    df = df.merge(games, on="GamePk", how="left")
    df = df.merge(umps, on="GamePk", how="left")
    pf = pd.read_parquet(F.STORES / "park_factors.parquet")[
        ["Venue", "Year", "pf_HR"]]
    df = df.merge(pf, left_on=["Venue", "_seas"],
                  right_on=["Venue", "Year"], how="left")
    uf = pd.read_parquet(F.STORES / "ump_factors.parquet")[
        ["HpUmpId", "Year", "uf_R"]]
    df = df.merge(uf, left_on=["HpUmpId", "_seas"],
                  right_on=["HpUmpId", "Year"], how="left",
                  suffixes=("", "_uf"))
    df["park_hr"] = df["pf_HR"].fillna(1.0)
    df["ump_r"] = df["uf_R"].fillna(1.0)
    return df


def _features(df):
    X = pd.DataFrame(index=df.index)
    for c in CTX_NUM:
        X[f"own_{c}"] = df[f"own_{c}"]
        X[f"opp_{c}"] = df[f"opp_{c}"]
    X["elo_diff"] = df["elo_diff"]
    for c in ("own_def_oaa", "opp_def_oaa", "own_frame", "opp_frame"):
        X[c] = df[c]
    X["home"] = df["Home"].astype(float)
    X["line"] = df["market"].map(_line_of)
    # extremity of the anchored price — shape context, not the anchor
    # itself (no sign, no level; the base margin cannot be reproduced)
    X["p_dist"] = (df["p_cal"] - 0.5).abs()
    X["park_hr"] = df["park_hr"]
    X["ump_r"] = df["ump_r"]
    return X


def train(min_rows=3000, holdout=0.25, seed=7):
    import lightgbm as lgb
    df = _load_rows()
    cal_path = ART / "output_calibrators.joblib"
    cal = joblib.load(cal_path) if cal_path.exists() else {}
    df["p_cal"] = df["p"]
    for fam in df["family"].unique():
        c = cal.get(fam)
        if c is not None:
            m = df["family"] == fam
            df.loc[m, "p_cal"] = np.clip(
                c.predict(df.loc[m, "p"].values), 1e-6, 1 - 1e-6)
    # per-line maps win over the family map exactly as at serve
    # (predict._cal) — the head's base margin must match serving
    for mkt, lc in (cal.get("_lines", {}) or {}).items():
        m = df["market"] == mkt
        if m.any():
            df.loc[m, "p_cal"] = np.clip(
                lc.predict(df.loc[m, "p"].values), 1e-6, 1 - 1e-6)

    heads, report = {}, []
    for fam, sub in df.groupby("family"):
        sub = sub.sort_values("Date")
        if len(sub) < min_rows or sub.y.nunique() < 2:
            report.append(dict(family=fam, n=len(sub), n_holdout=0,
                               best_iter=0, ll_platt=np.nan,
                               ll_head=np.nan, gain=np.nan,
                               max_adj=0.0, status="skipped"))
            continue
        dates = np.array(sorted(sub.Date.unique()))
        cut = dates[int(len(dates) * (1 - holdout))]
        tr, va = sub[sub.Date < cut], sub[sub.Date >= cut]
        Xt, Xv = _features(tr), _features(va)
        at, av = _logit(tr["p_cal"]), _logit(va["p_cal"])
        dtr = lgb.Dataset(Xt, label=tr["y"].values, init_score=at)
        dva = lgb.Dataset(Xv, label=va["y"].values, init_score=av,
                          reference=dtr)
        params = dict(objective="binary", metric="binary_logloss",
                      learning_rate=0.05, num_leaves=8, max_depth=3,
                      min_child_samples=200, feature_fraction=0.9,
                      lambda_l2=5.0, verbosity=-1, seed=seed)
        bst = lgb.train(params, dtr, num_boost_round=400,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30,
                                                      verbose=False)])
        best = int(bst.best_iteration or 0)
        # Booster.predict does NOT add init_score back — add it
        raw = (bst.predict(Xv, num_iteration=best, raw_score=True)
               if best > 0 else np.zeros(len(va)))
        p_head = 1.0 / (1.0 + np.exp(-(av + raw)))
        ll_platt = _ll(va["y"].values, va["p_cal"].values)
        ll_head = _ll(va["y"].values, p_head)
        max_adj = (float(np.max(np.abs(p_head - va["p_cal"].values)))
                   if best > 0 else 0.0)
        heads[fam] = dict(booster_str=bst.model_to_string(),
                          features=list(Xt.columns), best_iter=best,
                          trained_through=str(sub.Date.max()))
        report.append(dict(
            family=fam, n=len(sub), n_holdout=len(va), best_iter=best,
            ll_platt=round(ll_platt, 5), ll_head=round(ll_head, 5),
            gain=round(ll_platt - ll_head, 5),
            max_adj=round(max_adj, 4),
            status="identity" if best == 0 else "active"))

    rep = pd.DataFrame(report).sort_values("gain", ascending=False)
    print(f"\nRESIDUAL HEADS ({len(heads)} trained; identity = early-"
          f"stopped to 0 trees; gain = held-out logloss improvement "
          f"over Platt-only):")
    print(rep.to_string(index=False))
    F.write_artifact(ART / "residual_heads.joblib",
                     lambda p: joblib.dump(heads, p))
    F.write_artifact(ART / "heads_report.csv",
                     lambda p: rep.to_csv(p, index=False), backup=False)
    print(f"\nwrote residual_heads.joblib + heads_report.csv -> {ART}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--min-rows", type=int, default=3000)
    ap.add_argument("--holdout", type=float, default=0.25)
    args = ap.parse_args()
    if not args.train:
        ap.error("pass --train")
    train(min_rows=args.min_rows, holdout=args.holdout)
