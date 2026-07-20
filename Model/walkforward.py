"""Rolling-origin walk-forward evaluation of the PA-outcome model.

THE evaluation regime (replaces freeze/forward-test, user decision
2026-07-20): for each eval season Y the model is built exactly the way
production builds the serving model — trained on seasons <= Y-2 with
early stopping + vector scaling on season Y-1 — and scored on season Y,
which it has never seen in any form. Rolling the origin across seasons
gives several honest out-of-sample folds instead of one design year;
the aggregate is the model-quality number, the per-fold spread is its
stability.

This is PA-level (composite 8-class log loss vs the league and
batter-marginal baselines — the exact metric train.py optimizes).
Market-level walk-forward is structurally impossible before 2026 (no
archived odds, no projected lineups), which is why this harness scores
the model, not the bankroll.

Report-only: no serve artifact is ever written or touched.

Usage:
    python Model/walkforward.py                    # folds 2022..2026
    python Model/walkforward.py --years 2024 2025  # chosen eval years

Output: console + artifacts/walkforward_report.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F     # noqa: E402
import train as T        # noqa: E402

ART = F.ART
STORES = ART / "stores"
CLASSES = T.CLASSES


def _compose_eval(a1, a2, Xe):
    p1 = a1["t1"]["scaler"].transform(a1["t1"]["model"].predict_proba(Xe))
    p2 = a2["scaler"].transform(a2["model"].predict_proba(Xe))
    p3 = np.stack(
        [a1["t3"]["scaler"].transform(
            a1["t3"]["model"].predict_proba(T._t3_matrix(Xe, bi)))
         for bi in range(4)], axis=1)
    p8, _ = F.tree_compose(p1, p2, p3)
    return p8


def run(years):
    t0 = time.time()
    stores = F.load_stores()
    pa = pd.read_parquet(STORES / "pa_table.parquet")
    print(f"pa_table: {len(pa):,} rows, seasons "
          f"{int(pa.Season.min())}..{int(pa.Season.max())}", flush=True)
    X, cols = F.assemble_features(pa, stores)
    print(f"features assembled: {len(cols)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    y8 = pa["label"].map({c: i for i, c in enumerate(CLASSES)}).values

    folds = []
    for Y in years:
        cal = Y - 1
        need = pa.Season < cal
        ev = (pa.Season == Y).values
        if need.sum() < 100_000 or ev.sum() == 0:
            print(f"fold {Y}: skipped (train rows {need.sum():,}, "
                  f"eval rows {ev.sum():,})", flush=True)
            continue
        T._banner(f"WALK-FORWARD FOLD — train <= {cal - 1}, "
                  f"calibrate {cal}, evaluate {Y}")
        sub = (pa.Season <= cal).values
        pa_f = pa[sub].reset_index(drop=True)
        X_f = X[sub].reset_index(drop=True)
        a2 = T.fit_a2(pa_f, X_f, cal)
        a1 = T.fit_a1_tree(pa_f, X_f, cal, a2)
        del pa_f, X_f

        Xe = X[ev].astype(np.float32).to_numpy()
        p8 = _compose_eval(a1, a2, Xe)
        ll = T._report_a1(f"held-out {Y}", y8[ev], p8, CLASSES)
        ll_league, ll_marg = T._a1_baselines(pa, X, y8, sub, ev)
        T._kv("baseline: league", f"{ll_league:.5f}   "
              f"(gain {ll_league - ll:+.5f})")
        T._kv("baseline: batter-marginal", f"{ll_marg:.5f}   "
              f"(gain {ll_marg - ll:+.5f})")
        folds.append(dict(eval_year=Y, n_train=int((pa.Season < cal).sum()),
                          n_cal=int((pa.Season == cal).sum()),
                          n_eval=int(ev.sum()),
                          logloss=round(float(ll), 5),
                          league=round(float(ll_league), 5),
                          batter_marginal=round(float(ll_marg), 5),
                          gain_vs_league=round(float(ll_league - ll), 5),
                          gain_vs_marginal=round(float(ll_marg - ll), 5)))

    if not folds:
        sys.exit("no evaluable folds")
    df = pd.DataFrame(folds)
    T._banner("WALK-FORWARD SUMMARY (rolling origin)")
    print(df.to_string(index=False), flush=True)
    w = df.n_eval.to_numpy(dtype=float)
    agg = {k: float(np.average(df[k], weights=w))
           for k in ("logloss", "gain_vs_league", "gain_vs_marginal")}
    print(f"\nPA-weighted across folds: logloss {agg['logloss']:.5f}  "
          f"gain vs league {agg['gain_vs_league']:+.5f}  "
          f"vs batter-marginal {agg['gain_vs_marginal']:+.5f}",
          flush=True)
    print(f"fold spread (gain vs league): "
          f"{df.gain_vs_league.min():+.5f} .. "
          f"{df.gain_vs_league.max():+.5f}", flush=True)

    report = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                  scheme="train<=Y-2, calibrate Y-1, evaluate Y",
                  folds=folds, aggregate=agg,
                  minutes=round((time.time() - t0) / 60, 1))
    out = ART / "walkforward_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nreport -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2022, 2023, 2024, 2025, 2026],
                    help="eval seasons (each trains <=Y-2, "
                         "calibrates on Y-1)")
    args = ap.parse_args()
    run(sorted(args.years))
