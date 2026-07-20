"""Probability-quality evaluation, output calibration, and the CLV gate.

Three jobs, all driven by full serve-path replays of historical slates:

grade_replay        simulate a date range, grade every batter binary,
                    starter line and game market against the box scores;
                    per-market log loss / Brier vs the base rate.
fit_calibrators     fit ONE shared monotone map per market FAMILY
                    (Platt scaling in logit space) on replay rows and
                    write artifacts/output_calibrators.joblib. A single
                    monotone map per family preserves cross-line
                    coherence by construction — P(2+ hits) can never be
                    calibrated above P(1+ hit) — and the two-parameter
                    logistic cannot emit the hard 0/1 tails isotonic
                    calibration produced on sparse extremes.
market_gate         the sample-based CLV gate: grade the model against
                    the captured odds store (close AND open), per market
                    family — n graded prices, log-loss edge vs the
                    de-vigged close, date-block bootstrap CIs,
                    Benjamini-Hochberg control across families,
                    edge-bucket realization tables, and close-quality
                    columns (share of single-capture prices, median
                    open->close capture span). Verdicts are
                    PASS / NO-EDGE / INSUFFICIENT n — never elapsed
                    weeks.
ab_compare          the paired ship test: two replay-row ledgers over
                    the same slates joined row-for-row on
                    (GamePk, market, PlayerId), date-block bootstrap CI
                    on the per-row log-loss DIFFERENCE, BH control
                    across families — strictly more power than reading
                    two aggregate ledgers side by side.

Usage:
    python Model/evaluate.py --grade --start 2025-06-01 --end 2025-06-03
    python Model/evaluate.py --fit-calibrators --start 2025-05-01 \
        --end 2025-05-15
    python Model/evaluate.py --gate --start 2026-07-08 --end 2026-07-17
    python Model/evaluate.py --ab artifacts/rows_baseline.parquet \
        artifacts/calib_rows.parquet
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as B     # noqa: E402
import features as F     # noqa: E402
import odds as O         # noqa: E402
import predict as PR     # noqa: E402
import sim               # noqa: E402

ART = PR.ART

BAT_ACTUAL = {
    "HR": lambda r: r.HR >= 1, "Hit": lambda r: r.H >= 1,
    "2+ Hits": lambda r: r.H >= 2,
    "2+ TB": lambda r: r.TB >= 2, "3+ TB": lambda r: r.TB >= 3,
    "4+ TB": lambda r: r.TB >= 4,
    "Run": lambda r: r.R >= 1, "2+ Runs": lambda r: r.R >= 2,
    "RBI": lambda r: r.RBI >= 1, "2+ RBI": lambda r: r.RBI >= 2,
    "BB": lambda r: r.BB >= 1, "SB": lambda r: r.SB >= 1,
    "K": lambda r: r.SO >= 1, "2+ K": lambda r: r.SO >= 2,
    "3+ K": lambda r: r.SO >= 3,
    "Single": lambda r: (r.H - r["2B"] - r["3B"] - r.HR) >= 1,
    "Double": lambda r: r["2B"] >= 1,
    "Triple": lambda r: r["3B"] >= 1,
    "H+R+RBI 2+": lambda r: (r.H + r.R + r.RBI) >= 2,
    "H+R+RBI 3+": lambda r: (r.H + r.R + r.RBI) >= 3,
    "H+R+RBI 4+": lambda r: (r.H + r.R + r.RBI) >= 4,
}


def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    return float(np.mean((np.asarray(p, float)
                          - np.asarray(y, float)) ** 2))


def _load_actuals():
    gb = pd.read_csv(PR.DATA / "mlb_game_batting.csv",
                     encoding="utf-8-sig", low_memory=False,
                     usecols=["GamePk", "PlayerId", "PA", "H", "2B",
                              "3B", "HR", "R", "RBI", "BB", "SO", "SB",
                              "TB"])
    for c in gb.columns:
        gb[c] = pd.to_numeric(gb[c], errors="coerce")
    gp = pd.read_csv(PR.DATA / "mlb_game_pitching.csv",
                     encoding="utf-8-sig", low_memory=False,
                     usecols=["GamePk", "PlayerId", "GS", "IP", "H",
                              "BB", "SO", "ER"])
    for c in gp.columns:
        gp[c] = pd.to_numeric(gp[c], errors="coerce")
    ip = gp.IP.fillna(0)
    gp["OUTS"] = (ip.astype(int) * 3 + round((ip % 1) * 10)).astype(int)
    return gb, gp


def replay_rows(start, end, n_sims=4000, max_games=None, progress=True):
    """Replay slates through the full serve path and emit one long frame:
    [GamePk, Date, family, market, p, y] for batter binaries, starter
    lines and game markets — the raw material for grading, calibration
    and the gate."""
    P = PR.Predictor()
    # RAW rows by contract: replay ledgers feed fit_calibrators and
    # ab_compare, so the serve-time calibrators and heads must NOT be
    # baked in — a fit on already-calibrated rows composes on itself
    # and the written map is then wrong on raw serve probabilities
    # (the 2026-07-20 phantom +0.21 A/B artifact). The CLV gate path
    # (market_gate) keeps its own single _cal application.
    P.calib = {}
    P.heads = None
    games = P.stores.raw["games"]
    span = games[(games.Date >= pd.Timestamp(start))
                 & (games.Date <= pd.Timestamp(end))]
    if max_games:
        span = span.head(max_games)
    lineups, starters, umps, wx = B._spec_frames(P)
    gb, gp = _load_actuals()

    rows = []
    n_done = 0
    for _, g in span.iterrows():
        spec = B.build_spec(P, g, lineups, starters, umps, wx)
        if len(spec["away_lineup"]) < 9 or None in (
                spec["away_starter"], spec["home_starter"]):
            continue
        out = P.predict_slate([spec], n_sims=n_sims)
        f = PR.game_frame(out[0])
        act = gb[gb.GamePk == g.GamePk].set_index("PlayerId")
        pact = gp[(gp.GamePk == g.GamePk)
                  & (gp.GS == 1)].set_index("PlayerId")
        _emit_game_rows(rows, g, f, act, pact)
        n_done += 1
        if progress and n_done % 25 == 0:
            print(f"  replayed {n_done} games...", flush=True)
    return pd.DataFrame(rows, columns=ROW_COLS)


ROW_COLS = ["GamePk", "Date", "family", "market", "p", "y", "PlayerId",
            "Team", "Home"]


def _emit_game_rows(rows, g, f, act, pact):
    """Grade one simulated game frame against its box score and append
    the ledger rows — shared by the classic and batched replay paths."""
    pk = g.GamePk
    date = str(pd.Timestamp(g.Date).date())
    home_ab, away_ab = str(g.HomeTeam), str(g.AwayTeam)

    for br in f["bat"]:
        if br["ID"] not in act.index:
            continue
        ar = act.loc[br["ID"]]
        if isinstance(ar, pd.DataFrame):
            ar = ar.iloc[0]
        if not (ar.PA > 0):
            continue
        bid, bteam = int(br["ID"]), str(br["Team"])
        bhome = int(bteam == home_ab)
        for mkt, fn in BAT_ACTUAL.items():
            rows.append((pk, date, PR.COL_FAM[mkt], mkt,
                         float(br[mkt]), int(bool(fn(ar))),
                         bid, bteam, bhome))
    for pr_ in f["pit"]:
        if pr_["ID"] not in pact.index:
            continue
        ar = pact.loc[pr_["ID"]]
        if isinstance(ar, pd.DataFrame):
            ar = ar.iloc[0]
        pid_, pteam = int(pr_["ID"]), str(pr_["Team"])
        phome = int(pteam == home_ab)
        for x in PR.K_LINES:
            rows.append((pk, date, "pk", f"K > {x}",
                         float(pr_[f"K > {x}"]), int(ar.SO > x),
                         pid_, pteam, phome))
        for x in PR.OUT_LINES:
            rows.append((pk, date, "pout", f"Outs > {x}",
                         float(pr_[f"Outs > {x}"]),
                         int(ar.OUTS > x), pid_, pteam, phome))
        for x in PR.PHA_LINES:
            rows.append((pk, date, "pha", f"Hits > {x}",
                         float(pr_[f"Hits > {x}"]), int(ar.H > x),
                         pid_, pteam, phome))
        for x in PR.PBB_LINES:
            rows.append((pk, date, "pbb", f"BB > {x}",
                         float(pr_[f"BB > {x}"]), int(ar.BB > x),
                         pid_, pteam, phome))
        for x in PR.PER_LINES:
            rows.append((pk, date, "per", f"ER > {x}",
                         float(pr_[f"ER > {x}"]), int(ar.ER > x),
                         pid_, pteam, phome))
    tot = (pd.to_numeric(g.AwayScore, errors="coerce")
           + pd.to_numeric(g.HomeScore, errors="coerce"))
    hw = int(pd.to_numeric(g.HomeScore, errors="coerce")
             > pd.to_numeric(g.AwayScore, errors="coerce"))
    rows.append((pk, date, "ml", "home ML",
                 float(f["game"]["_home_wp"]), hw, -1, home_ab, 1))
    for x in PR.TOTAL_LINES:
        rows.append((pk, date, "tot", f"Runs > {x}",
                     float(f["game"][f"Runs > {x}"]),
                     int(tot > x), -1, home_ab, 1))
    for x in PR.TEAM_TOTAL_LINES:
        rows.append((pk, date, "tt", f"Away Runs > {x}",
                     float(f["game"][f"Away Runs > {x}"]),
                     int(pd.to_numeric(g.AwayScore,
                                       errors="coerce") > x),
                     -1, away_ab, 0))
        rows.append((pk, date, "tt", f"Home Runs > {x}",
                     float(f["game"][f"Home Runs > {x}"]),
                     int(pd.to_numeric(g.HomeScore,
                                       errors="coerce") > x),
                     -1, home_ab, 1))


def replay_rows_batch(start, end, n_sims=4000, chunk=64, progress=True,
                      backend="gpu", max_games=None):
    """replay_rows on the batched pipeline: sim_batch.prepare_games
    amortizes the pandas/model overhead across each chunk, run_batch
    runs the chunk as one device batch. Same rows, ~10x the
    throughput."""
    import sim_batch
    P = PR.Predictor()
    P.calib = {}     # RAW rows by contract — see replay_rows
    P.heads = None
    games = P.stores.raw["games"]
    span = games[(games.Date >= pd.Timestamp(start))
                 & (games.Date <= pd.Timestamp(end))]
    if max_games:
        span = span.head(max_games)
    lineups, starters, umps, wx = B._spec_frames(P)
    gb, gp = _load_actuals()
    gb_by = {pk: d for pk, d in gb.groupby("GamePk")}
    gp_by = {pk: d for pk, d in
             gp[gp.GS == 1].groupby("GamePk")}
    empty_b = gb.iloc[0:0].set_index("PlayerId")
    empty_p = gp.iloc[0:0].set_index("PlayerId")

    rows = []
    state = {"done": 0, "chunk": 0}
    pend_specs, pend_games = [], []

    def flush():
        if not pend_specs:
            return
        state["chunk"] += 1
        pm = sim_batch.prepare_games(P, pend_specs, n_sims=n_sims)
        res_list = sim_batch.run_batch(
            [p for p, _ in pm], n_sims=n_sims,
            seed=1000 + state["chunk"],
            seasons=[m["season"] for _, m in pm],
            is_dh=[bool(s.get("is_dh")) for s in pend_specs],
            backend=backend)
        for res, (_, meta), spec, g in zip(res_list, pm, pend_specs,
                                           pend_games):
            res["meta"] = meta
            res["spec"] = spec
            res["calib"] = P.calib
            f = PR.game_frame(res)
            act = gb_by.get(g.GamePk)
            act = (act.set_index("PlayerId") if act is not None
                   else empty_b)
            pact = gp_by.get(g.GamePk)
            pact = (pact.set_index("PlayerId") if pact is not None
                    else empty_p)
            _emit_game_rows(rows, g, f, act, pact)
            state["done"] += 1
            if progress and state["done"] % 100 == 0:
                print(f"  replayed {state['done']} games...",
                      flush=True)
        pend_specs.clear()
        pend_games.clear()

    for _, g in span.iterrows():
        spec = B.build_spec(P, g, lineups, starters, umps, wx)
        if len(spec["away_lineup"]) < 9 or None in (
                spec["away_starter"], spec["home_starter"]):
            continue
        pend_specs.append(spec)
        pend_games.append(g)
        if len(pend_specs) >= chunk:
            flush()
    flush()
    return pd.DataFrame(rows, columns=ROW_COLS)


def grade_replay(start, end, n_sims=4000, max_games=None):
    df = replay_rows(start, end, n_sims, max_games)
    print(f"\ngraded {df.GamePk.nunique()} games, {len(df):,} rows")
    rep = []
    for mkt, sub in df.groupby("market"):
        base = sub.y.mean()
        rep.append(dict(
            market=mkt, n=len(sub), rate=round(base, 4),
            mean_p=round(sub.p.mean(), 4),
            logloss=round(logloss(sub.y, sub.p), 5),
            ll_base=round(logloss(sub.y, np.full(len(sub), base)), 5),
            brier=round(brier(sub.y, sub.p), 5)))
    rep = pd.DataFrame(rep).sort_values("market")
    rep["gain"] = rep.ll_base - rep.logloss
    print(rep.to_string(index=False))
    return df


# families whose per-LINE calibration beat the shared family map on a
# held-out time split (W4.12 study, 2026-07-20: pout +0.0058 holdout
# logloss, the Outs>18.5 hot rung eliminated; per/pha were negative,
# pk/pbb within noise — re-assess them at future calibration refreshes)
LINE_CAL_FAMS = ("pout",)


def fit_calibrators(start, end, n_sims=4000, min_n=500,
                    max_games=None, reuse_rows=False, batched=False):
    """One shared Platt map (logit-space logistic) per family; identity
    (absent) below min_n, on a single-class sample, or on a
    non-positive slope. Families in LINE_CAL_FAMS additionally get one
    Platt map per LINE (stored under "_lines" keyed by market string;
    the line map wins at apply time, the family map is the fallback).
    Replay rows are cached to artifacts/calib_rows.parquet so a refit
    after a calibration-code change can skip the replay
    (--reuse-rows)."""
    cache = ART / "calib_rows.parquet"
    if reuse_rows and cache.exists():
        df = pd.read_parquet(cache)
        print(f"reusing {len(df):,} cached replay rows ({cache.name})")
    else:
        df = (replay_rows_batch(start, end, n_sims, max_games=max_games)
              if batched else
              replay_rows(start, end, n_sims, max_games))
        F.write_artifact(cache, df.to_parquet)
    out = {}
    print(f"\nfitting calibrators on {len(df):,} rows "
          f"({df.GamePk.nunique()} games)")
    for fam, sub in df.groupby("family"):
        if len(sub) < min_n:
            print(f"  {fam}: n={len(sub)} < {min_n} -> identity")
            continue
        if sub.y.nunique() < 2:
            print(f"  {fam}: single-class sample -> identity")
            continue
        p = np.clip(sub.p.values, 1e-6, 1 - 1e-6)
        z = (np.log(p) - np.log1p(-p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(z, sub.y.values)
        a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
        if b <= 0:
            print(f"  {fam}: non-positive slope b={b:.3f} -> identity")
            continue
        cal = F.PlattCal(a, b)
        before = logloss(sub.y, sub.p)
        after = logloss(sub.y, cal.predict(sub.p.values))
        out[fam] = cal
        print(f"  {fam}: n={len(sub):,} logloss {before:.5f} -> "
              f"{after:.5f} (in-sample; a={a:+.3f}, b={b:.3f})")
    lines = {}
    for fam in LINE_CAL_FAMS:
        for mkt, sub in df[df.family == fam].groupby("market"):
            if len(sub) < max(min_n, 3000) or sub.y.nunique() < 2:
                continue
            p = np.clip(sub.p.values, 1e-6, 1 - 1e-6)
            z = (np.log(p) - np.log1p(-p)).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(z, sub.y.values)
            a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
            if b <= 0:
                continue
            cal = F.PlattCal(a, b)
            before = logloss(sub.y, sub.p)
            after = logloss(sub.y, cal.predict(sub.p.values))
            lines[str(mkt)] = cal
            print(f"  line {mkt}: n={len(sub):,} logloss {before:.5f} "
                  f"-> {after:.5f} (in-sample; a={a:+.3f}, b={b:.3f})")
    if lines:
        out["_lines"] = lines
    # stamp the fit range so downstream evaluation can flag overlap
    # (every consumer looks families up by key, so "_meta" is inert)
    out["_meta"] = dict(fit_start=str(df.Date.min()),
                        fit_end=str(df.Date.max()),
                        n_rows=int(len(df)),
                        n_games=int(df.GamePk.nunique()))
    F.write_artifact(ART / "output_calibrators.joblib",
                     lambda p: joblib.dump(out, p))
    print(f"wrote {len(out) - 1} family calibrators "
          f"(fit range {out['_meta']['fit_start']}.."
          f"{out['_meta']['fit_end']}) -> "
          f"{ART / 'output_calibrators.joblib'}")
    return out


def _warn_calib_overlap(calib, start, end, context):
    """Loud warning when an evaluation range overlaps the output-
    calibrator fit range — calibrated probabilities are then partly
    in-sample. A warning, not a refusal: the in-sample view is sometimes
    wanted deliberately; the discipline just has to be visible."""
    meta = (calib or {}).get("_meta")
    if not meta or start is None or end is None:
        return
    fs, fe = str(meta.get("fit_start")), str(meta.get("fit_end"))
    s, e = str(start), str(end)
    if not (e < fs or s > fe):
        print(f"\n*** WARNING [{context}]: eval range {s}..{e} overlaps "
              f"the output-calibrator fit range {fs}..{fe} — calibrated "
              "probabilities are partly IN-SAMPLE here; use a disjoint "
              "range for an honest read. ***\n")


def skill_ledger(start=None, end=None):
    """The honest where-does-the-model-have-skill document, computed
    straight from artifacts/calib_rows.parquet — NEVER re-sims. Per
    family and per market: n, base rate, raw and calibrated log loss vs
    the base-rate baseline, AUC, Brier. Writes CSVs beside the parquet
    and prints both tables."""
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(ART / "calib_rows.parquet")
    if start:
        df = df[df.Date >= str(start)]
    if end:
        df = df[df.Date <= str(end)]
    cal_path = ART / "output_calibrators.joblib"
    cal = joblib.load(cal_path) if cal_path.exists() else {}
    df = df.copy()
    if len(df):
        _warn_calib_overlap(cal, start or df.Date.min(),
                            end or df.Date.max(), "skill ledger")
    df["p_cal"] = df["p"]
    for fam in df["family"].unique():
        c = cal.get(fam)
        if c is not None:
            mask = df["family"] == fam
            df.loc[mask, "p_cal"] = np.clip(
                c.predict(df.loc[mask, "p"].values), 1e-6, 1 - 1e-6)

    def _table(keys):
        rep = []
        for key, sub in df.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            base = sub.y.mean()
            try:
                auc = (roc_auc_score(sub.y, sub.p)
                       if sub.y.nunique() > 1 else np.nan)
            except ValueError:
                auc = np.nan
            rep.append(dict(
                **dict(zip(keys, key)),
                n=len(sub), rate=round(base, 4),
                mean_p=round(sub.p_cal.mean(), 4),
                ll_raw=round(logloss(sub.y, sub.p), 5),
                ll_cal=round(logloss(sub.y, sub.p_cal), 5),
                ll_base=round(logloss(sub.y,
                                      np.full(len(sub), base)), 5),
                auc=round(auc, 4) if auc == auc else np.nan,
                brier=round(brier(sub.y, sub.p_cal), 5)))
        rep = pd.DataFrame(rep)
        rep["gain"] = (rep.ll_base - rep.ll_cal).round(5)
        return rep.sort_values("gain", ascending=False)

    fam_rep = _table(["family"])
    mkt_rep = _table(["family", "market"])
    print(f"\nSKILL LEDGER: {len(df):,} rows, "
          f"{df.GamePk.nunique():,} games, {df.Date.nunique()} slates "
          f"({df.Date.min()}..{df.Date.max()})\n")
    print("per family (gain = base-rate logloss - calibrated logloss):")
    print(fam_rep.to_string(index=False))
    print("\nper market:")
    print(mkt_rep.to_string(index=False))
    F.write_artifact(ART / "skill_ledger_families.csv",
                     lambda p: fam_rep.to_csv(p, index=False), backup=False)
    F.write_artifact(ART / "skill_ledger_markets.csv",
                     lambda p: mkt_rep.to_csv(p, index=False), backup=False)
    print(f"\nwrote skill_ledger_families.csv / skill_ledger_markets.csv"
          f" -> {ART}")
    return fam_rep, mkt_rep


def ab_compare(path_a, path_b, start=None, end=None, boot=500,
               alpha=0.05, min_n=800):
    """Paired A/B between two replay-row ledgers graded over the same
    slates (calib_rows-style parquets; A = baseline, B = candidate —
    copy artifacts/calib_rows.parquet aside before replaying the new
    stack). Rows join on (GamePk, market, PlayerId) so every comparison
    is like-for-like; the mean per-row log-loss difference gets a
    date-block bootstrap CI and BH control across families. RAW serve
    probabilities are compared — each stack refits its own output
    calibrators, so pre-calibration p is the clean model-vs-model
    signal. Positive delta = B better."""
    key = ["GamePk", "market", "PlayerId"]
    a = pd.read_parquet(path_a).drop_duplicates(key)
    b = pd.read_parquet(path_b).drop_duplicates(key)
    if start:
        a, b = a[a.Date >= str(start)], b[b.Date >= str(start)]
    if end:
        a, b = a[a.Date <= str(end)], b[b.Date <= str(end)]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    bad = m.y_a != m.y_b
    if bad.any():
        print(f"WARNING: {int(bad.sum())} joined rows disagree on the "
              "realized outcome — dropped (ledgers graded from "
              "different box-score data?)")
        m = m[~bad]
    if m.empty:
        print("no shared rows between the two ledgers")
        return m
    print(f"paired A/B: {len(m):,} shared rows "
          f"({len(a) - len(m):,} only in A, {len(b) - len(m):,} only "
          f"in B), {m.Date_a.nunique()} slates "
          f"{m.Date_a.min()}..{m.Date_a.max()}")

    y = m.y_a.values.astype(float)
    pa = np.clip(m.p_a.values.astype(float), 1e-6, 1 - 1e-6)
    pb = np.clip(m.p_b.values.astype(float), 1e-6, 1 - 1e-6)
    m = m.assign(
        Date=m.Date_a, family=m.family_a,
        ll_a=-(y * np.log(pa) + (1 - y) * np.log(1 - pa)),
        ll_b=-(y * np.log(pb) + (1 - y) * np.log(1 - pb)))
    m["d"] = m.ll_a - m.ll_b

    rng = np.random.default_rng(7)

    def _boot(sub):
        """Date-block bootstrap of the mean per-row delta: resample
        slates, delta = sum of per-date delta sums / total n. Returns
        (ci_lo, ci_hi, two-sided p)."""
        g = sub.groupby("Date")["d"].agg(["sum", "size"])
        sums = g["sum"].values
        ns = g["size"].values.astype(float)
        k = len(sums)
        if k < 2:
            return np.nan, np.nan, 1.0
        idx = rng.integers(0, k, size=(boot, k))
        ds = sums[idx].sum(axis=1) / ns[idx].sum(axis=1)
        p_two = 2.0 * min(float((ds <= 0).mean()),
                          float((ds >= 0).mean()))
        return (float(np.quantile(ds, 0.05)),
                float(np.quantile(ds, 0.95)), min(p_two, 1.0))

    stats = []
    for fam, sub in m.groupby("family"):
        lo, hi, p_two = _boot(sub)
        stats.append(dict(
            family=fam, n=len(sub),
            ll_a=round(float(sub.ll_a.mean()), 5),
            ll_b=round(float(sub.ll_b.mean()), 5),
            delta=round(float(sub.d.mean()), 5),
            ci_lo=round(lo, 5), ci_hi=round(hi, 5), p_raw=p_two))
    rep = pd.DataFrame(stats).sort_values("p_raw")
    nf = len(rep)
    q = rep.p_raw.values * nf / np.arange(1, nf + 1)
    rep["p_bh"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
    rep["verdict"] = np.where(
        rep.n < min_n, "INSUFFICIENT n",
        np.where((rep.p_bh < alpha) & (rep.ci_lo > 0), "B BETTER",
                 np.where((rep.p_bh < alpha) & (rep.ci_hi < 0),
                          "A BETTER", "TIE")))
    lo, hi, p_two = _boot(m)
    rep = pd.concat([rep, pd.DataFrame([dict(
        family="ALL", n=len(m),
        ll_a=round(float(m.ll_a.mean()), 5),
        ll_b=round(float(m.ll_b.mean()), 5),
        delta=round(float(m.d.mean()), 5),
        ci_lo=round(lo, 5), ci_hi=round(hi, 5),
        p_raw=p_two, p_bh=np.nan, verdict="")])], ignore_index=True)
    print("\nper family (delta = ll_A - ll_B on raw p; positive = B "
          "better; 90% date-block bootstrap CI, BH across families):")
    print(rep.to_string(index=False))
    mk = (m.groupby(["family", "market"])
          .agg(n=("d", "size"), delta=("d", "mean")).reset_index())
    mk["delta"] = mk.delta.round(5)
    print("\nper market (point deltas only, no test):")
    print(mk.sort_values("delta", ascending=False)
          .to_string(index=False))

    # Goal board: ranking-only comparison on raw p — the family
    # calibrators are monotone, so within-market sort order on raw p IS
    # the served sort order; stated levels are a calibration question
    # and are not compared here.
    ga = goal_metrics(m[["Date", "market"]].assign(p=pa, y=y))
    gb_ = goal_metrics(m[["Date", "market"]].assign(p=pb, y=y))
    gsec = ga.merge(gb_, on="market", suffixes=("_a", "_b"))
    gsec["d_auc"] = (gsec.auc_b - gsec.auc_a).round(4)
    gsec["d_t10"] = (gsec.t10_hit_b - gsec.t10_hit_a).round(4)
    gsec = gsec[["market", "n_a", "auc_a", "auc_b", "d_auc",
                 "t10_hit_a", "t10_hit_b", "d_t10",
                 "trust_depth_a", "trust_depth_b"]].rename(
        columns={"n_a": "n"})
    print("\ngoal board (ranking-only, raw p; top-10/slate hit rate, "
          "AUC, trust depth; positive delta = B better):")
    print(gsec.sort_values("d_t10", ascending=False)
          .to_string(index=False))
    return rep


# --------------------------------------- goal-aligned (top-of-sort) board

GOAL_TOP_N = 10          # the workbook-sort test depth
GOAL_MAX_DEPTH = 15      # deepest trust depth considered
GOAL_OR_GATE = 1.5       # odds-ratio-lift LCB a depth must clear
GOAL_BOOT = 300


def goal_metrics(df, top_n=GOAL_TOP_N, boot=GOAL_BOOT, seed=7):
    """Per-market board for the workbook-sort goal: sort a column
    high->low and the top picks should hit; >50% cells should hit at
    their stated rate. Expects ledger-style rows (Date/market/p/y).
    Pass calibrated p for level-true stated columns; the ordering
    metrics (auc, hit rates, trust_depth) are unchanged by the
    monotone family calibrators either way.

    trust_depth = deepest per-slate top-d whose odds-ratio lift over
    the market's base rate holds its slate-block-bootstrap
    10th-percentile lower bound above GOAL_OR_GATE — how far down the
    sorted column selection power is PROVEN (0 = not even the top
    pick separates from a blind bet at this sample)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    out = []
    for mkt, g in df.groupby("market", sort=False):
        y = g.y.values.astype(float)
        rate = float(y.mean())
        auc = (float(roc_auc_score(y, g.p.values))
               if 0.0 < rate < 1.0 else np.nan)
        srt = g.sort_values("p", ascending=False)
        top = srt.groupby("Date", sort=False).head(top_n)
        hi = g[g.p > 0.5]
        depth = 0
        if boot:
            # per-slate cumulative hits down the sorted column; slates
            # with fewer than GOAL_MAX_DEPTH rows saturate at their size
            Hc, Cc, Ht, Ct = [], [], [], []
            for _, sub in srt.groupby("Date", sort=False):
                yy = sub.y.values[:GOAL_MAX_DEPTH].astype(float)
                k = len(yy)
                cs = np.cumsum(yy) if k else np.zeros(1)
                h = np.full(GOAL_MAX_DEPTH, float(cs[-1]))
                c = np.full(GOAL_MAX_DEPTH, float(k))
                h[:k], c[:k] = cs, np.arange(1, k + 1, dtype=float)
                Hc.append(h)
                Cc.append(c)
                Ht.append(float(sub.y.sum()))
                Ct.append(float(len(sub)))
            S = len(Hc)
            if S >= 8:
                Hc_, Cc_ = np.array(Hc), np.array(Cc)
                Ht_, Ct_ = np.array(Ht), np.array(Ct)
                idx = rng.integers(0, S, size=(boot, S))
                tc = np.maximum(Cc_[idx].sum(1), 1.0)
                bc = np.maximum(Ct_[idx].sum(1), 1.0)
                pr = np.clip(Hc_[idx].sum(1) / tc, 1e-6, 1 - 1e-6)
                br = np.clip(Ht_[idx].sum(1) / bc, 1e-6, 1 - 1e-6)
                orl = ((pr / (1 - pr))
                       / (br / (1 - br))[:, None])
                ok = np.where(np.quantile(orl, 0.10, axis=0)
                              >= GOAL_OR_GATE)[0]
                depth = int(ok.max() + 1) if len(ok) else 0
        out.append(dict(
            market=str(mkt), n=len(g), base=round(rate, 4),
            auc=round(auc, 4) if np.isfinite(auc) else np.nan,
            t10_stated=round(float(top.p.mean()), 4),
            t10_hit=round(float(top.y.mean()), 4),
            t10_gap=round(float(top.y.mean() - top.p.mean()), 4),
            hi_n=int(len(hi)),
            hi_stated=(round(float(hi.p.mean()), 4)
                       if len(hi) else np.nan),
            hi_hit=(round(float(hi.y.mean()), 4)
                    if len(hi) else np.nan),
            hi_gap=(round(float(hi.y.mean() - hi.p.mean()), 4)
                    if len(hi) else np.nan),
            trust_depth=depth))
    return pd.DataFrame(out)


def reliability_bands(df, lo=0.5, width=0.05):
    """Pooled stated-vs-hit ladder from `lo` upward — the '>50% cells
    hit at their number, more the higher they go' check. Pass
    calibrated p (raw p makes the levels meaningless)."""
    rows = []
    for e in np.arange(lo, 1.0 - 1e-9, width):
        b = df[(df.p >= e) & (df.p < e + width)]
        if len(b):
            rows.append(dict(
                band=f"[{e:.2f},{e + width:.2f})", n=len(b),
                stated=round(float(b.p.mean()), 4),
                hit=round(float(b.y.mean()), 4),
                gap=round(float(b.y.mean() - b.p.mean()), 4)))
    return pd.DataFrame(rows)


# ------------------------------------------------------- the CLV gate

def _odds_y(gb, gp, games_df, pk, pid, market, line):
    """Realized outcome for one captured price, or None (void/unplayed)."""
    if market in ("h2h", "totals"):
        g = games_df[games_df.GamePk == pk]
        if g.empty:
            return None
        aw = pd.to_numeric(g.AwayScore.iloc[0], errors="coerce")
        hm = pd.to_numeric(g.HomeScore.iloc[0], errors="coerce")
        if pd.isna(aw) or pd.isna(hm):
            return None
        return int(hm > aw) if market == "h2h" else \
            int(aw + hm > line)
    if market.startswith("pitcher"):
        r = gp[(gp.GamePk == pk) & (gp.PlayerId == pid)]
        if r.empty:
            return None
        r = r.iloc[0]
        stat = {"pitcher_strikeouts": r.SO, "pitcher_outs": r.OUTS,
                "pitcher_hits_allowed": r.H, "pitcher_walks": r.BB,
                "pitcher_earned_runs": r.ER}.get(market)
        return None if stat is None or pd.isna(stat) else int(stat > line)
    r = gb[(gb.GamePk == pk) & (gb.PlayerId == pid)]
    if r.empty or not (r.PA.iloc[0] > 0):
        return None
    r = r.iloc[0]
    stat = {"batter_hits": r.H, "batter_home_runs": r.HR,
            "batter_total_bases": r.TB, "batter_runs_scored": r.R,
            "batter_rbis": r.RBI, "batter_walks": r.BB,
            "batter_stolen_bases": r.SB,
            "batter_singles": r.H - r["2B"] - r["3B"] - r.HR,
            "batter_doubles": r["2B"],
            "batter_hits_runs_rbis": r.H + r.R + r.RBI}.get(market)
    return None if stat is None or pd.isna(stat) else int(stat > line)


def _gate_fingerprint(n_sims):
    """Cache key part that invalidates on any serving-stack change.
    Model artifacts are fingerprinted DIRECTLY (not just via
    manifest.json) so standalone refits — e.g. a bare fit_sb that
    bypasses the manifest — still invalidate the cache."""
    import os
    parts = [f"s{n_sims}"]
    for f_ in ("manifest.json", "output_calibrators.joblib",
               "residual_heads.joblib", "latent.json",
               "a1_model.joblib", "a2_model.joblib",
               "hazard_model.joblib", "sb_models.joblib"):
        p = ART / f_
        parts.append(str(int(os.path.getmtime(p))) if p.exists()
                     else "0")
    return "|".join(parts)


GATE_CACHE_COLS = ["key", "family", "Date", "y", "p_model", "p_close",
                   "p_open", "one_cap", "gap_min"]


def market_gate(start, end, n_sims=4000, min_n=800, boot=500,
                alpha=0.05):
    """Sample-based market-viability gate vs the captured odds store.
    Slates are re-simmed only when the serving stack or that date's
    captured odds changed since the last run (gate_rows_cache.parquet);
    unchanged slates load their graded rows from the cache."""
    P = PR.Predictor()
    _warn_calib_overlap(P.calib, start, end, "CLV gate")
    odds = pd.read_csv(O.DEFAULT_STORE, encoding="utf-8-sig",
                       low_memory=False)
    odds = odds[(odds.Date >= str(start)) & (odds.Date <= str(end))]
    games = P.stores.raw["games"]
    lineups, starters, umps, wx = B._spec_frames(P)
    gb, gp = _load_actuals()

    fp = _gate_fingerprint(n_sims)
    cache_path = ART / "gate_rows_cache.parquet"
    cache = (pd.read_parquet(cache_path) if cache_path.exists()
             else pd.DataFrame(columns=GATE_CACHE_COLS))
    if not set(GATE_CACHE_COLS) <= set(cache.columns):
        print("gate cache predates the close-quality columns — "
              "discarding (one-time re-sim)")
        cache = pd.DataFrame(columns=GATE_CACHE_COLS)
    new_frames, seen_keys = [], set()

    rows = []
    for date, day_odds in odds.groupby("Date"):
        day_games = games[games.Date == pd.Timestamp(date)]
        if day_games.empty:
            continue
        osig = int(pd.util.hash_pandas_object(
            day_odds[["PlayerId", "Team", "Market", "Line", "OverPrice",
                      "UnderPrice", "OpenOverPrice", "OpenUnderPrice"]]
            .astype(str)).sum() % 10 ** 12)
        key = f"{date}|{fp}|{len(day_odds)}|{osig}"
        seen_keys.add(key)
        hit = cache[cache.key == key] if len(cache) else cache
        if len(hit):
            rows.extend(hit[GATE_CACHE_COLS[1:]].to_dict("records"))
            print(f"  {date}: {len(hit)} rows from cache", flush=True)
            continue
        day_list = []
        # replay the slate once; map players/games to sim results
        by_pid, by_home, by_team = {}, {}, {}
        for _, g in day_games.iterrows():
            spec = B.build_spec(P, g, lineups, starters, umps, wx)
            if len(spec["away_lineup"]) < 9 or None in (
                    spec["away_starter"], spec["home_starter"]):
                continue
            res = P.predict_slate([spec], n_sims=n_sims)[0]
            by_home[g.HomeTeam] = (res, int(g.GamePk))
            by_team[str(g.AwayTeam)] = (res, int(g.GamePk), 0)
            by_team[str(g.HomeTeam)] = (res, int(g.GamePk), 1)
            for row_i, pid in enumerate(res["meta"]["players"]):
                if pid >= 0 and row_i < 20:
                    by_pid.setdefault(int(pid), (res, row_i,
                                                 int(g.GamePk)))
        print(f"  {date}: {len(by_home)} games replayed", flush=True)

        s = sim.SIDX
        # game markets group by TEAM (their PlayerId is blank — grouping
        # them by PlayerId would fold every game on the slate into one
        # group); player props group by PlayerId as before
        gm_mask = day_odds.Market.isin(("h2h", "totals", "team_totals"))
        grouped = list(day_odds[~gm_mask].groupby(
            ["PlayerId", "Market", "Line"], dropna=False)) + \
            list(day_odds[gm_mask].groupby(
                ["Team", "Market", "Line"], dropna=False))
        for (pid_s, market, line_s), grp in grouped:
            fam = PR.MKT_FAM.get(market)
            if fam is None:
                continue
            fair = O.sharp_fair(grp.to_dict("records"))
            op = [dict(OverPrice=r.get("OpenOverPrice"),
                       UnderPrice=r.get("OpenUnderPrice"),
                       Book=r.get("Book"))
                  for r in grp.to_dict("records")]
            fair_open = O.sharp_fair(op)
            if fair is None:
                continue
            # close quality: was this price ever re-captured, and how
            # long a span does open->close actually cover?
            cts = pd.to_datetime(grp.CapturedAt, errors="coerce")
            ots = pd.to_datetime(grp.OpenCapturedAt,
                                 errors="coerce").fillna(cts)
            gap_min = max(0.0, float(
                ((cts - ots).dt.total_seconds() / 60.0).fillna(0).max()))
            one_cap = int(gap_min <= 0)
            try:
                line = float(line_s)
            except (TypeError, ValueError):
                line = np.nan
            if market == "team_totals":
                if not np.isfinite(line):
                    continue
                hit = by_team.get(str(grp.Team.iloc[0]))
                if hit is None:
                    continue
                res, pk, side = hit
                p_model = PR._cal(res.get("calib"), "tt", float(
                    (res["score"][:, side] > line).mean()))
                grow = games[games.GamePk == pk]
                if grow.empty:
                    continue
                sc_ = pd.to_numeric(
                    grow.iloc[0].HomeScore if side
                    else grow.iloc[0].AwayScore, errors="coerce")
                y = None if pd.isna(sc_) else int(sc_ > line)
            elif market in ("h2h", "totals"):
                team = grp.Team.iloc[0]
                hit = by_home.get(team)
                if hit is None:
                    continue
                res, pk = hit
                if market == "h2h":
                    p_model = PR._cal(res.get("calib"), "ml", float(
                        (res["score"][:, 1] > res["score"][:, 0]).mean()))
                else:
                    p_model = PR._cal(res.get("calib"), "tot", float(
                        (res["score"].sum(axis=1) > line).mean()))
                y = _odds_y(gb, gp, games, pk, None, market, line)
            else:
                try:
                    pid = int(float(pid_s))
                except (TypeError, ValueError):
                    continue
                hit = by_pid.get(pid)
                if hit is None:
                    continue
                res, row_i, pk = hit
                t = res["tensor"]
                if market == "batter_total_bases":
                    counts = (t[:, row_i, s["B1"]]
                              + 2 * t[:, row_i, s["B2"]]
                              + 3 * t[:, row_i, s["B3"]]
                              + 4 * t[:, row_i, s["HR"]])
                elif market == "batter_hits_runs_rbis":
                    counts = (t[:, row_i, s["H"]] + t[:, row_i, s["R"]]
                              + t[:, row_i, s["RBI"]])
                else:
                    col = dict(PR._BAT_STAT, **PR._PIT_STAT).get(market)
                    if col is None:
                        continue
                    counts = t[:, row_i, s[col]]
                p_model = PR._cal(res.get("calib"), fam,
                                  float((counts > line).mean()),
                                  market=PR._line_market(market, line))
                y = _odds_y(gb, gp, games, pk, pid, market, line)
            if y is None:
                continue
            day_list.append(dict(family=fam, Date=date, y=y,
                                 p_model=p_model, p_close=fair,
                                 p_open=fair_open, one_cap=one_cap,
                                 gap_min=gap_min))
        rows.extend(day_list)
        if day_list:
            new_frames.append(
                pd.DataFrame(day_list).assign(key=key))

    # persist: rows for the current fingerprint only (a stack change
    # invalidates everything), atomically
    keep = cache[cache.key.isin(seen_keys)] if len(cache) else cache
    merged = pd.concat([keep] + new_frames, ignore_index=True) \
        if new_frames else keep
    if len(merged):
        F.write_artifact(cache_path,
                         lambda p: merged[GATE_CACHE_COLS].to_parquet(
                             p, index=False), backup=False)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no gradable prices in range")
        return df

    print(f"\nCLV GATE {start}..{end}: {len(df):,} graded prices, "
          f"{df.Date.nunique()} slates")
    rng = np.random.default_rng(5)
    dates = df.Date.unique()
    fam_stats = []
    for fam, sub in df.groupby("family"):
        d_model = logloss(sub.y, sub.p_model)
        d_close = logloss(sub.y, sub.p_close)
        delta = d_close - d_model
        boots = []
        for _ in range(boot):
            pick = rng.choice(dates, size=len(dates), replace=True)
            bs = pd.concat([sub[sub.Date == d] for d in pick])
            if len(bs) < 10:
                continue
            boots.append(logloss(bs.y, bs.p_close)
                         - logloss(bs.y, bs.p_model))
        boots = np.array(boots)
        pval = float((boots <= 0).mean()) if len(boots) else 1.0
        moved = sub[sub.p_open.notna()
                    & (np.abs(sub.p_open - sub.p_close) > 1e-9)]
        d_open = (logloss(moved.y, moved.p_open)
                  - logloss(moved.y, np.clip(moved.p_model, 0, 1))
                  ) if len(moved) >= 30 else np.nan
        recap = sub.gap_min[sub.gap_min > 0]
        fam_stats.append(dict(
            family=fam, n=len(sub),
            ll_model=round(d_model, 5), ll_close=round(d_close, 5),
            delta=round(delta, 5),
            ci_lo=round(float(np.quantile(boots, 0.05)), 5)
            if len(boots) else np.nan,
            p_raw=pval, n_moved=len(moved),
            d_vs_open=round(float(d_open), 5)
            if d_open == d_open else np.nan,
            pct_1cap=round(float(sub.one_cap.mean()), 3),
            med_gap_min=round(float(recap.median()), 1)
            if len(recap) else np.nan))
    rep = pd.DataFrame(fam_stats).sort_values("p_raw")
    # Benjamini-Hochberg across families: reverse cumulative minimum of
    # p * m / rank over the ascending-p ordering
    m = len(rep)
    q = (rep.p_raw.values * m / np.arange(1, m + 1))
    rep["p_bh"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
    rep["verdict"] = np.where(
        rep.n < min_n, "INSUFFICIENT n",
        np.where((rep.p_bh < alpha) & (rep.ci_lo > 0), "PASS",
                 "NO-EDGE"))
    print(rep.to_string(index=False))

    # edge buckets: does a bigger model-vs-close gap realize more often?
    df["edge"] = df.p_model - df.p_close
    df["bucket"] = pd.cut(df.edge, [-1, -0.10, -0.05, -0.03, 0.03,
                                    0.05, 0.10, 1])
    bt = df.groupby("bucket", observed=True).agg(
        n=("y", "size"), realized=("y", "mean"),
        close_implied=("p_close", "mean"),
        model_said=("p_model", "mean")).round(4)
    print("\nedge buckets (model - close):")
    print(bt.to_string())
    print("\nverdicts are per-family and sample-gated: PASS needs "
          f"n >= {min_n} AND BH-adjusted bootstrap CI above zero.")
    print("close quality: pct_1cap = share of graded prices captured "
          "only once (their 'close' is that lone capture — likely a "
          "soft early line, which flatters delta); med_gap_min = "
          "median open->close capture span among re-captured prices.")
    return df, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--fit-calibrators", action="store_true",
                    dest="fitcal")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--ledger", action="store_true",
                    help="skill ledger from calib_rows.parquet (no "
                         "re-sim; optional --start/--end filter)")
    ap.add_argument("--ab", nargs=2, metavar=("A_ROWS", "B_ROWS"),
                    help="paired A/B between two replay-row parquets "
                         "(A=baseline, B=candidate; optional "
                         "--start/--end filter)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--min-n", type=int, default=800)
    ap.add_argument("--reuse-rows", action="store_true",
                    help="refit from artifacts/calib_rows.parquet "
                         "instead of replaying")
    ap.add_argument("--batched", action="store_true",
                    help="batched replay path (sim_batch prepare_games "
                         "+ run_batch on CUDA)")
    args = ap.parse_args()
    needs_range = not (args.ledger or args.ab
                       or (args.fitcal and args.reuse_rows))
    if needs_range and not (args.start and args.end):
        ap.error("--start and --end are required for this mode")
    if args.ab:
        ab_compare(args.ab[0], args.ab[1], start=args.start,
                   end=args.end, min_n=args.min_n)
    elif args.ledger:
        skill_ledger(args.start, args.end)
    elif args.fitcal:
        fit_calibrators(args.start, args.end, n_sims=args.sims,
                        max_games=args.max_games,
                        reuse_rows=args.reuse_rows, batched=args.batched)
    elif args.gate:
        market_gate(args.start, args.end, n_sims=args.sims,
                    min_n=args.min_n)
    else:
        grade_replay(args.start, args.end, n_sims=args.sims,
                     max_games=args.max_games)
