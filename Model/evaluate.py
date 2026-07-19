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
                    Benjamini-Hochberg control across families, and
                    edge-bucket realization tables. Verdicts are
                    PASS / NO-EDGE / INSUFFICIENT n — never elapsed
                    weeks.

Usage:
    python Model/evaluate.py --grade --start 2025-06-01 --end 2025-06-03
    python Model/evaluate.py --fit-calibrators --start 2025-05-01 \
        --end 2025-05-15
    python Model/evaluate.py --gate --start 2026-07-08 --end 2026-07-17
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


def fit_calibrators(start, end, n_sims=4000, min_n=500,
                    max_games=None, reuse_rows=False, batched=False):
    """One shared Platt map (logit-space logistic) per family; identity
    (absent) below min_n, on a single-class sample, or on a
    non-positive slope. Replay rows are cached to
    artifacts/calib_rows.parquet so a refit after a calibration-code
    change can skip the replay (--reuse-rows)."""
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
    F.write_artifact(ART / "output_calibrators.joblib",
                     lambda p: joblib.dump(out, p))
    print(f"wrote {len(out)} family calibrators -> "
          f"{ART / 'output_calibrators.joblib'}")
    return out


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


def market_gate(start, end, n_sims=4000, min_n=800, boot=500,
                alpha=0.05):
    """Sample-based market-viability gate vs the captured odds store."""
    P = PR.Predictor()
    odds = pd.read_csv(O.DEFAULT_STORE, encoding="utf-8-sig",
                       low_memory=False)
    odds = odds[(odds.Date >= str(start)) & (odds.Date <= str(end))]
    games = P.stores.raw["games"]
    lineups, starters, umps, wx = B._spec_frames(P)
    gb, gp = _load_actuals()

    rows = []
    for date, day_odds in odds.groupby("Date"):
        day_games = games[games.Date == pd.Timestamp(date)]
        if day_games.empty:
            continue
        # replay the slate once; map players/games to sim results
        by_pid, by_home = {}, {}
        for _, g in day_games.iterrows():
            spec = B.build_spec(P, g, lineups, starters, umps, wx)
            if len(spec["away_lineup"]) < 9 or None in (
                    spec["away_starter"], spec["home_starter"]):
                continue
            res = P.predict_slate([spec], n_sims=n_sims)[0]
            by_home[g.HomeTeam] = (res, int(g.GamePk))
            for row_i, pid in enumerate(res["meta"]["players"]):
                if pid >= 0 and row_i < 20:
                    by_pid.setdefault(int(pid), (res, row_i,
                                                 int(g.GamePk)))
        print(f"  {date}: {len(by_home)} games replayed", flush=True)

        s = sim.SIDX
        for (pid_s, market, line_s), grp in day_odds.groupby(
                ["PlayerId", "Market", "Line"], dropna=False):
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
            try:
                line = float(line_s)
            except (TypeError, ValueError):
                line = np.nan
            if market in ("h2h", "totals"):
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
                                  float((counts > line).mean()))
                y = _odds_y(gb, gp, games, pk, pid, market, line)
            if y is None:
                continue
            rows.append(dict(family=fam, Date=date, y=y,
                             p_model=p_model, p_close=fair,
                             p_open=fair_open))
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
        fam_stats.append(dict(
            family=fam, n=len(sub),
            ll_model=round(d_model, 5), ll_close=round(d_close, 5),
            delta=round(delta, 5),
            ci_lo=round(float(np.quantile(boots, 0.05)), 5)
            if len(boots) else np.nan,
            p_raw=pval, n_moved=len(moved),
            d_vs_open=round(float(d_open), 5)
            if d_open == d_open else np.nan))
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
    needs_range = not (args.ledger or (args.fitcal and args.reuse_rows))
    if needs_range and not (args.start and args.end):
        ap.error("--start and --end are required for this mode")
    if args.ledger:
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
