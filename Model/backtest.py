"""As-of replay harness: simulate historical slates and compare against
what actually happened.

Three jobs:
  build_spec()          a historical game -> the same spec dict the GUI
                        serves (actual posted lineups, starters, park,
                        weather, ump), so the WHOLE serve path replays
  moment_match_latent() fit the game-level latent parameters (mu_env,
                        sigma_env, sigma_pitcher) by matching the sim's
                        run/strikeout distributions to the real ones on
                        a sample of games — staged coordinate search,
                        written to Model/artifacts/latent.json
  replay()              simulate a date range and grade sim probabilities
                        against actual box scores per market family
                        (evaluate.py renders the reports)

Replay uses actual lineups (pregame-projected lineups only exist in the
slate archive from 2026-07-19 forward) — quantify that gap forward, not
backward. Panels are as-of by construction; the A-model artifact trains
through 2024, so 2024 replay is near-in-sample and 2025 replay is the
honest holdout.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F     # noqa: E402
import sim               # noqa: E402

ART = F.ART


def _spec_frames(P):
    raw = P.stores.raw
    gb = raw["gb"]
    starters = raw["gp"][pd.to_numeric(raw["gp"].GS,
                                       errors="coerce") == 1]
    lineups = gb[gb.BattingOrder.notna()].copy()
    lineups["bo"] = pd.to_numeric(lineups.BattingOrder, errors="coerce")
    lineups = lineups[lineups.bo % 100 == 0]
    umps = raw["umps"].set_index("GamePk")["HpUmpId"]
    wx = pd.read_csv(F.DATA / "mlb_weather.csv", encoding="utf-8-sig")
    wx = wx.set_index("GamePk")
    return lineups, starters, umps, wx


def build_spec(P, game_row, lineups, starters, umps, wx):
    """One mlb_games row -> a serve-path spec with what was knowable."""
    pk = game_row.GamePk
    lu = lineups[lineups.GamePk == pk]
    st = starters[starters.GamePk == pk]
    spec = dict(
        date=str(pd.Timestamp(game_row.Date).date()),
        season=int(game_row.Season),
        away_team=game_row.AwayTeam, home_team=game_row.HomeTeam,
        venue=game_row.Venue, day_night=game_row.DayNight,
        temp=game_row.Temp, wind_speed=game_row.WindSpeed,
        wind_dir=game_row.WindDir, condition=game_row.Condition,
        humidity=wx.Humidity.get(pk), pressure=wx.Pressure.get(pk),
        hp_ump_id=umps.get(pk),
    )
    for side, team in (("away", game_row.AwayTeam),
                       ("home", game_row.HomeTeam)):
        rows = lu[lu.Team == team].sort_values("bo")
        spec[f"{side}_lineup"] = [(int(p), i + 1) for i, p in
                                  enumerate(rows.PlayerId.head(9))]
        s = st[st.Team == team]
        spec[f"{side}_starter"] = int(s.PlayerId.iloc[0]) if len(s) \
            else None
    return spec


def _game_actuals(P, pks):
    gb = P.stores.raw["gb"]
    gp = P.stores.raw["gp"]
    games = P.stores.raw["games"]
    g = games[games.GamePk.isin(pks)]
    total = dict(zip(g.GamePk,
                     pd.to_numeric(g.AwayScore, errors="coerce")
                     + pd.to_numeric(g.HomeScore, errors="coerce")))
    st = gp[(gp.GamePk.isin(pks))
            & (pd.to_numeric(gp.GS, errors="coerce") == 1)]
    return total, st


def sample_games(P, season=2024, n=60, seed=3):
    games = P.stores.raw["games"]
    pool = games[games.Season == season]
    return pool.sample(n=min(n, len(pool)), random_state=seed)


def sim_sample(P, sample, latent, n_sims=1000):
    """Sim each sampled game under the given latent params; per game:
    within-game total mean/sd, starter-K means/sds, HR/game, and total
    tail indicators."""
    lineups, starters, umps, wx = _spec_frames(P)
    rows = []
    preps = getattr(P, "_bt_preps", {})
    for _, g in sample.iterrows():
        pk = int(g.GamePk)
        if pk not in preps:
            spec = build_spec(P, g, lineups, starters, umps, wx)
            if len(spec["away_lineup"]) < 9 or spec["away_starter"] is \
                    None or spec["home_starter"] is None:
                continue
            preps[pk] = P.prepare_game(spec, n_sims=n_sims)
        prep, meta = preps[pk]
        prep.latent = latent
        res = sim.run(prep, n_sims=n_sims, seed=pk % 99991,
                      season=int(g.Season))
        t, s = res["tensor"], sim.SIDX
        tot = res["score"].sum(axis=1)
        rows.append(dict(
            GamePk=pk, Date=str(pd.Timestamp(g.Date).date()),
            m_total=float(tot.mean()), v_total=float(tot.var()),
            m_k=float((t[:, 18, s["PK_"]].mean()
                       + t[:, 19, s["PK_"]].mean()) / 2),
            v_k=float((t[:, 18, s["PK_"]].var()
                       + t[:, 19, s["PK_"]].var()) / 2),
            m_hr=float(t[..., s["HR"]].sum(axis=1).mean()),
            v_hr=float(t[..., s["HR"]].sum(axis=1).var()),
            tail_hi=float((tot > 11.5).mean()),
            tail_lo=float((tot <= 5.5).mean()),
        ))
    P._bt_preps = preps
    return pd.DataFrame(rows)


def _composite_sd(m_col, v_col, df):
    """Law-of-total-variance sd comparable to a cross-game historical
    sd: mean within-game variance + variance of within-game means."""
    return float(np.sqrt(df[v_col].mean() + df[m_col].var()))


def _hist_targets(P, sample, boot=200, seed=9):
    """Historical targets with date-block bootstrap CIs."""
    pks = set(sample.GamePk.astype(int))
    games = P.stores.raw["games"]
    g = games[games.GamePk.isin(pks)].copy()
    g["total"] = (pd.to_numeric(g.AwayScore, errors="coerce")
                  + pd.to_numeric(g.HomeScore, errors="coerce"))
    g["date"] = g.Date.astype(str)
    gp = pd.read_csv(F.DATA / "mlb_game_pitching.csv",
                     encoding="utf-8-sig",
                     usecols=["GamePk", "GS", "SO"])
    gp = gp[(gp.GamePk.isin(pks))
            & (pd.to_numeric(gp.GS, errors="coerce") == 1)]
    gp["SO"] = pd.to_numeric(gp.SO, errors="coerce")
    gb = pd.read_csv(F.DATA / "mlb_game_batting.csv",
                     encoding="utf-8-sig",
                     usecols=["GamePk", "HR"])
    gb = gb[gb.GamePk.isin(pks)]
    hr = gb.groupby("GamePk")["HR"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").sum())

    def calc(gg, so, hh):
        return dict(
            total_mean=float(gg.total.mean()),
            total_sd=float(gg.total.std()),
            k_mean=float(so.mean()), k_sd=float(so.std()),
            hr_mean=float(hh.mean()), hr_sd=float(hh.std()),
            tail_hi=float((gg.total > 11.5).mean()),
            tail_lo=float((gg.total <= 5.5).mean()))

    tgt = calc(g, gp.SO, hr)
    rng = np.random.default_rng(seed)
    dates = g.date.unique()
    boots = {k: [] for k in tgt}
    for _ in range(boot):
        pick = rng.choice(dates, size=len(dates), replace=True)
        gg = pd.concat([g[g.date == d] for d in pick])
        so = gp[gp.GamePk.isin(gg.GamePk)].SO
        hh = hr[hr.index.isin(gg.GamePk)]
        for k, v in calc(gg, so, hh).items():
            boots[k].append(v)
    ci = {k: [round(float(np.quantile(v, 0.05)), 3),
              round(float(np.quantile(v, 0.95)), 3)]
          for k, v in boots.items()}
    return tgt, ci


def moment_match_latent(n_games=60, n_sims=1000):
    """Staged coordinate search over the five latent knobs (mu_env,
    sigma_env, sigma_offense, sigma_pitcher, sigma_hr), matching the
    sim's composite (law-of-total-variance) dispersion against
    historical targets with date-block-bootstrap CIs. Teammate and
    opposing-total correlations are reported as diagnostics, not fitted
    — their historical estimates confound matchup heterogeneity with
    the within-game sharing the knobs control."""
    from predict import Predictor
    P = Predictor()
    sample = sample_games(P, n=n_games)
    targets, target_ci = _hist_targets(P, sample)
    print(f"targets: {targets}", flush=True)
    print(f"target CIs (date-block bootstrap): {target_ci}", flush=True)

    best = dict(mu_env=0.0, sigma_env=0.0, sigma_offense=0.0,
                sigma_pitcher=0.0, sigma_hr=0.0)

    def loss(latent):
        df = sim_sample(P, sample, latent, n_sims=n_sims)
        m = dict(total_mean=float(df.m_total.mean()),
                 total_sd=_composite_sd("m_total", "v_total", df),
                 k_mean=float(df.m_k.mean()),
                 k_sd=_composite_sd("m_k", "v_k", df),
                 hr_mean=float(df.m_hr.mean()),
                 hr_sd=_composite_sd("m_hr", "v_hr", df),
                 tail_hi=float(df.tail_hi.mean()),
                 tail_lo=float(df.tail_lo.mean()))
        w = dict(total_mean=1.0, total_sd=1.0, k_mean=0.5, k_sd=0.5,
                 hr_mean=0.5, hr_sd=0.3, tail_hi=2.0, tail_lo=2.0)
        L = sum(w[k] * ((m[k] - targets[k])
                        / max(abs(targets[k]), 1e-3)) ** 2 for k in m)
        print(f"  latent {latent} ->\n    {m}\n    loss {L:.4f}",
              flush=True)
        return L

    for param, grid in (("mu_env", [0.0, 0.05, 0.10, 0.15]),
                        ("sigma_env", [0.0, 0.06, 0.12]),
                        ("sigma_offense", [0.0, 0.08, 0.15]),
                        ("sigma_pitcher", [0.0, 0.10, 0.20]),
                        ("sigma_hr", [0.0, 0.15, 0.30])):
        scores = {}
        for v in grid:
            cand = dict(best)
            cand[param] = v
            scores[v] = loss(cand)
        best[param] = min(scores, key=scores.get)
        print(f"{param} -> {best[param]}", flush=True)

    best["fitted"] = True
    best["n_games"] = int(n_games)
    best["n_sims"] = int(n_sims)
    best["targets"] = targets
    best["target_ci"] = target_ci
    return best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-latent", action="store_true")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=1000)
    args = ap.parse_args()
    if args.fit_latent:
        res = moment_match_latent(args.games, args.sims)
        F.write_artifact(ART / "latent.json",
                         lambda p: p.write_text(json.dumps(res, indent=1)))
        print(f"latent.json written: {res}")
