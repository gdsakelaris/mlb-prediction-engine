"""Batched game preparation: prepare_game's per-game pandas/model
overhead amortized across a chunk of games.

prepare_game spends ~70% of replay wall time on fixed per-call costs:
one assemble_features call, two predict_proba calls, one hazard-grid
predict, two steal-model predicts, per game. This module builds the
SAME GamePrep objects (validated allclose against prepare_game) with
ONE call each across G games: the matchup frame is G x 1080 rows, the
hazard frame G x 2 x 451, the steal frame G x 360.

Only prepare_game's math is reproduced — pen selection, lineups, and
participation state reuse the Predictor's own (now cached) methods, so
there is a single source of truth for roster logic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F     # noqa: E402
import predict as PR     # noqa: E402
import sim               # noqa: E402


def _resolve_game(P, spec, n_sims):
    """The light per-game half of prepare_game: lineups, starters, pen,
    row layout, participation covariates, meta. Mirrors prepare_game
    exactly (same helpers)."""
    date = pd.Timestamp(spec["date"])
    season = int(spec.get("season") or date.year)
    away, home = spec["away_team"], spec["home_team"]

    def side_lineup(side, team):
        posted = sorted(spec.get(f"{side}_lineup") or [],
                        key=lambda x: x[1])
        pids = [int(p) for p, _ in posted][:9]
        if len(pids) < 9:
            for p in P._last_lineup(team):
                if p not in pids:
                    pids.append(p)
                if len(pids) == 9:
                    break
        while len(pids) < 9:
            pids.append(-1)
        return pids

    bat_away = side_lineup("away", away)
    bat_home = side_lineup("home", home)
    sp_away = int(spec.get("away_starter") or -2)
    sp_home = int(spec.get("home_starter") or -3)
    pen_away = P._pen_for(away, date)
    pen_home = P._pen_for(home, date)
    players = (bat_away + bat_home + [sp_away, sp_home]
               + [p for p, _ in pen_away]
               + [-1] * (PR.MAX_PEN - len(pen_away))
               + [p for p, _ in pen_home]
               + [-1] * (PR.MAX_PEN - len(pen_home))
               + [-1, -1])
    n_players = len(players)
    bench_rows = (n_players - 2, n_players - 1)
    pit_rows = [18, 19] + list(range(20, 20 + 2 * PR.MAX_PEN))
    pen_rows_away = list(range(20, 20 + len(pen_away)))
    pen_rows_home = list(range(20 + PR.MAX_PEN,
                               20 + PR.MAX_PEN + len(pen_home)))
    bat_rows_all = list(range(18)) + list(bench_rows)

    pen_order = np.zeros((1, 2, PR.MAX_PEN), dtype=np.int16) - 1
    pen_order[0, 0, :len(pen_rows_away)] = pen_rows_away
    pen_order[0, 1, :len(pen_rows_home)] = pen_rows_home

    side_code = {"L": 0, "R": 1, "S": 2}
    bat_side = np.array(
        [side_code.get(str(P._bats.get(players[i], "R")), 1)
         for i in range(18)], dtype=np.int8)
    pit_throws = np.array(
        [0 if str(P._throws.get(p, "R")) == "L" else 1
         for p in players], dtype=np.int8)
    slot_is_c = np.array(
        [1 if players[i] in P._catchers else 0 for i in range(18)],
        dtype=np.int8)
    meta = dict(players=players,
                names=[P._name(p) if p >= 0 else "League Avg"
                       for p in players],
                season=season, away=away, home=home,
                career_g=[P._career_g.get(p, 0) for p in players],
                career_gp=[P._career_gp.get(p, 0) for p in players])
    return dict(spec=spec, date=date, season=season, away=away,
                home=home, players=players, n_players=n_players,
                bench_rows=bench_rows, pit_rows=pit_rows,
                pen_rows_away=pen_rows_away,
                pen_rows_home=pen_rows_home, bat_rows_all=bat_rows_all,
                pen_order=pen_order, bat_side=bat_side,
                pit_throws=pit_throws, slot_is_c=slot_is_c, meta=meta)


def _matchup_frame(P, resolved):
    """G x 1080 matchup rows in prepare_game's exact nesting order
    (pitcher-major, batter, tto)."""
    blocks = []
    for rv in resolved:
        spec, date, season = rv["spec"], rv["date"], rv["season"]
        away, home = rv["away"], rv["home"]
        players = rv["players"]
        rest_by_pid = {p: P._rest(p, date)
                       for p in {players[r] for r in rv["pit_rows"]}}
        rows = []
        for prow in rv["pit_rows"]:
            ppid = players[prow]
            pthrows = str(P._throws.get(ppid, "R"))
            pthrows = pthrows if pthrows in ("L", "R") else "R"
            p_team = away if (prow == 18
                              or prow in rv["pen_rows_away"]) else home
            for brow in rv["bat_rows_all"]:
                bpid = players[brow]
                bat_is_home = (brow >= 9) if brow < 18 else \
                    (brow == rv["bench_rows"][1])
                stand = P._stand(bpid, pthrows)
                for tto in (1, 2, 3):
                    rows.append((date, season, bpid, ppid, stand,
                                 pthrows, tto, int(bat_is_home), p_team,
                                 spec.get("venue") or "",
                                 spec.get("day_night") or "day",
                                 spec.get("temp"),
                                 spec.get("wind_speed"),
                                 spec.get("wind_dir") or "",
                                 spec.get("condition") or "",
                                 spec.get("humidity"),
                                 spec.get("pressure"),
                                 spec.get("hp_ump_id"),
                                 rest_by_pid[ppid]))
        blocks.append(rows)
    cols = ["Date", "Season", "BatterId", "PitcherId", "stand",
            "p_throws", "tto", "home_bat", "fld_team", "Venue",
            "DayNight", "Temp", "WindSpeed", "WindDir", "Condition",
            "Humidity", "Pressure", "HpUmpId", "rest_p"]
    rdf = pd.DataFrame([r for b in blocks for r in b], columns=cols)
    rdf["Date"] = pd.to_datetime(rdf["Date"])
    return rdf


def _fill_avec(P, resolved, rdf, n_sims):
    """One assemble + one predict per model for the whole chunk, then
    scatter back into per-game avec/a2vec."""
    X, _ = F.assemble_features(rdf, P.fstores)
    X = X.reindex(columns=P.a1["features"]).astype(np.float32)
    p1 = P.a1["scaler"].transform(P.a1["model"].predict_proba(X))
    p2 = P.a2["scaler"].transform(P.a2["model"].predict_proba(X))
    per_game = 18 * 20 * 3
    out = []
    for gi, rv in enumerate(resolved):
        n_players = rv["n_players"]
        avec = np.full((n_players, 20, 3, 8), np.nan)
        a2vec = np.full((n_players, 20, 3, 4), np.nan)
        b1 = p1[gi * per_game:(gi + 1) * per_game].reshape(18, 20, 3, 8)
        b2 = p2[gi * per_game:(gi + 1) * per_game].reshape(18, 20, 3, 4)
        for j, prow in enumerate(rv["pit_rows"]):
            avec[prow] = b1[j]
            a2vec[prow] = b2[j]
        out.append((avec, a2vec))
    return out


def _start_hist(P):
    """Per-pitcher starter-only history + IL activations, cached."""
    cache = getattr(P, "_start_hist_cache", None)
    if cache is not None:
        return cache
    gp = P.stores.raw["gp"]
    st = gp[pd.to_numeric(gp.GS, errors="coerce") == 1].copy()
    ip = pd.to_numeric(st["IP"], errors="coerce").fillna(0)
    st["_outs"] = (ip.astype(int) * 3 + round((ip % 1) * 10))
    st["_np"] = pd.to_numeric(st["NP"], errors="coerce")
    st = st.sort_values("Date")
    hist = {int(p): (g["Date"].values, g["_np"].values,
                     g["_outs"].values)
            for p, g in st.groupby("PlayerId")}
    il = P.fstores.get("il_stints")
    il_hist = {}
    if il is not None:
        ils = il.sort_values("Date")
        il_hist = {int(p): g["act_date"].values
                   for p, g in ils.groupby("PlayerId")}
    cache = (hist, il_hist)
    P._start_hist_cache = cache
    return cache


def _hazard_grids(P, resolved):
    """Batched serve-side hazard grids: same fields as
    Predictor._hazard_grid, one model call for all starters."""
    leash = P.fstores.get("panel_leash")
    hist, il_hist = _start_hist(P)
    B_, R_ = np.meshgrid(np.arange(41), np.arange(11), indexing="ij")
    base_bf = B_.ravel()
    base_runs = R_.ravel()
    frames = []
    for rv in resolved:
        d = rv["date"]
        d64 = np.datetime64(d)
        for si, prow in enumerate((18, 19)):
            ppid = rv["players"][prow]
            lz = None
            if leash is not None:
                mine = leash[(leash.PlayerId == ppid)
                             & (leash.Date < d)]
                if len(mine):
                    lz = mine.sort_values("Date").iloc[-1]
            starts = float(lz["starts_d"]) if lz is not None else np.nan
            np_avg = (float(lz["np_sum_d"]) / max(starts, 1e-9)
                      if lz is not None else np.nan)
            bf_avg = (float(lz["bf_sum_d"]) / max(starts, 1e-9)
                      if lz is not None else np.nan)
            ppb = (np_avg / bf_avg) if np_avg and bf_avg and bf_avg > 0 \
                else 3.9
            outs_sd = np.nan
            if (lz is not None and "outs2_sum_d" in lz.index
                    and starts == starts and starts >= 5):
                mu_o = float(lz["outs_sum_d"]) / max(starts, 1e-9)
                var_o = (float(lz["outs2_sum_d"]) / max(starts, 1e-9)
                         - mu_o ** 2)
                outs_sd = float(np.sqrt(max(var_o, 0.0)))
            gap_days, ramp, prev_short = np.nan, 0.0, 0.0
            h = hist.get(int(ppid)) if ppid >= 0 else None
            if h is not None:
                i = int(np.searchsorted(h[0], d64, side="left"))
                if i > 0:
                    gap_days = float((d - pd.Timestamp(h[0][i - 1]))
                                     .days)
                    np_last = h[1][i - 1]
                    if np_last == np_last:
                        ramp = float(np_last < F.RAMP_NP)
                    outs_last = h[2][i - 1]
                    prev_short = float(outs_last
                                       <= F.SHORT_START_OUTS)
            il_ret30 = 0.0
            ih = il_hist.get(int(ppid)) if ppid >= 0 else None
            if ih is not None:
                i = int(np.searchsorted(ih, d64, side="right"))
                if i > 0:
                    il_ret30 = float(
                        (d - pd.Timestamp(ih[i - 1])).days <= 30)
            frames.append(pd.DataFrame(dict(
                bf=base_bf, cum_pitches=base_bf * ppb,
                tto=np.clip(1 + base_bf // 9, 1, 4),
                inning=1 + base_bf / 4.3, outs=1, score_diff=0,
                k_so_far=base_bf * 0.22, br_so_far=base_bf * 0.30,
                runs_so_far=base_runs,
                rest_p=P._rest(ppid, d),
                leash_np=np_avg, leash_bf=bf_avg, leash_starts=starts,
                season_idx=rv["season"] - 2015,
                gap_days=gap_days, ramp=ramp, prev_short=prev_short,
                il_ret30=il_ret30, outs_sd=outs_sd)))
    rows = pd.concat(frames, ignore_index=True)
    Xh = rows[P.hz["features"]].astype(np.float32)
    p = P.hz["iso"].predict(P.hz["model"].predict_proba(Xh)[:, 1])
    p = p.reshape(len(resolved), 2, 41, 11)
    return [p[i] for i in range(len(resolved))]


def _sb_matrices(P, resolved):
    """Batched steal matrices: one predict per model for all games'
    (runner, pitcher) pairs, then scatter into [n_players, n_players]
    per game — same order and values as Predictor._sb_matrices."""
    sprint = P.fstores["sprint"]
    deft = P.fstores["defense_team"]
    rows_a, rows_s, sizes = [], [], []
    for rv in resolved:
        season = rv["season"]
        away, home = rv["away"], rv["home"]
        players = rv["players"]
        spd = dict(zip(
            sprint.loc[sprint.Year == season, "PlayerId"],
            sprint.loc[sprint.Year == season, "SprintSpeed"]))
        if season not in getattr(P, "_sb_rate_cache", {}):
            P._sb_matrices(players, rv["pit_rows"][:1],
                           rv["bat_rows_all"][:1], rv["date"], season,
                           away, home)   # warms the season rate cache
        sbr, csr = P._sb_rate_cache[season]
        drow = {t: deft[(deft.Team == t) & (deft.Year == season)]
                for t in (away, home)}

        def team_pop(t):
            r = drow[t]
            return (float(r.PopTime.iloc[0]) if len(r)
                    and pd.notna(r.PopTime.iloc[0]) else np.nan)

        era_new = float(season >= 2023)
        n = 0
        for brow in rv["bat_rows_all"]:
            bpid = players[brow]
            for prow in rv["pit_rows"]:
                ppid = players[prow]
                fld = away if (prow == 18
                               or 20 <= prow < 20 + PR.MAX_PEN) else \
                    home
                lhp = float(str(P._throws.get(ppid, "R")) == "L")
                rows_a.append((
                    spd.get(bpid, np.nan) if bpid >= 0 else np.nan,
                    sbr.get(ppid, np.nan), csr.get(ppid, np.nan),
                    team_pop(fld), np.nan, 1, 1.0, era_new, lhp))
                rows_s.append((
                    spd.get(bpid, np.nan), csr.get(ppid, np.nan),
                    team_pop(fld), np.nan, lhp, era_new))
                n += 1
        sizes.append(n)
    A = pd.DataFrame(rows_a, columns=["SprintSpeed", "sb_allowed_rate",
                                      "cs_rate", "PopTime", "CSAA",
                                      "outs", "score_close", "era_new",
                                      "lhp"])[P.sb["att_features"]]
    S_ = pd.DataFrame(rows_s, columns=["SprintSpeed", "cs_rate",
                                       "PopTime", "CSAA", "lhp",
                                       "era_new"])[P.sb["suc_features"]]
    pa_all = P.sb["attempt"].predict_proba(A)[:, 1]
    ps_raw = P.sb["success"].predict_proba(S_)[:, 1]
    out = []
    off = 0
    for rv, n in zip(resolved, sizes):
        season = rv["season"]
        era = "post2023" if season >= 2023 else "pre2023"
        scale = P.sb["scale"][era]["attempt_scale"]
        shift = P.sb["success_logit_shift"][era]
        pa_ = pa_all[off:off + n] * scale
        praw = ps_raw[off:off + n]
        logit = np.log(np.clip(praw, 1e-6, 1 - 1e-6)
                       / np.clip(1 - praw, 1e-6, 1))
        ps_ = 1 / (1 + np.exp(-(logit + shift)))
        n_players = rv["n_players"]
        att = np.zeros((n_players, n_players))
        suc = np.zeros((n_players, n_players))
        ix = np.ix_(rv["bat_rows_all"], rv["pit_rows"])
        att[ix] = pa_.reshape(20, 18)
        suc[ix] = ps_.reshape(20, 18)
        out.append((att, suc))
        off += n
    return out


def prepare_games(P, specs, n_sims=4000):
    """Batched prepare_game: returns [(GamePrep, meta), ...] matching
    Predictor.prepare_game output for each spec."""
    resolved = [_resolve_game(P, s, n_sims) for s in specs]
    rdf = _matchup_frame(P, resolved)
    vecs = _fill_avec(P, resolved, rdf, n_sims)
    grids = _hazard_grids(P, resolved)
    sbs = _sb_matrices(P, resolved)
    out = []
    for rv, (avec, a2vec), haz, (att, suc) in zip(resolved, vecs,
                                                  grids, sbs):
        pen_order = np.broadcast_to(rv["pen_order"],
                                    (n_sims, 2, PR.MAX_PEN))
        lat = dict(P.latent)
        prep = sim.GamePrep(
            n_players=rv["n_players"], starters=[18, 19], avec=avec,
            a2vec=a2vec, haz_grid=haz, relief_exit=P._relief_exit,
            pen_order=pen_order, sb_att=att, sb_suc=suc,
            patterns=P.patterns, latent=lat,
            bench_rows=rv["bench_rows"], part_haz=P.part_haz,
            bat_side=rv["bat_side"], pit_throws=rv["pit_throws"],
            slot_is_c=rv["slot_is_c"],
            pre_wp=P.preevents["wp_pb_per_pa_runners_on"],
            pre_pk=P.preevents["pickoff_out_per_pa_runners_on"])
        out.append((prep, rv["meta"]))
    return out
