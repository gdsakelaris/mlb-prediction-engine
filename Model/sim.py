"""Vectorized Monte Carlo game simulator.

Plays N independent copies of one game simultaneously: every state
variable is an array over the sim axis, and each loop iteration advances
every unfinished sim by one plate appearance. All model evaluation
happens BEFORE the loop (predict.py precomputes per-matchup outcome
probabilities, hazard grids and steal probabilities), so the loop is pure
numpy gather/scatter — no model calls, no Python per-sim work.

Per PA, in order:
  1. pickoff / wild-pitch-passed-ball pre-events (league rates)
  2. steal-of-second window (per runner-vs-battery probabilities)
  3. starter-removal hazard (grid over batters faced x runs allowed);
     relievers change only at inning breaks via the stint-exit table
  4. outcome sample from the A1 8-class vector for (batter, pitcher,
     times-through-order), with the per-sim latent multipliers applied
  5. batted-ball type from A2 when in play
  6. advancement pattern sample keyed (class, bb-type, base state, outs):
     the empirical pattern moves runners, adds outs, scores runs, and
     carries RBI + earned-run counts measured from real play-by-play
  7. legality: walk-off truncation (non-HR capped at the winning run),
     home team skips the bottom 9th when leading, extra-innings ghost
     runner, inning/game accounting

Outputs a per-sim player-stat tensor plus game scores, first-5-inning
and first-inning runs — every market is a counting query over these.

Rules are per-season (replay needs old rules): ghost runner 2020+,
7-inning doubleheaders 2020-21.
"""

from pathlib import Path

import numpy as np

BBTYPES = ["ground_ball", "fly_ball", "line_drive", "popup"]
BB_CODE = {"": 0, **{b: i + 1 for i, b in enumerate(BBTYPES)}}


def pattern_key(cls, bb, state, outs):
    """Canonical key over (class idx, bb code 0-4, state 0-7, outs 0-2)."""
    return ((cls * 5 + bb) * 8 + state) * 3 + outs


# identity-tilt buckets: pattern sampling is biased by a standardized
# z (lead-runner speed/value net of OF arm on hits; batter GIDP
# propensity on ground-ball outs). z is quantized to 5 buckets and the
# tilted cumulative banks are precomputed per bucket, so runtime cost is
# one extra gather index. Bucket 2 (z=0) reproduces the league bank.
TILT_MUS = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
TILT_EDGES = np.array([-1.125, -0.375, 0.375, 1.125])
NB = len(TILT_MUS)


def _pattern_tilt(cls, bb, state, outs, arr, p, tilt):
    """Per-pattern tilt coefficient c for one key: exp(c * z) reweights
    pattern probabilities. c is centered (p-weighted) within the key so
    the neutral bucket keeps the league marginal exactly."""
    c = np.zeros(len(arr), dtype=float)
    if tilt is None:
        return c
    if cls in (3, 4, 5):                       # hits: advancement tilt
        station = {3: 1, 4: 2, 5: 3}[cls]
        adv = np.zeros(len(arr), dtype=float)
        bd = arr[:, 0].astype(int)
        adv += np.where((bd >= 1) & (bd <= 4), bd - station, 0
                        ).clip(min=0)
        for slot, base in ((1, 1), (2, 2), (3, 3)):
            if not (state & (1 << (base - 1))):
                continue
            d = arr[:, slot].astype(int)
            adv += np.where((d >= 1) & (d <= 4), d - base, 0)
        c = tilt.get("theta_adv", 0.0) * (adv - float(np.dot(p, adv)))
    elif cls == 7 and bb == 1 and (state & 1) and outs < 2:
        isdp = (arr[:, 4] >= 2).astype(float)  # GB out w/ r1: DP tilt
        c = tilt.get("theta_dp", 0.0) * (isdp - float(np.dot(p, isdp)))
    return c


def load_patterns(stores_dir=None):
    """pattern_table.parquet -> {key: (cum[NB, n], patterns[n, 8])}:
    per-bucket tilted cumulative probabilities (see TILT_MUS) with the
    same key encoding the sim uses. Tilt thetas come from
    advance_tilt.json when present; absent -> all buckets identical."""
    import json
    import pandas as pd
    stores_dir = Path(stores_dir) if stores_dir else \
        Path(__file__).resolve().parent / "artifacts" / "stores"
    tilt_path = stores_dir / "advance_tilt.json"
    tilt = json.loads(tilt_path.read_text()) if tilt_path.exists() \
        else None
    tab = pd.read_parquet(stores_dir / "pattern_table.parquet")
    cls_idx = {c: i for i, c in enumerate(
        ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "IPO"])}
    tab["_cls"] = tab["label"].map(cls_idx).astype(int)
    tab["_bb"] = tab["bb_type"].fillna("").map(BB_CODE).astype(int)
    tab["_k"] = pattern_key(tab["_cls"], tab["_bb"],
                            tab["state"].astype(int),
                            tab["outs"].astype(int))
    out = {}
    cols = ["b", "r1", "r2", "r3", "outs_added", "runs", "rbi", "earned"]
    for k, g in tab.groupby("_k"):
        arr = g[cols].to_numpy(dtype=np.int8)
        p = g["p"].to_numpy(dtype=float)
        p = p / p.sum()
        c = _pattern_tilt(int(g["_cls"].iat[0]), int(g["_bb"].iat[0]),
                          int(g["state"].iat[0]), int(g["outs"].iat[0]),
                          arr, p, tilt)
        w = p[None, :] * np.exp(np.outer(TILT_MUS, c))
        w /= w.sum(axis=1, keepdims=True)
        out[int(k)] = (np.cumsum(w, axis=1), arr)
    return out

# stat tensor columns (batters use the first block, pitchers the second;
# every player row uses the same width)
BAT_STATS = ["PA", "H", "B1", "B2", "B3", "HR", "BB", "HBP", "K", "R",
             "RBI", "SB", "CS"]
PIT_STATS = ["BF", "OUTS", "PK_", "PBB", "PH", "PHR", "PR", "PER"]
STATS = BAT_STATS + PIT_STATS
SIDX = {s: i for i, s in enumerate(STATS)}

CLASSES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "IPO"]
NCLS = len(CLASSES)
CI = {c: i for i, c in enumerate(CLASSES)}
ONBASE = np.array([CI[c] for c in ("BB", "HBP", "1B", "2B", "3B", "HR")])
HITS = {CI["1B"]: "B1", CI["2B"]: "B2", CI["3B"]: "B3", CI["HR"]: "HR"}

MAX_INNINGS = 19
PLATOON_WIN = 3          # pen entry: a same-hand arm within the first
                         # N slots of the leverage order jumps the queue
                         # (3-batter-minimum era: matching is real but
                         # managers don't reach past the top of the pen)


def rules_for(season, is_dh_game=False):
    return dict(
        ghost_runner=season >= 2020,
        regulation=7 if (is_dh_game and season in (2020, 2021)) else 9,
    )


class GamePrep:
    """Pure-array inputs for one game (built by predict.prepare_game).

    n_players rows: away batters 0-8, home batters 9-17, then pitchers.
    avec / a2vec:   [n_pitchers, 18, 3(tto)] -> 8 / 4 class probs
    haz_grid:       [2, 41, 11] starter removal prob by (bf, runs)
    relief_exit:    P(stint ends at inning break | outs in stint):
                    league [11] or per-pitcher [n_players, 11] (B4)
    pen_rank_cum:   optional [2, K] cumulative pmfs (hi, lo) over the
                    entry RANK among still-available pen arms (B6);
                    absent -> deterministic first-arm + platoon jump
    pen_order:      per sim, per team, the reliever entry order
                    [n_sims, 2, max_pen] as pitcher indices (-1 = none)
    sb_att/sb_suc:  [18, n_pitchers] steal-of-2B attempt/success prob for
                    the lineup player as runner vs that pitcher's battery
                    at the (outs=1, close-game) baseline
    sb_state:       [4] = (outs0, outs2, sc_far logit deltas, era
                    attempt scale) — live outs/score conditioning of
                    sb_att at sim time; None = frozen baseline (legacy)
    pattern bank:   flat arrays keyed by (class, bb, state, outs)
    latent sigmas:  dict(sigma_env, sigma_pitcher, sigma_hr, sigma_k)
                    sigma_k is a per-game per-STARTER strikeout-form
                    draw: multiplies the K class while the starter is in,
                    renormalization absorbs the mass (starter-K
                    dispersion knob; relievers unaffected)
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _sample_rows(P, u):
    """P: [n, k] probability rows; u: [n] uniforms -> class index."""
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0 + 1e-9
    return (u[:, None] > cum).sum(axis=1)


def bat_axis(rows):
    """Player row -> batter axis index for avec/a2vec/sb lookups: lineup
    rows 0-17 map to themselves, the two bench rows map to 18/19."""
    r = np.asarray(rows, dtype=np.int16)
    return r - 18 * (r >= 36)


def run(prep, n_sims=20000, seed=1, season=2026, is_dh_game=False):
    rng = np.random.default_rng(seed)
    rules = rules_for(season, is_dh_game)
    REG = rules["regulation"]
    S = n_sims

    n_players = prep.n_players
    tensor = np.zeros((S, n_players, len(STATS)), dtype=np.int16)
    score = np.zeros((S, 2), dtype=np.int16)          # away, home
    runs_f5 = np.zeros((S, 2), dtype=np.int16)
    runs_i1 = np.zeros((S, 2), dtype=np.int16)

    inning = np.ones(S, dtype=np.int8)
    half = np.zeros(S, dtype=np.int8)                 # 0 top (away bats)
    outs = np.zeros(S, dtype=np.int8)
    # bases hold the PLAYER ROW of the runner (0-17 lineup, 36/37 bench,
    # -1 empty) — identity-safe under substitutions; resp holds the
    # pitcher row charged with that runner (earned-run attribution)
    bases = -np.ones((S, 3), dtype=np.int16)
    resp = -np.ones((S, 3), dtype=np.int16)
    bat_ptr = np.zeros((S, 2), dtype=np.int8)
    cur_pit = np.tile(np.array(prep.starters, dtype=np.int16), (S, 1))
    pit_bf = np.zeros((S, 2), dtype=np.int16)
    pit_runs = np.zeros((S, 2), dtype=np.int16)
    starter_in = np.ones((S, 2), dtype=bool)
    pen_next = np.zeros((S, 2), dtype=np.int8)
    stint_outs = np.zeros((S, 2), dtype=np.int8)
    done = np.zeros(S, dtype=bool)

    # batter participation state: is each lineup slot's STARTER still in,
    # and how many PAs has the slot taken (the hazard's k index)
    active = np.ones((S, 18), dtype=bool)
    kpa = np.zeros((S, 18), dtype=np.int8)
    bench = np.asarray(prep.bench_rows, dtype=np.int16)  # [away, home]
    part = getattr(prep, "part_haz", None)
    bat_side = np.asarray(getattr(prep, "bat_side", np.ones(18)),
                          dtype=np.int8)                 # 0 L, 1 R, 2 S
    pit_throws = np.asarray(getattr(prep, "pit_throws",
                                    np.ones(n_players)), dtype=np.int8)
    slot_is_c = np.asarray(getattr(prep, "slot_is_c", np.zeros(18)),
                           dtype=np.int8)

    # per-sim latent draws (gap #2): shared game conditions. mu_env is
    # the run-environment mean offset the moment match fits alongside
    # the dispersion sigmas.
    sig = prep.latent
    z_env = rng.normal(sig.get("mu_env", 0.0),
                       sig.get("sigma_env", 0.0), S)
    z_hr = rng.normal(0, sig.get("sigma_hr", 0.0), S)
    z_pit = rng.normal(0, sig.get("sigma_pitcher", 0.0), (S, 2))
    z_off = rng.normal(0, sig.get("sigma_offense", 0.0), (S, 2))
    z_k = rng.normal(0, sig.get("sigma_k", 0.0), (S, 2))

    # identity adjusters (older preps lack them -> neutral defaults)
    zp32 = np.zeros(n_players, dtype=np.float32)
    run_z = np.asarray(getattr(prep, "run_z", zp32), dtype=np.float32)
    dp_z = np.asarray(getattr(prep, "dp_z", zp32), dtype=np.float32)
    arm_eff = np.asarray(getattr(prep, "arm_eff", np.zeros(2)),
                         dtype=np.float32)
    stretch_z = np.asarray(getattr(prep, "stretch_z", zp32),
                           dtype=np.float32)
    stre = getattr(prep, "stretch", None)
    pre_wp2 = np.asarray(prep.pre_wp, dtype=float).ravel()
    if pre_wp2.size == 1:
        pre_wp2 = np.repeat(pre_wp2, 2)

    # leverage-aware pen: two entry orders (player rows) chosen by game
    # state at entry, with a used-arm mask; absent -> legacy fixed order
    pen_hi = getattr(prep, "pen_hi", None)
    pen_lo = getattr(prep, "pen_lo", None)
    pen_used = None
    if pen_hi is not None and pen_lo is not None:
        pen_hi = np.asarray(pen_hi, dtype=np.int16)
        pen_lo = np.asarray(pen_lo, dtype=np.int16)
        pen_used = np.zeros((S, 2, pen_hi.shape[1]), dtype=bool)

    rank_cum = getattr(prep, "pen_rank_cum", None)
    if rank_cum is not None:
        rank_cum = np.asarray(rank_cum, dtype=np.float64)

    def _pick_pen(idx, side, hi_mask, due_stand=None):
        """Arm from the leverage-matched order; -1 = none. With a
        pen_rank_cum store (B6) the entry RANK among still-available
        arms is sampled per sim from the empirical manager-choice pmf
        (which subsumes platoon behavior); without it, first unused arm
        with the deterministic PLATOON_WIN same-hand jump."""
        order = np.where(hi_mask[:, None], pen_hi[side], pen_lo[side])
        npen = order.shape[1]
        slots = np.clip(order - 20 - 8 * side[:, None], 0, npen - 1)
        ok = (order >= 0) & ~pen_used[idx[:, None], side[:, None], slots]
        anyok = ok.any(axis=1)
        if rank_cum is not None:
            cum = rank_cum[np.where(hi_mask, 0, 1)]         # [n, K]
            cnt = ok.sum(axis=1)
            k = np.clip(np.minimum(cnt, cum.shape[1]) - 1, 0, None)
            tot = cum[np.arange(idx.size), k]
            u = rng.random(idx.size) * tot
            r = np.minimum((u[:, None] > cum).sum(axis=1),
                           np.maximum(cnt - 1, 0))
            pos = np.minimum((np.cumsum(ok, axis=1) <= r[:, None])
                             .sum(axis=1), npen - 1)
        else:
            pos = ok.argmax(axis=1)
            if due_stand is not None:
                hands = pit_throws[np.clip(order, 0, None)]
                match = (ok & (hands == due_stand[:, None])
                         & (due_stand[:, None] != 2))
                match[:, PLATOON_WIN:] = False
                m_any = match.any(axis=1)
                pos = np.where(m_any, match.argmax(axis=1), pos)
        chosen = np.where(anyok, order[np.arange(idx.size), pos],
                          -1).astype(np.int16)
        cslot = np.clip(chosen - 20 - 8 * side, 0, npen - 1)
        pen_used[idx[anyok], side[anyok], cslot[anyok]] = True
        return chosen

    pen_pick = _pick_pen if pen_used is not None else None

    pat = prep.patterns          # dict: key -> (cum[NB, n], arr[n, 8])
    empty_key_fallback = {}

    def occ_row(sims, bt_arr, slot_arr):
        """The row batting for a slot: the starter while active, else
        the batting team's bench bat."""
        idx18 = slot_arr.astype(np.int16) + 9 * bt_arr
        act = active[sims, idx18]
        return np.where(act, idx18, bench[bt_arr]), idx18

    for _step in range(400):
        act_mask = ~done
        if not act_mask.any():
            break
        bat_t = half.astype(np.int16)
        fld_t = 1 - bat_t
        a = np.flatnonzero(act_mask)
        bt, ft = bat_t[a], fld_t[a]
        slot = bat_ptr[a, bt]

        # ---- 1. pre-events: pickoff, then WP/PB advance
        r_on = (bases[a] >= 0)
        any_on = r_on.any(axis=1)
        u = rng.random(a.size)
        pk = any_on & (u < prep.pre_pk) & (bases[a, 0] >= 0)
        if pk.any():
            idx = a[pk]
            runner_row = bases[idx, 0]
            np.add.at(tensor, (idx, runner_row, SIDX["CS"]), 1)
            idx_fld = (1 - half[idx]).astype(np.int16)
            np.add.at(tensor, (idx, cur_pit[idx, idx_fld],
                               SIDX["OUTS"]), 1)
            outs[idx] += 1
            bases[idx, 0] = -1
            resp[idx, 0] = -1
        u = rng.random(a.size)
        wp = any_on & (u < pre_wp2[ft]) & (outs[a] < 3)
        if wp.any():
            idx = a[wp]
            for b in (2, 1, 0):                      # lead runners first
                mv = bases[idx, b] >= 0
                if not mv.any():
                    continue
                src = idx[mv]
                if b == 2:
                    runner_row = bases[src, 2]
                    np.add.at(tensor, (src, runner_row, SIDX["R"]), 1)
                    np.add.at(score, (src, bat_t[src]), 1)
                    f5m = inning[src] <= 5
                    np.add.at(runs_f5, (src[f5m], bat_t[src][f5m]), 1)
                    i1m = inning[src] == 1
                    np.add.at(runs_i1, (src[i1m], bat_t[src][i1m]), 1)
                    pr = resp[src, 2]
                    ok = pr >= 0
                    np.add.at(tensor, (src[ok], pr[ok], SIDX["PR"]), 1)
                    np.add.at(tensor, (src[ok], pr[ok], SIDX["PER"]), 1)
                    src_fld = (1 - half[src]).astype(np.int16)
                    np.add.at(pit_runs, (src, src_fld), 1)
                    bases[src, 2] = -1
                    resp[src, 2] = -1
                else:
                    open_ = bases[src, b + 1] < 0
                    mvs = src[open_]
                    bases[mvs, b + 1] = bases[mvs, b]
                    resp[mvs, b + 1] = resp[mvs, b]
                    bases[mvs, b] = -1
                    resp[mvs, b] = -1
            # a go-ahead run scoring on a wild pitch in the bottom of
            # the 9th or later is a walk-off
            wo_wp = idx[(half[idx] == 1) & (inning[idx] >= REG)
                        & (score[idx, 1] > score[idx, 0])]
            done[wo_wp] = True
            act_mask = ~done
            a = np.flatnonzero(act_mask)
            if a.size == 0:
                break
            bat_t = half.astype(np.int16)
            fld_t = 1 - bat_t
            bt, ft = bat_t[a], fld_t[a]
            slot = bat_ptr[a, bt]

        # ---- 2. steal of second (runner on 1B, 2B open, inning live)
        can = (bases[a, 0] >= 0) & (bases[a, 1] < 0) & (outs[a] < 3)
        if can.any():
            idx = a[can]
            r_row = bases[idx, 0]
            p_att = prep.sb_att[r_row, cur_pit[idx, fld_t[idx]]]
            st = getattr(prep, "sb_state", None)
            if st is not None:
                # recover the model logit (matrix stores scale*sigmoid at
                # outs=1/close), shift by the live state, re-scale
                base = np.clip(p_att / st[3], 1e-6, 1 - 1e-6)
                lg = np.log(base / (1 - base))
                o = outs[idx]
                lg = (lg + np.where(o == 0, st[0], 0.0)
                      + np.where(o == 2, st[1], 0.0))
                far = np.abs(score[idx, 0].astype(np.int32)
                             - score[idx, 1]) > 2
                lg = lg + np.where(far, st[2], 0.0)
                p_att = np.clip(st[3] / (1.0 + np.exp(-lg)), 0.0, 0.95)
            go = rng.random(idx.size) < p_att
            if go.any():
                gidx = idx[go]
                g_row = r_row[go]
                p_suc = prep.sb_suc[g_row, cur_pit[gidx, fld_t[gidx]]]
                win = rng.random(gidx.size) < p_suc
                w, l = gidx[win], gidx[~win]
                np.add.at(tensor, (w, g_row[win], SIDX["SB"]), 1)
                bases[w, 1] = bases[w, 0]
                resp[w, 1] = resp[w, 0]
                bases[w, 0] = -1
                resp[w, 0] = -1
                np.add.at(tensor, (l, g_row[~win], SIDX["CS"]), 1)
                l_fld = (1 - half[l]).astype(np.int16)
                np.add.at(tensor, (l, cur_pit[l, l_fld], SIDX["OUTS"]), 1)
                outs[l] += 1
                bases[l, 0] = -1
                resp[l, 0] = -1

        # an inning that ended on CS/PK settles NOW; every other sim
        # still takes its PA this iteration
        self_ended = np.zeros(S, dtype=bool)
        self_ended[a] = outs[a] >= 3
        if self_ended.any():
            _end_half(self_ended, inning, half, outs, bases, resp, score,
                      runs_f5, runs_i1, bat_ptr, done, stint_outs,
                      cur_pit, starter_in, pen_next, prep, rng, tensor,
                      REG, rules, active, bench, pen_pick)
            keep = ~self_ended[a] & ~done[a]
            a = a[keep]
            if a.size == 0:
                continue
            bt, ft = bat_t[a], fld_t[a]
            slot = bat_ptr[a, bt]

        # ---- 3. starter removal hazard (before this PA)
        chk = starter_in[a, ft] & (pit_bf[a, ft] > 0)
        if chk.any():
            idx = a[chk]
            ftc = ft[chk]
            bf = np.clip(pit_bf[idx, ftc], 0, 40)
            rn = np.clip(pit_runs[idx, ftc], 0, 10)
            hz = prep.haz_grid[ftc, bf, rn]
            out_now = rng.random(idx.size) < hz
            if out_now.any():
                oidx = idx[out_now]
                oft = ftc[out_now]
                starter_in[oidx, oft] = False
                if pen_pick is not None:
                    hi = ((inning[oidx] >= 7)
                          & (np.abs(score[oidx, 0].astype(np.int32)
                                    - score[oidx, 1]) <= 2))
                    s18d = (slot[chk][out_now].astype(np.int16)
                            + 9 * bt[chk][out_now])
                    due = np.where(active[oidx, s18d], bat_side[s18d],
                                   2).astype(np.int8)
                    nxt = pen_pick(oidx, oft, hi, due)
                else:
                    nxt = prep.pen_order[oidx, oft, pen_next[oidx, oft]]
                    pen_next[oidx, oft] = np.minimum(
                        pen_next[oidx, oft] + 1,
                        prep.pen_order.shape[2] - 1)
                has = nxt >= 0
                cur_pit[oidx[has], oft[has]] = nxt[has]
                stint_outs[oidx, oft] = 0
        pit = cur_pit[a, ft]

        # ---- 3b. batter participation hazard for the due slot: the
        # starter may be lifted (pinch hitter, blowout, injury); once
        # gone the slot is a generic bench bat for the rest of the game
        idx18 = slot.astype(np.int16) + 9 * bt
        if part is not None:
            live = active[a, idx18]
            if live.any():
                li = np.flatnonzero(live)
                sims = a[li]
                s18 = idx18[li]
                k = np.clip(kpa[sims, s18] + 1, 1, 6) - 1
                inn = inning[sims]
                inn_b = np.select([inn <= 6, inn == 7, inn == 8],
                                  [0, 1, 2], default=3)
                diff = (score[sims, bt[li]]
                        - score[sims, 1 - bt[li]]).astype(np.int16)
                margin_b = np.select([np.abs(diff) <= 1,
                                      np.abs(diff) <= 3], [0, 1],
                                     default=2)
                lead = (diff > 0).astype(np.int8)
                side = bat_side[s18]
                same = ((side == pit_throws[cur_pit[sims, ft[li]]])
                        & (side != 2)).astype(np.int8)
                isc = slot_is_c[s18]
                rate = part[k, inn_b, margin_b, lead, same, isc]
                gone = rng.random(sims.size) < rate
                if gone.any():
                    active[sims[gone], s18[gone]] = False
        batter_row, _ = occ_row(a, bt, slot)
        kpa[a, idx18] = np.minimum(kpa[a, idx18] + 1, 7)

        # ---- 4. outcome sample with latent multipliers
        tto = np.minimum(pit_bf[a, ft] // 9, 2)
        P = prep.avec[pit, bat_axis(batter_row), tto].copy()
        mult = np.ones_like(P)
        mult[:, ONBASE] *= np.exp(z_env[a])[:, None]
        mult[:, ONBASE] *= np.exp(z_off[a, bt])[:, None]
        mult[:, CI["HR"]] *= np.exp(z_hr[a])
        zp = z_pit[a, ft]
        mult[:, CI["K"]] *= np.exp(zp)
        mult[:, ONBASE] *= np.exp(-0.5 * zp)[:, None]
        # starter-only K-form draw; renormalization compensates
        mult[:, CI["K"]] *= np.exp(z_k[a, ft] * starter_in[a, ft])
        # base-state (stretch) conditioning: league K/BB offsets with
        # runners on vs empty + per-pitcher stretch-delta deviation
        if stre is not None:
            on_now = (bases[a] >= 0).any(axis=1)
            kadj = np.where(
                on_now, stre["dk_on"] + stre["b1k"] * stretch_z[pit],
                stre["dk_off"])
            mult[:, CI["K"]] *= np.exp(kadj)
            mult[:, CI["BB"]] *= np.exp(
                np.where(on_now, stre["db_on"], stre["db_off"]))
        P = P * mult
        P /= P.sum(axis=1, keepdims=True)
        cls = _sample_rows(P, rng.random(a.size))

        # ---- 5. batted-ball type when in play (class-conditional
        # under the contact tree: P(bb | outcome), 5-dim a2vec)
        bb = np.zeros(a.size, dtype=np.int8)
        inplay = np.isin(cls, [CI["1B"], CI["2B"], CI["3B"], CI["HR"],
                               CI["IPO"]])
        if inplay.any():
            if prep.a2vec.ndim == 5:
                P2 = prep.a2vec[pit[inplay],
                                bat_axis(batter_row[inplay]),
                                tto[inplay], cls[inplay]]
            else:
                P2 = prep.a2vec[pit[inplay],
                                bat_axis(batter_row[inplay]),
                                tto[inplay]]
            bb[inplay] = _sample_rows(P2, rng.random(int(inplay.sum())))

        # bookkeeping common to every outcome
        np.add.at(tensor, (a, batter_row, SIDX["PA"]), 1)
        np.add.at(tensor, (a, pit, SIDX["BF"]), 1)
        pit_bf[a, ft] += 1

        # ---- 6. advancement pattern per (class, bb, state, outs)
        st = ((bases[a, 0] >= 0).astype(np.int8)
              + 2 * (bases[a, 1] >= 0).astype(np.int8)
              + 4 * (bases[a, 2] >= 0).astype(np.int8))
        bb_code = np.where(inplay, bb.astype(np.int32) + 1, 0)
        key = pattern_key(cls.astype(np.int32), bb_code,
                          st.astype(np.int32), outs[a].astype(np.int32))
        dests = np.zeros((a.size, 4), dtype=np.int8)  # b, r1, r2, r3
        oadd = np.zeros(a.size, dtype=np.int8)
        pruns = np.zeros(a.size, dtype=np.int8)
        prbi = np.zeros(a.size, dtype=np.int8)
        pearn = np.zeros(a.size, dtype=np.int8)
        # identity-tilt bucket: lead-runner speed/value net of OF arm on
        # hits; batter GIDP propensity on GB outs with a runner on first
        hitm = np.isin(cls, [CI["1B"], CI["2B"], CI["3B"]])
        dpm = ((cls == CI["IPO"]) & (bb_code == 1) & ((st & 1) > 0)
               & (outs[a] < 2))
        lead = np.where(bases[a, 2] >= 0, bases[a, 2],
                        np.where(bases[a, 1] >= 0, bases[a, 1],
                                 bases[a, 0]))
        zsel = np.where(lead >= 0, run_z[np.maximum(lead, 0)],
                        run_z[batter_row]) + arm_eff[ft]
        zsel = np.where(dpm, dp_z[batter_row], zsel)
        bucket = np.where(hitm | dpm, np.digitize(zsel, TILT_EDGES), 2)
        u = rng.random(a.size)
        for k in np.unique(key):
            grp = key == k
            entry = pat.get(int(k))
            if entry is None:
                entry = _fallback_pattern(int(k), pat,
                                          empty_key_fallback)
            cum, arr = entry
            for b in np.unique(bucket[grp]):
                sub = grp & (bucket == b)
                pick = np.searchsorted(cum[b], u[sub], side="right")
                pick = np.minimum(pick, len(arr) - 1)
                dests[sub] = arr[pick, :4]
                oadd[sub] = arr[pick, 4]
                pruns[sub] = arr[pick, 5]
                prbi[sub] = arr[pick, 6]
                pearn[sub] = arr[pick, 7]

        # ---- walk-off capping: bottom of 9th+ non-HR hit that scores
        # more than needed only counts the winning run
        bot_late = (half[a] == 1) & (inning[a] >= REG)
        need = (score[a, 0] - score[a, 1] + 1).astype(np.int16)
        wo = bot_late & (pruns >= need) & (need > 0)
        non_hr = cls != CI["HR"]
        cap = wo & non_hr & (pruns > need)
        if cap.any():
            pruns[cap] = need[cap]
            prbi[cap] = np.minimum(prbi[cap], pruns[cap])
            pearn[cap] = np.minimum(pearn[cap], pruns[cap])
            # official scoring: only the winning run counts — demote the
            # trailing dest==4 movements (lead runner scores first) so
            # the per-player R ledger matches the capped scoreboard
            ci = np.flatnonzero(cap)
            dsub = dests[ci][:, [3, 2, 1, 0]].astype(np.int8)
            ksc = np.cumsum((dsub == 4).astype(np.int16), axis=1)
            demote = (dsub == 4) & (ksc > need[ci][:, None])
            fill = np.where(np.arange(4)[None, :] == 3, 1, 9)
            dsub = np.where(demote, fill, dsub).astype(np.int8)
            dests[ci[:, None], np.array([3, 2, 1, 0])[None, :]] = dsub

        _apply_pattern(a, bt, cls, dests, oadd, pruns, prbi, pearn,
                       bases, resp, outs, score, tensor, batter_row, pit)
        np.add.at(pit_runs, (a, ft), pruns.astype(np.int16))
        f5 = inning[a] <= 5
        np.add.at(runs_f5, (a[f5], bt[f5]), pruns[f5].astype(np.int16))
        i1 = inning[a] == 1
        np.add.at(runs_i1, (a[i1], bt[i1]), pruns[i1].astype(np.int16))

        bat_ptr[a, bt] = (slot + 1) % 9

        ended_wo = np.zeros(S, dtype=bool)
        ended_wo[a] = wo & (pruns >= need) & (need > 0)
        done |= ended_wo & ~done

        over = act_mask & (outs >= 3) & ~done
        if over.any():
            _end_half(over, inning, half, outs, bases, resp, score,
                      runs_f5, runs_i1, bat_ptr, done, stint_outs,
                      cur_pit, starter_in, pen_next, prep, rng, tensor,
                      REG, rules, active, bench, pen_pick)

    leftover = int((~done).sum())
    if leftover:
        done[:] = True

    return dict(tensor=tensor, score=score, runs_f5=runs_f5,
                runs_i1=runs_i1, leftover=leftover,
                stats=STATS, seed=seed)


def _fallback_pattern(k, pat, cache):
    """Unseen (class, bb, state, outs) key: fall back to the same class
    with bb stripped, then to bases-empty for the class."""
    if k in cache:
        return cache[k]
    rem, outs = divmod(k, 3)
    rem, st = divmod(rem, 8)
    cls, bb = divmod(rem, 5)
    for cand in (pattern_key(cls, 0, st, outs),
                 pattern_key(cls, bb, 0, outs),
                 pattern_key(cls, 0, 0, outs),
                 pattern_key(cls, 0, 0, 0)):
        if cand in pat:
            cache[k] = pat[cand]
            return pat[cand]
    arr = np.array([[1, 9, 9, 9, 0, 0, 0, 0]], dtype=np.int8)
    cache[k] = (np.ones((NB, 1)), arr)
    return cache[k]


def _apply_pattern(a, bt, cls, dests, oadd, pruns, prbi, pearn, bases,
                   resp, outs, score, tensor, batter_row, pit):
    outs[a] += oadd
    np.add.at(score, (a, bt), pruns.astype(np.int16))

    for ci, stat in HITS.items():
        hit = cls == ci
        if hit.any():
            np.add.at(tensor, (a[hit], batter_row[hit], SIDX["H"]), 1)
            np.add.at(tensor, (a[hit], batter_row[hit], SIDX[stat]), 1)
    for ci, stat in ((CI["BB"], "BB"), (CI["HBP"], "HBP"),
                     (CI["K"], "K")):
        m = cls == ci
        if m.any():
            np.add.at(tensor, (a[m], batter_row[m], SIDX[stat]), 1)
    np.add.at(tensor, (a, batter_row, SIDX["RBI"]), prbi)
    km = cls == CI["K"]
    np.add.at(tensor, (a[km], pit[km], SIDX["PK_"]), 1)
    bm = np.isin(cls, [CI["BB"], CI["HBP"]])
    np.add.at(tensor, (a[bm], pit[bm], SIDX["PBB"]), 1)
    hm = np.isin(cls, list(HITS))
    np.add.at(tensor, (a[hm], pit[hm], SIDX["PH"]), 1)
    hr = cls == CI["HR"]
    np.add.at(tensor, (a[hr], pit[hr], SIDX["PHR"]), 1)
    np.add.at(tensor, (a, pit, SIDX["OUTS"]), oadd)

    # runner movement, lead base first: higher bases only ever move UP,
    # so writing lead runners first leaves their origins free for the
    # trailing runners processed after them. bases hold PLAYER ROWS.
    new_bases = bases[a].copy()
    new_resp = resp[a].copy()
    earned_left = pearn.copy()
    for b in (2, 1, 0):                       # r3, r2, r1
        d = dests[:, b + 1]
        occ = bases[a, b] >= 0
        runner_row = bases[a, b]
        moved = occ & (d != 9)
        if not moved.any():
            continue
        scored = moved & (d == 4)
        if scored.any():
            np.add.at(tensor, (a[scored], runner_row[scored], SIDX["R"]),
                      1)
            pr = resp[a[scored], b]
            ok = pr >= 0
            np.add.at(tensor, (a[scored][ok], pr[ok], SIDX["PR"]), 1)
            has_e = earned_left[scored] > 0
            np.add.at(tensor, (a[scored][ok & has_e], pr[ok & has_e],
                               SIDX["PER"]), 1)
            earned_left[np.flatnonzero(scored)[has_e]] -= 1
        for tgt, code in ((0, 1), (1, 2), (2, 3)):
            mv = moved & (d == code)
            if mv.any():
                new_bases[mv, tgt] = runner_row[mv]
                new_resp[mv, tgt] = resp[a[mv], b]
        new_bases[moved, b] = np.where(
            new_bases[moved, b] == runner_row[moved], -1,
            new_bases[moved, b])
        new_resp[moved, b] = np.where(
            new_bases[moved, b] == -1, -1, new_resp[moved, b])

    bd = dests[:, 0]
    b_sc = bd == 4
    if b_sc.any():
        np.add.at(tensor, (a[b_sc], batter_row[b_sc], SIDX["R"]), 1)
        np.add.at(tensor, (a[b_sc], pit[b_sc], SIDX["PR"]), 1)
        has_e = earned_left[b_sc] > 0
        np.add.at(tensor, (a[b_sc][has_e], pit[b_sc][has_e],
                           SIDX["PER"]), 1)
    for tgt, code in ((0, 1), (1, 2), (2, 3)):
        m = bd == code
        if m.any():
            new_bases[m, tgt] = batter_row[m]
            new_resp[m, tgt] = pit[m]

    bases[a] = new_bases
    resp[a] = new_resp


def _end_half(mask, inning, half, outs, bases, resp, score, runs_f5,
              runs_i1, bat_ptr, done, stint_outs, cur_pit, starter_in,
              pen_next, prep, rng, tensor, REG, rules, active, bench,
              pen_pick=None):
    idx = np.flatnonzero(mask)
    was_top = half[idx] == 0
    top_done = idx[was_top]
    over_home = top_done[(inning[top_done] >= REG)
                         & (score[top_done, 1] > score[top_done, 0])]
    bot_done = idx[~was_top]
    over_away = bot_done[(inning[bot_done] >= REG)
                         & (score[bot_done, 0] != score[bot_done, 1])]
    hard_cap = bot_done[inning[bot_done] >= MAX_INNINGS]
    done[over_home] = True
    done[over_away] = True
    done[hard_cap] = True

    fld = np.where(was_top, 1, 0)
    non_start = ~starter_in[idx, fld]
    if non_start.any():
        ridx = idx[non_start]
        rfld = fld[non_start]
        stint_outs[ridx, rfld] += 3
        so = np.clip(stint_outs[ridx, rfld], 0, 10)
        ret = np.asarray(prep.relief_exit)
        exit_p = (ret[cur_pit[ridx, rfld], so] if ret.ndim == 2
                  else ret[so])
        leave = rng.random(ridx.size) < exit_p
        if leave.any():
            lidx, lfld = ridx[leave], rfld[leave]
            if pen_pick is not None:
                hi = ((inning[lidx] >= 7)
                      & (np.abs(score[lidx, 0].astype(np.int32)
                                - score[lidx, 1]) <= 2))
                # due batter = the OTHER side's pointer (it does not
                # advance during the intervening half-inning)
                bs = getattr(prep, "bat_side", None)
                due = None
                if bs is not None:
                    btf = (1 - lfld).astype(np.int16)
                    s18d = (bat_ptr[lidx, btf].astype(np.int16)
                            + 9 * btf)
                    due = np.where(active[lidx, s18d], bs[s18d],
                                   2).astype(np.int8)
                nxt = pen_pick(lidx, lfld, hi, due)
            else:
                nxt = prep.pen_order[lidx, lfld, pen_next[lidx, lfld]]
                pen_next[lidx, lfld] = np.minimum(
                    pen_next[lidx, lfld] + 1,
                    prep.pen_order.shape[2] - 1)
            has = nxt >= 0
            cur_pit[lidx[has], lfld[has]] = nxt[has]
            stint_outs[lidx, lfld] = 0

    outs[idx] = 0
    bases[idx] = -1
    resp[idx] = -1
    half[idx] = np.where(was_top, 1, 0)
    inning[idx] += (~was_top).astype(np.int8)

    live = idx[~done[idx]]
    ghost = live[(inning[live] > REG) if rules["ghost_runner"] else
                 np.zeros(live.size, dtype=bool)]
    if ghost.size:
        bt = half[ghost].astype(np.int16)
        prev = (bat_ptr[ghost, bt] - 1) % 9
        idx18 = prev.astype(np.int16) + 9 * bt
        act = active[ghost, idx18]
        bases[ghost, 1] = np.where(act, idx18, bench[bt])
        resp[ghost, 1] = cur_pit[ghost, 1 - bt]
