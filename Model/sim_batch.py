"""Batched replay path: preparation + Monte Carlo engine in one module.

PREPARATION (prepare_games): prepare_game's per-game pandas/model
overhead amortized across a chunk of games. prepare_game spends ~70%
of replay wall time on fixed per-call costs: one assemble_features
call, two predict_proba calls, one hazard-grid predict, two
steal-model predicts, per game. prepare_games builds the SAME GamePrep
objects (validated allclose against prepare_game) with ONE call each
across G games: the matchup frame is G x 1080 rows, the hazard frame
G x 2 x 451, the steal frame G x 360. Only prepare_game's math is
reproduced — pen selection, lineups, and participation state reuse the
Predictor's own (cached) methods, so there is a single source of truth
for roster logic.

ENGINE (run_batch): same game logic as sim.py (single source of truth
for rules and patterns — constants are imported), restructured for
throughput:

  - The sim axis holds G games x S sims = N rows at once. Per-game
    inputs (avec, hazard grids, steal matrices, pen order, bench rows,
    pre-event rates, rules) stack along a leading game axis and every
    row indexes them through its game id. One 30-game slate at 4k sims
    is a 120k-row batch — enough parallelism for a GPU to matter.
  - The pattern bank is DENSE: all 8x5x8x3 = 960 (class, bb, state,
    outs) keys are resolved at load (fallback chain included) into a
    padded cumulative matrix + outcome tensor that live on device. The
    per-step np.unique/host-dict loop in sim.py becomes two gathers.
  - The array module is a parameter: xp=numpy runs the identical
    batched logic on CPU (validates batching against sim.py), xp=cupy
    runs it on CUDA (validated distributionally against numpy).

Outputs per game match sim.run exactly (tensor/score/runs_f5/runs_i1),
so game_frame, calibration replays and the workbook path are unchanged.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F      # noqa: E402
import predict as PR      # noqa: E402
import sim as S_          # noqa: E402

CLASSES = S_.CLASSES
CI = S_.CI
NCLS = S_.NCLS
ONBASE = S_.ONBASE
HITS = S_.HITS
SIDX = S_.SIDX
STATS = S_.STATS
MAX_INNINGS = S_.MAX_INNINGS
N_KEYS = 8 * 5 * 8 * 3


def dense_patterns(pat):
    """sim.load_patterns dict -> (cum [960, NB, K], arr [960, K, 8])
    with every possible key resolved (fallback chain applied up front),
    the identity-tilt bucket axis preserved, and rows padded with
    cum=2.0 so a comparison-sum sample never lands on padding."""
    cache = {}
    kmax = max(c.shape[1] for c, _ in pat.values())
    nb = S_.NB
    cum = np.full((N_KEYS, nb, kmax), 2.0, dtype=np.float32)
    arr = np.zeros((N_KEYS, kmax, 8), dtype=np.int8)
    for k in range(N_KEYS):
        entry = pat.get(k)
        if entry is None:
            entry = S_._fallback_pattern(k, pat, cache)
        c, a = entry
        n = c.shape[1]
        cum[k, :, :n] = c
        arr[k, :n] = a
        if n:
            arr[k, n:] = a[-1]
    return cum, arr


class BatchPrep:
    """Stacked per-game arrays + rule vectors for one batch."""

    def __init__(self, preps, seasons, is_dh, xp):
        G = len(preps)
        self.G = G
        f32 = xp.float32
        self.avec = xp.asarray(
            np.stack([p.avec for p in preps]), dtype=f32)
        self.a2vec = xp.asarray(
            np.stack([p.a2vec for p in preps]), dtype=f32)
        self.haz = xp.asarray(
            np.stack([p.haz_grid for p in preps]), dtype=f32)
        # relief_exit: legacy league [11] or per-pitcher [n_players, 11]
        # rows (B4); broadcast legacy games so the batch is uniform 3-D
        rel = [np.asarray(p.relief_exit, dtype=np.float32)
               for p in preps]
        np0 = preps[0].n_players
        rel = [r if r.ndim == 2 else np.broadcast_to(r, (np0, r.size))
               for r in rel]
        self.relief = xp.asarray(np.stack(rel), dtype=f32)
        # pen_order in GamePrep is [n_sims, 2, P] broadcast from one
        # [1, 2, P] row — keep the per-game row only
        self.pen = xp.asarray(
            np.stack([np.asarray(p.pen_order[0]) for p in preps]),
            dtype=np.int16)
        self.sb_att = xp.asarray(
            np.stack([p.sb_att for p in preps]), dtype=f32)
        self.sb_suc = xp.asarray(
            np.stack([p.sb_suc for p in preps]), dtype=f32)
        sbs = [getattr(p, "sb_state", None) for p in preps]
        self.sb_state = (xp.asarray(np.stack(sbs), dtype=f32)
                         if all(s is not None for s in sbs) else None)
        self.bench = xp.asarray(
            np.stack([np.asarray(p.bench_rows) for p in preps]),
            dtype=np.int16)
        self.bat_side = xp.asarray(
            np.stack([np.asarray(p.bat_side) for p in preps]),
            dtype=np.int8)
        self.pit_throws = xp.asarray(
            np.stack([np.asarray(p.pit_throws) for p in preps]),
            dtype=np.int8)
        self.slot_is_c = xp.asarray(
            np.stack([np.asarray(p.slot_is_c) for p in preps]),
            dtype=np.int8)
        self.pre_pk = xp.asarray(
            np.array([p.pre_pk for p in preps]), dtype=f32)
        # pre_wp is per-fielding-side (catcher grain); scalars broadcast
        self.pre_wp = xp.asarray(np.stack([
            np.broadcast_to(np.asarray(p.pre_wp, dtype=np.float32)
                            .ravel() if np.size(p.pre_wp) > 1
                            else np.full(2, float(p.pre_wp)), (2,))
            for p in preps]), dtype=f32)
        np_ = preps[0].n_players
        z0 = np.zeros(np_, dtype=np.float32)
        self.run_z = xp.asarray(np.stack(
            [np.asarray(getattr(p, "run_z", z0)) for p in preps]),
            dtype=f32)
        self.dp_z = xp.asarray(np.stack(
            [np.asarray(getattr(p, "dp_z", z0)) for p in preps]),
            dtype=f32)
        self.arm_eff = xp.asarray(np.stack(
            [np.asarray(getattr(p, "arm_eff", np.zeros(2)))
             for p in preps]), dtype=f32)
        self.stretch_z = xp.asarray(np.stack(
            [np.asarray(getattr(p, "stretch_z", z0)) for p in preps]),
            dtype=f32)
        self.stretch = getattr(preps[0], "stretch", None)
        ph = [getattr(p, "pen_hi", None) for p in preps]
        pl = [getattr(p, "pen_lo", None) for p in preps]
        if all(x is not None for x in ph) and \
                all(x is not None for x in pl):
            self.pen_hi = xp.asarray(np.stack(ph), dtype=np.int16)
            self.pen_lo = xp.asarray(np.stack(pl), dtype=np.int16)
        else:
            self.pen_hi = self.pen_lo = None
        part = preps[0].part_haz
        self.part = (xp.asarray(part, dtype=f32)
                     if part is not None else None)
        rc = getattr(preps[0], "pen_rank_cum", None)
        self.rank_cum = (xp.asarray(rc, dtype=f32)
                         if rc is not None else None)
        # per-game latent params (B11 fix: was preps[0].latent for all)
        lats = [p.latent or {} for p in preps]
        self.lat = {
            k: xp.asarray(np.array([float(l.get(k, 0.0)) for l in lats]),
                          dtype=f32)
            for k in ("mu_env", "sigma_env", "sigma_hr",
                      "sigma_pitcher", "sigma_offense", "sigma_k")}
        self.n_players = preps[0].n_players
        rules = [S_.rules_for(s, d) for s, d in zip(seasons, is_dh)]
        self.reg = xp.asarray(
            np.array([r["regulation"] for r in rules]), dtype=np.int8)
        self.ghost = xp.asarray(
            np.array([r["ghost_runner"] for r in rules]), dtype=bool)
        cum, arr = dense_patterns(preps[0].patterns)
        self.pat_cum = xp.asarray(cum)
        self.pat_arr = xp.asarray(arr)


def _sample_dense(xp, cum_rows, u):
    """cum_rows: [n, K] padded cumulative probs; u: [n] -> pick idx."""
    return (u[:, None] > cum_rows).sum(axis=1).astype(xp.int32)


def run_batch(preps, n_sims=4000, seed=1, seasons=None, is_dh=None,
              backend="gpu"):
    """Simulate G games x n_sims each; returns a list of per-game dicts
    shaped exactly like sim.run's output."""
    if backend == "gpu":
        import warnings
        with warnings.catch_warnings():
            # pip cupy bundles its CUDA libs; the CUDA_PATH probe miss
            # is cosmetic and the sims prove the GPU path works
            warnings.filterwarnings(
                "ignore", message="CUDA path could not be detected.*")
            import cupy as xp
    else:
        xp = np
    G = len(preps)
    seasons = seasons if seasons is not None else [2026] * G
    is_dh = is_dh if is_dh is not None else [False] * G
    bp = BatchPrep(preps, seasons, is_dh, xp)
    rng = (xp.random.default_rng(seed) if xp is not np
           else np.random.default_rng(seed))
    S = n_sims
    N = G * S
    n_players = bp.n_players
    gidx = xp.repeat(xp.arange(G, dtype=xp.int32), S)

    # CuPy scatter_add rejects int16 targets — accumulate in int32 on
    # GPU, cast back to int16 at host exit (numpy path stays int16)
    it_acc = xp.int16 if xp is np else xp.int32
    tensor = xp.zeros((N, n_players, len(STATS)), dtype=it_acc)
    score = xp.zeros((N, 2), dtype=it_acc)
    runs_f5 = xp.zeros((N, 2), dtype=it_acc)
    runs_i1 = xp.zeros((N, 2), dtype=it_acc)

    inning = xp.ones(N, dtype=xp.int8)
    half = xp.zeros(N, dtype=xp.int8)
    outs = xp.zeros(N, dtype=xp.int8)
    bases = -xp.ones((N, 3), dtype=xp.int16)
    resp = -xp.ones((N, 3), dtype=xp.int16)
    bat_ptr = xp.zeros((N, 2), dtype=xp.int8)
    cur_pit = xp.tile(xp.asarray([18, 19], dtype=xp.int16), (N, 1))
    pit_bf = xp.zeros((N, 2), dtype=xp.int16)
    pit_runs = xp.zeros((N, 2), dtype=it_acc)
    starter_in = xp.ones((N, 2), dtype=bool)
    pen_next = xp.zeros((N, 2), dtype=xp.int8)
    stint_outs = xp.zeros((N, 2), dtype=xp.int8)
    done = xp.zeros(N, dtype=bool)
    active = xp.ones((N, 18), dtype=bool)
    kpa = xp.zeros((N, 18), dtype=xp.int8)

    if xp is np:
        def sadd(arr, idx, val):
            np.add.at(arr, idx, val)
    else:
        import cupyx

        def sadd(arr, idx, val):
            if not np.isscalar(val) and val.dtype != arr.dtype:
                val = val.astype(arr.dtype)
            cupyx.scatter_add(arr, idx, val)

    lat = bp.lat
    z_env = (lat["mu_env"][gidx] + lat["sigma_env"][gidx]
             * rng.standard_normal(N, dtype=xp.float32))
    z_hr = lat["sigma_hr"][gidx] * rng.standard_normal(
        N, dtype=xp.float32)
    z_pit = lat["sigma_pitcher"][gidx][:, None] * rng.standard_normal(
        (N, 2), dtype=xp.float32)
    z_off = lat["sigma_offense"][gidx][:, None] * rng.standard_normal(
        (N, 2), dtype=xp.float32)
    z_k = lat["sigma_k"][gidx][:, None] * rng.standard_normal(
        (N, 2), dtype=xp.float32)

    reg_n = bp.reg[gidx]                 # per-row regulation innings
    ghost_n = bp.ghost[gidx]
    tilt_edges = xp.asarray(S_.TILT_EDGES, dtype=xp.float32)

    # leverage-aware pen (absent orders -> legacy fixed sequence)
    lev = bp.pen_hi is not None
    pen_used = (xp.zeros((N, 2, bp.pen_hi.shape[2]), dtype=bool)
                if lev else None)

    def pick_pen(idx, side, hi_mask, due_stand=None):
        s32 = side.astype(xp.int32)
        order = xp.where(hi_mask[:, None], bp.pen_hi[gidx[idx], s32],
                         bp.pen_lo[gidx[idx], s32])
        npen = order.shape[1]
        slots = xp.clip(order - 20 - 8 * s32[:, None], 0,
                        npen - 1).astype(xp.int32)
        ok = (order >= 0) & ~pen_used[idx[:, None], s32[:, None], slots]
        anyok = ok.any(axis=1)
        if bp.rank_cum is not None:      # B6: pmf-sampled entry rank
            cum = bp.rank_cum[xp.where(hi_mask, 0, 1)]
            cnt = ok.sum(axis=1).astype(xp.int32)
            k = xp.clip(xp.minimum(cnt, cum.shape[1]) - 1, 0, None)
            tot = cum[xp.arange(int(idx.size)), k]
            u = rng.random(int(idx.size), dtype=xp.float32) * tot
            r = xp.minimum(
                (u[:, None] > cum).sum(axis=1).astype(xp.int32),
                xp.maximum(cnt - 1, 0))
            pos = xp.minimum((xp.cumsum(ok, axis=1) <= r[:, None])
                             .sum(axis=1), npen - 1).astype(xp.int32)
        else:
            pos = ok.argmax(axis=1).astype(xp.int32)
            if due_stand is not None:
                rows_ok = xp.clip(order, 0,
                                  bp.pit_throws.shape[1] - 1).astype(
                                      xp.int32)
                hands = bp.pit_throws[gidx[idx][:, None], rows_ok]
                match = (ok & (hands == due_stand[:, None])
                         & (due_stand[:, None] != 2))
                match[:, S_.PLATOON_WIN:] = False
                m_any = match.any(axis=1)
                pos = xp.where(m_any,
                               match.argmax(axis=1).astype(xp.int32),
                               pos)
        chosen = xp.where(anyok, order[xp.arange(int(idx.size)), pos],
                          -1).astype(xp.int16)
        cslot = xp.clip(chosen.astype(xp.int32) - 20 - 8 * s32, 0,
                        npen - 1)
        pen_used[idx[anyok], s32[anyok], cslot[anyok]] = True
        return chosen

    def bat_axis(rows):
        return rows - 18 * (rows >= 36).astype(rows.dtype)

    def end_half(mask):
        idx = xp.flatnonzero(mask)
        if idx.size == 0:
            return
        was_top = half[idx] == 0
        regi = reg_n[idx]
        top_done = idx[was_top]
        over_home = top_done[(inning[top_done] >= reg_n[top_done])
                             & (score[top_done, 1] > score[top_done, 0])]
        bot_done = idx[~was_top]
        over_away = bot_done[(inning[bot_done] >= reg_n[bot_done])
                             & (score[bot_done, 0]
                                != score[bot_done, 1])]
        hard_cap = bot_done[inning[bot_done] >= MAX_INNINGS]
        done[over_home] = True
        done[over_away] = True
        done[hard_cap] = True

        fld = xp.where(was_top, 1, 0).astype(xp.int16)
        non_start = ~starter_in[idx, fld]
        if bool(non_start.any()):
            ridx = idx[non_start]
            rfld = fld[non_start]
            stint_outs[ridx, rfld] += 3
            exit_p = bp.relief[gidx[ridx],
                               cur_pit[ridx, rfld].astype(xp.int32),
                               xp.clip(stint_outs[ridx, rfld],
                                       0, 10).astype(xp.int32)]
            leave = rng.random(int(ridx.size), dtype=xp.float32) < exit_p
            if bool(leave.any()):
                lidx, lfld = ridx[leave], rfld[leave]
                if lev:
                    hi = ((inning[lidx] >= 7)
                          & (xp.abs(score[lidx, 0].astype(xp.int32)
                                    - score[lidx, 1]) <= 2))
                    # due batter = other side's pointer (static during
                    # the intervening half-inning)
                    btf = (1 - lfld).astype(xp.int16)
                    s18d = (bat_ptr[lidx, btf].astype(xp.int16)
                            + 9 * btf).astype(xp.int32)
                    due = xp.where(active[lidx, s18d],
                                   bp.bat_side[gidx[lidx], s18d],
                                   2).astype(xp.int8)
                    nxt = pick_pen(lidx, lfld, hi, due)
                else:
                    nxt = bp.pen[gidx[lidx], lfld,
                                 pen_next[lidx, lfld].astype(xp.int32)]
                    pen_next[lidx, lfld] = xp.minimum(
                        pen_next[lidx, lfld] + 1, bp.pen.shape[2] - 1)
                has = nxt >= 0
                cur_pit[lidx[has], lfld[has]] = nxt[has]
                stint_outs[lidx, lfld] = 0

        outs[idx] = 0
        bases[idx] = -1
        resp[idx] = -1
        half[idx] = xp.where(was_top, 1, 0).astype(xp.int8)
        inning[idx] += (~was_top).astype(xp.int8)

        live = idx[~done[idx]]
        if live.size:
            gh = live[ghost_n[live] & (inning[live] > reg_n[live])]
            if gh.size:
                bt = half[gh].astype(xp.int16)
                prev = (bat_ptr[gh, bt] - 1) % 9
                idx18 = prev.astype(xp.int16) + 9 * bt
                act = active[gh, idx18]
                bases[gh, 1] = xp.where(act, idx18,
                                        bp.bench[gidx[gh], bt])
                resp[gh, 1] = cur_pit[gh, 1 - bt]

    for _step in range(400):
        act_mask = ~done
        if not bool(act_mask.any()):
            break
        a = xp.flatnonzero(act_mask)
        ga = gidx[a]
        bt = half[a].astype(xp.int16)
        ft = (1 - bt).astype(xp.int16)
        slot = bat_ptr[a, bt]

        # ---- 1. pre-events: pickoff, then WP/PB
        any_on = (bases[a] >= 0).any(axis=1)
        u = rng.random(int(a.size), dtype=xp.float32)
        pk = any_on & (u < bp.pre_pk[ga]) & (bases[a, 0] >= 0)
        if bool(pk.any()):
            idx = a[pk]
            runner_row = bases[idx, 0]
            sadd(tensor, (idx, runner_row.astype(xp.int32),
                          SIDX["CS"]), 1)
            idx_fld = (1 - half[idx]).astype(xp.int16)
            sadd(tensor, (idx, cur_pit[idx, idx_fld].astype(xp.int32),
                          SIDX["OUTS"]), 1)
            outs[idx] += 1
            bases[idx, 0] = -1
            resp[idx, 0] = -1
        u = rng.random(int(a.size), dtype=xp.float32)
        wp = any_on & (u < bp.pre_wp[ga, ft.astype(xp.int32)]) \
            & (outs[a] < 3)
        if bool(wp.any()):
            idx = a[wp]
            for b in (2, 1, 0):
                mv = bases[idx, b] >= 0
                if not bool(mv.any()):
                    continue
                src = idx[mv]
                if b == 2:
                    runner_row = bases[src, 2]
                    sadd(tensor, (src, runner_row.astype(xp.int32),
                                  SIDX["R"]), 1)
                    sadd(score, (src, half[src].astype(xp.int32)), 1)
                    f5m = inning[src] <= 5
                    sadd(runs_f5, (src[f5m],
                                   half[src][f5m].astype(xp.int32)), 1)
                    i1m = inning[src] == 1
                    sadd(runs_i1, (src[i1m],
                                   half[src][i1m].astype(xp.int32)), 1)
                    pr = resp[src, 2]
                    ok = pr >= 0
                    sadd(tensor, (src[ok], pr[ok].astype(xp.int32),
                                  SIDX["PR"]), 1)
                    sadd(tensor, (src[ok], pr[ok].astype(xp.int32),
                                  SIDX["PER"]), 1)
                    src_fld = (1 - half[src]).astype(xp.int32)
                    sadd(pit_runs, (src, src_fld), 1)
                    bases[src, 2] = -1
                    resp[src, 2] = -1
                else:
                    open_ = bases[src, b + 1] < 0
                    mvs = src[open_]
                    bases[mvs, b + 1] = bases[mvs, b]
                    resp[mvs, b + 1] = resp[mvs, b]
                    bases[mvs, b] = -1
                    resp[mvs, b] = -1
            wo_wp = idx[(half[idx] == 1) & (inning[idx] >= reg_n[idx])
                        & (score[idx, 1] > score[idx, 0])]
            done[wo_wp] = True
            act_mask = ~done
            a = xp.flatnonzero(act_mask)
            if a.size == 0:
                continue
            ga = gidx[a]
            bt = half[a].astype(xp.int16)
            ft = (1 - bt).astype(xp.int16)
            slot = bat_ptr[a, bt]

        # ---- 2. steal of second
        can = (bases[a, 0] >= 0) & (bases[a, 1] < 0) & (outs[a] < 3)
        if bool(can.any()):
            idx = a[can]
            r_row = bases[idx, 0].astype(xp.int32)
            pit_i = cur_pit[idx, (1 - half[idx]).astype(xp.int16)
                            ].astype(xp.int32)
            p_att = bp.sb_att[gidx[idx], r_row, pit_i]
            if bp.sb_state is not None:
                st = bp.sb_state[gidx[idx]]
                base = xp.clip(p_att / st[:, 3], 1e-6, 1 - 1e-6)
                lg = xp.log(base / (1 - base))
                o = outs[idx]
                lg = (lg + xp.where(o == 0, st[:, 0], 0.0)
                      + xp.where(o == 2, st[:, 1], 0.0))
                far = xp.abs(score[idx, 0].astype(xp.int32)
                             - score[idx, 1]) > 2
                lg = lg + xp.where(far, st[:, 2], 0.0)
                p_att = xp.clip(st[:, 3] / (1.0 + xp.exp(-lg)),
                                0.0, 0.95).astype(xp.float32)
            go = rng.random(int(idx.size), dtype=xp.float32) < p_att
            if bool(go.any()):
                gidx2 = idx[go]
                g_row = r_row[go]
                pit_g = pit_i[go]
                p_suc = bp.sb_suc[gidx[gidx2], g_row, pit_g]
                win = rng.random(int(gidx2.size),
                                 dtype=xp.float32) < p_suc
                w, l = gidx2[win], gidx2[~win]
                sadd(tensor, (w, g_row[win], SIDX["SB"]), 1)
                bases[w, 1] = bases[w, 0]
                resp[w, 1] = resp[w, 0]
                bases[w, 0] = -1
                resp[w, 0] = -1
                sadd(tensor, (l, g_row[~win], SIDX["CS"]), 1)
                l_fld = (1 - half[l]).astype(xp.int16)
                sadd(tensor, (l, cur_pit[l, l_fld].astype(xp.int32),
                              SIDX["OUTS"]), 1)
                outs[l] += 1
                bases[l, 0] = -1
                resp[l, 0] = -1

        self_ended = xp.zeros(N, dtype=bool)
        self_ended[a] = outs[a] >= 3
        if bool(self_ended.any()):
            end_half(self_ended)
            keep = ~self_ended[a] & ~done[a]
            a = a[keep]
            if a.size == 0:
                continue
            ga = gidx[a]
            bt = half[a].astype(xp.int16)
            ft = (1 - bt).astype(xp.int16)
            slot = bat_ptr[a, bt]

        # ---- 3. starter removal hazard
        chk = starter_in[a, ft] & (pit_bf[a, ft] > 0)
        if bool(chk.any()):
            idx = a[chk]
            ftc = ft[chk]
            bf = xp.clip(pit_bf[idx, ftc], 0, 40).astype(xp.int32)
            rn = xp.clip(pit_runs[idx, ftc], 0, 10).astype(xp.int32)
            hz = bp.haz[gidx[idx], ftc.astype(xp.int32), bf, rn]
            out_now = rng.random(int(idx.size), dtype=xp.float32) < hz
            if bool(out_now.any()):
                oidx = idx[out_now]
                oft = ftc[out_now]
                starter_in[oidx, oft] = False
                if lev:
                    hi = ((inning[oidx] >= 7)
                          & (xp.abs(score[oidx, 0].astype(xp.int32)
                                    - score[oidx, 1]) <= 2))
                    s18d = (slot[chk][out_now].astype(xp.int16)
                            + 9 * bt[chk][out_now]).astype(xp.int32)
                    due = xp.where(active[oidx, s18d],
                                   bp.bat_side[gidx[oidx], s18d],
                                   2).astype(xp.int8)
                    nxt = pick_pen(oidx, oft, hi, due)
                else:
                    nxt = bp.pen[gidx[oidx], oft,
                                 pen_next[oidx, oft].astype(xp.int32)]
                    pen_next[oidx, oft] = xp.minimum(
                        pen_next[oidx, oft] + 1, bp.pen.shape[2] - 1)
                has = nxt >= 0
                cur_pit[oidx[has], oft[has]] = nxt[has]
                stint_outs[oidx, oft] = 0
        pit = cur_pit[a, ft].astype(xp.int32)

        # ---- 3b. batter participation hazard
        idx18 = (slot.astype(xp.int16) + 9 * bt).astype(xp.int32)
        if bp.part is not None:
            live = active[a, idx18]
            if bool(live.any()):
                li = xp.flatnonzero(live)
                sims = a[li]
                s18 = idx18[li]
                k = xp.clip(kpa[sims, s18] + 1, 1, 6).astype(
                    xp.int32) - 1
                inn = inning[sims]
                inn_b = xp.select([inn <= 6, inn == 7, inn == 8],
                                  [xp.zeros_like(inn),
                                   xp.ones_like(inn),
                                   xp.full_like(inn, 2)],
                                  default=3).astype(xp.int32)
                diff = (score[sims, bt[li]]
                        - score[sims, 1 - bt[li]]).astype(xp.int16)
                margin_b = xp.select(
                    [xp.abs(diff) <= 1, xp.abs(diff) <= 3],
                    [xp.zeros_like(diff), xp.ones_like(diff)],
                    default=2).astype(xp.int32)
                lead = (diff > 0).astype(xp.int32)
                side = bp.bat_side[gidx[sims], s18]
                same = ((side == bp.pit_throws[
                    gidx[sims], cur_pit[sims, ft[li]].astype(xp.int32)])
                    & (side != 2)).astype(xp.int32)
                isc = bp.slot_is_c[gidx[sims], s18].astype(xp.int32)
                rate = bp.part[k, inn_b, margin_b, lead, same, isc]
                gone = rng.random(int(sims.size),
                                  dtype=xp.float32) < rate
                if bool(gone.any()):
                    active[sims[gone], s18[gone]] = False
        act18 = active[a, idx18]
        batter_row = xp.where(act18, idx18.astype(xp.int16),
                              bp.bench[ga, bt]).astype(xp.int32)
        kpa[a, idx18] = xp.minimum(kpa[a, idx18] + 1, 7)

        # ---- 4. outcome sample with latent multipliers
        tto = xp.minimum(pit_bf[a, ft] // 9, 2).astype(xp.int32)
        P = bp.avec[ga, pit, bat_axis(batter_row), tto].copy()
        mult = xp.ones_like(P)
        mult[:, ONBASE] *= xp.exp(z_env[a])[:, None]
        mult[:, ONBASE] *= xp.exp(z_off[a, bt])[:, None]
        mult[:, CI["HR"]] *= xp.exp(z_hr[a])
        zp = z_pit[a, ft]
        mult[:, CI["K"]] *= xp.exp(zp)
        mult[:, ONBASE] *= xp.exp(-0.5 * zp)[:, None]
        # starter-only K-form draw; renormalization compensates
        mult[:, CI["K"]] *= xp.exp(z_k[a, ft] * starter_in[a, ft])
        # base-state (stretch) conditioning
        if bp.stretch is not None:
            stz = bp.stretch_z[ga, pit]
            on_now = (bases[a] >= 0).any(axis=1)
            kadj = xp.where(
                on_now,
                bp.stretch["dk_on"] + bp.stretch["b1k"] * stz,
                xp.float32(bp.stretch["dk_off"]))
            mult[:, CI["K"]] *= xp.exp(kadj)
            mult[:, CI["BB"]] *= xp.exp(xp.where(
                on_now, xp.float32(bp.stretch["db_on"]),
                xp.float32(bp.stretch["db_off"])))
        P = P * mult
        P /= P.sum(axis=1, keepdims=True)
        cum = xp.cumsum(P, axis=1)
        cum[:, -1] = 1.0 + 1e-9
        u = rng.random(int(a.size), dtype=xp.float32)
        cls = (u[:, None] > cum).sum(axis=1).astype(xp.int32)

        # ---- 5. batted-ball type when in play (class-conditional
        # under the contact tree: P(bb | outcome), 6-dim a2vec)
        bb = xp.zeros(int(a.size), dtype=xp.int32)
        inplay = ((cls == CI["1B"]) | (cls == CI["2B"])
                  | (cls == CI["3B"]) | (cls == CI["HR"])
                  | (cls == CI["IPO"]))
        if bool(inplay.any()):
            if bp.a2vec.ndim == 6:
                P2 = bp.a2vec[ga[inplay], pit[inplay],
                              bat_axis(batter_row[inplay]),
                              tto[inplay], cls[inplay]]
            else:
                P2 = bp.a2vec[ga[inplay], pit[inplay],
                              bat_axis(batter_row[inplay]),
                              tto[inplay]]
            cum2 = xp.cumsum(P2, axis=1)
            cum2[:, -1] = 1.0 + 1e-9
            u2 = rng.random(int(inplay.sum()), dtype=xp.float32)
            bb[inplay] = (u2[:, None] > cum2).sum(axis=1).astype(
                xp.int32)

        sadd(tensor, (a, batter_row, SIDX["PA"]), 1)
        sadd(tensor, (a, pit, SIDX["BF"]), 1)
        pit_bf[a, ft] += 1

        # ---- 6. advancement pattern (dense bank: two gathers)
        st = ((bases[a, 0] >= 0).astype(xp.int32)
              + 2 * (bases[a, 1] >= 0).astype(xp.int32)
              + 4 * (bases[a, 2] >= 0).astype(xp.int32))
        bb_code = xp.where(inplay, bb + 1, 0)
        key = ((cls * 5 + bb_code) * 8 + st) * 3 + outs[a].astype(
            xp.int32)
        # identity-tilt bucket (see sim.py): lead-runner speed/value net
        # of OF arm on hits; batter GIDP propensity on GB outs w/ r1
        hitm = ((cls == CI["1B"]) | (cls == CI["2B"])
                | (cls == CI["3B"]))
        dpm = ((cls == CI["IPO"]) & (bb_code == 1) & ((st & 1) > 0)
               & (outs[a] < 2))
        lead = xp.where(bases[a, 2] >= 0, bases[a, 2],
                        xp.where(bases[a, 1] >= 0, bases[a, 1],
                                 bases[a, 0])).astype(xp.int32)
        zsel = xp.where(lead >= 0,
                        bp.run_z[ga, xp.maximum(lead, 0)],
                        bp.run_z[ga, batter_row]) \
            + bp.arm_eff[ga, ft.astype(xp.int32)]
        zsel = xp.where(dpm, bp.dp_z[ga, batter_row], zsel)
        b_idx = (zsel[:, None] > tilt_edges[None, :]).sum(
            axis=1).astype(xp.int32)
        bucket = xp.where(hitm | dpm, b_idx, 2)
        u = rng.random(int(a.size), dtype=xp.float32)
        pick = _sample_dense(xp, bp.pat_cum[key, bucket], u)
        pick = xp.minimum(pick, bp.pat_arr.shape[1] - 1)
        chosen = bp.pat_arr[key, pick]          # [n, 8] int8
        dests = chosen[:, :4]
        oadd = chosen[:, 4]
        pruns = chosen[:, 5].astype(xp.int16)
        prbi = chosen[:, 6].astype(xp.int16)
        pearn = chosen[:, 7].astype(xp.int16)

        # ---- walk-off capping
        bot_late = (half[a] == 1) & (inning[a] >= reg_n[a])
        need = (score[a, 0] - score[a, 1] + 1).astype(xp.int16)
        wo = bot_late & (pruns >= need) & (need > 0)
        cap = wo & (cls != CI["HR"]) & (pruns > need)
        if bool(cap.any()):
            pruns = xp.where(cap, need, pruns)
            prbi = xp.where(cap, xp.minimum(prbi, pruns), prbi)
            pearn = xp.where(cap, xp.minimum(pearn, pruns), pearn)
            # demote trailing dest==4 so the R ledger matches (see sim.py)
            ci = xp.flatnonzero(cap)
            colmap = xp.asarray([3, 2, 1, 0])
            dsub = dests[ci][:, colmap].astype(xp.int8)
            ksc = xp.cumsum((dsub == 4).astype(xp.int16), axis=1)
            demote = (dsub == 4) & (ksc > need[ci][:, None])
            fill = xp.where(xp.arange(4)[None, :] == 3, 1, 9)
            dsub = xp.where(demote, fill, dsub).astype(xp.int8)
            dests[ci[:, None], colmap[None, :]] = dsub

        # ---- apply pattern (stats, runners, outs, scores)
        outs[a] += oadd.astype(xp.int8)
        sadd(score, (a, bt.astype(xp.int32)), pruns)
        for ci, stat in HITS.items():
            hit = cls == ci
            if bool(hit.any()):
                sadd(tensor, (a[hit], batter_row[hit], SIDX["H"]), 1)
                sadd(tensor, (a[hit], batter_row[hit], SIDX[stat]), 1)
        for ci, stat in ((CI["BB"], "BB"), (CI["HBP"], "HBP"),
                         (CI["K"], "K")):
            m = cls == ci
            if bool(m.any()):
                sadd(tensor, (a[m], batter_row[m], SIDX[stat]), 1)
        sadd(tensor, (a, batter_row, SIDX["RBI"]), prbi)
        km = cls == CI["K"]
        sadd(tensor, (a[km], pit[km], SIDX["PK_"]), 1)
        bm = (cls == CI["BB"]) | (cls == CI["HBP"])
        sadd(tensor, (a[bm], pit[bm], SIDX["PBB"]), 1)
        hm = ((cls == CI["1B"]) | (cls == CI["2B"])
              | (cls == CI["3B"]) | (cls == CI["HR"]))
        sadd(tensor, (a[hm], pit[hm], SIDX["PH"]), 1)
        hr = cls == CI["HR"]
        sadd(tensor, (a[hr], pit[hr], SIDX["PHR"]), 1)
        sadd(tensor, (a, pit, SIDX["OUTS"]), oadd.astype(xp.int16))

        new_bases = bases[a].copy()
        new_resp = resp[a].copy()
        earned_left = pearn.copy()
        for b in (2, 1, 0):
            d = dests[:, b + 1]
            occ = bases[a, b] >= 0
            runner_row = bases[a, b].astype(xp.int32)
            moved = occ & (d != 9)
            if not bool(moved.any()):
                continue
            scored = moved & (d == 4)
            if bool(scored.any()):
                sadd(tensor, (a[scored], runner_row[scored],
                              SIDX["R"]), 1)
                pr = resp[a[scored], b]
                ok = pr >= 0
                sadd(tensor, (a[scored][ok], pr[ok].astype(xp.int32),
                              SIDX["PR"]), 1)
                has_e = earned_left[scored] > 0
                sadd(tensor, (a[scored][ok & has_e],
                              pr[ok & has_e].astype(xp.int32),
                              SIDX["PER"]), 1)
                sc_idx = xp.flatnonzero(scored)
                dec = sc_idx[has_e]
                earned_left[dec] -= 1
            for tgt, code in ((0, 1), (1, 2), (2, 3)):
                mv = moved & (d == code)
                if bool(mv.any()):
                    new_bases[mv, tgt] = runner_row[mv].astype(xp.int16)
                    new_resp[mv, tgt] = resp[a[mv], b]
            rr16 = runner_row.astype(xp.int16)
            new_bases[moved, b] = xp.where(
                new_bases[moved, b] == rr16[moved], -1,
                new_bases[moved, b])
            new_resp[moved, b] = xp.where(
                new_bases[moved, b] == -1, -1, new_resp[moved, b])

        bd = dests[:, 0]
        b_sc = bd == 4
        if bool(b_sc.any()):
            sadd(tensor, (a[b_sc], batter_row[b_sc], SIDX["R"]), 1)
            sadd(tensor, (a[b_sc], pit[b_sc], SIDX["PR"]), 1)
            has_e = earned_left[b_sc] > 0
            sadd(tensor, (a[b_sc][has_e], pit[b_sc][has_e],
                          SIDX["PER"]), 1)
        for tgt, code in ((0, 1), (1, 2), (2, 3)):
            m = bd == code
            if bool(m.any()):
                new_bases[m, tgt] = batter_row[m].astype(xp.int16)
                new_resp[m, tgt] = pit[m].astype(xp.int16)
        bases[a] = new_bases
        resp[a] = new_resp

        sadd(pit_runs, (a, ft.astype(xp.int32)), pruns)
        f5 = inning[a] <= 5
        sadd(runs_f5, (a[f5], bt[f5].astype(xp.int32)), pruns[f5])
        i1 = inning[a] == 1
        sadd(runs_i1, (a[i1], bt[i1].astype(xp.int32)), pruns[i1])

        bat_ptr[a, bt] = ((slot + 1) % 9).astype(xp.int8)

        ended_wo = xp.zeros(N, dtype=bool)
        ended_wo[a] = wo & (pruns >= need) & (need > 0)
        done |= ended_wo & ~done

        over = (~done) & (outs >= 3)
        if bool(over.any()):
            end_half(over)

    leftover = int((~done).sum())

    # ---- split back to per-game results on host
    def host(x):
        h = x.get() if xp is not np else x
        return h.astype(np.int16, copy=False)

    tensor_h = host(tensor)
    score_h = host(score)
    f5_h = host(runs_f5)
    i1_h = host(runs_i1)
    out = []
    for g in range(G):
        sl = slice(g * S, (g + 1) * S)
        out.append(dict(tensor=tensor_h[sl], score=score_h[sl],
                        runs_f5=f5_h[sl], runs_i1=i1_h[sl],
                        leftover=leftover, stats=STATS, seed=seed))
    return out


# ------------------------------------------------ batched preparation

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
    # pre-game Elo (spec ctx, else team-context store via game_pk) and
    # the actual fielders per side — A1 feature inputs
    ctx = spec.get("_ctx") or {}
    elo_a = ctx.get("away_elo", np.nan)
    elo_h = ctx.get("home_elo", np.nan)
    tcx = tuple(ctx.get(f"{s}_{c}", np.nan)
                for c in ("travel_km", "tz_shift", "day_after_night")
                for s in ("away", "home"))
    if not (elo_a == elo_a) and spec.get("game_pk") is not None \
            and P.fstores.get("teamctx") is not None:
        tcm = getattr(P, "_elo_by_pk", None)
        if tcm is None:
            tc = P.fstores["teamctx"]
            tcm = {int(k): (float(av), float(hv)) for k, av, hv in
                   zip(tc.GamePk, tc.away_elo, tc.home_elo)}
            P._elo_by_pk = tcm
            P._tcx_by_pk = {
                int(k): tuple(map(float, v)) for k, *v in zip(
                    tc.GamePk,
                    tc.away_travel_km, tc.home_travel_km,
                    tc.away_tz_shift, tc.home_tz_shift,
                    tc.away_day_after_night, tc.home_day_after_night)}
        elo_a, elo_h = tcm.get(int(spec["game_pk"]), (np.nan, np.nan))
        tcx = getattr(P, "_tcx_by_pk", {}).get(
            int(spec["game_pk"]), (np.nan,) * 6)
    fm_side = {away: P._fielder_map(bat_away),
               home: P._fielder_map(bat_home)}
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
                pit_throws=pit_throws, slot_is_c=slot_is_c, meta=meta,
                elo=(elo_a, elo_h), tcx=tcx, fm_side=fm_side)


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
            fm = rv["fm_side"][p_team]
            for brow in rv["bat_rows_all"]:
                bpid = players[brow]
                bat_is_home = (brow >= 9) if brow < 18 else \
                    (brow == rv["bench_rows"][1])
                stand = P._stand(bpid, pthrows)
                od_pid = (players[(brow + 1) % 9] if brow < 9 else
                          players[9 + (brow - 8) % 9] if brow < 18
                          else -1)
                elo_a, elo_h = rv["elo"]
                ta, th, za, zh, na_, nh = rv.get("tcx", (np.nan,) * 6)
                for tto in (1, 2, 3):
                    rows.append((date, season, bpid, ppid, stand,
                                 pthrows, od_pid, tto,
                                 int(bat_is_home), p_team,
                                 spec.get("venue") or "",
                                 spec.get("day_night") or "day",
                                 spec.get("temp"),
                                 spec.get("wind_speed"),
                                 spec.get("wind_dir") or "",
                                 spec.get("condition") or "",
                                 spec.get("humidity"),
                                 spec.get("pressure"),
                                 spec.get("hp_ump_id"),
                                 rest_by_pid[ppid],
                                 elo_h if bat_is_home else elo_a,
                                 elo_a if bat_is_home else elo_h,
                                 th if bat_is_home else ta,
                                 ta if bat_is_home else th,
                                 zh if bat_is_home else za,
                                 za if bat_is_home else zh,
                                 nh if bat_is_home else na_,
                                 na_ if bat_is_home else nh,
                                 fm[3], fm[4], fm[5], fm[6], fm[7],
                                 fm[8], fm[9]))
        blocks.append(rows)
    cols = ["Date", "Season", "BatterId", "PitcherId", "stand",
            "p_throws", "OnDeckId", "tto", "home_bat", "fld_team",
            "Venue",
            "DayNight", "Temp", "WindSpeed", "WindDir", "Condition",
            "Humidity", "Pressure", "HpUmpId", "rest_p",
            "b_elo", "p_elo", "b_travel_km", "p_travel_km",
            "b_tz_shift", "p_tz_shift", "b_day_after_night",
            "p_day_after_night", "fielder_3", "fielder_4", "fielder_5",
            "fielder_6", "fielder_7", "fielder_8", "fielder_9"]
    rdf = pd.DataFrame([r for b in blocks for r in b], columns=cols)
    rdf["Date"] = pd.to_datetime(rdf["Date"])
    return rdf


def _fill_avec(P, resolved, rdf, n_sims):
    """One assemble + one predict per model for the whole chunk, then
    scatter back into per-game avec/a2vec (a2vec is class-conditional
    under the contact tree)."""
    X, _ = F.assemble_features(rdf, P.fstores)
    p1, p2 = P._class_vecs(X)
    per_game = 18 * 20 * 3
    out = []
    for gi, rv in enumerate(resolved):
        n_players = rv["n_players"]
        avec = np.full((n_players, 20, 3, 8), np.nan)
        a2vec = np.full((n_players, 20, 3) + p2.shape[1:], np.nan)
        b1 = p1[gi * per_game:(gi + 1) * per_game].reshape(18, 20, 3, 8)
        b2 = p2[gi * per_game:(gi + 1) * per_game].reshape(
            (18, 20, 3) + p2.shape[1:])
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
        post = float(PR.spec_postseason(rv["spec"]))
        for si, prow in enumerate((18, 19)):
            ppid = rv["players"][prow]
            team = rv["away"] if si == 0 else rv["home"]
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
                il_ret30=il_ret30, outs_sd=outs_sd,
                pen_np3=P._pen_np3(team, d), post=post,
                team_hook=P._team_hook(team, d))))
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
        post = PR.spec_postseason(rv["spec"])
        spc = P.sb.get("speed_center", 27.3)
        n = 0
        for brow in rv["bat_rows_all"]:
            bpid = players[brow]
            for prow in rv["pit_rows"]:
                ppid = players[prow]
                fld = away if (prow == 18
                               or 20 <= prow < 20 + PR.MAX_PEN) else \
                    home
                lhp = float(str(P._throws.get(ppid, "R")) == "L")
                sspd = (spd.get(bpid, np.nan) if bpid >= 0 else np.nan)
                rows_a.append((
                    sspd, float(np.isnan(sspd)), (sspd - spc) ** 2,
                    sbr.get(ppid, np.nan), csr.get(ppid, np.nan),
                    team_pop(fld), np.nan, 1, 1.0, 1.0, era_new, lhp,
                    post))
                rows_s.append((
                    spd.get(bpid, np.nan), csr.get(ppid, np.nan),
                    team_pop(fld), np.nan, lhp, era_new, post))
                n += 1
        sizes.append(n)
    A = pd.DataFrame(rows_a, columns=["SprintSpeed", "speed_miss",
                                      "speed2", "sb_allowed_rate",
                                      "cs_rate", "PopTime", "CSAA",
                                      "outs", "outs1", "score_close",
                                      "era_new", "lhp",
                                      "post"])[P.sb["att_features"]]
    S2 = pd.DataFrame(rows_s, columns=["SprintSpeed", "cs_rate",
                                       "PopTime", "CSAA", "lhp",
                                       "era_new",
                                       "post"])[P.sb["suc_features"]]
    pa_all = P.sb["attempt"].predict_proba(A)[:, 1]
    ps_raw = P.sb["success"].predict_proba(S2)[:, 1]
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
        adj = P._sim_adjusters(rv["players"], rv["season"], rv["away"],
                               rv["home"], rv["date"],
                               rv["pen_rows_away"], rv["pen_rows_home"])
        P._apply_pen_fatigue(avec, adj["pen_fat_f"])
        lat = dict(P.latent)
        prep = S_.GamePrep(
            n_players=rv["n_players"], starters=[18, 19], avec=avec,
            a2vec=a2vec, haz_grid=haz, relief_exit=adj["relief_exit"],
            pen_order=pen_order, sb_att=att, sb_suc=suc,
            sb_state=P._sb_state(rv["season"]),
            patterns=P.patterns, latent=lat,
            bench_rows=rv["bench_rows"], part_haz=P.part_haz,
            bat_side=rv["bat_side"], pit_throws=rv["pit_throws"],
            slot_is_c=rv["slot_is_c"],
            run_z=adj["run_z"], dp_z=adj["dp_z"],
            arm_eff=adj["arm_eff"], stretch=adj["stretch"],
            stretch_z=adj["stretch_z"], pen_hi=adj["pen_hi"],
            pen_lo=adj["pen_lo"], pre_wp=adj["pre_wp"],
            pen_rank_cum=P._pen_rank_cum,
            pre_pk=P.preevents["pickoff_out_per_pa_runners_on"])
        out.append((prep, rv["meta"]))
    return out
