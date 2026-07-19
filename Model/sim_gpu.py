"""Batched Monte Carlo engine: many games x many sims in ONE tensor.

Same game logic as sim.py (single source of truth for rules and
patterns — constants are imported), restructured for throughput:

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

from pathlib import Path  # noqa: F401  (parity with sim.py imports)

import numpy as np

import sim as S_

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
    """sim.load_patterns dict -> (cum [960, K], arr [960, K, 8]) with
    every possible key resolved (fallback chain applied up front) and
    rows padded with cum=2.0 so a comparison-sum sample never lands on
    padding."""
    cache = {}
    kmax = max(len(c) for c, _ in pat.values())
    cum = np.full((N_KEYS, kmax), 2.0, dtype=np.float32)
    arr = np.zeros((N_KEYS, kmax, 8), dtype=np.int8)
    for k in range(N_KEYS):
        entry = pat.get(k)
        if entry is None:
            entry = S_._fallback_pattern(k, pat, cache)
        c, a = entry
        n = len(c)
        cum[k, :n] = c
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
        self.relief = xp.asarray(
            np.stack([p.relief_exit for p in preps]), dtype=f32)
        # pen_order in GamePrep is [n_sims, 2, P] broadcast from one
        # [1, 2, P] row — keep the per-game row only
        self.pen = xp.asarray(
            np.stack([np.asarray(p.pen_order[0]) for p in preps]),
            dtype=np.int16)
        self.sb_att = xp.asarray(
            np.stack([p.sb_att for p in preps]), dtype=f32)
        self.sb_suc = xp.asarray(
            np.stack([p.sb_suc for p in preps]), dtype=f32)
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
        self.pre_wp = xp.asarray(
            np.array([p.pre_wp for p in preps]), dtype=f32)
        part = preps[0].part_haz
        self.part = (xp.asarray(part, dtype=f32)
                     if part is not None else None)
        self.latent = preps[0].latent
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


def _normal(xp, rng, mu, sigma, size):
    return mu + sigma * rng.standard_normal(size, dtype=xp.float32)


def run_batch(preps, n_sims=4000, seed=1, seasons=None, is_dh=None,
              backend="gpu"):
    """Simulate G games x n_sims each; returns a list of per-game dicts
    shaped exactly like sim.run's output."""
    if backend == "gpu":
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

    sig = bp.latent
    z_env = _normal(xp, rng, float(sig.get("mu_env", 0.0)),
                    float(sig.get("sigma_env", 0.0)), N)
    z_hr = _normal(xp, rng, 0.0, float(sig.get("sigma_hr", 0.0)), N)
    z_pit = _normal(xp, rng, 0.0,
                    float(sig.get("sigma_pitcher", 0.0)), (N, 2))
    z_off = _normal(xp, rng, 0.0,
                    float(sig.get("sigma_offense", 0.0)), (N, 2))

    reg_n = bp.reg[gidx]                 # per-row regulation innings
    ghost_n = bp.ghost[gidx]

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
                               xp.clip(stint_outs[ridx, rfld],
                                       0, 10).astype(xp.int32)]
            leave = rng.random(int(ridx.size), dtype=xp.float32) < exit_p
            if bool(leave.any()):
                lidx, lfld = ridx[leave], rfld[leave]
                nxt = bp.pen[gidx[lidx], lfld,
                             pen_next[lidx, lfld].astype(xp.int32)]
                has = nxt >= 0
                cur_pit[lidx[has], lfld[has]] = nxt[has]
                pen_next[lidx, lfld] = xp.minimum(
                    pen_next[lidx, lfld] + 1, bp.pen.shape[2] - 1)
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
        wp = any_on & (u < bp.pre_wp[ga]) & (outs[a] < 3)
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
                nxt = bp.pen[gidx[oidx], oft,
                             pen_next[oidx, oft].astype(xp.int32)]
                has = nxt >= 0
                cur_pit[oidx[has], oft[has]] = nxt[has]
                pen_next[oidx, oft] = xp.minimum(
                    pen_next[oidx, oft] + 1, bp.pen.shape[2] - 1)
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
        P = P * mult
        P /= P.sum(axis=1, keepdims=True)
        cum = xp.cumsum(P, axis=1)
        cum[:, -1] = 1.0 + 1e-9
        u = rng.random(int(a.size), dtype=xp.float32)
        cls = (u[:, None] > cum).sum(axis=1).astype(xp.int32)

        # ---- 5. batted-ball type when in play
        bb = xp.zeros(int(a.size), dtype=xp.int32)
        inplay = ((cls == CI["1B"]) | (cls == CI["2B"])
                  | (cls == CI["3B"]) | (cls == CI["HR"])
                  | (cls == CI["IPO"]))
        if bool(inplay.any()):
            P2 = bp.a2vec[ga[inplay], pit[inplay],
                          bat_axis(batter_row[inplay]), tto[inplay]]
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
        u = rng.random(int(a.size), dtype=xp.float32)
        pick = _sample_dense(xp, bp.pat_cum[key], u)
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
