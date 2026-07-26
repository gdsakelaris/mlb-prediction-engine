"""Classic (sim.run) vs batched (sim_batch.run_batch) engine parity on
identical synthetic preps: the engines share constants but re-implement
the loop, so distributional agreement is the guard against the batched
rewrite silently diverging. GPU (cupy) parity runs when available."""
import numpy as np
import pytest

import sim as S
import synth


def _measure(res):
    t = np.asarray(res["tensor"], dtype=np.float64)
    sc = np.asarray(res["score"], dtype=np.float64)
    X = S.SIDX
    return dict(
        away=sc[:, 0], home=sc[:, 1], total=sc.sum(1),
        homewin=(sc[:, 1] > sc[:, 0]).astype(float),
        k=t[..., X["PK_"]].sum(1), hr=t[..., X["HR"]].sum(1),
        f5=np.asarray(res["runs_f5"], dtype=np.float64).sum(1),
        start_bf=t[:, 18, X["BF"]] + t[:, 19, X["BF"]],
        sb=t[..., X["SB"]].sum(1),
    )


def _assert_close(m1, m2, label):
    for k in m1:
        a, b = m1[k], m2[k]
        se = np.sqrt(a.var() / a.size + b.var() / b.size)
        diff = abs(a.mean() - b.mean())
        assert diff < max(4.5 * se, 1e-9), \
            f"{label}/{k}: {a.mean():.4f} vs {b.mean():.4f} " \
            f"(diff {diff:.4f}, 4.5se {4.5 * se:.4f})"


# per-family/per-market tail battery: the probabilities the engines
# actually SERVE (game_frame/build_bets price exactly these tail
# events), so parity is asserted on the betting quantities themselves,
# not just 9 slate-level means. Lines mirror the served ladders
# (predict.TOTAL_LINES / TEAM_TOTAL_LINES / K, Outs, ER, Hits, BB).
_TT_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
_TOT_LINES = (5.5, 6.5, 7.5, 8.5, 9.5, 10.5)
_PK_LINES = (2.5, 4.5, 6.5)
_POUT_LINES = (8.5, 11.5, 14.5, 17.5)
_PER_LINES = (1.5, 2.5, 3.5)
_PHA_LINES = (3.5, 5.5, 7.5)
_PBB_LINES = (0.5, 1.5)


def _market_battery(res):
    """{market: per-sim 0/1 indicator} for every served market family:
    ml, tot, tt, the starter ladders (pk/pout/per/pha/pbb), per-slot
    batter props (h/hr/tb/r/rbi/bb/bk/hrr/sb families), plus the
    distribution-shape events the mean lane can never see (walk-off
    frequency AND margin mix, extra innings, steal accounting)."""
    t = np.asarray(res["tensor"], dtype=np.float64)
    sc = np.asarray(res["score"], dtype=np.float64)
    X = S.SIDX
    ev = {}
    ev["ml/homewin"] = (sc[:, 1] > sc[:, 0]).astype(float)
    total = sc.sum(1)
    for x in _TOT_LINES:
        ev[f"tot/runs>{x}"] = (total > x).astype(float)
    for col, side in ((0, "away"), (1, "home")):
        for x in _TT_LINES:
            ev[f"tt/{side}>{x}"] = (sc[:, col] > x).astype(float)
    for row, tag in ((18, "a"), (19, "h")):
        for fam, stat, lines in (("pk", "PK_", _PK_LINES),
                                 ("pout", "OUTS", _POUT_LINES),
                                 ("per", "PER", _PER_LINES),
                                 ("pha", "PH", _PHA_LINES),
                                 ("pbb", "PBB", _PBB_LINES)):
            for x in lines:
                ev[f"{fam}/{tag}>{x}"] = \
                    (t[:, row, X[stat]] > x).astype(float)
    for row in range(18):
        sub = t[:, row]
        tb = (sub[:, X["B1"]] + 2 * sub[:, X["B2"]]
              + 3 * sub[:, X["B3"]] + 4 * sub[:, X["HR"]])
        hrr = sub[:, X["H"]] + sub[:, X["R"]] + sub[:, X["RBI"]]
        for fam, arr, x in (("h", sub[:, X["H"]], 0.5),
                            ("h", sub[:, X["H"]], 1.5),
                            ("hr", sub[:, X["HR"]], 0.5),
                            ("tb", tb, 1.5), ("tb", tb, 3.5),
                            ("r", sub[:, X["R"]], 0.5),
                            ("rbi", sub[:, X["RBI"]], 0.5),
                            ("bb", sub[:, X["BB"]], 0.5),
                            ("bk", sub[:, X["K"]], 0.5),
                            ("hrr", hrr, 1.5),
                            ("sb", sub[:, X["SB"]], 0.5)):
            ev[f"{fam}/b{row}>{x}"] = (arr > x).astype(float)
    # distribution shape: walk-off rate and margin mix (joint events, so
    # plain binomial bounds apply), extra innings, steal/CS accounting
    away_outs = t[:, synth.AWAY_STAFF, X["OUTS"]].sum(1)
    home_outs = t[:, synth.HOME_STAFF, X["OUTS"]].sum(1)
    walkoff = (sc[:, 1] > sc[:, 0]) & (away_outs % 3 != 0)
    ev["shape/walkoff"] = walkoff.astype(float)
    margin = sc[:, 1] - sc[:, 0]
    for k in (1, 2, 3, 4):
        ev[f"shape/walkoff_m{k}"] = (walkoff & (margin == k)).astype(float)
    ev["shape/extras"] = (home_outs > 27).astype(float)
    ev["shape/sb1"] = (t[..., X["SB"]].sum(1) > 0.5).astype(float)
    ev["shape/cs1"] = (t[..., X["CS"]].sum(1) > 0.5).astype(float)
    return ev


def _assert_battery_close(e1, e2, label):
    """Two-sample binomial bound per market: |p1-p2| < 5.0*SE. At
    n=20,000/engine that is ~2.5 points at p=0.5, ~1.5 at p=0.1 and
    ~0.7 at p=0.02 — the served-probability scale where divergence is
    money-relevant. z=5.0 (per-test alpha 5.7e-7) keeps the familywise
    false-alarm over the ~250 comparisons/game near 1.5e-4 under the
    null, and the seeds are fixed anyway; anything caught here is a
    real engine drift, not noise. The 1e-9 floor covers degenerate
    all-zero markets (e.g. sb in the quiet prep) where SE collapses."""
    for k in e1:
        a, b = e1[k], e2[k]
        p1, p2 = a.mean(), b.mean()
        se = np.sqrt(p1 * (1 - p1) / a.size + p2 * (1 - p2) / b.size)
        diff = abs(p1 - p2)
        assert diff < max(5.0 * se, 1e-9), \
            f"{label}/{k}: {p1:.4f} vs {p2:.4f} " \
            f"(diff {diff:.4f}, 5.0se {5.0 * se:.4f})"


LAT = dict(mu_env=0.05, sigma_env=0.06, sigma_offense=0.0,
           sigma_pitcher=0.0, sigma_hr=0.15, sigma_k=0.2)


def _preps():
    """Latent is per-game since the B11 fix (2026-07-20) — BatchPrep
    stacks each prep's latent params. patterns/part_haz/stretch are
    still read from preps[0] only (shared by prepare_games contract)."""
    quiet = synth.make_prep(latent=dict(LAT))
    busy = synth.make_prep(
        hazard=0.04, relief_exit=0.3, sb_att=0.012, sb_suc=0.72,
        pre_pk=0.002, pre_wp=0.004, latent=dict(LAT))
    return [quiet, busy]


def _run_parity(backend):
    import sim_batch as SB
    preps = _preps()
    # n=20,000 (raised from 6,000 with the per-market battery): the
    # slate-mean lane keeps its 4.5*SE gate — now ~0.2 runs on a total
    # instead of ~0.37 — and the battery bound above is what makes the
    # "probabilistically equivalent" contract mean served probabilities,
    # not just means
    n = 20000
    seasons = [2024, 2024]
    batch = SB.run_batch(preps, n_sims=n, seed=101, seasons=seasons,
                         is_dh=[False, False], backend=backend)
    for i, prep in enumerate(preps):
        classic = S.run(prep, n_sims=n, seed=202 + i, season=seasons[i])
        synth.ledger_checks(classic)
        synth.ledger_checks(batch[i])
        _assert_close(_measure(classic), _measure(batch[i]),
                      f"game{i}[{backend}]")
        _assert_battery_close(_market_battery(classic),
                              _market_battery(batch[i]),
                              f"game{i}[{backend}]")


def test_batch_cpu_matches_classic():
    _run_parity("cpu")


def test_batch_gpu_matches_classic():
    pytest.importorskip("cupy")
    _run_parity("gpu")


# full-feature latent: nonzero offense/pitcher sigmas too — LAT keeps
# them 0 (matching the current production latent.json), which left the
# per-team latent lanes untested; backtest.py grids nonzero values, so
# the next refit could activate untested twin code
LAT_FULL = dict(mu_env=0.05, sigma_env=0.06, sigma_offense=0.10,
                sigma_pitcher=0.12, sigma_hr=0.15, sigma_k=0.2)


def _run_parity_full(backend):
    """Parity with every twin-implemented subsystem ACTIVE (tilts,
    stretch, participation pulls, sb_state, catcher slots, leverage-pen
    split — synth._activate_full). The quiet/busy preps zero all of
    these, so this is the lane that actually guards the twin code the
    B11 class regressed."""
    import sim_batch as SB
    n = 20000
    preps = [synth.make_prep(sb_att=0.008, sb_suc=0.74,
                             latent=dict(LAT_FULL), full=True),
             synth.make_prep(hazard=0.04, relief_exit=0.3,
                             sb_att=0.012, sb_suc=0.72, pre_pk=0.002,
                             pre_wp=0.004, latent=dict(LAT_FULL),
                             full=True)]
    batch = SB.run_batch(preps, n_sims=n, seed=301,
                         seasons=[2024, 2024], is_dh=[False, False],
                         backend=backend)
    for i, prep in enumerate(preps):
        classic = S.run(prep, n_sims=n, seed=402 + i, season=2024)
        synth.ledger_checks(classic)
        synth.ledger_checks(batch[i])
        _assert_close(_measure(classic), _measure(batch[i]),
                      f"full{i}[{backend}]")
        _assert_battery_close(_market_battery(classic),
                              _market_battery(batch[i]),
                              f"full{i}[{backend}]")


def test_batch_cpu_matches_classic_full_features():
    _run_parity_full("cpu")


def test_batch_gpu_matches_classic_full_features():
    pytest.importorskip("cupy")
    _run_parity_full("gpu")


def test_batch_latent_is_per_game():
    """B11 FIXED 2026-07-20: BatchPrep stacks per-game latent params, so
    a mixed-latent batch applies each game's own latent (was: game 0's
    latent silently applied to every game)."""
    import sim_batch as SB
    zero = synth.make_prep()                      # all-zero latent
    hot = synth.make_prep(latent=dict(mu_env=0.30, sigma_env=0.0,
                                      sigma_offense=0.0,
                                      sigma_pitcher=0.0, sigma_hr=0.0,
                                      sigma_k=0.0))
    n = 4000
    out = SB.run_batch([zero, hot], n_sims=n, seed=55,
                       seasons=[2024, 2024], is_dh=[False, False],
                       backend="cpu")
    zero_classic = S.run(zero, n_sims=n, seed=56, season=2024)
    hot_classic = S.run(hot, n_sims=n, seed=57, season=2024)
    b0 = out[0]["score"].sum(1).mean()
    b1 = out[1]["score"].sum(1).mean()
    zc = zero_classic["score"].sum(1).mean()
    hc = hot_classic["score"].sum(1).mean()
    assert b1 - b0 > 1.0, "hot game's mu_env=0.3 must lift its totals"
    assert abs(b0 - zc) < 0.4, "zero-latent batch game drifted"
    assert abs(b1 - hc) < 0.4, "hot-latent batch game != hot classic"


def test_per_pitcher_exit_tables():
    """B4 (2026-07-20): a 2-D relief_exit [n_players, 11] is honored
    per reliever in both engines — an always-exit first arm hands off
    to a never-exit second arm, later arms never appear."""
    import sim_batch as SB
    re = np.zeros((synth.NP_, 11), dtype=np.float32)
    re[20] = 1.0
    re[28] = 1.0
    prep = synth.make_prep(hazard=1.0, relief_exit=re)
    n = 800
    res = S.run(prep, n_sims=n, seed=11, season=2024)
    X = S.SIDX
    outs = np.asarray(res["tensor"])[..., X["OUTS"]]
    assert (outs[:, 22:28] == 0).all() and (outs[:, 30:36] == 0).all()
    assert outs[:, 21].mean() > outs[:, 20].mean() + 5
    assert outs[:, 29].mean() > outs[:, 28].mean() + 5
    out = SB.run_batch([prep, prep], n_sims=n, seed=12,
                       seasons=[2024, 2024], is_dh=[False, False],
                       backend="cpu")
    for g in out:
        ob = np.asarray(g["tensor"])[..., X["OUTS"]]
        assert (ob[:, 22:28] == 0).all() and (ob[:, 30:36] == 0).all()
        assert abs(ob[:, 20].mean() - outs[:, 20].mean()) < 0.6
        assert abs(ob[:, 21].mean() - outs[:, 21].mean()) < 1.5


def test_pen_rank_pmf_pick():
    """B6: with pen_rank_cum forcing rank 1 (0-based) always, every
    entry picks the SECOND still-available arm — the first order slot
    is never used. Same behavior in both engines."""
    import sim_batch as SB
    pmf = np.zeros((2, 8))
    pmf[:, 1] = 1.0
    cum = np.cumsum(pmf, axis=1)
    # relief_exit=0: exactly ONE reliever enters per side (starter
    # hazard 1.0), so the sampled rank is observable exactly
    prep = synth.make_prep(hazard=1.0, relief_exit=0.0)
    prep.pen_rank_cum = cum
    n = 500
    res = S.run(prep, n_sims=n, seed=21, season=2024)
    X = S.SIDX
    outs = np.asarray(res["tensor"])[..., X["OUTS"]]
    assert outs[:, 20].sum() == 0 and outs[:, 28].sum() == 0
    assert (outs[:, 21] > 0).all() and (outs[:, 29] > 0).all()
    assert outs[:, 22:28].sum() == 0 and outs[:, 30:36].sum() == 0
    out = SB.run_batch([prep, prep], n_sims=n, seed=22,
                       seasons=[2024, 2024], is_dh=[False, False],
                       backend="cpu")
    for g in out:
        ob = np.asarray(g["tensor"])[..., X["OUTS"]]
        assert ob[:, 20].sum() == 0 and ob[:, 28].sum() == 0
        assert (ob[:, 21] > 0).all() and (ob[:, 29] > 0).all()
        assert ob[:, 22:28].sum() == 0 and ob[:, 30:36].sum() == 0


def test_batch_reproducible():
    import sim_batch as SB
    preps = _preps()
    r1 = SB.run_batch(preps, n_sims=300, seed=7, seasons=[2024, 2024],
                      is_dh=[False, False], backend="cpu")
    r2 = SB.run_batch(preps, n_sims=300, seed=7, seasons=[2024, 2024],
                      is_dh=[False, False], backend="cpu")
    for a, b in zip(r1, r2):
        assert (a["tensor"] == b["tensor"]).all()
        assert (a["score"] == b["score"]).all()


def test_batch_seven_inning_rule():
    import sim_batch as SB
    prep = synth.make_prep(away_vec=synth.class_vec(K=1.0))
    out = SB.run_batch([prep, prep], n_sims=1000, seed=8,
                       seasons=[2021, 2024], is_dh=[True, False],
                       backend="cpu")
    pa7 = out[0]["tensor"][:, synth.AWAY_BATS, S.SIDX["PA"]].sum(1)
    pa9 = out[1]["tensor"][:, synth.AWAY_BATS, S.SIDX["PA"]].sum(1)
    assert pa7.min() == 21 and (pa7 == 21).mean() > 0.7
    assert pa9.min() == 27 and (pa9 == 27).mean() > 0.8
