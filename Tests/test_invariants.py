"""Ledger and conservation invariants of the classic engine on synthetic
games — every identity in synth.ledger_checks over full-feature sims
(hazard, pen, steals, pre-events, latent sigmas all active)."""
import numpy as np

import sim as S
import synth


def _busy_prep():
    return synth.make_prep(
        hazard=0.04, relief_exit=0.3, sb_att=0.012, sb_suc=0.72,
        pre_pk=0.002, pre_wp=0.004,
        latent=dict(mu_env=0.0, sigma_env=0.06, sigma_offense=0.0,
                    sigma_pitcher=0.0, sigma_hr=0.15, sigma_k=0.2))


def test_ledger_modern_rules():
    res = S.run(_busy_prep(), n_sims=4000, seed=11, season=2024)
    info = synth.ledger_checks(res, reg=9)
    # a 4k-sim season must contain walk-offs; the ledger identities
    # above are the regression net for the capped-walk-off R bug
    assert info["walkoffs"] > 0


def test_ledger_seven_inning_dh():
    res = S.run(_busy_prep(), n_sims=2000, seed=12, season=2021,
                is_dh_game=True)
    synth.ledger_checks(res, reg=7)


def test_ledger_pre_ghost_era():
    res = S.run(_busy_prep(), n_sims=2000, seed=13, season=2019)
    synth.ledger_checks(res, reg=9)


def test_bench_never_bats_without_participation_hazard():
    res = S.run(_busy_prep(), n_sims=1000, seed=14, season=2024)
    t = res["tensor"]
    assert (t[:, 36:, S.SIDX["PA"]] == 0).all()


def test_reproducibility():
    p = _busy_prep()
    r1 = S.run(p, n_sims=500, seed=42, season=2024)
    r2 = S.run(p, n_sims=500, seed=42, season=2024)
    assert (r1["tensor"] == r2["tensor"]).all()
    assert (r1["score"] == r2["score"]).all()
    r3 = S.run(p, n_sims=500, seed=43, season=2024)
    assert (r1["score"] != r3["score"]).any()


def test_score_totals_match_run_ledger_globally():
    res = S.run(_busy_prep(), n_sims=2000, seed=15, season=2024)
    t, sc = res["tensor"], res["score"]
    assert t[..., S.SIDX["R"]].sum() == sc.sum()
    # every pitcher out was recorded to a real pitching row
    assert t[..., S.SIDX["OUTS"]].sum() \
        == t[:, synth.PITCHER_ROWS, S.SIDX["OUTS"]].sum()
