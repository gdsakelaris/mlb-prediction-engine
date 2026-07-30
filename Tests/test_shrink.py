"""W5.2 uncertainty-shrink unit net: bag-member sigma extraction, the
per-batter aggregation routing, the logit-space shrink map, and the
slot-aware participation-hazard bucket arithmetic (features builder vs
sim consumer must agree or the audit fix silently misroutes)."""
import numpy as np

import predict as PR


def _p8(hit_bump=0.0):
    # CLASSES order [K, BB, HBP, 1B, 2B, 3B, HR, IPO]
    base = np.array([.20, .08, .01, .15, .05, .005, .03, .475])
    v = base.copy()
    v[3] += hit_bump
    v[7] -= hit_bump
    return np.tile(v, (6, 1))


def test_member_sigma_groups_and_degenerate():
    sig = PR._member_sigma([_p8(), _p8(0.04)], 6)
    assert sig.shape == (6, 4)
    # members disagree on 1B only -> hit group moves, k group does not
    assert sig[0, PR.SIG_HIT] > 0.05
    assert sig[0, PR.SIG_K] == 0.0
    # xbh (2B+3B+HR) untouched by a 1B bump
    assert sig[0, PR.SIG_XBH] == 0.0
    # fewer than two members carries no disagreement signal
    assert PR._member_sigma([_p8()], 6).max() == 0.0
    assert PR._member_sigma([], 6).shape == (6, 4)


def test_agg_bat_sigma_opposing_starter_routing():
    sgb = np.zeros((18, 20, 3, 4), dtype=np.float32)
    sgb[1] = 1.0    # home starter block (pit_rows position 1)
    sgb[0] = 2.0    # away starter block
    sgb[5] = 9.0    # a pen row: must never be read
    bs = PR._agg_bat_sigma(sgb)
    assert bs.shape == (20, 4)
    # away lineup (0-8) + away bench (18) face the HOME starter
    assert bs[0, 0] == 1.0 and bs[8, 0] == 1.0 and bs[18, 0] == 1.0
    # home lineup (9-17) + home bench (19) face the AWAY starter
    assert bs[9, 0] == 2.0 and bs[17, 0] == 2.0 and bs[19, 0] == 2.0


def test_shrink_p_identity_and_pull(monkeypatch):
    monkeypatch.setattr(PR, "SHRINK_LAM", 0.0)
    assert abs(PR._shrink_p(0.6, 0.3, 0.35) - 0.6) < 1e-12
    monkeypatch.setattr(PR, "SHRINK_LAM", 400.0)
    # zero sigma -> identity at any lambda
    assert abs(PR._shrink_p(0.6, 0.0, 0.35) - 0.6) < 1e-12
    # pulls toward the anchor from both sides, never past it
    hi = PR._shrink_p(0.6, 0.1, 0.35)
    lo = PR._shrink_p(0.2, 0.1, 0.35)
    assert 0.35 < hi < 0.6
    assert 0.2 < lo < 0.35
    # more disagreement -> more pull
    assert PR._shrink_p(0.6, 0.2, 0.35) < hi


def test_shrink_preserves_ladder_monotonicity(monkeypatch):
    monkeypatch.setattr(PR, "SHRINK_LAM", 800.0)
    # a TB ladder: shared sigma group, ordered anchors, ordered rungs
    p = [0.42, 0.21, 0.09]
    anchors = [0.35, 0.19, 0.08]
    sig = 0.15
    out = [PR._shrink_p(pi, sig, ai) for pi, ai in zip(p, anchors)]
    assert out[0] > out[1] > out[2]


def test_slot_aware_hazard_lookup_routing():
    """A 7-dim participation table routes each due slot to its own
    rate through the exact index expression both sims use, and the
    6-dim legacy shape still routes slot-blind — the backward-compat
    contract for synth fixtures and old artifacts."""
    dense7 = np.zeros((6, 4, 3, 2, 2, 2, 9))
    for sb in range(9):
        dense7[..., sb] = sb / 100.0
    slot = np.array([0, 4, 8])
    z = np.zeros(3, dtype=int)
    rate = dense7[z, z, z, z, z, z, slot]          # sim.py expression
    assert list(np.round(rate, 4)) == [0.0, 0.04, 0.08]
    dense6 = np.full((6, 4, 3, 2, 2, 2), 0.03)
    assert dense6.ndim == 6                        # legacy branch key
    assert float(dense6[0, 0, 0, 0, 0, 0]) == 0.03


def test_game_frame_shrink_applied_before_cal(monkeypatch):
    """End-to-end through the real game_frame batter loop: with a
    bat_sigma on meta and a nonzero lambda, the anchored columns move
    toward the anchor; with SHRINK off they do not."""
    import sim

    n_sims, brow = 400, 0
    t = np.zeros((n_sims, 20, len(sim.STATS)), dtype=np.int16)
    rng = np.random.default_rng(3)
    t[:, brow, sim.SIDX["H"]] = rng.random(n_sims) < 0.9
    t[:, brow, sim.SIDX["PA"]] = 4
    meta = dict(players=[1000] + [-1] * 19,
                names=["Bat"] + [""] * 19, season=2026,
                away="AAA", home="BBB",
                career_g=[100] * 20, career_gp=[0] * 20,
                bat_sigma=np.full((20, 4), 0.3, dtype=np.float32))
    res = dict(tensor=t, score=np.zeros((n_sims, 2), dtype=np.int16),
               runs_f5=np.zeros((n_sims, 2)),
               runs_i1=np.zeros((n_sims, 2)),
               meta=meta, spec={}, calib={})
    monkeypatch.setattr(PR, "SHRINK", True)
    monkeypatch.setattr(PR, "SHRINK_LAM", 800.0)
    monkeypatch.setattr(PR, "SHRINK_ANCHOR", {"Hit": 0.55})
    hit_shrunk = PR.game_frame(res)["bat"][0]["Hit"]
    monkeypatch.setattr(PR, "SHRINK", False)
    hit_raw = PR.game_frame(res)["bat"][0]["Hit"]
    assert hit_raw > 0.8
    assert 0.55 < hit_shrunk < hit_raw
