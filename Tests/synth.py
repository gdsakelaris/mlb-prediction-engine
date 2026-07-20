"""Synthetic GamePrep builders: self-contained engine inputs.

Row layout mirrors predict.prepare_game exactly:
  0-8   away lineup          9-17  home lineup
  18/19 away/home starter    20-27 away pen   28-35 home pen
  36/37 away/home bench
so tests exercise the real index arithmetic (bat_axis, pen slot math,
bench substitution) without loading any artifact.

The pattern bank generated here is a *coherent* miniature of the real
pattern_table: every row's destination codes, outs_added and runs agree
with the base-state key (runs == number of dests hitting 4), so the
engine's ledger identities must hold exactly if the engine is correct.
"""
import numpy as np

import sim as S

NP_ = 38
AWAY_BATS = list(range(9)) + [36]
HOME_BATS = list(range(9, 18)) + [37]
AWAY_STAFF = [18] + list(range(20, 28))
HOME_STAFF = [19] + list(range(28, 36))
PITCHER_ROWS = list(range(18, 36))
NBAT = len(S.BAT_STATS)

# a "realistic" 8-class PA distribution (order = sim.CLASSES)
REAL8 = np.array([0.22, 0.08, 0.010, 0.150, 0.045, 0.004, 0.028, 0.463])

# batted-ball mix per outcome class [GB, FB, LD, PU]
BBMIX = {
    "1B": [0.55, 0.06, 0.38, 0.01],
    "2B": [0.20, 0.28, 0.52, 0.00],
    "3B": [0.02, 0.38, 0.60, 0.00],
    "HR": [0.00, 0.76, 0.24, 0.00],
    "IPO": [0.42, 0.34, 0.11, 0.13],
}


def class_vec(**over):
    """8-class prob vector; kwargs override by class name ("B1" for 1B)."""
    alias = {"B1": "1B", "B2": "2B", "B3": "3B"}
    if over:
        v = np.zeros(S.NCLS)
        for k, p in over.items():
            v[S.CI[alias.get(k, k)]] = p
        rem = 1.0 - v.sum()
        assert rem >= -1e-9
        v[S.CI["IPO"]] += max(rem, 0.0)
    else:
        v = REAL8.copy()
    return v / v.sum()


def _forced_walk(state):
    d = [1, 9, 9, 9]
    runs = 0
    if state & 1:
        d[1] = 2
        if state & 2:
            d[2] = 3
            if state & 4:
                d[3] = 4
                runs = 1
    return d, runs


def _hit_dests(cls, state):
    if cls == S.CI["1B"]:
        d = [1, 2 if state & 1 else 9, 3 if state & 2 else 9,
             4 if state & 4 else 9]
    elif cls == S.CI["2B"]:
        d = [2, 3 if state & 1 else 9, 4 if state & 2 else 9,
             4 if state & 4 else 9]
    elif cls == S.CI["3B"]:
        d = [3, 4 if state & 1 else 9, 4 if state & 2 else 9,
             4 if state & 4 else 9]
    else:                                            # HR
        d = [4, 4 if state & 1 else 9, 4 if state & 2 else 9,
             4 if state & 4 else 9]
    return d, sum(1 for x in d if x == 4)


def make_bank():
    """Full coherent pattern bank keyed like pattern_table."""
    bank = {}

    def put(cls, bb, state, outs, rows):
        p = np.array([r[-1] for r in rows], dtype=float)
        p /= p.sum()
        arr = np.array([r[:-1] for r in rows], dtype=np.int8)
        # coherence guard: runs column == number of scored dests
        for r in arr:
            assert r[5] == int((r[:4] == 4).sum()), (cls, bb, state, r)
        bank[S.pattern_key(cls, bb, state, outs)] = (
            np.tile(np.cumsum(p)[None, :], (S.NB, 1)), arr)

    for state in range(8):
        for outs in range(3):
            put(S.CI["K"], 0, state, outs,
                [[0, 9, 9, 9, 1, 0, 0, 0, 1.0]])
            d, r = _forced_walk(state)
            for cls in (S.CI["BB"], S.CI["HBP"]):
                put(cls, 0, state, outs, [d + [0, r, r, r, 1.0]])
            for cname in ("1B", "2B", "3B", "HR"):
                cls = S.CI[cname]
                d, r = _hit_dests(cls, state)
                for bb in (1, 2, 3, 4):
                    put(cls, bb, state, outs, [d + [0, r, r, r, 1.0]])
            for bb in (1, 2, 3, 4):
                rows = [[0, 9, 9, 9, 1, 0, 0, 0, 0.65]]
                if bb == 1 and (state & 1) and outs < 2:
                    rows.append([0, 0, 9, 9, 2, 0, 0, 0, 0.35])
                put(S.CI["IPO"], bb, state, outs, rows)
    return bank


def make_prep(away_vec=None, home_vec=None, hazard=0.0, relief_exit=0.0,
              sb_att=0.0, sb_suc=0.75, pre_pk=0.0, pre_wp=0.0,
              pen_empty=False, pit_throws=None, bat_side=None,
              latent=None, n_sims_pen=1):
    """Symmetric-by-default synthetic GamePrep. away_vec/home_vec are
    8-class vectors applied to that team's batter axes (incl. bench)."""
    away_vec = class_vec() if away_vec is None else away_vec
    home_vec = class_vec() if home_vec is None else home_vec

    avec = np.zeros((NP_, 20, 3, S.NCLS), dtype=np.float32)
    avec[:, list(range(9)) + [18]] = away_vec.astype(np.float32)
    avec[:, list(range(9, 18)) + [19]] = home_vec.astype(np.float32)

    a2vec = np.full((NP_, 20, 3, S.NCLS, 4), 0.25, dtype=np.float32)
    for cname, mix in BBMIX.items():
        a2vec[:, :, :, S.CI[cname]] = np.asarray(mix, dtype=np.float32)

    pen_rows = np.array([[*range(20, 28)], [*range(28, 36)]],
                        dtype=np.int16)
    if pen_empty:
        pen_hi = pen_lo = -np.ones((2, 8), dtype=np.int16)
    else:
        pen_hi = pen_lo = pen_rows

    return S.GamePrep(
        n_players=NP_,
        starters=[18, 19],
        bench_rows=[36, 37],
        avec=avec,
        a2vec=a2vec,
        haz_grid=np.full((2, 41, 11), hazard, dtype=np.float32),
        # scalar -> league-style [11]; arrays pass through (a 2-D
        # [n_players, 11] exercises the B4 per-pitcher exit path)
        relief_exit=(np.asarray(relief_exit, dtype=np.float32)
                     if np.ndim(relief_exit)
                     else np.full(11, relief_exit, dtype=np.float32)),
        pen_order=np.tile(pen_rows[None], (n_sims_pen, 1, 1)),
        pen_hi=pen_hi.copy(),
        pen_lo=pen_lo.copy(),
        sb_att=np.full((NP_, NP_), sb_att, dtype=np.float32),
        sb_suc=np.full((NP_, NP_), sb_suc, dtype=np.float32),
        sb_state=None,
        pre_pk=float(pre_pk),
        pre_wp=np.full(2, pre_wp, dtype=np.float32),
        latent=latent or dict(mu_env=0.0, sigma_env=0.0,
                              sigma_offense=0.0, sigma_pitcher=0.0,
                              sigma_hr=0.0, sigma_k=0.0),
        patterns=make_bank(),
        part_haz=None,
        bat_side=(np.ones(18, dtype=np.int8) if bat_side is None
                  else np.asarray(bat_side, dtype=np.int8)),
        pit_throws=(np.ones(NP_, dtype=np.int8) if pit_throws is None
                    else np.asarray(pit_throws, dtype=np.int8)),
        slot_is_c=np.zeros(18, dtype=np.int8),
        run_z=np.zeros(NP_, dtype=np.float32),
        dp_z=np.zeros(NP_, dtype=np.float32),
        arm_eff=np.zeros(2, dtype=np.float32),
        stretch=None,
        stretch_z=np.zeros(NP_, dtype=np.float32),
    )


def ledger_checks(res, reg=9):
    """Every conservation identity the engine must satisfy per sim.
    This is the regression net for the 2026-07-19 walk-off ledger bug
    (player-R exceeded the scoreboard in capped walk-offs)."""
    t = np.asarray(res["tensor"])
    sc = np.asarray(res["score"]).astype(np.int32)
    f5 = np.asarray(res["runs_f5"]).astype(np.int32)
    i1 = np.asarray(res["runs_i1"]).astype(np.int32)
    X = S.SIDX

    assert res["leftover"] == 0
    assert (t >= 0).all()

    # stat blocks live on the right rows only
    assert (t[:, PITCHER_ROWS, :NBAT] == 0).all(), "pitchers batted"
    bat_rows = AWAY_BATS + HOME_BATS
    assert (t[:, bat_rows, NBAT:] == 0).all(), "batters pitched"

    # scoreboard == player run ledger, per team
    for rows, side in ((AWAY_BATS, 0), (HOME_BATS, 1)):
        assert (t[:, rows, X["R"]].sum(1) == sc[:, side]).all()
        assert (t[:, rows, X["RBI"]].sum(1) <= sc[:, side]).all()

    # pitching staff ledgers vs the opposing scoreboard
    for staff, opp in ((HOME_STAFF, 0), (AWAY_STAFF, 1)):
        assert (t[:, staff, X["PR"]].sum(1) == sc[:, opp]).all()
        bat = AWAY_BATS if opp == 0 else HOME_BATS
        assert (t[:, staff, X["BF"]].sum(1)
                == t[:, bat, X["PA"]].sum(1)).all()
    assert (t[..., X["PER"]] <= t[..., X["PR"]]).all()

    # batter internal identities
    hsum = (t[..., X["B1"]] + t[..., X["B2"]] + t[..., X["B3"]]
            + t[..., X["HR"]])
    assert (t[..., X["H"]] == hsum).all()
    reach = (t[..., X["H"]] + t[..., X["BB"]] + t[..., X["HBP"]]
             + t[..., X["K"]])
    assert (t[..., X["PA"]] >= reach).all()

    # inning accounting: tops always complete; bottoms complete
    # whenever the away team won
    home_outs = t[:, HOME_STAFF, X["OUTS"]].sum(1)
    away_outs = t[:, AWAY_STAFF, X["OUTS"]].sum(1)
    assert (home_outs % 3 == 0).all(), "game ended mid-top-inning"
    away_won = sc[:, 0] > sc[:, 1]
    assert (away_outs[away_won] % 3 == 0).all()
    assert (away_outs[away_won] >= 3 * reg).all()

    # a home win that ended mid-inning is a walk-off: margin 1-4
    walkoff = (sc[:, 1] > sc[:, 0]) & (away_outs % 3 != 0)
    margin = (sc[:, 1] - sc[:, 0])[walkoff]
    assert (margin >= 1).all() and (margin <= 4).all()

    # partial-game run splits nest inside the final score
    assert (f5 <= sc).all() and (i1 <= f5).all()
    return dict(walkoffs=int(walkoff.sum()))
