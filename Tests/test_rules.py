"""Season rules and mechanism-level behavior: 7-inning doubleheaders,
ghost runner, walk-off legality, bullpen exhaustion, platoon pen picks,
reliever rotation, steal accounting."""
import numpy as np

import sim as S
import synth


def _shutout_away_prep(**kw):
    """Away bats strike out every PA; home hits a realistic mix — home
    leads almost every game, so game length is deterministic-ish."""
    return synth.make_prep(away_vec=synth.class_vec(K=1.0), **kw)


def test_regulation_length_nine_innings():
    res = S.run(_shutout_away_prep(), n_sims=1500, seed=21, season=2024)
    t = res["tensor"]
    pa = t[:, synth.AWAY_BATS, S.SIDX["PA"]].sum(1)
    k = t[:, synth.AWAY_BATS, S.SIDX["K"]].sum(1)
    assert (pa == k).all()                       # all-K offense
    assert (pa % 3 == 0).all()                   # only full innings
    assert pa.min() == 27                        # 9 regulation innings
    assert (pa == 27).mean() > 0.8               # extras are rare


def test_regulation_length_seven_inning_dh():
    res = S.run(_shutout_away_prep(), n_sims=1500, seed=22, season=2021,
                is_dh_game=True)
    pa = res["tensor"][:, synth.AWAY_BATS, S.SIDX["PA"]].sum(1)
    assert pa.min() == 21                        # 7-inning regulation
    assert (pa == 21).mean() > 0.7


def _end_half_state(n, inning, half, score, bat_ptr0=3):
    prep = synth.make_prep()
    st = dict(
        inning=np.full(n, inning, dtype=np.int8),
        half=np.full(n, half, dtype=np.int8),
        outs=np.full(n, 3, dtype=np.int8),
        bases=-np.ones((n, 3), dtype=np.int16),
        resp=-np.ones((n, 3), dtype=np.int16),
        score=np.array(score, dtype=np.int16),
        runs_f5=np.zeros((n, 2), dtype=np.int16),
        runs_i1=np.zeros((n, 2), dtype=np.int16),
        bat_ptr=np.full((n, 2), bat_ptr0, dtype=np.int8),
        done=np.zeros(n, dtype=bool),
        stint_outs=np.zeros((n, 2), dtype=np.int8),
        cur_pit=np.tile(np.array([18, 19], dtype=np.int16), (n, 1)),
        starter_in=np.ones((n, 2), dtype=bool),
        pen_next=np.zeros((n, 2), dtype=np.int8),
        tensor=np.zeros((n, synth.NP_, len(S.STATS)), dtype=np.int16),
        active=np.ones((n, 18), dtype=bool),
        bench=np.array([36, 37], dtype=np.int16),
    )
    return prep, st


def _call_end_half(prep, st, season, mask=None, postseason=False):
    rules = S.rules_for(season, postseason=postseason)
    mask = np.ones(len(st["inning"]), dtype=bool) if mask is None \
        else mask
    S._end_half(mask, st["inning"], st["half"], st["outs"], st["bases"],
                st["resp"], st["score"], st["runs_f5"], st["runs_i1"],
                st["bat_ptr"], st["done"], st["stint_outs"],
                st["cur_pit"], st["starter_in"], st["pen_next"], prep,
                np.random.default_rng(0), st["tensor"],
                rules["regulation"], rules, st["active"], st["bench"],
                None)
    return st


def test_ghost_runner_placed_in_extras():
    # bottom of the 9th ends tied -> top 10 opens with the previous
    # batter (slot bat_ptr-1) on second, charged to the fielding pitcher
    prep, st = _end_half_state(2, inning=9, half=1,
                               score=[[4, 4], [2, 2]], bat_ptr0=3)
    _call_end_half(prep, st, season=2024)
    assert (st["inning"] == 10).all() and (st["half"] == 0).all()
    assert not st["done"].any()
    assert (st["bases"][:, 1] == 2).all()        # away slot 2 = row 2
    assert (st["bases"][:, [0, 2]] == -1).all()
    assert (st["resp"][:, 1] == 19).all()        # home pitcher charged


def test_no_ghost_runner_before_2020():
    prep, st = _end_half_state(2, inning=9, half=1,
                               score=[[4, 4], [2, 2]])
    _call_end_half(prep, st, season=2019)
    assert (st["inning"] == 10).all()
    assert (st["bases"] == -1).all()


def test_rules_for_postseason_traditional():
    # the automatic runner and the 7-inning DH are regular-season only:
    # postseason=True must kill both in EVERY season they exist
    for season in (2020, 2021, 2024, 2026):
        assert S.rules_for(season)["ghost_runner"]
        r = S.rules_for(season, postseason=True)
        assert not r["ghost_runner"]
        assert r["regulation"] == 9
    assert not S.rules_for(2019, postseason=True)["ghost_runner"]
    assert S.rules_for(2021, is_dh_game=True)["regulation"] == 7
    assert S.rules_for(2021, is_dh_game=True,
                       postseason=True)["regulation"] == 9


def test_no_ghost_runner_in_postseason_extras():
    # same tied bottom-9 state as the 2024 placement test, but October:
    # top 10 must open with empty bases (season-only gating placed one)
    prep, st = _end_half_state(2, inning=9, half=1,
                               score=[[4, 4], [2, 2]])
    _call_end_half(prep, st, season=2024, postseason=True)
    assert (st["inning"] == 10).all() and (st["half"] == 0).all()
    assert not st["done"].any()
    assert (st["bases"] == -1).all()
    assert (st["resp"] == -1).all()


def test_batch_rules_postseason_vector():
    # run_batch's per-game postseason list must reach the rule vectors
    import sim_batch as SB
    prep = synth.make_prep()
    bp = SB.BatchPrep([prep, prep], [2024, 2024], [False, False],
                      [False, True], np)
    assert bp.ghost.tolist() == [True, False]


def test_postseason_extras_play_traditional():
    # walk-only offense + always-caught steals: every inning is
    # walk/CS x3 and the game rides 0-0 to the cap (see
    # test_steals_caught_only). In 2024 the ghost runner would occupy
    # 2B in extras, blocking the steal outs — walks then score without
    # end. postseason=True must reproduce the pre-2020 traditional ride.
    walks = synth.class_vec(BB=1.0)
    prep = synth.make_prep(away_vec=walks, home_vec=walks, sb_att=1.0,
                           sb_suc=0.0)
    res = S.run(prep, n_sims=100, seed=28, season=2024,
                postseason=True)
    t, sc = res["tensor"], res["score"]
    assert res["leftover"] == 0
    assert (sc.sum(axis=1) == 1).all()           # cap coin-flip only
    assert (t[..., S.SIDX["SB"]] == 0).all()
    cs = t[..., S.SIDX["CS"]].sum(1)
    assert (cs == 2 * 3 * S.MAX_INNINGS).all()   # no free runner ever


def test_game_end_conditions():
    # away leads after a completed bottom 9 -> over
    prep, st = _end_half_state(1, inning=9, half=1, score=[[5, 3]])
    _call_end_half(prep, st, season=2024)
    assert st["done"].all()
    # home leads after the top of the 9th -> bottom is skipped
    prep, st = _end_half_state(1, inning=9, half=0, score=[[3, 5]])
    _call_end_half(prep, st, season=2024)
    assert st["done"].all()
    # home trails after the top of the 9th -> bottom is played
    prep, st = _end_half_state(1, inning=9, half=0, score=[[5, 3]])
    _call_end_half(prep, st, season=2024)
    assert not st["done"].any()
    # hard cap: tied bottom of the 19th ends the sim
    prep, st = _end_half_state(1, inning=S.MAX_INNINGS, half=1,
                               score=[[6, 6]])
    _call_end_half(prep, st, season=2024)
    assert st["done"].all()


def test_bullpen_exhaustion_starter_stays():
    # hazard fires immediately but there is no pen: the starter must
    # finish the game (chosen = -1 keeps cur_pit)
    prep = synth.make_prep(hazard=1.0, pen_empty=True)
    res = S.run(prep, n_sims=400, seed=23, season=2024)
    t = res["tensor"]
    synth.ledger_checks(res)
    assert (t[:, 20:36, S.SIDX["BF"]] == 0).all()
    assert (t[:, 18, S.SIDX["BF"]] > 0).all()
    assert (t[:, 19, S.SIDX["BF"]] > 0).all()


def test_platoon_pen_pick_jumps_queue():
    # first pen slot throws L, second R; every batter hits R and the
    # starter exits after one batter -> the R arm (slot 2, within
    # PLATOON_WIN) must jump the L arm, which never pitches
    throws = np.ones(synth.NP_, dtype=np.int8)
    throws[[20, 28]] = 0
    prep = synth.make_prep(hazard=1.0, pit_throws=throws)
    res = S.run(prep, n_sims=400, seed=24, season=2024)
    t = res["tensor"]
    synth.ledger_checks(res)
    assert (t[:, 18, S.SIDX["BF"]] == 1).all()   # starters: 1 BF each
    assert (t[:, 19, S.SIDX["BF"]] == 1).all()
    assert (t[:, 20, S.SIDX["BF"]] == 0).all()   # L arms skipped
    assert (t[:, 28, S.SIDX["BF"]] == 0).all()
    assert (t[:, 21, S.SIDX["BF"]] > 0).all()    # R arms pitched
    assert (t[:, 29, S.SIDX["BF"]] > 0).all()


def test_reliever_rotation_on_exit():
    # exit prob 1 at every inning break -> a fresh arm each inning
    prep = synth.make_prep(hazard=1.0, relief_exit=1.0)
    res = S.run(prep, n_sims=400, seed=25, season=2024)
    t = res["tensor"]
    synth.ledger_checks(res)
    used = (t[:, 20:28, S.SIDX["BF"]] > 0).sum(axis=1)
    assert used.mean() > 6                       # deep into the pen
    assert (used >= 4).all()


def test_steals_caught_only():
    # walk-only offense, attempt prob 1, success prob 0: every inning is
    # walk/CS x3, nobody advances past first, no SB ever. Pre-2020 (no
    # ghost runner) the game rides 0-0 to the MAX_INNINGS safety cap,
    # where the fair-coin tiebreak (2026-07-25) awards exactly ONE
    # deciding run — a tie must never leak into (home > away) win
    # accounting, which counted capped ties as home losses.
    walks = synth.class_vec(BB=1.0)
    prep = synth.make_prep(away_vec=walks, home_vec=walks, sb_att=1.0,
                           sb_suc=0.0)
    res = S.run(prep, n_sims=200, seed=26, season=2019)
    t, sc = res["tensor"], res["score"]
    assert res["leftover"] == 0
    # tiebreak: every capped tie ends 1-0 (one run total, margin 1)
    assert (sc.sum(axis=1) == 1).all()
    assert (np.abs(sc[:, 1] - sc[:, 0]) == 1).all()
    hw = (sc[:, 1] > sc[:, 0]).mean()
    assert 0.0 < hw < 1.0                        # a fair coin, not a rule
    # the deciding run lands in the winner's batter-R ledger too, so
    # scoreboard == player run ledger still holds per team
    for rows, side in ((synth.AWAY_BATS, 0), (synth.HOME_BATS, 1)):
        assert (t[:, rows, S.SIDX["R"]].sum(1) == sc[:, side]).all()
    assert (t[..., S.SIDX["SB"]] == 0).all()
    cs = t[..., S.SIDX["CS"]].sum(1)
    outs = t[..., S.SIDX["OUTS"]].sum(1)
    assert (cs == outs).all()                    # every out was a CS
    assert (cs == 2 * 3 * S.MAX_INNINGS).all()   # 25 full tied innings


def test_steals_successful_advance():
    # realistic offense (outs still happen) but every steal succeeds
    prep = synth.make_prep(sb_att=1.0, sb_suc=1.0)
    res = S.run(prep, n_sims=100, seed=27, season=2024)
    t = res["tensor"]
    assert (t[..., S.SIDX["CS"]] == 0).all()
    assert t[..., S.SIDX["SB"]].sum() > 0
    synth.ledger_checks(res)
