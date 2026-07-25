"""write_pure_sidecar (Model/predict.py) — pure-model archive contract.

(a) a frame carrying a `pure` snapshot writes the PRE-blend values,
    not the blended live rows, for bat/pit/game sheets alike;
(b) a frame without a snapshot (blend off / no prices) falls back to
    its live rows — those are already pure;
(c) identity keys (ID, G#, Slot, Career G) and non-numeric cells are
    join keys only, never `p` rows; private floats (_home_wp) are kept;
(d) re-writing the same stem replaces the sidecar atomically and
    leaves no .tmp behind;
(e) game_frame's `pure` snapshot IS the pre-blend frame: bit-identical
    to serving the same sim result with the blend off, while the
    blended cells are exactly _blend_p(pure, fair) — the W4.22
    contract the sidecar exists to archive;
(f) the blend math itself: 50/50 logit-space _blend_p and the
    R_ONESIDED one-sided haircut in _soft_fair.
"""
import csv

import numpy as np

import predict as P
import sim as S
import synth


def _frame(pure_hit, live_hit):
    bat = [{"Game": "AAA@BBB", "Team": "AAA", "G#": 1, "Slot": 3,
            "Name": "Bat One", "ID": 111, "Career G": 250,
            "Hit": live_hit, "HR": 0.12}]
    pit = [{"Game": "AAA@BBB", "Team": "BBB", "G#": 1, "Name": "Arm",
            "ID": 222, "K > 4.5": 0.61}]
    game = {"Game": "AAA@BBB", "Date": "2026-07-25", "Venue": "Park",
            "Winner": "BBB", "Win Prob": 0.58, "_home_wp": 0.58}
    f = dict(bat=bat, pit=pit, game=game, blend_n=3)
    f["pure"] = dict(
        bat=[{**bat[0], "Hit": pure_hit}],
        pit=[{**pit[0], "K > 4.5": 0.70}],
        game={**game, "Win Prob": 0.55, "_home_wp": 0.55})
    return f


def _read(dest):
    with open(dest, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_pure_snapshot_wins_over_blended_rows(tmp_path):
    dest = P.write_pure_sidecar([_frame(0.42, 0.55)],
                                tmp_path / "2026-07-25.xlsx")
    assert dest == tmp_path / "pure" / "2026-07-25.csv"
    rows = _read(dest)
    by = {(r["sheet"], r["ID"], r["col"]): float(r["p"]) for r in rows}
    assert by[("bat", "111", "Hit")] == 0.42          # not 0.55
    assert by[("pit", "222", "K > 4.5")] == 0.70      # not 0.61
    assert by[("game", "", "_home_wp")] == 0.55       # private kept
    assert by[("game", "", "Win Prob")] == 0.55


def test_no_snapshot_falls_back_to_live_rows(tmp_path):
    f = _frame(0.42, 0.55)
    f["pure"] = None                    # blend off: live rows are pure
    rows = _read(P.write_pure_sidecar([f], tmp_path / "2026-07-25.xlsx"))
    by = {(r["sheet"], r["ID"], r["col"]): float(r["p"]) for r in rows}
    assert by[("bat", "111", "Hit")] == 0.55


def test_identity_and_text_cells_never_become_p_rows(tmp_path):
    rows = _read(P.write_pure_sidecar([_frame(0.42, 0.55)],
                                      tmp_path / "2026-07-25.xlsx"))
    cols = {r["col"] for r in rows}
    assert cols.isdisjoint({"ID", "G#", "Slot", "Career G",
                            "Game", "Team", "Name", "Date", "Venue",
                            "Winner"})
    bat = [r for r in rows if r["sheet"] == "bat"]
    assert {r["col"] for r in bat} == {"Hit", "HR"}
    assert bat[0]["Name"] == "Bat One" and bat[0]["ID"] == "111"


def test_rewrite_replaces_atomically_no_tmp_litter(tmp_path):
    xlsx = tmp_path / "2026-07-25.xlsx"
    P.write_pure_sidecar([_frame(0.42, 0.55)], xlsx)
    dest = P.write_pure_sidecar([_frame(0.30, 0.55)], xlsx)
    by = {(r["sheet"], r["ID"], r["col"]): float(r["p"])
          for r in _read(dest)}
    assert by[("bat", "111", "Hit")] == 0.30          # replaced
    assert list((tmp_path / "pure").glob("*.tmp")) == []
    assert list((tmp_path / "pure").iterdir()) == [dest]


# ---------------------------------------------------------------- (e)
# game_frame blend/purity contract on a real sim result

def make_res(n_sims=400, seed=5, gpk=777):
    """A real sim result dressed with the meta/spec game_frame needs —
    18 batters (ids 101-118), two starters (201/202), no calibrators,
    no heads. Shared with Tests/test_contracts.py."""
    prep = synth.make_prep()
    res = S.run(prep, n_sims=n_sims, seed=seed, season=2024)
    players = list(range(101, 119)) + [201, 202] + [-1] * 18
    res["meta"] = dict(
        away="AAA", home="BBB", season=2024, players=players,
        names=[f"Player {i}" for i in range(synth.NP_)],
        career_g=[200] * synth.NP_, career_gp=[200] * synth.NP_)
    res["spec"] = dict(date="2026-07-25", venue="Park",
                       game_pk=gpk, dh_game2=False, is_dh=False)
    res["calib"] = {}
    return res


def _fair_maps(gpk=777):
    """Market fairs for FIRST-rung columns only, so the served ladder
    clamps cannot move any neighbouring cell and every blended value is
    exactly _blend_p(pure, fair)."""
    return {"bat": {(101, "Hit"): {gpk: 0.90}},
            "pit": {(201, "K > 3.5"): {gpk: 0.85}},
            "game": {("BBB", "_home_wp"): {gpk: 0.75},
                     ("BBB", "Runs > 6.5"): {gpk: 0.97},
                     ("AAA", "TT > 2.5"): {gpk: 0.96}}}


def test_game_frame_pure_snapshot_is_the_preblend_frame():
    res = make_res()
    base = P.game_frame(res)             # no mktfair: blend never runs
    assert base["pure"] is None and base["blend_n"] == 0
    res["mktfair"] = _fair_maps()
    blended = P.game_frame(res)
    # the snapshot is bit-identical to the blend-off serve of the same
    # sims — the sidecar archives THE pure numbers, not near-copies
    assert blended["pure"]["bat"] == base["bat"]
    assert blended["pure"]["pit"] == base["pit"]
    assert blended["pure"]["game"] == base["game"]
    assert blended["blend_n"] == 5


def test_game_frame_blended_cells_are_exact_blend_p():
    res = make_res()
    base = P.game_frame(res)
    res["mktfair"] = _fair_maps()
    blended = P.game_frame(res)
    b0, p0 = blended["bat"][0], base["bat"][0]
    assert b0["ID"] == 101
    assert b0["Hit"] == P._blend_p(p0["Hit"], 0.90)
    assert b0["Hit"] != p0["Hit"]                    # discriminating
    # every other cell of the covered row is untouched
    assert {k: v for k, v in b0.items() if k != "Hit"} \
        == {k: v for k, v in p0.items() if k != "Hit"}
    # rows with no captured market stay pure model entirely
    assert blended["bat"][1:] == base["bat"][1:]
    bp, pp = blended["pit"][0], base["pit"][0]
    assert bp["K > 3.5"] == P._blend_p(pp["K > 3.5"], 0.85)
    assert blended["pit"][1] == base["pit"][1]
    bg, pg = blended["game"], base["game"]
    hw = P._blend_p(pg["_home_wp"], 0.75)
    assert bg["_home_wp"] == hw
    assert bg["Winner"] == ("BBB" if hw >= 0.5 else "AAA")
    assert bg["Win Prob"] == max(hw, 1 - hw)
    assert bg["Runs > 6.5"] == P._blend_p(pg["Runs > 6.5"], 0.97)
    assert bg["Away Runs > 2.5"] == P._blend_p(pg["Away Runs > 2.5"],
                                               0.96)
    # blending a first rung upward can never break ladder coherence
    for x, prev in zip(P.TOTAL_LINES[1:], P.TOTAL_LINES):
        assert bg[f"Runs > {x}"] <= bg[f"Runs > {prev}"]


def test_game_frame_wrong_gamepk_market_never_blends():
    # DH partial capture: a fair keyed to a DIFFERENT real GamePk (and
    # no -1 legacy key) must not blend — the other matchup's price
    res = make_res(gpk=777)
    base = P.game_frame(res)
    res["mktfair"] = {"bat": {(101, "Hit"): {888: 0.90}}, "pit": {},
                     "game": {}}
    blended = P.game_frame(res)
    assert blended["blend_n"] == 0
    assert blended["bat"] == base["bat"]


def test_sidecar_game_rows_carry_dh_identity(tmp_path):
    # grow gained G#/GamePk (2026-07-25) so DH days' game rows are
    # attributable; they must ride into the sidecar as identity keys,
    # never as `p` rows
    res = make_res(gpk=777)
    frame = P.game_frame(res)
    dest = P.write_pure_sidecar([frame], tmp_path / "2026-07-25.xlsx")
    rows = _read(dest)
    game = [r for r in rows if r["sheet"] == "game"]
    assert game and all(r["G#"] == "1" and r["GamePk"] == "777"
                        for r in game)
    assert {r["col"] for r in rows}.isdisjoint({"G#", "GamePk"})


# ---------------------------------------------------------------- (f)
# blend math: 50/50 logit blend and the R_ONESIDED haircut

def test_blend_p_is_5050_logit_blend():
    p, f = 0.62, 0.48
    z = 0.5 * np.log(p / (1 - p)) + 0.5 * np.log(f / (1 - f))
    assert P._blend_p(p, f) == float(1.0 / (1.0 + np.exp(-z)))
    assert P._blend_p(0.5, 0.5) == 0.5
    # weight override is on the MARKET side
    assert P._blend_p(0.62, 0.48, w=0.0) == 0.62
    assert abs(P._blend_p(0.62, 0.48, w=1.0) - 0.48) < 1e-12
    # clipped strictly inside (0, 1) even at degenerate inputs
    assert 0.0 < P._blend_p(0.0, 0.0) < P._blend_p(1.0, 1.0) < 1.0


def test_soft_fair_two_sided_beats_haircut():
    recs = [dict(OverPrice=-110, UnderPrice=-110, Book="pinnacle")]
    assert P._soft_fair(recs) == 0.5              # true de-vig, no cut


def test_soft_fair_one_sided_haircut():
    import odds as O
    # over-only board: median implied over x R_ONESIDED
    recs = [dict(OverPrice=-120, UnderPrice=None, Book="dk"),
            dict(OverPrice=-130, UnderPrice=None, Book="mgm"),
            dict(OverPrice=-140, UnderPrice=None, Book="fd")]
    med = O.american_to_prob(-130)
    assert P._soft_fair(recs) == float(np.clip(med * P.R_ONESIDED,
                                               1e-6, 1 - 1e-6))
    # under-only board: complement of the haircut under consensus
    recs = [dict(OverPrice=None, UnderPrice=-115, Book="dk")]
    mu = O.american_to_prob(-115)
    assert P._soft_fair(recs) == float(np.clip(1.0 - mu * P.R_ONESIDED,
                                               1e-6, 1 - 1e-6))
    assert P._soft_fair([dict(OverPrice=None, UnderPrice=None,
                              Book="dk")]) is None
