"""write_pure_sidecar (Model/predict.py) — pure-model archive contract.

(a) a frame carrying a `pure` snapshot writes the PRE-blend values,
    not the blended live rows, for bat/pit/game sheets alike;
(b) a frame without a snapshot (blend off / no prices) falls back to
    its live rows — those are already pure;
(c) identity keys (ID, G#, Slot, Career G) and non-numeric cells are
    join keys only, never `p` rows; private floats (_home_wp) are kept;
(d) re-writing the same stem replaces the sidecar atomically and
    leaves no .tmp behind.
"""
import csv

import predict as P


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
