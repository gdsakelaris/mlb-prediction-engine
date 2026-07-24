"""MIRROR tag inventory: the sim engine and its preparation pipeline are
DELIBERATELY maintained in two parallel implementations (numpy per-game
reference vs batched device path — a documented decision). Each twin
site carries a `# MIRROR[tag]` comment naming its partner; this test
pins the inventory so a twin can't be edited (or a tag deleted) without
tripping the fast suite.
"""
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "Model"

# tag -> the exact pair of files that must each carry it exactly once
EXPECTED = {
    "engine_loop": ("sim.py", "sim_batch.py"),
    "loop_pen_choice": ("sim.py", "sim_batch.py"),
    "loop_endhalf": ("sim.py", "sim_batch.py"),
    "prep_resolve": ("predict.py", "sim_batch.py"),
    "prep_matchup": ("predict.py", "sim_batch.py"),
    "prep_hazard": ("predict.py", "sim_batch.py"),
    "prep_sb": ("predict.py", "sim_batch.py"),
}


def test_mirror_tags_present_in_exactly_their_twin_pair():
    srcs = {n: (MODEL / n).read_text(encoding="utf-8")
            for n in ("sim.py", "sim_batch.py", "predict.py")}
    for tag, pair in EXPECTED.items():
        needle = f"MIRROR[{tag}]"
        for name in pair:
            n = srcs[name].count(needle)
            assert n == 1, (f"{needle} appears {n}x in {name} "
                            f"(expected exactly 1 — twin edited or tag "
                            f"deleted?)")
        for name, src in srcs.items():
            if name not in pair:
                assert needle not in src, (
                    f"{needle} leaked into {name}")


def test_no_unregistered_mirror_tags():
    import re
    for name in ("sim.py", "sim_batch.py", "predict.py"):
        src = (MODEL / name).read_text(encoding="utf-8")
        for tag in set(re.findall(r"MIRROR\[(\w+)\]", src)):
            assert tag in EXPECTED, (f"unregistered MIRROR tag "
                                     f"{tag!r} in {name} — add it to "
                                     f"EXPECTED with its twin pair")
