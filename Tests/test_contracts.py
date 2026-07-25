"""Core-contract tripwires the 2026-07-25 audit found untested.

(a) Bets/gate purity — the display market blend must never reach
    build_bets or market_gate (test_pure_sidecar covers the
    display/sidecar half of the W4.22 contract);
(b) replay rows raw — the three purity resets (calib {}, heads None,
    mkt_blend False) exist at every replay entry point in evaluate.py;
(c) isotonic ban — no isotonic calibrator anywhere in the model layer
    (Platt only, no hard 0/1); the shipped hazard artifact is checked
    once it postdates the retrain that removed isotonic;
(d) SERVE_CAL_YEAR — trees train strictly BEFORE the calibration year
    the manifest declares, calibrators fit ON it;
(e) wind-casing parity — every WindDir string Tools/1 can emit (and
    its boxscore-cased twin) resolves in features' wind map;
(f) market vocabulary — every batter label predict serves is gradeable
    by evaluate and Tools/4, and their graders agree on synthetic
    stat lines.
"""
import inspect
import json
import re
from pathlib import Path

import pandas as pd
import pytest

import evaluate as EV
import features as F
import predict as PR
from conftest import load_tool

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "Model"


# ------------------------------------------------ (a) Bets/gate purity

def test_build_bets_never_touches_the_blend():
    """The Bets sheet prices EV off pure model probabilities: the blend
    helpers must not appear anywhere in the bet-building path."""
    src = inspect.getsource(PR.build_bets)
    assert "_blend_market" not in src
    assert "_blend_p" not in src


def test_market_gate_forces_pure_model():
    """The CLV gate grades the pure model vs the market: it must reset
    mkt_blend before serving and never call the blend."""
    src = inspect.getsource(EV.market_gate)
    assert re.search(r"mkt_blend\s*=\s*False", src)
    assert "_blend_market" not in src


def test_display_blend_runs_after_pure_snapshot():
    """game_frame must snapshot the pure frame BEFORE mutating rows with
    the blend, or the sidecar archives blended values."""
    src = inspect.getsource(PR.game_frame)
    snap = src.find("pre-blend snapshot")
    blend = src.find("_blend_market(")
    assert snap != -1 and blend != -1 and snap < blend


# ------------------------------------------------ (b) replay rows raw

def test_every_replay_entry_point_resets_purity():
    """Each place evaluate.py builds replay/gate rows must strip
    calibrators, heads, and the blend — the 2026-07-19 calibration-
    composition bug and the W4.22 purity contract, pinned."""
    src = (MODEL / "evaluate.py").read_text(encoding="utf-8")
    resets = re.findall(r"P\.mkt_blend\s*=\s*False", src)
    assert len(resets) >= 3          # replay_rows x2 + market_gate
    assert len(re.findall(r"P\.calib\s*=\s*\{\}", src)) >= 2
    assert len(re.findall(r"P\.heads\s*=\s*None", src)) >= 2


# ------------------------------------------------ (c) isotonic ban

def test_no_isotonic_anywhere_in_model_layer():
    for name in ("train.py", "predict.py", "evaluate.py",
                 "features.py", "sim.py", "sim_batch.py", "heads.py"):
        src = (MODEL / name).read_text(encoding="utf-8")
        assert "IsotonicRegression" not in src, name
        assert "sklearn.isotonic" not in src, name


def test_shipped_hazard_calibrator_is_platt():
    """The live hazard artifact must carry a Platt map with strictly
    interior outputs. Skips only while the artifact still predates the
    fix (it refits at the next nightly retrain)."""
    p = MODEL / "artifacts" / "hazard_model.joblib"
    if not p.exists():
        pytest.skip("no hazard artifact on this machine")
    import joblib
    hz = joblib.load(p)
    cal = hz.get("iso")
    if not isinstance(cal, F.PlattCal):
        assert p.stat().st_mtime < (MODEL / "train.py").stat().st_mtime, \
            "hazard artifact retrained AFTER the fix but not Platt"
        pytest.skip("pre-retrain isotonic artifact; refits tonight")
    import numpy as np
    out = cal.predict(np.array([1e-9, 1e-4, 0.5, 1 - 1e-4, 1 - 1e-9]))
    assert (out > 0).all() and (out < 1).all()


# ------------------------------------------------ (d) SERVE_CAL_YEAR

def test_trees_never_train_on_the_calibration_year():
    """manifest.cal_year is the served calibration year; train.py must
    declare the same year and mask trees strictly before it."""
    man = json.loads((MODEL / "artifacts" / "manifest.json")
                     .read_text(encoding="utf-8"))
    src = (MODEL / "train.py").read_text(encoding="utf-8")
    m = re.search(r"^SERVE_CAL_YEAR\s*=\s*(\d{4})", src, re.M)
    assert m and int(m.group(1)) == int(man["cal_year"])
    # every fit site pairs a strict tree mask with an ==cal calib mask
    trees = len(re.findall(r'pa\["Season"\]\s*<\s*cal_year', src)) \
        + len(re.findall(r'hz\["Season"\]\s*<\s*cal_year', src))
    calib = len(re.findall(r'\["Season"\]\s*==\s*cal_year', src))
    assert trees >= 4 and calib >= trees


# ------------------------------------------------ (e) wind casing

def test_every_slate_wind_string_resolves():
    """Tools/1 emits title-cased sectors, boxscores emit 'Out To CF'
    style; both must resolve through the .str.title() normalization
    features.assemble_features applies before WIND_RADIAL."""
    T1 = load_tool("1) Get Todays Games")
    directional = [s for s in T1.WIND_SECTORS]
    boxscore = [re.sub(r"(Cf|Lf|Rf)$", lambda m: m.group(1).upper(), s)
                for s in directional]
    ser = pd.Series(directional + boxscore + ["Calm", "None", None])
    mapped = ser.astype("string").str.title().map(F.WIND_RADIAL)
    # every non-null string resolves; only the true null stays NaN
    assert mapped[ser.notna()].notna().all()
    assert F.WIND_RADIAL["Out To Cf"] == 1.0
    assert F.WIND_RADIAL["In From Cf"] == -1.0


# ------------------------------------------------ (f) market vocab

def _tools4_events():
    return load_tool("4) Grade Results").BAT_EVENTS


def test_every_served_batter_label_is_gradeable():
    served = {lbl for lbl, _ in PR.BAT_BINARIES}
    assert served <= set(EV.BAT_ACTUAL), \
        f"evaluate can't grade: {served - set(EV.BAT_ACTUAL)}"
    assert served <= set(_tools4_events()), \
        f"Tools/4 can't grade: {served - set(_tools4_events())}"


def test_evaluate_and_tools4_graders_agree():
    """Same synthetic stat lines through both vocabularies — a drifted
    threshold in either file fails here, not on a graded slate."""
    events = _tools4_events()
    lines = [
        dict(H=0, HR=0, TB=0, R=0, RBI=0, BB=0, SB=0, SO=0,
             **{"2B": 0, "3B": 0}),
        dict(H=1, HR=0, TB=1, R=0, RBI=0, BB=1, SB=1, SO=1,
             **{"2B": 0, "3B": 0}),
        dict(H=2, HR=1, TB=6, R=2, RBI=3, BB=0, SB=0, SO=2,
             **{"2B": 1, "3B": 0}),
        dict(H=3, HR=0, TB=5, R=1, RBI=1, BB=2, SB=2, SO=3,
             **{"2B": 1, "3B": 1}),
    ]
    shared = set(EV.BAT_ACTUAL) & set(events)
    assert len(shared) >= 20
    for stats in lines:
        row = pd.Series(stats)
        for mkt in shared:
            assert bool(EV.BAT_ACTUAL[mkt](row)) == \
                bool(events[mkt](stats)), (mkt, stats)
