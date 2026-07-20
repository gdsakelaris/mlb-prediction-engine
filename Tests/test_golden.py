"""Golden integration tests against real artifacts and stores: replay a
fixed historical game through the full serve path, assert the ledger
invariants hold on production preps, and check prepare_games row parity
against prepare_game. Slow: run explicitly with `pytest -m slow`."""
from pathlib import Path

import numpy as np
import pytest

import sim as S
import synth

MODEL = Path(__file__).resolve().parents[1] / "Model"
ART = MODEL / "artifacts"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (ART / "a1_model.joblib").exists()
        or not (ART / "stores" / "pattern_table.parquet").exists(),
        reason="model artifacts/stores not built"),
]


@pytest.fixture(scope="module")
def ctx():
    from predict import Predictor
    import backtest as B
    P = Predictor()
    lineups, starters, umps, wx = B._spec_frames(P)
    games = P.stores.raw["games"]
    pool = games[games.Season == 2025].sort_values("GamePk")
    spec = None
    for _, g in pool.iterrows():
        s = B.build_spec(P, g, lineups, starters, umps, wx)
        if len(s["away_lineup"]) == 9 and len(s["home_lineup"]) == 9 \
                and s["away_starter"] and s["home_starter"]:
            spec = s
            break
    assert spec is not None, "no complete 2025 game found"
    # mirror predict_slate: slate context (Elo/travel/park/ump) goes
    # into the spec before prepare_game; prepare_games resolves the
    # same context from stores, so parity needs it injected here too
    import features as F
    try:
        cx = F.slate_context([spec])[0]
        cx["park_hr"], cx["ump_r"] = P._game_effects(spec)
        spec["_ctx"] = cx
    except Exception as e:                    # noqa: BLE001
        pytest.skip(f"slate context unavailable: {e}")
    return P, spec


def test_golden_replay_ledger_and_sanity(ctx):
    P, spec = ctx
    prep, meta = P.prepare_game(spec, n_sims=2000)
    res = S.run(prep, n_sims=2000, seed=31, season=2025)
    synth.ledger_checks(res)
    total = np.asarray(res["score"]).sum(1)
    assert 5.0 < total.mean() < 13.0
    hw = (res["score"][:, 1] > res["score"][:, 0]).mean()
    assert 0.10 < hw < 0.90


def test_prepare_games_matches_prepare_game(ctx):
    import sim_batch as SB
    P, spec = ctx
    prep, _ = P.prepare_game(spec, n_sims=1000)
    bprep, _ = SB.prepare_games(P, [spec], n_sims=1000)[0]
    # unused matchup rows are NaN-padded identically in both paths
    for attr in ("avec", "a2vec", "haz_grid", "sb_att", "sb_suc"):
        a, b = getattr(prep, attr), getattr(bprep, attr)
        assert a.shape == b.shape, attr
        assert np.allclose(a, b, atol=1e-5, equal_nan=True), attr


def test_golden_engines_agree_on_real_prep(ctx):
    import sim_batch as SB
    P, spec = ctx
    prep, _ = P.prepare_game(spec, n_sims=4000)
    classic = S.run(prep, n_sims=4000, seed=32, season=2025)
    batch = SB.run_batch([prep], n_sims=4000, seed=33,
                         seasons=[2025], is_dh=[False],
                         backend="cpu")[0]
    for res in (classic, batch):
        synth.ledger_checks(res)
    t1 = np.asarray(classic["score"], dtype=float)
    t2 = np.asarray(batch["score"], dtype=float)
    for col in (0, 1):
        se = np.sqrt(t1[:, col].var() / len(t1)
                     + t2[:, col].var() / len(t2))
        assert abs(t1[:, col].mean() - t2[:, col].mean()) < 4.5 * se
