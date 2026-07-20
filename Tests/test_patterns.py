"""Pattern-bank machinery: key encoding, sampling, fallback chain, and
the dense (batched) bank matching the sparse (classic) bank."""
import numpy as np
import pytest

import sim as S
import synth


def test_pattern_key_bijective():
    seen = set()
    for cls in range(8):
        for bb in range(5):
            for state in range(8):
                for outs in range(3):
                    k = S.pattern_key(cls, bb, state, outs)
                    assert 0 <= k < 960
                    seen.add(k)
    assert len(seen) == 960


def test_sample_rows_bounds():
    P = np.array([[0.5, 0.5, 0.0], [0.3, 0.3, 0.3]])
    assert S._sample_rows(P, np.array([0.0, 0.0])).tolist() == [0, 0]
    assert S._sample_rows(P, np.array([0.49, 0.95])).tolist() == [0, 2]
    # never returns an index outside the row even at u ~ 1
    idx = S._sample_rows(P, np.array([0.999999, 0.999999]))
    assert (idx <= 2).all()


def test_bat_axis():
    rows = np.array([0, 8, 9, 17, 36, 37])
    assert S.bat_axis(rows).tolist() == [0, 8, 9, 17, 18, 19]


def test_fallback_chain():
    bank = synth.make_bank()
    # IPO never has bb=0 in the bank -> (cls, bb, 0, outs) is the first
    # fallback that exists for a deleted key
    k = S.pattern_key(S.CI["IPO"], 2, 5, 1)
    entry = bank.pop(k)
    del entry
    got = S._fallback_pattern(k, bank, {})
    want = bank[S.pattern_key(S.CI["IPO"], 2, 0, 1)]
    assert got is want


def test_fallback_last_resort():
    got = S._fallback_pattern(S.pattern_key(3, 2, 5, 1), {}, {})
    cum, arr = got
    assert cum.shape[0] == S.NB and arr.shape == (1, 8)
    assert arr[0, 0] == 1                      # batter to 1B, no outs


def test_dense_matches_sparse():
    import sim_batch as SB
    bank = synth.make_bank()
    cum, arr = SB.dense_patterns(bank)
    assert cum.shape[0] == 960 and arr.shape[0] == 960
    cache = {}
    for k in range(960):
        entry = bank.get(k) or S._fallback_pattern(k, bank, cache)
        c, a = entry
        n = c.shape[1]
        assert np.allclose(cum[k, :, :n], c, atol=1e-6)
        assert (arr[k, :n] == a).all()
        # padding must be unreachable: cumulative 2.0 > any uniform
        assert (cum[k, :, n:] >= 1.5).all()


@pytest.mark.skipif(
    not (S.Path(__file__).resolve().parents[1] / "Model" / "artifacts"
         / "stores" / "pattern_table.parquet").exists(),
    reason="real pattern_table not built")
def test_real_bank_probabilities_normalized():
    pat = S.load_patterns()
    for k, (cum, arr) in pat.items():
        assert cum.shape[0] == S.NB
        # each tilt bucket is a proper cumulative distribution
        assert np.allclose(cum[:, -1], 1.0, atol=1e-6), k
        assert (np.diff(cum, axis=1) >= -1e-9).all(), k
        assert (arr[:, 4] >= 0).all() and (arr[:, 4] <= 3).all()
