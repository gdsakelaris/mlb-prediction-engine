"""Same-game-parlay pricing off the persisted sim tensors.

Every served slate persists Predictions/<date>.sims.npz — the full
per-sim player-stat tensors. A parlay's joint probability is priced by
counting sims where EVERY leg hits, which embeds all the correlations
the engine simulated (lineup runs feeding RBIs, pitcher Ks suppressing
hits, totals coupling to team totals) — no copula, no independence
assumption.

IPF RE-ANCHORING GUARD: the engine's served single-leg probabilities are
Platt-calibrated per family, so the raw tensor marginals drift slightly
from the numbers actually published. Pricing the joint straight off raw
sims would embed those drifts multiplicatively. The fix is iterative
proportional fitting on the per-sim weights (raking): reweight sims so
every leg's implied marginal matches its CALIBRATED serve value, then
read the parlay probability under those weights. Correlation structure
survives; marginals re-anchor to the published numbers.

Leg mini-language (comma-separated, all legs one game):
    <pid>:<market>:<line>:<over|under>     player prop
    <TEAM>:h2h::<win>                      moneyline (team abbrev)
    :totals:<line>:<over|under>            game total

Markets: the odds-store keys (batter_hits, batter_total_bases,
batter_home_runs, batter_hits_runs_rbis, pitcher_strikeouts, ...).

Usage:
    python Model/sgp.py --date 2026-07-18 \
        --legs "665742:batter_home_runs:0.5:over,543037:pitcher_strikeouts:6.5:over,:totals:8.5:over"
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict as PR      # noqa: E402
import sim                # noqa: E402

MAX_IPF_ITERS = 200
IPF_TOL = 1e-9


def _amer(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return round(-100 * p / (1 - p)) if p >= 0.5 \
        else round(100 * (1 - p) / p)


def _leg_counts(npz, gi, pid, market, line):
    """Per-sim count array for one leg in game gi (None = no market)."""
    t = npz[f"tensor_{gi}"]
    s = sim.SIDX
    players = list(npz[f"players_{gi}"])
    if market == "totals":
        sc = npz[f"score_{gi}"]
        return sc.sum(axis=1), "tot"
    row_i = players.index(int(pid)) if int(pid) in players else None
    if row_i is None or row_i >= 20:
        return None, None
    fam = PR.MKT_FAM.get(market)
    if market == "batter_total_bases":
        c = (t[:, row_i, s["B1"]] + 2 * t[:, row_i, s["B2"]]
             + 3 * t[:, row_i, s["B3"]] + 4 * t[:, row_i, s["HR"]])
    elif market == "batter_hits_runs_rbis":
        c = (t[:, row_i, s["H"]] + t[:, row_i, s["R"]]
             + t[:, row_i, s["RBI"]])
    else:
        col = dict(PR._BAT_STAT, **PR._PIT_STAT).get(market)
        if col is None:
            return None, None
        c = t[:, row_i, s[col]]
    return c, fam


def price_sgp(date, legs, pred_dir=None, calib=None):
    """legs: list of (pid_or_team, market, line, side). Returns dict
    with raw/independent/anchored prices and per-leg detail."""
    pred_dir = Path(pred_dir or PR.PRED_DIR)
    cands = sorted(pred_dir.glob(f"{date}*.sims.npz"))
    if not cands:
        raise SystemExit(f"no sims npz for {date} under {pred_dir}")
    npz = np.load(cands[-1], allow_pickle=True)
    if calib is None:
        cp = PR.ART / "output_calibrators.joblib"
        calib = joblib.load(cp) if cp.exists() else {}

    n_games = len([k for k in npz.files if k.startswith("tensor_")])

    # locate the ONE game containing every leg
    def game_has(gi, pid, market):
        players = list(npz[f"players_{gi}"])
        if market in ("h2h", "totals"):
            return True
        return int(pid) in players and players.index(int(pid)) < 20

    gis = [gi for gi in range(n_games)
           if all(game_has(gi, p, m) for p, m, _, _ in legs
                  if m not in ("h2h", "totals"))]
    if not gis:
        raise SystemExit("no single game contains every player leg")
    gi = gis[0]

    ind, targets, detail = [], [], []
    for pid, market, line, side in legs:
        if market == "h2h":
            sc = npz[f"score_{gi}"]
            # team abbrev vs meta unavailable in npz: home = col 1;
            # side token 'home'/'away' selects the winner leg
            hit = (sc[:, 1] > sc[:, 0]) if side == "home" \
                else (sc[:, 0] > sc[:, 1])
            fam = "ml"
        else:
            counts, fam = _leg_counts(npz, gi, pid, market, line)
            if counts is None:
                raise SystemExit(f"unpriceable leg: {pid}:{market}")
            over = counts > float(line)
            hit = over if side != "under" else ~over
        raw = float(hit.mean())
        if fam and side != "under":
            tgt = PR._cal(calib, fam, raw)
        elif fam:
            tgt = 1.0 - PR._cal(calib, fam, 1.0 - raw)
        else:
            tgt = raw
        ind.append(hit.astype(np.float64))
        targets.append(min(max(tgt, 1e-6), 1 - 1e-6))
        detail.append(dict(pid=pid, market=market, line=line, side=side,
                           p_raw=round(raw, 4), p_anchor=round(tgt, 4)))

    H = np.stack(ind, axis=1)                  # [S, L]
    S = H.shape[0]
    w = np.full(S, 1.0 / S)
    p_raw_joint = float(H.all(axis=1).mean())
    p_indep = float(np.prod(targets))
    # raking IPF: reweight sims until every leg marginal matches its
    # calibrated target
    for _ in range(MAX_IPF_ITERS):
        moved = 0.0
        for j, tgt in enumerate(targets):
            m = float(w @ H[:, j])
            if m <= 0 or m >= 1:
                continue
            moved = max(moved, abs(m - tgt))
            w = np.where(H[:, j] > 0, w * (tgt / m),
                         w * ((1 - tgt) / (1 - m)))
        w /= w.sum()
        if moved < IPF_TOL:
            break
    p_sgp = float(w @ H.all(axis=1))
    lift = p_sgp / max(p_indep, 1e-12)
    return dict(game_index=gi, legs=detail,
                p_raw_joint=round(p_raw_joint, 5),
                p_independent=round(p_indep, 5),
                p_sgp=round(p_sgp, 5),
                correlation_lift=round(lift, 4),
                fair_american=_amer(p_sgp))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--legs", required=True,
                    help="pid:market:line:side, comma-separated")
    args = ap.parse_args()
    legs = []
    for tok in args.legs.split(","):
        pid, market, line, side = (tok.split(":") + ["", "", ""])[:4]
        legs.append((pid.strip(), market.strip(), line.strip(),
                     side.strip().lower() or "over"))
    res = price_sgp(args.date, legs)
    print(f"\nSGP {args.date} (game {res['game_index']}):")
    for lg in res["legs"]:
        print(f"  {lg['pid'] or 'game':>8} {lg['market']:<24} "
              f"{lg['line'] or '':>5} {lg['side']:<6} "
              f"raw {lg['p_raw']:.4f} -> anchored {lg['p_anchor']:.4f}")
    print(f"  raw joint          {res['p_raw_joint']:.5f}")
    print(f"  independent product {res['p_independent']:.5f}")
    print(f"  SGP (IPF-anchored) {res['p_sgp']:.5f}   "
          f"lift x{res['correlation_lift']:.3f}   "
          f"fair {res['fair_american']:+d}")


if __name__ == "__main__":
    main()
