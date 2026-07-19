"""Headless serve: Data/todays_games.json -> workbook + sims npz.

The scheduler's serve entry point — the exact equivalent of the GUI's
Predict button (same spec assembly the GUI does: games sorted by
start_et, lineups tuple-ified), so a scheduled serve and a GUI serve
are indistinguishable downstream. Each game's npz product tag
(projected vs confirmed) comes from the per-side lineup provenance the
slate scraper recorded; serving early in the day simply yields
projected-product games, and a later re-serve yields confirmed ones.

Usage:
    python Model/serve_slate.py                 # full slate, 20k sims
    python Model/serve_slate.py --sims 4000     # faster, for smoke tests
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict as PR     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=PR.N_SIMS)
    ap.add_argument("--json", default=str(PR.DATA / "todays_games.json"))
    args = ap.parse_args()

    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    games = payload.get("games") or []
    if not games:
        print("no games in the slate file; nothing to serve")
        return
    specs = sorted(games, key=lambda g: (g.get("start_et") or "99:99",
                                         g.get("away_team") or ""))
    for s in specs:
        for side in ("away", "home"):
            s[f"{side}_lineup"] = [tuple(x) for x in
                                   (s.get(f"{side}_lineup") or [])]

    P = PR.Predictor(progress=lambda m: print(m, flush=True))
    out = P.predict_slate(specs, n_sims=args.sims,
                          progress=lambda m: print(m, flush=True))
    path = PR.save_excel_slate(specs, out)
    n_conf = sum(1 for s in specs
                 if s.get("away_lineup_src", "mlb") == "mlb"
                 and s.get("home_lineup_src", "mlb") == "mlb")
    print(f"served {len(specs)} games at {args.sims} sims "
          f"({n_conf} confirmed-lineup, {len(specs) - n_conf} projected) "
          f"-> {path}")


if __name__ == "__main__":
    main()
