"""Paper-trading staking ledger — STAKING_DESIGN.md §3-§6 + §9 as code.

Status: PAPER ONLY. Nothing here authorizes real money (§7 must grade
first, and its criteria are being re-anchored). The point of shipping
the ledger NOW is evidence accrual: every served Bets row is recorded
at its captured price with the §4-shrunk probability, then settled with
outcome, realized PnL at the pre-registered stake, and CLV vs the final
close — the record §7.2 requires and win-counts can never provide.

Right-sizing (2026-07-23, per the audit roadmap): no family holds a
gate PASS today, so §5/§6 stakes are structurally ZERO for months —
rows carry Status='track' with the reason. The per-bet cap is
implemented; the §6 per-game/per-family/per-slate cap ORDERING (which
only matters once stakes are nonzero) is deferred to the first PASS.
The §10 constants below are pre-registered — do not change them before
the forward evidence grades.

Ledger: Ledger/staking_ledger.csv at the project root — kept OUT of
Model/artifacts on purpose so the planned artifacts relocation can
never move the one append-only permanent record off the synced path.
Append-only: corrections and re-serves write 'void' marker rows, never
edits (§9).

CLI:
    python Model/staking.py --settle 2026-07-24   # after Tools/4 grades
    python Model/staking.py --report
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import odds as O  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = Path(__file__).resolve().parent / "artifacts"
LEDGER = ROOT / "Ledger" / "staking_ledger.csv"
DATA = ROOT / "Data"

# §4-§6 pre-registered constants (§10: not changeable before grading)
SHRINK_W = 0.5          # p_bet = 0.5*p_model + 0.5*p_close
EDGE_FLOOR = 0.03       # §3.3, matches the gate's edge-bucket dead zone
LAMBDA = 0.25           # quarter-Kelly
PER_BET_CAP = 0.01      # min(f, 1% of bankroll)
START_BANKROLL = 100.0  # units; bankroll = START + cumsum(settled PnL)

LEDGER_COLS = [
    "Date", "Game", "GNum", "GamePk", "PlayerId", "PlayerName", "Team",
    "Market", "Prop", "Side", "Line", "Book", "PriceAmerican",
    "CapturedAt", "p_model", "p_close", "p_bet", "implied", "edge",
    "EV", "f_kelly", "stake_units", "stake_capped_by", "Status",
    "CloseAmerican", "p_close_final", "CLV", "Outcome", "PnL_units",
    "SettledAt",
]

_KEY = ["Date", "GamePk", "PlayerId", "Team", "Market", "Line", "Side"]


def _dec(american):
    p = O.american_to_prob(american)
    return None if p in (None, 0) else 1.0 / p


def outcome_y(gb, gp, games_df, pk, pid, market, line):
    """Realized outcome for one priced market: 1/0, or None for
    void/unplayed AND pushes (stat exactly on an integer line). The
    single source of settlement truth — evaluate._odds_y delegates
    here so the gate and the ledger can never grade differently."""
    if market in ("h2h", "totals"):
        g = games_df[games_df.GamePk == pk]
        if g.empty:
            return None
        aw = pd.to_numeric(g.AwayScore.iloc[0], errors="coerce")
        hm = pd.to_numeric(g.HomeScore.iloc[0], errors="coerce")
        if pd.isna(aw) or pd.isna(hm):
            return None
        if market == "h2h":
            return int(hm > aw)
        return None if aw + hm == line else int(aw + hm > line)
    if market.startswith("pitcher"):
        r = gp[(gp.GamePk == pk) & (gp.PlayerId == pid)]
        if r.empty:
            return None
        r = r.iloc[0]
        stat = {"pitcher_strikeouts": r.SO, "pitcher_outs": r.OUTS,
                "pitcher_hits_allowed": r.H, "pitcher_walks": r.BB,
                "pitcher_earned_runs": r.ER}.get(market)
        if stat is None or pd.isna(stat):
            return None
        return None if stat == line else int(stat > line)
    r = gb[(gb.GamePk == pk) & (gb.PlayerId == pid)]
    if r.empty or not (r.PA.iloc[0] > 0):
        return None
    r = r.iloc[0]
    stat = {"batter_hits": r.H, "batter_home_runs": r.HR,
            "batter_total_bases": r.TB, "batter_runs_scored": r.R,
            "batter_rbis": r.RBI, "batter_walks": r.BB,
            "batter_stolen_bases": r.SB,
            "batter_singles": r.H - r["2B"] - r["3B"] - r.HR,
            "batter_doubles": r["2B"],
            "batter_hits_runs_rbis": r.H + r.R + r.RBI}.get(market)
    if stat is None or pd.isna(stat):
        return None
    return None if stat == line else int(stat > line)


def _eligible_families():
    """Families holding a current gate PASS (§2), from the persisted
    verdict table. A missing report means NOTHING is eligible — the
    honest default, printed once so it can't pass silently."""
    p = ART / "market_gate_report.csv"
    if not p.exists():
        print("staking: no market_gate_report.csv — all rows tracked, "
              "none staked (run evaluate --gate)", flush=True)
        return set()
    rep = pd.read_csv(p)
    return set(rep.loc[rep.verdict == "PASS", "family"])


def enrich(bet_rows, date, mkt_fam):
    """Bets-sheet rows -> ledger rows at serve time (§3-§5 fields).
    Never both sides: only the higher-EV side of a market survives.
    Rows failing §3 record WHY in stake_capped_by and Status='track' —
    the evidence accrues either way."""
    passes = _eligible_families()
    best = {}
    for r in bet_rows:
        implied = O.american_to_prob(r.get("Best Odds"))
        dec = _dec(r.get("Best Odds"))
        pm, pc = r.get("Model %"), r.get("Mkt %")
        if None in (implied, dec, pm, pc):
            continue
        p_bet = SHRINK_W * pm + (1 - SHRINK_W) * pc
        edge = p_bet - implied
        ev = p_bet * dec - 1.0
        fam = mkt_fam.get(r.get("_market"), r.get("_market"))
        f_kelly = max(0.0, LAMBDA * (p_bet * dec - 1.0) / (dec - 1.0))
        if fam not in passes:
            status, capped, stake = "track", "family-not-PASS", 0.0
        elif edge < EDGE_FLOOR:
            status, capped, stake = "track", f"edge<{EDGE_FLOOR}", 0.0
        elif ev <= 0:
            status, capped, stake = "track", "EV<=0", 0.0
        elif f_kelly > PER_BET_CAP:
            status, capped = "paper", "per-bet-1%"
            stake = PER_BET_CAP * START_BANKROLL
        else:
            status, capped = "paper", ""
            stake = f_kelly * START_BANKROLL
        row = {
            "Date": str(date), "Game": r.get("Game"),
            "GNum": r.get("G#"), "GamePk": r.get("GamePk"),
            "PlayerId": r.get("PlayerId"),
            "PlayerName": r.get("Player"), "Team": r.get("Team"),
            "Market": r.get("_market"), "Prop": r.get("Prop"),
            "Side": r.get("Side"), "Line": r.get("Line"),
            "Book": r.get("Book"), "PriceAmerican": r.get("Best Odds"),
            "CapturedAt": pd.Timestamp.now().isoformat(
                timespec="seconds"),
            "p_model": round(float(pm), 6),
            "p_close": round(float(pc), 6),
            "p_bet": round(float(p_bet), 6),
            "implied": round(float(implied), 6),
            "edge": round(float(edge), 6), "EV": round(float(ev), 6),
            "f_kelly": round(float(f_kelly), 6),
            "stake_units": round(float(stake), 4),
            "stake_capped_by": capped, "Status": status,
            "CloseAmerican": "", "p_close_final": "", "CLV": "",
            "Outcome": "", "PnL_units": "", "SettledAt": "",
        }
        k = (row["Date"], row["GamePk"], row["PlayerId"], row["Team"],
             row["Market"], row["Line"])
        prev = best.get(k)
        if prev is None or ev > prev[0]:
            best[k] = (ev, row)
    return [row for _, row in best.values()]


def _load():
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    return pd.read_csv(LEDGER, encoding="utf-8-sig", dtype=str)


def _write(df):
    LEDGER.parent.mkdir(exist_ok=True)
    tmp = LEDGER.with_name(LEDGER.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, LEDGER)


def append(rows):
    """Append serve-time rows. A re-serve of the same slate supersedes:
    any UNSETTLED row with the same key first gets a 'void' marker row
    (append-only, §9 — never an edit), then the fresh row lands. A
    SETTLED row is history and is never superseded."""
    if not rows:
        return 0
    led = _load()
    new = pd.DataFrame(rows).astype(str)
    marks = []
    if len(led):
        live = led[(led.Status.isin(("paper", "track")))
                   & (led.Outcome.fillna("") == "")]
        newkeys = set(map(tuple, new[_KEY].itertuples(index=False,
                                                      name=None)))
        for _, r in live.iterrows():
            if tuple(r[k] for k in _KEY) in newkeys:
                m = r.copy()
                m["Status"] = "void"
                m["stake_units"] = "0.0"
                m["stake_capped_by"] = "superseded-by-re-serve"
                marks.append(m)
    out = pd.concat([led] + ([pd.DataFrame(marks)] if marks else [])
                    + [new], ignore_index=True)
    _write(out[LEDGER_COLS])
    n_paper = int((new.Status == "paper").sum())
    print(f"ledger: +{len(new)} rows ({n_paper} paper, "
          f"{len(new) - n_paper} track"
          + (f", {len(marks)} superseded" if marks else "") + ")",
          flush=True)
    return len(new)


def settle(date):
    """Fill Outcome/PnL/CLV for `date`'s unsettled rows from the graded
    box scores and the final captured close. Idempotent; unresolvable
    rows stay open for the next run."""
    led = _load()
    if not len(led):
        return
    mask = ((led.Date == str(date))
            & (led.Outcome.fillna("") == "")
            & led.Status.isin(("paper", "track")))
    if not mask.any():
        return
    games = pd.read_csv(DATA / "mlb_games.csv", encoding="utf-8-sig")
    games = games[games.Date == str(date)].dropna(
        subset=["AwayScore", "HomeScore"])
    gb = pd.read_csv(DATA / "mlb_game_batting.csv", encoding="utf-8-sig")
    gb = gb[gb.Date == str(date)]
    gp = pd.read_csv(DATA / "mlb_game_pitching.csv",
                     encoding="utf-8-sig")
    gp = gp[gp.Date == str(date)]
    try:
        store = pd.read_csv(O.DEFAULT_STORE, encoding="utf-8-sig",
                            low_memory=False)
        store = store[store.Date == str(date)]
    except OSError:
        store = pd.DataFrame(columns=O.ODDS_COLUMNS)

    n_set, wl = 0, {"win": 0, "lose": 0, "push": 0}
    for i in led.index[mask]:
        r = led.loc[i]
        try:
            pk = int(float(r.GamePk))
        except (TypeError, ValueError):
            pk = None
        market, side = r.Market, str(r.Side)
        try:
            line = float(r.Line)
        except (TypeError, ValueError):
            line = None
        if market == "h2h":
            g = games[games.GamePk == pk] if pk else games.iloc[0:0]
            if g.empty:
                continue
            hm = float(g.HomeScore.iloc[0]) > float(g.AwayScore.iloc[0])
            won = (str(r.Team) == str(g.HomeTeam.iloc[0])) == hm
            out = "win" if won else "lose"
        else:
            if pk is None or line is None:
                continue
            try:
                pid = int(float(r.PlayerId))
            except (TypeError, ValueError):
                pid = None
            y = outcome_y(gb, gp, games, pk, pid, market, line)
            if y is None:
                # exact-on-line = push; anything unresolvable stays open
                stat_known = not (games[games.GamePk == pk].empty)
                is_push = _is_push(gb, gp, games, pk, pid, market, line)
                if not (stat_known and is_push):
                    continue
                out = "push"
            else:
                over = y == 1
                out = ("win" if (side == "Over") == over else "lose")
        dec = _dec(r.PriceAmerican)
        stake = float(r.stake_units or 0.0)
        pnl = {"win": stake * (dec - 1.0) if dec else 0.0,
               "lose": -stake, "push": 0.0}[out]
        # CLV vs the final captured close for THIS market+side
        grp = store[(store.Market == market)
                    & (store.Line.astype(str) == str(r.Line))]
        if str(r.PlayerId) not in ("", "nan", "None"):
            grp = grp[pd.to_numeric(grp.PlayerId, errors="coerce")
                      == float(r.PlayerId)]
        else:
            grp = grp[(grp.Team == r.Team)
                      & pd.to_numeric(grp.PlayerId,
                                      errors="coerce").isna()]
        fair = O.sharp_fair(grp.to_dict("records")) if len(grp) else None
        if fair is not None:
            p_side = fair if side in ("Over", str(r.Team)) else 1 - fair
            led.loc[i, "p_close_final"] = f"{p_side:.6f}"
            led.loc[i, "CLV"] = f"{p_side - float(r.implied):.6f}"
            close = grp[grp.Book == r.Book]
            if len(close):
                col = ("OverPrice" if side in ("Over", str(r.Team))
                       else "UnderPrice")
                led.loc[i, "CloseAmerican"] = str(close[col].iloc[0])
        led.loc[i, "Outcome"] = out
        led.loc[i, "PnL_units"] = f"{pnl:.4f}"
        led.loc[i, "SettledAt"] = pd.Timestamp.now().isoformat(
            timespec="seconds")
        n_set += 1
        wl[out] += 1
    if n_set:
        _write(led[LEDGER_COLS])
    settled = led[led.Outcome.isin(("win", "lose", "push"))]
    pnl_all = pd.to_numeric(settled.PnL_units, errors="coerce").sum()
    clv = pd.to_numeric(settled.CLV, errors="coerce")
    print(f"ledger: {n_set} settled today ({wl['win']}-{wl['lose']}-"
          f"{wl['push']}); lifetime PnL {pnl_all:+.2f}u, bankroll "
          f"{START_BANKROLL + pnl_all:.1f}u, mean CLV "
          f"{clv.mean():+.4f}" if n_set or len(settled) else
          "ledger: nothing to settle", flush=True)


def _is_push(gb, gp, games, pk, pid, market, line):
    """Distinguish push (stat == line, a real result) from
    unresolvable (no box line yet) for the None returns of outcome_y."""
    if market == "totals":
        g = games[games.GamePk == pk]
        if g.empty:
            return False
        return (float(g.AwayScore.iloc[0])
                + float(g.HomeScore.iloc[0])) == line
    if market.startswith("pitcher"):
        r = gp[(gp.GamePk == pk) & (gp.PlayerId == pid)]
        if r.empty:
            return False
        stat = {"pitcher_strikeouts": "SO", "pitcher_outs": "OUTS",
                "pitcher_hits_allowed": "H", "pitcher_walks": "BB",
                "pitcher_earned_runs": "ER"}.get(market)
        return stat is not None and float(r.iloc[0][stat]) == line
    r = gb[(gb.GamePk == pk) & (gb.PlayerId == pid)]
    if r.empty or not (r.PA.iloc[0] > 0):
        return False
    row = r.iloc[0]
    stat = {"batter_hits": row.H, "batter_home_runs": row.HR,
            "batter_total_bases": row.TB, "batter_runs_scored": row.R,
            "batter_rbis": row.RBI, "batter_walks": row.BB,
            "batter_stolen_bases": row.SB,
            "batter_singles": row.H - row["2B"] - row["3B"] - row.HR,
            "batter_doubles": row["2B"],
            "batter_hits_runs_rbis": row.H + row.R + row.RBI
            }.get(market)
    return stat is not None and float(stat) == line


def report():
    led = _load()
    settled = led[led.Outcome.isin(("win", "lose", "push"))]
    if not len(settled):
        print("ledger: no settled rows yet")
        return
    settled = settled.assign(
        pnl=pd.to_numeric(settled.PnL_units, errors="coerce"),
        clv=pd.to_numeric(settled.CLV, errors="coerce"))
    g = settled.groupby("Market").agg(
        n=("Outcome", "size"),
        won=("Outcome", lambda s: int((s == "win").sum())),
        push=("Outcome", lambda s: int((s == "push").sum())),
        pnl=("pnl", "sum"), mean_clv=("clv", "mean")).round(4)
    print(g.to_string())
    pnl = settled.pnl.sum()
    print(f"\nlifetime: {len(settled)} settled, PnL {pnl:+.2f}u, "
          f"bankroll {START_BANKROLL + pnl:.1f}u, mean CLV "
          f"{settled.clv.mean():+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settle", metavar="DATE")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.settle:
        settle(a.settle)
    elif a.report:
        report()
    else:
        ap.error("pass --settle DATE or --report")
