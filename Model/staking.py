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
Ledger discipline (§9): money fields are never rewritten. A re-serve
appends a 'void' marker row for the audit trail AND stamps the
superseded original's Outcome='superseded' — the one in-place field
that makes it dead to settlement (a marker alone left the original
settling alongside its replacement, double-counting every re-served
bet). Player rows whose game went final without a stat line void on
the next settle run (the book's did-not-play rule).

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


def _key_norm(v):
    """One key field -> canonical string: '' for blank/None/NaN, ints
    without the float '.0' artifact ('777.0' == '777' == 777), floats
    without trailing zeros — so a CSV round trip (dtype=str load vs
    astype(str) write) can never break key equality."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    s = str(v).strip()
    if s in ("", "nan", "None", "none", "NaN", "<NA>"):
        return ""
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return s


def _bet_key(r):
    """Side-agnostic bet identity (§3: ONE bet per player/market/line
    per game). Team stays in the key for team-scoped markets (team
    totals) but NOT for h2h, where Team encodes the SIDE — an EV
    side-flip on re-serve must supersede the abandoned side, and both
    moneyline sides of one game are the same bet. Works on the fresh
    dict rows and the CSV-loaded Series alike via _key_norm."""
    get = r.get
    mkt = _key_norm(get("Market"))
    team = "" if mkt == "h2h" else _key_norm(get("Team"))
    pk = _key_norm(get("GamePk"))
    # pk-less rows (schedule-scrape hiccup slates): the matchup label
    # + game number stand in for the game identity, else every
    # unstamped h2h row on the slate shares one degenerate key and
    # distinct games cross-supersede. pk'd keys are unchanged.
    gm = ("", "") if pk != "" else (_key_norm(get("Game")),
                                    _key_norm(get("GNum")))
    return (_key_norm(get("Date")), pk,
            _key_norm(get("PlayerId")), team, mkt,
            _key_norm(get("Line"))) + gm


def _dec(american):
    p = O.american_to_prob(american)
    return None if p in (None, 0) else 1.0 / p


# Per-market stat access, evaluated LAZILY: the old eager dict literal
# touched every column of the box row (r.OUTS included), so any pitcher
# ledger row crashed the whole settle batch on the missing OUTS column
# even for markets that never needed it.
_PIT_STAT = {"pitcher_strikeouts": "SO", "pitcher_outs": "OUTS",
             "pitcher_hits_allowed": "H", "pitcher_walks": "BB",
             "pitcher_earned_runs": "ER"}
_BAT_STAT = {
    "batter_hits": lambda r: r.H,
    "batter_home_runs": lambda r: r.HR,
    "batter_total_bases": lambda r: r.TB,
    "batter_runs_scored": lambda r: r.R,
    "batter_rbis": lambda r: r.RBI,
    "batter_walks": lambda r: r.BB,
    "batter_stolen_bases": lambda r: r.SB,
    "batter_singles": lambda r: r.H - r["2B"] - r["3B"] - r.HR,
    "batter_doubles": lambda r: r["2B"],
    "batter_hits_runs_rbis": lambda r: r.H + r.R + r.RBI,
}


def _stat_of(r, market):
    """One box-score row's stat for `market`, or None when the market
    is unknown or a column it needs is absent (unresolvable, never a
    crash)."""
    col = _PIT_STAT.get(market)
    if col is not None:
        return r[col] if col in r.index else None
    fn = _BAT_STAT.get(market)
    if fn is None:
        return None
    try:
        return fn(r)
    except (AttributeError, KeyError):
        return None


def _with_outs(gp):
    """mlb_game_pitching.csv carries IP but no OUTS column; derive outs
    exactly as evaluate._load_boxes does (IP 5.2 -> 5*3+2 = 17) so the
    ledger settles pitcher ladders on the same numbers as the gate. A
    frame that already has OUTS (evaluate's, the tests') passes
    through untouched."""
    if "OUTS" in gp.columns or "IP" not in gp.columns:
        return gp
    ip = pd.to_numeric(gp.IP, errors="coerce").fillna(0)
    return gp.assign(OUTS=(ip.astype(int) * 3
                           + round((ip % 1) * 10)).astype(int))


def outcome_y(gb, gp, games_df, pk, pid, market, line, team=None):
    """Realized outcome for one priced market: 1/0, or None for
    void/unplayed AND pushes (stat exactly on an integer line). The
    single source of settlement truth — evaluate._odds_y delegates
    here so the gate and the ledger can never grade differently.
    `team` identifies the side for team-scoped markets (team_totals)."""
    if market == "team_totals":
        g = games_df[games_df.GamePk == pk]
        if g.empty or team is None:
            return None
        aw = pd.to_numeric(g.AwayScore.iloc[0], errors="coerce")
        hm = pd.to_numeric(g.HomeScore.iloc[0], errors="coerce")
        if pd.isna(aw) or pd.isna(hm):
            return None
        sc = hm if str(team) == str(g.HomeTeam.iloc[0]) else aw
        return None if sc == line else int(sc > line)
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
        stat = _stat_of(r.iloc[0], market)
        if stat is None or pd.isna(stat):
            return None
        return None if stat == line else int(stat > line)
    r = gb[(gb.GamePk == pk) & (gb.PlayerId == pid)]
    if r.empty or not (r.PA.iloc[0] > 0):
        return None
    stat = _stat_of(r.iloc[0], market)
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


def _bankroll():
    """Marked-to-market bankroll: START + cumulative settled PnL read
    from the ledger (§5: 'f applies to the current bankroll'). Falls
    back to START when there is no ledger or nothing has settled."""
    led = _load()
    if not len(led):
        return START_BANKROLL
    pnl = pd.to_numeric(led.PnL_units, errors="coerce").sum()
    return START_BANKROLL + (float(pnl) if pd.notna(pnl) else 0.0)


def enrich(bet_rows, date, mkt_fam):
    """Bets-sheet rows -> ledger rows at serve time (§3-§5 fields).
    Never both sides: only the higher-EV side of a bet identity
    survives (side-agnostic key, so both moneyline sides of one game
    collide too). Rows failing §3 record WHY in stake_capped_by and
    Status='track' — the evidence accrues either way."""
    passes = _eligible_families()
    bank = _bankroll()   # §5: stakes are a fraction of CURRENT bankroll
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
            stake = PER_BET_CAP * bank
        else:
            status, capped = "paper", ""
            stake = f_kelly * bank
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
        k = _bet_key(row)
        prev = best.get(k)
        if prev is None or ev > prev[0]:
            best[k] = (ev, row)
    return [row for _, row in best.values()]


def _load():
    """dtype=str + keep_default_na=False: every cell round-trips as a
    string ('' for blank), so key matching and the settle masks can
    never be broken by pandas NA inference (a totals row's blank Team
    once loaded as float NaN and silently defeated the supersede
    match for every game-market row)."""
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    return pd.read_csv(LEDGER, encoding="utf-8-sig", dtype=str,
                       keep_default_na=False)


def _write(df):
    LEDGER.parent.mkdir(exist_ok=True)
    tmp = LEDGER.with_name(LEDGER.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, LEDGER)


def append(rows):
    """Append serve-time rows. A re-serve of the same slate supersedes:
    any UNSETTLED row with the same side-agnostic bet identity (§3 —
    so an EV side-flip retires the abandoned side too) first gets a
    'void' marker row (audit trail, §9), and the original is stamped
    Outcome='superseded' so settle() can never grade both generations.
    A SETTLED row is history and is never superseded."""
    if not rows:
        return 0
    led = _load()
    new = pd.DataFrame(rows).astype(str)
    marks, sup = [], []
    if len(led):
        newkeys = {_bet_key(r) for r in rows}
        live = led.Status.isin(("paper", "track")) & (led.Outcome == "")
        for i in led.index[live]:
            if _bet_key(led.loc[i]) in newkeys:
                m = led.loc[i].copy()
                m["Status"] = "void"
                m["stake_units"] = "0.0"
                m["stake_capped_by"] = "superseded-by-re-serve"
                marks.append(m)
                sup.append(i)
        if sup:
            # neutralize the originals: a marker alone left the old
            # row inside settle()'s live mask, so every re-serve
            # double-counted outcomes/PnL/CLV in the permanent record.
            # Status keeps its serve-time value; the terminal Outcome
            # is what makes the row dead to settlement and report().
            led.loc[sup, "Outcome"] = "superseded"
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
    box scores and the final captured close, then void earlier dates'
    did-not-play stragglers. Idempotent; a bad row is skipped (logged)
    and can never abort the batch — everything that settled still
    writes. Unresolvable same-date rows stay open for the next run."""
    led = _load()
    if not len(led):
        return
    live = led.Status.isin(("paper", "track")) & (led.Outcome == "")
    # legacy re-serve stacks: before append() neutralized superseded
    # originals (and matched game-market keys at all), old generations
    # of a re-served bet stayed live. Newest generation wins; older
    # live duplicates of the same identity retire here, once, so the
    # historical ledger can never double-settle either.
    seen, dup = set(), []
    for i in reversed(led.index[live].tolist()):
        k = _bet_key(led.loc[i])
        if k in seen:
            dup.append(i)
        else:
            seen.add(k)
    if dup:
        led.loc[dup, "Outcome"] = "superseded"
        live = led.Status.isin(("paper", "track")) & (led.Outcome == "")
    mask = live & (led.Date == str(date))
    stale = live & (led.Date < str(date))
    if not (mask.any() or stale.any() or dup):
        return
    games_all = pd.read_csv(DATA / "mlb_games.csv", encoding="utf-8-sig")
    finals = games_all.dropna(subset=["AwayScore", "HomeScore"])
    games = finals[finals.Date == str(date)]
    gb_all = pd.read_csv(DATA / "mlb_game_batting.csv",
                         encoding="utf-8-sig")
    gb = gb_all[gb_all.Date == str(date)]
    gp_all = _with_outs(pd.read_csv(DATA / "mlb_game_pitching.csv",
                                    encoding="utf-8-sig"))
    gp = gp_all[gp_all.Date == str(date)]
    try:
        store = pd.read_csv(O.DEFAULT_STORE, encoding="utf-8-sig",
                            low_memory=False)
        store = store[store.Date == str(date)]
    except OSError:
        store = pd.DataFrame(columns=O.ODDS_COLUMNS)

    n_set, wl = 0, {"win": 0, "lose": 0, "push": 0}
    for i in led.index[mask]:
        try:
            out = _settle_one(led, i, games, gb, gp, store)
        except Exception as e:  # noqa: BLE001 — one bad row skips that
            # row only; the rows settled around it still write below
            print(f"ledger: settle skipped row {i} "
                  f"({led.loc[i, 'Market']}): {e}", flush=True)
            continue
        if out is not None:
            n_set += 1
            wl[out] += 1
    n_void = _void_dnp(led, stale, finals, gb_all, gp_all)
    if n_set or n_void or dup:
        _write(led[LEDGER_COLS])
    settled = led[led.Outcome.isin(("win", "lose", "push"))]
    if n_set or n_void or len(settled):
        pnl_all = pd.to_numeric(settled.PnL_units,
                                errors="coerce").sum()
        clv = pd.to_numeric(settled.CLV, errors="coerce").dropna()
        clv_txt = f"{clv.mean():+.4f}" if len(clv) else "n/a"
        print(f"ledger: {n_set} settled today ({wl['win']}-"
              f"{wl['lose']}-{wl['push']})"
              + (f", {n_void} DNP voided" if n_void else "")
              + (f", {len(dup)} legacy dupes retired" if dup else "")
              + f"; lifetime PnL {pnl_all:+.2f}u, bankroll "
              f"{START_BANKROLL + pnl_all:.1f}u, mean CLV {clv_txt}",
              flush=True)
    else:
        print("ledger: nothing to settle", flush=True)


def _settle_one(led, i, games, gb, gp, store):
    """Settle ledger row i in place (Outcome/PnL/CLV columns). Returns
    the outcome string when the row settled, None when it stays open."""
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
        g = (games[games.GamePk == pk] if pk is not None
             else games.iloc[0:0])
        if g.empty:
            return None
        hm = float(g.HomeScore.iloc[0]) > float(g.AwayScore.iloc[0])
        won = (str(r.Team) == str(g.HomeTeam.iloc[0])) == hm
        out = "win" if won else "lose"
    else:
        if pk is None or line is None:
            return None
        try:
            pid = int(float(r.PlayerId))
        except (TypeError, ValueError):
            pid = None
        tt_team = str(r.Team) if market == "team_totals" else None
        y = outcome_y(gb, gp, games, pk, pid, market, line, team=tt_team)
        if y is None:
            # exact-on-line = push; anything unresolvable stays open
            stat_known = not (games[games.GamePk == pk].empty)
            is_push = _is_push(gb, gp, games, pk, pid, market, line,
                               team=tt_team)
            if not (stat_known and is_push):
                return None
            out = "push"
        else:
            over = y == 1
            out = ("win" if (side == "Over") == over else "lose")
    dec = _dec(r.PriceAmerican)
    stake = float(r.stake_units or 0.0)
    pnl = {"win": stake * (dec - 1.0) if dec else 0.0,
           "lose": -stake, "push": 0.0}[out]
    _fill_clv(led, i, r, store, pk, market, side)
    led.loc[i, "Outcome"] = out
    led.loc[i, "PnL_units"] = f"{pnl:.4f}"
    led.loc[i, "SettledAt"] = pd.Timestamp.now().isoformat(
        timespec="seconds")
    return out


def _fill_clv(led, i, r, store, pk, market, side):
    """CLV vs the final captured close for THIS bet's market+side.

    Store keying (odds.ODDS_COLUMNS): player props by PlayerId; totals
    and h2h under the HOME club with blank PlayerId, h2h OverPrice =
    the home side. The ledger's totals rows carry Team='' and its h2h
    rows carry Team = Side = the bet's club, so the join goes through
    GamePk and the Game label ('AWY@HOM') — never r.Team equality
    against the store, which matched nothing for totals and away
    moneylines."""
    if not len(store):
        return
    grp = store[(store.Market == market)
                & (store.Line.map(_key_norm) == _key_norm(r.Line))]
    home = str(r.Game).split("@")[-1].strip()
    if _key_norm(r.PlayerId) != "":
        grp = grp[pd.to_numeric(grp.PlayerId, errors="coerce")
                  == float(r.PlayerId)]
    else:
        grp = grp[pd.to_numeric(grp.PlayerId, errors="coerce").isna()]
        if market == "h2h":
            grp = grp[grp.Team.astype(str) == home]
        elif market == "team_totals":
            grp = grp[grp.Team.astype(str) == str(r.Team)]
    # pk-exact on doubleheaders: without it the two games' prices merge
    # into one bogus fair. Legacy blank-pk store rows still match; an
    # ambiguous multi-pk group with no ledger pk stays blank — no CLV
    # beats wrong CLV.
    if len(grp):
        spk = grp.GamePk.map(_key_norm)
        if pk is not None:
            exact = grp[spk == str(pk)]
            grp = exact if len(exact) else grp[spk == ""]
        elif spk[spk != ""].nunique() > 1:
            grp = grp.iloc[0:0]
    fair = O.sharp_fair(grp.to_dict("records")) if len(grp) else None
    if fair is None:
        return
    # which store column is this bet's side? Over = the Over side for
    # totals/props, = the HOME club for h2h (an away bet takes the
    # complement of the two-sided no-vig fair)
    over_side = ((str(r.Team) == home) if market == "h2h"
                 else side == "Over")
    p_side = fair if over_side else 1 - fair
    led.loc[i, "p_close_final"] = f"{p_side:.6f}"
    imp = pd.to_numeric(r.implied, errors="coerce")
    if pd.notna(imp):
        led.loc[i, "CLV"] = f"{p_side - imp:.6f}"
    close = grp[grp.Book == r.Book]
    if len(close):
        col = "OverPrice" if over_side else "UnderPrice"
        led.loc[i, "CloseAmerican"] = str(close[col].iloc[0])


def _void_dnp(led, stale, finals, gb_all, gp_all):
    """Void (Status='void', Outcome='void', PnL 0) EARLIER dates'
    player rows whose game went final without a stat line for the
    player — the book's did-not-play rule. Without this a scratched
    player's row stays open forever (outcome_y returns None on every
    run). Same-date rows are deliberately exempt: a box score that
    lags tonight's final stays open for tomorrow's settle instead of
    voiding early."""
    n_void = 0
    for i in led.index[stale]:
        r = led.loc[i]
        try:
            pk = int(float(r.GamePk))
            pid = int(float(r.PlayerId))
        except (TypeError, ValueError):
            continue                  # game markets / no identity
        if finals[finals.GamePk == pk].empty:
            continue                  # not final — genuinely open
        box = (gp_all if str(r.Market).startswith("pitcher")
               else gb_all)
        gbox = box[box.GamePk == pk]
        if gbox.empty:
            continue                  # final in, box not scraped yet
        mine = gbox[gbox.PlayerId == pid]
        if len(mine):
            if str(r.Market).startswith("pitcher"):
                continue              # pitched: settle(date) grades it
            pa = (pd.to_numeric(mine.PA.iloc[0], errors="coerce")
                  if "PA" in mine.columns else None)
            if pa is None or pa > 0:
                continue              # batted: gradeable, stays open
        led.loc[i, "Status"] = "void"
        led.loc[i, "Outcome"] = "void"
        led.loc[i, "PnL_units"] = "0.0000"
        led.loc[i, "SettledAt"] = pd.Timestamp.now().isoformat(
            timespec="seconds")
        n_void += 1
    return n_void


def _is_push(gb, gp, games, pk, pid, market, line, team=None):
    """Distinguish push (stat == line, a real result) from
    unresolvable (no box line yet) for the None returns of outcome_y."""
    if market == "team_totals":
        g = games[games.GamePk == pk]
        if g.empty or team is None:
            return False
        sc = (float(g.HomeScore.iloc[0])
              if str(team) == str(g.HomeTeam.iloc[0])
              else float(g.AwayScore.iloc[0]))
        return sc == line
    if market == "totals":
        g = games[games.GamePk == pk]
        if g.empty:
            return False
        return (float(g.AwayScore.iloc[0])
                + float(g.HomeScore.iloc[0])) == line
    src = gp if market.startswith("pitcher") else gb
    r = src[(src.GamePk == pk) & (src.PlayerId == pid)]
    if r.empty:
        return False
    if not market.startswith("pitcher") and not (r.PA.iloc[0] > 0):
        return False
    stat = _stat_of(r.iloc[0], market)
    return (stat is not None and not pd.isna(stat)
            and float(stat) == line)


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
