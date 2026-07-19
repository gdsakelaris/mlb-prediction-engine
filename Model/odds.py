"""Canonical schema and market map for the sportsbook odds store.

Data/mlb_odds.csv is the one file in Data/ that cannot be rebuilt after
the fact — pregame prices vanish once games start — so its schema lives
here, in one place, shared by the capture tool ("Tools/2) Scrape Odds.py")
and every future consumer. This module defines:

  ODDS_COLUMNS     the store's column set. One row per (Date, PlayerId,
                   Market, Line, Book) — for game-market rows (blank
                   PlayerId: h2h, totals, team_totals) the Team column
                   is the identity instead.
                   OverPrice/UnderPrice/CapturedAt
                   are the LATEST capture (the closing side);
                   OpenOverPrice/OpenUnderPrice/OpenCapturedAt the
                   EARLIEST (the opening side) — so intraday recaptures
                   tighten the close without destroying the open, and
                   line movement stays measurable.
  PROP_MARKET      engine market key -> Odds API market for batter props
  STARTER_MARKET   the same for starting-pitcher props
  API_DEFAULT_LINE implied line for Yes/No markets that omit the point
  DEFAULT_STORE    where the store lives
  sharp_fair       Pinnacle-preferring no-vig fair probability from a
                   group of captured book prices

The store is for GRADING and model-vs-market evaluation only — never a
feature input to the models.
"""

from pathlib import Path

ODDS_COLUMNS = [
    "Date", "GamePk", "Team", "PlayerId", "PlayerName", "Market", "Line",
    "OverPrice", "UnderPrice", "Book", "CapturedAt",
    "OpenOverPrice", "OpenUnderPrice", "OpenCapturedAt",
]

DEFAULT_STORE = Path(__file__).resolve().parents[1] / "Data" / "mlb_odds.csv"

# Engine market key -> {api: Odds API market key, line: the canonical
# threshold the engine prices for that key (books may post others; every
# posted line is captured regardless)}. Several keys share one API market
# and differ only by line (hit/hits2, tb2/tb3/tb4, ...): the capture tool
# requests each DISTINCT api once and stores every returned line.
# Batter strikeouts (bk*) exist as keys so an explicit --markets request
# works, but no US book posts them — the capture tool's auto sets skip
# that api. There is no triples market at the API.
PROP_MARKET = {
    "hr":     {"api": "batter_home_runs",      "line": 0.5},
    "hit":    {"api": "batter_hits",           "line": 0.5},
    "hits2":  {"api": "batter_hits",           "line": 1.5},
    "single": {"api": "batter_singles",        "line": 0.5},
    "double": {"api": "batter_doubles",        "line": 0.5},
    "tb2":    {"api": "batter_total_bases",    "line": 1.5},
    "tb3":    {"api": "batter_total_bases",    "line": 2.5},
    "tb4":    {"api": "batter_total_bases",    "line": 3.5},
    "run":    {"api": "batter_runs_scored",    "line": 0.5},
    "run2":   {"api": "batter_runs_scored",    "line": 1.5},
    "rbi":    {"api": "batter_rbis",           "line": 0.5},
    "rbi2":   {"api": "batter_rbis",           "line": 1.5},
    "hrr2":   {"api": "batter_hits_runs_rbis", "line": 1.5},
    "hrr3":   {"api": "batter_hits_runs_rbis", "line": 2.5},
    "hrr4":   {"api": "batter_hits_runs_rbis", "line": 3.5},
    "bb":     {"api": "batter_walks",          "line": 0.5},
    "sb":     {"api": "batter_stolen_bases",   "line": 0.5},
    "bk":     {"api": "batter_strikeouts",     "line": 0.5},
    "bk2":    {"api": "batter_strikeouts",     "line": 1.5},
    "bk3":    {"api": "batter_strikeouts",     "line": 2.5},
}

# Starter props are book-lined per pitcher (no single canonical threshold).
STARTER_MARKET = {
    "k":    {"api": "pitcher_strikeouts",   "line": None},
    "outs": {"api": "pitcher_outs",         "line": None},
    "pha":  {"api": "pitcher_hits_allowed", "line": None},
    "pbb":  {"api": "pitcher_walks",        "line": None},
    "per":  {"api": "pitcher_earned_runs",  "line": None},
}

# Markets some books post as Yes/No without a point: the implied line.
API_DEFAULT_LINE = {
    "batter_home_runs": 0.5,
    "batter_stolen_bases": 0.5,
}

# Alternate-line API markets -> the base market they are stored AS. The
# engine prices deep lines (2+ hits, 3+/4+ TB, 2+ RBI, the pitcher
# ladders, every total) that books post mostly as alternates, so the
# capture tool requests these alongside each base market and normalizes
# the name away at parse time — one Market key per family in the store,
# and the gate grades alternate lines with zero downstream changes.
# Alternates whose extra lines the engine never prices (HR 1.5+, BB/SB
# deep lines) are deliberately absent: requesting them buys nothing.
# team_totals is its own market (Team column = which club the total is
# for); h2h and totals are event-featured and need no alternate to cover
# the main line.
ALT_MARKET = {
    "batter_hits_alternate": "batter_hits",
    "batter_total_bases_alternate": "batter_total_bases",
    "batter_rbis_alternate": "batter_rbis",
    "pitcher_strikeouts_alternate": "pitcher_strikeouts",
    "pitcher_hits_allowed_alternate": "pitcher_hits_allowed",
    "pitcher_walks_alternate": "pitcher_walks",
    "alternate_totals": "totals",
    "alternate_team_totals": "team_totals",
}

# Books treated as the sharp reference, in preference order: their no-vig
# price is the market-truth proxy a de-vig consensus defers to.
SHARP_BOOKS = ("pinnacle",)


def american_to_prob(price):
    """American odds -> implied probability (vig included); None when
    blank, zero, or non-finite (a NaN here would otherwise flow through
    every downstream de-vig untouched)."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p != p or p in (float("inf"), float("-inf")) or p == 0:
        return None
    return -p / (-p + 100.0) if p < 0 else 100.0 / (p + 100.0)


def no_vig(over_price, under_price):
    """Two-sided american prices -> (fair over prob, fair under prob) by
    proportional de-vig, or None when either side is missing."""
    po, pu = american_to_prob(over_price), american_to_prob(under_price)
    if po is None or pu is None or po + pu <= 0:
        return None
    return po / (po + pu), pu / (po + pu)


def sharp_fair(rows):
    """Fair over-probability for one (player, market, line) from captured
    book rows — dicts (or namedtuples) with OverPrice/UnderPrice/Book.

    A sharp book's two-sided no-vig price wins outright when present
    (SHARP_BOOKS order); otherwise the median no-vig price across the
    remaining two-sided books. None when no book has both sides."""
    fair = {}
    for r in rows:
        get = r.get if hasattr(r, "get") else lambda k, _r=r: getattr(_r, k)
        f = no_vig(get("OverPrice"), get("UnderPrice"))
        if f is not None:
            fair.setdefault(str(get("Book")).lower(), []).append(f[0])
    for book in SHARP_BOOKS:
        if book in fair:
            ps = fair[book]
            return sum(ps) / len(ps)
    ps = sorted(p for v in fair.values() for p in v)
    if not ps:
        return None
    mid = len(ps) // 2
    return ps[mid] if len(ps) % 2 else (ps[mid - 1] + ps[mid]) / 2.0
