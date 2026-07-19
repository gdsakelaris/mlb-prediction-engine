"""Capture real sportsbook lines into Data/mlb_odds.csv (canonical schema in
Model/odds.py), so the model can be graded against the real market instead
of only a naive base rate.

Source: The Odds API (https://the-odds-api.com). Set an API key once:

    setx ODDS_API_KEY "your_key_here"      # Windows, new shells pick it up
    # or pass --key on the command line

Then run it near game time to snapshot that day's closing-ish lines:

    python "Tools/2) Scrape Odds.py"                        # today, everything (default)
    python "Tools/2) Scrape Odds.py" --markets props        # player props only, no game markets
    python "Tools/2) Scrape Odds.py" --markets hr,pk,totals --date 2026-07-04
    python "Tools/2) Scrape Odds.py" --include-started      # includes started games (live odds)
                                     

The default 'all' captures every posted player prop the model predicts plus the
game markets (totals = over/under runs, h2h = moneyline / winner, team_totals
= per-club run totals — the engine's tt family), from the
DEFAULT_BOOKS list: the prop-posting US books plus Pinnacle, the sharp book a
de-vig consensus prefers as its reference — same credit
cost as the plain regions=us default. Every market drags its alternate-line
partner along (O.ALT_MARKET: hits/TB/RBI, pitcher K/hits/walks ladders,
totals and team totals) and the rows are stored under the BASE market name —
the engine prices those deep lines, and per-event calls bill on markets
RETURNED, so alternates and unposted markets (batter Ks) ride free until a
book actually posts them. Run it twice a day and it just works: player-prop markets cost 1 credit PER GAME while the
game markets are one flat bulk call, and a rerun near first pitch SKIPS games
already underway (their pregame line is final), so the second run only pays for
games not yet started. write_store keeps one row per (Date, PlayerId, Market,
Line, Book) holding BOTH ends of the capture history — the earliest capture in
the Open* columns and the latest in OverPrice/UnderPrice — so a morning run
pins the opening price, a rerun near first pitch tightens the closing one, and
the move between them (steam, line value) stays measurable forever.

IMPORTANT on the free tier: it covers moneyline/totals for current & upcoming
games, but player props ("additional markets") and historical snapshots are
paid add-ons, so the default set needs a paid key. The honest workflow is GOING
FORWARD capture — run this daily and you accumulate your own closing-line
history for model-vs-market grading to use. A purchased
historical dump or a scrape from elsewhere also works: just write rows in the
Model/odds.py schema to the same CSV.

Without a key the script explains this and exits 0 (nothing to do), so it is
safe to wire into a scheduler before you have props access.
"""

import argparse
import csv
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import odds as O  # noqa: E402  (canonical schema + market map)
# "1) Get Todays Games" starts with a digit and contains spaces, so a plain
# `import` is a syntax error — importlib loads it by name string instead
import importlib  # noqa: E402
_gtg = importlib.import_module("1) Get Todays Games")
NICKNAME_TO_ABBREV = _gtg.NICKNAME_TO_ABBREV
build_name_index = _gtg.build_name_index
norm_name = _gtg.norm_name

API = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
PROP_APIS = sorted({m["api"] for m in O.PROP_MARKET.values()} |
                   {m["api"] for m in O.STARTER_MARKET.values()})
GAME_MARKETS = ["totals", "h2h"]   # featured: ONE bulk call for the slate

# Game markets that exist only on the per-event endpoint; they ride
# along with each game's prop request (team_totals = per-club run
# totals, the engine's tt family; the alternate_* keys are extra lines
# that normalize into totals / team_totals at parse time, O.ALT_MARKET).
EVENT_GAME_APIS = ("team_totals", "alternate_totals",
                   "alternate_team_totals")

# Markets the model prices that no US book currently posts (batter
# strikeouts, once listed here). Per-event calls bill on markets
# RETURNED, not requested, so an unposted market rides along in the
# auto sets at zero cost — the day a book starts posting it, it is
# captured with no code change. Populate this set only if a market
# ever becomes expensive to request.
UNPOSTED_APIS = set()

# Which books to pay for. The Odds API bills the bookmakers param in GROUPS OF
# TEN (any group = one region-equivalent), so this list of 9 costs exactly what
# regions=us does while swapping in Pinnacle — the sharp, low-hold book (eu
# region) preferred as the de-vig reference. It replaces the
# whole-region 'us' default: the 8 US books here are the ones that actually
# post player props; the region's other books (mybookieag,
# lowvig, betus — soft offshore, game markets only) are dropped to keep the
# group at 9, leaving ONE free slot before every request doubles in cost.
# Pinnacle posts HR, total bases, pitcher outs/hits/ER
# pregame — whatever else it posts near first pitch is captured at no extra
# cost (per-event calls charge on markets RETURNED x book-groups).
DEFAULT_BOOKS = ",".join((
    "draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers",
    "fanatics", "bovada", "betonlineag", "pinnacle"))

# Player-prop markets cost 1 credit PER GAME per region; the game markets
# (totals, h2h) come from ONE flat bulk call (2 credits total, not per game).
# The default 'all' pulls every posted player prop the model predicts (10
# batter + 5 pitcher) plus game markets, so a 14-game slate is ~14x15 + 2 =
# ~212 credits. A rerun near first pitch skips games already underway (their
# pregame line is final), so it only pays for games not yet started. Use
# 'props' for player props only, or a list like hr,pk,totals to spend less.
DEFAULT_MARKETS = "all"


KEY_FILE = ROOT / ".odds_api_key"


def load_key(cli_key=None):
    """Resolve the API key from, in order: --key, a local .odds_api_key file
    (gitignored), then $ODDS_API_KEY. The FILE is checked BEFORE the env var on
    purpose: an ODDS_API_KEY set in an already-open terminal goes stale, and a
    wrong stale key silently burns the wrong account's quota.
    The file is the deliberate, persistent source of truth; set
    it once and it always wins. Use --key for a one-off override."""
    if cli_key:
        return cli_key
    if KEY_FILE.exists():
        k = KEY_FILE.read_text(encoding="utf-8").strip()
        if k:
            return k
    return os.environ.get("ODDS_API_KEY")


def _api_of(k):
    """Odds API market key for one of our market keys (batter prop, pitcher
    prop, team totals, or a game market that is already its own api key)."""
    if k in O.PROP_MARKET:
        return O.PROP_MARKET[k]["api"]
    if k in O.STARTER_MARKET:
        return O.STARTER_MARKET[k]["api"]
    if k in ("tt", "team_totals"):
        return "team_totals"
    return k


def resolve_markets(spec):
    """A market spec -> the ordered, de-duped list of Odds API market keys
    to request. `spec` is 'all' (every player prop + game markets incl.
    team totals), 'props' (every player prop the model predicts), or a
    comma list of our keys: batter props (PROP_MARKET), pitcher props
    (STARTER_MARKET), team_totals (alias tt), and/or the game markets
    totals, h2h. Every requested market drags its alternate-line partner
    along (O.ALT_MARKET) — the engine prices the deep lines books post
    as alternates, and per-event calls bill on markets RETURNED, so an
    alternate nobody posts costs nothing."""
    prop_keys = list(O.PROP_MARKET) + list(O.STARTER_MARKET)
    valid = prop_keys + GAME_MARKETS + ["team_totals", "tt"]
    s = spec.strip().lower()
    if s in ("all", "props"):
        auto = prop_keys + (GAME_MARKETS + ["team_totals"]
                            if s == "all" else [])
        keys = [k for k in auto if _api_of(k) not in UNPOSTED_APIS]
    else:
        keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in valid]
    if unknown:
        raise SystemExit(f"unknown market(s): {', '.join(unknown)}. valid: "
                         f"{', '.join(valid)}, or 'props'/'all'.")
    alt_of = {}
    for alt, base in O.ALT_MARKET.items():
        alt_of.setdefault(base, []).append(alt)
    apis, seen = [], set()
    for k in keys:
        api = _api_of(k)
        for a in [api] + alt_of.get(api, []):
            if a not in seen:
                seen.add(a)
                apis.append(a)
    return apis


def full_name_to_abbrev(full):
    """The Odds API uses full club names ('New York Yankees'); map to our
    abbreviations by matching the nickname suffix ('Yankees', 'Red Sox')."""
    for nick, ab in NICKNAME_TO_ABBREV.items():
        if full == nick or full.endswith(" " + nick) or full.endswith(nick):
            return ab
    return None


def _side(name):
    """Normalize an outcome label to 'over'/'under' (props use Over/Under or
    Yes/No; both mean the same 'does it clear the line' bet)."""
    n = (name or "").strip().lower()
    if n in ("over", "yes"):
        return "over"
    if n in ("under", "no"):
        return "under"
    return None


def make_resolver(idx, home_abbr, away_abbr):
    """name -> (PlayerId, team_abbr) via the layered build_name_index tiers.
    A team-scoped hit (exact / collapsed / last-name+first-initial) on either
    club gives the player's REAL team; a globally-unique name still resolves
    the PlayerId but leaves the team unknown (None) rather than guessing it
    (the prop outcome doesn't say which club the player is on). Loose tiers
    are consulted only when they land on a single pid, same as resolve()."""
    def r(name):
        n = norm_name(name)
        if not n:
            return None, None
        toks = n.split()
        c = "".join(toks)
        for ab in (home_abbr, away_abbr):
            if not ab:
                continue
            pid = idx["exact"].get((ab, n))
            if pid is not None:
                return pid, ab
            for m, key in ((idx["team_col"], (ab, c)),
                           (idx["team_lfi"], (ab, toks[-1], toks[0][0]))):
                hit = m.get(key)
                if hit and len(hit) == 1:
                    return next(iter(hit)), ab
        for m, key in ((idx["glob"], n), (idx["col"], c)):
            hit = m.get(key)
            if hit and len(hit) == 1:
                return next(iter(hit)), None
        return None, None
    return r


def parse_event_props(event, resolver, date, gamepk, captured_at,
                      prop_apis=PROP_APIS):
    """Pure: an Odds API event JSON -> list of canonical prop rows (one per
    player/market/line/book, with matched over & under prices). Unresolved
    player names are skipped. Alternate-line markets are stored under
    their BASE market name (O.ALT_MARKET) — one Market key per family in
    the store, extra lines indistinguishable from main-line captures."""
    rows = []
    for bk in event.get("bookmakers", []):
        book = bk.get("key")
        for mkt in bk.get("markets", []):
            if mkt.get("key") not in prop_apis:
                continue
            market = O.ALT_MARKET.get(mkt["key"], mkt["key"])
            # group the market's outcomes by (player, line) -> {over, under}
            pairs = {}
            for o in mkt.get("outcomes", []):
                side = _side(o.get("name"))
                player = o.get("description")
                if side is None or not player:
                    continue
                line = o.get("point")
                if line is None:  # Yes/No HR markets omit the point
                    line = O.API_DEFAULT_LINE.get(market, 0.5)
                key = (player, float(line))
                pairs.setdefault(key, {})[side] = o.get("price")
            for (player, line), pr in pairs.items():
                pid, team = resolver(player)
                if pid is None:
                    continue
                rows.append({
                    "Date": date, "GamePk": gamepk, "Team": team,
                    "PlayerId": pid, "PlayerName": player, "Market": market,
                    "Line": line, "OverPrice": pr.get("over"),
                    "UnderPrice": pr.get("under"), "Book": book,
                    "CapturedAt": captured_at,
                })
    return rows


def parse_event_games(event, home_abbr, away_abbr, date, gamepk, captured_at,
                      keys=("totals", "h2h")):
    """Game-level rows in the same schema, PlayerId blank. totals (every
    posted line, alternates normalized in) -> Team = home club; h2h ->
    OverPrice = home, UnderPrice = away, Team = home club; team_totals ->
    Team = the club the total is FOR (the outcome's description names
    it). `keys` picks which raw API markets to read — the bulk featured
    call passes the default, the per-event call passes EVENT_GAME_APIS."""
    rows = []
    for bk in event.get("bookmakers", []):
        book = bk.get("key")
        for mkt in bk.get("markets", []):
            k = mkt.get("key")
            if k not in keys:
                continue
            outs = mkt.get("outcomes", [])
            if k in ("totals", "alternate_totals"):
                pairs = {}
                for o in outs:
                    side = _side(o.get("name"))
                    if side is None or o.get("point") is None:
                        continue
                    pairs.setdefault(float(o["point"]),
                                     {})[side] = o.get("price")
                for line, pr in pairs.items():
                    if "over" in pr and "under" in pr:
                        rows.append(_game_row(date, gamepk, home_abbr,
                                              "totals", line, pr["over"],
                                              pr["under"], book,
                                              captured_at))
            elif k in ("team_totals", "alternate_team_totals"):
                pairs = {}
                for o in outs:
                    side = _side(o.get("name"))
                    ab = full_name_to_abbrev(o.get("description") or "")
                    if side is None or ab is None or o.get("point") is None:
                        continue
                    pairs.setdefault((ab, float(o["point"])),
                                     {})[side] = o.get("price")
                for (ab, line), pr in pairs.items():
                    if "over" in pr and "under" in pr:
                        rows.append(_game_row(date, gamepk, ab,
                                              "team_totals", line,
                                              pr["over"], pr["under"],
                                              book, captured_at))
            elif k == "h2h":
                price = {full_name_to_abbrev(o.get("name")): o.get("price")
                         for o in outs}
                rows.append(_game_row(date, gamepk, home_abbr, "h2h", None,
                                      price.get(home_abbr),
                                      price.get(away_abbr), book,
                                      captured_at))
    return rows


def _game_row(date, gamepk, team, market, line, over, under, book, captured):
    return {"Date": date, "GamePk": gamepk, "Team": team, "PlayerId": "",
            "PlayerName": "", "Market": market, "Line": line,
            "OverPrice": over, "UnderPrice": under, "Book": book,
            "CapturedAt": captured}


def fetch(url, params):
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    remain = r.headers.get("x-requests-remaining")
    return r.json(), remain


def write_store(rows, out):
    """Append rows, then collapse to one row per (Date, PlayerId, Market,
    Line, Book) keeping BOTH ends of the capture history: OverPrice/
    UnderPrice/CapturedAt hold the LATEST capture (the closing side) and
    OpenOverPrice/OpenUnderPrice/OpenCapturedAt the EARLIEST — so a
    morning run pins the open, a rerun near first pitch tightens the
    close, and the price move between them stays measurable forever.
    Rows written before the Open columns existed backfill their open from
    the one capture they hold.

    The store is the one file in Data/ that cannot be re-scraped after the
    fact (pregame lines are gone once games start), so it gets extra care:
    the existing copy is backed up to <dir>/backups/ before the rewrite,
    and the new file is written to a temp path then swapped in atomically —
    a crash mid-write can never truncate the only copy."""
    out = Path(out)
    existing = []
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    best = {}  # key -> [open (ts, over, under), close (ts, row)]
    for row in existing + rows:
        ident = str(row.get("PlayerId") or "").strip()
        if not ident:
            # game-market rows carry no PlayerId: the Team IS the
            # identity. Without it every game's h2h/totals rows on a
            # date collapsed to ONE surviving row per book (fixed
            # 2026-07-19; earlier slates lost all but one game).
            ident = "T:" + str(row.get("Team") or "")
        # a lineless market (h2h) is None in memory but "" once
        # round-tripped through the CSV — normalize or recaptures
        # never merge with their stored row
        line_v = row.get("Line")
        line_k = "" if line_v in (None, "") else str(line_v)
        key = (str(row.get("Date")), ident,
               row.get("Market"), line_k, row.get("Book"))
        close_ts = str(row.get("CapturedAt") or "")
        open_ = (str(row.get("OpenCapturedAt") or close_ts),
                 row.get("OpenOverPrice") or row.get("OverPrice"),
                 row.get("OpenUnderPrice") or row.get("UnderPrice"))
        prev = best.get(key)
        if prev is None:
            best[key] = [open_, (close_ts, row)]
            continue
        if open_[0] < prev[0][0]:
            prev[0] = open_
        if close_ts >= prev[1][0]:
            prev[1] = (close_ts, row)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        backup_dir = out.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(out, backup_dir / out.name)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=O.ODDS_COLUMNS)
        w.writeheader()
        for open_, (_, row) in best.values():
            merged = {c: row.get(c, "") for c in O.ODDS_COLUMNS}
            merged["OpenCapturedAt"] = open_[0]
            merged["OpenOverPrice"] = "" if open_[1] is None else open_[1]
            merged["OpenUnderPrice"] = "" if open_[2] is None else open_[2]
            w.writerow(merged)
    os.replace(tmp, out)
    return len(best)


def _commence(e):
    """An event's commence_time as a tz-aware UTC datetime, or None if absent
    or unparseable (treated as 'not yet started', so never skipped)."""
    t = e.get("commence_time")
    if not t:
        return None
    try:
        return dt.datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    except ValueError:
        return None


def _slate_date(e):
    """The LOCAL calendar date an event belongs to (first pitch converted to
    this machine's timezone), or None if commence_time is unusable. MLB slates
    are named by US-local date: a 9:40 PM PT game commences after midnight
    UTC, so bucketing by the raw UTC string would put West-Coast night games
    on TOMORROW's slate — today's --date would never fetch them and by
    tomorrow they are 'already started', i.e. never captured at all.
    Local-date bucketing matches the slate the model predicts
    and the Date the store rows carry."""
    c = _commence(e)
    return c.astimezone().date().isoformat() if c else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="slate date YYYY-MM-DD, the LOCAL calendar day of "
                         "first pitch (default today)")
    ap.add_argument("--key", default=None,
                    help="Odds API key; else $ODDS_API_KEY, else the local "
                         ".odds_api_key file (set once, survives forever)")
    ap.add_argument("--books", default=DEFAULT_BOOKS,
                    help="comma-separated Odds API bookmaker keys (billed in "
                         "groups of 10 = 1 region-equivalent). Default: the 8 "
                         "prop-posting US books + pinnacle. Pass '' to fall "
                         "back to whole-region fetching via --regions")
    ap.add_argument("--regions", default="us",
                    help="Odds API regions, used only with --books ''")
    ap.add_argument("--out", default=str(O.DEFAULT_STORE))
    ap.add_argument("--markets", default=DEFAULT_MARKETS,
                    help="comma-separated market keys (batter: "
                         + ", ".join(O.PROP_MARKET) + "; pitcher: "
                         + ", ".join(O.STARTER_MARKET) + "; game: "
                         + ", ".join(GAME_MARKETS) + "), or 'props' (all "
                         "player props) or 'all' (props + game markets). "
                         "Default: " + DEFAULT_MARKETS)
    ap.add_argument("--include-started", action="store_true",
                    help="also fetch games already underway (default: on "
                         "today's slate they are skipped — their pregame line "
                         "is final, so a rerun near first pitch pays only for "
                         "games not yet started)")
    args = ap.parse_args()
    args.key = load_key(args.key)
    markets = resolve_markets(args.markets)
    bulk_game = [m for m in markets if m in GAME_MARKETS]
    event_game = [m for m in markets if m in EVENT_GAME_APIS]
    prop_markets = [m for m in markets if m not in GAME_MARKETS
                    and m not in EVENT_GAME_APIS]

    if not args.key:
        print("No Odds API key found. Get one at https://the-odds-api.com, "
              "then pick ONE:\n"
              f"  persistent (recommended):  write the key into {KEY_FILE}\n"
              "     (one line, no quotes) — every run reads it, no env needed\n"
              "  env var:   setx ODDS_API_KEY \"...\"  then open a NEW terminal\n"
              "  one-off:   python \"Tools/2) Scrape Odds.py\" --key <key>\n"
              "Nothing to capture — exiting.")
        return  # exit 0: safe to schedule before you have a key

    captured_at = dt.datetime.now().isoformat(timespec="seconds")
    day = args.date
    try:
        events, remain = fetch(f"{API}/events", {"apiKey": args.key})
    except Exception as e:
        print(f"could not list events: {e}", file=sys.stderr)
        sys.exit(1)
    events = [e for e in events if _slate_date(e) == day]
    # bookmakers overrides regions at the API when both are sent, so exactly
    # one is used; billing counts each group of 10 books as one region.
    if args.books:
        scope = {"bookmakers": args.books}
        n_reg = (len([b for b in args.books.split(",") if b.strip()]) + 9) // 10
    else:
        scope = {"regions": args.regions}
        n_reg = len(args.regions.split(","))

    # Skip games already underway: their pregame prices are final, so a later
    # rerun spends credits only on games not yet started. write_store de-dupes
    # regardless, so this changes cost, never the stored data. Applied only to
    # today's slate (a past/future date is fetched in full).
    now = dt.datetime.now(dt.timezone.utc)
    pending = events
    if not args.include_started and day == dt.date.today().isoformat():
        pending = [e for e in events
                   if _commence(e) is None or _commence(e) > now]
    n_skip = len(events) - len(pending)
    pend_ids = {e.get("id") for e in pending}

    est = (len(prop_markets + event_game) * len(pending) * n_reg
           + len(bulk_game) * n_reg)
    print(f"{len(events)} MLB events on {day}"
          f"{f' ({n_skip} already started, skipped)' if n_skip else ''} "
          f"(api requests left: {remain})")
    print(f"markets [{', '.join(markets)}]: per-event x {len(pending)} "
          f"games{f' x {n_reg} region-equivs' if n_reg > 1 else ''}"
          f"{' + totals/h2h (1 bulk call)' if bulk_game else ''} "
          f"= at most ~{est} credits (billed on markets RETURNED — "
          f"alternates and unposted markets that come back empty are "
          f"free)")

    idx = build_name_index()
    all_rows, n_prop, n_game = [], 0, 0

    # Game markets (totals/h2h) are 'featured' markets on the bulk endpoint:
    # ONE call returns the whole slate for markets x regions credits (not per
    # game). Keep only pending games so a rerun never overwrites a captured
    # closing line with an in-play price.
    if bulk_game and pend_ids:
        try:
            slate, remain = fetch(
                f"{API}/odds",
                {"apiKey": args.key, **scope,
                 "markets": ",".join(bulk_game), "oddsFormat": "american"})
        except Exception as ex:
            print(f"  game markets fetch failed ({ex})", file=sys.stderr)
            slate = []
        for ev in slate:
            if ev.get("id") not in pend_ids:
                continue
            ha = full_name_to_abbrev(ev.get("home_team", ""))
            aa = full_name_to_abbrev(ev.get("away_team", ""))
            gm = parse_event_games(ev, ha, aa, day, "", captured_at)
            all_rows += gm
            n_game += len(gm)
        print(f"  game markets: {n_game} rows across the slate (left: {remain})")

    # Player props and per-event game markets (team totals, alternates)
    # live only on the per-event endpoint: one call per game covers both.
    ev_req = prop_markets + event_game
    base_req = [m for m in prop_markets if m not in O.ALT_MARKET]
    warned_422 = False
    for e in (pending if ev_req else []):
        home_abbr = full_name_to_abbrev(e.get("home_team", ""))
        away_abbr = full_name_to_abbrev(e.get("away_team", ""))
        resolver = make_resolver(idx, home_abbr, away_abbr)
        try:
            data, remain = fetch(
                f"{API}/events/{e['id']}/odds",
                {"apiKey": args.key, **scope,
                 "markets": ",".join(ev_req), "oddsFormat": "american"})
        except Exception as ex:
            # a 422 means some requested market key is unknown to the
            # API — salvage the game with the base prop markets rather
            # than losing the whole event's capture
            if "422" in str(ex) and base_req and base_req != ev_req:
                if not warned_422:
                    print("  API rejected the alternate/event-game "
                          "market list (422) — retrying base prop "
                          "markets only; check O.ALT_MARKET keys "
                          "against the Odds API docs", file=sys.stderr)
                    warned_422 = True
                try:
                    data, remain = fetch(
                        f"{API}/events/{e['id']}/odds",
                        {"apiKey": args.key, **scope,
                         "markets": ",".join(base_req),
                         "oddsFormat": "american"})
                except Exception as ex2:
                    print(f"  {away_abbr}@{home_abbr}: odds fetch failed "
                          f"({ex2})", file=sys.stderr)
                    continue
            else:
                print(f"  {away_abbr}@{home_abbr}: odds fetch failed "
                      f"({ex})", file=sys.stderr)
                continue
        pr = (parse_event_props(data, resolver, day, "", captured_at,
                                prop_apis=prop_markets)
              if prop_markets else [])
        gm = (parse_event_games(data, home_abbr, away_abbr, day, "",
                                captured_at, keys=EVENT_GAME_APIS)
              if event_game else [])
        all_rows += pr + gm
        n_prop += len(pr)
        n_game += len(gm)
        print(f"  {away_abbr}@{home_abbr}: {len(pr)} prop + {len(gm)} "
              f"game rows (left: {remain})")

    total = write_store(all_rows, args.out)
    print(f"\nwrote {n_prop} prop + {n_game} game rows this run; "
          f"store now holds {total} rows -> {args.out}")


if __name__ == "__main__":
    main()
