"""Team-game context tables for the residual heads: Elo, recent form,
rest/travel/schedule spots. NONE of this feeds the component models or
the sim — the simulator generates game structure from components; these
are exactly the reduced-form signals that belong in the residual heads,
where the sim probability stays the anchor and count coherence is never
at risk.

Writes Model/artifacts/stores/team_game_context.parquet: one row per
GamePk with away_/home_ prefixed columns, every value as-of strictly
before that game's first pitch (post-game updates happen after the row
is emitted; doubleheader game 2 legitimately sees game 1's result).

Usage:
    python Model/team_context.py --build
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ELO_K = 4.0            # per-game update size
ELO_HFA = 24.0         # home-field advantage, Elo points
ELO_REGRESS = 1 / 3    # season-start regression toward 1500
FORM_G = 10            # recent-form window (games)
PYTH_EXP = 1.83
TZ_DEG_PER_HR = 15.0
EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return float(2 * EARTH_R_KM * np.arcsin(np.sqrt(a)))


def _baseruns(h, bb, hbp, tb, hr, ab):
    """Team-game BaseRuns expectation (sequencing-free run estimate)."""
    a = h + bb + hbp - hr
    b = 1.1 * (1.4 * tb - 0.6 * h - 3.0 * hr + 0.1 * (bb + hbp))
    c = max(ab - h, 1.0)
    return a * b / max(b + c, 1e-9) + hr


def _team_game_maps():
    """(GamePk, Team) -> bullpen line (ER/outs/NP), starter outs, and
    the batting line BaseRuns needs. Everything is post-game truth,
    consumed only via the running pregame state."""
    gp = F.read_csv("mlb_game_pitching.csv",
                    usecols=["GamePk", "Team", "GS", "IP", "ER", "NP"])
    gp["GamePk"] = pd.to_numeric(gp["GamePk"], errors="coerce")
    ip = pd.to_numeric(gp["IP"], errors="coerce").fillna(0)
    gp["outs"] = (ip.astype(int) * 3 + round((ip % 1) * 10)).astype(float)
    gp["ER"] = pd.to_numeric(gp["ER"], errors="coerce").fillna(0)
    gp["NP"] = pd.to_numeric(gp["NP"], errors="coerce").fillna(0)
    gs = pd.to_numeric(gp["GS"], errors="coerce").fillna(0)
    pen = gp[gs == 0].groupby(["GamePk", "Team"]).agg(
        er=("ER", "sum"), outs=("outs", "sum"),
        np_=("NP", "sum"))
    st = gp[gs == 1].groupby(["GamePk", "Team"])["outs"].sum()
    gb = F.read_csv("mlb_game_batting.csv",
                    usecols=["GamePk", "Team", "AB", "H", "2B", "3B",
                             "HR", "BB", "HBP", "TB", "R"])
    gb["GamePk"] = pd.to_numeric(gb["GamePk"], errors="coerce")
    for c in ("AB", "H", "2B", "3B", "HR", "BB", "HBP", "TB", "R"):
        gb[c] = pd.to_numeric(gb[c], errors="coerce").fillna(0)
    bat = gb.groupby(["GamePk", "Team"])[["AB", "H", "HR", "BB", "HBP",
                                          "TB", "R"]].sum()
    return pen.to_dict("index"), st.to_dict(), bat.to_dict("index")


def _load_games():
    games = F.read_csv("mlb_games.csv")
    games["Date"] = pd.to_datetime(games["Date"])
    games["GamePk"] = pd.to_numeric(games["GamePk"], errors="coerce")
    for c in ("AwayScore", "HomeScore"):
        games[c] = pd.to_numeric(games[c], errors="coerce")
    return games.dropna(subset=["GamePk", "Date"]).sort_values(
        ["Date", "GamePk"]).reset_index(drop=True)


def _inputs():
    parks = F.read_csv("mlb_ballparks.csv")
    coords = {}
    for _, r in parks.iterrows():
        try:
            coords[r["Ballpark"]] = (float(r["Lat"]), float(r["Lon"]))
        except (TypeError, ValueError):
            pass
    return coords, _team_game_maps()


def _context_rows(games, coords, pen_map, st_map, bat_map):
    """The pre-game state machine, one emitted row per game. Games with
    NaN scores contribute no post-game update — which is exactly how an
    upcoming slate rides through: state from history, no result."""
    elo, season_seen, state = {}, {}, {}
    rows = []
    for g in games.itertuples(index=False):
        date, pk = g.Date, int(g.GamePk)
        side_vals = {}
        for side, team in (("away", g.AwayTeam), ("home", g.HomeTeam)):
            if season_seen.get(team) != g.Season:
                elo[team] = (1500.0 + (1 - ELO_REGRESS)
                             * (elo.get(team, 1500.0) - 1500.0))
                season_seen[team] = g.Season
                st = state.get(team)
                if st:
                    st["recent"].clear()
                    st["pen"].clear()
                    st["np3"].clear()
                    st["st_outs"].clear()
                    st["bsr"].clear()
                    st.update(rf=0.0, ra=0.0, w=0.0, n=0.0)
            st = state.setdefault(team, dict(
                prev_date=None, prev_venue=None, prev_dn=None,
                recent=deque(maxlen=FORM_G), rf=0.0, ra=0.0, w=0.0,
                n=0.0, pen=deque(maxlen=30), np3=deque(),
                st_outs=deque(maxlen=20), bsr=deque(maxlen=20)))
            while st["np3"] and (date - st["np3"][0][0]).days > 3:
                st["np3"].popleft()
            rest = ((date - st["prev_date"]).days
                    if st["prev_date"] is not None else np.nan)
            travel, tzs = 0.0, 0.0
            if st["prev_venue"] is not None:
                c0 = coords.get(st["prev_venue"])
                c1 = coords.get(g.Venue)
                if c0 and c1:
                    travel = haversine_km(c0[0], c0[1], c1[0], c1[1])
                    tzs = (c1[1] - c0[1]) / TZ_DEG_PER_HR
            dan = float(rest == 1 and st["prev_dn"] == "night"
                        and str(g.DayNight) == "day") \
                if rest == rest else 0.0
            rec = list(st["recent"])
            side_vals[side] = dict(
                elo=elo[team], rest=rest, b2b=float(rest == 1),
                travel_km=travel, tz_shift=tzs, day_after_night=dan,
                w10=np.mean([r[0] for r in rec]) if rec else np.nan,
                rf10=np.mean([r[1] for r in rec]) if rec else np.nan,
                ra10=np.mean([r[2] for r in rec]) if rec else np.nan,
                pyth=(st["rf"] ** PYTH_EXP
                      / max(st["rf"] ** PYTH_EXP
                            + st["ra"] ** PYTH_EXP, 1e-9)
                      if st["n"] >= 20 else np.nan),
                gp=st["n"],
                pen_era=(27.0 * sum(e for e, _ in st["pen"])
                         / max(sum(o for _, o in st["pen"]), 1.0)
                         if st["pen"] else np.nan),
                pen_np3=float(sum(v for _, v in st["np3"])),
                st_outs=(float(np.mean(st["st_outs"]))
                         if st["st_outs"] else np.nan),
                bsr_luck=(float(np.mean(st["bsr"]))
                          if st["bsr"] else np.nan))
        row = {"GamePk": pk, "Date": str(date.date()),
               "Season": int(g.Season),
               "away_team": g.AwayTeam, "home_team": g.HomeTeam}
        for side in ("away", "home"):
            for k, v in side_vals[side].items():
                row[f"{side}_{k}"] = v
        row["elo_diff"] = (side_vals["home"]["elo"]
                           - side_vals["away"]["elo"] + ELO_HFA)
        rows.append(row)

        # ---- post-game updates (never visible to this game's row)
        if g.HomeScore == g.HomeScore and g.AwayScore == g.AwayScore:
            hw = float(g.HomeScore > g.AwayScore)
            e_home = 1.0 / (1.0 + 10 ** (-(elo[g.HomeTeam] + ELO_HFA
                                           - elo[g.AwayTeam]) / 400))
            elo[g.HomeTeam] += ELO_K * (hw - e_home)
            elo[g.AwayTeam] -= ELO_K * (hw - e_home)
            for team, mine, theirs, won in (
                    (g.HomeTeam, g.HomeScore, g.AwayScore, hw),
                    (g.AwayTeam, g.AwayScore, g.HomeScore, 1 - hw)):
                st = state[team]
                st["recent"].append((won, mine, theirs))
                st["rf"] += mine
                st["ra"] += theirs
                st["w"] += won
                st["n"] += 1
                pg_ = pen_map.get((pk, team))
                if pg_:
                    st["pen"].append((pg_["er"], pg_["outs"]))
                    st["np3"].append((date, pg_["np_"]))
                so_ = st_map.get((pk, team))
                if so_ is not None:
                    st["st_outs"].append(so_)
                bg_ = bat_map.get((pk, team))
                if bg_:
                    st["bsr"].append(bg_["R"] - _baseruns(
                        bg_["H"], bg_["BB"], bg_["HBP"], bg_["TB"],
                        bg_["HR"], bg_["AB"]))
        for team in (g.AwayTeam, g.HomeTeam):
            st = state[team]
            st["prev_date"] = date
            st["prev_venue"] = g.Venue
            st["prev_dn"] = str(g.DayNight)
    return rows, elo


def build():
    games = _load_games()
    coords, (pen_map, st_map, bat_map) = _inputs()
    rows, elo = _context_rows(games, coords, pen_map, st_map, bat_map)
    out = pd.DataFrame(rows)
    out.to_parquet(F.STORES / "team_game_context.parquet", index=False)
    print(f"team_game_context: {len(out):,} games; current elo spread "
          f"sd {np.std(list(elo.values())):.0f} pts", flush=True)
    return out


_SLATE_PK = 2_000_000_000    # pseudo GamePks sort after any real pk


def slate_context(specs):
    """Pre-game context for an upcoming slate (spec dicts as loaded
    from todays_games.json). Runs the identical state machine over all
    completed games plus one scoreless pseudo-row per slate game, and
    returns row dicts (away_*/home_*/elo_diff keys) aligned to specs
    order. Doubleheader game 2 sees game 1 only as same-day prev state
    (no result exists yet — the correct pre-game truth)."""
    games = _load_games()
    pseudo = []
    for i, sp in enumerate(specs):
        d = pd.Timestamp(sp["date"])
        pseudo.append(dict(
            GamePk=_SLATE_PK + i, Date=d,
            Season=int(sp.get("season") or d.year),
            AwayTeam=str(sp["away_team"]),
            HomeTeam=str(sp["home_team"]),
            Venue=sp.get("venue") or "",
            DayNight=sp.get("day_night") or "day",
            AwayScore=np.nan, HomeScore=np.nan))
    allg = pd.concat([games, pd.DataFrame(pseudo)],
                     ignore_index=True).sort_values(
        ["Date", "GamePk"]).reset_index(drop=True)
    coords, (pen_map, st_map, bat_map) = _inputs()
    rows, _ = _context_rows(allg, coords, pen_map, st_map, bat_map)
    byk = {r["GamePk"]: r for r in rows}
    return [byk[_SLATE_PK + i] for i in range(len(specs))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    if not ap.parse_args().build:
        ap.error("pass --build")
    build()
