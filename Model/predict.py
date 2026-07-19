"""Serve path: slate specs -> simulated games -> workbook.

Predictor loads the trained artifacts and the feature warehouse once,
then predict_slate() prices any list of game specs (the GUI's or the
backtester's):

  1. resolve the spec into player rows (9 batters a side from the posted
     lineup, filled from the club's last posted order when short;
     starters; the bullpen from roster + recent-usage availability)
  2. assemble as-of features for every (pitcher, batter, times-through-
     order) matchup through features.assemble_features — the SAME code
     path training used — and evaluate the calibrated A1/A2 models
  3. build the starter hazard grids, steal matrices, pattern bank and
     latent sigmas into a sim.GamePrep
  4. run the vectorized sim and count every market off the tensor

save_excel_slate() writes the workbook the Tools expect (sheets: Batter
Props, Pitching Props, Games, Bets) to Predictions/, and persists the
per-sim tensors beside it so any new market can be priced later without
re-simming.
"""

import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F     # noqa: E402
import sim               # noqa: E402
import odds as O         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
ART = ROOT / "Model" / "artifacts"
PRED_DIR = ROOT / "Predictions"

N_SIMS = 20000
MAX_PEN = 8

K_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
OUT_LINES = [14.5, 15.5, 16.5, 17.5, 18.5]
PHA_LINES = [3.5, 4.5, 5.5, 6.5]
PBB_LINES = [0.5, 1.5, 2.5]
PER_LINES = [1.5, 2.5, 3.5, 4.5]
TOTAL_LINES = [6.5, 7.5, 8.5, 9.5, 10.5]
TEAM_TOTAL_LINES = [2.5, 3.5, 4.5, 5.5]

# workbook look (matches the established output format; 8-digit ARGB so
# the stored colors byte-match the format the grader expects)
NAVY = "FF041E42"
BETS_HEADER = "FF1E7B34"
BETS_ROW_TINT = "FFE7F3E2"      # the Bets rows (grading paints winners)
ROOKIE_RED = "FFC00000"
ROOKIE_G = 50                   # career games under this -> red + note
MIN_EV = 0.05                   # Bets sheet inclusion floor

BAT_BINARIES = [
    ("HR", lambda t, s: t[..., s["HR"]] >= 1),
    ("Hit", lambda t, s: t[..., s["H"]] >= 1),
    ("2+ Hits", lambda t, s: t[..., s["H"]] >= 2),
    ("Single", lambda t, s: t[..., s["B1"]] >= 1),
    ("Double", lambda t, s: t[..., s["B2"]] >= 1),
    ("Triple", lambda t, s: t[..., s["B3"]] >= 1),
    ("2+ TB", lambda t, s: _tb(t, s) >= 2),
    ("3+ TB", lambda t, s: _tb(t, s) >= 3),
    ("4+ TB", lambda t, s: _tb(t, s) >= 4),
    ("Run", lambda t, s: t[..., s["R"]] >= 1),
    ("2+ Runs", lambda t, s: t[..., s["R"]] >= 2),
    ("RBI", lambda t, s: t[..., s["RBI"]] >= 1),
    ("2+ RBI", lambda t, s: t[..., s["RBI"]] >= 2),
    ("H+R+RBI 2+", lambda t, s: _hrr(t, s) >= 2),
    ("H+R+RBI 3+", lambda t, s: _hrr(t, s) >= 3),
    ("H+R+RBI 4+", lambda t, s: _hrr(t, s) >= 4),
    ("BB", lambda t, s: t[..., s["BB"]] >= 1),
    ("SB", lambda t, s: t[..., s["SB"]] >= 1),
    ("K", lambda t, s: t[..., s["K"]] >= 1),
    ("2+ K", lambda t, s: t[..., s["K"]] >= 2),
    ("3+ K", lambda t, s: t[..., s["K"]] >= 3),
]


def _tb(t, s):
    return (t[..., s["B1"]] + 2 * t[..., s["B2"]] + 3 * t[..., s["B3"]]
            + 4 * t[..., s["HR"]])


def _hrr(t, s):
    return t[..., s["H"]] + t[..., s["R"]] + t[..., s["RBI"]]


class _Stores:
    def __init__(self, raw):
        self.raw = raw


def _load_raw():
    """The GUI-facing raw frames (contract: games/rosters/gb/gp/parks/
    umps)."""
    games = pd.read_csv(DATA / "mlb_games.csv", encoding="utf-8-sig",
                        low_memory=False)
    games["Date"] = pd.to_datetime(games["Date"])
    ros = pd.read_csv(DATA / "mlb_rosters.csv", encoding="utf-8-sig")
    gb = pd.read_csv(DATA / "mlb_game_batting.csv", encoding="utf-8-sig",
                     low_memory=False,
                     usecols=["GamePk", "Season", "Date", "PlayerId",
                              "Name", "Team", "BattingOrder", "Position",
                              "PA"])
    gb["Date"] = pd.to_datetime(gb["Date"])
    gp = pd.read_csv(DATA / "mlb_game_pitching.csv", encoding="utf-8-sig",
                     low_memory=False,
                     usecols=["GamePk", "Season", "Date", "PlayerId",
                              "Name", "Team", "GS", "GF", "IP", "BF",
                              "NP"])
    gp["Date"] = pd.to_datetime(gp["Date"])
    parks = pd.read_csv(DATA / "mlb_ballparks.csv", encoding="utf-8-sig")
    umps = pd.read_csv(DATA / "mlb_umpires.csv", encoding="utf-8-sig")
    return {"games": games, "rosters": ros, "gb": gb, "gp": gp,
            "parks": parks, "umps": umps}


class Predictor:
    def __init__(self, progress=None):
        tick = progress or (lambda m: None)
        tick("loading artifacts...")
        # artifacts trained by `python Model/train.py` pickled the scaler
        # under __main__; resolve it for those runs
        import __main__ as _m
        if not hasattr(_m, "VectorScaler"):
            _m.VectorScaler = F.VectorScaler
        self.a1 = joblib.load(ART / "a1_model.joblib")
        self.a2 = joblib.load(ART / "a2_model.joblib")
        self.hz = joblib.load(ART / "hazard_model.joblib")
        self.sb = joblib.load(ART / "sb_models.joblib")
        self.latent = json.loads((ART / "latent.json").read_text())
        tick("loading feature stores...")
        self.fstores = F.load_stores()
        self.patterns = sim.load_patterns(F.STORES)
        self.preevents = json.loads(
            (F.STORES / "preevents.json").read_text())
        tick("loading data frames...")
        self.stores = _Stores(_load_raw())
        self._names = {}
        for df, idc, nc in ((self.stores.raw["gb"], "PlayerId", "Name"),
                            (self.stores.raw["gp"], "PlayerId", "Name"),
                            (self.stores.raw["rosters"], "PlayerId",
                             "Name")):
            self._names.update(dict(zip(df[idc], df[nc])))
        gb_ = self.stores.raw["gb"]
        self._career_g = gb_[pd.to_numeric(gb_.PA, errors="coerce") > 0] \
            .groupby("PlayerId").size().to_dict()
        gp_ = self.stores.raw["gp"]
        self._career_gp = gp_.groupby("PlayerId").size().to_dict()
        hand = pd.read_csv(DATA / "mlb_handedness.csv",
                           encoding="utf-8-sig")
        self._bats = dict(zip(hand.PlayerId, hand.Bats))
        self._throws = dict(zip(hand.PlayerId, hand.Throws))
        ros = self.stores.raw["rosters"]
        self._bats.update({p: b for p, b in zip(ros.PlayerId, ros.B)
                           if p not in self._bats})
        self._throws.update({p: t for p, t in zip(ros.PlayerId, ros.T)
                             if p not in self._throws})
        # club abbrev -> full name -> roster rows
        bs = pd.read_csv(DATA / "mlb_batting_stats.csv",
                         encoding="utf-8-sig",
                         usecols=["Year", "Team", "TeamName"])
        pairs = bs[bs.Year == bs.Year.max()].drop_duplicates()
        self._full_of = dict(zip(pairs.Team, pairs.TeamName))
        ros = self.stores.raw["rosters"]
        self._catchers = set(
            ros.loc[ros.Position == "Catcher", "PlayerId"].astype(int))
        # batter participation hazard table -> dense lookup array
        self.part_haz = None
        part_path = F.STORES / "participation.parquet"
        if part_path.exists():
            pt = pd.read_parquet(part_path)
            km = pt.groupby("k").agg(ev=("ev", "sum"), n=("n", "sum"))
            km = (km.ev / km.n.clip(lower=1)).to_dict()
            dense = np.zeros((6, 4, 3, 2, 2, 2))
            for kk in range(1, 7):
                dense[kk - 1, ...] = km.get(kk, 0.0)
            for r in pt.itertuples(index=False):
                dense[int(r.k) - 1, int(r.inn_b), int(r.margin_b),
                      int(r.lead), int(r.same), int(r.isc)] = r.rate
            self.part_haz = dense
        # cross-line-coherent output calibrators (one shared monotone
        # map per market FAMILY; fit by evaluate.py --fit-calibrators)
        cal_path = ART / "output_calibrators.joblib"
        self.calib = joblib.load(cal_path) if cal_path.exists() else {}
        tick("computing bullpen availability...")
        self._relief_exit = self._reliever_exit_table()
        tick("ready")

    # ------------------------------------------------------ helpers

    def _name(self, pid):
        return self._names.get(pid, str(pid))

    def _stand(self, bat_pid, p_throws):
        b = str(self._bats.get(bat_pid, "R"))
        if b == "S":
            return "L" if p_throws == "R" else "R"
        return b if b in ("L", "R") else "R"

    def _reliever_exit_table(self):
        gp = self.stores.raw["gp"]
        rel = gp[(pd.to_numeric(gp.GS, errors="coerce") == 0)].copy()
        ip = pd.to_numeric(rel.IP, errors="coerce").fillna(0)
        outs = (ip.astype(int) * 3 + round((ip % 1) * 10)).astype(int)
        tab = np.ones(11)
        for k in range(1, 11):
            at_least = (outs >= k).sum()
            more = (outs > k).sum()
            tab[k] = 1.0 - (more / max(at_least, 1))
        tab[0] = 0.0
        # exit decisions happen at inning breaks; the per-outs exit prob
        # is evaluated there with the stint's completed outs
        return tab

    def _pen_for(self, team_abbrev, date):
        """Available relievers with usage weights: roster bullpen minus
        arms that threw on both of the last two days (or 25+ pitches
        yesterday)."""
        gp = self.stores.raw["gp"]
        ros = self.stores.raw["rosters"]
        full = self._full_of.get(team_abbrev)
        pen_ids = ros.loc[(ros.Team == full)
                          & (ros.Position == "Bullpen"), "PlayerId"]
        pen_ids = [int(p) for p in pen_ids]
        d = pd.Timestamp(date)
        recent = gp[(gp.Team == team_abbrev)
                    & (gp.Date >= d - pd.Timedelta(days=30))
                    & (gp.Date < d)
                    & (pd.to_numeric(gp.GS, errors="coerce") == 0)]
        y1 = recent[recent.Date == d - pd.Timedelta(days=1)]
        y2 = recent[recent.Date == d - pd.Timedelta(days=2)]
        np1 = dict(zip(y1.PlayerId, pd.to_numeric(y1.NP,
                                                  errors="coerce")))
        used_both = set(y1.PlayerId) & set(y2.PlayerId)
        out = []
        counts = recent.groupby("PlayerId").size()
        for pid in pen_ids:
            if pid in used_both or np1.get(pid, 0) >= 25:
                continue
            out.append((pid, 1.0 + counts.get(pid, 0)))
        # game-log fallback when the depth chart is thin
        if len(out) < 5:
            for pid, n in counts.sort_values(ascending=False).items():
                if pid not in {p for p, _ in out} and len(out) < MAX_PEN:
                    out.append((int(pid), 1.0 + n))
        out.sort(key=lambda t: -t[1])
        return out[:MAX_PEN]

    def _last_lineup(self, team):
        gb = self.stores.raw["gb"]
        rows = gb[(gb.Team == team) & gb.BattingOrder.notna()].copy()
        rows["bo"] = pd.to_numeric(rows.BattingOrder, errors="coerce")
        rows = rows[rows.bo % 100 == 0]
        if rows.empty:
            return []
        last = rows[rows.Date == rows.Date.max()].sort_values("bo")
        return [int(p) for p in last.PlayerId.head(9)]

    # -------------------------------------------------- game assembly

    def prepare_game(self, spec, n_sims=N_SIMS):
        date = pd.Timestamp(spec["date"])
        season = int(spec.get("season") or date.year)
        away, home = spec["away_team"], spec["home_team"]

        def side_lineup(side, team):
            posted = sorted(spec.get(f"{side}_lineup") or [],
                            key=lambda x: x[1])
            pids = [int(p) for p, _ in posted][:9]
            if len(pids) < 9:
                for p in self._last_lineup(team):
                    if p not in pids:
                        pids.append(p)
                    if len(pids) == 9:
                        break
            while len(pids) < 9:
                pids.append(-1)          # league-average phantom
            return pids

        bat_away = side_lineup("away", away)
        bat_home = side_lineup("home", home)
        sp_away = int(spec.get("away_starter") or -2)
        sp_home = int(spec.get("home_starter") or -3)
        pen_away = self._pen_for(away, date)
        pen_home = self._pen_for(home, date)

        # rows: 0-17 lineups, 18/19 starters, 20-35 pens, 36/37 = the
        # generic BENCH bats substitutions bring in (league-average
        # phantoms; the sim maps them to batter-axis slots 18/19)
        players = (bat_away + bat_home + [sp_away, sp_home]
                   + [p for p, _ in pen_away]
                   + [-1] * (MAX_PEN - len(pen_away))
                   + [p for p, _ in pen_home]
                   + [-1] * (MAX_PEN - len(pen_home))
                   + [-1, -1])
        n_players = len(players)         # 18 + 2 + 16 + 2 bench
        bench_rows = (n_players - 2, n_players - 1)
        pit_rows = [18, 19] + list(range(20, 20 + MAX_PEN)) \
            + list(range(20 + MAX_PEN, 20 + 2 * MAX_PEN))
        pen_rows_away = list(range(20, 20 + len(pen_away)))
        pen_rows_home = list(range(20 + MAX_PEN,
                                   20 + MAX_PEN + len(pen_home)))
        bat_rows_all = list(range(18)) + list(bench_rows)

        # ---- matchup feature rows for A1/A2 (18 lineup + 2 bench bats)
        rows = []
        for prow in pit_rows:
            ppid = players[prow]
            pthrows = str(self._throws.get(ppid, "R"))
            pthrows = pthrows if pthrows in ("L", "R") else "R"
            p_team = away if (prow == 18 or prow in pen_rows_away) \
                else home
            for brow in bat_rows_all:
                # rows exist for every (pitcher, batter) pair so the
                # avec indexing stays rectangular; the sim only ever
                # reads cross-team pairs
                bpid = players[brow]
                bat_is_home = (brow >= 9) if brow < 18 else \
                    (brow == bench_rows[1])
                stand = self._stand(bpid, pthrows)
                for tto in (1, 2, 3):
                    rows.append(dict(
                        Date=date, Season=season, BatterId=bpid,
                        PitcherId=ppid, stand=stand, p_throws=pthrows,
                        tto=tto, home_bat=int(bat_is_home),
                        fld_team=p_team, Venue=spec.get("venue") or "",
                        DayNight=spec.get("day_night") or "day",
                        Temp=spec.get("temp"),
                        WindSpeed=spec.get("wind_speed"),
                        WindDir=spec.get("wind_dir") or "",
                        Condition=spec.get("condition") or "",
                        Humidity=spec.get("humidity"),
                        Pressure=spec.get("pressure"),
                        HpUmpId=spec.get("hp_ump_id"),
                        rest_p=self._rest(ppid, date),
                    ))
        rdf = pd.DataFrame(rows)
        X, _ = F.assemble_features(rdf, self.fstores)
        feats = self.a1["features"]
        X = X.reindex(columns=feats).astype(np.float32)
        p1 = self.a1["scaler"].transform(self.a1["model"].predict_proba(X))
        p2 = self.a2["scaler"].transform(self.a2["model"].predict_proba(X))
        avec = np.full((n_players, 20, 3, 8), np.nan)
        a2vec = np.full((n_players, 20, 3, 4), np.nan)
        i = 0
        for prow in pit_rows:
            for bi, brow in enumerate(bat_rows_all):
                for t in range(3):
                    avec[prow, bi, t] = p1[i]
                    a2vec[prow, bi, t] = p2[i]
                    i += 1

        # ---- hazard grids for the two starters
        haz = np.zeros((2, 41, 11))
        for si, (prow, ppid) in enumerate(((18, sp_away), (19, sp_home))):
            haz[si] = self._hazard_grid(ppid, date, season)

        # ---- steal matrices (runner PLAYER ROW vs every pitcher row)
        sb_att, sb_suc = self._sb_matrices(players, pit_rows,
                                           bat_rows_all, date, season,
                                           away, home)

        pen_order = np.zeros((1, 2, MAX_PEN), dtype=np.int16) - 1
        pen_order[0, 0, :len(pen_rows_away)] = pen_rows_away
        pen_order[0, 1, :len(pen_rows_home)] = pen_rows_home
        pen_order = np.broadcast_to(pen_order, (n_sims, 2, MAX_PEN))

        # participation covariates: starter bat side (0 L / 1 R / 2 S),
        # every pitcher's throwing hand, and the catcher flag per slot
        side_code = {"L": 0, "R": 1, "S": 2}
        bat_side = np.array(
            [side_code.get(str(self._bats.get(players[i], "R")), 1)
             for i in range(18)], dtype=np.int8)
        pit_throws = np.array(
            [0 if str(self._throws.get(p, "R")) == "L" else 1
             for p in players], dtype=np.int8)
        slot_is_c = np.array(
            [1 if players[i] in self._catchers else 0
             for i in range(18)], dtype=np.int8)

        lat = dict(self.latent)
        prep = sim.GamePrep(
            n_players=n_players, starters=[18, 19], avec=avec,
            a2vec=a2vec, haz_grid=haz,
            relief_exit=self._relief_exit, pen_order=pen_order,
            sb_att=sb_att, sb_suc=sb_suc, patterns=self.patterns,
            latent=lat, bench_rows=bench_rows,
            part_haz=self.part_haz, bat_side=bat_side,
            pit_throws=pit_throws, slot_is_c=slot_is_c,
            pre_wp=self.preevents["wp_pb_per_pa_runners_on"],
            pre_pk=self.preevents["pickoff_out_per_pa_runners_on"])
        meta = dict(players=players, names=[self._name(p) if p >= 0
                                            else "League Avg"
                                            for p in players],
                    season=season, away=away, home=home,
                    career_g=[self._career_g.get(p, 0) for p in players],
                    career_gp=[self._career_gp.get(p, 0)
                               for p in players])
        return prep, meta

    def _rest(self, ppid, date):
        gp = self.stores.raw["gp"]
        rows = gp[(gp.PlayerId == ppid) & (gp.Date < date)]
        if rows.empty:
            return np.nan
        return float((date - rows.Date.max()).days)

    def _hazard_grid(self, ppid, date, season):
        leash = self.fstores.get("panel_leash")
        lz = None
        if leash is not None:
            mine = leash[(leash.PlayerId == ppid)
                         & (leash.Date < pd.Timestamp(date))]
            if len(mine):
                lz = mine.sort_values("Date").iloc[-1]
        starts = float(lz["starts_d"]) if lz is not None else np.nan
        np_avg = (float(lz["np_sum_d"]) / max(starts, 1e-9)
                  if lz is not None else np.nan)
        bf_avg = (float(lz["bf_sum_d"]) / max(starts, 1e-9)
                  if lz is not None else np.nan)
        ppb = (np_avg / bf_avg) if np_avg and bf_avg and bf_avg > 0 \
            else 3.9
        bf = np.arange(41)
        runs = np.arange(11)
        B, R = np.meshgrid(bf, runs, indexing="ij")
        rows = pd.DataFrame(dict(
            bf=B.ravel(), cum_pitches=(B * ppb).ravel(),
            tto=(1 + B // 9).clip(1, 4).ravel(),
            inning=(1 + B / 4.3).ravel(), outs=1,
            score_diff=0, k_so_far=(B * 0.22).ravel(),
            br_so_far=(B * 0.30).ravel(), runs_so_far=R.ravel(),
            rest_p=self._rest(ppid, pd.Timestamp(date)),
            leash_np=np_avg, leash_bf=bf_avg, leash_starts=starts,
            season_idx=season - 2015))
        Xh = rows[self.hz["features"]].astype(np.float32)
        p = self.hz["iso"].predict(
            self.hz["model"].predict_proba(Xh)[:, 1])
        return p.reshape(41, 11)

    def _sb_matrices(self, players, pit_rows, bat_rows_all, date, season,
                     away, home):
        n_players = len(players)
        sprint = self.fstores["sprint"]
        yr = season
        spd = dict(zip(
            sprint.loc[sprint.Year == yr, "PlayerId"],
            sprint.loc[sprint.Year == yr, "SprintSpeed"]))
        ps = pd.read_csv(DATA / "mlb_pitching_stats.csv",
                         encoding="utf-8-sig",
                         usecols=["Year", "PlayerId", "SB", "CS", "PK",
                                  "TBF"])
        ps = ps[pd.to_numeric(ps.Year, errors="coerce") == season - 1]
        for c in ("SB", "CS", "PK", "TBF"):
            ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0)
        sbr = dict(zip(ps.PlayerId, ps.SB / ps.TBF.clip(lower=50)))
        csr = dict(zip(ps.PlayerId, (ps.CS + ps.PK)
                       / (ps.SB + ps.CS + ps.PK).clip(lower=3)))
        deft = self.fstores["defense_team"]
        drow = {t: deft[(deft.Team == t) & (deft.Year == season)]
                for t in (away, home)}

        def team_pop(t):
            r = drow[t]
            return (float(r.PopTime.iloc[0]) if len(r) and
                    pd.notna(r.PopTime.iloc[0]) else np.nan)

        era_new = float(season >= 2023)
        att = np.zeros((n_players, n_players))
        suc = np.zeros((n_players, n_players))
        shift = self.sb["success_logit_shift"][
            "post2023" if era_new else "pre2023"]
        scale = self.sb["scale"][
            "post2023" if era_new else "pre2023"]["attempt_scale"]
        rows_a, rows_s, pos = [], [], []
        for brow in bat_rows_all:
            bpid = players[brow]
            for prow in pit_rows:
                ppid = players[prow]
                # the catcher backing this pitcher belongs to his club
                fld = away if (prow == 18 or
                               20 <= prow < 20 + MAX_PEN) else home
                lhp = float(str(self._throws.get(ppid, "R")) == "L")
                rows_a.append(dict(
                    SprintSpeed=spd.get(bpid, np.nan)
                    if bpid >= 0 else np.nan,
                    sb_allowed_rate=sbr.get(ppid, np.nan),
                    cs_rate=csr.get(ppid, np.nan),
                    PopTime=team_pop(fld), CSAA=np.nan,
                    outs=1, score_close=1.0, era_new=era_new, lhp=lhp))
                rows_s.append(dict(
                    SprintSpeed=spd.get(bpid, np.nan),
                    cs_rate=csr.get(ppid, np.nan),
                    PopTime=team_pop(fld), CSAA=np.nan, lhp=lhp,
                    era_new=era_new))
                pos.append((brow, prow))
        A = pd.DataFrame(rows_a)[self.sb["att_features"]]
        S_ = pd.DataFrame(rows_s)[self.sb["suc_features"]]
        pa_ = self.sb["attempt"].predict_proba(A)[:, 1] * scale
        p_raw = self.sb["success"].predict_proba(S_)[:, 1]
        logit = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6)
                       / np.clip(1 - p_raw, 1e-6, 1))
        ps_ = 1 / (1 + np.exp(-(logit + shift)))
        for (brow, prow), a_, s_ in zip(pos, pa_, ps_):
            att[brow, prow] = a_
            suc[brow, prow] = s_
        return att, suc

    # ------------------------------------------------------ slate run

    def predict_slate(self, specs, n_sims=N_SIMS, progress=None):
        tick = progress or (lambda m: None)
        out = []
        for gi, spec in enumerate(specs):
            tick(f"game {gi + 1}/{len(specs)}: preparing...")
            prep, meta = self.prepare_game(spec, n_sims=n_sims)
            tick(f"game {gi + 1}/{len(specs)}: simulating...")
            res = sim.run(prep, n_sims=n_sims,
                          seed=int(pd.Timestamp(spec["date"]).toordinal())
                          * 100 + gi,
                          season=meta["season"],
                          is_dh_game=bool(spec.get("is_dh")))
            res["meta"] = meta
            res["spec"] = spec
            res["calib"] = self.calib
            out.append(res)
        return out


# -------------------------------------------------------- aggregation

# market families for the cross-line-coherent output calibrators: ONE
# shared monotone map per family (fit by evaluate.py --fit-calibrators),
# so calibration can never break P(2+ hits) <= P(1+ hit)
COL_FAM = {"HR": "hr", "Hit": "h", "2+ Hits": "h", "Single": "b1",
           "Double": "b2", "Triple": "b3", "2+ TB": "tb", "3+ TB": "tb",
           "4+ TB": "tb", "Run": "r", "2+ Runs": "r", "RBI": "rbi",
           "2+ RBI": "rbi", "H+R+RBI 2+": "hrr", "H+R+RBI 3+": "hrr",
           "H+R+RBI 4+": "hrr", "BB": "bb", "SB": "sb", "K": "bk",
           "2+ K": "bk", "3+ K": "bk"}
PIT_FAM = {"K > ": "pk", "Outs > ": "pout", "Hits > ": "pha",
           "BB > ": "pbb", "ER > ": "per"}
MKT_FAM = {"batter_hits": "h", "batter_home_runs": "hr",
           "batter_total_bases": "tb", "batter_runs_scored": "r",
           "batter_rbis": "rbi", "batter_walks": "bb",
           "batter_stolen_bases": "sb", "batter_singles": "b1",
           "batter_doubles": "b2", "batter_hits_runs_rbis": "hrr",
           "pitcher_strikeouts": "pk", "pitcher_outs": "pout",
           "pitcher_hits_allowed": "pha", "pitcher_walks": "pbb",
           "pitcher_earned_runs": "per", "h2h": "ml", "totals": "tot"}


def _cal(calib, fam, p):
    """Apply a family's shared monotone calibrator (identity when the
    family has none fitted). Output is clamped strictly inside (0, 1):
    no calibrator may ever emit a hard 0/1 into grading or EV math."""
    cal = (calib or {}).get(fam)
    if cal is None or p is None:
        return p
    return float(np.clip(cal.predict(np.array([p]))[0],
                         1e-6, 1 - 1e-6))


def _dec(american):
    """American odds -> decimal payout multiplier (stake included)."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(a) or a == 0:
        return None
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def game_frame(res):
    """One sim result -> the workbook rows for each sheet, with the
    family output calibrators applied to every probability."""
    t = res["tensor"]
    s = sim.SIDX
    calib = res.get("calib") or {}
    meta, spec = res["meta"], res["spec"]
    away, home = meta["away"], meta["home"]
    label = f"{away}@{home}"
    gnum = 2 if spec.get("dh_game2") else 1

    bat_rows = []
    for brow in range(18):
        pid = meta["players"][brow]
        if pid < 0:
            continue
        sub = t[:, brow, :]
        tb = (sub[:, s["B1"]] + 2 * sub[:, s["B2"]]
              + 3 * sub[:, s["B3"]] + 4 * sub[:, s["HR"]])
        hrr = sub[:, s["H"]] + sub[:, s["R"]] + sub[:, s["RBI"]]
        row = {
            "Game": label,
            "Team": away if brow < 9 else home,
            "G#": gnum,
            "Slot": brow % 9 + 1,
            "Name": meta["names"][brow],
            "ID": pid,
            "Career G": meta["career_g"][brow],
            "HR": float((sub[:, s["HR"]] >= 1).mean()),
            "xTB": round(float(tb.mean()), 2),
            "2+ TB": float((tb >= 2).mean()),
            "3+ TB": float((tb >= 3).mean()),
            "4+ TB": float((tb >= 4).mean()),
            "xH": round(float(sub[:, s["H"]].mean()), 2),
            "Hit": float((sub[:, s["H"]] >= 1).mean()),
            "2+ Hits": float((sub[:, s["H"]] >= 2).mean()),
            "xRBI": round(float(sub[:, s["RBI"]].mean()), 2),
            "RBI": float((sub[:, s["RBI"]] >= 1).mean()),
            "2+ RBI": float((sub[:, s["RBI"]] >= 2).mean()),
            "xR": round(float(sub[:, s["R"]].mean()), 2),
            "Run": float((sub[:, s["R"]] >= 1).mean()),
            "2+ Runs": float((sub[:, s["R"]] >= 2).mean()),
            "xHRR": round(float(hrr.mean()), 2),
            "H+R+RBI 2+": float((hrr >= 2).mean()),
            "H+R+RBI 3+": float((hrr >= 3).mean()),
            "H+R+RBI 4+": float((hrr >= 4).mean()),
            "SB": float((sub[:, s["SB"]] >= 1).mean()),
            "xSO": round(float(sub[:, s["K"]].mean()), 2),
            "K": float((sub[:, s["K"]] >= 1).mean()),
            "2+ K": float((sub[:, s["K"]] >= 2).mean()),
            "3+ K": float((sub[:, s["K"]] >= 3).mean()),
            "Single": float((sub[:, s["B1"]] >= 1).mean()),
            "Double": float((sub[:, s["B2"]] >= 1).mean()),
            "Triple": float((sub[:, s["B3"]] >= 1).mean()),
            "xBB": round(float(sub[:, s["BB"]].mean()), 2),
            "BB": float((sub[:, s["BB"]] >= 1).mean()),
        }
        for c, fam in COL_FAM.items():
            row[c] = _cal(calib, fam, row[c])
        bat_rows.append(row)

    pit_rows = []
    for prow, team, opp in ((18, away, home), (19, home, away)):
        pid = meta["players"][prow]
        if pid < 0:
            continue
        sub = t[:, prow, :]
        row = {"Game": label, "Team": team, "Opponent": opp, "G#": gnum,
               "Name": meta["names"][prow], "ID": pid,
               "xK": round(float(sub[:, s["PK_"]].mean()), 2)}
        for x in K_LINES:
            row[f"K > {x}"] = float((sub[:, s["PK_"]] > x).mean())
        row["xER"] = round(float(sub[:, s["PER"]].mean()), 2)
        for x in PER_LINES:
            row[f"ER > {x}"] = float((sub[:, s["PER"]] > x).mean())
        row["xOuts"] = round(float(sub[:, s["OUTS"]].mean()), 2)
        for x in OUT_LINES:
            row[f"Outs > {x}"] = float((sub[:, s["OUTS"]] > x).mean())
        row["xHits"] = round(float(sub[:, s["PH"]].mean()), 2)
        for x in PHA_LINES:
            row[f"Hits > {x}"] = float((sub[:, s["PH"]] > x).mean())
        row["xBB"] = round(float(sub[:, s["PBB"]].mean()), 2)
        for x in PBB_LINES:
            row[f"BB > {x}"] = float((sub[:, s["PBB"]] > x).mean())
        for c in row:
            for pref, fam in PIT_FAM.items():
                if c.startswith(pref):
                    row[c] = _cal(calib, fam, row[c])
        pit_rows.append(row)

    sc = res["score"]
    total = sc.sum(axis=1)
    home_wp = _cal(calib, "ml",
                   float((sc[:, 1] > sc[:, 0]).mean()))
    lineup_hr = float(t[:, :18, s["HR"]].sum(axis=1).mean())
    grow = {"Game": label, "Date": str(spec.get("date", "")),
            "Venue": spec.get("venue", ""),
            "Winner": home if home_wp >= 0.5 else away,
            "Win Prob": max(home_wp, 1 - home_wp)}
    for x in TEAM_TOTAL_LINES:
        grow[f"Away Runs > {x}"] = _cal(calib, "tt",
                                        float((sc[:, 0] > x).mean()))
    for x in TEAM_TOTAL_LINES:
        grow[f"Home Runs > {x}"] = _cal(calib, "tt",
                                        float((sc[:, 1] > x).mean()))
    grow["Away Score"] = round(float(sc[:, 0].mean()), 2)
    grow["Home Score"] = round(float(sc[:, 1].mean()), 2)
    grow["Total Runs"] = round(float(total.mean()), 2)
    for x in TOTAL_LINES:
        grow[f"Runs > {x}"] = _cal(calib, "tot",
                                   float((total > x).mean()))
    grow["Lineup HRs"] = round(lineup_hr, 2)
    grow["_home_wp"] = home_wp
    grow["_nrfi"] = float((res["runs_i1"].sum(axis=1) == 0).mean())
    return dict(bat=bat_rows, pit=pit_rows, game=grow)


GAME_COLS = (["Game", "Date", "Venue", "Winner", "Win Prob",
              "Away Score"]
             + [f"Away Runs > {x}" for x in TEAM_TOTAL_LINES]
             + ["Home Score"]
             + [f"Home Runs > {x}" for x in TEAM_TOTAL_LINES]
             + ["Total Runs"]
             + [f"Runs > {x}" for x in TOTAL_LINES]
             + ["Lineup HRs"])

_BAT_STAT = {"batter_hits": "H", "batter_home_runs": "HR",
             "batter_runs_scored": "R", "batter_rbis": "RBI",
             "batter_walks": "BB", "batter_stolen_bases": "SB",
             "batter_singles": "B1", "batter_doubles": "B2"}
_PIT_STAT = {"pitcher_strikeouts": "PK_", "pitcher_outs": "OUTS",
             "pitcher_hits_allowed": "PH", "pitcher_walks": "PBB",
             "pitcher_earned_runs": "PER"}
_PIT_WORD = {"pitcher_strikeouts": "strikeouts", "pitcher_outs": "outs",
             "pitcher_hits_allowed": "hits allowed",
             "pitcher_walks": "walks", "pitcher_earned_runs":
             "earned runs"}
def _prop_name(market, line):
    n = int(line + 0.5)
    if market == "batter_hits":
        return "1+ hit" if n == 1 else f"{n}+ hits"
    if market == "batter_home_runs":
        return "1+ HR" if n == 1 else f"{n}+ HR"
    if market == "batter_total_bases":
        return f"{n}+ total bases"
    if market == "batter_runs_scored":
        return "run scored" if n == 1 else f"{n}+ runs scored"
    if market == "batter_rbis":
        return f"{n}+ RBI"
    if market == "batter_hits_runs_rbis":
        return f"{n}+ H+R+RBI"
    if market == "batter_walks":
        return "1+ walk" if n == 1 else f"{n}+ walks"
    if market == "batter_stolen_bases":
        return "stolen base"
    if market == "batter_singles":
        return "1+ single" if n == 1 else f"{n}+ singles"
    if market == "batter_doubles":
        return "1+ double" if n == 1 else f"{n}+ doubles"
    if market in _PIT_WORD:
        return f"pitcher {_PIT_WORD[market]} o{line}"
    return market


def _best_price(group, col):
    """Most generous posted price for one side across books."""
    best, book = None, ""
    for _, r in group.iterrows():
        d = _dec(r.get(col))
        if d is None:
            continue
        if best is None or d > best[0]:
            best = (d, r.get(col))
            book = r.get("Book", "")
    if best is None:
        return None, None, ""
    return best[0], int(float(best[1])), book


def build_bets(out, date):
    """Model vs the day's captured odds: every side clearing MIN_EV at
    the best posted price, sorted by EV descending."""
    store = Path(O.DEFAULT_STORE)
    if not store.exists():
        return []
    try:
        odds = pd.read_csv(store, encoding="utf-8-sig", low_memory=False)
    except Exception:                                # noqa: BLE001
        return []
    odds = odds[odds.Date.astype(str) == str(date)]
    if odds.empty:
        return []

    s = sim.SIDX
    by_pid, by_game = {}, {}
    for res in out:
        meta = res["meta"]
        label = f'{meta["away"]}@{meta["home"]}'
        by_game[meta["home"]] = (res, label)
        for row_i, pid in enumerate(meta["players"]):
            if pid >= 0 and row_i < 20:
                by_pid.setdefault(int(pid), []).append((res, row_i, label))

    rows = []

    def stat_counts(res, row_i, market):
        t = res["tensor"]
        if market == "batter_total_bases":
            return (t[:, row_i, s["B1"]] + 2 * t[:, row_i, s["B2"]]
                    + 3 * t[:, row_i, s["B3"]]
                    + 4 * t[:, row_i, s["HR"]])
        if market == "batter_hits_runs_rbis":
            return (t[:, row_i, s["H"]] + t[:, row_i, s["R"]]
                    + t[:, row_i, s["RBI"]])
        if market in _BAT_STAT:
            return t[:, row_i, s[_BAT_STAT[market]]]
        if market in _PIT_STAT:
            return t[:, row_i, s[_PIT_STAT[market]]]
        return None

    def emit(game, player, team, prop, side, line, p_model, fair_side,
             group, price_col, note_bits):
        # a de-vigged two-sided market is required: without it a stray
        # longshot price has no sanity anchor and EV explodes
        if fair_side is None or p_model is None:
            return
        dec_best, amer, book = _best_price(group, price_col)
        if dec_best is None:
            return
        ev = p_model * dec_best - 1
        if ev < MIN_EV:
            return
        books = group["Book"].nunique()
        bits = list(note_bits)
        if books == 1:
            bits.append("1 book")
        rows.append({
            "Game": game, "Player": player, "Team": team, "Prop": prop,
            "Side": side, "Line": line, "Model %": p_model,
            "Mkt %": fair_side, "Edge": p_model - fair_side,
            "Best Odds": amer, "Book": book, "EV%": ev, "Books": books,
            "Note": "; ".join(bits) if bits else None})

    # player props
    pl = odds[pd.to_numeric(odds.PlayerId, errors="coerce").notna()]
    for (pid, market, line), g in pl.groupby(
            [pd.to_numeric(pl.PlayerId, errors="coerce").astype("int64"),
             "Market", "Line"]):
        try:
            line = float(line)
        except (TypeError, ValueError):
            continue
        hits = by_pid.get(int(pid))
        if not hits:
            continue
        res, row_i, label = hits[0]
        counts = stat_counts(res, row_i, market)
        if counts is None:
            continue
        p_over = _cal(res.get("calib"), MKT_FAM.get(market),
                      float((counts > line).mean()))
        fair = O.sharp_fair(g.to_dict("records"))
        meta = res["meta"]
        name = meta["names"][row_i]
        team = (meta["away"] if (row_i < 9 or row_i == 18)
                else meta["home"])
        career = (meta["career_g"][row_i] if row_i < 18
                  else meta["career_gp"][row_i])
        note = ["rookie <50 G"] if career < ROOKIE_G else []
        prop = _prop_name(market, line)
        emit(label, name, team, prop, "Over", line, p_over,
             fair, g, "OverPrice", note)
        emit(label, name, team, prop, "Under", line,
             1 - p_over, (1 - fair) if fair is not None else None, g,
             "UnderPrice", note)

    # game markets: h2h (Over = home side) and totals
    gm = odds[odds.Market.isin(["h2h", "totals"])
              & pd.to_numeric(odds.PlayerId, errors="coerce").isna()]
    for (team, market, line), g in gm.groupby(["Team", "Market", "Line"],
                                              dropna=False):
        hit = by_game.get(team)
        if hit is None:
            continue
        res, label = hit
        meta = res["meta"]
        fair = O.sharp_fair(g.to_dict("records"))
        if market == "h2h":
            hw = _cal(res.get("calib"), "ml",
                      float((res["score"][:, 1]
                             > res["score"][:, 0]).mean()))
            note = ["winner: no proven edge vs. always-home"]
            emit(label, "", meta["home"], "moneyline", meta["home"], "",
                 hw, fair, g, "OverPrice", note)
            emit(label, "", meta["away"], "moneyline", meta["away"], "",
                 1 - hw, (1 - fair) if fair is not None else None, g,
                 "UnderPrice", note)
        else:
            try:
                line = float(line)
            except (TypeError, ValueError):
                continue
            total = res["score"].sum(axis=1)
            p_over = _cal(res.get("calib"), "tot",
                          float((total > line).mean()))
            emit(label, "", "", "total runs", "Over", line, p_over,
                 fair, g, "OverPrice", [])
            emit(label, "", "", "total runs", "Under", line, 1 - p_over,
                 (1 - fair) if fair is not None else None, g,
                 "UnderPrice", [])

    rows.sort(key=lambda r: -(r["EV%"] or 0))
    return rows


def save_excel_slate(specs, out, path=None):
    """Aggregate sim results into the workbook (Batter Props, Pitching
    Props, Games, Bets). `out` is predict_slate()'s list."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, \
        Side
    from openpyxl.utils import get_column_letter

    frames = [game_frame(r) for r in out]
    date = str(specs[0]["date"]) if specs else dt.date.today().isoformat()
    PRED_DIR.mkdir(exist_ok=True)
    if path is None:
        path = PRED_DIR / f"{date}.xlsx"
        k = 2
        while Path(path).exists():
            path = PRED_DIR / f"{date}_{k}.xlsx"
            k += 1

    bet_rows = build_bets(out, date)

    wb = Workbook()
    white_bold = Font(color="FFFFFFFF", bold=True)
    red_font = Font(color=ROOKIE_RED)
    row_tint = PatternFill("solid", fgColor=BETS_ROW_TINT)
    thin = Side(style="thin", color="FFB7B7B7")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")

    def write_sheet(ws, cols, rows, pct_cols, header_color=NAVY,
                    all_row_fill=None, width_cap=40):
        head_fill = PatternFill("solid", fgColor=header_color)
        widths = [len(str(c)) for c in cols]
        for j, c in enumerate(cols, start=1):
            cell = ws.cell(row=1, column=j, value=c)
            cell.fill = head_fill
            cell.font = white_bold
            cell.border = box
            cell.alignment = center
        for i, r in enumerate(rows, start=2):
            for j, c in enumerate(cols, start=1):
                v = r.get(c)
                cell = ws.cell(row=i, column=j, value=v)
                cell.border = box
                cell.alignment = center
                if all_row_fill is not None:
                    cell.fill = all_row_fill
                shown = "" if v is None else str(v)
                if c in pct_cols and isinstance(v, float):
                    cell.number_format = "0.0%"
                    shown = f"{v:.1%}"
                elif c == "Best Odds" and v is not None:
                    cell.number_format = "+0;-0"
                    shown = f"{int(v):+d}"
                elif isinstance(v, float):
                    shown = f"{v:.2f}"
                if c == "Career G" and isinstance(v, (int, float)) \
                        and v < ROOKIE_G:
                    cell.font = red_font
                widths[j - 1] = max(widths[j - 1], len(shown))
        ws.freeze_panes = "A2"
        # header dropdown arrows: sort/filter on every column
        ws.auto_filter.ref = (f"A1:{get_column_letter(len(cols))}"
                              f"{len(rows) + 1}")
        for j in range(1, len(cols) + 1):
            # +4 keeps content clear of the filter arrow
            ws.column_dimensions[get_column_letter(j)].width = \
                min(widths[j - 1] + 4, width_cap)

    ws = wb.active
    ws.title = "Batter Props"
    bat_rows = [r for f in frames for r in f["bat"]]
    # board default: most likely homer at the top
    bat_rows.sort(key=lambda r: -(r.get("HR") or 0.0))
    bat_cols = list(bat_rows[0].keys()) if bat_rows else []
    bat_pct = {c for c in bat_cols
               if c not in ("Game", "Team", "G#", "Slot", "Name", "ID",
                            "Career G") and not c.startswith("x")}
    write_sheet(ws, bat_cols, bat_rows, bat_pct)

    ws = wb.create_sheet("Pitching Props")
    pit_rows = [r for f in frames for r in f["pit"]]
    pit_cols = list(pit_rows[0].keys()) if pit_rows else []
    pit_pct = {c for c in pit_cols if ">" in c}
    write_sheet(ws, pit_cols, pit_rows, pit_pct)

    ws = wb.create_sheet("Games")
    game_rows = [f["game"] for f in frames]
    game_pct = {c for c in GAME_COLS if ">" in c or c == "Win Prob"}
    write_sheet(ws, GAME_COLS, game_rows, game_pct)

    ws = wb.create_sheet("Bets")
    bet_cols = ["Game", "Player", "Team", "Prop", "Side", "Line",
                "Model %", "Mkt %", "Edge", "Best Odds", "Book", "EV%",
                "Books", "Note"]
    write_sheet(ws, bet_cols, bet_rows,
                {"Model %", "Mkt %", "Edge", "EV%"},
                header_color=BETS_HEADER, all_row_fill=row_tint)
    if not bet_rows:
        cell = ws.cell(row=2, column=1,
                       value="no captured odds for this date — run "
                             "Tools/2_scrape_odds.py near game time")
        cell.border = box
        cell.alignment = center

    wb.save(path)

    npz = {}
    for gi, r in enumerate(out):
        npz[f"tensor_{gi}"] = r["tensor"]
        npz[f"score_{gi}"] = r["score"]
        npz[f"f5_{gi}"] = r["runs_f5"]
        npz[f"i1_{gi}"] = r["runs_i1"]
        npz[f"players_{gi}"] = np.array(r["meta"]["players"])
        # product tag: morning (projected lineups) vs confirmed — the
        # two ledgers are evaluated separately
        sp = r["spec"]
        confirmed = (sp.get("away_lineup_src", "mlb") == "mlb"
                     and sp.get("home_lineup_src", "mlb") == "mlb")
        npz[f"product_{gi}"] = np.array(
            "confirmed" if confirmed else "projected")
    np.savez_compressed(Path(path).with_suffix(".sims.npz"), **npz)
    return str(path)
