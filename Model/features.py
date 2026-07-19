"""Feature warehouse for the model layer.

Builds every derived store the component models train on and the serve
path reads, from the files in Data/. All stores land in
Model/artifacts/stores/ as parquet (plus small json), with a build
manifest recording counts and the data horizon.

Leakage rule enforced everywhere: a feature describing entity E on date D
uses only information from strictly before D. The decayed-panel machinery
guarantees it structurally (each panel row carries sums through the
PREVIOUS activity date), season-grain effects (park, ump, defense,
arsenal) are consumed from prior seasons only, and priors for thin
samples come from MiLB seasons strictly before the row's season.

Stores built (python Model/features.py --build):
  pa_table.parquet          one row per plate appearance 2015+, from the
                            raw pitch archive (last pitch of each PA):
                            8-class outcome label, batted-ball type,
                            base-out state, score, TTO, cumulative pitch
                            count, rest days, park/weather/ump context
  panel_*.parquet           per-entity DAILY decayed sufficient stats,
                            as-of-safe (see decayed_panel):
                            outcome classes (batter/pitcher, overall and
                            vs-hand), batted-ball quality, pitch-level
                            discipline/velocity, starter leash
  park_factors.parquet      per (Venue, Year) per-class multipliers from
                            the three PRIOR seasons, shrunk toward 1
  ump_factors.parquet       per (HpUmpId, Year) K/BB multipliers from
                            prior seasons, shrunk toward 1
  defense_team.parquet      prior-season team OAA + battery (framing,
                            pop time) serving surface
  catcher_player.parquet    prior-season per-catcher framing/throwing
  arsenal_*.parquet         prior-season pitcher usage and batter
                            run-value per pitch type (matchup dot product
                            is computed at assemble time)
  league_rates.parquet      per-season league class rates (priors),
                            overall and by pitcher hand / batter side
  eb_k.json                 empirical-Bayes shrinkage sample sizes per
                            class (method-of-moments fit)
  milb_priors.parquet       MiLB-translated per-player prior class rates
                            for thin-MLB-sample players, per (player,
                            first MLB-usable year)
  pattern_table.parquet     empirical advancement patterns: distribution
                            of (batter dest, runner dests, outs, runs,
                            RBI, earned) keyed by (class, batted-ball
                            type, base state, outs) — built from
                            mlb_pbp.csv terminal movements
  preevents.json            league rates for mid-PA events the patterns
                            exclude (WP/PB advance, pickoff) by base state
  sb_table.parquet          steal-of-2B opportunity rows with attempt /
                            success labels and battery covariates
  hazard_table.parquet      per-batter-faced starter rows with removal
                            labels (game-end censored rows dropped)
  forecast_error.json       forecast-vs-actual weather error summary
                            (insufficient-data flag until the archive
                            accumulates)

The shared feature assembly (assemble_features) is imported by train.py
and predict.py so the training matrix and the serve matrix can never
drift.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
RAW = DATA / "raw_pitches"
ART = ROOT / "Model" / "artifacts"
STORES = ART / "stores"

DECAY_HL = 90.0          # days; skill half-life for decayed panels
VELO_HL_FAST = 30.0      # short-window velo (in-season trend vs baseline)
PRIOR_SEASONS_PARK = 3   # park factors pool this many prior seasons
K_PARK = 2000            # PA of shrinkage toward 1 for park factors
K_UMP = 1500             # PA of shrinkage toward 1 for ump factors
MIN_MLB_PA_FOR_OWN = 400  # below this decayed PA, MiLB prior blends in

CLASSES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "IPO"]
EVENT_TO_CLASS = {
    "strikeout": "K", "strikeout_double_play": "K",
    "walk": "BB", "intent_walk": "BB",
    "hit_by_pitch": "HBP", "catcher_interf": "HBP",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "field_out": "IPO", "force_out": "IPO",
    "grounded_into_double_play": "IPO", "double_play": "IPO",
    "triple_play": "IPO", "fielders_choice": "IPO",
    "fielders_choice_out": "IPO", "field_error": "IPO",
    "sac_fly": "IPO", "sac_bunt": "IPO", "sac_fly_double_play": "IPO",
    "sac_bunt_double_play": "IPO", "batter_interference": "IPO",
    "other_out": "IPO",
}
BBTYPES = ["ground_ball", "fly_ball", "line_drive", "popup"]

# wind direction -> radial factor (+1 blowing out, -1 blowing in, 0 cross)
WIND_RADIAL = {"Out To CF": 1.0, "Out To LF": 0.8, "Out To RF": 0.8,
               "In From CF": -1.0, "In From LF": -0.8, "In From RF": -0.8,
               "L To R": 0.0, "R To L": 0.0, "Calm": 0.0, "None": 0.0,
               "Varies": 0.0}

PA_COLS = [
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number",
    "batter", "pitcher", "stand", "p_throws", "events", "bb_type",
    "on_1b", "on_2b", "on_3b", "outs_when_up", "inning", "inning_topbot",
    "bat_score", "fld_score", "n_thruorder_pitcher",
    "batter_days_since_prev_game", "pitcher_days_since_prev_game",
    "fielder_2", "home_team", "away_team",
]


class VectorScaler:
    """Multinomial calibration: logistic regression on a model's
    log-probabilities. Lives HERE (a stable import path) so pickled
    artifacts resolve it from any entry point."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(max_iter=2000, C=10.0)

    def fit(self, proba, y):
        self.lr.fit(np.log(np.clip(proba, 1e-9, 1)), y)
        return self

    def transform(self, proba):
        return self.lr.predict_proba(np.log(np.clip(proba, 1e-9, 1)))


class PlattCal:
    """Two-parameter logistic output calibrator applied in logit space:
    strictly monotone (so per-family use preserves cross-line
    coherence), smooth, and never exactly 0 or 1. Chosen over isotonic
    regression, whose flat tail segments emit hard 0/1 probabilities
    wherever the training tails are sparse. Lives HERE (a stable import
    path) so pickled artifacts resolve it from any entry point."""

    def __init__(self, a, b):
        self.a, self.b = float(a), float(b)

    def predict(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        z = np.log(p) - np.log1p(-p)
        return 1.0 / (1.0 + np.exp(-(self.a + self.b * z)))


# ---------------------------------------------------------------- io

def read_csv(name, **kw):
    return pd.read_csv(DATA / name, encoding="utf-8-sig",
                       low_memory=False, **kw)


def raw_seasons():
    return sorted(RAW.glob("pitches_*.parquet"))


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# ------------------------------------------------------- pa_table build

def build_pa_table():
    """One row per completed plate appearance from the raw pitch archive:
    the last pitch of each (game, at_bat_number) carries the outcome."""
    frames = []
    dropped = {}
    for path in raw_seasons():
        raw = pd.read_parquet(path, columns=PA_COLS)
        raw["game_pk"] = _num(raw["game_pk"])
        raw["at_bat_number"] = _num(raw["at_bat_number"])
        raw["pitch_number"] = _num(raw["pitch_number"])
        raw = raw.dropna(subset=["game_pk", "at_bat_number"])
        raw = raw.sort_values(["game_pk", "at_bat_number", "pitch_number"],
                              kind="mergesort")
        grp = raw.groupby(["game_pk", "at_bat_number"], sort=False,
                          as_index=False)
        last = grp.tail(1).copy()
        # first-pitch snapshot = the base-out state BEFORE any mid-PA
        # event (steals, wild pitches) — the state the SB opportunity
        # table and the sim's pre-PA checks condition on
        first = grp.head(1)[["game_pk", "at_bat_number", "on_1b",
                             "on_2b", "on_3b", "outs_when_up"]]
        first = first.rename(columns={"on_1b": "r1_0", "on_2b": "r2_0",
                                      "on_3b": "r3_0",
                                      "outs_when_up": "outs0"})
        last = last.merge(first, on=["game_pk", "at_bat_number"],
                          how="left")
        # per-PA pitch count and the pitcher's cumulative pitches BEFORE
        # this PA (last pitch_number of a PA == pitches thrown in it)
        last["pa_pitches"] = last["pitch_number"].fillna(1)
        last["cum_pitches"] = (
            last.groupby(["game_pk", "pitcher"])["pa_pitches"].cumsum()
            - last["pa_pitches"])
        cls = last["events"].map(EVENT_TO_CLASS)
        dropped[path.stem] = int((cls.isna()).sum())
        last = last[cls.notna()].copy()
        last["label"] = cls[cls.notna()]
        last["bb_type"] = last["bb_type"].where(
            last["bb_type"].isin(BBTYPES), "")
        # a non-in-play class never carries a batted-ball type
        last.loc[~last["label"].isin(["1B", "2B", "3B", "HR", "IPO"]),
                 "bb_type"] = ""
        frames.append(last)
    pa = pd.concat(frames, ignore_index=True)

    pa["Date"] = pd.to_datetime(pa["game_date"])
    pa["Season"] = _num(pa["game_year"]).astype("int32")
    pa["BatterId"] = _num(pa["batter"]).astype("int64")
    pa["PitcherId"] = _num(pa["pitcher"]).astype("int64")
    pa["CatcherId"] = _num(pa["fielder_2"])
    pa["tto"] = _num(pa["n_thruorder_pitcher"]).fillna(1).clip(1, 4)
    pa["outs"] = _num(pa["outs_when_up"]).fillna(0).clip(0, 2).astype("int8")
    for b, col in ((1, "on_1b"), (2, "on_2b"), (3, "on_3b")):
        pa[f"r{b}"] = _num(pa[col])
    pa["state"] = ((pa["r1"].notna()).astype(int)
                   + 2 * (pa["r2"].notna()).astype(int)
                   + 4 * (pa["r3"].notna()).astype(int)).astype("int8")
    for b in (1, 2, 3):
        pa[f"r{b}_0"] = _num(pa[f"r{b}_0"])
    pa["state0"] = ((pa["r1_0"].notna()).astype(int)
                    + 2 * (pa["r2_0"].notna()).astype(int)
                    + 4 * (pa["r3_0"].notna()).astype(int)).astype("int8")
    pa["outs0"] = _num(pa["outs0"]).fillna(0).clip(0, 2).astype("int8")
    pa["bat_sc"] = _num(pa["bat_score"])
    pa["score_diff"] = pa["bat_sc"] - _num(pa["fld_score"])
    pa["home_bat"] = (pa["inning_topbot"].astype(str).str.lower()
                      == "bot").astype("int8")
    pa["bat_team"] = np.where(pa["home_bat"] == 1, pa["home_team"],
                              pa["away_team"])
    pa["fld_team"] = np.where(pa["home_bat"] == 1, pa["away_team"],
                              pa["home_team"])
    pa["rest_b"] = _num(pa["batter_days_since_prev_game"])
    pa["rest_p"] = _num(pa["pitcher_days_since_prev_game"])

    games = read_csv("mlb_games.csv",
                     usecols=["GamePk", "Venue", "DayNight", "Temp",
                              "WindSpeed", "WindDir", "Condition"])
    games["GamePk"] = _num(games["GamePk"])
    pa = pa.merge(games, left_on="game_pk", right_on="GamePk", how="left")
    wx = read_csv("mlb_weather.csv",
                  usecols=["GamePk", "Humidity", "Pressure"])
    wx["GamePk"] = _num(wx["GamePk"])
    pa = pa.merge(wx, on="GamePk", how="left")
    umps = read_csv("mlb_umpires.csv", usecols=["GamePk", "HpUmpId"])
    umps["GamePk"] = _num(umps["GamePk"])
    pa = pa.merge(umps, on="GamePk", how="left")

    keep = ["game_pk", "Date", "Season", "at_bat_number", "BatterId",
            "PitcherId", "CatcherId", "stand", "p_throws", "label",
            "bb_type", "state", "outs", "r1", "r2", "r3", "state0",
            "outs0", "r1_0", "r2_0", "r3_0", "inning",
            "home_bat", "bat_team", "fld_team", "bat_sc", "score_diff",
            "tto", "cum_pitches", "pa_pitches", "rest_b", "rest_p", "Venue",
            "DayNight", "Temp", "WindSpeed", "WindDir", "Condition",
            "Humidity", "Pressure", "HpUmpId"]
    pa = pa[keep].sort_values(["Date", "game_pk", "at_bat_number"])
    STORES.mkdir(parents=True, exist_ok=True)
    pa.to_parquet(STORES / "pa_table.parquet", index=False)
    print(f"pa_table: {len(pa):,} PAs, {pa.game_pk.nunique():,} games, "
          f"{pa.Season.min()}-{pa.Season.max()} "
          f"(non-PA rows dropped/season: "
          f"{int(np.mean(list(dropped.values())))} avg)", flush=True)
    return pa


# ------------------------------------------------- decayed panel engine

def decayed_panel(daily, entity_cols, stat_cols, hl=DECAY_HL):
    """Daily per-entity stats -> as-of panel.

    Input: one row per entity(+key) per active date with raw daily sums.
    Output: same rows plus, per stat, `<stat>_d` = exp-decayed sum of all
    ACTIVITY THROUGH THAT DATE (inclusive). Consumers joining a row dated
    D must use the panel row at the latest date STRICTLY BEFORE D and
    decay it forward by the gap — merge_asof_panel below does exactly
    that, which is what makes every derived feature leakage-free.
    """
    df = daily.sort_values(entity_cols + ["Date"]).reset_index(drop=True)
    t0 = df.groupby(entity_cols)["Date"].transform("min")
    tp = (df["Date"] - t0).dt.days.astype(float)
    up = np.power(2.0, tp / hl)             # d^-t' with d = 2^(-1/hl)
    down = np.power(2.0, -tp / hl)
    for c in stat_cols:
        cum = (df[c].astype(float) * up).groupby(
            [df[k] for k in entity_cols]).cumsum()
        df[c + "_d"] = cum * down
    return df[entity_cols + ["Date"] + [c + "_d" for c in stat_cols]]


def merge_asof_panel(rows, panel, entity_cols, stat_cols, prefix,
                     hl=DECAY_HL):
    """As-of join: each row dated D gets the panel state from the latest
    panel date STRICTLY BEFORE D, decayed forward to D. Rows keep their
    original order."""
    left = rows[entity_cols + ["Date"]].copy()
    left["_ord"] = np.arange(len(left))
    left = left.sort_values("Date", kind="mergesort")
    right = panel.rename(columns={"Date": "_pdate"}).sort_values("_pdate")
    # shift the panel one day so an exact-date match lands on the row
    # BEFORE today's games (panel rows are inclusive-through-date)
    right["_join"] = right["_pdate"] + pd.Timedelta(days=1)
    merged = pd.merge_asof(left, right, left_on="Date", right_on="_join",
                           by=entity_cols, direction="backward")
    gap = (merged["Date"] - merged["_pdate"]).dt.days.astype(float)
    decay = np.power(2.0, -gap / hl)
    out = pd.DataFrame(index=merged.index)
    for c in stat_cols:
        out[prefix + c] = merged[c + "_d"] * decay
    out["_ord"] = merged["_ord"].values
    out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return out


# ------------------------------------------------------ panels builders

def _daily_class_counts(pa, ent, by_hand_col=None):
    keys = [ent, "Date"] + ([by_hand_col] if by_hand_col else [])
    g = pa.groupby(keys, sort=False)
    out = g.size().rename("pa").reset_index()
    lab = pd.get_dummies(pa["label"])
    lab[keys] = pa[keys].values
    counts = lab.groupby(keys, sort=False)[CLASSES].sum().reset_index()
    return out.merge(counts, on=keys, how="left")


def build_outcome_panels(pa):
    specs = [
        ("panel_bat_out", "BatterId", None),
        ("panel_bat_out_hand", "BatterId", "p_throws"),
        ("panel_pit_out", "PitcherId", None),
        ("panel_pit_out_hand", "PitcherId", "stand"),
    ]
    for name, ent, hand in specs:
        sub = pa.dropna(subset=[ent])
        daily = _daily_class_counts(sub, ent, hand)
        ents = [ent] + ([hand] if hand else [])
        panel = decayed_panel(daily, ents, ["pa"] + CLASSES)
        panel.to_parquet(STORES / f"{name}.parquet", index=False)
        print(f"{name}: {len(panel):,} entity-days", flush=True)


def build_bip_panels():
    bip = read_csv("mlb_statcast_bip.csv",
                   usecols=["Date", "BatterId", "PitcherId", "BBType",
                            "ExitVelo", "LSA"])
    bip["Date"] = pd.to_datetime(bip["Date"])
    bip["ev"] = _num(bip["ExitVelo"])
    bip["hard"] = (bip["ev"] >= 95).astype(float)
    bip["barrel"] = (_num(bip["LSA"]) == 6).astype(float)
    bip["gb"] = (bip["BBType"] == "ground_ball").astype(float)
    bip["air"] = bip["BBType"].isin(["fly_ball", "line_drive"]).astype(float)
    bip["ev_n"] = bip["ev"].notna().astype(float)
    bip["ev_sum"] = bip["ev"].fillna(0.0)
    bip["bip"] = 1.0
    stats = ["bip", "ev_n", "ev_sum", "hard", "barrel", "gb", "air"]
    for name, ent in (("panel_bat_bip", "BatterId"),
                      ("panel_pit_bip", "PitcherId")):
        daily = bip.groupby([ent, "Date"], sort=False)[stats].sum(
            ).reset_index()
        panel = decayed_panel(daily, [ent], stats)
        panel.to_parquet(STORES / f"{name}.parquet", index=False)
        print(f"{name}: {len(panel):,} entity-days", flush=True)


def build_pitchmetric_panels():
    b = read_csv("mlb_pitch_daily_batters.csv",
                 usecols=["PlayerId", "Date", "n", "sw_n", "wh_n", "oz_n",
                          "oz_sw", "oz_wh", "z_n", "cs_n",
                          "fb95_n", "fb95_sw", "fb95_wh",
                          "brk_n", "brk_sw", "brk_wh"])
    b["Date"] = pd.to_datetime(b["Date"])
    stats_b = [c for c in b.columns if c not in ("PlayerId", "Date")]
    panel = decayed_panel(b, ["PlayerId"], stats_b)
    panel.to_parquet(STORES / "panel_bat_pitchmet.parquet", index=False)
    print(f"panel_bat_pitchmet: {len(panel):,} entity-days", flush=True)

    p = read_csv("mlb_pitch_daily_pitchers.csv",
                 usecols=["PlayerId", "Date", "n", "sw_n", "wh_n", "oz_n",
                          "oz_sw", "z_n", "cs_n", "edge_n",
                          "fb_n", "fb_v", "fp_n", "fp_s"])
    p["Date"] = pd.to_datetime(p["Date"])
    stats_p = [c for c in p.columns if c not in ("PlayerId", "Date")]
    slow = decayed_panel(p, ["PlayerId"], stats_p)
    slow.to_parquet(STORES / "panel_pit_pitchmet.parquet", index=False)
    fast = decayed_panel(p[["PlayerId", "Date", "fb_n", "fb_v"]],
                         ["PlayerId"], ["fb_n", "fb_v"], hl=VELO_HL_FAST)
    fast = fast.rename(columns={"fb_n_d": "fb_n_fast_d",
                                "fb_v_d": "fb_v_fast_d"})
    fast.to_parquet(STORES / "panel_pit_velo_fast.parquet", index=False)
    print(f"panel_pit_pitchmet: {len(slow):,} entity-days (+fast velo)",
          flush=True)


def build_leash_panel():
    gp = read_csv("mlb_game_pitching.csv",
                  usecols=["PlayerId", "Date", "GS", "NP", "BF", "IP"])
    gp = gp[_num(gp["GS"]) == 1].copy()
    gp["Date"] = pd.to_datetime(gp["Date"])
    gp["starts"] = 1.0
    gp["np_sum"] = _num(gp["NP"]).fillna(0)
    gp["bf_sum"] = _num(gp["BF"]).fillna(0)
    ip = _num(gp["IP"]).fillna(0)
    gp["outs_sum"] = (ip.astype(int) * 3 + round((ip % 1) * 10)).astype(float)
    daily = gp.groupby(["PlayerId", "Date"], sort=False)[
        ["starts", "np_sum", "bf_sum", "outs_sum"]].sum().reset_index()
    panel = decayed_panel(daily, ["PlayerId"],
                          ["starts", "np_sum", "bf_sum", "outs_sum"])
    panel.to_parquet(STORES / "panel_leash.parquet", index=False)
    print(f"panel_leash: {len(panel):,} starter-days", flush=True)


# --------------------------------------- season-grain effect tables

def build_park_factors(pa):
    """Per (Venue, Year): class-rate multipliers vs league, pooled over
    the PRIOR `PRIOR_SEASONS_PARK` seasons and shrunk toward 1. The
    CURRENT season's row additionally pools its own season-to-date PAs
    (hierarchical in-season evidence — strictly before today at serve
    time, and never seen by training, which stops at the calibration
    year)."""
    cur = int(pa.Season.max())
    rates = pa.groupby(["Venue", "Season"], sort=False).agg(
        pa_n=("label", "size"),
        **{c: ("label", lambda s, c=c: (s == c).sum()) for c in CLASSES}
    ).reset_index()
    lg = pa.groupby("Season")["label"].value_counts(normalize=True)
    lg = lg.rename("lg_rate").reset_index()
    rows = []
    for year in range(int(pa.Season.min()) + 1, int(pa.Season.max()) + 2):
        in_win = ((rates.Season >= year - PRIOR_SEASONS_PARK)
                  & ((rates.Season < year)
                     | ((rates.Season == year) & (year == cur))))
        window = rates[in_win]
        if window.empty:
            continue
        agg = window.groupby("Venue")[["pa_n"] + CLASSES].sum().reset_index()
        lgw = lg[(lg.Season >= year - PRIOR_SEASONS_PARK)
                 & ((lg.Season < year)
                    | ((lg.Season == year) & (year == cur)))]
        lgr = lgw.groupby("label")["lg_rate"].mean()
        for c in CLASSES:
            obs = agg[c] / agg["pa_n"].clip(lower=1)
            f = obs / max(lgr.get(c, np.nan), 1e-9)
            w = agg["pa_n"] / (agg["pa_n"] + K_PARK)
            agg[f"pf_{c}"] = 1.0 + w * (f - 1.0)
        agg["Year"] = year
        rows.append(agg[["Venue", "Year"] + [f"pf_{c}" for c in CLASSES]])
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(STORES / "park_factors.parquet", index=False)
    print(f"park_factors: {len(out):,} venue-years", flush=True)


def build_ump_factors(pa):
    """Per (HpUmpId, Year) K/BB multipliers from prior seasons, shrunk
    toward 1; the CURRENT season's row also pools its own season-to-date
    games (matters most in the ABS-challenge era, where historical ump
    effects may compress)."""
    cur = int(pa.Season.max())
    sub = pa.dropna(subset=["HpUmpId"]).copy()
    sub["HpUmpId"] = _num(sub["HpUmpId"]).astype("int64")
    rates = sub.groupby(["HpUmpId", "Season"], sort=False).agg(
        pa_n=("label", "size"),
        K=("label", lambda s: (s == "K").sum()),
        BB=("label", lambda s: (s == "BB").sum())).reset_index()
    lg = sub.groupby("Season").agg(
        lgK=("label", lambda s: (s == "K").mean()),
        lgBB=("label", lambda s: (s == "BB").mean())).reset_index()
    rows = []
    for year in range(int(sub.Season.min()) + 1,
                      int(sub.Season.max()) + 2):
        in_win = ((rates.Season < year)
                  | ((rates.Season == year) & (year == cur)))
        window = rates[in_win]
        if window.empty:
            continue
        agg = window.groupby("HpUmpId")[["pa_n", "K", "BB"]].sum(
            ).reset_index()
        lgw = lg[(lg.Season < year)
                 | ((lg.Season == year) & (year == cur))][
            ["lgK", "lgBB"]].mean()
        w = agg["pa_n"] / (agg["pa_n"] + K_UMP)
        agg["uf_K"] = 1.0 + w * ((agg["K"] / agg["pa_n"].clip(lower=1))
                                 / max(lgw["lgK"], 1e-9) - 1.0)
        agg["uf_BB"] = 1.0 + w * ((agg["BB"] / agg["pa_n"].clip(lower=1))
                                  / max(lgw["lgBB"], 1e-9) - 1.0)
        agg["Year"] = year
        rows.append(agg[["HpUmpId", "Year", "uf_K", "uf_BB"]])
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(STORES / "ump_factors.parquet", index=False)
    print(f"ump_factors: {len(out):,} ump-years", flush=True)


def build_defense_tables():
    oaa = read_csv("mlb_oaa.csv")
    oaa["Year"] = _num(oaa["Year"]) + 1          # serve year = prior + 1
    oaa = oaa[["Year", "Team", "OAA_per162"]].rename(
        columns={"OAA_per162": "def_oaa"})
    cat = read_csv("mlb_catchers_team.csv")
    cat["Year"] = _num(cat["Year"]) + 1
    cat = cat[["Year", "Team", "FrameRV_pt", "PopTime", "CSAA_att",
               "SBAtt"]]
    out = oaa.merge(cat, on=["Year", "Team"], how="outer")
    out.to_parquet(STORES / "defense_team.parquet", index=False)
    catp = read_csv("mlb_catchers.csv")
    catp["Year"] = _num(catp["Year"]) + 1
    keep = [c for c in ("Year", "PlayerId", "FrameRV", "StrikePct",
                        "PopTime", "CSAA", "SBAtt") if c in catp.columns]
    catp[keep].to_parquet(STORES / "catcher_player.parquet", index=False)
    print(f"defense_team: {len(out):,} team-years; catcher_player: "
          f"{len(catp):,}", flush=True)


def build_arsenal_tables():
    ars = read_csv("mlb_pitch_arsenals.csv",
                   usecols=["Year", "PlayerId", "PitchType", "%"])
    ars["Year"] = _num(ars["Year"]) + 1
    ars["usage"] = _num(ars["%"]) / 100.0
    ars[["Year", "PlayerId", "PitchType", "usage"]].to_parquet(
        STORES / "arsenal_pitcher.parquet", index=False)
    arb = read_csv("mlb_pitch_arsenals_batters.csv",
                   usecols=["Year", "PlayerId", "PitchType", "RV/100",
                            "Pitches"])
    arb["Year"] = _num(arb["Year"]) + 1
    arb["rv100"] = _num(arb["RV/100"])
    arb["n"] = _num(arb["Pitches"])
    arb[["Year", "PlayerId", "PitchType", "rv100", "n"]].to_parquet(
        STORES / "arsenal_batter.parquet", index=False)
    print(f"arsenal tables: {ars.PlayerId.nunique():,} pitchers, "
          f"{arb.PlayerId.nunique():,} batters", flush=True)


# ------------------------------------------------- priors & shrinkage

def build_league_rates(pa):
    lg = (pa.groupby(["Season"])["label"].value_counts(normalize=True)
          .rename("rate").reset_index())
    lg_hand = (pa.groupby(["Season", "p_throws"])["label"]
               .value_counts(normalize=True).rename("rate").reset_index())
    lg_stand = (pa.groupby(["Season", "stand"])["label"]
                .value_counts(normalize=True).rename("rate").reset_index())
    lg.to_parquet(STORES / "league_rates.parquet", index=False)
    lg_hand.to_parquet(STORES / "league_rates_hand.parquet", index=False)
    lg_stand.to_parquet(STORES / "league_rates_stand.parquet", index=False)
    print("league_rates written", flush=True)


def build_eb_k(pa):
    """Method-of-moments shrinkage sample sizes: for each class, how many
    PA of league-average belief a player rate is shrunk toward."""
    ks = {}
    seas = pa.groupby(["BatterId", "Season"]).agg(
        n=("label", "size"),
        **{c: ("label", lambda s, c=c: (s == c).mean()) for c in CLASSES}
    ).reset_index()
    big = seas[seas.n >= 300]
    for c in CLASSES:
        p = big[c]
        pbar = float(p.mean())
        noise = float((pbar * (1 - pbar) / big.n).mean())
        var_true = max(float(p.var()) - noise, 1e-6)
        ks[c] = float(np.clip(pbar * (1 - pbar) / var_true, 20, 2000))
    (STORES / "eb_k.json").write_text(json.dumps(ks, indent=1))
    print(f"eb_k: {json.dumps({k: round(v) for k, v in ks.items()})}",
          flush=True)


def build_milb_priors():
    """MiLB-translated per-player class-rate priors. Level factors are fit
    on players with a AAA/AA season followed by a real MLB season; the
    prior for a player entering year Y blends his last two MiLB seasons
    strictly before Y through those factors."""
    milb = read_csv("milb_batting.csv")
    milb = milb[milb["Level"].isin(["AAA", "AA"])].copy()
    for c in ("PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP"):
        milb[c] = _num(milb[c]).fillna(0)
    milb["Year"] = _num(milb["Year"])
    milb["PlayerId"] = _num(milb["PlayerId"])
    milb = milb[milb.PA >= 100]
    milb["1B"] = milb["H"] - milb["2B"] - milb["3B"] - milb["HR"]
    mlb = read_csv("mlb_batting_stats.csv",
                   usecols=["Year", "PlayerId", "PA", "H", "2B", "3B",
                            "HR", "BB", "SO", "HBP"])
    for c in ("PA", "H", "2B", "3B", "HR", "BB", "SO", "HBP"):
        mlb[c] = _num(mlb[c]).fillna(0)
    mlb["1B"] = mlb["H"] - mlb["2B"] - mlb["3B"] - mlb["HR"]
    mlb = mlb[mlb.PA >= 150]

    rate_cols = {"K": "SO", "BB": "BB", "HBP": "HBP", "1B": "1B",
                 "2B": "2B", "3B": "3B", "HR": "HR"}
    pairs = milb.merge(mlb, left_on=["PlayerId"], right_on=["PlayerId"],
                       suffixes=("_m", "_M"))
    pairs = pairs[pairs["Year_M"] == pairs["Year_m"] + 1]
    # pooled-aggregate ratio (robust to zero-count seasons, unlike a
    # mean of per-player ratios): league MLB rate of the movers divided
    # by the same players' pooled MiLB rate
    factors = {}
    for lvl in ("AAA", "AA"):
        sub = pairs[pairs.Level == lvl]
        f = {}
        for cls, col in rate_cols.items():
            r_milb = sub[f"{col}_m"].sum() / max(sub["PA_m"].sum(), 1)
            r_mlb = sub[f"{col}_M"].sum() / max(sub["PA_M"].sum(), 1)
            f[cls] = float(np.clip(r_mlb / max(r_milb, 1e-6), 0.1, 3.0))
        factors[lvl] = f

    rows = []
    for pid, g in milb.groupby("PlayerId"):
        g = g.sort_values("Year")
        for entry_year in g.Year.unique() + 1:
            past = g[g.Year < entry_year].tail(2)
            if past.empty:
                continue
            w = past.PA.values
            row = {"PlayerId": int(pid), "Year": int(entry_year)}
            for cls, col in rate_cols.items():
                tr = [(r[col] / r.PA) * factors[r.Level][cls]
                      for _, r in past.iterrows()]
                row[f"prior_{cls}"] = float(np.average(tr, weights=w))
            row["prior_pa"] = float(past.PA.sum())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(STORES / "milb_priors.parquet", index=False)
    (STORES / "milb_level_factors.json").write_text(
        json.dumps(factors, indent=1))
    print(f"milb_priors: {len(out):,} player-entry-years; factors: "
          f"{ {l: {k: round(v, 2) for k, v in f.items()} for l, f in factors.items()} }",
          flush=True)


# --------------------------------------------- PBP-derived tables

DEST_HELD = 9  # runner did not move (no movement row in the play)


def build_pattern_table(pa):
    """Empirical advancement patterns from the terminal movements of each
    play: keyed by (label, bb_type, base state, outs), each pattern is
    (batter dest, r1 dest, r2 dest, r3 dest, outs added, runs, batter
    RBI, earned-run count) with dest codes 0=out 1/2/3=base 4=scored
    9=held. Sampling a pattern reproduces DP/FC/SF/ROE/error advancement
    — and the empirical RBI + earned/unearned split — by construction."""
    pbp = pd.read_csv(DATA / "mlb_pbp.csv", encoding="utf-8-sig",
                      usecols=["GamePk", "AtBatIndex", "PlayIndex",
                               "RunnerId", "StartBase", "EndBase",
                               "IsOut", "RBI", "Earned"],
                      low_memory=False)
    pbp["at_bat_number"] = _num(pbp["AtBatIndex"]) + 1
    pbp["game_pk"] = _num(pbp["GamePk"])
    # terminal movements only: the ones caused by the last event of the PA
    pbp["PlayIndex"] = _num(pbp["PlayIndex"]).fillna(-1)
    last = pbp.groupby(["game_pk", "at_bat_number"])["PlayIndex"]
    pbp = pbp[pbp["PlayIndex"] == last.transform("max")]

    # a runner can move in SEGMENTS within one play (single to second,
    # then home on a throwing error, as separate rows) — collapse each
    # runner to origin -> final disposition, or the second segment lands
    # on a slot that was empty at pitch time and the run vanishes
    pbp = pbp.reset_index(drop=True).reset_index(names="_row")
    grp = pbp.groupby(["game_pk", "at_bat_number", "RunnerId"],
                      sort=False)
    pbp = pbp.assign(
        _first=grp["_row"].transform("min"),
        _last=grp["_row"].transform("max"),
        _rbi=grp["RBI"].transform("max"),
        _earned=grp["Earned"].transform("max"),
        _out=grp["IsOut"].transform("max"),
    )
    start = pbp.loc[pbp["_row"] == pbp["_first"],
                    ["game_pk", "at_bat_number", "RunnerId", "StartBase"]]
    final = pbp.loc[pbp["_row"] == pbp["_last"],
                    ["game_pk", "at_bat_number", "RunnerId", "EndBase",
                     "_out", "_rbi", "_earned"]]
    pbp = start.merge(final, on=["game_pk", "at_bat_number", "RunnerId"])
    pbp = pbp.rename(columns={"_out": "IsOut", "_rbi": "RBI",
                              "_earned": "Earned"})

    dest = np.select(
        [pbp["IsOut"] == 1, pbp["EndBase"] == "H", pbp["EndBase"] == "1B",
         pbp["EndBase"] == "2B", pbp["EndBase"] == "3B"],
        [0, 4, 1, 2, 3], default=DEST_HELD)
    pbp = pbp.assign(dest=dest)
    slot = pbp["StartBase"].map({"": "b", "1B": "r1", "2B": "r2",
                                 "3B": "r3"})
    pbp = pbp.assign(slot=slot.fillna("b"))

    piv = pbp.pivot_table(index=["game_pk", "at_bat_number"],
                          columns="slot", values="dest",
                          aggfunc="last").reset_index()
    for col in ("b", "r1", "r2", "r3"):
        if col not in piv.columns:
            piv[col] = np.nan
    agg = pbp.groupby(["game_pk", "at_bat_number"]).agg(
        outs_added=("IsOut", "sum"),
        rbi=("RBI", "sum"),
        earned=("Earned", "sum"),
        runs=("dest", lambda d: int((d == 4).sum()))).reset_index()
    piv = piv.merge(agg, on=["game_pk", "at_bat_number"])

    core = pa[["game_pk", "at_bat_number", "label", "bb_type", "state",
               "outs"]]
    j = core.merge(piv, on=["game_pk", "at_bat_number"], how="inner")
    for col, base in (("r1", 1), ("r2", 2), ("r3", 4)):
        occupied = (j["state"].values & base) > 0
        j[col] = np.where(occupied, j[col].fillna(DEST_HELD), DEST_HELD)
    j["b"] = j["b"].fillna(DEST_HELD)

    keys = ["label", "bb_type", "state", "outs"]
    pat_cols = ["b", "r1", "r2", "r3", "outs_added", "runs", "rbi",
                "earned"]
    tab = (j.groupby(keys + pat_cols, sort=False).size()
           .rename("n").reset_index())
    tab["p"] = tab["n"] / tab.groupby(keys)["n"].transform("sum")
    tab.to_parquet(STORES / "pattern_table.parquet", index=False)
    print(f"pattern_table: {len(tab):,} patterns over "
          f"{tab.groupby(keys).ngroups:,} keys from {len(j):,} PAs",
          flush=True)

    # mid-PA pre-events the patterns exclude: WP/PB advancement and
    # pickoff outs, as per-PA rates given runners on
    mid = pd.read_csv(DATA / "mlb_pbp.csv", encoding="utf-8-sig",
                      usecols=["GamePk", "AtBatIndex", "EventType"],
                      low_memory=False)
    mid["game_pk"] = _num(mid["GamePk"])
    mid["at_bat_number"] = _num(mid["AtBatIndex"]) + 1
    wp = mid[mid.EventType.isin(["wild_pitch", "passed_ball"])]
    pk = mid[mid.EventType.str.startswith("pickoff", na=False)]
    on = pa[pa.state > 0]
    n_on = max(len(on), 1)
    pre = {"wp_pb_per_pa_runners_on":
           float(wp.drop_duplicates(["game_pk", "at_bat_number"]).shape[0]
                 / n_on),
           "pickoff_out_per_pa_runners_on":
           float(pk.drop_duplicates(["game_pk", "at_bat_number"]).shape[0]
                 / n_on)}
    (STORES / "preevents.json").write_text(json.dumps(pre, indent=1))
    print(f"preevents: {pre}", flush=True)


def build_sb_table(pa):
    """Steal-of-second opportunities (runner on 1B, 2B empty) with
    attempt/success labels from PBP mid-PA events, plus battery and
    runner covariates for train.py's attempt/success models."""
    pbp = pd.read_csv(DATA / "mlb_pbp.csv", encoding="utf-8-sig",
                      usecols=["GamePk", "AtBatIndex", "RunnerId",
                               "StartBase", "EventType"],
                      low_memory=False)
    pbp["game_pk"] = _num(pbp["GamePk"])
    pbp["at_bat_number"] = _num(pbp["AtBatIndex"]) + 1
    ev = pbp[pbp.StartBase.eq("1B")
             & pbp.EventType.isin(["stolen_base_2b", "caught_stealing_2b",
                                   "pickoff_caught_stealing_2b"])]
    ev = ev.assign(success=(ev.EventType == "stolen_base_2b").astype(int))
    ev = (ev.groupby(["game_pk", "at_bat_number"])
          .agg(attempt=("success", "size"), success=("success", "max"))
          .reset_index())

    # opportunity = runner on 1B with 2B open at the START of the PA
    # (the last-pitch state is post-steal and would erase the label)
    opp = pa[(pa.state0.isin([1, 5]))].copy()
    opp["r1"] = opp["r1_0"]                    # the runner who can go
    opp["outs"] = opp["outs0"]
    opp = opp.merge(ev, on=["game_pk", "at_bat_number"], how="left")
    opp["attempt"] = (opp["attempt"].fillna(0) > 0).astype(int)
    opp["success"] = opp["success"].fillna(0).astype(int)

    # inning-ending caught-steals void the batter's PA, which drops the
    # row from pa_table — those attempts are real but unmatched here, so
    # store per-era scale factors the serve path multiplies onto the
    # attempt model (true attempt-PAs from PBP / matched attempt-PAs)
    games = read_csv("mlb_games.csv", usecols=["GamePk", "Season"])
    games["game_pk"] = _num(games["GamePk"])
    ev_all = ev.merge(games[["game_pk", "Season"]], on="game_pk",
                      how="left")
    scale = {}
    for era, mask_true, mask_opp in (
            ("pre2023", ev_all.Season < 2023, opp.Season < 2023),
            ("post2023", ev_all.Season >= 2023, opp.Season >= 2023)):
        true_n = int(mask_true.sum())
        got_n = int(opp.loc[mask_opp, "attempt"].sum())
        scale[era] = {
            "attempt_scale": round(true_n / max(got_n, 1), 4),
            # truth from ALL attempt events incl. the voided-PA caught
            # steals the table can't hold — the matched sample's success
            # rate is biased high, so serving anchors to this mean
            "success_true": round(float(
                ev_all.loc[mask_true, "success"].mean()), 4),
        }
    (STORES / "sb_scale.json").write_text(json.dumps(scale, indent=1))

    sprint = read_csv("mlb_sprint_speed.csv",
                      usecols=["Year", "PlayerId", "SprintSpeed"])
    sprint["Year"] = _num(sprint["Year"]) + 1
    opp = opp.merge(sprint, left_on=["r1", "Season"],
                    right_on=["PlayerId", "Year"], how="left")
    pstats = read_csv("mlb_pitching_stats.csv",
                      usecols=["Year", "PlayerId", "SB", "CS", "PK",
                               "TBF"])
    for c in ("SB", "CS", "PK", "TBF"):
        pstats[c] = _num(pstats[c]).fillna(0)
    pstats["Year"] = _num(pstats["Year"]) + 1
    pstats["sb_allowed_rate"] = pstats["SB"] / pstats["TBF"].clip(lower=50)
    pstats["cs_rate"] = ((pstats["CS"] + pstats["PK"])
                         / (pstats["SB"] + pstats["CS"] + pstats["PK"]
                            ).clip(lower=3))
    opp = opp.merge(
        pstats[["Year", "PlayerId", "sb_allowed_rate", "cs_rate"]],
        left_on=["PitcherId", "Season"], right_on=["PlayerId", "Year"],
        how="left", suffixes=("", "_p"))
    catp = pd.read_parquet(STORES / "catcher_player.parquet")
    opp = opp.merge(catp, left_on=["CatcherId", "Season"],
                    right_on=["PlayerId", "Year"], how="left",
                    suffixes=("", "_c"))
    keep = ["game_pk", "at_bat_number", "Date", "Season", "r1",
            "PitcherId", "CatcherId", "p_throws", "outs", "score_diff",
            "inning", "attempt", "success", "SprintSpeed",
            "sb_allowed_rate", "cs_rate", "PopTime", "CSAA"]
    keep = [c for c in keep if c in opp.columns]
    opp[keep].to_parquet(STORES / "sb_table.parquet", index=False)
    print(f"sb_table: {len(opp):,} opportunities, "
          f"{opp.attempt.sum():,} attempts "
          f"({opp.attempt.mean():.3%}), success rate "
          f"{opp.loc[opp.attempt == 1, 'success'].mean():.1%}, "
          f"era scale {scale}", flush=True)


def build_hazard_table(pa):
    """Per-batter-faced rows for STARTERS with removal labels (1 = this
    was the starter's last batter; game-end-while-pitching censored rows
    dropped)."""
    gp = read_csv("mlb_game_pitching.csv",
                  usecols=["GamePk", "PlayerId", "GS"])
    gp["GamePk"] = _num(gp["GamePk"])
    starters = gp[_num(gp["GS"]) == 1][["GamePk", "PlayerId"]]
    starters.columns = ["game_pk", "PitcherId"]
    sub = pa.merge(starters, on=["game_pk", "PitcherId"], how="inner")
    sub = sub.sort_values(["game_pk", "PitcherId", "at_bat_number"]
                          ).reset_index(drop=True)
    g = sub.groupby(["game_pk", "PitcherId"], sort=False)
    sub["bf"] = g.cumcount() + 1
    sub["k_so_far"] = g["label"].transform(
        lambda s: (s == "K").cumsum() - (s == "K"))
    sub["br_so_far"] = g["label"].transform(
        lambda s: s.isin(["1B", "2B", "3B", "HR", "BB", "HBP"]).cumsum()
        - s.isin(["1B", "2B", "3B", "HR", "BB", "HBP"]))
    sub["runs_so_far"] = g["bat_sc"].transform(lambda s: s - s.iloc[0])
    # label: last batter this pitcher faced — but only when his TEAM's
    # pitching continued afterward (else the game ended: censored)
    half = pa.sort_values(["game_pk", "at_bat_number"])
    half = half.groupby(["game_pk", "fld_team"])["at_bat_number"].max()
    half = half.rename("team_last_ab").reset_index()
    half.columns = ["game_pk", "fld_team", "team_last_ab"]
    own_last = g["at_bat_number"].transform("max")
    sub["removed"] = (sub["at_bat_number"].values
                      == own_last.values).astype(int)
    sub = sub.merge(half, on=["game_pk", "fld_team"], how="left")
    censored = ((sub["removed"].values == 1)
                & (sub["at_bat_number"].values
                   == sub["team_last_ab"].values))
    sub = sub[~censored]
    keep = ["game_pk", "Date", "Season", "PitcherId", "at_bat_number",
            "bf", "cum_pitches", "tto", "inning", "outs", "score_diff",
            "k_so_far", "br_so_far", "runs_so_far", "rest_p", "removed"]
    sub[keep].to_parquet(STORES / "hazard_table.parquet", index=False)
    print(f"hazard_table: {len(sub):,} starter-BF rows, removal rate "
          f"{sub.removed.mean():.3%}", flush=True)


def build_participation(pa):
    """Batter participation hazard (universal-DH era, 2022+): the chance
    a lineup slot's STARTER is substituted out before each of the slot's
    plate appearances — pinch hitters, blowout rests, catcher days,
    injury exits. One empirical rate per (slot-PA index k, inning bucket,
    score-margin bucket, batting-team-leading flag, starter-vs-pitcher
    same-handedness, catcher flag), EB-shrunk toward the k marginal.

    Risk rows: every slot PA while the starter still occupies the slot,
    plus the first substitute PA (the event). Later sub PAs are not at
    risk (the starter is already gone). The same-hand covariate is the
    STARTER's side vs the pitcher on the mound at that PA."""
    gb = read_csv("mlb_game_batting.csv",
                  usecols=["GamePk", "Season", "PlayerId", "Team",
                           "BattingOrder", "Position"])
    gb = gb[_num(gb["Season"]) >= 2022].copy()
    gb["bo"] = _num(gb["BattingOrder"])
    gb = gb.dropna(subset=["bo"])
    gb["slot"] = (gb["bo"] // 100).astype(int)
    gb = gb[(gb["slot"] >= 1) & (gb["slot"] <= 9)]
    gb["is_start"] = (gb["bo"] % 100 == 0)
    gb["game_pk"] = _num(gb["GamePk"])
    slot_map = gb[["game_pk", "PlayerId", "Team", "slot", "is_start",
                   "Position"]]

    sub = pa[pa["Season"] >= 2022][
        ["game_pk", "at_bat_number", "BatterId", "bat_team", "stand",
         "p_throws", "inning", "score_diff"]].copy()
    j = sub.merge(slot_map, left_on=["game_pk", "BatterId", "bat_team"],
                  right_on=["game_pk", "PlayerId", "Team"], how="inner")
    j = j.sort_values(["game_pk", "bat_team", "slot", "at_bat_number"])
    keys = ["game_pk", "bat_team", "slot"]
    j["k"] = j.groupby(keys).cumcount() + 1
    j["sub_seq"] = (~j["is_start"]).groupby(
        [j[k] for k in keys]).cumsum()
    # starter identity per slot: side and catcher flag from his rows
    st = j[j["is_start"]].groupby(keys).agg(
        st_stand=("stand", "first"),
        st_pos=("Position", "first")).reset_index()
    j = j.merge(st, on=keys, how="left")

    risk = j[(j["is_start"]) | (~j["is_start"] & (j["sub_seq"] == 1))]
    risk = risk.dropna(subset=["st_stand"]).copy()
    risk["event"] = (~risk["is_start"]).astype(int)
    risk["k"] = risk["k"].clip(1, 6)
    margin = risk["score_diff"].abs()
    risk["margin_b"] = np.where(margin <= 1, 0,
                                np.where(margin <= 3, 1, 2))
    risk["lead"] = (risk["score_diff"] > 0).astype(int)
    inn = _num(risk["inning"]).fillna(1)
    risk["inn_b"] = np.select([inn <= 6, inn == 7, inn == 8],
                              [0, 1, 2], default=3)
    risk["same"] = (risk["st_stand"].astype(str)
                    == risk["p_throws"].astype(str)).astype(int)
    risk["isc"] = (risk["st_pos"].astype(str) == "C").astype(int)

    cells = ["k", "inn_b", "margin_b", "lead", "same", "isc"]
    grp = risk.groupby(cells).agg(n=("event", "size"),
                                  ev=("event", "sum")).reset_index()
    km = risk.groupby("k")["event"].mean()
    K_SHRINK = 300
    grp["rate"] = ((grp["ev"] + K_SHRINK * grp["k"].map(km))
                   / (grp["n"] + K_SHRINK))
    grp.to_parquet(STORES / "participation.parquet", index=False)
    print(f"participation: {len(risk):,} risk rows, "
          f"{int(risk.event.sum()):,} substitutions "
          f"({risk.event.mean():.3%}/slot-PA); per-k marginal "
          f"{ {int(k): round(v, 4) for k, v in km.items()} }", flush=True)


def build_forecast_error():
    fc = read_csv("mlb_weather_forecast.csv")
    out = {"n": 0, "sufficient": False}
    try:
        games = read_csv("mlb_games.csv",
                         usecols=["Date", "AwayTeam", "HomeTeam", "Temp",
                                  "WindSpeed"])
        fc = fc.sort_values("ScrapedAt").drop_duplicates(
            ["Date", "AwayTeam", "HomeTeam"], keep="last")
        j = fc.merge(games, on=["Date", "AwayTeam", "HomeTeam"],
                     suffixes=("_fc", "_act"))
        j["temp_err"] = _num(j["Temp_act"]) - _num(j["Temp_fc"])
        j["wind_err"] = _num(j["WindSpeed_act"]) - _num(j["WindSpeed_fc"])
        j = j.dropna(subset=["temp_err"])
        out["n"] = int(len(j))
        if len(j) >= 50:
            out.update(sufficient=True,
                       temp_sd=float(j.temp_err.std()),
                       wind_sd=float(j.wind_err.std()))
    except Exception as e:                       # noqa: BLE001
        out["error"] = str(e)
    (STORES / "forecast_error.json").write_text(json.dumps(out, indent=1))
    print(f"forecast_error: {out}", flush=True)


# ------------------------------------------------- feature assembly

BAT_RATES = ["K", "BB", "1B", "2B", "3B", "HR"]
PIT_RATES = ["K", "BB", "1B", "2B", "3B", "HR"]

FEATURE_COLS = None  # populated by assemble_features and saved by train


def _shrunk_rates(out, prefix, pa_col, classes, prior_lookup, k_map,
                  milb=None):
    """rate_c = (count_c + k_c * prior_c) / (pa + k_c); prior is the
    league rate for that season/hand, replaced by the player's
    MiLB-translated prior when his decayed MLB PA is thin."""
    pa_d = out[pa_col].fillna(0.0)
    for c in classes:
        prior = prior_lookup[c]
        if milb is not None and f"prior_{c}" in milb:
            thin = pa_d < MIN_MLB_PA_FOR_OWN
            mprior = milb[f"prior_{c}"]
            prior = np.where(thin & mprior.notna(), mprior, prior)
        k = k_map[c]
        out[f"{prefix}{c}_rate"] = ((out[f"{prefix}{c}"].fillna(0.0)
                                     + k * prior)
                                    / (pa_d + k))
    out[f"{prefix}log_pa"] = np.log1p(pa_d)
    return out


def load_stores():
    """Everything assemble_features needs, loaded once."""
    s = {}
    for name in ("panel_bat_out", "panel_bat_out_hand", "panel_pit_out",
                 "panel_pit_out_hand", "panel_bat_bip", "panel_pit_bip",
                 "panel_bat_pitchmet", "panel_pit_pitchmet",
                 "panel_pit_velo_fast", "panel_leash", "park_factors",
                 "ump_factors", "defense_team", "arsenal_pitcher",
                 "arsenal_batter", "league_rates", "league_rates_hand",
                 "league_rates_stand", "milb_priors"):
        s[name] = pd.read_parquet(STORES / f"{name}.parquet")
    s["eb_k"] = json.loads((STORES / "eb_k.json").read_text())
    ros = read_csv("mlb_rosters.csv", usecols=["PlayerId", "DOB"])
    ros["dob"] = pd.to_datetime(ros["DOB"], format="%m/%d/%Y",
                                errors="coerce")
    s["dob"] = ros.dropna(subset=["dob"]).drop_duplicates("PlayerId")
    spr = read_csv("mlb_sprint_speed.csv",
                   usecols=["Year", "PlayerId", "SprintSpeed"])
    spr["Year"] = _num(spr["Year"]) + 1
    s["sprint"] = spr
    return s


def assemble_features(rows, stores):
    """rows: DataFrame with at least [Date, Season, BatterId, PitcherId,
    stand, p_throws, tto, home_bat, fld_team, Venue, DayNight, Temp,
    WindSpeed, WindDir, Condition, Humidity, Pressure, HpUmpId, rest_p].
    Returns (X, feature_names) aligned to rows. Shared by training and
    serving — the one code path."""
    n = len(rows)
    out = rows[["Date", "Season"]].copy().reset_index(drop=True)
    rows = rows.reset_index(drop=True)

    # ---- decayed outcome panels: batter overall + vs pitcher hand
    m = merge_asof_panel(rows, stores["panel_bat_out"], ["BatterId"],
                         ["pa"] + CLASSES, "b_")
    out = pd.concat([out, m], axis=1)
    rows_h = rows.rename(columns={"p_throws": "p_throws_key"})
    ph = stores["panel_bat_out_hand"].rename(
        columns={"p_throws": "p_throws_key"})
    m = merge_asof_panel(rows_h, ph, ["BatterId", "p_throws_key"],
                         ["pa"] + CLASSES, "bh_")
    out = pd.concat([out, m], axis=1)
    m = merge_asof_panel(rows, stores["panel_pit_out"], ["PitcherId"],
                         ["pa"] + CLASSES, "p_")
    out = pd.concat([out, m], axis=1)
    rows_s = rows.rename(columns={"stand": "stand_key"})
    ps = stores["panel_pit_out_hand"].rename(columns={"stand": "stand_key"})
    m = merge_asof_panel(rows_s, ps, ["PitcherId", "stand_key"],
                         ["pa"] + CLASSES, "ph_")
    out = pd.concat([out, m], axis=1)

    # ---- EB-shrunk rates with league (and MiLB) priors
    lg = stores["league_rates"].pivot(index="Season", columns="label",
                                      values="rate")
    lgh = stores["league_rates_hand"].pivot(
        index=["Season", "p_throws"], columns="label", values="rate")
    lgs = stores["league_rates_stand"].pivot(
        index=["Season", "stand"], columns="label", values="rate")
    prior_seas = rows["Season"].clip(lower=int(lg.index.min()),
                                     upper=int(lg.index.max()))
    milb = stores["milb_priors"].sort_values(["PlayerId", "Year"])
    mrows = rows[["BatterId", "Season"]].copy()
    mrows["Season"] = mrows["Season"].astype("int64")
    mrows["BatterId"] = mrows["BatterId"].astype("int64")
    mrows["_ord"] = np.arange(n)
    mr = milb.rename(columns={"PlayerId": "BatterId", "Year": "Season"})
    mr["Season"] = mr["Season"].astype("int64")
    mr["BatterId"] = mr["BatterId"].astype("int64")
    mm = pd.merge_asof(
        mrows.sort_values("Season"), mr.sort_values("Season"),
        on="Season", by="BatterId", direction="backward")
    mm = mm.sort_values("_ord").reset_index(drop=True)

    def lgv(table, keys, c):
        idx = pd.MultiIndex.from_arrays(keys) if isinstance(keys, list) \
            else keys
        return table.reindex(idx)[c].values

    pri_b = {c: lgv(lgh, [prior_seas, rows["p_throws"]], c)
             for c in CLASSES}
    out = _shrunk_rates(out, "b_", "b_pa", BAT_RATES, pri_b,
                        stores["eb_k"], milb=mm)
    pri_bh = pri_b
    out = _shrunk_rates(out, "bh_", "bh_pa", BAT_RATES, pri_bh,
                        stores["eb_k"], milb=mm)
    pri_p = {c: lgv(lgs, [prior_seas, rows["stand"]], c) for c in CLASSES}
    out = _shrunk_rates(out, "p_", "p_pa", PIT_RATES, pri_p,
                        stores["eb_k"])
    out = _shrunk_rates(out, "ph_", "ph_pa", PIT_RATES, pri_p,
                        stores["eb_k"])

    # ---- batted-ball quality and pitch-level discipline
    for panel, ent, pref in (("panel_bat_bip", "BatterId", "bq_"),
                             ("panel_pit_bip", "PitcherId", "pq_")):
        cols = ["bip", "ev_n", "ev_sum", "hard", "barrel", "gb", "air"]
        m = merge_asof_panel(rows, stores[panel], [ent], cols, pref)
        out[pref + "ev"] = m[pref + "ev_sum"] / m[pref + "ev_n"].clip(
            lower=1e-9)
        out[pref + "hard"] = m[pref + "hard"] / m[pref + "bip"].clip(
            lower=1e-9)
        out[pref + "barrel"] = m[pref + "barrel"] / m[pref + "bip"].clip(
            lower=1e-9)
        out[pref + "gb"] = m[pref + "gb"] / m[pref + "bip"].clip(
            lower=1e-9)
        out[pref + "log_bip"] = np.log1p(m[pref + "bip"])
    bcols = ["n", "sw_n", "wh_n", "oz_n", "oz_sw", "oz_wh", "z_n",
             "cs_n", "fb95_n", "fb95_sw", "fb95_wh", "brk_n", "brk_sw",
             "brk_wh"]
    m = merge_asof_panel(rows.rename(columns={"BatterId": "PlayerId"}),
                         stores["panel_bat_pitchmet"], ["PlayerId"],
                         bcols, "bm_")
    out["bm_whiff"] = m["bm_wh_n"] / m["bm_sw_n"].clip(lower=1e-9)
    out["bm_chase"] = m["bm_oz_sw"] / m["bm_oz_n"].clip(lower=1e-9)
    out["bm_zcon"] = 1.0 - ((m["bm_wh_n"] - m["bm_oz_wh"])
                            / (m["bm_sw_n"] - m["bm_oz_sw"]).clip(
                                lower=1e-9))
    out["bm_fb95_whiff"] = m["bm_fb95_wh"] / m["bm_fb95_sw"].clip(
        lower=1e-9)
    out["bm_brk_whiff"] = m["bm_brk_wh"] / m["bm_brk_sw"].clip(lower=1e-9)
    pcols = ["n", "sw_n", "wh_n", "oz_n", "oz_sw", "z_n", "cs_n",
             "edge_n", "fb_n", "fb_v", "fp_n", "fp_s"]
    m = merge_asof_panel(rows.rename(columns={"PitcherId": "PlayerId"}),
                         stores["panel_pit_pitchmet"], ["PlayerId"],
                         pcols, "pm_")
    out["pm_whiff"] = m["pm_wh_n"] / m["pm_sw_n"].clip(lower=1e-9)
    out["pm_chase"] = m["pm_oz_sw"] / m["pm_oz_n"].clip(lower=1e-9)
    out["pm_zone"] = m["pm_z_n"] / m["pm_n"].clip(lower=1e-9)
    out["pm_edge"] = m["pm_edge_n"] / m["pm_n"].clip(lower=1e-9)
    out["pm_fstrike"] = m["pm_fp_s"] / m["pm_fp_n"].clip(lower=1e-9)
    out["pm_velo"] = m["pm_fb_v"] / m["pm_fb_n"].clip(lower=1e-9)
    mf = merge_asof_panel(rows.rename(columns={"PitcherId": "PlayerId"}),
                          stores["panel_pit_velo_fast"], ["PlayerId"],
                          ["fb_n_fast", "fb_v_fast"], "pv_",
                          hl=VELO_HL_FAST)
    velo_fast = mf["pv_fb_v_fast"] / mf["pv_fb_n_fast"].clip(lower=1e-9)
    out["pm_velo_trend"] = velo_fast - out["pm_velo"]

    # ---- arsenal matchup: sum over pitch types of prior-season pitcher
    # usage x batter run value per 100 vs that type
    ap = stores["arsenal_pitcher"]
    ab = stores["arsenal_batter"]
    key = rows[["PitcherId", "BatterId", "Season"]].copy()
    key["_ord"] = np.arange(n)
    ja = key.merge(ap, left_on=["PitcherId", "Season"],
                   right_on=["PlayerId", "Year"], how="left")
    ja = ja.merge(ab, left_on=["BatterId", "Season", "PitchType"],
                  right_on=["PlayerId", "Year", "PitchType"], how="left",
                  suffixes=("", "_b"))
    ja["contrib"] = ja["usage"] * ja["rv100"]
    ars = ja.groupby("_ord")["contrib"].sum(min_count=1)
    out["ars_matchup"] = ars.reindex(np.arange(n)).values

    # ---- park, ump, defense, era, weather, bio
    pf = stores["park_factors"]
    j = rows[["Venue", "Season"]].merge(
        pf, left_on=["Venue", "Season"], right_on=["Venue", "Year"],
        how="left")
    for c in ("HR", "1B", "K", "BB", "3B"):
        out[f"pf_{c}"] = j[f"pf_{c}"].fillna(1.0).values
    uf = stores["ump_factors"]
    j = rows[["HpUmpId", "Season"]].copy()
    j["HpUmpId"] = _num(j["HpUmpId"])
    j = j.merge(uf, left_on=["HpUmpId", "Season"],
                right_on=["HpUmpId", "Year"], how="left")
    out["uf_K"] = j["uf_K"].fillna(1.0).values
    out["uf_BB"] = j["uf_BB"].fillna(1.0).values
    dt_ = stores["defense_team"]
    j = rows[["fld_team", "Season"]].merge(
        dt_, left_on=["fld_team", "Season"], right_on=["Team", "Year"],
        how="left")
    out["def_oaa"] = j["def_oaa"].fillna(0.0).values
    out["frame_rv"] = j["FrameRV_pt"].fillna(0.0).values

    dob = stores["dob"]
    j = rows[["BatterId", "Date"]].merge(
        dob, left_on="BatterId", right_on="PlayerId", how="left")
    out["b_age"] = ((j["Date"] - j["dob"]).dt.days / 365.25).values
    j = rows[["PitcherId", "Date"]].merge(
        dob, left_on="PitcherId", right_on="PlayerId", how="left")
    out["p_age"] = ((j["Date"] - j["dob"]).dt.days / 365.25).values

    out["same_hand"] = (rows["stand"].astype(str)
                        == rows["p_throws"].astype(str)).astype(float)
    out["tto"] = _num(rows["tto"]).fillna(1.0)
    out["home_bat"] = _num(rows["home_bat"]).fillna(0.0)
    out["rest_p"] = _num(rows["rest_p"]).clip(upper=15)
    out["temp"] = _num(rows["Temp"])
    out["humidity"] = _num(rows["Humidity"])
    out["pressure"] = _num(rows["Pressure"])
    tK = (_num(rows["Temp"]) - 32) * 5 / 9 + 273.15
    out["air_density"] = _num(rows["Pressure"]) / tK
    dome = rows["Condition"].astype(str).str.contains(
        "Dome|Roof Closed", case=False, na=False)
    out["dome"] = dome.astype(float)
    radial = rows["WindDir"].map(WIND_RADIAL)
    out["wind_out"] = np.where(dome, 0.0,
                               _num(rows["WindSpeed"]) * radial)
    out["night"] = (rows["DayNight"].astype(str).str.lower()
                    == "night").astype(float)
    out["season_idx"] = rows["Season"] - 2015
    out["era_2020"] = (rows["Season"] == 2020).astype(float)
    out["era_sticky"] = (rows["Date"]
                         >= pd.Timestamp("2021-06-21")).astype(float)
    out["era_shiftban"] = (rows["Season"] >= 2023).astype(float)
    out["era_abs"] = (rows["Season"] >= 2026).astype(float)

    drop = ([c for c in out.columns
             if any(c == f"{p}{cl}" for p in ("b_", "bh_", "p_", "ph_")
                    for cl in CLASSES + ["pa", "HBP", "IPO"])]
            + ["Date", "Season"])
    X = out.drop(columns=[c for c in drop if c in out.columns])
    return X, list(X.columns)


# --------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true",
                    help="build every store")
    ap.add_argument("--only", default="",
                    help="comma list: pa,panels,effects,priors,pbp,"
                         "hazard,forecast")
    args = ap.parse_args()
    if not (args.build or args.only):
        ap.error("nothing to do: pass --build or --only ...")
    steps = set(args.only.split(",")) if args.only else {
        "pa", "panels", "effects", "priors", "pbp", "hazard", "part",
        "forecast"}

    STORES.mkdir(parents=True, exist_ok=True)
    pa = None
    if "pa" in steps:
        pa = build_pa_table()
    if pa is None and steps - {"pa", "forecast"}:
        pa = pd.read_parquet(STORES / "pa_table.parquet")
        print(f"pa_table loaded: {len(pa):,} rows", flush=True)
    if "panels" in steps:
        build_outcome_panels(pa)
        build_bip_panels()
        build_pitchmetric_panels()
        build_leash_panel()
    if "effects" in steps:
        build_park_factors(pa)
        build_ump_factors(pa)
        build_defense_tables()
        build_arsenal_tables()
    if "priors" in steps:
        build_league_rates(pa)
        build_eb_k(pa)
        build_milb_priors()
    if "pbp" in steps:
        build_pattern_table(pa)
        build_sb_table(pa)
    if "hazard" in steps:
        build_hazard_table(pa)
    if "part" in steps:
        build_participation(pa)
    if "forecast" in steps:
        build_forecast_error()

    manifest = {"built": date.today().isoformat(),
                "stores": sorted(p.name for p in STORES.iterdir())}
    (STORES / "build_manifest.json").write_text(
        json.dumps(manifest, indent=1))
    print("stores build complete", flush=True)


if __name__ == "__main__":
    main()
