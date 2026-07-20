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
  park_geo.parquet          static fence distances (LF/CF/RF) and
                            elevation by venue (porch/carry geometry)
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
import os
import shutil
from collections import deque
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
RAW = DATA / "raw_pitches"
ART = ROOT / "Model" / "artifacts"
STORES = ART / "stores"


def write_artifact(path, write_fn, backup=True):
    """Overwrite a serve artifact safely (the odds-store discipline):
    copy the existing file to <dir>/backups/ first (single-generation
    last-known-good, the rollback path after a bad retrain/replay), then
    write to a temp path and swap in atomically — a crash mid-write can
    never leave a truncated artifact serving. write_fn receives the temp
    Path. backup=False for cheap derived reports where atomicity is
    enough."""
    path = Path(path)
    if backup and path.exists():
        bdir = path.parent / "backups"
        bdir.mkdir(exist_ok=True)
        shutil.copy2(path, bdir / path.name)
    tmp = path.with_name(path.name + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)

DECAY_HL = 90.0          # days; skill half-life for decayed panels
VELO_HL_FAST = 30.0      # short-window velo (in-season trend vs baseline)
PRIOR_SEASONS_PARK = 3   # park factors pool this many prior seasons
K_PARK = 2000            # PA of shrinkage toward 1 for park factors
K_UMP = 1500             # PA of shrinkage toward 1 for ump factors
MIN_MLB_PA_FOR_OWN = 400  # below this decayed PA, MiLB prior blends in

# mechanical-consistency / regime / matchup-profile knobs
PULL_HL = 365.0          # pull tendency is stable; slow half-life
HRPT_HL = 730.0          # HR-by-pitch-class profile: two-year half-life
CONSIST_MIN_FB = 100     # tracked fastballs before velo SD means much
CONSIST_MIN_RP = 100     # release points before scatter means much
STRETCH_MIN_N = 30       # per-split fastballs before the delta counts
RELSEP_MIN_N = 50        # per-class release points before separation
PULL_DEG = 15.0          # spray beyond this = pulled (Savant convention)
PULL_K = 100.0           # air balls of shrinkage for pull share
HRPT_K = 15.0            # HRs of shrinkage for the HR pitch-class mix
RAMP_NP = 60             # last start under 60 pitches = ramping up
SHORT_START_OUTS = 12    # <= 12 outs = short start (opener / quick hook)
CARRY_CF_W = 0.7         # CF wind helps both pull sides at this weight
CARRY_OPP_W = 0.2        # opposite-field wind still helps a little

# form-trend / HR-quality / park-geometry knobs (2026-07-19 wave 3,
# mined from the old model's remaining families)
TREND_HL = 30.0          # fast half-life; trend = fast rate - 90d rate
HRQ_K = 10.0             # career HRs of shrinkage for HR EV/distance
LG_SPRINT = 27.0         # ft/s league sprint speed (leg-hit pivot)
SEA_LVL_FT = 0.005       # ft of HR carry per ft of park elevation
PORCH_REF = 330.0        # nominal pull-fence distance porch terms center on
AIR_REF = 0.1017         # Pressure/tempK at 29.92 inHg & 70F ("thin air" 0)
PITCH_CLASS = {          # arsenal codes -> fast / breaking / offspeed
    "FF": "fast", "SI": "fast", "FC": "fast",
    "SL": "brk", "ST": "brk", "CU": "brk", "KC": "brk", "SV": "brk",
    "KN": "brk", "CS": "brk", "SC": "brk",
    "CH": "off", "FS": "off", "FO": "off", "EP": "off"}

# wind direction -> (field it blows toward/from, out=+1/in=-1); both the
# boxscore casing ("Out To CF") and the live-slate casing ("Out To Cf")
_WIND_FIELD, _WIND_SIGN = {}, {}
for _d, _f, _s in (("Out To LF", "L", 1.0), ("In From LF", "L", -1.0),
                   ("Out To CF", "C", 1.0), ("In From CF", "C", -1.0),
                   ("Out To RF", "R", 1.0), ("In From RF", "R", -1.0)):
    for _k in (_d, _d.title()):
        _WIND_FIELD[_k] = _f
        _WIND_SIGN[_k] = _s

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
    "fielder_2", "fielder_3", "fielder_4", "fielder_5", "fielder_6",
    "fielder_7", "fielder_8", "fielder_9", "home_team", "away_team",
]


# ------------------------------------------------ contact-tree pieces
# Hierarchical PA model: T1 {K, BB, HBP, in-play} -> T2 batted-ball
# type (the A2 model) -> T3 outcome | bb-type {out, 1B, 2B, 3B, HR}
# conditioned on the same features (park/defense included). Composed
# back to the flat 8-class vector plus the CLASS-CONDITIONAL bb-type
# mix P(bb | outcome) the sim samples from (the flat design sampled bb
# independently of the outcome class — HRs drew ground balls).

T1_CLASSES = ["K", "BB", "HBP", "IP"]
T3_CLASSES = ["IPO", "1B", "2B", "3B", "HR"]
BB_DUMMIES = ["bb_fly", "bb_line", "bb_pop"]   # ground ball = baseline


def tree_compose(p1, p2, p3):
    """(T1 [n,4], T2 [n,4], T3 [n,4,5]) -> (p8 [n,8] in CLASSES order,
    a2cond [n,8,4] = P(bb | class); rows for K/BB/HBP carry the
    marginal, the sim never reads them)."""
    n = len(p1)
    joint = p2[:, :, None] * p3                 # [n, bb, T3 class]
    marg = joint.sum(axis=1)                    # [n, 5]
    ip = p1[:, 3]
    p8 = np.zeros((n, len(CLASSES)))
    p8[:, 0], p8[:, 1], p8[:, 2] = p1[:, 0], p1[:, 1], p1[:, 2]
    for j, c in enumerate(T3_CLASSES):
        p8[:, CLASSES.index(c)] = ip * marg[:, j]
    a2cond = np.zeros((n, len(CLASSES), 4))
    a2cond[:, :3, :] = p2[:, None, :]
    for j, c in enumerate(T3_CLASSES):
        a2cond[:, CLASSES.index(c), :] = (
            joint[:, :, j] / np.clip(marg[:, j:j + 1], 1e-12, None))
    return p8, a2cond


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
    for _fi in range(3, 10):        # actual defense on the field
        pa[f"fielder_{_fi}"] = _num(pa[f"fielder_{_fi}"])
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
            "Humidity", "Pressure", "HpUmpId",
            "fielder_3", "fielder_4", "fielder_5", "fielder_6",
            "fielder_7", "fielder_8", "fielder_9"]
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
    dec = {}
    for c in stat_cols:
        cum = (df[c].astype(float) * up).groupby(
            [df[k] for k in entity_cols]).cumsum()
        dec[c + "_d"] = cum * down
    return pd.concat([df[entity_cols + ["Date"]], pd.DataFrame(dec)],
                     axis=1)


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
    out = pd.DataFrame({prefix + c: merged[c + "_d"] * decay
                        for c in stat_cols})
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
        ("panel_bat_out", "BatterId", None, DECAY_HL),
        ("panel_bat_out_hand", "BatterId", "p_throws", DECAY_HL),
        ("panel_bat_out_loc", "BatterId", "home_bat", DECAY_HL),
        ("panel_pit_out", "PitcherId", None, DECAY_HL),
        ("panel_pit_out_hand", "PitcherId", "stand", DECAY_HL),
        ("panel_pit_out_tto", "PitcherId", "tto", DECAY_HL),
        # fast-decay twins of the overall panels: assemble emits the
        # DELTA vs the 90d rate (hot/cold form the slow panel smooths)
        ("panel_bat_out_fast", "BatterId", None, TREND_HL),
        ("panel_pit_out_fast", "PitcherId", None, TREND_HL),
    ]
    for name, ent, hand, hl in specs:
        sub = pa.dropna(subset=[ent])
        if hand == "tto":
            # per-times-through-order pitcher rates: the sim's TTO axis
            # gets pitcher-specific susceptibility instead of the
            # league-average TTO effect alone
            sub = sub.assign(tto=_num(sub["tto"]).clip(1, 3)
                             .astype("int64"))
        daily = _daily_class_counts(sub, ent, hand)
        ents = [ent] + ([hand] if hand else [])
        panel = decayed_panel(daily, ents, ["pa"] + CLASSES, hl=hl)
        panel.to_parquet(STORES / f"{name}.parquet", index=False)
        print(f"{name}: {len(panel):,} entity-days", flush=True)


def build_bip_panels():
    bip = read_csv("mlb_statcast_bip.csv",
                   usecols=["Date", "BatterId", "PitcherId", "BBType",
                            "ExitVelo", "LSA", "Events", "xBA",
                            "xwOBA"])
    bip["Date"] = pd.to_datetime(bip["Date"])
    bip["ev"] = _num(bip["ExitVelo"])
    bip["hard"] = (bip["ev"] >= 95).astype(float)
    bip["barrel"] = (_num(bip["LSA"]) == 6).astype(float)
    bip["gb"] = (bip["BBType"] == "ground_ball").astype(float)
    bip["air"] = bip["BBType"].isin(["fly_ball", "line_drive"]).astype(float)
    bip["ev_n"] = bip["ev"].notna().astype(float)
    bip["ev_sum"] = bip["ev"].fillna(0.0)
    # realized-vs-expected on contact: the luck-regression pair
    bip["hit"] = bip["Events"].isin(["single", "double", "triple",
                                     "home_run"]).astype(float)
    xba = _num(bip["xBA"])
    bip["xba_n"] = xba.notna().astype(float)
    bip["xba_sum"] = xba.fillna(0.0)
    xw = _num(bip["xwOBA"])
    bip["xw_n"] = xw.notna().astype(float)
    bip["xw_sum"] = xw.fillna(0.0)
    bip["hr"] = (bip["Events"] == "home_run").astype(float)
    bip["bip"] = 1.0
    stats = ["bip", "ev_n", "ev_sum", "hard", "barrel", "gb", "air",
             "hit", "xba_n", "xba_sum", "xw_n", "xw_sum", "hr"]
    for name, ent in (("panel_bat_bip", "BatterId"),
                      ("panel_pit_bip", "PitcherId")):
        daily = bip.groupby([ent, "Date"], sort=False)[stats].sum(
            ).reset_index()
        panel = decayed_panel(daily, [ent], stats)
        panel.to_parquet(STORES / f"{name}.parquet", index=False)
        print(f"{name}: {len(panel):,} entity-days", flush=True)
    # batter-vs-pitcher direct history: pairwise contact quality, slow
    # half-life, consumed as heavily-shrunk residuals off the batter's
    # own baseline (bvp_n carries the evidence mass)
    bip = bip.dropna(subset=["BatterId", "PitcherId"]).copy()
    bip["BatterId"] = _num(bip["BatterId"]).astype("int64")
    bip["PitcherId"] = _num(bip["PitcherId"]).astype("int64")
    pair = bip.groupby(["BatterId", "PitcherId", "Date"], sort=False)[
        ["bip", "xw_n", "xw_sum", "hr"]].sum().reset_index()
    bvp = decayed_panel(pair, ["BatterId", "PitcherId"],
                        ["bip", "xw_n", "xw_sum", "hr"], hl=HRPT_HL)
    bvp.to_parquet(STORES / "panel_bvp.parquet", index=False)
    print(f"panel_bvp: {len(bvp):,} pair-days", flush=True)


def build_pitchmetric_panels():
    b = read_csv("mlb_pitch_daily_batters.csv",
                 usecols=["PlayerId", "Date", "n", "sw_n", "wh_n", "oz_n",
                          "oz_sw", "oz_wh", "z_n", "cs_n",
                          "fb95_n", "fb95_sw", "fb95_wh",
                          "fbmid_n", "fbmid_sw", "fbmid_wh",
                          "fblo_n", "fblo_sw", "fblo_wh",
                          "brk_n", "brk_sw", "brk_wh",
                          "off_n", "off_sw", "off_wh",
                          "ts_n", "ts_sw", "ts_wh",
                          "f32_n", "f32_b", "fp_n", "fp_sw",
                          "con_n", "con_xw",
                          "fb95_bip", "fb95_xw", "fbmid_bip", "fbmid_xw",
                          "brk_bip", "brk_xw", "off_bip", "off_xw"])
    b["Date"] = pd.to_datetime(b["Date"])
    stats_b = [c for c in b.columns if c not in ("PlayerId", "Date")]
    panel = decayed_panel(b, ["PlayerId"], stats_b)
    panel.to_parquet(STORES / "panel_bat_pitchmet.parquet", index=False)
    print(f"panel_bat_pitchmet: {len(panel):,} entity-days", flush=True)

    p = read_csv("mlb_pitch_daily_pitchers.csv",
                 usecols=["PlayerId", "Date", "n", "sw_n", "wh_n", "oz_n",
                          "oz_sw", "oz_wh", "z_n", "cs_n", "edge_n",
                          "fb_n", "fb_v", "fp_n", "fp_s",
                          "fb95_n", "fbmid_n", "fblo_n",
                          "brk_n", "brk_sw", "brk_wh",
                          "off_n", "off_sw", "off_wh",
                          "ts_n", "ts_sw", "ts_wh",
                          "f32_n", "f32_z", "f32_b",
                          "c02_n", "c02_w",
                          "ah_n", "ah_brk", "ah_off",
                          "bh_n", "bh_brk", "bh_off",
                          "tr_n", "tr_same",
                          "ivb_n", "ivb_sum", "fbe_n", "fbe_sum",
                          "brkmov_n", "brkmov_sum", "fade_w", "fade_num",
                          "con_n", "con_xw",
                          "fb95_bip", "fb95_xw", "brk_bip", "brk_xw"])
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
    gp["outs2_sum"] = gp["outs_sum"] ** 2
    daily = gp.groupby(["PlayerId", "Date"], sort=False)[
        ["starts", "np_sum", "bf_sum", "outs_sum", "outs2_sum"]].sum(
        ).reset_index()
    panel = decayed_panel(daily, ["PlayerId"],
                          ["starts", "np_sum", "bf_sum", "outs_sum",
                           "outs2_sum"])
    panel.to_parquet(STORES / "panel_leash.parquet", index=False)
    print(f"panel_leash: {len(panel):,} starter-days", flush=True)


def build_consistency_panels():
    """Pitcher mechanical-consistency sufficient statistics, decayed:
    fastball velo variance (fb_v2), release-point scatter (rp_*2), the
    windup-vs-stretch velo split (fbstr_*), and the fastball-vs-breaking
    release centroids (rpf_/rpb_) whose distance is a deception tell.
    All pre-summed by scrape_pitches.py; the panel decays them so
    assemble_features can form as-of ratios."""
    cols = ["PlayerId", "Date", "fb_n", "fb_v", "fb_v2",
            "rp_n", "rp_x", "rp_x2", "rp_z", "rp_z2",
            "fbstr_n", "fbstr_v",
            "rpf_n", "rpf_x", "rpf_z", "rpb_n", "rpb_x", "rpb_z"]
    p = read_csv("mlb_pitch_daily_pitchers.csv", usecols=cols)
    p["Date"] = pd.to_datetime(p["Date"])
    stats = [c for c in cols if c not in ("PlayerId", "Date")]
    for c in stats:
        p[c] = _num(p[c]).fillna(0.0)
    panel = decayed_panel(p, ["PlayerId"], stats)
    panel.to_parquet(STORES / "panel_pit_consist.parquet", index=False)
    print(f"panel_pit_consist: {len(panel):,} pitcher-days", flush=True)


def build_il_table():
    """IL stint history -> one row per ACTIVATION (return to the
    roster), carrying stint length, rehab flag, IL60, and cumulative IL
    days within the activation season. Activation is public before that
    day's games, so an as-of backward join on Date is leakage-free."""
    il = read_csv("mlb_il.csv")
    il["PlayerId"] = _num(il["PlayerId"])
    il["Date"] = pd.to_datetime(il["ActDate"], errors="coerce")
    il = il.dropna(subset=["PlayerId", "Date"]).copy()
    il["last_len"] = _num(il["StintDays"]).fillna(0.0)
    il["rehab"] = _num(il["Rehab"]).fillna(0.0)
    il["il60"] = _num(il["IL60"]).fillna(0.0)
    il["act_season"] = il["Date"].dt.year
    il = il.sort_values(["PlayerId", "Date"])
    il["szn_days"] = il.groupby(["PlayerId", "act_season"])[
        "last_len"].cumsum()
    il["act_date"] = il["Date"]
    out = il[["PlayerId", "Date", "act_date", "act_season", "last_len",
              "rehab", "il60", "szn_days"]]
    out.to_parquet(STORES / "il_stints.parquet", index=False)
    print(f"il_stints: {len(out):,} activations", flush=True)


def build_bat_tracking_table():
    """Statcast bat tracking (2023+), consumed PRIOR-season like every
    other season-grain table. NaN before the tracked era — XGBoost
    handles missing natively."""
    bt = read_csv("mlb_bat_tracking.csv",
                  usecols=["Year", "PlayerId", "Swings", "BatSpeed",
                           "HardSwingRate", "SwingLength",
                           "SquaredUpPerSwing"])
    for c in bt.columns:
        bt[c] = _num(bt[c])
    bt = bt[bt["Swings"].fillna(0) >= 50].copy()
    bt["Year"] = bt["Year"] + 1              # serve year = prior + 1
    bt = bt.rename(columns={"BatSpeed": "bt_speed",
                            "SwingLength": "bt_swlen",
                            "HardSwingRate": "bt_hardsw",
                            "SquaredUpPerSwing": "bt_squp"})
    bt[["Year", "PlayerId", "bt_speed", "bt_swlen", "bt_hardsw",
        "bt_squp"]].to_parquet(STORES / "bat_tracking.parquet",
                               index=False)
    print(f"bat_tracking: {len(bt):,} player-years", flush=True)


def build_pull_table():
    """Batter pull tendency on AIR contact (fly/line), from the raw
    pitch archive's hit coordinates. Spray angle in the Savant frame
    (plate at (125.42, 198.27); negative = LF side); pulled = beyond
    PULL_DEG toward the batter's pull field. Slow half-life — pull is a
    stable trait that powers the wind-carry interaction."""
    import pyarrow.parquet as pq
    frames = []
    for f in sorted(RAW.glob("pitches_*.parquet")):
        t = pq.read_table(f, columns=["game_date", "batter", "stand",
                                      "bb_type", "hc_x", "hc_y"])
        d = t.to_pandas()
        d = d[d["bb_type"].isin(["fly_ball", "line_drive"])]
        frames.append(d.dropna(subset=["hc_x", "hc_y"]))
    bip = pd.concat(frames, ignore_index=True)
    ang = np.degrees(np.arctan2(_num(bip["hc_x"]) - 125.42,
                                198.27 - _num(bip["hc_y"])))
    right = bip["stand"].astype(str) == "R"
    bip["pull"] = np.where(right, ang < -PULL_DEG,
                           ang > PULL_DEG).astype(float)
    bip["air"] = 1.0
    bip["Date"] = pd.to_datetime(bip["game_date"])
    bip["BatterId"] = _num(bip["batter"]).astype("int64")
    daily = bip.groupby(["BatterId", "Date"], sort=False)[
        ["air", "pull"]].sum().reset_index()
    panel = decayed_panel(daily, ["BatterId"], ["air", "pull"],
                          hl=PULL_HL)
    panel.to_parquet(STORES / "panel_bat_pull.parquet", index=False)
    print(f"panel_bat_pull: {len(panel):,} batter-days "
          f"(league air-pull {bip['pull'].mean():.3f})", flush=True)


def build_hrpt_tables():
    """Batter HR mix by pitch class (fast/breaking/offspeed) as a slow
    panel, plus each pitcher's arsenal usage in the same classes: the
    serve-time dot product (batter class share vs league, weighted by
    what THIS pitcher throws) is the HR-matchup feature the generic
    run-value matchup can't see."""
    ars = read_csv("mlb_pitch_arsenals.csv",
                   usecols=["Year", "PlayerId", "PitchType", "Pitch",
                            "%"])
    disp2cls = {}
    for _, r in ars.drop_duplicates("Pitch").iterrows():
        cls = PITCH_CLASS.get(str(r["PitchType"]))
        if cls:
            disp2cls[str(r["Pitch"])] = cls
    hr = read_csv("mlb_homeruns.csv",
                  usecols=["BatterId", "Date", "Pitch", "Exit Velo",
                           "Distance", "Ballpark"])
    hr["Date"] = pd.to_datetime(hr["Date"], errors="coerce")
    hr["BatterId"] = _num(hr["BatterId"])
    hr = hr.dropna(subset=["BatterId", "Date"]).copy()
    hr["BatterId"] = hr["BatterId"].astype("int64")
    # HR quality: exit velo + sea-level-adjusted distance (each batter's
    # BEST contact — censored sample, so heavily shrunk at assemble).
    # Elevation credit keeps Coors homers from inflating raw power.
    bp = read_csv("mlb_ballparks.csv",
                  usecols=["Ballpark", "Elevation_ft"]
                  ).drop_duplicates("Ballpark")
    hr = hr.merge(bp, on="Ballpark", how="left")
    ev = _num(hr["Exit Velo"])
    dist = (_num(hr["Distance"])
            - SEA_LVL_FT * _num(hr["Elevation_ft"]).fillna(0.0))
    hr["ev_n"], hr["ev_sum"] = ev.notna().astype(float), ev.fillna(0.0)
    hr["dist_n"] = dist.notna().astype(float)
    hr["dist_sum"] = dist.fillna(0.0)
    hr["cls"] = hr["Pitch"].astype(str).map(disp2cls)
    hr = hr.dropna(subset=["cls"])
    for cls in ("fast", "brk", "off"):
        hr[f"hr_{cls}"] = (hr["cls"] == cls).astype(float)
    hr["hr_n"] = 1.0
    qcols = ["hr_n", "hr_fast", "hr_brk", "hr_off",
             "ev_n", "ev_sum", "dist_n", "dist_sum"]
    daily = hr.groupby(["BatterId", "Date"], sort=False)[
        qcols].sum().reset_index()
    panel = decayed_panel(daily, ["BatterId"], qcols, hl=HRPT_HL)
    panel.to_parquet(STORES / "panel_bat_hrmix.parquet", index=False)
    lg = {c: float(hr[f"hr_{c}"].mean()) for c in ("fast", "brk", "off")}
    lg["ev_mean"] = float((hr["ev_sum"].sum()
                           / max(hr["ev_n"].sum(), 1.0)))
    lg["dist_mean"] = float((hr["dist_sum"].sum()
                             / max(hr["dist_n"].sum(), 1.0)))
    (STORES / "hrmix_league.json").write_text(json.dumps(lg, indent=1))
    ars["Year"] = _num(ars["Year"]) + 1      # serve year = prior + 1
    ars["usage"] = _num(ars["%"]) / 100.0
    ars["cls"] = ars["PitchType"].astype(str).map(PITCH_CLASS)
    cu = (ars.dropna(subset=["cls"])
          .groupby(["Year", "PlayerId", "cls"])["usage"].sum()
          .unstack(fill_value=0.0).reset_index())
    for c in ("fast", "brk", "off"):
        if c not in cu.columns:
            cu[c] = 0.0
    cu = cu.rename(columns={"fast": "u_fast", "brk": "u_brk",
                            "off": "u_off"})
    cu[["Year", "PlayerId", "u_fast", "u_brk", "u_off"]].to_parquet(
        STORES / "arsenal_class_usage.parquet", index=False)
    print(f"hrmix: {len(panel):,} batter-days; usage: {len(cu):,} "
          f"pitcher-years; league mix "
          f"{ {k: round(v, 3) for k, v in lg.items()} }", flush=True)


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

    # handed HR factors: same windows split by batter stand — the porch
    # asymmetry the pooled factor can't see; half the shrink mass since
    # per-hand samples are half the size
    ratesh = pa.groupby(["Venue", "Season", "stand"], sort=False).agg(
        pa_n=("label", "size"),
        HR=("label", lambda s: (s == "HR").sum())).reset_index()
    lgh_ = pa.groupby(["Season", "stand"]).agg(
        n=("label", "size"),
        HR=("label", lambda s: (s == "HR").sum())).reset_index()
    lgh_["rate"] = lgh_["HR"] / lgh_["n"].clip(lower=1)
    rows_h = []
    for year in range(int(pa.Season.min()) + 1,
                      int(pa.Season.max()) + 2):
        in_win = ((ratesh.Season >= year - PRIOR_SEASONS_PARK)
                  & ((ratesh.Season < year)
                     | ((ratesh.Season == year) & (year == cur))))
        window = ratesh[in_win]
        if window.empty:
            continue
        agg = window.groupby(["Venue", "stand"])[["pa_n", "HR"]].sum(
            ).reset_index()
        lw = lgh_[(lgh_.Season >= year - PRIOR_SEASONS_PARK)
                  & ((lgh_.Season < year)
                     | ((lgh_.Season == year) & (year == cur)))]
        lr = lw.groupby("stand")["rate"].mean()
        obs = agg["HR"] / agg["pa_n"].clip(lower=1)
        f = obs / agg["stand"].map(lr).clip(lower=1e-9)
        w = agg["pa_n"] / (agg["pa_n"] + K_PARK / 2)
        agg["pf_HR_h"] = 1.0 + w * (f - 1.0)
        agg["Year"] = year
        rows_h.append(agg[["Venue", "Year", "stand", "pf_HR_h"]])
    outh = pd.concat(rows_h, ignore_index=True)
    outh.to_parquet(STORES / "park_hand.parquet", index=False)
    print(f"park_factors: {len(out):,} venue-years "
          f"(+{len(outh):,} handed HR rows)", flush=True)


def build_park_geometry(pa):
    """Static park geometry serving surface: fence distances by field
    plus elevation, keyed by Venue name. Geometry gives HR features the
    park factor can't: WHERE the fences are short lets batter-specific
    pull/porch fits exist. Unmatched venues (renames, international
    sites) come through as NaN and the trees route around them."""
    bp = read_csv("mlb_ballparks.csv",
                  usecols=["Ballpark", "LF", "CF", "RF", "Elevation_ft"])
    bp = bp.rename(columns={"Ballpark": "Venue"}).drop_duplicates("Venue")
    for c in ("LF", "CF", "RF", "Elevation_ft"):
        bp[c] = _num(bp[c])
    # historical names of CURRENT parks (sponsor renames — same physical
    # geometry). Departed parks (Oakland Coliseum, Turner Field, Globe
    # Life PARK, alternate sites) stay NaN deliberately.
    aliases = {"Miller Park": "American Family Field",
               "AT&T Park": "Oracle Park",
               "Safeco Field": "T-Mobile Park",
               "SunTrust Park": "Truist Park",
               "Angel Stadium of Anaheim": "Angel Stadium",
               "U.S. Cellular Field": "Rate Field",
               "Guaranteed Rate Field": "Rate Field",
               "Minute Maid Park": "Daikin Park",
               "Marlins Park": "loanDepot Park",
               "Oriole Park": "Oriole Park at Camden Yards"}
    amap = pd.DataFrame({"Venue": list(aliases),
                         "src": list(aliases.values())})
    extra = amap.merge(bp.rename(columns={"Venue": "src"}), on="src")
    bp = pd.concat([bp, extra[bp.columns]], ignore_index=True)
    bp.to_parquet(STORES / "park_geo.parquet", index=False)
    seen = pa["Venue"].dropna().unique()
    cov = float(np.isin(seen, bp["Venue"].values).mean()) if len(seen) \
        else 0.0
    print(f"park_geo: {len(bp)} parks; covers {cov:.0%} of "
          f"{len(seen)} pa venues", flush=True)


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

    # runs-per-game multiplier: the residual run environment an ump
    # carries beyond what his K/BB factors mediate (zone size at the
    # margins, hit-by-pitch temperament); same windowing as K/BB,
    # game-grain shrinkage (K_UMP_G games of league belief)
    K_UMP_G = 40
    g = read_csv("mlb_games.csv",
                 usecols=["GamePk", "Season", "AwayScore", "HomeScore"])
    u = read_csv("mlb_umpires.csv", usecols=["GamePk", "HpUmpId"])
    g = g.merge(u, on="GamePk", how="inner")
    g["runs"] = _num(g["AwayScore"]) + _num(g["HomeScore"])
    g = g.dropna(subset=["runs", "HpUmpId"])
    g["HpUmpId"] = _num(g["HpUmpId"]).astype("int64")
    g["Season"] = _num(g["Season"]).astype("int64")
    gr = g.groupby(["HpUmpId", "Season"], sort=False).agg(
        g_n=("runs", "size"), r_sum=("runs", "sum")).reset_index()
    lgr = g.groupby("Season")["runs"].mean().rename("lg_rg").reset_index()
    rrows = []
    for year in range(int(g.Season.min()) + 1, int(g.Season.max()) + 2):
        in_win = ((gr.Season < year)
                  | ((gr.Season == year) & (year == cur)))
        window = gr[in_win]
        if window.empty:
            continue
        agg = window.groupby("HpUmpId")[["g_n", "r_sum"]].sum(
            ).reset_index()
        lgw = float(lgr[(lgr.Season < year)
                        | ((lgr.Season == year) & (year == cur))][
            "lg_rg"].mean())
        w = agg["g_n"] / (agg["g_n"] + K_UMP_G)
        agg["uf_R"] = 1.0 + w * ((agg["r_sum"]
                                  / agg["g_n"].clip(lower=1))
                                 / max(lgw, 1e-9) - 1.0)
        agg["Year"] = year
        rrows.append(agg[["HpUmpId", "Year", "uf_R"]])
    out = out.merge(pd.concat(rrows, ignore_index=True),
                    on=["HpUmpId", "Year"], how="outer")
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
    # unearned-run rate: error-proneness OAA's range component misses
    # (prior season, (R-ER)*27/outs from the team's pitching lines)
    gp = read_csv("mlb_game_pitching.csv",
                  usecols=["Season", "Team", "IP", "R", "ER"])
    ip = _num(gp["IP"]).fillna(0.0)
    gp["outs"] = (ip.astype(int) * 3
                  + ((ip - ip.astype(int)) * 10).round().astype(int))
    gp["uer"] = _num(gp["R"]).fillna(0.0) - _num(gp["ER"]).fillna(0.0)
    uer = gp.groupby(["Season", "Team"], sort=False).agg(
        outs=("outs", "sum"), uer=("uer", "sum")).reset_index()
    uer["def_uer"] = uer["uer"] * 27.0 / uer["outs"].clip(lower=1)
    uer["Year"] = _num(uer["Season"]) + 1        # serve year = prior + 1
    out = out.merge(uer[["Year", "Team", "def_uer"]],
                    on=["Year", "Team"], how="left")
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
    # arsenal dynamics: usage entropy (repertoire breadth) and the
    # usage-weighted whiff trend vs the pitcher's own prior season
    full = read_csv("mlb_pitch_arsenals.csv",
                    usecols=["Year", "PlayerId", "PitchType", "%",
                             "Whiff %"])
    full["Year"] = _num(full["Year"])
    full["usage"] = _num(full["%"]) / 100.0
    full["whiff"] = _num(full["Whiff %"])
    meta = full.groupby(["Year", "PlayerId"]).apply(
        lambda d: pd.Series({
            "ars_entropy": float(-np.nansum(np.where(
                d.usage > 0, d.usage * np.log(d.usage), 0.0))),
            "ars_whiff": float(
                np.nansum(d.usage * d.whiff)
                / max(np.nansum(d.usage * d.whiff.notna()), 1e-9)),
        }), include_groups=False).reset_index()
    prev = meta[["Year", "PlayerId", "ars_whiff"]].rename(
        columns={"ars_whiff": "pw"})
    prev["Year"] = prev["Year"] + 1
    meta = meta.merge(prev, on=["Year", "PlayerId"], how="left")
    meta["ars_whiff_trend"] = meta["ars_whiff"] - meta["pw"]
    meta["Year"] = meta["Year"] + 1          # serve year = prior + 1
    meta[["Year", "PlayerId", "ars_entropy", "ars_whiff_trend"]
         ].to_parquet(STORES / "arsenal_meta.parquet", index=False)
    print(f"arsenal tables: {ars.PlayerId.nunique():,} pitchers, "
          f"{arb.PlayerId.nunique():,} batters "
          f"(+meta {len(meta):,} pitcher-years)", flush=True)


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


def build_bat_sched_panel():
    """Batter appearance schedule: cumulative games through each active
    date, for rest-days and games-in-last-N-days fatigue features.
    (Benched by the OLD model for its direct prop heads; retried here
    because this architecture is a different estimator — the A/B replay
    is the judge, not the old model's notes.)"""
    gb = read_csv("mlb_game_batting.csv",
                  usecols=["PlayerId", "Date", "PA"])
    gb = gb[_num(gb["PA"]).fillna(0) > 0].copy()
    gb["Date"] = pd.to_datetime(gb["Date"])
    gb["PlayerId"] = _num(gb["PlayerId"]).astype("int64")
    daily = (gb.groupby(["PlayerId", "Date"], sort=False).size()
             .rename("g").reset_index())
    daily = daily.sort_values(["PlayerId", "Date"])
    daily["cum_g"] = daily.groupby("PlayerId")["g"].cumsum().astype(
        float)
    daily["last_d"] = daily["Date"]
    daily = daily.rename(columns={"PlayerId": "BatterId"})
    daily[["BatterId", "Date", "cum_g", "last_d"]].to_parquet(
        STORES / "panel_bat_sched.parquet", index=False)
    print(f"panel_bat_sched: {len(daily):,} batter-days", flush=True)


def build_league_env(pa):
    """Daily cumulative league totals -> trailing-30-day environment
    rates at assemble time: the in-season offense drift (cold April,
    summer HR surge, September callups) that era flags and per-game
    weather cannot see."""
    d = pa.groupby("Date").agg(
        pa_n=("label", "size"),
        k=("label", lambda s: (s == "K").sum()),
        bb=("label", lambda s: (s == "BB").sum()),
        hr=("label", lambda s: (s == "HR").sum())).reset_index()
    d = d.sort_values("Date")
    for c in ("pa_n", "k", "bb", "hr"):
        d[c + "_c"] = d[c].cumsum().astype(float)
    g = read_csv("mlb_games.csv",
                 usecols=["Date", "AwayScore", "HomeScore"])
    g["Date"] = pd.to_datetime(g["Date"])
    gd = pd.DataFrame({
        "Date": g["Date"],
        "runs": _num(g["AwayScore"]).fillna(0)
        + _num(g["HomeScore"]).fillna(0),
        "g": 1.0})
    gd = gd.groupby("Date")[["runs", "g"]].sum().reset_index()
    gd = gd.sort_values("Date")
    gd["runs_c"] = gd["runs"].cumsum()
    gd["g_c"] = gd["g"].cumsum()
    env = d[["Date", "pa_n_c", "k_c", "bb_c", "hr_c"]].merge(
        gd[["Date", "runs_c", "g_c"]], on="Date",
        how="outer").sort_values("Date")
    env = env.ffill().fillna(0.0)
    env.to_parquet(STORES / "league_env_daily.parquet", index=False)
    print(f"league_env_daily: {len(env):,} dates", flush=True)


def build_milb_priors_pit():
    """MiLB-translated class-rate priors for PITCHERS (debut and callup
    starters otherwise serve on pure league priors). Translation factors
    are fit on AAA/AA pitchers with a real MLB season the following
    year; the season MLB file lacks 2B/3B allowed, so the hit classes
    share the pooled H factor (composition assumed to survive the
    jump)."""
    milb = read_csv("milb_pitching.csv")
    milb = milb[milb["Level"].isin(["AAA", "AA"])].copy()
    for c in ("TBF", "H", "2B", "3B", "HR", "BB", "SO"):
        milb[c] = _num(milb[c]).fillna(0)
    milb["Year"] = _num(milb["Year"])
    milb["PlayerId"] = _num(milb["PlayerId"])
    milb = milb[milb.TBF >= 150]
    milb["1B"] = milb["H"] - milb["2B"] - milb["3B"] - milb["HR"]
    mlb = read_csv("mlb_pitching_stats.csv",
                   usecols=["Year", "PlayerId", "TBF", "H", "HR", "BB",
                            "SO"])
    for c in ("TBF", "H", "HR", "BB", "SO"):
        mlb[c] = _num(mlb[c]).fillna(0)
    mlb["Year"] = _num(mlb["Year"])
    mlb = mlb[mlb.TBF >= 200]

    pairs = milb.merge(mlb, on="PlayerId", suffixes=("_m", "_M"))
    pairs = pairs[pairs["Year_M"] == pairs["Year_m"] + 1]
    fac_cols = {"K": "SO", "BB": "BB", "HR": "HR", "H": "H"}
    factors = {}
    for lvl in ("AAA", "AA"):
        sub = pairs[pairs.Level == lvl]
        f = {}
        for cls, col in fac_cols.items():
            r_m = sub[f"{col}_m"].sum() / max(sub["TBF_m"].sum(), 1)
            r_M = sub[f"{col}_M"].sum() / max(sub["TBF_M"].sum(), 1)
            f[cls] = float(np.clip(r_M / max(r_m, 1e-6), 0.1, 3.0))
        factors[lvl] = f

    rate_cols = {"K": "SO", "BB": "BB", "HR": "HR", "1B": "1B",
                 "2B": "2B", "3B": "3B"}
    rows = []
    for pid, g in milb.groupby("PlayerId"):
        g = g.sort_values("Year")
        for entry_year in g.Year.unique() + 1:
            past = g[g.Year < entry_year].tail(2)
            if past.empty:
                continue
            w = past.TBF.values
            row = {"PlayerId": int(pid), "Year": int(entry_year)}
            for cls, col in rate_cols.items():
                fkey = cls if cls in fac_cols else "H"
                tr = [(r[col] / r.TBF) * factors[r.Level][fkey]
                      for _, r in past.iterrows()]
                row[f"prior_{cls}"] = float(np.average(tr, weights=w))
            row["prior_pa"] = float(past.TBF.sum())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(STORES / "milb_priors_pit.parquet", index=False)
    (STORES / "milb_level_factors_pit.json").write_text(
        json.dumps(factors, indent=1))
    print(f"milb_priors_pit: {len(out):,} pitcher-entry-years; factors "
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
    games = read_csv("mlb_games.csv",
                     usecols=["GamePk", "Season", "GameType"])
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

    # postseason flag — October attempt selectivity runs opposite to
    # success selectivity, so the models condition on it explicitly
    opp = opp.merge(games[["game_pk", "GameType"]], on="game_pk",
                    how="left")
    opp["post"] = opp["GameType"].isin(["F", "D", "L", "W"]).astype(int)

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
            "inning", "post", "attempt", "success", "SprintSpeed",
            "sb_allowed_rate", "cs_rate", "PopTime", "CSAA"]
    keep = [c for c in keep if c in opp.columns]
    opp[keep].to_parquet(STORES / "sb_table.parquet", index=False)
    print(f"sb_table: {len(opp):,} opportunities, "
          f"{opp.attempt.sum():,} attempts "
          f"({opp.attempt.mean():.3%}), success rate "
          f"{opp.loc[opp.attempt == 1, 'success'].mean():.1%}, "
          f"era scale {scale}", flush=True)


# ------------------------------------------------- sim adjuster stores
#
# Per-player mechanisms the pattern-bank/pre-event layers consume at sim
# time (identity-aware advancement, GIDP, stretch splits, catcher WP).
# All stores are keyed by CONSUMPTION season: season Y rows are built
# from data strictly before Y, so serve lookups and the tilt fits are
# leakage-free by construction.

DP_K = 150.0             # r1-on GB opportunities of shrinkage for GIDP
CATCHER_WP_K = 1500.0    # runner-on PAs of shrinkage for catcher WP/PB
ADV_SITS = {             # extra-base situations: (label, StartBase) ->
    ("1B", "2B"): ("H",),          # station 3B; extra = scored
    ("1B", "1B"): ("3B", "H"),     # station 2B; extra = 3B or scored
    ("2B", "1B"): ("H",),          # station 3B; extra = scored
}


def _season_z(df, col, by, w=None, clip=2.5):
    """Within-group z-score (population weighted if w given)."""
    g = df.groupby(by)[col]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    return ((df[col] - mu) / sd).clip(-clip, clip).fillna(0.0)


def _pbp_terminal_outs():
    """Per-PA out count from the terminal play's movements (mid-PA
    CS/PK outs excluded — the same universe the pattern bank uses)."""
    m = pd.read_csv(DATA / "mlb_pbp.csv", encoding="utf-8-sig",
                    usecols=["GamePk", "AtBatIndex", "PlayIndex",
                             "IsOut"], low_memory=False)
    m["game_pk"] = _num(m["GamePk"])
    m["at_bat_number"] = _num(m["AtBatIndex"]) + 1
    m["PlayIndex"] = _num(m["PlayIndex"]).fillna(-1)
    last = m.groupby(["game_pk", "at_bat_number"])["PlayIndex"]
    m = m[m["PlayIndex"] == last.transform("max")]
    return (m.groupby(["game_pk", "at_bat_number"])["IsOut"].sum()
            .rename("nout").reset_index())


def build_runner_z(pa):
    """runner_z.parquet: per (Season, PlayerId) run_z — standardized
    blend of sprint speed and Savant extra-base runner value — and dp_z,
    the batter's EB-shrunk GIDP propensity given a GB with a runner on
    first and <2 outs. Consumption-season keyed (prior data only)."""
    spr = read_csv("mlb_sprint_speed.csv",
                   usecols=["Year", "PlayerId", "SprintSpeed"])
    spr["PlayerId"] = _num(spr["PlayerId"])
    spr["SprintSpeed"] = _num(spr["SprintSpeed"])
    spr = spr.dropna()
    spr["z_spr"] = _season_z(spr, "SprintSpeed", "Year")
    brr = read_csv("mlb_baserunning.csv",
                   usecols=["Year", "PlayerId", "RunnerRunsXB",
                            "Opportunities"])
    brr["PlayerId"] = _num(brr["PlayerId"])
    o = _num(brr["Opportunities"]).clip(lower=1)
    brr["xb_rate"] = _num(brr["RunnerRunsXB"]) / o
    brr = brr.dropna(subset=["PlayerId", "xb_rate"])
    brr.loc[o < 20, "xb_rate"] = 0.0     # tiny samples -> league
    brr["z_brr"] = _season_z(brr, "xb_rate", "Year")
    rz = spr[["Year", "PlayerId", "z_spr"]].merge(
        brr[["Year", "PlayerId", "z_brr"]],
        on=["Year", "PlayerId"], how="outer")
    rz["run_z"] = rz[["z_spr", "z_brr"]].mean(axis=1).clip(-2.5, 2.5)
    rz["Season"] = rz["Year"] + 1

    nout = _pbp_terminal_outs()
    opp = pa[((pa.state0.values & 1) > 0) & (pa.outs0 < 2)][
        ["game_pk", "at_bat_number", "Season", "BatterId", "label",
         "bb_type"]].merge(nout, on=["game_pk", "at_bat_number"],
                           how="left")
    opp["dp"] = ((opp.label == "IPO") & (opp.bb_type == "ground_ball")
                 & (opp.nout >= 2)).astype(int)
    per = (opp.groupby(["Season", "BatterId"])
           .agg(n=("dp", "size"), dp=("dp", "sum")).reset_index()
           .sort_values(["BatterId", "Season"]))
    g = per.groupby("BatterId")
    per["n_prior"] = g["n"].cumsum() - per["n"]
    per["dp_prior"] = g["dp"].cumsum() - per["dp"]
    lg = float(opp.dp.mean())
    per["dp_rate"] = ((per["dp_prior"] + DP_K * lg)
                      / (per["n_prior"] + DP_K))
    sd = per.loc[per.n_prior >= 50, "dp_rate"].std()
    per["dp_z"] = ((per["dp_rate"] - lg) / max(sd, 1e-6)).clip(-2.5, 2.5)
    out = rz[["Season", "PlayerId", "run_z"]].merge(
        per.rename(columns={"BatterId": "PlayerId"})[
            ["Season", "PlayerId", "dp_z"]],
        on=["Season", "PlayerId"], how="outer")
    out["run_z"] = out["run_z"].fillna(0.0)
    out["dp_z"] = out["dp_z"].fillna(0.0)
    out.to_parquet(STORES / "runner_z.parquet", index=False)
    print(f"runner_z: {len(out):,} player-seasons "
          f"(league GB-DP rate {lg:.3%})", flush=True)
    return out, opp


def build_arm_of(pa):
    """arm_of.parquet: per (Season, PlayerId) standardized OF arm
    strength, plus a per (Season, Team) fielding aggregate for the tilt
    fit and serve fallback. Arm data exists 2020+; earlier seasons z=0."""
    arm = read_csv("mlb_arm_strength.csv",
                   usecols=["Year", "PlayerId", "ArmOf"])
    arm["PlayerId"] = _num(arm["PlayerId"])
    arm["ArmOf"] = _num(arm["ArmOf"])
    arm = arm.dropna()
    arm["arm_z"] = _season_z(arm, "ArmOf", "Year")
    gbb = read_csv("mlb_game_batting.csv",
                   usecols=["Season", "PlayerId", "Team"])
    gbb["PlayerId"] = _num(gbb["PlayerId"])
    team = (gbb.groupby(["Season", "PlayerId"])["Team"]
            .agg(lambda s: s.mode().iat[0]).reset_index())
    at = arm.merge(team, left_on=["Year", "PlayerId"],
                   right_on=["Season", "PlayerId"], how="left")
    tz = (at.dropna(subset=["Team"])
          .groupby(["Year", "Team"])["arm_z"].mean()
          .rename("team_arm_z").reset_index())
    ply = arm[["Year", "PlayerId", "arm_z"]].copy()
    ply["Season"] = ply["Year"] + 1
    tz["Season"] = tz["Year"] + 1
    ply[["Season", "PlayerId", "arm_z"]].to_parquet(
        STORES / "arm_of.parquet", index=False)
    tz[["Season", "Team", "team_arm_z"]].to_parquet(
        STORES / "arm_of_team.parquet", index=False)
    print(f"arm_of: {len(ply):,} player-seasons, "
          f"{len(tz):,} team-seasons", flush=True)
    return tz


def build_advance_tilt(pa, rz, dp_opp, arm_team):
    """advance_tilt.json: logistic tilt coefficients the pattern bank
    applies at sim time — theta_adv (lead-runner speed/value per extra
    base), theta_arm (fielding OF arm), theta_dp (batter GIDP propensity
    on DP patterns). Fit on runner-level extra-base outcomes with
    situation/outs controls; z's are prior-season (as-of-honest)."""
    from sklearn.linear_model import LogisticRegression
    mv = pd.read_csv(DATA / "mlb_pbp.csv", encoding="utf-8-sig",
                     usecols=["GamePk", "AtBatIndex", "PlayIndex",
                              "RunnerId", "StartBase", "EndBase",
                              "IsOut"], low_memory=False)
    mv["game_pk"] = _num(mv["GamePk"])
    mv["at_bat_number"] = _num(mv["AtBatIndex"]) + 1
    mv["PlayIndex"] = _num(mv["PlayIndex"]).fillna(-1)
    last = mv.groupby(["game_pk", "at_bat_number"])["PlayIndex"]
    mv = mv[mv["PlayIndex"] == last.transform("max")]
    # collapse multi-SEGMENT movements to origin -> final disposition
    # (a scoring runner is 2B->3B then 3B->H as separate rows; without
    # this the first segment masks the extra base — same lesson as
    # build_pattern_table)
    mv = mv.reset_index(drop=True).reset_index(names="_row")
    grp = mv.groupby(["game_pk", "at_bat_number", "RunnerId"],
                     sort=False)
    mv = mv.assign(_first=grp["_row"].transform("min"),
                   _last=grp["_row"].transform("max"),
                   _out=grp["IsOut"].transform("max"))
    start = mv.loc[mv["_row"] == mv["_first"],
                   ["game_pk", "at_bat_number", "RunnerId", "StartBase"]]
    final = mv.loc[mv["_row"] == mv["_last"],
                   ["game_pk", "at_bat_number", "RunnerId", "EndBase",
                    "_out"]]
    mv = start.merge(final, on=["game_pk", "at_bat_number", "RunnerId"])
    mv = mv.rename(columns={"_out": "IsOut"})
    core = pa[["game_pk", "at_bat_number", "Season", "label",
               "fld_team", "outs0"]]
    mv = mv.merge(core, on=["game_pk", "at_bat_number"], how="inner")

    frames = []
    for i, ((lab, sb), extra_bases) in enumerate(ADV_SITS.items()):
        s = mv[(mv.label == lab) & (mv.StartBase == sb)].copy()
        s["extra"] = (s.EndBase.isin(extra_bases)
                      & (s.IsOut != 1)).astype(int)
        s["sit"] = i
        frames.append(s)
    ds = pd.concat(frames, ignore_index=True)
    ds = ds.merge(rz[["Season", "PlayerId", "run_z"]],
                  left_on=["Season", "RunnerId"],
                  right_on=["Season", "PlayerId"], how="left")
    ds = ds.merge(arm_team.rename(columns={"Team": "fld_team"}),
                  on=["Season", "fld_team"], how="left")
    ds["run_z"] = ds["run_z"].fillna(0.0)
    ds["team_arm_z"] = ds["team_arm_z"].fillna(0.0)
    X = np.column_stack([
        ds["run_z"].values, ds["team_arm_z"].values,
        (ds["outs0"] == 1).astype(float).values,
        (ds["outs0"] == 2).astype(float).values,
        (ds["sit"] == 1).astype(float).values,
        (ds["sit"] == 2).astype(float).values])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, ds["extra"])
    theta_adv = float(lr.coef_[0][0])
    theta_arm = float(lr.coef_[0][1])

    d = dp_opp.merge(rz[["Season", "PlayerId", "dp_z"]],
                     left_on=["Season", "BatterId"],
                     right_on=["Season", "PlayerId"], how="left")
    d = d[(d.label == "IPO") & (d.bb_type == "ground_ball")]
    d["dp_z"] = d["dp_z"].fillna(0.0)
    lr2 = LogisticRegression(C=1e6, max_iter=1000).fit(
        d[["dp_z"]].values, d["dp"])
    theta_dp = float(lr2.coef_[0][0])

    tilt = dict(theta_adv=round(theta_adv, 4),
                theta_arm=round(theta_arm, 4),
                theta_dp=round(theta_dp, 4),
                n_adv=int(len(ds)), n_dp=int(len(d)),
                extra_rate=round(float(ds.extra.mean()), 4),
                dp_given_gb=round(float(d.dp.mean()), 4))
    (STORES / "advance_tilt.json").write_text(json.dumps(tilt, indent=1))
    print(f"advance_tilt: {tilt}", flush=True)


def build_stretch_table(pa):
    """stretch.json + stretch_z.parquet: base-state K/BB conditioning.
    League log-offsets for K and BB with runners on vs empty (A1 has no
    state input — the sim applies these and renormalizes), plus a
    per-pitcher slope on the standardized stretch velo delta (fbstr
    split), prior-season keyed."""
    on = (pa.state0.values > 0)
    k, bb = (pa.label == "K").values, (pa.label == "BB").values
    pk_all, pb_all = float(k.mean()), float(bb.mean())
    off = ~on
    stretch = {
        "w_on": round(float(on.mean()), 4),
        "dk_on": round(float(np.log(k[on].mean() / pk_all)), 4),
        "dk_off": round(float(np.log(k[off].mean() / pk_all)), 4),
        "db_on": round(float(np.log(bb[on].mean() / pb_all)), 4),
        "db_off": round(float(np.log(bb[off].mean() / pb_all)), 4),
    }

    d = read_csv("mlb_pitch_daily_pitchers.csv",
                 usecols=["PlayerId", "Date", "fb_n", "fb_v", "fbstr_n",
                          "fbstr_v"])
    d["Season"] = pd.to_datetime(d["Date"]).dt.year
    for c in ("fb_n", "fb_v", "fbstr_n", "fbstr_v"):
        d[c] = _num(d[c]).fillna(0.0)
    ps = d.groupby(["Season", "PlayerId"])[
        ["fb_n", "fb_v", "fbstr_n", "fbstr_v"]].sum().reset_index()
    wn_n = ps["fb_n"] - ps["fbstr_n"]
    wn_v = ps["fb_v"] - ps["fbstr_v"]
    ok = (ps["fbstr_n"] >= STRETCH_MIN_N) & (wn_n >= STRETCH_MIN_N)
    ps["delta"] = np.where(
        ok, ps["fbstr_v"] / ps["fbstr_n"].clip(lower=1)
        - wn_v / wn_n.clip(lower=1), np.nan)
    ps = ps.dropna(subset=["delta"])
    ps["stretch_z"] = _season_z(ps, "delta", "Season")
    sz = ps[["Season", "PlayerId", "stretch_z"]].copy()
    sz["Season"] = sz["Season"] + 1

    # per-pitcher K logit split (runners on vs empty) vs career stretch z
    sub = pa[["PitcherId", "label"]].assign(on=on)
    agg = (sub.assign(k=(sub.label == "K").astype(int))
           .groupby(["PitcherId", "on"])
           .agg(n=("k", "size"), k=("k", "sum")).reset_index()
           .pivot(index="PitcherId", columns="on", values=["n", "k"]))
    agg.columns = ["n_off", "n_on", "k_off", "k_on"]
    agg = agg[(agg.n_on >= 400) & (agg.n_off >= 400)].reset_index()
    p_on = (agg.k_on / agg.n_on).clip(1e-3, 1 - 1e-3)
    p_off = (agg.k_off / agg.n_off).clip(1e-3, 1 - 1e-3)
    agg["dlogit"] = (np.log(p_on / (1 - p_on))
                     - np.log(p_off / (1 - p_off)))
    cz = ps.groupby("PlayerId")["stretch_z"].mean().rename("z")
    agg = agg.merge(cz, left_on="PitcherId", right_index=True,
                    how="inner")
    zc = agg["z"] - agg["z"].mean()
    dc = agg["dlogit"] - agg["dlogit"].mean()
    b1k = float((zc * dc).sum() / max((zc ** 2).sum(), 1e-9))
    stretch["b1k"] = round(b1k, 4)
    stretch["n_pitchers"] = int(len(agg))
    (STORES / "stretch.json").write_text(json.dumps(stretch, indent=1))
    sz.to_parquet(STORES / "stretch_z.parquet", index=False)
    print(f"stretch: {stretch}; z store {len(sz):,} pitcher-seasons",
          flush=True)


def build_catcher_wp(pa):
    """catcher_wp.parquet: per (Season, CatcherId) EB-shrunk WP+PB rate
    per runner-on PA (prior seasons only), replacing the league pre_wp
    scalar at serve. Pickoffs stay league-level (pitcher-driven)."""
    mid = pd.read_csv(DATA / "mlb_pbp.csv", encoding="utf-8-sig",
                      usecols=["GamePk", "AtBatIndex", "EventType"],
                      low_memory=False)
    mid["game_pk"] = _num(mid["GamePk"])
    mid["at_bat_number"] = _num(mid["AtBatIndex"]) + 1
    wp = (mid[mid.EventType.isin(["wild_pitch", "passed_ball"])]
          .drop_duplicates(["game_pk", "at_bat_number"]))
    wp = wp.assign(wp=1)[["game_pk", "at_bat_number", "wp"]]
    on = pa[pa.state > 0][["game_pk", "at_bat_number", "Season",
                           "CatcherId"]].merge(
        wp, on=["game_pk", "at_bat_number"], how="left")
    on["wp"] = on["wp"].fillna(0).astype(int)
    lg = float(on.wp.mean())
    per = (on.groupby(["Season", "CatcherId"])
           .agg(n=("wp", "size"), wp=("wp", "sum")).reset_index()
           .sort_values(["CatcherId", "Season"]))
    g = per.groupby("CatcherId")
    per["n_prior"] = g["n"].cumsum() - per["n"]
    per["wp_prior"] = g["wp"].cumsum() - per["wp"]
    per["wp_rate"] = ((per["wp_prior"] + CATCHER_WP_K * lg)
                      / (per["n_prior"] + CATCHER_WP_K))
    out = per.rename(columns={"CatcherId": "PlayerId"})[
        ["Season", "PlayerId", "wp_rate"]]
    out.to_parquet(STORES / "catcher_wp.parquet", index=False)
    print(f"catcher_wp: {len(out):,} catcher-seasons "
          f"(league {lg:.3%}/runner-on PA, rate spread "
          f"{out.wp_rate.min():.3%}..{out.wp_rate.max():.3%})",
          flush=True)


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
    # start-level regime context: previous start's date/pitch-count/
    # length (ramp-up and opener/quick-hook regimes), and IL-return
    # recency — the variance-wideners the leash averages can't see
    gpf = read_csv("mlb_game_pitching.csv",
                   usecols=["GamePk", "PlayerId", "GS", "Date", "NP",
                            "IP"])
    st = gpf[_num(gpf["GS"]) == 1].copy()
    st["Date"] = pd.to_datetime(st["Date"])
    ipf = _num(st["IP"]).fillna(0)
    st["outs_g"] = (ipf.astype(int) * 3
                    + round((ipf % 1) * 10)).astype(float)
    st["np_g"] = _num(st["NP"])
    st = st.sort_values(["PlayerId", "Date"])
    g2 = st.groupby("PlayerId")
    st["gap_days"] = (st["Date"] - g2["Date"].shift(1)).dt.days
    st["ramp"] = (g2["np_g"].shift(1) < RAMP_NP).astype(float)
    st["prev_short"] = (g2["outs_g"].shift(1)
                        <= SHORT_START_OUTS).astype(float)
    reg = st[["GamePk", "PlayerId", "Date", "gap_days", "ramp",
              "prev_short"]].copy()
    reg["GamePk"] = _num(reg["GamePk"])
    il = pd.read_parquet(STORES / "il_stints.parquet")
    ilr = reg.sort_values("Date").rename(columns={"Date": "d_"})
    ilj = pd.merge_asof(
        ilr, il[["PlayerId", "Date", "act_date"]].sort_values("Date"),
        left_on="d_", right_on="Date", by="PlayerId",
        direction="backward")
    ilj["il_ret30"] = ((ilj["d_"] - ilj["act_date"]).dt.days
                       <= 30).fillna(False).astype(float)

    # competing-risks CAUSE label for the removal event (the collapsed
    # all-cause hazard stays the serving model; the label separates
    # managerial hooks from exits the manager didn't choose):
    #   injury  IL placement within 3 days after the start
    #   ph      a pinch hitter appeared in the pitcher's batting slot
    #           (pitchers-bat games; upper bound — the PH may have hit
    #           for a later reliever)
    #   hook    everything else (incl. undetectable rain/suspensions)
    ilp = read_csv("mlb_il.csv", usecols=["PlayerId", "PlaceDate"])
    ilp["PlayerId"] = _num(ilp["PlayerId"])
    ilp["PlaceDate"] = pd.to_datetime(ilp["PlaceDate"], errors="coerce")
    ilp = ilp.dropna().sort_values("PlaceDate")
    inj = pd.merge_asof(
        ilj[["GamePk", "PlayerId", "d_"]].sort_values("d_"),
        ilp, left_on="d_", right_on="PlaceDate", by="PlayerId",
        direction="forward")
    inj["injury"] = ((inj["PlaceDate"] - inj["d_"]).dt.days
                     <= 3).fillna(False)
    ilj = ilj.merge(inj[["GamePk", "PlayerId", "injury"]],
                    on=["GamePk", "PlayerId"], how="left")

    gbb = read_csv("mlb_game_batting.csv",
                   usecols=["GamePk", "PlayerId", "Team",
                            "BattingOrder"])
    gbb["GamePk"] = _num(gbb["GamePk"])
    gbb["bo"] = _num(gbb["BattingOrder"])
    gbb = gbb.dropna(subset=["bo"])
    pslot = gbb[(gbb["bo"] % 100 == 0)].merge(
        ilj[["GamePk", "PlayerId"]], on=["GamePk", "PlayerId"],
        how="inner")
    pslot["slot"] = pslot["bo"] // 100
    subs = gbb[gbb["bo"] % 100 > 0].copy()
    subs["slot"] = subs["bo"] // 100
    subbed = subs[["GamePk", "Team", "slot"]].drop_duplicates()
    subbed["ph"] = True
    pslot = pslot.merge(subbed, on=["GamePk", "Team", "slot"],
                        how="left")
    ilj = ilj.merge(
        pslot[["GamePk", "PlayerId", "ph"]].drop_duplicates(),
        on=["GamePk", "PlayerId"], how="left")
    ilj["cause"] = np.select(
        [ilj["injury"].fillna(False).values,
         ilj["ph"].fillna(False).values],
        ["injury", "ph"], default="hook")

    reg = ilj[["GamePk", "PlayerId", "gap_days", "ramp", "prev_short",
               "il_ret30", "cause"]]
    reg.columns = ["game_pk", "PitcherId", "gap_days", "ramp",
                   "prev_short", "il_ret30", "cause"]
    sub = sub.merge(reg, on=["game_pk", "PitcherId"], how="left")
    sub["cause"] = sub["cause"].where(sub["removed"] == 1, "")

    # pen fatigue: a gassed pen stretches the starter's leash. Own-side
    # pen pitches over the last 3 days from the team context store
    # (as-of pre-game by construction; teamctx builds before hazard).
    sub["pen_np3"] = np.nan
    tc_path = STORES / "team_game_context.parquet"
    if tc_path.exists():
        tc = pd.read_parquet(tc_path, columns=[
            "GamePk", "away_pen_np3", "home_pen_np3"])
        tc["game_pk"] = _num(tc["GamePk"])
        gmap = read_csv("mlb_games.csv", usecols=["GamePk", "HomeTeam"])
        gmap["game_pk"] = _num(gmap["GamePk"])
        sub = sub.merge(tc[["game_pk", "away_pen_np3",
                            "home_pen_np3"]], on="game_pk", how="left")
        sub = sub.merge(gmap[["game_pk", "HomeTeam"]], on="game_pk",
                        how="left")
        sub["pen_np3"] = np.where(sub["fld_team"] == sub["HomeTeam"],
                                  sub["home_pen_np3"],
                                  sub["away_pen_np3"])

    keep = ["game_pk", "Date", "Season", "PitcherId", "at_bat_number",
            "bf", "cum_pitches", "tto", "inning", "outs", "score_diff",
            "k_so_far", "br_so_far", "runs_so_far", "rest_p", "removed",
            "gap_days", "ramp", "prev_short", "il_ret30", "cause",
            "pen_np3"]
    sub[keep].to_parquet(STORES / "hazard_table.parquet", index=False)
    mix = sub.loc[sub.removed == 1, "cause"].value_counts(normalize=True)
    print(f"hazard_table: {len(sub):,} starter-BF rows, removal rate "
          f"{sub.removed.mean():.3%}, cause mix "
          f"{ {k: round(float(v), 4) for k, v in mix.items()} }",
          flush=True)


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
                 "league_rates_stand", "milb_priors",
                 "panel_pit_consist", "il_stints", "bat_tracking",
                 "panel_bat_pull", "panel_bat_hrmix",
                 "arsenal_class_usage", "arsenal_meta",
                 "milb_priors_pit", "panel_bat_out_loc", "park_hand",
                 "panel_pit_out_tto", "panel_bvp", "panel_bat_sched",
                 "league_env_daily", "panel_bat_out_fast",
                 "panel_pit_out_fast", "park_geo"):
        s[name] = pd.read_parquet(STORES / f"{name}.parquet")
    s["eb_k"] = json.loads((STORES / "eb_k.json").read_text())
    s["hrmix_league"] = json.loads(
        (STORES / "hrmix_league.json").read_text())
    ros = read_csv("mlb_rosters.csv",
                   usecols=["PlayerId", "DOB", "Ht", "Wt"])
    ros["dob"] = pd.to_datetime(ros["DOB"], format="%m/%d/%Y",
                                errors="coerce")
    hw = ros["Ht"].astype(str).str.extract(r"(\d+)'\s*(\d+)")
    ros["ht_in"] = _num(hw[0]) * 12 + _num(hw[1])
    ros["wt"] = _num(ros["Wt"])
    s["dob"] = ros.dropna(subset=["dob"]).drop_duplicates("PlayerId")
    spr = read_csv("mlb_sprint_speed.csv",
                   usecols=["Year", "PlayerId", "SprintSpeed"])
    spr["Year"] = _num(spr["Year"]) + 1
    s["sprint"] = spr
    # team-strength context (Elo) — may not exist mid-first-build
    tc_p = STORES / "team_game_context.parquet"
    s["teamctx"] = (pd.read_parquet(
        tc_p, columns=["GamePk", "away_elo", "home_elo"])
        if tc_p.exists() else None)
    # player-grain OAA, prior-season consumption (actual fielders)
    oaa_p = DATA / "mlb_oaa_players.csv"
    if oaa_p.exists():
        poa = read_csv("mlb_oaa_players.csv",
                       usecols=["Year", "PlayerId", "OAA"])
        poa["Year"] = _num(poa["Year"]) + 1
        s["player_oaa"] = poa
    else:
        s["player_oaa"] = None
    return s


def assemble_features(rows, stores):
    """rows: DataFrame with at least [Date, Season, BatterId, PitcherId,
    stand, p_throws, tto, home_bat, fld_team, Venue, DayNight, Temp,
    WindSpeed, WindDir, Condition, Humidity, Pressure, HpUmpId, rest_p].
    Returns (X, feature_names) aligned to rows. Shared by training and
    serving — the one code path."""
    n = len(rows)
    rows = rows.reset_index(drop=True)
    # columns accumulate in a dict and the frame is built once at the
    # end: inserting hundreds of columns one-by-one fragments the block
    # manager (pandas PerformanceWarning) and copies it repeatedly
    out = {"Date": rows["Date"], "Season": rows["Season"]}

    # ---- decayed outcome panels: batter overall + vs pitcher hand
    m = merge_asof_panel(rows, stores["panel_bat_out"], ["BatterId"],
                         ["pa"] + CLASSES, "b_")
    out.update({c: m[c] for c in m.columns})
    rows_h = rows.rename(columns={"p_throws": "p_throws_key"})
    ph = stores["panel_bat_out_hand"].rename(
        columns={"p_throws": "p_throws_key"})
    m = merge_asof_panel(rows_h, ph, ["BatterId", "p_throws_key"],
                         ["pa"] + CLASSES, "bh_")
    out.update({c: m[c] for c in m.columns})
    m = merge_asof_panel(rows, stores["panel_pit_out"], ["PitcherId"],
                         ["pa"] + CLASSES, "p_")
    out.update({c: m[c] for c in m.columns})
    rows_s = rows.rename(columns={"stand": "stand_key"})
    ps = stores["panel_pit_out_hand"].rename(columns={"stand": "stand_key"})
    m = merge_asof_panel(rows_s, ps, ["PitcherId", "stand_key"],
                         ["pa"] + CLASSES, "ph_")
    out.update({c: m[c] for c in m.columns})

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
    milb_p = stores["milb_priors_pit"].sort_values(["PlayerId", "Year"])
    prow = rows[["PitcherId", "Season"]].copy()
    prow["Season"] = prow["Season"].astype("int64")
    prow["PitcherId"] = prow["PitcherId"].astype("int64")
    prow["_ord"] = np.arange(n)
    pr_ = milb_p.rename(columns={"PlayerId": "PitcherId",
                                 "Year": "Season"})
    pr_["Season"] = pr_["Season"].astype("int64")
    pr_["PitcherId"] = pr_["PitcherId"].astype("int64")
    pmm = pd.merge_asof(
        prow.sort_values("Season"), pr_.sort_values("Season"),
        on="Season", by="PitcherId", direction="backward")
    pmm = pmm.sort_values("_ord").reset_index(drop=True)
    pri_p = {c: lgv(lgs, [prior_seas, rows["stand"]], c) for c in CLASSES}
    out = _shrunk_rates(out, "p_", "p_pa", PIT_RATES, pri_p,
                        stores["eb_k"], milb=pmm)
    out = _shrunk_rates(out, "ph_", "ph_pa", PIT_RATES, pri_p,
                        stores["eb_k"], milb=pmm)
    # per-TTO pitcher rates: this row's times-through-order slice
    rows_t = rows[["PitcherId", "Date", "tto"]].copy()
    rows_t["tto_key"] = (_num(rows_t["tto"]).fillna(1).clip(1, 3)
                         .astype("int64"))
    pt = stores["panel_pit_out_tto"].rename(columns={"tto": "tto_key"})
    pt["tto_key"] = _num(pt["tto_key"]).astype("int64")
    m = merge_asof_panel(rows_t, pt, ["PitcherId", "tto_key"],
                         ["pa"] + CLASSES, "pt_")
    out.update({c: m[c] for c in m.columns})
    out = _shrunk_rates(out, "pt_", "pt_pa", PIT_RATES, pri_p,
                        stores["eb_k"])

    # ---- HBP rates (A1 predicts the class; give it more than the
    # league marginal — plunk-prone pitchers and crowd-the-plate
    # batters are real, stable axes)
    out = _shrunk_rates(out, "b_", "b_pa", ["HBP"], pri_b,
                        stores["eb_k"], milb=mm)
    out = _shrunk_rates(out, "p_", "p_pa", ["HBP"], pri_p,
                        stores["eb_k"], milb=pmm)

    # ---- outcome form trend: fast-decay (30d) shrunk rate minus the
    # 90d rate. Both arms shrink to the SAME prior, so thin samples and
    # rookies collapse to zero trend instead of a spurious one.
    m = merge_asof_panel(rows, stores["panel_bat_out_fast"],
                         ["BatterId"], ["pa"] + CLASSES, "btr_",
                         hl=TREND_HL)
    out.update({c: m[c] for c in m.columns})
    out = _shrunk_rates(out, "btr_", "btr_pa", ["K", "BB", "HR", "1B"],
                        pri_b, stores["eb_k"], milb=mm)
    for c in ("K", "BB", "HR", "1B"):
        out[f"b_tr_{c}"] = out[f"btr_{c}_rate"] - out[f"b_{c}_rate"]
    m = merge_asof_panel(rows, stores["panel_pit_out_fast"],
                         ["PitcherId"], ["pa"] + CLASSES, "ptr_",
                         hl=TREND_HL)
    out.update({c: m[c] for c in m.columns})
    out = _shrunk_rates(out, "ptr_", "ptr_pa", ["K", "BB", "HR", "1B"],
                        pri_p, stores["eb_k"], milb=pmm)
    for c in ("K", "BB", "HR", "1B"):
        out[f"p_tr_{c}"] = out[f"ptr_{c}_rate"] - out[f"p_{c}_rate"]

    # ---- batted-ball quality and pitch-level discipline
    for panel, ent, pref in (("panel_bat_bip", "BatterId", "bq_"),
                             ("panel_pit_bip", "PitcherId", "pq_")):
        cols = ["bip", "ev_n", "ev_sum", "hard", "barrel", "gb", "air",
                "hit", "xba_n", "xba_sum", "xw_n", "xw_sum", "hr"]
        m = merge_asof_panel(rows, stores[panel], [ent], cols, pref)
        out[pref + "ev"] = m[pref + "ev_sum"] / m[pref + "ev_n"].clip(
            lower=1e-9)
        out[pref + "hard"] = m[pref + "hard"] / m[pref + "bip"].clip(
            lower=1e-9)
        out[pref + "barrel"] = m[pref + "barrel"] / m[pref + "bip"].clip(
            lower=1e-9)
        out[pref + "gb"] = m[pref + "gb"] / m[pref + "bip"].clip(
            lower=1e-9)
        # realized hits minus expected on the same recent contact:
        # positive = running hot (mean-reversion signal)
        out[pref + "luck"] = ((m[pref + "hit"] - m[pref + "xba_sum"])
                              / m[pref + "bip"].clip(lower=1e-9))
        out[pref + "xw"] = (m[pref + "xw_sum"]
                            / m[pref + "xw_n"].clip(lower=1e-9))
        out[pref + "hr_bip"] = (m[pref + "hr"]
                                / m[pref + "bip"].clip(lower=1e-9))
        out[pref + "log_bip"] = np.log1p(m[pref + "bip"])
    bcols = ["n", "sw_n", "wh_n", "oz_n", "oz_sw", "oz_wh", "z_n",
             "cs_n", "fb95_n", "fb95_sw", "fb95_wh",
             "fbmid_sw", "fbmid_wh", "fblo_sw", "fblo_wh",
             "brk_n", "brk_sw", "brk_wh", "off_sw", "off_wh",
             "ts_sw", "ts_wh", "f32_n", "f32_b", "fp_n", "fp_sw",
             "con_n", "con_xw", "fb95_bip", "fb95_xw",
             "fbmid_bip", "fbmid_xw", "brk_bip", "brk_xw",
             "off_bip", "off_xw"]
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
    # wave 2: passivity, count leverage, banded whiff + damage axes
    out["bm_zsw"] = ((m["bm_sw_n"] - m["bm_oz_sw"])
                     / m["bm_z_n"].clip(lower=1e-9))
    out["bm_ts_wh"] = m["bm_ts_wh"] / m["bm_ts_sw"].clip(lower=1e-9)
    out["bm_f32_take"] = m["bm_f32_b"] / m["bm_f32_n"].clip(lower=1e-9)
    out["bm_fp_aggr"] = m["bm_fp_sw"] / m["bm_fp_n"].clip(lower=1e-9)
    out["bm_fbmid_whiff"] = (m["bm_fbmid_wh"]
                             / m["bm_fbmid_sw"].clip(lower=1e-9))
    out["bm_fblo_whiff"] = (m["bm_fblo_wh"]
                            / m["bm_fblo_sw"].clip(lower=1e-9))
    out["bm_off_whiff"] = m["bm_off_wh"] / m["bm_off_sw"].clip(lower=1e-9)
    fb_sw_b = m["bm_sw_n"] - m["bm_brk_sw"] - m["bm_off_sw"]
    fb_wh_b = m["bm_wh_n"] - m["bm_brk_wh"] - m["bm_off_wh"]
    out["bm_fb_whiff"] = fb_wh_b / fb_sw_b.clip(lower=1e-9)
    out["bm_con_xw"] = m["bm_con_xw"] / m["bm_con_n"].clip(lower=1e-9)
    out["bm_fb95_xw"] = m["bm_fb95_xw"] / m["bm_fb95_bip"].clip(
        lower=1e-9)
    out["bm_fbmid_xw"] = m["bm_fbmid_xw"] / m["bm_fbmid_bip"].clip(
        lower=1e-9)
    out["bm_brk_xw"] = m["bm_brk_xw"] / m["bm_brk_bip"].clip(lower=1e-9)
    out["bm_off_xw"] = m["bm_off_xw"] / m["bm_off_bip"].clip(lower=1e-9)
    pcols = ["n", "sw_n", "wh_n", "oz_n", "oz_sw", "oz_wh", "z_n",
             "cs_n", "edge_n", "fb_n", "fb_v", "fp_n", "fp_s",
             "fb95_n", "fbmid_n", "fblo_n", "brk_n", "brk_sw", "brk_wh",
             "off_n", "off_sw", "off_wh", "ts_sw", "ts_wh",
             "f32_n", "f32_z", "f32_b", "c02_n", "c02_w",
             "ah_n", "ah_brk", "ah_off", "bh_n", "bh_brk", "bh_off",
             "tr_n", "tr_same", "ivb_n", "ivb_sum", "fbe_n", "fbe_sum",
             "brkmov_n", "brkmov_sum", "fade_w", "fade_num",
             "con_n", "con_xw", "fb95_bip", "fb95_xw", "brk_bip",
             "brk_xw"]
    m = merge_asof_panel(rows.rename(columns={"PitcherId": "PlayerId"}),
                         stores["panel_pit_pitchmet"], ["PlayerId"],
                         pcols, "pm_")
    out["pm_whiff"] = m["pm_wh_n"] / m["pm_sw_n"].clip(lower=1e-9)
    out["pm_chase"] = m["pm_oz_sw"] / m["pm_oz_n"].clip(lower=1e-9)
    out["pm_zone"] = m["pm_z_n"] / m["pm_n"].clip(lower=1e-9)
    out["pm_edge"] = m["pm_edge_n"] / m["pm_n"].clip(lower=1e-9)
    out["pm_fstrike"] = m["pm_fp_s"] / m["pm_fp_n"].clip(lower=1e-9)
    out["pm_velo"] = m["pm_fb_v"] / m["pm_fb_n"].clip(lower=1e-9)
    # wave 2: stuff shape, count leverage, sequencing, usage mix
    out["pm_zwsw"] = ((m["pm_wh_n"] - m["pm_oz_wh"])
                      / (m["pm_sw_n"] - m["pm_oz_sw"]).clip(lower=1e-9))
    out["pm_putaway"] = m["pm_ts_wh"] / m["pm_ts_sw"].clip(lower=1e-9)
    out["pm_f32_zone"] = m["pm_f32_z"] / m["pm_f32_n"].clip(lower=1e-9)
    out["pm_f32_ball"] = m["pm_f32_b"] / m["pm_f32_n"].clip(lower=1e-9)
    out["pm_c02_waste"] = m["pm_c02_w"] / m["pm_c02_n"].clip(lower=1e-9)
    out["pm_ah_shift"] = ((m["pm_ah_brk"] + m["pm_ah_off"])
                          / m["pm_ah_n"].clip(lower=1e-9)
                          - (m["pm_bh_brk"] + m["pm_bh_off"])
                          / m["pm_bh_n"].clip(lower=1e-9))
    out["pm_tr_same"] = m["pm_tr_same"] / m["pm_tr_n"].clip(lower=1e-9)
    out["pm_ivb"] = m["pm_ivb_sum"] / m["pm_ivb_n"].clip(lower=1e-9)
    out["pm_ext"] = m["pm_fbe_sum"] / m["pm_fbe_n"].clip(lower=1e-9)
    out["pm_brkmov"] = (m["pm_brkmov_sum"]
                        / m["pm_brkmov_n"].clip(lower=1e-9))
    out["pm_fade"] = m["pm_fade_num"] / m["pm_fade_w"].clip(lower=1e-9)
    nn = m["pm_n"].clip(lower=1e-9)
    out["pm_fb95_u"] = m["pm_fb95_n"] / nn
    out["pm_fbmid_u"] = m["pm_fbmid_n"] / nn
    out["pm_fblo_u"] = m["pm_fblo_n"] / nn
    out["pm_brk_u"] = m["pm_brk_n"] / nn
    out["pm_off_u"] = m["pm_off_n"] / nn
    out["pm_brk_whiff"] = m["pm_brk_wh"] / m["pm_brk_sw"].clip(lower=1e-9)
    out["pm_off_whiff"] = m["pm_off_wh"] / m["pm_off_sw"].clip(lower=1e-9)
    out["pm_con_xw"] = m["pm_con_xw"] / m["pm_con_n"].clip(lower=1e-9)
    out["pm_fb95_xw"] = m["pm_fb95_xw"] / m["pm_fb95_bip"].clip(
        lower=1e-9)
    out["pm_brk_xw"] = m["pm_brk_xw"] / m["pm_brk_bip"].clip(lower=1e-9)
    mf = merge_asof_panel(rows.rename(columns={"PitcherId": "PlayerId"}),
                          stores["panel_pit_velo_fast"], ["PlayerId"],
                          ["fb_n_fast", "fb_v_fast"], "pv_",
                          hl=VELO_HL_FAST)
    velo_fast = mf["pv_fb_v_fast"] / mf["pv_fb_n_fast"].clip(lower=1e-9)
    out["pm_velo_trend"] = velo_fast - out["pm_velo"]
    # style-collision products: trees cannot multiply, so the strongest
    # log5-style matchups are materialized explicitly
    fb_u = (1.0 - out["pm_brk_u"] - out["pm_off_u"]).clip(lower=0.0)
    out["mx_band_whiff"] = (out["bm_fblo_whiff"] * out["pm_fblo_u"]
                            + out["bm_fbmid_whiff"] * out["pm_fbmid_u"]
                            + out["bm_fb95_whiff"] * out["pm_fb95_u"])
    out["mx_class_whiff"] = (out["bm_fb_whiff"] * fb_u
                             + out["bm_brk_whiff"] * out["pm_brk_u"]
                             + out["bm_off_whiff"] * out["pm_off_u"])
    out["mx_ts"] = out["pm_putaway"] * out["bm_ts_wh"]
    out["mx_ride"] = out["pm_ivb"] * out["bm_fb_whiff"]
    out["mx_class_dmg"] = (out["bm_fb95_xw"] * out["pm_fb95_u"]
                           + out["bm_brk_xw"] * out["pm_brk_u"]
                           + out["bm_off_xw"] * out["pm_off_u"])

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
    out["uf_R"] = j["uf_R"].fillna(1.0).values
    dt_ = stores["defense_team"]
    j = rows[["fld_team", "Season"]].merge(
        dt_, left_on=["fld_team", "Season"], right_on=["Team", "Year"],
        how="left")
    out["def_oaa"] = j["def_oaa"].fillna(0.0).values
    out["frame_rv"] = j["FrameRV_pt"].fillna(0.0).values
    out["def_uer"] = j["def_uer"].values

    dob = stores["dob"]
    j = rows[["BatterId", "Date"]].merge(
        dob, left_on="BatterId", right_on="PlayerId", how="left")
    out["b_age"] = ((j["Date"] - j["dob"]).dt.days / 365.25).values
    out["b_ht"] = j["ht_in"].values
    out["b_wt"] = j["wt"].values
    j = rows[["PitcherId", "Date"]].merge(
        dob, left_on="PitcherId", right_on="PlayerId", how="left")
    out["p_age"] = ((j["Date"] - j["dob"]).dt.days / 365.25).values
    out["p_ht"] = j["ht_in"].values
    out["p_wt"] = j["wt"].values

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

    # ---- batter-vs-pitcher direct history (heavily shrunk residuals
    # off the batter's own contact baseline; bvp_log_n = evidence mass)
    m = merge_asof_panel(rows, stores["panel_bvp"],
                         ["BatterId", "PitcherId"],
                         ["bip", "xw_n", "xw_sum", "hr"], "bvp_",
                         hl=HRPT_HL)
    bvp_n = m["bvp_bip"].fillna(0.0)
    bvp_xw = m["bvp_xw_sum"] / m["bvp_xw_n"].clip(lower=1e-9)
    out["bvp_log_n"] = np.log1p(bvp_n)
    w_xw = bvp_n / (bvp_n + 30.0)
    out["bvp_xw_resid"] = (w_xw * (bvp_xw - out["bq_xw"])).fillna(0.0)
    w_hr = bvp_n / (bvp_n + 50.0)
    out["bvp_hr_resid"] = (w_hr * (m["bvp_hr"]
                                   / bvp_n.clip(lower=1e-9)
                                   - out["bq_hr_bip"])).fillna(0.0)

    # ---- classic log5 outcome collisions + remaining style products
    out["mx_k5"] = out["b_K_rate"] * out["p_K_rate"]
    out["mx_bb5"] = out["b_BB_rate"] * out["p_BB_rate"]
    out["mx_hr5"] = out["b_HR_rate"] * out["p_HR_rate"]
    b_hit = (out["b_1B_rate"] + out["b_2B_rate"] + out["b_3B_rate"]
             + out["b_HR_rate"])
    p_hit = (out["p_1B_rate"] + out["p_2B_rate"] + out["p_3B_rate"]
             + out["p_HR_rate"])
    out["mx_hit5"] = b_hit * p_hit
    out["mx_chase"] = out["bm_chase"] * out["pm_chase"]
    out["mx_zone_con"] = out["pm_zone"] * (1.0 - out["bm_zcon"])
    out["mx_calledk"] = (1.0 - out["bm_zsw"]) * out["pm_edge"]

    # ---- IL recency (batter and pitcher)
    il = stores["il_stints"]
    for idc, pref in (("BatterId", "b_il_"), ("PitcherId", "p_il_")):
        left = rows[[idc, "Date"]].copy()
        left["_ord"] = np.arange(n)
        left["_id"] = _num(left[idc]).fillna(-1).astype("int64")
        r = il.copy()
        r["_id"] = _num(r["PlayerId"]).astype("int64")
        m = pd.merge_asof(
            left.sort_values("Date"),
            r[["_id", "Date", "act_date", "act_season", "last_len",
               "rehab", "szn_days"]].sort_values("Date"),
            on="Date", by="_id", direction="backward")
        m = m.sort_values("_ord").reset_index(drop=True)
        ret = (m["Date"] - m["act_date"]).dt.days.astype(float)
        out[pref + "ret_days"] = np.clip(ret, 0, 400)
        out[pref + "ret21"] = (ret <= 21).astype(float)
        out[pref + "last_len"] = m["last_len"]
        same_season = m["act_season"] == m["Date"].dt.year
        out[pref + "szn_days"] = np.where(same_season,
                                          m["szn_days"], 0.0)
        out[pref + "rehab"] = m["rehab"].fillna(0.0)

    # ---- pitcher mechanical consistency (velo SD, release scatter,
    # windup-vs-stretch velo delta, fastball-vs-breaking release sep)
    ccols = ["fb_n", "fb_v", "fb_v2", "rp_n", "rp_x", "rp_x2", "rp_z",
             "rp_z2", "fbstr_n", "fbstr_v", "rpf_n", "rpf_x", "rpf_z",
             "rpb_n", "rpb_x", "rpb_z"]
    m = merge_asof_panel(rows.rename(columns={"PitcherId": "PlayerId"}),
                         stores["panel_pit_consist"], ["PlayerId"],
                         ccols, "cz_")
    fbn = m["cz_fb_n"]
    mu_v = m["cz_fb_v"] / fbn.clip(lower=1e-9)
    var_v = m["cz_fb_v2"] / fbn.clip(lower=1e-9) - mu_v ** 2
    out["pm_velo_sd"] = np.where(fbn >= CONSIST_MIN_FB,
                                 np.sqrt(np.clip(var_v, 0, None)),
                                 np.nan)
    rpn = m["cz_rp_n"]
    vx = (m["cz_rp_x2"] / rpn.clip(lower=1e-9)
          - (m["cz_rp_x"] / rpn.clip(lower=1e-9)) ** 2)
    vz = (m["cz_rp_z2"] / rpn.clip(lower=1e-9)
          - (m["cz_rp_z"] / rpn.clip(lower=1e-9)) ** 2)
    out["pm_rel_scatter"] = np.where(
        rpn >= CONSIST_MIN_RP,
        np.sqrt(np.clip(vx, 0, None) + np.clip(vz, 0, None)), np.nan)
    stn = m["cz_fbstr_n"]
    wun = fbn - stn
    v_st = m["cz_fbstr_v"] / stn.clip(lower=1e-9)
    v_wu = (m["cz_fb_v"] - m["cz_fbstr_v"]) / wun.clip(lower=1e-9)
    out["pm_stretch_delta"] = np.where(
        (stn >= STRETCH_MIN_N) & (wun >= STRETCH_MIN_N),
        v_wu - v_st, np.nan)
    fn_, bn_ = m["cz_rpf_n"], m["cz_rpb_n"]
    sep = np.hypot(
        m["cz_rpf_x"] / fn_.clip(lower=1e-9)
        - m["cz_rpb_x"] / bn_.clip(lower=1e-9),
        m["cz_rpf_z"] / fn_.clip(lower=1e-9)
        - m["cz_rpb_z"] / bn_.clip(lower=1e-9))
    out["pm_rel_sep"] = np.where(
        (fn_ >= RELSEP_MIN_N) & (bn_ >= RELSEP_MIN_N), sep, np.nan)

    # ---- arsenal dynamics (prior-season entropy + whiff trend)
    am = stores["arsenal_meta"]
    j = rows[["PitcherId", "Season"]].merge(
        am, left_on=["PitcherId", "Season"],
        right_on=["PlayerId", "Year"], how="left")
    out["pars_entropy"] = j["ars_entropy"].values
    out["pars_whiff_trend"] = j["ars_whiff_trend"].values

    # ---- bat tracking (prior season, 2024+ serve years)
    bt = stores["bat_tracking"]
    j = rows[["BatterId", "Season"]].merge(
        bt, left_on=["BatterId", "Season"],
        right_on=["PlayerId", "Year"], how="left")
    for c in ("bt_speed", "bt_swlen", "bt_hardsw", "bt_squp"):
        out[c] = j[c].values

    # ---- pull tendency + wind-carry-to-pull-side interaction
    m = merge_asof_panel(rows, stores["panel_bat_pull"], ["BatterId"],
                         ["air", "pull"], "pl_", hl=PULL_HL)
    LG_PULL = 0.30
    pull_share = ((m["pl_pull"].fillna(0.0) + PULL_K * LG_PULL)
                  / (m["pl_air"].fillna(0.0) + PULL_K))
    out["b_pull"] = pull_share
    wf = rows["WindDir"].map(_WIND_FIELD)
    ws_ = rows["WindDir"].map(_WIND_SIGN)
    pull_field = np.where(rows["stand"].astype(str) == "R", "L", "R")
    wgt = np.where(wf == pull_field, 1.0,
                   np.where(wf == "C", CARRY_CF_W,
                            np.where(wf.notna(), CARRY_OPP_W, 0.0)))
    out["wind_pull"] = np.where(
        dome, 0.0,
        _num(rows["WindSpeed"]).fillna(0.0) * ws_.fillna(0.0) * wgt
        * pull_share)

    # ---- HR pitch-class matchup: batter HR mix vs THIS arsenal
    m = merge_asof_panel(rows, stores["panel_bat_hrmix"], ["BatterId"],
                         ["hr_n", "hr_fast", "hr_brk", "hr_off",
                          "ev_n", "ev_sum", "dist_n", "dist_sum"],
                         "hm_", hl=HRPT_HL)
    lgmix = stores["hrmix_league"]
    # HR quality: how hard/far this batter's homers go (censored — only
    # best contact is in the log — hence the heavy shrink); the distance
    # arm feeds the porch-margin geometry fit below
    out["b_hrq_ev"] = ((m["hm_ev_sum"].fillna(0.0)
                        + HRQ_K * lgmix["ev_mean"])
                       / (m["hm_ev_n"].fillna(0.0) + HRQ_K))
    out["b_hrq_dist"] = ((m["hm_dist_sum"].fillna(0.0)
                          + HRQ_K * lgmix["dist_mean"])
                         / (m["hm_dist_n"].fillna(0.0) + HRQ_K))
    cu = stores["arsenal_class_usage"]
    j = rows[["PitcherId", "Season"]].merge(
        cu, left_on=["PitcherId", "Season"],
        right_on=["PlayerId", "Year"], how="left")
    hrn = m["hm_hr_n"].fillna(0.0)
    hrpt = np.zeros(n)
    for cls in ("fast", "brk", "off"):
        share = ((m[f"hm_hr_{cls}"].fillna(0.0) + HRPT_K * lgmix[cls])
                 / (hrn + HRPT_K))
        usage = j[f"u_{cls}"].fillna(float(cu[f"u_{cls}"].mean())).values
        hrpt += (share.values - lgmix[cls]) * usage
    out["hrpt_score"] = hrpt

    # ---- batter home/away split rates (venue-affinity axis)
    rows_l = rows[["BatterId", "Date", "home_bat"]].copy()
    rows_l["hb_key"] = _num(rows_l["home_bat"]).fillna(0).astype("int64")
    pl_ = stores["panel_bat_out_loc"].copy()
    pl_["hb_key"] = _num(pl_["home_bat"]).astype("int64")
    m = merge_asof_panel(rows_l, pl_, ["BatterId", "hb_key"],
                         ["pa"] + CLASSES, "bl_")
    out.update({c: m[c] for c in m.columns})
    out = _shrunk_rates(out, "bl_", "bl_pa", BAT_RATES, pri_b,
                        stores["eb_k"])

    # ---- park x handedness HR factor (porch asymmetry)
    phh = stores["park_hand"]
    j = rows[["Venue", "Season", "stand"]].merge(
        phh, left_on=["Venue", "Season", "stand"],
        right_on=["Venue", "Year", "stand"], how="left")
    out["pf_HR_hand"] = j["pf_HR_h"].fillna(1.0).values

    # ---- park geometry + carry physics (trees can't multiply, so the
    # porch/carry collisions are materialized as mx_ products)
    pg = stores["park_geo"]
    j = rows[["Venue"]].merge(pg, on="Venue", how="left")
    elev_kft = j["Elevation_ft"].values / 1000.0
    out["park_elev"] = elev_kft
    pull_fence = np.where(rows["stand"].astype(str) == "R",
                          j["LF"].values, j["RF"].values)
    out["pull_fence"] = pull_fence
    porch = PORCH_REF - pull_fence
    # batter's adjusted HR distance vs the fence he pulls toward
    out["mx_porch_margin"] = out["b_hrq_dist"] - pull_fence
    out["mx_wind_porch"] = np.asarray(out["wind_pull"]) * porch
    out["mx_carry_elev"] = (np.asarray(out["temp"]) - 70.0) * elev_kft
    out["mx_air_porch"] = ((AIR_REF - np.asarray(out["air_density"]))
                           * porch)

    # ---- framing collisions: a good receiver hurts passive hitters
    # and pays off edge-heavy pitchers
    out["mx_frame_take"] = (np.asarray(out["frame_rv"])
                            * np.asarray(out["bm_f32_take"]))
    out["mx_frame_edge"] = (np.asarray(out["frame_rv"])
                            * np.asarray(out["pm_edge"]))

    # ---- sprint speed + leg-hit collision (fast + ground-ball profile
    # = infield hits the 1B-class rate alone can't attribute)
    spr = stores["sprint"].drop_duplicates(["PlayerId", "Year"])
    j = rows[["BatterId", "Season"]].merge(
        spr, left_on=["BatterId", "Season"],
        right_on=["PlayerId", "Year"], how="left")
    out["b_sprint"] = j["SprintSpeed"].values
    out["mx_leg_hits"] = ((j["SprintSpeed"].values - LG_SPRINT)
                          * np.asarray(out["bq_gb"]))

    # ---- batter schedule fatigue (rest days, games last 7/14 days)
    sch = stores["panel_bat_sched"].sort_values("Date")
    base_dates = rows["Date"].reset_index(drop=True)
    bid64 = rows["BatterId"].astype("int64").values

    def _sched_at(offset_days):
        at = base_dates - pd.Timedelta(days=offset_days)
        l2 = pd.DataFrame({"BatterId": bid64, "_ord": np.arange(n),
                           "_at": at.values}).sort_values("_at")
        mm2 = pd.merge_asof(l2, sch, left_on="_at", right_on="Date",
                            by="BatterId", direction="backward",
                            allow_exact_matches=False)
        return mm2.sort_values("_ord").reset_index(drop=True)

    m0, m7, m14 = _sched_at(0), _sched_at(7), _sched_at(14)
    out["b_rest"] = ((base_dates - m0["last_d"]).dt.days
                     .clip(upper=30).values)
    cum0 = m0["cum_g"].fillna(0.0)
    out["b_g_l7d"] = (cum0 - m7["cum_g"].fillna(0.0)).values
    out["b_g_l14d"] = (cum0 - m14["cum_g"].fillna(0.0)).values

    # ---- trailing-30-day league environment (in-season drift)
    env = stores["league_env_daily"].sort_values("Date")

    def _env_at(offset_days):
        at = base_dates - pd.Timedelta(days=offset_days)
        l2 = pd.DataFrame({"_ord": np.arange(n),
                           "_at": at.values}).sort_values("_at")
        mm2 = pd.merge_asof(l2, env, left_on="_at", right_on="Date",
                            direction="backward",
                            allow_exact_matches=False)
        return mm2.sort_values("_ord").reset_index(drop=True)

    e0, e30 = _env_at(0), _env_at(30)
    dpa = (e0["pa_n_c"] - e30["pa_n_c"]).clip(lower=1.0)
    out["lg_k30"] = ((e0["k_c"] - e30["k_c"]) / dpa).values
    out["lg_bb30"] = ((e0["bb_c"] - e30["bb_c"]) / dpa).values
    out["lg_hr30"] = ((e0["hr_c"] - e30["hr_c"]) / dpa).values
    dg = (e0["g_c"] - e30["g_c"]).clip(lower=1.0)
    out["lg_rpg30"] = ((e0["runs_c"] - e30["runs_c"]) / dg).values
    thin = (dpa < 5000).values
    for c in ("lg_k30", "lg_bb30", "lg_hr30"):
        out[c] = np.where(thin, np.nan, out[c])
    out["lg_rpg30"] = np.where((dg < 50).values, np.nan, out["lg_rpg30"])

    # ---- team strength (Elo) — reduced-form, as-of pre-game. Serve
    # rows carry b_elo/p_elo (predict injects from slate_context);
    # training rows carry game_pk and join the team-context store.
    b_elo = p_elo = None
    if "b_elo" in rows.columns:
        b_elo = pd.to_numeric(rows["b_elo"], errors="coerce")
        p_elo = pd.to_numeric(rows["p_elo"], errors="coerce")
    elif "game_pk" in rows.columns and stores.get("teamctx") is not None:
        tc = stores["teamctx"]
        j = rows[["game_pk", "home_bat"]].merge(
            tc, left_on="game_pk", right_on="GamePk", how="left")
        hb = _num(j["home_bat"]).fillna(0).values == 1
        b_elo = pd.Series(np.where(hb, j["home_elo"], j["away_elo"]))
        p_elo = pd.Series(np.where(hb, j["away_elo"], j["home_elo"]))
    if b_elo is not None:
        out["b_team_elo"] = (b_elo - 1500.0).values
        out["p_team_elo"] = (p_elo - 1500.0).values
        out["elo_diff"] = (b_elo - p_elo).values
    else:
        out["b_team_elo"] = np.full(n, np.nan)
        out["p_team_elo"] = np.full(n, np.nan)
        out["elo_diff"] = np.full(n, np.nan)

    # ---- actual-fielder defense: player-level OAA of the seven
    # non-battery fielders (train: Savant fielder_3..9; serve: the
    # fielding team's lineup mapped by roster position), aggregated
    # IF/OF and collided with the batter's ground/air profile
    if ("fielder_3" in rows.columns
            and stores.get("player_oaa") is not None):
        poa = stores["player_oaa"]
        omap = {(int(y), int(p)): float(v) for y, p, v in
                zip(poa.Year, poa.PlayerId, poa.OAA) if v == v}
        seas = _num(rows["Season"]).fillna(0).astype(int).values
        arrs = []
        for i in range(3, 10):
            fid = _num(rows[f"fielder_{i}"]).fillna(-1).astype(int).values
            arrs.append(np.array(
                [omap.get((s_, f_), np.nan)
                 for s_, f_ in zip(seas, fid)], dtype=float))
        arrs = np.vstack(arrs)
        with np.errstate(all="ignore"):
            if_oaa = np.nanmean(arrs[:4], axis=0)
            of_oaa = np.nanmean(arrs[4:], axis=0)
        out["def_oaa_if"] = if_oaa
        out["def_oaa_of"] = of_oaa
        gbs = np.asarray(pd.to_numeric(pd.Series(out["bq_gb"]),
                                       errors="coerce"))
        out["mx_oaa_gb"] = (gbs - 0.44) * if_oaa
        out["mx_oaa_air"] = (0.44 - gbs) * of_oaa
    else:
        for c in ("def_oaa_if", "def_oaa_of", "mx_oaa_gb",
                  "mx_oaa_air"):
            out[c] = np.full(n, np.nan)

    drop = {f"{p}{cl}"
            for p in ("b_", "bh_", "bl_", "p_", "ph_", "pt_",
                      "btr_", "ptr_")
            for cl in CLASSES + ["pa", "HBP", "IPO"]} | {"Date", "Season"}
    # fast-panel helper columns: only the b_tr_/p_tr_ DELTAS survive
    drop |= {f"{p}{c}_rate" for p in ("btr_", "ptr_")
             for c in ("K", "BB", "HR", "1B")}
    drop |= {"btr_log_pa", "ptr_log_pa"}
    X = pd.DataFrame({c: v for c, v in out.items() if c not in drop})
    return X, list(X.columns)


# ------------------------------------------- team-game context (heads)
# Elo, recent form, rest/travel/schedule spots for the residual heads.
# NONE of this feeds the component models or the sim — the simulator
# generates game structure from components; these are exactly the
# reduced-form signals that belong in the residual heads, where the sim
# probability stays the anchor and count coherence is never at risk.
# build_team_context writes stores/team_game_context.parquet (one row
# per GamePk, away_/home_ prefixed, every value as-of strictly before
# first pitch); slate_context runs the same state machine for an
# upcoming slate at serve time.

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
    gp = read_csv("mlb_game_pitching.csv",
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
    gb = read_csv("mlb_game_batting.csv",
                  usecols=["GamePk", "Team", "AB", "H", "2B", "3B",
                           "HR", "BB", "HBP", "TB", "R"])
    gb["GamePk"] = pd.to_numeric(gb["GamePk"], errors="coerce")
    for c in ("AB", "H", "2B", "3B", "HR", "BB", "HBP", "TB", "R"):
        gb[c] = pd.to_numeric(gb[c], errors="coerce").fillna(0)
    bat = gb.groupby(["GamePk", "Team"])[["AB", "H", "HR", "BB", "HBP",
                                          "TB", "R"]].sum()
    return pen.to_dict("index"), st.to_dict(), bat.to_dict("index")


def _load_games():
    games = read_csv("mlb_games.csv")
    games["Date"] = pd.to_datetime(games["Date"])
    games["GamePk"] = pd.to_numeric(games["GamePk"], errors="coerce")
    for c in ("AwayScore", "HomeScore"):
        games[c] = pd.to_numeric(games[c], errors="coerce")
    return games.dropna(subset=["GamePk", "Date"]).sort_values(
        ["Date", "GamePk"]).reset_index(drop=True)


def _ctx_inputs():
    parks = read_csv("mlb_ballparks.csv")
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


def build_team_context():
    games = _load_games()
    coords, (pen_map, st_map, bat_map) = _ctx_inputs()
    rows, elo = _context_rows(games, coords, pen_map, st_map, bat_map)
    out = pd.DataFrame(rows)
    out.to_parquet(STORES / "team_game_context.parquet", index=False)
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
    coords, (pen_map, st_map, bat_map) = _ctx_inputs()
    rows, _ = _context_rows(allg, coords, pen_map, st_map, bat_map)
    byk = {r["GamePk"]: r for r in rows}
    return [byk[_SLATE_PK + i] for i in range(len(specs))]


# --------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true",
                    help="build every store")
    ap.add_argument("--only", default="",
                    help="comma list: pa,panels,context,effects,priors,"
                         "pbp,simadj,hazard,part,teamctx,forecast")
    args = ap.parse_args()
    if not (args.build or args.only):
        ap.error("nothing to do: pass --build or --only ...")
    steps = set(args.only.split(",")) if args.only else {
        "pa", "panels", "context", "effects", "priors", "pbp", "simadj",
        "hazard", "part", "teamctx", "forecast"}

    STORES.mkdir(parents=True, exist_ok=True)
    pa = None
    if "pa" in steps:
        pa = build_pa_table()
    if pa is None and steps - {"pa", "teamctx", "forecast"}:
        pa = pd.read_parquet(STORES / "pa_table.parquet")
        print(f"pa_table loaded: {len(pa):,} rows", flush=True)
    if "panels" in steps:
        build_outcome_panels(pa)
        build_bip_panels()
        build_pitchmetric_panels()
        build_leash_panel()
    if "context" in steps:
        build_consistency_panels()
        build_il_table()
        build_bat_tracking_table()
        build_pull_table()
        build_hrpt_tables()
        build_bat_sched_panel()
        build_league_env(pa)
    if "effects" in steps:
        build_park_factors(pa)
        build_park_geometry(pa)
        build_ump_factors(pa)
        build_defense_tables()
        build_arsenal_tables()
    if "priors" in steps:
        build_league_rates(pa)
        build_eb_k(pa)
        build_milb_priors()
        build_milb_priors_pit()
    if "pbp" in steps:
        build_pattern_table(pa)
        build_sb_table(pa)
    if "simadj" in steps:
        rz, dp_opp = build_runner_z(pa)
        arm_team = build_arm_of(pa)
        build_advance_tilt(pa, rz, dp_opp, arm_team)
        build_stretch_table(pa)
        build_catcher_wp(pa)
    if "teamctx" in steps:      # before hazard: pen_np3 feeds the leash
        build_team_context()
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
