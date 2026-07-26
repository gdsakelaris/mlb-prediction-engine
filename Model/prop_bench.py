"""Model-vs-market top-K benchmark, per prop, batter AND pitcher sheets.

Same-universe comparison per (date, column): universe = rows in the served
workbook with a two-sided sharp_fair price for the matching (Market, Line).
Both sides ranked within that universe; top-K graded with Tools/4 mappings.
DH-duplicated IDs skipped. Read-only.

Segments: "pre" = pure-model comparison (dates < 2026-07-23, OR any date
with a Predictions/pure/<stem>.csv sidecar — written by real serves since
2026-07-25, model_p read from the sidecar instead of the blended workbook)
vs "post" = blended workbook with no sidecar (excluded from verdicts).
"""
import sys, glob, re
from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model"))
import odds as O

spec = importlib.util.spec_from_file_location(
    "g4", ROOT / "Tools" / "4) Grade Results.py")
g4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g4)

BAT_MAP = {
    ("batter_hits", 0.5): "Hit",
    ("batter_hits", 1.5): "2+ Hits",
    ("batter_home_runs", 0.5): "HR",
    ("batter_total_bases", 1.5): "2+ TB",
    ("batter_total_bases", 2.5): "3+ TB",
    ("batter_total_bases", 3.5): "4+ TB",
    ("batter_rbis", 0.5): "RBI",
    ("batter_rbis", 1.5): "2+ RBI",
    ("batter_runs_scored", 0.5): "Run",
    ("batter_runs_scored", 1.5): "2+ Runs",
    ("batter_hits_runs_rbis", 1.5): "H+R+RBI 2+",
    ("batter_hits_runs_rbis", 2.5): "H+R+RBI 3+",
    ("batter_hits_runs_rbis", 3.5): "H+R+RBI 4+",
    ("batter_walks", 0.5): "BB",
    ("batter_singles", 0.5): "Single",
    ("batter_doubles", 0.5): "Double",
    ("batter_stolen_bases", 0.5): "SB",
}
PIT_API = {"K": "pitcher_strikeouts", "Outs": "pitcher_outs",
           "Hits": "pitcher_hits_allowed", "BB": "pitcher_walks",
           "ER": "pitcher_earned_runs"}
TOP_KS = (1, 3, 5, 10)
MIN_UNI = {"bat": 15, "pit": 8}
BLEND_START = "2026-07-23"

odds = pd.read_csv(ROOT / "Data" / "mlb_odds.csv")


def newest_book(date):
    cands = sorted(glob.glob(str(ROOT / "Predictions" / f"{date}*.xlsx")))
    pat = re.compile(rf"{date}(_\d+)?\.xlsx$")
    cands = [c for c in cands if pat.search(Path(c).name)]
    return cands[-1] if cands else None


def fair_map(sub, open_side=False):
    """pid -> two-sided fair over prob (sharp_fair)."""
    if open_side:
        sub = sub.rename(columns={"OverPrice": "_c1", "UnderPrice": "_c2",
                                  "OpenOverPrice": "OverPrice",
                                  "OpenUnderPrice": "UnderPrice"})
    out = {}
    for pid, grp in sub.groupby("PlayerId"):
        f = O.sharp_fair(grp.to_dict("records"))
        if f is not None:
            out[int(pid)] = f
    return out


# Dates whose captured prices are NOT pregame market truth and must
# never enter the model-vs-market evidence. 2026-07-12: the day's only
# capture ran 14:39 local against an all-day-game slate — the stored
# prices are in-play (53% of two-sided batter_hits Overs plus-money vs
# ~12% on neighboring dates; two-sided universe 693 -> 322), carrying
# partial-outcome information the pregame market never offered.
# Tools/2 now skips commenced games at capture time, so the class
# cannot recur; this list quarantines the legacy damage.
IN_PLAY_DATES = {"2026-07-12"}

dates = sorted((set(odds["Date"]) - IN_PLAY_DATES) & {
    p.name[:10] for p in (ROOT / "Predictions").glob("2026-*.xlsx")})
print("dates:", dates, f"(quarantined in-play: {sorted(IN_PLAY_DATES)})")

rows_out = []
for date in dates:
    book = newest_book(date)
    if not book:
        continue
    try:
        batters, starters, games, bg, sg = g4.load_actuals(date)
    except Exception as e:
        print(f"  ! {date}: {e}")
        continue
    if not games:
        print(f"  - {date}: no finals, skipped")
        continue
    day_odds = odds[odds["Date"] == date]

    # pure-model sidecar (written by real serves since 2026-07-25):
    # {(kind, pid, col): pure p}; overrides the blended workbook cells
    side = ROOT / "Predictions" / "pure" / f"{Path(book).stem}.csv"
    pure = None
    if side.exists():
        sc = pd.read_csv(side)
        sc = sc[sc["sheet"].isin(["bat", "pit"])]
        sc["ID"] = pd.to_numeric(sc["ID"], errors="coerce")
        sc = sc.dropna(subset=["ID"])
        sc = sc[~sc.duplicated(["sheet", "ID", "col"], keep=False)]  # DH
        pure = {(r.sheet, int(r.ID), r.col): r.p
                for r in sc.itertuples()}

    sheets = {}
    bat = pd.read_excel(book, sheet_name="Batter Props")
    pit = pd.read_excel(book, sheet_name="Pitching Props")
    for tag, df_ in (("bat", bat), ("pit", pit)):
        df_ = df_[pd.to_numeric(df_["ID"], errors="coerce") > 0].copy()
        df_["ID"] = df_["ID"].astype(int)
        df_ = df_[~df_["ID"].duplicated(keep=False)]      # skip DH pids
        sheets[tag] = df_

    jobs = []   # (kind, col, api_mkt, line, event_fn)
    for (mkt, line), col in BAT_MAP.items():
        if col in sheets["bat"].columns:
            ev = g4.BAT_EVENTS[col]
            jobs.append(("bat", col, mkt, line,
                         lambda s, ev=ev: bool(ev(s))))
    for col in sheets["pit"].columns:
        m = g4.LINE_RE.match(str(col))
        if m:
            stat, line = g4.LINE_STAT[m.group(1)], float(m.group(2))
            jobs.append(("pit", col, PIT_API[m.group(1)], line,
                         lambda s, st=stat, ln=line: float(s[st]) > ln))

    for kind, col, mkt, line, event in jobs:
        sub = day_odds[(day_odds["Market"] == mkt) &
                       (np.isclose(day_odds["Line"].fillna(-99), line))]
        if sub.empty:
            continue
        fair_c = fair_map(sub)
        fair_o = fair_map(sub, open_side=True)
        uni = sheets[kind][sheets[kind]["ID"].isin(fair_c)].copy()
        if len(uni) < MIN_UNI[kind]:
            continue
        uni["mkt_p"] = uni["ID"].map(fair_c)
        uni["open_p"] = uni["ID"].map(fair_o)
        if pure is not None:
            uni["model_p"] = uni["ID"].map(
                lambda pid: pure.get((kind, int(pid), col)))
        else:
            uni["model_p"] = pd.to_numeric(uni[col], errors="coerce")
        uni = uni.dropna(subset=["model_p"])
        if len(uni) < MIN_UNI[kind]:
            continue
        day_stats = batters if kind == "bat" else starters

        def hit(pid):
            s = day_stats.get(int(pid))
            return event(s) if s is not None else False

        is_pure = date < BLEND_START or pure is not None
        rec = {"date": date, "kind": kind, "col": col, "n_uni": len(uni),
               "seg": "pre" if is_pure else "post"}
        for side, key in (("model", "model_p"), ("mkt", "mkt_p"),
                          ("open", "open_p")):
            u2 = uni.dropna(subset=[key])
            if len(u2) < MIN_UNI[kind]:
                for k in TOP_KS:
                    rec[f"{side}_t{k}"] = np.nan
                continue
            ranked = u2.sort_values(key, ascending=False)["ID"].tolist()
            for k in TOP_KS:
                rec[f"{side}_t{k}"] = np.mean([hit(p) for p in ranked[:k]])
        rows_out.append(rec)

df = pd.DataFrame(rows_out)
df.to_csv(ROOT / "Model" / "artifacts" / "prop_bench_rows.csv",
          index=False)
pd.set_option("display.width", 250)

for seg in ("pre", "post"):
    d = df[df.seg == seg]
    if d.empty:
        continue
    label = ("PURE MODEL (pre-blend dates + sidecar dates)" if seg == "pre"
             else "BLENDED, NO SIDECAR (excluded from verdicts)")
    print(f"\n================ {label} — {d.date.nunique()} slates ======")
    agg = d.groupby(["kind", "col"]).agg(
        days=("date", "nunique"), n=("n_uni", "mean"),
        **{f"{s}_t{k}": (f"{s}_t{k}", "mean")
           for s in ("model", "mkt") for k in TOP_KS})
    for k in TOP_KS:
        agg[f"d_t{k}"] = agg[f"model_t{k}"] - agg[f"mkt_t{k}"]
    agg["d_avg"] = agg[[f"d_t{k}" for k in TOP_KS]].mean(axis=1)
    # per-day win/loss at t3 (model vs close)
    wl = (d.assign(w=np.sign(d.model_t3 - d.mkt_t3))
          .groupby(["kind", "col"])["w"]
          .agg(w=lambda s: (s > 0).sum(), l=lambda s: (s < 0).sum()))
    agg = agg.join(wl)
    print(agg.round(3).sort_values(["kind", "d_avg"], ascending=[True, False])
          .to_string())
    if seg == "pre":
        # model vs OPEN, pooled per prop (robustness)
        ago = d.groupby(["kind", "col"]).apply(
            lambda g: pd.Series({
                "d_open_avg": np.mean([(g[f"model_t{k}"] - g[f"open_t{k}"]).mean()
                                        for k in TOP_KS])}),
            include_groups=False)
        print("\nmodel minus OPEN fair (pooled avg over t1/3/5/10):")
        print(ago.round(3).sort_values("d_open_avg", ascending=False)
              .to_string())
