"""Prop-trust rankings from the replay ledger (rewritten 2026-07-19).

The old version of this tool scored predictions against artifacts from a
retired evaluation pipeline (eval_paired_*.joblib / models_bt.joblib —
none of which the current engine produces). This version reads what the
engine actually maintains:

  Model/artifacts/calib_rows.parquet   every graded (market, p, y) row
                                       from the full-season replay, RAW
                                       engine probabilities
  Model/artifacts/output_calibrators.joblib   the live Platt family maps
  Model/artifacts/gate_rows_cache.parquet     CLV gate rows (optional)

and answers ONE question per served market: how much should you trust
it? Metrics per market:

  n         graded rows in the ledger
  rate      realized base rate
  mean_p    average calibrated probability (bias check vs rate)
  ll_cal    logloss of the calibrated engine
  ll_base   logloss of the constant base-rate predictor
  gain      ll_base - ll_cal  (skill; positive = engine beats naive)
  auc       rank discrimination
  lcb       5th-percentile day-block-bootstrap lower bound on gain —
            the number the tiers are built from (a market is only as
            trustworthy as its WORST plausible skill)
  vs_close  per-family logloss edge vs the de-vigged closing line from
            the gate cache (blank until enough prices accrue)

Tiers: A  lcb > +0.010      trust the number
       B  lcb > 0           real but thin edge
       C  gain > 0, lcb <=0 positive point estimate, unproven
       D  gain <= 0         do not trust in isolation

Output: console table + Tools/PROP_RANKINGS.xlsx (Rankings + Legend),
styled like the engine's serve workbooks.

Usage:
    python "Tools/5) Prop Rankings.py" [--boot 300] [--out PATH]
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model"))
import features as F                      # noqa: E402,F401  (PlattCal)

ART = ROOT / "Model" / "artifacts"
DEFAULT_OUT = ROOT / "Tools" / "PROP_RANKINGS.xlsx"

NAVY = "FF041E42"
TIER_TINT = {"A": "FFE7F3E2", "B": "FFF3F8EE", "C": "FFFFF6E0",
             "D": "FFFBE4E4"}


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _boot_lcb(df, boot, seed=11):
    """Day-block bootstrap 5th percentile of (ll_base - ll_cal)."""
    dates = df["Date"].astype(str).values
    uniq = np.unique(dates)
    if len(uniq) < 5:
        return np.nan
    idx_by = {d: np.flatnonzero(dates == d) for d in uniq}
    y = df["y"].values.astype(float)
    p = df["p_cal"].values
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[d] for d in pick])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():
            continue
        gains.append(_ll(yy, np.full(len(yy), yy.mean())) - _ll(yy, pp))
    return float(np.quantile(gains, 0.05)) if gains else np.nan


def build_table(boot=300):
    rows_p = ART / "calib_rows.parquet"
    if not rows_p.exists():
        raise SystemExit("calib_rows.parquet missing — run a replay "
                         "(evaluate.py --fit-calibrators) first")
    df = pd.read_parquet(rows_p)
    cal_p = ART / "output_calibrators.joblib"
    calib = joblib.load(cal_p) if cal_p.exists() else {}
    df["p_cal"] = df["p"]
    for fam, cal in calib.items():
        m = df["family"] == fam
        if m.any():
            df.loc[m, "p_cal"] = np.clip(
                cal.predict(df.loc[m, "p"].values), 1e-6, 1 - 1e-6)

    gate = {}
    gp = ART / "gate_rows_cache.parquet"
    if gp.exists():
        g = pd.read_parquet(gp)
        for fam, sub in g.groupby("family"):
            y = sub["y"].astype(float).values
            if len(sub) >= 50 and 0 < y.mean() < 1:
                gate[fam] = dict(
                    n=len(sub),
                    vs_close=_ll(y, sub["p_close"].values)
                    - _ll(y, sub["p_model"].values))

    out = []
    for (fam, mkt), sub in df.groupby(["family", "market"]):
        y = sub["y"].values.astype(float)
        if len(sub) < 200 or y.min() == y.max():
            continue
        p = sub["p_cal"].values
        ll_cal = _ll(y, p)
        ll_base = _ll(y, np.full(len(y), y.mean()))
        gv = gate.get(fam, {})
        out.append(dict(
            family=fam, market=str(mkt), n=len(sub),
            rate=float(y.mean()), mean_p=float(p.mean()),
            ll_cal=ll_cal, ll_base=ll_base, gain=ll_base - ll_cal,
            auc=_auc(y, p), lcb=_boot_lcb(sub, boot),
            gate_n=gv.get("n", np.nan),
            vs_close=gv.get("vs_close", np.nan)))
    t = pd.DataFrame(out)
    t["tier"] = np.select(
        [t["lcb"] > 0.010, t["lcb"] > 0, t["gain"] > 0],
        ["A", "B", "C"], default="D")
    return t.sort_values(["lcb"], ascending=False).reset_index(drop=True)


def save_excel(t, path):
    import openpyxl
    from openpyxl.styles import (Alignment, Border, Font, PatternFill,
                                 Side)
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rankings"
    cols = ["tier", "family", "market", "n", "rate", "mean_p", "ll_cal",
            "ll_base", "gain", "auc", "lcb", "gate_n", "vs_close"]
    ws.append(list(cols))
    thin = Side(style="thin", color="FFBFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    for j in range(1, len(cols) + 1):
        cell = ws.cell(row=1, column=j)
        cell.font = Font(bold=True, color="FFFFFFFF")
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = center
        cell.border = border
    for _, r in t.iterrows():
        ws.append([r[c] if pd.notna(r[c]) else "" for c in cols])
    for i in range(2, ws.max_row + 1):
        tint = TIER_TINT.get(ws.cell(row=i, column=1).value)
        for j in range(1, len(cols) + 1):
            cell = ws.cell(row=i, column=j)
            cell.alignment = center
            cell.border = border
            if tint:
                cell.fill = PatternFill("solid", fgColor=tint)
            if cols[j - 1] in ("rate", "mean_p", "auc"):
                cell.number_format = "0.000"
            elif cols[j - 1] in ("ll_cal", "ll_base", "gain", "lcb",
                                 "vs_close"):
                cell.number_format = "0.0000"
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}1"
    for j, c in enumerate(cols, 1):
        width = max([len(c)]
                    + [len(str(ws.cell(row=i, column=j).value))
                       for i in range(2, min(ws.max_row, 50) + 1)]) + 4
        ws.column_dimensions[get_column_letter(j)].width = min(width, 30)

    lg = wb.create_sheet("Legend")
    lines = [
        ("tier", "A trust / B thin edge / C unproven / D no skill — "
                 "built from lcb"),
        ("n", "graded ledger rows (full-season replay)"),
        ("rate / mean_p", "realized vs stated — bias check"),
        ("gain", "ll_base - ll_cal: skill over the naive base rate"),
        ("auc", "rank discrimination"),
        ("lcb", "5th pct day-block bootstrap of gain — the tier input"),
        ("gate_n / vs_close", "CLV gate: prices graded, logloss edge "
                              "vs de-vigged close (blank = too few)"),
    ]
    for k, v in lines:
        lg.append([k, v])
    for i in range(1, lg.max_row + 1):
        lg.cell(row=i, column=1).font = Font(bold=True)
    lg.column_dimensions["A"].width = 22
    lg.column_dimensions["B"].width = 90
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    t = build_table(boot=args.boot)
    with pd.option_context("display.width", 140,
                           "display.max_rows", 200):
        show = t.copy()
        for c in ("rate", "mean_p", "auc"):
            show[c] = show[c].round(3)
        for c in ("ll_cal", "ll_base", "gain", "lcb", "vs_close"):
            show[c] = show[c].round(4)
        print(show.to_string(index=False))
    counts = t.tier.value_counts().to_dict()
    print(f"\ntiers: { {k: counts.get(k, 0) for k in 'ABCD'} }")
    save_excel(t, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
