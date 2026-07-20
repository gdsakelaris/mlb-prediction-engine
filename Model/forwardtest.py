"""Pre-registered forward test: freeze the serving stack, declare an
evaluation window BEFORE it happens, grade it untouched afterward.

Why: 2025 is a clean holdout for the A-model only — calibrators, heads,
latent, and the tree-ship decision all consumed it. The defensible
performance claim is a forward window served by a declared, unchanged
stack. Daily A-model retrains are PART of the operated system (rolling
as-of data, fixed code/hyperparameters) and stay allowed; what must not
move during the window is everything fit by hand: output calibrators,
residual heads, latent.json, and the Model/ code itself.

    python Model/forwardtest.py --freeze          # declare the window
    python Model/forwardtest.py --check           # mid-window audit
    python Model/forwardtest.py --grade           # after window ends

Freeze writes artifacts/prereg/prereg_<start>_<end>.json with SHA-256
content hashes of the frozen artifacts + every Model/*.py file, the
window, and the pre-registered grading protocol. Grade refuses to run
early (--force overrides, clearly labeled), verifies the freeze is
intact, then runs the standard CLV gate + skill ledger over the window
and writes report_<start>_<end>.json next to the freeze file. A freeze
violation doesn't abort grading — it stamps the report VOIDED so the
result can't be quoted as pre-registered.
"""
import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F      # noqa: E402

ART = F.ART
PREREG = ART / "prereg"
FROZEN = ["output_calibrators.joblib", "residual_heads.joblib",
          "latent.json", "xgb_params.json"]

PROTOCOL = {
    "primary": "per-family logloss gain vs de-vigged close on captured "
               "odds (evaluate.market_gate), date-block bootstrap, "
               "Benjamini-Hochberg across families; PASS = n>=800 AND "
               "BH p<0.05 AND CI5>0",
    "secondary": "per-family logloss gain vs base rate (skill ledger); "
                 "confirmed-vs-projected product split when the slate "
                 "archive holds both",
    "allowed_during_window": "daily data scrape + A-model retrain "
                             "(fixed code, rolling as-of data); manual "
                             "slate serving and grading",
    "forbidden_during_window": "refits of output calibrators, residual "
                               "heads, or latent.json; any edit to "
                               "Model/*.py; any decision derived from "
                               "window results before grading",
}


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _hashes():
    out = {}
    for f_ in FROZEN:
        p = ART / f_
        out[f_] = _sha(p) if p.exists() else "MISSING"
    for p in sorted(Path(__file__).parent.glob("*.py")):
        out[f"code/{p.name}"] = _sha(p)
    return out


def _freeze_path(start, end):
    return PREREG / f"prereg_{start}_{end}.json"


def _latest_freeze():
    fps = sorted(PREREG.glob("prereg_*.json"))
    if not fps:
        sys.exit("no freeze found — run --freeze first")
    return json.loads(fps[-1].read_text()), fps[-1]


def freeze(start, end):
    PREREG.mkdir(parents=True, exist_ok=True)
    fp = _freeze_path(start, end)
    if fp.exists():
        sys.exit(f"{fp.name} already exists — a pre-registration is "
                 "immutable; declare a different window instead")
    doc = dict(declared=date.today().isoformat(), window_start=start,
               window_end=end, protocol=PROTOCOL, hashes=_hashes())
    fp.write_text(json.dumps(doc, indent=1))
    print(f"frozen: {fp}")
    print(f"window: {start} .. {end}  "
          f"({len(doc['hashes'])} hashed objects)")


def check(quiet=False):
    doc, fp = _latest_freeze()
    now = _hashes()
    bad = [k for k, v in doc["hashes"].items() if now.get(k) != v]
    new = [k for k in now if k not in doc["hashes"]]
    if not quiet:
        print(f"freeze {fp.name}: window {doc['window_start']} .. "
              f"{doc['window_end']}")
        if bad:
            print(f"VIOLATED — changed since freeze: {bad}")
        if new:
            print(f"note — new code files since freeze: {new}")
        if not bad:
            print("intact")
    return doc, bad + new


def supersede(n_sims, min_n):
    """Close the open window NOW because an improvement wave is about
    to land: grade start..today while the hashes are still intact,
    stamp the freeze superseded, and leave the partial record on file.
    Refreeze discipline: supersede at offline-validated milestones,
    NEVER on interim results — and every truncated window's record
    survives here (no silent discards)."""
    doc, fp = _latest_freeze()
    if doc.get("superseded"):
        sys.exit(f"{fp.name} already superseded {doc['superseded']}")
    _, violations = check(quiet=False)
    start, end = doc["window_start"], doc["window_end"]
    today = date.today().isoformat()
    gend = min(end, today)
    import evaluate as E
    df, rep = E.market_gate(start, gend, n_sims=n_sims, min_n=min_n)
    led = E.skill_ledger(start, gend)
    report = dict(
        graded=today, window=[start, end], graded_through=gend,
        status=(("VOIDED: " + "; ".join(violations)) if violations
                else f"SUPERSEDED (graded {start}..{gend})"),
        n_prices=int(len(df)) if df is not None else 0,
        families=rep.to_dict(orient="records")
        if rep is not None and len(rep) else [],
        ledger=(led.to_dict(orient="records")
                if hasattr(led, "to_dict") else None),
    )
    out = PREREG / f"report_{start}_{end}_superseded.json"
    out.write_text(json.dumps(report, indent=1))
    doc["superseded"] = today
    fp.write_text(json.dumps(doc, indent=1))
    print(f"\nreport written: {out}  status={report['status']}")
    print(f"{fp.name} marked superseded — declare the next window "
          f"with --freeze once the new stack is live")


def grade(n_sims, min_n, force=False):
    doc, violations = check(quiet=False)
    if doc.get("superseded"):
        sys.exit(f"window superseded {doc['superseded']} — its partial "
                 "record is the *_superseded.json report; freeze the "
                 "next window instead")
    start, end = doc["window_start"], doc["window_end"]
    if date.today().isoformat() <= end and not force:
        sys.exit(f"window ends {end} — grading early would un-register "
                 "the test (--force to override, result will be "
                 "labeled EARLY)")
    import evaluate as E
    df, rep = E.market_gate(start, end, n_sims=n_sims, min_n=min_n)
    led = E.skill_ledger(start, end)
    report = dict(
        graded=date.today().isoformat(), window=[start, end],
        status=("VOIDED: " + "; ".join(violations) if violations else
                ("EARLY" if date.today().isoformat() <= end else
                 "PRE-REGISTERED")),
        n_prices=int(len(df)) if df is not None else 0,
        families=rep.to_dict(orient="records")
        if rep is not None and len(rep) else [],
        ledger=(led.to_dict(orient="records")
                if hasattr(led, "to_dict") else None),
    )
    out = PREREG / f"report_{start}_{end}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nreport written: {out}  status={report['status']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--supersede", action="store_true",
                    help="close the open window at a milestone: grade "
                         "start..today, mark the freeze superseded "
                         "(partial record kept), ready for the next "
                         "--freeze")
    ap.add_argument("--start",
                    default=(date.today() + timedelta(days=1)
                             ).isoformat())
    ap.add_argument("--end",
                    default=(date.today() + timedelta(days=28)
                             ).isoformat())
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--min-n", type=int, default=800)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.freeze:
        freeze(args.start, args.end)
    elif args.check:
        check()
    elif args.grade:
        grade(args.sims, args.min_n, force=args.force)
    elif args.supersede:
        supersede(args.sims, args.min_n)
    else:
        ap.error("pass one of --freeze / --check / --grade / "
                 "--supersede")
