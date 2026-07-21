# Open Items

Trimmed 2026-07-20 (second trim, after W4-A/W4.16/W4-B/W4.12/W4.13 shipped — evidence and
verdicts live in Logs/*_2026-07-20.log, the goal-metric memory, and git history of this file).

**Ship discipline:** paired replay A/B (`evaluate --ab`) + `pytest` green before a change
serves — that is the whole per-ship gate. `Model/walkforward.py` (rolling-origin folds
2022–2026) runs at wave boundaries / ~monthly (with the calibration refresh) and for
STRUCTURAL changes (new model class, loss, sampling scheme, hyperparameter retune), not per
shipped feature (user policy 2026-07-20). Env gates `PEN_WAVE3` / `PEN_CHOICE` in predict.py
reproduce pre-pen-wave behavior ("0") for paired A/Bs. **Goal metric** (top-10/slate precision + >50% monotone
reliability + trust depth) is measured in Tools/5's Goal Board/Reliability sheets and the
`--ab` goal section. **Calibration refresh cadence:** rerun the extended replay →
`--fit-calibrators --reuse-rows` → `heads --train` at every wave boundary + ~monthly in-season
(~45 min GPU; procedure in the goal-metric memory).

## Watch (check in the next paired A/B or calibration refresh)

- [ ] **`per` (starter ER) lean under PEN_CHOICE** — −0.00046, raw p=.016, BH-tie twice;
  plausibly inherited-runner ER attribution. Needs a dedicated PEN_CHOICE=0 arm; `per`
  reliability on the W4.20 refreshed ledger shows no calibration-level distortion (bands
  alternate sign, all ≲1.3σ). Investigate only if it persists.
- [ ] **Batter-hits top-10 cluster after W4.20** — Hit −1.5 / HR −0.7 / Single −0.6 /
  3+ TB −0.6 pts on the goal board (same metric swung +1.7 in W4.18; AUC flat-to-positive on
  the same markets, so read as replay MC re-ranking noise at 4k sims). Recheck next A/B.
- [x] **pk / pbb per-line calibration** — RESOLVED at the W4.20 refresh (298-slate ledger,
  70/30 chrono holdout): pk per-line WORSE (−0.00012) — stays out; pbb +0.00019 borderline
  (3 lines) — stays out, flip only on a second consecutive positive; pout re-confirmed
  strongly (+0.006). `evaluate.LINE_CAL_FAMS` unchanged = ("pout",).
- [ ] **Batter 0.80–0.90 stated bands** — still +1.1 pts hot on the W4.20 ledger (n=1,847,
  ≈1.2σ); persistent lean but not decisive. Recheck at next refresh.

## Wave 4 — remaining queue

(W4.17 decay REJECTED by design sweep — all half-lives worse, verdict in the train.py config
comment; W4.18 3-seed bagging SHIPPED 2026-07-20 — A/B aggregate CI-positive +0.00009, 20/20
families non-negative, Hit top-10 +1.7 pts recovering the W4-B watch dip, walkforward better
in all 5 folds. W4.14 multi-library ensemble folded into the B7 annual slot — bagging captured
the cheap variance win; a second library is the expensive remainder.)

(W4.15 SHIPPED 2026-07-20: serve routes through sim_batch on GPU at 100k sims, chunked 25k/
device batch; 43/43 families within 2x MC noise vs per-game path, top-10 overlap .96, 100k
batched = HALF the wall of 20k per-game; SERVE_BATCH=0 reproduces the old path. W4.20 SHIPPED
2026-07-20: roof_open flag + b_pl_/p_pl_ platoon-split deltas, 276→285 features; A/B no-harm
tie aggregate lean-positive, walkforward better all 5 folds vs pre-wave. W4.19 verified: npz
retired 07-19, lineup provenance = away/home_lineup_src in slate JSON + serve printout; the
re-serve-on-confirmation habit is manual process — C2 automates the quantification later.)

- [ ] **W4.21 (research)** Within-slate ranking overlay (LambdaRank-style display reranker,
  served probabilities untouched) — only if W4-B/W4.14/18 leave top-10 precision short; nothing
  in the stack optimizes ranking directly.
- [ ] **W4.22 (user decision — DECLINED for now, may reconsider later)** Odds-blended workbook
  column (dual output: pure model for gate/CLV, blend for display). Do not implement without
  explicit user choice.

## Blocked (future unblock dates)

- [ ] **B2. Forecast-error weather sampling** — sample temp/wind per sim from the historical
  forecast-error distribution. Blocked: `forecast_error.json` has n=46, `sufficient: false`
  (checked 2026-07-20). The roof-state flag shipped separately in W4.20 (`roof_open`).
- [ ] **C2. Lineup-uncertainty quantification** — probability-weighted lineup distribution before
  confirmation. Blocked on slate-archive accrual, **revisit ~late Aug 2026**. The retired Tools/6
  archive wrapper was the data-collection half — bring it back if this runs.

## Annual retune slot only (do not do mid-season)

- [ ] **B7. Per-component Optuna + library bake-off** — separate studies for T1/T3/A2/hazard/SB
  plus a one-time XGB vs LightGBM/CatBoost/regularized-linear comparison. Wave 1 found a flat
  plateau (+0.0006); low expected value. W4.14/W4.17/W4.18 outcomes feed this slot.
- [ ] **B8. Multi-fold hyperparam scoring** — when B7 runs, score on aggregate out-of-fold log
  loss across chronological folds (…→2021 … …→2024) instead of the single 2024 design year.

## Evidence-gated / deferred research

- [ ] **B10. Bench/PH realism** — actual bench players + platoon-dependent PH selection. Revisit
  only with evidence of late-game prop residuals.
- [ ] **C1. State-space / dynamic latent skill** — Kalman-ish posterior mean+variance as A1
  inputs + posterior sampling in the sim. Bar: beat design-eval AND replay. Honest prior: EB +
  multi-horizon + velo/xw + age already approximates it.
- [ ] **D1. Latent correlation moments** — teammate/opponent/prop correlation diagnostics in
  moment_match at n=300+. Promote only if SGP returns or totals-tail residuals justify.
- [ ] **D2. Port moment_match to sim_batch** — do if latent refit cadence increases (~8 min/eval
  per-game engine is the bottleneck).
- [ ] **D3. Learned advancement-transition model** — pattern table + tilts stay primary; heavy
  for likely-small gain.

## Display caveat (known, low priority)

- Tools/5's cross-fit calibration is family-grain, so its pout rows understate the served
  stack (which adds per-line maps); line-aware cross-fit is the fix if the gap ever matters.
