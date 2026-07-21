# Open Items

Trimmed 2026-07-20 (second trim, after W4-A/W4.16/W4-B/W4.12/W4.13 shipped — evidence and
verdicts live in Logs/*_2026-07-20.log, the goal-metric memory, and git history of this file).

**Ship discipline:** paired replay A/B (`evaluate --ab`) + `pytest` green before a change
serves; `Model/walkforward.py` (rolling-origin folds 2022–2026) after material changes to the
trained models. Env gates `PEN_WAVE3` / `PEN_CHOICE` in predict.py reproduce pre-pen-wave
behavior ("0") for paired A/Bs. **Goal metric** (top-10/slate precision + >50% monotone
reliability + trust depth) is measured in Tools/5's Goal Board/Reliability sheets and the
`--ab` goal section. **Calibration refresh cadence:** rerun the extended replay →
`--fit-calibrators --reuse-rows` → `heads --train` at every wave boundary + ~monthly in-season
(~45 min GPU; procedure in the goal-metric memory).

## Watch (check in the next paired A/B or calibration refresh)

- [ ] **`per` (starter ER) lean under PEN_CHOICE** — −0.00046, raw p=.016, BH-tie twice;
  plausibly inherited-runner ER attribution. Investigate only if it persists.
- [x] **Hit / 2+ Hits top-10 dip after W4-B** — RESOLVED by W4.18 (goal board: Hit +1.7,
  2+ Hits +0.7 vs pre-bag; the dip was single-seed variance).
- [ ] **pk / pbb per-line calibration** — holdout-positive but within noise at 46 slates;
  re-assess at the next calibration refresh (`evaluate.LINE_CAL_FAMS` is the one-line switch).
- [ ] **Batter 0.80–0.90 stated bands run ~1–2 pts hot** (n≈1,800 pooled) — recheck on the
  next refreshed ledger.

## Wave 4 — remaining queue

(W4.17 decay REJECTED by design sweep — all half-lives worse, verdict in the train.py config
comment; W4.18 3-seed bagging SHIPPED 2026-07-20 — A/B aggregate CI-positive +0.00009, 20/20
families non-negative, Hit top-10 +1.7 pts recovering the W4-B watch dip, walkforward better
in all 5 folds. W4.14 multi-library ensemble folded into the B7 annual slot — bagging captured
the cheap variance win; a second library is the expensive remainder.)

- [ ] **W4.15** Serve sims 20k→50–100k by routing GUI/headless serve through the sim_batch GPU
  path — stabilizes top-of-sort ordering at similar wall time.
- [ ] **W4.19** Confirmed-lineup re-serve habit + measurement: grade projected-vs-confirmed
  serves separately (npz product tags exist); re-serve when lineups confirm on days you can.
  Process, not code (C2 automates the quantification later).
- [ ] **W4.20** Roof-state flag (the B1 survivor — open/closed Condition into park/weather
  features) + verify per-batter platoon-split SKILL panels exist (matchup platoon is in;
  per-batter EB split skill may not be).
- [ ] **W4.21 (research)** Within-slate ranking overlay (LambdaRank-style display reranker,
  served probabilities untouched) — only if W4-B/W4.14/18 leave top-10 precision short; nothing
  in the stack optimizes ranking directly.
- [ ] **W4.22 (user decision — DECLINED for now, may reconsider later)** Odds-blended workbook
  column (dual output: pure model for gate/CLV, blend for display). Do not implement without
  explicit user choice.

## Blocked (future unblock dates)

- [ ] **B2. Forecast-error weather sampling** — sample temp/wind per sim from the historical
  forecast-error distribution. Blocked: `forecast_error.json` has n=46, `sufficient: false`
  (checked 2026-07-20). Fold in the roof-open/closed Condition flag when this runs.
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
