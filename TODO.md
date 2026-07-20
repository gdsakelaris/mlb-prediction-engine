# Improvement Queue (running to-do list)

Triage of the 2026-07-20 ChatGPT repo review, reconciled against what's already built,
already queued (Wave 2), or already rejected with evidence. Check items off as they land.

**Evaluation regime (changed 2026-07-20, user decision):** the freeze/forward-test system is
RETIRED — too much ceremony, and market-beating proof is not a goal. forwardtest.py DELETED
2026-07-20 (recoverable from git history); the prereg window `prereg_2026-07-21_2026-08-17` is
moot and its freeze doc in artifacts/prereg/ is just a historical record.
**Standing regime: `Model/walkforward.py`** — rolling-origin walk-forward (train ≤Y−2,
calibrate Y−1, evaluate Y, folds 2022–2026), the production training scheme rolled backward.
Ship-a-change discipline: paired replay A/B (`evaluate --ab`) + `pytest` green BEFORE a change
serves; rerun walkforward.py after material model changes. Consequence: Model/*.py edits are no
longer gated on refreeze milestones — Section B items need only the offline validation above.

---

## A. Freeze-safe — can start any time (no frozen-file edits)

- [X] **A1. Simulation invariant + parity test suite** (ChatGPT #12) — **DONE 2026-07-20**: `Tests/`
  (31 tests: 28 fast synthetic + 3 slow golden vs real artifacts, `pytest` / `pytest -m slow`).
  Ledger conservation identities (the walk-off-bug regression net), season rules (7-inn DH, ghost
  runner, walk-off legality), pen exhaustion/platoon-jump/rotation, steal accounting,
  classic-vs-batch CPU AND GPU distribution parity, prepare_games row parity, reproducibility.
  Suite immediately caught the BatchPrep game0-latent landmine (see B11) — pinned in
  `test_batch_latent_comes_from_game0`.

- ~~**A2. Serve-time archive wrapper**~~ — built 2026-07-20 (Tools/6), then **RETIRED same day by
  user decision** along with the freeze apparatus (stashed in session scratchpad). One archive of
  2026-07-19 exists in `Data/serve_archive/`. If lineup-uncertainty quantification (C2) is ever
  wanted, this comes back — it was the data-collection half of that item.
- ~~**A3. Calibration diagnostics report**~~ — built 2026-07-20 (Tools/7), findings recorded, then
  **RETIRED same day by user decision** (stashed in session scratchpad). Findings stand: NO
  credible subgroup calibration drift (cluster-robust z by GamePk — independence z's were ~3x
  inflated; worst cell bk@Fenway z=−3.3, ~chance across 214 cells); b1 has 17% of rows within
  2 MC-SE of coin flip at 4k replay sims (supports 20k serve sims).

- [X] **A4. Cross-fitted calibration audit** (ChatGPT #2, audit half) — **DONE 2026-07-20**
  (Logs/audit_cf_2026-07-20.log, artifacts/audit_cf/report_audit_cf.json). Scoping finding:
  SERVE_CAL_YEAR=2025 → serving tree trains Season<=2024, so the calib window was NEVER in the
  model weights — ChatGPT's premise mostly moot by design. Audit (arm B: a1/a2 retrained on
  pa<=2025-04-30, swap-replay with SHA-verified restore): per-family Platt shifts max |Δp|
  .004–.026, BUT arm B's scalers anchored on ~1 month of April games → the shifts are an upper
  bound dominated by small-sample/seasonal scaler noise (systematic intercept pattern = April run
  environment). **Verdict: no evidence of material calibrator dishonesty; served p carries
  ~±0.01–0.02 systematic uncertainty from calibration-slice choice. B9 dropped.**

## B. Wave 3 candidates — need Model edits

- [X] **B1. Moist-air density** (ChatGPT #7) — **RESOLVED BY EVIDENCE 2026-07-20, dropped from
  Wave 3** (Logs/density_study_2026-07-20.log, 27,121 games). ChatGPT's premise was wrong twice:
  (1) scraped Pressure is already Open-Meteo `surface_pressure` (station-level, elevation
  embedded — no estimation needed); (2) the humidity correction the P/T proxy misses averages
  −0.65% (worst −2.0%) vs an 18.4% venue density spread already captured; spearman(proxy, moist)
  = 0.9972; humidity is ALSO a standalone A1 feature so XGB can learn the residual directly.
  Surviving sliver: **roof-open/closed state flag** (Condition string exists) — fold into B2's
  weather work if it ever runs.
- [ ] **B2. Forecast-error weather sampling** (ChatGPT #7b) — once forecast_error.json matures, sample
  temp/wind per sim from the historical forecast-error distribution instead of one fixed forecast.
  Pairs with the Wave 2 lineup-uncertainty item (same "input uncertainty" machinery).
  *Checked 2026-07-20: still blocked — store has n=46, self-reports `sufficient: false`.*
- [x] **B3. Continuous pen fatigue** (ChatGPT #5a) — **IMPLEMENTED 2026-07-20** (A/B pending, see
  below). Study (Logs/pen_fatigue_study_2026-07-20.log, 834k relief PAs, within-pitcher offsets):
  in the serving-relevant region (np1<25, no back-to-back — the arms availability doesn't already
  exclude) only BB is real (+0.040 log-odds per 20 NP yesterday, z=+3.7); K is NULL (its decay
  lives in the 25+/b2b arms `_pen_for` excludes); HR +0.016 ns. Shipped: `pen_fatigue.json` store
  (features.py build_pen_fatigue, refit each build; only |z|>=2 classes apply), avec class-odds
  offsets at prep (`_apply_pen_fatigue`), and graded fresh/mid(10+)/heavy(20+ or b2b) demotion
  tiers replacing the binary TIRED_NP sort key. **A/B VERDICT (pooled 2,975 games, 223 slates,
  2025-06..09 + 2026-04..07, Logs/pen_wave_ab_verdicts_2026-07-20.log): dead tie with B4
  (ALL −0.00004, CI [−0.00011, +0.00003], every family TIE) — no harm, realism kept. SHIPPED
  (PEN_WAVE3 default on).**
- [x] **B4. Reliever stint length conditioning** (ChatGPT #5c) — **IMPLEMENTED 2026-07-20** (A/B
  pending). Study (Logs/pen_exit_study_2026-07-20.log): huge pitcher heterogeneity (5th-95th pct
  mean stint outs 2.6→5.4); trailing-365d per-pitcher hazard blended with M=12 league
  pseudo-stints (holdout-tuned) beats league-only by 5.2% rel. on held-out per-break exit log
  loss. Shipped: relief_exit is now per-pitcher [n_players, 11] (predict `_pitcher_exit_table`;
  sim/sim_batch accept 1-D legacy or 2-D), pinned by test_parity::test_per_pitcher_exit_tables.
  Game-state (entry-margin) conditioning measured SMALL (±0.13 log-odds at k=3) and mostly
  pitcher-identity confounded (mop-up long men enter blowouts) — absorbed by per-pitcher tables,
  intentionally not double-counted. **A/B VERDICT: shipped with B3 (see above — pooled tie, no
  harm; component-level exit prediction is the win).**
- [x] **B5. Opener / bulk / tandem / short-rest detection** (ChatGPT #5d) — **RESOLVED 2026-07-20:
  already covered, no new code.** Verified the served hazard artifact conditions on gap_days
  (short rest), ramp (low prev NP), prev_short (opener-length last start), il_ret30, outs_sd —
  the starter-exit side of all four patterns. The pen side (bulk/tandem length) is B4's
  per-pitcher stints; entry of the long man after an early exit falls out of the lo-leverage
  order, which sorts by avg-outs merit. ChatGPT's premise (league-wide exit + no usage features)
  was stale.
- [x] **B6. Probabilistic manager reliever choice** (ChatGPT #5b) — **RESEARCHED + IMPLEMENTED
  2026-07-20** (ship gate = A/B win, pending). Study (Logs/pen_choice_study_2026-07-20.log, 7,954
  first-reliever entries 2024-25): the deterministic rank-1 pick matches the actual first arm in
  only 13.6% (uniform baseline 12.7%) — manager entry choice is near-flat over our order
  (pmf 13.6/16.8/17.2/15.5/13.8/10.2/7.5/5.5%). Shipped: `pen_choice.json` store (empirical hi/lo
  rank pmfs, features.py build_pen_choice), sim/sim_batch `pen_rank_cum` — entry rank sampled per
  sim from the pmf over still-available arms (subsumes the platoon jump, which only applies in
  the legacy deterministic path). Pinned by test_parity::test_pen_rank_pmf_pick.
  **A/B VERDICT — WIN, SHIPPED (PEN_CHOICE default on). Pooled B1-vs-B2 (2,975 games, 223
  slates): batter-K family B BETTER after BH (+0.00022, CI [+0.00010, +0.00032], p_bh=0.000 —
  exactly where reliever-identity spread should land) and aggregate CI-positive (+0.00009,
  CI [+0.00001, +0.00018]). The 2025-only window alone was a lean-positive TIE; 2026 pooling
  resolved it. WATCH: `per` (starter ER) shows a persistent negative lean (−0.00046, raw p=.016,
  BH-tie both windows) — plausibly inherited-runner ER attribution interacting with spread
  reliever quality; revisit if it survives the next A/B.**
- [ ] **B7. Per-component Optuna + library bake-off** (ChatGPT #3) — at the ANNUAL retune slot only
  (Wave 1 found a flat plateau, +0.0006 — retunes are annual by policy): separate studies for
  T1 / T3 / A2 / hazard / SB, and a one-time XGB vs LightGBM/CatBoost/regularized-linear
  comparison on the design split. Low expected value; do not do mid-season.
- [ ] **B8. Multi-fold walk-forward for annual retunes** (ChatGPT #2, tuning half) — when B7 runs,
  score hyperparams on aggregate out-of-fold log loss across chronological folds
  (…→2021, …→2022, …→2023, …→2024) instead of the single 2024 design year.
- [ ] **B10. Bench/PH realism** (ChatGPT #6, partial) — actual bench players + platoon-dependent PH
  selection instead of the generic bench batter after starter exit. Previously deprioritized as
  negligible EV; revisit only with evidence from A3 diagnostics (late-game prop residuals).
- [x] **B11. BatchPrep per-game latent** (found by A1 suite 2026-07-20) — **FIXED 2026-07-20**:
  BatchPrep now stacks per-game latent params (`bp.lat`, indexed by gidx at draw time) instead of
  broadcasting preps[0].latent. Pinned test flipped to the positive
  `test_parity.py::test_batch_latent_is_per_game` (mixed-latent batch matches per-game classic).
  Full suite green (28 fast + 3 golden). No distributional change under the shared-latent
  production contract — replay A/B not required.

## C. Already queued (Wave 2 — unchanged, do not duplicate)

- [ ] **C1. State-space / dynamic latent skill** (ChatGPT #4) — Kalman-ish posterior mean+variance as
  A1 inputs; posterior *sampling* in the sim is the ChatGPT addition — fold into the same
  experiment. Bar: beat design-eval AND replay. Honest prior: EB + multi-horizon + velo/xw + age
  already approximates it.
- [ ] **C2. Lineup-uncertainty quantification** (ChatGPT #1/#6) — probability-weighted lineup
  distribution before confirmation. BLOCKED on slate-archive accrual (~late Aug 2026); A2 above
  feeds this directly.

## D. Deferred research (no commitment; evidence required to promote)

- [ ] **D1. Latent correlation moments** (ChatGPT #9) — add teammate/opponent/prop correlation and
  tail moments as *diagnostics* to moment_match at n=300+. Main payoff was SGP joint accuracy and
  SGP was deleted by user decision — promote only if SGP returns or totals-tail residuals justify.
- [ ] **D2. Port moment_match to sim_batch** — ~8 min/eval per-game engine is the refit bottleneck;
  do this if latent refit cadence increases.
- [ ] **D3. Learned advancement-transition model** (ChatGPT #11) — pattern table + speed/arm/DP tilts
  stay primary; a learned model conditioned on identities/spray/park is heavy for likely-small
  gain. Keep pattern table as fallback/baseline if ever attempted.
