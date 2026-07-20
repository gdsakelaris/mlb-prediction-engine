# Open Items

Trimmed 2026-07-20 after the pen wave shipped (B3/B4/B5/B6/B11 + A1/A4/B1 before it — evidence
and verdicts live in Logs/*_2026-07-20.log and git history of this file).

**Ship discipline:** paired replay A/B (`evaluate --ab`) + `pytest` green before a change
serves; `Model/walkforward.py` (rolling-origin folds 2022–2026) after material changes to the
trained models. Env gates `PEN_WAVE3` / `PEN_CHOICE` in predict.py reproduce pre-wave behavior
("0") for future paired A/Bs.

## Watch

- [ ] **`per` (starter ER) family lean under PEN_CHOICE** — −0.00046, raw p=.016, BH-tie in both
  A/B windows; plausibly inherited-runner ER attribution interacting with spread reliever
  quality. Check it in the next paired A/B; investigate only if it persists.

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
  plateau (+0.0006); low expected value.
- [ ] **B8. Multi-fold hyperparam scoring** — when B7 runs, score on aggregate out-of-fold log
  loss across chronological folds (…→2021 … …→2024) instead of the single 2024 design year.

## Wave 4 — goal-aligned (top-of-sort) improvements (queued 2026-07-20)

**Goal metric (user-defined):** sort each workbook column high→low — most of the top 10 should
hit; cells >50% should hit at their stated rate, monotone the higher they go. Baseline
diagnostic on the 76-slate replay (2025-05-01..07-15, calibrators applied, heads NOT applied;
script: session scratchpad `goal_diag.py`): **>50% reliability already tight and monotone**
(every pooled batter band within ±0.01); **top-10/slate is honest** (most batter columns hit at
or above stated: 2+ Hits +3.4pts, Single +3.5, H+R+RBI 2+ +4.2). Binding constraint =
**discrimination** (batter AUCs .56–.70: Double .560, Single .566, Hit .581, HR .638). Sore
spots: pout high lines (Outs>18.5 stated .559 hit .441 pre-heads; Outs>15.5 top-10 −6.5pts),
batter BB >50% runs hot (n=203), **Triple never gets replay rows → b3 serves uncalibrated**.
Old-project mining (sanctioned 2026-07-20): `feature_keep.json` survival evidence backs the
transplants below; line refs are into `Desktop\MLB\Model\features.py` unless noted.

### W4-A. Measurement first — DONE 2026-07-20

- [x] **W4.1** DONE. `evaluate.goal_metrics` / `evaluate.reliability_bands` (single shared
  implementation): per-market top-10/slate stated-vs-hit, >50% region, and bootstrap-LCB
  trust depth (odds-ratio-lift 10th-pct LCB ≥ 1.5). Surfaced in Tools/5 (new "Goal Board" +
  "Reliability" sheets + console ladder; ledger load refactored into `_load_ledger`) and as a
  ranking-only section at the end of `evaluate --ab` (raw p is order-identical to served p —
  monotone family calibrators). Validated on the real prewave-vs-current A/B.
- [x] **W4.2** DONE. Triple added to `BAT_ACTUAL` — b3 rows flow into the next replay; the b3
  calibrator fits at the next `--fit-calibrators` (pairs with W4.16).
- [x] **W4.3** DONE. Heads-applied verdict (scratchpad `w43_heads_check.py`): the pout head
  NARROWS but does not fix the high-line miscalibration — Outs>18.5 >50%-region gap −13.2 →
  −10.4 pts on the honest holdout (−11.8 → −4.3 full-window/part-in-sample), Outs>15.5 stays
  ≈ −6.7 pts. **W4.12 confirmed necessary.** pbb's head fully repairs its family (BB>1.5
  top-10 gap → +1.0 pt holdout); tot heads fine. pytest 30/30 green.

### W4-B. A1 discrimination — SHIPPED 2026-07-20 (KEEP on no-harm tie + targeted goal-board gains)

**Recon finding:** W4.4 core log5s (`mx_k5/bb5/hr5/hit5`), W4.9 (per-TTO `pt_` rates), and W4.11
(BvP `bvp_xw_resid`/`bvp_hr_resid`, exact K=30/50 design) already existed at HEAD — memory/TODO
were stale; only the genuinely missing pieces were built. **Implemented:** W4.5 on-deck
protection (`_derive_ondeck` from PA stream at train, lineup slot+1 at serve in both engines;
`od_` shrunk rates + `od_obp/od_slg` + `mx_pitch_around`), W4.6 current-season bat tracking
(`panel_bat_track` from raw-pitch bat speed 2024+; `btk_speed/fast/swlen` + `btk_speed_dd`
drift vs prior-season baseline), W4.7 ump products (`mx_ump_k/bb/calledk`), W4.8 pitcher venue
splits (`panel_pit_out_loc` + `pv_` rates), W4.10 park XBH (`pf_2B` consumption was MISSING +
`mx_park_2b/3b`), W4.4b contact log5s (`mx_gb5/air5/brl5/xw5` + `mx_pullair`; `bq_air/pq_air`
now emitted). Features 223→276.

**Verdict (Logs/w4b_cycle_2026-07-20.log):** paired A/B on 1.71M shared rows = no-harm TIE
(ALL +0.00001; hr +0.00025 ci_lo>0 raw, BH-tie). Goal board: targeted columns up — Double
top-10 +1.0 pt, HR +0.9, 4+ TB +0.9, H+R+RBI 4+ +1.0, Run +1.0, Triple AUC +.003; give-back
Hit −1.3 / 2+ Hits −0.7 (~1.5 SE, WATCH next A/B). Walkforward stable all folds (+.028..+.032
vs league). pytest 30/30. Calibrators + heads refreshed on the new ledger (pout head +0.0045);
serve smoke PASS. Arm-A baseline: `artifacts/rows_prew4b.parquet`.

Original item list (statuses above):

- [ ] **W4.4** `mix_*` outcome log5 products (batter rate × starter-allowed rate: k/bb/hr/hit/
  gb/ld/pullair + contact-quality air/brl/xwcon; old `3102-3113`). Near-universal keep-list
  survivors; complementary to the arsenal collisions (those are pitch-level, these outcome-level).
- [ ] **W4.5** On-deck protection as per-PA A1 features: `ctx_behind_slg/obp` (+ decayed) and
  the `pitch_around` interaction (old `3855-3927`, `3196-3199`). This is a genuine per-PA causal
  channel the sim CANNOT emerge — A1 never sees who's on deck; affects BB/HR-avoidance.
- [ ] **W4.6** Current-season decayed bat-tracking panels + deltas (bt_ features are prior-season
  only today; scraper already pulls current data). Bat speed is the leading batter-power indicator.
- [ ] **W4.7** Ump × pitcher products: `ump_k_x_pk`, `ump_bb_x_pbb`, `ump_k_x_take` (old
  `2999-3002`, `3171-3195`; kept on ~15 heads). A1 has both terms separately; trees can't multiply.
- [ ] **W4.8** Starter venue splits `pvloc_*` (as-of home/road K/HR/ERA per BF, EB-shrunk; old
  `2241-2248`) — batter side exists (`bl_`), pitcher side doesn't.
- [ ] **W4.9** Per-pitcher TTO-decay skill (shrunk 3rd-vs-1st-pass degradation; old `1775-1779`).
  A1 has raw tto; this adds the per-pitcher susceptibility.
- [ ] **W4.10** `park_x_2b` / `park_x_3b` products (old `3118-3119`, `3158-3161`) — Double is the
  weakest column (AUC .560) and park geometry for 2B/3B isn't carried today.
- [ ] **W4.11** BvP shrunk residuals (`bvp_xwoba_resid` K=30, `bvp_hr_resid` K=50, `bvp_n`; old
  `1738-1772`) — design-eval judged. The "no BvP micro-samples" rule targeted raw rates; the
  residual-off-own-baseline encoding is the disciplined form and survived selection.

### W4-C. Calibration polish

- [x] **W4.12** DONE 2026-07-20 (pout only, holdout-gated). Study (scratchpad
  `w412_line_cal_study.py`, fit ≤05-31 / eval 06-01+): per-line Platt beat the family map on
  **pout +0.00578** holdout ll (Outs>18.5 +0.0156, hot rung eliminated; >15.5 gap −6.6→−2.1);
  **per −0.00013 / pha −0.00017 REJECTED; pk +0.00018 / pbb +0.00034 within noise — re-assess
  at future calibration refreshes** (`evaluate.LINE_CAL_FAMS` is the one-line switch).
  Implementation: per-line Platt under calibrator key `_lines` (market-string keyed);
  `predict._cal(market=)` line-map-wins lookup; game_frame passes column names + a
  calibration-stage ladder guard (heads path re-guards independently); bets/gate paths via
  `_line_market` (odds market → column); heads.py applies the identical maps to its base
  margin. **Post-ship evidence: pout head collapsed 47 trees/+0.0045 → identity (+0.00001) —
  the head had been compensating for line-level bias all along**; pout Platt-stage holdout ll
  0.5479→0.5421. Served Outs>18.5 map a=−0.728. pytest fast+slow green; serve smoke ladder
  monotone. Note: Tools/5 cross-fit stays family-grain, so its pout rows understate the served
  stack until line-aware cross-fit is added (display caveat only).
- [x] **W4.13** DONE 2026-07-20. `build_thresh_panel` (554,652 batter-days: per-game
  H+R+RBI≥2/3/4, 2+RBI, 2+Runs clear rates; 90d + career horizons; league priors in
  `thresh_league.json`) + shared `F.thresh_features` (EB K=40; zero history → league prior
  exactly) consumed by heads training (`_load_rows`) AND serving (`Predictor._thresh_map` →
  `_apply_heads(thr=)`) through the SAME builder — no train/serve drift possible. Held-out
  verdict (early-stop refereed): real structure found — sb +0.0007→+0.0012, bk +0.00001→
  +0.0006, bb +0.00004→+0.0005, h −0.00002→+0.0002, r/hrr/b1/pha all up; rbi ~0; hr/b2/b3
  identity; pout stays identity (line maps own it). Gotcha fixed: parquet Date round-trips
  as [s] — `thresh_features` normalizes both sides to [ns]. Serve smoke PASS, pytest 30/30.

### W4-D. Bigger swings (evidence-gated)

- [ ] **W4.14** Ensemble pull-forward from B7 (just the bake-off half, not the retune): LightGBM
  and/or CatBoost T1/T3 alongside XGB, average calibrated logits. Ensembling is the most reliable
  pure-AUC lever; user's goal is discrimination-bound.
- [ ] **W4.15** Serve sims 20k→50–100k by routing GUI/headless serve through the sim_batch GPU
  path — stabilizes top-of-sort ordering (MC-SE shrinks ~2x) at similar wall time.

### W4-E. Second-pass additions (adversarial "what am I not seeing" sweep, 2026-07-20)

- [x] **W4.16** DONE 2026-07-20. Ledger now 1.71M rows / 3,920 games / 298 slates (regular-season
  2025 + 2026-to-date; 2025 postseason excluded by date window). 20 calibrators refit (slopes
  .87–1.03; biggest level fixes pbb −0.32, b3 −0.22, pout −0.21, sb −0.20); **b3/Triple
  calibrated for the first time** (Tools/5 FAMILY_NAMES gained b3); heads retrained — pout head
  doubled to +0.0044 held-out, **ml head active for the first time** (n=3,920 clears the floor),
  pbb head collapsed to identity (the fresh calibrator absorbs it — correct division of labor).
  Serve smoke PASS (15 games); smoke workbook deleted. Baseline snapshot:
  `artifacts/pre_w416_2026-07-20/`. Post-refresh cross-fit goal board: batter top-10 gaps now
  −0.8..+1.3 pts (HR +0.8, Single +1.3), trust depth 15 almost everywhere — EXCEPT Double
  (depth 1, AUC .553 → W4.10) and pout (Outs>15.5 −5.6, >18.5 −9.7 → W4.12). **Refresh
  cadence (policy): rerun at every wave boundary + ~monthly in-season** (script pattern:
  snapshot → `replay_rows_batch` 2025 window + 2026-to-date → concat → `write_artifact`
  calib_rows → `fit_calibrators --reuse-rows` → `heads --train`; ~45 min GPU). WATCH: pooled
  batter bands 0.80–0.90 run ~1–2 pts hot (n≈1,800) — recheck after W4-B.
- [ ] **W4.17** Time-decay sample weights in A1 training (2015 PAs should not weigh like 2025
  PAs; era features capture regime, not relevance) + tune panel HALF-LIVES per stat family on
  design-eval (90d is a hand-set constant everywhere; K-skill and BABIP-luck decay differently).
- [ ] **W4.18** Seed-bagged A1 (train T1/T3 at 3-5 seeds, average probabilities) — cheap variance
  reduction, reliably +AUC at the top of the sort where ordering is tightest. Subsumed by W4.14
  if the multi-library ensemble ships; do whichever lands first.
- [ ] **W4.19** Confirmed-lineup re-serve habit + measurement: workbooks served off projected
  lineups carry avoidable error in every batter column; grade projected-vs-confirmed serves
  separately (npz product tags already exist) and, on days you can, re-serve once lineups
  confirm. Process, not code (C2 automates the quantification later).
- [ ] **W4.20** Roof-state flag (the B1 survivor — open/closed Condition into park/weather
  features) + verify per-batter platoon-split SKILL panels exist (platoon matchup features are
  in; per-batter EB-shrunk split skill may not be).
- [ ] **W4.21 (research)** Within-slate ranking overlay: a LambdaRank-style reranker (or simple
  isotonic-in-rank blend) trained on within-column ordering only, used to BREAK TIES in display
  order without touching served probabilities — the goal metric is literally a ranking metric,
  and nothing in the stack optimizes ranking directly. Evidence-gated; skip if W4-B lifts AUC
  enough.
- [ ] **W4.22 (user decision required)** Market-informed serving: the scraped odds are the
  strongest public predictor in existence and are currently grading-only BY DESIGN (CLV
  integrity). A dual-output mode — pure-model probabilities for the gate/CLV ledger, an
  odds-blended column (e.g. logit-average with de-vigged close) for the WORKBOOK — would
  measurably raise green-cell rates, at the cost of the workbook no longer being "the model's
  opinion." Philosophy call, not a technical one; do not implement without explicit user choice.

**Rejected on architecture (do NOT transplant):** `xpa_x_*` exposure products, runners-ahead
RBI-chance ctx, SB opportunity chain — the sim generates exposure/lineup/SB-opportunity
structure natively; the old game-grain heads needed them, A1+sim does not. **Expectation
setting:** for low-base columns the top-10 ceiling is bounded by reality (best HR spots ~.28–.32
true p) — "most of top 10 hitting" is reachable on Hit/K/H+R+RBI 2+/Single/2+ TB, not on
HR/SB/2+ Runs; there the goal is raising top-10 hit rate toward its ceiling.

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
