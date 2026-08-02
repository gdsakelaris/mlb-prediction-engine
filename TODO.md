# Open Items

Updated 2026-07-22 (post-W5.1: ensemble avenue closed, top-1 promoted to first-class goal,
served goal tracker live). Trimmed 2026-07-20 (second trim, after W4-A/W4.16/W4-B/W4.12/W4.13
shipped — evidence and verdicts live in Logs/*_2026-07-20.log, the goal-metric memory, and git
history of this file).

**Ship discipline:** paired replay A/B (`evaluate --ab`) + `pytest` green before a change
serves — that is the whole per-ship gate. `Model/walkforward.py` (rolling-origin folds
2022–2026) runs at wave boundaries / ~monthly (with the calibration refresh) and for
STRUCTURAL changes (new model class, loss, sampling scheme, hyperparameter retune), not per
shipped feature (user policy 2026-07-20). Env gates `PEN_WAVE3` / `PEN_CHOICE` in predict.py
reproduce pre-pen-wave behavior ("0") for paired A/Bs. **Goal metric** (top-10/slate precision + >50% monotone
reliability + trust depth, and since 2026-07-21 **top-1** as a first-class goal — gate ship
decisions on top-5/10, watch/report top-1 since it's 1 obs/market/slate) is measured in
Tools/5's Goal Board/Reliability sheets, the `--ab` goal section (t1/t3/t10), and the SERVED
tracker "Tools/6) Goal Tracker.py" → artifacts/served_goal_tracker.csv (run after each Tools/4
grade; scopes all + confirmed-lineup). **Calibration refresh cadence:** rerun the extended
replay → `--fit-calibrators --reuse-rows` → `heads --train` at every wave boundary + ~monthly
in-season (~45 min GPU; procedure in the goal-metric memory).

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
- [ ] **Batter top-1 weakness (served)** — first Tools/6 board (10 slates 07-07..07-20):
  pitcher top picks elite (K>4.5 / Outs>14.5 / K / K>3.5 all 10/10 at t1, med margin12
  0.15–0.34) but batter top slots weak (best 0.7, 2+ Hits 0.1; margins 0.02–0.09) — top-pick
  trust = pitcher markets, confirming the clear-leader asymmetry (serve smoke: 10 pitcher /
  0 batter clear-leader flags). Keep the tracker current after each grade; this series is the
  primary input to the late-Aug W4.21 decision.

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

(W5.1 heterogeneous ensemble REJECTED 2026-07-21, Logs/w5ens_cycle_2026-07-21.log: 3-seed
LGBM bag blended into the XGB bag via F.BlendClf — paired A/B on 1.71M rows/298 slates was an
aggregate TIE +0.00004, 0/20 families significant after BH, A1 composite dead even, and
goal-board top-1 NET NEGATIVE ≈ −0.2 pts/market (LGBM, weaker solo, dilutes the top slot);
no-harm ties ship only with ranking wins, so rolled back from artifacts/pre_w5ens_2026-07-21/.
Plumbing kept — train.py A1_LGB_SEEDS re-arms it. CatBoost probe same day
(Logs/cb_probe_2026-07-21.log): fails the pre-registered solo-parity gate by 3x (+0.00145
composite). **ENSEMBLE AVENUE CLOSED** — two data points; NN-as-member closed with it; re-open
only with a member at solo parity. KEPT from the cycle: --ab goal board t1/t3 columns,
predict.py clear-leader border flag (CLEAR_LEADER_LOGIT=0.25), Tools/6 served tracker +
10-slate backfill. Ledgers: rows_prew5.parquet = shipped-stack baseline for the next A/B;
rows_w5ens_reject.parquet = rejected blend rows.)

- [ ] **W4.21 (research — DEFERRED one measurement cycle, decision 2026-07-20)** Within-slate
  ranking overlay (LambdaRank-style display reranker, served probabilities untouched).
  Deferred because: (a) the measured top-10 shortfall comes from the 4k-sim goal board, which
  jitters ±1–1.5 pts from MC re-ranking noise, and the 100k-sim batched serve (W4.15) just
  removed exactly that noise from production — its effect is unmeasured; (b) ranking by
  calibrated p is already top-k optimal given the model's information, and every discrimination
  lever this wave moved AUC ≤ +0.002, so the rankable residual looks thin; (c) a display rank
  that disagrees with stated p breaks the workbook's sort=probability invariant; (d) 298
  slates is thin for LambdaRank. **Re-evaluate at the next calibration refresh (~late Aug):**
  grade SERVED workbooks' top-10 precision on confirmed-lineup slates (W4.19 split) + rerun
  the goal board; take up W4.21 only if Hit / K / Single / H+R+RBI 2+ / 2+ TB — the columns
  where "most of top 10" is honestly reachable — still sit short of their ceilings.
  (2026-07-21 update: the served-precision half of that evidence now accrues automatically in
  artifacts/served_goal_tracker.csv — Tools/6, confirmed scope; first board = batter top-1
  weakness watch above. Top-1 promotion raises the stakes of this decision but the deferral
  logic stands.)
- [x] **W4.22 SHIPPED 2026-07-22 (user approved same day, "blend everything possible")** —
  displayed workbook probabilities now blend 50/50 in logit space with the market fair price
  wherever one is captured for that exact (player/team, market, line): all three grid sheets
  (batter cols, pitcher ladders, Win Prob/totals/team totals). Pure model preserved by
  construction in the Bets sheet EV, the CLV gate, and every replay ledger (`P.mkt_blend=False`
  in evaluate.py replay/gate paths; `MKT_BLEND=0` serves pure model for paired comparisons;
  `MKT_BLEND_W=0.5` in predict.py). Fair = `sharp_fair` two-sided, else `R_ONESIDED=0.9336`
  haircut on the one-sided consensus (measured on 25,394 two-sided groups, stable 0.930–0.945
  across all 15 markets — books post hits/RBI boards Over-only some days; re-measure at
  refreshes). Ladder guards re-run post-blend. Evidence basis: market benchmark 2026-07-22
  (11 served slates, same-universe: market out-ranks model on batter top-K ~3–7 pts at
  t3/t5/t10 at BOTH open and close, Hit t1 .82 vs .46; memory market-benchmark-2026-07-22;
  rerun pattern scratchpad market_bench.py). Ship gate: pytest 30/30 + paired smoke serve on
  the 07-22 slate (blend-off vs blend-on, same seed: Bets bit-identical, pure cols identical,
  4,410 cells blended, 100% moved toward market, exact 50/50 logit blend, all ladders
  monotone). Replay A/B is blind to display-layer changes by design — the live judge is
  Tools/6: compare served top-K on post-07-22 slates vs the pre-blend series at the late-Aug
  review. Weight/haircut re-tune (per-family w once the odds archive supports it) rides the
  calibration-refresh cadence.
- [x] **W5.2 batter-flank wave SHIPPED 2026-07-27** (PA-count audit -> slot-aware participation
  hazard + bag-sigma shrink instrument + calibrator/heads refresh). (a) PA AUDIT (63 slates,
  15,048 starter-games, scratchpad pa_audit.py): the slot-blind hazard over-fed bottom-order
  starters their 4th PA (+6.7/+7.4/+9.0 pts P(4+PA) slots 7/8/9, slots 1-4 clean) — so
  top-of-sort batter overconfidence is NOT PA-driven (top picks are slots 1-5), but cumulative
  props of bottom-order bats were inflated. FIX: participation.parquet gained a 9-way zero-based
  slot axis (features build_participation; EB toward (k,slot) marginal; predict densify
  [6,4,3,2,2,2,9]; both sims index it, 6-dim legacy accepted). Re-audit: slot-gradient range
  [-1.6,+9.0] -> [+0.8,+5.0] pts, slots 8/9 +2.9/+2.1, P(5+PA) clean everywhere. RESIDUAL: a
  uniform ~+2pt P(4+PA) excess across ALL slots — RESOLVED by W5.3 (2026-08-02, below): it
  was a 2026 substitution regime shift, not extra team-PA. (b) W5.2 SHRINK: full plumbing shipped INERT — per-batter bag-member
  sigma at both prep choke points (_class_vecs(sigma=True), _agg_bat_sigma vs opposing starter),
  logit shrink in game_frame pre-calibration (display cols only, Bets untouched), SHRINK gate +
  SHRINK_LAM/SHRINK_ANCHOR constants, sigma recording in replay (collect_sigma). SWEEP VERDICT
  (rows_sigma_base/sig.parquet, 292 dev + 39 holdout slates, prereg objective d_t3+d_t10 weak
  set): lambda stays 0 — bias-vs-sigma gradient real but tiny (+0.2->+0.5pt/quintile; 3-seed
  bag sigma only 0.01-0.04 logit), dev gains noise, holdout tie [-0.38,+0.44] t10. Re-arm only
  with a diversity-bagged member (data/feature subsampling). want_sigma auto-off at lambda=0.
  (c) REFRESH: fresh 4k-sim GPU replay 2025-03-18..2026-07-26 (1.77M rows, 331 slates, 30 min)
  -> calib_rows (old snapshotted calib_rows_pre_w52) -> --fit-calibrators --reuse-rows --force
  (full-window precedent, 21 fams, tb slope 0.975 cooling) -> heads --train (b3/pha inactive).
  (d) A/B rows_prew5 vs new (298 shared slates): ALL +0.00039 [+0.00024,+0.00053] B BETTER;
  7 fams significant after BH — b1/hrr/h/bk/rbi/r/tb, ALL BATTER — the targeted flank; WATCH
  ml -0.00137 and pha -0.00081 (A better; retrain-drift vs hazard attribution unknown).
  (e) Fix-4 scope RESOLVED: in-season batter-vs-pitch-class already shipped in W4.15
  (panel_bat_pitchmet x arsenal usage = mx_class_whiff/dmg/ts/ride); xBA-grain T3 rebuild
  stays a late-Aug research item. Suite 236->242 fast + slow 3 green (Tests/test_shrink.py);
  rows_prew52.parquet = frozen baseline for the next A/B. Tools/5 board regenerated 07-27.
- [x] **W5.3 participation era anchor SHIPPED 2026-08-02** (closes the W5.2 turnover residual).
  (a) DIAGNOSIS (700-game paired replay Apr-Jul 2026, 1,414 team-games, scratchpad
  pa_turnover_diag.py, Logs/pa_turnover_diag_big_2026-08-02.log): team PA EXACT (+0.04±0.12),
  runs exact — but starters +0.287±0.115 PA/g, bench −0.250±0.046. Bench-PA/team-game by
  season: 2022 1.27, 2023-25 ~1.41, 2026 1.657 (+17%); the sim's 1.404 reproduced its 2022+
  pooled training era exactly. The "0.3 extra team-PA" framing was wrong — it was starter-vs-
  bench ALLOCATION (substitution under-firing vs the current season), not game length.
  (b) FIX: build_participation writes stores/participation_anchor.json — one EB-shrunk logit
  shift δ = w·(logit(act_cur) − logit(pred_cur)), w = n_cur/(n_cur+30k); 2026-08-02 δ=+0.1029
  (n 124,317, act 2.98% vs grid 2.63%). predict.py applies δ to the dense lookup at LOAD time
  (covers both engines and the (k,slot) ev/n marginal backfills). Self-tightens at every
  store rebuild. (c) VERIFY (same 700 games): bench deficit −0.250 → −0.117±0.046, starter
  excess +0.287 → +0.141, P(4+PA) gap ~halved per slot, team PA/runs still exact. Residual
  −0.117 = deliberate EB conservatism — WATCH, do not retune N0 to the diagnostic. Extras
  rate sim 9.6% vs actual 8.3% — separate small watch. (d) SHIP GATE: paired A/B rows_prew53
  vs rows_w53 (690,766 rows, 121 slates 2026-03-25..07-26, both 2026-only): ALL 20 families
  TIE after BH (r/pha/ml raw-positive, worst b2 −0.00012 n.s.); goal board t1 broadly
  positive, 2+ Hits +2.5/+2.5/+2.0 pts t1/t3/t10; suite 242 green. Shipped on no-harm-tie +
  generative correctness (contact-tree precedent). rows_w53.parquet basis = next A/B baseline.
  Same-day context: calibrators refreshed FRESH 2026-only (guard-compliant, fit_end 07-26
  preserved for gate OOS); heads leaner (h/hr/pha/pout/r/rbi now identity); walkforward +
  skill ledger + prop_bench + tracker all regenerated post-W5.2.

- [ ] **B2. Forecast-error weather sampling** — sample temp/wind per sim from the historical
  forecast-error distribution. Blocked: `forecast_error.json` has n=46, `sufficient: false`
  (checked 2026-07-20). The roof-state flag shipped separately in W4.20 (`roof_open`).
- [ ] **C2. Lineup-uncertainty quantification** — probability-weighted lineup distribution before
  confirmation. Blocked on slate-archive accrual, **revisit ~late Aug 2026**. The retired serve
  archiver (built/retired 2026-07-20; copy in that session's scratchpad retired_tools/) was the
  data-collection half — bring it back if this runs. (Its old "Tools/6" slot is now the Goal
  Tracker — unrelated tool.)

## Annual retune slot only (do not do mid-season)

- [ ] **B7. Per-component Optuna + library bake-off** — separate studies for T1/T3/A2/hazard/SB.
  Wave 1 found a flat plateau (+0.0006); low expected value. The library bake-off half is now
  largely RESOLVED by W5.1 (2026-07-21): LGBM blend rejected by full A/B (top-1 net negative),
  CatBoost fails solo parity 3x the pre-registered gate — ensemble avenue closed; what remains
  for this slot is per-component hyperparam studies only, plus re-opening the member question
  ONLY if some library reaches solo parity (probe pattern: scratchpad cb_probe.py).
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
