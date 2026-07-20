# Staking Layer — Pre-Registered Design

Declared: 2026-07-19. Status: **DESIGN ONLY — paper trading.**

> **2026-07-20 note:** the pre-registered forward-test regime this document's
> activation clause referenced was retired (forwardtest.py deleted; standing
> evaluation is `Model/walkforward.py`, rolling-origin). The §7 promotion
> criteria are therefore unanchored — if staking is ever activated, re-anchor
> them first (e.g., to sustained Tools/5 family tiers or a Sunday-gate
> equivalent). Until then nothing in this document authorizes real money.

---

## 1. Scope

Staking consumes the engine's served outputs; it never feeds back into them.
Inputs per candidate bet:

- `p_model` — the family-calibrated, head-adjusted probability from
  `predict.py` (clamped inside (0, 1); Platt calibrators, never isotonic).
- `price` — the actual American price offered at capture time.
- `p_close` — the de-vigged sharp fair probability (`odds.sharp_fair`:
  Pinnacle-preferred, median no-vig fallback), taken at the latest capture
  before first pitch.

Market families are exactly the `MKT_FAM` keys in `predict.py`:
`h hr tb r rbi bb sb b1 b2 b3 hrr bk pk pout pha pbb per ml tot tt`.

## 2. Eligibility (which families may be staked)

A family is stakeable only while it holds a **PASS** from
`evaluate.market_gate` — n ≥ 800 graded prices, Benjamini-Hochberg p < 0.05
across families, and bootstrap CI5 > 0 — computed on data **outside** any
window being graded. During the forward-test window, eligibility is fixed at
whatever the gate said on data through 2026-07-19; it is re-evaluated only
after grading.

Families at INSUFFICIENT n (e.g. `per` at n=159) are not stakeable and are
not paper-traded as if stakeable; they accrue gate sample only.

## 3. Bet selection

A candidate becomes a paper bet iff **all** of:

1. Family is eligible (§2).
2. The market has a two-sided de-vigged close available (no one-sided
   prices — same rule serving already enforces).
3. `edge = p_bet − p_market_implied` ≥ **0.03**, where `p_market_implied`
   is the vig-included implied probability of the offered price. The 0.03
   floor matches the gate's edge-bucket dead zone (−0.03, 0.03), inside
   which no realized edge has been demonstrated.
4. Expected value at the offered price is positive after §4 shrinkage:
   `EV = p_bet·(dec − 1) − (1 − p_bet) > 0`, `dec` = decimal payout.

Never both sides of the same market. One bet per (player, market, line) per
game, best available captured price.

## 4. Probability shrinkage

Raw `p_model` is not staked. The staked probability shrinks toward the
sharp close:

    p_bet = 0.5 · p_model + 0.5 · p_close

The 0.5 weight is pre-registered and deliberately conservative: the CLV gate
demonstrates rank/log-loss skill vs the close, not that the full magnitude of
(p_model − p_close) is real. After the forward test grades, the weight may be
refit per family from the gate's edge-bucket realization tables (declared
here so the refit is not a post-hoc choice).

## 5. Sizing — fractional Kelly

Per-bet fraction of bankroll:

    f = λ · (p_bet·dec − 1) / (dec − 1),   λ = 0.25 (quarter-Kelly)

subject to the caps in §6. Quarter-Kelly is pre-registered as protection
against residual calibration error and intra-slate correlation; λ may not be
raised before grading. Bankroll is marked to market daily; f applies to the
current bankroll (proportional staking).

## 6. Exposure caps (all pre-registered)

- **Per bet:** min(f, 1.0% of bankroll).
- **Per game:** total stake across all legs in one game ≤ 2.0% of bankroll.
  All props in a game are correlated through the game environment (the sim's
  own latent structure says so); the cap binds before any per-leg pruning,
  dropping the lowest-EV legs first.
- **Per slate (day):** total new stake ≤ 10% of bankroll.
- **Per family per slate:** ≤ 4% of bankroll, so one miscalibrated family
  cannot dominate a day.

## 7. Promotion to real money

All of the following, none waivable:

1. Forward test grades and the family is PASS under the pre-registered
   protocol (BH p < 0.05, CI5 > 0, n ≥ 800).
2. The family's paper-trade ledger over the window shows non-negative
   realized ROI **and** positive mean CLV (beat the close on average).
3. The projected-vs-confirmed lineup quantification has landed, and the
   family's paper record is not driven by projected-lineup slates.

Real-money λ starts at 0.25 and may only be raised after a further full
month of live PASS-grade results, one step at a time (0.25 → 0.35 → 0.50 max).

## 8. Demotion and kill-switches

- **Gate demotion:** monthly `market_gate` re-check on rolling data; a family
  that drops to NO-EDGE goes back to paper immediately.
- **Drawdown kill-switch:** live staking halts entirely at 20% peak-to-trough
  bankroll drawdown, pending a full gate + calibration review. The halt is
  mechanical, not judgment-based.
- **Data-integrity halt:** any day the odds capture, slate archive, or
  grading pipeline fails its health check, no new stakes that slate.

## 9. Ledger

Every paper (and later live) bet is appended to
`Model/artifacts/staking_ledger.csv`, one row per bet:

    Date, GameId, PlayerId, Market, Line, Side, Book, PriceAmerican,
    CapturedAt, p_model, p_close, p_bet, edge, EV, f_kelly, stake_frac,
    stake_capped_by, CloseAmerican, CLV, Outcome, PnL_units

Rows are written at bet time; `CloseAmerican/CLV/Outcome/PnL` fill at
grading. The ledger is append-only — corrections are new rows with a
`void` marker, never edits.

## 10. What this document forbids

- Changing λ, the 0.03 edge floor, the 0.5 shrinkage weight, or any cap in
  §6 before the forward test grades.
- Staking (even paper) a family the gate has not PASSed.
- Using window results to select which families "would have" been staked.
  Eligibility was fixed on 2026-07-19 data, before the window.
