# Roadmap & Status

_Updated 2026-05-29. Method + results: `PROPOSAL.md`; solver design: `admm-spec.md`;
pre-implementation review: `DESIGN_DECISIONS.md`. Code is the source of truth._

## Current state — analysis complete

**Method (validated).** FR-tDBN (`frtdbn/model.py`): fused multi-regime + Student-t
DBN, K=2, arbitrary lag `p`. Two solvers — the headline **consensus-ADMM with a
closed-form fused-lasso prox** (`solver="admm"`) and a smooth-L1 + L-BFGS baseline
(`solver="lbfgs_smooth"`, the DYNOTEARS analogue). Gaussian/Student-t loss;
uniform/adaptive fusion. 88 tests pass (`python -m pytest tests -q`).

**Synthetic (the contribution, ground-truth recovery).** Var-sortability-honest
benchmark (standardized → var-sortability = 0.50). Across change-edges (0–5),
samples (50–200), and tails (ν 3–30), FR-tDBN beats the DYNOTEARS-equivalent by
**+0.19–0.31 change-W AUROC** (gap widest at heavy tails); adaptive fusion is a
documented **negative** ablation.

**Real data (d = 23 Yahoo panel, 2024-05 → 2026-05).** Contemporaneous DAG is
justified (OOS `W≡0` gate: DAG ≫ SVAR; Student-t < Gaussian NLL). But the empirical
hypothesis is **not supported**: neither Δ_W nor Δ_A changes around macro events
beyond a volatility-matched null (global + edge-wise; robust across event type and
±1/2/4h windows). Hourly structure is overwhelmingly contemporaneous — lead-lag is
~20× weaker and negligible beyond 1h (so `p=1`).

**Honest framing.** A real methodological improvement (better Δ-*recovery* on
ground truth, driven mainly by the exact-prox solver), applied to crypto–macro
where it validates the DAG/heavy-tail modeling but yields a carefully-controlled
**null** empirical finding.

## Remaining
1. **NeurIPS write-up (≤ 8 pp)** — method (`PROPOSAL.md` §1–2 + `admm-spec.md`) +
   synthetic recovery (slide-1) + real-data: contemporaneous-dominance, OOS DAG
   gate, the controlled null (W and A), ablations, limits.
2. **(Optional rigor)** λ via BIC, γ via held-out LL; a rank-transform
   (Gaussian-copula) robustness column.

## Changelog (high level)
- Method + var-sortability-honest synthetic benchmark; **exact-prox ADMM solver**
  (≫ smooth-L1) + Student-t (helps) + fusion; adaptive fusion (negative ablation).
- Robustness/eval: OOS full-NLL `W≡0` ablation, block-bootstrap stability,
  vol-matched permutation null (Δ_W and Δ_A) + edge-wise + BH, time-ordered split.
- Data: pivoted Stooq → **Yahoo** (clean UTC); d=23 panel; verified event calendar.
- Real-data run → robust empirical **null**; lag-order ⇒ `p=1`; broadened synthetic
  sweep; baselines vs SVAR/DYNOTEARS; time-resolved figures (`outputs/figures/`).
