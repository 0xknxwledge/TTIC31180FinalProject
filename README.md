# FR-tDBN — Fused-Regime Student-t Dynamic Bayesian Networks

A method-paper extension of DYNOTEARS for **regime-structured, heavy-tailed** time
series: a **fused multi-regime penalty** (the scientific object is the sparse change
graph `Δ = W^event − W^ordinary`) and a **Student-t likelihood**, solved by
consensus ADMM with an exact closed-form fused-lasso proximal step. Application:
cross-asset dependency structure around scheduled macro events (CPI/NFP/FOMC) 
with a slight focus on crypto tokens and equities.

TTIC 31180 final project.

## Status — complete (paper written, 100 tests passing)

Docs live in **`docs/`**: `PROPOSAL.md` (method + results), `ROADMAP.md` (status +
remaining), `DESIGN_DECISIONS.md` (pre-implementation review), `admm-spec.md`
(solver design). The write-up is in **`paper/`** (`main.tex` + `references.bib`,
figures in `paper/figures/`, compiled `main.pdf`); rebuild with
`cd paper && latexmk -pdf main.tex`. It uses the NeurIPS LaTeX style for formatting
only (course project, not a submission — the conference notice is suppressed).

**Synthetic (the contribution — ground-truth recovery).** On a var-sortability-controlled
benchmark (standardized → var-sortability = 0.50), across change-edges (0–5), sample
size (50–200), and tail heaviness (ν 3–30), FR-tDBN recovers the change graph
**+0.19–0.31 change-W AUROC** better than the DYNOTEARS-equivalent (smooth-L1 +
Gaussian + independent), the gap widest under heavy tails. The exact-prox ADMM
solver is the dominant lever; Student-t and fusion help secondarily; an adaptive-
fusion variant is a documented **negative** ablation. The advantage survives a
distribution-free rank (Gaussian-copula) transform (+0.14), and principled
`(λ, γ)` selection by held-out likelihood chooses fusion (γ>0) in every seed.

**Real data (d = 23 Yahoo hourly panel, 2024-05 → 2026-05).** The contemporaneous
DAG is justified (out-of-sample `W≡0` gate: DAG ≫ SVAR; Student-t < Gaussian NLL),
but the empirical hypothesis is **not supported**: neither the contemporaneous
(Δ_W) nor the lagged (Δ_A) structure changes around macro events beyond a
volatility-matched null (global + edge-wise; robust across event type and
±1/2/4h windows). Hourly cross-asset structure is overwhelmingly contemporaneous
(lead-lag negligible beyond 1h ⇒ `p=1`). Principled selection corroborates this:
held-out likelihood drives the fusion penalty **γ→0** (BIC to a negligible 0.02) —
data-driven selection finds no change-graph to encode. A carefully-controlled null.

## Code map (`frtdbn/`)

| module | role |
|---|---|
| `model.py` | FR-tDBN estimator (`FitConfig`, `fit_fr_tdbn`): consensus-ADMM exact-prox solver + smooth-L1 baseline; Gaussian/Student-t loss; uniform/adaptive fusion; NOTEARS acyclicity (K=2, arbitrary lag `p`) |
| `prox.py` | closed-form K=2 fused-lasso prox (fuse-then-soft-threshold) |
| `synthetic.py` | two-regime DBN generator: realistic sparse `Δ` (add/remove/reweight), imbalanced regime sizes |
| `preprocess.py` | standardize, robust MAD scales, rolling past-only z-score, rank-Gaussianize |
| `metrics.py` | var-sortability (Reisach), change scores, AUROC, returned-graph diagnostics |
| `benchmark.py` | solver × loss × fusion × change-edge grid + `summarize_grid` (mean ± se) |
| `evaluation.py` | full-density NLLs, `W≡0` SVAR baseline, out-of-sample `svar_vs_dag_oos` |
| `selection.py` | principled `(λ, γ)` selection: held-out NLL + BIC, nonzero-param count |
| `robustness.py` | bootstrap stability, vol-matched permutation null + edge-wise + BH (Δ_W or Δ_A) |
| `splitting.py` | time-ordered train/test split, regime partition, time blocks |
| `data.py` | Yahoo hourly fetch + return panel (legacy: ccxt / Stooq loaders) |
| `panel.py` | `DEFAULT_PANEL` (d=23) + assemble returns → z-score → lag → regime design |
| `events.py` | event-calendar loader (ET→UTC) + regime labeling |
| `viz.py` | regime heatmaps, change network, structure-over-time |

## Running

```bash
python -m pytest tests -q                                  # tests
python scripts/fetch_yahoo_hourly.py --symbols SPY QQQ ... # fetch panel (needs network)
python scripts/run_synthetic_sweep.py                      # synthetic recovery sweep + figure
python scripts/run_real_panel.py                           # OOS W=0 gate + first fit
python scripts/run_real_robustness.py --n-edgewise 100     # stability + permutation/edge-wise null
python scripts/run_selection_robustness.py                 # (λ,γ) selection + rank-transform column
python scripts/make_figures.py                             # paper figures
```

Other scripts: `run_lag_selection.py`, `run_lagged_analysis.py`, `run_sensitivity.py`,
`compare_baselines.py`, `explore_viz.py`, `run_synthetic_benchmark.py`.
Figures land in `outputs/figures/`.

## Data

Hourly OHLCV from Yahoo Finance (`scripts/fetch_yahoo_hourly.py`, via `yfinance` +
`curl_cffi`), cached under `data/raw/` (gitignored; re-downloadable). Usable common
window ~2024-05 → 2026-05. The d=23 panel is `frtdbn.panel.DEFAULT_PANEL`; the
verified macro event calendar is `data/events.csv`.
