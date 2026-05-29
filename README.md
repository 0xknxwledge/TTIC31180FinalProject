# FR-tDBN — Fused-Regime Student-t Dynamic Bayesian Networks

A method-paper extension of DYNOTEARS for **regime-structured, heavy-tailed** time
series: a **fused multi-regime penalty** (the scientific object is the sparse change
graph `Δ = W^event − W^ordinary`) plus a **Student-t likelihood**. Application:
crypto–macro dependency structure around scheduled macro events (CPI/NFP/FOMC).
TTIC 31180 final project.

## Status

Methods validated on synthetic data; empirical (real-data) half in progress. See
`PROPOSAL_v2.md` §0 for goals/results, `TODO.md` for current state + next steps,
`DESIGN_DECISIONS.md` for the pre-implementation review, and
`docs/superpowers/specs/` for the ADMM design spec. `python -m pytest tests -q` →
62 passing.

**Headline (synthetic, change-`W` AUROC, standardized so var-sortability = 0.50):**
exact-prox ADMM ≫ smooth-L1 (~+0.13); Student-t > Gaussian (~+0.10); fusion helps
the weaker estimator. Full tables in `PROPOSAL_v2.md` §0 / `outputs/headline_*`.

## Code map (`frtdbn/`)

| module | role |
|---|---|
| `model.py` | FR-tDBN estimator (`FitConfig`, `FitResult`, `fit_fr_tdbn`): consensus-ADMM exact-prox solver + smooth-L1 baseline; Gaussian/Student-t loss; uniform/adaptive fusion; NOTEARS acyclicity |
| `prox.py` | closed-form K=2 fused-lasso prox (fuse-then-soft-threshold) |
| `synthetic.py` | two-regime DBN generator: realistic sparse `Δ` (add/remove/reweight), imbalanced regime sizes |
| `preprocess.py` | full-sample standardize, robust MAD scales, rolling past-only z-score, rank-Gaussianize |
| `metrics.py` | var-sortability (Reisach), change scores, AUROC, returned-graph diagnostics |
| `benchmark.py` | solver × loss × fusion grid + `summarize_grid` (mean ± se) |
| `evaluation.py` | full-density NLLs, `W≡0` SVAR baseline, out-of-sample `svar_vs_dag_oos` |
| `robustness.py` | restarts + top-k Jaccard, block-bootstrap stability selection, vol-matched permutation null |
| `splitting.py` | time-ordered train/test split, time-block indices |
| `data.py` | ccxt OHLCV fetch/cache; Stooq hourly loader + coverage audit; return panel |

## Scripts

- `scripts/run_synthetic_benchmark.py` — main synthetic benchmark (grid, mean ± se).
- `scripts/run_synthetic_smoke.py` — quick single-fit sanity check.
- `scripts/audit_data_coverage.py` — Stooq hourly coverage/timezone audit.
- `scripts/fetch_crypto_smoke.py` — ccxt crypto fetch smoke.

## Running

```bash
python -m pytest tests -q                                   # tests
python scripts/run_synthetic_benchmark.py --d 12 --n 50 100 --seeds 0 1 2
python scripts/audit_data_coverage.py --symbols spy qqq tlt ief gld btc.v eth.v
```

## Data

Hourly OHLCV from Yahoo Finance (`scripts/fetch_yahoo_hourly.py`), cached locally
under `data/raw/` (gitignored; re-downloadable). Usable span ~2024-05 → 2026-05.
