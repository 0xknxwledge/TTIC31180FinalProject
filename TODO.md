# Next Steps Plan

## Current State

- Fixed-`nu` fused-regime Student-t DBN MVP exists in `frtdbn/model.py`.
- Synthetic two-regime data generator exists in `frtdbn/synthetic.py`.
- Smoke tests pass with `python -m pytest tests -q`.
- Synthetic smoke script runs end-to-end.
- Crypto data path works through `ccxt`.
- Coinbase spot hourly OHLCV smoke fetch works for `BTC/USD` and `ETH/USD`.
- Binance public endpoints are blocked without VPN from this location with HTTP 451.
- With VPN connected to Switzerland, Binance USDT-margined perp OHLCV smoke fetch works for `BTC/USDT:USDT` and `ETH/USDT:USDT`.

## Guiding MVP

Start narrow and make each layer defensible:

1. Synthetic benchmark first.
2. Binance perp hourly return panel second.
3. Event calendar and RTH alignment third.
4. Real-data graph fits only after the synthetic benchmark and data coverage audit are stable.

Target first real panel: `K=2`, `p=1`, fixed `nu=5`, fixed robust scales, `d≈10-20` crypto-only/perp universe. Expand to macro assets only after the pipeline is mechanically reliable.

## Phase 1: Synthetic Benchmark Hardening

- Add baselines:
  - independent per-regime fit: set `gamma_w=gamma_a=0`
  - fused fit: `gamma_w=gamma_a>0`
  - later: Gaussian loss version
- Run a small repeated benchmark:

```bash
python scripts/run_synthetic_benchmark.py \
  --d 12 \
  --p 1 \
  --n 50 100 \
  --seeds 0 1 2 \
  --max-iter 20 \
  --output outputs/synthetic_benchmark.csv
```

- Add summary script/table:
  - mean AUROC for change-edge recovery by `n` and `gamma`
  - final max acyclicity violation
  - runtime per fit
- Tune sparsity thresholds and penalties until recovered graphs are not obviously dense.

Success criterion: fused model improves or at least matches independent model on change-edge AUROC in small-`n` regimes, with acceptable acyclicity violations.

## Phase 2: Binance Perp Data Audit

- Keep VPN on Switzerland before Binance fetches.
- Fetch a short coverage sample for candidate symbols:

```bash
python scripts/fetch_crypto_smoke.py \
  --exchange binance \
  --market-type future \
  --start 2026-05-01 \
  --end 2026-05-02 \
  --symbols BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT \
  --cache-dir data/raw/binance_perps
```

- Add a coverage-audit script that reports, per symbol:
  - first timestamp
  - last timestamp
  - expected hourly rows
  - observed rows
  - missing-row count
  - close-price null count
- Decide the first core crypto universe from actual coverage, not from the proposal list.

Candidate first core:

```text
BTC, ETH, BNB, XRP, SOL, DOGE, ADA, LINK, LTC, BCH
```

Success criterion: at least 8-10 liquid symbols with clean hourly history over the chosen initial window.

## Phase 3: Event Calendar and Panel Construction

- Build event calendar module for CPI, NFP, and FOMC.
- Start with manual CSV if API/calendar scraping becomes slow.
- Columns:
  - `event_time_utc`
  - `event_type`
  - `event_name`
  - `release_time_et`
- Build label function:
  - ordinary regime
  - event regime within configurable `±h` window
- For crypto-only first pass, use full 24/7 hourly data.
- For cross-asset macro/equity pass, add RTH alignment separately.

Success criterion: a wide return panel plus a same-index regime-label series with no timezone ambiguity.

## Phase 4: First Real-Data Fit

- Fit crypto-only panel first:
  - `K=2`
  - `p=1`
  - fixed `nu=5`
  - fixed robust scales
  - small `d`
- Save outputs:
  - `W_ordinary`
  - `W_event`
  - `A_ordinary`
  - `A_event`
  - `Delta_W`
  - `Delta_A`
  - objective/history metadata
- Add plotting or CSV export for top changed edges.

Success criterion: one reproducible real-data fit that completes in reasonable time and produces inspectable changed-edge rankings.

## Phase 5: Expand Toward Proposal

- Add macro/equity/rates data only after crypto-only fit is stable.
- Start with broad ETFs and rates proxies before single names:
  - `SPY`, `QQQ`, `IWM`
  - `TLT`, `IEF`, `SHY`
  - `GLD`, `USO`
  - `VIX` proxy if available
- Revisit RTH-only versus futures/pre-market alignment for CPI/NFP.
- Add stability selection and permutation null only after one real fit is fast enough.

## Progress (2026-05-28)

See `DESIGN_DECISIONS.md` for the full review, locked decisions, and fix queue.

- **[P0-A done]** Var-sortability honesty. Added `frtdbn/metrics.var_sortability`
  (Reisach et al. 2021) and `frtdbn/preprocess.standardize_columns`; the
  benchmark now standardizes each regime before fitting. Confirmed: the
  generator is var-sortable (raw ≈ 0.90 at ν=5, ≈ 0.71 at ν=3) and standardizing
  drives it to exactly 0.50.
- **[P0-B done]** 2×2 grid. Added a `loss` switch (`student_t` | `gaussian`) to
  the model and `frtdbn/benchmark.run_grid`; the CLI runs
  {Gaussian, Student-t} × {independent, fused} and reports change-W/A AUROC,
  var-sortability, max_h, runtime. 14 tests green.
- **[P1-F done]** Synthetic Δ cleanup. `make_regime_pair` now stabilizes the base
  once and applies a single stable *step* to a sparse delta (additions, removals,
  reweights via `change_types`), so no differential global rescale contaminates
  unchanged edges. Change masks are read off the *realized* `W1−W0` / `A1−A0`.
  19 tests green. (Note: the old contamination was seed-dependent — it only
  fired when the event graph destabilized the VAR — so part of the earlier
  AUROC_A < 0.5 was also just tiny additive A-changes near the noise floor.)
- **Re-run on valid Δ (the picture changed): both contributions now show signal.**
  - ν=5, n∈{50,100}: **Student-t lifts AUROC_W 0.730 → 0.792** (indep) /
    0.734 → 0.797 (fused). Fusion ≈ +0.004. AUROC_A ≈ 0.60 (now valid).
  - ν=3, n∈{20,35}: **fusion lifts AUROC_W 0.559 → 0.613** (Gaussian), matching
    the small-n hypothesis — but Student-t *under*performs Gaussian here
    (0.52 vs 0.56), which is counterintuitive at the heaviest tail.
  - Numbers are modest (0.5–0.8) and from ≤12 fits — need error bars (more seeds)
    and a γ/λ sweep before claiming anything. Contributions don't clearly *stack*.

## Open threads

- **Student-t under-performs at ν=3** (should be its best case). Suspects: pooled
  MAD scales (`_robust_scales`) noisy at small n; optimizer non-convexity; or just
  noise. Investigate with error bars + per-regime scales.

## ADMM exact-prox solver (Option C) — built and it works decisively

Spec: `docs/superpowers/specs/2026-05-28-admm-fused-fr-tdbn-design.md`. Added
`frtdbn/prox.py` (closed-form K=2 fused prox), an `admm` branch in `model.py`
(`FitConfig.solver`), and `solver` as a benchmark axis. 32 tests green.

**Headline change-W AUROC, mean (se) over 24 fits/cell, standardized (varsort=0.50).**
`summarize_grid` + `outputs/headline_nu{5,3}_summary.csv`.

ν=5, n∈{50,100}:

| solver | loss | indep | fused |
|---|---|---|---|
| admm | gaussian | 0.714 (.030) | 0.794 (.032) |
| admm | student_t | 0.838 (.027) | **0.853 (.031)** |
| smooth | gaussian | 0.584 | 0.582 |
| smooth | student_t | 0.617 | 0.626 |

ν=3, n∈{20,35}:

| solver | loss | indep | fused |
|---|---|---|---|
| admm | gaussian | 0.565 | 0.626 |
| admm | student_t | **0.666** | 0.662 |
| smooth | (both) | ~0.53 | ~0.55 |

Robust claims (≈2+ se): **exact-prox ADMM ≫ smooth-L1** (~+0.13 everywhere);
**Student-t helps under ADMM** (gaussian→student_t +0.10–0.12); the ν=3 Student-t
anomaly was a smooth-solver artifact. `max_h ≈ 4e-3` (no ADMM divergence).

**Nuance (matters for D):** fusion's *marginal* benefit on top of Student-t is
within noise here. A regime-imbalance probe (ordinary n=200 fixed, admm gaussian)
shows why — uniform-L1 fusion **hurts** when the event regime is small:

| event n | indep | fused (γ=.08) | gain |
|---|---|---|---|
| 20 | 0.738 | 0.682 | **−0.056** |
| 50 | 0.807 | 0.801 | −0.006 |
| 200 | 0.854 | 0.917 | **+0.063** |

Uniform L1 shrinks the *true* Δ along with the spurious differences; when the
event sample is small (the realistic case) the true changes are weak and get
shrunk below detection. **This is exactly the failure mode Option D targets**
(adaptive / non-convex penalty: shrink small diffs hard, leave large true diffs
unbiased). Two synthetic-design facts exposed: (a) `make_regime_pair` gives both
regimes *equal* n — need separate `n_ordinary`/`n_event`; (b) the realistic
small-event regime is where the method must prove itself.

## Option D (adaptive fused lasso) — built, tested, and it FAILS (negative ablation)

Added imbalanced regime sizes (`make_regime_pair(n_event=...)`), per-edge prox
weights, `FitConfig.fusion="adaptive"` + `adaptive_pilot ∈ {uniform, independent}`
(`adaptive_eps`). 42 tests green. Weights `γ_ij = γ₀/(|Δ̂_ij| + ε)`.

**Verdict: adaptive reweighting consistently *hurts* on every metric** (ordinary
n=200, small event n, admm gaussian, 8 seeds):

| event n | metric | uniform | adaptive (best variant) |
|---|---|---|---|
| 20 | AUROC_W | 0.656 | 0.51–0.55 |
| 20 | F1(Δ support) | 0.222 | 0.00–0.04 |
| 35 | F1(Δ support) | 0.253 | 0.04–0.09 |

**Why:** adaptive-lasso assumes a root-n *unbiased* pilot. Here the only pilots
available (uniform-fused, or independent at small event-n) already shrink the
true Δ toward 0, so `γ_ij = γ₀/(|Δ̂|+ε)` puts a *large* weight on the true-change
edges → stage 2 zeros them → Δ collapses to ~empty (P=R=0). Reweighting amplifies
the under-estimation instead of correcting it. Kept in the codebase as a
documented negative ablation; **`fusion="uniform"` (Option C) stays the headline.**

**Side finding:** even uniform fusion has low *precision* at small event-n
(F1≈0.22, P≈0.17) — exact support recovery is genuinely hard here. Ranking
(AUROC) is the honest lens, and this motivates **stability selection** (bootstrap
edge frequencies) over trusting a single fit's support on real data.

## Immediate Next Work Session

1. **Lock Option C as the headline** (uniform exact-prox ADMM fusion + Student-t).
   Optionally: clean slide-1 figure (already have `*_summary.csv`).
2. **AUROC_A wrinkle** (A-fusion hurts under uniform, 0.660→0.616) — decouple
   γ_A from γ_W, or just report it.
3. **Crypto-macro data pipeline** (Massive/Alpaca, `d≈15–20`) — the empirical half.
4. Real-data eval leans on **stability selection + permutation null**, not
   single-fit support (per the side finding above).

## Data Pipeline Questions To Resolve Before Planning

The first real analysis panel should be crypto-macro, not crypto-only. Crypto-only
is still useful as a fetch/model smoke test, but the empirical claim is about
spillovers linking crypto, equities, rates, FX, and volatility.

Open questions to settle before building the full pipeline:

1. **Near-24h macro coverage.** Which macro instruments are actually available
   hourly from Massive/Alpaca? CPI/NFP releases occur at 08:30 ET, so cash ETFs
   only capture post-open reaction.
2. **Futures vs ETFs.** Do we use index/rate futures to capture premarket macro
   reactions, or accept an ETF panel whose event window starts at the cash open?
3. **Continuous futures construction.** If using futures, how do we roll/stitch
   contracts and avoid artificial return jumps?
4. **Event calendar source.** Manual CSV vs API/scrape; must include exact UTC
   timestamps for CPI, NFP, FOMC statement, and possibly FOMC press conference.
5. **Hourly bar convention.** For an 08:30 ET release, which hourly bar receives
   the event label, and are returns close-to-close, open-to-close, or event-time
   aligned?
6. **Standardization.** Real data should use rolling, past-only z-scores; the
   synthetic full-sample standardization is only for benchmark fairness.
7. **Missingness policy.** Drop variables with incomplete history; forward-fill
   prices only before computing returns; never forward-fill returns.
8. **First panel size.** Pick `d≈15–20` from actual coverage and data quality
   before coding the full event-label pipeline.
9. **Crypto data source.** Binance perps work with Switzerland VPN; decide
   fallback behavior when VPN/API access fails.

## Robustness Infrastructure Completed

- Fit results now include returned-graph diagnostics:
  - `h_returned_max`
  - per-regime `h_returned_k`
  - `W`/`A` nonzero counts
  - `Delta_W`/`Delta_A` nonzero counts
  - final primal/dual residuals when available
- Added full held-out likelihood evaluation with constants:
  - Student-t full NLL
  - Gaussian full NLL
- Added `W≡0` structural-VAR-only baseline (`fit_svar_only`) for early
  contemporaneous-DAG sanity checks.
- Added restart robustness helpers:
  - multiple random restarts
  - top-k changed-edge Jaccard stability
- Added block-bootstrap stability selection helpers for edge-frequency reporting.
- Added volatility-matched permutation-null helper for `||Delta_W||_1`.
- Added real-data preprocessing robustness helpers:
  - rolling past-only z-scores
  - rank/Gaussian-copula transform
  - MAD robust scales
- Benchmark CLI now supports:
  - multiple `nu` values
  - multiple `p` values
  - separate `gamma_w` and `gamma_a`

Validation: `python -m pytest tests -q` passes with 54 tests.

## Update 2026-05-29 — robustness review, OOS fix, and Stooq coverage audit (62 tests)

### Robustness work reviewed — solid; one correctness fix applied
Verified: full-constant NLLs correct; diagnostics computed on the *returned sparse*
graph; block bootstrap aligns target+lag rows; vol-matched permutation preserves
label counts. The team's cautions (single-run smoke; null underpowered, use ≥20–100
perms; dense support → report frequencies/rankings; tune γ_W/γ_A separately) are all
valid. **Main gap fixed: the W≡0 / held-out-NLL comparison was in-sample**, which
favors the DAG mechanically (it has more parameters). Added genuine out-of-sample:
- `frtdbn/splitting.py`: `train_test_split_regimes` (time-ordered) + `time_block_indices`.
- `evaluation.heldout_nll` → renamed **`full_nll`**; added **`svar_vs_dag_oos`**
  (fit FR-tDBN + SVAR on train, compare full NLL on a disjoint test split with
  train-derived scales). Re-run the W≡0 ablation through this before claiming W helps.

### Coverage audit (first pipeline deliverable) — `scripts/audit_data_coverage.py`
`frtdbn/data.{load_stooq_txt, find_stooq_files, audit_symbol_coverage}`.
Candidate-panel result (`outputs/coverage_audit.csv`):

| block | tickers found | span | hours (tz clue) |
|---|---|---|---|
| equity ETF | spy qqq iwm | 2024-05-10 → 2026-05-28 | 15–22 |
| rates ETF | tlt ief shy | same | 15–22 |
| commodity | gld | same | 15–22 |
| crypto | btc.v eth.v sol.v | same, 24h | 1–23 |
| FX | usdeur | same, 24h | 1–23 |
| rates yield | 10yusy.b 2yusy.b | same | 4–23 |
| vol | `^vix` **not found** | — | — |

**Two big constraints the audit exposed:**
1. **Stooq hourly history is ~2 years (2024-05 → 2026-05), not 5.** Revise the
   window to ~2y (≈24 CPI, 16 FOMC, 24 NFP → ~64 events). Silver lining: much less
   non-stationarity to worry about. If 5y is required, supplement via Massive/Alpaca.
2. **Timestamps are NOT ET** — ETF bars at hours 15–22 imply CET/UTC-ish (US RTH
   in ET would be 9–16). Resolve the tz (likely CET = UTC+1) before mapping the
   08:30 ET CPI release to a bar. Crypto/FX are 24h (1–23) as expected; closes are
   clean (0 nonpositive). `^vix` index isn't in the dump — use a VIX ETF proxy
   (VIXY/UVXY) or find the Stooq code.

### Resolved data-pipeline questions (locked guidance)
1. Coverage audit first ✓. 2. Futures for event windows, else caveat ETF windows as
"post-open reaction." 3. Stitch **returns** across futures rolls, not prices.
4. Hand-curated event CSV in UTC; separate FOMC statement (14:00 ET) / presser
(14:30 ET). 5. Log returns, close-to-close on a fixed hourly grid, window relative
to the release timestamp; report ±1/2/4h sensitivity. 6. `rolling_zscore_past`,
applied before regime split, never per-regime. 7. Drop sparse symbols; ffill
*prices* over short gaps then compute returns; never ffill returns. 8. d≈12–15 from
actual coverage. 9. Binance → Coinbase/Alpaca fallback, log source per symbol.
**(+) Split:** train 2024-05→2025-12, test 2026-01→2026-05 (time-ordered ~80/20).
**(+) Non-stationarity:** report per-sub-period Δ stability via `time_block_indices`.

### Next
- Re-run W≡0 ablation via `svar_vs_dag_oos` on the first real panel (early sanity).
- Build the panel loader: Stooq → tz-resolved hourly grid → log returns → rolling
  z-score → regime labels from the event CSV; pick d≈12–15 from the audit.
