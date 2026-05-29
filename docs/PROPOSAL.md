# Fused-Regime Student-t Dynamic Bayesian Networks for Crypto–Macro Spillover Structure

**John Beecher · TTIC 31180 (Probabilistic Graphical Models) · Final Project (living document)**

This supersedes `BeecherJohn_TTIC31180_ProjectProposal.pdf`. The contribution lives
on the **structure-learning** side (a method paper with a financial application);
the HRP idea was dropped. Companion docs (all in `docs/`): `ROADMAP.md` (status +
remaining work), `DESIGN_DECISIONS.md` (pre-implementation review + how each item
resolved), `admm-spec.md` (detailed solver design). **The code is the source of
truth; this document is reconciled to it (2026-05-29).**

---

## 0. Project status & results (updated 2026-05-29)

### Goals

1. **Methodological (primary, the gradable contribution).** Extend DYNOTEARS
   (Pamfil et al. 2020) in two ways for regime-structured, heavy-tailed data:
   a **fused multi-regime penalty** (most structure is shared; the scientific
   object is the sparse *change graph* `Δ = W^event − W^ordinary`) and a
   **Student-t likelihood** (heavy-tailed financial residuals). Prove the value
   on ground-truth synthetic data.
2. **Empirical (secondary).** On a small but complete crypto–macro panel, ask
   whether the contemporaneous + lagged dependency structure changes in
   measurable, stable ways around scheduled macro events (CPI/NFP/FOMC), and
   which edges flip.

### Completed

- **FR-tDBN estimator** (`frtdbn/model.py`): fixed-`ν` Student-t / Gaussian loss,
  per-regime intra-slice DAG `W^k` + inter-slice `A^k`, fused + sparse penalties,
  NOTEARS acyclicity. **Two solvers**: a smooth-L1 + L-BFGS baseline
  (`lbfgs_smooth`) and the headline **consensus-ADMM solver with a closed-form
  fused-lasso prox** (`admm`) — see §2.
- **Var-sortability-honest synthetic benchmark** (`frtdbn/{synthetic,preprocess,
  metrics,benchmark}.py`): realistic sparse `Δ` (additions/removals/reweights),
  per-regime standardization, imbalanced regime sizes, and a
  solver × loss × fusion grid with change-edge AUROC and mean ± se.
- **Robustness & evaluation infrastructure** (`frtdbn/{evaluation,robustness,
  splitting}.py`): full-density NLLs; time-ordered train/test split; an
  **out-of-sample `W≡0` SVAR ablation** (`svar_vs_dag_oos`); block-bootstrap
  stability selection; volatility-matched permutation null; restart stability;
  rolling past-only z-scores + rank-Gaussianize for real data.
- **Real data pipeline + analysis** (`frtdbn/{data,panel,events}.py`, `scripts/`):
  a **d = 23 Yahoo Finance hourly panel** (2024-05 → 2026-05, UTC), verified
  CPI/NFP/FOMC event calendar, OOS `W≡0` ablation, stability selection,
  global + edge-wise permutation nulls (on Δ_W **and** Δ_A), sensitivity,
  baselines, and time-resolved visualizations — see §4–5.
- **88 tests pass** (`python -m pytest tests -q`).

### Headline results (synthetic, change-`W` AUROC, mean (se) over 24 fits/cell, standardized so var-sortability = 0.50)

ν = 5, n ∈ {50, 100}:

| solver | loss | independent | fused |
|---|---|---|---|
| **admm** | gaussian | 0.714 (.030) | 0.794 (.032) |
| **admm** | student-t | 0.838 (.027) | **0.853 (.031)** |
| smooth | gaussian | 0.584 | 0.582 |
| smooth | student-t | 0.617 | 0.626 |

ν = 3, n ∈ {20, 35}:

| solver | loss | independent | fused |
|---|---|---|---|
| **admm** | gaussian | 0.565 | 0.626 |
| **admm** | student-t | **0.666** | 0.662 |
| smooth | (both) | ~0.53 | ~0.55 |

**Findings (robust, ≥ ~2 se):**
- **Exact-prox ADMM ≫ smooth-L1** everywhere (+0.13); even ADMM-independent beats
  smooth-fused. The solver is the dominant lever.
- **Student-t helps** under ADMM (+0.10–0.12 over Gaussian) in both regimes; an
  earlier "Student-t hurts at ν=3" reading was a smooth-solver artifact.
- **Fusion helps the weaker estimator** (Gaussian; +0.06–0.08) and the
  balanced-sample regime; its *marginal* gain on top of Student-t is within noise.
- The full **FR-tDBN (admm + student-t + fused)** is the best cell in both regimes.
- **Negative ablations (kept honestly):** (i) smoothed-L1 fusion (`√(x²+ε)`) never
  produces exact zeros and so does *not* select `Δ` — fusion gains ≈ 0 under it;
  (ii) an **adaptive fused-lasso** reweighting (Option D) was implemented and
  *fails* — at small event-`n` every available pilot already shrinks the true `Δ`,
  so `γ_ij = γ₀/(|Δ̂_ij|+ε)` over-penalizes the true edges and collapses `Δ` to
  near-empty. Adaptive-lasso's unbiased-pilot assumption is violated here.
- **Practical caveat:** exact *support* recovery (F1) is hard at small event-`n`
  even for uniform fusion (F1 ≈ 0.22, precision ≈ 0.17); ranking (AUROC) is the
  honest lens, motivating **stability selection** over single-fit support on real data.

### First real-data results (2026-05-29) — mixed, reported honestly

Panel: **d=23**, 2,962 hourly bars, 2024-05 → 2026-05 (Yahoo, UTC), 104 event-regime
bars (3.6%); event labels from a verified CPI/NFP/FOMC calendar (`data/events.csv`).

- **OOS `W≡0` gate — passes decisively.** On a held-out test split (train-derived
  scales), the contemporaneous DAG beats the `W≡0` SVAR: Gaussian Δ test-NLL
  −7,048, Student-t −4,448. So the hourly contemporaneous DAG genuinely carries
  the dependency structure (and Student-t < Gaussian NLL — heavy tails confirmed
  on real data). Read as explanatory power of the joint structure, not forecasting.
- **Bootstrap stability — the stable change edges are on-thesis.** Across 25 block
  bootstraps, the most stable `Δ_W` edges (freq 1.0) are crypto-internal and
  cross-asset: ETH↔SOL/LINK/XRP, BTC↔XRP, BTC→VIX, NVDA→COIN, MSTR→SPY/QQQ,
  DXY→JPY. The crypto block reliably reorganizes around macro events.
- **Permutation null — does NOT reject.** Observed `‖Δ_W‖₁ = 8.10` vs a
  vol/block-matched regime-label null (20 perms) mean 8.91 (p ≈ 0.81). The
  *global magnitude* of change is not beyond random vol-matched relabeling.
- **Edge-wise permutation test — also null (and definitive).** Per-edge `|Δ|` vs
  its own vol-matched null (100 perms): only **10** edges reach raw p<0.05 vs
  **~25 expected by chance**, and **0 survive BH-FDR**. So the change is not
  concentrated in specific edges either — the bootstrap-stable crypto edges are
  *sampling*-stable but not *event*-specific.

**Honest status:** the methods contribution (synthetic benchmark + ADMM solver)
and the DAG/heavy-tail justification are solid; but the empirical hypothesis —
that scheduled macro events reorganize the conditional dependency structure — is
**not supported** at either the global or edge-wise level, against a
volatility-matched null. The defensible empirical statement is the *null* itself:
events raise volatility (which the matched null absorbs) but do not measurably
change the conditional dependency graph at hourly RTH frequency over this window.
Figures: `outputs/figures/{regime_heatmaps,change_network,structure_over_time}.png`
(`scripts/make_figures.py`). Reproduce via `scripts/run_real_panel.py` and
`scripts/run_real_robustness.py`.

### Lagged structure & lag order (2026-05-29, `scripts/run_lag_selection.py`, `run_lagged_analysis.py`)

All models use **p=1** (1-hour lag), and that is justified: OOS NLL is flat across
p∈{1..6} (range 0.2%, non-monotone) and lags ≥2 carry negligible weight
(‖A_ℓ‖₁≈0.01 vs ‖A_1‖≈0.76 vs ‖W‖≈14.8). **The hourly cross-asset dependency is
overwhelmingly contemporaneous**; lead-lag is ~20× weaker and 2–6h lead-lag ≈ 0
(real lead-lag lives at minutes, which this 2-year hourly panel can't resolve).
The lag-1 lead-lag graph has a few sensible persistent edges (SPY→VIX, MSTR→BTC,
rates→JPY; `outputs/figures/lagged_a_persistence.png`). **Event change in the
lagged structure is also null** — edge-wise Δ_A: 6 edges p<0.05 vs ~26 by chance,
0 survive BH; global ‖Δ_A‖₁ p=0.095 is an artifact (the 104-row event regime
can't estimate the tiny lead-lag, so A_event≈0 and Δ_A≈−A_ordinary). So **neither
W nor A reorganizes around events beyond a vol-matched null.**

### Broadened synthetic sweep (2026-05-29, 12 seeds, `scripts/run_synthetic_sweep.py`)

FR-tDBN (admm + Student-t + fused) vs the DYNOTEARS-equivalent (smooth-L1 +
Gaussian + independent) on change-W AUROC, swept one axis at a time:
change-edges 1→5: 0.85–0.99 vs 0.60–0.69 (**+0.25–0.30**); samples 50→200:
0.82–0.90 vs 0.55–0.61; tail ν 3→30: gap **+0.31 (heavy) → +0.19 (≈Gaussian)** —
largest under heavy tails, as the Student-t likelihood predicts. change-edges=0 is
a clean null control (Δ floor ≈ 0.27). Figure:
`outputs/figures/synthetic_sweep_change_edges.png`.

### Baselines & sensitivity (2026-05-29)

- **Null is robust** (`scripts/run_sensitivity.py`): no event type (FOMC/CPI/NFP)
  and no window (±1h/2h/4h) rejects — every observed `‖Δ‖₁` ≤ its matched-null
  mean (all p ≥ 0.52).
- **vs SVAR (`W≡0`):** decisive — all DAG models beat it by ~4.4k nats (Student-t)
  / ~7k (Gaussian) OOS.
- **vs DYNOTEARS (`scripts/compare_baselines.py`):** split, stated honestly.
  *Synthetic ground-truth recovery* — FR-tDBN beats the DYNOTEARS-equivalent
  (smooth-L1 + Gaussian + independent) by **+0.12 to +0.23 change-W AUROC**,
  dominated by the **exact-prox ADMM solver** (Student-t / fusion secondary).
  *Real-data OOS predictive NLL* — FR-tDBN does **not** win; pooled Gaussian
  DYNOTEARS is slightly better (~160–360 nats), as the thin event regime gives
  fusion little to exploit out of sample. **The defensible claim is better
  change-graph *recovery* (ground truth), not better held-out density.** Solver
  is the analogue of Danaher et al. 2014's joint-graphical-lasso ADMM (`papers/`).
- **Descriptive structure-over-time** (`scripts/explore_viz.py`): a persistent
  dependency backbone (IEF→TLT rates, DXY→EUR dollar, QQQ→SPY equity, SLV→GLD
  metals) with crypto links strengthening through 2025
  (`outputs/figures/edge_persistence.png`, `top_edges_per_block.png`).

### What's next

The analysis is complete — synthetic recovery is defended across the design space,
and the real-data empirical question is answered with a robust, controlled null.
What remains is the **NeurIPS write-up** (≤ 8 pp) and optional rigor (BIC/held-out
λ,γ selection; a rank-transform robustness column). See `ROADMAP.md`.

---

## 1. Problem statement

DYNOTEARS learns a single dynamic Bayesian network from a stationary series under
Gaussian-equivalent squared loss. Two settings it does not address well: (a) data
with known but heterogeneous regimes that share most of their structure, and (b)
heavy-tailed noise where squared loss is a mismatched likelihood. The empirical
question — do crypto↔equity↔rates↔FX↔vol dependencies change around scheduled
macro events, and which edges flip — is exactly a regime-conditioned DBN problem
with ~100 heavy-tailed variables and calendar-known regime indicators.

## 2. Method: Fused-Regime Student-t DBN (FR-tDBN)

### 2.1 Model and notation

Regimes `k ∈ {0, 1}` (0 = ordinary, 1 = event); `d` variables; lag order `p`;
`n_k` observations in regime `k`; `N = Σ_k n_k`. Per regime the lagged design is
`X^k ∈ ℝ^{n_k×d}` (contemporaneous targets) and `Y^k_ℓ ∈ ℝ^{n_k×d}` (series lagged
by `ℓ = 1..p`). Parameters `Θ = (W^0, W^1, A^0_{1:p}, A^1_{1:p})` with intra-slice
DAGs `W^k` (`diag = 0`, entry `W^k[i,j]` = directed edge `i→j`, column `j` = the
structural equation for variable `j`) and inter-slice `A^k_ℓ`. Residuals

```
E^k = X^k − X^k W^k − Σ_{ℓ=1}^{p} Y^k_ℓ A^k_ℓ          (n_k × d)
```

are modeled i.i.d. Student-t with shared `ν` (fixed) and per-variable scale `σ_j`
(fixed, pooled robust MAD).

### 2.2 Objective

Minimize `F(Θ) = f(Θ) + g(Θ)` subject to acyclicity:

```
f(Θ) = (1/N) Σ_k Σ_{t,j} ℓ( E^k[t,j] ; ν, σ_j )                       (smooth data term)

g(Θ) = λ_W (‖W^0‖₁ + ‖W^1‖₁) + λ_A Σ_ℓ (‖A^0_ℓ‖₁ + ‖A^1_ℓ‖₁)         (within-regime sparsity)
     + γ_W ‖W^1 − W^0‖₁       + γ_A Σ_ℓ ‖A^1_ℓ − A^0_ℓ‖₁              (cross-regime fusion)

s.t.  h(W^k) = 0,  diag(W^k) = 0,   k ∈ {0,1},   h(W) = tr(exp(W∘W)) − d
```

with, at fixed `ν, σ`:

```
Student-t:  ℓ(e; ν, σ) = log σ + ((ν+1)/2) · log(1 + e²/(ν σ²))
Gaussian :  ℓ(e; σ)    = log σ + e²/(2 σ²)
```

`f` is smooth, `g` is convex but nonsmooth, `h` is smooth but nonconvex.
The fusion term `γ‖W^1−W^0‖₁` is the heart: edges that do not change pay no cost,
so the model borrows strength from the large ordinary regime to pin shared
structure, and the reported object is the sparse `Δ_W = W^1 − W^0`.

### 2.3 Optimization (the process we run)

We separate the three term types: **L-BFGS** for the smooth part, an
**augmented-Lagrangian outer loop** for the nonconvex acyclicity, and a
**closed-form proximal operator** for the nonsmooth penalties via **consensus
ADMM**. Penalty copies `Z = (V^0, V^1, B^0_{1:p}, B^1_{1:p})` of `Θ`, scaled dual
`U`, constraint `Θ = Z`. The smooth part the L-BFGS step sees is

```
S(Θ) = f(Θ) + Σ_k [ (ρ_h/2) h(W^k)² + α_k h(W^k) ]
```

**Algorithm (begin → end):**

```
init Θ ~ small random (diag W = 0); Z ← Θ; U ← 0; α ← 0; ρ_h ← 1; h_prev ← ∞.

for outer m = 1..M:                                  # acyclicity augmented Lagrangian
    repeat T_admm times:                             # consensus ADMM
        # (1) Θ-update — smooth, L-BFGS:
        Θ ← argmin_Θ  S(Θ) + (ρ/2)‖Θ − Z + U‖²_F
        # (2) Z-update — closed-form fused prox (exact zeros), per coordinate:
        Z ← FusedProx2( Θ + U ; λ/ρ , γ/ρ )           # diag(V)=0; A uses (λ_A,γ_A)
        # (3) dual:
        U ← U + Θ − Z
    h_max ← max_k |h(W^k_Θ)|
    if h_max > 0.25·h_prev and ρ_h < ρ_h_max:  ρ_h ← 10·ρ_h
    else:                                       α_k ← α_k + ρ_h·h(W^k_Θ);  h_prev ← h_max
    if h_max ≤ h_tol: break

report W^k ← V^k (exactly sparse), A^k_ℓ ← B^k_ℓ, Δ_W = V^1 − V^0, Δ_A = B^1 − B^0;
log h(W^k), primal ‖Θ−Z‖_F and dual ρ‖Z−Z_prev‖_F residuals.
```

The Θ-update is smooth (the only nonconvexity is `h²`), so L-BFGS with strong-Wolfe
line search is well-behaved. All sparsity/fusion lives in the Z-update.

**Closed-form K=2 fused prox** (Friedman et al. 2007; Danaher et al. 2014). For
each coordinate, `FusedProx2(a^0, a^1; λ', γ')` solves
`min ½(v^0−a^0)² + ½(v^1−a^1)² + λ'(|v^0|+|v^1|) + γ'|v^1−v^0|` by **fuse, then
soft-threshold**:

```
Step 1 (fuse; the mean (a^0+a^1)/2 is preserved):
    diff = a^1 − a^0
    if  diff >  2γ' : t^0 = a^0 + γ',  t^1 = a^1 − γ'
    elif diff < −2γ': t^0 = a^0 − γ',  t^1 = a^1 + γ'
    else            : t^0 = t^1 = (a^0 + a^1)/2
Step 2 (soft-threshold): v^k = sign(t^k)·max(|t^k| − λ', 0)
```

The difference is soft-thresholded by `2γ'`, which forces `W^0 = W^1` (hence exact
zeros in `Δ`) on unchanged edges — the selection behavior that smoothed-L1 lacks.
Empirically (§0) `max_k h(W^k) ≈ 4e-3` at convergence; ADMM × the nonconvex
acyclicity did not require the warm-start fallback in the spec.

### 2.4 Solver variants and ablations (one codebase, fair comparisons)

- **Loss** (`FitConfig.loss`): `student_t` | `gaussian` — isolates the heavy-tail
  contribution.
- **Fusion strength** `γ`: `γ=0` (independent per-regime fits) vs `γ>0` (fused).
- **Solver** (`FitConfig.solver`): `admm` (exact prox, headline) vs `lbfgs_smooth`
  (smoothed `√(x²+ε)` penalties, the DYNOTEARS-style baseline) — isolates the prox.
- **Adaptive fusion** (`fusion="adaptive"`, `adaptive_pilot ∈ {uniform, independent}`):
  per-edge `γ_ij = γ₀/(|Δ̂_ij| + ε)` (Option D). Retained as a **documented negative
  ablation** — it collapses `Δ` at small event-`n` (§0).

## 3. Synthetic benchmark (the methods core)

Real financial graphs have no ground-truth DAG, so the methods claim is earned on
simulation. `make_regime_pair` samples an Erdős–Rényi ordinary DAG (topologically
ordered), stabilizes the reduced-form VAR once, then applies a **single stable
step** to a **sparse delta** (additions, removals, reweights) — so the change
masks equal the *realized* `W^1−W^0` / `A^1−A^0` (no rescale contamination), `Δ`
stays sparse, and regime sizes can be **imbalanced** (`n_event ≪ n_ordinary`).

**Var-sortability honesty (Reisach et al. 2021).** The generator is var-sortable
by construction (variance grows down the topological order; raw score ≈ 0.83 at
ν=5). We therefore **standardize each regime** before fitting, which pins the
measured var-sortability at exactly **0.50** — so recovery cannot ride the variance
gradient and the synthetic result transfers to the standardized real pipeline.

**Grid & metrics.** `run_grid` runs solver × loss × fusion over seeds and `n`,
reporting change-edge AUROC for `W` and `A`, var-sortability (raw and fitted),
`max_h`, and runtime; `summarize_grid` aggregates to mean ± se. Results: §0.

## 4. Data (as built)

Hourly OHLCV from **Yahoo Finance** (`scripts/fetch_yahoo_hourly.py`, via
`yfinance` + a `curl_cffi` browser session to clear Yahoo's rate limiter), cached
under `data/raw/yahoo/` (gitignored). Yahoo gives clean **UTC** timestamps and
~2.8y of hourly history; the usable common window after alignment is
**2024-05-30 → 2026-05-28** (bounded by the crypto series' start). An earlier Stooq
hourly dump was explored but abandoned — ambiguous CET/UTC timestamps and no spot
VIX (the Stooq loader/audit in `frtdbn/data.py` + `scripts/audit_data_coverage.py`
remain as a cross-check).

### 4.1 Panel (`frtdbn.panel.DEFAULT_PANEL`, d = 23)

| block | members |
|---|---|
| equity / crypto-equity | SPY, QQQ, IWM, NVDA, COIN, MSTR |
| rates / credit | TLT, IEF, SHY, HYG |
| commodities | GLD, SLV, USO |
| FX | DX-Y.NYB (DXY), EURUSD=X, GBPUSD=X, JPY=X |
| vol | ^VIX |
| crypto (spot) | BTC, ETH, SOL, XRP, LINK (-USD) |

Dropped after auditing coverage: **CNY=X** (sparse during US RTH — halved the
common grid), **^TNX** (redundant with IEF; its :20 stamp trimmed the grid),
**IBIT** (redundant with BTC-USD).

### 4.2 Panel construction (`frtdbn/{data,panel}.py`)

1. `build_return_panel`: per-symbol close → **floor each timestamp to the hour**
   (Yahoo stamps equities at :30, crypto/FX/VIX at :00) → align on the union grid
   → **log returns** → drop rows with any missing return. Drop-any-NaN restricts
   to the common trading grid (equity RTH, where 24/7 crypto is also active):
   **2,962 hourly bars, ~6/day, 499 trading days, zero NaNs**.
2. `build_lagged_design`: **rolling past-only z-score** (window 250, min-periods
   60; no look-ahead) → drop warmup → lagged design at **p = 1**. Lag order is
   justified empirically (§0): lead-lag beyond 1 hour is negligible.

### 4.3 Event regime

`data/events.csv` holds CPI / NFP / FOMC release dates in ET (08:30 for CPI/NFP,
14:00 for FOMC statements), **verified against the official BLS/Fed schedules**
(incl. the 2025 lapse-driven exceptions: Sep-25 CPI on 10-24, Oct-25 CPI canceled,
Nov-25 CPI on 12-18, Jan-26 CPI on 02-13). `load_event_calendar` converts ET→UTC
(DST-aware via `America/New_York`); `label_event_regime` tags bars within **±2h** of
a release → **104 event bars (3.6%)** — the small-`n_event` borrow-strength regime
the fusion targets. Caveat: 08:30-ET CPI/NFP precede the cash open, so on the RTH
panel their window captures the **post-open reaction**; FOMC (14:00 ET) is
mid-session and captured directly.

## 5. Evaluation protocol (executed; numbers in §0)

No ground-truth DAG on real data, so (helpers in `frtdbn/{evaluation,robustness,
splitting}.py`):

- **Time-ordered train/test split** (`train_test_split_regimes`, last 20% = test),
  with **train-derived scales** for every held-out number (no leakage).
- **Out-of-sample full-density NLL** (`full_nll`, `Γ`/`½log(νπ)` constants
  included) for valid t-vs-Gaussian and DAG-vs-`W≡0` comparison. The `W≡0` SVAR
  ablation (`svar_vs_dag_oos`) is the frequency/acyclicity sanity — only an
  out-of-sample win counts, since the DAG has strictly more parameters. **Result:
  DAG ≫ SVAR** (§0); this also serves as the hourly "is W meaningful" check.
- **Stability selection** via block bootstrap (`stability_selection`) — Δ edge
  frequencies (the reliable lens given low single-fit precision at small event-`n`).
- **Global + edge-wise permutation null** on Δ_W **and** Δ_A
  (`permutation_null_delta_norm`, `edgewise_permutation_test`, `delta="W"|"A"`),
  block- and **volatility-matched** (event windows are mechanically high-vol).
  **Result: null** at both global and edge-wise level, for W and A (§0).
- **Sensitivity** (`scripts/run_sensitivity.py`): per-event-type (FOMC/CPI/NFP)
  and ±1/2/4h windows — null robust everywhere. **Lag-order selection**
  (`run_lag_selection.py`). **Baselines** vs SVAR and DYNOTEARS-per-regime/pooled
  (`compare_baselines.py`).
- **Time-resolved visualizations** (`make_figures.py`, `explore_viz.py`,
  `run_lagged_analysis.py`): regime heatmaps, change network, structure-over-time,
  edge persistence (W and lag-1 A).
- Community detection (Louvain) and the HRP "dessert" were **not** pursued.

## 6. Remaining work

The analysis is complete; what remains is the **NeurIPS write-up (≤ 8 pp)** and
optional rigor (BIC/held-out λ,γ selection; a rank-transform robustness column).
See `ROADMAP.md`.

## References

NOTEARS (Zheng et al. 2018); DYNOTEARS (Pamfil et al. 2020); fused/group graphical
lasso ADMM (Danaher et al. 2014); fused-lasso decomposition (Friedman et al. 2007);
ADMM (Boyd et al. 2011); var-sortability (Reisach et al. 2021); adaptive lasso
(Zou 2006); reweighted ℓ1 (Candès et al. 2008).
