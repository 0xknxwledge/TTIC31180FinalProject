# Fused-Regime Student-t Dynamic Bayesian Networks for Crypto–Macro Spillover Structure

**John Beecher · TTIC 31180 (Probabilistic Graphical Models) · Final Project (living document)**

This supersedes `BeecherJohn_TTIC31180_ProjectProposal.pdf`. The contribution lives
on the **structure-learning** side (a method paper with a financial application);
HRP is at most a one-figure illustration. This document is kept current with
progress; the canonical remaining-work list is `TODO.md`, and the original ADMM
design spec is `docs/superpowers/specs/2026-05-28-admm-fused-fr-tdbn-design.md`.

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
- **Data coverage audit** (`frtdbn/data.py` + `scripts/audit_data_coverage.py`):
  Stooq hourly loader/auditor. Finding: usable hourly history is **~2 years
  (2024-05 → 2026-05)**, not 5, and Stooq timestamps are **not ET** (≈ CET/UTC) —
  both must shape the real panel. `^vix` absent (needs a proxy / different source).
- **62 tests pass** (`python -m pytest tests -q`). The core formulation (§2) is
  unchanged since the ADMM solver landed; this round added evaluation/robustness
  methodology and the data audit, not new model results.

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

Broaden the synthetic benchmark (more samples / change-edge counts / regimes /
seeds) to make the recovery claim maximally defensible, then the **LaTeX writeup**
— a methods paper (exact-prox fused Student-t DBN, validated on ground-truth
recovery) whose empirical section reports an honest, carefully-controlled null.
Task list in **`TODO.md`**.

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

## 4. Data design (forward plan)

### 4.1 Universe — start small and complete

The first real panel is a **small but complete crypto–macro panel** (`d ≈ 15–20`),
because the spillover claim *requires* macro assets — they are constitutive, not a
later expansion. Candidate: a few liquid crypto perps (BTC, ETH, SOL, …) + equity
index proxies (SPY/QQQ or ES/NQ) + rates proxies (TLT/IEF/SHY or ZN/ZB) + DXY +
GLD + VIX. The larger `d ≈ 89` universe in v1 (single-name tech, sector ETFs,
late-listing alts like ARB/SUI/WIF) is a stretch only after the small panel works.

### 4.2 Sampling and macro alignment

- **Hourly bars** as the primary frequency (≈ 8,200 RTH obs over 5y). Hourly makes
  intra-slice `W` defensible and aligns with event resolution.
- **CPI/NFP release at 08:30 ET, before cash-equity RTH.** Resolution: run the
  event-window analysis on near-24h instruments (crypto + index/rate futures + FX)
  so the reaction is actually captured, rather than only post-open bars.
- **Rates at hourly resolution need ETF/futures proxies — FRED yields are
  daily-only** and unusable here.
- Standardize per variable with **rolling, past-only z-scores** (avoids look-ahead
  and preserves the event-window variance shifts); report a rank (Gaussian-copula)
  variant as a robustness check.

### 4.3 History and events

A coverage audit of the local Stooq hourly dump (`scripts/audit_data_coverage.py`)
shows usable history of **~2 years (2024-05 → 2026-05)** for the candidate panel —
not the 5 years originally planned (free Stooq hourly is short). That window holds
≈ 24 CPI, 16 FOMC, 24 NFP (~64 events); with ±2h hourly windows the event regime is
a few hundred observations — `n_event ≪ n_ordinary`, the borrow-strength regime the
fusion penalty targets. A 2-year window also *reduces* the non-stationarity risk; we
still report per-sub-period Δ stability (`time_block_indices`). For a 5-year version,
supplement Stooq with Massive/Alpaca. **Timezone:** Stooq timestamps are not ET (ETF
bars fall at hours 15–22, consistent with CET/UTC), so the tz must be resolved before
aligning the 08:30-ET releases.

### 4.4 Acyclicity sanity check

Ablation with `W ≡ 0` (pure structural VAR). If held-out predictive likelihood
barely changes, contemporaneous DAG claims are weak at hourly frequency — flag
honestly. Worth running early on a tiny panel before committing the pipeline.

## 5. Real-data evaluation (forward plan)

No ground-truth DAG, so (helpers in `frtdbn/{evaluation,robustness,splitting}.py`):
- **Time-ordered train/test split** (`train_test_split_regimes`): train
  2024-05→2025-12, test 2026-01→2026-05 (~80/20). All held-out numbers use the
  test split with **train-derived** scales (no leakage).
- **Out-of-sample full-density NLL** (`full_nll`, with the `Γ` / `½log(νπ)`
  constants) for valid t-vs-Gaussian and FR-tDBN-vs-`W≡0` comparison. The `W≡0`
  SVAR ablation runs through `svar_vs_dag_oos` — only an *out-of-sample* win
  counts as evidence the contemporaneous DAG carries weight (an in-sample
  comparison favors the DAG mechanically, since it has strictly more parameters).
- **Stability selection** via block bootstrap (`stability_selection`); report `Δ`
  edge frequencies; headline = edges surviving > 70%. This is the reliable lens
  given the §0 finding that single-fit support is hard at small event-`n`.
- **Permutation null on regime labels** (`permutation_null_delta_norm`), block-
  and volatility-matched (event windows are mechanically high-volatility, so a
  naive relabel rejects trivially); ≥ 20 (ideally 100) permutations for usable
  p-value resolution.
- **Restart stability** (`fit_restarts` / top-k Δ Jaccard) and per-sub-period Δ
  stability (`time_block_indices`).
- **Community shift** (Louvain on `|W^k|`) and the optional **HRP** figure remain
  secondary / dessert.

## 6. Remaining work

See **`TODO.md`** for the live task list. Near-term: lock Option C as the headline
(optional clean slide-1 figure + γ_A decoupling for the minor A-fusion wrinkle),
then build the crypto–macro data pipeline.

## References

NOTEARS (Zheng et al. 2018); DYNOTEARS (Pamfil et al. 2020); fused/group graphical
lasso ADMM (Danaher et al. 2014); fused-lasso decomposition (Friedman et al. 2007);
ADMM (Boyd et al. 2011); var-sortability (Reisach et al. 2021); adaptive lasso
(Zou 2006); reweighted ℓ1 (Candès et al. 2008).
