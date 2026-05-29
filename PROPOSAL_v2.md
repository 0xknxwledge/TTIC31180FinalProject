# Fused-Regime Student-t Dynamic Bayesian Networks for Crypto–Macro Spillover Structure

**John Beecher · TTIC 31180 (Probabilistic Graphical Models) · Final Project (v2)**

This document supersedes `BeecherJohn_TTIC31180_ProjectProposal.pdf`. The change in framing is that the project's contribution lives on the **structure-learning** side (a method paper with a financial application), not on the portfolio-construction side. HRP is demoted to an optional one-figure illustration.

---

## 1. Problem statement

Two intertwined questions:

1. **Methodological.** DYNOTEARS (Pamfil et al., 2020) learns a single dynamic Bayesian network from a stationary time series under Gaussian-equivalent squared loss. Two practical settings it does not address well: (a) data with known but heterogeneous regimes that share most of their structure, and (b) heavy-tailed noise where Gaussian-equivalent loss neither matches the data nor exploits the identifiability that non-Gaussian errors would provide.

2. **Empirical.** Around scheduled macro information events (CPI, NFP, FOMC), does the contemporaneous + lagged dependency structure linking crypto, equities, rates, FX, and volatility instruments change in measurable, stable ways — and if so, which edges flip on?

These questions plug into each other: regime-conditioned DBN learning is exactly the tool the empirical question needs, and the empirical setting (~100 heavy-tailed financial variables with calendar-known regime indicators) is where the methodological extension matters.

## 2. Method: Fused-Regime Student-t DBN (FR-tDBN)

Let regimes be indexed `k ∈ {1, …, K}` with calendar-known assignments `z_t ∈ {1, …, K}`. Per regime: intra-slice DAG `W^(k) ∈ ℝ^{d×d}` (with the standard NOTEARS acyclicity surrogate `h(W) = tr e^{W∘W} − d` and `diag(W^(k)) = 0`), and inter-slice matrices `A^(k)_1, …, A^(k)_p ∈ ℝ^{d×d}`. Residuals `z_{m,t}` are i.i.d. Student-t with shared degrees-of-freedom `ν` and per-variable scale `σ_j`.

Joint objective:

```
min_{W^(k), A^(k), ν, σ}
    Σ_k  Σ_{j,t ∈ Ω_k}  ℓ_t( x_{j,t} − (X_t W^(k))_j − (Y_t A^(k))_j ; ν, σ_j )
  + λ_W Σ_k ‖W^(k)‖_1   +  λ_A Σ_k ‖A^(k)‖_1                       # within-regime sparsity
  + γ_W Σ_{k<l} ‖W^(k) − W^(l)‖_1   +  γ_A Σ_{k<l} ‖A^(k) − A^(l)‖_1   # cross-regime fusion
  + (ρ/2) Σ_k h(W^(k))²   +  α Σ_k h(W^(k))                          # acyclicity (aug. Lagrangian)
```

where the Student-t negative log-density must include the scale and degrees-of-freedom terms whenever `ν` and `σ` are estimated:

```
ℓ_t(z ; ν, σ_j)
  = log σ_j
  + ((ν+1)/2) · log(1 + z² / (ν σ_j²))
  + log Γ(ν/2) + 0.5 log(νπ) - log Γ((ν+1)/2).
```

Use `σ_j = softplus(s_j) + ε` and `ν = 2 + softplus(η)` if finite residual variance is required. If `ν` is fixed during early experiments, the final Gamma terms can be dropped, but `log σ_j` cannot.

**Two contributions over DYNOTEARS:**

- **Fused penalty (γ terms).** Edges that are stable across regimes pay no fusion cost; edges that genuinely differ are kept distinct. The change graph `Δ_{WW} = W^(event) − W^(ordinary)` is what we actually report. This borrows strength from the much larger ordinary-regime sample to anchor what does not change, addressing the small-n problem inside event windows.
- **Student-t likelihood.** DYNOTEARS explicitly discusses identifiability under either non-Gaussian errors or a standardized Gaussian SEM; the conservative claim here is not that DYNOTEARS is unidentified, but that squared loss is a mismatched likelihood for heavy-tailed financial residuals. Switching to a Student-t log-likelihood (i) matches the heavy tails of crypto returns more honestly, (ii) gives valid out-of-sample predictive-likelihood comparisons, and (iii) should improve structure recovery in the synthetic regime where the data-generating errors are actually Student-t.

**Optimization.** Smooth `|x|` with `√(x² + ε)`, optimize via L-BFGS-B or a bound-free reparameterized L-BFGS on the joint problem, augmented-Lagrangian outer loop on the K acyclicity constraints — the same scheme as DYNOTEARS but K times wider plus the fused-lasso terms. `ν` and `σ_j` updated jointly via the same L-BFGS pass after a fixed-`ν` version works. If using pure PyTorch, note that `torch.optim.LBFGS` is not L-BFGS-B; enforce positivity and zero diagonals by reparameterization/masking, or use SciPy L-BFGS-B with PyTorch-provided gradients. For `K=2`, `d=120`, `p=2`, raw structural parameter count is `K(1+p)d² = 86,400` before positivity splits or optimizer state, not ≈30k; the laptop-feasible headline should be `d≈50–80`, with `d≈120` as a computational stretch.

**Implementation language.** PyTorch (autograd handles all smooth surrogates), CPU-only.

## 3. Data design

### 3.1 Universe (`d ≈ 89`, extensible to ~150 via per-asset features)

| Block | Count | Examples |
|---|---|---|
| Crypto perps (returns) | 25 | BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, MATIC, TON, TRX, LTC, BCH, ATOM, NEAR, APT, ARB, OP, SUI, INJ, TIA, SEI, ORDI, WIF |
| Crypto derivatives features | 15 | Funding rates + perp-spot basis for top 5; aggregate OI deltas; USDT/USDC mcap deltas; DVOL |
| Crypto-equity hybrids | 6 | COIN, MSTR, MARA, RIOT, IBIT, ETHA |
| Equity sectors / factors | 15 | XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB, XLRE, XLC, SPY, QQQ, IWM, MTUM |
| Single-name tech anchors | 8 | NVDA, AAPL, MSFT, META, GOOGL, AMZN, TSLA, AMD |
| Rates | 8 | 2Y, 5Y, 10Y, 30Y yields; 2s10s, 5s30s spreads; SOFR; 10Y TIPS |
| FX | 5 | DXY, EUR, JPY, CNH, KRW |
| Commodities | 4 | GLD, USO, HG (copper), UNG |
| Vol | 3 | VIX, MOVE, VVIX |
| **Total** | **89** | |

Optional feature axis (return, 5-min realized vol, signed flow imbalance per asset) would push `d` to ~150; deferred to a stretch goal — only added if base panel works. The first implementation should use a survivorship-clean core panel (`d≈50–60`) with full history and hourly data availability; `d≈89` is the second-stage universe after data gaps are audited.

### 3.2 Sampling — dual-clock design

- **Primary frequency: hourly bars, US RTH only** (9:30–16:00 ET). Five-year history yields ≈ 8,200 observations. Hourly resolution makes intra-slice `W` defensible (fast enough for cross-asset propagation, slow enough that lead-lag is meaningful), aligns naturally with macro event resolution (8:30 ET CPI vs. 10:00 ET reaction), and avoids the macro-stale-forward-fill problem during overnight crypto-only hours. Important caveat: CPI/NFP releases at 8:30 ET fall before cash equity RTH, so a ±2h RTH-only event window contributes only the post-open reaction bars unless futures/pre-market data are included.
- **Secondary frequency: 5-minute bars inside ±2h event windows** for a separate within-event decay analysis (replaces the original proposal's "sliding window inside event periods").

### 3.3 History and event counts

- Window: **2021-01-01 → 2026-05-01** (5 years post-COVID-floor, captures 2022–24 macro-shock regime).
- Event releases: ≈ 65 CPI, ≈ 40 FOMC, ≈ 60 NFP → ≈ 165 event days. With ±2h hourly windows: ~660 event observations before RTH filtering; fewer for 8:30 ET CPI/NFP under a cash-market-only panel. With ±1 trading day: ~1300. Small enough that the fused penalty is doing real work.

### 3.4 Macro alignment

Restrict the structure-learning panel to US RTH hours throughout. Yields/futures sampled at the same hourly grid as crypto. Drop overnight macro bars rather than forward-filling. 24/7 crypto used only for descriptive plots, not for the panel.

### 3.5 Defenses against known structure-learning failure modes

- **var-sortability** (Reisach et al. 2021). Standardize per-variable with rolling, past-only z-scores before regime splitting; avoid regime-wise normalization that leaks future/event information or removes the very variance shifts the event analysis is trying to measure. Report a rank-transformed (Gaussian copula) variant as a robustness check.
- **Acyclicity sanity check.** Ablation with `W ≡ 0` (pure structural VAR, all edges in `A`). If predictive likelihood barely changes, contemporaneous DAG claims are weak — flag honestly.

## 4. Synthetic benchmark (non-negotiable)

The methods contribution is only credible if it beats baselines on ground-truth data. This is the slide-1 figure.

- `d = 100, p = 2, K = 2, n_k ∈ {200, 500, 1000}`
- Sample `W^(O)` as an Erdős–Rényi DAG with mean degree 4, topologically ordered. Set `W^(E) = W^(O)` on ~90% of edges, plus ~10 new "regime-specific" edges concentrated in a macro→crypto sub-block.
- Inter-slice `A^(O), A^(E)` analogously.
- Noise: Student-t with `ν ∈ {3, 5, ∞ (Gaussian)}`.
- **Algorithms compared:**
  1. DYNOTEARS per regime, independently (the paper baseline)
  2. DYNOTEARS pooled (no regime split)
  3. Fused-regime Gaussian DYNOTEARS (fusion only)
  4. Independent Student-t DYNOTEARS per regime (heavy-tail loss only)
  5. **FR-tDBN — ours**
  6. Fused graphical lasso on residuals (Danaher et al. 2014) — undirected descriptive baseline, not a direct SHD competitor
- **Metrics:** SHD on `W^(O)`, on `W^(E)`, and on `Δ = W^(E) − W^(O)`. AUROC on change-edge recovery. Wall-clock time.
- **Expected story.** FR-tDBN dominates on Δ recovery when per-regime `n` is small (the regime where vanilla per-regime DYNOTEARS overfits), with the biggest gains at `ν = 3` (heavy tail).

## 5. Real-data evaluation

SHD requires ground truth; on real data we use:

- **Out-of-sample Student-t log-likelihood** on held-out 2025 data (~12 months).
- **Stability selection** via block-bootstrap (preserving time order), initially 25-50 resamples; report edge frequencies. Headline real-data findings = edges in `Δ` surviving at > 70% frequency. Increase toward 100 only if one fit is fast enough.
- **Permutation null on regime labels**: shuffle event-window assignments in blocks, refit, build a null distribution for `‖Δ‖_1`. Compare against the observed value. Start with 20 permutations; full 100-permutation null is a stretch.
- **Community structure shift.** Run Louvain on `|W^(k)|` per regime; ask whether crypto nodes migrate from a crypto-internal community into a macro community during event windows.

## 6. Optional dessert: structured-covariance HRP

If real-data results are strong, *one figure*: use the FR-tDBN model-implied covariance `Σ̂(W^(k), A^(k))` per regime as the input correlation for HRP, compare cluster dendrograms across regimes. This requires converting the structural model into a stable reduced-form VAR and solving for the implied covariance; if the fitted VAR is not stable, use empirical residual covariance by regime instead. This makes the PGM causally upstream of the portfolio object (a clean fix for the broken-bridge problem in the original proposal). Hard-cut if time slips.

## 7. Revised timeline

- **Week 1** — Data pipeline: hourly bars for a core `d≈50–60` panel, RTH-aligned panel, event-window labels, caching layer. Implement FR-tDBN in PyTorch (objective, smooth surrogates, augmented-Lagrangian loop), first with fixed `ν` and fixed robust scales.
- **Week 2** — Synthetic benchmark. Hyperparameter sweep (`λ_W, λ_A, γ_W, γ_A, ν`). Validate structure recovery empirically under Gaussian and Student-t errors. **This is the methods-paper core.**
- **Week 3** — Real-data application. Fit FR-tDBN to the 5y core panel first; expand toward the 89-var panel only if timing and data coverage permit. Stability selection, permutation null, community shift. Ablations: Gaussian vs. t loss, fused vs. unfused, `W ≡ 0` SVAR-only.
- **Week 4** — Writeup (NeurIPS style, ≤ 8pp) + presentation. HRP dessert only if real-data results are clean.

## 8. Open questions to resolve before code is written

1. **Crypto data source.** Plan: Binance perp klines + funding (free, possibly via `ccxt`). The uninterrupted 2021-2026 top-25 perp panel is not realistic because several listed assets launched later or changed liquidity regimes. *Decision needed:* either use a smaller full-history core universe or use a later common start date for the expanded universe.
2. **`K = 2` vs. `K = 3`.** Two-state (event / ordinary) is the main result. Three-state (pre-event / event / post-event) captures information leakage and digestion but each regime gets ~⅓ the samples. *Tentative:* K=2 as headline, K=3 as a Section 5 extension if time permits.
3. **Pooled event regime.** Initially pool CPI + FOMC + NFP into one "event" regime; split into K=4 (one per event type + ordinary) if `‖Δ‖_1` results warrant.
4. **Event window width.** Default ±2h hourly bars around scheduled release time. Sensitivity to ±1h, ±4h reported in appendix.
5. **Hyperparameter selection.** BIC for `λ_W, λ_A`. Cross-validated held-out log-likelihood for `γ_W, γ_A, ν`. Open question: whether to share `ν` across regimes (yes for parsimony, no if event tails are visibly fatter).
6. **Equity / rates / FX data source.** Polygon for equities + ETFs; consider FRED for rates / DXY; Yahoo Financials as fallback. The QTS project used Nasdaq Data Link QUOTEMEDIA — still has a quota? Need to confirm before committing.
7. **Identifiability empirics.** Does the Student-t likelihood actually recover the true `W` better than Gaussian + Huber on synthetic Student-t data, holding everything else fixed? Need to verify before claiming this as a contribution.
8. **Computational ceiling.** L-BFGS-B on `K=2, d=120, p=2` means 86,400 raw structural parameters, plus scale/df parameters, positivity splits if using the NOTEARS implementation trick, and optimizer state. Estimate ~hours per fit; stability selection and permutation tests are the true bottleneck. Headline plan should start at `d≈50–60`, then expand only after timing one full augmented-Lagrangian fit.
9. **Acyclicity at hourly frequency.** Still worth pressure-testing: is hourly really slow enough for intra-slice DAGs to be defensible in this universe? The `W ≡ 0` ablation will answer this empirically.

## 9. What changed from v1

| v1 | v2 |
|---|---|
| Apply DYNOTEARS + Huber loss | Replace with FR-tDBN: fused multi-regime + Student-t likelihood |
| `d ≈ 10–15` (BTC, ETH, SOL + macro) | `d ≈ 89` (extensible to ~150) |
| Daily / unspecified frequency | Hourly RTH primary + 5-min event-window secondary |
| HRP bake-off (4 portfolios) as half the project | HRP demoted to optional one-figure dessert; PGM bridge fixed via structured covariance |
| SHD on real data | SHD on synthetic only; real data uses stability selection + permutation null + held-out log-likelihood |
| No synthetic benchmark | Synthetic benchmark is the methods-paper core (week 2) |
| No defense against var-sortability / acyclicity failure modes | Both addressed explicitly |

## 10. Critique and additions before implementation

### 10.1 What v2 gets right

- The reframing is much stronger than v1. The old proposal had two projects fighting for space: DYNOTEARS graph estimation and HRP portfolio construction. V2 correctly makes structure learning the contribution and leaves HRP as an optional illustration.
- The fused-regime idea is well matched to the empirical setting. Event windows are sparse, ordinary windows are plentiful, and the scientific object is the change graph `Δ`; a fusion penalty targets exactly that.
- The synthetic benchmark is now essential rather than decorative. Since real financial graphs have no ground-truth DAG, the methods claim must be earned on simulated data with known `W`, `A`, and `Δ`.
- The proposal now acknowledges two major causal-structure failure modes: var-sortability and the possibility that contemporaneous `W` is not empirically useful once lagged `A` is present.

### 10.2 Main risks

1. **The original Student-t objective was under-specified.** If `σ_j` is learned, dropping `log σ_j` makes the optimization degenerate. If `ν` is learned, the Gamma-function normalization matters too. This is the most important mathematical fix.
2. **The identifiability claim should be softened.** DYNOTEARS already discusses non-Gaussian identifiability and a standardized Gaussian case. The stronger, defensible claim is that Student-t loss is a better specified likelihood for heavy-tailed financial residuals and should improve recovery under matched heavy-tailed simulations.
3. **The data universe is too optimistic.** ARB, OP, SUI, TIA, SEI, ORDI, WIF, IBIT, and ETHA do not support a clean five-year common panel. A smaller core universe or a shorter expanded sample is necessary.
4. **RTH-only event timing weakens CPI/NFP coverage.** CPI and NFP are usually 8:30 ET releases, before cash equity RTH. Either include liquid futures/pre-market proxies or explicitly define event windows as post-open reaction windows.
5. **Computation is the binding constraint.** The original parameter count was too low. The feasible version is a core panel, fixed `ν` first, fewer bootstraps/permutations, and only then expansion.
6. **The optimizer details matter.** The DYNOTEARS paper uses L-BFGS-B after splitting positive and negative weights. Pure PyTorch L-BFGS does not provide box constraints, so the implementation needs explicit masks/reparameterizations or a SciPy bridge.
7. **Some baselines need to isolate contributions.** Pooled DYNOTEARS and independent per-regime DYNOTEARS are not enough. Add fused-Gaussian and independent Student-t variants so the report can say whether gains come from fusion, heavy-tail likelihood, or both.
8. **Real-data evaluation needs discipline.** Louvain communities and HRP are nice but secondary. The report should prioritize held-out likelihood, stability frequencies, permutation null, and interpretable high-confidence `Δ` edges.

### 10.3 Minimum viable final-report spine

1. **Problem and model.** Start with DYNOTEARS, then introduce known regimes, fused penalties, and Student-t residuals.
2. **Synthetic evidence.** Show that FR-tDBN improves change-edge recovery when event-regime `n` is small and errors are heavy-tailed. This is the main contribution.
3. **Crypto-macro application.** Fit the core panel, report stable event-specific edges, and test whether observed `‖Δ‖_1` beats the permuted-regime null.
4. **Ablations.** Gaussian vs. Student-t, fused vs. independent, pooled vs. regime-specific, and `W≡0` SVAR-only.
5. **Limits.** State clearly that directed financial edges are model-based dependency claims, not trading-grade causal claims, and that results depend on sampling frequency and event-window design.
