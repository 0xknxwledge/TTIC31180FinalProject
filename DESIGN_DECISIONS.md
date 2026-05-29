# Design Decisions & Review Notes

Living record of the pre-implementation review of `PROPOSAL_v2.md` + `TODO.md`,
the decisions made, and the prioritized fix queue. Companion to `TODO.md`
(phase plan) and `PROPOSAL_v2.md` (the method/empirical spec).

## Locked decisions (2026-05-28)

1. **Empirical target:** a *small but complete* crypto–macro panel (`d≈15–20`),
   not crypto-only-then-expand. The headline claim ("spillover linking crypto,
   equities, rates, FX, vol") requires macro assets in the panel, so they are
   constitutive, not deferred. Build it *after* the P0 method/benchmark fixes.
2. **Where the grade lives:** the *novel contribution* (the method + a
   var-sortability-honest synthetic benchmark + one defensible real fit).
   Community detection and the HRP "dessert" are appendix-only.
3. **Data sources (hourly):**
   - `MASSIVE_KEY` in `.env` — Massive (formerly Polygon) → equities/ETFs/index
     futures, hourly. Use ETF/futures proxies for rates (TLT/IEF/SHY or ZN/ZB),
     **not** FRED (FRED yields are daily, unusable at hourly resolution).
   - `ALPACA_KEY` in `.env` — Alpaca → equities/crypto fallback.
   - Binance perps via `ccxt` (needs Switzerland VPN; 451 otherwise) for crypto.
4. **Compute:** laptop CPU for dev; AWS + Azure credits (~$10k each, YC student)
   available for GPU when we scale `d`, bootstraps, and the permutation null.
   Pick `d` and the null size to fit a sane wall-clock budget; the null (20–100
   refits over K regimes) is the true bottleneck.

## Prioritized fix queue

### P0 — foundation for the slide-1 figure (in progress)

- **[P0-A] Var-sortability honesty.** The synthetic generator
  (`frtdbn/synthetic.py`) builds upper-triangular DAGs (`i→j` for `i<j`), so the
  natural index order *is* a valid topological order and marginal variance grows
  monotonically down the order — the textbook Reisach et al. (2021)
  var-sortability setup, where NOTEARS looks good *for the wrong reason*. The
  benchmark also never standardizes, while the real pipeline (`PROPOSAL_v2.md`
  §3.5) z-scores → the synthetic "win" may not transfer.
  Fixes: (1) a `var_sortability` diagnostic to *measure and report* the score;
  (2) standardize the design before fitting so the score sits at ~0.5 and the
  method cannot ride the variance gradient; (3) consistent random node-relabel as
  hygiene (note: relabeling alone does NOT change var-sortability — it is a graph
  property; standardization is the real fix).
- **[P0-B] 2×2 baseline grid.** The benchmark currently only varies `gamma`
  (fused vs independent), both Student-t. To attribute gains we need
  {Gaussian, Student-t} × {independent (γ=0), fused (γ>0)} in the *same*
  optimizer. Add a `loss` switch (`student_t` | `gaussian`) to `FitConfig`.

### P1 — correctness / fairness before real data

- **[P1-C] Identifiability claim.** Free per-variable `σ_j` forfeits the
  equal-variance Gaussian identifiability (Peters & Bühlmann 2014); Student-t
  here is a robust M-estimator, not LiNGAM-style identifiability. Reframe to
  "more *efficient/robust* recovery under heavy tails + valid predictive-LL
  comparison." Either drop the identifiability claim or test it head-on.
- **[P1-D] Held-out / cross-model log-likelihood normalization.**
  `_student_t_nll` drops the `Γ` and `½log(νπ)` constants — fine for the
  fixed-`ν` training argmin, *wrong* for comparing t vs Gaussian held-out LL or
  for a learned `ν`. Add a proper-log-density eval function.
- **[P1-E] Pooled shared robust scales.** `_robust_scales` pools across regimes,
  so the larger ordinary regime dominates and the event regime's genuinely
  larger residuals get down-weighted as outliers — possibly erasing the event
  signal. Add a per-regime-scale sensitivity check.
- **[P1-F] Broaden synthetic Δ.** Changes are additive-only (new edges). Add
  weight changes, sign flips, removals, and pure-A (no-W) changes.

### P2 — empirical discipline

- **[P2-G] Permutation null.** `‖Δ‖₁` is mechanically inflated by event-window
  volatility — use a volatility-matched / block null, not a plain relabel.
- **[P2-H] Penalty scaling.** `λ`,`γ`,`h` don't transfer across `n`/`d`
  (data ~ `d·n`, L1 ~ `d²`, `h` undivided). Re-tune per scale; add a final
  hard-threshold / DAG-projection so reported graphs are guaranteed acyclic.
- **[P2-I] 5-year non-stationarity** (2021 bull → 2022 crash → 2023–24 → 2025–26)
  can confound the event/ordinary contrast. Consider a time-block control.
- **[P2-J] Event timing.** CPI/NFP release 08:30 ET (pre cash-equity RTH). Run
  the event-window analysis on near-24h instruments (crypto + index/rate futures
  + FX) so the reaction is actually captured at release.
- **[P2-K] Frequency pre-check.** Run the `W≡0` SVAR ablation *first* on a tiny
  panel: if hourly contemporaneous `W` carries no weight over lagged `A`, the
  "DBN" framing needs rethinking — better to learn that in week 1.
- **[P2-L] Naming.** The "DYNOTEARS baseline" is our own code at γ=0 + Gaussian
  loss — label it "Gaussian DYNOTEARS-equivalent (our implementation)."

## Things v2 already got right (not re-litigated)

Reframing to a method paper, the fused penalty matched to sparse event windows,
the synthetic benchmark as core, and §10's self-critique (under-specified
objective, optimistic universe, compute ceiling) are all sound.
