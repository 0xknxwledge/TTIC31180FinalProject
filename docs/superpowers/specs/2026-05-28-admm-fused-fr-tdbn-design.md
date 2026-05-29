# Spec: ADMM solver with exact fused-lasso prox for FR-tDBN (Option C)

**Date:** 2026-05-28 · **Status:** approved design, implementation starting ·
**Companion:** `DESIGN_DECISIONS.md`, `PROPOSAL_v2.md`

## 1. Goal

Replace the smoothed-L1 fusion (`√(x²+ε)`, no exact zeros) with a solver that
produces an **exactly sparse change graph** `Δ = W¹ − W⁰`. We do this with
consensus ADMM: the smooth data + acyclicity terms are solved by L-BFGS, and the
nonsmooth sparsity + fusion penalties are applied by a closed-form fused-lasso
proximal operator. This is the directed, regime-conditioned, heavy-tailed
analogue of the fused graphical lasso (Danaher et al. 2014), built on the
NOTEARS/DYNOTEARS acyclicity machinery (Zheng et al. 2018; Pamfil et al. 2020)
and the K=2 fused-lasso closed form (Friedman et al. 2007).

Scope for this iteration: **K = 2** regimes (ordinary `k=0`, event `k=1`),
**fixed `ν`**, **fixed pooled per-variable scales `σ_j`**. K>2, learned `ν`, and
per-regime scales are out of scope (noted in §10). The existing smoothed solver
is retained as an ablation baseline.

## 2. Notation and model

- Regimes `k ∈ {0, 1}`; `d` variables; lag order `p`; `n_k` observations in
  regime `k`; `N = n_0 + n_1`.
- Per regime, the lagged design (built by `prepare_lagged_design`):
  - `X^k ∈ ℝ^{n_k×d}` — contemporaneous targets (row `t` = observation `xₜ`).
  - `Y^k_ℓ ∈ ℝ^{n_k×d}`, `ℓ = 1..p` — the series lagged by `ℓ`.
- Parameters `Θ = (W⁰, W¹, A⁰_{1:p}, A¹_{1:p})`:
  - intra-slice `W^k ∈ ℝ^{d×d}`, `diag(W^k) = 0`; entry `W^k[i,j]` is the
    directed edge `i → j` (column `j` is the structural equation for variable `j`).
  - inter-slice `A^k_ℓ ∈ ℝ^{d×d}`.
- Structural form (matches existing code: predicted `X = X W + Σ_ℓ Y_ℓ A_ℓ`):

  ```
  E^k = X^k − X^k W^k − Σ_{ℓ=1}^{p} Y^k_ℓ A^k_ℓ              (residuals, n_k×d)
  ```

- Noise: each residual entry `E^k[t,j] ~ Student-t(ν, scale σ_j)` i.i.d.,
  `ν` fixed, `σ_j` shared across regimes.

## 3. Objective (constrained, composite)

Minimize `F(Θ) = f(Θ) + g(Θ)` subject to acyclicity, where:

**Smooth data term** (per-observation normalized so penalties transfer across
`n`; the large ordinary regime carries proportionally more weight, which is the
borrow-strength mechanism):

```
f(Θ) = (1/N) Σ_{k∈{0,1}} Σ_{t=1}^{n_k} Σ_{j=1}^{d} ℓ( E^k[t,j] ; ν, σ_j )
```

with, at fixed `ν, σ` (additive constants in `ν` dropped — see §9 for the
evaluation-time full density):

```
Student-t:  ℓ(e; ν, σ) = log σ + ((ν+1)/2) · log(1 + e² / (ν σ²))
Gaussian:   ℓ(e; σ)    = log σ + e² / (2 σ²)
```

**Nonsmooth penalty term** (within-regime sparsity + cross-regime fusion):

```
g(Θ) = λ_W (‖W⁰‖₁ + ‖W¹‖₁) + λ_A Σ_ℓ (‖A⁰_ℓ‖₁ + ‖A¹_ℓ‖₁)
     + γ_W ‖W¹ − W⁰‖₁       + γ_A Σ_ℓ ‖A¹_ℓ − A⁰_ℓ‖₁
```

(`‖·‖₁` = sum of absolute entries; `W` diagonals excluded.)

**Constraints:** `h(W^k) = 0` for `k ∈ {0,1}`, with the NOTEARS surrogate
`h(W) = tr(exp(W ∘ W)) − d`, plus `diag(W^k) = 0`.

`f` is smooth in `Θ`. `g` is convex but nonsmooth. `h` is smooth but nonconvex.
We separate these three with ADMM (for `g`) wrapped in an augmented-Lagrangian
outer loop (for `h`).

## 4. Acyclicity via augmented Lagrangian (outer loop)

Per regime, dual `α_k` and penalty `ρ_h`. The smooth part the L-BFGS step sees is

```
S(Θ) = f(Θ) + Σ_{k} [ (ρ_h/2) · h(W^k)² + α_k · h(W^k) ]
```

Outer schedule (NOTEARS-style), iteration `m`:
1. (Approximately) solve the penalized subproblem `min_Θ S(Θ) + g(Θ)` by running
   `T_admm` ADMM sweeps (§5) at the current `(α, ρ_h)`.
2. Let `h_max = max_k |h(W^k_Θ)|` (evaluated on the L-BFGS iterate `Θ`).
3. If `h_max > c · h_prev` (default `c = 0.25`) and `ρ_h < ρ_h_max`:
   `ρ_h ← η · ρ_h` (default `η = 10`). Else: `α_k ← α_k + ρ_h · h(W^k)`,
   `h_prev ← h_max`.
4. Stop when `h_max ≤ h_tol` or `m` reaches `outer_max_iter`.

## 5. ADMM for the nonsmooth penalties (inner)

Introduce penalty copies `Z = (V⁰, V¹, B⁰_{1:p}, B¹_{1:p})` of the same shapes as
`Θ`, with constraint `Θ = Z`. Solve `min_{Θ,Z} S(Θ) + g(Z)` s.t. `Θ = Z`.
Scaled-dual augmented Lagrangian, ADMM penalty `ρ` (distinct from `ρ_h`), scaled
dual `U`:

```
L_ρ(Θ, Z, U) = S(Θ) + g(Z) + (ρ/2) ‖Θ − Z + U‖²_F − (ρ/2) ‖U‖²_F
```

One ADMM sweep:

**(1) Θ-update — smooth, L-BFGS.** `g` is absent here, so this subproblem is
smooth (the only nonconvexity is `h²`, as in current code):

```
Θ ← argmin_Θ  S(Θ) + (ρ/2) ‖Θ − Z + U‖²_F
```

`W^k` is masked to zero diagonal before entering `E^k` and `h`. Gradients via
PyTorch autograd; `torch.optim.LBFGS` with strong-Wolfe line search,
`lbfgs_max_iter` steps. `Z, U` are constants here.

**(2) Z-update — closed-form fused prox (exact zeros).** Separable per
coordinate. For the W-copies, off-diagonal `(i,j)`, with targets
`a^k = Θ_{W^k}[i,j] + U_{W^k}[i,j]`:

```
(V⁰[i,j], V¹[i,j]) = FusedProx2( a⁰, a¹ ; λ_W/ρ , γ_W/ρ )
```

`diag(V^k) = 0`. For the A-copies, same with `(λ_A/ρ, γ_A/ρ)`. (Derivation:
dividing the per-coordinate Z-subproblem `min_z (ρ/2)Σ_k(z^k−a^k)² + λ(|z⁰|+|z¹|)
+ γ|z¹−z⁰|` by `ρ` gives a unit-quadratic fused-lasso problem with
`λ' = λ/ρ, γ' = γ/ρ`.)

**(3) Dual update.** `U ← U + Θ − Z`.

**Residuals / optional ρ adaptation** (Boyd et al. 2011):
`r = ‖Θ − Z‖_F` (primal), `s = ρ ‖Z − Z_prev‖_F` (dual). Optionally
`if r > μ·s: ρ ← τ·ρ`; `elif s > μ·r: ρ ← ρ/τ` (defaults `μ=10, τ=2`), rescaling
`U ← U·(ρ_old/ρ_new)`.

## 6. The fused-lasso prox (K = 2), exact closed form

`FusedProx2(a⁰, a¹; λ', γ')` solves
`min_{v⁰,v¹} ½(v⁰−a⁰)² + ½(v¹−a¹)² + λ'(|v⁰|+|v¹|) + γ'|v¹−v⁰|`.
By the fused-lasso decomposition (Friedman et al. 2007; Danaher et al. 2014, K=2):
**fuse, then soft-threshold.**

```
Step 1 (fuse the difference; the mean (a⁰+a¹)/2 is preserved):
    diff = a¹ − a⁰
    if  diff >  2γ' :  t⁰ = a⁰ + γ' ,  t¹ = a¹ − γ'
    elif diff < −2γ' :  t⁰ = a⁰ − γ' ,  t¹ = a¹ + γ'
    else             :  t⁰ = t¹ = (a⁰ + a¹) / 2

Step 2 (soft-threshold each by λ'):
    v^k = sign(t^k) · max(|t^k| − λ', 0)
```

Derivation of Step 1: write `v^k = m ∓ g/2`, `g = v¹−v⁰`. The mean stationarity
gives `m = (a⁰+a¹)/2`; the difference stationarity gives
`g = (a¹−a⁰) − 2γ'·sign(g)`, i.e. the difference is soft-thresholded by `2γ'`
(capped at 0 when `|a¹−a⁰| ≤ 2γ'`). Step 2 is exact because the fusion solution
and the L1 prox commute for the 1D/K=2 fused lasso. Vectorized elementwise over
the whole matrix. **This is the mechanism that yields exact equality of `W⁰` and
`W¹` (hence exact zeros in `Δ`) on unchanged edges.**

## 7. Full algorithm (beginning to end)

```
Inputs: targets {X^k}, lags {Y^k_ℓ}, loss ∈ {student_t, gaussian}, ν, σ (pooled MAD),
        λ_W, λ_A, γ_W, γ_A, ρ (ADMM), ρ_h init, η, c, ρ_h_max, h_tol,
        outer_max_iter, T_admm, lbfgs_max_iter.

Init: Θ ← small random (diag W = 0); Z ← Θ; U ← 0; α ← 0; ρ_h ← ρ_h_init; h_prev ← ∞.

for m in 1..outer_max_iter:                      # acyclicity augmented-Lagrangian
    repeat T_admm times:                         # ADMM
        Θ ← LBFGS_argmin  S(Θ; α, ρ_h) + (ρ/2)‖Θ − Z + U‖²     # smooth
        Z_prev ← Z
        Z ← FusedProx2 elementwise on (Θ + U) with (λ/ρ, γ/ρ)  # exact prox; diag V = 0
        U ← U + Θ − Z
        (optional) adapt ρ from primal/dual residuals
    h_max ← max_k |h(W^k_Θ)|
    if h_max > c·h_prev and ρ_h < ρ_h_max:  ρ_h ← η·ρ_h
    else:                                    α_k ← α_k + ρ_h·h(W^k_Θ);  h_prev ← h_max
    if h_max ≤ h_tol: break

Post-process: report W^k ← V^k (exactly sparse), A^k_ℓ ← B^k_ℓ.
              Δ_W ← V¹ − V⁰; Δ_A_ℓ ← B¹_ℓ − B⁰_ℓ (exactly sparse).
              Record h(V^k), final primal/dual residuals, objective history.
              If h(V^k) > h_tol (consensus gap), apply a final hard-threshold to
              tiny |V^k| entries and re-check acyclicity (DAG-projection guard).
```

## 8. Outputs

`FitResult` extended with: `Delta_W`, `Delta_A` (the sparse change graphs),
per-outer `history` rows including `loss`, `max_h`, `primal_res`, `dual_res`,
`rho`, `rho_h`. Existing fields (`W`, `A`, `history`) preserved.

## 9. Held-out likelihood note (deferred, but fixed in spec)

For cross-model comparison (t vs Gaussian) the dropped constants matter. The full
Student-t per-element negative log density is
`ℓ_full = log σ + ½log(νπ) + log Γ(ν/2) − log Γ((ν+1)/2) + ((ν+1)/2) log(1+e²/(νσ²))`.
A separate evaluation function will use `ℓ_full`; training at fixed `ν, σ` may use
the constant-free form in §3. (Tracked as P1-D.)

## 10. Out of scope (this iteration)

- **K > 2:** the §6 closed form is K=2 only; K>2 needs the general
  fused-lasso-signal-approximator prox or pairwise ADMM (reference regime
  `Δ^k = W^k − W^ord`). Deferred.
- **Learned `ν`, per-regime `σ_j`** (P1-E). The pooled-MAD-downweights-the-event
  -regime concern is logged separately.
- **Option D** (adaptive / non-convex penalty on `Δ`): added only if C shows the
  fusion benefit.

## 11. Module structure

- `frtdbn/prox.py` — `fused_lasso_prox_pair(a0, a1, lam, gam)` (vectorized,
  NumPy), `soft_threshold(x, t)`. Pure functions.
- `frtdbn/model.py` — add `FitConfig.solver ∈ {"lbfgs_smooth", "admm"}` (default
  `"lbfgs_smooth"`), `FitConfig.rho`, `T_admm`, residual settings; add the ADMM
  branch reusing `_acyclicity`, `_student_t_nll`, `_gaussian_nll`. **Align the
  existing smooth solver to the §3 normalization** so the smooth-vs-ADMM ablation
  is apples-to-apples (penalties no longer double-divided; acyclicity via the
  same aug-Lag).
- `frtdbn/benchmark.py` — add `solver` as a grid axis so we can report
  smooth-fusion vs prox-fusion on Δ-AUROC.

## 12. Testing plan (TDD)

1. **Prox unit (`tests/test_prox.py`)** — the heart:
   - `γ'=0` ⇒ plain soft-threshold by `λ'`.
   - equal targets, or `|a¹−a⁰| ≤ 2γ'` ⇒ fully fused (`v⁰ = v¹`).
   - `|a¹−a⁰| > 2γ'` ⇒ difference shrunk by exactly `2γ'`, then soft-thresholded.
   - exact zeros produced; vectorized matrix call matches scalar calls.
   - optimality cross-check: prox output beats small random perturbations on the
     scalar objective.
2. **Solver behavior (`tests/test_model.py`)**:
   - `γ_W, γ_A → ∞` ⇒ `‖W¹ − W⁰‖` ≈ 0 (regimes fused) on a synthetic pair.
   - `γ = 0` ⇒ two independent sparse fits (matches per-regime fit within tol).
   - output `W^k` finite, near-acyclic (`h ≤ tol`), `diag = 0`; `Δ` has exact zeros.
   - unknown `solver` raises `ValueError`.
3. **Payoff experiment (not a unit test)** — benchmark: ADMM-prox fusion ≥
   smoothed fusion on Δ-AUROC across seeds; record in `outputs/`.

## 13. Main risk + fallback

ADMM × nonconvex `h²` can diverge if `ρ_h` outruns `ρ`. Mitigations: cap
`ρ_h_max`; report primal/dual residuals + `h` each outer step; **warm-start**
fallback — run `lbfgs_smooth` to near-convergence, init `Θ` from it, then run
ADMM only to sparsify `Δ`. Build with conservative defaults (`T_admm` small,
`ρ` fixed first; enable ρ-adaptation once stable).

## References

NOTEARS (Zheng et al. 2018); DYNOTEARS (Pamfil et al. 2020); fused/group
graphical lasso ADMM (Danaher et al. 2014); fused-lasso decomposition (Friedman
et al. 2007); ADMM (Boyd et al. 2011); var-sortability (Reisach et al. 2021).
