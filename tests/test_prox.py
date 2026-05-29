import numpy as np

from frtdbn.prox import fused_lasso_prox_pair, soft_threshold


def _objective(v0, v1, a0, a1, lam, gam):
    return (
        0.5 * (v0 - a0) ** 2
        + 0.5 * (v1 - a1) ** 2
        + lam * (abs(v0) + abs(v1))
        + gam * abs(v1 - v0)
    )


def test_soft_threshold_shrinks_toward_zero():
    assert soft_threshold(3.0, 1.0) == 2.0
    assert soft_threshold(-2.0, 1.0) == -1.0
    assert soft_threshold(0.5, 1.0) == 0.0


def test_gamma_zero_reduces_to_soft_threshold():
    v0, v1 = fused_lasso_prox_pair(3.0, -2.0, lam=1.0, gam=0.0)
    assert np.isclose(v0, 2.0)
    assert np.isclose(v1, -1.0)


def test_small_difference_fully_fuses_to_the_mean():
    # |a1 - a0| = 1 <= 2*gam = 2  ->  fused to mean 1.5, then no L1
    v0, v1 = fused_lasso_prox_pair(1.0, 2.0, lam=0.0, gam=1.0)
    assert np.isclose(v0, 1.5)
    assert np.isclose(v1, 1.5)


def test_large_difference_is_shrunk_by_exactly_two_gamma():
    # diff = 5 > 2*gam = 2  ->  move each toward the other by gam=1
    v0, v1 = fused_lasso_prox_pair(0.0, 5.0, lam=0.0, gam=1.0)
    assert np.isclose(v0, 1.0)
    assert np.isclose(v1, 4.0)
    assert np.isclose((v1 - v0), 5.0 - 2.0)


def test_negative_difference_handled_symmetrically():
    v0, v1 = fused_lasso_prox_pair(5.0, 0.0, lam=0.0, gam=1.0)
    assert np.isclose(v0, 4.0)
    assert np.isclose(v1, 1.0)


def test_produces_exact_zeros():
    v0, v1 = fused_lasso_prox_pair(0.3, -0.2, lam=0.5, gam=0.0)
    assert v0 == 0.0
    assert v1 == 0.0


def test_vectorized_matches_scalar():
    rng = np.random.default_rng(0)
    a0 = rng.normal(size=(5, 5))
    a1 = rng.normal(size=(5, 5))
    V0, V1 = fused_lasso_prox_pair(a0, a1, lam=0.2, gam=0.3)
    for i in range(5):
        for j in range(5):
            s0, s1 = fused_lasso_prox_pair(a0[i, j], a1[i, j], lam=0.2, gam=0.3)
            assert np.isclose(V0[i, j], s0)
            assert np.isclose(V1[i, j], s1)


def test_per_edge_gamma_array_fuses_selectively():
    # edge 0 gets no fusion; edge 1 gets strong fusion (gam >= |diff|/2 -> fuse to mean)
    a0 = np.array([0.0, 0.0])
    a1 = np.array([1.0, 1.0])
    gam = np.array([0.0, 1.0])
    v0, v1 = fused_lasso_prox_pair(a0, a1, lam=0.0, gam=gam)
    assert np.isclose(v0[0], 0.0) and np.isclose(v1[0], 1.0)   # unfused
    assert np.isclose(v0[1], 0.5) and np.isclose(v1[1], 0.5)   # fused to mean


def test_prox_minimizes_its_objective():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a0, a1 = rng.normal(scale=2.0, size=2)
        lam, gam = rng.uniform(0.0, 1.0, size=2)
        v0, v1 = fused_lasso_prox_pair(a0, a1, lam=lam, gam=gam)
        best = _objective(v0, v1, a0, a1, lam, gam)
        for _ in range(200):
            p0 = v0 + rng.normal(scale=0.05)
            p1 = v1 + rng.normal(scale=0.05)
            assert _objective(p0, p1, a0, a1, lam, gam) >= best - 1e-9
