import numpy as np

from frtdbn.synthetic import (
    _companion_radius,
    _reduced_form_coefficients,
    make_regime_pair,
)

TOL = 1e-8


def test_change_masks_equal_realized_parameter_differences():
    """Ground-truth Delta must be the actual W/A difference, not the intent."""
    s = make_regime_pair(d=12, p=2, n_per_regime=20, change_edges=4, seed=0)

    realized_w = np.abs(s.W[1] - s.W[0]) > TOL
    assert np.array_equal(s.changed_W, realized_w)

    diff_a = np.stack([np.abs(a1 - a0) for a1, a0 in zip(s.A[1], s.A[0])])
    realized_a = (diff_a > TOL).any(axis=0)
    assert np.array_equal(s.changed_A, realized_a)


def test_change_graph_is_sparse():
    s = make_regime_pair(d=12, p=1, n_per_regime=20, change_edges=4, seed=0)
    assert 0 < int(s.changed_W.sum()) <= 12
    assert int(s.changed_A.sum()) <= 12


def test_delta_includes_additions_and_reductions():
    s = make_regime_pair(
        d=16, p=1, n_per_regime=20, change_edges=6, seed=1,
        change_types=("add", "remove", "reweight"),
    )
    w0, w1 = np.abs(s.W[0]), np.abs(s.W[1])
    added = (w0 <= TOL) & (w1 > TOL)
    reduced = (w0 > TOL) & (w1 < w0 - TOL)
    assert added.any()
    assert reduced.any()


def test_both_regimes_are_stable():
    s = make_regime_pair(d=12, p=2, n_per_regime=20, change_edges=4, seed=2)
    for W, A in zip(s.W, s.A):
        radius = _companion_radius(_reduced_form_coefficients(W, A))
        assert radius <= 0.95


def test_both_regime_intra_slice_graphs_stay_acyclic():
    s = make_regime_pair(d=12, p=1, n_per_regime=20, change_edges=4, seed=3)
    for W in s.W:
        assert np.allclose(np.tril(W), 0.0)


def test_supports_imbalanced_regime_sizes():
    s = make_regime_pair(d=12, p=1, n_per_regime=200, n_event=20, seed=0)
    assert s.X[0].shape[0] == 200 + 1   # ordinary
    assert s.X[1].shape[0] == 20 + 1    # event (small)


def test_defaults_to_balanced_regime_sizes():
    s = make_regime_pair(d=12, p=2, n_per_regime=50, seed=0)
    assert s.X[0].shape[0] == s.X[1].shape[0] == 50 + 2
