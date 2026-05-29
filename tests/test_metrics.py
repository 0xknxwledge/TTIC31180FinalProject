import numpy as np

from frtdbn.metrics import var_sortability


def _chain_W():
    """Directed chain 0 -> 1 -> 2 (edge i->j stored at W[i, j])."""
    W = np.zeros((3, 3))
    W[0, 1] = 1.0
    W[1, 2] = 1.0
    return W


def _scaled_columns(scales):
    """Three zero-mean columns with population variances equal to scales**2."""
    pattern = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # mean 0, var 1
    return np.stack([s * pattern for s in scales], axis=1)


def test_var_sortability_is_one_when_variance_increases_along_edges():
    X = _scaled_columns([1.0, 2.0, 3.0])  # var 1 < 4 < 9 down the chain
    assert var_sortability(X, _chain_W()) == 1.0


def test_var_sortability_is_zero_when_variance_decreases_along_edges():
    X = _scaled_columns([3.0, 2.0, 1.0])  # var 9 > 4 > 1 down the chain
    assert var_sortability(X, _chain_W()) == 0.0


def test_var_sortability_is_half_for_standardized_equal_variance_data():
    X = _scaled_columns([1.0, 1.0, 1.0])  # all variances tie
    assert var_sortability(X, _chain_W()) == 0.5
