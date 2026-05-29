import numpy as np

from frtdbn.metrics import acyclicity_numpy, graph_diagnostics, var_sortability


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


def test_acyclicity_numpy_is_zero_for_dag_positive_for_cycle():
    dag = _chain_W()
    cyc = dag.copy()
    cyc[2, 0] = 1.0

    assert np.isclose(acyclicity_numpy(dag), 0.0)
    assert acyclicity_numpy(cyc) > 0.0


def test_graph_diagnostics_reports_h_and_delta_counts():
    W = [np.zeros((3, 3)), np.zeros((3, 3))]
    W[1][0, 1] = 0.2
    A = [np.zeros((3, 3)), np.ones((3, 3))]

    diag = graph_diagnostics(W, A, threshold=0.1)

    assert diag["h_returned_max"] == 0.0
    assert diag["delta_w_nnz"] == 1.0
    assert diag["delta_a_nnz"] == 9.0
