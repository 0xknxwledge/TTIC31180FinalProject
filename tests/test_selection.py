import math

import numpy as np
import pytest

from frtdbn.evaluation import gaussian_nll_full
from frtdbn.model import FitConfig, FitResult
from frtdbn.selection import bic_score, count_nonzero_params, select_hyperparameters
from frtdbn.splitting import train_test_split_regimes
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def _prep(synth, p=1):
    targets, lags = [], []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=p)
        targets.append(target)
        lags.append(lagged)
    return targets, lags


def test_count_nonzero_params_ignores_W_diagonal_but_counts_A():
    W = [np.zeros((3, 3)), np.zeros((3, 3))]
    W[1][0, 1] = 0.5  # one off-diagonal contemporaneous edge -> counts
    W[1][2, 2] = 9.0  # diagonal of W is structurally zero -> ignored
    A = [np.zeros((3, 3)), np.zeros((3, 3))]
    A[1][0, 0] = 0.3  # lagged self-edge is a real free parameter -> counts
    A[1][1, 2] = 0.2
    res = FitResult(W=W, A=A)

    assert count_nonzero_params(res) == 1 + 2


def test_bic_score_is_two_nll_when_no_free_parameters():
    # W=A=0 -> residuals equal the targets; k=0 -> no parameter penalty.
    targets = [np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])]
    lags = [[np.zeros((3, 2))]]
    res = FitResult(W=[np.zeros((2, 2))], A=[np.zeros((2, 2))])
    scales = np.array([1.0, 1.0])

    bic = bic_score(targets, lags, res, loss="gaussian", scales=scales)

    assert np.isclose(bic, 2.0 * gaussian_nll_full(targets[0], scales))


def test_bic_score_adds_log_n_penalty_per_nonzero_parameter():
    targets = [np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])]
    lags = [[np.zeros((3, 2))]]
    scales = np.array([1.0, 1.0])
    dense = FitResult(W=[np.array([[0.0, 0.1], [0.0, 0.0]])], A=[np.zeros((2, 2))])

    bic = bic_score(targets, lags, dense, loss="gaussian", scales=scales)

    resid = targets[0] - targets[0] @ dense.W[0]
    expected = 2.0 * gaussian_nll_full(resid, scales) + 1 * math.log(3)
    assert np.isclose(bic, expected)


def test_select_hyperparameters_returns_argmin_and_full_trace():
    synth = make_regime_pair(d=6, p=1, n_per_regime=80, seed=3)
    targets, lags = _prep(synth)
    tr_t, tr_l, te_t, te_l = train_test_split_regimes(targets, lags, test_fraction=0.25)
    base = FitConfig(p=1, solver="admm", lbfgs_max_iter=10, outer_max_iter=2, seed=3)

    out = select_hyperparameters(
        tr_t, tr_l, te_t, te_l, base,
        lambdas=(0.01, 0.2), gammas=(0.0, 0.1), criterion="heldout_nll", loss="gaussian",
    )

    assert len(out["trace"]) == 4
    best = min(out["trace"], key=lambda r: r["score"])
    assert (out["lambda"], out["gamma"]) == (best["lambda"], best["gamma"])
    assert np.isclose(out["best_score"], best["score"])


def test_select_hyperparameters_supports_bic_criterion():
    synth = make_regime_pair(d=6, p=1, n_per_regime=80, seed=7)
    targets, lags = _prep(synth)
    tr_t, tr_l, te_t, te_l = train_test_split_regimes(targets, lags, test_fraction=0.25)
    base = FitConfig(p=1, solver="admm", lbfgs_max_iter=10, outer_max_iter=2, seed=7)

    out = select_hyperparameters(
        tr_t, tr_l, te_t, te_l, base,
        lambdas=(0.01, 0.2), gammas=(0.0, 0.1), criterion="bic", loss="gaussian",
    )

    assert out["criterion"] == "bic"
    assert np.isfinite(out["best_score"])
    best = min(out["trace"], key=lambda r: r["score"])
    assert (out["lambda"], out["gamma"]) == (best["lambda"], best["gamma"])


def test_select_hyperparameters_rejects_unknown_criterion():
    synth = make_regime_pair(d=6, p=1, n_per_regime=40, seed=1)
    targets, lags = _prep(synth)
    tr_t, tr_l, te_t, te_l = train_test_split_regimes(targets, lags, test_fraction=0.25)
    base = FitConfig(p=1, solver="admm", lbfgs_max_iter=6, outer_max_iter=1, seed=1)

    with pytest.raises(ValueError):
        select_hyperparameters(tr_t, tr_l, te_t, te_l, base, lambdas=(0.1,), gammas=(0.0,), criterion="bogus")
