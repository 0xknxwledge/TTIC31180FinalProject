import numpy as np
import pytest
import torch

from frtdbn.model import FitConfig, _adaptive_weights, _gaussian_nll, fit_fr_tdbn
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def test_gaussian_nll_matches_closed_form():
    resid = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    scales = torch.tensor([1.0, 2.0], dtype=torch.float64)

    got = float(_gaussian_nll(resid, scales))
    expected = float((0.5 * (resid / scales) ** 2 + torch.log(scales)).sum())

    assert np.isclose(got, expected)


def test_fit_accepts_gaussian_loss_and_returns_finite_graphs():
    synth = make_regime_pair(d=8, p=1, n_per_regime=40, change_edges=2, seed=7)
    targets, lags = [], []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=1)
        targets.append(target)
        lags.append(lagged)

    cfg = FitConfig(p=1, loss="gaussian", lbfgs_max_iter=10, outer_max_iter=2, seed=7)
    result = fit_fr_tdbn(targets, lags, cfg)

    assert result.W[0].shape == (8, 8)
    assert all(np.all(np.isfinite(w)) for w in result.W)
    assert all(np.all(np.isfinite(a)) for a in result.A)


def test_fit_rejects_unknown_loss():
    synth = make_regime_pair(d=6, p=1, n_per_regime=20, change_edges=1, seed=1)
    targets, lags = [], []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=1)
        targets.append(target)
        lags.append(lagged)

    cfg = FitConfig(p=1, loss="laplace", lbfgs_max_iter=5, outer_max_iter=1)
    with pytest.raises(ValueError):
        fit_fr_tdbn(targets, lags, cfg)


def _prep(synth, p):
    targets, lags = [], []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=p)
        targets.append(target)
        lags.append(lagged)
    return targets, lags


def test_admm_large_gamma_fuses_regimes():
    synth = make_regime_pair(d=8, p=1, n_per_regime=40, change_edges=3, seed=5)
    targets, lags = _prep(synth, 1)
    cfg = FitConfig(
        p=1, solver="admm", lambda_w=0.0, lambda_a=0.0, gamma_w=1e3, gamma_a=1e3,
        lbfgs_max_iter=15, outer_max_iter=3, t_admm=4, seed=5,
    )
    res = fit_fr_tdbn(targets, lags, cfg)
    assert np.max(np.abs(res.W[1] - res.W[0])) < 1e-6
    assert np.allclose(res.Delta_W, 0.0)


def test_admm_zero_gamma_allows_regime_differences():
    synth = make_regime_pair(d=8, p=1, n_per_regime=60, change_edges=3, seed=6)
    targets, lags = _prep(synth, 1)
    cfg = FitConfig(
        p=1, solver="admm", lambda_w=0.02, lambda_a=0.02, gamma_w=0.0, gamma_a=0.0,
        lbfgs_max_iter=20, outer_max_iter=4, t_admm=4, seed=6,
    )
    res = fit_fr_tdbn(targets, lags, cfg)
    assert np.max(np.abs(res.W[1] - res.W[0])) > 1e-3


def test_admm_outputs_acyclic_sparse_and_finite():
    synth = make_regime_pair(d=8, p=1, n_per_regime=50, change_edges=3, seed=7)
    targets, lags = _prep(synth, 1)
    cfg = FitConfig(
        p=1, solver="admm", lambda_w=0.05, lambda_a=0.05, gamma_w=0.05, gamma_a=0.05,
        lbfgs_max_iter=20, outer_max_iter=6, t_admm=3, seed=7,
    )
    res = fit_fr_tdbn(targets, lags, cfg)
    d = res.W[0].shape[0]
    offdiag = ~np.eye(d, dtype=bool)
    for W in res.W:
        assert np.all(np.isfinite(W))
        assert np.allclose(np.diag(W), 0.0)
    # exact off-diagonal zeros are the whole point of the prox (vs smooth-L1)
    assert int((res.W[0][offdiag] == 0.0).sum()) > 0
    assert int((res.Delta_W[offdiag] == 0.0).sum()) > 0
    assert res.history[-1]["max_h"] < 1e-1


def test_fit_rejects_unknown_solver():
    synth = make_regime_pair(d=6, p=1, n_per_regime=20, change_edges=1, seed=1)
    targets, lags = _prep(synth, 1)
    cfg = FitConfig(p=1, solver="newton", lbfgs_max_iter=5, outer_max_iter=1)
    with pytest.raises(ValueError):
        fit_fr_tdbn(targets, lags, cfg)


def test_adaptive_weights_downweight_large_pilot_changes():
    delta = np.array([[0.0, 1.0], [0.5, 0.0]])
    w = _adaptive_weights(delta, gamma0=0.1, eps=0.05)
    assert np.isclose(w[0, 0], 0.1 / 0.05)         # no pilot change -> max penalty
    assert np.isclose(w[0, 1], 0.1 / (1.0 + 0.05))  # large change -> small penalty
    assert w[0, 1] < w[1, 0] < w[0, 0]             # bigger change -> smaller penalty


def test_admm_adaptive_runs_sparse_acyclic_finite():
    synth = make_regime_pair(d=8, p=1, n_per_regime=80, n_event=20, change_edges=3, seed=8)
    targets, lags = _prep(synth, 1)
    cfg = FitConfig(
        p=1, solver="admm", fusion="adaptive",
        lambda_w=0.04, lambda_a=0.04, gamma_w=0.08, gamma_a=0.08,
        lbfgs_max_iter=20, outer_max_iter=5, t_admm=3, seed=8,
    )
    res = fit_fr_tdbn(targets, lags, cfg)
    d = res.W[0].shape[0]
    offdiag = ~np.eye(d, dtype=bool)
    for W in res.W:
        assert np.all(np.isfinite(W))
        assert np.allclose(np.diag(W), 0.0)
    assert int((res.W[0][offdiag] == 0.0).sum()) > 0
    assert res.Delta_W is not None
    assert res.history[-1]["max_h"] < 1e-1


def test_fit_rejects_unknown_fusion():
    synth = make_regime_pair(d=6, p=1, n_per_regime=20, change_edges=1, seed=1)
    targets, lags = _prep(synth, 1)
    cfg = FitConfig(p=1, solver="admm", fusion="bogus", lbfgs_max_iter=5, outer_max_iter=1)
    with pytest.raises(ValueError):
        fit_fr_tdbn(targets, lags, cfg)


def test_fit_rejects_unknown_adaptive_pilot():
    synth = make_regime_pair(d=6, p=1, n_per_regime=20, change_edges=1, seed=1)
    targets, lags = _prep(synth, 1)
    cfg = FitConfig(
        p=1, solver="admm", fusion="adaptive", adaptive_pilot="ols",
        lbfgs_max_iter=5, outer_max_iter=1,
    )
    with pytest.raises(ValueError):
        fit_fr_tdbn(targets, lags, cfg)
