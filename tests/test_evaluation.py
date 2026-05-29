import numpy as np

from frtdbn.evaluation import (
    fit_svar_only,
    full_nll,
    gaussian_nll_full,
    residuals_for_regime,
    student_t_nll_full,
    svar_vs_dag_oos,
)
from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.splitting import train_test_split_regimes
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def _prep(synth, p=1):
    targets, lags = [], []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=p)
        targets.append(target)
        lags.append(lagged)
    return targets, lags


def test_full_nlls_are_finite_and_different():
    resid = np.array([[0.0, 1.0], [2.0, -1.0]])
    scales = np.array([1.0, 2.0])

    t_nll = student_t_nll_full(resid, scales, nu=5.0)
    g_nll = gaussian_nll_full(resid, scales)

    assert np.isfinite(t_nll)
    assert np.isfinite(g_nll)
    assert not np.isclose(t_nll, g_nll)


def test_fit_svar_only_returns_zero_W_and_full_nll():
    synth = make_regime_pair(d=6, p=1, n_per_regime=30, seed=4)
    targets, lags = _prep(synth)
    svar = fit_svar_only(targets, lags)
    assert np.allclose(svar.W[0], 0.0)
    assert np.isfinite(full_nll(targets, lags, svar, loss="gaussian"))


def test_residuals_for_fitted_model_have_expected_shape():
    synth = make_regime_pair(d=6, p=1, n_per_regime=25, seed=5)
    targets, lags = _prep(synth)
    fit = fit_fr_tdbn(targets, lags, FitConfig(p=1, lbfgs_max_iter=5, outer_max_iter=1, seed=5))
    resid = residuals_for_regime(targets[0], lags[0], fit.W[0], fit.A[0])
    assert resid.shape == targets[0].shape


def test_svar_vs_dag_oos_is_out_of_sample_and_consistent():
    synth = make_regime_pair(d=8, p=1, n_per_regime=120, seed=6)
    targets, lags = _prep(synth)
    tr_t, tr_l, te_t, te_l = train_test_split_regimes(targets, lags, test_fraction=0.25)
    cfg = FitConfig(p=1, solver="admm", lbfgs_max_iter=10, outer_max_iter=2, seed=6)

    res = svar_vs_dag_oos(tr_t, tr_l, te_t, te_l, cfg, loss="gaussian")

    for key in ("dag_test_nll", "svar_test_nll", "dag_minus_svar", "dag_better"):
        assert key in res
    assert np.isfinite(res["dag_test_nll"]) and np.isfinite(res["svar_test_nll"])
    assert np.isclose(res["dag_minus_svar"], res["dag_test_nll"] - res["svar_test_nll"])
    assert res["dag_better"] == (res["dag_minus_svar"] < 0)
