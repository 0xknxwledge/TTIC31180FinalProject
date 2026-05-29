import numpy as np

from frtdbn.model import FitConfig
from frtdbn.robustness import (
    bh_rejected,
    block_bootstrap_indices,
    edgewise_permutation_test,
    edgewise_pvalues,
    fit_restarts,
    permutation_null_delta_norm,
    resample_lagged_regimes,
    restart_summary,
    stability_selection,
    top_k_jaccard,
    top_k_mask,
)
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def _tiny_data():
    synth = make_regime_pair(d=6, p=1, n_per_regime=20, seed=11)
    targets, lags = [], []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=1)
        targets.append(target)
        lags.append(lagged)
    return targets, lags


def test_top_k_mask_and_jaccard():
    scores = np.array([[1.0, 3.0], [2.0, 0.0]])
    mask = top_k_mask(scores, 2)
    assert mask.sum() == 2
    assert top_k_jaccard([scores, scores], 2) == 1.0


def test_block_bootstrap_and_resample_shapes():
    targets, lags = _tiny_data()
    idx = block_bootstrap_indices(10, 3, np.random.default_rng(0))
    assert idx.shape == (10,)
    bt, bl = resample_lagged_regimes(targets, lags, block_size=3, seed=0)
    assert bt[0].shape == targets[0].shape
    assert bl[0][0].shape == lags[0][0].shape


def test_fit_restarts_and_summary_run():
    targets, lags = _tiny_data()
    cfg = FitConfig(p=1, solver="admm", lbfgs_max_iter=3, outer_max_iter=1, seed=0)
    results = fit_restarts(targets, lags, cfg, seeds=[0, 1])
    summary = restart_summary(results, top_k=3)

    assert summary["n_restarts"] == 2.0
    assert 0.0 <= summary["top_k_jaccard"] <= 1.0


def test_edgewise_pvalues_counts_exceedances():
    obs = np.array([[5.0, 0.0]])
    perms = [np.array([[1.0, 0.0]]), np.array([[2.0, 0.0]]), np.array([[6.0, 0.0]])]
    p = edgewise_pvalues(obs, perms)
    assert np.isclose(p[0, 0], (1 + 1) / (1 + 3))   # only the 6 >= 5
    assert np.isclose(p[0, 1], (1 + 3) / (1 + 3))   # all 0 >= 0


def test_bh_rejected_controls_fdr():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    rej = bh_rejected(p, alpha=0.05)
    assert rej.tolist() == [True, True, False, False, False, False, False, False]
    assert bh_rejected(np.array([0.9, 0.8, 0.7]), alpha=0.05).tolist() == [False, False, False]


def test_edgewise_permutation_test_smoke():
    targets, lags = _tiny_data()
    cfg = FitConfig(p=1, solver="admm", lbfgs_max_iter=2, outer_max_iter=1, seed=0)
    res = edgewise_permutation_test(targets, lags, cfg, n_permutations=2, seed=0, block_size=4)
    assert res["pvalues"].shape == (6, 6)
    assert np.all((res["pvalues"] >= 0) & (res["pvalues"] <= 1))


def test_stability_selection_and_permutation_null_smoke():
    targets, lags = _tiny_data()
    cfg = FitConfig(p=1, solver="admm", lbfgs_max_iter=2, outer_max_iter=1, seed=0)
    stability = stability_selection(targets, lags, cfg, n_bootstrap=2, block_size=4, seed=0)
    null = permutation_null_delta_norm(targets, lags, cfg, n_permutations=2, seed=0, block_size=4)

    assert stability["delta_w_frequency"].shape == (6, 6)
    assert null.shape == (2,)
    assert np.all(np.isfinite(null))
