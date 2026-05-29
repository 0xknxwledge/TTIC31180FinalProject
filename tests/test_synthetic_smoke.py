from frtdbn.model import FitConfig, fit_fr_tdbn
from frtdbn.synthetic import make_regime_pair, prepare_lagged_design


def test_fixed_nu_fr_tdbn_smoke_runs():
    synth = make_regime_pair(d=8, p=1, n_per_regime=30, change_edges=2, seed=123)
    targets = []
    lags = []
    for x in synth.X:
        target, lagged = prepare_lagged_design(x, p=1)
        targets.append(target)
        lags.append(lagged)

    cfg = FitConfig(
        p=1,
        lambda_w=0.05,
        lambda_a=0.05,
        gamma_w=0.05,
        gamma_a=0.05,
        lbfgs_max_iter=5,
        outer_max_iter=2,
        seed=123,
    )
    result = fit_fr_tdbn(targets, lags, cfg)

    assert len(result.W) == 2
    assert result.W[0].shape == synth.W[0].shape
    assert result.A[0].shape == (8, 8)
    assert result.history
