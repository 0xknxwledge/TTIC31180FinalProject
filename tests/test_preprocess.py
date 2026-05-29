import numpy as np

from frtdbn.preprocess import rank_gaussianize, robust_scales, rolling_zscore_past, standardize_columns


def test_standardize_columns_yields_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=[5.0, -2.0, 100.0], scale=[1.0, 3.0, 0.1], size=(500, 3))

    z = standardize_columns(x)

    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-9)


def test_standardize_columns_handles_constant_column_without_nan():
    x = np.column_stack([np.ones(10), np.arange(10.0)])

    z = standardize_columns(x)

    assert np.all(np.isfinite(z))
    assert np.allclose(z[:, 0], 0.0)


def test_standardize_columns_preserves_shape():
    x = np.zeros((7, 4))
    assert standardize_columns(x).shape == (7, 4)


def test_robust_scales_returns_positive_vector():
    x = np.column_stack([np.ones(10), np.arange(10.0)])
    scales = robust_scales(x)
    assert scales.shape == (2,)
    assert np.all(scales > 0)


def test_rolling_zscore_past_uses_only_history():
    x = np.arange(8.0).reshape(-1, 1)
    z = rolling_zscore_past(x, window=4, min_periods=2)
    assert np.isnan(z[0, 0])
    assert np.isnan(z[1, 0])
    # At t=2, history is [0, 1], mean=.5, std=.5, so z=(2-.5)/.5=3.
    assert np.isclose(z[2, 0], 3.0)


def test_rank_gaussianize_preserves_order_and_shape():
    x = np.array([[3.0], [1.0], [2.0]])
    z = rank_gaussianize(x)
    assert z.shape == x.shape
    assert list(np.argsort(z[:, 0])) == [1, 2, 0]
