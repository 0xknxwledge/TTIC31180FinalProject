import numpy as np

from frtdbn.preprocess import standardize_columns


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
