import numpy as np

from frtdbn.splitting import (
    split_panel_by_regime,
    time_block_indices,
    train_test_split_regimes,
)


def _regimes():
    rng = np.random.default_rng(0)
    targets = [rng.normal(size=(100, 4)), rng.normal(size=(50, 4))]
    lags = [[t.copy()] for t in targets]  # one lag matrix per regime
    return targets, lags


def test_train_test_split_is_time_ordered_and_contiguous():
    targets, lags = _regimes()
    tr_t, tr_l, te_t, te_l = train_test_split_regimes(targets, lags, test_fraction=0.2)

    # regime 0: 100 rows -> 80 train / 20 test; train is the EARLIER rows
    assert tr_t[0].shape[0] == 80 and te_t[0].shape[0] == 20
    assert tr_t[1].shape[0] == 40 and te_t[1].shape[0] == 10
    assert np.array_equal(tr_t[0], targets[0][:80])
    assert np.array_equal(te_t[0], targets[0][80:])
    # lag rows stay aligned with their targets
    assert np.array_equal(te_l[0][0], lags[0][0][80:])


def test_train_test_split_rejects_bad_fraction():
    targets, lags = _regimes()
    for frac in (0.0, 1.0, -0.1):
        try:
            train_test_split_regimes(targets, lags, test_fraction=frac)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_split_panel_by_regime_partitions_rows_by_label():
    target = np.arange(12).reshape(6, 2).astype(float)
    lags = [target + 100.0]
    labels = np.array([0, 1, 0, 1, 0, 1])

    tbr, lbr = split_panel_by_regime(target, lags, labels)

    assert tbr[0].shape == (3, 2) and tbr[1].shape == (3, 2)   # [ordinary, event]
    assert np.array_equal(tbr[1], target[[1, 3, 5]])
    assert np.array_equal(lbr[0][0], (target + 100.0)[[0, 2, 4]])  # lag rows stay aligned


def test_time_block_indices_partition_is_complete_and_contiguous():
    blocks = time_block_indices(10, 3)
    assert len(blocks) == 3
    assert np.array_equal(np.concatenate(blocks), np.arange(10))  # complete, ordered
    for b in blocks:
        assert np.array_equal(b, np.arange(b[0], b[-1] + 1))      # contiguous
