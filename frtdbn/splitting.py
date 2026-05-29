"""Time-ordered splits for out-of-sample evaluation and stationarity checks.

Financial panels are serially dependent, so all splits are *contiguous in time*
(no shuffling): the test set is the most recent block, and stationarity blocks
are contiguous sub-periods.
"""

from __future__ import annotations

import numpy as np


def train_test_split_regimes(
    targets_by_regime: list[np.ndarray],
    lags_by_regime: list[list[np.ndarray]],
    test_fraction: float = 0.2,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[np.ndarray], list[list[np.ndarray]]]:
    """Split each regime by time: earliest rows train, latest `test_fraction` test.

    Target row `t` and its lag rows stay aligned across the split.
    """

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1).")

    train_t: list[np.ndarray] = []
    train_l: list[list[np.ndarray]] = []
    test_t: list[np.ndarray] = []
    test_l: list[list[np.ndarray]] = []
    for target, lags in zip(targets_by_regime, lags_by_regime):
        n = target.shape[0]
        cut = int(round(n * (1.0 - test_fraction)))
        cut = min(max(cut, 1), n - 1)  # keep both sides non-empty
        train_t.append(target[:cut])
        test_t.append(target[cut:])
        train_l.append([lag[:cut] for lag in lags])
        test_l.append([lag[cut:] for lag in lags])
    return train_t, train_l, test_t, test_l


def time_block_indices(n: int, n_blocks: int) -> list[np.ndarray]:
    """Partition range(n) into `n_blocks` contiguous, near-equal index blocks."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if not 1 <= n_blocks <= n:
        raise ValueError("n_blocks must be in [1, n].")
    return [np.array(b) for b in np.array_split(np.arange(n), n_blocks)]
