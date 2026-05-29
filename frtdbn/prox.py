"""Proximal operators for the fused-lasso penalties in the ADMM solver.

The K=2 fused lasso has an exact closed form: fuse the difference, then
soft-threshold each coordinate (Friedman et al. 2007; Danaher et al. 2014). This
is what produces exact zeros in the change graph ``Delta = W^1 - W^0``.
"""

from __future__ import annotations

import numpy as np


def soft_threshold(x, t):
    """Elementwise soft-threshold: sign(x) * max(|x| - t, 0)."""

    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def fused_lasso_prox_pair(a0, a1, lam: float, gam: float):
    """Exact prox of the K=2 fused lasso, applied elementwise.

    Solves, for each coordinate independently,
        min_{v0,v1}  1/2 (v0-a0)^2 + 1/2 (v1-a1)^2
                     + lam (|v0| + |v1|) + gam |v1 - v0|.

    Step 1 fuses the difference (the mean is preserved): each value moves toward
    the other by ``gam``, capped when |a1 - a0| <= 2*gam (then both take the
    mean). Step 2 soft-thresholds each by ``lam``. Accepts scalars or arrays.
    """

    a0 = np.asarray(a0, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    diff = a1 - a0

    # Move each toward the other by gam, but never past the shared mean.
    shrink = np.minimum(gam, 0.5 * np.abs(diff))
    step = np.sign(diff) * shrink
    t0 = a0 + step
    t1 = a1 - step

    v0 = soft_threshold(t0, lam)
    v1 = soft_threshold(t1, lam)
    if np.isscalar(a0) or np.ndim(v0) == 0:
        return float(v0), float(v1)
    return v0, v1
