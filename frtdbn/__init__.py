"""Fused-regime Student-t dynamic Bayesian network MVP."""

from frtdbn.model import FitConfig, FitResult, fit_fr_tdbn
from frtdbn.synthetic import SyntheticDBN, make_regime_pair, prepare_lagged_design

__all__ = [
    "FitConfig",
    "FitResult",
    "SyntheticDBN",
    "fit_fr_tdbn",
    "make_regime_pair",
    "prepare_lagged_design",
]
