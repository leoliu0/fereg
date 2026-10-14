"""fereg: fast high-dimensional fixed-effects regressions with pyfixest-compatible results.

The estimation itself (fixed-effect absorption, OLS, IV/2SLS, Poisson, logit, variance
matrices, fit statistics) runs in the Rust core; this package parses formulas, builds the
design columns and presents the results with pyfixest 0.60.0 names and conventions.
"""

from ._core import __version__
from ._vcov import VcovTypeNotSupportedError
from .estimation import feglm, feols, fepois, ssc
from .etable import etable
from .results import Fegaussian, Feglm, Feiv, Felogit, Feols, Fepois, FixestMulti
from .summary import summary

__all__ = [
    "__version__",
    "feols",
    "fepois",
    "feglm",
    "ssc",
    "etable",
    "summary",
    "Feols",
    "Feiv",
    "Fepois",
    "Feglm",
    "Felogit",
    "Fegaussian",
    "FixestMulti",
    "VcovTypeNotSupportedError",
]
