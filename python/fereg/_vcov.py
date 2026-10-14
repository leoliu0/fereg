"""Parsing of pyfixest ``vcov`` / ``vcov_kwargs`` arguments."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["VcovSpec", "VcovTypeNotSupportedError", "parse_vcov", "MAX_CLUSTER_DIMS"]

MAX_CLUSTER_DIMS = 4
# fixest `check_set_cutoff`: miles to km.
_KM_PER_MILE = 1.60934
# fixest `check_set_cutoff`: an integer, an optional space and a unit.
_CUTOFF_RE = re.compile(r"^([0-9]+) ?((m|mi|mil|mile|miles)|(k|km))?$")

_KWARGS_KEYS = ("lag", "time_id", "panel_id", "lat", "lon", "cutoff", "pixel", "distance")
_KWARGS_KEYS_MSG = (
    "The function argument `vcov_kwargs` must be a dictionary with keys 'lag', 'time_id', or "
    "'panel_id' (or fereg's Conley keys 'lat', 'lon', 'cutoff', 'pixel' and 'distance')."
)
_VCOV_STRINGS = ("iid", "hetero", "HC1", "HC2", "HC3", "NW", "DK", "conley")


class VcovTypeNotSupportedError(ValueError):
    """pyfixest ``VcovTypeNotSupportedError``: the variance type is not defined for the model."""


@dataclass(frozen=True)
class VcovSpec:
    """A parsed ``vcov`` request with its ``vcov_kwargs``."""

    kind: str  # core estimator: iid | hetero | hc2 | hc3 | cluster | crv3 | nw | dk | conley
    vcov_type: str  # pyfixest ``_vcov_type``: iid | hetero | CRV | HAC (fereg: conley)
    detail: str  # pyfixest ``_vcov_type_detail``: iid | hetero | HC1-3 | CRV1 | CRV3 | NW | DK (conley)
    cluster: tuple[str, ...] = ()  # cluster dimensions; "a^b" = interaction
    lag: int | None = None  # NW / DK lag; None = pyfixest's default floor(T^0.25)
    time_id: str | None = None
    panel_id: str | None = None
    lat: str | None = None  # Conley coordinates (degrees)
    lon: str | None = None
    cutoff: float | None = None  # Conley cutoff in km; None = fixest's deduced cutoff
    pixel: float = 0.0  # Conley pixel size in km; 0 = none
    spherical: bool = False  # Conley great-circle distance (else fixest's "triangular")

    @property
    def is_clustered(self) -> bool:
        return self.kind in ("cluster", "crv3")

    @property
    def is_hac(self) -> bool:
        return self.kind in ("nw", "dk")

    def columns(self) -> dict[str, str | None]:
        """The time / panel / coordinate columns this estimator reads (None = not used)."""
        hac, conley = self.is_hac, self.kind == "conley"
        return {
            "time": self.time_id if hac else None,
            "panel": self.panel_id if hac else None,
            "lat": self.lat if conley else None,
            "lon": self.lon if conley else None,
        }

    def core_options(self) -> dict:
        """Keyword arguments of the core's ``fit`` / ``FitState.vcov`` for this estimator."""
        return {
            "lag": self.lag if self.is_hac else None,
            "cutoff": self.cutoff,
            "pixel": self.pixel,
            "spherical": self.spherical,
        }


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _kwarg_error(what: str, key: str) -> ValueError:
    return ValueError(
        f"The function argument `vcov_kwargs` must be a dictionary with {what} values for '{key}' "
        "if explicitly provided."
    )


def _parse_cutoff(value) -> tuple[float | None, float]:
    """Conley cutoff in km and the km per unit of ``pixel`` (fixest: pixel shares the cutoff's unit)."""
    if value is None:
        return None, 1.0
    if _is_number(value) and value >= 0:
        return float(value), 1.0
    if isinstance(value, str):
        m = _CUTOFF_RE.match(value)
        if m:
            scale = _KM_PER_MILE if m.group(3) else 1.0
            return float(m.group(1)) * scale, scale
    raise ValueError(
        "The function argument `vcov_kwargs` must have a non-negative number (km) or a string such "
        "as '100km' or '60mi' for 'cutoff'."
    )


def _parse_kwargs(vcov_kwargs: dict | None, columns) -> dict:
    """pyfixest ``_estimation_input_checks`` on ``vcov_kwargs`` plus fereg's Conley keys."""
    if vcov_kwargs is None:
        return {}
    if not isinstance(vcov_kwargs, dict) or not all(k in _KWARGS_KEYS for k in vcov_kwargs):
        raise ValueError(_KWARGS_KEYS_MSG)
    kw = dict(vcov_kwargs)
    if "lag" in kw and not isinstance(kw["lag"], int):
        raise _kwarg_error("integer", "lag")
    for key in ("time_id", "panel_id", "lat", "lon"):
        if key in kw:
            if not isinstance(kw[key], str):
                raise _kwarg_error("string", key)
            if columns is not None and kw[key] not in columns:
                raise ValueError(f"The variable '{kw[key]}' is not in the data.")
    cutoff, scale = _parse_cutoff(kw.get("cutoff"))
    kw["cutoff"] = cutoff
    pixel = kw.get("pixel", 0)
    if not (_is_number(pixel) and pixel >= 0):
        raise _kwarg_error("non-negative numeric", "pixel")
    kw["pixel"] = float(pixel) * scale
    distance = kw.get("distance", "triangular")
    if distance not in ("triangular", "spherical"):
        raise ValueError("The 'distance' of `vcov_kwargs` must be 'triangular' or 'spherical'.")
    kw["spherical"] = distance == "spherical"
    return kw


def parse_vcov(vcov: str | dict[str, str] | None, vcov_kwargs: dict | None = None, columns=None) -> VcovSpec:
    """Parses a pyfixest ``vcov`` (``None`` = iid) and its ``vcov_kwargs``.

    ``columns`` (the data's column names), when given, are checked to contain the
    ``vcov_kwargs`` variables.
    """
    kw = _parse_kwargs(vcov_kwargs, columns)
    opts = {
        "lag": kw.get("lag"),
        "time_id": kw.get("time_id"),
        "panel_id": kw.get("panel_id"),
        "lat": kw.get("lat"),
        "lon": kw.get("lon"),
        "cutoff": kw.get("cutoff"),
        "pixel": kw.get("pixel", 0.0),
        "spherical": kw.get("spherical", False),
    }
    if vcov is None:
        return VcovSpec("iid", "iid", "iid", **opts)
    if isinstance(vcov, dict):
        if len(vcov) != 1:
            raise ValueError("vcov dict must have exactly one key, e.g. {'CRV1': 'firm'}.")
        ((detail, value),) = vcov.items()
        if detail not in ("CRV1", "CRV3"):
            raise ValueError("vcov dict key must be CRV1 or CRV3")
        if not isinstance(value, str):
            raise TypeError("vcov dict value must be a string")
        dims = tuple(v.replace(" ", "") for v in value.split("+"))
        if any(not d for d in dims):
            raise ValueError(f"invalid cluster specification '{value}'.")
        if len(dims) > MAX_CLUSTER_DIMS:
            raise ValueError(f"at most {MAX_CLUSTER_DIMS} cluster dimensions are supported.")
        if len(set(dims)) != len(dims):
            raise ValueError(f"duplicate cluster variable in '{value}'.")
        return VcovSpec("cluster" if detail == "CRV1" else "crv3", "CRV", detail, dims, **opts)
    if not isinstance(vcov, str):
        raise TypeError("arg vcov needs to be a dict, string or None")
    if vcov == "nid":
        # pyfixest `Feols._vcov_nid`: quantile regression only.
        raise NotImplementedError("Only models of type Quantreg support a variance-covariance matrix of type 'nid'.")
    if vcov not in _VCOV_STRINGS:
        raise ValueError(
            "vcov string must be iid, hetero, HC1, HC2, HC3, NW, or DK (or fereg's conley), or for "
            "quantile regression, 'nid'."
        )
    if vcov == "iid":
        return VcovSpec("iid", "iid", "iid", **opts)
    if vcov in ("hetero", "HC1"):
        return VcovSpec("hetero", "hetero", vcov, **opts)
    if vcov in ("HC2", "HC3"):
        return VcovSpec(vcov.lower(), "hetero", vcov, **opts)
    if vcov in ("NW", "DK"):
        # pyfixest `_check_vcov_input`.
        if opts["time_id"] is None:
            raise ValueError("Missing required 'time_id' for NW/DK vcov")
        return VcovSpec(vcov.lower(), "HAC", vcov, **opts)
    if opts["lat"] is None or opts["lon"] is None:
        raise ValueError("Conley inference needs the latitude and longitude columns: vcov_kwargs={'lat': ..., 'lon': ...}.")
    return VcovSpec("conley", "conley", "conley", **opts)
