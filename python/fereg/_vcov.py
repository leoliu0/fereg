"""Parsing of pyfixest ``vcov`` arguments."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["VcovSpec", "parse_vcov", "MAX_CLUSTER_DIMS"]

MAX_CLUSTER_DIMS = 4


@dataclass(frozen=True)
class VcovSpec:
    """A parsed ``vcov`` request."""

    kind: str  # core estimator: "iid" | "hetero" | "cluster"
    vcov_type: str  # pyfixest ``_vcov_type``: "iid" | "hetero" | "CRV"
    detail: str  # pyfixest ``_vcov_type_detail``: "iid" | "hetero" | "HC1" | "CRV1"
    cluster: tuple[str, ...] = ()  # cluster dimensions; "a^b" = interaction

    @property
    def is_clustered(self) -> bool:
        return self.kind == "cluster"


_UNSUPPORTED = {
    "HC2": "HC2 inference is not supported by fereg.",
    "HC3": "HC3 inference is not supported by fereg.",
    "CRV3": "CRV3 inference is not supported by fereg.",
    "NW": "Newey-West (HAC) inference is not supported by fereg.",
    "DK": "Driscoll-Kraay (HAC) inference is not supported by fereg.",
    "nid": "nid inference is not supported by fereg.",
}


def parse_vcov(vcov: str | dict[str, str] | None) -> VcovSpec:
    """Parses ``None`` (iid), ``"iid"``, ``"hetero"``/``"HC1"`` or ``{"CRV1": "a+b"}``."""
    if vcov is None:
        return VcovSpec("iid", "iid", "iid")
    if isinstance(vcov, dict):
        if len(vcov) != 1:
            raise ValueError("vcov dict must have exactly one key, e.g. {'CRV1': 'firm'}.")
        ((detail, value),) = vcov.items()
        if detail in _UNSUPPORTED:
            raise NotImplementedError(_UNSUPPORTED[detail])
        if detail != "CRV1":
            raise ValueError(f"vcov dict key must be 'CRV1', got '{detail}'.")
        if not isinstance(value, str):
            raise TypeError("the cluster variable(s) of vcov must be given as a string, e.g. 'a+b'.")
        dims = tuple(v.replace(" ", "") for v in value.split("+"))
        if any(not d for d in dims):
            raise ValueError(f"invalid cluster specification '{value}'.")
        if len(dims) > MAX_CLUSTER_DIMS:
            raise ValueError(f"at most {MAX_CLUSTER_DIMS} cluster dimensions are supported.")
        if len(set(dims)) != len(dims):
            raise ValueError(f"duplicate cluster variable in '{value}'.")
        return VcovSpec("cluster", "CRV", "CRV1", dims)
    if isinstance(vcov, str):
        if vcov == "iid":
            return VcovSpec("iid", "iid", "iid")
        if vcov in ("hetero", "HC1"):
            return VcovSpec("hetero", "hetero", vcov)
        if vcov in _UNSUPPORTED:
            raise NotImplementedError(_UNSUPPORTED[vcov])
        raise ValueError(
            f"vcov '{vcov}' is not supported; use 'iid', 'hetero', 'HC1' or {{'CRV1': 'clustervar'}}."
        )
    raise TypeError("arg vcov needs to be a dict, string or None")
