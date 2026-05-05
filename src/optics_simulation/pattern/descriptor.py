"""Gaussian dimple pattern descriptor (dict <-> dataclass).

Converts a :class:`GaussianDimplePattern` to a JSON-compatible dict
and back. The descriptor is the reproducibility / report / pre-export
artifact between pattern generation and downstream consumers (mesh
displacement, STL export, optimization history, report generation).

This module performs **no file I/O**. ``json.dumps`` /
``yaml.safe_dump`` are caller responsibilities. The descriptor is a
plain Python dict containing only ``int`` / ``float`` / ``str`` /
``None`` / ``list`` / ``dict`` — never numpy arrays or numpy scalar
types — so any caller-supplied serializer can consume it.

depth_field is **not** stored in the descriptor; reconstruction
recomputes it from ``dimples`` and ``resolution`` via
:func:`evaluate_gaussian_dimple_field` with ``clip=True``.

Out of scope
------------
JSON / YAML file writing, mesh displacement, patterned STL or CAD
export, pattern optimization, residual hotspot updates, thermal
modeling, visualization, config wiring, actual PET STL handling,
Open3D backends, and schema versioning are intentionally not
implemented here.
"""
from __future__ import annotations

import math
from typing import Any

from optics_simulation.pattern.gaussian import (
    GaussianDimple,
    GaussianDimplePattern,
    PatternError,
    _validate_dimple,
    _validate_resolution,
    evaluate_gaussian_dimple_field,
)


_PATTERN_FAMILY = "gaussian_dimple"
_COORDINATE_SYSTEM = "normalized_cylindrical_uv"
_DEPTH_FIELD_TYPE = "normalized_eta"

_REQUIRED_KEYS = (
    "pattern_family",
    "coordinate_system",
    "depth_field_type",
    "resolution",
    "max_depth",
    "seed",
    "dimple_count",
    "dimples",
)
_REQUIRED_DIMPLE_KEYS = (
    "center_u",
    "center_v",
    "amplitude",
    "sigma_u",
    "sigma_v",
)


def _is_strict_int(value: Any) -> bool:
    """True iff value is a Python int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_finite(name: str, value: float, *, context: str = "") -> None:
    if not math.isfinite(float(value)):
        prefix = f"{context}." if context else ""
        raise PatternError(
            f"{prefix}{name} must be finite (no NaN or inf); got {value}"
        )


def gaussian_pattern_to_descriptor(
    pattern: GaussianDimplePattern,
) -> dict:
    """Convert a :class:`GaussianDimplePattern` to a JSON-compatible dict.

    The returned dict contains only Python ``int`` / ``float`` /
    ``str`` / ``None`` / ``list`` / ``dict``; ``depth_field`` is not
    included (reconstruction recomputes it). Raises
    :class:`PatternError` if ``pattern`` is not a
    :class:`GaussianDimplePattern` or if any numeric field is not
    finite.
    """
    if not isinstance(pattern, GaussianDimplePattern):
        raise PatternError(
            f"pattern must be a GaussianDimplePattern; "
            f"got {type(pattern).__name__}"
        )

    nv = int(pattern.resolution[0])
    nu = int(pattern.resolution[1])
    if nv <= 0 or nu <= 0:
        raise PatternError(
            f"resolution entries must be positive integers; "
            f"got (nv, nu)=({nv}, {nu})"
        )

    max_depth = float(pattern.max_depth)
    _check_finite("max_depth", max_depth)
    if max_depth <= 0.0:
        raise PatternError(f"max_depth must be > 0; got {max_depth}")

    if pattern.seed is None:
        seed_out: int | None = None
    elif _is_strict_int(pattern.seed):
        seed_out = int(pattern.seed)
    else:
        raise PatternError(
            f"pattern.seed must be int or None; got {type(pattern.seed).__name__}"
        )

    dimples_list: list[dict] = []
    for i, d in enumerate(pattern.dimples):
        if not isinstance(d, GaussianDimple):
            raise PatternError(
                f"pattern.dimples[{i}] must be a GaussianDimple; "
                f"got {type(d).__name__}"
            )
        cu = float(d.center_u)
        cv = float(d.center_v)
        amp = float(d.amplitude)
        su = float(d.sigma_u)
        sv = float(d.sigma_v)
        for name, val in (
            ("center_u", cu),
            ("center_v", cv),
            ("amplitude", amp),
            ("sigma_u", su),
            ("sigma_v", sv),
        ):
            _check_finite(name, val, context=f"pattern.dimples[{i}]")
        dimples_list.append(
            {
                "center_u": cu,
                "center_v": cv,
                "amplitude": amp,
                "sigma_u": su,
                "sigma_v": sv,
            }
        )

    return {
        "pattern_family": _PATTERN_FAMILY,
        "coordinate_system": _COORDINATE_SYSTEM,
        "depth_field_type": _DEPTH_FIELD_TYPE,
        "resolution": [nv, nu],
        "max_depth": max_depth,
        "seed": seed_out,
        "dimple_count": len(pattern.dimples),
        "dimples": dimples_list,
    }


def _validate_top_level(descriptor: dict) -> None:
    if not isinstance(descriptor, dict):
        raise PatternError(
            f"descriptor must be a dict; got {type(descriptor).__name__}"
        )
    missing = [k for k in _REQUIRED_KEYS if k not in descriptor]
    if missing:
        raise PatternError(f"descriptor missing required keys: {missing}")

    if descriptor["pattern_family"] != _PATTERN_FAMILY:
        raise PatternError(
            f"descriptor.pattern_family must be {_PATTERN_FAMILY!r}; "
            f"got {descriptor['pattern_family']!r}"
        )
    if descriptor["coordinate_system"] != _COORDINATE_SYSTEM:
        raise PatternError(
            f"descriptor.coordinate_system must be {_COORDINATE_SYSTEM!r}; "
            f"got {descriptor['coordinate_system']!r}"
        )
    if descriptor["depth_field_type"] != _DEPTH_FIELD_TYPE:
        raise PatternError(
            f"descriptor.depth_field_type must be {_DEPTH_FIELD_TYPE!r}; "
            f"got {descriptor['depth_field_type']!r}"
        )


def _parse_resolution(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)):
        raise PatternError(
            f"descriptor.resolution must be a list or tuple; "
            f"got {type(raw).__name__}"
        )
    if len(raw) != 2:
        raise PatternError(
            f"descriptor.resolution must have exactly 2 entries; "
            f"got {len(raw)}"
        )
    nv_raw, nu_raw = raw[0], raw[1]
    if not _is_strict_int(nv_raw) or not _is_strict_int(nu_raw):
        raise PatternError(
            f"descriptor.resolution entries must be int; "
            f"got types ({type(nv_raw).__name__}, {type(nu_raw).__name__})"
        )
    return _validate_resolution((int(nv_raw), int(nu_raw)))


def _parse_seed(raw: Any) -> int | None:
    if raw is None:
        return None
    if _is_strict_int(raw):
        return int(raw)
    raise PatternError(
        f"descriptor.seed must be int or None; got {type(raw).__name__}"
    )


def _parse_max_depth(raw: Any) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise PatternError(
            f"descriptor.max_depth must be a number; "
            f"got {type(raw).__name__}"
        )
    md = float(raw)
    _check_finite("max_depth", md)
    if md <= 0.0:
        raise PatternError(f"descriptor.max_depth must be > 0; got {md}")
    return md


def _parse_dimples(
    dimples_raw: Any, declared_count: int
) -> tuple[GaussianDimple, ...]:
    if not isinstance(dimples_raw, (list, tuple)):
        raise PatternError(
            f"descriptor.dimples must be a list or tuple; "
            f"got {type(dimples_raw).__name__}"
        )
    if declared_count != len(dimples_raw):
        raise PatternError(
            f"descriptor.dimple_count ({declared_count}) does not match "
            f"len(descriptor.dimples) ({len(dimples_raw)})"
        )

    built: list[GaussianDimple] = []
    for i, dd in enumerate(dimples_raw):
        if not isinstance(dd, dict):
            raise PatternError(
                f"descriptor.dimples[{i}] must be a dict; "
                f"got {type(dd).__name__}"
            )
        miss = [k for k in _REQUIRED_DIMPLE_KEYS if k not in dd]
        if miss:
            raise PatternError(
                f"descriptor.dimples[{i}] missing keys: {miss}"
            )
        for k in _REQUIRED_DIMPLE_KEYS:
            v = dd[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise PatternError(
                    f"descriptor.dimples[{i}].{k} must be a number; "
                    f"got {type(v).__name__}"
                )
            _check_finite(k, float(v), context=f"descriptor.dimples[{i}]")

        d = GaussianDimple(
            center_u=float(dd["center_u"]),
            center_v=float(dd["center_v"]),
            amplitude=float(dd["amplitude"]),
            sigma_u=float(dd["sigma_u"]),
            sigma_v=float(dd["sigma_v"]),
        )
        _validate_dimple(d, index=i)
        built.append(d)

    return tuple(built)


def gaussian_pattern_from_descriptor(
    descriptor: dict,
) -> GaussianDimplePattern:
    """Reconstruct a :class:`GaussianDimplePattern` from a descriptor dict.

    Validates the schema literals (``pattern_family``,
    ``coordinate_system``, ``depth_field_type``), shape and types of
    the metadata fields, dimple-count consistency, and per-dimple
    field constraints (delegated to the same private validators
    used by :mod:`pattern.gaussian`). Recomputes ``depth_field`` via
    :func:`evaluate_gaussian_dimple_field` with ``clip=True``.

    Raises :class:`PatternError` on any schema or value violation.
    """
    _validate_top_level(descriptor)

    nv, nu = _parse_resolution(descriptor["resolution"])
    max_depth = _parse_max_depth(descriptor["max_depth"])
    seed = _parse_seed(descriptor["seed"])

    if not _is_strict_int(descriptor["dimple_count"]):
        raise PatternError(
            f"descriptor.dimple_count must be int; "
            f"got {type(descriptor['dimple_count']).__name__}"
        )
    declared_count = int(descriptor["dimple_count"])
    if declared_count < 0:
        raise PatternError(
            f"descriptor.dimple_count must be >= 0; got {declared_count}"
        )

    dimples = _parse_dimples(descriptor["dimples"], declared_count)
    depth_field = evaluate_gaussian_dimple_field(
        dimples, resolution=(nv, nu), clip=True
    )

    return GaussianDimplePattern(
        dimples=dimples,
        depth_field=depth_field,
        resolution=(nv, nu),
        max_depth=max_depth,
        seed=seed,
    )
