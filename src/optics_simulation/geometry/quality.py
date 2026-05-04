"""Mesh quality report.

Pure data + pure function: takes a ``trimesh.Trimesh`` and returns a
:class:`MeshQualityReport`. Performs no I/O. STL files do not encode
units, so ``units`` must be supplied by the caller (typically from
``geometry.units`` in config).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class MeshQualityReport:
    source_path: str | None
    vertex_count: int
    face_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    extents: tuple[float, float, float]
    centroid: tuple[float, float, float]
    is_watertight: bool
    is_empty: bool
    units: str | None = None


def _vec3(arr: Any) -> tuple[float, float, float]:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError(f"expected 3-vector, got size {a.size}")
    return (float(a[0]), float(a[1]), float(a[2]))


def create_mesh_quality_report(
    mesh: trimesh.Trimesh,
    source_path: str | Path | None = None,
    units: str | None = None,
) -> MeshQualityReport:
    is_empty = len(mesh.vertices) == 0 or len(mesh.faces) == 0

    if is_empty:
        zero = (0.0, 0.0, 0.0)
        bounds_min = bounds_max = extents = centroid = zero
        is_watertight = False
    else:
        bounds = np.asarray(mesh.bounds, dtype=float)
        bounds_min = _vec3(bounds[0])
        bounds_max = _vec3(bounds[1])
        extents = _vec3(mesh.extents)
        centroid = _vec3(mesh.centroid)
        is_watertight = bool(mesh.is_watertight)

    return MeshQualityReport(
        source_path=str(source_path) if source_path is not None else None,
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        extents=extents,
        centroid=centroid,
        is_watertight=is_watertight,
        is_empty=is_empty,
        units=units,
    )
