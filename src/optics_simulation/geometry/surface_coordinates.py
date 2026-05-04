"""Normalized cylindrical surface coordinates.

Maps 3D points or mesh vertices to (u, v) where:

- ``u in [0, 1)``: circumferential coordinate, ``atan2`` around the z-axis,
  scaled by 1 / (2*pi) and wrapped.
- ``v in [0, 1]``: height coordinate normalized between ``z_min`` and ``z_max``.

Limitations
-----------
- Assumes the input STL/mesh is **z-axis aligned**. Automatic
  bottle-axis alignment (e.g. PCA / inertia tensor) is not implemented
  in this module — that belongs to a later task.
- For points lying exactly on the chosen center axis (``x == cx`` and
  ``y == cy``), ``u`` is mathematically undefined. We follow numpy's
  ``arctan2`` convention which returns 0 in that case, so ``u = 0``. PET
  bottle wall vertices do not lie on the axis, so this corner does not
  occur in practice.
- ``v`` is **not clamped**. If a caller passes a point with z outside
  ``[z_min, z_max]``, ``v`` will fall outside ``[0, 1]``. The caller
  decides whether to clamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import trimesh

from optics_simulation.geometry.mesh_io import GeometryError


@dataclass(frozen=True)
class SurfaceCoordinateMap:
    u: np.ndarray
    v: np.ndarray
    point_count: int
    z_min: float
    z_max: float
    center_xy: tuple[float, float]
    coordinate_type: str = "normalized_cylindrical"


def _check_axis_range(z_min: float, z_max: float) -> None:
    if z_min == z_max:
        raise GeometryError(
            f"z_min equals z_max ({z_min}); cannot normalize v over zero-height axis"
        )
    if z_max < z_min:
        raise GeometryError(
            f"z_max ({z_max}) must be greater than z_min ({z_min})"
        )


def _u_from_xy(x: np.ndarray, y: np.ndarray, cx: float, cy: float) -> np.ndarray:
    angle = np.arctan2(y - cy, x - cx)
    return np.mod(angle / (2.0 * np.pi) + 1.0, 1.0)


def normalize_cylindrical_coordinates(
    points: np.ndarray | Sequence[Sequence[float]],
    z_min: float | None = None,
    z_max: float | None = None,
    center_xy: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (u, v) for an array of 3D points around the z-axis.

    ``points`` must have shape (N, 3). If ``z_min`` / ``z_max`` are
    ``None`` they are derived from ``points[:, 2]``. If ``center_xy``
    is ``None`` it defaults to ``(0.0, 0.0)`` — auto-detection of the
    bottle axis is intentionally not performed here. See module
    docstring for limitations.
    """
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise GeometryError(
            f"points must have shape (N, 3); got {arr.shape}"
        )

    if center_xy is None:
        cx, cy = 0.0, 0.0
    else:
        cx, cy = float(center_xy[0]), float(center_xy[1])

    if z_min is None:
        z_min = float(arr[:, 2].min())
    if z_max is None:
        z_max = float(arr[:, 2].max())
    _check_axis_range(float(z_min), float(z_max))

    u = _u_from_xy(arr[:, 0], arr[:, 1], cx, cy)
    v = (arr[:, 2] - z_min) / (z_max - z_min)
    return u, v


def point_to_normalized_cylindrical(
    point: Sequence[float],
    z_min: float,
    z_max: float,
    center_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Compute (u, v) for a single 3D point. Convenience wrapper."""
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size != 3:
        raise GeometryError(
            f"point must have 3 components; got {arr.size}"
        )
    _check_axis_range(float(z_min), float(z_max))

    cx, cy = float(center_xy[0]), float(center_xy[1])
    angle = float(np.arctan2(arr[1] - cy, arr[0] - cx))
    u = float(np.mod(angle / (2.0 * np.pi) + 1.0, 1.0))
    v = float((arr[2] - z_min) / (z_max - z_min))
    return u, v


def create_vertex_surface_coordinates(
    mesh: trimesh.Trimesh,
) -> SurfaceCoordinateMap:
    """Compute (u, v) for every mesh vertex using the mesh bounding box.

    Assumes the mesh is **z-axis aligned**. Derives ``z_min`` / ``z_max``
    from ``mesh.bounds[:, 2]`` and ``center_xy`` from the x/y
    bounding-box midpoint. Auto-alignment via PCA or inertia tensor is
    not performed — see module docstring.
    """
    bounds = np.asarray(mesh.bounds, dtype=float)
    z_min = float(bounds[0, 2])
    z_max = float(bounds[1, 2])
    cx = float(0.5 * (bounds[0, 0] + bounds[1, 0]))
    cy = float(0.5 * (bounds[0, 1] + bounds[1, 1]))

    vertices = np.asarray(mesh.vertices, dtype=float)
    u, v = normalize_cylindrical_coordinates(
        vertices, z_min=z_min, z_max=z_max, center_xy=(cx, cy)
    )
    return SurfaceCoordinateMap(
        u=u,
        v=v,
        point_count=int(len(vertices)),
        z_min=z_min,
        z_max=z_max,
        center_xy=(cx, cy),
    )
