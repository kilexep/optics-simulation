"""Actual-STL body-region vertex mask heuristic.

Actual STL patterned optical-to-thermal risk smoke check; **not a
physical PET-bottle validation**. Builds a per-vertex boolean mask
that selects an "outer body" region of an actual STL bottle mesh
under the z-axis assumption: the middle z band (configurable via
``z_min_fraction`` / ``z_max_fraction``) intersected with vertices
whose radial distance from the bounding-box xy center is at or
above a configurable radial quantile (default median).

Limitations
-----------
- This mask is a **diagnostic body-region heuristic**, not a
  verified manufacturing surface.
- The mask uses the ``z_min`` / ``z_max`` bounding-box span only.
  Automatic PCA / inertia-tensor alignment is not performed.
- The mask does **not** validate physical wall thickness, neck or
  base geometry, or any manufacturability constraint.
- ``mesh`` is read but never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.mesh_io import GeometryError


@dataclass(frozen=True)
class ActualBottleBodyMaskReport:
    include_mask: np.ndarray
    vertex_count: int
    selected_count: int
    selected_fraction: float
    z_min: float
    z_max: float
    body_z_min: float
    body_z_max: float
    radial_min: float
    radial_max: float
    radial_median: float
    radial_threshold: float
    notes: tuple[str, ...]
    report_type: str = "actual_bottle_body_mask_report"


def create_actual_bottle_body_vertex_mask(
    mesh: trimesh.Trimesh,
    *,
    z_min_fraction: float = 0.15,
    z_max_fraction: float = 0.85,
    radial_quantile: float = 0.50,
) -> ActualBottleBodyMaskReport:
    """Compute a diagnostic body-region include-mask for an actual STL.

    Actual STL patterned optical-to-thermal risk smoke check;
    **not a physical PET-bottle validation**. Returns an
    :class:`ActualBottleBodyMaskReport` whose ``include_mask`` is
    ``True`` for vertices that lie in the middle z band (between
    ``z_min + z_min_fraction * height`` and
    ``z_min + z_max_fraction * height``) **and** whose radial
    distance from the bounding-box xy center is at or above the
    configured radial quantile (default median). The intent is to
    rough out neck, base, and inner-ish low-radius vertices so
    downstream patterning sees an "outer body"-ish surface.

    Raises
    ------
    GeometryError
        On non-Trimesh / empty input, non-finite vertices,
        ``z_min_fraction`` / ``z_max_fraction`` outside ``[0, 1]``
        or with ``z_min_fraction >= z_max_fraction``, or
        ``radial_quantile`` outside ``(0, 1)``.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise GeometryError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )

    vertices = np.asarray(mesh.vertices, dtype=float)
    n = int(vertices.shape[0])
    if n == 0:
        raise GeometryError(
            "mesh has no vertices; cannot build body-region mask"
        )
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise GeometryError(
            f"mesh.vertices must have shape (N, 3); got {vertices.shape}"
        )
    if not np.isfinite(vertices).all():
        raise GeometryError(
            "mesh.vertices contain non-finite values"
        )

    z_lo = float(z_min_fraction)
    z_hi = float(z_max_fraction)
    if not (np.isfinite(z_lo) and np.isfinite(z_hi)):
        raise GeometryError(
            f"z_min_fraction and z_max_fraction must be finite; "
            f"got ({z_lo}, {z_hi})"
        )
    if not (0.0 <= z_lo < z_hi <= 1.0):
        raise GeometryError(
            f"require 0 <= z_min_fraction < z_max_fraction <= 1; "
            f"got ({z_lo}, {z_hi})"
        )

    rq = float(radial_quantile)
    if not (np.isfinite(rq) and 0.0 < rq < 1.0):
        raise GeometryError(
            f"radial_quantile must be a finite float in (0, 1); "
            f"got {rq}"
        )

    bounds = np.asarray(mesh.bounds, dtype=float)
    if not np.isfinite(bounds).all():
        raise GeometryError("mesh.bounds contain non-finite values")
    z_min = float(bounds[0, 2])
    z_max = float(bounds[1, 2])
    height = float(z_max - z_min)
    if height <= 0.0:
        raise GeometryError(
            f"mesh z extent must be > 0; got height={height}"
        )

    body_z_min = z_min + z_lo * height
    body_z_max = z_min + z_hi * height

    cx = float(0.5 * (bounds[0, 0] + bounds[1, 0]))
    cy = float(0.5 * (bounds[0, 1] + bounds[1, 1]))

    z_band = (
        (vertices[:, 2] >= body_z_min)
        & (vertices[:, 2] <= body_z_max)
    )
    r = np.sqrt(
        (vertices[:, 0] - cx) ** 2 + (vertices[:, 1] - cy) ** 2
    )

    if int(z_band.sum()) >= 4:
        r_in_band = r[z_band]
        radial_threshold = float(np.quantile(r_in_band, rq))
    else:
        radial_threshold = float(np.quantile(r, rq))

    include_mask = z_band & (r >= radial_threshold)
    selected_count = int(include_mask.sum())
    selected_fraction = float(selected_count) / float(n)

    notes = (
        "Actual STL body-region heuristic; not arbitrary STL "
        "semantic segmentation.",
        "Patterning applied via include_mask is diagnostic, not a "
        "verified manufacturing surface.",
        "z-axis assumption: PCA / arbitrary-axis alignment is out "
        "of scope.",
    )

    return ActualBottleBodyMaskReport(
        include_mask=include_mask.astype(bool, copy=True),
        vertex_count=n,
        selected_count=selected_count,
        selected_fraction=selected_fraction,
        z_min=z_min,
        z_max=z_max,
        body_z_min=float(body_z_min),
        body_z_max=float(body_z_max),
        radial_min=float(r.min()),
        radial_max=float(r.max()),
        radial_median=float(np.median(r)),
        radial_threshold=float(radial_threshold),
        notes=notes,
    )
