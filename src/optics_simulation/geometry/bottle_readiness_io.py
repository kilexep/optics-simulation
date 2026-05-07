"""File-based bottle mesh readiness loader.

Bottle STL readiness runner; **not a physical PET-bottle validation**.
Wraps :func:`load_mesh` and
:func:`compute_bottle_mesh_readiness_report` into a single
file-path-driven entry point that returns a
:class:`BottleMeshReadinessFileReport`. The returned report is
purely diagnostic: it does **not** repair meshes, does **not** infer
material regions automatically, does **not** perform optical
simulation, does **not** perform thermal simulation, and does
**not** prove fire prevention or PET-bottle safety. Synthetic STL
round-trip is only a file-path smoke check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optics_simulation.geometry.bottle_readiness import (
    BottleMeshReadinessReport,
    compute_bottle_mesh_readiness_report,
)
from optics_simulation.geometry.mesh_io import GeometryError, load_mesh


@dataclass(frozen=True)
class BottleMeshReadinessFileReport:
    mesh_path: str
    readiness_report: BottleMeshReadinessReport
    load_success: bool
    load_error: str | None
    file_exists: bool
    scale_factor: float
    report_type: str = "bottle_mesh_readiness_file_report"


def load_bottle_mesh_readiness_report(
    mesh_path: str | Path,
    *,
    scale_factor: float = 1.0,
    expected_height_range: tuple[float, float] | None = None,
    expected_radius_range: tuple[float, float] | None = None,
    require_watertight: bool = False,
    require_winding_consistent: bool = False,
) -> BottleMeshReadinessFileReport:
    """Load an STL/mesh file and return a bottle-readiness file report.

    Bottle STL readiness runner; **not a physical PET-bottle
    validation**. The supplied path is loaded by
    :func:`load_mesh` and the resulting :class:`trimesh.Trimesh` is
    forwarded to :func:`compute_bottle_mesh_readiness_report`. The
    function does **not** repair meshes, does **not** align axes,
    does **not** perform optical simulation, does **not** perform
    thermal simulation, and does **not** infer material regions.
    The mesh file on disk is read but never written or modified.

    Failure modes are raised as :class:`GeometryError`:

    - ``mesh_path`` is not a ``str`` or :class:`pathlib.Path`.
    - the file does not exist.
    - :func:`load_mesh` cannot decode the file.
    - any downstream validation
      (``scale_factor`` / expected ranges / etc.) fails.

    On success the returned object always has ``load_success ==
    True`` and ``load_error is None``; the failure-only fields are
    present so future soft-load extensions can populate them
    without a schema change.

    Parameters
    ----------
    mesh_path
        Filesystem path to an STL (or other trimesh-supported)
        mesh file. Must be a ``str`` or :class:`pathlib.Path`.
    scale_factor
        Multiplicative scale applied to ``mesh.vertices`` for the
        reported copy. Forwarded verbatim to
        :func:`compute_bottle_mesh_readiness_report`. Must be a
        finite float ``> 0``.
    expected_height_range, expected_radius_range
        Optional ``(min, max)`` ranges in post-scaling units
        forwarded verbatim.
    require_watertight, require_winding_consistent
        Forwarded verbatim.

    Raises
    ------
    GeometryError
        On invalid path type, missing file, mesh decode failure,
        or any downstream validation failure.
    """
    if not isinstance(mesh_path, (str, Path)):
        raise GeometryError(
            "mesh_path must be a str or pathlib.Path; "
            f"got {type(mesh_path).__name__}"
        )

    sf = float(scale_factor)
    if not np.isfinite(sf) or sf <= 0.0:
        raise GeometryError(
            f"scale_factor must be a finite float > 0; got {sf}"
        )

    path = Path(mesh_path)
    if not path.is_file():
        raise GeometryError(f"mesh file not found: {path}")

    mesh = load_mesh(path)

    readiness_report = compute_bottle_mesh_readiness_report(
        mesh,
        scale_factor=sf,
        expected_height_range=expected_height_range,
        expected_radius_range=expected_radius_range,
        require_watertight=require_watertight,
        require_winding_consistent=require_winding_consistent,
    )

    return BottleMeshReadinessFileReport(
        mesh_path=str(path),
        readiness_report=readiness_report,
        load_success=True,
        load_error=None,
        file_exists=True,
        scale_factor=sf,
    )
