"""Bottle mesh readiness diagnostic.

Synthetic/loaded bottle mesh readiness diagnostic; **not a physical
PET-bottle validation**. This module diagnoses whether a mesh
resembles the z-axis bottle-body assumptions used by the rest of the
simulator. It does not repair meshes, does not infer material
regions automatically, does not validate wall thickness for arbitrary
STL, and does not prove fire prevention or PET-bottle safety. It is
a pre-simulation readiness report.

Two reports are produced:

- :class:`CylindricalFitReport` summarizes how cylinder-like a mesh
  appears under the z-axis assumption: bounds, height, estimated
  radius from the bounding-box center axis, radial coefficient of
  variation, and a coarse fit-quality label.
- :class:`BottleMeshReadinessReport` wraps the cylindrical fit
  together with :class:`MeshQualityReport` and turns optional
  caller-supplied expected ranges, watertightness requirement, and
  winding-consistency requirement into a list of blocking issues
  and warnings.

Limitations
-----------
- Only the z-axis is considered. Automatic PCA / inertia-tensor
  alignment is intentionally not implemented.
- The cylindrical fit is diagnostic; it is **not** a least-squares
  cylinder fit.
- Empty meshes raise :class:`GeometryError`; non-empty meshes that
  fail other checks populate ``blocking_issues`` instead of raising.
- ``compute_bottle_mesh_readiness_report`` builds a non-mutating
  scaled copy for reporting when ``scale_factor != 1.0``; the input
  ``trimesh.Trimesh`` is read but never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.mesh_io import GeometryError
from optics_simulation.geometry.quality import (
    MeshQualityReport,
    create_mesh_quality_report,
)


_AXIS_RADIUS_EPSILON = 1e-9
_BODY_FALLBACK_MIN_VERTICES = 8


@dataclass(frozen=True)
class CylindricalFitReport:
    vertex_count: int
    face_count: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    height: float
    center_xy: tuple[float, float]
    radial_min: float
    radial_max: float
    radial_mean: float
    radial_std: float
    estimated_radius: float
    radius_cv: float
    z_min: float
    z_max: float
    z_axis_aligned: bool
    is_empty: bool
    is_watertight: bool
    is_winding_consistent: bool
    fit_quality_label: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class BottleMeshReadinessReport:
    quality_report: MeshQualityReport
    cylindrical_fit: CylindricalFitReport
    scale_factor_applied: float
    expected_height_range: tuple[float, float] | None
    expected_radius_range: tuple[float, float] | None
    simulation_ready: bool
    blocking_issues: tuple[str, ...]
    warnings: tuple[str, ...]
    report_type: str = "bottle_mesh_readiness_diagnostic"


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise GeometryError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise GeometryError(
            "mesh is empty; cannot compute cylindrical fit"
        )


def _validate_finite_positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise GeometryError(
            f"{name} must be a finite float > 0; got {v}"
        )
    return v


def _validate_expected_range(
    value: tuple[float, float] | None, name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    seq = tuple(value)
    if len(seq) != 2:
        raise GeometryError(
            f"{name} must have length 2; got length {len(seq)}"
        )
    lo = float(seq[0])
    hi = float(seq[1])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise GeometryError(
            f"{name} bounds must be finite; got ({lo}, {hi})"
        )
    if lo >= hi:
        raise GeometryError(
            f"{name} requires min < max; got ({lo}, {hi})"
        )
    return (lo, hi)


def compute_cylindrical_fit_report(
    mesh: trimesh.Trimesh,
    *,
    radial_sample_exclude_z_fraction: float = 0.05,
    z_axis_tolerance: float = 1e-6,
    radius_cv_cylindrical_threshold: float = 0.08,
    radius_cv_weak_threshold: float = 0.20,
) -> CylindricalFitReport:
    """Diagnose how z-axis cylinder-like a mesh appears.

    Synthetic/loaded bottle mesh readiness diagnostic; **not a
    physical PET-bottle validation**. Computes radial distance from
    the bounding-box center axis (``x = bx_mid``, ``y = by_mid``)
    for vertices in an interior z band, and turns the resulting
    radial coefficient of variation into a coarse fit-quality
    label. The mesh is read but never mutated. PCA / inertia-tensor
    alignment is intentionally **not** performed; an arbitrary-axis
    bottle would need to be aligned by the caller before being
    passed in.

    Parameters
    ----------
    mesh
        Input :class:`trimesh.Trimesh`. Must be non-empty and have
        finite vertices.
    radial_sample_exclude_z_fraction
        Fraction of the total z extent to exclude from the top and
        bottom of the mesh when picking body vertices for radial
        statistics. Must be in ``[0, 0.5)``. Default 0.05 trims the
        top and bottom 5%; cap-center vertices that lie on the
        rotation axis are always excluded regardless of this band.
    z_axis_tolerance
        Minimum height for the report to mark the mesh as z-axis
        aligned. Must be ``> 0``.
    radius_cv_cylindrical_threshold
        Coefficient of variation at or below which the mesh is
        labeled ``"cylindrical_like"``. Must be ``> 0``.
    radius_cv_weak_threshold
        Coefficient of variation at or below which the mesh is
        labeled ``"weakly_cylindrical"`` (when above the
        cylindrical threshold). Must satisfy
        ``radius_cv_cylindrical_threshold <= radius_cv_weak_threshold``.

    Raises
    ------
    GeometryError
        On non-Trimesh input, empty mesh, non-finite vertices, or
        any out-of-range parameter.
    """
    _validate_mesh(mesh)

    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise GeometryError(
            f"mesh.vertices must have shape (N, 3); got {vertices.shape}"
        )
    if not np.all(np.isfinite(vertices)):
        raise GeometryError("mesh vertices contain non-finite values")

    rsf = float(radial_sample_exclude_z_fraction)
    if not np.isfinite(rsf) or rsf < 0.0 or rsf >= 0.5:
        raise GeometryError(
            f"radial_sample_exclude_z_fraction must be a finite float "
            f"in [0, 0.5); got {rsf}"
        )

    z_tol = float(z_axis_tolerance)
    if not np.isfinite(z_tol) or z_tol <= 0.0:
        raise GeometryError(
            f"z_axis_tolerance must be a finite float > 0; got {z_tol}"
        )

    cv_cyl = float(radius_cv_cylindrical_threshold)
    cv_weak = float(radius_cv_weak_threshold)
    if not (np.isfinite(cv_cyl) and np.isfinite(cv_weak)):
        raise GeometryError(
            f"cv thresholds must be finite; got "
            f"({cv_cyl}, {cv_weak})"
        )
    if cv_cyl <= 0.0 or cv_weak <= 0.0:
        raise GeometryError(
            f"cv thresholds must be > 0; got ({cv_cyl}, {cv_weak})"
        )
    if cv_cyl > cv_weak:
        raise GeometryError(
            f"radius_cv_cylindrical_threshold ({cv_cyl}) must be "
            f"<= radius_cv_weak_threshold ({cv_weak})"
        )

    bounds = np.asarray(mesh.bounds, dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise GeometryError("mesh bounds contain non-finite values")
    bounds_min = bounds[0].astype(float, copy=True)
    bounds_max = bounds[1].astype(float, copy=True)

    z_min = float(bounds_min[2])
    z_max = float(bounds_max[2])
    height = float(z_max - z_min)

    cx = float(0.5 * (bounds_min[0] + bounds_max[0]))
    cy = float(0.5 * (bounds_min[1] + bounds_max[1]))

    r_all = np.sqrt(
        (vertices[:, 0] - cx) ** 2 + (vertices[:, 1] - cy) ** 2
    )

    # Always exclude vertices that sit on the rotation axis (e.g.,
    # cap-center vertices produced by ``trimesh.creation.cylinder``);
    # they would bias the mean radius toward zero.
    non_axis_mask = r_all > _AXIS_RADIUS_EPSILON

    if height > 0.0:
        z_low = z_min + rsf * height
        z_high = z_max - rsf * height
        in_z_band = (
            (vertices[:, 2] >= z_low) & (vertices[:, 2] <= z_high)
        )
    else:
        in_z_band = np.ones(vertices.shape[0], dtype=bool)

    body_mask = in_z_band & non_axis_mask
    if int(body_mask.sum()) < _BODY_FALLBACK_MIN_VERTICES:
        body_mask = non_axis_mask
    if int(body_mask.sum()) == 0:
        body_mask = np.ones(vertices.shape[0], dtype=bool)

    r = r_all[body_mask]
    radial_min = float(r.min())
    radial_max = float(r.max())
    radial_mean = float(r.mean())
    radial_std = float(r.std())
    estimated_radius = radial_mean
    if radial_mean > 0.0:
        radius_cv = float(radial_std / radial_mean)
    else:
        radius_cv = float("inf")

    if not np.isfinite(radius_cv):
        fit_quality_label = "not_cylindrical"
    elif radius_cv <= cv_cyl:
        fit_quality_label = "cylindrical_like"
    elif radius_cv <= cv_weak:
        fit_quality_label = "weakly_cylindrical"
    else:
        fit_quality_label = "not_cylindrical"

    z_axis_aligned = bool(
        height > z_tol
        and np.all(np.isfinite(bounds_min))
        and np.all(np.isfinite(bounds_max))
    )

    notes = (
        "z-axis assumption: report assumes the mesh is z-axis "
        "aligned.",
        "PCA/automatic alignment not implemented; arbitrary-axis "
        "bottle alignment is out of scope of this report.",
        "fit is diagnostic, not mesh repair.",
    )

    return CylindricalFitReport(
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        height=height,
        center_xy=(cx, cy),
        radial_min=radial_min,
        radial_max=radial_max,
        radial_mean=radial_mean,
        radial_std=radial_std,
        estimated_radius=estimated_radius,
        radius_cv=radius_cv,
        z_min=z_min,
        z_max=z_max,
        z_axis_aligned=z_axis_aligned,
        is_empty=False,
        is_watertight=bool(mesh.is_watertight),
        is_winding_consistent=bool(mesh.is_winding_consistent),
        fit_quality_label=fit_quality_label,
        notes=notes,
    )


def compute_bottle_mesh_readiness_report(
    mesh: trimesh.Trimesh,
    *,
    scale_factor: float = 1.0,
    expected_height_range: tuple[float, float] | None = None,
    expected_radius_range: tuple[float, float] | None = None,
    require_watertight: bool = False,
    require_winding_consistent: bool = False,
) -> BottleMeshReadinessReport:
    """Wrap mesh-quality and cylindrical-fit reports in a readiness summary.

    Synthetic/loaded bottle mesh readiness diagnostic; **not a
    physical PET-bottle validation**. Builds a non-mutating scaled
    copy of the mesh (when ``scale_factor != 1.0``), forwards it to
    :func:`create_mesh_quality_report` and
    :func:`compute_cylindrical_fit_report`, and turns the optional
    expected-range / watertightness / winding-consistency
    requirements into a tuple of blocking issues and a tuple of
    warnings. ``simulation_ready`` is ``True`` iff the blocking
    list is empty. The input ``trimesh.Trimesh`` is read but never
    mutated; this function does not repair meshes, does not infer
    material regions, does not validate wall thickness, does not
    perform optical or thermal simulation, and does not prove fire
    prevention or PET-bottle safety.

    Parameters
    ----------
    mesh
        Input :class:`trimesh.Trimesh`. Must be non-empty.
    scale_factor
        Multiplicative scale applied to ``mesh.vertices`` for the
        reported copy (e.g. unit conversion). Must be a finite
        float ``> 0``. Default 1.0 leaves the mesh unscaled.
    expected_height_range
        Optional ``(min, max)`` range, in the post-scaling units,
        that the bounding-box height must fall within. Length 2,
        finite, ``min < max``.
    expected_radius_range
        Optional ``(min, max)`` range, in the post-scaling units,
        that ``cylindrical_fit.estimated_radius`` must fall within.
        Length 2, finite, ``min < max``.
    require_watertight
        If ``True``, a non-watertight mesh produces a blocking
        issue. If ``False``, a non-watertight mesh produces a
        warning only.
    require_winding_consistent
        If ``True``, an inconsistent-winding mesh produces a
        blocking issue. If ``False``, a warning only.

    Raises
    ------
    GeometryError
        On non-Trimesh input, empty mesh, non-finite ``scale_factor``,
        non-positive ``scale_factor``, or invalid expected ranges.
    """
    _validate_mesh(mesh)
    sf = _validate_finite_positive(scale_factor, "scale_factor")
    eh_range = _validate_expected_range(
        expected_height_range, "expected_height_range"
    )
    er_range = _validate_expected_range(
        expected_radius_range, "expected_radius_range"
    )

    if sf != 1.0:
        scaled_vertices = (
            np.asarray(mesh.vertices, dtype=float) * sf
        )
        scaled_mesh = trimesh.Trimesh(
            vertices=scaled_vertices,
            faces=np.asarray(mesh.faces),
            process=False,
        )
    else:
        scaled_mesh = mesh

    quality_report = create_mesh_quality_report(scaled_mesh)
    cylindrical_fit = compute_cylindrical_fit_report(scaled_mesh)

    blocking: list[str] = []
    warnings: list[str] = []

    if cylindrical_fit.is_empty:
        blocking.append("mesh_is_empty")

    scaled_vertices_arr = np.asarray(scaled_mesh.vertices, dtype=float)
    if not np.all(np.isfinite(scaled_vertices_arr)):
        blocking.append("non_finite_vertices")

    if cylindrical_fit.height <= 0.0:
        blocking.append(
            f"non_positive_height:{cylindrical_fit.height:.6f}"
        )

    if cylindrical_fit.fit_quality_label == "not_cylindrical":
        blocking.append(
            f"fit_quality_not_cylindrical:radius_cv="
            f"{cylindrical_fit.radius_cv:.6f}"
        )

    if require_watertight and not cylindrical_fit.is_watertight:
        blocking.append("require_watertight_violated")

    if (
        require_winding_consistent
        and not cylindrical_fit.is_winding_consistent
    ):
        blocking.append("require_winding_consistent_violated")

    if eh_range is not None:
        h = cylindrical_fit.height
        if h < eh_range[0] or h > eh_range[1]:
            blocking.append(
                f"height_out_of_range:{h:.6f}_not_in_"
                f"[{eh_range[0]:.6f},{eh_range[1]:.6f}]"
            )

    if er_range is not None:
        r_est = cylindrical_fit.estimated_radius
        if r_est < er_range[0] or r_est > er_range[1]:
            blocking.append(
                f"estimated_radius_out_of_range:{r_est:.6f}_not_in_"
                f"[{er_range[0]:.6f},{er_range[1]:.6f}]"
            )

    if cylindrical_fit.fit_quality_label == "weakly_cylindrical":
        warnings.append(
            f"fit_quality_weakly_cylindrical:radius_cv="
            f"{cylindrical_fit.radius_cv:.6f}"
        )
    if (
        not cylindrical_fit.is_watertight
        and not require_watertight
    ):
        warnings.append("mesh_not_watertight")
    if (
        not cylindrical_fit.is_winding_consistent
        and not require_winding_consistent
    ):
        warnings.append("mesh_winding_inconsistent")
    if sf != 1.0:
        warnings.append(f"scale_factor_applied:{sf:.6f}")
    if eh_range is None:
        warnings.append("expected_height_range_not_provided")
    if er_range is None:
        warnings.append("expected_radius_range_not_provided")

    simulation_ready = len(blocking) == 0

    return BottleMeshReadinessReport(
        quality_report=quality_report,
        cylindrical_fit=cylindrical_fit,
        scale_factor_applied=sf,
        expected_height_range=eh_range,
        expected_radius_range=er_range,
        simulation_ready=simulation_ready,
        blocking_issues=tuple(blocking),
        warnings=tuple(warnings),
    )
