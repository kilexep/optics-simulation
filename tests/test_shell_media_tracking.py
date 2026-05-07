import os
from pathlib import Path

import numpy as np
import pytest

from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_shell,
)
from optics_simulation.optics import (
    OpticsError,
    RayBundle,
    ShellHitSurfaceClassification,
    ShellMediumTrackingResult,
    ShellMediumTrackingStepSummary,
    classify_synthetic_shell_hit_surfaces,
    expected_shell_surface_sequence,
    make_ray_bundle,
    run_surface_classified_shell_trace,
)


_OUTER_RADIUS = 30.0
_WALL_THICKNESS = 1.0
_HEIGHT = 120.0
_SECTIONS = 96
_HEIGHT_SEGMENTS = 24
_TRACE_EPSILON = 0.1
_TRACE_RADIAL_TOL = 0.1


def _build_shell():
    return create_subdivided_synthetic_bottle_shell(
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        sections=_SECTIONS,
        height_segments=_HEIGHT_SEGMENTS,
    )


def _build_side_incidence_rays(
    *,
    origin_x: float,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    ny: int,
    nz: int,
    direction: tuple[float, float, float] = (-1.0, 0.0, 0.0),
) -> RayBundle:
    ys = np.linspace(y_range[0], y_range[1], ny)
    zs = np.linspace(z_range[0], z_range[1], nz)
    yv, zv = np.meshgrid(ys, zs, indexing="xy")
    origins = np.column_stack([
        np.full(yv.size, float(origin_x)),
        yv.ravel(),
        zv.ravel(),
    ])
    directions = np.broadcast_to(
        np.asarray(direction, dtype=float).reshape(1, 3),
        origins.shape,
    ).copy()
    return make_ray_bundle(origins, directions)


def test_classify_returns_shell_hit_surface_classification() -> None:
    pts = np.array([[30.0, 0.0, 0.0]])
    mask = np.array([True])
    out = classify_synthetic_shell_hit_surfaces(
        pts, mask,
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
    )
    assert isinstance(out, ShellHitSurfaceClassification)
    assert out.ray_count == 1


def test_outer_lateral_points_classified_correctly() -> None:
    pts = np.array([
        [30.0, 0.0, 0.0],
        [0.0, -30.0, 10.0],
        [-30.0, 0.0, -20.0],
    ])
    mask = np.array([True, True, True])
    out = classify_synthetic_shell_hit_surfaces(
        pts, mask,
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
    )
    assert int(out.outer_lateral_mask.sum()) == 3
    assert (out.surface_kind == "outer_lateral").all()


def test_inner_lateral_points_classified_correctly() -> None:
    pts = np.array([
        [29.0, 0.0, 0.0],
        [0.0, -29.0, 10.0],
        [-29.0, 0.0, 5.0],
    ])
    mask = np.array([True, True, True])
    out = classify_synthetic_shell_hit_surfaces(
        pts, mask,
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
    )
    assert int(out.inner_lateral_mask.sum()) == 3
    assert (out.surface_kind == "inner_lateral").all()


def test_z_boundary_points_classified_correctly() -> None:
    pts = np.array([
        [29.5, 0.0, 60.0],
        [0.0, 29.5, -60.0],
    ])
    mask = np.array([True, True])
    out = classify_synthetic_shell_hit_surfaces(
        pts, mask,
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
    )
    assert int(out.z_boundary_mask.sum()) == 2
    assert (out.surface_kind == "z_boundary").all()


def test_miss_rays_classified_as_miss() -> None:
    pts = np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ])
    mask = np.array([False, False])
    out = classify_synthetic_shell_hit_surfaces(
        pts, mask,
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
    )
    assert int(out.miss_mask.sum()) == 2
    assert (out.surface_kind == "miss").all()


def test_unknown_hit_point_classified_as_unknown() -> None:
    pts = np.array([
        [15.0, 0.0, 0.0],   # r=15, far from outer/inner/z
    ])
    mask = np.array([True])
    out = classify_synthetic_shell_hit_surfaces(
        pts, mask,
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
    )
    assert int(out.unknown_mask.sum()) == 1
    assert (out.surface_kind == "unknown").all()


def test_invalid_geometry_params_raise() -> None:
    pts = np.array([[30.0, 0.0, 0.0]])
    mask = np.array([True])
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(OpticsError, match="outer_radius"):
            classify_synthetic_shell_hit_surfaces(
                pts, mask,
                outer_radius=bad,
                wall_thickness=_WALL_THICKNESS,
                height=_HEIGHT,
            )
        with pytest.raises(OpticsError, match="wall_thickness"):
            classify_synthetic_shell_hit_surfaces(
                pts, mask,
                outer_radius=_OUTER_RADIUS,
                wall_thickness=bad,
                height=_HEIGHT,
            )
        with pytest.raises(OpticsError, match="height"):
            classify_synthetic_shell_hit_surfaces(
                pts, mask,
                outer_radius=_OUTER_RADIUS,
                wall_thickness=_WALL_THICKNESS,
                height=bad,
            )
    with pytest.raises(OpticsError, match="wall_thickness"):
        classify_synthetic_shell_hit_surfaces(
            pts, mask,
            outer_radius=_OUTER_RADIUS,
            wall_thickness=_OUTER_RADIUS,
            height=_HEIGHT,
        )
    for bad in (0.0, -1e-6, float("nan"), float("inf")):
        with pytest.raises(OpticsError, match="radial_tolerance"):
            classify_synthetic_shell_hit_surfaces(
                pts, mask,
                outer_radius=_OUTER_RADIUS,
                wall_thickness=_WALL_THICKNESS,
                height=_HEIGHT,
                radial_tolerance=bad,
            )
    with pytest.raises(OpticsError, match="z_tolerance"):
        classify_synthetic_shell_hit_surfaces(
            pts, mask,
            outer_radius=_OUTER_RADIUS,
            wall_thickness=_WALL_THICKNESS,
            height=_HEIGHT,
            z_tolerance=-1.0,
        )


def test_invalid_hit_points_shape_raises() -> None:
    bad_2d = np.array([[1.0, 2.0]])  # (N, 2) instead of (N, 3)
    mask = np.array([True])
    with pytest.raises(OpticsError, match="hit_points"):
        classify_synthetic_shell_hit_surfaces(
            bad_2d, mask,
            outer_radius=_OUTER_RADIUS,
            wall_thickness=_WALL_THICKNESS,
            height=_HEIGHT,
        )

    pts = np.array([[30.0, 0.0, 0.0], [29.0, 0.0, 0.0]])
    bad_mask = np.array([True])  # mismatched length
    with pytest.raises(OpticsError, match="hit_mask"):
        classify_synthetic_shell_hit_surfaces(
            pts, bad_mask,
            outer_radius=_OUTER_RADIUS,
            wall_thickness=_WALL_THICKNESS,
            height=_HEIGHT,
        )

    nan_pts = np.array([[np.nan, np.nan, np.nan]])
    hit_mask = np.array([True])
    with pytest.raises(OpticsError, match="finite"):
        classify_synthetic_shell_hit_surfaces(
            nan_pts, hit_mask,
            outer_radius=_OUTER_RADIUS,
            wall_thickness=_WALL_THICKNESS,
            height=_HEIGHT,
        )


def test_expected_shell_surface_sequence_air() -> None:
    seq = expected_shell_surface_sequence(fill_medium="air")
    assert len(seq) == 4
    assert seq == (
        "outer_lateral", "inner_lateral",
        "inner_lateral", "outer_lateral",
    )


def test_expected_shell_surface_sequence_water() -> None:
    seq = expected_shell_surface_sequence(fill_medium="water")
    assert len(seq) == 4
    assert seq == (
        "outer_lateral", "inner_lateral",
        "inner_lateral", "outer_lateral",
    )


def test_expected_shell_surface_sequence_invalid_fill_medium_raises() -> None:
    with pytest.raises(OpticsError, match="fill_medium"):
        expected_shell_surface_sequence(fill_medium="oil")
    with pytest.raises(OpticsError, match="fill_medium"):
        expected_shell_surface_sequence(fill_medium="")


def test_run_surface_classified_shell_trace_returns_result() -> None:
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    out = run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="air",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    assert isinstance(out, ShellMediumTrackingResult)
    assert all(
        isinstance(s, ShellMediumTrackingStepSummary)
        for s in out.step_summaries
    )


def test_air_fill_trace_medium_path() -> None:
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    out = run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="air",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    assert out.medium_path == ("air", "pet", "air", "pet", "air")
    assert out.expected_surface_sequence == (
        "outer_lateral", "inner_lateral",
        "inner_lateral", "outer_lateral",
    )


def test_water_fill_trace_medium_path() -> None:
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    out = run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="water",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    assert out.medium_path == ("air", "pet", "water", "pet", "air")


def test_step_summaries_length_matches_trace_step_count() -> None:
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    out = run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="air",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    assert len(out.step_summaries) == out.trace_result.step_count


def test_validation_passed_for_side_incidence_synthetic_smoke() -> None:
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    for fill in ("air", "water"):
        out = run_surface_classified_shell_trace(
            mesh, rays,
            fill_medium=fill,
            outer_radius=_OUTER_RADIUS,
            wall_thickness=_WALL_THICKNESS,
            height=_HEIGHT,
            radial_tolerance=_TRACE_RADIAL_TOL,
            epsilon=_TRACE_EPSILON,
        )
        assert out.validation_passed is True
        assert out.unexpected_surface_total == 0


def test_final_ray_weights_align_with_trace_result() -> None:
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    out = run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="air",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    np.testing.assert_allclose(
        out.final_ray_weights,
        out.trace_result.final_ray_weights,
    )
    # And confirm copy: mutating one does not affect the other
    snapshot = out.trace_result.final_ray_weights.copy()
    out.final_ray_weights[:] = -1.0
    np.testing.assert_array_equal(
        out.trace_result.final_ray_weights, snapshot,
    )


def test_final_source_ray_indices_align_with_trace_result() -> None:
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    out = run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="air",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    np.testing.assert_array_equal(
        out.final_source_ray_indices,
        out.trace_result.final_source_ray_indices,
    )
    snapshot = out.trace_result.final_source_ray_indices.copy()
    out.final_source_ray_indices[:] = -1
    np.testing.assert_array_equal(
        out.trace_result.final_source_ray_indices, snapshot,
    )


def test_wrapper_final_ray_count_matches_trace_result() -> None:
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    out = run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="air",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    assert int(out.final_ray_count) == int(
        out.trace_result.final_rays.ray_count
    )


def test_mesh_vertices_and_faces_not_mutated() -> None:
    mesh = _build_shell()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="water",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_no_file_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    mesh = _build_shell()
    rays = _build_side_incidence_rays(
        origin_x=100.0,
        y_range=(-50.0, 50.0),
        z_range=(-70.0, 70.0),
        ny=7, nz=7,
    )
    before = set(os.listdir(tmp_path))
    run_surface_classified_shell_trace(
        mesh, rays,
        fill_medium="air",
        outer_radius=_OUTER_RADIUS,
        wall_thickness=_WALL_THICKNESS,
        height=_HEIGHT,
        radial_tolerance=_TRACE_RADIAL_TOL,
        epsilon=_TRACE_EPSILON,
    )
    after = set(os.listdir(tmp_path))
    assert before == after
