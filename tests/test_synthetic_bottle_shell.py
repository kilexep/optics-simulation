import numpy as np
import pytest
import trimesh

from optics_simulation.geometry import (
    GeometryError,
    SurfaceCoordinateMap,
    create_subdivided_synthetic_bottle_shell,
    create_vertex_surface_coordinates,
)
from optics_simulation.optics import (
    RayBundle,
    create_shell_medium_preset,
    make_ray_bundle,
    run_multi_step_trace,
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


def test_returns_trimesh() -> None:
    mesh = create_subdivided_synthetic_bottle_shell()
    assert isinstance(mesh, trimesh.Trimesh)


def test_has_vertices_and_faces() -> None:
    mesh = create_subdivided_synthetic_bottle_shell()
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_default_mesh_is_watertight() -> None:
    mesh = create_subdivided_synthetic_bottle_shell()
    assert mesh.is_watertight is True


def test_z_range_matches_height() -> None:
    height = 80.0
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=20.0,
        wall_thickness=0.5,
        height=height,
        sections=64,
        height_segments=10,
    )
    bounds = np.asarray(mesh.bounds, dtype=float)
    assert bounds[1, 2] - bounds[0, 2] == pytest.approx(height, rel=1e-9)
    assert bounds[0, 2] == pytest.approx(-height / 2.0, abs=1e-9)
    assert bounds[1, 2] == pytest.approx(+height / 2.0, abs=1e-9)


def test_outer_radius_matches_outer_radius() -> None:
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=25.0,
        wall_thickness=1.5,
    )
    xy_radii = np.linalg.norm(np.asarray(mesh.vertices)[:, :2], axis=1)
    assert xy_radii.max() == pytest.approx(25.0, abs=1e-9)


def test_inner_radius_matches_outer_minus_wall_thickness() -> None:
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=25.0,
        wall_thickness=1.5,
    )
    xy_radii = np.linalg.norm(np.asarray(mesh.vertices)[:, :2], axis=1)
    assert xy_radii.min() == pytest.approx(23.5, abs=1e-9)


def test_invalid_outer_radius_raises() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(GeometryError, match="outer_radius"):
            create_subdivided_synthetic_bottle_shell(outer_radius=bad)


def test_invalid_wall_thickness_raises() -> None:
    for bad in (0.0, -0.5):
        with pytest.raises(GeometryError, match="wall_thickness"):
            create_subdivided_synthetic_bottle_shell(wall_thickness=bad)


def test_wall_thickness_not_less_than_outer_radius_raises() -> None:
    with pytest.raises(GeometryError, match="wall_thickness"):
        create_subdivided_synthetic_bottle_shell(
            outer_radius=10.0, wall_thickness=10.0,
        )
    with pytest.raises(GeometryError, match="wall_thickness"):
        create_subdivided_synthetic_bottle_shell(
            outer_radius=10.0, wall_thickness=15.0,
        )


def test_invalid_height_raises() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(GeometryError, match="height"):
            create_subdivided_synthetic_bottle_shell(height=bad)


def test_invalid_sections_raises() -> None:
    for bad in (7, 1, 0, -3):
        with pytest.raises(GeometryError, match="sections"):
            create_subdivided_synthetic_bottle_shell(sections=bad)


def test_invalid_height_segments_raises() -> None:
    for bad in (0, -2):
        with pytest.raises(GeometryError, match="height_segments"):
            create_subdivided_synthetic_bottle_shell(height_segments=bad)


def test_surface_coordinate_compatibility() -> None:
    mesh = create_subdivided_synthetic_bottle_shell()
    surface_map = create_vertex_surface_coordinates(mesh)
    assert isinstance(surface_map, SurfaceCoordinateMap)
    assert surface_map.point_count == len(mesh.vertices)


def test_surface_coordinate_uv_ranges_valid() -> None:
    mesh = create_subdivided_synthetic_bottle_shell()
    surface_map = create_vertex_surface_coordinates(mesh)
    assert (surface_map.u >= 0.0).all()
    assert (surface_map.u < 1.0 + 1e-12).all()
    assert (surface_map.v >= -1e-12).all()
    assert (surface_map.v <= 1.0 + 1e-12).all()


def test_side_incidence_smoke_with_empty_shell_preset() -> None:
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=30.0, wall_thickness=1.0, height=120.0,
        sections=64, height_segments=12,
    )
    rays = _build_side_incidence_rays(
        origin_x=80.0,
        y_range=(-20.0, 20.0),
        z_range=(-30.0, 30.0),
        ny=5,
        nz=5,
    )
    preset = create_shell_medium_preset(fill_medium="air")
    assert len(preset.interface_sequence) == 4
    trace = run_multi_step_trace(
        mesh, rays, list(preset.interface_sequence),
    )
    assert trace.final_rays.ray_count >= 0
    assert trace.final_rays.ray_count <= rays.ray_count


def test_side_incidence_smoke_with_water_filled_preset() -> None:
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=30.0, wall_thickness=1.0, height=120.0,
        sections=64, height_segments=12,
    )
    rays = _build_side_incidence_rays(
        origin_x=80.0,
        y_range=(-20.0, 20.0),
        z_range=(-30.0, 30.0),
        ny=5,
        nz=5,
    )
    preset = create_shell_medium_preset(fill_medium="water")
    assert len(preset.interface_sequence) == 4
    trace = run_multi_step_trace(
        mesh, rays, list(preset.interface_sequence),
    )
    assert trace.final_rays.ray_count >= 0
    assert trace.final_rays.ray_count <= rays.ray_count


def test_mesh_not_mutated_by_trace() -> None:
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=30.0, wall_thickness=1.0, height=120.0,
        sections=64, height_segments=12,
    )
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)

    rays = _build_side_incidence_rays(
        origin_x=80.0,
        y_range=(-20.0, 20.0),
        z_range=(-30.0, 30.0),
        ny=4,
        nz=4,
    )
    preset = create_shell_medium_preset(fill_medium="air")
    run_multi_step_trace(mesh, rays, list(preset.interface_sequence))

    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)
