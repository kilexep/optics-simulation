import numpy as np
import pytest
import trimesh

from optics_simulation.angle_scan import run_baseline_angle_scan
from optics_simulation.geometry import (
    GeometryError,
    SurfaceCoordinateMap,
    create_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.metrics import OpticalMetrics
from optics_simulation.optics import (
    create_detector_grid,
    create_detector_plane,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
SLAB_INTERFACES = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]


def test_returns_trimesh() -> None:
    mesh = create_synthetic_bottle_body()
    assert isinstance(mesh, trimesh.Trimesh)


def test_has_vertices_and_faces() -> None:
    mesh = create_synthetic_bottle_body()
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_default_mesh_is_watertight() -> None:
    mesh = create_synthetic_bottle_body()
    assert mesh.is_watertight is True


def test_z_range_matches_height() -> None:
    height = 120.0
    mesh = create_synthetic_bottle_body(radius=30.0, height=height, sections=96)
    bounds = np.asarray(mesh.bounds, dtype=float)
    z_min = float(bounds[0, 2])
    z_max = float(bounds[1, 2])
    assert z_max - z_min == pytest.approx(height, rel=1e-9)
    assert z_min == pytest.approx(-height / 2.0, abs=1e-9)
    assert z_max == pytest.approx(+height / 2.0, abs=1e-9)


def test_xy_radius_matches() -> None:
    radius = 30.0
    mesh = create_synthetic_bottle_body(radius=radius, height=120.0, sections=96)
    bounds = np.asarray(mesh.bounds, dtype=float)
    x_extent = float(bounds[1, 0] - bounds[0, 0])
    y_extent = float(bounds[1, 1] - bounds[0, 1])
    assert x_extent == pytest.approx(2.0 * radius, rel=1e-6)
    assert y_extent == pytest.approx(2.0 * radius, rel=1e-6)
    vertices = np.asarray(mesh.vertices, dtype=float)
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    assert float(radii.max()) <= radius + 1e-9


def test_centroid_near_origin() -> None:
    mesh = create_synthetic_bottle_body()
    centroid = np.asarray(mesh.centroid, dtype=float)
    assert np.allclose(centroid, np.zeros(3), atol=1e-9)


def test_invalid_radius_raises() -> None:
    with pytest.raises(GeometryError):
        create_synthetic_bottle_body(radius=0.0)
    with pytest.raises(GeometryError):
        create_synthetic_bottle_body(radius=-1.0)


def test_invalid_height_raises() -> None:
    with pytest.raises(GeometryError):
        create_synthetic_bottle_body(height=0.0)
    with pytest.raises(GeometryError):
        create_synthetic_bottle_body(height=-5.0)


def test_sections_below_minimum_raises() -> None:
    with pytest.raises(GeometryError):
        create_synthetic_bottle_body(sections=7)
    with pytest.raises(GeometryError):
        create_synthetic_bottle_body(sections=0)
    with pytest.raises(GeometryError):
        create_synthetic_bottle_body(sections=-3)


def test_sections_at_minimum_is_accepted() -> None:
    mesh = create_synthetic_bottle_body(sections=8)
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) > 0


def test_surface_coordinates_compatible() -> None:
    mesh = create_synthetic_bottle_body()
    coords = create_vertex_surface_coordinates(mesh)
    assert isinstance(coords, SurfaceCoordinateMap)
    assert coords.point_count == len(mesh.vertices)


def test_uv_in_unit_ranges() -> None:
    mesh = create_synthetic_bottle_body()
    coords = create_vertex_surface_coordinates(mesh)
    assert float(coords.u.min()) >= 0.0
    assert float(coords.u.max()) < 1.0 + 1e-12
    assert float(coords.v.min()) >= -1e-12
    assert float(coords.v.max()) <= 1.0 + 1e-12


def test_run_baseline_angle_scan_smoke() -> None:
    mesh = create_synthetic_bottle_body(radius=30.0, height=120.0, sections=48)
    ray_grid_config = {
        "origin_plane_z": 200.0,
        "x_range": (-50.0, 50.0),
        "y_range": (-70.0, 70.0),
        "nx": 7,
        "ny": 7,
    }
    detector = create_detector_plane(
        center=(0.0, 0.0, -100.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=200.0,
        height=200.0,
    )
    detector_grid = create_detector_grid(
        width=200.0, height=200.0, resolution=(20, 20)
    )

    result = run_baseline_angle_scan(
        mesh=mesh,
        angles_degrees=[0.0, 15.0],
        ray_grid_config=ray_grid_config,
        interface_sequence=SLAB_INTERFACES,
        detector=detector,
        detector_grid=detector_grid,
    )

    assert result.angle_count == 2
    assert len(result.per_angle) == 2
    for entry in result.per_angle:
        assert isinstance(entry.metrics, OpticalMetrics)
