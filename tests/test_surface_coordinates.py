import numpy as np
import pytest
import trimesh

from optics_simulation.geometry import (
    GeometryError,
    SurfaceCoordinateMap,
    create_vertex_surface_coordinates,
    normalize_cylindrical_coordinates,
    point_to_normalized_cylindrical,
)


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=15.0, height=200.0, sections=64)


def test_u_at_positive_x() -> None:
    u, v = point_to_normalized_cylindrical((1.0, 0.0, 5.0), z_min=0.0, z_max=10.0)
    assert u == pytest.approx(0.0, abs=1e-9)
    assert v == pytest.approx(0.5)


def test_u_at_positive_y() -> None:
    u, _ = point_to_normalized_cylindrical((0.0, 1.0, 5.0), z_min=0.0, z_max=10.0)
    assert u == pytest.approx(0.25, abs=1e-9)


def test_u_at_negative_x() -> None:
    u, _ = point_to_normalized_cylindrical((-1.0, 0.0, 5.0), z_min=0.0, z_max=10.0)
    assert u == pytest.approx(0.5, abs=1e-9)


def test_u_at_negative_y() -> None:
    u, _ = point_to_normalized_cylindrical((0.0, -1.0, 5.0), z_min=0.0, z_max=10.0)
    assert u == pytest.approx(0.75, abs=1e-9)


def test_v_at_z_min_is_zero() -> None:
    _, v = point_to_normalized_cylindrical((1.0, 0.0, 0.0), z_min=0.0, z_max=10.0)
    assert v == pytest.approx(0.0)


def test_v_at_z_max_is_one() -> None:
    _, v = point_to_normalized_cylindrical((1.0, 0.0, 10.0), z_min=0.0, z_max=10.0)
    assert v == pytest.approx(1.0)


def test_v_midpoint() -> None:
    _, v = point_to_normalized_cylindrical((1.0, 0.0, 5.0), z_min=0.0, z_max=10.0)
    assert v == pytest.approx(0.5)


def test_z_min_equals_z_max_raises_on_point() -> None:
    with pytest.raises(GeometryError, match="z_min"):
        point_to_normalized_cylindrical((1.0, 0.0, 5.0), z_min=5.0, z_max=5.0)


def test_z_min_equals_z_max_raises_on_array() -> None:
    pts = np.array([[1.0, 0.0, 5.0]])
    with pytest.raises(GeometryError, match="z_min"):
        normalize_cylindrical_coordinates(pts, z_min=5.0, z_max=5.0)


def test_center_xy_offset_applied() -> None:
    cx, cy = 10.0, 5.0
    u, _ = point_to_normalized_cylindrical(
        (cx + 1.0, cy, 5.0), z_min=0.0, z_max=10.0, center_xy=(cx, cy),
    )
    assert u == pytest.approx(0.0, abs=1e-9)


def test_center_xy_offset_changes_result() -> None:
    point = (1.0, 1.0, 5.0)
    u_no_offset, _ = point_to_normalized_cylindrical(point, 0.0, 10.0)
    u_with_offset, _ = point_to_normalized_cylindrical(
        point, 0.0, 10.0, center_xy=(1.0, 0.0),
    )
    assert u_no_offset == pytest.approx(0.125, abs=1e-9)
    assert u_with_offset == pytest.approx(0.25, abs=1e-9)


def test_array_function_shapes_and_values() -> None:
    pts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 5.0],
        [-1.0, 0.0, 10.0],
        [0.0, -1.0, 7.5],
    ])
    u, v = normalize_cylindrical_coordinates(pts, z_min=0.0, z_max=10.0)
    assert u.shape == (4,)
    assert v.shape == (4,)
    np.testing.assert_allclose(u, [0.0, 0.25, 0.5, 0.75], atol=1e-9)
    np.testing.assert_allclose(v, [0.0, 0.5, 1.0, 0.75])


def test_array_function_derives_z_bounds_from_points() -> None:
    pts = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 10.0],
    ])
    _, v = normalize_cylindrical_coordinates(pts)
    np.testing.assert_allclose(v, [0.0, 1.0])


def test_u_strictly_below_one_around_full_circle() -> None:
    angles = np.linspace(-np.pi, np.pi, 1000, endpoint=True)
    pts = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    u, _ = normalize_cylindrical_coordinates(pts, z_min=0.0, z_max=1.0)
    assert (u >= 0.0).all()
    assert (u < 1.0).all()


def test_cylinder_mesh_uv_in_range(cylinder_mesh: trimesh.Trimesh) -> None:
    smap = create_vertex_surface_coordinates(cylinder_mesh)
    assert isinstance(smap, SurfaceCoordinateMap)
    assert smap.u.shape == (smap.point_count,)
    assert smap.v.shape == (smap.point_count,)
    assert (smap.u >= 0.0).all()
    assert (smap.u < 1.0).all()
    assert (smap.v >= 0.0).all()
    assert (smap.v <= 1.0).all()


def test_surface_coordinate_map_fields(cylinder_mesh: trimesh.Trimesh) -> None:
    smap = create_vertex_surface_coordinates(cylinder_mesh)
    assert smap.coordinate_type == "normalized_cylindrical"
    assert smap.point_count == len(cylinder_mesh.vertices)
    assert smap.z_min == pytest.approx(-100.0, abs=0.5)
    assert smap.z_max == pytest.approx(100.0, abs=0.5)
    assert smap.center_xy[0] == pytest.approx(0.0, abs=0.5)
    assert smap.center_xy[1] == pytest.approx(0.0, abs=0.5)


def test_invalid_points_shape_raises() -> None:
    bad = np.zeros((5, 2))
    with pytest.raises(GeometryError, match="shape"):
        normalize_cylindrical_coordinates(bad, z_min=0.0, z_max=1.0)


def test_invalid_points_ndim_raises() -> None:
    bad = np.zeros(9)
    with pytest.raises(GeometryError, match="shape"):
        normalize_cylindrical_coordinates(bad, z_min=0.0, z_max=1.0)


def test_invalid_point_size_raises() -> None:
    with pytest.raises(GeometryError, match="3 components"):
        point_to_normalized_cylindrical((1.0, 2.0), z_min=0.0, z_max=1.0)
