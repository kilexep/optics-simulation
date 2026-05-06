import numpy as np
import pytest
import trimesh

from optics_simulation.geometry import (
    GeometryError,
    SurfaceCoordinateMap,
    create_subdivided_synthetic_bottle_body,
    create_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.pattern import (
    GaussianDimple,
    GaussianDimplePattern,
    compute_vertex_displacement_amounts,
    evaluate_gaussian_dimple_field,
)


def test_returns_trimesh() -> None:
    mesh = create_subdivided_synthetic_bottle_body()
    assert isinstance(mesh, trimesh.Trimesh)


def test_vertex_count_exceeds_simple_cylinder_for_same_sections() -> None:
    sections = 96
    mesh_simple = create_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=sections
    )
    mesh_sub = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=sections, height_segments=24
    )
    assert len(mesh_sub.vertices) > len(mesh_simple.vertices)
    assert len(mesh_sub.vertices) == 25 * sections + 2


def test_face_count_positive() -> None:
    mesh = create_subdivided_synthetic_bottle_body()
    assert len(mesh.faces) > 0


def test_default_mesh_is_watertight() -> None:
    mesh = create_subdivided_synthetic_bottle_body()
    assert mesh.is_watertight is True


def test_default_mesh_winding_is_consistent() -> None:
    mesh = create_subdivided_synthetic_bottle_body()
    assert mesh.is_winding_consistent is True


def test_side_face_normals_point_radially_outward() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=8, height_segments=2
    )
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    face_centers = np.asarray(mesh.triangles_center, dtype=float)
    half_h = 60.0
    margin = 1e-6
    side_mask = (
        (face_centers[:, 2] > -half_h + margin)
        & (face_centers[:, 2] < +half_h - margin)
    )
    assert side_mask.any()
    radial_xy = face_centers[side_mask, :2]
    radial_xy_norm = np.linalg.norm(radial_xy, axis=1, keepdims=True)
    radial_unit = radial_xy / np.maximum(radial_xy_norm, 1e-12)
    side_normals_xy = face_normals[side_mask, :2]
    dotp = (side_normals_xy * radial_unit).sum(axis=1)
    assert (dotp > 0.5).all()


def test_z_range_matches_height() -> None:
    height = 120.0
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=height, sections=96, height_segments=24
    )
    bounds = np.asarray(mesh.bounds, dtype=float)
    assert float(bounds[1, 2] - bounds[0, 2]) == pytest.approx(
        height, rel=1e-9
    )
    assert float(bounds[0, 2]) == pytest.approx(-height / 2.0, abs=1e-9)
    assert float(bounds[1, 2]) == pytest.approx(+height / 2.0, abs=1e-9)


def test_xy_radius_matches() -> None:
    radius = 30.0
    mesh = create_subdivided_synthetic_bottle_body(
        radius=radius, height=120.0, sections=96, height_segments=12
    )
    bounds = np.asarray(mesh.bounds, dtype=float)
    assert float(bounds[1, 0] - bounds[0, 0]) == pytest.approx(
        2.0 * radius, rel=1e-6
    )
    assert float(bounds[1, 1] - bounds[0, 1]) == pytest.approx(
        2.0 * radius, rel=1e-6
    )
    vertices = np.asarray(mesh.vertices, dtype=float)
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    assert float(radii.max()) <= radius + 1e-9


def test_centroid_near_origin() -> None:
    mesh = create_subdivided_synthetic_bottle_body()
    centroid = np.asarray(mesh.centroid, dtype=float)
    assert np.allclose(centroid, np.zeros(3), atol=1e-9)


def test_invalid_radius_raises() -> None:
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(radius=0.0)
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(radius=-1.0)


def test_invalid_height_raises() -> None:
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(height=0.0)
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(height=-5.0)


def test_invalid_sections_raises() -> None:
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(sections=7)
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(sections=0)
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(sections=-3)


def test_invalid_height_segments_raises() -> None:
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(height_segments=0)
    with pytest.raises(GeometryError):
        create_subdivided_synthetic_bottle_body(height_segments=-1)


def test_height_segments_at_minimum_is_accepted() -> None:
    mesh = create_subdivided_synthetic_bottle_body(height_segments=1)
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight is True


def test_surface_coordinates_compatible() -> None:
    mesh = create_subdivided_synthetic_bottle_body()
    coords = create_vertex_surface_coordinates(mesh)
    assert isinstance(coords, SurfaceCoordinateMap)
    assert coords.point_count == len(mesh.vertices)


def test_v_includes_interior_values() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=96, height_segments=24
    )
    coords = create_vertex_surface_coordinates(mesh)
    v = np.asarray(coords.v, dtype=float)
    interior = (v > 1e-6) & (v < 1.0 - 1e-6)
    assert int(interior.sum()) > 0


def test_uv_in_unit_ranges() -> None:
    mesh = create_subdivided_synthetic_bottle_body()
    coords = create_vertex_surface_coordinates(mesh)
    assert float(coords.u.min()) >= 0.0
    assert float(coords.u.max()) < 1.0 + 1e-12
    assert float(coords.v.min()) >= -1e-12
    assert float(coords.v.max()) <= 1.0 + 1e-12


def test_active_vertex_count_positive_with_central_dimple() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=48, height_segments=12
    )
    surface_map = create_vertex_surface_coordinates(mesh)

    dimple = GaussianDimple(
        center_u=0.5, center_v=0.5,
        amplitude=1.0, sigma_u=0.1, sigma_v=0.1,
    )
    resolution = (8, 16)
    depth_field = evaluate_gaussian_dimple_field(
        [dimple], resolution=resolution, clip=True
    )
    pattern = GaussianDimplePattern(
        dimples=(dimple,),
        depth_field=depth_field,
        resolution=resolution,
        max_depth=0.25,
        seed=0,
    )

    result = compute_vertex_displacement_amounts(
        mesh,
        surface_map,
        pattern,
        active_threshold=0.0,
        exclude_v_boundary_epsilon=0.02,
    )
    assert int(result.active_mask.sum()) > 0


def test_existing_simple_function_still_works() -> None:
    sections = 96
    mesh = create_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=sections
    )
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) == 2 * sections + 2
    assert mesh.is_watertight is True
