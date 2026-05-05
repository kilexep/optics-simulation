import numpy as np
import pytest
import trimesh

from optics_simulation.geometry import (
    GeometryError,
    HitSurfaceCoordinates,
    SurfaceCoordinateMap,
    create_synthetic_bottle_body,
    create_vertex_surface_coordinates,
    hit_to_surface_coordinates,
)
from optics_simulation.optics import (
    IntersectionResult,
    intersect_rays,
    parallel_ray_grid,
)


def _single_triangle_mesh() -> trimesh.Trimesh:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _surface_map_from_arrays(
    u: np.ndarray,
    v: np.ndarray,
    *,
    z_min: float = 0.0,
    z_max: float = 1.0,
    center_xy: tuple[float, float] = (0.0, 0.0),
) -> SurfaceCoordinateMap:
    return SurfaceCoordinateMap(
        u=np.asarray(u, dtype=float),
        v=np.asarray(v, dtype=float),
        point_count=int(len(u)),
        z_min=float(z_min),
        z_max=float(z_max),
        center_xy=center_xy,
    )


def _intersection_with_one_hit(
    hit_point: tuple[float, float, float],
    primitive_id: int,
    *,
    ray_count: int = 1,
    hit_index: int = 0,
) -> IntersectionResult:
    t_hit = np.full(ray_count, np.inf, dtype=float)
    hit_mask = np.zeros(ray_count, dtype=bool)
    hit_points = np.full((ray_count, 3), np.nan, dtype=float)
    hit_normals = np.full((ray_count, 3), np.nan, dtype=float)
    primitive_ids = np.full(ray_count, -1, dtype=np.int64)

    hit_mask[hit_index] = True
    t_hit[hit_index] = 0.0
    hit_points[hit_index] = hit_point
    hit_normals[hit_index] = (0.0, 0.0, 1.0)
    primitive_ids[hit_index] = primitive_id

    return IntersectionResult(
        t_hit=t_hit,
        hit_mask=hit_mask,
        hit_points=hit_points,
        hit_normals=hit_normals,
        primitive_ids=primitive_ids,
        ray_count=ray_count,
    )


def _bottle_mesh_with_grid_intersection():
    mesh = create_synthetic_bottle_body(radius=30.0, height=120.0, sections=48)
    rays = parallel_ray_grid(
        origin_plane_z=200.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-50.0, 50.0),
        y_range=(-70.0, 70.0),
        nx=7,
        ny=7,
    )
    intersection = intersect_rays(mesh, rays)
    surface_map = create_vertex_surface_coordinates(mesh)
    return mesh, rays, intersection, surface_map


def test_returns_n_aligned_arrays() -> None:
    mesh, rays, intersection, surface_map = _bottle_mesh_with_grid_intersection()
    coords = hit_to_surface_coordinates(mesh, intersection, surface_map)
    assert isinstance(coords, HitSurfaceCoordinates)
    assert coords.ray_count == intersection.ray_count
    assert coords.u.shape == (intersection.ray_count,)
    assert coords.v.shape == (intersection.ray_count,)
    assert coords.barycentric_weights.shape == (intersection.ray_count, 3)


def test_hit_mask_and_primitive_ids_match_intersection() -> None:
    mesh, _rays, intersection, surface_map = _bottle_mesh_with_grid_intersection()
    coords = hit_to_surface_coordinates(mesh, intersection, surface_map)
    assert np.array_equal(coords.hit_mask, intersection.hit_mask)
    assert np.array_equal(coords.primitive_ids, intersection.primitive_ids)


def test_hit_uv_in_unit_ranges() -> None:
    mesh, _rays, intersection, surface_map = _bottle_mesh_with_grid_intersection()
    coords = hit_to_surface_coordinates(mesh, intersection, surface_map)
    hits = intersection.hit_mask
    assert hits.any(), "expected at least one hit for this fixture"
    u_hits = coords.u[hits]
    v_hits = coords.v[hits]
    assert float(u_hits.min()) >= 0.0
    assert float(u_hits.max()) < 1.0 + 1e-12
    assert float(v_hits.min()) >= -1e-12
    assert float(v_hits.max()) <= 1.0 + 1e-12


def test_miss_rays_have_nan_uv_and_bary() -> None:
    mesh, _rays, intersection, surface_map = _bottle_mesh_with_grid_intersection()
    coords = hit_to_surface_coordinates(mesh, intersection, surface_map)
    miss = ~intersection.hit_mask
    if not miss.any():
        pytest.skip("fixture happens to have no misses")
    assert np.isnan(coords.u[miss]).all()
    assert np.isnan(coords.v[miss]).all()
    assert np.isnan(coords.barycentric_weights[miss]).all()


def test_barycentric_sum_is_one_for_hits() -> None:
    mesh, _rays, intersection, surface_map = _bottle_mesh_with_grid_intersection()
    coords = hit_to_surface_coordinates(mesh, intersection, surface_map)
    hits = intersection.hit_mask
    assert hits.any()
    bary_hits = coords.barycentric_weights[hits]
    sums = bary_hits.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_vertex_count_mismatch_raises() -> None:
    mesh = _single_triangle_mesh()
    bad_map = _surface_map_from_arrays(
        u=np.array([0.0, 0.0]),
        v=np.array([0.0, 0.0]),
    )
    intersection = _intersection_with_one_hit(
        hit_point=(1.0 / 3.0, 1.0 / 3.0, 0.0), primitive_id=0
    )
    with pytest.raises(GeometryError, match="point_count"):
        hit_to_surface_coordinates(mesh, intersection, bad_map)


def test_invalid_primitive_id_raises() -> None:
    mesh = _single_triangle_mesh()
    surface_map = _surface_map_from_arrays(
        u=np.array([0.1, 0.2, 0.3]),
        v=np.array([0.4, 0.5, 0.6]),
    )
    intersection = _intersection_with_one_hit(
        hit_point=(1.0 / 3.0, 1.0 / 3.0, 0.0), primitive_id=99
    )
    with pytest.raises(GeometryError, match="out-of-range face index"):
        hit_to_surface_coordinates(mesh, intersection, surface_map)


def test_degenerate_triangle_raises() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    surface_map = _surface_map_from_arrays(
        u=np.array([0.0, 0.5, 1.0 - 1e-12]),
        v=np.array([0.0, 0.5, 1.0]),
    )
    intersection = _intersection_with_one_hit(
        hit_point=(0.5, 0.0, 0.0), primitive_id=0
    )
    with pytest.raises(GeometryError, match="degenerate triangle"):
        hit_to_surface_coordinates(mesh, intersection, surface_map)


def test_triangle_center_interpolates_to_vertex_uv_average() -> None:
    mesh = _single_triangle_mesh()
    surface_map = _surface_map_from_arrays(
        u=np.array([0.1, 0.2, 0.3]),
        v=np.array([0.4, 0.5, 0.6]),
    )
    intersection = _intersection_with_one_hit(
        hit_point=(1.0 / 3.0, 1.0 / 3.0, 0.0), primitive_id=0
    )
    coords = hit_to_surface_coordinates(mesh, intersection, surface_map)
    assert coords.u[0] == pytest.approx(0.2, abs=1e-9)
    assert coords.v[0] == pytest.approx(0.5, abs=1e-9)
    assert np.allclose(
        coords.barycentric_weights[0],
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        atol=1e-9,
    )


def test_u_wrap_interpolation_near_zero() -> None:
    mesh = _single_triangle_mesh()
    surface_map = _surface_map_from_arrays(
        u=np.array([0.99, 0.01, 0.02]),
        v=np.array([0.5, 0.5, 0.5]),
    )
    intersection = _intersection_with_one_hit(
        hit_point=(1.0 / 3.0, 1.0 / 3.0, 0.0), primitive_id=0
    )
    coords = hit_to_surface_coordinates(mesh, intersection, surface_map)
    u_hit = float(coords.u[0])
    assert min(u_hit, 1.0 - u_hit) < 0.05
    assert coords.v[0] == pytest.approx(0.5, abs=1e-9)


def test_synthetic_bottle_integration_smoke() -> None:
    mesh, _rays, intersection, surface_map = _bottle_mesh_with_grid_intersection()
    coords = hit_to_surface_coordinates(mesh, intersection, surface_map)
    assert coords.ray_count == 7 * 7
    assert int(coords.hit_mask.sum()) > 0
    hits = intersection.hit_mask
    assert (coords.u[hits] >= 0.0).all()
    assert (coords.u[hits] < 1.0 + 1e-12).all()
    assert (coords.v[hits] >= -1e-12).all()
    assert (coords.v[hits] <= 1.0 + 1e-12).all()
    miss = ~hits
    if miss.any():
        assert np.isnan(coords.u[miss]).all()
        assert np.isnan(coords.v[miss]).all()
