import numpy as np
import pytest
import trimesh

from optics_simulation.optics import (
    IntersectionResult,
    OpticsError,
    RayBundle,
    intersect_rays,
    make_ray_bundle,
    parallel_ray_grid,
)


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=15.0, height=200.0, sections=64)


def test_make_ray_bundle_normalizes_directions() -> None:
    bundle = make_ray_bundle(
        origins=[[0.0, 0.0, 0.0]],
        directions=[[0.0, 0.0, 5.0]],
    )
    assert isinstance(bundle, RayBundle)
    assert bundle.ray_count == 1
    np.testing.assert_allclose(np.linalg.norm(bundle.directions, axis=1), [1.0])
    np.testing.assert_allclose(bundle.directions[0], [0.0, 0.0, 1.0])


def test_make_ray_bundle_preserves_origins() -> None:
    origins = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    bundle = make_ray_bundle(origins, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(bundle.origins, origins)
    assert bundle.ray_count == 2


def test_make_ray_bundle_shape_mismatch_raises() -> None:
    with pytest.raises(OpticsError, match="same shape"):
        make_ray_bundle(
            origins=[[0.0, 0.0, 0.0]],
            directions=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        )


def test_make_ray_bundle_wrong_dim_raises() -> None:
    with pytest.raises(OpticsError, match=r"shape \(N, 3\)"):
        make_ray_bundle(
            origins=[[0.0, 0.0]],
            directions=[[0.0, 1.0]],
        )


def test_make_ray_bundle_zero_direction_raises() -> None:
    with pytest.raises(OpticsError, match="zero length"):
        make_ray_bundle(
            origins=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            directions=[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        )


def test_parallel_ray_grid_count_and_norm() -> None:
    rays = parallel_ray_grid(
        origin_plane_z=10.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-1.0, 1.0),
        y_range=(-2.0, 2.0),
        nx=4,
        ny=3,
    )
    assert rays.ray_count == 12
    assert rays.origins.shape == (12, 3)
    assert rays.directions.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(rays.directions, axis=1), 1.0)
    np.testing.assert_allclose(rays.directions, np.tile([0.0, 0.0, -1.0], (12, 1)))
    np.testing.assert_allclose(rays.origins[:, 2], 10.0)
    assert rays.origins[:, 0].min() == pytest.approx(-1.0)
    assert rays.origins[:, 0].max() == pytest.approx(1.0)
    assert rays.origins[:, 1].min() == pytest.approx(-2.0)
    assert rays.origins[:, 1].max() == pytest.approx(2.0)


def test_parallel_ray_grid_invalid_count_raises() -> None:
    with pytest.raises(OpticsError, match="nx and ny"):
        parallel_ray_grid(
            origin_plane_z=0.0,
            direction=(0.0, 0.0, -1.0),
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            nx=0,
            ny=1,
        )


def test_parallel_ray_grid_zero_direction_raises() -> None:
    with pytest.raises(OpticsError, match="zero length"):
        parallel_ray_grid(
            origin_plane_z=0.0,
            direction=(0.0, 0.0, 0.0),
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            nx=2,
            ny=2,
        )


def test_intersect_rays_box_top_face_hit(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    result = intersect_rays(box_mesh, rays)
    assert isinstance(result, IntersectionResult)
    assert result.ray_count == 9
    assert result.hit_mask.all()
    np.testing.assert_allclose(result.hit_points[:, 2], 5.0, atol=1e-6)
    np.testing.assert_allclose(result.t_hit, 15.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result.hit_normals, axis=1), 1.0)
    np.testing.assert_allclose(result.hit_normals, np.tile([0.0, 0.0, 1.0], (9, 1)),
                                atol=1e-9)
    assert (result.primitive_ids >= 0).all()


def test_intersect_rays_box_pointing_away_misses(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, 1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=2,
        ny=2,
    )
    result = intersect_rays(box_mesh, rays)
    assert result.ray_count == 4
    assert not result.hit_mask.any()
    assert np.isinf(result.t_hit).all()
    assert np.isnan(result.hit_points).all()
    assert np.isnan(result.hit_normals).all()
    assert (result.primitive_ids == -1).all()


def test_intersect_rays_mixed_hit_and_miss(box_mesh: trimesh.Trimesh) -> None:
    origins = np.array([
        [0.0, 0.0, 20.0],
        [100.0, 100.0, 20.0],
        [2.0, 2.0, 20.0],
    ])
    directions = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ])
    rays = make_ray_bundle(origins, directions)
    result = intersect_rays(box_mesh, rays)
    assert result.ray_count == 3
    np.testing.assert_array_equal(result.hit_mask, [True, False, True])
    assert result.t_hit[0] == pytest.approx(15.0, abs=1e-6)
    assert np.isinf(result.t_hit[1])
    assert result.t_hit[2] == pytest.approx(15.0, abs=1e-6)
    assert np.isnan(result.hit_points[1]).all()
    assert np.isnan(result.hit_normals[1]).all()
    assert result.primitive_ids[1] == -1
    assert result.primitive_ids[0] >= 0
    assert result.primitive_ids[2] >= 0


def test_intersect_rays_cylinder_radial_hit(cylinder_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle(
        origins=[[100.0, 0.0, 0.0]],
        directions=[[-1.0, 0.0, 0.0]],
    )
    result = intersect_rays(cylinder_mesh, rays)
    assert result.hit_mask[0]
    radial = float(np.hypot(result.hit_points[0, 0], result.hit_points[0, 1]))
    assert radial == pytest.approx(15.0, rel=0.02)
    assert result.t_hit[0] == pytest.approx(100.0 - 15.0, rel=0.02)
    assert np.isclose(np.linalg.norm(result.hit_normals[0]), 1.0)


def test_intersect_rays_result_shape_invariants(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=4,
        ny=5,
    )
    result = intersect_rays(box_mesh, rays)
    n = rays.ray_count
    assert result.ray_count == n
    assert result.t_hit.shape == (n,)
    assert result.hit_mask.shape == (n,)
    assert result.hit_points.shape == (n, 3)
    assert result.hit_normals.shape == (n, 3)
    assert result.primitive_ids.shape == (n,)
    assert result.primitive_ids.dtype == np.int64


def test_intersect_rays_rejects_non_bundle(box_mesh: trimesh.Trimesh) -> None:
    with pytest.raises(OpticsError, match="RayBundle"):
        intersect_rays(box_mesh, object())  # type: ignore[arg-type]


def test_intersect_rays_empty_bundle(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle(
        origins=np.zeros((0, 3)),
        directions=np.zeros((0, 3)),
    )
    result = intersect_rays(box_mesh, rays)
    assert result.ray_count == 0
    assert result.t_hit.shape == (0,)
    assert result.hit_mask.shape == (0,)
    assert result.hit_points.shape == (0, 3)
    assert result.hit_normals.shape == (0, 3)
    assert result.primitive_ids.shape == (0,)
