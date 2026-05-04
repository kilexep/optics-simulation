import numpy as np
import pytest
import trimesh

from optics_simulation.optics import (
    IntersectionResult,
    OpticsError,
    PropagationResult,
    RayBundle,
    intersect_rays,
    make_ray_bundle,
    parallel_ray_grid,
    propagate_through_interface,
)

IOR_AIR = 1.00028
IOR_PET = 1.575


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=15.0, height=200.0, sections=64)


def _shoot(box: trimesh.Trimesh, origins, directions):
    rays = make_ray_bundle(origins, directions)
    inter = intersect_rays(box, rays)
    return rays, inter


def test_normal_incidence_air_to_pet_single_ray(box_mesh: trimesh.Trimesh) -> None:
    rays, inter = _shoot(box_mesh, [[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)
    assert isinstance(result, PropagationResult)
    assert result.next_rays.ray_count == 1
    assert result.active_mask.tolist() == [True]
    assert result.tir_mask.tolist() == [False]
    np.testing.assert_array_equal(result.source_ray_indices, [0])
    np.testing.assert_allclose(
        result.next_rays.directions[0], [0.0, 0.0, -1.0], atol=1e-12
    )
    assert result.reflectance[0] + result.transmittance[0] == pytest.approx(
        1.0, abs=1e-12
    )


def test_next_ray_origin_is_offset_along_refracted_direction(
    box_mesh: trimesh.Trimesh,
) -> None:
    epsilon = 0.01
    rays, inter = _shoot(box_mesh, [[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = propagate_through_interface(
        rays, inter, IOR_AIR, IOR_PET, epsilon=epsilon
    )
    src = result.source_ray_indices[0]
    expected_origin = inter.hit_points[src] + epsilon * result.next_rays.directions[0]
    np.testing.assert_allclose(result.next_rays.origins[0], expected_origin, atol=1e-12)


def test_next_directions_are_unit_norm(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    inter = intersect_rays(box_mesh, rays)
    result = propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)
    norms = np.linalg.norm(result.next_rays.directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_mixed_hit_and_miss_compact_layout(box_mesh: trimesh.Trimesh) -> None:
    origins = np.array(
        [[0.0, 0.0, 20.0], [100.0, 100.0, 20.0], [2.0, 2.0, 20.0]]
    )
    directions = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
    )
    rays, inter = _shoot(box_mesh, origins, directions)
    result = propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)
    assert result.next_rays.ray_count == 2
    np.testing.assert_array_equal(result.active_mask, [True, False, True])
    np.testing.assert_array_equal(result.tir_mask, [False, False, False])
    np.testing.assert_array_equal(result.source_ray_indices, [0, 2])
    assert np.isnan(result.reflectance[1])
    assert np.isnan(result.transmittance[1])
    assert not np.isnan(result.reflectance[0])
    assert not np.isnan(result.reflectance[2])


def test_dense_r_t_align_with_active_mask(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=2,
        ny=2,
    )
    inter = intersect_rays(box_mesh, rays)
    result = propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)
    active_r = result.reflectance[result.active_mask]
    active_t = result.transmittance[result.active_mask]
    np.testing.assert_allclose(active_r + active_t, 1.0, atol=1e-12)


def test_total_internal_reflection_excludes_from_next_rays(
    box_mesh: trimesh.Trimesh,
) -> None:
    # PET -> AIR at 60 deg incidence -> TIR for every ray.
    # Origins are kept close to the top face (z = 5) so the steep ray
    # actually hits the box rather than overshooting in x.
    angle = np.deg2rad(60.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    rays, inter = _shoot(
        box_mesh,
        [[0.0, 0.0, 6.0], [-1.0, 0.0, 6.0]],
        [[sx, 0.0, -sz], [sx, 0.0, -sz]],
    )
    assert inter.hit_mask.all()  # both hit the top face
    result = propagate_through_interface(rays, inter, IOR_PET, IOR_AIR)
    assert result.next_rays.ray_count == 0
    assert result.active_mask.any() == False  # noqa: E712
    assert result.tir_mask.tolist() == [True, True]
    np.testing.assert_array_equal(result.reflectance, [1.0, 1.0])
    np.testing.assert_array_equal(result.transmittance, [0.0, 0.0])


def test_partial_tir_and_transmit(box_mesh: trimesh.Trimesh) -> None:
    # ray 0: normal incidence (transmits), ray 1: 60 deg (TIR for PET->AIR)
    angle = np.deg2rad(60.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    rays, inter = _shoot(
        box_mesh,
        [[0.0, 0.0, 20.0], [-1.0, 0.0, 6.0]],
        [[0.0, 0.0, -1.0], [sx, 0.0, -sz]],
    )
    assert inter.hit_mask.all()
    result = propagate_through_interface(rays, inter, IOR_PET, IOR_AIR)
    np.testing.assert_array_equal(result.active_mask, [True, False])
    np.testing.assert_array_equal(result.tir_mask, [False, True])
    assert result.next_rays.ray_count == 1
    np.testing.assert_array_equal(result.source_ray_indices, [0])
    assert result.reflectance[1] == 1.0
    assert result.transmittance[1] == 0.0


def test_all_miss_produces_empty_result(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, 1.0),  # pointing away
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        nx=2,
        ny=2,
    )
    inter = intersect_rays(box_mesh, rays)
    result = propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)
    assert result.next_rays.ray_count == 0
    assert result.source_ray_indices.shape == (0,)
    assert result.source_ray_indices.dtype == np.int64
    assert not result.active_mask.any()
    assert not result.tir_mask.any()
    assert np.isnan(result.reflectance).all()
    assert np.isnan(result.transmittance).all()


def test_empty_input_bundle(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle(np.zeros((0, 3)), np.zeros((0, 3)))
    inter = intersect_rays(box_mesh, rays)
    result = propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)
    assert result.next_rays.ray_count == 0
    assert result.active_mask.shape == (0,)
    assert result.tir_mask.shape == (0,)
    assert result.reflectance.shape == (0,)
    assert result.transmittance.shape == (0,)
    assert result.source_ray_indices.shape == (0,)


def test_cylinder_radial_hit_propagates(cylinder_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[100.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]])
    inter = intersect_rays(cylinder_mesh, rays)
    assert inter.hit_mask[0]
    result = propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)
    assert result.next_rays.ray_count == 1
    np.testing.assert_array_equal(result.source_ray_indices, [0])
    assert np.linalg.norm(result.next_rays.directions[0]) == pytest.approx(1.0)


def test_ray_count_mismatch_raises(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    inter = intersect_rays(
        box_mesh,
        make_ray_bundle(
            [[0.0, 0.0, 20.0], [1.0, 0.0, 20.0]],
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
        ),
    )
    with pytest.raises(OpticsError, match="ray_count"):
        propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)


def test_negative_epsilon_raises(box_mesh: trimesh.Trimesh) -> None:
    rays, inter = _shoot(box_mesh, [[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="epsilon"):
        propagate_through_interface(rays, inter, IOR_AIR, IOR_PET, epsilon=-1.0)


def test_zero_epsilon_allowed(box_mesh: trimesh.Trimesh) -> None:
    rays, inter = _shoot(box_mesh, [[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = propagate_through_interface(
        rays, inter, IOR_AIR, IOR_PET, epsilon=0.0
    )
    src = result.source_ray_indices[0]
    np.testing.assert_allclose(
        result.next_rays.origins[0], inter.hit_points[src], atol=1e-12
    )


def test_nonpositive_eta_raises(box_mesh: trimesh.Trimesh) -> None:
    rays, inter = _shoot(box_mesh, [[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="positive"):
        propagate_through_interface(rays, inter, 0.0, IOR_PET)
    with pytest.raises(OpticsError, match="positive"):
        propagate_through_interface(rays, inter, IOR_AIR, -1.0)


def test_mask_invariants(box_mesh: trimesh.Trimesh) -> None:
    # Mix of miss, transmit, and TIR rays in a single bundle.
    angle = np.deg2rad(60.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    origins = np.array(
        [
            [0.0, 0.0, 20.0],       # transmit
            [100.0, 100.0, 20.0],   # miss
            [-1.0, 0.0, 6.0],       # TIR (with PET->AIR)
        ]
    )
    directions = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [sx, 0.0, -sz],
        ]
    )
    rays, inter = _shoot(box_mesh, origins, directions)
    result = propagate_through_interface(rays, inter, IOR_PET, IOR_AIR)

    assert np.all(result.active_mask <= inter.hit_mask)
    assert np.all(result.tir_mask <= inter.hit_mask)
    assert not np.any(result.active_mask & result.tir_mask)


def test_schema_shape_and_dtype_invariants(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=4,
    )
    inter = intersect_rays(box_mesh, rays)
    result = propagate_through_interface(rays, inter, IOR_AIR, IOR_PET)
    n = rays.ray_count
    m = int(result.active_mask.sum())
    assert result.active_mask.shape == (n,)
    assert result.tir_mask.shape == (n,)
    assert result.reflectance.shape == (n,)
    assert result.transmittance.shape == (n,)
    assert result.source_ray_indices.shape == (m,)
    assert result.source_ray_indices.dtype == np.int64
    assert result.next_rays.origins.shape == (m, 3)
    assert result.next_rays.directions.shape == (m, 3)
    assert result.next_rays.ray_count == m
