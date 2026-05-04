import numpy as np
import pytest
import trimesh

from optics_simulation.optics import (
    OpticsError,
    RayBundle,
    SingleInterfacePipelineResult,
    make_ray_bundle,
    parallel_ray_grid,
    run_single_interface_pipeline,
)

IOR_AIR = 1.00028
IOR_PET = 1.575


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=15.0, height=200.0, sections=64)


def _assert_invariants(result: SingleInterfacePipelineResult) -> None:
    """Check every count invariant promised by the pipeline contract."""
    assert result.ray_count == result.rays.ray_count
    assert result.hit_count + result.miss_count == result.ray_count
    assert result.transmitted_count == int(result.propagation.active_mask.sum())
    assert result.tir_count == int(result.propagation.tir_mask.sum())
    assert result.transmitted_count == result.propagation.next_rays.ray_count
    np.testing.assert_array_equal(
        result.propagation.source_ray_indices,
        np.where(result.propagation.active_mask)[0],
    )
    assert (
        result.transmitted_count + result.tir_count + result.miss_count
        == result.ray_count
    )


def test_box_partial_hit_and_miss_grid(box_mesh: trimesh.Trimesh) -> None:
    # 4x4 grid spanning x,y in [-8, 8] from z=20 going -z; some rays
    # land on the box (|x|,|y| <= 5) and some miss.
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-8.0, 8.0),
        y_range=(-8.0, 8.0),
        nx=4,
        ny=4,
    )
    result = run_single_interface_pipeline(box_mesh, rays, IOR_AIR, IOR_PET)
    assert isinstance(result, SingleInterfacePipelineResult)
    assert result.hit_count > 0
    assert result.miss_count > 0
    assert result.tir_count == 0  # normal incidence, no TIR
    assert result.transmitted_count == result.hit_count
    assert result.propagation.next_rays.ray_count == result.transmitted_count
    _assert_invariants(result)


def test_box_all_miss_grid(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, 1.0),  # pointing away
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        nx=2,
        ny=2,
    )
    result = run_single_interface_pipeline(box_mesh, rays, IOR_AIR, IOR_PET)
    assert result.ray_count == 4
    assert result.hit_count == 0
    assert result.miss_count == 4
    assert result.transmitted_count == 0
    assert result.tir_count == 0
    assert result.propagation.next_rays.ray_count == 0
    _assert_invariants(result)


def test_box_all_hit_grid(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    result = run_single_interface_pipeline(box_mesh, rays, IOR_AIR, IOR_PET)
    assert result.hit_count == 9
    assert result.miss_count == 0
    assert result.transmitted_count == 9
    assert result.tir_count == 0
    _assert_invariants(result)


def test_cylinder_radial_ray_transmits(cylinder_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[100.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]])
    result = run_single_interface_pipeline(
        cylinder_mesh, rays, IOR_AIR, IOR_PET
    )
    assert result.hit_count == 1
    assert result.transmitted_count == 1
    assert result.tir_count == 0
    assert result.miss_count == 0
    next_dir = result.propagation.next_rays.directions[0]
    assert np.linalg.norm(next_dir) == pytest.approx(1.0, abs=1e-12)
    _assert_invariants(result)


def test_pet_to_air_60deg_all_tir(box_mesh: trimesh.Trimesh) -> None:
    # Build two rays that hit the +z face of the box (z=5) at exactly
    # 60 deg from the surface normal. Origin is held close to the
    # face so the steep ray cannot overshoot in x.
    angle = np.deg2rad(60.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    direction = np.array([sx, 0.0, -sz])

    rays = make_ray_bundle(
        origins=[[0.0, 0.0, 6.0], [-1.0, 0.0, 6.0]],
        directions=[direction.tolist(), direction.tolist()],
    )
    result = run_single_interface_pipeline(box_mesh, rays, IOR_PET, IOR_AIR)

    assert result.hit_count == 2
    # confirm geometry: the actual incidence angle is exactly 60 deg
    # against the recovered hit normal (top face -> +z).
    for i in range(2):
        normal = result.intersection.hit_normals[i]
        cos_i = float(-np.dot(rays.directions[i], normal))
        assert cos_i == pytest.approx(np.cos(angle), abs=1e-9)
        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-9)

    assert result.tir_count == 2
    assert result.transmitted_count == 0
    assert result.propagation.next_rays.ray_count == 0
    _assert_invariants(result)


def test_pet_to_air_partial_tir_and_transmit(box_mesh: trimesh.Trimesh) -> None:
    # ray 0: normal incidence (transmits), ray 1: 60 deg (TIR)
    angle = np.deg2rad(60.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    rays = make_ray_bundle(
        origins=[[0.0, 0.0, 20.0], [-1.0, 0.0, 6.0]],
        directions=[[0.0, 0.0, -1.0], [sx, 0.0, -sz]],
    )
    result = run_single_interface_pipeline(box_mesh, rays, IOR_PET, IOR_AIR)
    assert result.hit_count == 2
    assert result.transmitted_count == 1
    assert result.tir_count == 1
    assert result.miss_count == 0
    _assert_invariants(result)


def test_invalid_mesh_type_raises(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="trimesh.Trimesh"):
        run_single_interface_pipeline("not a mesh", rays, IOR_AIR, IOR_PET)  # type: ignore[arg-type]


def test_invalid_rays_type_raises(box_mesh: trimesh.Trimesh) -> None:
    with pytest.raises(OpticsError, match="RayBundle"):
        run_single_interface_pipeline(box_mesh, object(), IOR_AIR, IOR_PET)  # type: ignore[arg-type]


def test_nonpositive_eta_raises(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="positive"):
        run_single_interface_pipeline(box_mesh, rays, 0.0, IOR_PET)
    with pytest.raises(OpticsError, match="positive"):
        run_single_interface_pipeline(box_mesh, rays, IOR_AIR, -0.5)


def test_negative_epsilon_raises(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="epsilon"):
        run_single_interface_pipeline(
            box_mesh, rays, IOR_AIR, IOR_PET, epsilon=-1.0
        )


def test_empty_bundle(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle(np.zeros((0, 3)), np.zeros((0, 3)))
    result = run_single_interface_pipeline(box_mesh, rays, IOR_AIR, IOR_PET)
    assert result.ray_count == 0
    assert result.hit_count == 0
    assert result.miss_count == 0
    assert result.transmitted_count == 0
    assert result.tir_count == 0
    assert result.propagation.next_rays.ray_count == 0
    _assert_invariants(result)


def test_invariants_on_mixed_fixture(box_mesh: trimesh.Trimesh) -> None:
    # 1 transmit + 1 miss + 1 TIR in a single bundle, exercising every
    # branch of the count invariants.
    angle = np.deg2rad(60.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    rays = make_ray_bundle(
        origins=[
            [0.0, 0.0, 20.0],       # transmit (normal incidence)
            [100.0, 100.0, 20.0],   # miss (outside box footprint)
            [-1.0, 0.0, 6.0],       # TIR (60 deg with PET->AIR)
        ],
        directions=[
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [sx, 0.0, -sz],
        ],
    )
    result = run_single_interface_pipeline(box_mesh, rays, IOR_PET, IOR_AIR)
    assert result.ray_count == 3
    assert result.hit_count == 2
    assert result.miss_count == 1
    assert result.transmitted_count == 1
    assert result.tir_count == 1
    _assert_invariants(result)


def test_pipeline_result_artifacts_consistent(box_mesh: trimesh.Trimesh) -> None:
    # Pipeline should return the *same* RayBundle instance the caller
    # passed in, plus internally consistent intersection / propagation.
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=2,
        ny=2,
    )
    result = run_single_interface_pipeline(box_mesh, rays, IOR_AIR, IOR_PET)
    assert result.rays is rays
    assert isinstance(result.rays, RayBundle)
    assert result.intersection.ray_count == rays.ray_count
    _assert_invariants(result)
