import numpy as np
import pytest
import trimesh

from optics_simulation.optics import (
    MultiMeshTraceResult,
    MultiMeshTraceStepSpec,
    OpticsError,
    make_ray_bundle,
    parallel_ray_grid,
    run_multi_mesh_trace,
    run_multi_step_trace,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
SLAB_INTERFACES = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))


def _slab_step_specs(
    mesh: trimesh.Trimesh,
) -> list[MultiMeshTraceStepSpec]:
    return [
        MultiMeshTraceStepSpec(
            mesh=mesh, eta_i=IOR_AIR, eta_t=IOR_PET, label="enter",
        ),
        MultiMeshTraceStepSpec(
            mesh=mesh, eta_i=IOR_PET, eta_t=IOR_AIR, label="exit",
        ),
    ]


def test_empty_step_specs_returns_no_interfaces(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = run_multi_mesh_trace(rays, [])
    assert isinstance(result, MultiMeshTraceResult)
    assert result.step_count == 0
    assert result.termination_reason == "no_interfaces"
    assert result.final_rays is rays
    expected = np.arange(rays.ray_count, dtype=np.int64)
    assert np.array_equal(result.final_source_ray_indices, expected)
    np.testing.assert_array_equal(
        result.final_ray_weights, np.ones(rays.ray_count, dtype=float),
    )


def test_max_steps_zero_returns_max_steps_reached(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = run_multi_mesh_trace(
        rays, _slab_step_specs(box_mesh), max_steps=0,
    )
    assert result.step_count == 0
    assert result.termination_reason == "max_steps_reached"
    assert result.final_rays is rays
    np.testing.assert_array_equal(
        result.final_ray_weights, np.ones(rays.ray_count, dtype=float),
    )


def test_two_step_same_mesh_matches_run_multi_step_trace(
    box_mesh: trimesh.Trimesh,
) -> None:
    angle = np.deg2rad(20.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(sx, 0.0, -sz),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    canonical = run_multi_step_trace(
        box_mesh, rays, SLAB_INTERFACES,
    )
    multi = run_multi_mesh_trace(rays, _slab_step_specs(box_mesh))

    assert multi.step_count == canonical.step_count
    assert multi.termination_reason == canonical.termination_reason
    assert multi.final_rays.ray_count == canonical.final_rays.ray_count
    np.testing.assert_allclose(
        multi.final_rays.origins,
        canonical.final_rays.origins,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        multi.final_rays.directions,
        canonical.final_rays.directions,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        multi.final_source_ray_indices,
        canonical.final_source_ray_indices,
    )
    np.testing.assert_allclose(
        multi.final_ray_weights,
        canonical.final_ray_weights,
        atol=1e-12,
    )


def test_final_source_ray_indices_length_matches_final_rays(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    result = run_multi_mesh_trace(rays, _slab_step_specs(box_mesh))
    assert result.final_source_ray_indices.shape == (
        result.final_rays.ray_count,
    )
    assert result.final_source_ray_indices.dtype == np.int64


def test_final_ray_weights_length_matches_final_rays(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    result = run_multi_mesh_trace(rays, _slab_step_specs(box_mesh))
    assert result.final_ray_weights.shape == (
        result.final_rays.ray_count,
    )
    assert result.final_ray_weights.dtype == float


def test_manual_weight_chain_matches_final_ray_weights(
    box_mesh: trimesh.Trimesh,
) -> None:
    angle = np.deg2rad(20.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(sx, 0.0, -sz),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    result = run_multi_mesh_trace(rays, _slab_step_specs(box_mesh))
    expected = np.ones(rays.ray_count, dtype=float)
    for step in result.steps:
        src = step.propagation.source_ray_indices
        expected = expected[src] * step.propagation.transmittance[src]
    np.testing.assert_allclose(
        result.final_ray_weights, expected, atol=1e-12,
    )


def test_no_active_rays_returns_empty_weights_and_indices(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, 1.0),  # away from box
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        nx=2,
        ny=2,
    )
    result = run_multi_mesh_trace(rays, _slab_step_specs(box_mesh))
    assert result.termination_reason == "no_active_rays"
    assert result.final_rays.ray_count == 0
    assert result.final_source_ray_indices.shape == (0,)
    assert result.final_ray_weights.shape == (0,)


def test_invalid_spec_raises_optics_error(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="MultiMeshTraceStepSpec"):
        run_multi_mesh_trace(rays, ["not a spec"])  # type: ignore[list-item]
    with pytest.raises(OpticsError, match="trimesh.Trimesh"):
        run_multi_mesh_trace(
            rays,
            [
                MultiMeshTraceStepSpec(
                    mesh="not a mesh",  # type: ignore[arg-type]
                    eta_i=IOR_AIR,
                    eta_t=IOR_PET,
                ),
            ],
        )
    with pytest.raises(OpticsError, match="non-positive"):
        run_multi_mesh_trace(
            rays,
            [
                MultiMeshTraceStepSpec(
                    mesh=box_mesh, eta_i=0.0, eta_t=IOR_PET,
                ),
            ],
        )
    with pytest.raises(OpticsError, match="non-finite"):
        run_multi_mesh_trace(
            rays,
            [
                MultiMeshTraceStepSpec(
                    mesh=box_mesh,
                    eta_i=float("nan"),
                    eta_t=IOR_PET,
                ),
            ],
        )


def test_input_rays_not_mutated(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    origins_before = np.array(rays.origins, copy=True)
    directions_before = np.array(rays.directions, copy=True)
    run_multi_mesh_trace(rays, _slab_step_specs(box_mesh))
    assert np.array_equal(np.asarray(rays.origins), origins_before)
    assert np.array_equal(
        np.asarray(rays.directions), directions_before,
    )


def test_meshes_not_mutated(box_mesh: trimesh.Trimesh) -> None:
    vertices_before = np.array(box_mesh.vertices, copy=True)
    faces_before = np.array(box_mesh.faces, copy=True)
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    run_multi_mesh_trace(rays, _slab_step_specs(box_mesh))
    assert np.array_equal(np.asarray(box_mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(box_mesh.faces), faces_before)


def test_invalid_rays_type_raises(box_mesh: trimesh.Trimesh) -> None:
    with pytest.raises(OpticsError, match="RayBundle"):
        run_multi_mesh_trace(
            object(), _slab_step_specs(box_mesh),  # type: ignore[arg-type]
        )


def test_max_steps_greater_than_specs_raises(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="exceeds step_specs length"):
        run_multi_mesh_trace(
            rays, _slab_step_specs(box_mesh), max_steps=5,
        )
