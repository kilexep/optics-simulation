import numpy as np
import pytest
import trimesh

from optics_simulation.optics import (
    MultiStepTraceResult,
    OpticsError,
    RayBundle,
    SingleInterfacePipelineResult,
    make_ray_bundle,
    parallel_ray_grid,
    run_multi_step_trace,
)

IOR_AIR = 1.00028
IOR_PET = 1.575
SLAB_INTERFACES = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))


def _assert_step_count_invariants(step: SingleInterfacePipelineResult) -> None:
    assert step.hit_count + step.miss_count == step.ray_count
    assert step.transmitted_count == step.propagation.next_rays.ray_count
    assert step.transmitted_count + step.tir_count + step.miss_count == step.ray_count


def test_two_step_slab_normal_incidence(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle(
        origins=[[0.0, 0.0, 20.0]],
        directions=[[0.0, 0.0, -1.0]],
    )
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)

    assert isinstance(result, MultiStepTraceResult)
    assert result.step_count == 2
    assert len(result.steps) == 2
    assert result.termination_reason == "completed_interfaces"
    assert result.final_rays.ray_count == 1
    np.testing.assert_allclose(
        result.final_rays.directions[0], [0.0, 0.0, -1.0], atol=1e-12
    )


def test_oblique_two_step_slab_preserves_direction(box_mesh: trimesh.Trimesh) -> None:
    # Convention check: refraction.refract_direction does not swap
    # eta_i / eta_t under back-face normal flip. Therefore an oblique
    # ray that passes through a parallel-faced slab must exit parallel
    # to the incident direction (slab transmission preserves angle —
    # only a lateral shift occurs).
    angle = np.deg2rad(20.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    wi = np.array([sx, 0.0, -sz])
    rays = make_ray_bundle(origins=[[0.0, 0.0, 6.0]], directions=[wi.tolist()])

    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert result.step_count == 2
    assert result.termination_reason == "completed_interfaces"
    assert result.final_rays.ray_count == 1

    step1_dir = result.steps[0].propagation.next_rays.directions[0]
    step2_dir = result.steps[1].propagation.next_rays.directions[0]

    # Step 1, AIR -> PET: ray bends toward the surface normal (transverse
    # component shrinks, |z|-component grows).
    assert abs(step1_dir[0]) < abs(wi[0])
    assert abs(step1_dir[2]) > abs(wi[2])

    # Step 2, PET -> AIR: ray bends away from the normal (transverse
    # component grows back, |z|-component shrinks back).
    assert abs(step2_dir[0]) > abs(step1_dir[0])
    assert abs(step2_dir[2]) < abs(step1_dir[2])

    # Each refracted direction stays unit-norm.
    assert np.linalg.norm(step1_dir) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(step2_dir) == pytest.approx(1.0, abs=1e-12)

    # Slab parallel-face property: final exit direction equals initial
    # incident direction within numerical tolerance.
    np.testing.assert_allclose(result.final_rays.directions[0], wi, atol=1e-9)


def test_all_miss_early_termination(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, 1.0),  # pointing away from box
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        nx=2,
        ny=2,
    )
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert result.step_count == 1
    assert result.termination_reason == "no_active_rays"
    assert result.final_rays.ray_count == 0


def test_max_steps_zero_with_non_empty_sequence(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES, max_steps=0)
    assert result.steps == ()
    assert result.step_count == 0
    assert result.final_rays is rays
    assert result.termination_reason == "max_steps_reached"


def test_empty_interface_sequence_takes_priority_over_max_steps(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    for max_steps in (None, 0):
        result = run_multi_step_trace(
            box_mesh, rays, [], max_steps=max_steps
        )
        assert result.steps == ()
        assert result.step_count == 0
        assert result.final_rays is rays
        assert result.termination_reason == "no_interfaces"


def test_max_steps_greater_than_sequence_raises(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="exceeds interface_sequence length"):
        run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES, max_steps=3)


def test_negative_max_steps_raises(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="max_steps"):
        run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES, max_steps=-1)


def test_nonpositive_eta_raises_with_index(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])

    # First-pair fault.
    with pytest.raises(OpticsError, match=r"interface_sequence\[0\]"):
        run_multi_step_trace(box_mesh, rays, [(0.0, IOR_PET), (IOR_PET, IOR_AIR)])

    # Second-pair fault — confirms the index reporting.
    with pytest.raises(OpticsError, match=r"interface_sequence\[1\]"):
        run_multi_step_trace(box_mesh, rays, [(IOR_AIR, IOR_PET), (IOR_PET, -1.0)])


def test_negative_epsilon_raises(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="epsilon"):
        run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES, epsilon=-1.0)


def test_per_step_count_invariants(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert result.step_count == 2
    for step in result.steps:
        _assert_step_count_invariants(step)


def test_ray_handoff_invariant(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=2,
        ny=2,
    )
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    for i in range(len(result.steps) - 1):
        prev_next = result.steps[i].propagation.next_rays
        following = result.steps[i + 1].rays
        assert following is prev_next
        assert following.ray_count == prev_next.ray_count


def test_final_rays_invariant_with_and_without_steps(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])

    # Zero-step path: final_rays must be the exact same instance.
    zero = run_multi_step_trace(box_mesh, rays, [], max_steps=None)
    assert zero.step_count == 0
    assert zero.final_rays is rays

    # Non-zero-step path: final_rays must equal the last step's
    # propagation.next_rays by identity.
    nonzero = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert nonzero.step_count > 0
    assert nonzero.final_rays is nonzero.steps[-1].propagation.next_rays


def test_tir_termination_with_caller_provided_eta(box_mesh: trimesh.Trimesh) -> None:
    # Mathematical fixture, not geometric medium tracking: the caller
    # asserts (PET, AIR) at the first interface and (AIR, PET) at the
    # second. The first step encounters 60-degree incidence on the box
    # top face and total-internally-reflects under the supplied
    # (PET, AIR) labelling, so the loop terminates before the second
    # interface is reached. This exercises the caller-authoritative
    # convention (no automatic medium tracking).
    angle = np.deg2rad(60.0)
    sx, sz = float(np.sin(angle)), float(np.cos(angle))
    rays = make_ray_bundle(
        origins=[[0.0, 0.0, 6.0], [-1.0, 0.0, 6.0]],
        directions=[[sx, 0.0, -sz], [sx, 0.0, -sz]],
    )
    result = run_multi_step_trace(
        box_mesh, rays, [(IOR_PET, IOR_AIR), (IOR_AIR, IOR_PET)]
    )
    assert result.step_count == 1
    assert result.termination_reason == "no_active_rays"
    assert result.final_rays.ray_count == 0
    assert result.steps[0].tir_count == 2
    assert result.steps[0].transmitted_count == 0


def test_empty_input_bundle(box_mesh: trimesh.Trimesh) -> None:
    rays = make_ray_bundle(np.zeros((0, 3)), np.zeros((0, 3)))
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert result.step_count == 1
    assert result.termination_reason == "no_active_rays"
    assert result.final_rays.ray_count == 0


def test_max_steps_equal_to_sequence_length_is_allowed(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = run_multi_step_trace(
        box_mesh, rays, SLAB_INTERFACES, max_steps=len(SLAB_INTERFACES)
    )
    assert result.step_count == 2
    assert result.termination_reason in {"completed_interfaces", "no_active_rays"}


def test_max_steps_below_sequence_length_terminates_with_max_steps_reached(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES, max_steps=1)
    assert result.step_count == 1
    assert result.termination_reason == "max_steps_reached"


def test_invalid_mesh_type_raises() -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="trimesh.Trimesh"):
        run_multi_step_trace("not a mesh", rays, SLAB_INTERFACES)  # type: ignore[arg-type]


def test_invalid_rays_type_raises(box_mesh: trimesh.Trimesh) -> None:
    with pytest.raises(OpticsError, match="RayBundle"):
        run_multi_step_trace(box_mesh, object(), SLAB_INTERFACES)  # type: ignore[arg-type]
