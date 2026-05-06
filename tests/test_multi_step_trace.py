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


def _grid_rays_over_box() -> RayBundle:
    return parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )


def test_zero_step_lineage_is_arange(box_mesh: trimesh.Trimesh) -> None:
    rays = _grid_rays_over_box()
    result = run_multi_step_trace(box_mesh, rays, [], max_steps=None)
    assert result.step_count == 0
    assert result.termination_reason == "no_interfaces"
    expected = np.arange(rays.ray_count, dtype=np.int64)
    assert result.final_source_ray_indices.dtype == np.int64
    assert np.array_equal(result.final_source_ray_indices, expected)


def test_max_steps_zero_lineage_is_arange(box_mesh: trimesh.Trimesh) -> None:
    rays = _grid_rays_over_box()
    result = run_multi_step_trace(
        box_mesh, rays, SLAB_INTERFACES, max_steps=0
    )
    assert result.step_count == 0
    assert result.termination_reason == "max_steps_reached"
    expected = np.arange(rays.ray_count, dtype=np.int64)
    assert result.final_source_ray_indices.dtype == np.int64
    assert np.array_equal(result.final_source_ray_indices, expected)


def test_no_active_rays_lineage_is_empty(box_mesh: trimesh.Trimesh) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, 1.0),  # away from the box
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        nx=2,
        ny=2,
    )
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert result.termination_reason == "no_active_rays"
    assert result.final_rays.ray_count == 0
    assert result.final_source_ray_indices.shape == (0,)
    assert result.final_source_ray_indices.dtype == np.int64


def test_lineage_length_and_dtype_match_final_rays(box_mesh: trimesh.Trimesh) -> None:
    rays = _grid_rays_over_box()
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert result.final_source_ray_indices.shape == (result.final_rays.ray_count,)
    assert result.final_source_ray_indices.dtype == np.int64


def test_lineage_values_are_valid_initial_indices(box_mesh: trimesh.Trimesh) -> None:
    rays = _grid_rays_over_box()
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    if result.final_source_ray_indices.size > 0:
        assert int(result.final_source_ray_indices.min()) >= 0
        assert int(result.final_source_ray_indices.max()) < rays.ray_count


def test_lineage_unique_when_no_splitting(box_mesh: trimesh.Trimesh) -> None:
    rays = _grid_rays_over_box()
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    indices = result.final_source_ray_indices.tolist()
    assert len(indices) == len(set(indices))


def test_lineage_matches_manual_chain(box_mesh: trimesh.Trimesh) -> None:
    rays = _grid_rays_over_box()
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    expected = np.arange(rays.ray_count, dtype=np.int64)
    for step in result.steps:
        expected = expected[step.propagation.source_ray_indices]
    assert np.array_equal(result.final_source_ray_indices, expected)


def test_partial_miss_lineage_identifies_surviving_rays(
    box_mesh: trimesh.Trimesh,
) -> None:
    # Box footprint is x,y in [-5, 5]. A 5x5 grid spanning [-10, 10]
    # in both axes has 9 rays whose origin (x, y) falls outside the
    # box footprint and miss; the central 16 rays land on the top
    # face. (Corners of the grid are at +-10, well outside the box.)
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-10.0, 10.0),
        y_range=(-10.0, 10.0),
        nx=5,
        ny=5,
    )
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    half_w = 5.0
    half_h = 5.0
    surviving = result.final_source_ray_indices
    assert surviving.size > 0
    assert surviving.size < rays.ray_count
    surviving_origins = rays.origins[surviving]
    assert np.all(np.abs(surviving_origins[:, 0]) <= half_w + 1e-9)
    assert np.all(np.abs(surviving_origins[:, 1]) <= half_h + 1e-9)


def test_detector_hotspot_mapping_via_final_source_ray_indices(
    box_mesh: trimesh.Trimesh,
) -> None:
    from optics_simulation.metrics import select_hotspot_rays
    from optics_simulation.optics import (
        accumulate_detector_hits,
        create_detector_grid,
        create_detector_plane,
        intersect_detector_plane,
    )

    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=4,
        ny=4,
    )
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)

    detector = create_detector_plane(
        center=(0.0, 0.0, -15.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )
    detector_hits = intersect_detector_plane(result.final_rays, detector)
    grid = create_detector_grid(width=10.0, height=10.0, resolution=(8, 8))
    accum = accumulate_detector_hits(detector_hits, grid)

    selection = select_hotspot_rays(detector_hits, accum, top_percent=10.0)

    mapped_initial = result.final_source_ray_indices[
        selection.selected_ray_indices
    ]
    assert mapped_initial.dtype == np.int64
    if mapped_initial.size > 0:
        assert int(mapped_initial.min()) >= 0
        assert int(mapped_initial.max()) < rays.ray_count
    # uniqueness across selected rays follows from final lineage
    # uniqueness for non-splitting traces.
    assert len(set(mapped_initial.tolist())) == mapped_initial.size


# ---------------------------------------------------------------------------
# Cumulative Fresnel transmission weighting (final_ray_weights)
# ---------------------------------------------------------------------------


def test_zero_step_trace_has_ones_final_ray_weights(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = _grid_rays_over_box()  # 9 rays
    result = run_multi_step_trace(box_mesh, rays, [], max_steps=None)
    assert result.termination_reason == "no_interfaces"
    assert result.final_ray_weights.shape == (rays.ray_count,)
    assert result.final_ray_weights.dtype == float
    np.testing.assert_array_equal(
        result.final_ray_weights, np.ones(rays.ray_count, dtype=float)
    )


def test_max_steps_zero_has_ones_final_ray_weights(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = _grid_rays_over_box()
    result = run_multi_step_trace(
        box_mesh, rays, SLAB_INTERFACES, max_steps=0
    )
    assert result.termination_reason == "max_steps_reached"
    assert result.final_ray_weights.shape == (rays.ray_count,)
    assert result.final_ray_weights.dtype == float
    np.testing.assert_array_equal(
        result.final_ray_weights, np.ones(rays.ray_count, dtype=float)
    )


def test_no_active_rays_gives_empty_final_ray_weights(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, 1.0),  # pointing away from box: all miss
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        nx=2,
        ny=2,
    )
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert result.termination_reason == "no_active_rays"
    assert result.final_ray_weights.shape == (0,)
    assert result.final_ray_weights.dtype == float


def test_two_step_slab_final_ray_weights_length_matches_final_rays(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert result.final_ray_weights.shape == (result.final_rays.ray_count,)
    assert result.final_ray_weights.shape == (1,)


def test_final_ray_weights_finite_and_nonneg(
    box_mesh: trimesh.Trimesh,
) -> None:
    angles = (0.0, 5.0, 15.0, 25.0)
    for deg in angles:
        theta = np.deg2rad(deg)
        wi = [float(np.sin(theta)), 0.0, -float(np.cos(theta))]
        rays = parallel_ray_grid(
            origin_plane_z=20.0,
            direction=tuple(wi),
            x_range=(-2.0, 2.0),
            y_range=(-2.0, 2.0),
            nx=3,
            ny=3,
        )
        result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
        w = result.final_ray_weights
        assert np.isfinite(w).all()
        assert (w >= 0.0).all()


def test_final_ray_weights_bounded_by_one_in_passive_slab(
    box_mesh: trimesh.Trimesh,
) -> None:
    angles = (0.0, 10.0, 20.0, 30.0)
    for deg in angles:
        theta = np.deg2rad(deg)
        wi = [float(np.sin(theta)), 0.0, -float(np.cos(theta))]
        rays = parallel_ray_grid(
            origin_plane_z=20.0,
            direction=tuple(wi),
            x_range=(-2.0, 2.0),
            y_range=(-2.0, 2.0),
            nx=3,
            ny=3,
        )
        result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
        w = result.final_ray_weights
        if w.size > 0:
            assert float(w.max()) <= 1.0 + 1e-12
            # Slab transmission at non-grazing angles is also strictly
            # below 1 (Fresnel R > 0). Sanity-check normal incidence:
            # cumulative T = (1 - R_air_pet) * (1 - R_pet_air).
            assert float(w.min()) > 0.0


def test_manual_chain_matches_final_ray_weights(
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
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)

    expected = np.ones(rays.ray_count, dtype=float)
    for step in result.steps:
        src = step.propagation.source_ray_indices
        expected = expected[src] * step.propagation.transmittance[src]

    assert expected.shape == result.final_ray_weights.shape
    np.testing.assert_allclose(
        result.final_ray_weights, expected, atol=1e-12
    )


def test_final_ray_weights_aligned_with_final_source_ray_indices(
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
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert (
        result.final_ray_weights.shape
        == result.final_source_ray_indices.shape
    )
    assert result.final_ray_weights.shape == (result.final_rays.ray_count,)
    assert result.final_source_ray_indices.dtype == np.int64
    assert result.final_ray_weights.dtype == float


def test_final_ray_weights_does_not_alias_internal_state(
    box_mesh: trimesh.Trimesh,
) -> None:
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    result = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    snapshot = result.final_ray_weights.copy()
    # Mutating the returned array must not affect another call's output.
    result.final_ray_weights[:] = 0.0
    result2 = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    np.testing.assert_array_equal(result2.final_ray_weights, snapshot)


def test_mesh_not_mutated_by_weighted_trace(
    box_mesh: trimesh.Trimesh,
) -> None:
    vertices_before = np.array(box_mesh.vertices, copy=True)
    faces_before = np.array(box_mesh.faces, copy=True)
    rays = make_ray_bundle([[0.0, 0.0, 20.0]], [[0.0, 0.0, -1.0]])
    run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    assert np.array_equal(np.asarray(box_mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(box_mesh.faces), faces_before)
