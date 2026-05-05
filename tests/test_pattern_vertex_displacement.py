import numpy as np
import pytest
import trimesh

from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry import (
    create_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.geometry.surface_coordinates import SurfaceCoordinateMap
from optics_simulation.pattern import (
    GaussianDimple,
    GaussianDimplePattern,
    PatternError,
    VertexPatternDisplacement,
    compute_vertex_displacement_amounts,
    create_gaussian_dimple_pattern,
    evaluate_gaussian_pattern_at_points,
    gaussian_pattern_from_descriptor,
    gaussian_pattern_to_descriptor,
)


def _make_risk_map(prob_2d: np.ndarray) -> RiskMap:
    prob = np.asarray(prob_2d, dtype=float)
    total = float(prob.sum())
    normalized = prob / total if total > 0.0 else np.zeros_like(prob)
    return RiskMap(
        risk_map=prob.copy(),
        probability_map=normalized,
        active_mask=prob > 0.0,
        total_risk=total,
        active_count=int((prob > 0.0).sum()),
        epsilon=0.0,
        threshold=None,
    )


def _make_one_dimple_pattern(
    *,
    center_u: float = 0.5,
    center_v: float = 0.5,
    sigma_u: float = 0.05,
    sigma_v: float = 0.05,
    amplitude: float = 1.0,
    max_depth: float = 0.25,
    resolution: tuple[int, int] = (16, 16),
) -> GaussianDimplePattern:
    dimple = GaussianDimple(
        center_u=center_u, center_v=center_v,
        amplitude=amplitude, sigma_u=sigma_u, sigma_v=sigma_v,
    )
    return GaussianDimplePattern(
        dimples=(dimple,),
        depth_field=np.zeros(resolution, dtype=float),
        resolution=resolution,
        max_depth=max_depth,
        seed=0,
    )


def _empty_pattern(resolution: tuple[int, int] = (8, 8)) -> GaussianDimplePattern:
    return GaussianDimplePattern(
        dimples=(),
        depth_field=np.zeros(resolution, dtype=float),
        resolution=resolution,
        max_depth=0.25,
        seed=None,
    )


def test_evaluate_returns_shape_n() -> None:
    pattern = _make_one_dimple_pattern()
    u = np.array([0.1, 0.5, 0.9])
    v = np.array([0.2, 0.5, 0.8])
    field = evaluate_gaussian_pattern_at_points(pattern, u, v)
    assert field.shape == (3,)


def test_one_dimple_at_same_uv_gives_high_depth() -> None:
    pattern = _make_one_dimple_pattern(
        center_u=0.5, center_v=0.5, sigma_u=0.05, sigma_v=0.05,
        amplitude=1.0,
    )
    u = np.array([0.5])
    v = np.array([0.5])
    field = evaluate_gaussian_pattern_at_points(pattern, u, v)
    assert float(field[0]) == pytest.approx(1.0, abs=1e-12)


def test_far_point_gives_lower_depth_than_center() -> None:
    pattern = _make_one_dimple_pattern(
        center_u=0.5, center_v=0.5, sigma_u=0.05, sigma_v=0.05,
    )
    u = np.array([0.5, 0.2])
    v = np.array([0.5, 0.5])
    field = evaluate_gaussian_pattern_at_points(pattern, u, v)
    assert float(field[1]) < float(field[0])


def test_circular_u_distance_works_near_wrap_boundary() -> None:
    pattern = _make_one_dimple_pattern(
        center_u=0.99, center_v=0.5, sigma_u=0.05, sigma_v=0.05,
    )
    u = np.array([0.01, 0.5])
    v = np.array([0.5, 0.5])
    field = evaluate_gaussian_pattern_at_points(pattern, u, v, clip=False)
    assert float(field[0]) > float(field[1])
    assert float(field[0]) > 0.5


def test_empty_dimples_produce_all_zero_depth() -> None:
    pattern = _empty_pattern()
    u = np.array([0.1, 0.5, 0.9])
    v = np.array([0.2, 0.5, 0.8])
    field = evaluate_gaussian_pattern_at_points(pattern, u, v)
    assert field.shape == (3,)
    assert float(np.abs(field).max()) == 0.0


def test_clip_true_bounds_field_to_unit_interval() -> None:
    big_dimples = tuple(
        GaussianDimple(0.5, 0.5, 1.0, 0.5, 0.5) for _ in range(3)
    )
    pattern = GaussianDimplePattern(
        dimples=big_dimples,
        depth_field=np.zeros((8, 8), dtype=float),
        resolution=(8, 8),
        max_depth=0.25,
        seed=0,
    )
    u = np.array([0.5])
    v = np.array([0.5])
    field = evaluate_gaussian_pattern_at_points(pattern, u, v, clip=True)
    assert float(field[0]) <= 1.0 + 1e-12


def test_clip_false_can_exceed_one_for_overlapping_dimples() -> None:
    big_dimples = tuple(
        GaussianDimple(0.5, 0.5, 1.0, 0.5, 0.5) for _ in range(3)
    )
    pattern = GaussianDimplePattern(
        dimples=big_dimples,
        depth_field=np.zeros((8, 8), dtype=float),
        resolution=(8, 8),
        max_depth=0.25,
        seed=0,
    )
    u = np.array([0.5])
    v = np.array([0.5])
    field = evaluate_gaussian_pattern_at_points(pattern, u, v, clip=False)
    assert float(field[0]) > 1.0


def test_invalid_u_v_shape_raises() -> None:
    pattern = _make_one_dimple_pattern()
    with pytest.raises(PatternError, match="same shape"):
        evaluate_gaussian_pattern_at_points(
            pattern, np.array([0.1, 0.5]), np.array([0.2, 0.5, 0.8])
        )
    with pytest.raises(PatternError, match="1D"):
        evaluate_gaussian_pattern_at_points(
            pattern, np.array([[0.1]]), np.array([[0.2]])
        )


def test_nan_inf_uv_raises() -> None:
    pattern = _make_one_dimple_pattern()
    with pytest.raises(PatternError, match="u contains"):
        evaluate_gaussian_pattern_at_points(
            pattern, np.array([np.nan]), np.array([0.5])
        )
    with pytest.raises(PatternError, match="v contains"):
        evaluate_gaussian_pattern_at_points(
            pattern, np.array([0.5]), np.array([np.inf])
        )


def test_out_of_range_uv_raises() -> None:
    pattern = _make_one_dimple_pattern()
    with pytest.raises(PatternError, match="u must be"):
        evaluate_gaussian_pattern_at_points(
            pattern, np.array([1.5]), np.array([0.5])
        )
    with pytest.raises(PatternError, match="u must be"):
        evaluate_gaussian_pattern_at_points(
            pattern, np.array([-0.1]), np.array([0.5])
        )
    with pytest.raises(PatternError, match="v must be"):
        evaluate_gaussian_pattern_at_points(
            pattern, np.array([0.5]), np.array([1.5])
        )
    with pytest.raises(PatternError, match="v must be"):
        evaluate_gaussian_pattern_at_points(
            pattern, np.array([0.5]), np.array([-0.1])
        )


def test_invalid_pattern_type_raises() -> None:
    with pytest.raises(PatternError, match="GaussianDimplePattern"):
        evaluate_gaussian_pattern_at_points(
            object(), np.array([0.5]), np.array([0.5])
        )


def _make_synthetic_setup() -> tuple[
    trimesh.Trimesh, SurfaceCoordinateMap, GaussianDimplePattern
]:
    mesh = create_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=48
    )
    surface_map = create_vertex_surface_coordinates(mesh)
    rm = _make_risk_map(np.ones((8, 8)))
    pattern = create_gaussian_dimple_pattern(
        rm, count=10, amplitude=1.0, sigma_u=0.05, sigma_v=0.05,
        max_depth=0.25, seed=42,
    )
    return mesh, surface_map, pattern


def test_compute_returns_vertex_pattern_displacement() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    result = compute_vertex_displacement_amounts(mesh, sm, pattern)
    assert isinstance(result, VertexPatternDisplacement)
    assert result.coordinate_system == "normalized_cylindrical_uv"
    assert result.application_mode == "amount_only"


def test_normalized_and_physical_depth_shapes_match_vertex_count() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    n = len(mesh.vertices)
    result = compute_vertex_displacement_amounts(mesh, sm, pattern)
    assert result.vertex_count == n
    assert result.normalized_depth.shape == (n,)
    assert result.physical_depth.shape == (n,)
    assert result.active_mask.shape == (n,)


def test_normalized_depth_in_unit_interval() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    result = compute_vertex_displacement_amounts(mesh, sm, pattern)
    assert float(result.normalized_depth.min()) >= 0.0
    assert float(result.normalized_depth.max()) <= 1.0 + 1e-12


def test_physical_depth_equals_normalized_times_max_depth() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    result = compute_vertex_displacement_amounts(mesh, sm, pattern)
    assert np.allclose(
        result.physical_depth,
        result.normalized_depth * pattern.max_depth,
        atol=1e-15,
    )
    assert result.max_depth == pytest.approx(pattern.max_depth)


def test_active_mask_follows_active_threshold() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    res_low = compute_vertex_displacement_amounts(
        mesh, sm, pattern, active_threshold=0.0
    )
    res_high = compute_vertex_displacement_amounts(
        mesh, sm, pattern, active_threshold=0.5
    )
    assert int(res_high.active_mask.sum()) <= int(res_low.active_mask.sum())
    expected_high = res_high.normalized_depth > 0.5
    assert np.array_equal(res_high.active_mask, expected_high)


def test_exclude_v_boundary_epsilon_zeros_boundary_depth() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    eps = 0.05
    result = compute_vertex_displacement_amounts(
        mesh, sm, pattern, exclude_v_boundary_epsilon=eps
    )
    v = np.asarray(sm.v, dtype=float)
    boundary = (v <= eps) | (v >= 1.0 - eps)
    assert boundary.any(), "fixture should contain boundary vertices"
    assert float(np.abs(result.normalized_depth[boundary]).max()) == 0.0
    assert float(np.abs(result.physical_depth[boundary]).max()) == 0.0
    assert not bool(result.active_mask[boundary].any())


def test_invalid_active_threshold_raises() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    with pytest.raises(PatternError, match="active_threshold"):
        compute_vertex_displacement_amounts(
            mesh, sm, pattern, active_threshold=-0.1
        )


def test_invalid_v_boundary_epsilon_raises() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    for bad in (-0.1, 0.5, 0.7):
        with pytest.raises(PatternError, match="exclude_v_boundary_epsilon"):
            compute_vertex_displacement_amounts(
                mesh, sm, pattern, exclude_v_boundary_epsilon=bad
            )


def test_surface_map_point_count_mismatch_raises() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    bad_sm = SurfaceCoordinateMap(
        u=np.zeros(3, dtype=float),
        v=np.zeros(3, dtype=float),
        point_count=3,
        z_min=sm.z_min,
        z_max=sm.z_max,
        center_xy=sm.center_xy,
    )
    with pytest.raises(PatternError, match="point_count"):
        compute_vertex_displacement_amounts(mesh, bad_sm, pattern)


def test_empty_mesh_raises() -> None:
    empty_mesh = trimesh.Trimesh(
        vertices=np.zeros((0, 3), dtype=float),
        faces=np.zeros((0, 3), dtype=np.int64),
        process=False,
    )
    sm = SurfaceCoordinateMap(
        u=np.zeros(0, dtype=float),
        v=np.zeros(0, dtype=float),
        point_count=0,
        z_min=0.0,
        z_max=1.0,
        center_xy=(0.0, 0.0),
    )
    pattern = _empty_pattern()
    with pytest.raises(PatternError, match="no vertices"):
        compute_vertex_displacement_amounts(empty_mesh, sm, pattern)


def test_invalid_input_types_raise() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    with pytest.raises(PatternError, match="trimesh.Trimesh"):
        compute_vertex_displacement_amounts(object(), sm, pattern)
    with pytest.raises(PatternError, match="SurfaceCoordinateMap"):
        compute_vertex_displacement_amounts(mesh, object(), pattern)
    with pytest.raises(PatternError, match="GaussianDimplePattern"):
        compute_vertex_displacement_amounts(mesh, sm, object())


def test_mesh_vertices_and_faces_not_mutated() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    compute_vertex_displacement_amounts(
        mesh, sm, pattern,
        active_threshold=0.1, exclude_v_boundary_epsilon=0.05,
    )
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_synthetic_bottle_integration_smoke() -> None:
    mesh = create_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=48
    )
    sm = create_vertex_surface_coordinates(mesh)
    rm = _make_risk_map(np.ones((8, 8)))
    pattern = create_gaussian_dimple_pattern(
        rm, count=5, amplitude=1.0, sigma_u=0.05, sigma_v=0.05,
        max_depth=0.25, seed=7,
    )
    result = compute_vertex_displacement_amounts(mesh, sm, pattern)
    assert result.vertex_count == len(mesh.vertices)
    assert np.isfinite(result.normalized_depth).all()
    assert np.isfinite(result.physical_depth).all()


def test_descriptor_round_trip_pattern_works_in_compute() -> None:
    mesh, sm, pattern = _make_synthetic_setup()
    descriptor = gaussian_pattern_to_descriptor(pattern)
    restored = gaussian_pattern_from_descriptor(descriptor)
    result_orig = compute_vertex_displacement_amounts(mesh, sm, pattern)
    result_round = compute_vertex_displacement_amounts(mesh, sm, restored)
    assert np.allclose(
        result_round.normalized_depth, result_orig.normalized_depth,
        atol=1e-12,
    )
    assert np.allclose(
        result_round.physical_depth, result_orig.physical_depth,
        atol=1e-12,
    )
    assert np.array_equal(result_round.active_mask, result_orig.active_mask)
