import numpy as np
import pytest
import trimesh

from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry import (
    create_mesh_quality_report,
    create_subdivided_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.pattern import (
    DisplacedMeshResult,
    PatternError,
    VertexPatternDisplacement,
    compute_vertex_displacement_amounts,
    create_displaced_mesh_copy,
    create_gaussian_dimple_pattern,
)


def _uniform_risk_map(shape: tuple[int, int] = (8, 16)) -> RiskMap:
    risk = np.ones(shape, dtype=float)
    total = float(risk.sum())
    return RiskMap(
        risk_map=risk,
        probability_map=risk / total,
        active_mask=np.ones(shape, dtype=bool),
        total_risk=total,
        active_count=int(risk.size),
        epsilon=0.0,
        threshold=None,
    )


def _build_synthetic_setup() -> tuple[trimesh.Trimesh, VertexPatternDisplacement]:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=48, height_segments=12,
    )
    surface_map = create_vertex_surface_coordinates(mesh)
    risk = _uniform_risk_map()
    pattern = create_gaussian_dimple_pattern(
        risk, count=10, amplitude=1.0,
        sigma_u=0.05, sigma_v=0.05,
        max_depth=0.25, seed=42,
    )
    displacement = compute_vertex_displacement_amounts(
        mesh, surface_map, pattern,
        active_threshold=0.01,
        exclude_v_boundary_epsilon=0.02,
    )
    return mesh, displacement


def _make_displacement(
    n: int,
    *,
    physical_depth: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    vertex_count: int | None = None,
    max_depth: float = 0.25,
) -> VertexPatternDisplacement:
    pd = (
        np.full(n, 0.5 * max_depth, dtype=float)
        if physical_depth is None else np.asarray(physical_depth, dtype=float)
    )
    am = (
        np.ones(n, dtype=bool) if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    vc = n if vertex_count is None else int(vertex_count)
    return VertexPatternDisplacement(
        normalized_depth=np.full(n, 0.5, dtype=float),
        physical_depth=pd,
        active_mask=am,
        vertex_count=vc,
        max_depth=max_depth,
    )


def test_returns_displaced_mesh_result() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement)
    assert isinstance(result, DisplacedMeshResult)
    assert result.mode == "inward"
    assert result.original_vertex_count == int(len(mesh.vertices))


def test_displaced_mesh_is_new_trimesh_instance() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement)
    assert isinstance(result.displaced_mesh, trimesh.Trimesh)
    assert result.displaced_mesh is not mesh


def test_original_mesh_vertices_and_faces_unchanged() -> None:
    mesh, displacement = _build_synthetic_setup()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    create_displaced_mesh_copy(mesh, displacement, mode="inward")
    create_displaced_mesh_copy(mesh, displacement, mode="outward")
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_displaced_mesh_faces_are_preserved() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement)
    assert np.array_equal(
        np.asarray(result.displaced_mesh.faces),
        np.asarray(mesh.faces),
    )


def test_displacement_vectors_shape() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement)
    n = int(len(mesh.vertices))
    assert result.displacement_vectors.shape == (n, 3)


def test_displacement_magnitudes_shape() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement)
    n = int(len(mesh.vertices))
    assert result.displacement_magnitudes.shape == (n,)


def test_moved_mask_shape() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement)
    n = int(len(mesh.vertices))
    assert result.moved_mask.shape == (n,)


def test_inactive_vertices_have_zero_displacement_vectors() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement)
    inactive = ~np.asarray(displacement.active_mask, dtype=bool)
    assert bool(inactive.any()), (
        "fixture should contain at least one inactive vertex"
    )
    assert float(np.abs(result.displacement_vectors[inactive]).max()) == 0.0
    assert float(np.abs(result.displacement_magnitudes[inactive]).max()) == 0.0
    assert not bool(result.moved_mask[inactive].any())


def test_active_vertices_with_positive_depth_have_nonzero_displacement() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement)
    active = np.asarray(displacement.active_mask, dtype=bool)
    pd = np.asarray(displacement.physical_depth, dtype=float)
    interesting = active & (pd > 0.0)
    assert bool(interesting.any()), (
        "fixture should contain at least one active vertex with depth > 0"
    )
    assert (result.displacement_magnitudes[interesting] > 0.0).all()
    assert bool(result.moved_mask[interesting].all())


def test_inward_mode_moves_vertices_opposite_normals() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement, mode="inward")
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    moved = np.asarray(result.moved_mask, dtype=bool)
    assert bool(moved.any())
    dotp = (result.displacement_vectors[moved] * normals[moved]).sum(axis=1)
    assert (dotp < 0.0).all()


def test_outward_mode_moves_vertices_along_normals() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement, mode="outward")
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    moved = np.asarray(result.moved_mask, dtype=bool)
    assert bool(moved.any())
    dotp = (result.displacement_vectors[moved] * normals[moved]).sum(axis=1)
    assert (dotp > 0.0).all()


def test_invalid_mode_raises() -> None:
    mesh, displacement = _build_synthetic_setup()
    with pytest.raises(PatternError, match="mode must be"):
        create_displaced_mesh_copy(mesh, displacement, mode="diagonal")


def test_mismatched_vertex_count_raises() -> None:
    mesh, _ = _build_synthetic_setup()
    n = int(len(mesh.vertices))
    bad = _make_displacement(n + 1)
    with pytest.raises(PatternError, match="vertex_count"):
        create_displaced_mesh_copy(mesh, bad)


def test_negative_physical_depth_raises() -> None:
    mesh, _ = _build_synthetic_setup()
    n = int(len(mesh.vertices))
    pd = np.full(n, 0.05, dtype=float)
    pd[0] = -0.01
    bad = _make_displacement(n, physical_depth=pd)
    with pytest.raises(PatternError, match="non-negative"):
        create_displaced_mesh_copy(mesh, bad)


def test_non_finite_physical_depth_raises() -> None:
    mesh, _ = _build_synthetic_setup()
    n = int(len(mesh.vertices))
    pd_nan = np.full(n, 0.05, dtype=float)
    pd_nan[0] = np.nan
    with pytest.raises(PatternError, match="NaN or inf"):
        create_displaced_mesh_copy(
            mesh, _make_displacement(n, physical_depth=pd_nan)
        )

    pd_inf = np.full(n, 0.05, dtype=float)
    pd_inf[1] = np.inf
    with pytest.raises(PatternError, match="NaN or inf"):
        create_displaced_mesh_copy(
            mesh, _make_displacement(n, physical_depth=pd_inf)
        )


def test_invalid_mesh_type_raises() -> None:
    _, displacement = _build_synthetic_setup()
    with pytest.raises(PatternError, match="trimesh.Trimesh"):
        create_displaced_mesh_copy(object(), displacement)


def test_invalid_displacement_type_raises() -> None:
    mesh, _ = _build_synthetic_setup()
    with pytest.raises(PatternError, match="VertexPatternDisplacement"):
        create_displaced_mesh_copy(mesh, object())


def test_synthetic_subdivided_bottle_integration() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=48, height_segments=12,
    )
    surface_map = create_vertex_surface_coordinates(mesh)
    risk = _uniform_risk_map()
    pattern = create_gaussian_dimple_pattern(
        risk, count=10, amplitude=1.0,
        sigma_u=0.05, sigma_v=0.05,
        max_depth=0.25, seed=42,
    )
    displacement = compute_vertex_displacement_amounts(
        mesh, surface_map, pattern,
        active_threshold=0.01,
        exclude_v_boundary_epsilon=0.02,
    )

    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)

    result = create_displaced_mesh_copy(mesh, displacement, mode="inward")

    assert int(len(result.displaced_mesh.vertices)) == int(len(mesh.vertices))
    assert int(len(result.displaced_mesh.faces)) == int(len(mesh.faces))
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)
    assert int(result.moved_mask.sum()) > 0
    assert float(result.max_displacement) > 0.0


def test_mesh_quality_smoke_on_displaced_mesh() -> None:
    mesh, displacement = _build_synthetic_setup()
    result = create_displaced_mesh_copy(mesh, displacement, mode="inward")
    report = create_mesh_quality_report(result.displaced_mesh)
    assert int(report.vertex_count) > 0
    assert int(report.face_count) > 0
    # ``is_watertight`` is recorded for visibility but not required to
    # remain true after displacement; manufacturability / repair are
    # explicitly out of scope for this foundation.
    assert isinstance(report.is_watertight, bool)
