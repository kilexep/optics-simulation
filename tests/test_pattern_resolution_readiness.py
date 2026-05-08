import numpy as np
import pytest

from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.pattern import (
    MeshSubdivisionReport,
    PatternError,
    PatternResolutionReadinessReport,
    compute_pattern_resolution_readiness,
    create_subdivided_mesh_copy_for_patterning,
)


def _mesh():
    return create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=64, height_segments=12,
    )


def test_compute_returns_report() -> None:
    mesh = _mesh()
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    report = compute_pattern_resolution_readiness(
        mesh, include_mask=mask,
    )
    assert isinstance(report, PatternResolutionReadinessReport)
    assert report.report_type == (
        "pattern_resolution_readiness_diagnostic"
    )


def test_selected_vertex_count_matches_mask() -> None:
    mesh = _mesh()
    mask = np.zeros(int(len(mesh.vertices)), dtype=bool)
    mask[: len(mask) // 2] = True
    report = compute_pattern_resolution_readiness(
        mesh, include_mask=mask, min_vertices_required=10,
    )
    assert int(report.selected_vertex_count) == int(mask.sum())
    assert int(report.vertex_count) == int(len(mesh.vertices))


def test_edge_length_stats_are_positive_for_valid_mesh() -> None:
    mesh = _mesh()
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    report = compute_pattern_resolution_readiness(
        mesh, include_mask=mask,
    )
    assert int(report.selected_edge_count) > 0
    assert float(report.edge_length_min) > 0.0
    assert float(report.edge_length_median) > 0.0
    assert float(report.edge_length_max) >= float(
        report.edge_length_median
    )


def test_low_selected_vertex_count_gives_insufficient_label() -> None:
    mesh = _mesh()
    n = int(len(mesh.vertices))
    mask = np.zeros(n, dtype=bool)
    # Two adjacent vertices guarantees at least one selected edge,
    # while still being well below min_vertices_required.
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    mask[edges[0, 0]] = True
    mask[edges[0, 1]] = True
    report = compute_pattern_resolution_readiness(
        mesh, include_mask=mask,
        min_vertices_required=200,
    )
    assert (
        report.readiness_label
        in (
            "pattern_resolution_insufficient",
            "pattern_resolution_marginal",
        )
    )
    assert any(
        "min_vertices_required" in w for w in report.warnings
    )


def test_small_sigma_gives_warning() -> None:
    mesh = _mesh()
    surface = create_vertex_surface_coordinates(mesh)
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    report = compute_pattern_resolution_readiness(
        mesh,
        include_mask=mask,
        surface_map=surface,
        sigma_u=0.0001,
        sigma_v=0.0001,
        max_depth=0.05,
        min_vertices_required=10,
    )
    assert any(
        "uv_spacing_median" in w for w in report.warnings
    )


def test_large_max_depth_gives_warning() -> None:
    mesh = _mesh()
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    report = compute_pattern_resolution_readiness(
        mesh,
        include_mask=mask,
        sigma_u=0.05, sigma_v=0.05,
        max_depth=1.0e6,
        min_vertices_required=10,
    )
    assert any(
        "max_depth" in w for w in report.warnings
    )


def test_surface_map_populates_uv_spacing_fields() -> None:
    mesh = _mesh()
    surface = create_vertex_surface_coordinates(mesh)
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    report = compute_pattern_resolution_readiness(
        mesh, include_mask=mask, surface_map=surface,
        min_vertices_required=10,
    )
    assert report.estimated_uv_spacing_median is not None
    assert float(report.estimated_uv_spacing_median) > 0.0
    assert report.sigma_u_to_uv_spacing_ratio is not None
    assert report.sigma_v_to_uv_spacing_ratio is not None


def test_no_surface_map_leaves_uv_spacing_none() -> None:
    mesh = _mesh()
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    report = compute_pattern_resolution_readiness(
        mesh, include_mask=mask,
        min_vertices_required=10,
    )
    assert report.estimated_uv_spacing_median is None
    assert report.sigma_u_to_uv_spacing_ratio is None
    assert report.sigma_v_to_uv_spacing_ratio is None


def test_invalid_include_mask_shape_raises() -> None:
    mesh = _mesh()
    bad_mask = np.zeros(7, dtype=bool)
    with pytest.raises(PatternError, match="include_mask"):
        compute_pattern_resolution_readiness(
            mesh, include_mask=bad_mask,
        )


def test_invalid_sigma_raises() -> None:
    mesh = _mesh()
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    with pytest.raises(PatternError, match="sigma_u"):
        compute_pattern_resolution_readiness(
            mesh, include_mask=mask, sigma_u=0.0,
        )


def test_invalid_max_depth_raises() -> None:
    mesh = _mesh()
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    with pytest.raises(PatternError, match="max_depth"):
        compute_pattern_resolution_readiness(
            mesh, include_mask=mask, max_depth=-1.0,
        )


def test_mesh_is_not_mutated() -> None:
    mesh = _mesh()
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    compute_pattern_resolution_readiness(
        mesh, include_mask=mask, min_vertices_required=10,
    )
    assert np.array_equal(
        np.asarray(mesh.vertices), vertices_before
    )
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_subdivision_returns_mesh_and_report() -> None:
    mesh = _mesh()
    sub, report = create_subdivided_mesh_copy_for_patterning(
        mesh, iterations=1,
    )
    assert isinstance(report, MeshSubdivisionReport)
    assert int(len(sub.vertices)) == int(
        report.subdivided_vertex_count
    )
    assert int(len(sub.faces)) == int(report.subdivided_face_count)


def test_subdivision_increases_vertex_and_face_count() -> None:
    mesh = _mesh()
    sub, report = create_subdivided_mesh_copy_for_patterning(
        mesh, iterations=1,
    )
    assert int(report.subdivided_vertex_count) > int(
        report.original_vertex_count
    )
    assert int(report.subdivided_face_count) > int(
        report.original_face_count
    )


def test_subdivision_invalid_iterations_raises() -> None:
    mesh = _mesh()
    with pytest.raises(PatternError, match="iterations"):
        create_subdivided_mesh_copy_for_patterning(
            mesh, iterations=0,
        )


def test_no_file_output(tmp_path) -> None:
    mesh = _mesh()
    mask = np.ones(int(len(mesh.vertices)), dtype=bool)
    before = sorted(tmp_path.iterdir())
    compute_pattern_resolution_readiness(
        mesh, include_mask=mask, min_vertices_required=10,
    )
    create_subdivided_mesh_copy_for_patterning(mesh, iterations=1)
    after = sorted(tmp_path.iterdir())
    assert before == after
