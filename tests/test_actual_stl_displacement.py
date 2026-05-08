import numpy as np
import pytest
import trimesh

from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.pattern import (
    ActualSTLPatternedMeshResult,
    GaussianDimple,
    GaussianDimplePattern,
    PatternError,
    VertexPatternDisplacement,
    compute_vertex_displacement_amounts,
    create_actual_stl_patterned_mesh_copy,
    evaluate_gaussian_dimple_field,
)


def _build_central_pattern_displacement(mesh: trimesh.Trimesh):
    surface_map = create_vertex_surface_coordinates(mesh)
    dimples = (
        GaussianDimple(
            center_u=0.5, center_v=0.5,
            amplitude=1.0, sigma_u=0.2, sigma_v=0.2,
        ),
    )
    nv, nu = 32, 32
    pattern = GaussianDimplePattern(
        dimples=dimples,
        depth_field=evaluate_gaussian_dimple_field(
            dimples, resolution=(nv, nu), clip=True,
        ),
        resolution=(nv, nu),
        max_depth=0.5,
        seed=0,
    )
    return compute_vertex_displacement_amounts(
        mesh, surface_map, pattern, active_threshold=0.0,
    )


def _solid_cylinder() -> trimesh.Trimesh:
    return create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=48, height_segments=8,
    )


def test_returns_dataclass() -> None:
    mesh = _solid_cylinder()
    disp = _build_central_pattern_displacement(mesh)
    result = create_actual_stl_patterned_mesh_copy(mesh, disp)
    assert isinstance(result, ActualSTLPatternedMeshResult)
    assert result.mode == "actual_stl_inward_auto"


def test_patterned_mesh_is_new_object() -> None:
    mesh = _solid_cylinder()
    disp = _build_central_pattern_displacement(mesh)
    result = create_actual_stl_patterned_mesh_copy(mesh, disp)
    assert result.patterned_mesh is not mesh
    assert (
        int(len(result.patterned_mesh.vertices))
        == int(len(mesh.vertices))
    )


def test_original_mesh_unchanged() -> None:
    mesh = _solid_cylinder()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    disp = _build_central_pattern_displacement(mesh)
    create_actual_stl_patterned_mesh_copy(mesh, disp)
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_moved_vertex_count_positive_with_nonzero_displacement() -> None:
    mesh = _solid_cylinder()
    disp = _build_central_pattern_displacement(mesh)
    result = create_actual_stl_patterned_mesh_copy(mesh, disp)
    assert int(result.moved_vertex_count) > 0
    assert int(result.moved_mask.sum()) == int(result.moved_vertex_count)


def test_selected_offset_sign_is_minus_or_plus() -> None:
    mesh = _solid_cylinder()
    disp = _build_central_pattern_displacement(mesh)
    result = create_actual_stl_patterned_mesh_copy(mesh, disp)
    assert float(result.selected_offset_sign) in (-1.0, 1.0)


def test_inward_detected_for_outward_normal_synthetic_cylinder() -> None:
    mesh = _solid_cylinder()
    disp = _build_central_pattern_displacement(mesh)
    result = create_actual_stl_patterned_mesh_copy(mesh, disp)
    assert result.selected_offset_sign == pytest.approx(-1.0, abs=1e-12)
    assert result.inward_displacement_detected is True


def test_inward_detected_for_inward_normal_inverted_mesh() -> None:
    mesh = _solid_cylinder()
    inverted = mesh.copy()
    inverted.invert()
    disp = _build_central_pattern_displacement(inverted)
    result = create_actual_stl_patterned_mesh_copy(inverted, disp)
    assert result.selected_offset_sign == pytest.approx(+1.0, abs=1e-12)
    assert result.inward_displacement_detected is True


def test_displacement_magnitudes_shape_matches_vertex_count() -> None:
    mesh = _solid_cylinder()
    disp = _build_central_pattern_displacement(mesh)
    result = create_actual_stl_patterned_mesh_copy(mesh, disp)
    n = int(len(mesh.vertices))
    assert result.displacement_magnitudes.shape == (n,)
    assert result.displacement_vectors.shape == (n, 3)
    assert result.moved_mask.shape == (n,)


def test_invalid_inputs_raise_pattern_error() -> None:
    mesh = _solid_cylinder()
    disp = _build_central_pattern_displacement(mesh)
    with pytest.raises(PatternError, match="trimesh.Trimesh"):
        create_actual_stl_patterned_mesh_copy(
            "not a mesh", disp,  # type: ignore[arg-type]
        )
    with pytest.raises(PatternError, match="VertexPatternDisplacement"):
        create_actual_stl_patterned_mesh_copy(
            mesh, "not a displacement",  # type: ignore[arg-type]
        )


def test_no_file_output(tmp_path) -> None:
    mesh = _solid_cylinder()
    disp = _build_central_pattern_displacement(mesh)
    before = sorted(tmp_path.iterdir())
    create_actual_stl_patterned_mesh_copy(mesh, disp)
    after = sorted(tmp_path.iterdir())
    assert before == after
