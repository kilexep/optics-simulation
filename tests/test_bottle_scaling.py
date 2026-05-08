import numpy as np
import pytest
import trimesh

from optics_simulation.geometry import (
    BottleTargetScaleReport,
    GeometryError,
    InnerOffsetMeshReport,
    compute_target_dimension_scale,
    create_inner_offset_mesh_from_vertex_normals,
    create_subdivided_synthetic_bottle_shell,
    create_target_scaled_mesh_copy,
)


def _shell() -> trimesh.Trimesh:
    return create_subdivided_synthetic_bottle_shell(
        outer_radius=30.0,
        wall_thickness=1.0,
        height=120.0,
        sections=64,
        height_segments=12,
    )


def test_compute_target_dimension_scale_matches_target_dimensions() -> None:
    mesh = _shell()
    scale_xyz = compute_target_dimension_scale(
        mesh, target_height=225.6, target_diameter=72.1,
    )
    assert isinstance(scale_xyz, tuple)
    assert len(scale_xyz) == 3
    assert scale_xyz[0] == scale_xyz[1]

    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    expected_xy = 72.1 / float(max(extents[0], extents[1]))
    expected_z = 225.6 / float(extents[2])
    assert scale_xyz[0] == pytest.approx(expected_xy, rel=1e-9)
    assert scale_xyz[2] == pytest.approx(expected_z, rel=1e-9)


def test_create_target_scaled_mesh_copy_does_not_mutate_input() -> None:
    mesh = _shell()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    create_target_scaled_mesh_copy(
        mesh, target_height=225.6, target_diameter=72.1,
    )
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_scaled_mesh_height_equals_target_height() -> None:
    mesh = _shell()
    scaled, report = create_target_scaled_mesh_copy(
        mesh, target_height=225.6, target_diameter=72.1,
    )
    bounds = np.asarray(scaled.bounds, dtype=float)
    height = float(bounds[1, 2] - bounds[0, 2])
    assert height == pytest.approx(225.6, rel=1e-9)
    assert report.scaled_height == pytest.approx(225.6, rel=1e-9)


def test_scaled_xy_extent_equals_target_diameter() -> None:
    mesh = _shell()
    scaled, report = create_target_scaled_mesh_copy(
        mesh, target_height=225.6, target_diameter=72.1,
    )
    bounds = np.asarray(scaled.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    xy_extent = float(max(extents[0], extents[1]))
    assert xy_extent == pytest.approx(72.1, rel=1e-9)
    assert report.scaled_xy_extent == pytest.approx(72.1, rel=1e-9)


def test_invalid_target_values_raise_geometry_error() -> None:
    mesh = _shell()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="target_height"):
            compute_target_dimension_scale(
                mesh, target_height=bad, target_diameter=72.1,
            )
        with pytest.raises(GeometryError, match="target_diameter"):
            compute_target_dimension_scale(
                mesh, target_height=225.6, target_diameter=bad,
            )


def test_create_inner_offset_mesh_returns_mesh_and_report() -> None:
    mesh = _shell()
    inner, report = create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3,
    )
    assert isinstance(inner, trimesh.Trimesh)
    assert isinstance(report, InnerOffsetMeshReport)


def test_inner_mesh_has_same_vertex_and_face_counts() -> None:
    mesh = _shell()
    inner, report = create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3,
    )
    assert int(len(inner.vertices)) == int(len(mesh.vertices))
    assert int(len(inner.faces)) == int(len(mesh.faces))
    assert report.vertex_count == int(len(mesh.vertices))
    assert report.face_count == int(len(mesh.faces))


def test_source_mesh_not_mutated_by_inner_offset() -> None:
    mesh = _shell()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    create_inner_offset_mesh_from_vertex_normals(mesh, thickness=0.3)
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_invalid_thickness_raises() -> None:
    mesh = _shell()
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="thickness"):
            create_inner_offset_mesh_from_vertex_normals(
                mesh, thickness=bad,
            )


def test_inner_offset_report_fields_valid() -> None:
    mesh = _shell()
    _, report = create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3, invert=True,
    )
    assert report.thickness == pytest.approx(0.3, rel=1e-9)
    assert report.inverted is True
    assert report.report_type == "inner_offset_mesh_report"
    assert report.source_mesh_watertight is bool(mesh.is_watertight)
    assert np.isfinite(report.original_radial_stat)
    assert np.isfinite(report.minus_radial_stat)
    assert np.isfinite(report.plus_radial_stat)
    assert np.isfinite(report.selected_radial_stat)
    assert np.isfinite(report.radial_delta)
    assert isinstance(report.notes, tuple)
    assert len(report.notes) >= 1


def test_inner_offset_invert_false_keeps_original_winding() -> None:
    mesh = _shell()
    _, report = create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3, invert=False,
    )
    assert report.inverted is False


def test_offset_mode_minus_normals_preserves_legacy_formula() -> None:
    mesh = _shell()
    inner, report = create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3, invert=False,
        offset_mode="minus_normals",
    )
    expected_vertices = (
        np.asarray(mesh.vertices, dtype=float)
        - 0.3 * np.asarray(mesh.vertex_normals, dtype=float)
    )
    np.testing.assert_allclose(
        np.asarray(inner.vertices), expected_vertices, atol=1e-12,
    )
    assert report.offset_mode == "minus_normals"
    assert report.selected_offset_sign == pytest.approx(-1.0, abs=1e-12)


def test_offset_mode_plus_normals_uses_plus_formula() -> None:
    mesh = _shell()
    inner, report = create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3, invert=False,
        offset_mode="plus_normals",
    )
    expected_vertices = (
        np.asarray(mesh.vertices, dtype=float)
        + 0.3 * np.asarray(mesh.vertex_normals, dtype=float)
    )
    np.testing.assert_allclose(
        np.asarray(inner.vertices), expected_vertices, atol=1e-12,
    )
    assert report.offset_mode == "plus_normals"
    assert report.selected_offset_sign == pytest.approx(+1.0, abs=1e-12)


def test_offset_mode_auto_chooses_minus_for_outward_normal_cylinder() -> None:
    mesh = trimesh.creation.cylinder(
        radius=30.0, height=120.0, sections=64,
    )
    _, report = create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3, offset_mode="auto",
    )
    assert report.offset_mode == "auto"
    assert report.selected_offset_sign == pytest.approx(-1.0, abs=1e-12)
    assert report.inward_offset_detected is True
    assert report.selected_radial_stat < report.original_radial_stat


def test_offset_mode_auto_chooses_plus_for_inward_normal_mesh() -> None:
    mesh = trimesh.creation.cylinder(
        radius=30.0, height=120.0, sections=64,
    )
    inverted_source = mesh.copy()
    inverted_source.invert()
    _, report = create_inner_offset_mesh_from_vertex_normals(
        inverted_source, thickness=0.3, offset_mode="auto",
    )
    assert report.offset_mode == "auto"
    assert report.selected_offset_sign == pytest.approx(+1.0, abs=1e-12)
    assert report.inward_offset_detected is True
    assert report.selected_radial_stat < report.original_radial_stat


def test_auto_offset_radial_stats_finite_and_consistent() -> None:
    mesh = trimesh.creation.cylinder(
        radius=30.0, height=120.0, sections=64,
    )
    _, report = create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3, offset_mode="auto",
    )
    assert np.isfinite(report.original_radial_stat)
    assert np.isfinite(report.minus_radial_stat)
    assert np.isfinite(report.plus_radial_stat)
    assert np.isfinite(report.selected_radial_stat)
    if report.selected_offset_sign < 0.0:
        assert report.selected_radial_stat == pytest.approx(
            report.minus_radial_stat, abs=1e-12,
        )
    else:
        assert report.selected_radial_stat == pytest.approx(
            report.plus_radial_stat, abs=1e-12,
        )
    assert report.radial_delta == pytest.approx(
        report.selected_radial_stat - report.original_radial_stat,
        abs=1e-12,
    )


def test_invalid_offset_mode_raises_geometry_error() -> None:
    mesh = _shell()
    with pytest.raises(GeometryError, match="offset_mode"):
        create_inner_offset_mesh_from_vertex_normals(
            mesh, thickness=0.3, offset_mode="something_else",
        )


def test_target_scale_report_dataclass_returned() -> None:
    mesh = _shell()
    _, report = create_target_scaled_mesh_copy(
        mesh, target_height=225.6, target_diameter=72.1,
    )
    assert isinstance(report, BottleTargetScaleReport)
    assert report.report_type == "bottle_target_dimension_scale_report"
    assert isinstance(report.original_bounds_min, np.ndarray)
    assert isinstance(report.scaled_bounds_max, np.ndarray)
    assert report.scale_xyz[0] == report.scale_xyz[1]


def test_no_file_output(tmp_path) -> None:
    mesh = _shell()
    before = sorted(tmp_path.iterdir())
    create_target_scaled_mesh_copy(
        mesh, target_height=225.6, target_diameter=72.1,
    )
    create_inner_offset_mesh_from_vertex_normals(
        mesh, thickness=0.3,
    )
    after = sorted(tmp_path.iterdir())
    assert before == after
