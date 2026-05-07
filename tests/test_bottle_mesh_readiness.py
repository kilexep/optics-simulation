import numpy as np
import pytest
import trimesh

from optics_simulation.geometry import (
    BottleMeshReadinessReport,
    CylindricalFitReport,
    GeometryError,
    MeshQualityReport,
    compute_bottle_mesh_readiness_report,
    compute_cylindrical_fit_report,
    create_subdivided_synthetic_bottle_body,
    create_subdivided_synthetic_bottle_shell,
    create_synthetic_bottle_body,
)


def _make_inconsistent_winding_mesh() -> trimesh.Trimesh:
    """Two faces sharing edge 1-2 traversed in the same direction."""
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _make_non_watertight_mesh() -> trimesh.Trimesh:
    mesh = create_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=64,
    )
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=faces[:-1].copy(),
        process=False,
    )


def test_compute_cylindrical_fit_report_returns_dataclass() -> None:
    mesh = create_synthetic_bottle_body()
    report = compute_cylindrical_fit_report(mesh)
    assert isinstance(report, CylindricalFitReport)
    assert isinstance(report.bounds_min, np.ndarray)
    assert isinstance(report.bounds_max, np.ndarray)
    assert report.bounds_min.shape == (3,)
    assert report.bounds_max.shape == (3,)


def test_synthetic_cylinder_is_cylindrical_like() -> None:
    mesh = create_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=96,
    )
    report = compute_cylindrical_fit_report(mesh)
    assert report.fit_quality_label == "cylindrical_like"


def test_subdivided_synthetic_bottle_body_is_cylindrical_like() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=96, height_segments=24,
    )
    report = compute_cylindrical_fit_report(mesh)
    assert report.fit_quality_label == "cylindrical_like"


def test_synthetic_shell_is_cylindrical_like_or_weakly() -> None:
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=30.0,
        wall_thickness=1.0,
        height=120.0,
        sections=96,
        height_segments=24,
    )
    report = compute_cylindrical_fit_report(mesh)
    assert report.fit_quality_label in {
        "cylindrical_like", "weakly_cylindrical",
    }
    assert report.fit_quality_label != "not_cylindrical"


def test_box_mesh_is_weakly_cylindrical_or_not_cylindrical() -> None:
    box = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    box = box.subdivide()
    report = compute_cylindrical_fit_report(box)
    assert report.fit_quality_label in {
        "weakly_cylindrical", "not_cylindrical",
    }


def test_estimated_radius_near_expected_for_synthetic_cylinder() -> None:
    radius = 30.0
    mesh = create_subdivided_synthetic_bottle_body(
        radius=radius, height=120.0, sections=128, height_segments=24,
    )
    report = compute_cylindrical_fit_report(mesh)
    assert report.estimated_radius == pytest.approx(radius, rel=1e-3)


def test_height_near_expected_for_synthetic_cylinder() -> None:
    height = 120.0
    mesh = create_synthetic_bottle_body(
        radius=30.0, height=height, sections=96,
    )
    report = compute_cylindrical_fit_report(mesh)
    assert report.height == pytest.approx(height, rel=1e-9)


def test_radius_cv_nonnegative_finite_for_valid_cylinder() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=96, height_segments=12,
    )
    report = compute_cylindrical_fit_report(mesh)
    assert np.isfinite(report.radius_cv)
    assert report.radius_cv >= 0.0


def test_report_includes_z_axis_and_no_pca_note() -> None:
    mesh = create_synthetic_bottle_body()
    report = compute_cylindrical_fit_report(mesh)
    note_blob = "\n".join(report.notes).lower()
    assert "z-axis" in note_blob
    assert "pca" in note_blob


def test_compute_bottle_mesh_readiness_report_returns_dataclass() -> None:
    mesh = create_synthetic_bottle_body()
    report = compute_bottle_mesh_readiness_report(mesh)
    assert isinstance(report, BottleMeshReadinessReport)
    assert isinstance(report.quality_report, MeshQualityReport)
    assert isinstance(report.cylindrical_fit, CylindricalFitReport)
    assert report.report_type == "bottle_mesh_readiness_diagnostic"


def test_synthetic_cylinder_readiness_simulation_ready_true() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=96, height_segments=24,
    )
    report = compute_bottle_mesh_readiness_report(
        mesh,
        expected_height_range=(115.0, 125.0),
        expected_radius_range=(28.0, 32.0),
    )
    assert report.simulation_ready is True
    assert report.blocking_issues == ()


def test_expected_height_range_violation_creates_blocking_issue() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=96, height_segments=24,
    )
    report = compute_bottle_mesh_readiness_report(
        mesh,
        expected_height_range=(200.0, 300.0),
        expected_radius_range=(28.0, 32.0),
    )
    assert report.simulation_ready is False
    assert any(
        issue.startswith("height_out_of_range")
        for issue in report.blocking_issues
    )


def test_expected_radius_range_violation_creates_blocking_issue() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=96, height_segments=24,
    )
    report = compute_bottle_mesh_readiness_report(
        mesh,
        expected_height_range=(115.0, 125.0),
        expected_radius_range=(50.0, 70.0),
    )
    assert report.simulation_ready is False
    assert any(
        issue.startswith("estimated_radius_out_of_range")
        for issue in report.blocking_issues
    )


def test_require_watertight_blocks_non_watertight_mesh() -> None:
    mesh = _make_non_watertight_mesh()
    assert mesh.is_watertight is False
    report = compute_bottle_mesh_readiness_report(
        mesh, require_watertight=True,
    )
    assert report.simulation_ready is False
    assert "require_watertight_violated" in report.blocking_issues


def test_require_winding_consistent_blocks_inconsistent_mesh() -> None:
    mesh = _make_inconsistent_winding_mesh()
    if mesh.is_winding_consistent:
        pytest.skip(
            "inconsistent winding fixture unavailable on this trimesh "
            "version"
        )
    report = compute_bottle_mesh_readiness_report(
        mesh, require_winding_consistent=True,
    )
    assert report.simulation_ready is False
    assert (
        "require_winding_consistent_violated" in report.blocking_issues
    )


def test_scale_factor_changes_report_without_mutating_input() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0, sections=96, height_segments=24,
    )
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)

    base = compute_bottle_mesh_readiness_report(mesh, scale_factor=1.0)
    scaled = compute_bottle_mesh_readiness_report(
        mesh, scale_factor=2.0,
    )

    assert base.cylindrical_fit.height == pytest.approx(120.0, rel=1e-9)
    assert scaled.cylindrical_fit.height == pytest.approx(
        240.0, rel=1e-6,
    )
    assert scaled.cylindrical_fit.estimated_radius == pytest.approx(
        2.0 * base.cylindrical_fit.estimated_radius, rel=1e-6,
    )
    assert scaled.scale_factor_applied == pytest.approx(2.0, rel=1e-9)
    assert any(
        w.startswith("scale_factor_applied") for w in scaled.warnings
    )

    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_invalid_scale_factor_raises() -> None:
    mesh = create_synthetic_bottle_body()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="scale_factor"):
            compute_bottle_mesh_readiness_report(mesh, scale_factor=bad)


def test_invalid_expected_range_raises() -> None:
    mesh = create_synthetic_bottle_body()
    with pytest.raises(GeometryError, match="expected_height_range"):
        compute_bottle_mesh_readiness_report(
            mesh, expected_height_range=(120.0, 100.0),
        )
    with pytest.raises(GeometryError, match="expected_radius_range"):
        compute_bottle_mesh_readiness_report(
            mesh, expected_radius_range=(30.0, 30.0),
        )
    with pytest.raises(GeometryError, match="expected_height_range"):
        compute_bottle_mesh_readiness_report(
            mesh, expected_height_range=(float("nan"), 100.0),
        )
    with pytest.raises(GeometryError, match="expected_radius_range"):
        compute_bottle_mesh_readiness_report(
            mesh, expected_radius_range=(0.0, 30.0, 60.0),
        )


def test_empty_mesh_raises_geometry_error() -> None:
    """Empty meshes raise; this is documented in the public API."""
    empty_mesh = trimesh.Trimesh(
        vertices=np.zeros((0, 3), dtype=float),
        faces=np.zeros((0, 3), dtype=np.int64),
        process=False,
    )
    with pytest.raises(GeometryError, match="empty"):
        compute_cylindrical_fit_report(empty_mesh)
    with pytest.raises(GeometryError, match="empty"):
        compute_bottle_mesh_readiness_report(empty_mesh)


def test_input_mesh_vertices_and_faces_not_mutated() -> None:
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=30.0, wall_thickness=1.0, height=120.0,
        sections=64, height_segments=12,
    )
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)

    compute_cylindrical_fit_report(mesh)
    compute_bottle_mesh_readiness_report(
        mesh,
        scale_factor=1.5,
        expected_height_range=(150.0, 200.0),
        expected_radius_range=(40.0, 50.0),
        require_watertight=True,
    )

    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_no_file_output(tmp_path) -> None:
    mesh = create_synthetic_bottle_body()
    before = sorted(tmp_path.iterdir())
    compute_cylindrical_fit_report(mesh)
    compute_bottle_mesh_readiness_report(mesh)
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_non_trimesh_input_raises() -> None:
    with pytest.raises(GeometryError, match="trimesh.Trimesh"):
        compute_cylindrical_fit_report("not a mesh")  # type: ignore[arg-type]
    with pytest.raises(GeometryError, match="trimesh.Trimesh"):
        compute_bottle_mesh_readiness_report(42)  # type: ignore[arg-type]


def test_invalid_radial_sample_exclude_z_fraction_raises() -> None:
    mesh = create_synthetic_bottle_body()
    with pytest.raises(GeometryError, match="radial_sample"):
        compute_cylindrical_fit_report(
            mesh, radial_sample_exclude_z_fraction=-0.1,
        )
    with pytest.raises(GeometryError, match="radial_sample"):
        compute_cylindrical_fit_report(
            mesh, radial_sample_exclude_z_fraction=0.5,
        )


def test_invalid_cv_thresholds_raise() -> None:
    mesh = create_synthetic_bottle_body()
    with pytest.raises(GeometryError, match="cv"):
        compute_cylindrical_fit_report(
            mesh,
            radius_cv_cylindrical_threshold=0.30,
            radius_cv_weak_threshold=0.20,
        )
    with pytest.raises(GeometryError, match="cv"):
        compute_cylindrical_fit_report(
            mesh, radius_cv_cylindrical_threshold=-0.1,
        )
