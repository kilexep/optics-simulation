from pathlib import Path

import pytest
import trimesh

from optics_simulation.geometry import (
    BottleMeshReadinessFileReport,
    BottleMeshReadinessReport,
    GeometryError,
    create_subdivided_synthetic_bottle_shell,
    load_bottle_mesh_readiness_report,
)


SHELL_OUTER_RADIUS = 30.0
SHELL_WALL_THICKNESS = 1.0
SHELL_HEIGHT = 120.0
SHELL_SECTIONS = 64
SHELL_HEIGHT_SEGMENTS = 12


def _export_synthetic_shell_stl(tmp_path: Path) -> Path:
    mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=SHELL_OUTER_RADIUS,
        wall_thickness=SHELL_WALL_THICKNESS,
        height=SHELL_HEIGHT,
        sections=SHELL_SECTIONS,
        height_segments=SHELL_HEIGHT_SEGMENTS,
    )
    path = tmp_path / "synthetic_shell.stl"
    mesh.export(path)
    return path


def test_load_bottle_mesh_readiness_report_returns_dataclass(
    tmp_path: Path,
) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    file_report = load_bottle_mesh_readiness_report(stl_path)
    assert isinstance(file_report, BottleMeshReadinessFileReport)
    assert isinstance(
        file_report.readiness_report, BottleMeshReadinessReport
    )
    assert (
        file_report.report_type == "bottle_mesh_readiness_file_report"
    )


def test_temp_synthetic_shell_stl_loads_successfully(
    tmp_path: Path,
) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    file_report = load_bottle_mesh_readiness_report(stl_path)
    fit = file_report.readiness_report.cylindrical_fit
    qr = file_report.readiness_report.quality_report
    assert int(qr.vertex_count) > 0
    assert int(qr.face_count) > 0
    assert float(fit.height) == pytest.approx(SHELL_HEIGHT, rel=1e-6)
    assert float(fit.estimated_radius) > 0.0


def test_load_success_is_true_for_valid_temp_stl(
    tmp_path: Path,
) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    file_report = load_bottle_mesh_readiness_report(stl_path)
    assert file_report.load_success is True
    assert file_report.load_error is None
    assert file_report.file_exists is True
    assert file_report.mesh_path == str(stl_path)


def test_simulation_ready_true_with_reasonable_expected_ranges(
    tmp_path: Path,
) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    file_report = load_bottle_mesh_readiness_report(
        stl_path,
        expected_height_range=(115.0, 125.0),
        expected_radius_range=(25.0, 31.0),
    )
    assert file_report.readiness_report.simulation_ready is True
    assert file_report.readiness_report.blocking_issues == ()


def test_scale_factor_propagates_into_readiness_report(
    tmp_path: Path,
) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    base = load_bottle_mesh_readiness_report(stl_path, scale_factor=1.0)
    scaled = load_bottle_mesh_readiness_report(
        stl_path, scale_factor=2.0,
    )
    assert scaled.scale_factor == pytest.approx(2.0, rel=1e-9)
    assert (
        scaled.readiness_report.scale_factor_applied
        == pytest.approx(2.0, rel=1e-9)
    )
    assert (
        scaled.readiness_report.cylindrical_fit.height
        == pytest.approx(
            2.0 * base.readiness_report.cylindrical_fit.height,
            rel=1e-6,
        )
    )
    assert (
        scaled.readiness_report.cylindrical_fit.estimated_radius
        == pytest.approx(
            2.0
            * base.readiness_report.cylindrical_fit.estimated_radius,
            rel=1e-6,
        )
    )


def test_missing_file_raises_geometry_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.stl"
    with pytest.raises(GeometryError, match="not found"):
        load_bottle_mesh_readiness_report(missing)


def test_invalid_mesh_path_type_raises_geometry_error() -> None:
    with pytest.raises(GeometryError, match="mesh_path"):
        load_bottle_mesh_readiness_report(42)  # type: ignore[arg-type]
    with pytest.raises(GeometryError, match="mesh_path"):
        load_bottle_mesh_readiness_report(None)  # type: ignore[arg-type]


def test_invalid_scale_factor_raises_geometry_error(
    tmp_path: Path,
) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(GeometryError, match="scale_factor"):
            load_bottle_mesh_readiness_report(
                stl_path, scale_factor=bad,
            )


def test_expected_height_range_violation_appears_as_blocking_issue(
    tmp_path: Path,
) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    file_report = load_bottle_mesh_readiness_report(
        stl_path,
        expected_height_range=(200.0, 300.0),
        expected_radius_range=(25.0, 31.0),
    )
    readiness = file_report.readiness_report
    assert readiness.simulation_ready is False
    assert any(
        issue.startswith("height_out_of_range")
        for issue in readiness.blocking_issues
    )


def test_no_mesh_repair_is_performed(tmp_path: Path) -> None:
    """Loaded mesh metrics must match the on-disk mesh without repair."""
    stl_path = _export_synthetic_shell_stl(tmp_path)
    direct = trimesh.load(stl_path, force="mesh")
    direct_vertex_count = int(len(direct.vertices))
    direct_face_count = int(len(direct.faces))
    direct_height = float(direct.bounds[1, 2] - direct.bounds[0, 2])

    file_report = load_bottle_mesh_readiness_report(stl_path)
    qr = file_report.readiness_report.quality_report
    fit = file_report.readiness_report.cylindrical_fit

    assert int(qr.vertex_count) == direct_vertex_count
    assert int(qr.face_count) == direct_face_count
    assert float(fit.height) == pytest.approx(direct_height, rel=1e-9)


def test_no_extra_file_output_in_tmp_path(tmp_path: Path) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    snapshot_before = sorted(p.name for p in tmp_path.iterdir())

    load_bottle_mesh_readiness_report(
        stl_path,
        scale_factor=2.0,
        expected_height_range=(200.0, 300.0),
        expected_radius_range=(50.0, 70.0),
        require_watertight=True,
    )

    snapshot_after = sorted(p.name for p in tmp_path.iterdir())
    assert snapshot_after == snapshot_before


def test_input_temp_stl_remains_on_disk_after_call(
    tmp_path: Path,
) -> None:
    stl_path = _export_synthetic_shell_stl(tmp_path)
    bytes_before = stl_path.read_bytes()

    load_bottle_mesh_readiness_report(stl_path)

    assert stl_path.is_file()
    bytes_after = stl_path.read_bytes()
    assert bytes_after == bytes_before
