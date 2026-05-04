from pathlib import Path

import pytest
import trimesh

from optics_simulation.geometry import (
    GeometryError,
    MeshLoadError,
    MeshQualityReport,
    create_mesh_quality_report,
    load_mesh,
    load_mesh_with_report,
)


@pytest.fixture
def box_stl(tmp_path: Path) -> Path:
    mesh = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    path = tmp_path / "box.stl"
    mesh.export(path)
    return path


@pytest.fixture
def cylinder_stl(tmp_path: Path) -> Path:
    mesh = trimesh.creation.cylinder(radius=15.0, height=200.0, sections=64)
    path = tmp_path / "cylinder.stl"
    mesh.export(path)
    return path


def test_load_mesh_returns_trimesh(box_stl: Path) -> None:
    mesh = load_mesh(box_stl)
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) > 0


def test_load_mesh_accepts_str_path(box_stl: Path) -> None:
    mesh = load_mesh(str(box_stl))
    assert isinstance(mesh, trimesh.Trimesh)


def test_load_mesh_with_report_returns_pair(box_stl: Path) -> None:
    mesh, report = load_mesh_with_report(box_stl, units="mm")
    assert isinstance(mesh, trimesh.Trimesh)
    assert isinstance(report, MeshQualityReport)
    assert report.units == "mm"
    assert report.source_path == str(box_stl)


def test_box_report_counts_positive(box_stl: Path) -> None:
    _, report = load_mesh_with_report(box_stl)
    assert report.vertex_count > 0
    assert report.face_count > 0
    assert not report.is_empty


def test_cylinder_report_counts_positive(cylinder_stl: Path) -> None:
    _, report = load_mesh_with_report(cylinder_stl)
    assert report.vertex_count > 0
    assert report.face_count > 0
    assert not report.is_empty


def test_cylinder_report_geometry_tuples(cylinder_stl: Path) -> None:
    _, report = load_mesh_with_report(cylinder_stl)
    for field in (report.bounds_min, report.bounds_max, report.extents, report.centroid):
        assert isinstance(field, tuple)
        assert len(field) == 3
        assert all(isinstance(v, float) for v in field)
    for lo, hi in zip(report.bounds_min, report.bounds_max):
        assert lo <= hi


def test_cylinder_extents_match_creation(cylinder_stl: Path) -> None:
    _, report = load_mesh_with_report(cylinder_stl)
    ex, ey, ez = report.extents
    # cylinder: radius=15 (diameter ~30) in the radial plane, height=200 along axis.
    # trimesh.creation.cylinder is axis-aligned along z by default.
    assert ex == pytest.approx(30.0, abs=0.5)
    assert ey == pytest.approx(30.0, abs=0.5)
    assert ez == pytest.approx(200.0, abs=0.5)


def test_box_is_watertight(box_stl: Path) -> None:
    _, report = load_mesh_with_report(box_stl)
    assert report.is_watertight is True


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MeshLoadError, match="not found"):
        load_mesh(tmp_path / "does_not_exist.stl")


def test_invalid_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.stl"
    bad.write_text("this is not an STL file at all\n", encoding="utf-8")
    with pytest.raises(MeshLoadError):
        load_mesh(bad)


def test_mesh_load_error_is_geometry_error() -> None:
    assert issubclass(MeshLoadError, GeometryError)


def test_units_passthrough_default_none(box_stl: Path) -> None:
    mesh = load_mesh(box_stl)
    report = create_mesh_quality_report(mesh, source_path=box_stl)
    assert report.units is None
    assert report.source_path == str(box_stl)


def test_create_report_without_source_path(box_stl: Path) -> None:
    mesh = load_mesh(box_stl)
    report = create_mesh_quality_report(mesh)
    assert report.source_path is None
