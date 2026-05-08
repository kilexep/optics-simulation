import numpy as np
import pytest

from optics_simulation.geometry import (
    ActualBottleBodyMaskReport,
    GeometryError,
    create_actual_bottle_body_vertex_mask,
    create_subdivided_synthetic_bottle_body,
)


def _mesh():
    return create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=64, height_segments=12,
    )


def test_returns_report_dataclass() -> None:
    report = create_actual_bottle_body_vertex_mask(_mesh())
    assert isinstance(report, ActualBottleBodyMaskReport)
    assert report.report_type == "actual_bottle_body_mask_report"


def test_include_mask_shape_equals_vertex_count() -> None:
    mesh = _mesh()
    report = create_actual_bottle_body_vertex_mask(mesh)
    assert report.include_mask.shape == (int(len(mesh.vertices)),)
    assert report.include_mask.dtype == bool
    assert int(report.vertex_count) == int(len(mesh.vertices))


def test_selected_count_positive_for_synthetic_body() -> None:
    report = create_actual_bottle_body_vertex_mask(_mesh())
    assert int(report.selected_count) > 0


def test_selected_fraction_in_unit_interval() -> None:
    report = create_actual_bottle_body_vertex_mask(_mesh())
    assert 0.0 <= float(report.selected_fraction) <= 1.0


def test_z_band_excludes_extremes() -> None:
    mesh = _mesh()
    report = create_actual_bottle_body_vertex_mask(
        mesh, z_min_fraction=0.20, z_max_fraction=0.80,
    )
    vertices = np.asarray(mesh.vertices, dtype=float)
    selected_z = vertices[report.include_mask, 2]
    assert (
        float(selected_z.min())
        >= float(report.body_z_min) - 1e-9
    )
    assert (
        float(selected_z.max())
        <= float(report.body_z_max) + 1e-9
    )


def test_radial_quantile_affects_selected_count() -> None:
    mesh = _mesh()
    low = create_actual_bottle_body_vertex_mask(
        mesh, radial_quantile=0.10,
    )
    high = create_actual_bottle_body_vertex_mask(
        mesh, radial_quantile=0.90,
    )
    assert int(low.selected_count) >= int(high.selected_count)


def test_invalid_z_fractions_raise() -> None:
    mesh = _mesh()
    with pytest.raises(GeometryError, match="z_min_fraction"):
        create_actual_bottle_body_vertex_mask(
            mesh, z_min_fraction=0.6, z_max_fraction=0.5,
        )
    with pytest.raises(GeometryError, match="z_min_fraction"):
        create_actual_bottle_body_vertex_mask(
            mesh, z_min_fraction=-0.1, z_max_fraction=0.5,
        )
    with pytest.raises(GeometryError, match="z_min_fraction"):
        create_actual_bottle_body_vertex_mask(
            mesh, z_min_fraction=0.1, z_max_fraction=1.5,
        )


def test_invalid_radial_quantile_raises() -> None:
    mesh = _mesh()
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(GeometryError, match="radial_quantile"):
            create_actual_bottle_body_vertex_mask(
                mesh, radial_quantile=bad,
            )


def test_input_mesh_not_mutated() -> None:
    mesh = _mesh()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    create_actual_bottle_body_vertex_mask(mesh)
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_no_file_output(tmp_path) -> None:
    mesh = _mesh()
    before = sorted(tmp_path.iterdir())
    create_actual_bottle_body_vertex_mask(mesh)
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_invalid_mesh_type_raises() -> None:
    with pytest.raises(GeometryError, match="trimesh.Trimesh"):
        create_actual_bottle_body_vertex_mask("not a mesh")  # type: ignore[arg-type]
