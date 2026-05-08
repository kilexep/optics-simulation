import numpy as np
import pytest
import trimesh

from optics_simulation.geometry import (
    GeometryError,
    create_subdivided_synthetic_bottle_shell,
)
from optics_simulation.optics import (
    LegacyPetWaterTraceSetup,
    MultiMeshTraceStepSpec,
    OpticsError,
    create_legacy_pet_water_trace_setup,
)


EXPECTED_LABELS = (
    "air_to_pet_outer_shell",
    "pet_to_water_inner_surface",
    "water_to_pet_inner_surface",
    "pet_to_air_outer_shell",
)


def _shell() -> trimesh.Trimesh:
    return create_subdivided_synthetic_bottle_shell(
        outer_radius=30.0,
        wall_thickness=1.0,
        height=120.0,
        sections=64,
        height_segments=12,
    )


def test_create_legacy_pet_water_trace_setup_returns_dataclass() -> None:
    setup = create_legacy_pet_water_trace_setup(_shell())
    assert isinstance(setup, LegacyPetWaterTraceSetup)


def test_step_specs_length_is_four() -> None:
    setup = create_legacy_pet_water_trace_setup(_shell())
    assert len(setup.step_specs) == 4
    for spec in setup.step_specs:
        assert isinstance(spec, MultiMeshTraceStepSpec)


def test_labels_match_expected_four_stages() -> None:
    setup = create_legacy_pet_water_trace_setup(_shell())
    labels = tuple(spec.label for spec in setup.step_specs)
    assert labels == EXPECTED_LABELS


def test_shell_mesh_and_water_mesh_are_distinct_objects() -> None:
    setup = create_legacy_pet_water_trace_setup(_shell())
    assert setup.shell_mesh is not setup.water_mesh
    assert setup.step_specs[0].mesh is setup.shell_mesh
    assert setup.step_specs[1].mesh is setup.water_mesh
    assert setup.step_specs[2].mesh is setup.water_mesh
    assert setup.step_specs[3].mesh is setup.shell_mesh


def test_water_mesh_face_and_vertex_count_match_shell_mesh() -> None:
    setup = create_legacy_pet_water_trace_setup(_shell())
    assert int(len(setup.water_mesh.vertices)) == int(
        len(setup.shell_mesh.vertices)
    )
    assert int(len(setup.water_mesh.faces)) == int(
        len(setup.shell_mesh.faces)
    )


def test_scale_report_target_dimensions_match_requested() -> None:
    setup = create_legacy_pet_water_trace_setup(
        _shell(),
        target_height=200.0,
        target_diameter=80.0,
    )
    assert setup.target_height == pytest.approx(200.0, rel=1e-9)
    assert setup.target_diameter == pytest.approx(80.0, rel=1e-9)
    assert setup.scale_report.target_height == pytest.approx(
        200.0, rel=1e-9,
    )
    assert setup.scale_report.target_diameter == pytest.approx(
        80.0, rel=1e-9,
    )
    assert setup.scale_report.scaled_height == pytest.approx(
        200.0, rel=1e-9,
    )
    assert setup.scale_report.scaled_xy_extent == pytest.approx(
        80.0, rel=1e-9,
    )


def test_inner_offset_report_thickness_matches_requested() -> None:
    setup = create_legacy_pet_water_trace_setup(
        _shell(), wall_thickness=0.5,
    )
    assert setup.wall_thickness == pytest.approx(0.5, rel=1e-9)
    assert setup.inner_offset_report.thickness == pytest.approx(
        0.5, rel=1e-9,
    )
    assert setup.inner_offset_report.inverted is True


def test_invalid_wall_thickness_raises() -> None:
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises((GeometryError, OpticsError)):
            create_legacy_pet_water_trace_setup(
                _shell(), wall_thickness=bad,
            )


def test_invalid_ior_raises_optics_error() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(OpticsError):
            create_legacy_pet_water_trace_setup(
                _shell(), ior_pet=bad,
            )
        with pytest.raises(OpticsError):
            create_legacy_pet_water_trace_setup(
                _shell(), ior_water=bad,
            )


def test_step_specs_carry_correct_eta_pairs() -> None:
    setup = create_legacy_pet_water_trace_setup(
        _shell(),
        ior_air=1.0,
        ior_pet=1.57,
        ior_water=1.333,
    )
    pairs = [(spec.eta_i, spec.eta_t) for spec in setup.step_specs]
    assert pairs[0] == (1.0, 1.57)
    assert pairs[1] == (1.57, 1.333)
    assert pairs[2] == (1.333, 1.57)
    assert pairs[3] == (1.57, 1.0)


def test_no_file_output(tmp_path) -> None:
    before = sorted(tmp_path.iterdir())
    create_legacy_pet_water_trace_setup(_shell())
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_input_mesh_not_mutated() -> None:
    mesh = _shell()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    create_legacy_pet_water_trace_setup(mesh)
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_default_inner_offset_mode_is_auto() -> None:
    setup = create_legacy_pet_water_trace_setup(_shell())
    assert setup.inner_offset_report.offset_mode == "auto"


def test_inner_offset_mode_minus_normals_passes_through() -> None:
    setup = create_legacy_pet_water_trace_setup(
        _shell(), inner_offset_mode="minus_normals",
    )
    assert setup.inner_offset_report.offset_mode == "minus_normals"
    assert setup.inner_offset_report.selected_offset_sign == pytest.approx(
        -1.0, abs=1e-12,
    )


def test_inner_offset_mode_plus_normals_passes_through() -> None:
    setup = create_legacy_pet_water_trace_setup(
        _shell(), inner_offset_mode="plus_normals",
    )
    assert setup.inner_offset_report.offset_mode == "plus_normals"
    assert setup.inner_offset_report.selected_offset_sign == pytest.approx(
        +1.0, abs=1e-12,
    )


def test_inner_offset_mode_invalid_raises() -> None:
    with pytest.raises((GeometryError, OpticsError)):
        create_legacy_pet_water_trace_setup(
            _shell(), inner_offset_mode="bogus",
        )


def test_water_mesh_radial_smaller_than_shell_for_outward_cylinder() -> None:
    cyl = trimesh.creation.cylinder(
        radius=30.0, height=120.0, sections=64,
    )
    setup = create_legacy_pet_water_trace_setup(
        cyl,
        target_height=120.0,
        target_diameter=60.0,
        wall_thickness=0.3,
        inner_offset_mode="auto",
    )
    assert setup.inner_offset_report.inward_offset_detected is True
    assert (
        setup.inner_offset_report.selected_radial_stat
        < setup.inner_offset_report.original_radial_stat
    )


def test_water_mesh_radial_smaller_for_inward_normal_mesh() -> None:
    cyl = trimesh.creation.cylinder(
        radius=30.0, height=120.0, sections=64,
    )
    cyl.invert()
    setup = create_legacy_pet_water_trace_setup(
        cyl,
        target_height=120.0,
        target_diameter=60.0,
        wall_thickness=0.3,
        inner_offset_mode="auto",
    )
    assert setup.inner_offset_report.inward_offset_detected is True
    assert setup.inner_offset_report.selected_offset_sign == pytest.approx(
        +1.0, abs=1e-12,
    )
