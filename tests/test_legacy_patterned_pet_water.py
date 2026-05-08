import numpy as np
import pytest

from optics_simulation.geometry import (
    ActualBottleBodyMaskReport,
    InnerOffsetMeshReport,
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.optics import (
    LegacyPatternedPetWaterSetup,
    LegacyPetWaterTraceSetup,
    MultiMeshTraceStepSpec,
    create_legacy_patterned_pet_water_setup,
)
from optics_simulation.pattern import (
    ActualSTLPatternedMeshResult,
    VertexPatternDisplacement,
)


def _setup():
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=48, height_segments=10,
    )
    return create_legacy_patterned_pet_water_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
        pattern_count=20,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=42,
        active_threshold=0.001,
    )


def test_returns_dataclass() -> None:
    setup = _setup()
    assert isinstance(setup, LegacyPatternedPetWaterSetup)
    assert isinstance(setup.original_setup, LegacyPetWaterTraceSetup)
    assert isinstance(setup.body_mask_report, ActualBottleBodyMaskReport)
    assert isinstance(setup.displacement_result, VertexPatternDisplacement)
    assert isinstance(
        setup.patterned_mesh_result, ActualSTLPatternedMeshResult,
    )
    assert isinstance(
        setup.patterned_inner_offset_report, InnerOffsetMeshReport,
    )


def test_original_step_specs_length_is_four() -> None:
    setup = _setup()
    assert len(setup.original_setup.step_specs) == 4


def test_patterned_step_specs_length_is_four() -> None:
    setup = _setup()
    assert len(setup.patterned_step_specs) == 4
    for spec in setup.patterned_step_specs:
        assert isinstance(spec, MultiMeshTraceStepSpec)


def test_body_mask_selected_count_positive() -> None:
    setup = _setup()
    assert int(setup.body_mask_report.selected_count) > 0


def test_displacement_active_or_moved_count_positive() -> None:
    setup = _setup()
    active_count = int(setup.displacement_result.active_mask.sum())
    moved_count = int(setup.patterned_mesh_result.moved_vertex_count)
    assert (active_count > 0) or (moved_count > 0)


def test_patterned_shell_distinct_from_original_shell() -> None:
    setup = _setup()
    assert (
        setup.patterned_shell_mesh
        is not setup.original_setup.shell_mesh
    )


def test_patterned_water_distinct_from_patterned_shell() -> None:
    setup = _setup()
    assert setup.patterned_water_mesh is not setup.patterned_shell_mesh


def test_patterned_inner_offset_inward_detected_true() -> None:
    setup = _setup()
    assert (
        setup.patterned_inner_offset_report.inward_offset_detected is True
    )


def test_no_file_output(tmp_path) -> None:
    before = sorted(tmp_path.iterdir())
    _setup()
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_input_mesh_not_mutated() -> None:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=48, height_segments=10,
    )
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    create_legacy_patterned_pet_water_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)
