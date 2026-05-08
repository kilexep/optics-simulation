import numpy as np
import pytest

from optics_simulation.contribution import RiskMap
from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.optics import (
    LegacyPatternedPetWaterSetup,
    LegacyPetWaterTraceSetup,
    MultiMeshTraceStepSpec,
    SubdividedLegacyPatternSetup,
    create_legacy_pet_water_trace_setup,
    create_legacy_pet_water_trace_setup_from_shell_mesh,
    create_subdivided_risk_guided_legacy_pattern_setup,
)
from optics_simulation.pattern import (
    PatternResolutionReadinessReport,
)


def _input_mesh():
    return create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=48, height_segments=10,
    )


def _uniform_risk_map(nv: int = 32, nu: int = 64):
    risk = np.ones((nv, nu), dtype=float)
    total = float(risk.sum())
    return RiskMap(
        risk_map=risk.copy(),
        probability_map=(risk / total).astype(float, copy=True),
        active_mask=np.ones((nv, nu), dtype=bool),
        total_risk=total,
        active_count=int(nv * nu),
        epsilon=0.0,
        threshold=None,
    )


def test_setup_from_shell_mesh_returns_setup() -> None:
    mesh = _input_mesh()
    base = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    setup = create_legacy_pet_water_trace_setup_from_shell_mesh(
        base.shell_mesh,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    assert isinstance(setup, LegacyPetWaterTraceSetup)


def test_setup_from_shell_mesh_has_four_step_specs() -> None:
    mesh = _input_mesh()
    base = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    setup = create_legacy_pet_water_trace_setup_from_shell_mesh(
        base.shell_mesh,
    )
    assert len(setup.step_specs) == 4
    for spec in setup.step_specs:
        assert isinstance(spec, MultiMeshTraceStepSpec)


def test_setup_from_shell_mesh_does_not_rescale() -> None:
    mesh = _input_mesh()
    base = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    setup = create_legacy_pet_water_trace_setup_from_shell_mesh(
        base.shell_mesh,
    )
    # shell_mesh is the same object (no scaling applied here).
    assert setup.shell_mesh is base.shell_mesh
    assert setup.scale_report.scale_xyz == (1.0, 1.0, 1.0)


def test_subdivided_risk_guided_returns_setup() -> None:
    mesh = _input_mesh()
    risk = _uniform_risk_map()
    setup = create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
        subdivision_iterations=1,
        risk_map=risk,
        pattern_count=10,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=1,
    )
    assert isinstance(setup, SubdividedLegacyPatternSetup)
    assert isinstance(
        setup.original_subdivided_setup, LegacyPetWaterTraceSetup,
    )
    assert isinstance(
        setup.patterned_subdivided_setup,
        LegacyPatternedPetWaterSetup,
    )
    assert isinstance(
        setup.original_readiness, PatternResolutionReadinessReport,
    )
    assert isinstance(
        setup.subdivided_readiness,
        PatternResolutionReadinessReport,
    )


def test_subdivided_shell_has_more_vertices_than_base() -> None:
    mesh = _input_mesh()
    risk = _uniform_risk_map()
    setup = create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        risk_map=risk,
        subdivision_iterations=1,
        pattern_count=5,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=2,
    )
    assert int(len(setup.subdivided_shell.vertices)) > int(
        len(setup.base_scaled_shell.vertices)
    )
    assert int(setup.subdivision_report.subdivided_vertex_count) > (
        int(setup.subdivision_report.original_vertex_count)
    )


def test_subdivided_readiness_selected_count_not_smaller_than_original() -> None:
    mesh = _input_mesh()
    risk = _uniform_risk_map()
    setup = create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        risk_map=risk,
        subdivision_iterations=1,
        pattern_count=5,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=3,
    )
    assert int(
        setup.subdivided_readiness.selected_vertex_count,
    ) >= int(setup.original_readiness.selected_vertex_count)


def test_body_mask_selected_count_positive() -> None:
    mesh = _input_mesh()
    risk = _uniform_risk_map()
    setup = create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        risk_map=risk,
        subdivision_iterations=1,
        pattern_count=5,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=4,
    )
    assert int(setup.body_mask_report.selected_count) > 0


def test_moved_vertex_count_positive() -> None:
    mesh = _input_mesh()
    risk = _uniform_risk_map()
    setup = create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        risk_map=risk,
        subdivision_iterations=1,
        pattern_count=20,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=5,
    )
    assert int(setup.moved_vertex_count) > 0


def test_step_specs_lengths_are_four() -> None:
    mesh = _input_mesh()
    risk = _uniform_risk_map()
    setup = create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        risk_map=risk,
        subdivision_iterations=1,
        pattern_count=5,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=6,
    )
    assert len(setup.original_subdivided_setup.step_specs) == 4
    assert len(
        setup.patterned_subdivided_setup.patterned_step_specs,
    ) == 4


def test_input_mesh_not_mutated() -> None:
    mesh = _input_mesh()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    risk = _uniform_risk_map()
    create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        risk_map=risk,
        subdivision_iterations=1,
        pattern_count=5,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=7,
    )
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_no_file_output(tmp_path) -> None:
    mesh = _input_mesh()
    risk = _uniform_risk_map()
    before = sorted(tmp_path.iterdir())
    create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        risk_map=risk,
        subdivision_iterations=1,
        pattern_count=5,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=8,
    )
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_invalid_subdivision_iterations_raises() -> None:
    mesh = _input_mesh()
    risk = _uniform_risk_map()
    with pytest.raises(Exception):
        create_subdivided_risk_guided_legacy_pattern_setup(
            mesh,
            risk_map=risk,
            subdivision_iterations=0,
            pattern_count=5,
            pattern_sigma_u=0.05, pattern_sigma_v=0.05,
            pattern_max_depth=0.05,
            pattern_seed=9,
        )
