import numpy as np
import pytest

from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.optics import (
    LegacyOpticalScanEntry,
    LegacyOpticalScanResult,
    LegacyParityScanSummary,
    LegacyParitySchedule,
    OpticsError,
    RayBundle,
    create_legacy_experiment_schedule,
    create_legacy_pet_water_trace_setup,
    create_oriented_parallel_ray_grid,
    run_legacy_pet_water_angle_distance_scan,
    summarize_legacy_optical_scan,
)


def _setup():
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=32, height_segments=8,
    )
    return create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6,
        target_diameter=72.1,
        wall_thickness=0.3,
        inner_offset_mode="auto",
    )


def test_create_oriented_parallel_ray_grid_returns_ray_bundle() -> None:
    rays = create_oriented_parallel_ray_grid(
        center=(100.0, 0.0, 0.0),
        direction=(-1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        width=50.0, height=80.0,
        sample_count_y=4, sample_count_z=5,
    )
    assert isinstance(rays, RayBundle)


def test_oriented_grid_ray_count_matches_sample_product() -> None:
    rays = create_oriented_parallel_ray_grid(
        center=(100.0, 0.0, 0.0),
        direction=(-1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        width=50.0, height=80.0,
        sample_count_y=4, sample_count_z=5,
    )
    assert int(rays.ray_count) == 4 * 5


def test_oriented_grid_directions_are_unit_vectors() -> None:
    rays = create_oriented_parallel_ray_grid(
        center=(100.0, 0.0, 0.0),
        direction=(-2.0, 0.0, 0.0),  # non-unit, will be normalized
        up=(0.0, 0.0, 1.0),
        width=50.0, height=80.0,
        sample_count_y=3, sample_count_z=3,
    )
    norms = np.linalg.norm(rays.directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)
    # All directions are equal (parallel grid).
    for d in rays.directions:
        np.testing.assert_allclose(d, [-1.0, 0.0, 0.0], atol=1e-12)


def test_oriented_grid_origins_span_width_and_height() -> None:
    rays = create_oriented_parallel_ray_grid(
        center=(100.0, 0.0, 0.0),
        direction=(-1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        width=50.0, height=80.0,
        sample_count_y=4, sample_count_z=5,
    )
    # right = cross(up, direction) = cross(+z, -x) = -y axis.
    # So along right axis (right = -y), spans [-25, 25] in -y direction.
    y_min = float(rays.origins[:, 1].min())
    y_max = float(rays.origins[:, 1].max())
    z_min = float(rays.origins[:, 2].min())
    z_max = float(rays.origins[:, 2].max())
    assert y_max - y_min == pytest.approx(50.0, abs=1e-9)
    assert z_max - z_min == pytest.approx(80.0, abs=1e-9)
    # Centered at x=100, y=0, z=0
    np.testing.assert_allclose(rays.origins[:, 0], 100.0, atol=1e-12)


def test_oriented_grid_up_parallel_to_direction_raises() -> None:
    with pytest.raises(OpticsError, match="parallel"):
        create_oriented_parallel_ray_grid(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            width=10.0, height=10.0,
            sample_count_y=2, sample_count_z=2,
        )


def test_oriented_grid_invalid_width_height_raises() -> None:
    for bad_w in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(OpticsError):
            create_oriented_parallel_ray_grid(
                center=(0.0, 0.0, 0.0),
                direction=(-1.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                width=bad_w, height=10.0,
                sample_count_y=2, sample_count_z=2,
            )
    for bad_h in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(OpticsError):
            create_oriented_parallel_ray_grid(
                center=(0.0, 0.0, 0.0),
                direction=(-1.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                width=10.0, height=bad_h,
                sample_count_y=2, sample_count_z=2,
            )


def test_oriented_grid_invalid_sample_counts_raise() -> None:
    for bad in (0, -1, -5):
        with pytest.raises(OpticsError, match="sample_count"):
            create_oriented_parallel_ray_grid(
                center=(0.0, 0.0, 0.0),
                direction=(-1.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                width=10.0, height=10.0,
                sample_count_y=bad, sample_count_z=2,
            )
        with pytest.raises(OpticsError, match="sample_count"):
            create_oriented_parallel_ray_grid(
                center=(0.0, 0.0, 0.0),
                direction=(-1.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                width=10.0, height=10.0,
                sample_count_y=2, sample_count_z=bad,
            )


def test_run_legacy_scan_returns_result_dataclass() -> None:
    setup = _setup()
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0,),
        detector_distances=(120.0,),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    assert isinstance(result, LegacyOpticalScanResult)
    for entry in result.entries:
        assert isinstance(entry, LegacyOpticalScanEntry)


def test_empty_angles_returns_empty_result() -> None:
    setup = _setup()
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(),
        detector_distances=(120.0, 200.0),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    assert result.entries == ()
    assert result.angle_count == 0
    assert result.detector_count == 2
    assert result.max_relative_irradiance is None
    assert result.max_c99 is None


def test_empty_distances_returns_empty_result() -> None:
    setup = _setup()
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0, 15.0),
        detector_distances=(),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    assert result.entries == ()
    assert result.angle_count == 2
    assert result.detector_count == 0
    assert result.max_relative_irradiance is None
    assert result.max_c99 is None


def test_result_count_equals_angles_times_distances() -> None:
    setup = _setup()
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0, 15.0, 30.0),
        detector_distances=(120.0, 200.0),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    assert len(result.entries) == 3 * 2
    assert result.angle_count == 3
    assert result.detector_count == 2


def test_entries_are_angle_major() -> None:
    setup = _setup()
    angles = (0.0, 15.0, 30.0)
    distances = (120.0, 200.0)
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=angles,
        detector_distances=distances,
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    expected_pairs = [
        (a, d) for a in angles for d in distances
    ]
    actual_pairs = [
        (e.angle_degrees, e.detector_distance)
        for e in result.entries
    ]
    assert actual_pairs == expected_pairs


def test_each_entry_c99_and_cmax_nonnegative() -> None:
    setup = _setup()
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0, 15.0),
        detector_distances=(120.0, 200.0),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    for entry in result.entries:
        assert entry.c99 >= 0.0
        assert entry.cmax >= 0.0
        assert entry.max_relative_irradiance >= 0.0
        assert entry.detector_hits >= 0
        assert entry.final_ray_count >= 0


def test_max_relative_irradiance_aggregate_matches_manual_argmax() -> None:
    setup = _setup()
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0, 15.0, 30.0),
        detector_distances=(120.0, 200.0),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    expected_idx = max(
        range(len(result.entries)),
        key=lambda i: result.entries[i].max_relative_irradiance,
    )
    expected_entry = result.entries[expected_idx]
    assert result.max_relative_irradiance == pytest.approx(
        expected_entry.max_relative_irradiance, abs=1e-12,
    )
    assert result.max_relative_irradiance_angle == pytest.approx(
        expected_entry.angle_degrees, abs=1e-12,
    )
    assert (
        result.max_relative_irradiance_detector_distance
        == pytest.approx(expected_entry.detector_distance, abs=1e-12)
    )


def test_max_c99_aggregate_matches_manual_argmax() -> None:
    setup = _setup()
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0, 15.0, 30.0),
        detector_distances=(120.0, 200.0),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    expected_idx = max(
        range(len(result.entries)),
        key=lambda i: result.entries[i].c99,
    )
    expected_entry = result.entries[expected_idx]
    assert result.max_c99 == pytest.approx(
        expected_entry.c99, abs=1e-12,
    )
    assert result.max_c99_angle == pytest.approx(
        expected_entry.angle_degrees, abs=1e-12,
    )
    assert result.max_c99_detector_distance == pytest.approx(
        expected_entry.detector_distance, abs=1e-12,
    )


def test_setup_meshes_not_mutated_by_scan() -> None:
    setup = _setup()
    shell_v_before = np.array(setup.shell_mesh.vertices, copy=True)
    shell_f_before = np.array(setup.shell_mesh.faces, copy=True)
    water_v_before = np.array(setup.water_mesh.vertices, copy=True)
    water_f_before = np.array(setup.water_mesh.faces, copy=True)

    run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0, 15.0),
        detector_distances=(120.0, 200.0),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    assert np.array_equal(
        np.asarray(setup.shell_mesh.vertices), shell_v_before,
    )
    assert np.array_equal(
        np.asarray(setup.shell_mesh.faces), shell_f_before,
    )
    assert np.array_equal(
        np.asarray(setup.water_mesh.vertices), water_v_before,
    )
    assert np.array_equal(
        np.asarray(setup.water_mesh.faces), water_f_before,
    )


def test_scan_does_not_write_files(tmp_path) -> None:
    setup = _setup()
    before = sorted(tmp_path.iterdir())
    run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0, 15.0),
        detector_distances=(120.0, 200.0),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    after = sorted(tmp_path.iterdir())
    assert after == before


def test_invalid_setup_type_raises() -> None:
    with pytest.raises(OpticsError, match="LegacyPetWaterTraceSetup"):
        run_legacy_pet_water_angle_distance_scan(
            setup="not a setup",  # type: ignore[arg-type]
            angles_degrees=(0.0,),
            detector_distances=(120.0,),
            source_width=80.0, source_height=240.0,
            sample_count_y=4, sample_count_z=4,
            detector_size=400.0, detector_resolution=(20, 20),
        )


def test_invalid_detector_distance_raises() -> None:
    setup = _setup()
    with pytest.raises(OpticsError, match="detector_distances"):
        run_legacy_pet_water_angle_distance_scan(
            setup=setup,
            angles_degrees=(0.0,),
            detector_distances=(0.0,),
            source_width=80.0, source_height=240.0,
            sample_count_y=4, sample_count_z=4,
            detector_size=400.0, detector_resolution=(20, 20),
        )
    with pytest.raises(OpticsError, match="detector_distances"):
        run_legacy_pet_water_angle_distance_scan(
            setup=setup,
            angles_degrees=(0.0,),
            detector_distances=(float("nan"),),
            source_width=80.0, source_height=240.0,
            sample_count_y=4, sample_count_z=4,
            detector_size=400.0, detector_resolution=(20, 20),
        )


# ---------------------------------------------------------------------------
# Legacy parity schedule and summary
# ---------------------------------------------------------------------------


def _empty_result() -> LegacyOpticalScanResult:
    return LegacyOpticalScanResult(
        entries=(),
        angle_count=0,
        detector_count=0,
        max_relative_irradiance_angle=None,
        max_relative_irradiance_detector_distance=None,
        max_relative_irradiance=None,
        max_c99_angle=None,
        max_c99_detector_distance=None,
        max_c99=None,
    )


def test_create_legacy_experiment_schedule_returns_dataclass() -> None:
    schedule = create_legacy_experiment_schedule()
    assert isinstance(schedule, LegacyParitySchedule)


def test_default_schedule_has_72_angles_and_20_distances() -> None:
    schedule = create_legacy_experiment_schedule()
    assert schedule.angle_count == 72
    assert schedule.detector_count == 20
    assert len(schedule.angles_degrees) == 72
    assert len(schedule.detector_distances) == 20


def test_default_schedule_first_two_angles_are_0_and_5() -> None:
    schedule = create_legacy_experiment_schedule()
    assert schedule.angles_degrees[0] == pytest.approx(0.0, abs=1e-12)
    assert schedule.angles_degrees[1] == pytest.approx(5.0, abs=1e-12)


def test_default_schedule_first_distance_100_last_distance_480() -> None:
    schedule = create_legacy_experiment_schedule()
    assert schedule.detector_distances[0] == pytest.approx(
        100.0, abs=1e-12,
    )
    assert schedule.detector_distances[-1] == pytest.approx(
        480.0, abs=1e-12,
    )


def test_invalid_angle_count_raises() -> None:
    for bad in (0, -1, -10):
        with pytest.raises(OpticsError, match="angle_count"):
            create_legacy_experiment_schedule(angle_count=bad)


def test_invalid_detector_count_raises() -> None:
    for bad in (0, -1, -10):
        with pytest.raises(OpticsError, match="detector_count"):
            create_legacy_experiment_schedule(detector_count=bad)


def test_invalid_detector_spacing_raises() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(OpticsError, match="detector_spacing"):
            create_legacy_experiment_schedule(detector_spacing=bad)


def test_invalid_sample_counts_raise() -> None:
    for bad in (0, -1):
        with pytest.raises(OpticsError, match="sample_count_y"):
            create_legacy_experiment_schedule(sample_count_y=bad)
        with pytest.raises(OpticsError, match="sample_count_z"):
            create_legacy_experiment_schedule(sample_count_z=bad)


def test_summarize_legacy_optical_scan_returns_summary() -> None:
    setup = _setup()
    schedule = create_legacy_experiment_schedule(
        angle_count=2, angle_step_degrees=15.0,
        detector_count=2, detector_start=120.0, detector_spacing=40.0,
        sample_count_y=4, sample_count_z=4,
    )
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=schedule.angles_degrees,
        detector_distances=schedule.detector_distances,
        source_width=schedule.source_width,
        source_height=schedule.source_height,
        sample_count_y=schedule.sample_count_y,
        sample_count_z=schedule.sample_count_z,
        detector_size=schedule.detector_size,
        detector_resolution=(20, 20),
        source_radius=schedule.source_radius,
    )
    summary = summarize_legacy_optical_scan(result, schedule)
    assert isinstance(summary, LegacyParityScanSummary)
    assert (
        summary.summary_type == "legacy_experiment_parity_scan_summary"
    )
    assert summary.result is result
    assert summary.schedule is schedule


def test_summary_total_entries_equals_result_entry_count() -> None:
    setup = _setup()
    schedule = create_legacy_experiment_schedule(
        angle_count=2, angle_step_degrees=15.0,
        detector_count=3, detector_start=120.0, detector_spacing=40.0,
        sample_count_y=4, sample_count_z=4,
    )
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=schedule.angles_degrees,
        detector_distances=schedule.detector_distances,
        source_width=schedule.source_width,
        source_height=schedule.source_height,
        sample_count_y=schedule.sample_count_y,
        sample_count_z=schedule.sample_count_z,
        detector_size=schedule.detector_size,
        detector_resolution=(20, 20),
        source_radius=schedule.source_radius,
    )
    summary = summarize_legacy_optical_scan(result, schedule)
    assert summary.total_entries == len(result.entries)
    assert summary.total_entries == 2 * 3


def test_summary_total_detector_hits_equals_manual_sum() -> None:
    setup = _setup()
    schedule = create_legacy_experiment_schedule(
        angle_count=2, angle_step_degrees=15.0,
        detector_count=2, detector_start=120.0, detector_spacing=40.0,
        sample_count_y=4, sample_count_z=4,
    )
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=schedule.angles_degrees,
        detector_distances=schedule.detector_distances,
        source_width=schedule.source_width,
        source_height=schedule.source_height,
        sample_count_y=schedule.sample_count_y,
        sample_count_z=schedule.sample_count_z,
        detector_size=schedule.detector_size,
        detector_resolution=(20, 20),
        source_radius=schedule.source_radius,
    )
    summary = summarize_legacy_optical_scan(result, schedule)
    expected_total = sum(
        int(e.detector_hits) for e in result.entries
    )
    assert summary.total_detector_hits == expected_total
    expected_nonzero = sum(
        1 for e in result.entries if int(e.detector_hits) > 0
    )
    assert summary.nonzero_entry_count == expected_nonzero
    assert (
        summary.zero_hit_entry_count
        == summary.total_entries - expected_nonzero
    )


def test_summary_max_detector_hits_matches_manual_argmax() -> None:
    setup = _setup()
    schedule = create_legacy_experiment_schedule(
        angle_count=3, angle_step_degrees=15.0,
        detector_count=2, detector_start=120.0, detector_spacing=40.0,
        sample_count_y=4, sample_count_z=4,
    )
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=schedule.angles_degrees,
        detector_distances=schedule.detector_distances,
        source_width=schedule.source_width,
        source_height=schedule.source_height,
        sample_count_y=schedule.sample_count_y,
        sample_count_z=schedule.sample_count_z,
        detector_size=schedule.detector_size,
        detector_resolution=(20, 20),
        source_radius=schedule.source_radius,
    )
    summary = summarize_legacy_optical_scan(result, schedule)
    expected_idx = max(
        range(len(result.entries)),
        key=lambda i: int(result.entries[i].detector_hits),
    )
    expected_entry = result.entries[expected_idx]
    assert summary.max_detector_hits == int(
        expected_entry.detector_hits,
    )
    assert summary.max_detector_hits_angle == pytest.approx(
        expected_entry.angle_degrees, abs=1e-12,
    )
    assert summary.max_detector_hits_distance == pytest.approx(
        expected_entry.detector_distance, abs=1e-12,
    )


def test_empty_result_summary_works() -> None:
    schedule = create_legacy_experiment_schedule(
        angle_count=1, detector_count=1,
    )
    summary = summarize_legacy_optical_scan(_empty_result(), schedule)
    assert summary.total_entries == 0
    assert summary.total_detector_hits == 0
    assert summary.nonzero_entry_count == 0
    assert summary.zero_hit_entry_count == 0
    assert summary.max_detector_hits == 0
    assert summary.max_detector_hits_angle is None
    assert summary.max_detector_hits_distance is None
    assert summary.max_relative_irradiance is None
    assert summary.max_c99 is None


def test_source_radius_kwarg_changes_origins_but_not_shape() -> None:
    setup = _setup()
    common = dict(
        setup=setup,
        angles_degrees=(0.0,),
        detector_distances=(120.0,),
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
    )
    base = run_legacy_pet_water_angle_distance_scan(
        source_radius=200.0, **common,
    )
    far = run_legacy_pet_water_angle_distance_scan(
        source_radius=400.0, **common,
    )
    assert len(base.entries) == len(far.entries) == 1
    assert base.entries[0].ray_count == far.entries[0].ray_count
    # final_ray_count >= 0 in both cases
    assert base.entries[0].final_ray_count >= 0
    assert far.entries[0].final_ray_count >= 0


def test_summary_no_file_output(tmp_path) -> None:
    setup = _setup()
    schedule = create_legacy_experiment_schedule(
        angle_count=1, detector_count=1,
        sample_count_y=4, sample_count_z=4,
    )
    before = sorted(tmp_path.iterdir())
    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=schedule.angles_degrees,
        detector_distances=schedule.detector_distances,
        source_width=schedule.source_width,
        source_height=schedule.source_height,
        sample_count_y=schedule.sample_count_y,
        sample_count_z=schedule.sample_count_z,
        detector_size=schedule.detector_size,
        detector_resolution=(20, 20),
        source_radius=schedule.source_radius,
    )
    summarize_legacy_optical_scan(result, schedule)
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_summarize_invalid_inputs_raise() -> None:
    schedule = create_legacy_experiment_schedule(
        angle_count=1, detector_count=1,
    )
    with pytest.raises(OpticsError, match="LegacyOpticalScanResult"):
        summarize_legacy_optical_scan("nope", schedule)  # type: ignore[arg-type]
    with pytest.raises(OpticsError, match="LegacyParitySchedule"):
        summarize_legacy_optical_scan(_empty_result(), "nope")  # type: ignore[arg-type]
