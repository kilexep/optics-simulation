import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from optics_simulation.angle_scan import (
    AngleDistanceScanResult,
    PerAngleDistanceResult,
    run_angle_distance_sweep,
)
from optics_simulation.metrics import (
    DetectorIrradianceSurrogate,
    OpticalMetrics,
)
from optics_simulation.optics import create_detector_grid
from optics_simulation.thermal import (
    AngleDistanceHeatingScanResult,
    PerAngleDistanceHeatingResult,
    ThermalError,
    run_lumped_heating_over_angle_distance_scan,
)


_THERMAL_KWARGS = dict(
    nominal_incident_irradiance_w_m2=1000.0,
    duration_s=10.0,
    dt_s=0.5,
    areal_heat_capacity_j_m2k=1200.0,
    absorptivity=0.8,
    h_conv_w_m2k=10.0,
    emissivity=0.9,
    ambient_temp_k=293.15,
    threshold_temp_k=295.0,
)


def _build_metrics(c99: float = 0.0, cmax: float = 0.0) -> OpticalMetrics:
    return OpticalMetrics(
        peak_value=float(cmax),
        total_value=0.0,
        mean_value=0.0,
        cmax=float(cmax),
        c99=float(c99),
        top_percent=1.0,
        incident_reference=1.0,
        thresholds=(2.0,),
        eexceed={2.0: 0.0},
        ahot={2.0: 0},
        pixel_count=4,
    )


def _build_surrogate(
    relative_irradiance_max: float,
    *,
    detector_resolution: tuple[int, int] = (2, 2),
) -> DetectorIrradianceSurrogate:
    grid = create_detector_grid(
        width=10.0, height=10.0, resolution=detector_resolution,
    )
    rel_map = np.zeros(detector_resolution, dtype=float)
    rel_map[0, 0] = float(relative_irradiance_max)
    detector_power_map = rel_map * grid.pixel_width * grid.pixel_height
    return DetectorIrradianceSurrogate(
        detector_power_map=detector_power_map.copy(),
        relative_irradiance_map=rel_map.copy(),
        pixel_area=float(grid.pixel_width * grid.pixel_height),
        source_area=100.0,
        incident_irradiance=1.0,
        total_incident_power=100.0,
        incident_power_per_ray=1.0,
        total_detector_power=float(detector_power_map.sum()),
        ray_count=100,
        detector_resolution=detector_resolution,
    )


def _build_entry(
    *,
    angle: float,
    z: float,
    relative_max: float,
    c99: float = 0.0,
    cmax: float = 0.0,
    detector_hits: int = 5,
    final_ray_count: int = 5,
    ray_count: int = 100,
    termination_reason: str = "completed_interfaces",
    surrogate: DetectorIrradianceSurrogate | None = None,
) -> PerAngleDistanceResult:
    if surrogate is None and relative_max is not None:
        surrogate = _build_surrogate(relative_max)
    return PerAngleDistanceResult(
        angle_degrees=float(angle),
        detector_z=float(z),
        direction=np.array([0.0, 0.0, -1.0], dtype=float),
        ray_count=int(ray_count),
        final_ray_count=int(final_ray_count),
        detector_hits=int(detector_hits),
        metrics=_build_metrics(c99=c99, cmax=cmax),
        termination_reason=termination_reason,
        irradiance_surrogate=surrogate,
    )


def _build_scan(entries: list[PerAngleDistanceResult]) -> AngleDistanceScanResult:
    if not entries:
        return AngleDistanceScanResult(
            per_result=(),
            angle_count=0,
            detector_count=0,
            max_c99_angle=None,
            max_c99_detector_z=None,
            max_c99=None,
        )
    angles = sorted({p.angle_degrees for p in entries})
    distances = sorted({p.detector_z for p in entries}, reverse=True)
    c99_arr = np.array([p.metrics.c99 for p in entries], dtype=float)
    idx = int(np.argmax(c99_arr))
    return AngleDistanceScanResult(
        per_result=tuple(entries),
        angle_count=len(angles),
        detector_count=len(distances),
        max_c99_angle=float(entries[idx].angle_degrees),
        max_c99_detector_z=float(entries[idx].detector_z),
        max_c99=float(entries[idx].metrics.c99),
    )


def test_returns_angle_distance_heating_scan_result() -> None:
    scan = _build_scan([
        _build_entry(angle=0.0, z=-100.0, relative_max=1.0),
    ])
    out = run_lumped_heating_over_angle_distance_scan(scan, **_THERMAL_KWARGS)
    assert isinstance(out, AngleDistanceHeatingScanResult)
    assert isinstance(out.per_result[0], PerAngleDistanceHeatingResult)
    assert out.scalar_mode == "max_relative_irradiance"


def test_empty_scan_returns_empty_thermal_result() -> None:
    out = run_lumped_heating_over_angle_distance_scan(
        _build_scan([]), **_THERMAL_KWARGS,
    )
    assert out.result_count == 0
    assert out.per_result == ()
    assert out.max_temperature_angle is None
    assert out.max_temperature_detector_z is None
    assert out.max_temperature_k is None
    assert out.max_temperature_rise_k is None
    assert out.max_incident_flux_w_m2 is None


def test_requires_angle_distance_scan_result_type() -> None:
    with pytest.raises(ThermalError, match="AngleDistanceScanResult"):
        run_lumped_heating_over_angle_distance_scan(
            object(),  # type: ignore[arg-type]
            **_THERMAL_KWARGS,
        )


def test_requires_irradiance_surrogate_on_nonempty_entries() -> None:
    bad_entry = _build_entry(angle=0.0, z=-100.0, relative_max=None)
    bad_entry = PerAngleDistanceResult(
        angle_degrees=bad_entry.angle_degrees,
        detector_z=bad_entry.detector_z,
        direction=bad_entry.direction,
        ray_count=bad_entry.ray_count,
        final_ray_count=bad_entry.final_ray_count,
        detector_hits=bad_entry.detector_hits,
        metrics=bad_entry.metrics,
        termination_reason=bad_entry.termination_reason,
        irradiance_surrogate=None,
    )
    with pytest.raises(ThermalError, match="irradiance_surrogate"):
        run_lumped_heating_over_angle_distance_scan(
            _build_scan([bad_entry]),
            **_THERMAL_KWARGS,
        )


def test_invalid_scalar_mode_raises() -> None:
    scan = _build_scan([
        _build_entry(angle=0.0, z=-100.0, relative_max=1.0),
    ])
    with pytest.raises(ThermalError, match="scalar_mode"):
        run_lumped_heating_over_angle_distance_scan(
            scan, scalar_mode="bogus", **_THERMAL_KWARGS,
        )


def test_invalid_nominal_incident_irradiance_raises() -> None:
    scan = _build_scan([
        _build_entry(angle=0.0, z=-100.0, relative_max=1.0),
    ])
    bad_kwargs = dict(_THERMAL_KWARGS)
    bad_kwargs["nominal_incident_irradiance_w_m2"] = -1.0
    with pytest.raises(
        ThermalError, match="nominal_incident_irradiance_w_m2"
    ):
        run_lumped_heating_over_angle_distance_scan(scan, **bad_kwargs)


def test_max_relative_irradiance_extracted_from_relative_map() -> None:
    scan = _build_scan([
        _build_entry(angle=0.0, z=-100.0, relative_max=2.5),
        _build_entry(angle=10.0, z=-100.0, relative_max=4.0),
    ])
    out = run_lumped_heating_over_angle_distance_scan(scan, **_THERMAL_KWARGS)
    assert out.per_result[0].max_relative_irradiance == pytest.approx(2.5)
    assert out.per_result[1].max_relative_irradiance == pytest.approx(4.0)


def test_incident_flux_equals_nominal_times_max_relative() -> None:
    scan = _build_scan([
        _build_entry(angle=0.0, z=-100.0, relative_max=2.5),
    ])
    out = run_lumped_heating_over_angle_distance_scan(
        scan, **_THERMAL_KWARGS,
    )
    expected = (
        _THERMAL_KWARGS["nominal_incident_irradiance_w_m2"] * 2.5
    )
    assert out.per_result[0].incident_flux_w_m2 == pytest.approx(expected)


def test_per_result_ordering_follows_scan_ordering() -> None:
    entries = [
        _build_entry(angle=0.0, z=-80.0, relative_max=1.0),
        _build_entry(angle=0.0, z=-100.0, relative_max=2.0),
        _build_entry(angle=10.0, z=-80.0, relative_max=3.0),
        _build_entry(angle=10.0, z=-100.0, relative_max=4.0),
    ]
    out = run_lumped_heating_over_angle_distance_scan(
        _build_scan(entries), **_THERMAL_KWARGS,
    )
    pairs = [(p.angle_degrees, p.detector_z) for p in out.per_result]
    assert pairs == [
        (0.0, -80.0), (0.0, -100.0), (10.0, -80.0), (10.0, -100.0),
    ]


def test_max_temperature_fields_match_argmax() -> None:
    entries = [
        _build_entry(angle=0.0, z=-80.0, relative_max=1.0),
        _build_entry(angle=0.0, z=-100.0, relative_max=2.0),
        _build_entry(angle=10.0, z=-80.0, relative_max=4.0),
        _build_entry(angle=10.0, z=-100.0, relative_max=3.0),
    ]
    out = run_lumped_heating_over_angle_distance_scan(
        _build_scan(entries), **_THERMAL_KWARGS,
    )
    max_temps = np.array(
        [p.heating_result.max_temperature_k for p in out.per_result],
        dtype=float,
    )
    idx = int(np.argmax(max_temps))
    chosen = out.per_result[idx]
    assert out.max_temperature_k == pytest.approx(
        float(chosen.heating_result.max_temperature_k)
    )
    assert out.max_temperature_angle == chosen.angle_degrees
    assert out.max_temperature_detector_z == chosen.detector_z
    assert out.max_incident_flux_w_m2 == pytest.approx(
        chosen.incident_flux_w_m2
    )


def test_threshold_crossing_preserved() -> None:
    # High flux fixture + low threshold = crossing in both entries.
    scan = _build_scan([
        _build_entry(angle=0.0, z=-100.0, relative_max=10.0),
    ])
    kw = dict(_THERMAL_KWARGS)
    kw["threshold_temp_k"] = 295.0
    kw["duration_s"] = 30.0
    out = run_lumped_heating_over_angle_distance_scan(scan, **kw)
    h = out.per_result[0].heating_result
    assert h.time_to_threshold_s is not None
    assert 0.0 <= h.time_to_threshold_s <= 30.0


def test_optical_fields_copied_from_entries() -> None:
    entry = _build_entry(
        angle=15.0, z=-120.0, relative_max=2.0,
        c99=0.5, cmax=0.7,
        detector_hits=42, final_ray_count=60,
        termination_reason="no_active_rays",
    )
    out = run_lumped_heating_over_angle_distance_scan(
        _build_scan([entry]), **_THERMAL_KWARGS,
    )
    pr = out.per_result[0]
    assert pr.optical_c99 == pytest.approx(0.5)
    assert pr.optical_cmax == pytest.approx(0.7)
    assert pr.detector_hits == 42
    assert pr.final_ray_count == 60
    assert pr.termination_reason == "no_active_rays"


def test_temperature_arrays_finite() -> None:
    entries = [
        _build_entry(angle=0.0, z=-80.0, relative_max=1.0),
        _build_entry(angle=10.0, z=-80.0, relative_max=2.5),
    ]
    out = run_lumped_heating_over_angle_distance_scan(
        _build_scan(entries), **_THERMAL_KWARGS,
    )
    for p in out.per_result:
        assert np.isfinite(p.heating_result.temperature_k).all()


def test_no_file_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    scan = _build_scan([
        _build_entry(angle=0.0, z=-100.0, relative_max=1.0),
    ])
    before = set(os.listdir(tmp_path))
    run_lumped_heating_over_angle_distance_scan(scan, **_THERMAL_KWARGS)
    after = set(os.listdir(tmp_path))
    assert before == after


def test_scan_result_is_not_mutated() -> None:
    entry = _build_entry(angle=0.0, z=-100.0, relative_max=2.5)
    scan = _build_scan([entry])
    snapshot_map = (
        scan.per_result[0].irradiance_surrogate
        .relative_irradiance_map.copy()
    )
    snapshot_max_c99 = scan.max_c99
    run_lumped_heating_over_angle_distance_scan(scan, **_THERMAL_KWARGS)
    np.testing.assert_array_equal(
        scan.per_result[0].irradiance_surrogate.relative_irradiance_map,
        snapshot_map,
    )
    assert scan.max_c99 == snapshot_max_c99


def test_integration_smoke_with_real_angle_distance_sweep() -> None:
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    sweep = run_angle_distance_sweep(
        mesh=mesh,
        angles_degrees=[0.0, 10.0],
        detector_z_values=[-15.0, -25.0],
        ray_grid_config={
            "origin_plane_z": 20.0,
            "x_range": (-7.5, 7.5),
            "y_range": (-7.5, 7.5),
            "nx": 7,
            "ny": 7,
        },
        interface_sequence=[(1.00028, 1.575), (1.575, 1.00028)],
        detector_width=10.0,
        detector_height=10.0,
        detector_resolution=(10, 10),
        thresholds=(2.0, 5.0, 10.0),
        use_power_weights=True,
        use_relative_irradiance=True,
    )
    out = run_lumped_heating_over_angle_distance_scan(
        sweep, **_THERMAL_KWARGS,
    )
    assert out.result_count == len(sweep.per_result)
    for sweep_entry, heat_entry in zip(sweep.per_result, out.per_result):
        assert heat_entry.angle_degrees == sweep_entry.angle_degrees
        assert heat_entry.detector_z == sweep_entry.detector_z
        assert heat_entry.detector_hits == sweep_entry.detector_hits
        assert heat_entry.final_ray_count == sweep_entry.final_ray_count
        assert np.isfinite(
            heat_entry.heating_result.temperature_k
        ).all()
