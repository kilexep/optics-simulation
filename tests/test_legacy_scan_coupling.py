import numpy as np
import pytest

from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.optics import (
    LegacyOpticalScanEntry,
    LegacyOpticalScanResult,
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.thermal import (
    LegacyThermalRiskEntry,
    LegacyThermalRiskScanResult,
    ThermalError,
    ThermalRiskMetrics,
    compute_thermal_risk_over_legacy_optical_scan,
)


def _setup_optical_result(
    *,
    store_irradiance_surrogate: bool = True,
    angles=(0.0, 15.0),
    distances=(120.0, 200.0),
) -> LegacyOpticalScanResult:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=32, height_segments=8,
    )
    setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    return run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=angles,
        detector_distances=distances,
        source_width=80.0, source_height=240.0,
        sample_count_y=4, sample_count_z=4,
        detector_size=400.0, detector_resolution=(20, 20),
        store_irradiance_surrogate=store_irradiance_surrogate,
    )


_THERMAL_KWARGS = dict(
    nominal_incident_irradiance_w_m2=1000.0,
    duration_s=10.0,
    dt_s=1.0,
    areal_heat_capacity_j_m2k=1200.0,
    absorptivity=0.8,
    h_conv_w_m2k=10.0,
    emissivity=0.9,
    ambient_temp_k=293.15,
    threshold_temp_k=373.15,
    top_percent=1.0,
)


def test_compute_thermal_risk_returns_dataclass() -> None:
    optical = _setup_optical_result()
    result = compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    assert isinstance(result, LegacyThermalRiskScanResult)
    assert result.result_type == "legacy_stl_thermal_risk_scan"


def test_empty_optical_result_returns_empty_thermal_result() -> None:
    empty = LegacyOpticalScanResult(
        entries=(),
        angle_count=0, detector_count=0,
        max_relative_irradiance_angle=None,
        max_relative_irradiance_detector_distance=None,
        max_relative_irradiance=None,
        max_c99_angle=None,
        max_c99_detector_distance=None,
        max_c99=None,
    )
    result = compute_thermal_risk_over_legacy_optical_scan(
        empty, **_THERMAL_KWARGS,
    )
    assert result.entries == ()
    assert result.entry_count == 0
    assert result.max_temperature_k is None
    assert result.max_temperature_angle is None
    assert result.max_threshold_exceeded_count is None


def test_missing_irradiance_surrogate_raises_thermal_error() -> None:
    optical = _setup_optical_result(store_irradiance_surrogate=False)
    with pytest.raises(ThermalError, match="irradiance_surrogate"):
        compute_thermal_risk_over_legacy_optical_scan(
            optical, **_THERMAL_KWARGS,
        )


def test_entry_count_matches_optical_entries() -> None:
    optical = _setup_optical_result()
    result = compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    assert result.entry_count == len(optical.entries)
    assert len(result.entries) == len(optical.entries)


def test_ordering_is_preserved() -> None:
    optical = _setup_optical_result(
        angles=(0.0, 15.0, 30.0), distances=(120.0, 200.0),
    )
    result = compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    optical_pairs = [
        (e.angle_degrees, e.detector_distance) for e in optical.entries
    ]
    thermal_pairs = [
        (e.angle_degrees, e.detector_distance) for e in result.entries
    ]
    assert thermal_pairs == optical_pairs


def test_incident_flux_map_max_equals_nominal_times_max_rel_irradiance() -> None:
    optical = _setup_optical_result()
    result = compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    nominal = float(_THERMAL_KWARGS["nominal_incident_irradiance_w_m2"])
    for thermal_entry, optical_entry in zip(
        result.entries, optical.entries,
    ):
        assert isinstance(thermal_entry, LegacyThermalRiskEntry)
        assert thermal_entry.incident_flux_map_max_w_m2 == pytest.approx(
            nominal * float(optical_entry.max_relative_irradiance),
            abs=1e-9,
        )


def test_thermal_metrics_are_thermal_risk_metrics() -> None:
    optical = _setup_optical_result()
    result = compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    for entry in result.entries:
        assert isinstance(entry.thermal_metrics, ThermalRiskMetrics)
        assert np.isfinite(entry.thermal_metrics.max_temperature_k)
        assert np.isfinite(
            entry.thermal_metrics.top_percent_max_temperature_rise_k
        )


def test_max_temperature_aggregate_matches_manual_argmax() -> None:
    optical = _setup_optical_result(
        angles=(0.0, 15.0, 30.0), distances=(120.0, 200.0, 280.0),
    )
    result = compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    expected_idx = max(
        range(len(result.entries)),
        key=lambda i: float(
            result.entries[i].thermal_metrics.max_temperature_k
        ),
    )
    expected_entry = result.entries[expected_idx]
    assert result.max_temperature_k == pytest.approx(
        float(expected_entry.thermal_metrics.max_temperature_k),
        abs=1e-12,
    )
    assert result.max_temperature_angle == pytest.approx(
        float(expected_entry.angle_degrees), abs=1e-12,
    )
    assert result.max_temperature_distance == pytest.approx(
        float(expected_entry.detector_distance), abs=1e-12,
    )


def test_max_top_percent_temperature_rise_aggregate_matches() -> None:
    optical = _setup_optical_result(
        angles=(0.0, 15.0, 30.0), distances=(120.0, 200.0),
    )
    result = compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    expected_idx = max(
        range(len(result.entries)),
        key=lambda i: float(
            result.entries[i].thermal_metrics
            .top_percent_max_temperature_rise_k
        ),
    )
    expected_entry = result.entries[expected_idx]
    assert (
        result.max_top_percent_temperature_rise_k
        == pytest.approx(
            float(
                expected_entry.thermal_metrics
                .top_percent_max_temperature_rise_k
            ),
            abs=1e-12,
        )
    )
    assert (
        result.max_top_percent_temperature_rise_angle
        == pytest.approx(float(expected_entry.angle_degrees), abs=1e-12)
    )
    assert (
        result.max_top_percent_temperature_rise_distance
        == pytest.approx(
            float(expected_entry.detector_distance), abs=1e-12,
        )
    )


def test_max_threshold_exceeded_count_aggregate_matches() -> None:
    optical = _setup_optical_result(
        angles=(0.0, 15.0, 30.0), distances=(120.0, 200.0),
    )
    result = compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    expected_idx = max(
        range(len(result.entries)),
        key=lambda i: int(
            result.entries[i].thermal_metrics.threshold_exceeded_count
        ),
    )
    expected_entry = result.entries[expected_idx]
    assert result.max_threshold_exceeded_count == int(
        expected_entry.thermal_metrics.threshold_exceeded_count,
    )
    assert result.max_threshold_exceeded_count_angle == pytest.approx(
        float(expected_entry.angle_degrees), abs=1e-12,
    )
    assert result.max_threshold_exceeded_count_distance == pytest.approx(
        float(expected_entry.detector_distance), abs=1e-12,
    )


def test_invalid_optical_result_type_raises() -> None:
    with pytest.raises(ThermalError, match="LegacyOpticalScanResult"):
        compute_thermal_risk_over_legacy_optical_scan(
            "not a result",  # type: ignore[arg-type]
            **_THERMAL_KWARGS,
        )


def test_invalid_nominal_incident_irradiance_raises() -> None:
    optical = _setup_optical_result()
    bad_kwargs = dict(_THERMAL_KWARGS)
    for bad in (-1.0, float("nan"), float("inf")):
        bad_kwargs["nominal_incident_irradiance_w_m2"] = bad
        with pytest.raises(
            ThermalError, match="nominal_incident_irradiance",
        ):
            compute_thermal_risk_over_legacy_optical_scan(
                optical, **bad_kwargs,
            )


def test_no_file_output(tmp_path) -> None:
    optical = _setup_optical_result()
    before = sorted(tmp_path.iterdir())
    compute_thermal_risk_over_legacy_optical_scan(
        optical, **_THERMAL_KWARGS,
    )
    after = sorted(tmp_path.iterdir())
    assert before == after
