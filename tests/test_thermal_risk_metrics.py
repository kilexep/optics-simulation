import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optics_simulation.thermal import (
    LumpedTargetHeatingMapResult,
    ThermalError,
    ThermalRiskMetrics,
    compute_thermal_risk_metrics,
    simulate_lumped_target_heating_map,
)


_BASE_MAP_KWARGS = dict(
    duration_s=10.0,
    dt_s=0.5,
    areal_heat_capacity_j_m2k=1200.0,
    absorptivity=0.8,
    h_conv_w_m2k=10.0,
    emissivity=0.9,
    ambient_temp_k=293.15,
    threshold_temp_k=295.0,
    top_percent=1.0,
)


def _heating_map(
    flux: np.ndarray | None = None,
    **overrides,
) -> LumpedTargetHeatingMapResult:
    if flux is None:
        flux = np.array(
            [[2000.0, 1000.0], [500.0, 0.0]], dtype=float,
        )
    kw = dict(_BASE_MAP_KWARGS)
    kw.update(overrides)
    return simulate_lumped_target_heating_map(flux, **kw)


def test_returns_thermal_risk_metrics() -> None:
    out = compute_thermal_risk_metrics(_heating_map())
    assert isinstance(out, ThermalRiskMetrics)
    assert out.metric_type == "thermal_risk_surrogate_metrics"


def test_max_temperature_matches_map_max() -> None:
    hm = _heating_map()
    out = compute_thermal_risk_metrics(hm)
    assert out.max_temperature_k == pytest.approx(
        float(hm.max_temperature_map_k.max())
    )


def test_max_temperature_rise_matches_map_max() -> None:
    hm = _heating_map()
    out = compute_thermal_risk_metrics(hm)
    assert out.max_temperature_rise_k == pytest.approx(
        float(hm.max_temperature_rise_map_k.max())
    )


def test_final_temperature_uses_hottest_final_pixel() -> None:
    hm = _heating_map()
    out = compute_thermal_risk_metrics(hm)
    assert out.final_temperature_k == pytest.approx(
        float(hm.final_temperature_map_k.max())
    )
    assert out.final_temperature_rise_k == pytest.approx(
        float(hm.final_temperature_rise_map_k.max())
    )


def test_mean_fields_match_numpy_means() -> None:
    hm = _heating_map()
    out = compute_thermal_risk_metrics(hm)
    assert out.mean_max_temperature_k == pytest.approx(
        float(hm.max_temperature_map_k.mean())
    )
    assert out.mean_max_temperature_rise_k == pytest.approx(
        float(hm.max_temperature_rise_map_k.mean())
    )
    assert out.mean_final_temperature_k == pytest.approx(
        float(hm.final_temperature_map_k.mean())
    )


def test_top_percent_matches_manual_top_k_mean() -> None:
    flux = np.array(
        [[100.0, 200.0, 300.0, 400.0, 500.0]], dtype=float,
    )
    hm = _heating_map(flux=flux)
    out = compute_thermal_risk_metrics(hm, top_percent=20.0)
    flat = np.sort(hm.max_temperature_rise_map_k.ravel())
    n = flat.size
    k = max(1, int(np.ceil(n * 20.0 / 100.0)))
    expected = float(flat[-k:].mean())
    assert out.top_percent_max_temperature_rise_k == pytest.approx(
        expected
    )


def test_threshold_exceeded_count_copied_correctly() -> None:
    hm = _heating_map()
    out = compute_thermal_risk_metrics(hm)
    assert out.threshold_exceeded_count == int(
        hm.threshold_exceeded_count
    )


def test_threshold_exceeded_fraction_equals_count_over_pixel_count() -> None:
    hm = _heating_map()
    out = compute_thermal_risk_metrics(hm)
    expected = float(hm.threshold_exceeded_count) / float(
        hm.max_temperature_map_k.size
    )
    assert out.threshold_exceeded_fraction == pytest.approx(expected)


def test_threshold_exceeded_area_equals_count_times_pixel_area() -> None:
    hm = _heating_map()
    out = compute_thermal_risk_metrics(hm, pixel_area=2.5)
    expected = float(hm.threshold_exceeded_count) * 2.5
    assert out.threshold_exceeded_area == pytest.approx(expected)


def test_threshold_exceeded_area_is_none_when_pixel_area_none() -> None:
    hm = _heating_map()
    out = compute_thermal_risk_metrics(hm, pixel_area=None)
    assert out.threshold_exceeded_area is None
    assert out.pixel_area is None


def test_finite_threshold_time_stats_are_computed() -> None:
    flux = np.array([[2000.0, 0.0]], dtype=float)
    hm = _heating_map(
        flux=flux,
        duration_s=30.0,
        threshold_temp_k=300.0,
    )
    out = compute_thermal_risk_metrics(hm)
    finite = hm.time_to_threshold_map_s[
        np.isfinite(hm.time_to_threshold_map_s)
    ]
    assert finite.size > 0
    assert out.min_time_to_threshold_s == pytest.approx(
        float(finite.min())
    )
    assert out.mean_time_to_threshold_s == pytest.approx(
        float(finite.mean())
    )
    assert out.max_time_to_threshold_s == pytest.approx(
        float(finite.max())
    )


def test_threshold_time_stats_are_none_when_no_finite_times() -> None:
    flux = np.full((2, 2), 1.0, dtype=float)
    hm = _heating_map(
        flux=flux,
        duration_s=1.0,
        threshold_temp_k=10000.0,
    )
    out = compute_thermal_risk_metrics(hm)
    assert out.min_time_to_threshold_s is None
    assert out.mean_time_to_threshold_s is None
    assert out.max_time_to_threshold_s is None


def test_exceedance_duration_and_degree_seconds_are_none() -> None:
    out = compute_thermal_risk_metrics(_heating_map())
    assert out.exceedance_duration_sum_s is None
    assert out.exceedance_degree_seconds_sum_ks is None


def test_invalid_top_percent_raises() -> None:
    hm = _heating_map()
    for bad in (0.0, -1.0, 100.1, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="top_percent"):
            compute_thermal_risk_metrics(hm, top_percent=bad)


def test_invalid_pixel_area_raises() -> None:
    hm = _heating_map()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="pixel_area"):
            compute_thermal_risk_metrics(hm, pixel_area=bad)


def test_invalid_heating_map_type_raises() -> None:
    with pytest.raises(
        ThermalError, match="LumpedTargetHeatingMapResult"
    ):
        compute_thermal_risk_metrics(object())  # type: ignore[arg-type]


def test_inconsistent_map_shapes_raise() -> None:
    base = _heating_map()
    bad = replace(
        base,
        max_temperature_map_k=base.max_temperature_map_k[:1, :],
    )
    with pytest.raises(ThermalError, match="shape"):
        compute_thermal_risk_metrics(bad)


def test_no_file_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    hm = _heating_map()
    before = set(os.listdir(tmp_path))
    compute_thermal_risk_metrics(hm)
    after = set(os.listdir(tmp_path))
    assert before == after
