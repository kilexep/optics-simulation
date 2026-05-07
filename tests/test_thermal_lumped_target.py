import os
from pathlib import Path

import numpy as np
import pytest

from optics_simulation.thermal import (
    LumpedTargetHeatingMapResult,
    LumpedTargetHeatingResult,
    ThermalError,
    simulate_lumped_target_heating,
    simulate_lumped_target_heating_map,
)


_BASE_KWARGS = dict(
    incident_flux_w_m2=1000.0,
    duration_s=10.0,
    dt_s=0.1,
    areal_heat_capacity_j_m2k=1200.0,
    absorptivity=0.8,
    h_conv_w_m2k=0.0,
    emissivity=0.0,
    ambient_temp_k=293.15,
    initial_temp_k=None,
    threshold_temp_k=None,
)


def _kwargs(**overrides):
    out = dict(_BASE_KWARGS)
    out.update(overrides)
    return out


def test_returns_lumped_target_heating_result() -> None:
    out = simulate_lumped_target_heating(**_kwargs())
    assert isinstance(out, LumpedTargetHeatingResult)
    assert out.model_type == "lumped_target_heating_surrogate"


def test_time_starts_at_zero_and_ends_at_duration() -> None:
    out = simulate_lumped_target_heating(**_kwargs(duration_s=10.0, dt_s=0.7))
    assert float(out.time_s[0]) == 0.0
    assert float(out.time_s[-1]) == 10.0
    # Strictly monotonic non-decreasing.
    assert (np.diff(out.time_s) > 0).all()


def test_temperature_shape_matches_time_shape() -> None:
    out = simulate_lumped_target_heating(**_kwargs())
    assert out.temperature_k.shape == out.time_s.shape


def test_zero_flux_with_initial_equal_to_ambient_keeps_constant() -> None:
    out = simulate_lumped_target_heating(**_kwargs(
        incident_flux_w_m2=0.0,
        h_conv_w_m2k=0.0,
        emissivity=0.0,
        initial_temp_k=293.15,
        ambient_temp_k=293.15,
    ))
    assert np.allclose(out.temperature_k, 293.15, atol=1e-12)
    assert out.max_temperature_rise_k == pytest.approx(0.0, abs=1e-12)
    assert out.final_temperature_rise_k == pytest.approx(0.0, abs=1e-12)


def test_higher_flux_gives_higher_max_temperature() -> None:
    low = simulate_lumped_target_heating(**_kwargs(incident_flux_w_m2=500.0))
    high = simulate_lumped_target_heating(**_kwargs(incident_flux_w_m2=2000.0))
    assert high.max_temperature_k > low.max_temperature_k


def test_absorptivity_zero_with_initial_equal_to_ambient_keeps_constant() -> None:
    out = simulate_lumped_target_heating(**_kwargs(
        absorptivity=0.0,
        initial_temp_k=293.15,
        ambient_temp_k=293.15,
        h_conv_w_m2k=0.0,
        emissivity=0.0,
    ))
    assert np.allclose(out.temperature_k, 293.15, atol=1e-12)


def test_convection_reduces_final_temperature() -> None:
    no_conv = simulate_lumped_target_heating(**_kwargs(h_conv_w_m2k=0.0))
    with_conv = simulate_lumped_target_heating(**_kwargs(h_conv_w_m2k=20.0))
    assert with_conv.final_temperature_k < no_conv.final_temperature_k


def test_radiation_reduces_final_temperature() -> None:
    no_rad = simulate_lumped_target_heating(**_kwargs(
        emissivity=0.0,
        duration_s=60.0,
        incident_flux_w_m2=2000.0,
    ))
    with_rad = simulate_lumped_target_heating(**_kwargs(
        emissivity=0.9,
        duration_s=60.0,
        incident_flux_w_m2=2000.0,
    ))
    assert with_rad.final_temperature_k < no_rad.final_temperature_k


def test_threshold_crossing_returns_finite_time() -> None:
    out = simulate_lumped_target_heating(**_kwargs(
        incident_flux_w_m2=2000.0,
        duration_s=30.0,
        threshold_temp_k=300.0,
    ))
    assert out.time_to_threshold_s is not None
    assert 0.0 <= out.time_to_threshold_s <= 30.0


def test_subthreshold_returns_none() -> None:
    out = simulate_lumped_target_heating(**_kwargs(
        incident_flux_w_m2=10.0,
        duration_s=1.0,
        threshold_temp_k=1000.0,
    ))
    assert out.time_to_threshold_s is None


def test_threshold_none_returns_none() -> None:
    out = simulate_lumped_target_heating(**_kwargs(
        incident_flux_w_m2=2000.0,
        threshold_temp_k=None,
    ))
    assert out.time_to_threshold_s is None


def test_final_temperature_equals_temperature_last() -> None:
    out = simulate_lumped_target_heating(**_kwargs())
    assert out.final_temperature_k == float(out.temperature_k[-1])


def test_max_temperature_equals_temperature_max() -> None:
    out = simulate_lumped_target_heating(**_kwargs())
    assert out.max_temperature_k == float(out.temperature_k.max())


def test_absorbed_flux_equals_absorptivity_times_incident() -> None:
    out = simulate_lumped_target_heating(**_kwargs(
        incident_flux_w_m2=1234.0,
        absorptivity=0.7,
    ))
    assert out.absorbed_flux_w_m2 == pytest.approx(0.7 * 1234.0)


def test_invalid_incident_flux_raises() -> None:
    for bad in (-1.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ThermalError, match="incident_flux_w_m2"):
            simulate_lumped_target_heating(**_kwargs(incident_flux_w_m2=bad))


def test_invalid_duration_raises() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ThermalError, match="duration_s"):
            simulate_lumped_target_heating(**_kwargs(duration_s=bad))


def test_invalid_dt_raises() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="dt_s"):
            simulate_lumped_target_heating(**_kwargs(dt_s=bad))


def test_dt_greater_than_duration_raises() -> None:
    with pytest.raises(ThermalError, match="dt_s must be <= duration_s"):
        simulate_lumped_target_heating(**_kwargs(duration_s=1.0, dt_s=2.0))


def test_invalid_areal_heat_capacity_raises() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="areal_heat_capacity_j_m2k"):
            simulate_lumped_target_heating(
                **_kwargs(areal_heat_capacity_j_m2k=bad)
            )


def test_invalid_absorptivity_raises() -> None:
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="absorptivity"):
            simulate_lumped_target_heating(**_kwargs(absorptivity=bad))


def test_invalid_h_conv_raises() -> None:
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="h_conv_w_m2k"):
            simulate_lumped_target_heating(**_kwargs(h_conv_w_m2k=bad))


def test_invalid_emissivity_raises() -> None:
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="emissivity"):
            simulate_lumped_target_heating(**_kwargs(emissivity=bad))


def test_invalid_temperatures_raise() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="ambient_temp_k"):
            simulate_lumped_target_heating(**_kwargs(ambient_temp_k=bad))
        with pytest.raises(ThermalError, match="initial_temp_k"):
            simulate_lumped_target_heating(**_kwargs(initial_temp_k=bad))
        with pytest.raises(ThermalError, match="threshold_temp_k"):
            simulate_lumped_target_heating(**_kwargs(threshold_temp_k=bad))


def test_returned_arrays_do_not_alias_internal_state() -> None:
    out = simulate_lumped_target_heating(**_kwargs())
    snapshot_T = out.temperature_k.copy()
    snapshot_t = out.time_s.copy()
    out.temperature_k[:] = -1.0
    out.time_s[:] = -1.0
    out2 = simulate_lumped_target_heating(**_kwargs())
    np.testing.assert_array_equal(out2.temperature_k, snapshot_T)
    np.testing.assert_array_equal(out2.time_s, snapshot_t)


def test_no_file_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    simulate_lumped_target_heating(**_kwargs())
    after = set(os.listdir(tmp_path))
    assert before == after


# -----------------------------------------------------------------
# Per-pixel thermal-map surrogate tests
# -----------------------------------------------------------------

_MAP_BASE_KWARGS = dict(
    duration_s=10.0,
    dt_s=0.5,
    areal_heat_capacity_j_m2k=1200.0,
    absorptivity=0.8,
    h_conv_w_m2k=0.0,
    emissivity=0.0,
    ambient_temp_k=293.15,
    initial_temp_k=None,
    threshold_temp_k=None,
    top_percent=1.0,
)


def _map_kwargs(**overrides):
    out = dict(_MAP_BASE_KWARGS)
    out.update(overrides)
    return out


def _default_flux_map() -> np.ndarray:
    return np.array(
        [[1000.0, 500.0], [200.0, 0.0]], dtype=float,
    )


def test_map_returns_lumped_target_heating_map_result() -> None:
    out = simulate_lumped_target_heating_map(
        _default_flux_map(), **_map_kwargs(),
    )
    assert isinstance(out, LumpedTargetHeatingMapResult)
    assert out.model_type == "per_pixel_lumped_target_heating_surrogate"


def test_map_shapes_match_input_flux_map() -> None:
    flux = _default_flux_map()
    out = simulate_lumped_target_heating_map(flux, **_map_kwargs())
    assert out.final_temperature_map_k.shape == flux.shape
    assert out.max_temperature_map_k.shape == flux.shape
    assert out.final_temperature_rise_map_k.shape == flux.shape
    assert out.max_temperature_rise_map_k.shape == flux.shape
    assert out.time_to_threshold_map_s.shape == flux.shape
    assert out.threshold_exceeded_mask.shape == flux.shape
    assert out.incident_flux_map_w_m2.shape == flux.shape
    assert out.absorbed_flux_map_w_m2.shape == flux.shape


def test_map_time_starts_at_zero_and_ends_at_duration() -> None:
    out = simulate_lumped_target_heating_map(
        _default_flux_map(), **_map_kwargs(duration_s=10.0, dt_s=0.7),
    )
    assert float(out.time_s[0]) == 0.0
    assert float(out.time_s[-1]) == 10.0
    assert (np.diff(out.time_s) > 0).all()


def test_map_zero_flux_with_initial_equal_to_ambient_keeps_constant() -> None:
    flux = np.zeros((3, 4), dtype=float)
    out = simulate_lumped_target_heating_map(
        flux,
        **_map_kwargs(
            h_conv_w_m2k=0.0,
            emissivity=0.0,
            initial_temp_k=293.15,
            ambient_temp_k=293.15,
        ),
    )
    assert np.allclose(out.final_temperature_map_k, 293.15, atol=1e-12)
    assert np.allclose(out.max_temperature_map_k, 293.15, atol=1e-12)
    assert np.allclose(
        out.max_temperature_rise_map_k, 0.0, atol=1e-12,
    )
    assert np.allclose(
        out.final_temperature_rise_map_k, 0.0, atol=1e-12,
    )


def test_map_higher_flux_pixel_gives_higher_max_temperature() -> None:
    flux = np.array([[100.0, 5000.0]], dtype=float)
    out = simulate_lumped_target_heating_map(flux, **_map_kwargs())
    assert (
        float(out.max_temperature_map_k[0, 1])
        > float(out.max_temperature_map_k[0, 0])
    )
    assert (
        float(out.max_temperature_rise_map_k[0, 1])
        > float(out.max_temperature_rise_map_k[0, 0])
    )


def test_map_all_zero_flux_returns_uniform_initial_map() -> None:
    flux = np.zeros((4, 5), dtype=float)
    out = simulate_lumped_target_heating_map(
        flux,
        **_map_kwargs(
            initial_temp_k=300.0,
            ambient_temp_k=300.0,
        ),
    )
    assert np.allclose(out.final_temperature_map_k, 300.0, atol=1e-12)
    assert np.allclose(out.max_temperature_map_k, 300.0, atol=1e-12)


def test_map_absorptivity_zero_with_initial_equal_to_ambient_keeps_constant() -> None:
    flux = _default_flux_map()
    out = simulate_lumped_target_heating_map(
        flux,
        **_map_kwargs(
            absorptivity=0.0,
            initial_temp_k=293.15,
            ambient_temp_k=293.15,
            h_conv_w_m2k=0.0,
            emissivity=0.0,
        ),
    )
    assert np.allclose(out.final_temperature_map_k, 293.15, atol=1e-12)
    assert np.allclose(out.max_temperature_map_k, 293.15, atol=1e-12)


def test_map_convection_reduces_final_temperature() -> None:
    flux = np.array([[2000.0, 1500.0]], dtype=float)
    no_conv = simulate_lumped_target_heating_map(
        flux, **_map_kwargs(h_conv_w_m2k=0.0),
    )
    with_conv = simulate_lumped_target_heating_map(
        flux, **_map_kwargs(h_conv_w_m2k=20.0),
    )
    assert (
        with_conv.final_temperature_map_k
        < no_conv.final_temperature_map_k
    ).all()


def test_map_radiation_reduces_final_temperature() -> None:
    flux = np.full((2, 2), 2000.0, dtype=float)
    no_rad = simulate_lumped_target_heating_map(
        flux,
        **_map_kwargs(
            emissivity=0.0,
            duration_s=60.0,
        ),
    )
    with_rad = simulate_lumped_target_heating_map(
        flux,
        **_map_kwargs(
            emissivity=0.9,
            duration_s=60.0,
        ),
    )
    assert (
        with_rad.final_temperature_map_k
        < no_rad.final_temperature_map_k
    ).all()


def test_map_threshold_crossing_returns_finite_time_in_map() -> None:
    flux = np.array([[2000.0, 0.0], [0.0, 0.0]], dtype=float)
    out = simulate_lumped_target_heating_map(
        flux,
        **_map_kwargs(
            duration_s=30.0,
            threshold_temp_k=300.0,
        ),
    )
    assert np.isfinite(out.time_to_threshold_map_s[0, 0])
    assert 0.0 <= float(out.time_to_threshold_map_s[0, 0]) <= 30.0
    assert out.threshold_exceeded_mask[0, 0]
    assert out.threshold_exceeded_count >= 1


def test_map_threshold_not_reached_gives_nan() -> None:
    flux = np.full((2, 2), 10.0, dtype=float)
    out = simulate_lumped_target_heating_map(
        flux,
        **_map_kwargs(
            duration_s=1.0,
            threshold_temp_k=1000.0,
        ),
    )
    assert np.isnan(out.time_to_threshold_map_s).all()
    assert out.threshold_exceeded_count == 0
    assert not out.threshold_exceeded_mask.any()


def test_map_threshold_none_gives_all_nan_and_zero_count() -> None:
    flux = _default_flux_map()
    out = simulate_lumped_target_heating_map(
        flux,
        **_map_kwargs(threshold_temp_k=None),
    )
    assert np.isnan(out.time_to_threshold_map_s).all()
    assert out.threshold_exceeded_count == 0
    assert not out.threshold_exceeded_mask.any()


def test_map_max_geq_final_elementwise() -> None:
    out = simulate_lumped_target_heating_map(
        _default_flux_map(), **_map_kwargs(),
    )
    assert (
        out.max_temperature_map_k >= out.final_temperature_map_k - 1e-12
    ).all()


def test_map_final_rise_equals_final_minus_initial() -> None:
    out = simulate_lumped_target_heating_map(
        _default_flux_map(),
        **_map_kwargs(initial_temp_k=290.0),
    )
    np.testing.assert_allclose(
        out.final_temperature_rise_map_k,
        out.final_temperature_map_k - 290.0,
        atol=1e-12,
    )


def test_map_max_rise_equals_max_minus_initial() -> None:
    out = simulate_lumped_target_heating_map(
        _default_flux_map(),
        **_map_kwargs(initial_temp_k=290.0),
    )
    np.testing.assert_allclose(
        out.max_temperature_rise_map_k,
        out.max_temperature_map_k - 290.0,
        atol=1e-12,
    )


def test_map_top_percent_matches_manual_top_k_mean() -> None:
    flux = np.array(
        [[100.0, 200.0, 300.0, 400.0, 500.0]], dtype=float,
    )
    out = simulate_lumped_target_heating_map(
        flux, **_map_kwargs(top_percent=20.0),
    )
    flat = np.sort(out.max_temperature_rise_map_k.ravel())
    n = flat.size
    k = max(1, int(np.ceil(n * 20.0 / 100.0)))
    expected = float(flat[-k:].mean())
    assert out.top_percent_max_temperature_rise_k == pytest.approx(
        expected
    )


def test_map_invalid_flux_shape_raises() -> None:
    bad_1d = np.array([1.0, 2.0, 3.0], dtype=float)
    with pytest.raises(ThermalError, match="2D"):
        simulate_lumped_target_heating_map(bad_1d, **_map_kwargs())
    bad_3d = np.zeros((2, 2, 2), dtype=float)
    with pytest.raises(ThermalError, match="2D"):
        simulate_lumped_target_heating_map(bad_3d, **_map_kwargs())
    bad_empty = np.zeros((0, 0), dtype=float)
    with pytest.raises(ThermalError, match="non-empty"):
        simulate_lumped_target_heating_map(bad_empty, **_map_kwargs())


def test_map_nan_or_inf_flux_raises() -> None:
    for bad in (float("nan"), float("inf"), -float("inf")):
        flux = np.array([[1.0, bad]], dtype=float)
        with pytest.raises(
            ThermalError, match="incident_flux_map_w_m2"
        ):
            simulate_lumped_target_heating_map(flux, **_map_kwargs())


def test_map_negative_flux_raises() -> None:
    flux = np.array([[1.0, -1.0]], dtype=float)
    with pytest.raises(
        ThermalError, match="incident_flux_map_w_m2"
    ):
        simulate_lumped_target_heating_map(flux, **_map_kwargs())


def test_map_invalid_top_percent_raises() -> None:
    flux = _default_flux_map()
    for bad in (0.0, -1.0, 100.1, float("nan"), float("inf")):
        with pytest.raises(ThermalError, match="top_percent"):
            simulate_lumped_target_heating_map(
                flux, **_map_kwargs(top_percent=bad),
            )


def test_map_returned_arrays_do_not_alias_input() -> None:
    flux = _default_flux_map()
    flux_snapshot = flux.copy()
    out = simulate_lumped_target_heating_map(flux, **_map_kwargs())
    out.incident_flux_map_w_m2[:] = -1.0
    out.final_temperature_map_k[:] = -1.0
    out.max_temperature_map_k[:] = -1.0
    out.absorbed_flux_map_w_m2[:] = -1.0
    np.testing.assert_array_equal(flux, flux_snapshot)


def test_map_no_file_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    simulate_lumped_target_heating_map(
        _default_flux_map(), **_map_kwargs(),
    )
    after = set(os.listdir(tmp_path))
    assert before == after
