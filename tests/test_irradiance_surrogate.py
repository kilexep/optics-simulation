import os
from pathlib import Path

import numpy as np
import pytest

from optics_simulation.metrics import (
    DetectorIrradianceSurrogate,
    MetricsError,
    compute_relative_irradiance_surrogate,
)
from optics_simulation.optics import create_detector_grid
from optics_simulation.optics.detector_accumulation import (
    DetectorAccumulationResult,
    DetectorGrid,
)


def _build_accum(weight_map: np.ndarray) -> DetectorAccumulationResult:
    wm = np.asarray(weight_map, dtype=float)
    count_map = (wm > 0.0).astype(np.int64)
    n_hits = int(count_map.sum())
    return DetectorAccumulationResult(
        count_map=count_map,
        weight_map=wm.copy(),
        hit_pixel_indices=np.full((max(n_hits, 0), 2), -1, dtype=np.int64),
        hit_mask=np.ones((max(n_hits, 0),), dtype=bool),
        total_hits=n_hits,
        total_weight=float(wm.sum()),
    )


def _build_grid(
    width: float = 10.0,
    height: float = 10.0,
    resolution: tuple[int, int] = (4, 4),
) -> DetectorGrid:
    return create_detector_grid(
        width=width, height=height, resolution=resolution
    )


def test_returns_detector_irradiance_surrogate() -> None:
    grid = _build_grid()
    accum = _build_accum(np.ones((4, 4)))
    out = compute_relative_irradiance_surrogate(
        accum, grid,
        ray_count=16,
        source_area=100.0,
        incident_irradiance=1.0,
    )
    assert isinstance(out, DetectorIrradianceSurrogate)
    assert out.normalization_mode == "source_area_per_ray_over_pixel_area"


def test_map_shapes_match_weight_map_shape() -> None:
    grid = _build_grid(resolution=(3, 5))
    wm = np.ones((3, 5))
    accum = _build_accum(wm)
    out = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=15, source_area=10.0,
    )
    assert out.detector_power_map.shape == wm.shape
    assert out.relative_irradiance_map.shape == wm.shape
    assert out.detector_resolution == (3, 5)


def test_pixel_area_equals_grid_pixel_dimensions() -> None:
    grid = _build_grid(width=10.0, height=20.0, resolution=(4, 5))
    accum = _build_accum(np.zeros((4, 5)))
    out = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=10, source_area=1.0,
    )
    assert out.pixel_area == pytest.approx(
        grid.pixel_width * grid.pixel_height, abs=0.0
    )


def test_total_incident_power_equals_irradiance_times_source_area() -> None:
    grid = _build_grid()
    accum = _build_accum(np.zeros((4, 4)))
    for I, A in [(1.0, 100.0), (2.5, 40.0), (0.7, 1.5)]:
        out = compute_relative_irradiance_surrogate(
            accum, grid, ray_count=10, source_area=A, incident_irradiance=I,
        )
        assert out.total_incident_power == pytest.approx(I * A)


def test_incident_power_per_ray_equals_total_over_ray_count() -> None:
    grid = _build_grid()
    accum = _build_accum(np.zeros((4, 4)))
    out = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=25, source_area=50.0, incident_irradiance=2.0,
    )
    assert out.incident_power_per_ray == pytest.approx(
        out.total_incident_power / 25
    )


def test_detector_power_map_equals_weight_times_per_ray() -> None:
    grid = _build_grid()
    wm = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [0.5, 0.0, 0.0, 1.5],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.75],
        ]
    )
    accum = _build_accum(wm)
    out = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=10, source_area=20.0, incident_irradiance=1.0,
    )
    expected = wm * out.incident_power_per_ray
    np.testing.assert_allclose(out.detector_power_map, expected, atol=1e-15)


def test_relative_irradiance_map_formula() -> None:
    grid = _build_grid()
    wm = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [0.5, 0.0, 0.0, 1.5],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.75],
        ]
    )
    accum = _build_accum(wm)
    out = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=10, source_area=20.0, incident_irradiance=1.5,
    )
    expected = (
        out.detector_power_map / out.pixel_area / out.incident_irradiance
    )
    np.testing.assert_allclose(
        out.relative_irradiance_map, expected, atol=1e-15
    )


def test_all_zero_weight_map_returns_zero_maps() -> None:
    grid = _build_grid()
    accum = _build_accum(np.zeros((4, 4)))
    out = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=10, source_area=20.0,
    )
    assert float(np.abs(out.detector_power_map).max()) == 0.0
    assert float(np.abs(out.relative_irradiance_map).max()) == 0.0
    assert out.total_detector_power == 0.0


def test_total_detector_power_equals_map_sum() -> None:
    grid = _build_grid()
    wm = np.array(
        [[1.0, 2.0, 3.0, 4.0]] * 4, dtype=float
    )
    accum = _build_accum(wm)
    out = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=10, source_area=20.0, incident_irradiance=1.5,
    )
    assert out.total_detector_power == float(out.detector_power_map.sum())


def test_ray_density_normalization_invariant() -> None:
    grid = _build_grid(width=4.0, height=4.0, resolution=(1, 1))
    accum_a = _build_accum(np.array([[10.0]]))
    accum_b = _build_accum(np.array([[20.0]]))
    out_a = compute_relative_irradiance_surrogate(
        accum_a, grid, ray_count=10, source_area=4.0, incident_irradiance=1.0,
    )
    out_b = compute_relative_irradiance_surrogate(
        accum_b, grid, ray_count=20, source_area=4.0, incident_irradiance=1.0,
    )
    np.testing.assert_allclose(
        out_a.relative_irradiance_map,
        out_b.relative_irradiance_map,
        atol=1e-15,
    )


def test_detector_pixel_area_effect() -> None:
    wm = np.array([[5.0]])
    accum = _build_accum(wm)
    big_grid = _build_grid(width=10.0, height=10.0, resolution=(1, 1))
    small_grid = _build_grid(width=5.0, height=5.0, resolution=(1, 1))
    out_big = compute_relative_irradiance_surrogate(
        accum, big_grid, ray_count=10, source_area=100.0,
    )
    out_small = compute_relative_irradiance_surrogate(
        accum, small_grid, ray_count=10, source_area=100.0,
    )
    # Smaller pixel area -> larger relative irradiance for the same
    # detector_power_map.
    assert (
        float(out_small.relative_irradiance_map[0, 0])
        > float(out_big.relative_irradiance_map[0, 0])
    )


def test_incident_irradiance_cancellation() -> None:
    grid = _build_grid()
    wm = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [0.5, 0.0, 0.0, 1.5],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.75],
        ]
    )
    accum = _build_accum(wm)
    out_low = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=10, source_area=20.0, incident_irradiance=1.0,
    )
    out_high = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=10, source_area=20.0, incident_irradiance=10.0,
    )
    # relative_irradiance_map cancels out incident_irradiance
    np.testing.assert_allclose(
        out_low.relative_irradiance_map,
        out_high.relative_irradiance_map,
        atol=1e-15,
    )
    # detector_power_map scales linearly with incident_irradiance
    np.testing.assert_allclose(
        out_high.detector_power_map,
        10.0 * out_low.detector_power_map,
        atol=1e-15,
    )
    # total_incident_power scales linearly with incident_irradiance
    assert out_high.total_incident_power == pytest.approx(
        10.0 * out_low.total_incident_power
    )


def test_invalid_ray_count_raises() -> None:
    grid = _build_grid()
    accum = _build_accum(np.zeros((4, 4)))
    for bad in (0, -1):
        with pytest.raises(MetricsError, match="ray_count"):
            compute_relative_irradiance_surrogate(
                accum, grid, ray_count=bad, source_area=10.0,
            )
    for bad in (1.5, True, False, None, "10"):
        with pytest.raises(MetricsError, match="ray_count"):
            compute_relative_irradiance_surrogate(
                accum, grid, ray_count=bad, source_area=10.0,  # type: ignore[arg-type]
            )


def test_invalid_source_area_raises() -> None:
    grid = _build_grid()
    accum = _build_accum(np.zeros((4, 4)))
    for bad in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(MetricsError, match="source_area"):
            compute_relative_irradiance_surrogate(
                accum, grid, ray_count=10, source_area=bad,
            )


def test_invalid_incident_irradiance_raises() -> None:
    grid = _build_grid()
    accum = _build_accum(np.zeros((4, 4)))
    for bad in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(MetricsError, match="incident_irradiance"):
            compute_relative_irradiance_surrogate(
                accum, grid, ray_count=10, source_area=10.0,
                incident_irradiance=bad,
            )


def test_non_finite_weight_map_raises() -> None:
    grid = _build_grid()
    bad_nan = np.zeros((4, 4))
    bad_nan[0, 0] = np.nan
    with pytest.raises(MetricsError, match="NaN or inf"):
        compute_relative_irradiance_surrogate(
            _build_accum(bad_nan), grid, ray_count=10, source_area=10.0,
        )
    bad_inf = np.zeros((4, 4))
    bad_inf[1, 2] = np.inf
    with pytest.raises(MetricsError, match="NaN or inf"):
        compute_relative_irradiance_surrogate(
            _build_accum(bad_inf), grid, ray_count=10, source_area=10.0,
        )


def test_negative_weight_map_raises() -> None:
    grid = _build_grid()
    bad = np.zeros((4, 4))
    bad[2, 1] = -0.01
    with pytest.raises(MetricsError, match="non-negative"):
        compute_relative_irradiance_surrogate(
            _build_accum(bad), grid, ray_count=10, source_area=10.0,
        )


def test_shape_mismatch_raises() -> None:
    grid = _build_grid(resolution=(4, 4))
    accum = _build_accum(np.zeros((3, 3)))
    with pytest.raises(MetricsError, match="shape"):
        compute_relative_irradiance_surrogate(
            accum, grid, ray_count=10, source_area=10.0,
        )


def test_input_weight_map_not_mutated() -> None:
    grid = _build_grid()
    wm = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [0.5, 0.0, 0.0, 1.5],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.75],
        ]
    )
    accum = _build_accum(wm)
    snapshot = np.array(accum.weight_map, copy=True)
    out = compute_relative_irradiance_surrogate(
        accum, grid, ray_count=10, source_area=20.0,
    )
    # No alias to internal weight_map.
    assert out.detector_power_map is not accum.weight_map
    assert out.relative_irradiance_map is not accum.weight_map
    # Mutating the returned array must not corrupt the input.
    out.detector_power_map[:] = 0.0
    assert np.array_equal(np.asarray(accum.weight_map), snapshot)


def test_no_file_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    grid = _build_grid()
    accum = _build_accum(np.ones((4, 4)))
    before = set(os.listdir(tmp_path))
    compute_relative_irradiance_surrogate(
        accum, grid, ray_count=16, source_area=100.0,
    )
    after = set(os.listdir(tmp_path))
    assert before == after


def test_invalid_accumulation_type_raises() -> None:
    grid = _build_grid()
    with pytest.raises(MetricsError, match="DetectorAccumulationResult"):
        compute_relative_irradiance_surrogate(
            object(), grid, ray_count=10, source_area=10.0,  # type: ignore[arg-type]
        )


def test_invalid_detector_grid_type_raises() -> None:
    accum = _build_accum(np.zeros((4, 4)))
    with pytest.raises(MetricsError, match="DetectorGrid"):
        compute_relative_irradiance_surrogate(
            accum, object(), ray_count=10, source_area=10.0,  # type: ignore[arg-type]
        )
