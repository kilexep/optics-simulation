import numpy as np
import pytest
import trimesh

from optics_simulation.metrics import (
    MetricsError,
    OpticalMetrics,
    compute_optical_metrics,
)
from optics_simulation.optics import (
    accumulate_detector_hits,
    create_detector_grid,
    create_detector_plane,
    intersect_detector_plane,
    parallel_ray_grid,
    run_multi_step_trace,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
SLAB_INTERFACES = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]


def test_basic_peak_total_mean() -> None:
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    m = compute_optical_metrics(arr)
    assert isinstance(m, OpticalMetrics)
    assert m.peak_value == pytest.approx(9.0)
    assert m.total_value == pytest.approx(45.0)
    assert m.mean_value == pytest.approx(5.0)
    assert m.pixel_count == 9


def test_cmax_uses_incident_reference() -> None:
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    m = compute_optical_metrics(arr, incident_reference=2.0)
    assert m.cmax == pytest.approx(9.0 / 2.0)


def test_c99_top_one_percent_on_100_pixel_map() -> None:
    arr = np.arange(100, dtype=float).reshape(10, 10)
    m = compute_optical_metrics(arr, incident_reference=1.0, top_percent=1.0)
    # Top 1% of 100 pixels = 1 pixel = 99 (the max).
    assert m.c99 == pytest.approx(99.0)


def test_c99_uses_at_least_one_pixel_on_small_map() -> None:
    # 25 pixels, top_percent=1.0 -> ceil(0.25)=1 pixel -> c99 = peak / Iref.
    arr = np.arange(25, dtype=float).reshape(5, 5)
    m = compute_optical_metrics(arr, top_percent=1.0)
    assert m.c99 == pytest.approx(24.0)


def test_eexceed_per_threshold() -> None:
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = compute_optical_metrics(
        arr, incident_reference=1.0, thresholds=(3.0, 5.0)
    )
    # threshold=3.0 -> excess = [[0,0,0],[1,2,3]] sum=6
    # threshold=5.0 -> excess = [[0,0,0],[0,0,1]] sum=1
    assert m.eexceed[3.0] == pytest.approx(6.0)
    assert m.eexceed[5.0] == pytest.approx(1.0)


def test_ahot_per_threshold_strict_greater_than() -> None:
    # Pixel value exactly equal to threshold * incident_reference must
    # NOT count as hot. Here threshold=3.0, Iref=1.0, cutoff=3.0; the
    # pixel with value 3.0 must be excluded.
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = compute_optical_metrics(
        arr, incident_reference=1.0, thresholds=(3.0, 5.0)
    )
    assert m.ahot[3.0] == 3  # 4, 5, 6 only
    assert m.ahot[5.0] == 1  # 6 only


def test_ahot_boundary_pixel_excluded_with_nontrivial_iref() -> None:
    # cutoff = threshold * incident_reference = 2.0 * 1.5 = 3.0.
    # Pixel value exactly 3.0 must be excluded.
    arr = np.array([[3.0, 3.0, 4.5], [4.5, 4.5, 4.5]])
    m = compute_optical_metrics(
        arr, incident_reference=1.5, thresholds=(2.0,)
    )
    assert m.ahot[2.0] == 4  # only the four 4.5 pixels


def test_all_zero_map_returns_zero_metrics_without_error() -> None:
    arr = np.zeros((4, 4))
    m = compute_optical_metrics(arr, thresholds=(1.0, 2.0))
    assert m.peak_value == 0.0
    assert m.total_value == 0.0
    assert m.mean_value == 0.0
    assert m.cmax == 0.0
    assert m.c99 == 0.0
    assert m.eexceed[1.0] == 0.0
    assert m.eexceed[2.0] == 0.0
    assert m.ahot[1.0] == 0
    assert m.ahot[2.0] == 0


def test_invalid_shape_raises() -> None:
    with pytest.raises(MetricsError, match="2D"):
        compute_optical_metrics(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(MetricsError, match="2D"):
        compute_optical_metrics(np.zeros((2, 2, 2)))
    with pytest.raises(MetricsError, match="2D"):
        compute_optical_metrics(np.array(5.0))


def test_nan_or_inf_raises() -> None:
    with pytest.raises(MetricsError, match="NaN or inf"):
        compute_optical_metrics(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(MetricsError, match="NaN or inf"):
        compute_optical_metrics(np.array([[1.0, np.inf], [2.0, 3.0]]))
    with pytest.raises(MetricsError, match="NaN or inf"):
        compute_optical_metrics(np.array([[1.0, -np.inf], [2.0, 3.0]]))


def test_negative_values_raise() -> None:
    with pytest.raises(MetricsError, match="negative"):
        compute_optical_metrics(np.array([[-0.1, 0.0], [0.0, 0.0]]))


def test_incident_reference_invalid_raises() -> None:
    arr = np.ones((3, 3))
    with pytest.raises(MetricsError, match="incident_reference"):
        compute_optical_metrics(arr, incident_reference=0.0)
    with pytest.raises(MetricsError, match="incident_reference"):
        compute_optical_metrics(arr, incident_reference=-1.0)


def test_top_percent_invalid_raises() -> None:
    arr = np.ones((3, 3))
    with pytest.raises(MetricsError, match="top_percent"):
        compute_optical_metrics(arr, top_percent=0.0)
    with pytest.raises(MetricsError, match="top_percent"):
        compute_optical_metrics(arr, top_percent=-1.0)
    with pytest.raises(MetricsError, match="top_percent"):
        compute_optical_metrics(arr, top_percent=101.0)


def test_thresholds_invalid_raises() -> None:
    arr = np.ones((3, 3))
    with pytest.raises(MetricsError, match="thresholds"):
        compute_optical_metrics(arr, thresholds=(0.0,))
    with pytest.raises(MetricsError, match="thresholds"):
        compute_optical_metrics(arr, thresholds=(-1.0,))
    with pytest.raises(MetricsError, match="thresholds"):
        compute_optical_metrics(arr, thresholds=(1.0, 2.0, 0.0))


def test_thresholds_input_order_preserved() -> None:
    arr = np.ones((3, 3))
    m = compute_optical_metrics(arr, thresholds=(10.0, 2.0, 5.0))
    assert m.thresholds == (10.0, 2.0, 5.0)


def test_weight_map_smoke_from_detector_accumulation() -> None:
    # Build a small detector accumulation result and feed its
    # weight_map into compute_optical_metrics. The metrics module
    # should accept it without error.
    detector = create_detector_plane(
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )
    grid = create_detector_grid(width=10.0, height=10.0, resolution=(10, 10))
    rays = parallel_ray_grid(
        origin_plane_z=5.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=5,
        ny=5,
    )
    hits = intersect_detector_plane(rays, detector)
    accum = accumulate_detector_hits(hits, grid)

    m = compute_optical_metrics(accum.weight_map, incident_reference=1.0)
    assert m.pixel_count == 100
    assert m.total_value == pytest.approx(float(accum.weight_map.sum()))
    assert m.peak_value >= 0.0


def test_count_map_and_weight_map_both_accepted() -> None:
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-3.0, 3.0),
        y_range=(-3.0, 3.0),
        nx=5,
        ny=5,
    )
    trace = run_multi_step_trace(box, rays, SLAB_INTERFACES)
    detector = create_detector_plane(
        center=(0.0, 0.0, -15.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )
    hits = intersect_detector_plane(trace.final_rays, detector)
    grid = create_detector_grid(width=10.0, height=10.0, resolution=(20, 20))
    accum = accumulate_detector_hits(hits, grid)

    m_count = compute_optical_metrics(accum.count_map, incident_reference=1.0)
    m_weight = compute_optical_metrics(accum.weight_map, incident_reference=1.0)

    assert isinstance(m_count, OpticalMetrics)
    assert isinstance(m_weight, OpticalMetrics)
    assert m_count.pixel_count == m_weight.pixel_count
    # With default unit weights both maps should have identical totals.
    assert m_count.total_value == pytest.approx(m_weight.total_value)


def test_top_percent_100_equals_mean_over_iref() -> None:
    arr = np.arange(16, dtype=float).reshape(4, 4)
    m = compute_optical_metrics(arr, top_percent=100.0, incident_reference=2.0)
    assert m.c99 == pytest.approx(arr.mean() / 2.0)


def test_single_pixel_map() -> None:
    arr = np.array([[7.0]])
    m = compute_optical_metrics(arr, incident_reference=2.0)
    assert m.peak_value == 7.0
    assert m.total_value == 7.0
    assert m.mean_value == 7.0
    assert m.cmax == pytest.approx(7.0 / 2.0)
    assert m.c99 == pytest.approx(7.0 / 2.0)
    assert m.pixel_count == 1


def test_total_value_matches_sum() -> None:
    arr = np.array([[0.5, 1.5], [2.5, 3.5]])
    m = compute_optical_metrics(arr)
    assert m.total_value == pytest.approx(arr.sum())


def test_empty_2d_map_raises() -> None:
    with pytest.raises(MetricsError, match="empty"):
        compute_optical_metrics(np.zeros((0, 0)))
    with pytest.raises(MetricsError, match="empty"):
        compute_optical_metrics(np.zeros((0, 5)))
