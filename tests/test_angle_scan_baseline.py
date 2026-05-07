import numpy as np
import pytest
import trimesh

from optics_simulation.angle_scan import (
    AngleScanError,
    AngleScanResult,
    PerAngleResult,
    direction_from_incident_angle,
    run_baseline_angle_scan,
)
from optics_simulation.metrics import OpticalMetrics
from optics_simulation.optics import (
    create_detector_grid,
    create_detector_plane,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
SLAB_INTERFACES = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]
THRESHOLDS = (2.0, 5.0, 10.0)


def _slab_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))


def _ray_grid_config() -> dict:
    return {
        "origin_plane_z": 20.0,
        "x_range": (-7.5, 7.5),
        "y_range": (-7.5, 7.5),
        "nx": 7,
        "ny": 7,
    }


def _detector():
    return create_detector_plane(
        center=(0.0, 0.0, -15.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )


def _detector_grid():
    return create_detector_grid(width=10.0, height=10.0, resolution=(10, 10))


def test_direction_zero_is_normal_incidence() -> None:
    d = direction_from_incident_angle(0.0)
    assert np.allclose(d, np.array([0.0, 0.0, -1.0]))


def test_direction_30_degrees_xz() -> None:
    d = direction_from_incident_angle(30.0)
    expected = np.array([np.sin(np.deg2rad(30.0)), 0.0, -np.cos(np.deg2rad(30.0))])
    assert np.allclose(d, expected)


def test_direction_is_unit_norm() -> None:
    for angle in (0.0, 15.0, 30.0, 45.0, 60.0, 80.0, -45.0):
        d = direction_from_incident_angle(angle)
        assert np.linalg.norm(d) == pytest.approx(1.0)


def test_direction_invalid_plane_raises() -> None:
    with pytest.raises(AngleScanError):
        direction_from_incident_angle(0.0, plane="yz")
    with pytest.raises(AngleScanError):
        direction_from_incident_angle(0.0, plane="bogus")


def test_empty_angles_returns_empty_result() -> None:
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=[],
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    assert isinstance(result, AngleScanResult)
    assert result.per_angle == ()
    assert result.angle_count == 0
    assert result.max_c99_angle is None
    assert result.max_c99 is None


def test_single_angle_scan_returns_one_result() -> None:
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=[0.0],
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    assert result.angle_count == 1
    assert len(result.per_angle) == 1
    assert isinstance(result.per_angle[0], PerAngleResult)
    assert result.per_angle[0].angle_degrees == 0.0


def test_multiple_angles_match_count_and_have_metrics() -> None:
    angles = [0.0, 5.0, 10.0]
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=angles,
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    assert result.angle_count == len(angles)
    assert len(result.per_angle) == len(angles)
    for entry, angle in zip(result.per_angle, angles):
        assert entry.angle_degrees == angle
        assert isinstance(entry.metrics, OpticalMetrics)
        assert entry.detector_hits >= 0
        assert entry.ray_count == 7 * 7
        assert entry.final_ray_count >= 0


def test_max_c99_angle_matches_argmax_over_per_angle() -> None:
    angles = [0.0, 5.0, 10.0, 20.0]
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=angles,
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    c99s = np.array([p.metrics.c99 for p in result.per_angle], dtype=float)
    expected_idx = int(np.argmax(c99s))
    assert result.max_c99_angle == result.per_angle[expected_idx].angle_degrees
    assert result.max_c99 == pytest.approx(float(c99s[expected_idx]))


def test_normal_incidence_produces_detector_hits() -> None:
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=[0.0],
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    assert result.per_angle[0].detector_hits > 0


def test_scan_does_not_modify_input_mesh() -> None:
    mesh = _slab_mesh()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    run_baseline_angle_scan(
        mesh=mesh,
        angles_degrees=[0.0, 10.0],
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_thresholds_propagate_into_metrics() -> None:
    custom_thresholds = (1.5, 3.0, 7.5)
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=[0.0],
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=custom_thresholds,
    )
    metrics = result.per_angle[0].metrics
    assert metrics.thresholds == custom_thresholds
    assert tuple(metrics.eexceed.keys()) == custom_thresholds
    assert tuple(metrics.ahot.keys()) == custom_thresholds


def test_per_angle_direction_is_unit_norm_and_matches_helper() -> None:
    angles = [0.0, 12.0, 25.0]
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=angles,
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    for entry, angle in zip(result.per_angle, angles):
        assert np.linalg.norm(entry.direction) == pytest.approx(1.0)
        assert np.allclose(
            entry.direction, direction_from_incident_angle(angle)
        )


# ---------------------------------------------------------------------------
# Power-weighted detector accumulation (use_power_weights)
# ---------------------------------------------------------------------------


def _scan(*, use_power_weights: bool, angles=(0.0, 10.0)):
    return run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=list(angles),
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
        use_power_weights=use_power_weights,
    )


def test_use_power_weights_default_is_false_backward_compatible() -> None:
    explicit = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=[0.0, 10.0],
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
        use_power_weights=False,
    )
    implicit = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=[0.0, 10.0],
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    for ex, im in zip(explicit.per_angle, implicit.per_angle):
        assert ex.detector_hits == im.detector_hits
        assert ex.metrics.c99 == pytest.approx(im.metrics.c99)
        assert ex.metrics.cmax == pytest.approx(im.metrics.cmax)


def test_use_power_weights_true_smoke_normal_incidence() -> None:
    weighted = _scan(use_power_weights=True, angles=(0.0,))
    unweighted = _scan(use_power_weights=False, angles=(0.0,))
    we = weighted.per_angle[0]
    ue = unweighted.per_angle[0]

    # Hit count is a count of rays reaching the detector — invariant
    # under weight choice.
    assert we.detector_hits == ue.detector_hits

    # If any rays reached the detector, weighted Cmax / C99 must be
    # strictly less than unweighted because passive Fresnel T < 1.
    if ue.detector_hits > 0:
        assert we.metrics.cmax < ue.metrics.cmax
        assert we.metrics.c99 <= ue.metrics.c99
        # And weighted Cmax stays bounded by 1.0 (no per-pixel multi-hit
        # in this fixture's slab + 7x7 normal-incidence grid).
        assert we.metrics.cmax <= 1.0 + 1e-9


def test_use_power_weights_true_oblique_reduces_metrics() -> None:
    angle = 30.0
    weighted = _scan(use_power_weights=True, angles=(angle,))
    unweighted = _scan(use_power_weights=False, angles=(angle,))
    we = weighted.per_angle[0]
    ue = unweighted.per_angle[0]
    if ue.detector_hits > 0:
        assert we.metrics.c99 < ue.metrics.c99


def test_use_power_weights_true_does_not_mutate_mesh_or_config() -> None:
    mesh = _slab_mesh()
    cfg = _ray_grid_config()
    cfg["direction"] = (0.5, 0.5, -0.5)
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    cfg_before = dict(cfg)

    run_baseline_angle_scan(
        mesh=mesh,
        angles_degrees=[0.0, 10.0],
        ray_grid_config=cfg,
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
        use_power_weights=True,
    )

    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)
    assert cfg == cfg_before


# ---------------------------------------------------------------------------
# Relative irradiance surrogate integration (use_relative_irradiance)
# ---------------------------------------------------------------------------


from optics_simulation.metrics import (
    DetectorIrradianceSurrogate,
    MetricsError,
    compute_optical_metrics,
    compute_relative_irradiance_surrogate,
)
from optics_simulation.optics import (
    accumulate_detector_hits,
    intersect_detector_plane,
    parallel_ray_grid,
    run_multi_step_trace,
)


def _common_kwargs(**overrides):
    base = dict(
        mesh=_slab_mesh(),
        angles_degrees=[0.0],
        ray_grid_config=_ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
    )
    base.update(overrides)
    return base


def test_default_use_relative_irradiance_is_false_backward_compatible() -> None:
    explicit = run_baseline_angle_scan(**_common_kwargs(
        use_relative_irradiance=False,
    ))
    implicit = run_baseline_angle_scan(**_common_kwargs())
    for ex, im in zip(explicit.per_angle, implicit.per_angle):
        assert ex.detector_hits == im.detector_hits
        assert ex.metrics.c99 == pytest.approx(im.metrics.c99)
        assert ex.metrics.cmax == pytest.approx(im.metrics.cmax)
        assert ex.irradiance_surrogate is None
        assert im.irradiance_surrogate is None


def test_use_relative_irradiance_true_populates_surrogate_field() -> None:
    result = run_baseline_angle_scan(**_common_kwargs(
        use_relative_irradiance=True,
    ))
    for entry in result.per_angle:
        assert isinstance(
            entry.irradiance_surrogate, DetectorIrradianceSurrogate
        )


def test_use_relative_irradiance_false_keeps_surrogate_none() -> None:
    result = run_baseline_angle_scan(**_common_kwargs(
        use_relative_irradiance=False,
    ))
    for entry in result.per_angle:
        assert entry.irradiance_surrogate is None


def test_relative_mode_metrics_match_manual_recomputation() -> None:
    cfg = _ray_grid_config()
    detector = _detector()
    detector_grid = _detector_grid()
    angles = [0.0, 10.0]
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=angles,
        ray_grid_config=cfg,
        interface_sequence=SLAB_INTERFACES,
        detector=detector,
        detector_grid=detector_grid,
        thresholds=THRESHOLDS,
        use_relative_irradiance=True,
    )
    expected_source_area = (
        (cfg["x_range"][1] - cfg["x_range"][0])
        * (cfg["y_range"][1] - cfg["y_range"][0])
    )
    grid_kwargs = {k: v for k, v in cfg.items() if k != "direction"}
    for entry, angle in zip(result.per_angle, angles):
        rays = parallel_ray_grid(
            direction=tuple(entry.direction), **grid_kwargs
        )
        trace = run_multi_step_trace(
            _slab_mesh(), rays, SLAB_INTERFACES
        )
        hits = intersect_detector_plane(trace.final_rays, detector)
        accum = accumulate_detector_hits(hits, detector_grid)
        surrogate = compute_relative_irradiance_surrogate(
            accum, detector_grid,
            ray_count=int(rays.ray_count),
            source_area=expected_source_area,
            incident_irradiance=1.0,
        )
        expected_metrics = compute_optical_metrics(
            surrogate.relative_irradiance_map,
            incident_reference=1.0,
            thresholds=THRESHOLDS,
            top_percent=1.0,
        )
        assert entry.metrics.c99 == pytest.approx(expected_metrics.c99)
        assert entry.metrics.cmax == pytest.approx(expected_metrics.cmax)


def test_source_area_inferred_from_ray_grid_config() -> None:
    cfg = _ray_grid_config()
    expected_source_area = (
        (cfg["x_range"][1] - cfg["x_range"][0])
        * (cfg["y_range"][1] - cfg["y_range"][0])
    )
    result = run_baseline_angle_scan(**_common_kwargs(
        use_relative_irradiance=True,
    ))
    for entry in result.per_angle:
        assert entry.irradiance_surrogate.source_area == pytest.approx(
            expected_source_area
        )


def test_explicit_source_area_overrides_inferred_value() -> None:
    cfg = _ray_grid_config()
    inferred = (
        (cfg["x_range"][1] - cfg["x_range"][0])
        * (cfg["y_range"][1] - cfg["y_range"][0])
    )
    explicit_value = inferred * 2.0 + 7.0
    result = run_baseline_angle_scan(**_common_kwargs(
        use_relative_irradiance=True,
        source_area=explicit_value,
    ))
    for entry in result.per_angle:
        assert entry.irradiance_surrogate.source_area == pytest.approx(
            explicit_value
        )
        assert entry.irradiance_surrogate.source_area != pytest.approx(
            inferred
        )


def test_invalid_explicit_source_area_raises() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises((AngleScanError, MetricsError)):
            run_baseline_angle_scan(**_common_kwargs(
                use_relative_irradiance=True,
                source_area=bad,
            ))


def test_invalid_ray_grid_ranges_raise_in_relative_mode() -> None:
    # Inverted range: x_max == x_min, source-plane area would be zero.
    bad_range_cfg = _ray_grid_config()
    bad_range_cfg["x_range"] = (5.0, 5.0)
    with pytest.raises(AngleScanError, match="max > min"):
        run_baseline_angle_scan(
            mesh=_slab_mesh(),
            angles_degrees=[0.0],
            ray_grid_config=bad_range_cfg,
            interface_sequence=SLAB_INTERFACES,
            detector=_detector(),
            detector_grid=_detector_grid(),
            thresholds=THRESHOLDS,
            use_relative_irradiance=True,
        )


def test_invalid_ray_grid_ranges_silent_in_unweighted_mode() -> None:
    # Same malformed range but with use_relative_irradiance=False:
    # ray_grid_config validation only kicks in for the relative path.
    # parallel_ray_grid will still validate its own kwargs, so use a
    # valid ray_grid_config and only confirm that no source_area
    # validation is performed.
    cfg = _ray_grid_config()
    # Pass a bogus source_area; it should be ignored when relative mode
    # is off.
    result = run_baseline_angle_scan(
        mesh=_slab_mesh(),
        angles_degrees=[0.0],
        ray_grid_config=cfg,
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
        use_relative_irradiance=False,
        source_area=-1.0,  # ignored
    )
    assert result.per_angle[0].irradiance_surrogate is None


def test_use_relative_irradiance_with_use_power_weights_smoke() -> None:
    weighted_relative = run_baseline_angle_scan(**_common_kwargs(
        use_power_weights=True,
        use_relative_irradiance=True,
    ))
    for entry in weighted_relative.per_angle:
        assert isinstance(
            entry.irradiance_surrogate, DetectorIrradianceSurrogate
        )
        assert entry.metrics.c99 >= 0.0
        assert entry.metrics.cmax >= 0.0


def test_relative_mode_does_not_mutate_mesh_or_ray_grid_config() -> None:
    mesh = _slab_mesh()
    cfg = _ray_grid_config()
    cfg["direction"] = (0.5, 0.5, -0.5)
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    cfg_before = dict(cfg)

    run_baseline_angle_scan(
        mesh=mesh,
        angles_degrees=[0.0, 10.0],
        ray_grid_config=cfg,
        interface_sequence=SLAB_INTERFACES,
        detector=_detector(),
        detector_grid=_detector_grid(),
        thresholds=THRESHOLDS,
        use_power_weights=True,
        use_relative_irradiance=True,
    )

    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)
    assert cfg == cfg_before
