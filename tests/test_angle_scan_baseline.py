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
