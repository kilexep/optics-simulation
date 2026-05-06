import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from optics_simulation.angle_scan import (
    AngleDistanceScanResult,
    AngleScanError,
    PerAngleDistanceResult,
    direction_from_incident_angle,
    run_angle_distance_sweep,
)
from optics_simulation.metrics import OpticalMetrics
from optics_simulation.optics import OpticsError


IOR_AIR = 1.00028
IOR_PET = 1.575
SLAB_INTERFACES = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]
THRESHOLDS = (2.0, 5.0, 10.0)
RAY_GRID_NX = 7
RAY_GRID_NY = 7
EXPECTED_RAY_COUNT = RAY_GRID_NX * RAY_GRID_NY


def _slab_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))


def _ray_grid_config() -> dict:
    return {
        "origin_plane_z": 20.0,
        "x_range": (-7.5, 7.5),
        "y_range": (-7.5, 7.5),
        "nx": RAY_GRID_NX,
        "ny": RAY_GRID_NY,
    }


def _detector_kwargs(width: float = 10.0, height: float = 10.0,
                     resolution: tuple[int, int] = (10, 10)) -> dict:
    return {
        "detector_width": width,
        "detector_height": height,
        "detector_resolution": resolution,
    }


def _run(
    *,
    angles: list[float],
    distances: list[float],
    mesh: trimesh.Trimesh | None = None,
    ray_grid_config: dict | None = None,
    thresholds: tuple[float, ...] = THRESHOLDS,
    detector_overrides: dict | None = None,
) -> AngleDistanceScanResult:
    return run_angle_distance_sweep(
        mesh=mesh if mesh is not None else _slab_mesh(),
        angles_degrees=angles,
        detector_z_values=distances,
        ray_grid_config=ray_grid_config if ray_grid_config is not None
        else _ray_grid_config(),
        interface_sequence=SLAB_INTERFACES,
        thresholds=thresholds,
        **(detector_overrides if detector_overrides is not None
           else _detector_kwargs()),
    )


def test_returns_angle_distance_scan_result_type() -> None:
    result = _run(angles=[0.0], distances=[-15.0])
    assert isinstance(result, AngleDistanceScanResult)
    assert isinstance(result.per_result[0], PerAngleDistanceResult)


def test_empty_angles_returns_empty_result() -> None:
    result = _run(angles=[], distances=[-15.0, -20.0])
    assert result.per_result == ()
    assert result.angle_count == 0
    assert result.detector_count == 2
    assert result.max_c99_angle is None
    assert result.max_c99_detector_z is None
    assert result.max_c99 is None


def test_empty_detector_z_values_returns_empty_result() -> None:
    result = _run(angles=[0.0], distances=[])
    assert result.per_result == ()
    assert result.angle_count == 1
    assert result.detector_count == 0
    assert result.max_c99_angle is None
    assert result.max_c99_detector_z is None
    assert result.max_c99 is None


def test_single_angle_single_distance_returns_one_result() -> None:
    result = _run(angles=[0.0], distances=[-15.0])
    assert result.angle_count == 1
    assert result.detector_count == 1
    assert len(result.per_result) == 1
    entry = result.per_result[0]
    assert entry.angle_degrees == 0.0
    assert entry.detector_z == -15.0
    assert isinstance(entry.metrics, OpticalMetrics)


def test_multiple_angles_and_distances_total_count() -> None:
    result = _run(angles=[0.0, 10.0], distances=[-15.0, -20.0, -25.0])
    assert result.angle_count == 2
    assert result.detector_count == 3
    assert len(result.per_result) == 2 * 3


def test_result_order_is_angle_major() -> None:
    result = _run(angles=[0.0, 10.0], distances=[-15.0, -20.0, -25.0])
    pairs = [(p.angle_degrees, p.detector_z) for p in result.per_result]
    assert pairs == [
        (0.0, -15.0), (0.0, -20.0), (0.0, -25.0),
        (10.0, -15.0), (10.0, -20.0), (10.0, -25.0),
    ]


def test_duplicate_detector_z_preserved_in_input_order() -> None:
    result = _run(angles=[0.0], distances=[-15.0, -20.0, -15.0])
    assert [p.detector_z for p in result.per_result] == [-15.0, -20.0, -15.0]


def test_direction_matches_helper() -> None:
    angles = [0.0, 12.0, 25.0]
    result = _run(angles=angles, distances=[-15.0, -20.0])
    for entry in result.per_result:
        assert np.linalg.norm(entry.direction) == pytest.approx(1.0)
        assert np.allclose(
            entry.direction,
            direction_from_incident_angle(entry.angle_degrees),
        )


def test_detector_z_is_preserved() -> None:
    distances = [-15.0, -22.5, -30.0]
    result = _run(angles=[0.0, 10.0], distances=distances)
    seen = [p.detector_z for p in result.per_result]
    assert seen == distances + distances


def test_each_metrics_is_optical_metrics() -> None:
    result = _run(angles=[0.0, 10.0], distances=[-15.0, -20.0])
    for entry in result.per_result:
        assert isinstance(entry.metrics, OpticalMetrics)


def test_detector_hits_nonneg() -> None:
    result = _run(angles=[0.0, 10.0], distances=[-15.0, -25.0])
    for entry in result.per_result:
        assert entry.detector_hits >= 0


def test_final_ray_count_within_bounds() -> None:
    result = _run(angles=[0.0, 10.0], distances=[-15.0, -25.0])
    for entry in result.per_result:
        assert 0 <= entry.final_ray_count <= entry.ray_count


def test_ray_count_matches_grid() -> None:
    result = _run(angles=[0.0, 10.0], distances=[-15.0, -25.0])
    for entry in result.per_result:
        assert entry.ray_count == EXPECTED_RAY_COUNT


def test_max_c99_matches_argmax_over_per_result() -> None:
    result = _run(angles=[0.0, 5.0, 10.0], distances=[-15.0, -20.0, -25.0])
    c99_arr = np.array(
        [p.metrics.c99 for p in result.per_result], dtype=float
    )
    idx = int(np.argmax(c99_arr))
    chosen = result.per_result[idx]
    assert result.max_c99 == pytest.approx(float(c99_arr[idx]))
    assert result.max_c99_angle == chosen.angle_degrees
    assert result.max_c99_detector_z == chosen.detector_z


def test_tie_break_uses_first_occurrence() -> None:
    # Reuse the same z twice; deterministically the c99 of the first and
    # third entries are equal, and np.argmax must pick the first.
    result = _run(angles=[0.0], distances=[-15.0, -25.0, -15.0])
    c99_arr = np.array(
        [p.metrics.c99 for p in result.per_result], dtype=float
    )
    # First and third entries see identical detector setups; their c99
    # values must match.
    assert c99_arr[0] == pytest.approx(c99_arr[2])
    expected_idx = int(np.argmax(c99_arr))
    assert expected_idx == 0  # first occurrence wins on a tie
    assert result.max_c99_detector_z == result.per_result[0].detector_z


def test_thresholds_propagate_into_metrics() -> None:
    custom_thresholds = (1.5, 3.0, 7.5)
    result = _run(
        angles=[0.0, 10.0],
        distances=[-15.0, -25.0],
        thresholds=custom_thresholds,
    )
    for entry in result.per_result:
        assert entry.metrics.thresholds == custom_thresholds
        assert tuple(entry.metrics.eexceed.keys()) == custom_thresholds
        assert tuple(entry.metrics.ahot.keys()) == custom_thresholds


def test_invalid_angle_nan_inf_raises() -> None:
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(AngleScanError, match="angles_degrees"):
            _run(angles=[0.0, bad], distances=[-15.0])


def test_invalid_detector_z_nan_inf_raises() -> None:
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(AngleScanError, match="detector_z_values"):
            _run(angles=[0.0], distances=[-15.0, bad])


def test_invalid_detector_center_xy_raises() -> None:
    with pytest.raises(AngleScanError, match="detector_center_xy"):
        run_angle_distance_sweep(
            mesh=_slab_mesh(),
            angles_degrees=[0.0],
            detector_z_values=[-15.0],
            ray_grid_config=_ray_grid_config(),
            interface_sequence=SLAB_INTERFACES,
            detector_width=10.0,
            detector_height=10.0,
            detector_resolution=(10, 10),
            detector_center_xy=(0.0, 0.0, 0.0),  # wrong length
        )
    with pytest.raises(AngleScanError, match="detector_center_xy"):
        run_angle_distance_sweep(
            mesh=_slab_mesh(),
            angles_degrees=[0.0],
            detector_z_values=[-15.0],
            ray_grid_config=_ray_grid_config(),
            interface_sequence=SLAB_INTERFACES,
            detector_width=10.0,
            detector_height=10.0,
            detector_resolution=(10, 10),
            detector_center_xy=(0.0, float("nan")),
        )


def test_invalid_detector_geometry_propagates_validator_error() -> None:
    with pytest.raises((OpticsError, AngleScanError)):
        _run(
            angles=[0.0],
            distances=[-15.0],
            detector_overrides=_detector_kwargs(width=-1.0),
        )
    with pytest.raises((OpticsError, AngleScanError)):
        _run(
            angles=[0.0],
            distances=[-15.0],
            detector_overrides=_detector_kwargs(resolution=(0, 10)),
        )


def test_mesh_is_not_mutated() -> None:
    mesh = _slab_mesh()
    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)
    _run(
        mesh=mesh,
        angles=[0.0, 10.0],
        distances=[-15.0, -25.0],
    )
    assert np.array_equal(np.asarray(mesh.vertices), vertices_before)
    assert np.array_equal(np.asarray(mesh.faces), faces_before)


def test_ray_grid_config_is_not_mutated() -> None:
    cfg = _ray_grid_config()
    cfg["direction"] = (0.5, 0.5, -0.5)  # something the function should override
    cfg_before = dict(cfg)
    _run(
        angles=[0.0, 10.0],
        distances=[-15.0, -25.0],
        ray_grid_config=cfg,
    )
    assert cfg == cfg_before


def test_normal_incidence_smoke_has_nonneg_hits() -> None:
    result = _run(angles=[0.0], distances=[-15.0])
    assert result.per_result[0].detector_hits >= 0


def test_no_file_output_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    _run(angles=[0.0, 10.0], distances=[-15.0, -25.0])
    after = set(os.listdir(tmp_path))
    assert before == after
