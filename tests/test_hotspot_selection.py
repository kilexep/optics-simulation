import numpy as np
import pytest
import trimesh

from optics_simulation.metrics import (
    HotspotSelection,
    MetricsError,
    select_hotspot_rays,
)
from optics_simulation.optics import (
    DetectorAccumulationResult,
    DetectorHitResult,
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


def _make_synthetic_inputs(
    weight_map: np.ndarray,
    ray_pixels: list[tuple[int, int] | None],
) -> tuple[DetectorHitResult, DetectorAccumulationResult]:
    """Build a matching (DetectorHitResult, DetectorAccumulationResult).

    ``ray_pixels[i]`` is ``(row, col)`` for hit rays, ``None`` for
    miss rays. ``weight_map`` is taken as-is (caller controls
    pixel values directly). ``count_map`` is rebuilt from the
    actual ``ray_pixels`` so the accumulation invariants stay
    consistent.
    """
    ny, nx = weight_map.shape
    n = len(ray_pixels)
    hit_mask = np.zeros(n, dtype=bool)
    hit_pixel_indices = np.full((n, 2), -1, dtype=np.int64)
    count_map = np.zeros((ny, nx), dtype=np.int64)
    for i, rp in enumerate(ray_pixels):
        if rp is not None:
            hit_mask[i] = True
            hit_pixel_indices[i, 0] = rp[0]
            hit_pixel_indices[i, 1] = rp[1]
            count_map[rp[0], rp[1]] += 1

    hits = DetectorHitResult(
        t_hit=np.where(hit_mask, 1.0, np.inf),
        hit_mask=hit_mask,
        hit_points=np.full((n, 3), np.nan, dtype=float),
        local_xy=np.full((n, 2), np.nan, dtype=float),
        ray_count=n,
    )
    accum = DetectorAccumulationResult(
        count_map=count_map,
        weight_map=np.asarray(weight_map, dtype=float).copy(),
        hit_pixel_indices=hit_pixel_indices,
        hit_mask=np.array(hit_mask, dtype=bool, copy=True),
        total_hits=int(hit_mask.sum()),
        total_weight=float(weight_map.sum()),
    )
    return hits, accum


def test_threshold_mode_selects_above_threshold() -> None:
    weight_map = np.array(
        [[0.0, 1.0, 5.0],
         [2.0, 8.0, 0.0]],
        dtype=float,
    )
    ray_pixels = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, threshold=1.5)
    assert isinstance(sel, HotspotSelection)
    assert sel.selection_mode == "threshold"
    expected_pixel_mask = np.array(
        [[False, False, True], [True, True, False]], dtype=bool
    )
    assert np.array_equal(sel.pixel_mask, expected_pixel_mask)
    assert sel.selected_count == 3
    assert sorted(sel.selected_ray_indices.tolist()) == [2, 3, 4]


def test_threshold_mode_uses_strict_greater_than() -> None:
    weight_map = np.array([[2.0, 2.0, 3.0]], dtype=float)
    ray_pixels = [(0, 0), (0, 1), (0, 2)]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, threshold=2.0)
    assert sel.selected_count == 1
    assert sel.selected_ray_indices.tolist() == [2]


def test_threshold_mode_with_count_map_when_use_weight_map_false() -> None:
    weight_map = np.array(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=float,
    )
    ray_pixels = [(0, 0), (0, 0), (1, 2), None]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(
        hits, accum, threshold=0.5, use_weight_map=False
    )
    expected_pixel_mask = np.zeros((2, 3), dtype=bool)
    expected_pixel_mask[0, 0] = True
    expected_pixel_mask[1, 2] = True
    assert np.array_equal(sel.pixel_mask, expected_pixel_mask)
    assert sorted(sel.selected_ray_indices.tolist()) == [0, 1, 2]


def test_top_percent_selects_at_least_one_nonzero() -> None:
    weight_map = np.array(
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 2.0, 0.0],
         [0.0, 3.0, 4.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        dtype=float,
    )
    ray_pixels = [(1, 1), (1, 2), (2, 1), (2, 2), (0, 0)]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, top_percent=10.0)
    assert sel.selection_mode == "top_percent"
    assert sel.selected_count >= 1
    assert (weight_map[sel.pixel_mask] > 0.0).all()


def test_top_percent_all_zero_map_selects_none() -> None:
    weight_map = np.zeros((3, 3), dtype=float)
    ray_pixels = [(0, 0), (1, 1), (2, 2)]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, top_percent=10.0)
    assert sel.selection_mode == "top_percent"
    assert sel.selected_count == 0
    assert not sel.pixel_mask.any()
    assert not sel.selected_mask.any()


def test_miss_rays_never_selected() -> None:
    weight_map = np.array([[10.0, 10.0]], dtype=float)
    ray_pixels = [(0, 0), None, (0, 1), None]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, threshold=0.5)
    assert sel.selected_mask[1] == False  # noqa: E712
    assert sel.selected_mask[3] == False  # noqa: E712
    assert set(sel.selected_ray_indices.tolist()) == {0, 2}


def test_selected_mask_shape_equals_ray_count() -> None:
    weight_map = np.array([[1.0, 0.0]], dtype=float)
    ray_pixels = [(0, 0), (0, 1), None]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, threshold=0.5)
    assert sel.selected_mask.shape == (hits.ray_count,)
    assert sel.ray_count == hits.ray_count


def test_selected_ray_indices_equals_where_mask() -> None:
    weight_map = np.array([[3.0, 1.0, 5.0]], dtype=float)
    ray_pixels = [(0, 0), (0, 1), (0, 2)]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, threshold=2.0)
    assert np.array_equal(
        sel.selected_ray_indices, np.flatnonzero(sel.selected_mask)
    )


def test_selected_pixel_indices_match_hit_pixel_indices() -> None:
    weight_map = np.array(
        [[0.0, 5.0],
         [3.0, 0.0]],
        dtype=float,
    )
    ray_pixels = [(0, 1), (1, 0), (0, 0), None]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, threshold=1.0)
    expected = accum.hit_pixel_indices[sel.selected_ray_indices]
    assert np.array_equal(sel.selected_pixel_indices, expected)


def test_selected_count_matches_indices_and_mask() -> None:
    weight_map = np.array([[2.0, 4.0, 6.0]], dtype=float)
    ray_pixels = [(0, 0), (0, 1), (0, 2)]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, threshold=1.0)
    assert sel.selected_count == len(sel.selected_ray_indices)
    assert sel.selected_count == int(sel.selected_mask.sum())


def test_selected_weights_are_all_ones_placeholder() -> None:
    weight_map = np.array([[1.0, 2.0, 3.0]], dtype=float)
    ray_pixels = [(0, 0), (0, 1), (0, 2)]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    sel = select_hotspot_rays(hits, accum, threshold=0.5)
    assert sel.selected_weights.shape == (sel.selected_count,)
    assert np.all(sel.selected_weights == 1.0)


def test_invalid_threshold_raises() -> None:
    weight_map = np.array([[1.0]], dtype=float)
    hits, accum = _make_synthetic_inputs(weight_map, [(0, 0)])

    with pytest.raises(MetricsError, match="threshold"):
        select_hotspot_rays(hits, accum, threshold=0.0)
    with pytest.raises(MetricsError, match="threshold"):
        select_hotspot_rays(hits, accum, threshold=-1.0)


def test_invalid_top_percent_raises() -> None:
    weight_map = np.array([[1.0]], dtype=float)
    hits, accum = _make_synthetic_inputs(weight_map, [(0, 0)])

    for bad in (0.0, -5.0, 100.5):
        with pytest.raises(MetricsError, match="top_percent"):
            select_hotspot_rays(hits, accum, top_percent=bad)


def test_both_threshold_and_top_percent_raises() -> None:
    weight_map = np.array([[1.0]], dtype=float)
    hits, accum = _make_synthetic_inputs(weight_map, [(0, 0)])

    with pytest.raises(MetricsError, match="exactly one"):
        select_hotspot_rays(hits, accum, threshold=1.0, top_percent=10.0)


def test_neither_threshold_nor_top_percent_raises() -> None:
    weight_map = np.array([[1.0]], dtype=float)
    hits, accum = _make_synthetic_inputs(weight_map, [(0, 0)])

    with pytest.raises(MetricsError, match="exactly one"):
        select_hotspot_rays(hits, accum)


def test_mismatched_hit_masks_raises() -> None:
    weight_map = np.array([[1.0, 2.0]], dtype=float)
    ray_pixels = [(0, 0), (0, 1)]
    hits, accum = _make_synthetic_inputs(weight_map, ray_pixels)

    flipped_mask = np.array(accum.hit_mask, copy=True)
    flipped_mask[0] = not bool(flipped_mask[0])
    bad_accum = DetectorAccumulationResult(
        count_map=accum.count_map,
        weight_map=accum.weight_map,
        hit_pixel_indices=accum.hit_pixel_indices,
        hit_mask=flipped_mask,
        total_hits=accum.total_hits,
        total_weight=accum.total_weight,
    )
    with pytest.raises(MetricsError, match="hit_mask"):
        select_hotspot_rays(hits, bad_accum, threshold=0.5)


def test_mismatched_hit_mask_length_raises() -> None:
    weight_map = np.array([[1.0]], dtype=float)
    hits, accum = _make_synthetic_inputs(weight_map, [(0, 0), (0, 0)])

    short_accum = DetectorAccumulationResult(
        count_map=accum.count_map,
        weight_map=accum.weight_map,
        hit_pixel_indices=accum.hit_pixel_indices,
        hit_mask=accum.hit_mask[:1],
        total_hits=accum.total_hits,
        total_weight=accum.total_weight,
    )
    with pytest.raises(MetricsError, match="length"):
        select_hotspot_rays(hits, short_accum, threshold=0.5)


def test_integration_smoke_with_detector_demo_pipeline() -> None:
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-7.5, 7.5),
        y_range=(-7.5, 7.5),
        nx=11,
        ny=11,
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

    sel = select_hotspot_rays(hits, accum, top_percent=10.0)
    assert isinstance(sel, HotspotSelection)
    assert sel.selection_mode == "top_percent"
    assert sel.selected_mask.shape == (hits.ray_count,)
    assert sel.selected_count >= 0
    assert sel.selected_count == int(sel.selected_mask.sum())
    assert not sel.selected_mask[~hits.hit_mask].any()
