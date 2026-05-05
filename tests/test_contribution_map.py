import numpy as np
import pytest

from optics_simulation.contribution import (
    ContributionError,
    ContributionMap,
    create_contribution_map,
)
from optics_simulation.geometry import (
    HitSurfaceCoordinates,
    create_synthetic_bottle_body,
    create_vertex_surface_coordinates,
    hit_to_surface_coordinates,
)
from optics_simulation.metrics.hotspot_selection import HotspotSelection
from optics_simulation.optics import (
    intersect_rays,
    parallel_ray_grid,
)


def _make_hit_coords(
    u: np.ndarray,
    v: np.ndarray,
) -> HitSurfaceCoordinates:
    n = int(u.size)
    return HitSurfaceCoordinates(
        u=np.asarray(u, dtype=float),
        v=np.asarray(v, dtype=float),
        barycentric_weights=np.full((n, 3), np.nan, dtype=float),
        hit_mask=~(np.isnan(u) | np.isnan(v)),
        primitive_ids=np.full(n, -1, dtype=np.int64),
        ray_count=n,
    )


def _make_selection(
    *,
    selected_indices: np.ndarray,
    weights: np.ndarray,
    ray_count: int,
) -> HotspotSelection:
    indices = np.asarray(selected_indices, dtype=np.int64)
    selected_mask = np.zeros(ray_count, dtype=bool)
    if indices.size > 0:
        selected_mask[indices] = True
    return HotspotSelection(
        selected_mask=selected_mask,
        selected_ray_indices=indices,
        selected_pixel_indices=np.full((indices.size, 2), -1, dtype=np.int64),
        selected_weights=np.asarray(weights, dtype=float),
        pixel_mask=np.zeros((1, 1), dtype=bool),
        ray_count=int(ray_count),
        selected_count=int(indices.size),
        selection_mode="threshold",
    )


def test_simple_points_bin_into_expected_cells() -> None:
    u = np.array([0.0, 0.5, 0.9])
    v = np.array([0.0, 0.5, 1.0])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0, 1, 2]),
        weights=np.array([1.0, 1.0, 1.0]),
        ray_count=3,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4))
    assert isinstance(cm, ContributionMap)
    assert cm.count_map.shape == (4, 4)
    assert cm.count_map[0, 0] == 1
    assert cm.count_map[2, 2] == 1
    assert cm.count_map[3, 3] == 1
    assert cm.total_selected == 3


def test_u_zero_maps_to_col_zero() -> None:
    hits = _make_hit_coords(np.array([0.0]), np.array([0.5]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 8))
    assert int(cm.u_bin_indices[0]) == 0


def test_u_close_to_one_maps_to_last_col() -> None:
    u_val = float(np.nextafter(1.0, 0.0))
    hits = _make_hit_coords(np.array([u_val]), np.array([0.5]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 8))
    assert int(cm.u_bin_indices[0]) == 7


def test_v_zero_maps_to_row_zero() -> None:
    hits = _make_hit_coords(np.array([0.5]), np.array([0.0]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 8))
    assert int(cm.v_bin_indices[0]) == 0


def test_v_one_maps_to_last_row() -> None:
    hits = _make_hit_coords(np.array([0.5]), np.array([1.0]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 8))
    assert int(cm.v_bin_indices[0]) == 3


def test_duplicate_rays_increment_count() -> None:
    u = np.array([0.25, 0.25, 0.25])
    v = np.array([0.75, 0.75, 0.75])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0, 1, 2]),
        weights=np.array([1.0, 1.0, 1.0]),
        ray_count=3,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4))
    nonzero = np.argwhere(cm.count_map > 0)
    assert nonzero.shape[0] == 1
    r, c = int(nonzero[0, 0]), int(nonzero[0, 1])
    assert cm.count_map[r, c] == 3


def test_weights_accumulate_correctly() -> None:
    u = np.array([0.25, 0.25])
    v = np.array([0.75, 0.75])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0, 1]),
        weights=np.array([1.5, 2.5]),
        ray_count=2,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4))
    assert float(cm.weight_map.sum()) == pytest.approx(4.0)
    assert cm.total_weight == pytest.approx(4.0)


def test_negative_selected_weight_raises() -> None:
    hits = _make_hit_coords(np.array([0.5]), np.array([0.5]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([-0.5]),
        ray_count=1,
    )
    with pytest.raises(ContributionError, match="negative"):
        create_contribution_map(sel, hits, resolution=(4, 4))


def test_inf_selected_u_raises() -> None:
    hits = _make_hit_coords(np.array([np.inf]), np.array([0.5]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    with pytest.raises(ContributionError, match="inf u"):
        create_contribution_map(sel, hits, resolution=(4, 4))


def test_inf_selected_v_raises() -> None:
    hits = _make_hit_coords(np.array([0.5]), np.array([np.inf]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    with pytest.raises(ContributionError, match="inf v"):
        create_contribution_map(sel, hits, resolution=(4, 4))


def test_selected_ray_with_nan_uv_is_skipped() -> None:
    u = np.array([np.nan, 0.5])
    v = np.array([0.5, 0.5])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0, 1]),
        weights=np.array([1.0, 1.0]),
        ray_count=2,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4))
    assert cm.total_selected == 1
    assert int(cm.count_map.sum()) == 1
    assert cm.selected_ray_indices.tolist() == [1]


def test_nonselected_nan_uv_does_not_matter() -> None:
    u = np.array([0.5, np.nan, np.inf])
    v = np.array([0.5, np.nan, 0.5])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([2.0]),
        ray_count=3,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4))
    assert cm.total_selected == 1
    assert cm.total_weight == pytest.approx(2.0)


def test_normalized_map_max_is_one_for_positive_weights() -> None:
    u = np.array([0.1, 0.6, 0.6])
    v = np.array([0.1, 0.6, 0.6])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0, 1, 2]),
        weights=np.array([1.0, 1.0, 1.0]),
        ray_count=3,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4), normalize=True)
    assert float(cm.normalized_map.max()) == pytest.approx(1.0)


def test_normalized_map_is_zero_when_no_valid_rays() -> None:
    u = np.array([np.nan, np.nan])
    v = np.array([np.nan, np.nan])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0, 1]),
        weights=np.array([1.0, 1.0]),
        ray_count=2,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4), normalize=True)
    assert float(cm.normalized_map.max()) == 0.0
    assert cm.total_selected == 0


def test_normalize_false_returns_raw_weight_map_copy() -> None:
    u = np.array([0.5])
    v = np.array([0.5])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([7.5]),
        ray_count=1,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4), normalize=False)
    assert np.array_equal(cm.normalized_map, cm.weight_map)
    assert cm.normalized_map is not cm.weight_map
    assert float(cm.normalized_map.max()) == pytest.approx(7.5)


def test_mismatched_ray_count_raises() -> None:
    hits = _make_hit_coords(np.array([0.1, 0.2]), np.array([0.1, 0.2]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=3,
    )
    with pytest.raises(ContributionError, match="ray_count"):
        create_contribution_map(sel, hits, resolution=(4, 4))


def test_invalid_resolution_raises() -> None:
    hits = _make_hit_coords(np.array([0.5]), np.array([0.5]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    for bad in [(0, 5), (-1, 5), (5, 0), (5, -2)]:
        with pytest.raises(ContributionError, match="resolution"):
            create_contribution_map(sel, hits, resolution=bad)
    with pytest.raises(ContributionError, match="resolution"):
        create_contribution_map(sel, hits, resolution=(5,))
    with pytest.raises(ContributionError, match="resolution"):
        create_contribution_map(sel, hits, resolution=(5, 5, 5))


def test_selected_weights_shape_mismatch_raises() -> None:
    hits = _make_hit_coords(np.array([0.5, 0.6]), np.array([0.5, 0.6]))
    sel = HotspotSelection(
        selected_mask=np.array([True, True]),
        selected_ray_indices=np.array([0, 1], dtype=np.int64),
        selected_pixel_indices=np.full((2, 2), -1, dtype=np.int64),
        selected_weights=np.array([1.0]),
        pixel_mask=np.zeros((1, 1), dtype=bool),
        ray_count=2,
        selected_count=2,
        selection_mode="threshold",
    )
    with pytest.raises(ContributionError, match="selected_weights"):
        create_contribution_map(sel, hits, resolution=(4, 4))


def test_selected_ray_indices_out_of_range_raises() -> None:
    hits = _make_hit_coords(np.array([0.5, 0.6, 0.7]), np.array([0.5, 0.6, 0.7]))
    sel = HotspotSelection(
        selected_mask=np.zeros(3, dtype=bool),
        selected_ray_indices=np.array([5], dtype=np.int64),
        selected_pixel_indices=np.full((1, 2), -1, dtype=np.int64),
        selected_weights=np.array([1.0]),
        pixel_mask=np.zeros((1, 1), dtype=bool),
        ray_count=3,
        selected_count=1,
        selection_mode="threshold",
    )
    with pytest.raises(ContributionError, match="out-of-range"):
        create_contribution_map(sel, hits, resolution=(4, 4))


def test_u_out_of_range_raises() -> None:
    hits = _make_hit_coords(np.array([1.5]), np.array([0.5]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    with pytest.raises(ContributionError, match=r"u outside"):
        create_contribution_map(sel, hits, resolution=(4, 4))


def test_v_out_of_range_raises() -> None:
    hits = _make_hit_coords(np.array([0.5]), np.array([-0.1]))
    sel = _make_selection(
        selected_indices=np.array([0]),
        weights=np.array([1.0]),
        ray_count=1,
    )
    with pytest.raises(ContributionError, match=r"v outside"):
        create_contribution_map(sel, hits, resolution=(4, 4))


def test_compact_output_lengths_match_total_selected() -> None:
    u = np.array([0.1, np.nan, 0.6])
    v = np.array([0.1, 0.5, 0.6])
    hits = _make_hit_coords(u, v)
    sel = _make_selection(
        selected_indices=np.array([0, 1, 2]),
        weights=np.array([1.0, 1.0, 1.0]),
        ray_count=3,
    )
    cm = create_contribution_map(sel, hits, resolution=(4, 4))
    assert cm.selected_ray_indices.shape[0] == cm.total_selected
    assert cm.u_bin_indices.shape[0] == cm.total_selected
    assert cm.v_bin_indices.shape[0] == cm.total_selected
    assert float(cm.weight_map.sum()) == pytest.approx(cm.total_weight)


def test_integration_smoke_with_synthetic_bottle_pipeline() -> None:
    # u/v binning compatibility test only — this is NOT an actual
    # detector hotspot contribution. The synthetic HotspotSelection
    # is built from intersection.hit_mask, not from a true hot
    # detector pixel selection. No pattern generation is performed.
    mesh = create_synthetic_bottle_body(radius=30.0, height=120.0, sections=48)
    rays = parallel_ray_grid(
        origin_plane_z=200.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-50.0, 50.0),
        y_range=(-70.0, 70.0),
        nx=7,
        ny=7,
    )
    intersection = intersect_rays(mesh, rays)
    surface_map = create_vertex_surface_coordinates(mesh)
    hit_coords = hit_to_surface_coordinates(mesh, intersection, surface_map)

    selected_indices = np.flatnonzero(intersection.hit_mask).astype(np.int64)
    weights = np.ones(selected_indices.size, dtype=float)
    selection = HotspotSelection(
        selected_mask=np.array(intersection.hit_mask, dtype=bool, copy=True),
        selected_ray_indices=selected_indices,
        selected_pixel_indices=np.full(
            (selected_indices.size, 2), -1, dtype=np.int64
        ),
        selected_weights=weights,
        pixel_mask=np.zeros((1, 1), dtype=bool),
        ray_count=int(rays.ray_count),
        selected_count=int(selected_indices.size),
        selection_mode="threshold",
    )

    cm = create_contribution_map(selection, hit_coords, resolution=(32, 64))
    assert isinstance(cm, ContributionMap)
    assert cm.count_map.shape == (32, 64)
    assert cm.total_selected >= 0
    assert cm.total_selected <= int(selection.selected_count)
    assert float(cm.weight_map.sum()) == pytest.approx(cm.total_weight)
