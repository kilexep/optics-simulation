import numpy as np
import pytest
import trimesh

from optics_simulation.optics import (
    DetectorAccumulationResult,
    DetectorGrid,
    DetectorHitResult,
    OpticsError,
    accumulate_detector_hits,
    create_detector_grid,
    create_detector_plane,
    intersect_detector_plane,
    make_ray_bundle,
    parallel_ray_grid,
    run_multi_step_trace,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
SLAB_INTERFACES = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]


def _grid_10x10(nx: int = 10, ny: int = 10, w: float = 10.0, h: float = 10.0) -> DetectorGrid:
    return create_detector_grid(width=w, height=h, resolution=(ny, nx))


def _hits_from_local_xy(
    local_xy_list: list[tuple[float, float]],
    hit_flags: list[bool] | None = None,
) -> DetectorHitResult:
    """Synthesize a DetectorHitResult straight from (local_x, local_y) pairs.

    Bypasses ray-plane geometry so tests can lock pixel-mapping
    behavior directly. Miss rows carry NaN local_xy as per the
    detector-plane sentinel convention.
    """
    n = len(local_xy_list)
    flags = [True] * n if hit_flags is None else hit_flags
    assert len(flags) == n
    local_xy = np.full((n, 2), np.nan, dtype=float)
    hit_points = np.full((n, 3), np.nan, dtype=float)
    t_hit = np.full(n, np.inf, dtype=float)
    hit_mask = np.zeros(n, dtype=bool)
    for i, (xy, hit) in enumerate(zip(local_xy_list, flags)):
        if hit:
            local_xy[i, 0] = xy[0]
            local_xy[i, 1] = xy[1]
            hit_points[i] = (xy[0], xy[1], 0.0)
            t_hit[i] = 1.0
            hit_mask[i] = True
    return DetectorHitResult(
        t_hit=t_hit,
        hit_mask=hit_mask,
        hit_points=hit_points,
        local_xy=local_xy,
        ray_count=n,
    )


def test_create_grid_pixel_dimensions() -> None:
    grid = create_detector_grid(width=10.0, height=20.0, resolution=(40, 20))
    assert grid.width == 10.0
    assert grid.height == 20.0
    assert grid.resolution == (40, 20)
    assert grid.pixel_width == pytest.approx(0.5)
    assert grid.pixel_height == pytest.approx(0.5)


def test_create_grid_invalid_inputs_raise() -> None:
    with pytest.raises(OpticsError, match="positive"):
        create_detector_grid(width=0.0, height=10.0, resolution=(10, 10))
    with pytest.raises(OpticsError, match="positive"):
        create_detector_grid(width=10.0, height=-1.0, resolution=(10, 10))
    with pytest.raises(OpticsError, match="resolution"):
        create_detector_grid(width=10.0, height=10.0, resolution=(0, 10))
    with pytest.raises(OpticsError, match="resolution"):
        create_detector_grid(width=10.0, height=10.0, resolution=(10, -1))
    with pytest.raises(OpticsError, match="resolution"):
        create_detector_grid(width=10.0, height=10.0, resolution=(10,))  # type: ignore[arg-type]


def test_center_hit_lands_in_center_pixel() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(0.0, 0.0)])
    result = accumulate_detector_hits(hits, grid)
    assert isinstance(result, DetectorAccumulationResult)
    # local_x=0, half_w=5, pixel_width=1.0 -> col = floor(5/1) = 5
    assert result.hit_pixel_indices[0, 0] == 5
    assert result.hit_pixel_indices[0, 1] == 5
    assert result.count_map[5, 5] == 1
    assert result.total_hits == 1
    assert result.total_weight == pytest.approx(1.0)


def test_negative_boundary_maps_to_pixel_zero() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(-5.0, -5.0)])
    result = accumulate_detector_hits(hits, grid)
    assert result.hit_pixel_indices[0, 0] == 0
    assert result.hit_pixel_indices[0, 1] == 0
    assert result.count_map[0, 0] == 1


def test_positive_boundary_clamps_to_last_pixel() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(5.0, 5.0)])
    result = accumulate_detector_hits(hits, grid)
    ny, nx = grid.resolution
    assert result.hit_pixel_indices[0, 0] == ny - 1
    assert result.hit_pixel_indices[0, 1] == nx - 1
    assert result.count_map[ny - 1, nx - 1] == 1


def test_local_x_maps_monotonically_to_columns() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(-2.5, 0.0), (0.0, 0.0), (2.5, 0.0)])
    result = accumulate_detector_hits(hits, grid)
    cols = result.hit_pixel_indices[:, 1]
    assert cols[0] < cols[1] < cols[2]
    # Concrete values: floor((x + 5)/1.0)
    assert cols[0] == 2  # floor(2.5)
    assert cols[1] == 5
    assert cols[2] == 7  # floor(7.5)


def test_local_y_maps_with_row_zero_at_lowest_y() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(0.0, -2.5), (0.0, 0.0), (0.0, 2.5)])
    result = accumulate_detector_hits(hits, grid)
    rows = result.hit_pixel_indices[:, 0]
    # Math-oriented convention: smaller local_y -> smaller row index
    assert rows[0] < rows[1] < rows[2]
    assert rows[0] == 2
    assert rows[1] == 5
    assert rows[2] == 7


def test_miss_rows_get_negative_one_and_are_not_accumulated() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy(
        [(0.0, 0.0), (1.0, 1.0), (-2.0, 2.0)],
        hit_flags=[True, False, True],
    )
    result = accumulate_detector_hits(hits, grid)
    assert result.hit_pixel_indices[0].tolist() != [-1, -1]
    assert result.hit_pixel_indices[1].tolist() == [-1, -1]
    assert result.hit_pixel_indices[2].tolist() != [-1, -1]
    assert result.total_hits == 2
    assert int(result.count_map.sum()) == 2


def test_count_map_increments_for_multiple_rays_in_same_pixel() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
    result = accumulate_detector_hits(hits, grid)
    assert result.count_map[5, 5] == 3
    assert result.total_hits == 3
    assert int(result.count_map.sum()) == 3


def test_weight_map_sums_caller_weights() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
    weights = np.array([2.0, 3.0, 1.5])
    result = accumulate_detector_hits(hits, grid, weights=weights)
    assert result.weight_map[5, 5] == pytest.approx(6.5)
    assert result.total_weight == pytest.approx(6.5)
    assert result.count_map[5, 5] == 3


def test_default_weights_behave_like_ones() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(-2.0, 1.0), (0.0, 0.0), (2.0, -1.0)])
    result = accumulate_detector_hits(hits, grid)
    assert result.weight_map.sum() == pytest.approx(float(result.count_map.sum()))
    assert result.total_weight == pytest.approx(float(result.total_hits))


def test_invalid_weight_shape_raises() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(OpticsError, match="shape"):
        accumulate_detector_hits(hits, grid, weights=np.ones(3))


def test_non_finite_weights_raise() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(OpticsError, match="finite"):
        accumulate_detector_hits(hits, grid, weights=np.array([1.0, np.nan]))
    with pytest.raises(OpticsError, match="finite"):
        accumulate_detector_hits(hits, grid, weights=np.array([1.0, np.inf]))


def test_empty_hit_result_returns_zero_maps() -> None:
    grid = _grid_10x10()
    hits = DetectorHitResult(
        t_hit=np.zeros(0),
        hit_mask=np.zeros(0, dtype=bool),
        hit_points=np.zeros((0, 3)),
        local_xy=np.zeros((0, 2)),
        ray_count=0,
    )
    result = accumulate_detector_hits(hits, grid)
    ny, nx = grid.resolution
    assert result.count_map.shape == (ny, nx)
    assert result.weight_map.shape == (ny, nx)
    assert int(result.count_map.sum()) == 0
    assert result.weight_map.sum() == pytest.approx(0.0)
    assert result.hit_pixel_indices.shape == (0, 2)
    assert result.total_hits == 0
    assert result.total_weight == pytest.approx(0.0)


def test_detector_plane_integration() -> None:
    detector = create_detector_plane(
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )
    grid = _grid_10x10()
    rays = make_ray_bundle(
        origins=[
            [0.0, 0.0, 5.0],     # hit center
            [3.0, -2.0, 5.0],    # hit off-center
            [100.0, 0.0, 5.0],   # miss out-of-bounds
        ],
        directions=[
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
    )
    hits = intersect_detector_plane(rays, detector)
    result = accumulate_detector_hits(hits, grid)
    assert result.total_hits == int(hits.hit_mask.sum())
    assert int(result.count_map.sum()) == result.total_hits
    assert result.total_hits == 2  # two hit, one out-of-bounds


def test_multi_step_final_rays_accumulate_smoke() -> None:
    box_mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        nx=3,
        ny=3,
    )
    trace = run_multi_step_trace(box_mesh, rays, SLAB_INTERFACES)
    detector = create_detector_plane(
        center=(0.0, 0.0, -15.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )
    hits = intersect_detector_plane(trace.final_rays, detector)
    grid = create_detector_grid(width=10.0, height=10.0, resolution=(20, 20))
    result = accumulate_detector_hits(hits, grid)
    assert result.total_hits > 0
    assert int(result.count_map.sum()) == result.total_hits
    assert result.count_map.shape == (20, 20)


def test_hit_pixel_indices_dtype_and_alignment() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy(
        [(-3.0, -3.0), (1.0, 2.0), (4.5, 4.5)],
    )
    result = accumulate_detector_hits(hits, grid)
    assert result.hit_pixel_indices.dtype == np.int64
    # Caller can use indices to fancy-index count_map without reshape:
    rows = result.hit_pixel_indices[:, 0]
    cols = result.hit_pixel_indices[:, 1]
    np.testing.assert_array_equal(result.count_map[rows, cols], [1, 1, 1])


def test_non_square_grid_pixel_mapping() -> None:
    grid = create_detector_grid(width=10.0, height=20.0, resolution=(8, 16))
    # pixel_width = 10/16 = 0.625; pixel_height = 20/8 = 2.5
    assert grid.pixel_width == pytest.approx(0.625)
    assert grid.pixel_height == pytest.approx(2.5)
    hits = _hits_from_local_xy([(0.0, 0.0)])
    result = accumulate_detector_hits(hits, grid)
    # col = floor((0 + 5) / 0.625) = 8
    # row = floor((0 + 10) / 2.5) = 4
    assert result.hit_pixel_indices[0, 0] == 4
    assert result.hit_pixel_indices[0, 1] == 8
    assert result.count_map.shape == (8, 16)


def test_invalid_input_types_raise() -> None:
    grid = _grid_10x10()
    with pytest.raises(OpticsError, match="DetectorHitResult"):
        accumulate_detector_hits(object(), grid)  # type: ignore[arg-type]

    hits = _hits_from_local_xy([(0.0, 0.0)])
    with pytest.raises(OpticsError, match="DetectorGrid"):
        accumulate_detector_hits(hits, object())  # type: ignore[arg-type]


def test_hit_mask_is_copied_not_aliased() -> None:
    grid = _grid_10x10()
    hits = _hits_from_local_xy([(0.0, 0.0), (1.0, 1.0)], hit_flags=[True, False])
    result = accumulate_detector_hits(hits, grid)
    # Mutating the result's hit_mask must not affect the source.
    assert result.hit_mask is not hits.hit_mask
    np.testing.assert_array_equal(result.hit_mask, hits.hit_mask)
