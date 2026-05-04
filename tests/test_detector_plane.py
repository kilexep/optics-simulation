import numpy as np
import pytest
import trimesh

from optics_simulation.optics import (
    DetectorHitResult,
    DetectorPlane,
    OpticsError,
    create_detector_plane,
    intersect_detector_plane,
    make_ray_bundle,
    parallel_ray_grid,
    run_multi_step_trace,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
SLAB_INTERFACES = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]


def _xy_detector(width: float = 10.0, height: float = 10.0) -> DetectorPlane:
    return create_detector_plane(
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=width,
        height=height,
    )


def test_normal_incidence_ray_hits_center() -> None:
    detector = _xy_detector()
    rays = make_ray_bundle([[0.0, 0.0, 5.0]], [[0.0, 0.0, -1.0]])
    result = intersect_detector_plane(rays, detector)

    assert isinstance(result, DetectorHitResult)
    assert result.ray_count == 1
    np.testing.assert_array_equal(result.hit_mask, [True])
    assert result.t_hit[0] == pytest.approx(5.0, abs=1e-12)
    np.testing.assert_allclose(result.hit_points[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.local_xy[0], [0.0, 0.0], atol=1e-12)


def test_off_center_ray_maps_to_local_xy() -> None:
    detector = _xy_detector()
    rays = make_ray_bundle([[3.0, 2.0, 5.0]], [[0.0, 0.0, -1.0]])
    result = intersect_detector_plane(rays, detector)

    assert result.hit_mask[0]
    np.testing.assert_allclose(result.hit_points[0], [3.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.local_xy[0], [3.0, 2.0], atol=1e-12)


def test_ray_outside_bounds_is_miss() -> None:
    detector = _xy_detector(width=4.0, height=4.0)  # half = 2.0
    rays = make_ray_bundle([[3.0, 0.0, 5.0]], [[0.0, 0.0, -1.0]])
    result = intersect_detector_plane(rays, detector)

    assert not result.hit_mask[0]
    assert np.isinf(result.t_hit[0])
    assert np.isnan(result.hit_points[0]).all()
    assert np.isnan(result.local_xy[0]).all()


def test_ray_pointing_away_is_miss() -> None:
    detector = _xy_detector()
    rays = make_ray_bundle([[0.0, 0.0, 5.0]], [[0.0, 0.0, 1.0]])  # +z away from plane
    result = intersect_detector_plane(rays, detector)

    assert not result.hit_mask[0]
    assert np.isinf(result.t_hit[0])
    assert np.isnan(result.hit_points[0]).all()
    assert np.isnan(result.local_xy[0]).all()


def test_ray_parallel_to_plane_is_miss() -> None:
    detector = _xy_detector()
    rays = make_ray_bundle([[0.0, 0.0, 5.0]], [[1.0, 0.0, 0.0]])
    result = intersect_detector_plane(rays, detector)

    assert not result.hit_mask[0]
    assert np.isinf(result.t_hit[0])
    assert np.isnan(result.hit_points[0]).all()
    assert np.isnan(result.local_xy[0]).all()


def test_nonpositive_width_raises() -> None:
    for width in (0.0, -1.0):
        with pytest.raises(OpticsError, match="width"):
            create_detector_plane(
                center=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                up=(0.0, 1.0, 0.0),
                width=width,
                height=1.0,
            )


def test_nonpositive_height_raises() -> None:
    for height in (0.0, -1.0):
        with pytest.raises(OpticsError, match="height"):
            create_detector_plane(
                center=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                up=(0.0, 1.0, 0.0),
                width=1.0,
                height=height,
            )


def test_negative_t_min_raises() -> None:
    detector = _xy_detector()
    rays = make_ray_bundle([[0.0, 0.0, 5.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="t_min"):
        intersect_detector_plane(rays, detector, t_min=-0.1)


def test_up_parallel_to_normal_raises() -> None:
    with pytest.raises(OpticsError, match="parallel to normal"):
        create_detector_plane(
            center=(0.0, 0.0, 0.0),
            normal=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0),
            width=10.0,
            height=10.0,
        )


def test_basis_vectors_are_unit_norm() -> None:
    detector = _xy_detector()
    assert np.linalg.norm(detector.normal) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(detector.up) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(detector.right) == pytest.approx(1.0, abs=1e-12)


def test_right_equals_cross_up_normal_and_basis_orthogonal() -> None:
    # Locks the convention: right = cross(up, normal). With normal=+z
    # and up=+y this gives right=+x.
    detector = _xy_detector()
    np.testing.assert_allclose(
        detector.right,
        np.cross(detector.up, detector.normal),
        atol=1e-12,
    )
    np.testing.assert_allclose(detector.right, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(detector.up, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(detector.normal, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.dot(detector.normal, detector.up) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(detector.normal, detector.right) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(detector.up, detector.right) == pytest.approx(0.0, abs=1e-12)


def test_empty_bundle_returns_empty_result() -> None:
    detector = _xy_detector()
    rays = make_ray_bundle(np.zeros((0, 3)), np.zeros((0, 3)))
    result = intersect_detector_plane(rays, detector)

    assert result.ray_count == 0
    assert result.t_hit.shape == (0,)
    assert result.hit_mask.shape == (0,)
    assert result.hit_points.shape == (0, 3)
    assert result.local_xy.shape == (0, 2)


def test_miss_sentinels_uniform_across_miss_modes() -> None:
    detector = _xy_detector(width=4.0, height=4.0)
    # row 0: parallel; row 1: behind plane; row 2: out of bounds
    rays = make_ray_bundle(
        origins=[
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [3.0, 0.0, 5.0],
        ],
        directions=[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
    )
    result = intersect_detector_plane(rays, detector)
    assert not result.hit_mask.any()
    assert np.isinf(result.t_hit).all()
    assert np.isnan(result.hit_points).all()
    assert np.isnan(result.local_xy).all()


def test_dense_alignment_on_mixed_bundle() -> None:
    detector = _xy_detector(width=4.0, height=4.0)  # half = 2.0
    rays = make_ray_bundle(
        origins=[
            [0.0, 0.0, 5.0],   # 0: hit center
            [3.0, 0.0, 5.0],   # 1: miss out-of-bounds
            [0.0, 0.0, 5.0],   # 2: miss behind (going +z)
            [1.0, -1.0, 5.0],  # 3: hit at (1,-1)
        ],
        directions=[
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
    )
    result = intersect_detector_plane(rays, detector)
    assert result.ray_count == 4
    np.testing.assert_array_equal(result.hit_mask, [True, False, False, True])

    # Hit rows preserve their input index.
    np.testing.assert_allclose(result.local_xy[0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.local_xy[3], [1.0, -1.0], atol=1e-12)

    # Miss rows (1, 2) carry sentinels.
    for i in (1, 2):
        assert np.isinf(result.t_hit[i])
        assert np.isnan(result.hit_points[i]).all()
        assert np.isnan(result.local_xy[i]).all()


def test_multi_step_final_rays_reach_detector_smoke() -> None:
    # End-to-end smoke: parallel ray grid -> slab (box) two-step trace
    # -> detector at z=-15 facing +z. We only assert that at least one
    # ray lands on the detector and that the result schema is sane.
    # No irradiance map or pixel binning is constructed.
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
    assert trace.step_count == 2
    assert trace.final_rays.ray_count > 0

    detector = create_detector_plane(
        center=(0.0, 0.0, -15.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )
    result = intersect_detector_plane(trace.final_rays, detector)
    assert result.ray_count == trace.final_rays.ray_count
    assert result.hit_mask.any()
    # Hits land on the plane z = -15.
    hit_idx = np.flatnonzero(result.hit_mask)
    np.testing.assert_allclose(result.hit_points[hit_idx, 2], -15.0, atol=1e-9)


def test_invalid_rays_type_raises() -> None:
    detector = _xy_detector()
    with pytest.raises(OpticsError, match="RayBundle"):
        intersect_detector_plane(object(), detector)  # type: ignore[arg-type]


def test_invalid_detector_type_raises() -> None:
    rays = make_ray_bundle([[0.0, 0.0, 5.0]], [[0.0, 0.0, -1.0]])
    with pytest.raises(OpticsError, match="DetectorPlane"):
        intersect_detector_plane(rays, object())  # type: ignore[arg-type]
