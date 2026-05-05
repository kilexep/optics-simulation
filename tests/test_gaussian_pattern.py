import numpy as np
import pytest

from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.pattern import (
    GaussianDimple,
    GaussianDimplePattern,
    PatternError,
    create_gaussian_dimple_pattern,
    evaluate_gaussian_dimple_field,
    sample_dimple_centers_from_risk,
)


def _make_risk_map(prob_2d: np.ndarray) -> RiskMap:
    prob = np.asarray(prob_2d, dtype=float)
    total = float(prob.sum())
    if total > 0.0:
        normalized = prob / total
    else:
        normalized = np.zeros_like(prob)
    risk = prob.copy()
    active = prob > 0.0
    return RiskMap(
        risk_map=risk,
        probability_map=normalized,
        active_mask=active,
        total_risk=total,
        active_count=int(active.sum()),
        epsilon=0.0,
        threshold=None,
    )


def test_count_zero_returns_empty_centers_and_zero_field() -> None:
    rm = _make_risk_map(np.array([[1.0, 1.0], [1.0, 1.0]]))
    centers = sample_dimple_centers_from_risk(rm, count=0, seed=0)
    assert centers.shape == (0, 2)
    pattern = create_gaussian_dimple_pattern(rm, count=0, seed=0)
    assert pattern.dimples == ()
    assert pattern.depth_field.shape == (2, 2)
    assert float(pattern.depth_field.sum()) == 0.0


def test_invalid_count_raises() -> None:
    rm = _make_risk_map(np.array([[1.0]]))
    with pytest.raises(PatternError, match="count"):
        sample_dimple_centers_from_risk(rm, count=-1, seed=0)
    with pytest.raises(PatternError, match="count"):
        create_gaussian_dimple_pattern(rm, count=-3, seed=0)


def test_zero_probability_with_count_positive_raises() -> None:
    rm = _make_risk_map(np.zeros((4, 4)))
    with pytest.raises(PatternError, match="zero total mass"):
        sample_dimple_centers_from_risk(rm, count=5, seed=0)
    with pytest.raises(PatternError, match="zero total mass"):
        create_gaussian_dimple_pattern(rm, count=5, seed=0)


def test_same_seed_gives_same_centers() -> None:
    rng_state = np.random.default_rng(0).random((6, 6))
    rm = _make_risk_map(rng_state)
    a = sample_dimple_centers_from_risk(rm, count=20, seed=42)
    b = sample_dimple_centers_from_risk(rm, count=20, seed=42)
    assert np.array_equal(a, b)


def test_one_hot_probability_samples_one_hot_pixel_center() -> None:
    nv, nu = 8, 8
    prob = np.zeros((nv, nu))
    prob[3, 5] = 1.0
    rm = _make_risk_map(prob)
    centers = sample_dimple_centers_from_risk(rm, count=10, seed=7)
    expected_u = (5 + 0.5) / nu
    expected_v = (3 + 0.5) / nv
    assert np.allclose(centers[:, 0], expected_u)
    assert np.allclose(centers[:, 1], expected_v)


def test_sampled_centers_are_in_valid_ranges() -> None:
    rm = _make_risk_map(np.ones((5, 7)))
    centers = sample_dimple_centers_from_risk(rm, count=50, seed=1)
    assert (centers[:, 0] >= 0.0).all()
    assert (centers[:, 0] < 1.0).all()
    assert (centers[:, 1] >= 0.0).all()
    assert (centers[:, 1] <= 1.0).all()


def test_field_from_one_dimple_has_max_above_zero() -> None:
    dimple = GaussianDimple(
        center_u=0.5, center_v=0.5,
        amplitude=1.0, sigma_u=0.1, sigma_v=0.1,
    )
    field = evaluate_gaussian_dimple_field(
        [dimple], resolution=(16, 16), clip=True
    )
    assert float(field.max()) > 0.0


def test_depth_field_shape_matches_resolution() -> None:
    dimple = GaussianDimple(
        center_u=0.5, center_v=0.5,
        amplitude=1.0, sigma_u=0.1, sigma_v=0.1,
    )
    field = evaluate_gaussian_dimple_field(
        [dimple], resolution=(16, 32), clip=True
    )
    assert field.shape == (16, 32)


def test_clip_true_bounds_field_to_unit_interval() -> None:
    dimples = [
        GaussianDimple(
            center_u=0.5, center_v=0.5,
            amplitude=1.0, sigma_u=0.5, sigma_v=0.5,
        ),
        GaussianDimple(
            center_u=0.5, center_v=0.5,
            amplitude=1.0, sigma_u=0.5, sigma_v=0.5,
        ),
        GaussianDimple(
            center_u=0.5, center_v=0.5,
            amplitude=1.0, sigma_u=0.5, sigma_v=0.5,
        ),
    ]
    field_clipped = evaluate_gaussian_dimple_field(
        dimples, resolution=(8, 8), clip=True
    )
    assert float(field_clipped.max()) <= 1.0 + 1e-12


def test_clip_false_can_exceed_one_for_overlapping_dimples() -> None:
    dimples = [
        GaussianDimple(
            center_u=0.5, center_v=0.5,
            amplitude=1.0, sigma_u=0.5, sigma_v=0.5,
        ),
        GaussianDimple(
            center_u=0.5, center_v=0.5,
            amplitude=1.0, sigma_u=0.5, sigma_v=0.5,
        ),
        GaussianDimple(
            center_u=0.5, center_v=0.5,
            amplitude=1.0, sigma_u=0.5, sigma_v=0.5,
        ),
    ]
    field_raw = evaluate_gaussian_dimple_field(
        dimples, resolution=(8, 8), clip=False
    )
    assert float(field_raw.max()) > 1.0


def test_circular_u_distance_works_near_wrap_boundary() -> None:
    dimple = GaussianDimple(
        center_u=0.99, center_v=0.5,
        amplitude=1.0, sigma_u=0.05, sigma_v=0.05,
    )
    field = evaluate_gaussian_dimple_field(
        [dimple], resolution=(64, 64), clip=False
    )
    nu = 64
    u_axis = (np.arange(nu) + 0.5) / nu
    near_zero_col = int(np.argmin(np.abs(u_axis - 0.01)))
    far_col = int(np.argmin(np.abs(u_axis - 0.5)))
    row = int(np.argmin(np.abs((np.arange(64) + 0.5) / 64 - 0.5)))
    val_near = float(field[row, near_zero_col])
    val_far = float(field[row, far_col])
    assert val_near > val_far
    assert val_near > 0.1


def test_invalid_amplitude_raises() -> None:
    with pytest.raises(PatternError, match="amplitude"):
        evaluate_gaussian_dimple_field(
            [GaussianDimple(0.5, 0.5, amplitude=0.0, sigma_u=0.1, sigma_v=0.1)],
            resolution=(4, 4),
        )
    rm = _make_risk_map(np.array([[1.0]]))
    with pytest.raises(PatternError, match="amplitude"):
        create_gaussian_dimple_pattern(rm, count=1, amplitude=-1.0, seed=0)


def test_invalid_sigma_raises() -> None:
    with pytest.raises(PatternError, match="sigma_u"):
        evaluate_gaussian_dimple_field(
            [GaussianDimple(0.5, 0.5, amplitude=1.0, sigma_u=0.0, sigma_v=0.1)],
            resolution=(4, 4),
        )
    with pytest.raises(PatternError, match="sigma_v"):
        evaluate_gaussian_dimple_field(
            [GaussianDimple(0.5, 0.5, amplitude=1.0, sigma_u=0.1, sigma_v=-0.1)],
            resolution=(4, 4),
        )
    rm = _make_risk_map(np.array([[1.0]]))
    with pytest.raises(PatternError, match="sigma_u"):
        create_gaussian_dimple_pattern(rm, count=1, sigma_u=0.0, seed=0)
    with pytest.raises(PatternError, match="sigma_v"):
        create_gaussian_dimple_pattern(rm, count=1, sigma_v=-0.1, seed=0)


def test_invalid_max_depth_raises() -> None:
    rm = _make_risk_map(np.array([[1.0]]))
    with pytest.raises(PatternError, match="max_depth"):
        create_gaussian_dimple_pattern(rm, count=1, max_depth=0.0, seed=0)
    with pytest.raises(PatternError, match="max_depth"):
        create_gaussian_dimple_pattern(rm, count=1, max_depth=-0.5, seed=0)


def test_invalid_center_u_raises() -> None:
    with pytest.raises(PatternError, match="center_u"):
        evaluate_gaussian_dimple_field(
            [GaussianDimple(1.0, 0.5, amplitude=1.0, sigma_u=0.1, sigma_v=0.1)],
            resolution=(4, 4),
        )
    with pytest.raises(PatternError, match="center_u"):
        evaluate_gaussian_dimple_field(
            [GaussianDimple(-0.01, 0.5, amplitude=1.0, sigma_u=0.1, sigma_v=0.1)],
            resolution=(4, 4),
        )


def test_invalid_center_v_raises() -> None:
    with pytest.raises(PatternError, match="center_v"):
        evaluate_gaussian_dimple_field(
            [GaussianDimple(0.5, 1.5, amplitude=1.0, sigma_u=0.1, sigma_v=0.1)],
            resolution=(4, 4),
        )
    with pytest.raises(PatternError, match="center_v"):
        evaluate_gaussian_dimple_field(
            [GaussianDimple(0.5, -0.1, amplitude=1.0, sigma_u=0.1, sigma_v=0.1)],
            resolution=(4, 4),
        )


def test_invalid_resolution_raises() -> None:
    dimple = GaussianDimple(
        center_u=0.5, center_v=0.5,
        amplitude=1.0, sigma_u=0.1, sigma_v=0.1,
    )
    for bad in [(0, 4), (-1, 4), (4, 0), (4, -2)]:
        with pytest.raises(PatternError, match="resolution"):
            evaluate_gaussian_dimple_field([dimple], resolution=bad)
    with pytest.raises(PatternError, match="resolution"):
        evaluate_gaussian_dimple_field([dimple], resolution=(4,))
    with pytest.raises(PatternError, match="resolution"):
        evaluate_gaussian_dimple_field([dimple], resolution=(4, 4, 4))


def test_create_gaussian_dimple_pattern_returns_pattern() -> None:
    rm = _make_risk_map(np.ones((4, 4)))
    pattern = create_gaussian_dimple_pattern(
        rm, count=5, amplitude=1.0, sigma_u=0.1, sigma_v=0.1,
        max_depth=0.3, seed=0,
    )
    assert isinstance(pattern, GaussianDimplePattern)
    assert pattern.resolution == (4, 4)
    assert pattern.max_depth == pytest.approx(0.3)
    assert pattern.seed == 0


def test_number_of_dimples_equals_count() -> None:
    rm = _make_risk_map(np.ones((6, 6)))
    pattern = create_gaussian_dimple_pattern(rm, count=12, seed=3)
    assert len(pattern.dimples) == 12


def test_depth_field_finite_and_nonneg() -> None:
    rm = _make_risk_map(np.ones((8, 8)))
    pattern = create_gaussian_dimple_pattern(rm, count=10, seed=4)
    assert np.isfinite(pattern.depth_field).all()
    assert (pattern.depth_field >= 0.0).all()
    assert float(pattern.depth_field.max()) <= 1.0 + 1e-12


def test_empty_dimples_evaluates_to_zero_field() -> None:
    field = evaluate_gaussian_dimple_field([], resolution=(8, 16))
    assert field.shape == (8, 16)
    assert float(field.sum()) == 0.0


def test_integration_smoke_with_synthetic_risk_map() -> None:
    # Smoke check: synthetic RiskMap -> pattern foundation only.
    # No mesh displacement, no STL export, no optimization performed.
    prob = np.array(
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 2.0, 0.0],
         [0.0, 3.0, 4.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        dtype=float,
    )
    rm = _make_risk_map(prob)
    pattern = create_gaussian_dimple_pattern(
        rm, count=8, amplitude=1.0, sigma_u=0.05, sigma_v=0.05,
        max_depth=0.25, seed=11,
    )
    assert isinstance(pattern, GaussianDimplePattern)
    assert len(pattern.dimples) == 8
    assert pattern.depth_field.shape == prob.shape
    assert np.isfinite(pattern.depth_field).all()
    assert (pattern.depth_field >= 0.0).all()
    assert float(pattern.depth_field.max()) <= 1.0 + 1e-12
    for d in pattern.dimples:
        assert 0.0 <= d.center_u < 1.0
        assert 0.0 <= d.center_v <= 1.0
