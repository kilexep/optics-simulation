import numpy as np
import pytest

from optics_simulation.contribution import (
    ContributionError,
    RiskMap,
    RiskMapRingTransformResult,
    create_ring_offset_risk_map,
)


def _single_center_risk(nv: int = 8, nu: int = 16, *,
                        v: int = 4, u: int = 8,
                        weight: float = 1.0) -> RiskMap:
    risk = np.zeros((nv, nu), dtype=float)
    risk[v, u] = float(weight)
    active = risk > 0.0
    total = float(risk.sum())
    prob = (risk / total).astype(float, copy=True) if total > 0 else (
        np.zeros_like(risk)
    )
    return RiskMap(
        risk_map=risk.copy(),
        probability_map=prob,
        active_mask=active.copy(),
        total_risk=total,
        active_count=int(active.sum()),
        epsilon=0.0,
        threshold=None,
    )


def test_returns_result() -> None:
    rm = _single_center_risk()
    result = create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=3, epsilon=0.0,
    )
    assert isinstance(result, RiskMapRingTransformResult)
    assert isinstance(result.transformed_risk_map, RiskMap)


def test_transformed_shape_matches_input() -> None:
    rm = _single_center_risk(nv=10, nu=20)
    result = create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=2, epsilon=0.0,
    )
    assert result.transformed_risk_map.risk_map.shape == (10, 20)
    assert result.transformed_risk_map.probability_map.shape == (10, 20)
    assert result.transformed_risk_map.active_mask.shape == (10, 20)
    assert result.ring_weight_map.shape == (10, 20)


def test_transformed_finite_and_nonnegative() -> None:
    rm = _single_center_risk()
    result = create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=3, epsilon=0.01,
    )
    arr = result.transformed_risk_map.risk_map
    assert np.isfinite(arr).all()
    assert (arr >= 0.0).all()


def test_center_pixel_removed_when_inner_radius_one() -> None:
    rm = _single_center_risk(nv=8, nu=16, v=4, u=8, weight=1.0)
    result = create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=3, epsilon=0.0,
    )
    # Original center should not be active in the ring map.
    assert result.transformed_risk_map.active_mask[4, 8] == False
    assert float(result.transformed_risk_map.risk_map[4, 8]) == 0.0


def test_ring_pixels_active_around_single_center() -> None:
    rm = _single_center_risk(nv=8, nu=16, v=4, u=8, weight=1.0)
    result = create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=2, epsilon=0.0,
    )
    mask = result.transformed_risk_map.active_mask
    # Direct 4-neighbors at distance 1 should be active.
    assert mask[3, 8]
    assert mask[5, 8]
    assert mask[4, 7]
    assert mask[4, 9]
    # A pixel far outside the outer radius should not be active.
    assert mask[0, 0] == False


def test_u_wrapping_at_boundary() -> None:
    nv, nu = 8, 16
    # Place the active pixel at u=0 so that u-1 wraps to u=nu-1.
    rm = _single_center_risk(nv=nv, nu=nu, v=4, u=0, weight=1.0)
    result = create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=2,
        wrap_u=True, epsilon=0.0,
    )
    mask = result.transformed_risk_map.active_mask
    # u-1 wraps to last column.
    assert mask[4, nu - 1]
    # u+1 stays within range.
    assert mask[4, 1]


def test_v_does_not_wrap() -> None:
    nv, nu = 8, 16
    # Place the active pixel at v=0 so that v-1 would wrap if v wrapped.
    rm = _single_center_risk(nv=nv, nu=nu, v=0, u=8, weight=1.0)
    result = create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=2,
        wrap_u=True, epsilon=0.0,
    )
    mask = result.transformed_risk_map.active_mask
    # v=nv-1 (the bottom row) must NOT be activated by v=0's
    # weight; v does not wrap.
    assert mask[nv - 1, 8] == False
    # v+1 stays within range.
    assert mask[1, 8]


def test_invalid_radii_raises() -> None:
    rm = _single_center_risk()
    with pytest.raises(ContributionError, match="inner_radius_px"):
        create_ring_offset_risk_map(
            rm, inner_radius_px=-1, outer_radius_px=3,
        )
    with pytest.raises(ContributionError, match="outer_radius_px"):
        create_ring_offset_risk_map(
            rm, inner_radius_px=2, outer_radius_px=2,
        )
    with pytest.raises(ContributionError, match="outer_radius_px"):
        create_ring_offset_risk_map(
            rm, inner_radius_px=3, outer_radius_px=1,
        )


def test_invalid_epsilon_raises() -> None:
    rm = _single_center_risk()
    with pytest.raises(ContributionError, match="epsilon"):
        create_ring_offset_risk_map(
            rm, inner_radius_px=1, outer_radius_px=2,
            epsilon=-0.1,
        )


def test_invalid_input_type_raises() -> None:
    with pytest.raises(ContributionError, match="RiskMap"):
        create_ring_offset_risk_map(
            "not a risk map",  # type: ignore[arg-type]
            inner_radius_px=1, outer_radius_px=2,
        )


def test_input_risk_map_not_mutated() -> None:
    rm = _single_center_risk(nv=8, nu=16, v=4, u=8, weight=1.0)
    risk_before = np.array(rm.risk_map, copy=True)
    prob_before = np.array(rm.probability_map, copy=True)
    mask_before = np.array(rm.active_mask, copy=True)
    create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=3, epsilon=0.05,
    )
    assert np.array_equal(rm.risk_map, risk_before)
    assert np.array_equal(rm.probability_map, prob_before)
    assert np.array_equal(rm.active_mask, mask_before)


def test_no_file_output(tmp_path) -> None:
    rm = _single_center_risk()
    before = sorted(tmp_path.iterdir())
    create_ring_offset_risk_map(
        rm, inner_radius_px=1, outer_radius_px=3, epsilon=0.01,
    )
    after = sorted(tmp_path.iterdir())
    assert before == after
