import numpy as np
import pytest

from optics_simulation.contribution import (
    ContributionError,
    ContributionMap,
    RiskMap,
    build_risk_map,
)


def _make_contribution(
    weight_map: np.ndarray,
    normalized_map: np.ndarray | None = None,
) -> ContributionMap:
    w = np.asarray(weight_map, dtype=float)
    if normalized_map is None:
        mx = float(w.max()) if w.size > 0 else 0.0
        if mx > 0.0:
            n = w / mx
        else:
            n = np.zeros_like(w)
    else:
        n = np.asarray(normalized_map, dtype=float)
    nv, nu = w.shape
    return ContributionMap(
        count_map=(w > 0).astype(np.int64),
        weight_map=w,
        normalized_map=n,
        resolution=(int(nv), int(nu)),
        total_selected=int((w > 0).sum()),
        total_weight=float(w.sum()),
        selected_ray_indices=np.empty(0, dtype=np.int64),
        u_bin_indices=np.empty(0, dtype=np.int64),
        v_bin_indices=np.empty(0, dtype=np.int64),
    )


def test_uses_normalized_map_by_default() -> None:
    weight_map = np.array([[0.0, 2.0], [4.0, 0.0]])
    normalized_map = weight_map / 4.0
    contribution = _make_contribution(weight_map, normalized_map)
    rm = build_risk_map(contribution)
    assert isinstance(rm, RiskMap)
    expected_active = np.array([[False, True], [True, False]])
    assert np.array_equal(rm.active_mask, expected_active)
    assert rm.risk_map[0, 1] == pytest.approx(0.5)
    assert rm.risk_map[1, 0] == pytest.approx(1.0)


def test_uses_weight_map_when_use_normalized_false() -> None:
    weight_map = np.array([[0.0, 2.0], [4.0, 0.0]])
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(contribution, use_normalized=False)
    assert rm.risk_map[0, 1] == pytest.approx(2.0)
    assert rm.risk_map[1, 0] == pytest.approx(4.0)


def test_threshold_filters_low_values() -> None:
    weight_map = np.array([[0.0, 1.0, 5.0], [2.0, 8.0, 0.0]])
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(contribution, use_normalized=False, threshold=1.5)
    expected_active = np.array(
        [[False, False, True], [True, True, False]]
    )
    assert np.array_equal(rm.active_mask, expected_active)
    assert rm.risk_map[0, 0] == 0.0
    assert rm.risk_map[0, 1] == 0.0
    assert rm.risk_map[1, 2] == 0.0


def test_threshold_uses_strict_greater_than() -> None:
    weight_map = np.array([[2.0, 2.0, 3.0]])
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(contribution, use_normalized=False, threshold=2.0)
    assert rm.active_count == 1
    assert rm.active_mask.tolist() == [[False, False, True]]


def test_threshold_zero_equals_default_none() -> None:
    weight_map = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    contribution = _make_contribution(weight_map)
    rm_none = build_risk_map(contribution, use_normalized=False)
    rm_zero = build_risk_map(contribution, use_normalized=False, threshold=0.0)
    assert np.array_equal(rm_none.active_mask, rm_zero.active_mask)
    assert np.allclose(rm_none.risk_map, rm_zero.risk_map)


def test_epsilon_added_only_to_active_pixels() -> None:
    weight_map = np.array([[0.0, 2.0]])
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(
        contribution, use_normalized=False, epsilon=0.5
    )
    assert rm.risk_map[0, 0] == 0.0
    assert rm.risk_map[0, 1] == pytest.approx(2.5)


def test_all_zero_map_with_epsilon_still_zero_probability() -> None:
    weight_map = np.zeros((4, 4))
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(
        contribution, use_normalized=False, epsilon=0.5
    )
    assert rm.active_count == 0
    assert float(rm.risk_map.sum()) == 0.0
    assert float(rm.probability_map.sum()) == 0.0


def test_probability_map_sums_to_one_when_total_risk_positive() -> None:
    weight_map = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 0.0]])
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(contribution, use_normalized=False)
    assert rm.total_risk > 0.0
    assert float(rm.probability_map.sum()) == pytest.approx(1.0, abs=1e-12)


def test_probability_map_all_zero_when_total_risk_zero() -> None:
    weight_map = np.array([[0.0, 1.0]])
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(
        contribution, use_normalized=False, threshold=10.0
    )
    assert rm.total_risk == 0.0
    assert float(rm.probability_map.sum()) == 0.0
    assert rm.active_count == 0


def test_all_zero_contribution_map_handled() -> None:
    weight_map = np.zeros((3, 3))
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(contribution)
    assert rm.total_risk == 0.0
    assert rm.active_count == 0
    assert float(rm.probability_map.sum()) == 0.0
    assert rm.risk_map.shape == (3, 3)
    assert rm.probability_map.shape == (3, 3)


def test_invalid_threshold_raises() -> None:
    contribution = _make_contribution(np.array([[1.0]]))
    with pytest.raises(ContributionError, match="threshold"):
        build_risk_map(contribution, threshold=-0.1)


def test_invalid_epsilon_raises() -> None:
    contribution = _make_contribution(np.array([[1.0]]))
    with pytest.raises(ContributionError, match="epsilon"):
        build_risk_map(contribution, epsilon=-0.001)


def test_nan_in_base_raises() -> None:
    weight_map = np.array([[0.0, np.nan]])
    normalized_map = np.array([[0.0, 0.5]])
    contribution = _make_contribution(weight_map, normalized_map)
    with pytest.raises(ContributionError, match="NaN or inf"):
        build_risk_map(contribution, use_normalized=False)


def test_inf_in_base_raises() -> None:
    weight_map = np.array([[0.0, np.inf]])
    normalized_map = np.array([[0.0, 0.5]])
    contribution = _make_contribution(weight_map, normalized_map)
    with pytest.raises(ContributionError, match="NaN or inf"):
        build_risk_map(contribution, use_normalized=False)


def test_negative_base_raises() -> None:
    weight_map = np.array([[0.0, -1.0]])
    normalized_map = np.array([[0.0, 0.5]])
    contribution = _make_contribution(weight_map, normalized_map)
    with pytest.raises(ContributionError, match="negative"):
        build_risk_map(contribution, use_normalized=False)


def test_shape_mismatch_raises() -> None:
    weight_map = np.array([[1.0, 2.0]])
    normalized_map = np.array([[0.5, 1.0]])
    contribution = ContributionMap(
        count_map=np.array([[1, 1, 1]], dtype=np.int64),  # wrong shape
        weight_map=weight_map,
        normalized_map=normalized_map,
        resolution=(1, 2),
        total_selected=2,
        total_weight=3.0,
        selected_ray_indices=np.empty(0, dtype=np.int64),
        u_bin_indices=np.empty(0, dtype=np.int64),
        v_bin_indices=np.empty(0, dtype=np.int64),
    )
    with pytest.raises(ContributionError, match="inconsistent shapes"):
        build_risk_map(contribution)


def test_invalid_input_type_raises() -> None:
    with pytest.raises(ContributionError, match="ContributionMap"):
        build_risk_map(object())  # type: ignore[arg-type]


def test_input_contribution_is_not_mutated() -> None:
    weight_map = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 0.0]])
    normalized_map = weight_map / 4.0
    contribution = _make_contribution(weight_map, normalized_map)
    weight_before = np.array(contribution.weight_map, copy=True)
    normalized_before = np.array(contribution.normalized_map, copy=True)
    count_before = np.array(contribution.count_map, copy=True)

    build_risk_map(contribution, use_normalized=True, epsilon=0.5)
    build_risk_map(
        contribution, use_normalized=False, threshold=1.0, epsilon=0.25
    )

    assert np.array_equal(contribution.weight_map, weight_before)
    assert np.array_equal(contribution.normalized_map, normalized_before)
    assert np.array_equal(contribution.count_map, count_before)


def test_integration_smoke_with_synthetic_contribution_map() -> None:
    # Smoke check that a synthetic ContributionMap flows end-to-end
    # through build_risk_map. No pattern generation is performed here.
    weight_map = np.array(
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 2.0, 0.0],
         [0.0, 3.0, 4.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]]
    )
    contribution = _make_contribution(weight_map)
    rm = build_risk_map(
        contribution, use_normalized=True, threshold=0.1, epsilon=0.05
    )
    assert rm.risk_map.shape == weight_map.shape
    assert rm.probability_map.shape == weight_map.shape
    assert rm.active_count >= 1
    assert rm.total_risk > 0.0
    assert float(rm.probability_map.sum()) == pytest.approx(1.0, abs=1e-12)
