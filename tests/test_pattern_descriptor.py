import json

import numpy as np
import pytest

from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.pattern import (
    GaussianDimple,
    GaussianDimplePattern,
    PatternError,
    create_gaussian_dimple_pattern,
    evaluate_gaussian_dimple_field,
    gaussian_pattern_from_descriptor,
    gaussian_pattern_to_descriptor,
)


REQUIRED_TOP_KEYS = {
    "pattern_family",
    "coordinate_system",
    "depth_field_type",
    "resolution",
    "max_depth",
    "seed",
    "dimple_count",
    "dimples",
}
REQUIRED_DIMPLE_KEYS = {
    "center_u", "center_v", "amplitude", "sigma_u", "sigma_v",
}


def _make_risk_map(prob_2d: np.ndarray) -> RiskMap:
    prob = np.asarray(prob_2d, dtype=float)
    total = float(prob.sum())
    normalized = prob / total if total > 0.0 else np.zeros_like(prob)
    return RiskMap(
        risk_map=prob.copy(),
        probability_map=normalized,
        active_mask=prob > 0.0,
        total_risk=total,
        active_count=int((prob > 0.0).sum()),
        epsilon=0.0,
        threshold=None,
    )


def _make_synthetic_pattern(
    *, count: int = 5, seed: int = 42
) -> GaussianDimplePattern:
    rm = _make_risk_map(np.ones((4, 8)))
    return create_gaussian_dimple_pattern(
        rm,
        count=count,
        amplitude=1.0,
        sigma_u=0.05,
        sigma_v=0.05,
        max_depth=0.25,
        seed=seed,
    )


def _make_empty_pattern() -> GaussianDimplePattern:
    rm = _make_risk_map(np.ones((4, 8)))
    return create_gaussian_dimple_pattern(
        rm,
        count=0,
        amplitude=1.0,
        sigma_u=0.05,
        sigma_v=0.05,
        max_depth=0.25,
        seed=7,
    )


def _walk_no_numpy(node) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _walk_no_numpy(v)
    elif isinstance(node, list):
        for v in node:
            _walk_no_numpy(v)
    else:
        assert not isinstance(node, np.ndarray), "numpy array in descriptor"
        assert not isinstance(node, np.generic), "numpy scalar in descriptor"


def test_descriptor_is_dict() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    assert isinstance(d, dict)


def test_descriptor_contains_required_keys() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    assert set(d.keys()) >= REQUIRED_TOP_KEYS
    for dd in d["dimples"]:
        assert set(dd.keys()) >= REQUIRED_DIMPLE_KEYS


def test_descriptor_coordinate_system_value() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    assert d["coordinate_system"] == "normalized_cylindrical_uv"
    assert d["pattern_family"] == "gaussian_dimple"
    assert d["depth_field_type"] == "normalized_eta"


def test_descriptor_is_json_serializable() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    blob = json.dumps(d)
    reloaded = json.loads(blob)
    assert reloaded["pattern_family"] == "gaussian_dimple"
    assert reloaded["dimple_count"] == len(pattern.dimples)


def test_descriptor_contains_no_numpy_arrays_or_scalars() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    _walk_no_numpy(d)


def test_dimple_count_equals_dimples_length() -> None:
    pattern = _make_synthetic_pattern(count=12)
    d = gaussian_pattern_to_descriptor(pattern)
    assert d["dimple_count"] == len(d["dimples"]) == 12


def test_round_trip_preserves_dimple_parameters() -> None:
    pattern = _make_synthetic_pattern(count=8)
    d = gaussian_pattern_to_descriptor(pattern)
    rebuilt = gaussian_pattern_from_descriptor(d)
    assert len(rebuilt.dimples) == len(pattern.dimples)
    for orig, new in zip(pattern.dimples, rebuilt.dimples):
        assert new.center_u == pytest.approx(orig.center_u)
        assert new.center_v == pytest.approx(orig.center_v)
        assert new.amplitude == pytest.approx(orig.amplitude)
        assert new.sigma_u == pytest.approx(orig.sigma_u)
        assert new.sigma_v == pytest.approx(orig.sigma_v)


def test_round_trip_preserves_metadata() -> None:
    pattern = _make_synthetic_pattern(count=8, seed=42)
    d = gaussian_pattern_to_descriptor(pattern)
    rebuilt = gaussian_pattern_from_descriptor(d)
    assert rebuilt.resolution == pattern.resolution
    assert rebuilt.max_depth == pytest.approx(pattern.max_depth)
    assert rebuilt.seed == pattern.seed


def test_round_trip_recomputes_depth_field_with_same_shape() -> None:
    pattern = _make_synthetic_pattern(count=8)
    d = gaussian_pattern_to_descriptor(pattern)
    rebuilt = gaussian_pattern_from_descriptor(d)
    assert rebuilt.depth_field.shape == pattern.depth_field.shape


def test_round_trip_depth_field_allclose() -> None:
    pattern = _make_synthetic_pattern(count=8)
    d = gaussian_pattern_to_descriptor(pattern)
    rebuilt = gaussian_pattern_from_descriptor(d)
    assert np.allclose(rebuilt.depth_field, pattern.depth_field, atol=1e-12)


def test_wrong_pattern_family_raises() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    d["pattern_family"] = "other_family"
    with pytest.raises(PatternError, match="pattern_family"):
        gaussian_pattern_from_descriptor(d)


def test_wrong_coordinate_system_raises() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    d["coordinate_system"] = "xy"
    with pytest.raises(PatternError, match="coordinate_system"):
        gaussian_pattern_from_descriptor(d)


def test_wrong_depth_field_type_raises() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    d["depth_field_type"] = "raw"
    with pytest.raises(PatternError, match="depth_field_type"):
        gaussian_pattern_from_descriptor(d)


def test_dimple_count_mismatch_raises() -> None:
    pattern = _make_synthetic_pattern(count=5)
    d = gaussian_pattern_to_descriptor(pattern)
    d["dimple_count"] = 3
    with pytest.raises(PatternError, match="dimple_count"):
        gaussian_pattern_from_descriptor(d)


def test_missing_required_top_level_key_raises() -> None:
    pattern = _make_synthetic_pattern()
    base = gaussian_pattern_to_descriptor(pattern)
    for key in REQUIRED_TOP_KEYS:
        bad = {k: v for k, v in base.items() if k != key}
        with pytest.raises(PatternError, match="missing"):
            gaussian_pattern_from_descriptor(bad)


def test_missing_required_dimple_key_raises() -> None:
    pattern = _make_synthetic_pattern(count=2)
    d = gaussian_pattern_to_descriptor(pattern)
    del d["dimples"][0]["sigma_u"]
    with pytest.raises(PatternError, match="missing"):
        gaussian_pattern_from_descriptor(d)


def test_invalid_resolution_raises() -> None:
    pattern = _make_synthetic_pattern()
    base = gaussian_pattern_to_descriptor(pattern)
    bad_values = [
        [0, 5], [-1, 5], [5, 0], [5, -2],
        [5], [5, 5, 5],
    ]
    for r in bad_values:
        d = dict(base)
        d["resolution"] = r
        with pytest.raises(PatternError, match="resolution"):
            gaussian_pattern_from_descriptor(d)
    d = dict(base)
    d["resolution"] = "not a list"
    with pytest.raises(PatternError, match="resolution"):
        gaussian_pattern_from_descriptor(d)


def test_non_integer_resolution_values_raise() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    d["resolution"] = [4.0, 8]
    with pytest.raises(PatternError, match="resolution"):
        gaussian_pattern_from_descriptor(d)
    d["resolution"] = [4, "8"]
    with pytest.raises(PatternError, match="resolution"):
        gaussian_pattern_from_descriptor(d)


def test_invalid_seed_type_raises() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    d["seed"] = 1.5
    with pytest.raises(PatternError, match="seed"):
        gaussian_pattern_from_descriptor(d)
    d["seed"] = "42"
    with pytest.raises(PatternError, match="seed"):
        gaussian_pattern_from_descriptor(d)


def test_bool_seed_raises() -> None:
    pattern = _make_synthetic_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    d["seed"] = True
    with pytest.raises(PatternError, match="seed"):
        gaussian_pattern_from_descriptor(d)


def test_export_rejects_bool_seed_on_dataclass() -> None:
    # Bypass the public factory to inject a bool seed into the
    # dataclass; export must reject it strictly.
    pattern = GaussianDimplePattern(
        dimples=(),
        depth_field=np.zeros((4, 8), dtype=float),
        resolution=(4, 8),
        max_depth=0.25,
        seed=True,  # type: ignore[arg-type]
    )
    with pytest.raises(PatternError, match="seed"):
        gaussian_pattern_to_descriptor(pattern)


def test_invalid_max_depth_raises() -> None:
    pattern = _make_synthetic_pattern()
    base = gaussian_pattern_to_descriptor(pattern)
    for bad in [0.0, -0.5]:
        d = dict(base)
        d["max_depth"] = bad
        with pytest.raises(PatternError, match="max_depth"):
            gaussian_pattern_from_descriptor(d)
    d = dict(base)
    d["max_depth"] = "0.25"
    with pytest.raises(PatternError, match="max_depth"):
        gaussian_pattern_from_descriptor(d)
    d = dict(base)
    d["max_depth"] = True
    with pytest.raises(PatternError, match="max_depth"):
        gaussian_pattern_from_descriptor(d)


def test_nan_inf_numeric_field_raises_on_import() -> None:
    pattern = _make_synthetic_pattern(count=2)
    base = gaussian_pattern_to_descriptor(pattern)

    d = json.loads(json.dumps(base))
    d["max_depth"] = float("nan")
    with pytest.raises(PatternError, match="finite"):
        gaussian_pattern_from_descriptor(d)

    d = json.loads(json.dumps(base))
    d["max_depth"] = float("inf")
    with pytest.raises(PatternError, match="finite"):
        gaussian_pattern_from_descriptor(d)

    d = json.loads(json.dumps(base))
    d["dimples"][0]["amplitude"] = float("nan")
    with pytest.raises(PatternError, match="finite"):
        gaussian_pattern_from_descriptor(d)

    d = json.loads(json.dumps(base))
    d["dimples"][0]["sigma_u"] = float("-inf")
    with pytest.raises(PatternError, match="finite"):
        gaussian_pattern_from_descriptor(d)


def test_nan_inf_numeric_field_raises_on_export() -> None:
    bad_dimple = GaussianDimple(
        center_u=0.5, center_v=0.5,
        amplitude=1.0, sigma_u=0.1, sigma_v=0.1,
    )
    object.__setattr__(bad_dimple, "amplitude", float("nan"))
    pattern = GaussianDimplePattern(
        dimples=(bad_dimple,),
        depth_field=np.zeros((4, 8), dtype=float),
        resolution=(4, 8),
        max_depth=0.25,
        seed=0,
    )
    with pytest.raises(PatternError, match="finite"):
        gaussian_pattern_to_descriptor(pattern)


def test_invalid_dimple_field_raises() -> None:
    pattern = _make_synthetic_pattern(count=1)
    d = gaussian_pattern_to_descriptor(pattern)
    d["dimples"][0]["amplitude"] = 0.0
    with pytest.raises(PatternError, match="amplitude"):
        gaussian_pattern_from_descriptor(d)

    d = gaussian_pattern_to_descriptor(pattern)
    d["dimples"][0]["sigma_u"] = -0.1
    with pytest.raises(PatternError, match="sigma_u"):
        gaussian_pattern_from_descriptor(d)

    d = gaussian_pattern_to_descriptor(pattern)
    d["dimples"][0]["center_u"] = 1.0
    with pytest.raises(PatternError, match="center_u"):
        gaussian_pattern_from_descriptor(d)


def test_invalid_dimple_field_type_raises() -> None:
    pattern = _make_synthetic_pattern(count=1)
    d = gaussian_pattern_to_descriptor(pattern)
    d["dimples"][0]["amplitude"] = "1.0"
    with pytest.raises(PatternError, match="amplitude"):
        gaussian_pattern_from_descriptor(d)

    d = gaussian_pattern_to_descriptor(pattern)
    d["dimples"][0]["sigma_u"] = True
    with pytest.raises(PatternError, match="sigma_u"):
        gaussian_pattern_from_descriptor(d)


def test_dimples_must_be_list_or_tuple() -> None:
    pattern = _make_synthetic_pattern(count=1)
    d = gaussian_pattern_to_descriptor(pattern)
    d["dimples"] = {"not": "a list"}
    with pytest.raises(PatternError, match="dimples"):
        gaussian_pattern_from_descriptor(d)


def test_empty_pattern_round_trip() -> None:
    pattern = _make_empty_pattern()
    d = gaussian_pattern_to_descriptor(pattern)
    assert d["dimple_count"] == 0
    assert d["dimples"] == []
    rebuilt = gaussian_pattern_from_descriptor(d)
    assert rebuilt.dimples == ()
    assert rebuilt.depth_field.shape == pattern.depth_field.shape
    assert float(rebuilt.depth_field.sum()) == 0.0
    assert rebuilt.resolution == pattern.resolution
    assert rebuilt.max_depth == pytest.approx(pattern.max_depth)
    assert rebuilt.seed == pattern.seed


def test_seed_none_round_trip() -> None:
    rm = _make_risk_map(np.ones((4, 8)))
    pattern = create_gaussian_dimple_pattern(
        rm, count=3, amplitude=1.0, sigma_u=0.05, sigma_v=0.05,
        max_depth=0.25, seed=None,
    )
    d = gaussian_pattern_to_descriptor(pattern)
    assert d["seed"] is None
    rebuilt = gaussian_pattern_from_descriptor(d)
    assert rebuilt.seed is None
