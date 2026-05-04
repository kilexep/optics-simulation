import numpy as np
import pytest

from optics_simulation.optics import (
    OpticsError,
    RefractionResult,
    fresnel_unpolarized,
    normalize_vector,
    refract_direction,
)

# Mitsuba dielectric preset values, used as Snell/Fresnel sanity baselines.
IOR_AIR = 1.00028
IOR_PET = 1.575
IOR_WATER = 1.3330


def test_normalize_vector_returns_unit_norm() -> None:
    v = normalize_vector([3.0, 0.0, 4.0])
    assert np.linalg.norm(v) == pytest.approx(1.0)
    np.testing.assert_allclose(v, [0.6, 0.0, 0.8])


def test_normalize_vector_accepts_tuple_and_ndarray() -> None:
    a = normalize_vector((0.0, 0.0, 2.0))
    b = normalize_vector(np.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(a, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(b, [0.0, 0.0, 1.0])


def test_normalize_vector_zero_raises() -> None:
    with pytest.raises(OpticsError, match="zero-length"):
        normalize_vector([0.0, 0.0, 0.0])


def test_normalize_vector_wrong_size_raises() -> None:
    with pytest.raises(OpticsError, match="3 components"):
        normalize_vector([1.0, 2.0])


def test_fresnel_normal_incidence_air_to_pet_matches_closed_form() -> None:
    r, t = fresnel_unpolarized(
        cos_i=1.0, cos_t=1.0, eta_i=IOR_AIR, eta_t=IOR_PET
    )
    expected = ((IOR_PET - IOR_AIR) / (IOR_PET + IOR_AIR)) ** 2
    assert r == pytest.approx(expected, abs=1e-9)
    assert t == pytest.approx(1.0 - expected, abs=1e-9)


def test_fresnel_normal_incidence_pet_to_air_reciprocity() -> None:
    r_forward, _ = fresnel_unpolarized(1.0, 1.0, IOR_AIR, IOR_PET)
    r_reverse, _ = fresnel_unpolarized(1.0, 1.0, IOR_PET, IOR_AIR)
    assert r_forward == pytest.approx(r_reverse, abs=1e-9)


def test_fresnel_r_plus_t_equals_one_oblique() -> None:
    cases = [
        (0.95, 0.97, IOR_AIR, IOR_PET),
        (0.5, np.sqrt(1.0 - (IOR_AIR / IOR_PET) ** 2 * (1.0 - 0.5 ** 2)),
         IOR_AIR, IOR_PET),
        (0.7, np.sqrt(1.0 - (IOR_WATER / IOR_PET) ** 2 * (1.0 - 0.7 ** 2)),
         IOR_WATER, IOR_PET),
    ]
    for ci, ct, ni, nt in cases:
        r, t = fresnel_unpolarized(ci, ct, ni, nt)
        assert r + t == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= t <= 1.0


def test_fresnel_nonpositive_eta_raises() -> None:
    with pytest.raises(OpticsError, match="positive"):
        fresnel_unpolarized(1.0, 1.0, 0.0, IOR_PET)
    with pytest.raises(OpticsError, match="positive"):
        fresnel_unpolarized(1.0, 1.0, IOR_AIR, -1.0)


def test_fresnel_zero_denominator_raises() -> None:
    # cos_i = cos_t = 0 with any eta drives both denominators to zero.
    with pytest.raises(OpticsError, match="denominator"):
        fresnel_unpolarized(0.0, 0.0, IOR_AIR, IOR_PET)


def test_refract_direction_normal_incidence_air_to_pet_keeps_direction() -> None:
    result = refract_direction(
        wi=[0.0, 0.0, -1.0],
        normal=[0.0, 0.0, 1.0],
        eta_i=IOR_AIR,
        eta_t=IOR_PET,
    )
    assert isinstance(result, RefractionResult)
    assert not result.total_internal_reflection
    assert result.direction is not None
    np.testing.assert_allclose(result.direction, [0.0, 0.0, -1.0], atol=1e-12)
    assert np.linalg.norm(result.direction) == pytest.approx(1.0)
    assert result.cos_i == pytest.approx(1.0)
    assert result.cos_t == pytest.approx(1.0)
    assert result.reflectance + result.transmittance == pytest.approx(1.0, abs=1e-12)


def test_refract_direction_oblique_air_to_pet_unit_norm_and_bends_toward_normal() -> None:
    sin_i = 0.5
    wi = np.array([sin_i, 0.0, -np.sqrt(1.0 - sin_i ** 2)])
    result = refract_direction(
        wi=wi,
        normal=[0.0, 0.0, 1.0],
        eta_i=IOR_AIR,
        eta_t=IOR_PET,
    )
    assert not result.total_internal_reflection
    assert result.direction is not None
    assert np.linalg.norm(result.direction) == pytest.approx(1.0, abs=1e-12)
    # bending toward normal: |z| component grows as ray enters denser medium
    assert abs(result.direction[2]) > abs(wi[2])
    # transverse component shrinks per Snell
    assert abs(result.direction[0]) < abs(wi[0])


def test_refract_direction_total_internal_reflection_pet_to_air() -> None:
    # critical angle = arcsin(1/1.575) ≈ 39.4°; use 60° (well above critical)
    angle = np.deg2rad(60.0)
    wi = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    result = refract_direction(
        wi=wi,
        normal=[0.0, 0.0, 1.0],
        eta_i=IOR_PET,
        eta_t=IOR_AIR,
    )
    assert result.total_internal_reflection is True
    assert result.direction is None
    assert result.cos_t is None
    assert result.reflectance == pytest.approx(1.0)
    assert result.transmittance == pytest.approx(0.0)
    assert result.cos_i >= 0.0


def test_refract_direction_below_critical_angle_pet_to_air_transmits() -> None:
    angle = np.deg2rad(20.0)  # well below critical
    wi = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    result = refract_direction(
        wi=wi,
        normal=[0.0, 0.0, 1.0],
        eta_i=IOR_PET,
        eta_t=IOR_AIR,
    )
    assert result.total_internal_reflection is False
    assert result.direction is not None
    assert np.linalg.norm(result.direction) == pytest.approx(1.0, abs=1e-12)
    assert result.reflectance + result.transmittance == pytest.approx(1.0, abs=1e-12)


def test_refract_direction_r_plus_t_equals_one_for_non_tir() -> None:
    angle = np.deg2rad(30.0)
    wi = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    result = refract_direction(wi, [0.0, 0.0, 1.0], IOR_AIR, IOR_PET)
    assert not result.total_internal_reflection
    assert result.reflectance + result.transmittance == pytest.approx(1.0, abs=1e-12)


def test_refract_direction_back_face_flip_preserves_eta() -> None:
    # New convention: when the supplied normal is back-facing
    # (cos_i < 0 against wi), the function flips the normal for
    # geometric correctness only. eta_i (current medium) and eta_t
    # (next medium) are caller-authoritative and must NOT be swapped.
    wi = np.array([0.3, 0.0, -np.sqrt(1.0 - 0.09)])
    result = refract_direction(
        wi=wi,
        normal=[0.0, 0.0, -1.0],  # back-facing for this incident ray
        eta_i=IOR_AIR,
        eta_t=IOR_PET,
    )
    assert result.cos_i >= 0.0
    # Eta is preserved exactly as the caller supplied (no swap):
    assert result.eta_i == pytest.approx(IOR_AIR)
    assert result.eta_t == pytest.approx(IOR_PET)
    assert result.reflectance + result.transmittance == pytest.approx(1.0, abs=1e-12)


def test_refract_direction_zero_wi_raises() -> None:
    with pytest.raises(OpticsError, match="zero-length"):
        refract_direction([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], IOR_AIR, IOR_PET)


def test_refract_direction_zero_normal_raises() -> None:
    with pytest.raises(OpticsError, match="zero-length"):
        refract_direction([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], IOR_AIR, IOR_PET)


def test_sanity_normal_incidence_reflectances_match_textbook() -> None:
    # Spot-check normal-incidence R for three common interfaces.
    r_air_pet, _ = fresnel_unpolarized(1.0, 1.0, IOR_AIR, IOR_PET)
    r_air_water, _ = fresnel_unpolarized(1.0, 1.0, IOR_AIR, IOR_WATER)
    r_water_pet, _ = fresnel_unpolarized(1.0, 1.0, IOR_WATER, IOR_PET)
    assert r_air_pet == pytest.approx(0.04965, abs=5e-4)
    assert r_air_water == pytest.approx(0.02035, abs=5e-4)
    assert r_water_pet == pytest.approx(0.00650, abs=5e-4)


def test_refract_direction_pet_to_air_oblique_bends_away_from_normal() -> None:
    # PET -> AIR below the critical angle: the transmitted ray bends
    # AWAY from the normal (angle from normal grows). Equivalently
    # |z|-component shrinks and the transverse component grows.
    angle = np.deg2rad(20.0)
    wi = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    result = refract_direction(
        wi=wi,
        normal=[0.0, 0.0, 1.0],
        eta_i=IOR_PET,
        eta_t=IOR_AIR,
    )
    assert not result.total_internal_reflection
    assert result.direction is not None
    assert np.linalg.norm(result.direction) == pytest.approx(1.0, abs=1e-12)
    assert abs(result.direction[2]) < abs(wi[2])
    assert abs(result.direction[0]) > abs(wi[0])
    assert result.eta_i == pytest.approx(IOR_PET)
    assert result.eta_t == pytest.approx(IOR_AIR)


def test_refract_direction_pet_to_air_above_critical_is_tir() -> None:
    # New-convention regression: PET -> AIR at 60 deg (well above the
    # ~39.4 deg critical angle for n=1.575) must still TIR, with eta
    # echoed back unchanged.
    angle = np.deg2rad(60.0)
    wi = np.array([np.sin(angle), 0.0, -np.cos(angle)])
    result = refract_direction(
        wi=wi,
        normal=[0.0, 0.0, 1.0],
        eta_i=IOR_PET,
        eta_t=IOR_AIR,
    )
    assert result.total_internal_reflection is True
    assert result.direction is None
    assert result.cos_t is None
    assert result.reflectance == pytest.approx(1.0)
    assert result.transmittance == pytest.approx(0.0)
    # Eta is caller-authoritative — preserved on TIR too.
    assert result.eta_i == pytest.approx(IOR_PET)
    assert result.eta_t == pytest.approx(IOR_AIR)


def test_refract_direction_eta_preserved_under_both_normal_orientations() -> None:
    # Same wi, same caller-supplied (eta_i, eta_t), two normal
    # orientations: outward-facing and back-facing. The function must
    # report the caller's eta values unchanged in both calls.
    wi = [0.3, 0.0, -float(np.sqrt(0.91))]
    forward = refract_direction(
        wi=wi, normal=[0.0, 0.0, 1.0], eta_i=IOR_AIR, eta_t=IOR_PET,
    )
    backward = refract_direction(
        wi=wi, normal=[0.0, 0.0, -1.0], eta_i=IOR_AIR, eta_t=IOR_PET,
    )
    for r in (forward, backward):
        assert r.cos_i >= 0.0
        assert r.eta_i == pytest.approx(IOR_AIR)
        assert r.eta_t == pytest.approx(IOR_PET)
        assert not r.total_internal_reflection
        assert r.reflectance + r.transmittance == pytest.approx(1.0, abs=1e-12)
    # Geometric correction makes the back-face case agree with the
    # forward case at the cos_i level.
    assert forward.cos_i == pytest.approx(backward.cos_i, abs=1e-12)


def test_refract_direction_result_eta_equals_caller_eta_for_pet_to_air() -> None:
    # Direct lock on requirements 6 and 7: result.eta_i / result.eta_t
    # must echo the caller-supplied values for the PET -> AIR direction
    # as well, including under back-face normal flip.
    wi = [0.3, 0.0, -float(np.sqrt(0.91))]
    result = refract_direction(
        wi=wi,
        normal=[0.0, 0.0, -1.0],  # back-face for this ray
        eta_i=IOR_PET,
        eta_t=IOR_AIR,
    )
    assert result.eta_i == pytest.approx(IOR_PET)
    assert result.eta_t == pytest.approx(IOR_AIR)
    assert result.cos_i >= 0.0
