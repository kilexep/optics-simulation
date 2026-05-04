"""Snell refraction and unpolarized Fresnel coefficients (scalar).

Pure-function module. Given an incident direction, surface normal, and
two refractive indices, returns the transmitted (refracted) direction
together with unpolarized Fresnel reflectance / transmittance and a
total-internal-reflection flag.

Convention
----------
- ``wi`` points **toward** the surface (i.e. it is the direction of
  propagation of the incident ray as it approaches the interface).
- ``normal`` is the geometric surface normal. It is *ideally* directed
  from the incident medium into the transmitting medium so that
  ``cos_i = -dot(wi, normal) > 0``. When the caller passes the
  opposite-facing normal (back-face hit), this function detects that
  via ``cos_i < 0`` and internally flips the normal **and** swaps
  ``eta_i`` ↔ ``eta_t``. The returned ``RefractionResult`` reports
  ``cos_i``, ``eta_i``, ``eta_t`` after that swap, so the values
  describe the actual incident / transmitting media for this hit.
- The returned ``direction`` is the **transmitted** ray direction
  (unit length). On total internal reflection it is ``None``. The
  reflected direction is intentionally not returned here — it belongs
  to the multi-bounce tracer that consumes this result.

Limitations
-----------
- **Unpolarized only.** Stokes / Jones polarization is not modeled.
- **Real refractive index only.** Complex (absorbing) indices are not
  supported; the formulas will be inaccurate for strongly absorbing
  media.
- **Smooth dielectric only.** Rough-dielectric / micro-facet BTDF
  (e.g. Mitsuba ``roughdielectric``) is not modeled.
- **No dispersion.** A single wavelength is assumed; ``eta_i`` /
  ``eta_t`` must already be evaluated at the wavelength of interest.
- **Scalar interface.** This module operates on a single ray at a
  time. Vectorization over a ray bundle is intentionally deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from optics_simulation.optics.ray import OpticsError


@dataclass(frozen=True)
class RefractionResult:
    direction: np.ndarray | None
    reflectance: float
    transmittance: float
    total_internal_reflection: bool
    cos_i: float
    cos_t: float | None
    eta_i: float
    eta_t: float


_FRESNEL_DENOM_EPS = 1e-12


def normalize_vector(
    v: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return a unit-length copy of ``v``. Raises OpticsError on zero / wrong size."""
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size != 3:
        raise OpticsError(f"vector must have 3 components; got size {arr.size}")
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        raise OpticsError("zero-length vector cannot be normalized")
    return arr / n


def fresnel_unpolarized(
    cos_i: float,
    cos_t: float,
    eta_i: float,
    eta_t: float,
) -> tuple[float, float]:
    """Unpolarized Fresnel reflectance and transmittance.

    Returns ``(R, T)`` with ``R`` clamped to ``[0, 1]`` and
    ``T = 1.0 - R`` for numerical stability. Raises OpticsError if
    either index is non-positive or if a Fresnel denominator collapses
    to zero (a degenerate interface).
    """
    if eta_i <= 0.0 or eta_t <= 0.0:
        raise OpticsError(
            f"refractive indices must be positive; got eta_i={eta_i}, eta_t={eta_t}"
        )

    ci = float(cos_i)
    ct = float(cos_t)
    ni = float(eta_i)
    nt = float(eta_t)

    denom_s = ni * ci + nt * ct
    denom_p = nt * ci + ni * ct
    if abs(denom_s) < _FRESNEL_DENOM_EPS or abs(denom_p) < _FRESNEL_DENOM_EPS:
        raise OpticsError(
            "Fresnel denominator is numerically zero; degenerate interface "
            f"(eta_i={ni}, eta_t={nt}, cos_i={ci}, cos_t={ct})"
        )

    rs = ((ni * ci - nt * ct) / denom_s) ** 2
    rp = ((nt * ci - ni * ct) / denom_p) ** 2
    r = 0.5 * (rs + rp)
    r_clamped = float(min(1.0, max(0.0, r)))
    t_clamped = 1.0 - r_clamped
    return r_clamped, t_clamped


def refract_direction(
    wi: np.ndarray | Sequence[float],
    normal: np.ndarray | Sequence[float],
    eta_i: float,
    eta_t: float,
) -> RefractionResult:
    """Snell refraction + Fresnel for a single incident ray.

    See module docstring for the sign / direction convention. Back-face
    hits (``cos_i < 0`` against the supplied normal) are auto-corrected
    by flipping the normal and swapping ``eta_i`` / ``eta_t``; the
    returned result reflects the post-swap media.
    """
    wi_n = normalize_vector(wi)
    n = normalize_vector(normal)

    cos_i = float(-np.dot(wi_n, n))
    if cos_i < 0.0:
        n = -n
        eta_i, eta_t = eta_t, eta_i
        cos_i = float(-np.dot(wi_n, n))

    eta = float(eta_i) / float(eta_t)
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)

    if sin2_t > 1.0:
        return RefractionResult(
            direction=None,
            reflectance=1.0,
            transmittance=0.0,
            total_internal_reflection=True,
            cos_i=cos_i,
            cos_t=None,
            eta_i=float(eta_i),
            eta_t=float(eta_t),
        )

    cos_t = float(np.sqrt(max(0.0, 1.0 - sin2_t)))
    wo = eta * wi_n + (eta * cos_i - cos_t) * n
    wo = normalize_vector(wo)

    r, t = fresnel_unpolarized(cos_i, cos_t, float(eta_i), float(eta_t))
    return RefractionResult(
        direction=wo,
        reflectance=r,
        transmittance=t,
        total_internal_reflection=False,
        cos_i=cos_i,
        cos_t=cos_t,
        eta_i=float(eta_i),
        eta_t=float(eta_t),
    )
