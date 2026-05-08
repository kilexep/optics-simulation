"""Risk-map ring-offset transform for diagnostic pattern placement.

Actual STL offset-ring risk-guided pattern diagnostic; **not a
physical PET-bottle validation**. Earlier pattern-induced hotspot
diagnostics showed that a centered risk-guided Gaussian dimple
pattern (one dimple sampled directly on top of each baseline
hotspot contribution bin) tends to *strengthen* the same caustic
location instead of scattering it: at the worst worsened
``(angle, distance)`` entry, every patterned hotspot bin
overlapped with the baseline bins (``candidate_only_bins == 0``).

This module exposes a small **diagnostic** transformation that
moves probability mass from each active risk pixel to an annular
ring of pixels around it. The transformed :class:`RiskMap` can
then drive a Gaussian dimple sampler that places dimples in a
ring around the original hotspot bins instead of on top of them.

The goal is purely diagnostic: it tests whether avoiding direct
dimple placement on hotspot contribution centers changes hotspot
behavior. It is **not** optimization, **not** mesh repair, and
**not** manufacturing validation.

Coordinate convention
---------------------
- ``u`` (circumferential, axis 1) wraps via :func:`numpy.roll`
  when ``wrap_u=True`` (default). This matches the unit-circle
  ``[0, 1)`` convention used by
  :class:`SurfaceCoordinateMap` and
  :class:`ContributionMap`.
- ``v`` (height, axis 0) does **not** wrap; v-shifts that fall
  off the top or bottom row are dropped.

Limitations
-----------
- The transform uses integer pixel offsets and treats each pixel
  as having unit area. It is a discrete diagnostic, not a
  continuous Green's function.
- ``epsilon`` is added only to active pixels of the **transformed**
  map, mirroring :func:`build_risk_map`'s policy.
- This module does **not** prove physical causality between any
  pattern-placement strategy and any thermal-risk reduction.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** manufacturing-ready.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.contribution.map import ContributionError
from optics_simulation.contribution.risk_map import RiskMap


@dataclass(frozen=True)
class RiskMapRingTransformResult:
    source_risk_map: RiskMap
    transformed_risk_map: RiskMap
    ring_weight_map: np.ndarray
    source_active_count: int
    transformed_active_count: int
    inner_radius_px: int
    outer_radius_px: int
    wrap_u: bool
    result_type: str = "risk_map_ring_offset_transform"


def _validate_radii(inner: int, outer: int) -> tuple[int, int]:
    if isinstance(inner, bool) or not isinstance(inner, int):
        raise ContributionError(
            f"inner_radius_px must be a non-negative int; "
            f"got {inner!r}"
        )
    if isinstance(outer, bool) or not isinstance(outer, int):
        raise ContributionError(
            f"outer_radius_px must be a positive int; got {outer!r}"
        )
    if int(inner) < 0:
        raise ContributionError(
            f"inner_radius_px must be >= 0; got {int(inner)}"
        )
    if int(outer) <= int(inner):
        raise ContributionError(
            f"outer_radius_px must be > inner_radius_px; got "
            f"inner={int(inner)}, outer={int(outer)}"
        )
    return int(inner), int(outer)


def _shift_u(arr: np.ndarray, du: int, *, wrap: bool) -> np.ndarray:
    if du == 0:
        return arr.copy()
    if wrap:
        return np.roll(arr, shift=int(du), axis=1)
    out = np.zeros_like(arr)
    nu = arr.shape[1]
    if du > 0:
        out[:, du:nu] = arr[:, 0:nu - du]
    else:
        absdu = -du
        out[:, 0:nu - absdu] = arr[:, absdu:nu]
    return out


def _shift_v(arr: np.ndarray, dv: int) -> np.ndarray:
    if dv == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    nv = arr.shape[0]
    if dv > 0:
        out[dv:nv, :] = arr[0:nv - dv, :]
    else:
        absdv = -dv
        out[0:nv - absdv, :] = arr[absdv:nv, :]
    return out


def create_ring_offset_risk_map(
    risk_map: RiskMap,
    *,
    inner_radius_px: int = 1,
    outer_radius_px: int = 3,
    wrap_u: bool = True,
    epsilon: float = 0.01,
) -> RiskMapRingTransformResult:
    """Move risk-map probability mass to a ring around each active pixel.

    Actual STL offset-ring risk-guided pattern diagnostic; **not a
    physical PET-bottle validation**. Reads the source
    :class:`RiskMap`'s ``risk_map`` array, accumulates each active
    pixel's value into a ring of pixels around it (offsets
    ``(dv, du)`` with ``inner_radius_px**2 <= dv**2 + du**2 <=
    outer_radius_px**2``), and returns a new :class:`RiskMap` whose
    ``risk_map`` field is the accumulated ring weights plus
    ``epsilon`` on active pixels (and zero elsewhere). The
    ``probability_map`` is renormalized accordingly.

    With ``inner_radius_px >= 1`` the original center pixel is
    excluded from the offset set, so the transformed map's active
    pixels lie strictly around the source's active pixels rather
    than on top of them. With ``inner_radius_px == 0`` and
    ``outer_radius_px > 0`` the center is included as well (the
    transform is a disk blur).

    The input ``risk_map`` is read but never mutated. No file I/O.

    Raises
    ------
    ContributionError
        On invalid risk-map type, non-finite source, invalid
        radii (``inner < 0``, ``outer <= inner``), or negative
        ``epsilon``.
    """
    if not isinstance(risk_map, RiskMap):
        raise ContributionError(
            "risk_map must be a RiskMap; got "
            f"{type(risk_map).__name__}"
        )
    inner, outer = _validate_radii(
        inner_radius_px, outer_radius_px,
    )
    eps = float(epsilon)
    if eps < 0.0:
        raise ContributionError(
            f"epsilon must be >= 0; got {eps}"
        )

    source = np.asarray(risk_map.risk_map, dtype=float)
    if source.ndim != 2:
        raise ContributionError(
            f"risk_map.risk_map must be 2-D; got shape {source.shape}"
        )
    if source.size == 0:
        raise ContributionError(
            "risk_map.risk_map must be non-empty"
        )
    if not np.isfinite(source).all():
        raise ContributionError(
            "risk_map.risk_map contains NaN or inf values"
        )
    if (source < 0.0).any():
        raise ContributionError(
            "risk_map.risk_map contains negative values"
        )

    inner_sq = inner * inner
    outer_sq = outer * outer
    accumulator = np.zeros_like(source, dtype=float)
    for dv in range(-outer, outer + 1):
        for du in range(-outer, outer + 1):
            r2 = dv * dv + du * du
            if r2 < inner_sq or r2 > outer_sq:
                continue
            shifted_u = _shift_u(source, du, wrap=bool(wrap_u))
            shifted = _shift_v(shifted_u, dv)
            accumulator = accumulator + shifted

    active_mask = accumulator > 0.0
    new_risk = np.zeros_like(accumulator, dtype=float)
    new_risk[active_mask] = accumulator[active_mask] + eps
    total_risk = float(new_risk.sum())
    if total_risk > 0.0:
        probability_map = new_risk / total_risk
    else:
        probability_map = np.zeros_like(new_risk, dtype=float)

    transformed = RiskMap(
        risk_map=new_risk.astype(float, copy=True),
        probability_map=probability_map.astype(float, copy=True),
        active_mask=active_mask.astype(bool, copy=True),
        total_risk=total_risk,
        active_count=int(active_mask.sum()),
        epsilon=eps,
        threshold=None,
    )

    return RiskMapRingTransformResult(
        source_risk_map=risk_map,
        transformed_risk_map=transformed,
        ring_weight_map=accumulator.astype(float, copy=True),
        source_active_count=int(risk_map.active_count),
        transformed_active_count=int(active_mask.sum()),
        inner_radius_px=int(inner),
        outer_radius_px=int(outer),
        wrap_u=bool(wrap_u),
    )
