"""Risk-map post-processing for contribution maps.

Takes a :class:`ContributionMap` and produces a :class:`RiskMap`
suitable as the input distribution for downstream pattern sampling
(``Rrisk_norm(u, v) + epsilon`` per CLAUDE.md). This module is the
post-processing step between contribution-map binning and dimple
center sampling; it does **not** perform any sampling itself.

Policy
------
- ``use_normalized=True`` (default) reads ``contribution.normalized_map``.
  ``use_normalized=False`` reads ``contribution.weight_map``.
- ``threshold`` is a strict greater-than cutoff applied to ``base``.
  ``threshold=None`` is equivalent to ``threshold=0``: the active
  region is exactly the support of the base map (``base > 0``).
- ``epsilon`` is added **only to active pixels**. Inactive pixels
  remain zero. This keeps a downstream sampler from leaking mass
  outside the active region while still avoiding zero-mass spikes
  inside it.
- Probability map: ``risk_map / total_risk`` when ``total_risk > 0``,
  otherwise all zeros. An all-zero base with ``epsilon > 0`` still
  yields an all-zero probability map because no pixel is active.

Out of scope
------------
Gaussian dimple center sampling, Poisson disk sampling, pattern
generation, pattern displacement, optimization, residual hotspot
update, thermal modeling, visualization, file export, config wiring,
SciPy-based smoothing, Open3D backend, and actual PET STL handling
are intentionally not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.contribution.map import (
    ContributionError,
    ContributionMap,
)


@dataclass(frozen=True)
class RiskMap:
    """Risk / probability map derived from a :class:`ContributionMap`.

    Fields
    ------
    risk_map
        Same shape as the contribution map. Equals ``base + epsilon``
        on active pixels and ``0`` everywhere else.
    probability_map
        Same shape. ``risk_map / total_risk`` if ``total_risk > 0``,
        otherwise an all-zero map.
    active_mask
        Boolean map of pixels that survive the ``threshold`` /
        ``base > 0`` filter. This is the final active region:
        downstream pattern sampling must remain inside this mask.
    total_risk
        ``float(risk_map.sum())``.
    active_count
        ``int(active_mask.sum())``.
    epsilon
        Echoed back from input.
    threshold
        Echoed back from input (``None`` or non-negative float).
    """

    risk_map: np.ndarray
    probability_map: np.ndarray
    active_mask: np.ndarray
    total_risk: float
    active_count: int
    epsilon: float
    threshold: float | None


def _validate_contribution_shapes(contribution: ContributionMap) -> None:
    cs = contribution.count_map.shape
    ws = contribution.weight_map.shape
    ns = contribution.normalized_map.shape
    if not (cs == ws == ns):
        raise ContributionError(
            f"contribution maps have inconsistent shapes: "
            f"count_map={cs}, weight_map={ws}, normalized_map={ns}"
        )


def _validate_base(base: np.ndarray) -> None:
    if not np.isfinite(base).all():
        raise ContributionError("base map contains NaN or inf values")
    if (base < 0.0).any():
        raise ContributionError("base map contains negative values")


def build_risk_map(
    contribution: ContributionMap,
    *,
    use_normalized: bool = True,
    threshold: float | None = None,
    epsilon: float = 0.0,
) -> RiskMap:
    """Post-process a :class:`ContributionMap` into a :class:`RiskMap`.

    See module docstring for the threshold / epsilon policy.

    Raises
    ------
    ContributionError
        On invalid inputs: ``contribution`` is not a
        :class:`ContributionMap`; the contribution maps' shapes
        disagree; ``threshold`` is negative; ``epsilon`` is
        negative; the base map contains NaN, inf, or negative
        values.
    """
    if not isinstance(contribution, ContributionMap):
        raise ContributionError(
            f"contribution must be a ContributionMap; "
            f"got {type(contribution).__name__}"
        )

    _validate_contribution_shapes(contribution)

    if threshold is not None and float(threshold) < 0.0:
        raise ContributionError(
            f"threshold must be >= 0 when provided; got {threshold}"
        )
    if float(epsilon) < 0.0:
        raise ContributionError(f"epsilon must be >= 0; got {epsilon}")

    if use_normalized:
        base = np.asarray(contribution.normalized_map, dtype=float)
    else:
        base = np.asarray(contribution.weight_map, dtype=float)

    _validate_base(base)

    if threshold is None:
        active_mask = base > 0.0
    else:
        active_mask = base > float(threshold)

    risk_map = np.zeros_like(base, dtype=float)
    risk_map[active_mask] = base[active_mask] + float(epsilon)

    total_risk = float(risk_map.sum())
    if total_risk > 0.0:
        probability_map = risk_map / total_risk
    else:
        probability_map = np.zeros_like(risk_map, dtype=float)

    return RiskMap(
        risk_map=risk_map,
        probability_map=probability_map,
        active_mask=active_mask,
        total_risk=total_risk,
        active_count=int(active_mask.sum()),
        epsilon=float(epsilon),
        threshold=None if threshold is None else float(threshold),
    )
