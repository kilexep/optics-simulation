"""Contribution map foundation.

Bins the surface ``(u, v)`` coordinates of selected hotspot rays
into a 2D grid to produce a caustic contribution map ``Rrisk(u, v)``
surrogate. Pure numpy: takes a :class:`HotspotSelection` (which
selected rays are hot) and a :class:`HitSurfaceCoordinates` (where
each ray hit on the mesh surface), and returns a
:class:`ContributionMap` with ``count_map``, ``weight_map``, and
``normalized_map``.

Coordinate convention
---------------------
- ``resolution`` is ``(nv, nu)``: rows along ``v``, columns along
  ``u`` — same row/column convention as detector accumulation.
- ``u`` is the circumferential coordinate in ``[0, 1)``; ``u = 0``
  maps to column 0 and ``u`` close to 1 maps to column ``nu - 1``.
- ``v`` is the height coordinate in ``[0, 1]``; ``v = 0`` maps to
  row 0 and ``v = 1`` maps to row ``nv - 1``.
- Bin indices are ``floor(u * nu)`` and ``floor(v * nv)``, then
  clipped into the valid range so that ``v = 1`` and FP-edge
  ``u`` values land in the last column / row instead of overflowing.

NaN / inf / range policy
------------------------
- A selected ray with ``NaN`` ``u`` or ``v`` is silently skipped
  (does not count toward ``total_selected``). Non-selected rays
  with ``NaN`` / ``inf`` ``u`` or ``v`` are ignored entirely — they
  are never inspected.
- A selected ray with ``inf`` ``u`` or ``v`` raises
  :class:`ContributionError` rather than being silently skipped:
  ``inf`` is unambiguously a bug upstream, not a "miss" sentinel.
- A selected ray with ``u`` or ``v`` outside the unit interval by
  more than ``1e-12`` raises :class:`ContributionError`. Values
  inside that tolerance are clipped into ``[0, nextafter(1, 0)]``
  for ``u`` and ``[0, 1]`` for ``v`` before binning.

Normalization
-------------
- ``normalize=True`` with positive ``weight_map.max()``:
  ``normalized_map = weight_map / weight_map.max()``.
- ``normalize=True`` with zero ``weight_map.max()``: zero map.
- ``normalize=False``: ``normalized_map`` is a plain copy of
  ``weight_map`` (raw weight, no scaling).

Out of scope
------------
Pattern generation (Gaussian dimples), pattern optimization,
residual hotspot updates, thermal modeling, visualization,
ray-history Parquet/CSV, file export, and detector / hotspot /
hit-coordinate / angle-scan modifications are intentionally not
implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.geometry.hit_coordinates import HitSurfaceCoordinates
from optics_simulation.metrics.hotspot_selection import HotspotSelection


_TOL = 1e-12


class ContributionError(Exception):
    """Raised for invalid contribution-map inputs."""


@dataclass(frozen=True)
class ContributionMap:
    count_map: np.ndarray            # (nv, nu) int64
    weight_map: np.ndarray           # (nv, nu) float64
    normalized_map: np.ndarray       # (nv, nu) float64
    resolution: tuple[int, int]      # (nv, nu)
    total_selected: int              # number of valid selected rays binned
    total_weight: float              # weight_map.sum()
    selected_ray_indices: np.ndarray  # (M_valid,) int64; NaN-skipped subset
    u_bin_indices: np.ndarray        # (M_valid,) int64
    v_bin_indices: np.ndarray        # (M_valid,) int64


def _validate_resolution(resolution: tuple[int, int]) -> tuple[int, int]:
    res = tuple(resolution)
    if len(res) != 2:
        raise ContributionError(
            f"resolution must have exactly 2 entries (nv, nu); got {res}"
        )
    nv = int(res[0])
    nu = int(res[1])
    if nv <= 0 or nu <= 0:
        raise ContributionError(
            f"resolution entries must be positive integers; "
            f"got (nv, nu)=({nv}, {nu})"
        )
    return nv, nu


def _validate_inputs(
    selection: HotspotSelection,
    hit_coordinates: HitSurfaceCoordinates,
) -> None:
    if not isinstance(selection, HotspotSelection):
        raise ContributionError(
            f"selection must be a HotspotSelection; "
            f"got {type(selection).__name__}"
        )
    if not isinstance(hit_coordinates, HitSurfaceCoordinates):
        raise ContributionError(
            f"hit_coordinates must be a HitSurfaceCoordinates; "
            f"got {type(hit_coordinates).__name__}"
        )

    if int(selection.ray_count) != int(hit_coordinates.ray_count):
        raise ContributionError(
            f"selection.ray_count ({selection.ray_count}) does not match "
            f"hit_coordinates.ray_count ({hit_coordinates.ray_count})"
        )
    if selection.selected_mask.shape != hit_coordinates.u.shape:
        raise ContributionError(
            f"selection.selected_mask shape {selection.selected_mask.shape} "
            f"does not match hit_coordinates.u shape {hit_coordinates.u.shape}"
        )

    indices = np.asarray(selection.selected_ray_indices, dtype=np.int64)
    weights = np.asarray(selection.selected_weights, dtype=float)
    if weights.shape[0] != indices.shape[0]:
        raise ContributionError(
            f"selection.selected_weights length ({weights.shape[0]}) "
            f"does not match selection.selected_ray_indices length "
            f"({indices.shape[0]})"
        )

    n = int(selection.ray_count)
    if indices.size > 0:
        if int(indices.min()) < 0 or int(indices.max()) >= n:
            raise ContributionError(
                f"selection.selected_ray_indices contain out-of-range entries "
                f"(min={int(indices.min())}, max={int(indices.max())}, "
                f"ray_count={n})"
            )

    if not np.isfinite(weights).all():
        raise ContributionError(
            "selection.selected_weights must be finite (no NaN or inf)"
        )
    if (weights < 0.0).any():
        bad = int(np.argmin(weights))
        raise ContributionError(
            f"selection.selected_weights contain a negative value "
            f"(index {bad}, value {float(weights[bad])})"
        )


def create_contribution_map(
    selection: HotspotSelection,
    hit_coordinates: HitSurfaceCoordinates,
    *,
    resolution: tuple[int, int] = (64, 128),
    normalize: bool = True,
) -> ContributionMap:
    """Bin the (u, v) of selected hotspot rays into a 2D contribution map.

    Selected rays come from ``selection.selected_ray_indices``; this
    is the authoritative source. ``selection.selected_mask`` is only
    used for shape consistency. See the module docstring for the
    coordinate, NaN/inf/range, and normalization policies.

    Raises
    ------
    ContributionError
        On mismatched ray counts or shapes, invalid resolution,
        invalid selected_weights (length mismatch, non-finite, or
        negative), out-of-range selected indices, ``inf`` ``u`` or
        ``v`` on a selected ray, or ``u`` / ``v`` of a selected
        non-NaN ray that falls outside the unit interval by more
        than ``1e-12``.
    """
    _validate_inputs(selection, hit_coordinates)
    nv, nu = _validate_resolution(resolution)

    indices = np.asarray(selection.selected_ray_indices, dtype=np.int64)
    weights_all = np.asarray(selection.selected_weights, dtype=float)

    count_map = np.zeros((nv, nu), dtype=np.int64)
    weight_map = np.zeros((nv, nu), dtype=float)

    if indices.size == 0:
        normalized_map = (
            weight_map.copy() if not normalize else np.zeros_like(weight_map)
        )
        return ContributionMap(
            count_map=count_map,
            weight_map=weight_map,
            normalized_map=normalized_map,
            resolution=(nv, nu),
            total_selected=0,
            total_weight=0.0,
            selected_ray_indices=np.empty(0, dtype=np.int64),
            u_bin_indices=np.empty(0, dtype=np.int64),
            v_bin_indices=np.empty(0, dtype=np.int64),
        )

    u_sel = np.asarray(hit_coordinates.u, dtype=float)[indices]
    v_sel = np.asarray(hit_coordinates.v, dtype=float)[indices]

    if np.isposinf(u_sel).any() or np.isneginf(u_sel).any():
        bad = int(np.argmax(np.isinf(u_sel)))
        raise ContributionError(
            f"selected ray has inf u (index {int(indices[bad])}, u={float(u_sel[bad])})"
        )
    if np.isposinf(v_sel).any() or np.isneginf(v_sel).any():
        bad = int(np.argmax(np.isinf(v_sel)))
        raise ContributionError(
            f"selected ray has inf v (index {int(indices[bad])}, v={float(v_sel[bad])})"
        )

    valid = ~(np.isnan(u_sel) | np.isnan(v_sel))
    if not valid.any():
        normalized_map = (
            weight_map.copy() if not normalize else np.zeros_like(weight_map)
        )
        return ContributionMap(
            count_map=count_map,
            weight_map=weight_map,
            normalized_map=normalized_map,
            resolution=(nv, nu),
            total_selected=0,
            total_weight=0.0,
            selected_ray_indices=np.empty(0, dtype=np.int64),
            u_bin_indices=np.empty(0, dtype=np.int64),
            v_bin_indices=np.empty(0, dtype=np.int64),
        )

    u_v = u_sel[valid]
    v_v = v_sel[valid]
    w_v = weights_all[valid]
    idx_v = indices[valid]

    if (u_v < -_TOL).any() or (u_v >= 1.0 + _TOL).any():
        bad = int(
            np.argmax((u_v < -_TOL) | (u_v >= 1.0 + _TOL))
        )
        raise ContributionError(
            f"selected ray has u outside [0, 1) tolerance "
            f"(ray index {int(idx_v[bad])}, u={float(u_v[bad])})"
        )
    if (v_v < -_TOL).any() or (v_v > 1.0 + _TOL).any():
        bad = int(
            np.argmax((v_v < -_TOL) | (v_v > 1.0 + _TOL))
        )
        raise ContributionError(
            f"selected ray has v outside [0, 1] tolerance "
            f"(ray index {int(idx_v[bad])}, v={float(v_v[bad])})"
        )

    u_clipped = np.clip(u_v, 0.0, np.nextafter(1.0, 0.0))
    v_clipped = np.clip(v_v, 0.0, 1.0)

    u_bin = np.floor(u_clipped * nu).astype(np.int64)
    v_bin = np.floor(v_clipped * nv).astype(np.int64)
    u_bin = np.clip(u_bin, 0, nu - 1)
    v_bin = np.clip(v_bin, 0, nv - 1)

    np.add.at(count_map, (v_bin, u_bin), 1)
    np.add.at(weight_map, (v_bin, u_bin), w_v)

    if normalize:
        mx = float(weight_map.max())
        if mx > 0.0:
            normalized_map = weight_map / mx
        else:
            normalized_map = np.zeros_like(weight_map)
    else:
        normalized_map = weight_map.copy()

    return ContributionMap(
        count_map=count_map,
        weight_map=weight_map,
        normalized_map=normalized_map,
        resolution=(nv, nu),
        total_selected=int(idx_v.size),
        total_weight=float(weight_map.sum()),
        selected_ray_indices=idx_v.astype(np.int64, copy=True),
        u_bin_indices=u_bin.astype(np.int64, copy=True),
        v_bin_indices=v_bin.astype(np.int64, copy=True),
    )
