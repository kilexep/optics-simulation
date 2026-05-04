"""Detector pixel grid and per-pixel hit / weight accumulation.

Builds a finite, regular pixel grid on a detector plane and bins
:class:`DetectorHitResult` rows into ``(ny, nx)`` count and weight
maps. Operates entirely on the local 2D coordinates produced by
:func:`intersect_detector_plane`; this module knows nothing about
3D geometry.

Coordinate / pixel convention
-----------------------------
- ``local_x`` (the detector's ``right`` axis) maps to **column**
  index. ``col = floor((local_x + width/2) / pixel_width)``.
- ``local_y`` (the detector's ``up`` axis) maps to **row** index.
  ``row = floor((local_y + height/2) / pixel_height)``.
- ``row = 0`` corresponds to the lowest ``local_y`` (bottom of the
  detector). ``row = ny - 1`` is the top. This is a math-oriented
  layout: it lines up with ``imshow(count_map, origin='lower')``
  for plotting, and avoids hidden y-flips in numerical work.
- ``hit_pixel_indices`` columns are ``(row, col)``, so callers can
  index ``count_map[indices[:, 0], indices[:, 1]]`` directly.
- Boundary inclusivity follows :mod:`detector`: a hit exactly on
  ``local_x = +width/2`` is clamped into the last column, and a
  hit exactly on ``local_x = -width/2`` falls in column 0. The
  same logic applies on the y axis.

Out of scope
------------
Irradiance unit conversion (W/m² normalization), C99 / Eexceed /
Ahot, BTDF entropy, contribution maps, ray-history logging, and
visualization are intentionally not implemented here. This module
is the pixel-binning foundation those layers compose on top of.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.optics.detector import DetectorHitResult
from optics_simulation.optics.ray import OpticsError


@dataclass(frozen=True)
class DetectorGrid:
    width: float
    height: float
    resolution: tuple[int, int]   # (ny, nx)
    pixel_width: float
    pixel_height: float


@dataclass(frozen=True)
class DetectorAccumulationResult:
    count_map: np.ndarray            # (ny, nx) int64
    weight_map: np.ndarray           # (ny, nx) float64
    hit_pixel_indices: np.ndarray    # (N, 2) int64; columns = (row, col); -1 for miss
    hit_mask: np.ndarray             # (N,) bool, copied from input
    total_hits: int
    total_weight: float


def create_detector_grid(
    width: float,
    height: float,
    resolution: tuple[int, int],
) -> DetectorGrid:
    """Build a regular pixel grid covering a ``width`` x ``height`` plane.

    ``resolution`` is ``(ny, nx)``: rows along the local y axis,
    columns along the local x axis. See module docstring for the
    row/column convention.
    """
    if width <= 0.0 or height <= 0.0:
        raise OpticsError(
            f"detector grid width and height must be positive; got "
            f"width={width}, height={height}"
        )

    res = tuple(resolution)
    if len(res) != 2:
        raise OpticsError(
            f"resolution must have exactly 2 entries (ny, nx); got {res}"
        )
    ny = int(res[0])
    nx = int(res[1])
    if ny <= 0 or nx <= 0:
        raise OpticsError(
            f"resolution entries must be positive integers; got (ny, nx)=({ny}, {nx})"
        )

    pixel_width = float(width) / nx
    pixel_height = float(height) / ny
    return DetectorGrid(
        width=float(width),
        height=float(height),
        resolution=(ny, nx),
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )


def accumulate_detector_hits(
    hits: DetectorHitResult,
    grid: DetectorGrid,
    *,
    weights: np.ndarray | None = None,
) -> DetectorAccumulationResult:
    """Bin detector hits into pixel count and weight maps.

    Only rows with ``hits.hit_mask == True`` contribute. Miss rows
    are reported with ``hit_pixel_indices == (-1, -1)`` and do not
    affect ``count_map`` or ``weight_map``. When ``weights`` is
    ``None`` every hit contributes a weight of 1.0; otherwise
    ``weights`` must have shape ``(hits.ray_count,)`` and contain
    only finite values.
    """
    if not isinstance(hits, DetectorHitResult):
        raise OpticsError(
            f"hits must be a DetectorHitResult; got {type(hits).__name__}"
        )
    if not isinstance(grid, DetectorGrid):
        raise OpticsError(
            f"grid must be a DetectorGrid; got {type(grid).__name__}"
        )

    n = int(hits.ray_count)
    ny, nx = grid.resolution

    if weights is None:
        weights_arr = np.ones(n, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != (n,):
            raise OpticsError(
                f"weights must have shape ({n},); got {weights_arr.shape}"
            )
        if not np.isfinite(weights_arr).all():
            raise OpticsError("weights must be finite (no NaN or inf)")

    count_map = np.zeros((ny, nx), dtype=np.int64)
    weight_map = np.zeros((ny, nx), dtype=float)
    hit_pixel_indices = np.full((n, 2), -1, dtype=np.int64)
    hit_mask_copy = np.array(hits.hit_mask, dtype=bool, copy=True)

    if n == 0:
        return DetectorAccumulationResult(
            count_map=count_map,
            weight_map=weight_map,
            hit_pixel_indices=hit_pixel_indices,
            hit_mask=hit_mask_copy,
            total_hits=0,
            total_weight=0.0,
        )

    hit_idx = np.flatnonzero(hits.hit_mask)
    if hit_idx.size == 0:
        return DetectorAccumulationResult(
            count_map=count_map,
            weight_map=weight_map,
            hit_pixel_indices=hit_pixel_indices,
            hit_mask=hit_mask_copy,
            total_hits=0,
            total_weight=0.0,
        )

    lx = hits.local_xy[hit_idx, 0]
    ly = hits.local_xy[hit_idx, 1]

    half_w = grid.width / 2.0
    half_h = grid.height / 2.0

    col = np.floor((lx + half_w) / grid.pixel_width).astype(np.int64)
    row = np.floor((ly + half_h) / grid.pixel_height).astype(np.int64)

    # Clamp inclusive boundaries: +w/2 maps to nx (overflow) and tiny
    # negative float noise around -w/2 can underflow to -1.
    col = np.clip(col, 0, nx - 1)
    row = np.clip(row, 0, ny - 1)

    hit_pixel_indices[hit_idx, 0] = row
    hit_pixel_indices[hit_idx, 1] = col

    hit_weights = weights_arr[hit_idx]
    np.add.at(count_map, (row, col), 1)
    np.add.at(weight_map, (row, col), hit_weights)

    return DetectorAccumulationResult(
        count_map=count_map,
        weight_map=weight_map,
        hit_pixel_indices=hit_pixel_indices,
        hit_mask=hit_mask_copy,
        total_hits=int(hit_idx.size),
        total_weight=float(hit_weights.sum()),
    )
