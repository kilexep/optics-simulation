"""Hotspot ray selection on detector pixel maps.

Selects "hot" pixels on a detector grid (either above an absolute
threshold, or in the top-percent of pixel values) and back-maps the
selection to the input ray indices that landed in those pixels.

Out of scope
------------
Contribution-map generation, surface (u, v) accumulation,
``Rrisk(u, v)`` back-tracking, ray-history Parquet/CSV, pattern
generation, optimization, thermal modeling, and visualization are
intentionally not implemented here. This module only answers
"which detector pixels are hot, and which input rays produced
them?".

Per-ray weight limitation
-------------------------
:class:`DetectorAccumulationResult` does not store the per-ray
weights it consumed during accumulation, so this module cannot
recover them. Until a ray-history layer is added, every selected
ray contributes ``selected_weights = 1.0`` as a placeholder. This
is **not** a true per-ray irradiance weight; callers that need
true energy-weighted hotspot rays must wait for the ray-history
follow-up task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.metrics.optical import MetricsError
from optics_simulation.optics.detector import DetectorHitResult
from optics_simulation.optics.detector_accumulation import (
    DetectorAccumulationResult,
)


@dataclass(frozen=True)
class HotspotSelection:
    selected_mask: np.ndarray          # (N,) bool, dense ray-aligned
    selected_ray_indices: np.ndarray   # (M,) int64 = np.flatnonzero(selected_mask)
    selected_pixel_indices: np.ndarray  # (M, 2) int64; columns = (row, col)
    selected_weights: np.ndarray       # (M,) float64; placeholder all-ones
    pixel_mask: np.ndarray             # (ny, nx) bool, selected detector pixels
    ray_count: int                     # = hits.ray_count
    selected_count: int                # = M
    selection_mode: str                # "threshold" | "top_percent"


def _validate_inputs(
    hits: DetectorHitResult,
    accumulation: DetectorAccumulationResult,
    threshold: float | None,
    top_percent: float | None,
) -> None:
    if not isinstance(hits, DetectorHitResult):
        raise MetricsError(
            f"hits must be a DetectorHitResult; got {type(hits).__name__}"
        )
    if not isinstance(accumulation, DetectorAccumulationResult):
        raise MetricsError(
            f"accumulation must be a DetectorAccumulationResult; "
            f"got {type(accumulation).__name__}"
        )

    n = int(hits.ray_count)
    if int(accumulation.hit_mask.shape[0]) != n:
        raise MetricsError(
            f"accumulation.hit_mask length ({int(accumulation.hit_mask.shape[0])}) "
            f"does not match hits.ray_count ({n})"
        )
    if not np.array_equal(accumulation.hit_mask, hits.hit_mask):
        raise MetricsError(
            "accumulation.hit_mask does not match hits.hit_mask"
        )

    has_threshold = threshold is not None
    has_top_percent = top_percent is not None
    if has_threshold and has_top_percent:
        raise MetricsError(
            "provide exactly one of threshold or top_percent; got both"
        )
    if not has_threshold and not has_top_percent:
        raise MetricsError(
            "provide exactly one of threshold or top_percent; got neither"
        )

    if has_threshold and float(threshold) <= 0.0:
        raise MetricsError(
            f"threshold must be positive; got {threshold}"
        )
    if has_top_percent and not (0.0 < float(top_percent) <= 100.0):
        raise MetricsError(
            f"top_percent must be in (0, 100]; got {top_percent}"
        )


def _pixel_mask_from_threshold(
    map_value: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return map_value > float(threshold)


def _pixel_mask_from_top_percent(
    map_value: np.ndarray,
    top_percent: float,
) -> np.ndarray:
    pixel_count = int(map_value.size)
    k = max(1, int(np.ceil(pixel_count * float(top_percent) / 100.0)))

    if float(map_value.max()) <= 0.0:
        return np.zeros_like(map_value, dtype=bool)

    flat = map_value.ravel()
    if k >= pixel_count:
        cutoff = float(flat.min())
    else:
        partitioned = np.partition(flat, -k)
        cutoff = float(partitioned[-k])

    return (map_value >= cutoff) & (map_value > 0.0)


def select_hotspot_rays(
    hits: DetectorHitResult,
    accumulation: DetectorAccumulationResult,
    *,
    threshold: float | None = None,
    top_percent: float | None = None,
    use_weight_map: bool = True,
) -> HotspotSelection:
    """Select rays that landed in hot detector pixels.

    Exactly one of ``threshold`` or ``top_percent`` must be provided.

    threshold mode
    --------------
    A pixel is hot when ``map_value > threshold`` (strict
    greater-than, matching :func:`compute_optical_metrics`'s
    ``ahot`` convention). ``threshold`` must be strictly positive.
    To pick pixels with at least one hit on the integer
    ``count_map``, pass ``use_weight_map=False`` and
    ``threshold=0.5``.

    top_percent mode
    ----------------
    Pick the ``k = max(1, ceil(pixel_count * top_percent / 100))``
    largest pixel values; ``top_percent`` must lie in ``(0, 100]``.
    Zero-valued pixels are never selected unless the entire map is
    zero, in which case nothing is selected. The cutoff uses
    ``>=``, so under cutoff ties more than ``k`` pixels may be
    selected; this is intentional for the foundation and can be
    tightened in a follow-up task.

    Selection
    ---------
    A ray is selected iff it is a detector hit (``hits.hit_mask``
    is True) and its ``hit_pixel_indices`` row falls inside the
    selected ``pixel_mask``. Miss rays are never selected.

    Output
    ------
    Returns a :class:`HotspotSelection` whose ``selected_mask`` is
    dense and aligned with ``hits.ray_count`` and whose
    ``selected_ray_indices`` equals ``np.flatnonzero(selected_mask)``.
    ``selected_weights`` is an all-ones placeholder; see the module
    docstring for why per-ray weights are not yet recoverable.

    Raises
    ------
    MetricsError
        On invalid threshold / top_percent, on hit-mask length or
        equality mismatch between ``hits`` and ``accumulation``, or
        when neither / both selection modes are provided.
    """
    _validate_inputs(hits, accumulation, threshold, top_percent)

    if use_weight_map:
        map_value = np.asarray(accumulation.weight_map, dtype=float)
    else:
        map_value = np.asarray(accumulation.count_map, dtype=float)

    if threshold is not None:
        pixel_mask = _pixel_mask_from_threshold(map_value, float(threshold))
        selection_mode = "threshold"
    else:
        pixel_mask = _pixel_mask_from_top_percent(
            map_value, float(top_percent)
        )
        selection_mode = "top_percent"

    n = int(hits.ray_count)
    selected_mask = np.zeros(n, dtype=bool)

    if n > 0 and bool(hits.hit_mask.any()) and bool(pixel_mask.any()):
        hit_idx = np.flatnonzero(hits.hit_mask)
        rows = accumulation.hit_pixel_indices[hit_idx, 0]
        cols = accumulation.hit_pixel_indices[hit_idx, 1]
        picked = pixel_mask[rows, cols]
        selected_mask[hit_idx[picked]] = True

    selected_ray_indices = np.flatnonzero(selected_mask).astype(np.int64)
    selected_pixel_indices = accumulation.hit_pixel_indices[
        selected_ray_indices
    ].astype(np.int64, copy=True)
    selected_weights = np.ones(selected_ray_indices.size, dtype=float)

    return HotspotSelection(
        selected_mask=selected_mask,
        selected_ray_indices=selected_ray_indices,
        selected_pixel_indices=selected_pixel_indices,
        selected_weights=selected_weights,
        pixel_mask=pixel_mask,
        ray_count=n,
        selected_count=int(selected_ray_indices.size),
        selection_mode=selection_mode,
    )
