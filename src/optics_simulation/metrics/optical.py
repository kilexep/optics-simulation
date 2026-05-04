"""Optical caustic-risk metrics on a 2D detector map.

Pure-numpy module. Takes a 2D non-negative array (typically a
:class:`DetectorAccumulationResult.count_map` or ``weight_map``) and
computes the basic caustic-risk indicators used by CLAUDE.md:
``peak_value`` / ``total_value`` / ``mean_value`` / ``Cmax`` /
``C99`` / ``Eexceed[t]`` / ``Ahot[t]``.

Out of scope
------------
This module operates on **relative irradiance surrogate** values
only. Pixel-area normalization (W -> W/m^2), spectral integration,
contribution-map back-tracking, BTDF entropy, angle scan, pattern
generation, and visualization are intentionally not implemented
here.

Note
----
``MetricsError`` is defined locally for now. If the metrics package
grows to several modules with shared error semantics, factor it
out to a ``metrics/errors.py`` in a follow-up task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class MetricsError(Exception):
    """Raised when metric inputs are invalid (shape, NaN/inf, negative,
    non-positive incident_reference, out-of-range top_percent, or
    non-positive thresholds)."""


@dataclass(frozen=True)
class OpticalMetrics:
    """Container for caustic-risk metrics computed from a 2D map.

    Fields
    ------
    peak_value
        ``arr.max()`` — maximum pixel value.
    total_value
        ``arr.sum()`` — sum across all pixels (irradiance integral
        surrogate; pixel area is **not** applied in this task).
    mean_value
        ``arr.mean()``.
    cmax
        ``peak_value / incident_reference``. Secondary metric per
        CLAUDE.md (less robust than ``c99``).
    c99
        Mean of the top ``top_percent`` percent of pixels divided by
        ``incident_reference``. With the default ``top_percent=1.0``
        this is the project's primary **C99** caustic metric. The
        top-pixel count is ``max(1, ceil(N * top_percent / 100))``,
        so even a very small top_percent on a small map keeps at
        least one pixel.
    top_percent
        The ``top_percent`` value actually used (echoed back).
    incident_reference
        The ``incident_reference`` value actually used.
    thresholds
        Threshold tuple, **input order preserved**.
    eexceed
        ``{t: sum(max(arr - t * incident_reference, 0))}``.
    ahot
        ``{t: count(arr > t * incident_reference)}`` — strict
        greater-than (a pixel exactly equal to the threshold is
        **not** counted as hot).
    pixel_count
        ``arr.size``.
    """

    peak_value: float
    total_value: float
    mean_value: float
    cmax: float
    c99: float
    top_percent: float
    incident_reference: float
    thresholds: tuple[float, ...]
    eexceed: dict[float, float]
    ahot: dict[float, int]
    pixel_count: int


def compute_optical_metrics(
    map_2d: np.ndarray,
    *,
    incident_reference: float = 1.0,
    thresholds: Sequence[float] = (2.0, 5.0, 10.0),
    top_percent: float = 1.0,
) -> OpticalMetrics:
    """Compute :class:`OpticalMetrics` from a 2D non-negative map.

    See :class:`OpticalMetrics` for the meaning of each returned
    field. Raises :class:`MetricsError` for invalid shape, NaN / inf,
    negative pixel values, ``incident_reference <= 0``, ``top_percent``
    outside ``(0, 100]``, or any non-positive entry in ``thresholds``.
    """
    arr = np.asarray(map_2d, dtype=float)
    if arr.ndim != 2:
        raise MetricsError(
            f"map_2d must be a 2D array; got shape {arr.shape}"
        )
    if arr.size == 0:
        raise MetricsError(f"map_2d is empty; got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise MetricsError("map_2d contains NaN or inf values")
    if (arr < 0).any():
        raise MetricsError("map_2d contains negative values")

    if incident_reference <= 0.0:
        raise MetricsError(
            f"incident_reference must be positive; got {incident_reference}"
        )

    if not (0.0 < top_percent <= 100.0):
        raise MetricsError(
            f"top_percent must be in (0, 100]; got {top_percent}"
        )

    thresholds_tuple = tuple(float(t) for t in thresholds)
    for i, t in enumerate(thresholds_tuple):
        if t <= 0.0:
            raise MetricsError(
                f"thresholds[{i}] must be positive; got {t}"
            )

    pixel_count = int(arr.size)
    peak = float(arr.max())
    total = float(arr.sum())
    mean = float(arr.mean())
    cmax = peak / float(incident_reference)

    k = max(1, int(np.ceil(pixel_count * top_percent / 100.0)))
    if k >= pixel_count:
        top_mean = mean
    else:
        flat = arr.ravel()
        partitioned = np.partition(flat, -k)
        top_mean = float(partitioned[-k:].mean())
    c99 = top_mean / float(incident_reference)

    eexceed: dict[float, float] = {}
    ahot: dict[float, int] = {}
    for t in thresholds_tuple:
        cutoff = t * float(incident_reference)
        excess = arr - cutoff
        eexceed[t] = float(np.maximum(excess, 0.0).sum())
        ahot[t] = int((arr > cutoff).sum())

    return OpticalMetrics(
        peak_value=peak,
        total_value=total,
        mean_value=mean,
        cmax=cmax,
        c99=c99,
        top_percent=float(top_percent),
        incident_reference=float(incident_reference),
        thresholds=thresholds_tuple,
        eexceed=eexceed,
        ahot=ahot,
        pixel_count=pixel_count,
    )
