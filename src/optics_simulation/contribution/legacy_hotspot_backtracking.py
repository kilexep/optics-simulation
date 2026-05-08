"""Detector hotspot backtracking to first shell-surface contribution map.

Actual STL hotspot-backtracked risk-guided pattern smoke check;
**not a physical PET-bottle validation**. Maps a
:class:`LegacyDetailedTraceResult`'s detector hotspot rays back to
their corresponding initial-ray indices via the trace's
``final_source_ray_indices`` lineage, then bins those initial
rays' first-shell-hit ``(u, v)`` coordinates into a
:class:`ContributionMap` and post-processes the result into a
:class:`RiskMap`. The resulting risk map can drive a risk-guided
Gaussian dimple pattern that targets the surface regions
responsible for the baseline caustic peak.

Limitations
-----------
- This is a **diagnostic** map, not a measured physical risk
  field.
- Detector pixel weighting comes from
  :class:`HotspotSelection.selected_weights`, which is currently
  an all-ones placeholder; a future ray-history layer can replace
  it with true Fresnel-weighted hotspot weights.
- The current trace runner does not split rays, so
  ``final_source_ray_indices`` is one-to-one and duplicates are
  not expected. If they appear, contribution-map binning will
  count the duplicated initial-ray hits multiple times — that is
  the documented behavior; this module does **not** silently
  deduplicate.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** manufacturing-ready.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optics_simulation.contribution.map import (
    ContributionError,
    ContributionMap,
    create_contribution_map,
)
from optics_simulation.contribution.risk_map import (
    RiskMap,
    build_risk_map,
)
from optics_simulation.metrics.hotspot_selection import (
    HotspotSelection,
    select_hotspot_rays,
)
from optics_simulation.metrics.optical import MetricsError
from optics_simulation.optics.legacy_detailed_trace import (
    LegacyDetailedTraceResult,
)


@dataclass(frozen=True)
class LegacyHotspotContributionResult:
    hotspot_selection: HotspotSelection
    initial_ray_indices: np.ndarray
    initial_aligned_selection: HotspotSelection
    contribution_map: ContributionMap
    risk_map: RiskMap
    selected_final_ray_count: int
    mapped_initial_ray_count: int
    valid_surface_hit_count: int
    contribution_resolution: tuple[int, int]
    selection_mode: str


def build_legacy_hotspot_contribution_map(
    detailed: LegacyDetailedTraceResult,
    *,
    top_percent: float = 10.0,
    contribution_resolution: tuple[int, int] = (32, 64),
    risk_threshold: float | None = None,
    risk_epsilon: float = 0.01,
) -> LegacyHotspotContributionResult:
    """Backtrack detector hotspots to a shell-surface contribution map.

    Actual STL hotspot-backtracked risk-guided pattern smoke
    check; **not a physical PET-bottle validation**.

    Steps:

    1. Pick the top-``top_percent`` detector pixels and the rays
       that landed in them via :func:`select_hotspot_rays` on the
       detailed trace's ``detector_hits`` and ``accumulation``.
    2. Map the surviving final-ray indices back to their initial
       indices through ``trace_result.final_source_ray_indices``.
    3. Wrap those initial indices in a fresh
       :class:`HotspotSelection` aligned with the initial ray
       count so :func:`create_contribution_map` accepts it
       alongside ``detailed.first_shell_hit_coordinates``.
    4. Bin the selected initial rays' shell-surface ``(u, v)`` into
       a :class:`ContributionMap` at ``contribution_resolution``.
    5. Run :func:`build_risk_map` over the contribution map with
       the supplied ``risk_threshold`` / ``risk_epsilon``.

    Raises
    ------
    MetricsError, ContributionError
        On invalid inputs or downstream validation failures.
    """
    if not isinstance(detailed, LegacyDetailedTraceResult):
        raise ContributionError(
            "detailed must be a LegacyDetailedTraceResult; got "
            f"{type(detailed).__name__}"
        )

    tp = float(top_percent)
    if not (math.isfinite(tp) and 0.0 < tp <= 100.0):
        raise MetricsError(
            f"top_percent must be a finite float in (0, 100]; "
            f"got {top_percent}"
        )

    selection = select_hotspot_rays(
        detailed.detector_hits,
        detailed.accumulation,
        top_percent=tp,
    )

    final_indices = np.asarray(
        selection.selected_ray_indices, dtype=np.int64,
    )
    initial_lineage = np.asarray(
        detailed.trace_result.final_source_ray_indices, dtype=np.int64,
    )

    if final_indices.size > 0:
        if (
            int(final_indices.max())
            >= int(initial_lineage.shape[0])
        ):
            raise ContributionError(
                "selected final ray indices exceed lineage length; "
                "trace_result.final_source_ray_indices is "
                "inconsistent with detector_hits.ray_count"
            )
        initial_indices = initial_lineage[final_indices].astype(
            np.int64, copy=True,
        )
    else:
        initial_indices = np.empty(0, dtype=np.int64)

    initial_ray_count = int(detailed.rays.ray_count)
    selected_mask = np.zeros(initial_ray_count, dtype=bool)
    if initial_indices.size > 0:
        selected_mask[initial_indices] = True

    selected_pixel_indices = np.asarray(
        selection.selected_pixel_indices, dtype=np.int64,
    ).copy()
    selected_weights = np.asarray(
        selection.selected_weights, dtype=float,
    ).copy()
    pixel_mask = np.asarray(selection.pixel_mask, dtype=bool).copy()

    initial_aligned_selection = HotspotSelection(
        selected_mask=selected_mask,
        selected_ray_indices=initial_indices,
        selected_pixel_indices=selected_pixel_indices,
        selected_weights=selected_weights,
        pixel_mask=pixel_mask,
        ray_count=initial_ray_count,
        selected_count=int(initial_indices.size),
        selection_mode=str(selection.selection_mode),
    )

    contribution_map = create_contribution_map(
        initial_aligned_selection,
        detailed.first_shell_hit_coordinates,
        resolution=contribution_resolution,
    )
    risk_map_obj = build_risk_map(
        contribution_map,
        threshold=risk_threshold,
        epsilon=float(risk_epsilon),
    )

    return LegacyHotspotContributionResult(
        hotspot_selection=selection,
        initial_ray_indices=initial_indices,
        initial_aligned_selection=initial_aligned_selection,
        contribution_map=contribution_map,
        risk_map=risk_map_obj,
        selected_final_ray_count=int(final_indices.size),
        mapped_initial_ray_count=int(initial_indices.size),
        valid_surface_hit_count=int(contribution_map.total_selected),
        contribution_resolution=tuple(contribution_resolution),
        selection_mode=str(selection.selection_mode),
    )
