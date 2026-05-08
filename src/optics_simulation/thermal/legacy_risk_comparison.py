"""Legacy thermal-risk scan comparison for actual STL screening.

Actual STL pattern-family thermal-risk screening smoke check;
**not a physical PET-bottle validation**. Sibling of
:func:`compare_angle_distance_thermal_risk_scans` for the
:class:`LegacyThermalRiskScanResult` shape produced by the actual
STL legacy PET-water optical-to-thermal-risk pipeline. Walks
``baseline.entries`` and ``candidate.entries`` in lock-step, calls
:func:`compare_thermal_risk_metrics` once per entry, and
aggregates pass / fail / tradeoff counts together with the best
and worst deltas across the scan.

Research framing
----------------
- Guardrails are diagnostic filters, **not** safety
  certification.
- A candidate can improve one metric and worsen another; the
  ``tradeoff_label`` exposes that explicitly.
- Delta values are comparison diagnostics only; a negative delta
  is **not required**.
- Do **not** call ``passes_guardrails`` "safe".
- Do **not** claim fire prevention or PET-bottle safety.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.thermal.legacy_scan_coupling import (
    LegacyThermalRiskScanResult,
)
from optics_simulation.thermal.lumped_target import ThermalError
from optics_simulation.thermal.risk_comparison import (
    ThermalRiskGuardrailConfig,
    ThermalRiskMetricComparison,
    compare_thermal_risk_metrics,
)


@dataclass(frozen=True)
class LegacyThermalRiskComparisonEntry:
    angle_degrees: float
    detector_distance: float
    comparison: ThermalRiskMetricComparison


@dataclass(frozen=True)
class LegacyThermalRiskComparisonResult:
    entries: tuple[LegacyThermalRiskComparisonEntry, ...]
    entry_count: int
    pass_count: int
    fail_count: int
    mixed_count: int
    improved_count: int
    worsened_count: int
    unchanged_count: int
    worst_delta_max_temperature_k: float | None
    worst_delta_max_temperature_angle: float | None
    worst_delta_max_temperature_distance: float | None
    best_delta_max_temperature_k: float | None
    best_delta_max_temperature_angle: float | None
    best_delta_max_temperature_distance: float | None
    best_delta_threshold_count: int | None
    best_delta_threshold_count_angle: float | None
    best_delta_threshold_count_distance: float | None
    metric_type: str = "legacy_thermal_risk_comparison"


def _empty_legacy_comparison() -> LegacyThermalRiskComparisonResult:
    return LegacyThermalRiskComparisonResult(
        entries=(),
        entry_count=0,
        pass_count=0,
        fail_count=0,
        mixed_count=0,
        improved_count=0,
        worsened_count=0,
        unchanged_count=0,
        worst_delta_max_temperature_k=None,
        worst_delta_max_temperature_angle=None,
        worst_delta_max_temperature_distance=None,
        best_delta_max_temperature_k=None,
        best_delta_max_temperature_angle=None,
        best_delta_max_temperature_distance=None,
        best_delta_threshold_count=None,
        best_delta_threshold_count_angle=None,
        best_delta_threshold_count_distance=None,
    )


def compare_legacy_thermal_risk_scans(
    baseline: LegacyThermalRiskScanResult,
    candidate: LegacyThermalRiskScanResult,
    *,
    guardrails: ThermalRiskGuardrailConfig | None = None,
) -> LegacyThermalRiskComparisonResult:
    """Compare two :class:`LegacyThermalRiskScanResult` entry-by-entry.

    Actual STL pattern-family thermal-risk screening smoke check;
    **not a physical PET-bottle validation**. ``baseline`` and
    ``candidate`` must already be in identical
    ``(angle_degrees, detector_distance)`` order with matching
    ``entry_count``. Per-entry comparison is delegated to
    :func:`compare_thermal_risk_metrics`; aggregate fields use
    first-occurrence tie breaks (``np.argmax`` / ``np.argmin``).

    Empty scans return an empty result with all counts zero and
    aggregate fields ``None``.

    Raises
    ------
    ThermalError
        On invalid scan types, mismatched ``entry_count``, or
        per-entry ``(angle_degrees, detector_distance)`` ordering
        that does not match.
    """
    if not isinstance(baseline, LegacyThermalRiskScanResult):
        raise ThermalError(
            "baseline must be a LegacyThermalRiskScanResult; got "
            f"{type(baseline).__name__}"
        )
    if not isinstance(candidate, LegacyThermalRiskScanResult):
        raise ThermalError(
            "candidate must be a LegacyThermalRiskScanResult; got "
            f"{type(candidate).__name__}"
        )

    if not baseline.entries and not candidate.entries:
        return _empty_legacy_comparison()

    if int(baseline.entry_count) != int(candidate.entry_count):
        raise ThermalError(
            f"baseline.entry_count ({int(baseline.entry_count)}) "
            f"must equal candidate.entry_count "
            f"({int(candidate.entry_count)})"
        )
    if len(baseline.entries) != len(candidate.entries):
        raise ThermalError(
            f"baseline.entries length ({len(baseline.entries)}) "
            f"must equal candidate.entries length "
            f"({len(candidate.entries)})"
        )

    per_entry: list[LegacyThermalRiskComparisonEntry] = []
    pass_count = 0
    fail_count = 0
    mixed_count = 0
    improved_count = 0
    worsened_count = 0
    unchanged_count = 0

    for i, (b, c) in enumerate(
        zip(baseline.entries, candidate.entries)
    ):
        if (
            float(b.angle_degrees) != float(c.angle_degrees)
            or float(b.detector_distance)
            != float(c.detector_distance)
        ):
            raise ThermalError(
                f"entries[{i}] ordering mismatch: baseline "
                f"(angle={b.angle_degrees}, "
                f"distance={b.detector_distance}) vs candidate "
                f"(angle={c.angle_degrees}, "
                f"distance={c.detector_distance})"
            )

        comparison = compare_thermal_risk_metrics(
            b.thermal_metrics,
            c.thermal_metrics,
            guardrails=guardrails,
        )
        per_entry.append(
            LegacyThermalRiskComparisonEntry(
                angle_degrees=float(b.angle_degrees),
                detector_distance=float(b.detector_distance),
                comparison=comparison,
            )
        )

        if comparison.passes_guardrails:
            pass_count += 1
        else:
            fail_count += 1
        if comparison.tradeoff_label == "improved":
            improved_count += 1
        elif comparison.tradeoff_label == "worsened":
            worsened_count += 1
        elif comparison.tradeoff_label == "unchanged":
            unchanged_count += 1
        else:
            mixed_count += 1

    delta_max_t = np.array(
        [
            float(p.comparison.delta_max_temperature_k)
            for p in per_entry
        ],
        dtype=float,
    )
    delta_count = np.array(
        [
            int(p.comparison.delta_threshold_exceeded_count)
            for p in per_entry
        ],
        dtype=np.int64,
    )

    idx_worst = int(np.argmax(delta_max_t))
    idx_best_max = int(np.argmin(delta_max_t))
    idx_best_count = int(np.argmin(delta_count))
    e_worst = per_entry[idx_worst]
    e_best_max = per_entry[idx_best_max]
    e_best_count = per_entry[idx_best_count]

    return LegacyThermalRiskComparisonResult(
        entries=tuple(per_entry),
        entry_count=len(per_entry),
        pass_count=int(pass_count),
        fail_count=int(fail_count),
        mixed_count=int(mixed_count),
        improved_count=int(improved_count),
        worsened_count=int(worsened_count),
        unchanged_count=int(unchanged_count),
        worst_delta_max_temperature_k=float(
            e_worst.comparison.delta_max_temperature_k
        ),
        worst_delta_max_temperature_angle=float(
            e_worst.angle_degrees
        ),
        worst_delta_max_temperature_distance=float(
            e_worst.detector_distance
        ),
        best_delta_max_temperature_k=float(
            e_best_max.comparison.delta_max_temperature_k
        ),
        best_delta_max_temperature_angle=float(
            e_best_max.angle_degrees
        ),
        best_delta_max_temperature_distance=float(
            e_best_max.detector_distance
        ),
        best_delta_threshold_count=int(
            e_best_count.comparison.delta_threshold_exceeded_count
        ),
        best_delta_threshold_count_angle=float(
            e_best_count.angle_degrees
        ),
        best_delta_threshold_count_distance=float(
            e_best_count.detector_distance
        ),
    )
