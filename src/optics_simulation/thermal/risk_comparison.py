"""Synthetic thermal-risk comparison and guardrail smoke foundation.

Synthetic thermal-risk comparison and guardrail smoke check;
**not a physical PET-bottle validation**. Provides reusable
``baseline``-vs-``candidate`` comparison utilities for
:class:`ThermalRiskMetrics` and
:class:`AngleDistanceThermalRiskScanResult`. The primary use is
to expose the situation where a candidate pattern reduces the
threshold-exceeded footprint while increasing peak / top-percent
temperature, so the comparison must be reported as a **mixed
tradeoff** rather than as an improvement.

Why this exists
---------------
Earlier outer-surface patterned-shell smoke runs revealed
distribution-sensitive but conflicting deltas: threshold-exceeded
count down, max temperature up. Reporting only one of those
deltas would be a research-logic error. This module supplies the
canonical reduction so that downstream demos and screening loops
score patterns against multiple metrics under explicit guardrail
limits.

Research framing (also enforced in demos)
-----------------------------------------
- Guardrails are conservative diagnostic filters, **not** safety
  certification.
- Threshold values are illustrative unless calibrated by
  experiment.
- Delta values are comparison diagnostics only.
- A candidate can reduce threshold footprint while increasing
  peak temperature; such a candidate must be reported as a mixed
  tradeoff, not as an improvement.
- Do **not** call ``passes_guardrails`` "safe".
- Do **not** call ``tradeoff_label`` "success".
- Do **not** claim fire prevention or PET-bottle safety.

Tradeoff label policy
---------------------
For the four key metrics
(``delta_max_temperature_k``,
``delta_top_percent_max_temperature_rise_k``,
``delta_threshold_exceeded_count``,
``delta_threshold_exceeded_area`` when available):

- ``"improved"`` when every available key delta is
  ``<= -tolerance`` and no key metric worsens.
- ``"worsened"`` when every available key delta is
  ``>= +tolerance`` and no key metric improves.
- ``"unchanged"`` when every key delta is within
  ``[-tolerance, +tolerance]``.
- ``"mixed"`` otherwise.

Out of scope
------------
Numerical optimization, manufacturing-readiness scoring, ignition
validation, calibrated W/m^2 anchoring, fire-prevention proof,
and any safety certification are intentionally not implemented
here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.thermal.lumped_target import ThermalError
from optics_simulation.thermal.risk_metrics import ThermalRiskMetrics
from optics_simulation.thermal.scan_coupling import (
    AngleDistanceThermalRiskScanResult,
)


@dataclass(frozen=True)
class ThermalRiskGuardrailConfig:
    max_temperature_increase_limit_k: float = 0.0
    top_percent_rise_increase_limit_k: float = 0.0
    threshold_count_increase_limit: int = 0
    threshold_area_increase_limit: float | None = 0.0
    tolerance: float = 1e-12


@dataclass(frozen=True)
class ThermalRiskMetricComparison:
    baseline: ThermalRiskMetrics
    candidate: ThermalRiskMetrics
    delta_max_temperature_k: float
    delta_max_temperature_rise_k: float
    delta_top_percent_max_temperature_rise_k: float
    delta_threshold_exceeded_count: int
    delta_threshold_exceeded_fraction: float
    delta_threshold_exceeded_area: float | None
    delta_mean_max_temperature_rise_k: float
    delta_mean_final_temperature_k: float
    guardrail_violations: tuple[str, ...]
    passes_guardrails: bool
    tradeoff_label: str
    metric_type: str = "thermal_risk_metric_comparison"


@dataclass(frozen=True)
class PerAngleDistanceThermalRiskComparison:
    angle_degrees: float
    detector_z: float
    comparison: ThermalRiskMetricComparison


@dataclass(frozen=True)
class AngleDistanceThermalRiskComparisonResult:
    per_result: tuple[PerAngleDistanceThermalRiskComparison, ...]
    result_count: int
    pass_count: int
    fail_count: int
    mixed_count: int
    improved_count: int
    worsened_count: int
    unchanged_count: int
    worst_delta_max_temperature_k: float | None
    worst_delta_max_temperature_angle: float | None
    worst_delta_max_temperature_detector_z: float | None
    best_delta_threshold_count: int | None
    best_delta_threshold_count_angle: float | None
    best_delta_threshold_count_detector_z: float | None
    metric_type: str = "angle_distance_thermal_risk_comparison"


def _check_finite_nonneg(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalError(
            f"{name} must be a finite float >= 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise ThermalError(f"{name} must be finite; got {v}")
    if v < 0.0:
        raise ThermalError(f"{name} must be >= 0; got {v}")
    return v


def _check_int_nonneg(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ThermalError(
            f"{name} must be a non-negative int; got {value!r}"
        )
    iv = int(value)
    if iv < 0:
        raise ThermalError(f"{name} must be >= 0; got {iv}")
    return iv


def _validate_guardrails(
    guardrails: ThermalRiskGuardrailConfig | None,
) -> ThermalRiskGuardrailConfig:
    if guardrails is None:
        return ThermalRiskGuardrailConfig()
    if not isinstance(guardrails, ThermalRiskGuardrailConfig):
        raise ThermalError(
            f"guardrails must be a ThermalRiskGuardrailConfig or None; "
            f"got {type(guardrails).__name__}"
        )
    _check_finite_nonneg(
        guardrails.max_temperature_increase_limit_k,
        name="max_temperature_increase_limit_k",
    )
    _check_finite_nonneg(
        guardrails.top_percent_rise_increase_limit_k,
        name="top_percent_rise_increase_limit_k",
    )
    _check_int_nonneg(
        guardrails.threshold_count_increase_limit,
        name="threshold_count_increase_limit",
    )
    if guardrails.threshold_area_increase_limit is not None:
        _check_finite_nonneg(
            guardrails.threshold_area_increase_limit,
            name="threshold_area_increase_limit",
        )
    _check_finite_nonneg(guardrails.tolerance, name="tolerance")
    return guardrails


def _classify_tradeoff(
    deltas: list[float], tolerance: float,
) -> str:
    if not deltas:
        return "unchanged"
    tol = float(tolerance)
    improved = [d <= -tol for d in deltas]
    worsened = [d >= tol for d in deltas]
    unchanged = [(-tol <= d <= tol) for d in deltas]
    if all(unchanged):
        return "unchanged"
    if all(improved):
        return "improved"
    if all(worsened):
        return "worsened"
    return "mixed"


def compare_thermal_risk_metrics(
    baseline: ThermalRiskMetrics,
    candidate: ThermalRiskMetrics,
    *,
    guardrails: ThermalRiskGuardrailConfig | None = None,
) -> ThermalRiskMetricComparison:
    """Compare a candidate to a baseline ThermalRiskMetrics value.

    Synthetic thermal-risk comparison and guardrail smoke check;
    **not a physical PET-bottle validation**. Computes per-metric
    deltas (``candidate - baseline``), assigns a coarse
    ``tradeoff_label``, and applies a conservative guardrail
    policy to flag candidates whose key metrics increased beyond
    the configured limits.

    Guardrails are conservative diagnostic filters, **not** safety
    certification. ``passes_guardrails == True`` does **not** mean
    the candidate is safe, manufacturable, or fire-prevention
    validated; it only means none of the configured key metric
    limits were exceeded under the synthetic surrogate setup.

    Parameters
    ----------
    baseline, candidate
        :class:`ThermalRiskMetrics` produced by
        :func:`compute_thermal_risk_metrics`. Must both be
        :class:`ThermalRiskMetrics`.
    guardrails
        Optional :class:`ThermalRiskGuardrailConfig`. ``None``
        installs the default (conservative) policy: any positive
        delta on max temperature, top-percent rise,
        threshold-exceeded count, or threshold-exceeded area is
        treated as a violation. The threshold-area guardrail is
        skipped when either side reports
        ``threshold_exceeded_area is None`` or the configured
        ``threshold_area_increase_limit`` is ``None``.

    Raises
    ------
    ThermalError
        On invalid ``baseline`` / ``candidate`` types, invalid
        ``guardrails`` type, or non-finite / negative guardrail
        limits.
    """
    if not isinstance(baseline, ThermalRiskMetrics):
        raise ThermalError(
            f"baseline must be a ThermalRiskMetrics; "
            f"got {type(baseline).__name__}"
        )
    if not isinstance(candidate, ThermalRiskMetrics):
        raise ThermalError(
            f"candidate must be a ThermalRiskMetrics; "
            f"got {type(candidate).__name__}"
        )
    cfg = _validate_guardrails(guardrails)
    tol = float(cfg.tolerance)

    delta_max_t = float(
        candidate.max_temperature_k - baseline.max_temperature_k
    )
    delta_max_t_rise = float(
        candidate.max_temperature_rise_k
        - baseline.max_temperature_rise_k
    )
    delta_top = float(
        candidate.top_percent_max_temperature_rise_k
        - baseline.top_percent_max_temperature_rise_k
    )
    delta_count = int(
        int(candidate.threshold_exceeded_count)
        - int(baseline.threshold_exceeded_count)
    )
    delta_fraction = float(
        candidate.threshold_exceeded_fraction
        - baseline.threshold_exceeded_fraction
    )
    if (
        baseline.threshold_exceeded_area is None
        or candidate.threshold_exceeded_area is None
    ):
        delta_area: float | None = None
    else:
        delta_area = float(
            float(candidate.threshold_exceeded_area)
            - float(baseline.threshold_exceeded_area)
        )
    delta_mean_max_rise = float(
        candidate.mean_max_temperature_rise_k
        - baseline.mean_max_temperature_rise_k
    )
    delta_mean_final = float(
        candidate.mean_final_temperature_k
        - baseline.mean_final_temperature_k
    )

    violations: list[str] = []
    if delta_max_t > float(cfg.max_temperature_increase_limit_k) + tol:
        violations.append("max_temperature_k")
    if (
        delta_top
        > float(cfg.top_percent_rise_increase_limit_k) + tol
    ):
        violations.append("top_percent_max_temperature_rise_k")
    if (
        delta_count
        > int(cfg.threshold_count_increase_limit)
    ):
        violations.append("threshold_exceeded_count")
    if (
        delta_area is not None
        and cfg.threshold_area_increase_limit is not None
        and delta_area
            > float(cfg.threshold_area_increase_limit) + tol
    ):
        violations.append("threshold_exceeded_area")

    passes = len(violations) == 0

    key_deltas: list[float] = [
        float(delta_max_t),
        float(delta_top),
        float(delta_count),
    ]
    if delta_area is not None:
        key_deltas.append(float(delta_area))
    label = _classify_tradeoff(key_deltas, tolerance=tol)

    return ThermalRiskMetricComparison(
        baseline=baseline,
        candidate=candidate,
        delta_max_temperature_k=delta_max_t,
        delta_max_temperature_rise_k=delta_max_t_rise,
        delta_top_percent_max_temperature_rise_k=delta_top,
        delta_threshold_exceeded_count=delta_count,
        delta_threshold_exceeded_fraction=delta_fraction,
        delta_threshold_exceeded_area=delta_area,
        delta_mean_max_temperature_rise_k=delta_mean_max_rise,
        delta_mean_final_temperature_k=delta_mean_final,
        guardrail_violations=tuple(violations),
        passes_guardrails=bool(passes),
        tradeoff_label=label,
    )


def _empty_scan_comparison() -> AngleDistanceThermalRiskComparisonResult:
    return AngleDistanceThermalRiskComparisonResult(
        per_result=(),
        result_count=0,
        pass_count=0,
        fail_count=0,
        mixed_count=0,
        improved_count=0,
        worsened_count=0,
        unchanged_count=0,
        worst_delta_max_temperature_k=None,
        worst_delta_max_temperature_angle=None,
        worst_delta_max_temperature_detector_z=None,
        best_delta_threshold_count=None,
        best_delta_threshold_count_angle=None,
        best_delta_threshold_count_detector_z=None,
    )


def compare_angle_distance_thermal_risk_scans(
    baseline_scan: AngleDistanceThermalRiskScanResult,
    candidate_scan: AngleDistanceThermalRiskScanResult,
    *,
    guardrails: ThermalRiskGuardrailConfig | None = None,
) -> AngleDistanceThermalRiskComparisonResult:
    """Compare two thermal-risk scan results entry-by-entry.

    Synthetic thermal-risk comparison and guardrail smoke check;
    **not a physical PET-bottle validation**. Walks every
    :class:`PerAngleDistanceThermalRiskResult` in
    ``baseline_scan`` and ``candidate_scan`` in lock-step (the two
    scans must already be in identical
    ``(angle_degrees, detector_z)`` order with matching
    ``result_count``), calls
    :func:`compare_thermal_risk_metrics` once per entry, and
    aggregates the per-entry tradeoff labels and guardrail
    pass/fail counts.

    Aggregates
    ----------
    - ``worst_delta_max_temperature_k`` is ``max(delta_max_temperature_k)``
      across per_result entries (largest positive increase wins on
      ties by first occurrence).
    - ``best_delta_threshold_count`` is ``min(delta_threshold_exceeded_count)``
      across per_result entries (most negative wins on ties by first
      occurrence).

    Empty scans return an empty result with all counts zero and
    aggregate fields ``None``.

    Parameters
    ----------
    baseline_scan, candidate_scan
        :class:`AngleDistanceThermalRiskScanResult` from
        :func:`compute_thermal_risk_metrics_over_angle_distance_scan`.
    guardrails
        Forwarded to every per-entry
        :func:`compare_thermal_risk_metrics` call.

    Raises
    ------
    ThermalError
        On invalid scan types, mismatched ``result_count``,
        mismatched per-entry ``(angle_degrees, detector_z)``
        ordering, or any downstream guardrail validation error.
    """
    if not isinstance(baseline_scan, AngleDistanceThermalRiskScanResult):
        raise ThermalError(
            f"baseline_scan must be an "
            f"AngleDistanceThermalRiskScanResult; "
            f"got {type(baseline_scan).__name__}"
        )
    if not isinstance(candidate_scan, AngleDistanceThermalRiskScanResult):
        raise ThermalError(
            f"candidate_scan must be an "
            f"AngleDistanceThermalRiskScanResult; "
            f"got {type(candidate_scan).__name__}"
        )
    cfg = _validate_guardrails(guardrails)

    if not baseline_scan.per_result and not candidate_scan.per_result:
        return _empty_scan_comparison()

    if baseline_scan.result_count != candidate_scan.result_count:
        raise ThermalError(
            f"baseline_scan.result_count "
            f"({baseline_scan.result_count}) must equal "
            f"candidate_scan.result_count "
            f"({candidate_scan.result_count})"
        )
    if len(baseline_scan.per_result) != len(candidate_scan.per_result):
        raise ThermalError(
            f"baseline_scan.per_result length "
            f"({len(baseline_scan.per_result)}) must equal "
            f"candidate_scan.per_result length "
            f"({len(candidate_scan.per_result)})"
        )

    per_results: list[PerAngleDistanceThermalRiskComparison] = []
    pass_count = 0
    fail_count = 0
    mixed_count = 0
    improved_count = 0
    worsened_count = 0
    unchanged_count = 0

    for i, (b, c) in enumerate(
        zip(baseline_scan.per_result, candidate_scan.per_result)
    ):
        if (
            float(b.angle_degrees) != float(c.angle_degrees)
            or float(b.detector_z) != float(c.detector_z)
        ):
            raise ThermalError(
                f"per_result[{i}] ordering mismatch: baseline "
                f"(angle={b.angle_degrees}, z={b.detector_z}) "
                f"vs candidate "
                f"(angle={c.angle_degrees}, z={c.detector_z})"
            )
        comparison = compare_thermal_risk_metrics(
            b.thermal_metrics,
            c.thermal_metrics,
            guardrails=cfg,
        )
        per_results.append(
            PerAngleDistanceThermalRiskComparison(
                angle_degrees=float(b.angle_degrees),
                detector_z=float(b.detector_z),
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

    delta_max_temps = np.array(
        [
            float(p.comparison.delta_max_temperature_k)
            for p in per_results
        ],
        dtype=float,
    )
    idx_worst = int(np.argmax(delta_max_temps))
    chosen_worst = per_results[idx_worst]

    delta_counts = np.array(
        [
            int(p.comparison.delta_threshold_exceeded_count)
            for p in per_results
        ],
        dtype=np.int64,
    )
    idx_best = int(np.argmin(delta_counts))
    chosen_best = per_results[idx_best]

    return AngleDistanceThermalRiskComparisonResult(
        per_result=tuple(per_results),
        result_count=len(per_results),
        pass_count=int(pass_count),
        fail_count=int(fail_count),
        mixed_count=int(mixed_count),
        improved_count=int(improved_count),
        worsened_count=int(worsened_count),
        unchanged_count=int(unchanged_count),
        worst_delta_max_temperature_k=float(
            chosen_worst.comparison.delta_max_temperature_k
        ),
        worst_delta_max_temperature_angle=float(
            chosen_worst.angle_degrees
        ),
        worst_delta_max_temperature_detector_z=float(
            chosen_worst.detector_z
        ),
        best_delta_threshold_count=int(
            chosen_best.comparison.delta_threshold_exceeded_count
        ),
        best_delta_threshold_count_angle=float(
            chosen_best.angle_degrees
        ),
        best_delta_threshold_count_detector_z=float(
            chosen_best.detector_z
        ),
    )
