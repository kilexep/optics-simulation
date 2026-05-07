"""Synthetic guardrail-sensitivity diagnostic foundation.

Synthetic guardrail-sensitivity and pattern-family screening
smoke check; **not a physical PET-bottle validation**. Re-runs
:func:`compare_thermal_risk_metrics` (and its scan-level
sibling) against a sequence of named
:class:`GuardrailScenario` configurations so callers can see
how the **same** baseline-vs-candidate pair scores under strict
vs numerical-tolerance vs measurement-tolerance vs relaxed
guardrail policies.

Why this exists
---------------
The strict default :class:`ThermalRiskGuardrailConfig` flags any
positive delta as a violation. That is correct as the
conservative research default, but a candidate with a
sub-millikelvin positive delta can be a numerical-tolerance
artefact rather than a physical worsening. This module exposes
the sensitivity directly: every candidate is reported under
multiple named guardrail policies, with the diagnostic bucket
counts kept independent across scenarios.

Research framing (also enforced in demos)
-----------------------------------------
- Guardrail sensitivity is a diagnostic, **not** safety
  certification.
- Relaxed guardrails do **not** prove safety.
- A pass under one guardrail does **not** mean the pattern is
  safe.
- Threshold values are illustrative unless calibrated by
  experiment.
- Candidate screening is **not** optimization.
- A candidate that reduces threshold footprint while increasing
  peak temperature must remain ``"mixed"``, not ``"improved"``.
- No fire-prevention or PET-bottle-safety claim is allowed.

Out of scope
------------
Numerical optimization, manufacturing-readiness scoring, ignition
validation, calibrated W/m^2 anchoring, fire-prevention proof,
and any safety certification are intentionally not implemented
here. This module does **not** modify
:class:`ThermalRiskGuardrailConfig`,
:func:`compare_thermal_risk_metrics`, or
:func:`compare_angle_distance_thermal_risk_scans`; it only
sequences calls to them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from optics_simulation.thermal.lumped_target import ThermalError
from optics_simulation.thermal.risk_comparison import (
    AngleDistanceThermalRiskComparisonResult,
    ThermalRiskGuardrailConfig,
    ThermalRiskMetricComparison,
    compare_angle_distance_thermal_risk_scans,
    compare_thermal_risk_metrics,
)
from optics_simulation.thermal.risk_metrics import ThermalRiskMetrics
from optics_simulation.thermal.scan_coupling import (
    AngleDistanceThermalRiskScanResult,
)


@dataclass(frozen=True)
class GuardrailScenario:
    name: str
    guardrails: ThermalRiskGuardrailConfig
    description: str


@dataclass(frozen=True)
class GuardrailSensitivityEntry:
    scenario_name: str
    comparison: ThermalRiskMetricComparison
    passes_guardrails: bool
    tradeoff_label: str
    guardrail_violations: tuple[str, ...]
    delta_max_temperature_k: float
    delta_top_percent_max_temperature_rise_k: float
    delta_threshold_exceeded_count: int
    delta_threshold_exceeded_area: float | None


@dataclass(frozen=True)
class GuardrailSensitivityResult:
    entries: tuple[GuardrailSensitivityEntry, ...]
    scenario_count: int
    pass_count: int
    fail_count: int
    mixed_count: int
    improved_count: int
    worsened_count: int
    unchanged_count: int
    metric_type: str = "guardrail_sensitivity_diagnostic"


def _validate_scenarios(
    scenarios: Sequence[GuardrailScenario],
) -> tuple[GuardrailScenario, ...]:
    if not isinstance(scenarios, (list, tuple)):
        raise ThermalError(
            f"scenarios must be a list or tuple of GuardrailScenario; "
            f"got {type(scenarios).__name__}"
        )
    out: list[GuardrailScenario] = []
    for i, s in enumerate(scenarios):
        if not isinstance(s, GuardrailScenario):
            raise ThermalError(
                f"scenarios[{i}] must be a GuardrailScenario; "
                f"got {type(s).__name__}"
            )
        if not isinstance(s.guardrails, ThermalRiskGuardrailConfig):
            raise ThermalError(
                f"scenarios[{i}].guardrails must be a "
                f"ThermalRiskGuardrailConfig; "
                f"got {type(s.guardrails).__name__}"
            )
        out.append(s)
    return tuple(out)


def _empty_sensitivity_result() -> GuardrailSensitivityResult:
    return GuardrailSensitivityResult(
        entries=(),
        scenario_count=0,
        pass_count=0,
        fail_count=0,
        mixed_count=0,
        improved_count=0,
        worsened_count=0,
        unchanged_count=0,
    )


def evaluate_guardrail_sensitivity(
    baseline: ThermalRiskMetrics,
    candidate: ThermalRiskMetrics,
    scenarios: Sequence[GuardrailScenario],
) -> GuardrailSensitivityResult:
    """Re-run thermal-risk comparison under multiple guardrail policies.

    Synthetic guardrail-sensitivity and pattern-family screening
    smoke check; **not a physical PET-bottle validation**.
    Sequences :func:`compare_thermal_risk_metrics` calls — one per
    :class:`GuardrailScenario` — and aggregates per-scenario
    pass/fail and tradeoff-label counts. The per-metric deltas
    are identical across scenarios (they depend only on
    ``baseline`` and ``candidate``), but ``passes_guardrails``,
    ``tradeoff_label``, and ``guardrail_violations`` may differ
    because each scenario applies its own
    :class:`ThermalRiskGuardrailConfig`.

    Empty ``scenarios`` returns an empty result with all counts
    zero.

    Parameters
    ----------
    baseline, candidate
        :class:`ThermalRiskMetrics` produced by
        :func:`compute_thermal_risk_metrics`.
    scenarios
        Ordered sequence of :class:`GuardrailScenario`. Order is
        preserved in :attr:`GuardrailSensitivityResult.entries`.

    Raises
    ------
    ThermalError
        On invalid ``baseline`` / ``candidate`` types, an invalid
        ``scenarios`` container, an invalid scenario type, or any
        downstream :class:`ThermalError` from
        :func:`compare_thermal_risk_metrics`.
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
    scenario_tuple = _validate_scenarios(scenarios)
    if not scenario_tuple:
        return _empty_sensitivity_result()

    entries: list[GuardrailSensitivityEntry] = []
    pass_count = 0
    fail_count = 0
    mixed_count = 0
    improved_count = 0
    worsened_count = 0
    unchanged_count = 0

    for scenario in scenario_tuple:
        comparison = compare_thermal_risk_metrics(
            baseline, candidate, guardrails=scenario.guardrails,
        )
        entries.append(
            GuardrailSensitivityEntry(
                scenario_name=str(scenario.name),
                comparison=comparison,
                passes_guardrails=bool(comparison.passes_guardrails),
                tradeoff_label=str(comparison.tradeoff_label),
                guardrail_violations=tuple(
                    comparison.guardrail_violations
                ),
                delta_max_temperature_k=float(
                    comparison.delta_max_temperature_k
                ),
                delta_top_percent_max_temperature_rise_k=float(
                    comparison.delta_top_percent_max_temperature_rise_k
                ),
                delta_threshold_exceeded_count=int(
                    comparison.delta_threshold_exceeded_count
                ),
                delta_threshold_exceeded_area=(
                    None
                    if comparison.delta_threshold_exceeded_area is None
                    else float(
                        comparison.delta_threshold_exceeded_area
                    )
                ),
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

    return GuardrailSensitivityResult(
        entries=tuple(entries),
        scenario_count=len(scenario_tuple),
        pass_count=int(pass_count),
        fail_count=int(fail_count),
        mixed_count=int(mixed_count),
        improved_count=int(improved_count),
        worsened_count=int(worsened_count),
        unchanged_count=int(unchanged_count),
    )


def evaluate_scan_guardrail_sensitivity(
    baseline_scan: AngleDistanceThermalRiskScanResult,
    candidate_scan: AngleDistanceThermalRiskScanResult,
    scenarios: Sequence[GuardrailScenario],
) -> tuple[
    tuple[str, AngleDistanceThermalRiskComparisonResult], ...
]:
    """Run scan-level thermal-risk comparison under multiple guardrails.

    Synthetic guardrail-sensitivity and pattern-family screening
    smoke check; **not a physical PET-bottle validation**.
    Sequences :func:`compare_angle_distance_thermal_risk_scans`
    calls — one per :class:`GuardrailScenario` — and returns a
    tuple of ``(scenario_name, comparison_result)`` pairs in the
    same order as ``scenarios``. The per-entry deltas are
    identical across scenarios; only the per-entry guardrail
    pass/fail and tradeoff label may change.

    Empty ``scenarios`` returns ``()``.

    Parameters
    ----------
    baseline_scan, candidate_scan
        :class:`AngleDistanceThermalRiskScanResult` from
        :func:`compute_thermal_risk_metrics_over_angle_distance_scan`.
        Must have matching length and per-entry
        ``(angle_degrees, detector_z)`` ordering — same precondition
        as :func:`compare_angle_distance_thermal_risk_scans`.
    scenarios
        Ordered sequence of :class:`GuardrailScenario`.

    Raises
    ------
    ThermalError
        On invalid scan types, an invalid ``scenarios`` container,
        an invalid scenario type, or any downstream
        :class:`ThermalError` from
        :func:`compare_angle_distance_thermal_risk_scans`.
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
    scenario_tuple = _validate_scenarios(scenarios)
    if not scenario_tuple:
        return ()

    pairs: list[
        tuple[str, AngleDistanceThermalRiskComparisonResult]
    ] = []
    for scenario in scenario_tuple:
        comparison = compare_angle_distance_thermal_risk_scans(
            baseline_scan, candidate_scan,
            guardrails=scenario.guardrails,
        )
        pairs.append((str(scenario.name), comparison))
    return tuple(pairs)
