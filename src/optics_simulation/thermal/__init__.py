"""Thermal package: simplified target-heating surrogate.

Simplified target-heating surrogate; **not an ignition
validation model**. The thermal model consumes an optical flux
surrogate; the input is **not calibrated W/m^2** unless the
caller provides calibrated incident irradiance. This is **not
CFD**, **not pyrolysis chemistry**, and does **not** prove fire
prevention or PET-bottle safety.
"""
from optics_simulation.thermal.lumped_target import (
    LumpedTargetHeatingMapResult,
    LumpedTargetHeatingResult,
    ThermalError,
    simulate_lumped_target_heating,
    simulate_lumped_target_heating_map,
)
from optics_simulation.thermal.risk_comparison import (
    AngleDistanceThermalRiskComparisonResult,
    PerAngleDistanceThermalRiskComparison,
    ThermalRiskGuardrailConfig,
    ThermalRiskMetricComparison,
    compare_angle_distance_thermal_risk_scans,
    compare_thermal_risk_metrics,
)
from optics_simulation.thermal.risk_metrics import (
    ThermalRiskMetrics,
    compute_thermal_risk_metrics,
)
from optics_simulation.thermal.scan_coupling import (
    AngleDistanceHeatingMapScanResult,
    AngleDistanceHeatingScanResult,
    AngleDistanceThermalRiskScanResult,
    PerAngleDistanceHeatingMapResult,
    PerAngleDistanceHeatingResult,
    PerAngleDistanceThermalRiskResult,
    compute_thermal_risk_metrics_over_angle_distance_scan,
    run_lumped_heating_maps_over_angle_distance_scan,
    run_lumped_heating_over_angle_distance_scan,
)

__all__ = [
    "AngleDistanceHeatingMapScanResult",
    "AngleDistanceHeatingScanResult",
    "AngleDistanceThermalRiskComparisonResult",
    "AngleDistanceThermalRiskScanResult",
    "LumpedTargetHeatingMapResult",
    "LumpedTargetHeatingResult",
    "PerAngleDistanceHeatingMapResult",
    "PerAngleDistanceHeatingResult",
    "PerAngleDistanceThermalRiskComparison",
    "PerAngleDistanceThermalRiskResult",
    "ThermalError",
    "ThermalRiskGuardrailConfig",
    "ThermalRiskMetricComparison",
    "ThermalRiskMetrics",
    "compare_angle_distance_thermal_risk_scans",
    "compare_thermal_risk_metrics",
    "compute_thermal_risk_metrics",
    "compute_thermal_risk_metrics_over_angle_distance_scan",
    "run_lumped_heating_maps_over_angle_distance_scan",
    "run_lumped_heating_over_angle_distance_scan",
    "simulate_lumped_target_heating",
    "simulate_lumped_target_heating_map",
]
