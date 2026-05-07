"""Thermal package: simplified target-heating surrogate.

Simplified target-heating surrogate; **not an ignition
validation model**. The thermal model consumes an optical flux
surrogate; the input is **not calibrated W/m^2** unless the
caller provides calibrated incident irradiance. This is **not
CFD**, **not pyrolysis chemistry**, and does **not** prove fire
prevention or PET-bottle safety.
"""
from optics_simulation.thermal.lumped_target import (
    LumpedTargetHeatingResult,
    ThermalError,
    simulate_lumped_target_heating,
)
from optics_simulation.thermal.scan_coupling import (
    AngleDistanceHeatingScanResult,
    PerAngleDistanceHeatingResult,
    run_lumped_heating_over_angle_distance_scan,
)

__all__ = [
    "AngleDistanceHeatingScanResult",
    "LumpedTargetHeatingResult",
    "PerAngleDistanceHeatingResult",
    "ThermalError",
    "run_lumped_heating_over_angle_distance_scan",
    "simulate_lumped_target_heating",
]
