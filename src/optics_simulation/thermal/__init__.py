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

__all__ = [
    "LumpedTargetHeatingResult",
    "ThermalError",
    "simulate_lumped_target_heating",
]
