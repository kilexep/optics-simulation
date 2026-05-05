"""Angle scan package: baseline incident-angle sweep over the synthetic-mesh optical pipeline."""
from optics_simulation.angle_scan.baseline import (
    AngleScanError,
    AngleScanResult,
    PerAngleResult,
    direction_from_incident_angle,
    run_baseline_angle_scan,
)

__all__ = [
    "AngleScanError",
    "AngleScanResult",
    "PerAngleResult",
    "direction_from_incident_angle",
    "run_baseline_angle_scan",
]
