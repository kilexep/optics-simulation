"""Angle scan package: baseline incident-angle sweep over the synthetic-mesh optical pipeline."""
from optics_simulation.angle_scan.baseline import (
    AngleScanError,
    AngleScanResult,
    PerAngleResult,
    direction_from_incident_angle,
    run_baseline_angle_scan,
)
from optics_simulation.angle_scan.distance_sweep import (
    AngleDistanceScanResult,
    PerAngleDistanceResult,
    run_angle_distance_sweep,
)

__all__ = [
    "AngleDistanceScanResult",
    "AngleScanError",
    "AngleScanResult",
    "PerAngleDistanceResult",
    "PerAngleResult",
    "direction_from_incident_angle",
    "run_angle_distance_sweep",
    "run_baseline_angle_scan",
]
