"""Metrics package: optical caustic-risk metrics on detector maps."""
from optics_simulation.metrics.hotspot_selection import (
    HotspotSelection,
    select_hotspot_rays,
)
from optics_simulation.metrics.optical import (
    MetricsError,
    OpticalMetrics,
    compute_optical_metrics,
)
from optics_simulation.metrics.irradiance import (
    DetectorIrradianceSurrogate,
    compute_relative_irradiance_surrogate,
)

__all__ = [
    "DetectorIrradianceSurrogate",
    "HotspotSelection",
    "MetricsError",
    "OpticalMetrics",
    "compute_optical_metrics",
    "compute_relative_irradiance_surrogate",
    "select_hotspot_rays",
]
