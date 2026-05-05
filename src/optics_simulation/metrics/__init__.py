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

__all__ = [
    "HotspotSelection",
    "MetricsError",
    "OpticalMetrics",
    "compute_optical_metrics",
    "select_hotspot_rays",
]
