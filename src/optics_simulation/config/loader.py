"""YAML config loader and structured AppConfig.

This module is the only place that knows the filenames and top-level
keys of the research config foundation. Downstream modules (geometry,
optics, pattern, ...) should consume an :class:`AppConfig` instance
rather than re-reading YAML directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CONFIG_FILES: dict[str, str] = {
    "project.yaml": "project",
    "geometry.yaml": "geometry",
    "materials.yaml": "materials",
    "optics.yaml": "optics",
    "angle_scan.yaml": "angle_scan",
    "pattern.yaml": "pattern",
    "optimization.yaml": "optimization",
    "thermal.yaml": "thermal",
    "validation.yaml": "validation",
    "tracking.yaml": "tracking",
}


class ConfigError(Exception):
    """Raised when config files are missing, malformed, or fail validation."""


@dataclass
class AppConfig:
    project: dict[str, Any]
    geometry: dict[str, Any]
    materials: dict[str, Any]
    optics: dict[str, Any]
    angle_scan: dict[str, Any]
    pattern: dict[str, Any]
    optimization: dict[str, Any]
    thermal: dict[str, Any]
    validation: dict[str, Any]
    tracking: dict[str, Any]
    mode: str | None
    config_dir: Path
    raw: dict[str, dict[str, Any]]

    def ray_count_for_mode(self, mode: str | None = None) -> int:
        resolved = self._resolve_mode(mode)
        ray_counts = self.optics.get("ray_counts")
        if not isinstance(ray_counts, dict) or resolved not in ray_counts:
            raise ConfigError(
                f"optics.yaml: ray_counts has no entry for mode '{resolved}'"
            )
        return int(ray_counts[resolved])

    def trials_for_mode(self, mode: str | None = None) -> int:
        resolved = self._resolve_mode(mode)
        trials = self.optimization.get("trials")
        if not isinstance(trials, dict) or resolved not in trials:
            raise ConfigError(
                f"optimization.yaml: trials has no entry for mode '{resolved}'"
            )
        return int(trials[resolved])

    def _resolve_mode(self, mode: str | None) -> str:
        chosen = mode if mode is not None else self.mode
        if chosen is None:
            raise ConfigError(
                "mode not provided: pass `mode=` explicitly or load via "
                "load_config(..., mode=...)"
            )
        return chosen


def load_config(
    config_dir: str | Path = "config",
    mode: str | None = None,
) -> AppConfig:
    """Load and validate the research config foundation.

    Validates: required files exist, each YAML parses to a dict, the
    expected top-level key is present and maps to a dict, and (if
    ``mode`` is given) the mode key exists in both
    ``optics.yaml::optics.ray_counts`` and
    ``optimization.yaml::optimization.trials``.
    """
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise ConfigError(f"Config directory not found: {config_dir}")

    raw: dict[str, dict[str, Any]] = {}
    inner: dict[str, dict[str, Any]] = {}

    for filename, top_key in REQUIRED_CONFIG_FILES.items():
        path = config_dir / filename
        if not path.is_file():
            raise ConfigError(f"Missing config file: {path}")

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ConfigError(
                f"{path}: expected top-level dict, got {type(data).__name__}"
            )
        if top_key not in data:
            raise ConfigError(
                f"{path}: missing top-level key '{top_key}'"
            )
        if not isinstance(data[top_key], dict):
            raise ConfigError(
                f"{path}: top-level key '{top_key}' must map to a dict, "
                f"got {type(data[top_key]).__name__}"
            )

        raw[top_key] = data
        inner[top_key] = data[top_key]

    if mode is not None:
        ray_counts = inner["optics"].get("ray_counts")
        if not isinstance(ray_counts, dict) or mode not in ray_counts:
            raise ConfigError(
                f"optics.yaml: ray_counts has no entry for mode '{mode}'"
            )
        trials = inner["optimization"].get("trials")
        if not isinstance(trials, dict) or mode not in trials:
            raise ConfigError(
                f"optimization.yaml: trials has no entry for mode '{mode}'"
            )

    return AppConfig(
        project=inner["project"],
        geometry=inner["geometry"],
        materials=inner["materials"],
        optics=inner["optics"],
        angle_scan=inner["angle_scan"],
        pattern=inner["pattern"],
        optimization=inner["optimization"],
        thermal=inner["thermal"],
        validation=inner["validation"],
        tracking=inner["tracking"],
        mode=mode,
        config_dir=config_dir,
        raw=raw,
    )
