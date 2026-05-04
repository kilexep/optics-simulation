"""Config loader for optics-simulation.

Reads YAML files under ``config/``, validates required files and
top-level keys, and exposes a structured :class:`AppConfig` object.
"""
from optics_simulation.config.loader import AppConfig, ConfigError, load_config

__all__ = ["AppConfig", "ConfigError", "load_config"]
