from pathlib import Path


REQUIRED_CONFIG_FILES = [
    "project.yaml",
    "geometry.yaml",
    "materials.yaml",
    "optics.yaml",
    "angle_scan.yaml",
    "pattern.yaml",
    "optimization.yaml",
    "thermal.yaml",
    "validation.yaml",
    "tracking.yaml",
]


def test_required_config_files_exist_and_are_not_empty() -> None:
    config_dir = Path("config")

    for file_name in REQUIRED_CONFIG_FILES:
        path = config_dir / file_name
        assert path.exists(), f"Missing config file: {path}"
        assert path.stat().st_size > 0, f"Empty config file: {path}"
