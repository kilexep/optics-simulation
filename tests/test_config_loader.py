import shutil
from pathlib import Path

import pytest

from optics_simulation.config import AppConfig, ConfigError, load_config


REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _copy_config(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for src in REPO_CONFIG_DIR.glob("*.yaml"):
        shutil.copy(src, dst / src.name)
    return dst


def test_load_config_from_default_dir() -> None:
    cfg = load_config(REPO_CONFIG_DIR)
    assert isinstance(cfg, AppConfig)
    assert cfg.mode is None
    assert cfg.config_dir == REPO_CONFIG_DIR


def test_appconfig_exposes_inner_dicts() -> None:
    cfg = load_config(REPO_CONFIG_DIR)
    assert cfg.project["name"] == "optics-simulation"
    assert "project" not in cfg.project
    assert cfg.geometry["units"] == "mm"
    assert "ray_counts" in cfg.optics
    assert "comparison_groups" in cfg.validation


def test_raw_preserves_full_yaml() -> None:
    cfg = load_config(REPO_CONFIG_DIR)
    expected = {
        "project", "geometry", "materials", "optics", "angle_scan",
        "pattern", "optimization", "thermal", "validation", "tracking",
    }
    assert set(cfg.raw.keys()) == expected
    assert cfg.raw["project"]["project"]["name"] == "optics-simulation"


def test_missing_file_raises(tmp_path: Path) -> None:
    dst = tmp_path / "config"
    dst.mkdir()
    for name in ("project.yaml", "geometry.yaml"):
        shutil.copy(REPO_CONFIG_DIR / name, dst / name)
    with pytest.raises(ConfigError, match="Missing config file"):
        load_config(dst)


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist")


def test_invalid_top_level_key_raises(tmp_path: Path) -> None:
    dst = _copy_config(tmp_path / "config")
    (dst / "project.yaml").write_text("wrong_key:\n  name: foo\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top-level key 'project'"):
        load_config(dst)


def test_yaml_not_dict_raises(tmp_path: Path) -> None:
    dst = _copy_config(tmp_path / "config")
    (dst / "project.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top-level dict"):
        load_config(dst)


def test_top_level_key_not_dict_raises(tmp_path: Path) -> None:
    dst = _copy_config(tmp_path / "config")
    (dst / "project.yaml").write_text("project: just_a_string\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must map to a dict"):
        load_config(dst)


def test_load_config_with_valid_mode() -> None:
    cfg = load_config(REPO_CONFIG_DIR, mode="debug")
    assert cfg.mode == "debug"
    assert cfg.ray_count_for_mode() == 10000
    assert cfg.trials_for_mode() == 10


def test_load_config_with_invalid_mode_raises() -> None:
    with pytest.raises(ConfigError, match="'bogus'"):
        load_config(REPO_CONFIG_DIR, mode="bogus")


def test_ray_count_for_mode_explicit_arg() -> None:
    cfg = load_config(REPO_CONFIG_DIR)
    assert cfg.ray_count_for_mode("screening") == 100000
    assert cfg.ray_count_for_mode("final") == 1000000


def test_trials_for_mode_explicit_arg() -> None:
    cfg = load_config(REPO_CONFIG_DIR)
    assert cfg.trials_for_mode("debug") == 10
    assert cfg.trials_for_mode("final") == 500


def test_ray_count_for_mode_without_mode_raises() -> None:
    cfg = load_config(REPO_CONFIG_DIR)
    with pytest.raises(ConfigError, match="mode not provided"):
        cfg.ray_count_for_mode()


def test_trials_for_mode_without_mode_raises() -> None:
    cfg = load_config(REPO_CONFIG_DIR)
    with pytest.raises(ConfigError, match="mode not provided"):
        cfg.trials_for_mode()


def test_ray_count_for_mode_explicit_arg_overrides_loaded_mode() -> None:
    cfg = load_config(REPO_CONFIG_DIR, mode="debug")
    assert cfg.ray_count_for_mode("final") == 1000000
    assert cfg.mode == "debug"


def test_nested_dict_not_overwritten_by_mode() -> None:
    cfg = load_config(REPO_CONFIG_DIR, mode="debug")
    assert isinstance(cfg.optics["ray_counts"], dict)
    assert set(cfg.optics["ray_counts"].keys()) >= {"debug", "screening", "final"}
    assert isinstance(cfg.optimization["trials"], dict)
    assert set(cfg.optimization["trials"].keys()) >= {"debug", "screening", "final"}


def test_config_dir_accepts_str(tmp_path: Path) -> None:
    dst = _copy_config(tmp_path / "config")
    cfg = load_config(str(dst))
    assert cfg.project["name"] == "optics-simulation"


def test_unknown_nested_keys_are_allowed(tmp_path: Path) -> None:
    dst = _copy_config(tmp_path / "config")
    project_path = dst / "project.yaml"
    text = project_path.read_text(encoding="utf-8")
    project_path.write_text(text + "\nextra_unknown_section:\n  free_form: true\n", encoding="utf-8")
    cfg = load_config(dst)
    assert cfg.raw["project"]["extra_unknown_section"]["free_form"] is True
