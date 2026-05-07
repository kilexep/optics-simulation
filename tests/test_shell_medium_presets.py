import dataclasses

import pytest

from optics_simulation.optics import (
    OpticsError,
    ShellMediumPreset,
    create_shell_medium_preset,
)


_IOR_AIR = 1.00028
_IOR_PET = 1.575
_IOR_WATER = 1.333


def test_air_preset_has_four_interfaces() -> None:
    preset = create_shell_medium_preset(fill_medium="air")
    assert isinstance(preset, ShellMediumPreset)
    assert len(preset.interface_sequence) == 4


def test_water_preset_has_four_interfaces() -> None:
    preset = create_shell_medium_preset(fill_medium="water")
    assert isinstance(preset, ShellMediumPreset)
    assert len(preset.interface_sequence) == 4


def test_air_preset_sequence_is_air_pet_air_pet_air() -> None:
    preset = create_shell_medium_preset(
        fill_medium="air",
        ior_air=_IOR_AIR,
        ior_pet=_IOR_PET,
        ior_water=_IOR_WATER,
    )
    expected = (
        (_IOR_AIR, _IOR_PET),
        (_IOR_PET, _IOR_AIR),
        (_IOR_AIR, _IOR_PET),
        (_IOR_PET, _IOR_AIR),
    )
    assert preset.interface_sequence == expected
    assert preset.fill_ior == pytest.approx(_IOR_AIR)


def test_water_preset_sequence_is_air_pet_water_pet_air() -> None:
    preset = create_shell_medium_preset(
        fill_medium="water",
        ior_air=_IOR_AIR,
        ior_pet=_IOR_PET,
        ior_water=_IOR_WATER,
    )
    expected = (
        (_IOR_AIR, _IOR_PET),
        (_IOR_PET, _IOR_WATER),
        (_IOR_WATER, _IOR_PET),
        (_IOR_PET, _IOR_AIR),
    )
    assert preset.interface_sequence == expected
    assert preset.fill_ior == pytest.approx(_IOR_WATER)


def test_invalid_fill_medium_raises() -> None:
    with pytest.raises(OpticsError, match="fill_medium"):
        create_shell_medium_preset(fill_medium="oil")
    with pytest.raises(OpticsError, match="fill_medium"):
        create_shell_medium_preset(fill_medium="")


def test_invalid_ior_values_raise() -> None:
    bad_values = (0.0, -1.0, float("nan"), float("inf"), -float("inf"))
    for bad in bad_values:
        with pytest.raises(OpticsError, match="ior_air"):
            create_shell_medium_preset(ior_air=bad)
        with pytest.raises(OpticsError, match="ior_pet"):
            create_shell_medium_preset(ior_pet=bad)
        with pytest.raises(OpticsError, match="ior_water"):
            create_shell_medium_preset(
                fill_medium="water", ior_water=bad,
            )


def test_preset_is_immutable_dataclass() -> None:
    preset = create_shell_medium_preset(fill_medium="air")
    assert dataclasses.is_dataclass(preset)
    with pytest.raises(dataclasses.FrozenInstanceError):
        preset.fill_ior = 9.99  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        preset.name = "tampered"  # type: ignore[misc]


def test_description_mentions_synthetic_and_no_auto_medium_tracking() -> None:
    air_preset = create_shell_medium_preset(fill_medium="air")
    water_preset = create_shell_medium_preset(fill_medium="water")
    for preset in (air_preset, water_preset):
        text = preset.description.lower()
        assert "synthetic" in text
        assert "not automatic medium tracking" in text
