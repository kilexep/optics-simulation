"""Synthetic shell-fixture medium / interface presets.

Synthetic shell/fill-state optical-to-thermal smoke check; **not
a physical PET-bottle validation**. Builds caller-facing
:class:`ShellMediumPreset` value objects that bundle a fill
medium label, the matching ``interface_sequence`` for
:func:`run_multi_step_trace`, and a short description for
demo/test output.

Why this exists
---------------
The multi-step trace orchestrator does **not** auto-track media —
it consumes the ``interface_sequence`` it is given verbatim. For
the synthetic cylindrical-shell fixture
(:func:`create_subdivided_synthetic_bottle_shell`) a single
side-incidence ray crosses **four** interfaces:

- empty shell:
  ``air -> PET -> air -> PET -> air``
  ``[(air, PET), (PET, air), (air, PET), (PET, air)]``
- water-filled shell:
  ``air -> PET -> water -> PET -> air``
  ``[(air, PET), (PET, water), (water, PET), (PET, air)]``

This module produces those two presets (and only those two) for
the synthetic side-incidence smoke setup. The presets are not a
substitute for automatic medium tracking: callers using a
different geometry, a different incidence direction, or a
different fill medium must build their own sequence. The presets
do not change the refraction convention; they do not modify
:mod:`optics_simulation.optics.refraction` or
:mod:`optics_simulation.optics.multi_step`; and they do not
re-interpret ``(eta_i, eta_t)`` ordering.

Out of scope
------------
Automatic medium tracking, real PET STL handling, neck / shoulder
/ base petaloid geometry, cap geometry, water volume meshing,
physical water absorption, spectral dispersion, polarization,
calibrated W/m^2 anchoring, and fire-prevention or PET-bottle
safety claims are intentionally not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.optics.ray import OpticsError


_SUPPORTED_FILL_MEDIA = ("air", "water")


@dataclass(frozen=True)
class ShellMediumPreset:
    name: str
    interface_sequence: tuple[tuple[float, float], ...]
    fill_ior: float
    description: str


def _check_positive_ior(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise OpticsError(
            f"{name} must be a finite positive float; got {value!r}"
        )
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise OpticsError(
            f"{name} must be a finite positive float; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise OpticsError(f"{name} must be finite; got {v}")
    if v <= 0.0:
        raise OpticsError(f"{name} must be > 0; got {v}")
    return v


def create_shell_medium_preset(
    *,
    fill_medium: str = "air",
    ior_air: float = 1.00028,
    ior_pet: float = 1.575,
    ior_water: float = 1.333,
) -> ShellMediumPreset:
    """Build a synthetic shell-fixture medium / interface preset.

    Synthetic shell/fill-state optical-to-thermal smoke check;
    **not a physical PET-bottle validation**. Returns a frozen
    :class:`ShellMediumPreset` whose ``interface_sequence`` is
    a fixed 4-interface preset for the synthetic
    cylindrical-shell side-incidence setup. The preset is **not**
    automatic medium tracking — callers using a different geometry
    or incidence direction must build their own sequence.

    Parameters
    ----------
    fill_medium
        ``"air"`` (empty shell) or ``"water"`` (water-filled
        shell). Any other value raises :class:`OpticsError`.
    ior_air, ior_pet, ior_water
        Refractive indices of air, PET, and water. All must be
        finite positive floats; ``bool`` values are rejected.

    Returns
    -------
    ShellMediumPreset
        Immutable preset whose ``interface_sequence`` is::

            empty shell:
                ((air, PET), (PET, air), (air, PET), (PET, air))
            water-filled shell:
                ((air, PET), (PET, water), (water, PET), (PET, air))

    Raises
    ------
    OpticsError
        On invalid ``fill_medium`` or any non-positive / non-finite
        IOR value.
    """
    air = _check_positive_ior(ior_air, name="ior_air")
    pet = _check_positive_ior(ior_pet, name="ior_pet")
    water = _check_positive_ior(ior_water, name="ior_water")

    if fill_medium not in _SUPPORTED_FILL_MEDIA:
        raise OpticsError(
            f"fill_medium must be one of {_SUPPORTED_FILL_MEDIA}; "
            f"got {fill_medium!r}"
        )

    if fill_medium == "air":
        seq: tuple[tuple[float, float], ...] = (
            (air, pet),
            (pet, air),
            (air, pet),
            (pet, air),
        )
        fill_ior = air
        description = (
            "Synthetic empty-shell preset for the cylindrical "
            "shell side-incidence smoke setup; not automatic "
            "medium tracking and not a physical PET-bottle "
            "validation."
        )
    else:
        seq = (
            (air, pet),
            (pet, water),
            (water, pet),
            (pet, air),
        )
        fill_ior = water
        description = (
            "Synthetic water-filled shell preset for the "
            "cylindrical shell side-incidence smoke setup; not "
            "automatic medium tracking and not a physical "
            "PET-bottle validation."
        )

    return ShellMediumPreset(
        name=f"shell_{fill_medium}_fill",
        interface_sequence=seq,
        fill_ior=float(fill_ior),
        description=description,
    )
