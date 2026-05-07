"""Simplified target-heating surrogate.

Simplified target-heating surrogate; **not an ignition validation
model**. Solves a 0-D lumped-capacitance areal heat balance for a
flat absorbing target driven by a constant optical-flux input,
with optional convective and grey-body radiative loss to a fixed
ambient. The thermal model consumes an optical flux surrogate;
the input is **not calibrated W/m^2** unless the caller provides
calibrated incident irradiance. This is **not CFD**, **not
pyrolysis chemistry**, and does **not** prove fire prevention or
PET-bottle safety.

Energy balance (per unit area)
------------------------------
    dT/dt = (absorbed_flux
             - h_conv * (T - T_ambient)
             - emissivity * sigma * (T^4 - T_ambient^4)) / C_areal

with:
    absorbed_flux = absorptivity * incident_flux_w_m2
    sigma         = Stefan-Boltzmann constant (5.670374419e-8 W/m^2/K^4)
    C_areal       = areal heat capacity in J/m^2/K

The integrator is explicit Euler on a time grid that starts at
``0.0`` and lands exactly on ``duration_s``. Most steps have
length ``dt_s``; if ``duration_s`` is not an exact multiple of
``dt_s`` the **final** step is shorter so the last sample is
exactly ``duration_s``.

Out of scope
------------
Pyrolysis, smoldering, flaming, ignition criteria, material
spectral absorptivity, moisture, spatial heat conduction,
multi-layer / multi-material targets, time-varying incident
flux, ignition-test database, manufacturing claims, and physical
fire-prevention validation are all intentionally not modeled
here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann, W/m^2/K^4
DEFAULT_AMBIENT_TEMP_K = 293.15


class ThermalError(Exception):
    """Raised on invalid thermal-surrogate inputs."""


@dataclass(frozen=True)
class LumpedTargetHeatingResult:
    time_s: np.ndarray
    temperature_k: np.ndarray
    max_temperature_k: float
    final_temperature_k: float
    max_temperature_rise_k: float
    final_temperature_rise_k: float
    time_to_threshold_s: float | None
    incident_flux_w_m2: float
    absorbed_flux_w_m2: float
    absorptivity: float
    areal_heat_capacity_j_m2k: float
    h_conv_w_m2k: float
    emissivity: float
    ambient_temp_k: float
    initial_temp_k: float
    duration_s: float
    dt_s: float
    model_type: str = "lumped_target_heating_surrogate"


def _check_finite_positive(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalError(
            f"{name} must be a finite float > 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise ThermalError(f"{name} must be finite; got {v}")
    if v <= 0.0:
        raise ThermalError(f"{name} must be > 0; got {v}")
    return v


def _check_finite_nonneg(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalError(
            f"{name} must be a finite float >= 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise ThermalError(f"{name} must be finite; got {v}")
    if v < 0.0:
        raise ThermalError(f"{name} must be >= 0; got {v}")
    return v


def _check_finite_unit_interval(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalError(
            f"{name} must be a finite float in [0, 1]; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise ThermalError(f"{name} must be finite; got {v}")
    if v < 0.0 or v > 1.0:
        raise ThermalError(f"{name} must be in [0, 1]; got {v}")
    return v


def _build_time_grid(duration_s: float, dt_s: float) -> np.ndarray:
    duration = float(duration_s)
    dt = float(dt_s)
    n_full = int(np.floor(duration / dt + 1e-9))
    body = np.arange(n_full + 1, dtype=float) * dt
    if body.size == 0 or body[-1] + dt * 1e-9 < duration:
        body = np.append(body, duration)
    else:
        body = body.copy()
        body[-1] = duration
    return body


def simulate_lumped_target_heating(
    *,
    incident_flux_w_m2: float,
    duration_s: float,
    dt_s: float,
    areal_heat_capacity_j_m2k: float,
    absorptivity: float = 1.0,
    h_conv_w_m2k: float = 0.0,
    emissivity: float = 0.0,
    ambient_temp_k: float = DEFAULT_AMBIENT_TEMP_K,
    initial_temp_k: float | None = None,
    threshold_temp_k: float | None = None,
) -> LumpedTargetHeatingResult:
    """Run the lumped-target heating surrogate to completion.

    Simplified target-heating surrogate; **not an ignition
    validation model**. Integrates ``dT/dt = (absorbed_flux -
    convective_loss - radiative_loss) / areal_heat_capacity`` from
    ``t = 0`` to ``t = duration_s`` with explicit Euler. The
    ``incident_flux_w_m2`` input is treated as a scalar optical
    flux surrogate; it is **not calibrated W/m^2** unless the
    caller provides a calibrated incident irradiance.

    Parameters
    ----------
    incident_flux_w_m2
        Constant scalar optical flux input (``>= 0``).
    duration_s
        Total integration time (``> 0``).
    dt_s
        Nominal integration step size (``> 0``, ``<= duration_s``).
        The final step may be shorter so the last time sample is
        exactly ``duration_s``.
    areal_heat_capacity_j_m2k
        Lumped areal heat capacity ``C_areal`` (``> 0``).
    absorptivity
        Fraction of incident flux absorbed by the target, in
        ``[0, 1]``. Defaults to ``1.0``.
    h_conv_w_m2k
        Linear convective coefficient (``>= 0``). Defaults to
        ``0.0`` (no convective loss).
    emissivity
        Grey-body emissivity in ``[0, 1]``. Defaults to ``0.0``
        (no radiative loss).
    ambient_temp_k
        Reference ambient temperature in K (``> 0``). Defaults to
        ``293.15`` K.
    initial_temp_k
        Initial target temperature in K (``> 0``). If ``None``,
        defaults to ``ambient_temp_k``.
    threshold_temp_k
        Optional finite positive threshold (``> 0``). If
        provided, the time at which ``T(t) >= threshold_temp_k``
        first holds is returned in ``time_to_threshold_s``;
        otherwise ``None``.

    Raises
    ------
    ThermalError
        On any invalid input (non-finite, out-of-range, or
        ``dt_s > duration_s``).
    """
    incident = _check_finite_nonneg(
        incident_flux_w_m2, name="incident_flux_w_m2"
    )
    duration = _check_finite_positive(duration_s, name="duration_s")
    dt = _check_finite_positive(dt_s, name="dt_s")
    if dt > duration:
        raise ThermalError(
            f"dt_s must be <= duration_s; got dt_s={dt}, "
            f"duration_s={duration}"
        )
    capacity = _check_finite_positive(
        areal_heat_capacity_j_m2k, name="areal_heat_capacity_j_m2k"
    )
    alpha = _check_finite_unit_interval(absorptivity, name="absorptivity")
    h_conv = _check_finite_nonneg(h_conv_w_m2k, name="h_conv_w_m2k")
    eps = _check_finite_unit_interval(emissivity, name="emissivity")
    t_amb = _check_finite_positive(ambient_temp_k, name="ambient_temp_k")
    if initial_temp_k is None:
        t_init = t_amb
    else:
        t_init = _check_finite_positive(
            initial_temp_k, name="initial_temp_k"
        )
    threshold: float | None
    if threshold_temp_k is None:
        threshold = None
    else:
        threshold = _check_finite_positive(
            threshold_temp_k, name="threshold_temp_k"
        )

    times = _build_time_grid(duration, dt)
    T = np.empty(times.size, dtype=float)
    T[0] = t_init

    absorbed = alpha * incident
    for i in range(times.size - 1):
        step = times[i + 1] - times[i]
        Ti = T[i]
        conv_loss = h_conv * (Ti - t_amb)
        rad_loss = eps * SIGMA_SB * (Ti ** 4 - t_amb ** 4)
        dT_dt = (absorbed - conv_loss - rad_loss) / capacity
        T[i + 1] = Ti + step * dT_dt

    max_T = float(T.max())
    final_T = float(T[-1])

    time_to_threshold: float | None
    if threshold is None:
        time_to_threshold = None
    else:
        crossings = np.where(T >= threshold)[0]
        if crossings.size > 0:
            time_to_threshold = float(times[int(crossings[0])])
        else:
            time_to_threshold = None

    return LumpedTargetHeatingResult(
        time_s=times.copy(),
        temperature_k=T.copy(),
        max_temperature_k=max_T,
        final_temperature_k=final_T,
        max_temperature_rise_k=float(max_T - t_init),
        final_temperature_rise_k=float(final_T - t_init),
        time_to_threshold_s=time_to_threshold,
        incident_flux_w_m2=float(incident),
        absorbed_flux_w_m2=float(absorbed),
        absorptivity=float(alpha),
        areal_heat_capacity_j_m2k=float(capacity),
        h_conv_w_m2k=float(h_conv),
        emissivity=float(eps),
        ambient_temp_k=float(t_amb),
        initial_temp_k=float(t_init),
        duration_s=float(duration),
        dt_s=float(dt),
    )
