"""Circular metal waveguide

This suite of functions implements some basic formulas for standard hollow,
i.e. air-filled circular waveguide with metal walls. The equations are taken
from the Waveguide Handbook by Marcuvitz.
"""

import numpy as np
import scipy.special
from scipy.constants import speed_of_light, epsilon_0, mu_0
from aux_funcs import metal_resistance


def calc_cutoff_frequency(wvg_diameter: float, mode: str,
                          mode_n: int, mode_m: int) -> float:
    """Calculate cutoff frequency

    This function calculates the cutoff frequency, i.e. the lowest frequency
    for which a particular mode will propagate, in a specified waveguide. The
    mode of interest needs to be specified in advance.

    Args:
        wvg_diameter: A `float` with the diameter of the waveguide which
                      is being checked. Units are metres.
        mode: A `str` specifying what type of mode is propagating along the
              waveguide. Valid values are TE or TM.
        mode_n: The `n` index of the mode of interest
        mode_m: The `m` index of the mode of interest.

    Returns:
        The cutoff frequency in GHz as a `float` number.

    Raises:
        ValueError: In case of a mode different than TE or TM
        ZeroDivisionError: In case the waveguide diameter is given as zero
    """
    if 'te' == mode.lower():
        bessel_root = scipy.special.jnp_zeros(mode_n, mode_m)[mode_m - 1]
    elif 'tm' == mode.lower():
        bessel_root = scipy.special.jn_zeros(mode_n, mode_m)[mode_m - 1]
    else:
        raise ValueError('Propagation mode must be TE or TM')

    wavelength = np.pi * wvg_diameter / bessel_root

    try:
        freq = speed_of_light / wavelength
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide diameter must be > 0'). \
              with_traceback(error.__traceback__)

    freq /= 1e9

    return freq


def calc_guide_wavelength(freq: float, wvg_diameter: float, mode: str,
                          mode_n: int, mode_m: int) -> float:
    """Calculate guide wavelength

    This function calculates the guide wavelength for a particular propagating
    mode at a specified frequency in a specified circular waveguide.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_diameter: A `float` with the diameter of the waveguide which
                      is being checked. Units are metres.
        mode: A `str` specifying what type of mode is propagating along the
              waveguide. Valid values are TE or TM.
        mode_n: The `n` index of the mode of interest
        mode_m: The `m` index of the mode of interest.

    Returns:
        The guide wavelength in metres as a `float` number.

    Raises:
        ZeroDivisionError: In case the frequency is given as zero
    """
    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    # ! `calc_cutoff_frequency` returns the cutoff frequency in GHz, so we
    # ! have to convert it to Hz first and then to a wavelength in metres.
    cutoff_wavelength = calc_cutoff_frequency(
        wvg_diameter, mode, mode_n, mode_m
    )
    cutoff_wavelength *= 1e9
    cutoff_wavelength = speed_of_light / cutoff_wavelength

    guide_wavelength = 1 - np.float_power(wavelength / cutoff_wavelength, 2)
    guide_wavelength = 1 / np.sqrt(guide_wavelength)

    return guide_wavelength


def calc_attenuation_constant(freq: float, wvg_diameter: float,
                              mode: str, mode_n: int, mode_m: int,
                              metal_conductivity: float,
                              metal_permeability: float):
    """Calculate attenuation constant

    This function calculates the attenuation constant of a particular mode
    at a particular frequency in a particular circular waveguide, which is
    due to the finite conductivity of the metal used for the waveguide
    construction.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_diameter: A `float` with the diameter of the waveguide which
                      is being checked. Units are metres.
        mode: A `str` specifying what type of mode is propagating along the
              waveguide. Valid values are TE or TM.
        mode_n: The `n` index of the mode of interest.
        mode_m: The `m` index of the mode of interest.
        metal_conductivity: A `float` with the conductivity of the material
                            of which the waveguide walls are made. Units are
                            S/m or mhos/m.
        metal_permeability: A `float` with the real part of the relative
                            permeability of the waveguide wall material.

    Returns:
        The attenuation constant in Np/m as a `float` number.

    Raises:
        RuntimeError: If a mode differen than TE or TM is specified and not
                      caught by the `calc_cutoff_frequency` function.
    """

    z_freespace = np.sqrt(mu_0 / epsilon_0)

    r_metal = metal_resistance(
        freq, metal_conductivity, metal_permeability
    )

    # ! `calc_cutoff_frequency` returns the cutoff frequency in GHz, so we
    # ! have to convert it to Hz first and then to a wavelength in metres.
    cutoff_wavelength = calc_cutoff_frequency(
        wvg_diameter, mode, mode_n, mode_m
    )
    cutoff_wavelength *= 1e9
    cutoff_wavelength = speed_of_light / cutoff_wavelength

    freq *= 1e9
    wavelength = speed_of_light / freq

    alpha_1 = r_metal / (z_freespace * wvg_diameter / 2)

    alpha_2 = 1 - np.float_power(wavelength / cutoff_wavelength, 2)
    alpha_2 = 1 / np.sqrt(alpha_2)

    if 'te' == mode.lower():
        bessel_root = scipy.special.jnp_zeros(mode_n, mode_m)[mode_m - 1]
        alpha_3 = np.float_power(mode_n, 2)
        alpha_3 /= (np.float_power(bessel_root, 2) - np.float_power(mode_n, 2))
        alpha_3 += np.float_power(wavelength / cutoff_wavelength, 2)
    elif 'tm' == mode.lower():
        alpha_3 = 1
    else:
        raise RuntimeError('Something terrible has happened')

    alpha = alpha_1 * alpha_3 * alpha_2

    return alpha


def calc_phase_constant(freq: float, wvg_diameter: float, mode: str,
                        mode_n: int, mode_m: int) -> float:
    """Calculate phase constant

    This function calculates the phase constant of a particular mode
    at a particular frequency in a particular circular waveguide.

    Notes:
        1. There is an implicit assumption here for a lossless propagation.
        While that is not strictly true due to the finite conductivity of the
        material used to make the waveguide, it is a close enough for practical
        purposes approximation.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_diameter: A `float` with the diameter of the waveguide which
                      is being checked. Units are metres.
        mode: A `str` specifying what type of mode is propagating along the
              waveguide. Valid values are TE or TM.
        mode_n: The `n` index of the mode of interest.
        mode_m: The `m` index of the mode of interest.

    Returns:
        The attenuation constant in Np/m as a `float` number.

    Raises:
        RuntimeError: If a mode differen than TE or TM is specified and not
                      caught by the `calc_cutoff_frequency` function.
    """

    guide_wavelength = calc_guide_wavelength(freq, wvg_diameter, mode,
                                             mode_n, mode_m)

    beta = 2 * np.pi / guide_wavelength

    return beta
