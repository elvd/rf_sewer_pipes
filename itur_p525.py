"""Implementation of some of the ITU-R P.525 formulas

ITU-R Recommendation P.525 (08/2019) Calculation of Free-Space Attenuation
is a short Recommendation which specifies how to calculate free-space path
loss. In addition it contains several formulas for conversion between
electric field strength and power flux at a given distance from a transmitter.

Some extra formulas are included here that fit thematically, such as converting
from W/m2 to V/m
"""

import numpy as np
from scipy.constants import speed_of_light, epsilon_0


np.seterr(divide='raise')


def free_space_path_loss(freq: float, distance: float) -> float:
    """Calculate free-space path loss

    Standard formula, also known as (one of) the Friis formula(e).

    Args:
        freq: A `float` with the frequency of the transmitter.
              Unites are GHz.
        distance: A `float` with the path length. Units are metres.

    Returns:
        The path loss in dB as a `float` number.

    Raises:
        ZeroDivisionError: In case the frequency is given as zero.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    path_loss = 20 * np.log10(4 * np.pi * distance / wavelength)

    return path_loss


def field_strength_at_distance(power: float, distance: float,
                               mode: str = 'dBW') -> float:
    """Calculate E-field strength at a specified distance

    Sometimes it is useful to know the field strength at a particular
    distance away from a transmitter.

    Args:
        power: A `float` with the EIRP of the transmitter. See `mode`
               for more info on the units.
        distance: A `float` with the distance of interes. Units are km.
        mode: A `str` specifying the units of `power`. Supported ones are
              dBW, dBm, W, and mW. Default is dBW.

    Returns:
        The electric field strength in dBuV/m as a `float` number.

    Raises:
        RuntimeError: In case an unsupported unit is given for the `power`.
        ValueError: In case distance is given as a negative number.
    """

    if 'dbm' == mode.lower():
        power -= 30
    elif 'mw' == mode.lower():
        power /= 1e3
        power = 10 * np.log10(power)
    elif 'w' == mode.lower():
        power = 10 * np.log10(power)
    elif 'dbw' == mode.lower():
        pass
    else:
        raise RuntimeError('Unsupported power unit')

    if 0 > distance:
        raise ValueError('Distance must be > 0')

    field_strength = power - 20 * np.log10(distance) + 74.8

    return field_strength


def power_flux_at_distance(power: float, distance: float,
                           mode: str = 'dBW') -> float:
    """Calculate power flux at a specified distance

    Another way to represent how much EM energy there is a specific
    distance away from a transmitter. Calls `field_strength_at_distance()`
    internally.

    Args:
        power: A `float` with the EIRP of the transmitter. See `mode`
               for more info on the units.
        distance: A `float` with the distance of interes. Units are km.
        mode: A `str` specifying the units of `power`. Supported ones are
              dBW, dBm, W, and mW. Default is dBW.

    Returns:
        The power flux in dBW/m2 as a `float` number.

    Raises:
        RuntimeError: In case an unsupported unit is given for the `power`.
        ValueError: In case distance is given as a negative number.
    """

    field_strength = field_strength_at_distance(power, distance, mode)

    power_flux = field_strength - 145.8

    return power_flux


def intensity_to_field_strength(intensity: float) -> float:
    """Calculate E-field strength

    A quick and simple conversion between EM field intensity in W/m2
    to electric field strength in V/m.

    Args:
        intensity: A `float` with the EM field intensity in W/m2

    Returns:
        The electric field strength in V/m as a `float` number.

    Raises:
        Nothing
    """

    field_strength = (2 * intensity) / (speed_of_light * epsilon_0)
    field_strength = np.sqrt(field_strength)

    return field_strength
