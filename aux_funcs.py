"""Helper functions for Python-based EM simulations

A library of utility functions useful when setting up and running various
EM simulations using Pythin tools, such as gprMax, scikit-rf, or generic
RF/microwave calculations.

Currently mainly contains functions for swapping between different ways
of representing the complex relative permittivity of materials.

Other functions are used to calculate current through a Hertzian dipole
based on required radiated power, and the equivalent relative permittivity
of a multi-layer medium.

Added 26.VI.2020:
    - Conversion between dB and Np
    - Skin depth calculator
    - Far field distance calculator
    - Fresnel zone calculator
    - Maximum antenna separation for a given first Fresnel zone radius
    - Propagation constant for a plane wave through homogeneous medium
"""

import warnings
from typing import List, Tuple
import numpy as np
from scipy.constants import epsilon_0, mu_0, speed_of_light


np.seterr(divide='raise')


def conductivity_to_tan_delta(freq: float, conductivity: float,
                              real_permittivity: float) -> float:
    """Converts between conductivity and loss tangent at a specific frequency

    This is a simple and straightforward conversion between the value of
    conductivity, in S/m, at a particular frequency, and a loss tangent.

    Args:
        freq: A `float` with the frequency, in GHz, at which to do the
              conversion
        conductivity: A `float` value for the conductivity in S/m
        real_permittivity: A `float` value for the real part of the
                           complex relative permittivity.

    Returns:
        The value for the loss tangent, as a `float` number.

    Raises:
        ZeroDivisionError: If you specify 0 Hz, i.e. DC, for the frequency
    """

    try:
        tan_delta = 17.97591 * conductivity / (real_permittivity * freq)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Real part and frequency must be > 0'). \
              with_traceback(error.__traceback__)

    return tan_delta


def tan_delta_to_conductivity(freq: float, real_permittivity: float,
                              tan_delta: float) -> float:
    """Converts between loss tangent and conductivity

    This is a simple and straightforward conversion between the loss tangent
    and the value of conductivity, in S/m, at a particular frequency.

    Args:
        freq: A `float` with the frequency, in GHz, at which to do the
              conversion
        real_permittivity: A `float` value for the real part of the complex
                           relative permittivity.
        tan_delta: A `float` value for the loss tangent.

    Returns:
        The value for conductivity in S/m, which is equivalent to mho/m.

    Raises:
        ZeroDivisionError: If you specify 0 Hz, i.e. DC, for the frequency
    """

    if np.isclose(freq, 0):
        raise ZeroDivisionError('Frequency must be > 0')

    conductivity = 0.05563 * real_permittivity * tan_delta * freq

    return conductivity


def complex_permittivity_to_tan_delta(real_permittivity: float,
                                      imag_permittivity: float) -> float:
    """Computes loss tangent from complex relative permittivity

    This is a simple and straightforward calculation of a material's loss
    tangent from the real and imaginary parts of its complex relative
    permittivity.

    Args:
        real_permittivity: A `float` value for te real part of the
                           complex relative permittivity.
        imag_permittivity: A `float` value for the imaginary part of the
                           complex relative permittivity.

    Returns:
        The value for the loss tangent.

    Raises:
        ZeroDivisionError: If you specify 0 Hz, i.e. DC, for the frequency
    """

    try:
        tan_delta = imag_permittivity / real_permittivity
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Real part must be > 0'). \
              with_traceback(error.__traceback__)

    return tan_delta


def imaginary_permittivity_to_conductivity(freq: float,
                                           imag_permittivity: float) -> float:
    """Converts between imaginary permittivity and conductivity

    This is a simple and straightforward conversion between the imaginary
    part of the complex relative permittivity and the value of conductivity,
    in S/m, at a particular frequency.

    Args:
        freq: A `float` with the frequency, in GHz, at which to do the
              conversion
        imag_permittivity: A `float` value for the imaginary part of the
                           complex relative permittivity.

    Returns:
        The value for conductivity in S/m, which is equivalent to mho/m.

    Raises:
        ZeroDivisionError: If you specify 0 Hz, i.e. DC, for the frequency
    """

    if np.isclose(freq, 0):
        raise ZeroDivisionError('Frequency must be > 0')

    conductivity = 0.05563 * freq * imag_permittivity

    return conductivity


def conductivity_to_imaginary_permittivity(freq: float,
                                           conductivity: float) -> float:
    """Converts between conductivity and imaginary permittivity

    This is a simple and straightforward conversion between the value
    of conductivity, in S/m, and the imaginary part of the complex
    relative permittivity, at a particular frequency.

    Args:
        freq: A `float` with the frequency, in GHz, at which to do the
              conversion
        conductivity: A `float` value for the conductivity in S/m

    Returns:
        The unitless value for the imaginary part of the complex
        relative permittivity.

    Raises:
        ZeroDivisionError: If you specify 0 Hz, i.e. DC, for the frequency
    """

    try:
        imag_permittivity = 17.97591 * conductivity / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    return imag_permittivity


def equivalent_relative_permittivity(epsilon_real: List[float],
                                     thicknesses: List[float]) -> float:
    """Calculate equivalent relative permittivity of multi-layer media

    Uses the low-frequency approximation for equivalent real part of the
    relative permittivity of a multi-layer dielectric media. This approximation
    is valid up to about 100 GHz.

    Args:
        epsilon_real: A `List` of `float` values for the real part of the
                      relative permittivity of the individual layers.
        thicknesses: A `List` of `float` values for the thickness of the
                     individual layers. Units do not matter as long as they
                     are all the same.

    Returns:
        A `float` with the equivalent relative permittivity.

    Raises:
        ZeroDivisionError: In case the total thickness is zero, or a
                           particular relative permittivity is given as
                           zero.
    """

    total_thickness = np.sum(thicknesses)
    epsilon_real_eff = 0.0

    try:
        for er_eff, ind_thickness in zip(epsilon_real, thicknesses):
            epsilon_real_eff += (ind_thickness / (er_eff * total_thickness))
        epsilon_real_eff = 1 / epsilon_real_eff
    except ZeroDivisionError as error:
        raise ZeroDivisionError('One or more arguments evaluate to zero'). \
              with_traceback(error.__traceback__)

    return epsilon_real_eff


def hertzian_dipole_current(freq: float, power: float, length: float,
                            units: str = 'dBm') -> float:
    """Convert Hertzian dipole radiated power to current

    This is a short helper function for gprMax that converts desired radiated
    power by a Hertzian dipole into the current amplitude that is required
    to be passed through the dipole.

    Notes:
        1. Strictly speaking, this formula is valid for a time-harmonic current
        excitation. Other, stranger excitations require more maths.
        2. The units for `wavelength` and `length` must be the same.

    Args:
        freq: A `float` with the frequency at which the dipole is radiating.
              Units are GHz.
        power: A `float` with the desired radiated power. Can be in either
               dBm or W units.
        length: A `float` with the physical length of the Hertzian dipole.
                Units are metres.
        units: A `str` with the units for `power`. Supported ones are 'W' and
               'dBm', with the latter converted to the former internally.

    Returns:
        A `float` value with the required current, in A, to be passed through
        the dipole to result in the desired radiated power.

    Raises:
        RuntimeWarning: If the physical length of the dipole is not much
                        smaller than the wavelength.
        ZeroDivisionError: If `power` or `freq` evaluate to zero.
        RuntimeError: If the `dipole_current` evaluates to a negative number.
    """

    if 'dbm' == units.lower():
        power = np.float_power(10, power / 10)
        power /= 1e3
    elif 'w' == units.lower():
        if 0 == power:
            raise ZeroDivisionError('Power in absolute units must be > 0')
    else:
        raise RuntimeError('Unsupported power units')

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    if length > (wavelength / 10):
        warnings.warn('Dipole is not electrically small')

    try:
        dipole_current = 40 * np.float_power(np.pi, 2) * \
                         np.float_power(length / wavelength, 2)
        dipole_current = power / dipole_current
    except (ZeroDivisionError, FloatingPointError) as error:
        raise ZeroDivisionError('Dipole length must be > 0'). \
              with_traceback(error.__traceback__)

    dipole_current = np.sqrt(dipole_current)

    if np.isnan(dipole_current):
        raise RuntimeError('Dipole current somehow ended negative')

    return dipole_current


def max_antenna_separation_full(freq: float, radius: float,
                                mode: str = 'normal') -> float:
    """Maximum antenna separation for a given 1st Fresnel Zone

    This function calculates the maximum allowable distance between two
    antennas such that the radius of the first Fresnel zone is less than or
    equal to a specified value.

    Notes:
        1. This is derived from the full equation for the first Fresnel zone
        radius, as opposed to using the approximate formula for large antenna
        separation.
        2. Depending on the combination of input values you might get negative
        separation. While mathematically correct, this does not have any
        physical meaning.

    Args:
        freq: A `float` with the frequency of interest. Units are GHz.
        radius: A `float` with the maximum desired radius of the first
                Fresnel zone. Units are metres.
        mode: A `str` specifying the mode, either 'normal', i.e. we want the
              zone to be 100% free, or 'cheeky', for 60% of the zone.

    Returns:
        A single `float` number with the maximum separation between the two
        antennas, with units in metres.

    Raises:
        ZeroDivisionError: If the frequency has been given as zero.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    # * Assume the pipe radius is 60% of the first Fresnel zone
    # * as opposed to 100%
    if 'cheeky' == mode.lower():
        radius /= 0.6

    separation = 16 * np.float_power(radius, 2) - np.float_power(wavelength, 2)
    separation /= (8 * wavelength)

    return separation


def max_antenna_separation_approx(freq: float, radius: float,
                                  mode: str = 'normal') -> float:
    """Maximum antenna separation for a given 1st Fresnel Zone

    This function calculates the maximum allowable distance between two
    antennas such that the radius of the first Fresnel zone is less than or
    equal to a specified value.

    Notes:
        1. This is derived from the approximate equation for the first Fresnel
        zone radius, which is the better-known and used one. However it assumes
        that the distance between transmitter and receiver is much, much larger
        than the wavelength, e.g. kilometres vs centimetres.

    Args:
        freq: A `float` with the frequency of interest. Units are GHz.
        radius: A `float` with the maximum desired radius of the first
                Fresnel zone. Units are metres.
        mode: A `str` specifying the mode, either 'normal', i.e. we want the
              zone to be 100% free, or 'cheeky', for 60% of the zone.

    Returns:
        A single `float` number with the maximum separation between the two
        antennas, with units in metres.

    Raises:
        ZeroDivisionError: If the frequency has been given as zero.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    # * Assume the pipe radius is 60% of the first Fresnel zone
    # * as opposed to 100%
    if 'cheeky' == mode.lower():
        radius /= 0.6

    separation = 2 * np.float_power(radius, 2)
    separation /= wavelength

    return separation


def fresnel_zone_radius(freq: float, distance_1: float,
                        distance_2: float) -> float:
    """Calculates the radius of the 1st Fresnel zone

    Uses the well-known and used formula to find the radius of the
    first Fresnel zone at a specified point between two antennas.

    Notes:
        1. The assumption is that the distances between the antennas
        and the point of interest are much, much larger than the
        wavelength.
        2. The distances do not have to be the same, i.e. this is valid
        for any point along the length of the wireless link.

    Args:
        freq: A `float` with the frequency of interest. Units are GHz.
        distance_1: A `float` with the distance from the first antenna
                    to the point of interest. Units are in metres.
        distance_2: A `float` with the distance from the second antenna
                    to the point of interest. Units are in metres.

    Returns:
        A single `float` with the radius of the 1st Fresnel zone.
        Units are metres.

    Raises:
        ZeroDivisionError: In case the frequency or both distances are
                           given as zero.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        radius = wavelength * distance_1 * distance_2
        radius /= (distance_1 + distance_2)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Distances must be > 0'). \
              with_traceback(error.__traceback__)

    radius = np.sqrt(radius)

    return radius


def far_field_distance(freq: float, antenna_dimension: float,
                       antenna_type: str = 'array') -> float:
    """Calculates far field boundary for an antenna

    Uses the well-established formula for far field region based on the
    largest physical size of an antenna.

    Notes:
        1. There is support for three common antenna types - monopole, dipole,
        and antenna array with half-wavelength spacing.
        2. The default behaviour assumes an array in which case the antenna
        dimension is the number of elements along x and/or y.
        3. In case of a monopole or dipole the antenna dimension is ignored as
        the antenna size is calculated from the free-space wavelength.
        4. Otherwise the units for the antenna dimension should be metres.

    Args:
        freq: A `float` with the frequency at which the far fiel distance
              is being calculated. Units are GHz.
        antenna_dimension : A `float` with the largest physical size of the
                            antenna. See Notes for further information
                            on units.
        antenna_type : A `str` with the type of antenna. Can be `monopole`,
                       `dipole`, or `array`.

    Returns:
        A single `float` with the minimum far field distance, with units in m.

    Raises:
        ZeroDivisionError: If a frequency of zero is given.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    # * A monopole is assumed to be a quarter-wavelength resonator, and
    # * a dipole to be a half-wavelength resonator
    if 'monopole' == antenna_type.lower():
        dimension = wavelength / 4.0
    elif 'dipole' == antenna_type.lower():
        dimension = wavelength / 2.0
    elif 'array' == antenna_type.lower():
        dimension = antenna_dimension * (wavelength / 2)
    else:
        dimension = antenna_dimension

    distance = 2 * np.float_power(dimension, 2)
    distance /= wavelength

    return distance


def nepers_to_db(nepers: float) -> float:
    return (nepers * 8.685889638)


def db_to_nepers(db: float) -> float:
    return (db * 0.115129255)


def skin_depth(freq: float, conductivity: float,
               real_permeability: float) -> float:
    """Calculates skin depth for a particular metal at a particular frequency

    Uses the well-known formula for metal skin depth, with some additional
    error-checking to prevent runtime errors.

    Args:
        freq: A `float` with the frequency at which we want to know the skin
              depth. Units are GHz.
        conductivity: A `float` with the value for the metal's conductivity.
                      Units are S/m.
        real_permeability: A `float` with the relative permeability of the
                           metal. Unitless.

    Returns:
        A single `float` with the skin depth in metres, due to using base
        units in the function.

    Raises:
        RuntimeError: If for whatever reason one or more of the input
                      variables are negative.
        ZeroDivisionError: If for whatever reason one or more of the input
                           variables are zero.
    """

    # ? Add database of metals and their properties

    freq *= 1e9

    delta = np.pi * freq * conductivity * mu_0 * real_permeability
    delta = np.sqrt(delta)

    if np.isnan(delta):
        raise RuntimeError('All variables must be > 0')

    try:
        delta = 1 / delta
    except (ZeroDivisionError, FloatingPointError) as error:
        raise ZeroDivisionError('Variable values must be > 0'). \
              with_traceback(error.__traceback__)

    return delta


def plane_wave_prop_const(freq: float, real_permittivity: float,
                          imag_permittivity: float,
                          real_permeability: float) -> Tuple[float, float]:
    """Calculate the complex propagation constant in homogeneous medium

    The general-case formula is used to find the attenuation constant in Np/m
    and the phase constant in rad/m of a planar EM wave in homogeneous medium.

    Args:
        freq: A `float` with the frequency of interest. Units are GHz.
        real_permittivity: A `float` with the relative permittivity of the
                           medium. Unitless.
        imag_permittivity: A `float` with the imaginary part of the complex
                           relative pertmittivity of the medium. Unitless.
        real_permeability: A `float` with the relative permeability of the
                           medium. Unitless.

    Returns:
        A `tuple` consisting of `float` values for the attenuation constant
        `alpha` and the phase constant `beta`. Units are Np/m and rad/m.

    Raises:
        ZeroDivisionError: If the real part of the relative permittivity is
                           given as zero.
        RuntimeError: If any of the square root arguments turn out to be
                      negative
    """

    real_permittivity *= epsilon_0
    imag_permittivity *= epsilon_0
    real_permeability *= mu_0

    freq *= 1e9
    ang_freq = 2 * np.pi * freq

    try:
        common_root = (
            1 + np.float_power(imag_permittivity / real_permittivity, 2)
        )
    except (ZeroDivisionError, FloatingPointError) as error:
        raise ZeroDivisionError('Real relative permittivity must be >= 1'). \
              with_traceback(error.__traceback__)

    common_root = np.sqrt(common_root)

    if np.isnan(common_root):
        raise RuntimeError('All variables must be > 0')

    common_multiplier = (real_permeability * real_permittivity) / 2.0

    alpha = ang_freq * np.sqrt(common_multiplier * (common_root - 1))

    if np.isnan(alpha):
        raise RuntimeError('All variables must be > 0')

    beta = ang_freq * np.sqrt(common_multiplier * (common_root + 1))

    if np.isnan(beta):
        raise RuntimeError('All variables must be > 0')

    return (alpha, beta)
