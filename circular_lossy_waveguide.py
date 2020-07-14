"""Electrically large lossy circular waveguide

This suite of functions implements the approximate solutions for a lossy,
air-filled circular waveguide embedded in a homogeneous medium. This solution
is only valid for electrically large tunnels.

The formulas are based on the following paper:

E. A. J. Marcatili and R. A. Schmeltzer,
Hollow metallic and dielectric waveguides for long distance optical
transmission and lasers,
The Bell System Technical Journal, vol. 43, no. 4, pp. 1783–1809, Jul. 1964,
doi: 10.1002/j.1538-7305.1964.tb04108.x.
"""

import warnings
import numpy as np
import scipy.special
from scipy.constants import speed_of_light


np.seterr(divide='raise')


def check_electrical_size(freq: float, wvg_diameter: float,
                          permittivity: complex,
                          mode_n: int, mode_m: int,
                          largeness_factor: int = 10) -> float:
    """Electrical size check for circular waveguide

    Uses the formula in Marcatilli and Schmeltzer's 1964 paper to determine
    if a given circular waveguide is electrically large at a given frequency
    and waveguide propagation mode combination. This is also dependent on the
    complex relative permittivity of the surrounding medium.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_diameter: A `float` with the diameter of the waveguide which
                      is being checked. Units are metres.
        permittivity: A `complex` value of the relative permittivity of
                      the material surrounding the circular waveguide.
        mode_n: The `n` index of the mode of interest
        mode_m: The `m` index of the mode of interest.
        largeness_factor: An `int` with a multiplication factor used to turn
                          the 'much greater than' inequality into a simple
                          'greater than or equal to'. Unitless.

    Returns:
        A single `float` value showing to what extend the waveguide is large
        electrically compared to the wavelength.

    Raises:
        ZeroDivisionError: In case the `freq` parameter is given as zero.
        ValueError: In case the `permittivity` is specified as a negative
                    real number.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    check_value_1 = (np.pi * wvg_diameter) / wavelength

    refr_index = np.sqrt(permittivity)

    if np.isnan(refr_index):
        raise ValueError('Material permittivity cannot be real and < 0')

    # ! The `jn_zeros` function returns a list of length `mode_m`, but we are
    # ! only interested in the mth zero, i.e. index mode_m-1
    bessel_root = scipy.special.jn_zeros(mode_n-1, mode_m)[mode_m-1]

    check_value_2 = np.abs(refr_index) * bessel_root
    check_value_2 *= largeness_factor

    try:
        check_result = check_value_1 / check_value_2
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Largeness factor must be > 0'). \
              with_traceback(error.__traceback__)

    return check_result


def calc_mode_refr_index(permittivity: complex, mode: str) -> complex:
    """Helper function for propagation constant calculation

    This is a helper function that calculates a mode-specific refractive
    index, something necessary to determine the attenuation and phase
    constants of a particular mode in a lossy circular waveguide.

    Args:
        permittivity: A `complex` value of the relative permittivity of
                      the material surrounding the circular waveguide.
        mode: A `str` specifying what type of mode is propagating along the
              waveguide. Valid values are TE, TM, HE, or EH.

    Returns:
        A `complex` number with the mode-specific refractive index.

    Raises:
        ValueError: In case the `permittivity` is specified as a negative
                    real number.
        ZeroDivisionError: In case the `permittivity` is that of free space,
                           i.e. 1 + 0j.
    """

    common_root = permittivity - 1
    common_root = np.sqrt(common_root)

    if np.isnan(common_root):
        raise ValueError('Soil permittivity cannot be real and < 0')

    if 'tm' == mode.lower():
        numerator = permittivity
    elif 'te' == mode.lower():
        numerator = 1
    elif ('he' == mode.lower()) or ('eh' == mode.lower()):
        numerator = (permittivity + 1) / 2
    else:
        raise ValueError('Mode must be TE, TM, EH, or HE')

    try:
        mode_refr_index = numerator / common_root
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Surrounding material cannot be free space'). \
              with_traceback(error.__traceback__)

    return mode_refr_index


def calc_attenuation_constant(freq: float, wvg_diameter: float,
                              permittivity: complex,
                              mode: str, mode_n: int, mode_m: int,
                              largeness_factor: int = 10) -> float:
    """Calculate attenuation constant of electrically large circular waveguide

    This function calculates the attenuation constant for a particular mode
    in an electrically large circular waveguide. It uses other functions
    from this module internally.

    Note:
        1. This function does not do any error handling, this is done in the
        other functions in the module. In case of an error there this function
        will simply re-raise the error.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_diameter: A `float` with the diameter of the waveguide which
                      is being checked. Units are metres.
        permittivity: A `complex` value of the relative permittivity of
                      the material surrounding the circular waveguide.
        mode: A `str` specifying what type of mode is propagating along the
              waveguide. Valid values are TE, TM, HE, or EH.
        mode_n: The `n` index of the mode of interest
        mode_m: The `m` index of the mode of interest.
        largeness_factor: An `int` with a multiplication factor used to turn
                          the 'much greater than' inequality into a simple
                          'greater than or equal to'. Unitless.

    Returns:
        The attenuation rate in Np/m as a `float` number.

    Raises:
        RuntimeWarning: In case the waveguide is not electrically large.
    """

    elec_size_check = check_electrical_size(freq, wvg_diameter, permittivity,
                                            mode_n, mode_m, largeness_factor)
    if elec_size_check <= 1.0:
        warnings.warn('Waveguide is not electrically large',
                      category=RuntimeWarning)

    mode_refr_index = calc_mode_refr_index(permittivity, mode)
    mode_refr_index = mode_refr_index.real

    freq *= 1e9
    wavelength = speed_of_light / freq

    # ! The `jn_zeros` function returns a list of length `mode_m`, but we are
    # ! only interested in the mth zero, i.e. index mode_m-1
    bessel_root = scipy.special.jn_zeros(mode_n - 1, mode_m)[mode_m - 1]

    alpha_1 = np.float_power(bessel_root / (2 * np.pi), 2)

    alpha_2 = np.float_power(wavelength, 2)
    alpha_2 /= np.float_power(wvg_diameter / 2, 3)

    alpha = alpha_1 * alpha_2 * mode_refr_index

    return alpha


def calc_phase_constant(freq: float, wvg_diameter: float,
                        permittivity: complex,
                        mode: str, mode_n: int, mode_m: int,
                        largeness_factor: int = 10) -> float:
    """Calculate phase constant of large circular waveguide

    This function calculates the phase constant for a particular mode
    in an electrically large circular waveguide. It uses other functions
    from this module internally.

    Note:
        1. This function does not do any error handling, this is done in the
        other functions in the module. In case of an error there this function
        will simply re-raise the error.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_diameter: A `float` with the diameter of the waveguide which
                      is being checked. Units are metres.
        permittivity: A `complex` value of the relative permittivity of
                      the material surrounding the circular waveguide.
        mode: A `str` specifying what type of mode is propagating along the
              waveguide. Valid values are TE, TM, HE, or EH.
        mode_n: The `n` index of the mode of interest
        mode_m: The `m` index of the mode of interest.
        largeness_factor: An `int` with a multiplication factor used to turn
                          the 'much greater than' inequality into a simple
                          'greater than or equal to'. Unitless.

    Returns:
        The phase constant in rad/m as a `float` number.

    Raises:
        RuntimeWarning: In case the waveguide is not electrically large.
    """

    elec_size_check = check_electrical_size(freq, wvg_diameter, permittivity,
                                            mode_n, mode_m, largeness_factor)
    if elec_size_check <= 1.0:
        warnings.warn('Waveguide is not electrically large',
                      category=RuntimeWarning)

    mode_refr_index = calc_mode_refr_index(permittivity, mode)

    freq *= 1e9
    wavelength = speed_of_light / freq

    # ! The `jn_zeros` function returns a list of length `mode_m`, but we are
    # ! only interested in the mth zero, i.e. index mode_m-1
    bessel_root = scipy.special.jn_zeros(mode_n - 1, mode_m)[mode_m - 1]

    beta_0 = 2 * np.pi / wavelength

    beta_1 = mode_refr_index * wavelength / (np.pi * wvg_diameter / 2)
    beta_1 = beta_1.imag + 1

    beta_2 = bessel_root * wavelength / (np.pi * wvg_diameter)
    beta_2 = (np.float_power(beta_2, 2)) / 2

    beta = beta_0 * (1 - beta_2 * beta_1)

    return beta
