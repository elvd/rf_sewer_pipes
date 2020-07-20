"""Electrically large lossy rectangular waveguide

This suite of functions implements the approximate modal solutions for a lossy,
air-filled rectangular waveguide embedded in a dielectric medium. The medium
does not have to be homogeneous, i.e. the material along the top and bottom
walls can be different to that along the left and right walls.

The formulas are based on several papers dealing with mining and railway
tunnels. Unlike ITU-R P.2040, these are generalised for higher-order modes.
"""

import numpy as np
from scipy.constants import speed_of_light


np.seterr(divide='raise')


def check_electrical_size(freq: float, wvg_dimension: float, mode_idx: int,
                          largeness_factor: int) -> float:
    """Electrical size check for a rectangular waveguide

    This check is used to evaluate to what extent the lossy rectangular
    waveguide is electrically large. The check does not explicitly depend
    on the value of permittivity of the surrounding media, even though there
    is a limit to said value.

    Notes:
        1. This check should be invoked separately for the two waveguide
        dimensions, i.e. the height and the width.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_dimension: A `float` with the dimension of the waveguide which
                       is being checked. Units are metres.
        mode_idx: The modal index along the dimension of the waveguide which
                  is being checked.
        largeness_factor: An `int` with a multiplication factor used to turn
                          the 'much greater than' inequality into a simple
                          'greater than or equal to'. Unitless.

    Returns:
        A single `float` value showing to what extend the waveguide is large
        electrically compared to the wavelength.

    Raises:
        ZeroDivisionError: In case the `freq` or the `wvg_dimension parameters
                           are given as zero.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        check_result = (mode_idx * wavelength) / (2 * wvg_dimension)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide dimension must be > 0'). \
              with_traceback(error.__traceback__)

    check_result *= largeness_factor
    check_result = 1 / check_result

    return check_result


def calc_attenuation_constant(freq: float, permittivity_width: complex,
                              permittivity_height: complex, wvg_width: float,
                              wvg_height: float, mode_n: int, mode_m: int,
                              polarisation: str = 'vertical') -> float:
    """Calculate the attenuation constant of a lossy rectangular waveguide

    This functionc calculates the attenuation constant for a particular mode
    with a given polarisation in an electrically large rectangular waveguide.

    Notes:
        1. This function does not cuurently do any checks for electrical size.
        2. `Vertical` polarisation refers to a Y-polarised mode, i.e. the `E`
        vector is oriented parallel to the side walls.
        3. `Horizontal` polarisation refers to an X-polarised mode, i.e. the
        `E` vector is oriented parallel to the top and bottom walls.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        permittivity_width: A `complex` value of the relative permittivity of
                            the material along the top and bottom walls of the
                            waveguide.
        permittivity_height: A `complex` value of the relative permittivity of
                             the material along the side walls of the
                             waveguide.
        wvg_height: A `float` with the height of the waveguide which is being
                    checked. Units are metres.
        wvg_width: A `float` with the width of the waveguide which is being
                   checked. Units are metres.
        mode_n: The `n` index of the mode of interest
        mode_m: The `m` index of the mode of interest.
        polarisation: A `str` specifying the polarisation of the mode of
                      interest. Valid values are `horizontal` or `vertical`.

    Returns:
        The attenuation rate in Np/m as a `float` number.

    Raises:
        ZeroDivisionError: In case any variable is given as zero.
        RuntimeError: In case an invalid polarisation is given.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    e_r_width_sqrt = np.sqrt(permittivity_width - 1)
    e_r_height_sqrt = np.sqrt(permittivity_height - 1)

    try:
        factor_width = (mode_n * wavelength) / (2 * wvg_width)
        factor_width = (2 / wvg_width) * np.float_power(factor_width, 2)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide width must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        factor_height = (mode_m * wavelength) / (2 * wvg_height)
        factor_height = (2 / wvg_height) * np.float_power(factor_height, 2)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide height must be > 0'). \
              with_traceback(error.__traceback__)

    if 'vertical' == polarisation.lower():
        alpha_width = factor_width * np.real(1 / e_r_width_sqrt)

        alpha_height = (
            factor_height * np.real(permittivity_height / e_r_height_sqrt)
        )
    elif 'horizontal' == polarisation.lower():
        alpha_width = (
            factor_width * np.real(permittivity_width / e_r_width_sqrt)
        )

        alpha_height = factor_height * np.real(1 / e_r_height_sqrt)
    else:
        raise RuntimeError('Polarisation must be horizontal or vertical')

    alpha = alpha_width + alpha_height

    return alpha


def calc_phase_constant(freq: float, wvg_width: float, wvg_height: float,
                        mode_n: int, mode_m: int) -> float:
    """Calculate the phase constant of a lossy rectangular waveguide

    This function calculates the phase constant for a particular mode in an
    electrically large rectangular waveguide.

    Notes:
        1. This function does not cuurently do any checks for electrical size.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_height: A `float` with the height of the waveguide which is being
                    checked. Units are metres.
        wvg_width: A `float` with the width of the waveguide which is being
                   checked. Units are metres.
        mode_n: The `n` index of the mode of interest
        mode_m: The `m` index of the mode of interest.

    Returns:
        The phase constant in rad/m as a `float` number.

    Raises:
        ZeroDivisionError: In case any variable is given as zero.
        RuntimeError: In case an invalid polarisation is given.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    beta_0 = 2 * np.pi / wavelength

    try:
        beta_width = (mode_n * wavelength) / (2 * wvg_width)
        beta_width = 0.5 * np.float_power(beta_width, 2)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide width must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        beta_height = (mode_m * wavelength) / (2 * wvg_height)
        beta_height = 0.5 * np.float_power(beta_height, 2)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide height must be > 0'). \
              with_traceback(error.__traceback__)

    beta = beta_0 * (1 - beta_width - beta_height)

    return beta


def calc_phase_constant_alt(freq: float, wvg_width: float, wvg_height: float,
                            mode_n: int, mode_m: int) -> float:
    """
        This is an alternative implementation of `calc_phase_constant`. Once
        they have been demonstrated to yield the same results, this function
        will be removed.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    beta_0 = 2 * np.pi / wavelength
    beta_0 = np.float_power(beta_0, 2)

    beta_1 = (mode_n * np.pi) / (wvg_width)
    beta_1 = np.float_power(beta_1, 2)

    beta_2 = (mode_m * np.pi) / (wvg_height)
    beta_2 = np.float_power(beta_2, 2)

    beta = beta_0 - beta_1 - beta_2
    beta = np.sqrt(beta)

    return beta
