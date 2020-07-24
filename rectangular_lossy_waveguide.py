"""Electrically large lossy rectangular waveguide

This suite of functions implements the approximate modal solutions for a lossy,
air-filled rectangular waveguide embedded in a dielectric medium. The medium
does not have to be homogeneous, i.e. the material along the top and bottom
walls can be different to that along the left and right walls.

The formulas are based on several papers dealing with mining and railway
tunnels. Unlike ITU-R P.2040, these are generalised for higher-order modes.
"""

import warnings
import numpy as np
from scipy.constants import speed_of_light
from aux_funcs import db_to_mag, mag_to_db


np.seterr(divide='raise', invalid='raise')


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
        wvg_width: A `float` with the width of the waveguide which is being
                   checked. Units are metres.
        wvg_height: A `float` with the height of the waveguide which is being
                    checked. Units are metres.
        mode_n: The `n` index of the mode of interest.
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


def mode_phase(mode_idx: int) -> float:
    """Returns mode phase constant

    This function returns the phase constant depending on the mode number.
    Currently this implementation is for vertical polarisation only.

    Args:
        mode_idx: An `int` with the mode number

    Returns:
        Either 0 or pi/2 depending on the mode number

    Raises:
        Nothing
    """

    if (mode_idx & 1):
        return (np.pi / 2)
    else:
        return 0


def calc_mode_weight(wvg_width: float, wvg_height: float,
                     tx_x: float, tx_y: float,
                     rx_x: float, rx_y: float,
                     mode_n: int, mode_m: int) -> float:
    """Calculates the mode eigenfunction

    This function returns the mode eigenfunction, which reflects the influence
    of both the Tx and Rx antenna positions on power distribution within the
    waveguide.

    Notes:
        1. The origin of the x-y coordinate system is the geometric centre of
        rectangular cross-section of the waveguide.

    Args:
        wvg_width: A `float` with the width of the waveguide. Units are metres.
        wvg_height: A `float` with the height of the waveguide. Units are
                    metres.
        tx_x: A `float` with the x coordinate of the transmitter. Units are
              metres.
        tx_y: A `float` with the y coordinate of the transmitter. Units are
              metres.
        rx_x: A `float` with the x coordinate of the receiver. Units are
              metres.
        rx_y: A `float` with the y coordinate of the receiver. Units are
              metres.
        mode_n: An `int` with the mode index along the width of the waveguide.
        mode_m: An `int` with the mode index along the height of the waveguide.

    Returns:
        The eigenfunction value as a `float` number.

    Raises:
        ZeroDivisionError: In case one or both of the waveguide dimensions are
                           given as zero.
    """

    try:
        sin_width = mode_n * np.pi / wvg_width
        sin_height = mode_m * np.pi / wvg_height
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide dimensions must be > 0'). \
              with_traceback(error.__traceback__)

    alpha_nm = np.sin(sin_width * rx_x + mode_phase(mode_n))
    alpha_nm *= np.sin(sin_width * tx_x + mode_phase(mode_n))
    alpha_nm *= np.sin(sin_height * rx_y + mode_phase(mode_m))
    alpha_nm *= np.sin(sin_height * tx_y + mode_phase(mode_m))

    return alpha_nm


def calc_electric_field(freq: float, distance: float,
                        er_width: complex, er_height: complex,
                        wvg_width: float, wvg_height: float,
                        tx_x: float, tx_y: float, rx_x: float, rx_y: float,
                        mode_n_max: int, mode_m_max: int) -> float:
    """Electric field amplitude at a location inside a waveguide

    This function calculates the electric field amplitude at a given position
    inside the waveguide, taking into account contributions from all specified
    propagating modes.

    Notes:
        1. Currently this only considers vertically polarised fields, i.e. the
        electric field is parallel to the height of the waveguide.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        distance: A `float` with how far along the waveguide the receiver is.
                  Units are metres.
        er_width: A `complex` value of the relative permittivity of
                  the material along the top and bottom walls of the waveguide.
        er_height: A `complex` value of the relative permittivity of
                   the material along the side walls of the waveguide.
        wvg_width: A `float` with the width of the waveguide which is being
                   checked. Units are metres.
        wvg_height: A `float` with the height of the waveguide which is being
                    checked. Units are metres.
        tx_x: A `float` with the x coordinate of the transmitter. Units are
              metres.
        tx_y: A `float` with the y coordinate of the transmitter. Units are
              metres.
        rx_x: A `float` with the x coordinate of the receiver. Units are
              metres.
        rx_y: A `float` with the y coordinate of the receiver. Units are
              metres.
        mode_n_max: An `int` with the maximum number of modes to consider
                    along the width of the waveguide.
        mode_m_max: An `int` with the maximum number of modes to consider
                    along the height of the waveguide.

    Returns:
        The electric field amplitude in V/m2 as a `float` number.

    Raises:
        ZeroDivisionError: In case either or both of the waveguide dimensions
                           are given as zero.
    """

    common_multiplier = -1j * 2 * np.pi
    try:
        common_multiplier /= ((wvg_height / 2) * (wvg_width / 2))
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide dimensions must be > 0'). \
              with_traceback(error.__traceback__)

    field = 0.0

    for mode_n in range(1, mode_n_max + 1):
        for mode_m in range(1, mode_m_max + 1):
            mode_weight = calc_mode_weight(
                wvg_width, wvg_height, tx_x, tx_y, rx_x, rx_y, mode_n, mode_m
            )

            mode_alpha = calc_attenuation_constant(
                freq, er_width, er_height, wvg_width, wvg_height,
                mode_n, mode_m
            )

            mode_beta = calc_phase_constant(
                freq, wvg_width, wvg_height, mode_n, mode_m
            )

            mode_gamma = complex(mode_alpha, mode_beta)

            mode_field = np.exp(-(mode_gamma * distance))
            mode_field /= mode_beta
            mode_field *= mode_weight

            field += mode_field

    field *= common_multiplier

    return field


def antenna_insertion_loss(freq: float, wvg_width: float, wvg_height: float,
                           antenna_gain_db: float, antenna_x: float,
                           antenna_y: float) -> float:
    """Calculates antenna inserion loss for a rectangular lossy waveguide

    This function calculates the coupling loss between an antenna placed at a
    specific location in a rectangular lossy waveguide, and the mode that has
    been excited as a result.

    Notes:
        1. The formula is inaccurte when close to the waveguide walls.

    Args:
        freq: A `float` with he frequency at which to calculate the insertion
              loss. Units are GHz.
        wvg_width: A `float` with the width of the waveguide. Units are metres.
        wvg_height: A `float` with the height of the waveguide. Units are
                    metres.
        antenna_gain_db: A `float` with the free-space gain of the antenna.
                         Units are dB.
        antenna_x: A `float` with the x coordinate of the antenna position,
                   relative to the geometric centre of the waveguide
                   cross-section. Units are metres.
        antenna_y: A `float` with the y coordinate of the antenna position,
                   relative to the geometric centre of the waveguide
                   cross-section. Units are metres.
    Returns:
        The insertion loss in dB as a `float` number.

    Raises:
        RuntimeWarning: In case the antenna is placed close to either wall.
        ZeroDivisionError: In case any relevant variable is given as zero.

    """

    if (antenna_x <= (wvg_width / 20)) or (antenna_y <= (wvg_height / 20)):
        warnings.warn('Antenna too close to waveguide walls',
                      category=RuntimeWarning)

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    antenna_gain = db_to_mag(antenna_gain_db)

    loss_1 = 2 * np.pi * wvg_width * wvg_height
    loss_1 /= (antenna_gain * np.float_power(wavelength, 2))

    try:
        loss_cos_1 = np.cos(np.pi * antenna_x / wvg_width)
        loss_cos_2 = np.cos(np.pi * antenna_y / wvg_height)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide dimensions must be > 0'). \
              with_traceback(error.__traceback__)

    loss_cos_1 = np.float_power(loss_cos_1, 2)
    loss_cos_2 = np.float_power(loss_cos_2, 2)

    try:
        loss = loss_1 * (1 / loss_cos_1) * (1 / loss_cos_2)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Bad antenna position'). \
              with_traceback(error.__traceback__)

    loss = mag_to_db(loss)

    return loss
