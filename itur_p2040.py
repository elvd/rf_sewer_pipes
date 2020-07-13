"""Implementation of some of the ITU-R P.2040 formulas

ITU-R Recommendation P.2040-1 (07/2015) Effects of Building Materials and
Structures on Radiowave Propagation Above About 100 MHz contains a lot of
data and formulas on EM properties of materials, building entry loss, and
EM propagation within hallways.

Currently the main implemeneted functionality is the calculation of relative
permittivity and conductivity for the materials listed in the Recommendation.
The generic formulas for the loss within a dielectric waveguide are implemented
with some limitations - the same material is assumed on all sides. The formulas
for extra loss in the case of rectangular waveguide are also implemented.

Other functionality includes a few functions to check if a waveguide is large
compared to the wavelength. Finally there are a couple of functions to compute
losses in homogeneous materials and through a slab of material.

Note:
    Unless specified otherwise, *ALL* functions here assume a vertically-
    polarised HE11 mode.
"""

import warnings
from collections import namedtuple
from typing import Tuple
import numpy as np
from scipy.constants import speed_of_light


np.seterr(divide='raise')


Material = namedtuple(
    'Material', ['a', 'b', 'c', 'd', 'freq_min', 'freq_max']
)

WaveguideShape = namedtuple(
    'WaveguideShape', ['horizontal', 'vertical']
)

MATERIAL_CONSTANTS = {
    'vacuum': Material(1, 0, 0, 0, 0.001, 100),
    'concrete': Material(5.31, 0, 0.0326, 0.8095, 1, 100),
    'brick': Material(3.75, 0, 0.038, 0, 1, 10),
    'plasterboard': Material(2.94, 0, 0.0116, 0.7076, 1, 100),
    'wood': Material(1.99, 0, 0.0047, 1.0718, 0.001, 100),
    'glass': Material(6.27, 0, 0.0043, 1.1925, 0.1, 100),
    'ceiling_board': Material(1.50, 0, 0.0005, 1.1634, 1, 100),
    'chipboard': Material(2.58, 0, 0.0217, 0.78, 1, 100),
    'floorboard': Material(3.66, 0, 0.0044, 1.3515, 50, 100),
    'metal': Material(1, 0, 10e7, 0, 1, 100),
    'very_dry_ground': Material(3, 0, 0.00015, 2.52, 1, 10),
    'medium_dry_ground': Material(15, -0.1, 0.035, 1.63, 1, 10),
    'wet_ground': Material(30, -0.4, 0.15, 1.3, 1, 10)
}

SHAPE_CONSTANTS = {
    'circle': WaveguideShape(5.09, 5.09),
    'ellipse': WaveguideShape(4.45, 4.40),
    'square': WaveguideShape(4.34, 4.34),
    'arch_backed': WaveguideShape(5.13, 5.09)
}


def material_permittivity(freq: float = 1.0,
                          material_name: str = 'vacuum') -> float:
    """Calculates real part of complex permittivity

    Uses the aggregate formulas in ITU-R Recommendation P.2040-1 for the
    real part of the complex relative permittivity of several common
    building materials. More specifically, it uses Equation (57) on
    page 24 of the 2015 revision of the Recommendation.

    Args:
        freq: A `float` with the frequency at which the permittivity should be
              calculated. Units are GHz.
        material_name: A `str` identifying the material. Has to be one of the
                       materials in the Recommendation, and also present in the
                      `MATERIAL_CONSTANTS`, defined in this module.

    Returns:
        A single `float` number with the real part of the relative permittivity
        of the specified material at the specified frequency.

    Raises:
        KeyError: If a material that is not included in the Recommendation
                  and/or in this module.
        RuntimeWarning: If a frequency outside the validity range, as defined
                        in the Recommendation, is requested.
    """

    try:
        material = MATERIAL_CONSTANTS[material_name]
    except KeyError as error:
        raise KeyError('Material not found in database'). \
              with_traceback(error.__traceback__)

    if (material.freq_min > freq) or (material.freq_max < freq):
        warnings.warn('Frequency specified outside range of vailidity',
                      category=RuntimeWarning)

    permittivity = material.a * (freq ** material.b)

    return permittivity


def material_conductivity(freq: float = 1.0,
                          material_name: str = 'vacuum') -> float:
    """Calculates conductivity of a material

    Uses the aggregate formulas in ITU-R Recommendation P.2040-1 for the
    conductivity of several common building materials. More specifically,
    it uses Equation (58) on page 24 of the 2015 revision of the
    Recommendation.

    Args:
        freq: A `float` with the frequency at which the conductivity should be
              calculated. Units are GHz.
        material_name: A `str` identifying the material. Has to be one of the
                       materials in the Recommendation, and also present in the
                       `MATERIAL_CONSTANTS`, defined in this module.

    Returns:
        A single `float` number with the conductivity of the specified material
        at the specified frequency. The units are S/m, which are identical to
        the old style mho/m.

    Raises:
        KeyError: If a material that is not included in the Recommendation
                  and/or in this module.
        RuntimeWarning: If a frequency outside the validity range, as defined
                        in the Recommendation, is requested.
    """

    try:
        material = MATERIAL_CONSTANTS[material_name]
    except KeyError as error:
        raise KeyError('Material not found in database'). \
              with_traceback(error.__traceback__)

    if (material.freq_min > freq) or (material.freq_max < freq):
        warnings.warn('Frequency specified outside range of vailidity',
                      category=RuntimeWarning)

    conductivity = material.c * (freq ** material.d)

    return conductivity


def dielectric_wvg_loss(freq: float, width: float, height: float,
                        real_permittivity: float,
                        imaginary_permittivity: float,
                        polarisation: str = 'vertical',
                        shape: str = 'square') -> float:
    """Calculate loss in dB/m in a dielectric waveguide

    Uses the methodology in ITU-R Recommendation P.2040-1 to calculate the
    loss per unit length (dB/m) for an air-filled waveguide with dielectric
    walls. This is based off measurements in buildings, so applicability to
    sewer pipes is untested. The methodology supports waveguides of different
    shapes, and calculates the propagation loss separately for horizontal
    and vertical polarisation.

    Notes:
        1. While the methodology allows for different dielectrics on left/right
        and top/bottom, this implementation assumes a single, homogenous one.
        2. The additional losses for the square case are not implemented.
        3. The frequency range of validity for this model is 0.2 GHz to 12 GHz.
        4. The imaginary part of the complex relative permittivity has a
        negative sign applied internally in this function.

    Args:
        freq: A `float` with the frequency at which to calculate the loss.
              Units are GHz and are converted to Hz internally.
        width: A `float` with the width, i.e. dimension parallel to the Earth,
               units are metres.
        height: A `float` with the height, i.e. dimension normal to the Earth,
                units are metres.
        real_permittivity: A `float` with the real part of the complex relative
                           permittivity.
        imaginary_permittivity: A `float` with the imaginary part of the
                                complex relative permittivity.
        polarisation: A `str` with the polarisation of the propagating wave.
                      Has to be `vertical` or `horizontal`.
        shape: A `str` with the shape of the waveguide. Has to be one defined
               in the ITU Recommendation.

    Returns:
        A single `float` number representing the calculated loss per unit
        length, in dB/m, at the specified frequency for the specified
        waveguide.

    Raises:
        KeyError: If the waveguide shape specified is not in the Recommendation
        AttributeError: If a polarisation different than horizontal or vertical
                        is specified
        ZeroDivisionError: If you specify 0 Hz, i.e. DC, for the frequency
        RuntimeError: Negative square root argument or failed checks
        RuntimeWarning: If the dimensions of the waveguide are too small when
                        compared to the wavelength; or if the frequency is out-
                        side the validity range.
    """

    try:
        wvg_shape_constant = SHAPE_CONSTANTS[shape]
        wvg_shape_constant = getattr(wvg_shape_constant, polarisation)
    except KeyError as error:
        raise KeyError('Waveguide shape not found in database'). \
              with_traceback(error.__traceback__)
    except AttributeError as error:
        raise AttributeError('Polarisation must be horizontal or vertical'). \
              with_traceback(error.__traceback__)

    if (freq < 0.2) or (freq > 12.0):
        warnings.warn('Frequency outside of range of validity',
                      category=RuntimeWarning)

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        if check_electrical_size(wavelength, width, height, real_permittivity):
            warnings.warn('Dielectric waveguide is not electrically large',
                          category=RuntimeWarning)
    except (ZeroDivisionError, RuntimeError) as error:
        raise RuntimeError('Check values and units of input parameters'). \
              with_traceback(error.__traceback__)

    permittivity = complex(real_permittivity, -imaginary_permittivity)

    real_denum_1 = np.float_power(width, 3) * np.sqrt(permittivity - 1)
    real_denum_2 = np.float_power(height, 3) * np.sqrt(permittivity - 1)

    if np.isnan(real_denum_1) or np.isnan(real_denum_2):
        raise RuntimeError('Square root argument negative')

    imag_denum_1 = np.float_power(width, 4) * (permittivity - 1)
    imag_denum_2 = np.float_power(height, 4) * (permittivity - 1)

    try:
        if 'horizontal' == polarisation.lower():
            real_part = permittivity / real_denum_1 + 1 / real_denum_2
            imag_part = (
                np.float_power(np.abs(permittivity), 2) / imag_denum_1
                + 1 / imag_denum_2
            )

        elif 'vertical' == polarisation.lower():
            real_part = 1 / real_denum_1 + permittivity / real_denum_2
            imag_part = (
                np.float_power(np.abs(permittivity), 2) / imag_denum_2 +
                1 / imag_denum_1
            )
        else:
            raise RuntimeError('The code should never get to here')
    except (ZeroDivisionError, FloatingPointError) as error:
        raise ZeroDivisionError('Values of parameters must be > 0'). \
              with_traceback(error.__traceback__)

    norm_wavelength = wavelength / (2 * np.pi)
    loss_per_m = wvg_shape_constant * np.float_power(wavelength, 2)
    loss_per_m *= (np.real(real_part) - norm_wavelength * np.imag(imag_part))

    return loss_per_m


def rough_walls_loss(freq: float, width: float, height: float,
                     roughness_width: float, roughness_height: float,
                     polarisation: str) -> float:
    """Calculate additional losses from rough walls

    These are additional losses specified in ITU-R P.2040 for the special case
    of a square/rectangular waveguide. They are valid only in that case and as
    such this formula does not take an argument for the shape of a waveguide.

    Notes:
        There is no check for an electrically large waveguide, it is assumed
        this has been performed previously, by either calling
        `check_electrical_size()` or by running `dielectric_wvg_loss()` first.

    Args:
        freq: A `float` with the frequency at which to calculate the loss.
              Units are GHz.
        width: A `float` with the width of the waveguide. Units are metres.
        height: A `float` with the height of the waveguide. Units are metres.
        roughness_width: A `float` with the RMS of the surface roughness of
                         the top and bottom walls of the waveguide.
        roughness_height: A `float` with the RMS of the surface roughness of
                          the sidewalls of the waveguide.
        polarisation: A `str` with the polarisation of the propagating wave.
                      Has to be `vertical` or `horizontal`.

    Returns:
        A single `float` number with the additional loss due to rough walls.
        Units are dB/m.

    Raises:
        AttributeError: If a polarisation different than horizontal or vertical
                        is specified
        ZeroDivisionError: If you specify 0 Hz, i.e. DC, for the frequency
        RuntimeError: Negative square root argument or failed checks
        RuntimeWarning: If the frequency is outside the validity range.
    """

    wvg_shape_constant = SHAPE_CONSTANTS['square']

    try:
        wvg_shape_constant = getattr(wvg_shape_constant, polarisation)
    except AttributeError as error:
        raise AttributeError('Polarisation must be horizontal or vertical'). \
              with_traceback(error.__traceback__)

    if (freq < 0.2) or (freq > 12.0):
        warnings.warn('Frequency outside of range of validity',
                      category=RuntimeWarning)

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        width_factor = np.float_power(
            (roughness_width / np.float_power(width, 2)), 2
        )
        height_facotr = np.float_power(
            (roughness_height / np.float_power(height, 2)), 2
        )
    except (ZeroDivisionError, FloatingPointError) as error:
        raise ZeroDivisionError('Waveguide dimensions must be > 0'). \
              with_traceback(error.__traceback__)

    loss_per_m = width_factor + height_facotr
    loss_per_m *= (wvg_shape_constant * np.float_power(np.pi, 2) * wavelength)

    return loss_per_m


def tilt_angle_loss(freq: float, tilt_angle: float,
                    polarisation: str = 'vertical') -> float:
    """Calculate additional losses from tilt angle

    These are additional losses specified in ITU-R P.2040 for the special case
    of a square/rectangular waveguide. They are valid only in that case and as
    such this formula does not take an argument for the shape of a waveguide.

    Notes:
        There is no check for an electrically large waveguide, it is assumed
        this has been performed previously, by either calling
        `check_electrical_size()` or by running `dielectric_wvg_loss()` first.

    Args:
        freq: A `float` with the frequency at which to calculate the loss.
              Units are GHz.
        tilt_angle: A `float` with the tilt angle. Units are degrees.
        polarisation: A `str` with the polarisation of the propagating wave.
                      Has to be `vertical` or `horizontal`.

    Returns:
        A single `float` number with the additional loss due to tilt angle.
        Units are dB/m.

    Raises:
        AttributeError: If a polarisation different than horizontal or vertical
                        is specified
        ZeroDivisionError: If you specify 0 Hz, i.e. DC, for the frequency
        RuntimeWarning: If the frequency is outside the validity range.
    """

    wvg_shape_constant = SHAPE_CONSTANTS['square']

    try:
        wvg_shape_constant = getattr(wvg_shape_constant, polarisation)
    except AttributeError as error:
        raise AttributeError('Polarisation must be horizontal or vertical'). \
              with_traceback(error.__traceback__)

    if (freq < 0.2) or (freq > 12.0):
        warnings.warn('Frequency outside of range of validity',
                      category=RuntimeWarning)

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    loss_per_m = np.float_power((np.pi * tilt_angle), 2) / wavelength
    loss_per_m *= wvg_shape_constant

    return loss_per_m


def check_electrical_size(wavelength: float,
                          width: float, height: float,
                          real_permittivity: float,
                          largeness_factor: int = 10) -> bool:
    """Check if the waveguide is electrically large - ITU-R P.2040 main

    The formulas in ITU-R P.2040 are only valid under certain conditions, which
    generally mean that the dielectric waveguide is electrically large. There
    are several ways to quantify that, with this function implementing the
    one given in the Recommendation.

    Notes:
        1. This check assumes a rectangular waveguide cross-section, however
        a circular one can be approximated by setting the `width` equal to the
        `height` variable.
        2. This check assumes the waveguide is embedded in a homogeneous
        medium, i.e. uses the same relative permittivity for all sides.

    Args:
        wavelength: A `float` with the free-space wavelength corresponding to
                    the frequency of interest. Units should be the same as
                    those for the `width` and `height`.
        width: A `float` with the width of the dielectric waveguide. Units
               should be the same as those for the `height` and `wavelength`.
        height: A `float` with the height of the dielectric waveguide. Units
                should be the same as those for the `width` and `wavelength`.
        real_permittivity: A `float` with the real part of the complex relative
                           permittivity of the surrounding medium. Unitless.
        largeness_factor: An `int` used to turn the "much greather than"
                          condition in the standard to a regular inequality.

    Returns:
        `True` or `False` depending on whether the checks pass and the
        waveguide cross-section is electrically large.

    Raises:
        ZeroDivisionError: If the real part of the permittivity is specified
                           as zero.
        RuntimeError: If the square root argument evaluates to a
                      negative number.
    """

    try:
        check_value_width = (
            (np.pi * width * np.sqrt(real_permittivity - 1)) /
            real_permittivity
        )

        check_value_height = (
            np.pi * height * np.sqrt(real_permittivity - 1)
        )
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Permittivity must be >= 1'). \
              with_traceback(error.__traceback__)

    if np.isnan(check_value_height) or np.isnan(check_value_width):
        raise RuntimeError('Permittivity must be >= 1')

    try:
        check_value_width /= largeness_factor
        check_value_height /= largeness_factor
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Largeness factor must be > 0'). \
              with_traceback(error.__traceback__)

    check_result = (wavelength > check_value_width) and \
                   (wavelength > check_value_height)

    return check_result


def check_electrical_size_wavelength(freq: float,
                                     wvg_dimension: float,
                                     largeness_factor: int = 10) -> float:
    """Check if the waveguide is electrically large - no material

    The formulas in `check_electrical_size` have an implicit condition
    for the relationship between a waveguide/tunnel dimension and the
    free-space wavelength. This function extracts that and makes it
    explicit. It is also independent of the relative permittivity of the
    surrounding material.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_dimension: A `float` with the dimension of the waveguide which
                       is being checked. Units are metres.
        largeness_factor: An `int` with a multiplication factor used to turn
                          the 'much greater than' inequality into a simple
                          'greater than or equal to'. Unitless.

    Returns:
        A single `float` value showing to what extent the waveguide is large
        electrically compared to the wavelength.

    Raises:
        ZeroDivisionError: In case either the `freq` or `largeness_factor`
                           parameters are given as zero.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        check_result = (wvg_dimension * np.pi) / (2 * largeness_factor)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Largeness factor must be > 0'). \
              with_traceback(error.__traceback__)

    check_result /= wavelength

    return check_result


def relative_permittivity_bounds(freq: float, wvg_dimension: float,
                                 largeness_factor: int) -> Tuple[float, float]:
    """Find minimum and maximum allowed permittivity for surrounding medium

    The real part of the relative permittivity of the surrounding medium has
    to fall within certain bounds. There are two checks in the ITU-R document,
    one of which results in a quadratic equation with two possible solutions.
    This function finds these solutions.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        wvg_dimension: A `float` with the dimension of the waveguide which
                       is being checked. Units are metres.
        largeness_factor: An `int` with a multiplication factor used to turn
                          the 'much greater than' inequality into a simple
                          'greater than or equal to'. Unitless.

    Returns:
        A `Tuple` with two `float` values, corresponding to the minimum and
        maximum real relative permittivity that the surrounding medium
        can have.

    Raises:
        ZeroDivisionError: In case either the `freq` or `largeness_factor`
                           parameters are given as zero.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        t = (largeness_factor * wavelength) / (np.pi * wvg_dimension)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide dimension must be > 0'). \
              with_traceback(error.__traceback__)

    t = np.float_power(t, 2)
    determinant = np.sqrt(1 - 4 * t)

    try:
        permittivity_1 = (1 + determinant) / (2 * t)
        permittivity_2 = (1 - determinant) / (2 * t)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Largeness factor must be > 0'). \
              with_traceback(error.__traceback__)

    return (permittivity_1, permittivity_2)


def check_material_relative_permittivity(freq: float, width: float,
                                         height: float, largeness_factor: int,
                                         real_permittivity: float) -> bool:
    """Performs the main check for medium permittivity in ITU-R P.2040

    There are two conditions in the ITU-R P.2040 for the value of the real
    part of the complex relative permittivity. This functions performs
    these checks.

    Notes:
        1. This apperas to be specifically for the case of a square/rectangular
        waveguide. While it can be used for other shapes with a square or
        rectangular approximation, the results may not be accurate.
        2. This implementation assumes that the surrounding material is
        homogeneous.

    Args:
        freq: A `float` with the frequency at which to perform the check.
              Units are GHz.
        width: A `float` with the width of the waveguide. Units are metres.
        height: A `float` with the height of the waveguide. Units are metres.
        largeness_factor: An `int` with a multiplication factor used to turn
                          the 'much greater than' inequality into a simple
                          'greater than or equal to'. Unitless.
        real_permittivity: A `float` with the real part of the complex relative
                           permittivity of the surrounding medium.

    Returns:
        `True` or `False` depending on whether the checks pass.

    Raises:
        ZeroDivisionError: In case the frequency or waveguide dimensions are
                           given as zero
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    try:
        t_1 = (largeness_factor * wavelength) / (np.pi * width)
        t_2 = (largeness_factor * wavelength) / (np.pi * height)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Waveguide dimensions must be > 0'). \
              with_traceback(error.__traceback__)

    e_r_sqrt = np.sqrt(real_permittivity - 1)

    check_1 = (e_r_sqrt / real_permittivity) > t_1
    check_2 = e_r_sqrt > t_2

    check_result = check_1 and check_2

    return check_result


def material_attenuation_rate(freq: float, real_permittivity: float,
                              loss_tangent: float) -> float:
    """Calculates EM wave attenuation in a homogeneous material

    Uses the formulas given in ITU-R P.2040 to calculate the attenuation
    constant for homogeneous material.

    Notes:
        1. These are approximate/practical formulas and should not be used
        for precise calculations.
        2. These are only valid in two cases, a dielectric and good conductor.
        The distinction is based on the value of the loss tangent, either
        < 0.5 for a dielectric or > 15 for a good conductor.

    Args:
        freq: A `float` with the frequency at which to calculate the
              attenuation constant. Units are GHz.
        real_permittivity: A `float` with the real part of the complex
                           relative permittivity of the material of
                           interest.
        loss_tangent: A `float` with the loss tangent, or tan delta, of
                      the material of interest.

    Returns:
        A single `float` with the attenuation constant, in dB/m.

    Raises:
        ZeroDivisionError: In case the frequency is given as zero.
        RuntimeError: In case the material cannot be categorised as
                      a dielectric or a good conductor.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    wave_number = 2 * np.pi / wavelength

    foo = 1 / (wave_number * np.sqrt(real_permittivity))
    bar = 2 / loss_tangent

    if loss_tangent > 15:
        delta_distance = foo * np.sqrt(bar)
    elif loss_tangent < 0.5:
        delta_distance = foo * bar
    else:
        raise RuntimeError('Material properties outside validity region')

    attenuation = 8.686 / delta_distance

    return attenuation


def single_layer_slab_coefficients(freq: float, thickness: float,
                                   permittivity: complex, angle: float,
                                   polarisation: str = 'TE') -> Tuple[complex,
                                                                      complex]:
    """Calculate the reflection and transmission for a single-layer slab

    ITU-R P.2040 provides a way to calculate the reflection and transmission
    coefficients for an EM wave travelling through air and encountering a slab
    of homogeneous material. The slab is of finite thickness, with air on
    either side of it.

    Args:
        freq: A `float` with the frequency of interes. Units are GHz.
        thickness: A `float` with the thickness of the slab. Units are metres.
        permittivity: A `complex` number with the relative permittivity of the
                      material of which the slab is made.
        angle: A `float` number with the angle at which the EM wave is
               impinging the slab. Units are degrees.
        polarisation: A `str` with the polarisation of the impinging EM wave.
                      Valid values are either TE or TM.

    Returns:
        A `Tuple` with two `complex` values, representing the reflection
        and transmission coefficients, respectively.

    Raises:
        ZeroDivisionError: In case the frequency is given as zero.
        RuntimeError: In case an unsupported polarisation is given.
    """

    freq *= 1e9

    try:
        wavelength = speed_of_light / freq
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency must be > 0'). \
              with_traceback(error.__traceback__)

    angle = np.deg2rad(angle)
    common_root = np.sqrt(permittivity - np.float_power(np.sin(angle), 2))

    if 'te' == polarisation.lower():
        refl_coeff_prime = np.cos(angle) - common_root
        refl_coeff_prime /= (np.cos(angle) + common_root)
    elif 'tm' == polarisation.lower():
        refl_coeff_prime = permittivity * np.cos(angle) - common_root
        refl_coeff_prime /= (permittivity * np.cos(angle) + common_root)
    else:
        raise RuntimeError('Polarisation must be TE or TM')

    q_coeff = ((2 * np.pi * thickness) / wavelength) * common_root

    q_coeff_exp_2 = np.exp(-1j * 2 * q_coeff)
    q_coeff_exp_1 = np.exp(-1j * q_coeff)

    refl_coeff = refl_coeff_prime * (1 - q_coeff_exp_2)
    refl_coeff /= (1 - np.float_power(refl_coeff_prime, 2) * q_coeff_exp_2)

    tx_coeff = (1 - np.float_power(refl_coeff_prime, 2)) * q_coeff_exp_1
    tx_coeff /= (1 - np.float_power(refl_coeff_prime, 2) * q_coeff_exp_2)

    return (refl_coeff, tx_coeff)
