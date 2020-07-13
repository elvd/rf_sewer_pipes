"""Implementation of some of the ITU-R P.527 formulas

ITU-R Recommendation P.527-4 (06/2017) Electrical Characteristics of the
Surface of the Earth gives formulas for the calculation of the frequency
dependent complex relative permittivity of various parts of the surface of
the Earth.

Currently, only the formulas for pure water and soil are implemented, as they
are the only ones of interest right now. Ice, vegetation, sea water might be
added later.

There is a dictionary containing some example soil mixtures in terms of sand,
clay, and silt percentages, however the functions are desinged in such a way
to allow people to specify their own mixtures.
"""

import warnings
from collections import namedtuple
import numpy as np


np.seterr(divide='raise')


Soil = namedtuple(
    'Soil', ['p_sand', 'p_clay', 'p_silt']
)

SOILS = {
    'clay': Soil(20.0, 60.0, 20.0),
    'sandy_clay': Soil(50.0, 40.0, 10.0),
    'silty_clay': Soil(10.0, 45.0, 45.0),
    'clay_loam': Soil(35.0, 30.0, 35.0),
    'sandy_clay_loam': Soil(60.0, 25.0, 15.0),
    'silty_clay_loam': Soil(15.0, 32.5, 52.5),
    'loam': Soil(40.0, 20.0, 40.0),
    'silty_loam': Soil(22.5, 15.0, 62.5),
    'sandy_loam': Soil(65.0, 10.0, 25.0),
    'sand': Soil(90.0, 5.0, 5.0),
    'loamy_sand': Soil(80.0, 10.0, 10.0),
    'silt': Soil(10.0, 10.0, 80.0)
}


def pure_water_permittivity(freq: float,
                            temperature: float) -> complex:
    """Calculate complex relative permittivity of pure water

    Uses the methodology described in ITU-R Recommendation P.527-4 to
    calculate the complex relative permittivity of pure water at a
    given frequency and at a particular temperature

    Args:
        freq: A `float` with the frequency of interest. Units are GHz.
        temperature: A `float` with the temperature. Units are degrees
                     Celsius.

    Returns:
        A complex number of the form `e_real - j * e_imag`. Please note the
        imaginary part has not got the negative sign applied.

    Raises:
        Nothing
    """

    try:
        theta = 300 / (temperature + 273.15) - 1
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Temperature must be > 0 K'). \
              with_traceback(error.__traceback__)

    debye_relax_freq_1 = (
        20.20 - 146.4 * theta + 316 * np.float_power(theta, 2)
    )
    debye_relax_freq_2 = 39.8 * debye_relax_freq_1

    epsilon_inf = 3.52 - 7.52 * theta
    epsilon_static = 77.66 + 103.3 * theta
    epsilon_pole = 0.0671 * epsilon_static

    try:
        denum_freq_1 = 1 + np.float_power(freq / debye_relax_freq_1, 2)
        denum_freq_2 = 1 + np.float_power(freq / debye_relax_freq_2, 2)
    except (ZeroDivisionError, FloatingPointError) as error:
        raise ZeroDivisionError('Check temperature value'). \
              with_traceback(error.__traceback__)

    epsilon_pw_real = (epsilon_static - epsilon_pole) / denum_freq_1
    epsilon_pw_real += (epsilon_pole - epsilon_inf) / denum_freq_2
    epsilon_pw_real += epsilon_inf

    epsilon_pw_imag = (
        (freq / debye_relax_freq_1) * (epsilon_static - epsilon_pole) /
        denum_freq_1
    )
    epsilon_pw_imag += (
        (freq / debye_relax_freq_2) * (epsilon_pole - epsilon_inf) /
        denum_freq_2
    )

    epsilon_pw = complex(epsilon_pw_real, epsilon_pw_imag)

    return epsilon_pw


def soil_permittivity(freq: float, temperature: float,
                      p_sand: float, p_clay: float, p_silt: float,
                      water_vol: float,
                      rho_s: float = 2.65) -> complex:
    """Calculate complex relative permittivity of soil

    Uses the methodology described in ITU-R Recommendation P.527-4 to calculate
    the complex relative permittivity of a specific soil at a specific
    frequency and at a specific temperature. The soil is characterised by a
    mixture of sand, clay, and silt, as well as the volumetric water content.

    Args:
        freq: A `float` with the frequency of interest. Units are GHz.
        temperature: A `float` with the temperature at which the soil is at.
                     Units are degree Celsius.
        p_sand: A `float` with the percentage of sand in the soil
        p_clay: A `float` with the percentage of clay in the soil
        p_silt: A `float` with the percentage of silt in the soil
        water_vol: A `float` with the volumetric water content, as a ratio
        rho_s: A `float` with the specific gravity of the dry mixture of
               soil constituents. Dependent on soil composition, normally
               between 2.5 and 2.7.

    Returns:
        A complex number of the form `e_real - j * e_imag`. Please note the
        imaginary part has not got the negative sign applied.

    Raises:
        RuntimeWarning: If the percentage of any of sand, clay, or silt are
                        less than 1%
        RuntimeError: If the sum of all percentages is greater than 100, or
                      a percentage is negative
        ZeroDivisionError: If a percentage is zero
    """

    if not np.isclose((p_sand + p_clay + p_silt), 100):
        raise RuntimeError('The constituent percentages must sum to 100')

    if (p_sand < 1.0):
        warnings.warn('Sand percentage too low', category=RuntimeWarning)
    if (p_clay < 1.0):
        warnings.warn('Clay percentage too low', category=RuntimeWarning)
    if (p_silt < 1.0):
        warnings.warn('Silt percentage too low', category=RuntimeWarning)

    try:
        rho_b = (1.072560 +
                 0.078886 * np.log(p_sand) +
                 0.038753 * np.log(p_clay) +
                 0.032732 * np.log(p_silt))
    except (ZeroDivisionError, FloatingPointError) as error:
        raise ZeroDivisionError('Percentage must be > 0'). \
              with_traceback(error.__traceback__)

    if np.isnan(rho_b):
        raise RuntimeError('Percentage cannot be negative')

    sigma_1 = (0.0467 + 0.2204 * rho_b - 0.004111 * p_sand -
               0.006614 * p_clay)
    sigma_2 = (-1.645 + 1.939 * rho_b - 0.0225622 * p_sand +
               0.01594 * p_clay)

    sigma_common = (sigma_1 - sigma_2) / (1 + np.float_power(freq / 1.35, 2))
    sigma_eff_prime = sigma_common * (freq / 1.35)
    sigma_eff_second = sigma_common + sigma_2

    try:
        epsilon_fw_corr = (rho_s - rho_b) / (rho_s * water_vol)
        epsilon_fw_corr *= (18 / freq)
    except ZeroDivisionError as error:
        raise ZeroDivisionError('Frequency and water volume must be > 0'). \
              with_traceback(error.__traceback__)

    epsilon_pw = pure_water_permittivity(freq, temperature)
    epsilon_fw_real = epsilon_pw.real + (sigma_eff_prime * epsilon_fw_corr)
    epsilon_fw_imag = epsilon_pw.imag + (sigma_eff_second * epsilon_fw_corr)

    alpha = 0.65

    beta_prime = 1.2748 - 0.00519 * p_sand - 0.00152 * p_clay
    beta_second = 1.33797 - 0.00603 * p_sand - 0.00166 * p_clay

    epsilon_sm = np.float_power(1.01 + 0.44 * rho_s, 2) - 0.062

    # ! This is needed for an edge case where we need to raise a negative
    # ! number to a fractional power. This is not supported by NumPy, neither
    # ! in `np.power` nor in `np.float_power`, so we need to transition to
    # ! built-in Python data types and mathematical operations.
    epsilon_sm = float(epsilon_sm)
    epsilon_fw_real = float(epsilon_fw_real)
    epsilon_fw_imag = float(epsilon_fw_imag)

    epsilon_soil_imag = (water_vol ** beta_second)
    epsilon_soil_imag *= (epsilon_fw_imag ** alpha)
    epsilon_soil_imag = (epsilon_soil_imag ** (1 / alpha))

    epsilon_soil_real = 1 - water_vol
    epsilon_soil_real += (
        (water_vol ** beta_prime) * (epsilon_fw_real ** alpha)
    )
    epsilon_soil_real += (
        (rho_b / rho_s) * ((epsilon_sm ** alpha) - 1)
    )
    epsilon_soil_real = (epsilon_soil_real ** (1 / alpha))

    epsilon_soil = complex(epsilon_soil_real, epsilon_soil_imag)

    return epsilon_soil
