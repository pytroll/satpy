"""Test CREFL rayleigh correction functions."""
import unittest

import numpy as np
import pytest


class TestCreflUtils(unittest.TestCase):
    """Test crefl_utils."""

    def test_get_atm_variables_abi(self):
        """Test getting atmospheric variables for ABI."""
        from satpy.modifiers._crefl_utils import _ABIAtmosphereVariables
        atm_vars = _ABIAtmosphereVariables(
            21.71342113, 77.14385758, 56.214566960,
            0.17690244, 6.123234e-17, 530.61332168, 405.,
            0.0043149700000000004, 0.0037296, 0.014107995000000002, 0.052349,
        )
        sphalb, rhoray, TtotraytH2O, tOG = atm_vars()
        assert abs(np.array(sphalb) - 0.045213532544630494) < 1e-10
        assert abs(rhoray - 2.2030281148621356) < 1e-10
        assert abs(TtotraytH2O - 0.30309880915889087) < 1e-10
        assert abs(tOG - 0.5969089524560548) < 1e-10


# METimage band name, wavelength range, matching MODIS band, LUT index
METIMAGE_TO_MODIS_BANDS = [
    ("vii_668", (0.658, 0.668, 0.678), 1, 0),
    ("vii_865", (0.855, 0.865, 0.875), 2, 1),
    ("vii_443", (0.428, 0.443, 0.458), 3, 2),
    ("vii_555", (0.545, 0.555, 0.565), 4, 3),
    ("vii_1240", (1.230, 1.240, 1.250), 5, 4),
    ("vii_1630", (1.620, 1.630, 1.640), 6, 5),
    ("vii_2250", (2.225, 2.250, 2.275), 7, 6),
]


@pytest.mark.parametrize(("vii_name", "vii_wavelength", "modis_band", "lut_index"), METIMAGE_TO_MODIS_BANDS)
def test_metimage_coefficients_match_modis(vii_name, vii_wavelength, modis_band, lut_index):
    """Test that each METimage band gets the coefficients of its matching MODIS band."""
    from satpy.modifiers._crefl_utils import _METimageCoefficients, _MODISCoefficients

    exp_coeffs = _MODISCoefficients(str(modis_band), 1000)()
    np.testing.assert_allclose(_METimageCoefficients(vii_wavelength, 500)(), exp_coeffs)
    np.testing.assert_allclose(_METimageCoefficients(vii_name, 500)(), exp_coeffs)
    np.testing.assert_allclose(exp_coeffs, [lut_array[lut_index] for lut_array in _MODISCoefficients.LUTS])


@pytest.mark.parametrize("sensor", ["seviri", {"viirs", "modis"}])
def test_runner_class_for_unknown_sensor(sensor):
    """Test that an unsupported sensor produces a useful error."""
    from satpy.modifiers._crefl_utils import _runner_class_for_sensor

    with pytest.raises(NotImplementedError, match="Don't know how to apply CREFL"):
        _runner_class_for_sensor(sensor)


@pytest.mark.parametrize("sensor", ["metimage", {"metimage"}, frozenset({"METimage"})])
def test_runner_class_for_metimage(sensor):
    """Test that the sensor name may be a string or a one-element collection."""
    from satpy.modifiers._crefl_utils import _METimageCREFLRunner, _runner_class_for_sensor

    assert _runner_class_for_sensor(sensor) is _METimageCREFLRunner
