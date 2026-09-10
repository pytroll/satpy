"""Test CREFL rayleigh correction functions."""
import unittest


class TestCreflUtils(unittest.TestCase):
    """Test crefl_utils."""

    def test_get_atm_variables_abi(self):
        """Test getting atmospheric variables for ABI."""
        import numpy as np

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
