import unittest


class testCreflUtils(unittest.TestCase):
    """Test crefl_utils."""

    def test_GetATMVariablesABI(self):
        import numpy as np
        from satpy.composites.crefl_utils import get_atm_variables_abi
        sphalb, rhoray, TtotraytH2O, tOG = get_atm_variables_abi(0.17690244, 6.123234e-17, 530.61332168, 405.,
                                                                 (0.0043149700000000004, 0.0037296,
                                                                  0.014107995000000002, 0.052349), 21.71342113,
                                                                 77.14385758, 56.21456696)
        self.assertEqual(np.array(sphalb), 0.045213532544630494)
        self.assertEqual(rhoray, 2.2030281148621356)
        self.assertEqual(TtotraytH2O, 0.30309880915889087)
        self.assertEqual(tOG, 0.5969089524560548)


def suite():
    """The test suite for test_crefl_utils."""

    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(testCreflUtils))
    return mysuite

