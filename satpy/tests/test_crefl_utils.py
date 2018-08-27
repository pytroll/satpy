import unittest


class TestCreflUtils(unittest.TestCase):
    """Test crefl_utils."""

    def test_get_atm_variables_abi(self):
        import numpy as np
        from satpy.composites.crefl_utils import get_atm_variables_abi
        sphalb, rhoray, TtotraytH2O, tOG = get_atm_variables_abi(0.17690244, 6.123234e-17, 530.61332168, 405.,
                                                                 21.71342113, 77.14385758, 56.214566960,
                                                                 0.0043149700000000004, 0.0037296,
                                                                 0.014107995000000002, 0.052349)
        self.assertLess(abs(np.array(sphalb) - 0.045213532544630494), 1e-10)
        self.assertLess(abs(rhoray - 2.2030281148621356), 1e-10)
        self.assertLess(abs(TtotraytH2O - 0.30309880915889087), 1e-10)
        self.assertLess(abs(tOG - 0.5969089524560548), 1e-10)


def suite():
    """The test suite for test_crefl_utils."""

    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestCreflUtils))
    return mysuite
