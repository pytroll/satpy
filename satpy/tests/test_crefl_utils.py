import unittest


class TestCreflUtils(unittest.TestCase):
    """Test crefl_utils."""

    def test_get_atm_variables_abi(self):
        import numpy as np
        from satpy.composites.crefl_utils import get_atm_variables_abi
        sphalb, rhoray, TtotraytH2O, tOG = get_atm_variables_abi(0.17690244, 6.123234e-17, 530.61332168, 405.,
                                                                 0.0043149700000000004, 0.0037296,
                                                                 0.014107995000000002, 0.052349, 21.71342113,
                                                                 77.14385758, 56.21456696)
        if abs(np.array(sphalb) - 0.045213532544630494) >= 1e-10:
            raise AssertionError('{} is not within {} of {}'.format(np.array(sphalb), 1e-10, 0.045213532544630494))
        if abs(rhoray - 2.2030281148621356) >= 1e-10:
            raise AssertionError('{} is not within {} of {}'.format(rhoray, 1e-10, 2.2030281148621356))
        if abs(TtotraytH2O - 0.30309880915889087) >= 1e-10:
            raise AssertionError('{} is not within {} of {}'.format(TtotraytH2O, 1e-10, 0.30309880915889087))
        if abs(tOG - 0.5969089524560548) >= 1e-10:
            raise AssertionError('{} is not within {} of {}'.format(tOG, 1e-10, 0.5969089524560548))


def suite():
    """The test suite for test_crefl_utils."""

    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestCreflUtils))
    return mysuite
