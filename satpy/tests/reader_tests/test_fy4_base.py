"""The fy4_base reader tests package."""

from unittest import mock

import pytest

from satpy.readers.core.fy4 import FY4Base
from satpy.tests.reader_tests.test_agri_l1 import FakeHDF5FileHandler2


class Test_FY4Base:
    """Tests for the FengYun4 base class for the components missed by AGRI/GHI tests."""

    def setup_method(self):
        """Initialise the tests."""
        self.p = mock.patch.object(FY4Base, "__bases__", (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

        self.file_type = {"file_type": "agri_l1_0500m"}

    def teardown_method(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_badsensor(self):
        """Test case where we pass a bad sensor name, must be GHI or AGRI."""
        fy4 = FY4Base(None, {"platform_id": "FY4A", "instrument": "FCI"}, self.file_type)
        with pytest.raises(ValueError, match="Unsupported sensor type: FCI"):
            fy4.calibrate_to_reflectance(None, None, None)
        with pytest.raises(ValueError, match="Error, sensor must be GHI or AGRI."):
            fy4.calibrate_to_bt(None, None, None)

    def test_badcalibration(self):
        """Test case where we pass a bad calibration type, radiance is not supported."""
        fy4 = FY4Base(None, {"platform_id": "FY4A", "instrument": "AGRI"}, self.file_type)
        with pytest.raises(NotImplementedError):
            fy4.calibrate(None, {"calibration": "radiance"}, None, None)

    def test_badplatform(self):
        """Test case where we pass a bad calibration type, radiance is not supported."""
        with pytest.raises(KeyError):
            FY4Base(None, {"platform_id": "FY3D", "instrument": "AGRI"}, self.file_type)
