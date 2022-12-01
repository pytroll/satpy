#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""The fy4_base reader tests package."""

from unittest import mock

import pytest

from satpy.readers.fy4_base import FY4Base
from satpy.tests.reader_tests.test_agri_l1 import FakeHDF5FileHandler2


class Test_FY4Base:
    """Tests for the FengYun4 base class for the components missed by AGRI/GHI tests."""

    def setup(self):
        """Initialise the tests."""
        self.p = mock.patch.object(FY4Base, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

        self.file_type = {'file_type': 'agri_l1_0500m'}

    def teardown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_badsensor(self):
        """Test case where we pass a bad sensor name, must be GHI or AGRI."""
        fy4 = FY4Base(None, {'platform_id': 'FY4A', 'instrument': 'FCI'}, self.file_type)
        with pytest.raises(ValueError):
            fy4.calibrate_to_reflectance(None, None, None)
        with pytest.raises(ValueError):
            fy4.calibrate_to_bt(None, None, None)

    def test_badcalibration(self):
        """Test case where we pass a bad calibration type, radiance is not supported."""
        fy4 = FY4Base(None, {'platform_id': 'FY4A', 'instrument': 'AGRI'}, self.file_type)
        with pytest.raises(NotImplementedError):
            fy4.calibrate(None, {'calibration': 'radiance'}, None, None)

    def test_badplatform(self):
        """Test case where we pass a bad calibration type, radiance is not supported."""
        with pytest.raises(KeyError):
            FY4Base(None, {'platform_id': 'FY3D', 'instrument': 'AGRI'}, self.file_type)
