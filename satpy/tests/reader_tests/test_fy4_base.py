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

import pytest

from satpy.readers.fy4_base import FY4Base


class Test_FY4Base:
    """Tests for the FengYun4 base class for the components missed by AGRI/GHI tests."""
    def setup(self):
        self.fy4 = FY4Base
        self.fy4.sensor = 'Bad'

    def test_badsensor(self):
        """Test case where we pass a bad sensor name, must be GHI or AGRI."""
        with pytest.raises(ValueError):
            self.fy4.calibrate_to_reflectance(self.fy4, None, None, None)
        with pytest.raises(ValueError):
            self.fy4.calibrate_to_bt(self.fy4, None, None, None)

    def test_badcalibration(self):
        """Test case where we pass a bad calibration type, radiance is not supported."""
        with pytest.raises(NotImplementedError):
            self.fy4.calibrate(self.fy4, None, {'calibration': 'radiance'}, None, None)
