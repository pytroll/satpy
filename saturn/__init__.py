#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of the mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Saturn Package initializer.
"""

import logging
import os

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep)[:-1])

class NullHandler(logging.Handler):
    """Not-doing-anything handler for logging, so as to let the user choose if
    logging should be enabled.
    """
    def emit(self, record):
        """Do no write anything.
        """
        pass

LOG = logging.getLogger("saturn").addHandler(NullHandler())

