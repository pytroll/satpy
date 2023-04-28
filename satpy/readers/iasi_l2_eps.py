# Copyright (c) 2023 Satpy developers
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

"""Reader for EPS level 2 IASI data. Uses xml files as a format description."""

from .eps_base import EPSFile


class EPSIASIFile(EPSFile):
    """EPS level 2 reader for IASI data.

    Reader for the IASI Level 2 combined sounding products in native format.

    Overview of the data including links to the product user guide, product format
    specification, validation reports, and other documents, can be found at the
    EUMETSAT Data Services at https://data.eumetsat.int/product/EO:EUM:DAT:METOP:IASSND02

    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialise Filehandler."""
        raise NotImplementedError()
