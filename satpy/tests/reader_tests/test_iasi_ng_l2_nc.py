# Copyright (c) 2022 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Unit tests on the IASI NG L2 reader using the conventional mock constructed context."""
import pytest
import logging
import os
from unittest import mock
from satpy.readers.iasi_ng_l2_nc import IASINGL2NCFileHandler

from satpy.readers import load_reader

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler

logger = logging.getLogger(__name__)


class FakeIASINGFileHandlerBase(FakeNetCDF4FileHandler):
    """Fake base class for IASI NG handler"""

    def get_test_content(self, _filename, _filename_info, _filetype_info):
        """Get the content of the test data.

        Here we generate the default content we want to provide depending
        on the provided filename infos.
        """

        dset = {}
        return dset


class TestIASINGL2NCReader:
    """Main test class for the IASI NG L2 reader."""

    reader_name = "iasi_ng_l2_nc"

    def setup_method(self):
        """Setup the reade config"""
        from satpy._config import config_search_paths

        self.reader_configs = config_search_paths(
            os.path.join("readers", self.reader_name + ".yaml")
        )

    @pytest.fixture(autouse=True, scope="class")
    def fake_handler(self):
        """Wrap NetCDF4 FileHandler with our own fake handler."""
        patch_ctx = mock.patch.object(
            IASINGL2NCFileHandler, "__bases__", (FakeIASINGFileHandlerBase,)
        )

        with patch_ctx:
            patch_ctx.is_local = True
            yield patch_ctx

    def _create_file_handler(self, filename):
        """Create an handler for the given file checking that it can
        be parsed correctly"""

        reader = load_reader(self.reader_configs)

        # Test if the file is recognized by the reader
        files = reader.select_files_from_pathnames([filename])
        assert len(files) == 1, "File should be recognized by the reader"

        # Create the file handler:
        reader.create_filehandlers(files)

        # We should have our handler now:
        assert len(reader.file_handlers) == 1

        # logger.info("File handlers are: %s", reader.file_handlers)
        assert self.reader_name in reader.file_handlers

        handlers = reader.file_handlers[self.reader_name]

        # We should have a single handler for a single file:
        assert len(handlers) == 1
        assert isinstance(handlers[0], IASINGL2NCFileHandler)

        return handlers[0]

    def test_filename_matching(self):
        """Test filename matching against some random name"""

        # Example filename
        filename = "W_fr-meteo-sat,GRAL,MTI1-IASING-2-l2p_C_EUMS_20220101120000_LEO_O_D_20220101115425_20220101115728_____W______.nc"

        self._create_file_handler(filename)

    def test_real_filename_matching(self):
        """Test that we will match an actual IASI NG L2 product file name"""

        # Below we test the TWV,CLD,GHG and SFC products:
        filenames = {
            "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-TWV_C_EUMT_20170616120000_G_V_20070912084329_20070912084600_O_N____.nc",
            "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-CLD_C_EUMT_20170616120000_G_V_20070912094037_20070912094308_O_N____.nc",
            "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-GHG_C_EUMT_20170616120000_G_V_20070912090651_20070912090922_O_N____.nc",
            "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-SFC_C_EUMT_20170616120000_G_V_20070912100911_20070912101141_O_N____.nc",
        }

        for filename in filenames:
            self._create_file_handler(filename)
