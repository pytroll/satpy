import os
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pytest

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.tests.utils import convert_file_content_to_data_array
from satpy.readers import load_reader

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(
    DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1], dtype=DEFAULT_FILE_DTYPE
).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeNetCDF4FileHandlerVIIRSL2(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        dt = filename_info.get("start_time", datetime(2023, 12, 30, 22, 30, 0))
        file_type = filename[:6]
        num_lines = DEFAULT_FILE_SHAPE[0]
        num_pixels = DEFAULT_FILE_SHAPE[1]
        num_scans = 5
        file_content = {
            "/dimension/number_of_scans": num_scans,
            "/dimension/number_of_lines": num_lines,
            "/dimension/number_of_pixels": num_pixels,
            "/attr/time_coverage_start": dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "/attr/time_coverage_end": (dt + timedelta(minutes=6)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            ),
            "/attr/orbit_number": 26384,
            "/attr/instrument": "VIIRS",
            "/attr/platform": "Suomi-NPP",
        }
        self._fill_contents_with_default_data(file_content, file_type)
        convert_file_content_to_data_array(file_content)
        return file_content

    def _fill_contents_with_default_data(self, file_content, file_type):
        """Fill file contents with default data."""
        if file_type.startswith("CLD"):
            file_content["geolocation_data/latitude"] = DEFAULT_LAT_DATA
            file_content["geolocation_data/longitude"] = DEFAULT_LON_DATA
            if file_type == "CLDPRO":
                file_content["geophysical_data/Cloud_Top_Height"] = DEFAULT_FILE_DATA
            elif file_type == "CLDMSK":
                file_content[
                    "geophysical_data/Clear_Sky_Confidence"
                ] = DEFAULT_FILE_DATA
        elif file_type == "AERDB_":
            file_content["Latitude"] = DEFAULT_LAT_DATA
            file_content["Longitude"] = DEFAULT_LON_DATA
            file_content["Angstrom_Exponent_Land_Ocean"] = DEFAULT_FILE_DATA
            file_content["Aerosol_Optical_Thickness_550_Land_Ocean"] = DEFAULT_FILE_DATA


class TestVIIRSL2FileHandler:
    """Test VIIRS_L2 Reader"""

    yaml_file = "viirs_l2.yaml"

    def setup_method(self):
        """Wrap NetCDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.viirs_l2 import VIIRSL2FileHandler

        self.reader_configs = config_search_paths(
            os.path.join("readers", self.yaml_file)
        )
        self.p = mock.patch.object(
            VIIRSL2FileHandler, "__bases__", (FakeNetCDF4FileHandlerVIIRSL2,)
        )
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def teardown_method(self):
        """Stop wrapping the NetCDF4 file handler."""
        self.p.stop()

    @pytest.mark.parametrize(
        "filename",
        [
            ("CLDPROP_L2_VIIRS_SNPP.A2023364.2230.011.2023365115856.nc"),
            ("CLDMSK_L2_VIIRS_SNPP.A2023364.2230.001.2023365105952.nc"),
            ("AERDB_L2_VIIRS_SNPP.A2023364.2230.011.2023365113427.nc"),
        ],
    )
    def test_init(self, filename):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader

        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([filename])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_load_aerdb(self):
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames(
            ["AERDB_L2_VIIRS_SNPP.A2023364.2230.011.2023365113427.nc"]
        )
        r.create_filehandlers(loadables)
        datasets = r.load(
            ["Aerosol_Optical_Thickness_550_Land_Ocean", "Angstrom_Exponent_Land_Ocean"]
        )
        assert len(datasets) == 2
        for d in datasets.values():
            assert d.shape == DEFAULT_FILE_SHAPE
            assert d.dims == ("y", "x")
            assert d.attrs["sensor"] == "viirs"

    def test_load_cldprop(self):
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames(
            ["CLDPROP_L2_VIIRS_SNPP.A2023364.2230.011.2023365115856.nc"]
        )
        r.create_filehandlers(loadables)
        datasets = r.load(["Cloud_Top_Height"])
        assert len(datasets) == 1
        for d in datasets.values():
            assert d.shape == DEFAULT_FILE_SHAPE
            assert d.dims == ("y", "x")
            assert d.attrs["sensor"] == "viirs"

    def test_load_cldmsk(self):
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames(
            ["CLDMSK_L2_VIIRS_SNPP.A2023364.2230.001.2023365105952.nc"]
        )
        r.create_filehandlers(loadables)
        datasets = r.load(["Clear_Sky_Confidence"])
        assert len(datasets) == 1
        for d in datasets.values():
            assert d.shape == DEFAULT_FILE_SHAPE
            assert d.dims == ("y", "x")
            assert d.attrs["sensor"] == "viirs"
