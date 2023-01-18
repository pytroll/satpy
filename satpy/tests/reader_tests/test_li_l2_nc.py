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
"""Unit tests on the LI L2 reader using the conventional mock constructed context."""
import os
from datetime import datetime
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from pyproj import Proj

from satpy._config import config_search_paths
from satpy.readers.li_base_nc import LINCFileHandler
from satpy.readers.li_l2_nc import LI_GRID_SHAPE, LIL2NCFileHandler
from satpy.readers.yaml_reader import load_yaml_configs
from satpy.tests.reader_tests._li_test_utils import (
    FakeLIFileHandlerBase,
    extract_filetype_info,
    get_product_schema,
    products_dict,
)
from satpy.tests.utils import make_dataid


@pytest.fixture(name="filetype_infos")
def std_filetype_infos():
    """Return standard filetype info for LI L2."""
    cpaths = config_search_paths(os.path.join("readers", "li_l2_nc.yaml"))

    cfg = load_yaml_configs(cpaths[0])

    # get the li_l2 filetype:
    ftypes = cfg['file_types']

    yield ftypes


# Note: the helper class below has some missing abstract class implementation,
# but that is not critical to us, so ignoring them for now.
class TestLIL2():
    """Main test class for the LI L2 reader."""

    @pytest.fixture(autouse=True, scope="class")
    def fake_handler(self):
        """Wrap NetCDF4 FileHandler with our own fake handler."""
        patch_ctx = mock.patch.object(
            LINCFileHandler,
            "__bases__",
            (FakeLIFileHandlerBase,))

        with patch_ctx:
            patch_ctx.is_local = True
            yield patch_ctx

    def _test_dataset_single_variable(self, vname, desc, settings, handler):
        """Check the validity of a given variable."""
        dname = vname

        dims = settings.get('dimensions', {})

        var_path = settings.get('variable_path', '')

        # Compute shape from dimensions:
        if desc['shape'] == ():
            # scalar case, dim should have been added in the code by validate_array_dimensions
            shape = (1,)
        else:
            shape = tuple([dims[dim_name] for dim_name in desc['shape']])

        dataset_info = {
            'name': dname,
            'variable_name': vname,
            'use_rescaling': False,
        }
        var_params = [dataset_info, desc, dname, handler, shape, var_path]
        self._test_dataset_variable(var_params)

    def _test_dataset_variables(self, settings, ds_desc, handler):
        """Check the loading of the non in sector variables."""
        assert 'variables' in ds_desc
        all_vars = ds_desc['variables']

        variables = settings.get('variables')
        for vname, desc in variables.items():
            # variable should be in list of dataset:
            assert vname in all_vars
            self._test_dataset_single_variable(vname, desc, settings, handler)

    def _test_dataset_single_sector_variable(self, names, desc, settings, handler):
        """Check the validity of a given sector variable."""
        sname, vname = names[0], names[1]

        dname = f"{vname}_{sname}_sector"

        dims = settings.get('dimensions', {})

        var_path = settings.get('variable_path', '')

        shape = tuple([dims[dim_name] for dim_name in desc['shape']])

        dataset_info = {
            'name': dname,
            'variable_name': vname,
            'sector_name': sname,
            'use_rescaling': False,
        }
        var_params = [dataset_info, desc, vname, handler, shape, var_path]
        self._test_dataset_variable(var_params, sname=sname)

    def _test_dataset_variable(self, var_params, sname=""):
        """Test the validity of a given (sector) variable."""
        dataset_info, desc, dname, handler, shape, var_path = var_params
        res = self.get_variable_dataset(dataset_info, dname, handler)
        assert res.shape == shape
        assert res.dims[0] == 'y'
        # Should retrieve content with fullname key:
        full_name = self.create_fullname_key(desc, var_path, dname, sname=sname)
        # Note: 'content' is not recognized as a valid member of the class below
        # since it is silently injected in from our patching fake base netcdf4 file handler class.
        # But for now, we don't need to actually extend the class itself as this is only
        # needed for testing.
        assert np.all(res.values == handler.content[full_name])  # pylint: disable=no-member

    def get_variable_dataset(self, dataset_info, dname, handler):
        """Get the dataset of a given (sector) variable."""
        dataset_id = make_dataid(name=dname)
        res = handler.get_dataset(dataset_id, dataset_info)
        return res

    def create_fullname_key(self, desc, var_path, vname, sname=''):
        """Create full name key for sector/non-sector content retrieval."""
        vpath = desc.get('path', var_path)
        if vpath != "" and vpath[-1] != '/':
            vpath += '/'
        if sname != "":
            sname += '/'
        full_name = f"{vpath}{sname}{vname}"
        return full_name

    def _test_dataset_sector_variables(self, settings, ds_desc, handler):
        """Check the loading of the in sector variables."""
        sector_vars = settings.get('sector_variables')
        sectors = settings.get('sectors', ['north', 'east', 'south', 'west'])

        assert 'sector_variables' in ds_desc
        all_vars = ds_desc['sector_variables']

        for sname in sectors:
            for vname, desc in sector_vars.items():
                # variable should be in list of dataset:
                assert vname in all_vars
                self._test_dataset_single_sector_variable([sname, vname], desc, settings, handler)

    def test_dataset_loading(self, filetype_infos):
        """Test loading of all datasets from all products."""
        # Iterate on all the available product types:
        for ptype, pinfo in products_dict.items():
            ftype = pinfo['ftype']
            filename_info = {
                'start_time': "0000",
                'end_time': "1000"
            }

            handler = LIL2NCFileHandler('filename', filename_info, extract_filetype_info(filetype_infos, ftype))
            ds_desc = handler.ds_desc

            # retrieve the schema that what used to generate the content for that product:
            settings = get_product_schema(ptype)

            # Now we check all the variables are available:
            if 'variables' in settings:
                self._test_dataset_variables(settings, ds_desc, handler)

                # check the sector variables:
            if 'sector_variables' in settings:
                self._test_dataset_sector_variables(settings, ds_desc, handler)

    def test_unregistered_dataset_loading(self, filetype_infos):
        """Test loading of an unregistered dataset."""
        # Iterate on all the available product types:

        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_af_nc'))

        dataset_id = make_dataid(name='test_dataset')
        with pytest.raises(KeyError):
            handler.get_dataset(dataset_id)

    def test_dataset_not_in_provided_dataset(self, filetype_infos):
        """Test loading of a dataset that is not provided."""
        # Iterate on all the available product types:

        dataset_dict = {'name': 'test_dataset'}

        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_af_nc'))

        dataset_id = make_dataid(name='test_dataset')

        assert handler.get_dataset(dataset_id, ds_info=dataset_dict) is None

    def test_filename_infos(self, filetype_infos):
        """Test settings retrieved from filename."""
        filename_info = {
            'start_time': "20101112131415",
            'end_time': "20101112131416"
        }

        handler = LIL2NCFileHandler('filename', filename_info, extract_filetype_info(filetype_infos, 'li_l2_af_nc'))

        # Start and end time should come from filename info:
        assert handler.start_time == "20101112131415"
        assert handler.end_time == "20101112131416"

        # internal vars should be initialized:
        assert handler.search_paths is not None
        assert handler.dataset_infos is not None

        # calling register_available_datasets again should not change things (early return)
        ds_infos_current = handler.dataset_infos.copy()
        handler.register_available_datasets()
        assert handler.dataset_infos == ds_infos_current

        # Should have some datasets:
        assert len(handler.provided_datasets) > 0

        # Sensor names should be just 'li'
        assert handler.sensor_names == {'li'}

        # check product type:
        assert handler.product_type == '2-AF'

    def test_var_path_exists(self, filetype_infos):
        """Test variable_path_exists from li reader."""
        filename_info = {
            'start_time': "20101112131415",
            'end_time': "20101112131416",
        }

        handler = LIL2NCFileHandler('filename', filename_info, extract_filetype_info(filetype_infos, 'li_l2_lef_nc'))

        # Check variable paths:
        assert handler.variable_path_exists("dummy") is False

        assert handler.variable_path_exists("state/processor/l1b_geolocation_warning") is False
        assert handler.variable_path_exists("data/l1b_geolocation_warning") is True
        assert handler.variable_path_exists("data/north/event_id") is True
        assert handler.variable_path_exists("data/none/event_id") is False
        assert handler.variable_path_exists("/attr") is False
        assert handler.variable_path_exists("data/l1b_geolocation_warning/dtype") is False
        assert handler.variable_path_exists("data/l1b_geolocation_warning/shape") is False
        assert handler.variable_path_exists("data/l1b_geolocation_warning/dimensions") is False

    def test_get_first_valid_variable(self, filetype_infos):
        """Test get_first_valid_variable from li reader."""
        filename_info = {
            'start_time': "20101112131415",
            'end_time': "20101112131416",
        }

        handler = LIL2NCFileHandler('filename', filename_info, extract_filetype_info(filetype_infos, 'li_l2_lef_nc'))

        # Check variable paths:
        var1 = handler.get_first_valid_variable(["dummy/path", "data/north/event_id"])
        var2 = handler.get_first_valid_variable(["dummy/path", "data/east/event_id"])
        var3 = handler.get_first_valid_variable(["dummy/path", "data/south/group_id"])
        var4 = handler.get_first_valid_variable(["dummy/path", "data/west/group_id"])

        assert isinstance(var1, xr.DataArray)
        assert isinstance(var2, xr.DataArray)
        assert isinstance(var3, xr.DataArray)
        assert isinstance(var4, xr.DataArray)

        assert id(var1) != id(var2)
        assert id(var2) != id(var3)
        assert id(var3) != id(var4)

        mix1 = handler.get_first_valid_variable(["dummy/path",
                                                 "data/north/event_id",
                                                 "data/east/event_id",
                                                 "data/south/group_id"])

        mix2 = handler.get_first_valid_variable(["dummy/path",
                                                 "data/west/group_id",
                                                 "data/north/event_id",
                                                 "data/east/event_id",
                                                 "data/south/group_id"])

        # first mix should give us var1 and the second one var4:
        assert id(mix1) == id(var1)
        assert id(mix2) == id(var4)

        # get the measured variables now:
        # Note that we must specify fill_value==None below otherwise
        # a new array is generated filling the invalid values:
        meas1 = handler.get_measured_variable("east/event_id", fill_value=None)
        meas2 = handler.get_measured_variable("south/group_id", fill_value=None)

        assert id(meas1) == id(var2)
        assert id(meas2) == id(var3)

        # We should have a fill value on those variables:
        assert var1.attrs.get('_FillValue') == 65535
        assert var2.attrs.get('_FillValue') == 65535

    def test_get_first_valid_variable_not_found(self, filetype_infos):
        """Test get_first_valid_variable from li reader if the variable is not found."""
        filename_info = {
            'start_time': "20101112131415",
            'end_time': "20101112131416",
        }

        handler = LIL2NCFileHandler('filename', filename_info, extract_filetype_info(filetype_infos, 'li_l2_lef_nc'))

        with pytest.raises(KeyError):
            handler.get_first_valid_variable(["dummy/path", "data/test/test_var"])

    def test_available_datasets(self, filetype_infos):
        """Test available_datasets from li reader."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_lef_nc'))

        # get current ds_infos. These should all be returned by the available_datasets
        ds_infos_to_compare = handler.dataset_infos.copy()

        # now add a dummy configured dataset to make sure that it is included in the available_datasets output
        ds_info_dummy = {'test': 'test'}
        conf_ds_dummy = [(True, ds_info_dummy)]
        ds_infos_to_compare.insert(0, ds_info_dummy)

        assert ds_infos_to_compare == [ds[1] for ds in handler.available_datasets(configured_datasets=conf_ds_dummy)]

    def test_variable_scaling(self, filetype_infos):
        """Test automatic rescaling with offset and scale attributes."""
        filename_info = {
            'start_time': "20101112131415",
            'end_time': "20101112131416"
        }

        handler = LIL2NCFileHandler('filename', filename_info, extract_filetype_info(filetype_infos, 'li_l2_lfl_nc'))

        # Get the raw variable without rescaling:
        vname = "latitude"
        rawlat = handler.get_measured_variable(vname)

        # Get the dataset without rescaling:
        dataset_info = {
            'name': vname,
            'variable_name': vname,
            'use_rescaling': False,
        }

        dataset_id = make_dataid(name=vname)
        lat_noscale = handler.get_dataset(dataset_id, dataset_info)
        assert np.all(lat_noscale.values == rawlat)

        # Now get the dataset with scaling:
        dataset_info['use_rescaling'] = True
        lat_scaled = handler.get_dataset(dataset_id, dataset_info)

        # By default we write data in the ranges [-88.3/0.0027, 88.3/0.0027] for latitude and longitude:
        assert abs(np.nanmax(lat_scaled.values) - 88.3) < 1e-2
        assert abs(np.nanmin(lat_scaled.values) + 88.3) < 1e-2

    def test_swath_coordinates(self, filetype_infos):
        """Test that swath coordinates are used correctly to assign coordinates to some datasets."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_lfl_nc'))

        # Check latitude:
        dsid = make_dataid(name="latitude")
        dset = handler.get_dataset(dsid)
        assert 'coordinates' not in dset.attrs

        # get_area_def should raise exception:
        with pytest.raises(NotImplementedError):
            handler.get_area_def(dsid)

        # Check radiance:
        dsid = make_dataid(name="radiance")
        dset = handler.get_dataset(dsid)
        assert 'coordinates' in dset.attrs
        assert dset.attrs['coordinates'][0] == "longitude"
        assert dset.attrs['coordinates'][1] == "latitude"

        with pytest.raises(NotImplementedError):
            handler.get_area_def(dsid)

    def test_report_datetimes(self, filetype_infos):
        """Should report time variables as numpy datetime64 type and time durations as timedelta64."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_le_nc'))

        # Check epoch_time:
        dsid = make_dataid(name="epoch_time_north_sector")
        dset = handler.get_dataset(dsid)
        assert dset.values.dtype == np.dtype('datetime64[ns]')

        # The default epoch_time should be 1.234 seconds after epoch:
        ref_time = np.datetime64(datetime(2000, 1, 1, 0, 0, 1, 234000))
        assert np.all(dset.values == ref_time)

        # Check time_offset:
        dsid = make_dataid(name="time_offset_east_sector")
        dset = handler.get_dataset(dsid)
        assert dset.values.dtype == np.dtype('timedelta64[ns]')

        # The default time_offset should be: np.linspace(0.0, 1000.0, nobs)
        # but then we first multiply by 1e6 to generate us times:
        # Note that below no automatic transform to np.float64 is happening:
        nobs = dset.shape[0]
        ref_data = np.linspace(0.0, 1000.0, nobs).astype(np.float32)
        ref_data = (ref_data * 1e9).astype('timedelta64[ns]')

        # And not absolutely sure why, but we always get the timedelta in ns from the dataset:
        # ref_data = (ref_data).astype('timedelta64[ns]')

        assert np.all(dset.values == ref_data)

    def test_milliseconds_to_timedelta(self, filetype_infos):
        """Should covert milliseconds to timedelta."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_lfl_nc'))

        # Check flash_duration:
        dsid = make_dataid(name="flash_duration")
        dset = handler.get_dataset(dsid)
        assert dset.values.dtype == np.dtype('timedelta64[ns]')

        nobs = dset.shape[0]
        ref_data = np.linspace(0, 1000, nobs).astype('u2')
        ref_data = (ref_data * 1e6).astype('timedelta64[ns]')

        assert np.all(dset.values == ref_data)

    def test_apply_accumulate_index_offset(self, filetype_infos):
        """Should accumulate index offsets."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_le_nc'))

        # Check time offset:
        dsid = make_dataid(name="l1b_chunk_offsets_north_sector")
        dset = handler.get_dataset(dsid)

        nobs = dset.shape[0]
        ref_data = (np.arange(nobs)).astype('u4')
        # check first execution without offset
        assert np.all(dset.values == ref_data)
        # check that the offset is being stored
        assert handler.current_ds_info['__index_offset'] == 123

        # check execution with offset value
        # this simulates the case where we are loading this variable from multiple files and concatenating it
        dset = handler.get_dataset(dsid, handler.current_ds_info)
        assert np.all(dset.values == ref_data + 123)

    def test_combine_info(self, filetype_infos):
        """Test overridden combine_info."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_le_nc'))

        # get a dataset including the index_offset in the ds_info
        dsid = make_dataid(name="l1b_chunk_offsets_north_sector")
        ds_info = {'name': 'l1b_chunk_offsets_north_sector',
                   'variable_name': 'l1b_chunk_offsets',
                   'sector_name': 'north',
                   '__index_offset': 1000,
                   'accumulate_index_offset': "{sector_name}/l1b_window"}
        dset = handler.get_dataset(dsid, ds_info=ds_info)
        handler.combine_info([dset.attrs])
        # combine_info should have removed the index_offset key from the ds_info passed to get_dataset
        assert '__index_offset' not in ds_info
        # and reset the current_ds_info dict, in order to avoid failures if we call combine_info again
        assert handler.current_ds_info is None

    def test_coordinates_projection(self, filetype_infos):
        """Should automatically generate lat/lon coords from projection data."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_af_nc'))

        dsid = make_dataid(name="flash_accumulation")
        dset = handler.get_dataset(dsid)
        assert 'coordinates' in dset.attrs

        assert dset.attrs['coordinates'][0] == "longitude"
        assert dset.attrs['coordinates'][1] == "latitude"

        with pytest.raises(NotImplementedError):
            handler.get_area_def(dsid)

        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_afr_nc'))

        dsid = make_dataid(name="flash_radiance")
        dset = handler.get_dataset(dsid)
        assert 'coordinates' in dset.attrs

        assert dset.attrs['coordinates'][0] == "longitude"
        assert dset.attrs['coordinates'][1] == "latitude"

        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_afa_nc'))

        dsid = make_dataid(name="accumulated_flash_area")
        dset = handler.get_dataset(dsid)
        assert 'coordinates' in dset.attrs

        assert dset.attrs['coordinates'][0] == "longitude"
        assert dset.attrs['coordinates'][1] == "latitude"

    def test_generate_coords_on_accumulated_prods(self, filetype_infos):
        """Test daskified generation of coords."""
        accumulated_products = ['li_l2_af_nc', 'li_l2_afr_nc', 'li_l2_afa_nc']
        coordinate_datasets = ['longitude', 'latitude']

        for accum_prod in accumulated_products:
            for ds_name in coordinate_datasets:
                handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, accum_prod))
                dsid = make_dataid(name=ds_name)
                dset = handler.get_dataset(dsid)
                # Check dataset type
                assert isinstance(dset, xr.DataArray)
                vals = dset.values
                assert vals is not None

    def test_generate_coords_on_lon_lat(self, filetype_infos):
        """Test getting lon/lat dataset on accumulated product."""
        accumulated_products = ['li_l2_af_nc', 'li_l2_afr_nc', 'li_l2_afa_nc']
        coordinate_datasets = ['longitude', 'latitude']

        for accum_prod in accumulated_products:
            for ds_name in coordinate_datasets:
                handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, accum_prod))
                dsid = make_dataid(name=ds_name)
                handler.generate_coords_from_scan_angles = mock.MagicMock(
                    side_effect=handler.generate_coords_from_scan_angles)
                handler.get_dataset(dsid)
                assert handler.generate_coords_from_scan_angles.called

    def test_generate_coords_inverse_proj(self, filetype_infos):
        """Test inverse_projection execution delayed until .values is called on the dataset."""
        accumulated_products = ['li_l2_af_nc', 'li_l2_afr_nc', 'li_l2_afa_nc']
        coordinate_datasets = ['longitude', 'latitude']

        for accum_prod in accumulated_products:
            for ds_name in coordinate_datasets:
                handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, accum_prod))
                dsid = make_dataid(name=ds_name)
                handler.inverse_projection = mock.MagicMock(side_effect=handler.inverse_projection)
                dset = handler.get_dataset(dsid)
                assert not handler.inverse_projection.called
                vals = dset.values
                assert vals is not None
                assert handler.inverse_projection.called

    def test_generate_coords_not_called_on_non_coord_dataset(self, filetype_infos):
        """Test that the method is not called when getting non-coord dataset."""
        handler = self.generate_coords(filetype_infos, 'li_l2_af_nc', 'flash_accumulation')
        assert not handler.generate_coords_from_scan_angles.called

    def test_generate_coords_not_called_on_non_accum_dataset(self, filetype_infos):
        """Test that the method is not called when getting non-accum dataset."""
        handler = self.generate_coords(filetype_infos, 'li_l2_lef_nc', 'latitude_north_sector')
        assert not handler.generate_coords_from_scan_angles.called

    def generate_coords(self, filetype_infos, file_type_name, variable_name):
        """Generate file handler and mimic coordinate generator call."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, file_type_name))
        dsid = make_dataid(name=variable_name)
        handler.generate_coords_from_scan_angles = mock.MagicMock(
            side_effect=handler.generate_coords_from_scan_angles)
        handler.get_dataset(dsid)
        return handler

    def test_generate_coords_called_once(Self, filetype_infos):
        """Test that the method is called only once."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_af_nc'))
        # check internal variable is empty
        assert len(handler.internal_variables) == 0
        coordinate_datasets = ['longitude', 'latitude']
        handler.generate_coords_from_scan_angles = mock.MagicMock(side_effect=handler.generate_coords_from_scan_angles)

        for ds_name in coordinate_datasets:
            dsid = make_dataid(name=ds_name)
            dset = handler.get_dataset(dsid)
            # Check dataset type
            assert isinstance(dset, xr.DataArray)
            assert len(handler.internal_variables) == 2
            assert handler.generate_coords_from_scan_angles.called

    def test_coords_generation(self, filetype_infos):
        """Compare daskified coords generation results with non-daskified."""
        # Prepare dummy (but somewhat realistic) arrays of azimuth/elevation values.
        products = ['li_l2_af_nc',
                    'li_l2_afr_nc',
                    'li_l2_afa_nc']

        for prod in products:
            handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, prod))

            # Get azimuth/elevation arrays from handler
            azimuth = handler.get_measured_variable(handler.swath_coordinates['azimuth'])
            azimuth = handler.apply_use_rescaling(azimuth)

            elevation = handler.get_measured_variable(handler.swath_coordinates['elevation'])
            elevation = handler.apply_use_rescaling(elevation)

            # Initialize proj_dict
            proj_var = handler.swath_coordinates['projection']
            geos_proj = handler.get_measured_variable(proj_var, fill_value=None)
            major_axis = float(geos_proj.attrs["semi_major_axis"])
            point_height = 35786400.0  # float(geos_proj.attrs["perspective_point_height"])
            inv_flattening = float(geos_proj.attrs["inverse_flattening"])
            lon_0 = float(geos_proj.attrs["longitude_of_projection_origin"])
            sweep = str(geos_proj.attrs["sweep_angle_axis"])
            proj_dict = {'a': major_axis,
                         'lon_0': lon_0,
                         'h': point_height,
                         "rf": inv_flattening,
                         'proj': 'geos',
                         'units': 'm',
                         "sweep": sweep}

            # Compute reference values
            projection = Proj(proj_dict)
            azimuth_vals = azimuth.values * point_height
            elevation_vals = elevation.values * point_height
            lon_ref, lat_ref = projection(azimuth_vals, elevation_vals, inverse=True)
            # Convert to float32:
            lon_ref = lon_ref.astype(np.float32)
            lat_ref = lat_ref.astype(np.float32)

            handler.generate_coords_from_scan_angles()
            lon = handler.internal_variables['longitude'].values
            lat = handler.internal_variables['latitude'].values

            # Compare the arrays, should be the same:
            np.testing.assert_equal(lon, lon_ref)
            np.testing.assert_equal(lat, lat_ref)

    def test_get_area_def_acc_products(self, filetype_infos):
        """Test retrieval of area def for accumulated products."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_af_nc'),
                                    with_area_definition=True)

        dsid = make_dataid(name="flash_accumulation")

        area_def = handler.get_area_def(dsid)
        assert area_def.shape == LI_GRID_SHAPE

        # Should throw for non-gridded variables:
        with pytest.raises(NotImplementedError):
            handler.get_area_def(make_dataid(name="accumulation_offsets"))

    def test_get_area_def_non_acc_products(self, filetype_infos):
        """Test retrieval of area def for non-accumulated products."""
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_lgr_nc'),
                                    with_area_definition=True)
        # Should throw for non-accum products:
        with pytest.raises(NotImplementedError):
            handler.get_area_def(make_dataid(name="radiance"))

    @staticmethod
    def param_provider(_filename, filename_info, _fileype_info):
        """Provide parameters."""

        def write_flash_accum(_vname, _ocname, _settings):
            """Write the flash accumulation array."""
            return np.arange(1234, dtype=np.float32) + 0.5

        # We return the settings we want to use here to generate our custom/fixed product content:
        return {
            'num_obs': 1234,
            'providers': {
                'flash_accumulation': write_flash_accum,
            }
        }

    def test_without_area_def(self, filetype_infos):
        """Test accumulated products data array without area definition."""
        # without area definition
        handler_without_area_def = LIL2NCFileHandler(
            'filename', {}, extract_filetype_info(filetype_infos, 'li_l2_af_nc'), with_area_definition=False)

        dsid = make_dataid(name="flash_accumulation")

        # Keep the data array:
        data = handler_without_area_def.get_dataset(dsid).values
        assert data.shape == (1234,)

    def test_with_area_def(self, filetype_infos):
        """Test accumulated products data array with area definition."""
        handler = self.handler_with_area(filetype_infos, 'li_l2_af_nc')
        dsid = make_dataid(name="flash_accumulation")
        # Retrieve the 2D array:
        arr = handler.get_dataset(dsid).values
        assert arr.shape == LI_GRID_SHAPE

    def test_get_on_fci_grid_exc(self, filetype_infos):
        """Test the execution of the get_on_fci_grid function for an accumulated gridded variable."""
        handler = self.handler_with_area(filetype_infos, 'li_l2_af_nc')
        handler.get_array_on_fci_grid = mock.MagicMock(side_effect=handler.get_array_on_fci_grid)
        dsid = make_dataid(name="flash_accumulation")
        handler.get_dataset(dsid)
        assert handler.get_array_on_fci_grid.called

    def test_get_on_fci_grid_exc_non_grid(self, filetype_infos):
        """Test the non-execution of the get_on_fci_grid function for an accumulated non-gridded variable."""
        handler = self.handler_with_area(filetype_infos, 'li_l2_af_nc')
        handler.get_array_on_fci_grid = mock.MagicMock(side_effect=handler.get_array_on_fci_grid)
        dsid = make_dataid(name="accumulation_offsets")
        handler.get_dataset(dsid)
        assert not handler.get_array_on_fci_grid.called

    def test_get_on_fci_grid_exc_non_accum(self, filetype_infos):
        """Test the non-execution of the get_on_fci_grid function for a non-accumulated variable."""
        handler = self.handler_with_area(filetype_infos, 'li_l2_lef_nc')
        handler.get_array_on_fci_grid = mock.MagicMock(side_effect=handler.get_array_on_fci_grid)
        dsid = make_dataid(name="radiance_north_sector")
        handler.get_dataset(dsid)
        assert not handler.get_array_on_fci_grid.called

    def test_with_area_def_vars_with_no_pattern(self, filetype_infos):
        """Test accumulated products variable with no patterns and with area definition."""
        handler = self.handler_with_area(filetype_infos, 'li_l2_af_nc')
        # variable with no patterns
        dsid = make_dataid(name="accumulation_offsets")
        assert handler.get_dataset(dsid).shape == (1,)

    def handler_with_area(self, filetype_infos, product_name):
        """Create handler with area definition."""
        # Note: we need a test param provider here to ensure we write the same values for both handlers below:
        FakeLIFileHandlerBase.schema_parameters = TestLIL2.param_provider
        # with area definition
        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, product_name),
                                    with_area_definition=True)
        return handler

    def test_with_area_def_pixel_placement(self, filetype_infos):
        """Test the placements of pixel value with area definition."""
        # with area definition
        FakeLIFileHandlerBase.schema_parameters = TestLIL2.param_provider

        handler = LIL2NCFileHandler('filename', {}, extract_filetype_info(filetype_infos, 'li_l2_af_nc'),
                                    with_area_definition=True)
        dsid = make_dataid(name="flash_accumulation")

        # Retrieve the 2D array:a
        arr = handler.get_dataset(dsid).values

        # Retrieve the x/y coordinates:
        xarr = handler.get_measured_variable('x').values.astype(int)
        yarr = handler.get_measured_variable('y').values.astype(int)

        handler_without_area_def = LIL2NCFileHandler(
            'filename', {}, extract_filetype_info(filetype_infos, 'li_l2_af_nc'), with_area_definition=False)

        FakeLIFileHandlerBase.schema_parameters = None

        # prepare reference array
        data = handler_without_area_def.get_dataset(dsid).values
        ref_arr = np.empty(LI_GRID_SHAPE, dtype=arr.dtype)
        ref_arr[:] = np.nan
        rows = (LI_GRID_SHAPE[0] - yarr)
        cols = xarr - 1
        ref_arr[rows, cols] = data

        # Check all nan values are at the same locations:
        assert np.all(np.isnan(arr) == np.isnan(ref_arr))

        # Check all finite values are the same:
        assert np.all(arr[np.isfinite(arr)] == ref_arr[np.isfinite(ref_arr)])
