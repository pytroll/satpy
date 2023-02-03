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
r"""Base class used for the MTG Lighting Imager netCDF4 readers.

The base LI reader class supports generating the available datasets
programmatically: to achieve this, each LI product type should provide a
\"file description\" which is itself retrieved directly from the YAML
configuration file for the reader of interest, as a custom
``file_desc`` entry inside the \'file_type\' section
corresponding to that product type.

Each of the ``file_desc`` entry describes what are the
variables that are available into that product that should be used to
register the available satpy datasets.

Each of those description entries may contain the following elements:

- **product_type** \[required\]:

  Indicate the processing_level / product_type name to use internally for
  that type of product file. This should correspond to the
  ``{processing_level}-{product_type}`` part of the full
  file_pattern.

- **search_paths** \[optional\]:

  A list of the possible paths that should be prefixed to a given variable
  name when searching for that variable in the NetCDF file to register a
  dataset on it. The list is given in priority order. If no search path is
  provided (or an empty array is provided) then the variables will only be
  searched directly in the root group of the NetCDF structure.

- **swath_coordinates** \[required\]:

  The LI reader will use a ``SwathDefinition`` object to define
  the area/coordinates of each of the provided datasets depending on the
  content of this entry. The user can either:

    - Specify a
      ``swath_coordinates`` entry directly with
      ``latitude`` and ``longitude`` entries, in which
      case, the datasets that will match one of the
      ``'variable_patterns'`` provided will use those lat/lon
      variables as coordinate providers.
    - Specify a
      ``swath_coordinates`` entry directly with
      ``projection``, ``azimuth`` and
      ``elevation`` entries instead, in which case, the reader will
      first use the variables pointed by those 3 entries compute the
      corresponding latitude/longitude data from the scan angles contained in
      the product file. And then, continue with assigned those lat/lon
      datasets as coordinates for datasets that will match one of the
      ``variable_patterns`` provided.

  **Note:** It is acceptable to specify an empty array for the list of
  ``variable_patterns``, in this case, the swath coordinates will not be
  assigned to any dataset.

- **sectors** \[optional\]:

  The custom dataset description mechanism makes a distinction between
  \"ordinary\" variables which should be used to create a \"single
  dataset\" and \"sectored variables\" which will be found per sector and
  will thus be used to generate as many datasets as there are sectors (see below). So
  this entry is used to specify the list of sector names there should be
  available in the NetCDF structure.

- **sector_variables** \[optional\]:

  This entry is used to provide a list of the variables that are available
  **per sector** in the NetCDF file. Thus, assuming the
  ``sectors`` entry is set to the standard list ``['north',
  'east', 'south', 'west']``, 4 separated datasets will
  be registered for each variable listed here (using the conventional
  suffix ``"{sector_name}_sector"``)

- **variables** \[optional\]:

  This entry is used to provide a list of \"ordinary variables\" (ie.
  variables that are not available **per sector**). Each of those
  variables will be used to register one dataset.

  **Note:** A single product may provide both the \"variables\" and the
  \"sector_variables\" at the same time (as this is the case for LI LEF
  for instance)

- **variable_transforms** \[optional\]:

  This entry is may be used to provide specific additional entries **per
  variable name** (ie. will apply to both in sector or out of sector
  variables) that should be added to the dataset infos when registering a
  dataset with that variable. While any kind of info could be added this
  way to the final dataset infos, we are currently using the entry mainly
  to provide our LI reader with the following traits which
  will then be used to \"transform\" the data of the dataset as requested
  on loading:

    - ``broadcast_to``: if this extra info is found in a
      dataset_info on dataset loading, then the initial data array will be
      broadcast to the shape of the variable found under the variable path
      specified as value for that entry. Note that, if the pattern
      ``{sector_name}`` if found in this entry value, then the
      reader will assume that we are writing a dataset from an in sector
      variable, and use the current sector name to find the appropriate
      alternate variable that will be used as reference to broadcast the
      current variable data.

    - ``seconds_to_datetime``: This transformation is used to
      internally convert variables provided as float values to the
      ``np.datetime64`` data type. The value specified for this
      entry should be the reference epoch time used as offsets for the
      elapsed seconds when converting the data.

    - ``seconds_to_timedelta``: This transformation is used to
      internally convert variables (assumed to use a \"second\" unit)
      provided as float values to the ``np.timedelta64`` data
      type. This entry should be set to ``true`` to activate this
      transform. During the conversion, we internally use a
      nanosecond resolution on the input floating point second values.

    - ``milliseconds_to_timedelta``: Same kind of transformation
      as ``seconds_to_timedelta`` except that the source data is
      assumed to contain millisecond float values.

    - ``accumulate_index_offset``: if this extra info is found in
      a ``dataset_info`` on dataset loading, then we will consider that the
      dataset currently being generated is an array of indices inside the
      variable pointed by the path provided as value for that entry. Note
      that the same usage of the pattern ``{sector_name}``
      mentioned for the entry \"broadcast_to\" will also apply here. This
      behavior is useful when multiple input files are loaded together in a
      single satpy scene, in which case, the variables from each files will
      be concatenated to produce a single dataset for each variable, and
      thus the need to correct the reported indices accordingly.

      An example of usage of this entry is as follows:

      ..  code-block:: yaml

          variable_transforms:
            integration_frame_index:
              accumulate_index_offset: "{sector_name}/exposure_time"



      In the example above the integration_frame_index from each sector (i.e.
      optical channel) provides a list of indices in the corresponding
      exposure_time array from that same sector. The final indices will thus
      correctly take into account that the final exposure_time array contains
      all the values concatenated from all the input files in the scene.

    - ``use_rescaling``: By default, we currently
      apply variable rescaling as soon as we find one (or more) of the
      attributes named ``'scale_factor'``,
      ``'scaling_factor'`` or ``'add_offset'`` in
      the source netcdf variable. This automatic transformation can be
      disabled for a given variable specifying a value of false for
      this extra info element, for instance:

      ..  code-block:: yaml

          variable_transforms:
            latitude:
              use_rescaling: false


      **Note:** We are currently not disabling rescaling for any dataset, so
      that entry is not used in the current version of the YAML config files
      for the LI readers.

"""

import logging
import re

import dask.array as da
import netCDF4
import numpy as np
import xarray as xr
from pyproj import Proj

from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


class LINCFileHandler(NetCDF4FileHandler):
    """Base class used as parent for the concrete LI reader classes."""

    def __init__(self, filename, filename_info, filetype_info, cache_handle=True):
        """Initialize LINCFileHandler."""
        super().__init__(filename, filename_info, filetype_info,
                         cache_var_size=10000,
                         cache_handle=cache_handle
                         )

        # decode_times should be disabled for xr.open_dataset access (cache_handle=False):
        # Note: the default dict assignment is need to avoid error when using the fake
        # netcdf4 file handler in mock unit tests:
        self._xarray_kwargs = getattr(self, "_xarray_kwargs", {})
        self._xarray_kwargs['decode_times'] = False
        self._xarray_kwargs['mask_and_scale'] = False

        # Processing level that should be set by derived classes.
        self.processing_level = filetype_info.get('processing_level', 'L0')

        # This class will only provide support for the LI sensor:
        self.sensors = {'li'}

        # Set of dataset names explicitly provided by this file handler:
        # This set is required to filter the retrieval of datasets later in the
        # get_dataset() method, for instance when building a Scene from multiple
        # different product files (example: 1 L1B BCK file + 1 L1B LE file):
        # the dataset loading mechanism will still request from the LE specific
        # file handler if it can load a 'timestamp_vis_08_north_sector' dataset
        # for instance.
        # And when concatenating multiple BCK files into a single scene, usually
        # only one of the file handler will be able to load a specific timestamp.
        # => We could recompute the availability of a dataset from the provided
        # ds_info in get_dataset(), but it seems a better/easier solution to just
        # cache the set of available dataset names as generated in 'available_datasets()'
        # directly here:
        self.provided_datasets = set()

        self.ds_desc = filetype_info['file_desc']
        # Store the extra infos available on specific variables:
        # Write the correct product type here:
        self.product_type = self.ds_desc['product_type']
        logger.debug("Product type is: %s", self.product_type)

        self.variable_transforms = self.ds_desc.get('variable_transforms', {})

        # Store the pattern for the default swath coordinates:
        # Note that we should always have this swath coordinates entry now:
        self.swath_coordinates = self.ds_desc.get('swath_coordinates', {})
        patterns = self.swath_coordinates.get('variable_patterns', [])
        self.swath_coordinates['patterns'] = [re.compile(pstr) for pstr in patterns]

        # check if the current product is in an accumulation grid
        self.prod_in_accumulation_grid = self.is_prod_in_accumulation_grid()

        # list of paths where we should be looking for data when trying to retrieve
        # "measured variables" from the netcdf file attached to this file handler.
        self.search_paths = None

        # Storage for the registered datasets provided in this file handler: will only be
        # initialized once in `register_available_datasets()`
        self.dataset_infos = None

        # Storage for the current ds_infos in use in a call to get_dataset()
        self.current_ds_info = None

        # Ordered list of transform operations supported in this file handler:
        # those transforms are applied if requested in the 'apply_transforms' method below
        self.transform_names = ['use_rescaling', 'seconds_to_timedelta', 'milliseconds_to_timedelta',
                                'seconds_to_datetime', 'broadcast_to', 'accumulate_index_offset']

        # store internal variables
        self.internal_variables = {}

        # We register all the available datasets on creation:
        self.register_available_datasets()

    @property
    def start_time(self):
        """Get the start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get the end time."""
        return self.filename_info['end_time']

    @property
    def sensor_names(self):
        """List of sensors represented in this file."""
        return self.sensors

    def is_prod_in_accumulation_grid(self):
        """Check if the current product is an accumulated product in geos grid."""
        in_grid = self.swath_coordinates.get('projection', None) == 'mtg_geos_projection'
        return in_grid

    def get_latlon_names(self):
        """Retrieve the user specified names for latitude/longitude coordinates.

        Use default 'latitude' / 'longitude' if not specified.
        """
        lon_name = self.swath_coordinates.setdefault('longitude', 'longitude')
        lat_name = self.swath_coordinates.setdefault('latitude', 'latitude')
        return lat_name, lon_name

    def get_projection_config(self):
        """Retrieve the projection configuration details."""
        # We retrieve the projection variable name directly from our swath settings:
        proj_var = self.swath_coordinates['projection']

        geos_proj = self.get_measured_variable(proj_var, fill_value=None)
        # cast projection attributes to float/str:
        major_axis = float(geos_proj.attrs["semi_major_axis"])
        # TODO reinstate reading from file when test data issue is fixed
        point_height = 35786400.0  # float(geos_proj.attrs["perspective_point_height"])
        inv_flattening = float(geos_proj.attrs["inverse_flattening"])
        lon_0 = float(geos_proj.attrs["longitude_of_projection_origin"])
        sweep = str(geos_proj.attrs["sweep_angle_axis"])

        # use a (semi-major axis) and rf (reverse flattening) to define ellipsoid as recommended by EUM
        proj_dict = {'a': major_axis,
                     'lon_0': lon_0,
                     'h': point_height,
                     "rf": inv_flattening,
                     'proj': 'geos',
                     'units': 'm',
                     "sweep": sweep}

        return proj_dict

    def get_daskified_lon_lat(self, proj_dict):
        """Get daskified lon and lat array using map_blocks."""
        # Get our azimuth/elevation arrays,
        azimuth = self.get_measured_variable(self.swath_coordinates['azimuth'])
        azimuth = self.apply_use_rescaling(azimuth)

        elevation = self.get_measured_variable(self.swath_coordinates['elevation'])
        elevation = self.apply_use_rescaling(elevation)

        # Daskify inverse projection computation:
        lon, lat = da.map_blocks(self.inverse_projection, azimuth, elevation, proj_dict,
                                 chunks=(2, azimuth.shape[0]),
                                 meta=np.array((), dtype=azimuth.dtype),
                                 dtype=azimuth.dtype,
                                 )
        return lon, lat

    def generate_coords_from_scan_angles(self):
        """Generate the latitude/longitude coordinates from the scan azimuth and elevation angles."""
        proj_cfg = self.get_projection_config()
        lon, lat = self.get_daskified_lon_lat(proj_cfg)

        # Retrieve the names we should use for the generated lat/lon datasets:
        lat_name, lon_name = self.get_latlon_names()

        # Finally, we should store those arrays as internal variables for later retrieval as
        # standard datasets:
        self.internal_variables[lon_name] = xr.DataArray(
            da.asarray(lon), dims=['y'], attrs={'standard_name': 'longitude'})
        self.internal_variables[lat_name] = xr.DataArray(
            da.asarray(lat), dims=['y'], attrs={'standard_name': 'latitude'})

    def inverse_projection(self, azimuth, elevation, proj_dict):
        """Compute inverse projection."""
        # Initialise Proj object:
        projection = Proj(proj_dict)

        # Retrieve the point height from the projection config:
        point_height = proj_dict['h']

        # Convert scan angles to projection coordinates by multiplying with perspective point height
        azimuth = azimuth.values * point_height
        elevation = elevation.values * point_height

        lon, lat = projection(azimuth, elevation, inverse=True)

        return np.stack([lon.astype(azimuth.dtype), lat.astype(elevation.dtype)])

    def register_coords_from_scan_angles(self):
        """Register lat lon datasets in this reader."""
        lat_name, lon_name = self.get_latlon_names()
        self.register_dataset(lon_name)
        self.register_dataset(lat_name)

    def variable_path_exists(self, var_path):
        """Check if a given variable path is available in the underlying netCDF file.

        All we really need to do here is to access the file_content dictionary and
        check if we have a variable under that var_path key.
        """
        # but we ignore attributes: or sub properties:
        if var_path.startswith("/attr") or var_path.endswith(("/dtype", "/shape", "/dimensions")):
            return False

        # Check if the path is found:
        if var_path in self.file_content:
            # This is only a valid variable if it is not a netcdf group:
            return not isinstance(self.file_content[var_path], netCDF4.Group)

        # Var path not in file_content:
        return False

    def get_first_valid_variable(self, var_paths):
        """Select the first valid path for a variable from the given input list and returns the data."""
        for vpath in var_paths:
            if self.variable_path_exists(vpath):
                return self[vpath]

        # We could not find a variable with that path, this might be an error:
        raise KeyError(f"Could not find variable with paths: {var_paths}")

    def get_measured_variable(self, var_paths, fill_value=np.nan):
        """Retrieve a measured variable path taking into account the potential old data formatting schema.

        And also replace the missing values with the provided fill_value (except if this is explicitly
        set to None).
        Also, if a slice index is provided, only that slice of the array (on the axis=0) is retrieved
        (before filling the missing values).
        """
        # convert the var_paths to a list in case it is a single string:
        if isinstance(var_paths, str):
            var_paths = [var_paths]

        # then we may return one of the internal variables:
        # We only really need to check the first variable name in the list below:
        # it doesn't really make sense to mix internal variables and multi var
        # names anyway
        for vname, arr in self.internal_variables.items():
            if var_paths[0].endswith(vname):
                return arr

        # Get the search paths from our dataset descriptions:
        all_var_paths = self.get_variable_search_paths(var_paths)

        arr = self.get_first_valid_variable(all_var_paths)

        # Also handle fill value here (but only if it is not None, so that we can still bypass this
        # step if needed)
        arr = self.apply_fill_value(arr, fill_value)

        return arr

    def apply_fill_value(self, arr, fill_value):
        """Apply fill values, unless it is None."""
        if fill_value is not None:
            if np.isnan(fill_value):
                fill_value = np.float32(np.nan)
            arr = arr.where(arr != arr.attrs.get('_FillValue'), fill_value)
        return arr

    def get_variable_search_paths(self, var_paths):
        """Get the search paths from the dataset descriptions."""
        if len(self.search_paths) == 0:
            all_var_paths = var_paths
        else:
            all_var_paths = [f"{folder}/{var_path}"
                             for folder in self.search_paths
                             for var_path in var_paths]
        return all_var_paths

    def add_provided_dataset(self, ds_infos):
        """Add a provided dataset to our internal list."""
        # Check if we have extra infos for that variable:
        # Note that if available we should use the alias name instead here:
        vname = ds_infos["alias_name"] if 'alias_name' in ds_infos else ds_infos["variable_name"]
        self.check_variable_extra_info(ds_infos, vname)

        # We check here if we should include the default coordinates on that dataset:
        if self.swath_coordinates is not None and 'coordinates' not in ds_infos:

            # Check if the variable corresponding to this dataset will match one of the valid patterns
            # for the swath usage:
            if any([p.search(vname) is not None for p in self.swath_coordinates['patterns']]):

                # Get the target coordinate names, applying the sector name as needed:
                lat_coord_name, lon_coord_name = self.get_coordinate_names(ds_infos)

                # Ensure we do not try to add the coordinates on the coordinates themself:
                dname = ds_infos['name']
                if dname != lat_coord_name and dname != lon_coord_name:
                    ds_infos['coordinates'] = [lon_coord_name, lat_coord_name]
        self.dataset_infos.append(ds_infos)
        self.provided_datasets.add(ds_infos['name'])

    def check_variable_extra_info(self, ds_infos, vname):
        """Check if we have extra infos for that variable."""
        if vname in self.variable_transforms:
            extras = self.variable_transforms[vname]

            # extend the ds_infos:
            ds_infos.update(extras)

    def get_coordinate_names(self, ds_infos):
        """Get the target coordinate names, applying the sector name as needed."""
        lat_coord_name, lon_coord_name = self.get_latlon_names()
        if 'sector_name' in ds_infos:
            sname = ds_infos['sector_name']
            lat_coord_name = lat_coord_name.replace("{sector_name}", sname)
            lon_coord_name = lon_coord_name.replace("{sector_name}", sname)
        return lat_coord_name, lon_coord_name

    def get_dataset_infos(self, dname):
        """Retrieve the dataset infos corresponding to one of the registered datasets."""
        for dsinfos in self.dataset_infos:
            if dsinfos['name'] == dname:
                return dsinfos

        # nothing found.
        return None

    def register_dataset(self, var_name, oc_name=None):
        """Register a simple dataset given name elements."""
        # generate our default dataset name:

        ds_name = var_name if oc_name is None else f"{var_name}_{oc_name}_sector"

        ds_info = {
            'name': ds_name,
            'variable_name': var_name,
            'sensor': 'li',
            'file_type': self.filetype_info['file_type']
        }

        # add the sector name:
        if oc_name is not None:
            ds_info['sector_name'] = oc_name

        self.add_provided_dataset(ds_info)

    def register_available_datasets(self):
        """Register all the available dataset that should be made available from this file handler."""
        if self.dataset_infos is not None:
            return

        # Otherwise, we need to perform the registration:
        self.dataset_infos = []

        # Assign the search paths for this product type:
        self.search_paths = self.ds_desc.get('search_paths', [])

        # Register our coordinates from azimuth/elevation data
        # if the product is accumulated
        if self.prod_in_accumulation_grid:
            self.register_coords_from_scan_angles()

        # First we check if we have support for sectors for this product:
        self.register_sector_datasets()

        # Retrieve the list of "raw" (ie not in sectors) variables provided in this description:
        self.register_variable_datasets()

        logger.debug("Adding %d datasets for %s input product.",
                     len(self.dataset_infos), self.product_type)

    def register_variable_datasets(self):
        """Register all the available raw (i.e. not in sectors)."""
        if 'variables' in self.ds_desc:
            all_vars = self.ds_desc['variables']
            # No sector to handle so we write simple datasets from the variables:
            for var_name in all_vars:
                self.register_dataset(var_name)

    def register_sector_datasets(self):
        """Register all the available sector datasets."""
        if 'sectors' in self.ds_desc:
            sectors = self.ds_desc['sectors']
            sector_vars = self.ds_desc['sector_variables']
            # We should generate the datasets per sector:
            for oc_name in sectors:
                for var_name in sector_vars:
                    self.register_dataset(var_name, oc_name)

    def available_datasets(self, configured_datasets=None):
        """Determine automatically the datasets provided by this file.

        Uses a per product type dataset registration mechanism using the dataset descriptions declared in the reader
        construction above.
        """
        # pass along existing datasets
        for is_avail, ds_info in (configured_datasets or []):
            yield is_avail, ds_info

        for ds_info in self.dataset_infos:
            yield True, ds_info

    def apply_use_rescaling(self, data_array, ds_info=None):
        """Apply the use_rescaling transform on a given array."""
        # Here we should apply the rescaling except if it is explicitly requested not to rescale
        if ds_info is not None and ds_info.get("use_rescaling", True) is not True:
            return data_array

        # Check if we have the scaling elements:
        attribs = data_array.attrs
        if 'scale_factor' in attribs or 'scaling_factor' in attribs or 'add_offset' in attribs:
            # TODO remove scaling_factor fallback after issue in NetCDF is fixed
            scale_factor = attribs.setdefault('scale_factor', attribs.get('scaling_factor', 1))
            add_offset = attribs.setdefault('add_offset', 0)

            data_array = (data_array * scale_factor) + add_offset

            # rescale the valid range accordingly
            if 'valid_range' in attribs.keys():
                attribs['valid_range'] = attribs['valid_range'] * scale_factor + add_offset

        data_array.attrs.update(attribs)

        return data_array

    def apply_broadcast_to(self, data_array, ds_info):
        """Apply the broadcast_to transform on a given array."""
        ref_var = self.get_transform_reference('broadcast_to', ds_info)

        logger.debug("Broascasting %s to shape %s", ds_info['name'], ref_var.shape)
        new_array = da.broadcast_to(data_array, ref_var.shape)
        dims = data_array.dims if data_array.ndim > 0 else ('y',)
        data_array = xr.DataArray(new_array, coords=data_array.coords, dims=dims, name=data_array.name,
                                  attrs=data_array.attrs)
        return data_array

    def apply_accumulate_index_offset(self, data_array, ds_info):
        """Apply the accumulate_index_offset transform on a given array."""
        # retrieve the __index_offset here, or create it if missing:
        # And keep track of the shared ds_info dict to reset it later in combine_info()
        self.current_ds_info = ds_info
        offset = ds_info.setdefault('__index_offset', 0)

        ref_var = self.get_transform_reference('accumulate_index_offset', ds_info)

        # Apply the current index_offset already reached on the indices we have in the current dataset:
        data_array = data_array + offset

        # Now update the __index_offset adding the number of elements in the reference array:
        ds_info['__index_offset'] = offset + ref_var.size
        logger.debug("Adding %d elements for index offset, new value is: %d",
                     ref_var.size, ds_info['__index_offset'])

        return data_array

    def apply_seconds_to_datetime(self, data_array, ds_info):
        """Apply the seconds_to_datetime transform on a given array."""
        # Retrieve the epoch timestamp:
        epoch_ts = np.datetime64('2000-01-01T00:00:00.000000')

        # And add our values as delta times in seconds:
        # note that we use a resolution of 1ns here:
        data_array = epoch_ts + (data_array * 1e9).astype('timedelta64[ns]')
        return data_array

    def apply_seconds_to_timedelta(self, data_array, _ds_info):
        """Apply the seconds_to_timedelta transform on a given array."""
        # Apply the type conversion in place in the data_array:
        # note that we use a resolution of 1ns here:
        data_array = (data_array * 1e9).astype('timedelta64[ns]')
        return data_array

    def apply_milliseconds_to_timedelta(self, data_array, _ds_info):
        """Apply the milliseconds_to_timedelta transform on a given array."""
        # Apply the type conversion in place in the data_array:
        # note that we use a resolution of 1ns here:
        data_array = (data_array * 1e6).astype('timedelta64[ns]')
        return data_array

    def get_transform_reference(self, transform_name, ds_info):
        """Retrieve a variable that should be used as reference during a transform."""
        var_path = ds_info[transform_name]

        if "{sector_name}" in var_path:
            # We really expect to have a sector name for that variable:
            var_path = var_path.replace("{sector_name}", ds_info['sector_name'])

        # get the variable on that path:
        ref_var = self.get_measured_variable(var_path)

        return ref_var

    def apply_transforms(self, data_array, ds_info):
        """Apply all transformations requested in the ds_info on the provided data array."""
        # Rescaling should be enabled by default:
        ds_info.setdefault("use_rescaling", True)
        for tname in self.transform_names:
            if tname in ds_info:
                # Retrieve the transform function:
                transform = getattr(self, f'apply_{tname}')
                # Apply the transformation on the dataset:
                data_array = transform(data_array, ds_info)
        return data_array

    def combine_info(self, all_infos):
        """Re-implement combine_info.

        This is to be able to reset our __index_offset attribute in the shared ds_info currently being updated.
        """
        if self.current_ds_info is not None:
            del self.current_ds_info['__index_offset']
            self.current_ds_info = None

        return super().combine_info(all_infos)

    def get_transformed_dataset(self, ds_info):
        """Retrieve a dataset with all transformations applied on it."""
        # Extract base variable name:
        vname = ds_info['variable_name']

        # Note that the sector name might be None below:
        sname = ds_info.get('sector_name', None)

        # Use the sector name as prefix for the variable path if applicable:
        var_paths = vname if sname is None else f"{sname}/{vname}"

        # Note that this includes the case where sname == None:
        data_array = self.get_measured_variable(var_paths)
        data_array = self.apply_transforms(data_array, ds_info)
        return data_array

    def validate_array_dimensions(self, data_array, ds_info=None):
        """Ensure that the dimensions of the provided data_array are valid."""
        # We also need a special handling of the ndim==0 case (i.e. reading scalar values)
        # in order to potentially support data array combination in a satpy scene:
        if data_array.ndim == 0:
            # If we have no dimension, we should force creating one here:
            data_array = data_array.expand_dims({'y': 1})

        data_array = data_array.rename({data_array.dims[0]: 'y'})

        return data_array

    def update_array_attributes(self, data_array, ds_info):
        """Inject the attributes from the ds_info structure into the final data array, ignoring the internal entries."""
        # ignore some internal processing only entries:
        ignored_attribs = ["__index_offset", "broadcast_to", 'accumulate_index_offset',
                           'seconds_to_timedelta', 'seconds_to_datetime']
        for key, value in ds_info.items():
            if key not in ignored_attribs:
                data_array.attrs[key] = value

        return data_array

    def get_dataset(self, dataset_id, ds_info=None):
        """Get a dataset."""
        # Retrieve default infos if missing:
        if ds_info is None:
            ds_info = self.get_dataset_infos(dataset_id['name'])

        # check for potential error:
        if ds_info is None:
            raise KeyError(f"No dataset registered for {dataset_id}")

        ds_name = ds_info['name']
        # In case this dataset name is not explicitly provided by this file handler then we
        # should simply return None.
        if ds_name not in self.provided_datasets:
            return None

        # Generate our coordinates from azimuth/elevation data if needed.
        # It shall be called only when a corresponding dataset is being requested
        # (i.e. longitude and latitude for accumulated products)
        coord_names = self.get_latlon_names()
        is_coord = ds_name in coord_names
        # call only when internal variable is empty, to avoid multiple call.
        if ds_name not in self.internal_variables and is_coord and self.prod_in_accumulation_grid:
            self.generate_coords_from_scan_angles()

        # Retrieve the transformed data array:
        data_array = self.get_transformed_dataset(ds_info)

        # Validate the dimensions:
        data_array = self.validate_array_dimensions(data_array, ds_info)

        # Update the attributes in the final array:
        data_array = self.update_array_attributes(data_array, ds_info)

        # Return the resulting array:
        return data_array
