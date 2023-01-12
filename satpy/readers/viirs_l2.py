# Copyright (c) 2022-2023 Satpy developers
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
"""Interface to VIIRS L2 files."""

from datetime import datetime

from satpy.readers.netcdf_utils import NetCDF4FileHandler


class VIIRSCloudMaskFileHandler(NetCDF4FileHandler):
    """VIIRS L2 Cloud Mask reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the file handler."""
        super().__init__(filename, filename_info, filetype_info, cache_handle=True)

    def _parse_datetime(self, datestr):
        """Parse datetime."""
        return datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%SZ")

    @property
    def start_orbit_number(self):
        """Get start orbit number."""
        return int(self['/attr/start_orbit_number'])

    @property
    def end_orbit_number(self):
        """Get end orbit number."""
        return int(self['/attr/end_orbit_number'])

    @property
    def platform_name(self):
        """Get platform name."""
        res = self.filename_info['platform_shortname']

        return {
            'npp': 'Suomi-NPP',
            'j01': 'NOAA-20',
            'j02': 'NOAA-21',
        }.get(res, res)

    @property
    def sensor_name(self):
        """Get sensor name."""
        return self['/attr/instrument_name'].lower()

    def get_shape(self, ds_id, ds_info):
        """Get shape."""
        return self.get(ds_id['name'] + '/shape', 1)

    @property
    def start_time(self):
        """Get start time."""
        return self._parse_datetime(self['/attr/time_coverage_start'])

    @property
    def end_time(self):
        """Get end time."""
        return self._parse_datetime(self['/attr/time_coverage_end'])

    def get_metadata(self, dataset_id, ds_info):
        """Get metadata."""
        var_path = ds_info['file_key']
        shape = self.get_shape(dataset_id, ds_info)
        file_units = ds_info.get('file_units')

        attr = getattr(self[var_path], 'attrs', {})
        attr.update(ds_info)
        attr.update(dataset_id.to_dict())
        attr.update({
            "shape": shape,
            "units": ds_info.get("units", file_units),
            "file_units": file_units,
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
        })
        attr.update(dataset_id.to_dict())
        return attr

    def get_dataset(self, dataset_id, ds_info):
        """Get dataset."""
        var_path = ds_info['file_key']
        metadata = self.get_metadata(dataset_id, ds_info)

        valid_min, valid_max = self._get_dataset_valid_range(var_path)
        data = self[var_path]
        data.attrs.update(metadata)

        if valid_min is not None and valid_max is not None:
            data = data.where((data >= valid_min) & (data <= valid_max))

        if isinstance(data.attrs.get('flag_meanings'), str):
            data.attrs['flag_meanings'] = data.attrs['flag_meanings'].split(' ')

        # rename dimensions to correspond to satpy's 'y' and 'x' standard
        if 'Rows' in data.dims:
            data = data.rename({'Rows': 'y', 'Columns': 'x'})
        return data

    def _get_dataset_valid_range(self, var_path):
        valid_range = self.get(var_path + '/attr/valid_range')
        valid_min = valid_range[0]
        valid_max = valid_range[1]

        return valid_min, valid_max
