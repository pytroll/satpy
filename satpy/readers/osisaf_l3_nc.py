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
"""A reader for OSI-SAF level 3 products in netCDF format."""

import datetime as dt
import logging

from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


class OSISAFL3NCFileHandler(NetCDF4FileHandler):
    """Reader for the OSISAF l3 netCDF format."""
    def _get_ease_grid(self):
        """Set up the EASE grid."""
        from pyresample import create_area_def

        proj4str = self["Lambert_Azimuthal_Grid/attr/proj4_string"]
        x_size = self["/dimension/xc"]
        y_size = self["/dimension/yc"]
        p_lowerleft_lat = self["lat"].values[y_size - 1, 0]
        p_lowerleft_lon = self["lon"].values[y_size - 1, 0]
        p_upperright_lat = self["lat"].values[0, x_size - 1]
        p_upperright_lon = self["lon"].values[0, x_size - 1]
        area_extent = [p_lowerleft_lon, p_lowerleft_lat, p_upperright_lon, p_upperright_lat]
        area_def = create_area_def(area_id="osisaf_lambert_azimuthal_equal_area",
                                   description="osisaf_lambert_azimuthal_equal_area",
                                   proj_id="osisaf_lambert_azimuthal_equal_area",
                                   projection=proj4str, width=x_size, height=y_size, area_extent=area_extent,
                                   units="deg")
        return area_def

    def _get_geographic_grid(self):
        """Set up the EASE grid."""
        from pyresample import create_area_def

        x_size = self["/dimension/lon"]
        y_size = self["/dimension/lat"]
        lat_0 = self["lat"].min()
        lon_0 = self["lon"].min()
        lat_1 = self["lat"].max()
        lon_1 = self["lon"].max()
        area_extent = [lon_0, lat_1, lon_1, lat_0]
        area_def = create_area_def(area_id="osisaf_geographic_area",
                                   description="osisaf_geographic_area",
                                   proj_id="osisaf_geographic_area",
                                   projection="+proj=lonlat", width=x_size, height=y_size, area_extent=area_extent,
                                   units="deg")
        return area_def

    def _get_polar_stereographic_grid(self):
        """Set up the polar stereographic grid."""
        from pyresample import create_area_def
        try:
            proj4str = self["Polar_Stereographic_Grid/attr/proj4_string"]
        except KeyError:
            # Some products don't have the proj str, so we construct it ourselves
            sma = self["Polar_Stereographic_Grid/attr/semi_major_axis"]
            smb = self["Polar_Stereographic_Grid/attr/semi_minor_axis"]
            lon_0 = self["Polar_Stereographic_Grid/attr/straight_vertical_longitude_from_pole"]
            lat_0 = self["Polar_Stereographic_Grid/attr/latitude_of_projection_origin"]
            lat_ts = self["Polar_Stereographic_Grid/attr/standard_parallel"]
            proj4str = f"+a={sma} +b={smb} +lat_ts={lat_ts} +lon_0={lon_0} +proj=stere +lat_0={lat_0}"
        x_size = self["/dimension/xc"]
        y_size = self["/dimension/yc"]
        p_lowerleft_lat = self["lat"].values[y_size - 1, 0]
        p_lowerleft_lon = self["lon"].values[y_size - 1, 0]
        p_upperright_lat = self["lat"].values[0, x_size - 1]
        p_upperright_lon = self["lon"].values[0, x_size - 1]
        area_extent = [p_lowerleft_lon, p_lowerleft_lat, p_upperright_lon, p_upperright_lat]
        area_def = create_area_def(area_id="osisaf_polar_stereographic",
                                   description="osisaf_polar_stereographic",
                                   proj_id="osisaf_polar_stereographic",
                                   projection=proj4str, width=x_size, height=y_size, area_extent=area_extent,
                                   units="deg")
        return area_def


    def _get_finfo_grid(self):
        """Get grid in case of filename info being used."""
        if self.filename_info["grid"] == "ease":
            self.area_def = self._get_ease_grid()
            return self.area_def
        elif self.filename_info["grid"] == "polstere" or self.filename_info["grid"] == "stere":
            self.area_def = self._get_polar_stereographic_grid()
            return self.area_def
        else:
            raise ValueError(f"Unknown grid type: {self.filename_info['grid']}")

    def _get_ftype_grid(self):
        """Get grid in case of filetype info being used."""
        if self.filetype_info["file_type"] == "osi_radflux_grid":
            self.area_def = self._get_geographic_grid()
            return self.area_def
        elif self.filetype_info["file_type"] in ["osi_sst", "osi_sea_ice_conc"]:
            self.area_def = self._get_polar_stereographic_grid()
            return self.area_def

    def get_area_def(self, area_id):
        """Get the area definition, which varies depending on file type and structure."""
        if "grid" in self.filename_info:
            return self._get_finfo_grid()
        else:
            return self._get_ftype_grid()


    def _get_ds_units(self, ds_info, var_path):
        """Find the units of the datasets."""
        file_units = ds_info.get("file_units")
        if file_units is None:
            file_units = self.get(var_path + "/attr/units")
            if file_units is None:
                file_units = 1
        return file_units

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        logger.debug(f"Reading {dataset_id['name']} from {self.filename}")
        var_path = ds_info.get("file_key", f"{dataset_id['name']}")

        shape = self[var_path + "/shape"]
        data = self[var_path]
        if shape[0] == 1:
            # Remove the time dimension from dataset
            data = data[0]

        file_units = self._get_ds_units(ds_info, var_path)

        # Try to get the valid limits for the data.
        # Not all datasets have these, so fall back on assuming no limits.
        valid_min = self.get(var_path + "/attr/valid_min")
        valid_max = self.get(var_path + "/attr/valid_max")
        if valid_min is not None and valid_max is not None:
            data = data.where(data >= valid_min)
            data = data.where(data <= valid_max)

        # Try to get the fill value for the data.
        # If there isn't one, assume all remaining pixels are valid.
        fill_value = self.get(var_path + "/attr/_FillValue")
        if fill_value is not None:
            data = data.where(data != fill_value)

        # Try to get the scale and offset for the data.
        # As above, not all datasets have these, so fall back on assuming no limits.
        scale_factor = self.get(var_path + "/attr/scale_factor")
        scale_offset = self.get(var_path + "/attr/add_offset")
        if scale_offset is not None and scale_factor is not None:
            data = (data * scale_factor + scale_offset)

        # Set proper dimension names
        if self.filetype_info["file_type"] == "osi_radflux_grid":
            data = data.rename({"lon": "x", "lat": "y"})
        else:
            data = data.rename({"xc": "x", "yc": "y"})

        ds_info.update({
            "units": ds_info.get("units", file_units),
            "platform_name": self._get_platname(),
            "sensor": self._get_instname()
        })
        ds_info.update(dataset_id.to_dict())
        data.attrs.update(ds_info)
        return data

    def _get_instname(self):
        """Get instrument name."""
        try:
            return self["/attr/instrument_name"]
        except KeyError:
            try:
                return self["/attr/sensor"]
            except KeyError:
                return "unknown_sensor"

    def _get_platname(self):
        """Get platform name."""
        try:
            return self["/attr/platform_name"]
        except KeyError:
            return self["/attr/platform"]

    @staticmethod
    def _parse_datetime(datestr):
        for dt_format in ("%Y-%m-%d %H:%M:%S","%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return dt.datetime.strptime(datestr, dt_format)
            except ValueError:
                continue
        raise ValueError(f"Unsupported date format: {datestr}")

    @property
    def start_time(self):
        """Get the start time."""
        poss_names = ["/attr/start_date", "/attr/start_time", "/attr/time_coverage_start"]
        for name in poss_names:
            start_t = self.get(name)
            if start_t is not None:
                break
        if start_t is None:
            raise ValueError("Unknown start time attribute.")
        return self._parse_datetime(start_t)

    @property
    def end_time(self):
        """Get the end time."""
        poss_names = ["/attr/stop_date", "/attr/stop_time", "/attr/time_coverage_end"]
        for name in poss_names:
            end_t = self.get(name)
            if end_t is not None:
                break
        if end_t is None:
            raise ValueError("Unknown stop time attribute.")
        return self._parse_datetime(end_t)
