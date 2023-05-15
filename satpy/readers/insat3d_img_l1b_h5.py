"""File handler for Insat 3D L1B data in hdf5 format."""
from contextlib import suppress
from datetime import datetime
from functools import cached_property

import dask.array as da
import numpy as np
import xarray as xr

from satpy.utils import import_error_helper

with import_error_helper("xarray-datatree"):
    from datatree import DataTree

from satpy.readers.file_handlers import BaseFileHandler

LUT_SUFFIXES = {"vis": ("RADIANCE", "ALBEDO"),
                "swir": ("RADIANCE",),
                "mir": ("RADIANCE", "TEMP"),
                "tir1": ("RADIANCE", "TEMP"),
                "tir2": ("RADIANCE", "TEMP"),
                "wv": ("RADIANCE", "TEMP"),
                }

CHANNELS_BY_RESOLUTION = {1000: ["vis", "swir"],
                          4000: ["mir", "tir1", "tir2"],
                          8000: ["wv"],
                          }


def apply_lut(data, lut):
    """Apply a lookup table."""
    return lut[data]


def decode_lut_arr(arr, lut):
    """Decode an array using a lookup table."""
    dtype = lut.dtype
    lut_attrs = lut.attrs

    attrs = arr.attrs
    attrs["units"] = lut_attrs["units"]
    attrs["long_name"] = lut_attrs["long_name"]
    new_darr = da.map_blocks(apply_lut, arr.data, lut=np.asanyarray(lut), dtype=dtype)
    new_arr = xr.DataArray(new_darr, dims=arr.dims, attrs=attrs, coords=arr.coords)
    new_arr = new_arr.where(arr.data != attrs["_FillValue"])
    return new_arr


def get_lonlat_suffix(resolution):
    """Get the lonlat variable suffix from the resolution."""
    if resolution == 1000:
        lonlat_suffix = "_VIS"
    elif resolution == 8000:
        lonlat_suffix = "_WV"
    else:
        lonlat_suffix = ""
    return lonlat_suffix


def open_dataset(filename, resolution=1000):
    """Open a dataset for a given resolution."""
    if resolution not in [1000, 4000, 8000]:
        raise ValueError(f"Resolution {resolution} not available. Available resolutions: 1000, 4000, 8000")

    h5ds = xr.open_dataset(filename, engine="h5netcdf", chunks="auto")
    h5ds_raw = xr.open_dataset(filename, engine="h5netcdf", chunks="auto", mask_and_scale=False)
    ds = xr.Dataset()
    ds.attrs = h5ds.attrs
    for channel in CHANNELS_BY_RESOLUTION[resolution]:
        var_name = "IMG_" + channel.upper()
        channel_data = h5ds_raw[var_name]
        ds[var_name] = channel_data

        for name in [var_name + "_" + suffix for suffix in LUT_SUFFIXES[channel]]:
            lut = h5ds[name]
            decoded = decode_lut_arr(channel_data, lut)
            ds[name] = decoded

        lonlat_suffix = get_lonlat_suffix(resolution)

        for coord in ["Longitude", "Latitude"]:
            var_name = coord + lonlat_suffix
            ds[var_name] = h5ds[var_name]

    ds = _rename_dims(ds)
    return ds


def _rename_dims(ds):
    """Rename dimensions to satpy standards."""
    for x_dim in ["GeoX", "GeoX1", "GeoX2"]:
        with suppress(ValueError):
            ds = ds.rename({x_dim: "x"})
    for y_dim in ["GeoY", "GeoY1", "GeoY2"]:
        with suppress(ValueError):
            ds = ds.rename({y_dim: "y"})
    for lons in ["Longitude_VIS", "Longitude_WV"]:
        with suppress(ValueError):
            ds = ds.rename({lons: "Longitude"})
    for lats in ["Latitude_VIS", "Latitude_WV"]:
        with suppress(ValueError):
            ds = ds.rename({lats: "Latitude"})
    return ds


def open_datatree(filename):
    """Open a datatree."""
    datasets = {}
    for resolution in [1000, 4000, 8000]:
        datasets[str(resolution)] = open_dataset(filename, resolution)
    dt = DataTree.from_dict(datasets)
    dt.attrs = dt["1000"].attrs
    return dt


class Insat3DIMGL1BH5FileHandler(BaseFileHandler):
    """File handler for insat 3d imager data."""

    @property
    def start_time(self):
        """Get the start time."""
        start_time = datetime.strptime(self.datatree.attrs['Acquisition_Start_Time'], '%d-%b-%YT%H:%M:%S')
        return start_time

    @property
    def end_time(self):
        """Get the end time."""
        end_time = datetime.strptime(self.datatree.attrs['Acquisition_End_Time'], '%d-%b-%YT%H:%M:%S')
        return end_time

    @cached_property
    def datatree(self):
        """Create the datatree."""
        return open_datatree(self.filename)

    def get_dataset(self, ds_id, ds_info):
        """Get a data array."""
        resolution = ds_id["resolution"]
        ds = self.datatree[str(resolution)]
        if ds_id["name"] in ["longitude", "latitude"]:
            darr = ds[ds_id["name"].capitalize()]

            return darr

        if ds_id["calibration"] == "counts":
            calibration = ""
        elif ds_id["calibration"] == "radiance":
            calibration = "_RADIANCE"
        elif ds_id["calibration"] == "reflectance":
            calibration = "_ALBEDO"
        elif ds_id["calibration"] == "brightness_temperature":
            calibration = "_TEMP"

        darr = ds["IMG_" + ds_id["name"] + calibration]

        nlat, nlon = ds.attrs['Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude']
        darr.attrs["orbital_parameters"] = dict(satellite_nominal_longitude=float(nlon),
                                                satellite_nominal_latitude=float(nlat),
                                                satellite_nominal_altitude=float(ds.attrs["Nominal_Altitude(km)"]),
                                                satellite_actual_altitude=float(ds.attrs["Observed_Altitude(km)"]))
        darr.attrs["platform_name"] = "insat-3d"
        darr.attrs["sensor"] = "imager"
        darr = darr.squeeze()

        return darr

    def get_area_def(self, ds_id):
        """Get the area definition."""
        from satpy.readers._geos_area import get_area_definition, get_area_extent
        darr = self.get_dataset(ds_id, None)
        shape = darr.shape
        lines = shape[-2]
        cols = shape[-1]

        fov = self.datatree.attrs["Field_of_View(degrees)"]
        cfac = 2 ** 16 / (fov / cols)
        lfac = 2 ** 16 / (fov / lines)

        h = self.datatree.attrs["Observed_Altitude(km)"] * 1000
        # WGS 84
        a = 6378137.0
        b = 6356752.314245

        pdict = {
            'cfac': cfac,
            'lfac': lfac,
            'coff': cols / 2,
            'loff': lines / 2,
            'ncols': cols,
            'nlines': lines,
            'scandir': 'N2S',
            'a': a,
            'b': b,
            'h': h,
            'ssp_lon': 82.0,
            'a_name': "insat3d82",
            'a_desc': "insat3d82",
            'p_id': 'geosmsg'
        }
        area_extent = get_area_extent(pdict)
        adef = get_area_definition(pdict, area_extent)
        return adef
