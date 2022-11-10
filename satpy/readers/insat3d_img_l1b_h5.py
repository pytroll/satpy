"""File handler for Insat 3D L1B data in hdf5 format."""

from functools import partial

import h5netcdf
import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.coding.variables import CFScaleOffsetCoder, lazy_elemwise_func, unpack_for_decoding
from xarray.core import indexing

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


def decode_lut(var, lut):
    """Decode a variable using a lookup table."""
    dtype = lut.dtype
    lut_attrs = lut.attrs
    dims, data, attrs, encoding = unpack_for_decoding(var)
    transform = partial(apply_lut, lut=np.asanyarray(lut))
    attrs["units"] = lut_attrs["units"]
    attrs["long_name"] = lut_attrs["long_name"]
    return xr.Variable(dims, lazy_elemwise_func(data, transform, dtype), attrs, encoding)


class H5Array(BackendArray):
    """Wrapper for h5netcdf variables."""

    def __init__(self, array):
        """Set up the wrapper."""
        self.shape = array.shape
        self.dtype = array.dtype
        self.array = array

    def __getitem__(self, key):
        """Get a slice of data."""
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key):
        return self.array[key]


def _create_variable_from_h5_array(h5_arr):
    chunks = dict(zip(h5_arr.dimensions, h5_arr.chunks))
    var = xr.Variable(h5_arr.dimensions,
                      indexing.LazilyIndexedArray(H5Array(h5_arr)),
                      encoding={"preferred_chunks": chunks},
                      attrs=h5_arr.attrs)
    return var


def get_lonlat_suffix(resolution):
    """Get the lonlat variable suffix from the resolution."""
    if resolution == 1000:
        lonlat_suffix = "_VIS"
    elif resolution == 8000:
        lonlat_suffix = "_WV"
    else:
        lonlat_suffix = ""
    return lonlat_suffix


class I3DBackend(BackendEntrypoint):
    """Backend for Insat 3D in hdf5 format."""

    def open_dataset(self, filename, *, drop_variables=None, resolution=None):
        """Open the dataset."""
        h5f = h5netcdf.File(filename, mode="r")
        ds = xr.Dataset()
        scale_offset_coder = CFScaleOffsetCoder()
        if resolution in [1000, 4000, 8000]:
            for channel in CHANNELS_BY_RESOLUTION[resolution]:
                var_name = "IMG_" + channel.upper()
                var = _create_variable_from_h5_array(h5f[var_name])
                ds[var_name] = var

                for name in [var_name + "_" + suffix for suffix in LUT_SUFFIXES[channel]]:
                    lut = h5f[name]
                    decoded = decode_lut(var, lut)
                    ds[name] = decoded

                lonlat_suffix = get_lonlat_suffix(resolution)

                for coord in ["Longitude", "Latitude"]:
                    var_name = coord + lonlat_suffix
                    var = _create_variable_from_h5_array(h5f[var_name])
                    ds[coord] = scale_offset_coder.decode(var)
        else:
            raise ValueError(f"Resolution {resolution} not availble. Available resolutions: 1000, 4000, 8000")

        # this makes sure the file isn't closed at the end of the function.
        ds.attrs["_filehandle"] = h5f
        ds.attrs.update(h5f.attrs)
        return ds


class Insat3DIMGL1BH5FileHandler(BaseFileHandler):
    """File handler for insat 3d imager data."""

    def get_dataset(self, ds_id, ds_info):
        """Get a data array."""
        resolution = ds_id["resolution"]
        ds = xr.open_dataset(self.filename, engine=I3DBackend, chunks={}, resolution=resolution)
        # this makes sure the file isn't closed at the end of the function.
        self._h5filehandle = ds.attrs["_filehandle"]

        if ds_id["name"] in ["Longitude", "Latitude"]:
            return ds[ds_id["name"]]

        if ds_id["calibration"] == "counts":
            calibration = ""
        elif ds_id["calibration"] == "radiance":
            calibration = "_RADIANCE"
        elif ds_id["calibration"] == "reflectance":
            calibration = "_ALBEDO"
        darr = ds["IMG_" + ds_id["name"] + calibration]
        return darr

    def get_area_def(self, ds_id):
        """Get the area definition."""
        from satpy.readers._geos_area import get_area_definition, get_area_extent
        resolution = ds_id["resolution"]
        ds = xr.open_dataset(self.filename, engine=I3DBackend, chunks={}, resolution=resolution)
        darr = ds["IMG_" + ds_id.name]
        shape = darr.shape
        lines = shape[-2]
        cols = shape[-1]
        fov = ds.attrs["Field_of_View(degrees)"]
        cfac = 2 ** 16 / (fov / cols)
        lfac = 2 ** 16 / (fov / lines)
        # HRV on MSG
        # lfac = cfac = 40927014
        pdict = {
            'cfac': cfac,
            'lfac': lfac,
            'coff': cols / 2,
            'loff': lines / 2,
            'ncols': cols,
            'nlines': lines,
            'scandir': 'N2S',
            'a': 6378137.0,
            'b': 6356752.314245,
            'h': 35778490.219,
            'ssp_lon': 82.0,
            'a_name': "insat3d82",
            'a_desc': "insat3d82",
            'p_id': 'geosmsg'
        }
        area_extent = get_area_extent(pdict)
        adef = get_area_definition(pdict, area_extent)
        return adef
