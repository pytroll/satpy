
"""Reader for the EPS-SG Microwave Sounder (MWS) level-1b data.

Documentation: https://www.eumetsat.int/media/44139
"""

import datetime as dt
import logging

import numpy as np
import xarray as xr
from netCDF4 import default_fillvals

from satpy.readers.core.netcdf import NetCDF4FileHandler

logger = logging.getLogger(__name__)


# dict containing all available auxiliary data parameters to be read using the index map. Keys are the
# parameter name and values are the paths to the variable inside the netcdf

AUX_DATA = {
    "scantime_utc": "data/navigation/mws_scantime_utc",
    "solar_azimuth": "data/navigation/mws_solar_azimuth_angle",
    "solar_zenith": "data/navigation/mws_solar_zenith_angle",
    "satellite_azimuth": "data/navigation/mws_satellite_azimuth_angle",
    "satellite_zenith": "data/navigation/mws_satellite_zenith_angle",
    "surface_type": "data/navigation/mws_surface_type",
    "terrain_elevation": "data/navigation/mws_terrain_elevation",
    "mws_lat": "data/navigation/mws_lat",
    "mws_lon": "data/navigation/mws_lon",
}

MWS_CHANNEL_NAMES_TO_NUMBER = {"1": 1, "2": 2, "3": 3, "4": 4,
                               "5": 5, "6": 6, "7": 7, "8": 8,
                               "9": 9, "10": 10, "11": 11, "12": 12,
                               "13": 13, "14": 14, "15": 15, "16": 16,
                               "17": 17, "18": 18, "19": 19, "20": 20,
                               "21": 21, "22": 22, "23": 23, "24": 24}

MWS_CHANNEL_NAMES = list(MWS_CHANNEL_NAMES_TO_NUMBER.keys())
MWS_CHANNELS = set(MWS_CHANNEL_NAMES)

# netCDF attributes describing how the data is packed on disk.  They stop being
# true once the data has been masked and scaled, so they are not passed on.
PACKING_ATTRIBUTES = ("FillValue", "_FillValue", "missing_value",
                      "valid_range", "valid_min", "valid_max",
                      "scale_factor", "add_offset")


def get_channel_index_from_name(chname):
    """Get the MWS channel index from the channel name."""
    chindex = MWS_CHANNEL_NAMES_TO_NUMBER.get(chname, 0) - 1
    if 0 <= chindex < 24:
        return chindex
    raise AttributeError(f"Channel name {chname!r} not supported")


def _get_fill_value(attrs, dtype):
    """Get the value that marks a missing measurement.

    The netCDF ``FillValue``/``_FillValue`` conventions take precedence.  Real
    MWS files use ``missing_value`` instead, and files that provide none of
    them fall back to the netCDF default for the data type.

    """
    for attr_name in ("FillValue", "_FillValue", "missing_value"):
        if attr_name in attrs:
            return attrs[attr_name]
    return default_fillvals.get(dtype.str[1:], np.nan)


def _get_valid_range(attrs):
    """Get the range of valid measurements, in the units they are stored in.

    Real MWS files use ``valid_min``/``valid_max`` rather than ``valid_range``.

    """
    if "valid_range" in attrs:
        return attrs["valid_range"]
    return [attrs.get("valid_min", -np.inf), attrs.get("valid_max", np.inf)]


def mask_and_scale(data, keep_raw_counts=False):
    """Mask missing measurements and apply the scale factor and offset.

    The netCDF packing attributes describe the data as it is stored on disk and
    no longer describe the array once it has been masked and scaled, so they
    are removed rather than passed on.  Data left as raw counts keeps a
    ``_FillValue`` attribute instead, since NaN can only mark a missing
    measurement in a float array.

    """
    attrs = data.attrs
    fill_value = _get_fill_value(attrs, data.dtype)
    valid_min, valid_max = _get_valid_range(attrs)
    scale_factor = attrs.get("scale_factor")
    add_offset = attrs.get("add_offset")

    # the fill value and the valid range describe the data as it is packed, so
    # the masking has to happen before the data is scaled
    new_fill = data.dtype.type(fill_value) if keep_raw_counts else np.float32(np.nan)
    with xr.set_options(keep_attrs=True):
        data = data.where(data != fill_value, new_fill)
        data = data.where((data >= valid_min) & (data <= valid_max), new_fill)
        if not keep_raw_counts:
            data = data.astype(np.float32)
            if scale_factor is not None:
                data = data * np.float32(scale_factor)
            if add_offset is not None:
                data = data + np.float32(add_offset)

    for attr_name in PACKING_ATTRIBUTES:
        data.attrs.pop(attr_name, None)
    if keep_raw_counts:
        data.attrs["_FillValue"] = fill_value
    return data


def _get_aux_data_name_from_dsname(dsname):
    aux_data_name = [key for key in AUX_DATA.keys() if key in dsname]
    if len(aux_data_name) > 0:
        return aux_data_name[0]


class MWSL1BFile(NetCDF4FileHandler):
    """Class implementing the EPS-SG-A1 MWS L1b Filehandler.

    This class implements the European Polar System Second Generation (EPS-SG)
    Microwave Sounder (MWS) Level-1b NetCDF reader.  It is designed to be used
    through the :class:`Scene <satpy.scene.Scene>` class using the :mod:`Scene.load <satpy.scene.Scene.load>`
    method with the reader ``"mws_l1b_nc"``.

    """

    _platform_name_translate = {
        "SGA1": "Metop-SG-A1",
        "SGA2": "Metop-SG-A2",
        "SGA3": "Metop-SG-A3"}

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super().__init__(filename, filename_info,
                         filetype_info,
                         cache_var_size=10000,
                         cache_handle=True)
        logger.debug("Reading: {}".format(self.filename))
        logger.debug("Start: {}".format(self.start_time))
        logger.debug("End: {}".format(self.end_time))

        self._cache = {}

        self._channel_names = MWS_CHANNEL_NAMES

    @property
    def start_time(self):
        """Get start time."""
        return dt.datetime.strptime(self["/attr/sensing_start_time_utc"],
                                 "%Y-%m-%d %H:%M:%S.%f")

    @property
    def end_time(self):
        """Get end time."""
        return dt.datetime.strptime(self["/attr/sensing_end_time_utc"],
                                 "%Y-%m-%d %H:%M:%S.%f")

    @property
    def sensor(self):
        """Get the sensor name."""
        # Satpy sensor names are lowercase by convention
        return self["/attr/instrument"].lower()

    @property
    def platform_name(self):
        """Get the platform name."""
        return self._platform_name_translate.get(self["/attr/spacecraft"])

    @property
    def sub_satellite_longitude_start(self):
        """Get the longitude of sub-satellite point at start of the product."""
        return self["status/satellite/subsat_longitude_start"].data.item()

    @property
    def sub_satellite_latitude_start(self):
        """Get the latitude of sub-satellite point at start of the product."""
        return self["status/satellite/subsat_latitude_start"].data.item()

    @property
    def sub_satellite_longitude_end(self):
        """Get the longitude of sub-satellite point at end of the product."""
        return self["status/satellite/subsat_longitude_end"].data.item()

    @property
    def sub_satellite_latitude_end(self):
        """Get the latitude of sub-satellite point at end of the product."""
        return self["status/satellite/subsat_latitude_end"].data.item()

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using file_key in dataset_info."""
        logger.debug("Reading {} from {}".format(dataset_id["name"], self.filename))

        var_key = dataset_info["file_key"]
        if _get_aux_data_name_from_dsname(dataset_id["name"]) is not None:
            variable = self._get_dataset_aux_data(dataset_id["name"])
        elif any(lb in dataset_id["name"] for lb in MWS_CHANNELS):
            logger.debug(f"Reading in file to get dataset with key {var_key}.")
            variable = self._get_dataset_channel(dataset_id, dataset_info)
        else:
            logger.warning(f"Could not find key {var_key} in NetCDF file, no valid Dataset created")  # noqa: E501
            return None

        variable = self._manage_attributes(variable, dataset_info)
        variable = self._drop_coords(variable)
        variable = self._standardize_dims(variable)
        return variable

    @staticmethod
    def _standardize_dims(variable):
        """Standardize dims to y, x."""
        if "n_scans" in variable.dims:
            variable = variable.rename({"n_fovs": "x", "n_scans": "y"})
        if variable.dims[0] == "x":
            variable = variable.transpose("y", "x")
        return variable

    @staticmethod
    def _drop_coords(variable):
        """Drop coords that are not in dims."""
        for coord in variable.coords:
            if coord not in variable.dims:
                variable = variable.drop_vars(coord)
        return variable

    def _manage_attributes(self, variable, dataset_info):
        """Manage attributes of the dataset."""
        variable.attrs.setdefault("units", None)
        variable.attrs.update(dataset_info)
        variable.attrs.update(self._get_global_attributes())
        return variable

    def _get_dataset_channel(self, key, dataset_info):
        """Load dataset corresponding to channel measurement.

        Load a dataset when the key refers to a measurand, whether uncalibrated
        (counts) or calibrated in terms of brightness temperature or radiance.

        """
        grp_pth = dataset_info["file_key"]
        channel_index = get_channel_index_from_name(key["name"])

        data = self[grp_pth][:, :, channel_index]
        data = mask_and_scale(data, keep_raw_counts=key["calibration"] == "counts")

        # the remaining attributes are added by _manage_attributes
        data.attrs["orbital_parameters"] = {
            "sub_satellite_latitude_start": self.sub_satellite_latitude_start,
            "sub_satellite_longitude_start": self.sub_satellite_longitude_start,
            "sub_satellite_latitude_end": self.sub_satellite_latitude_end,
            "sub_satellite_longitude_end": self.sub_satellite_longitude_end,
        }

        try:
            data.attrs.update(key.to_dict())
        except AttributeError:
            data.attrs.update(key)

        return data

    def _get_dataset_aux_data(self, dsname):
        """Get the auxiliary data arrays using the index map."""
        # Geolocation and navigation data:
        if dsname in ["mws_lat", "mws_lon",
                      "solar_azimuth", "solar_zenith",
                      "satellite_azimuth", "satellite_zenith",
                      "surface_type", "terrain_elevation"]:
            var_key = AUX_DATA.get(dsname)
        else:
            raise NotImplementedError(f"Dataset {dsname!r} not supported!")

        try:
            variable = self[var_key]
        except KeyError:
            logger.exception("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
            raise

        return mask_and_scale(variable)

    def _get_global_attributes(self):
        """Create a dictionary of global attributes."""
        return {
            "filename": self.filename,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "spacecraft_name": self.platform_name,
            "sensor": self.sensor,
            "filename_start_time": self.filename_info["start_time"],
            "filename_end_time": self.filename_info["end_time"],
            "platform_name": self.platform_name,
            "quality_group": self._get_quality_attributes(),
        }

    def _get_quality_attributes(self):
        """Get quality attributes."""
        quality_group = self["quality"]
        quality_dict = {}
        for key in quality_group:
            # Add the values (as Numpy array) of each variable in the group
            # where possible
            try:
                quality_dict[key] = quality_group[key].values
            except ValueError:
                quality_dict[key] = None

        quality_dict.update(quality_group.attrs)
        return quality_dict
