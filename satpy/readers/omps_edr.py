
"""Interface to the NOAA JPSS OMPS EDR format.

This reader supports the official NOAA JPSS OMPS EDR format which are in
the NetCDF4 file format.

Error Flag Filtering
^^^^^^^^^^^^^^^^^^^^

The Total Ozone (V8TOZ) and Total SO2 (V8TOS) files include a variable called
"ErrorFlag" with values marking various retrieval conditions. The
``error_flags_to_keep`` keyword argument keeps only the pixels whose
"ErrorFlag" value is one of the listed values and sets every other pixel to
the fill value (NaN). The same "ErrorFlag" variable is used for every product
in a file, but different products may need different "ErrorFlag" values to be
considered valid. So ``error_flags_to_keep`` is a mapping of variable name to
the "ErrorFlag" values to keep for that variable.

.. code-block:: python

   error_flags_to_keep = {
       "ColumnAmountO3": [0, 1, 2],
       "s_ColumnamountSO2_STL": [0, 1, 5],
   }
   scn = Scene(reader="omps_edr", filenames=[...], reader_kwargs={"error_flags_to_keep": error_flags_to_keep})

In this example any "ColumnAmountO3" pixel whose "ErrorFlag" is not 0, 1, or 2
is set to NaN. Only the variables in the mapping are filtered by "ErrorFlag".
To apply the same filtering to every variable use a
:class:`collections.defaultdict`:

.. code-block:: python

   from collections import defaultdict
   error_flags_to_keep = defaultdict(lambda: [0, 1])
   scn = Scene(reader="omps_edr", filenames=[...], reader_kwargs={"error_flags_to_keep": error_flags_to_keep})

Geolocation variables (longitude and latitude) are never filtered. By default
no "ErrorFlag" filtering is done. Refer to algorithm documentation for more
details on the "ErrorFlag" variable.

Eclipse Filtering
^^^^^^^^^^^^^^^^^

The Total Ozone (V8TOZ) and Total SO2 (V8TOS) files include a variable called
"EclipseFlag" which is 0 for pixels not affected by a solar eclipse. Passing
``filter_eclipse=True`` will set any pixel where "EclipseFlag" is not 0 to the
fill value (NaN) for every variable except geolocation and the "EclipseFlag"
variable itself. Files without an "EclipseFlag" variable are not filtered.

.. code-block:: python

   scn = Scene(reader="omps_edr", filenames=[...], reader_kwargs={"filter_eclipse": True})

By default no eclipse filtering is done.

Negative SO2 Filtering
^^^^^^^^^^^^^^^^^^^^^^

The Total SO2 files (V8TOS) include column amount variables whose valid range
includes negative values (retrieval noise around zero). These negative values
are typically not useful for display or analysis. Passing
``filter_negative_so2=True`` will set any value below 0 to the fill value (NaN)
in addition to the normal valid range filtering. At the time of writing the
affected variables are "s_ColumnamountSO2_PBL", "s_ColumnamountSO2_TRL",
"s_ColumnamountSO2_TRM", and "s_ColumnamountSO2_STL".

.. code-block:: python

   scn = Scene(reader="omps_edr", filenames=[...], reader_kwargs={"filter_negative_so2": True})

By default no filtering of negative values is done.

"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.remote import open_file_or_filename

LOG = logging.getLogger(__name__)


def _rename_unit_attr(attrs: dict) -> dict:
    """Get a copy of ``attrs`` with the non-standard "Unit" attribute renamed to "units".

    The SO2 variables in the V8TOS files use "Unit" instead of the CF standard
    "units" used by the rest of the variables.

    """
    attrs = dict(attrs)
    unit = attrs.pop("Unit", None)
    if unit is not None and "units" not in attrs:
        attrs["units"] = unit
    return attrs


class EDRFileHandler(BaseFileHandler):
    """File handler for NOAA JPSS OMPS EDR files."""

    y_dim_name = "nTimes"
    x_dim_name = "nIFOV"
    error_flag_var_name = "ErrorFlag"
    eclipse_flag_var_name = "EclipseFlag"
    negative_filter_var_prefix = "s_ColumnamountSO2"

    def __init__(
        self,
        filename,
        filename_info,
        filetype_info,
        error_flags_to_keep: Mapping[str, Sequence[int]] | None = None,
        filter_negative_so2: bool = False,
        filter_eclipse: bool = False,
        **kwargs,
    ):
        """Initialize the geo filehandler."""
        super(EDRFileHandler, self).__init__(filename, filename_info, filetype_info)

        # don't copy the mapping so special mappings like a defaultdict are preserved
        self.error_flags_to_keep = error_flags_to_keep
        self.filter_negative_so2 = filter_negative_so2
        self.filter_eclipse = filter_eclipse

        drop_variables = filetype_info.get("drop_variables", None)
        f_obj = open_file_or_filename(self.filename)
        self.nc = xr.open_dataset(
            f_obj,
            decode_cf=filetype_info.get("decode_cf", True),
            mask_and_scale=True,
            drop_variables=drop_variables,
            chunks=-1,  # do one big chunk given size of the data
        )

    @property
    def start_orbit_number(self):
        """Get the start orbit number."""
        return self.nc.start_orbit_number

    @property
    def end_orbit_number(self):
        """Get the end orbit number."""
        return self.nc.end_orbit_number

    @property
    def platform_name(self):
        """Get the platform name."""
        platform_dict = {
            "NPP": "Suomi-NPP",
            "NOAA20": "NOAA-20",
            "NOAA21": "NOAA-21",
            "NOAA22": "NOAA-22",
            "NOAA23": "NOAA-23",
            "NOAA-20": "NOAA-20",
            "NOAA-21": "NOAA-21",
            "NOAA-22": "NOAA-22",
            "NOAA-23": "NOAA-23",
            "J01": "NOAA-20",
            "J02": "NOAA-21",
            # J04 will launch before J03
            "J04": "NOAA-22",
            "J03": "NOAA-23",
        }
        if hasattr(self.nc, "platform"):
            return platform_dict[self.nc.platform]
        return platform_dict[self.nc.platform_name]

    @property
    def sensor_name(self):
        """Get the sensor name."""
        if hasattr(self.nc, "instrument"):
            return self.nc.instrument.lower()
        return self.nc.instrument_name.lower()

    def get_metadata(self, dataset_id, ds_info):
        """Get the metadata."""
        var_path = ds_info.get("file_key", "{}".format(dataset_id["name"]))
        info = _rename_unit_attr(getattr(self.nc[var_path], "attrs", {}))
        info.update(ds_info)

        info.update(
            {
                "platform_name": self.platform_name,
                "sensor": self.sensor_name,
                "orbital_parameters": {
                    "start_orbit": self.start_orbit_number,
                    "end_orbit": self.end_orbit_number,
                },
            }
        )
        info.update(dataset_id.to_dict())
        return info

    def get_dataset(self, dataset_id, ds_info):
        """Get the dataset."""
        var_path = ds_info.get("file_key", "{}".format(dataset_id["name"]))
        metadata = self.get_metadata(dataset_id, ds_info)
        data_arr = self.nc[var_path]
        data_arr = data_arr.rename({self.y_dim_name: "y", self.x_dim_name: "x"})
        data_arr.attrs = _rename_unit_attr(data_arr.attrs)
        data_arr = self._mask_invalid(data_arr, var_path, metadata)
        data_arr.attrs.update(metadata)
        return data_arr

    def _mask_invalid(self, data_arr: xr.DataArray, var_name: str, ds_info: dict) -> xr.DataArray:
        keep_mask = self._flag_keep_mask(var_name, ds_info)
        if keep_mask is not None:
            data_arr.data = np.where(keep_mask, data_arr.data, np.nan)
        # xarray auto mask and scale handled any fills from the file
        valid_range = ds_info.get("valid_range", data_arr.attrs.get("valid_range"))
        if isinstance(valid_range, np.ndarray):
            valid_range = valid_range.tolist()
            # data array attrs are updated with ds_info later
            ds_info["valid_range"] = valid_range
        if "valid_min" in data_arr.attrs and valid_range is None:
            valid_range = (data_arr.attrs["valid_min"], data_arr.attrs["valid_max"])
        valid_cond = None
        if valid_range is not None:
            valid_cond = (valid_range[0] <= data_arr) & (data_arr <= valid_range[1])
        if self._filters_negative_values(var_name):
            neg_cond = data_arr >= 0
            valid_cond = neg_cond if valid_cond is None else (valid_cond & neg_cond)
        if valid_cond is not None:
            # NOTE: may modify attrs in place
            fill_value = self._handle_fill_value(data_arr)
            return data_arr.where(valid_cond, fill_value)
        return data_arr

    def _flag_keep_mask(self, var_name: str, ds_info: dict) -> Any:
        """Get a boolean mask of the pixels to keep based on the flag variables or None to keep everything."""
        if ds_info.get("standard_name") in ("longitude", "latitude"):
            # never mask geolocation
            return None
        keep_mask = None
        error_flag_values = self._error_flag_values_to_keep(var_name)
        if error_flag_values:
            keep_mask = np.isin(self.nc[self.error_flag_var_name].data, error_flag_values)
        if self._filters_eclipse(var_name):
            not_eclipsed = self.nc[self.eclipse_flag_var_name].data == 0
            keep_mask = not_eclipsed if keep_mask is None else (keep_mask & not_eclipsed)
        return keep_mask

    def _error_flag_values_to_keep(self, var_name: str) -> Sequence[int] | None:
        # NOTE: An empty defaultdict is falsy so explicitly check for None
        if self.error_flags_to_keep is None:
            return None
        try:
            # use __getitem__ so things like defaultdict produce their default
            return self.error_flags_to_keep[var_name]
        except KeyError:
            return None

    def _filters_eclipse(self, var_name: str) -> bool:
        return (
            self.filter_eclipse
            and var_name != self.eclipse_flag_var_name
            and self.eclipse_flag_var_name in self.nc
        )

    def _filters_negative_values(self, var_name: str) -> bool:
        return self.filter_negative_so2 and var_name.startswith(self.negative_filter_var_prefix)

    def _handle_fill_value(self, data_arr: xr.DataArray) -> Any:
        if "_FillValue" in data_arr.attrs and not np.issubdtype(data_arr.dtype, np.floating):
            fill_value = data_arr.attrs["_FillValue"]
            if hasattr(fill_value, "shape"):
                fill_value = fill_value.item()
                data_arr.attrs["_FillValue"] = fill_value
        else:
            # fill value is being overwritten with NaN
            data_arr.attrs.pop("_FillValue", None)
            fill_value = xr.core.dtypes.NA
        return fill_value

    def available_datasets(self, configured_datasets=None):
        """Get information of available datasets in this file.

        Args:
            configured_datasets (list): Series of (bool or None, dict) in the
                same way as is returned by this method (see below). The bool
                is whether the dataset is available from at least one
                of the current file handlers. It can also be ``None`` if
                no file handler before us knows how to handle it.
                The dictionary is existing dataset metadata. The dictionaries
                are typically provided from a YAML configuration file and may
                be modified, updated, or used as a "template" for additional
                available datasets. This argument could be the result of a
                previous file handler's implementation of this method.

        Returns:
            Iterator of (bool or None, dict) pairs where dict is the
            dataset's metadata. If the dataset is available in the current
            file type then the boolean value should be ``True``, ``False``
            if we **know** about the dataset but it is unavailable, or
            ``None`` if this file object is not responsible for it.

        """
        # keep track of what variables the YAML has configured, so we don't
        # duplicate entries for them in the dynamic portion
        handled_var_names = set()
        for is_avail, ds_info in configured_datasets or []:
            file_key = ds_info.get("file_key", ds_info["name"])
            # we must add all variables here even if another file handler has
            # claimed the variable. It could be another instance of this file
            # type and we don't want to add that variable dynamically if the
            # other file handler defined it by the YAML definition.
            handled_var_names.add(file_key)
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            if self.file_type_matches(ds_info["file_type"]) is None:
                # this is not the file type for this dataset
                yield None, ds_info
                continue
            yield file_key in self.nc, ds_info

        yield from self._dynamic_variables_from_file(handled_var_names)

    def _dynamic_variables_from_file(self, handled_var_names: set) -> Iterable[tuple[bool, dict]]:
        coords: dict[str, dict] = {}
        for is_avail, ds_info in self._generate_dynamic_metadata(self.nc.variables.keys(), coords):
            var_name = ds_info["file_key"]
            if var_name in handled_var_names and var_name not in ("Longitude", "Latitude"):
                continue
            handled_var_names.add(var_name)
            yield is_avail, ds_info

        for coord_info in coords.values():
            yield True, coord_info

    def _generate_dynamic_metadata(self, variable_names: Iterable[str], coords: dict) -> Iterable[tuple[bool, dict]]:
        ftype = self.filetype_info["file_type"]

        for var_name in variable_names:
            data_arr = self.nc[var_name]
            if data_arr.ndim != 2 or data_arr.dims != (self.y_dim_name, self.x_dim_name):
                # only 2D arrays supported at this time
                continue
            res = 50000  # meters
            coords_names = (f"longitude_{ftype}", f"latitude_{ftype}")
            ds_info = {
                "file_key": var_name,
                "file_type": self.filetype_info["file_type"],
                "name": var_name,
                "resolution": res,
                "coordinates": coords_names,
            }

            is_lon = "longitude" in var_name.lower()
            is_lat = "latitude" in var_name.lower()
            if not (is_lon or is_lat):
                yield True, ds_info
                continue

            ds_info["standard_name"] = "longitude" if is_lon else "latitude"
            ds_info["units"] = "degrees_east" if is_lon else "degrees_north"
            # recursive coordinate/SwathDefinitions are not currently handled well in the base reader
            del ds_info["coordinates"]
            yield True, ds_info

            # "standard" geolocation coordinate (assume shorter variable name is "better")
            new_name = coords_names[int(not is_lon)]
            if new_name not in coords or len(var_name) < len(coords[new_name]["file_key"]):
                ds_info = ds_info.copy()
                ds_info["name"] = new_name
                coords[ds_info["name"]] = ds_info
