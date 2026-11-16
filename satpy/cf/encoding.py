"""CF encoding."""
import copy
import logging

import numpy as np
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

logger = logging.getLogger(__name__)

# Keys of a variable's own ``encoding`` that are passed on to xarray. xarray validates encodings passed explicitly
# to ``to_netcdf`` strictly, but silently drops what it can't use from a variable's ``encoding``, for example
# ``source``, ``original_shape`` or ``preferred_chunks`` added when reading a file. So only keys supported by xarray's
# CF encoding and by both the netCDF4 and h5netcdf backends are taken over. ``contiguous`` is left out since it
# conflicts with the chunking and compression this writer may add.
_PASSED_ON_ENCODING_KEYS = frozenset({
    "dtype", "_FillValue", "missing_value", "scale_factor", "add_offset", "units", "calendar",
    "chunksizes", "zlib", "compression", "complevel", "shuffle", "fletcher32", "significant_digits", "quantize_mode",
})

# Keys that can contradict each other when mixed from different sources (e.g. ``zlib`` from the variable's encoding
# and ``compression`` from the ``encoding`` keyword argument). Each group is taken as a whole from the source with
# the highest precedence that sets any of its keys.
_ENCODING_KEY_GROUPS = (
    frozenset({"chunksizes", "contiguous"}),
    frozenset({"zlib", "compression", "complevel"}),
)


def _get_default_chunks(dataset):
    """Get default chunk sizes preserving the current dask chunks."""
    encoding = {}
    for var_name, variable in dataset.variables.items():
        if variable.chunks:
            chunks = tuple(
                np.stack([variable.data.chunksize,
                          variable.shape]).min(axis=0)
            )  # Chunksize may not exceed shape
            encoding[var_name] = {"chunksizes": chunks}
    return encoding


def _get_coordinate_fill_values(dataset):
    """Get fill values for dimension coordinates.

    Avoid _FillValue attribute being added to coordinate variables
    (https://github.com/pydata/xarray/issues/1865).
    """
    coord_vars = set()
    for data_array in dataset.values():
        coord_vars.update(set(data_array.dims).intersection(data_array.coords))
    return {coord_var: {"_FillValue": None} for coord_var in coord_vars}


def _get_default_time_encoding(dataset):
    """Get default time encoding.

    Default is xarray's CF datetime encoding of the first time.
    """
    try:
        dtnp64 = dataset["time"].data[0]
    except IndexError:
        dtnp64 = dataset["time"].data

    default = CFDatetimeCoder().encode(xr.DataArray(dtnp64))
    return {"units": default.attrs["units"],
            "dtype": "float64",
            "calendar": default.attrs["calendar"]}


def _get_time_bounds_encoding(time_encoding):
    """Get time bounds encoding, making sure that time coordinates and bounds have the same units."""
    return {"units": time_encoding["units"],
            "dtype": "float64",
            "calendar": time_encoding["calendar"],
            "_FillValue": None}  # FUTURE: Not required anymore with xarray-0.14+


def _get_own_encoding(dataset, var_name):
    """Get the part of a variable's own encoding that can be passed on to xarray explicitly.

    Keys that are also attributes of the variable are dropped as xarray refuses to overwrite attributes.
    """
    if var_name not in dataset.variables:
        return {}
    variable = dataset.variables[var_name]
    encoding = {key: value for key, value in variable.encoding.items()
                if key in _PASSED_ON_ENCODING_KEYS and key not in variable.attrs}
    if not _chunksizes_fit(variable, encoding.get("chunksizes")):
        encoding.pop("chunksizes", None)
    return encoding


def _chunksizes_fit(variable, chunksizes):
    """Check if chunk sizes from a variable's encoding can still be used for its data.

    Like xarray, don't use chunk sizes that are larger than the data or that were read from a file whose
    data have since been reshaped, for example by slicing.
    """
    if chunksizes is None:
        return False
    original_shape = tuple(variable.encoding.get("original_shape", variable.shape))
    return (original_shape == variable.shape and len(chunksizes) == variable.ndim and
            all(chunk <= size for chunk, size in zip(chunksizes, variable.shape)))


def _merge_encodings(*encodings):
    """Merge encodings of one variable, later ones taking precedence key by key.

    Keys in each of ``_ENCODING_KEY_GROUPS`` are taken from the last encoding setting any of them.
    """
    merged = {}
    for encoding in encodings:
        for group in _ENCODING_KEY_GROUPS:
            if not group.isdisjoint(encoding):
                for key in group:
                    merged.pop(key, None)
        merged.update(encoding)
    return merged


def _update_encoding_dataset_names(encoding, dataset, numeric_name_prefix):
    """Ensure variable names of the encoding dictionary account for numeric_name_prefix.

    A lot of channel names in satpy starts with a digit.
    When preparing CF-compliant datasets, these channels are prefixed with numeric_name_prefix.

    If variables names in the encoding dictionary are numeric digits, their name is prefixed
    with numeric_name_prefix
    """
    for var_name in list(dataset.variables):
        if not numeric_name_prefix or not var_name.startswith(numeric_name_prefix):
            continue
        orig_var_name = var_name.replace(numeric_name_prefix, "")
        if orig_var_name in encoding:
            encoding[var_name] = encoding.pop(orig_var_name)
    return encoding


def update_encoding(dataset, to_engine_kwargs, numeric_name_prefix="CHANNEL_"):
    """Update encoding.

    Preserve dask chunks, avoid fill values in coordinate variables and make sure that
    time & time bounds have the same units.

    xarray replaces a variable's ``encoding`` attribute with its entry in the ``encoding`` passed to
    ``to_netcdf``. To not lose it, the ``encoding`` attribute is merged into the returned encoding. The
    precedence per variable and key is, from lowest to highest:

    1. Defaults of the writer: chunk sizes matching the dask chunks and time units.
    2. The variable's ``encoding`` attribute, limited to keys that xarray accepts for writing netCDF.
       Chunk sizes are ignored if they don't fit the data anymore.
    3. No ``_FillValue`` for dimension coordinates.
    4. The ``encoding`` entry in ``to_engine_kwargs``.

    Chunking (``chunksizes``, ``contiguous``) and compression (``zlib``, ``compression``, ``complevel``) are
    each taken as a whole from the source with the highest precedence that sets any of them. Time bounds
    always get the units and calendar of the time coordinate.
    """
    other_to_engine_kwargs = to_engine_kwargs.copy()
    user_encoding = copy.deepcopy(other_to_engine_kwargs.pop("encoding", None) or {})
    user_encoding = _update_encoding_dataset_names(user_encoding, dataset, numeric_name_prefix)
    default_encoding = _get_default_chunks(dataset)
    if "time" in dataset:
        default_encoding.setdefault("time", {}).update(_get_default_time_encoding(dataset))
    coord_encoding = _get_coordinate_fill_values(dataset)

    encoding = {}
    for var_name in default_encoding.keys() | coord_encoding.keys() | user_encoding.keys():
        encoding[var_name] = _merge_encodings(default_encoding.get(var_name, {}),
                                              _get_own_encoding(dataset, var_name),
                                              coord_encoding.get(var_name, {}),
                                              user_encoding.get(var_name, {}))
    if "time" in dataset:
        encoding["time_bnds"] = _get_time_bounds_encoding(encoding["time"])
    return encoding, other_to_engine_kwargs
