# Copyright (c) 2023 Satpy developers
# Copyright (c) 2017-2022, European Organisation for the Exploitation of
# Meteorological Satellites (EUMETSAT)
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

# Parts of this module are based on source code obtained from the
# epct_plugin_gis package developed by B-Open Solutions srl for EUMETSAT under
# contract EUM/C0/17/4600001943/0PN and released under Apache License
# Version 2.0, January 2004, http://www.apache.org/licenses/.  The original
# source including revision history and details on authorship can be found at
# https://gitlab.eumetsat.int/open-source/data-tailor-plugins/epct_plugin_gis

"""Reader for EPS level 2 IASI data. Uses xml files as a format description."""

import collections
import datetime
import itertools

import dask.array as da
import numpy as np
import xarray as xr

from . import epsnative_reader
from .file_handlers import BaseFileHandler


class EPSIASIL2FileHandler(BaseFileHandler):
    """EPS level 2 reader for IASI data.

    Reader for the IASI Level 2 combined sounding products in native format.

    Overview of the data including links to the product user guide, product format
    specification, validation reports, and other documents, can be found at the
    EUMETSAT Data Services at https://data.eumetsat.int/product/EO:EUM:DAT:METOP:IASSND02

    Product format specification for IASI L2 data: https://www.eumetsat.int/media/41105

    Generic EPS product format specification: https://www.eumetsat.int/media/40048
    """

    # TODO:
    # - make dask-friendly / lazy
    # - only read variables that are requested

    _nc = None

    def get_dataset(self, dataid, dataset_info):
        """Get dataset."""
        if self._nc is None:
            self._nc = self._get_netcdf_dataset()
        da = self._nc[dataid["name"]]
        da = da * da.attrs.pop("scale_factor", 1)
        return da

    def _get_netcdf_dataset(self):
        """Get full NetCDF dataset."""
        input_product = self.filename
        descriptor = epsnative_reader.assemble_descriptor("IASISND02")
        ipr_sequence = epsnative_reader.read_ipr_sequence(input_product)
        first_mdr_class = [cl for cl in ipr_sequence if cl["class"] == ("mdr", 1)][0]
        first_mdr_offset = first_mdr_class["offset"]
        with open(input_product, "rb") as epsfile_obj:
            data_before_errors, algo_data, errors_data = read_product_data(
                epsfile_obj, descriptor, first_mdr_offset, self.start_time,
                self.end_time)
        giadr = read_giadr(input_product, descriptor)
        ds = create_netcdf_dataset(data_before_errors, algo_data, errors_data, giadr)
        return ds

    def available_datasets(self, configured_datasets):
        """Get available datasets."""
        # FIXME: do this without converting/reading the file — maybe hardcode
        # still?
        common = {"file_type": "iasi_l2_eps", "resolution": 12000}
        if self._nc is None:
            self._nc = self._get_netcdf_dataset()
        for var in self._nc.data_vars:
            yield (True, {"name": var} | common | self._nc[var].attrs)


missing_values = {
    "CO_CP_AIR": -2,
    "CO_CP_CO_A": -2,
    "CO_X_CO": 9.96921e36,
    "CO_H_EIGENVALUES": 9.96921e36,
    "CO_H_EIGENVECTORS": 9.96921e36,
}
var_2nd_dim = {
    "co_cp_air": "co_nbr",
    "co_cp_co_a": "co_nbr",
    "co_x_co": "co_nbr",
    "co_h_eigenvalues": "co_nbr",
    "co_h_eigenvectors": "co_nbr",
    "ozone_error": "nerr",
    "temperature_error": "nerr",
    "water_vapour_error": "nerr",
}
var_3rd_dim = {
    "fg_atmospheric_temperature": "nlt",
    "fg_atmospheric_water_vapour": "nlq",
    "fg_atmospheric_ozone": "nlo",
    "atmospheric_temperature": "nlt",
    "atmospheric_water_vapour": "nlq",
    "atmospheric_ozone": "nlo",
}
# supported major version format
MAJOR_VERSION = 11
NR_FOV = 120


def read_values(eps_fileobj, row, reshape=False):
    """FIXME DOC.

    :param _io.BinaryIO eps_fileobj:
    :param pandas.core.series.Series row:
    :param bool reshape:
    :return numpy.ndarray:
    """
    dtype = epsnative_reader.reckon_dtype(row["TYPE"].strip())
    shape = [int(row[k]) for k in row.keys() if k.lower().startswith("dim")]
    shape = [k for k in shape if k > 1]
    count = int(np.product(shape))
    if isinstance(eps_fileobj, np.memmap):
        values = eps_fileobj[:(np.dtype(dtype).itemsize*count)].view(dtype)
    else:
        values = np.fromfile(eps_fileobj, dtype, count)
    # if np.isfinite(row['SF']):
    #     values = values / 10 ** row['SF']
    if row["TYPE"].startswith("v-"):
        values = values["f1"] * 10.0 ** -values["f0"]
    if row["TYPE"].strip().lower() == "short_cds_time":
        values = [epsnative_reader.scds_to_datetime(*v) for v in values]
        values = values[0] if len(values) == 1 else values
    if reshape:
        try:
            values = values.reshape(tuple(reversed(shape))).T
        except AttributeError:
            values = values
    return values


def read_giadr(input_product, descriptor, **kwargs):
    """FIXME DOC.

    :param input_product:
    :param descriptor:
    :param kwargs:
    :return:
    """
    grh_size = 20  # bytes
    ipr_sequence = epsnative_reader.read_ipr_sequence(input_product)
    giadr_class = [cl for cl in ipr_sequence if "giadr" in cl["class"]][0]
    with open(input_product, "rb") as eps_fileobj:
        eps_fileobj.seek(giadr_class["offset"] + grh_size)
        class_data = collections.OrderedDict()
        giadr_descriptor = descriptor.get(("giadr", 1, 4))
        for _, row in giadr_descriptor.iterrows():
            # empty row or general header
            if np.isnan(row["FIELD_SIZE"]) or row["FIELD"] == "RECORD_HEADER":
                continue
            else:
                class_data[row["FIELD"].strip()] = {
                    "description": row["DESCRIPTION"],
                    "units": row["UNITS"],
                    "values": read_values(eps_fileobj, row, **kwargs),
                }
    return class_data


def set_values_in_mdr_descriptor(epsfile_mmap, mdr_descriptor, mdr_class_offset, field_name):
    """FIXME DOC.

    :param _io.BinaryIO epsfile_obj:
    :param pandas.core.frame.DataFrame mdr_descriptor:
    :param int mdr_class_offset:
    :param str field_name:
    :return list:
    """
    row = mdr_descriptor.loc[mdr_descriptor["FIELD"] == field_name]
    row_idx = row.index[0]
    row = row.squeeze()
    # read values
    epsfile_mmap_subset = epsfile_mmap[mdr_class_offset + int(row["OFFSET"]):]
    value = read_values(epsfile_mmap_subset, row).astype(int)[0]
    # set the read values in MDR-descriptor
    mdr_descriptor.loc[mdr_descriptor["DIM2"] == field_name.lower(), "DIM2"] = value
    return value, row_idx


def update_mdr_descriptor(mdr_descriptor, start_row_idx):
    """FIXME DOC.

    :param pandas.core.frame.DataFrame mdr_descriptor:
    :param int start_row_idx:
    :return pandas.core.frame.DataFrame:
    """
    for idx in range(start_row_idx, len(mdr_descriptor)):
        row = mdr_descriptor.iloc[idx]
        if np.isnan(row["FIELD_SIZE"]) or np.isnan(row["OFFSET"]):
            previous_row = mdr_descriptor.iloc[idx - 1]
            try:
                if np.isnan(row["FIELD_SIZE"]):
                    dims = np.int_(row.loc[["DIM1", "DIM2", "DIM3"]].values)
                    field_size = np.prod(dims) * int(row["TYPE_SIZE"])
                    # set the FILED_SIZE value
                    mdr_descriptor.at[idx, "FIELD_SIZE"] = field_size
                if np.isnan(row["OFFSET"]):
                    # set the OFFSET value
                    offset = previous_row["OFFSET"] + previous_row["FIELD_SIZE"]
                    mdr_descriptor.at[idx, "OFFSET"] = int(offset)
            except ValueError:
                continue
    return mdr_descriptor


def update_mdr_descriptor_for_all_parameters(epsfile_obj, mdr_descriptor, mdr_class_offset):
    """FIXME DOC.

    :param _io.BinaryIo epsfile_obj:
    :param pandas.core.frame.DataFrame mdr_descriptor:
    :param int mdr_class_offset:
    :return (dict, pandas.core.frame.DataFrame):
    """
    params = {}
    for field_name in ["NERR", "CO_NBR", "HNO3_NBR", "O3_NBR"]:
        value, row_idx = set_values_in_mdr_descriptor(
            epsfile_obj, mdr_descriptor, mdr_class_offset, field_name
        )
        params[field_name] = value
        update_mdr_descriptor(mdr_descriptor, row_idx)
    return params, mdr_descriptor


def read_errors(epsfile_obj, mdr_descriptor, mdr_class_offset, max_nerr):
    """FIXME DOC.

    :param _io.BinaryIo epsfile_obj:
    :param pandas.core.frame.DataFrame mdr_descriptor:
    :param int mdr_class_offset:
    :param int max_nerr:
    :return dict:
    """
    missing_value = -2147483646
    fields_to_read = [
        "TEMPERATURE_ERROR",
        "WATER_VAPOUR_ERROR",
        "OZONE_ERROR",
        "SURFACE_Z",
    ]
    errors = {}
    for field in fields_to_read:
        row = mdr_descriptor.loc[mdr_descriptor["FIELD"] == field].squeeze()
        epsfile_mmap_subset = epsfile_obj[mdr_class_offset + int(row["OFFSET"]):]
        values = read_values(epsfile_mmap_subset, row, reshape=True)
        if field != "SURFACE_Z":
            if row["DIM2"] == 0:
                values = np.full((row["DIM1"], max_nerr), missing_value)
            else:
                values = values.reshape(-1, 1) if values.ndim == 1 else values
                values = np.pad(
                    values,
                    ((0, 0), (0, max_nerr - values.shape[1])),
                    mode="constant",
                    constant_values=missing_value,
                )
        errors[field] = values
    return errors


def read_algorithm_sections(
    epsfile_obj, mdr_descriptor, mdr_class_offset, vars_must_be_extend_to_fov, max_nerr
):
    """FIXME DOC.

    Read the following algorithm sections from an MDR record (i.e. row): FORLI-CO, FORLI-HNO3,
    FORLI-O3, BRESCIA-SO2

    :param _io.BinaryIO epsfile_obj:
    :param pandas.core.frame.DataFrame mdr_descriptor:
    :param int mdr_class_offset:
    :param list vars_must_be_extend_to_fov:
    :param int max_nerr:
    :return (dict, dict):
    """
    params, mdr_descriptor = update_mdr_descriptor_for_all_parameters(
        epsfile_obj, mdr_descriptor.copy(), mdr_class_offset
    )
    errors = read_errors(epsfile_obj, mdr_descriptor, mdr_class_offset, max_nerr)
    res = {"params": params}
    # data about HNO3 and O3 are skipped because the NetCDF file does not include those data
    for prefix in ["CO_"]:  # ["CO", "HNO3_", "O3_"]
        param_name = [k for k in params.keys() if k.startswith(prefix)][0]
        if params[param_name] != 0:
            section_rows = mdr_descriptor.loc[mdr_descriptor["FIELD"].str.startswith(prefix)]
            section = {}
            for _, row in section_rows.iterrows():
                epsfile_mmap_subset = epsfile_obj[mdr_class_offset + int(row["OFFSET"]):]
                values = read_values(epsfile_mmap_subset, row, reshape=True)
                if row["FIELD"] in vars_must_be_extend_to_fov:
                    values = values.reshape(-1, 1) if values.ndim == 1 else values
                    values = np.pad(
                        values,
                        ((0, 0), (0, NR_FOV - values.shape[1])),
                        mode="constant",
                        constant_values=missing_values[row["FIELD"]],
                    )
                section[row["FIELD"]] = values
            res[prefix[:-1]] = section
    return res, errors


def reckon_valid_algorithm_sections(algorithms_data):
    """FIXME DOC.

    Return the value for each MDR-descriptor parameter for each row. An algorithm section is
    valid only if at least 1 correlated parameter (e.g 'CO_NBR' for FORLI-CO algorithm) is not
    equal to zero

    :param list algorithms_data:
    :return dict:
    """
    # data about HNO3 and O3 are skipped because the NetCDF file does not include those data
    params_data = {
        "CO_NBR": {"values": [], "is_valid": None},
        # "HNO3_NBR": {"values": [], "is_valid": None},
        # "O3_NBR": {"values": [], "is_valid": None},
    }
    for row in algorithms_data:
        for param_name in params_data:
            params_data[param_name]["values"].append(row["params"][param_name])
    for param_name in params_data:
        total = sum(params_data[param_name]["values"])
        params_data[param_name]["is_valid"] = total > 0
    return params_data


def fill_invalid_rows(algorithms_data):
    """FIXME DOC.

    :param list algorithms_data:
    :return list:
    """
    params_data = reckon_valid_algorithm_sections(algorithms_data)
    for param, data in params_data.items():
        # only valid algorithms will be saved in the NetCDF file
        if data["is_valid"]:
            first_valid_row_idx = np.where(np.array(data["values"]) != 0)[0][0]
            section_name = param.split("_")[0]
            dummy_section = algorithms_data[first_valid_row_idx][section_name].copy()
            for key, values in dummy_section.items():
                dummy_section[key] = np.full(values.shape, np.nan)
            invalid_rows = np.where(np.array(data["values"]) == 0)[0]
            for row_idx in invalid_rows:
                algorithms_data[row_idx][section_name] = dummy_section
    return algorithms_data


def initialize_stacked_algorithms_output(algorithms_data):
    """FIXME DOC.

    :param list algorithms_data:
    :return dict:
    """
    stacked_data = {}
    for algorithm in algorithms_data[0]:
        if algorithm == "params":
            continue
        stacked_data[algorithm] = algorithms_data[0][algorithm].copy()
        for var in stacked_data[algorithm]:
            stacked_data[algorithm][var] = []
    return stacked_data


def stack_algorithm_sections_along_rows(algorithms_data, stacked_data):
    """FIXME DOC.

    :param list algorithms_data:
    :param dict stacked_data:
    :return:
    """
    # collect values from each row
    for idx, row in enumerate(algorithms_data):
        for algorithm in row.keys():
            if algorithm == "params":
                continue
            for var, values in row[algorithm].items():
                stacked_data[algorithm][var].append(values)
        # free memory deleting read values
        algorithms_data[idx] = None
    # stack data along rows
    for algotirthm in stacked_data:
        for var, values in stacked_data[algotirthm].items():
            stacked_data[algotirthm][var] = np.stack(values, axis=0)
    return stacked_data


def get_vars_must_be_extend_to_fov(mdr_descriptor):
    """FIXME DOC.

    :param pandas.core.frame.DataFrame mdr_descriptor:
    :return list:
    """
    co_nbr_rows = list(mdr_descriptor.loc[mdr_descriptor["DIM2"] == "co_nbr", "FIELD"])
    hno3_nbr_rows = list(mdr_descriptor.loc[mdr_descriptor["DIM2"] == "hno3_nbr", "FIELD"])
    o3_nbr_rows = list(mdr_descriptor.loc[mdr_descriptor["DIM2"] == "o3_nbr", "FIELD"])
    vars_must_be_extend_to_fov = list(
        itertools.chain.from_iterable([co_nbr_rows, hno3_nbr_rows, o3_nbr_rows])
    )
    return vars_must_be_extend_to_fov


def stack_non_algorithm_data(data):
    """FIXME DOC.

    :param list data:
    :return dict:
    """
    stacked_data = {k: [] for k in data[0]}
    for _, row in enumerate(data):
        for key in stacked_data:
            stacked_data[key].append(row[key])
    for key, content in stacked_data.items():
        if getattr(content[0], "size", 0) > 10:
            stacked_data[key] = da.stack(content, axis=0)
        else:
            stacked_data[key] = np.stack(content, axis=0)
    return stacked_data


def read_records_before_error_section(epsfile_mmap, mdr_descriptor, mdr_class_offset):
    """FIXME DOC.

    :param _io.BinaryIO epsfile_obj:
    :param mdr_descriptor:
    :param mdr_class_offset:
    :return dict:
    """
    data = {}
    for _, row in mdr_descriptor.iterrows():
        epsfile_mmap_subset = epsfile_mmap[mdr_class_offset + int(row["OFFSET"]):]
        reshape_flag = False if row["FIELD"] in ["EARTH_LOCATION", "ANGULAR_RELATION"] else True
        values = read_values(epsfile_mmap_subset, row, reshape=reshape_flag)
        data[row["FIELD"]] = values
    return data


def datetime_to_second_since_2000(date):
    """FIXME DOC.

    :param datetime.datetime date:
    :return float:
    """
    era = datetime.datetime(2000, 1, 1)
    return (date - era).total_seconds()


def read_nerr_values(epsfile_mmap, mdr_descriptor, mdr_class_offset):
    """FIXME DOC.

    Return a numpy array of all NERR values as red from each MDR class (i.e. data row). The
    maximum values of NERR is used as second dimension of the errors variables (with the exclusion
    of SURFACE_Z variable), so it is necessary read all values to reckon the max.

    :param epsfile_obj:
    :param mdr_descriptor:
    :param mdr_class_offset:
    :return:
    """
    nerr_row = mdr_descriptor.loc[mdr_descriptor["FIELD"] == "NERR"].iloc[0]
    nerr_values = []
    epsfile_mmap_subset = epsfile_mmap[mdr_class_offset:]
    while True:
        grh = epsnative_reader.grh_reader(epsfile_mmap_subset)
        if grh:
            if grh[0:3] == ("mdr", 1, 4):
                new_offset = mdr_class_offset + int(nerr_row["OFFSET"])
                epsfile_mmap_subset = epsfile_mmap[new_offset:]
                nerr_values.append(read_values(
                    epsfile_mmap_subset, nerr_row, reshape=False)[0])
                mdr_class_offset += grh[3]
            epsfile_mmap_subset = epsfile_mmap[mdr_class_offset:]
        else:
            break
    return np.array(nerr_values)


def read_all_rows(epsfile_mmap, descriptor, mdr_class_offset, sensing_start, sensing_stop):
    """FIXME DOC.

    :param _io.BinaryIO epsfile_obj:
    :param dict descriptor:
    :param int mdr_class_offset:
    :param datetime.datetime sensing_start:
    :param datetime.datetime sensing_stop:
    :return (list, list, list):
    """
    mdr_descriptor = descriptor[("mdr", 1, 4)]
    last_constant_row = mdr_descriptor.loc[mdr_descriptor["FIELD"] == "ERROR_DATA_INDEX"].index[0]
    mdr_descr_constant_offsets = mdr_descriptor[1:last_constant_row + 1]
    vars_must_be_extend_to_fov = get_vars_must_be_extend_to_fov(mdr_descriptor)
    max_nerr = read_nerr_values(epsfile_mmap, mdr_descriptor, mdr_class_offset).max()
    algorithms_data = []
    data_before_errors_section = []
    errors_data = []
    epsfile_mmap_subset = epsfile_mmap[mdr_class_offset:]
    while True:
        grh = epsnative_reader.grh_reader(epsfile_mmap_subset)
        if grh:
            if grh[0:3] == ("mdr", 1, 4):
                overlap = reckon_overlap(
                    sensing_start, sensing_stop, grh[-2], grh[-1]
                )
                if overlap > 0:
                    record_start_time = datetime_to_second_since_2000(grh[-2])
                    record_stop_time = datetime_to_second_since_2000(grh[-1])

                    records_before_error_section = read_records_before_error_section(
                       epsfile_mmap, mdr_descr_constant_offsets, mdr_class_offset
                    )
                    records_before_error_section["record_start_time"] = record_start_time
                    records_before_error_section["record_stop_time"] = record_stop_time
                    data_before_errors_section.append(records_before_error_section)
                    algorithm_data, errors = read_algorithm_sections(
                        epsfile_mmap,
                        mdr_descriptor,
                        mdr_class_offset,
                        vars_must_be_extend_to_fov,
                        max_nerr,
                    )
                    algorithms_data.append(algorithm_data)
                    errors_data.append(errors)
                    mdr_class_offset += grh[3]
                else:
                    mdr_class_offset += grh[3]
            else:
                algorithms_data.append("dummy_mdr")
            epsfile_mmap_subset = epsfile_mmap[mdr_class_offset:]
        else:
            break
    return data_before_errors_section, algorithms_data, errors_data


def read_product_data(epsfile_obj, descriptor, mdr_class_offset, sensing_start, sensing_stop):
    """FIXME DOC.

    :param _io.BinaryIO epsfile_obj:
    :param descriptor:
    :param mdr_class_offset:
    :param datetime.datetime sensing_start:
    :param datetime.datetime sensing_stop:
    :return (dict, dict, dict):
    """
    epsfile_mmap = np.memmap(epsfile_obj, mode="r")
    data_before_errors, algorithms_data, errors = read_all_rows(
        epsfile_mmap, descriptor, mdr_class_offset, sensing_start, sensing_stop
    )
    algorithms_data = fill_invalid_rows(algorithms_data)
    stacked_algo_data = initialize_stacked_algorithms_output(algorithms_data)
    stacked_algo_data = stack_algorithm_sections_along_rows(algorithms_data, stacked_algo_data)
    stacked_data_before_errors = stack_non_algorithm_data(data_before_errors)
    stacked_errors_data = stack_non_algorithm_data(errors)
    return stacked_data_before_errors, stacked_algo_data, stacked_errors_data


def add_angular_relations(data_before_errors_section):
    """FIXME DOC.

    :param data_before_errors_section:
    :return:
    """
    angular_relation = data_before_errors_section.pop("ANGULAR_RELATION")
    solar_zenith = angular_relation[:, ::4]
    satellite_zenith = angular_relation[:, 1::4]
    solar_azimuth = angular_relation[:, 2::4]
    satellite_azimuth = angular_relation[:, 3::4]
    data_before_errors_section["solar_zenith"] = solar_zenith
    data_before_errors_section["satellite_zenith"] = satellite_zenith
    data_before_errors_section["solar_azimuth"] = solar_azimuth
    data_before_errors_section["satellite_azimuth"] = satellite_azimuth
    return data_before_errors_section


def add_latitude_longitude(data_before_errors_section):
    """FIXME DOC.

    :param data_before_errors_section:
    :return:
    """
    earth_location = data_before_errors_section.pop("EARTH_LOCATION")
    data_before_errors_section["lat"] = earth_location[:, ::2]
    data_before_errors_section["lon"] = earth_location[:, 1::2]
    return data_before_errors_section


def get_var_dimensions(var_name, values, dimensions):
    """FIXME DOC.

    :param str var_name:
    :param numpy.ndarray values:
    :param dict dimensions:
    :return dict:
    """
    # nlt, nlq, nlo are all equals to 101, so the automatic detection fails
    dims_keys = list(dimensions.keys())
    sizes = list(dimensions.values())
    dims = {}
    for idx, k in enumerate(values.shape):
        if idx == 0:
            dims["y"] = dimensions["y"]
        elif idx == 1:
            if var_name in var_2nd_dim:
                dims[var_2nd_dim[var_name]] = k
            else:
                dims["x"] = k
        else:
            if var_name in var_3rd_dim:
                dims[var_3rd_dim[var_name]] = k
            else:
                dim_name = dims_keys[sizes.index(k)]
                dims[dim_name] = k
    return dims


def add_giadr_variables(dataset, giadr, band_to_record, dimensions, coordinates):
    """FIXME DOC."""
    var_to_read = {
        "pressure_levels_humidity": "nlq",
        "pressure_levels_ozone": "nlo",
        "pressure_levels_temp": "nlt",
        "surface_emissivity_wavelengths": "new",
    }
    for var_name, dim_name in var_to_read.items():
        dims = {dim_name: dimensions[dim_name]}
        coords = {k: coordinates[k] for k in dims if k in coordinates}
        dataset[var_name] = xr.DataArray(
            giadr[var_name.upper()]["values"], dims=dims, coords=coords
        )
        metadata = band_to_record.get(var_name, {}).get("metadata", {})
        if metadata.get("scale_factor"):
            metadata["scale_factor"] = 10 ** -metadata["scale_factor"]
        dataset[var_name].attrs = metadata
    return


def add_errors_variables(dataset, errors_data, band_to_record, dimensions, coordinates):
    """FIXME DOC."""
    for var, values in errors_data.items():
        var_name = var.lower()
        values = np.rollaxis(values, -1, 1)
        dims = get_var_dimensions(var_name, values, dimensions)
        coords = {k: coordinates[k] for k in dims if k in coordinates}
        dataset[var_name] = xr.DataArray(values, dims=dims, coords=coords)
        metadata = band_to_record.get(var_name, {}).get("metadata", {})
        if metadata.get("scale_factor"):
            metadata["scale_factor"] = 10 ** -metadata["scale_factor"]
        dataset[var_name].attrs = metadata
    return


def assemble_dimensions(giadr, nr_rows, max_nerr):
    """See Parameter Table (8.6) in EUM/OPS-EPS/MAN/04/0033.

    :param dict giadr:
    :param int nr_rows:
    :param int max_nerr:
    :return dict:
    """
    dimensions = {
        "x": NR_FOV,
        "y": nr_rows,
        "cloud_formations": 3,
        "co_nbr": NR_FOV,  # pad to FOV number; real value for CO_NBR is provided in MDR
        "nerr": max_nerr,  # max value of NERR values provided in MDR
        "nerro": giadr["NUM_OZONE_PCS"]["values"][0]
        * (giadr["NUM_OZONE_PCS"]["values"][0] + 1)
        / 2,
        "nerrt": giadr["NUM_TEMPERATURE_PCS"]["values"][0]
        * (giadr["NUM_TEMPERATURE_PCS"]["values"][0] + 1)
        / 2,
        "nerrw": giadr["NUM_WATER_VAPOUR_PCS"]["values"][0]
        * (giadr["NUM_WATER_VAPOUR_PCS"]["values"][0] + 1)
        / 2,
        "neva_co": round(giadr["FORLI_NUM_LAYERS_CO"]["values"][0] / 2),
        "neve_co": round(giadr["FORLI_NUM_LAYERS_CO"]["values"][0] / 2)
        * giadr["FORLI_NUM_LAYERS_CO"]["values"][0],
        "new": giadr["NUM_SURFACE_EMISSIVITY_WAVELENGTHS"]["values"][0],
        "nl_co": giadr["FORLI_NUM_LAYERS_CO"]["values"][0],
        "nl_hno3": giadr["FORLI_NUM_LAYERS_HNO3"]["values"][0],
        "nl_o3": giadr["FORLI_NUM_LAYERS_O3"]["values"][0],
        "nl_so2": giadr["BRESCIA_NUM_ALTITUDES_SO2"]["values"][0],
        "nlo": giadr["NUM_PRESSURE_LEVELS_OZONE"]["values"][0],
        "nlq": giadr["NUM_PRESSURE_LEVELS_HUMIDITY"]["values"][0],
        "nlt": giadr["NUM_PRESSURE_LEVELS_TEMP"]["values"][0],
        "npco": giadr["NUM_OZONE_PCS"]["values"][0],
        "npct": giadr["NUM_TEMPERATURE_PCS"]["values"][0],
        "npcw": giadr["NUM_WATER_VAPOUR_PCS"]["values"][0],
    }
    return dimensions


def add_variable_to_nc(ds, var_name, values, dimensions, coordinates, band_to_record):
    """FIXME DOC."""
    dims = get_var_dimensions(var_name, values, dimensions)
    coords = {k: coordinates[k] for k in dims if k in coordinates}
    ds[var_name] = xr.DataArray(values, dims=dims, coords=coords)
    metadata = band_to_record.get(var_name, {}).get("metadata", {})
    if metadata.get("scale_factor") is not None:
        metadata["scale_factor"] = 10.0 ** -metadata["scale_factor"]
    ds[var_name].attrs = metadata
    return ds


def create_netcdf_dataset(data_before_errors_section, algorithms_data, errors_data, giadr):
    """FIXME DOC.

    :param data_before_errors_section:
    :param algorithms_data:
    :param errors_data:
    :param giadr:
    :return:
    """
    band_to_record = epsnative_reader.bands_to_records_reader("IASISND02")
    add_latitude_longitude(data_before_errors_section)
    add_angular_relations(data_before_errors_section)
    nr_rows = data_before_errors_section["lat"].shape[0]
    max_nerr = data_before_errors_section["NERR"].max()
    dimensions = assemble_dimensions(giadr, nr_rows, max_nerr)
    coordinates = {
        "new": np.arange(giadr["NUM_SURFACE_EMISSIVITY_WAVELENGTHS"]["values"]),
        "nl_co": np.arange(giadr["FORLI_NUM_LAYERS_CO"]["values"]),
        "nl_hno3": np.arange(giadr["FORLI_NUM_LAYERS_HNO3"]["values"]),
        "nl_o3": np.arange(giadr["FORLI_NUM_LAYERS_O3"]["values"]),
        "nl_so2": np.arange(giadr["BRESCIA_NUM_ALTITUDES_SO2"]["values"]),
        "nlo": np.arange(giadr["NUM_PRESSURE_LEVELS_OZONE"]["values"]),
        "nlq": np.arange(giadr["NUM_PRESSURE_LEVELS_HUMIDITY"]["values"]),
        "nlt": np.arange(giadr["NUM_PRESSURE_LEVELS_TEMP"]["values"]),
        "npco": np.arange(giadr["NUM_OZONE_PCS"]["values"]),
        "npct": np.arange(giadr["NUM_TEMPERATURE_PCS"]["values"]),
        "npcw": np.arange(giadr["NUM_WATER_VAPOUR_PCS"]["values"]),
    }
    ds = xr.Dataset()
    for var, values in data_before_errors_section.items():
        var_name = var.lower().replace("flg_", "flag_").replace("nerr", "nerr_values")
        if values.ndim > 2:
            values = np.rollaxis(values, -1, 1)
        add_variable_to_nc(ds, var_name, values, dimensions, coordinates, band_to_record)
    for _, content in algorithms_data.items():
        for var, values in content.items():
            var_name = f"{var.lower()}_values" if var == "CO_NBR" else var.lower()
            if values.ndim > 2:
                values = np.rollaxis(values, -1, 1)
            add_variable_to_nc(ds, var_name, values, dimensions, coordinates, band_to_record)
    add_giadr_variables(ds, giadr, band_to_record, dimensions, coordinates)
    add_errors_variables(ds, errors_data, band_to_record, dimensions, coordinates)
    ds["cloud_formation"] = xr.DataArray(np.arange(3), dims={"cloud_formation": 3})
    ds["cloud_formation"].attrs = band_to_record["cloud_formation"]["metadata"]
    return ds


def reckon_overlap(sensing_start, sensing_end, class_start, class_stop):
    """FIXME DOC.

    :param datetime.datetime sensing_start:
    :param datetime.datetime sensing_end:
    :param datetime.datetime class_start:
    :param datetime.datetime class_stop:
    :return float:
    """
    latest_start = max(sensing_start, class_start)
    earliest_end = min(sensing_end, class_stop)
    delta = (earliest_end - latest_start).total_seconds()
    overlap = max(0.0, delta)
    return overlap
