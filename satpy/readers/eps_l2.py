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

"""Reader for EPS level 2 data. Uses xml files as a format description."""

import logging

import numpy as np

from satpy.readers.eps_base import EPSBaseFileHandler, create_xarray

from . import epsnative_reader

logger = logging.getLogger(__name__)

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

NR_FOV = 120


class EPSIASIL2FileHandler(EPSBaseFileHandler):
    """EPS level 2 reader for IASI data.

    Reader for the IASI Level 2 combined sounding products in native format.

    Overview of the data including links to the product user guide, product format
    specification, validation reports, and other documents, can be found at the
    EUMETSAT Data Services at https://data.eumetsat.int/product/EO:EUM:DAT:METOP:IASSND02

    """

    xml_conf = "eps_iasil2_9.0.xml"
    mdr_subclass = 1

    def __init__(self, filename, filename_info, filetype_info):
        """Initialise handler."""
        super().__init__(filename, filename_info, filetype_info)
        self.descr = epsnative_reader.assemble_descriptor("IASISND02")

    def get_dataset(self, key, info):
        """Get calibrated data."""
        dsname = key["name"].upper()
        vals = self[dsname]
        dims = get_var_dimensions(dsname, vals, self.dimensions)
        # FIXME: get variable attributes
        return create_xarray(vals, dims=list(dims.keys()))

    def available_datasets(self, configured_datasets=None):
        """Get available datasets."""
        if self.sections is None:
            self._read_all()
        common = {"file_type": "iasi_l2_eps", "resolution": 12000}
        for key in self.keys():
            yield (True, {"name": key.lower()} | common)

    def _read_all(self):
        """Read all variables, lazily."""
        super()._read_all()
        self.giadr = read_giadr(self.filename, self.descr)
        self.dimensions = assemble_dimensions(
                self.giadr,
                self.scanlines,
                self["NUM_ERROR_DATA"].max())  # Correct?


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
        class_data = {}
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


def assemble_dimensions(giadr, nr_rows, max_nerr):
    """See Parameter Table (8.6) in EUM/OPS-EPS/MAN/04/0033.

    :param dict giadr:
    :param int nr_rows:
    :param int max_nerr:
    :return dict:
    """
    dimensions = {
        "across_track": NR_FOV,
        "along_track": nr_rows,
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
            dims["y"] = dimensions["along_track"]
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
