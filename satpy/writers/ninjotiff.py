#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""Writer for TIFF images compatible with the NinJo visualization tool (NinjoTIFFs).

NinjoTIFFs can be color images or monochromatic. For monochromatic images, the
physical units and scale and offsets to retrieve the physical values are
provided. Metadata is also recorded in the file.

In order to write ninjotiff files, some metadata needs to be provided to the
writer. Here is an example on how to write a color image::

    chn = "airmass"
    ninjoRegion = load_area("areas.def", "nrEURO3km")

    filenames = glob("data/*__")
    global_scene = Scene(reader="hrit_msg", filenames=filenames)
    global_scene.load([chn])
    local_scene = global_scene.resample(ninjoRegion)
    local_scene.save_dataset(chn, filename="airmass.tif", writer='ninjotiff',
                          sat_id=6300014,
                          chan_id=6500015,
                          data_cat='GPRN',
                          data_source='EUMCAST',
                          nbits=8)

Here is an example on how to write a color image::

    chn = "IR_108"
    ninjoRegion = load_area("areas.def", "nrEURO3km")

    filenames = glob("data/*__")
    global_scene = Scene(reader="hrit_msg", filenames=filenames)
    global_scene.load([chn])
    local_scene = global_scene.resample(ninjoRegion)
    local_scene.save_dataset(chn, filename="msg.tif", writer='ninjotiff',
                          sat_id=6300014,
                          chan_id=900015,
                          data_cat='GORN',
                          data_source='EUMCAST',
                          physic_unit='K',
                          nbits=8)

The metadata to provide to the writer can also be stored in a configuration file
(see pyninjotiff), so that the previous example can be rewritten as::

    chn = "IR_108"
    ninjoRegion = load_area("areas.def", "nrEURO3km")

    filenames = glob("data/*__")
    global_scene = Scene(reader="hrit_msg", filenames=filenames)
    global_scene.load([chn])
    local_scene = global_scene.resample(ninjoRegion)
    local_scene.save_dataset(chn, filename="msg.tif", writer='ninjotiff',
                          # ninjo product name to look for in .cfg file
                          ninjo_product_name="IR_108",
                          # custom configuration file for ninjo tiff products
                          # if not specified PPP_CONFIG_DIR is used as config file directory
                          ninjo_product_file="/config_dir/ninjotiff_products.cfg")


.. _ninjotiff: http://www.ssec.wisc.edu/~davidh/polar2grid/misc/NinJo_Satellite_Import_Formats.html

"""

import logging

import numpy as np
import pyninjotiff.ninjotiff as nt
import xarray as xr
from trollimage.xrimage import invert_scale_offset

from satpy.writers import ImageWriter

logger = logging.getLogger(__name__)


def convert_units(dataset, in_unit, out_unit):
    """Convert units of *dataset*.

    Convert dataset units for the benefit of writing NinJoTIFF.  The main
    background here is that NinJoTIFF would like brightness temperatures in °C,
    but satellite data files are in K.  For simplicity of implementation, this
    function can only convert from K to °C.

    This function will convert input data from K to °C and write the new unit
    in the ``"units"`` attribute.  When output and input units are equal, it
    returns the input dataset.

    Args:
        dataset (xarray DataArray):
            Dataarray for which to convert the units.
        in_unit (str):
            Unit for input data.
        out_unit (str):
            Unit for output data.

    Returns:
        dataset, possibly with new units.
    """
    if in_unit == out_unit:
        return dataset

    if in_unit.lower() in {"k", "kelvin"} and out_unit.lower() in {"c", "celsius"}:
        logger.debug("Converting temperature units from K to °C")
        with xr.set_options(keep_attrs=True):
            new_dataset = dataset - 273.15
        new_dataset.attrs["units"] = out_unit
        return new_dataset

    # Other units not implemented.  Before Satpy 0.16.1 there was a
    # non-working implementation based on pint here.

    raise NotImplementedError(
            "NinJoTIFF unit conversion only implemented between K and C, not "
            f"between {in_unit!s} and {out_unit!s}")


class NinjoTIFFWriter(ImageWriter):
    """Writer for NinjoTiff files."""

    def __init__(self, tags=None, **kwargs):
        """Inititalize the writer."""
        ImageWriter.__init__(
            self, default_config_filename="writers/ninjotiff.yaml", **kwargs
        )

        self.tags = self.info.get("tags", None) if tags is None else tags
        if self.tags is None:
            self.tags = {}
        elif not isinstance(self.tags, dict):
            # if it's coming from a config file
            self.tags = dict(tuple(x.split("=")) for x in self.tags.split(","))

    def save_image(self, img, filename=None, compute=True, **kwargs):  # floating_point=False,
        """Save the image to the given *filename* in ninjotiff_ format.

        .. _ninjotiff: http://www.ssec.wisc.edu/~davidh/polar2grid/misc/NinJo_Satellite_Import_Formats.html
        """
        filename = filename or self.get_filename(**img.data.attrs)
        if img.mode.startswith("L") and (
            "ch_min_measurement_unit" not in kwargs
            or "ch_max_measurement_unit" not in kwargs
        ):
            try:
                scale, offset = img.get_scaling_from_history()
                scale, offset = invert_scale_offset(scale, offset)
            except ValueError as err:
                logger.warning(str(err))
            else:
                try:
                    # Here we know that the data if the image is scaled between 0 and 1
                    dmin = offset
                    dmax = scale + offset
                    ch_min_measurement_unit, ch_max_measurement_unit = np.minimum(dmin, dmax), np.maximum(dmin, dmax)
                    kwargs["ch_min_measurement_unit"] = ch_min_measurement_unit
                    kwargs["ch_max_measurement_unit"] = ch_max_measurement_unit
                except KeyError:
                    raise NotImplementedError(
                        "Don't know how to handle non-scale/offset-based enhancements yet."
                    )
        if img.mode.startswith("P"):
            img.data = img.data.astype(np.uint8)
        return nt.save(img, filename, data_is_scaled_01=True, compute=compute, **kwargs)

    def save_dataset(
        self, dataset, filename=None, fill_value=None, compute=True,
        convert_temperature_units=True, **kwargs
    ):
        """Save a dataset to ninjotiff format.

        This calls `save_image` in turn, but first preforms some unit conversion
        if necessary and desired.  Unit conversion can be suppressed by passing
        ``convert_temperature_units=False``.
        """
        nunits = kwargs.get("physic_unit", None)
        if nunits is None:
            try:
                options = nt.get_product_config(
                    kwargs["ninjo_product_name"], True, kwargs["ninjo_product_file"]
                )
                nunits = options["physic_unit"]
            except KeyError:
                pass
        if nunits is not None:
            try:
                units = dataset.attrs["units"]
            except KeyError:
                logger.warning(
                    "Saving to physical ninjo file without units defined in dataset!"
                )
            else:
                if convert_temperature_units:
                    dataset = convert_units(dataset, units, nunits)
                else:
                    logger.debug("Omitting unit conversion")
        return super(NinjoTIFFWriter, self).save_dataset(
            dataset, filename=filename, compute=compute, fill_value=fill_value, **kwargs
        )
