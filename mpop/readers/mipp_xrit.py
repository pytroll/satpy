#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2013, 2014, 2015.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to Eumetcast level 1.5 HRIT/LRIT format. Uses the MIPP reader.
"""
import ConfigParser
import os
from pyproj import Proj

from mipp import xrit
from mipp import CalibrationError, ReaderError

from mpop import CONFIG_PATH
import logging
from trollsift.parser import Parser

from mpop.satin.helper_functions import area_defs_to_extent
from mpop.projectable import Projectable

LOGGER = logging.getLogger(__name__)


try:
    # Work around for on demand import of pyresample. pyresample depends
    # on scipy.spatial which memory leaks on multiple imports
    IS_PYRESAMPLE_LOADED = False
    from pyresample import geometry
    from mpop.projector import get_area_def
    IS_PYRESAMPLE_LOADED = True
except ImportError:
    LOGGER.warning("pyresample missing. Can only work in satellite projection")

from mpop.readers import Reader


class XritReader(Reader):

    '''Class for reading XRIT data.
    '''
    pformat = "mipp_xrit"

    def __init__(self, *args, **kwargs):
        Reader.__init__(self, *args, **kwargs)

    def load(self, datasets_to_load, calibrate=True, areas=None, **kwargs):
        """Read imager data from file and return projectables.
        """
        LOGGER.debug("Channels to load: %s" % datasets_to_load)

        # Compulsory global attributes according to CF
        # satscene.info["title"] = (satscene.satname.capitalize() + satscene.number +
        #                           " satellite, " +
        #                           satscene.instrument_name.capitalize() +
        #                           " instrument.")
        # satscene.info["institution"] = "Original data disseminated by EumetCast."
        # satscene.add_to_history("HRIT/LRIT data read by mipp/mpop.")
        # satscene.info["references"] = "No reference."
        # satscene.info["comments"] = "No comment."

        area_converted_to_extent = False
        # FIXME: that's way too specific...
        filename = list(self.filenames)[0]
        pattern = self.file_patterns[0]

        parser = Parser(pattern)

        file_info = parser.parse(filename)

        platforms = {"MSG1": "Meteosat-8",
                     "MSG2": "Meteosat-9",
                     "MSG3": "Meteosat-10",
                     "MSG4": "Meteosat-11",
                     }

        short_name = file_info["platform_shortname"]
        fullname = platforms.get(short_name, short_name)
        projectables = {}
        area_extent = None
        for ds in datasets_to_load:

            # Convert area definitions to maximal area_extent
            if not area_converted_to_extent and areas is not None:
                metadata = xrit.sat.load(fullname, self.start_time,
                                         ds, only_metadata=True)
                # otherwise use the default value (MSG3 extent at
                # lon0=0.0), that is, do not pass default_extent=area_extent
                area_extent = area_defs_to_extent(areas, metadata.proj4_params)
                area_converted_to_extent = True

            try:
                image = xrit.sat.load(fullname,
                                      self.start_time,
                                      ds,
                                      mask=True,
                                      calibrate=calibrate)
                if area_extent:
                    metadata, data = image(area_extent)
                else:
                    metadata, data = image()
            except CalibrationError:
                LOGGER.warning(
                    "Loading non calibrated data since calibration failed.")
                image = xrit.sat.load(fullname,
                                      self.start_time,
                                      ds,
                                      mask=True,
                                      calibrate=False)
                if area_extent:
                    metadata, data = image(area_extent)
                else:
                    metadata, data = image()

            except ReaderError, err:
                # if dataset can't be found, go on with next dataset
                LOGGER.error(str(err))
                continue

            projectable = Projectable(data,
                                      name=ds,
                                      units=metadata.calibration_unit,
                                      wavelength_range=self.datasets[ds]["wavelength_range"],
                                      sensor=self.datasets[ds]["sensor"],
                                      start_time=self.start_time)

            # satscene[ds] = data
            #
            # satscene[ds].info['units'] = metadata.calibration_unit
            # satscene[ds].info['satname'] = satscene.satname
            # satscene[ds].info['satnumber'] = satscene.number
            # satscene[ds].info['instrument_name'] = satscene.instrument_name
            # satscene[ds].info['time'] = satscene.time_slot

            # Build an area on the fly from the mipp metadata
            proj_params = getattr(metadata, "proj4_params").split(" ")
            proj_dict = {}
            for param in proj_params:
                key, val = param.split("=")
                proj_dict[key] = val

            if IS_PYRESAMPLE_LOADED:
                # Build area_def on-the-fly
                projectable.info["area"] = geometry.AreaDefinition(
                    #satscene.satname + satscene.instrument_name +
                    str(metadata.area_extent) +
                    str(data.shape),
                    "On-the-fly area",
                    proj_dict["proj"],
                    proj_dict,
                    data.shape[1],
                    data.shape[0],
                    metadata.area_extent)
            else:
                LOGGER.info("Could not build area, pyresample missing...")

            projectables[ds] = projectable
        return projectables



