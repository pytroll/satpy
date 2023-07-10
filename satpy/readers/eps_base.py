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

"""Reader for eps level 1b data. Uses xml files as a format description."""

import logging

import dask.array as da
import numpy as np
import xarray as xr
from pyresample.geometry import SwathDefinition

from satpy._config import get_config_path
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.xmlformat import XMLFormat
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

C1 = 1.191062e-05  # mW/(m2*sr*cm-4)
C2 = 1.4387863  # K/cm-1


record_class = ["Reserved", "mphr", "sphr",
                "ipr", "geadr", "giadr",
                "veadr", "viadr", "mdr"]


def read_records(filename, format_definition):
    """Read ``filename`` without scaling it afterwards.

    Args:
        filename (str): Filename to read.
        format_definition (str): XML file containing the format definition.
            Must be findable in the Satpy config path.
    """
    format_fn = get_config_path(format_definition)
    form = XMLFormat(format_fn)

    grh_dtype = np.dtype([("record_class", "|i1"),
                          ("INSTRUMENT_GROUP", "|i1"),
                          ("RECORD_SUBCLASS", "|i1"),
                          ("RECORD_SUBCLASS_VERSION", "|i1"),
                          ("RECORD_SIZE", ">u4"),
                          ("RECORD_START_TIME", "S6"),
                          ("RECORD_STOP_TIME", "S6")])

    max_lines = np.floor((CHUNK_SIZE ** 2) / 2048)

    dtypes = []
    cnt = 0
    counts = []
    classes = []
    prev = None
    with open(filename, "rb") as fdes:
        while True:
            grh = np.fromfile(fdes, grh_dtype, 1)
            if grh.size == 0:
                break
            rec_class = record_class[int(grh["record_class"])]
            sub_class = grh["RECORD_SUBCLASS"][0]

            expected_size = int(grh["RECORD_SIZE"])
            bare_size = expected_size - grh_dtype.itemsize
            try:
                the_type = form.dtype((rec_class, sub_class))
                # the_descr = grh_dtype.descr + the_type.descr
            except KeyError:
                the_type = np.dtype([('unknown', 'V%d' % bare_size)])
            the_descr = grh_dtype.descr + the_type.descr
            the_type = np.dtype(the_descr)
            if the_type.itemsize < expected_size:
                padding = [('unknown%d' % cnt, 'V%d' % (expected_size - the_type.itemsize))]
                cnt += 1
                the_descr += padding
            new_dtype = np.dtype(the_descr)
            key = (rec_class, sub_class)
            if key == prev:
                counts[-1] += 1
            else:
                dtypes.append(new_dtype)
                counts.append(1)
                classes.append(key)
                prev = key
            fdes.seek(expected_size - grh_dtype.itemsize, 1)

        sections = {}
        offset = 0
        for dtype, count, rec_class in zip(dtypes, counts, classes):
            fdes.seek(offset)
            if rec_class == ('mdr', 2):
                record = da.from_array(np.memmap(fdes, mode='r', dtype=dtype, shape=count, offset=offset),
                                       chunks=(max_lines,))
            else:
                record = np.fromfile(fdes, dtype=dtype, count=count)
            offset += dtype.itemsize * count
            if rec_class in sections:
                logger.debug('Multiple records for ', str(rec_class))
                sections[rec_class] = np.hstack((sections[rec_class], record))
            else:
                sections[rec_class] = record

    return sections, form


def create_xarray(arr, dims=("y", "x"), attrs=None):
    """Create xarray with correct dimensions."""
    res = arr
    if attrs is None:
        attrs = {}
    res = xr.DataArray(res, dims=dims, attrs=attrs)
    return res


class EPSBaseFileHandler(BaseFileHandler):
    """Base class for EPS level 1b readers."""

    spacecrafts = {"M01": "Metop-B",
                   "M02": "Metop-A",
                   "M03": "Metop-C", }

    xml_conf: str
    mdr_subclass: int

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize FileHandler."""
        super().__init__(filename, filename_info, filetype_info)

        self.area = None
        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']
        self.form = None
        self.scanlines = None
        self.sections = None

    def _read_all(self):
        logger.debug("Reading %s", self.filename)
        self.sections, self.form = read_records(self.filename, self.xml_conf)
        self.scanlines = self['TOTAL_MDR']
        if self.scanlines != len(self.sections[('mdr', self.mdr_subclass)]):
            logger.warning("Number of declared records doesn't match number of scanlines in the file.")
            self.scanlines = len(self.sections[('mdr', self.mdr_subclass)])

    def __getitem__(self, key):
        """Get value for given key."""
        for altkey in self.form.scales:
            try:
                try:
                    return self.sections[altkey][key] * self.form.scales[altkey][key]
                except TypeError:
                    val = self.sections[altkey][key].item().decode().split("=")[1]
                    try:
                        return float(val) * self.form.scales[altkey][key].item()
                    except ValueError:
                        return val.strip()
            except (KeyError, ValueError):
                continue
        raise KeyError("No matching value for " + str(key))

    def keys(self):
        """List of reader's keys."""
        keys = []
        for val in self.form.scales.values():
            keys += val.dtype.fields.keys()
        return keys

    def get_lonlats(self):
        """Get lonlats."""
        if self.area is None:
            lons, lats = self.get_full_lonlats()
            self.area = SwathDefinition(lons, lats)
            self.area.name = '_'.join([self.platform_name, str(self.start_time),
                                       str(self.end_time)])
        return self.area

    @property
    def platform_name(self):
        """Get platform name."""
        return self.spacecrafts[self["SPACECRAFT_ID"]]

    @property
    def sensor_name(self):
        """Get sensor name."""
        return self.sensors[self["INSTRUMENT_ID"]]

    @property
    def start_time(self):
        """Get start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get end time."""
        return self._end_time
