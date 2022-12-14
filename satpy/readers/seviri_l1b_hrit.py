#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2019 Satpy developers
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
r"""SEVIRI HRIT format reader.

Introduction
------------

The ``seviri_l1b_hrit`` reader reads and calibrates MSG-SEVIRI L1.5 image data in HRIT format. The format is explained
in the `MSG Level 1.5 Image Data Format Description`_. The files are usually named as
follows:

.. code-block:: none

    H-000-MSG4__-MSG4________-_________-PRO______-201903011200-__
    H-000-MSG4__-MSG4________-IR_108___-000001___-201903011200-__
    H-000-MSG4__-MSG4________-IR_108___-000002___-201903011200-__
    H-000-MSG4__-MSG4________-IR_108___-000003___-201903011200-__
    H-000-MSG4__-MSG4________-IR_108___-000004___-201903011200-__
    H-000-MSG4__-MSG4________-IR_108___-000005___-201903011200-__
    H-000-MSG4__-MSG4________-IR_108___-000006___-201903011200-__
    H-000-MSG4__-MSG4________-IR_108___-000007___-201903011200-__
    H-000-MSG4__-MSG4________-IR_108___-000008___-201903011200-__
    H-000-MSG4__-MSG4________-_________-EPI______-201903011200-__

Each image is decomposed into 24 segments (files) for the high-resolution-visible (HRV) channel and 8 segments for other
visible (VIS) and infrared (IR) channels. Additionally, there is one prologue and one epilogue file for the entire scan
which contain global metadata valid for all channels.

Reader Arguments
----------------
Some arguments can be provided to the reader to change its behaviour. These are
provided through the `Scene` instantiation, eg::

  Scene(reader="seviri_l1b_hrit", filenames=fnames, reader_kwargs={'fill_hrv': False})

To see the full list of arguments that can be provided, look into the documentation
of :class:`HRITMSGFileHandler`.

Compression
-----------

This reader accepts compressed HRIT files, ending in ``C_`` as other HRIT readers, see
:class:`satpy.readers.hrit_base.HRITFileHandler`.

This reader also accepts bzipped file with the extension ``.bz2`` for the prologue,
epilogue, and segment files.


Example
-------
Here is an example how to read the data in satpy:

.. code-block:: python

    from satpy import Scene
    import glob

    filenames = glob.glob('data/H-000-MSG4__-MSG4________-*201903011200*')
    scn = Scene(filenames=filenames, reader='seviri_l1b_hrit')
    scn.load(['VIS006', 'IR_108'])
    print(scn['IR_108'])


Output:

.. code-block:: none

    <xarray.DataArray (y: 3712, x: 3712)>
    dask.array<shape=(3712, 3712), dtype=float32, chunksize=(464, 3712)>
    Coordinates:
        acq_time  (y) datetime64[ns] NaT NaT NaT NaT NaT NaT ... NaT NaT NaT NaT NaT
      * x         (x) float64 5.566e+06 5.563e+06 5.56e+06 ... -5.566e+06 -5.569e+06
      * y         (y) float64 -5.566e+06 -5.563e+06 ... 5.566e+06 5.569e+06
    Attributes:
        orbital_parameters:       {'projection_longitude': 0.0, 'projection_latit...
        platform_name:            Meteosat-11
        georef_offset_corrected:  True
        standard_name:            brightness_temperature
        raw_metadata:             {'file_type': 0, 'total_header_length': 6198, '...
        wavelength:               (9.8, 10.8, 11.8)
        units:                    K
        sensor:                   seviri
        platform_name:            Meteosat-11
        start_time:               2019-03-01 12:00:09.716000
        end_time:                 2019-03-01 12:12:42.946000
        area:                     Area ID: some_area_name\\nDescription: On-the-fl...
        name:                     IR_108
        resolution:               3000.403165817
        calibration:              brightness_temperature
        polarization:             None
        level:                    None
        modifiers:                ()
        ancillary_variables:      []

The `filenames` argument can either be a list of strings, see the example above, or a list of
:class:`satpy.readers.FSFile` objects. FSFiles can be used in conjunction with `fsspec`_,
e.g. to handle in-memory data:

.. code-block:: python

    import glob

    from fsspec.implementations.memory import MemoryFile, MemoryFileSystem
    from satpy import Scene
    from satpy.readers import FSFile

    # In this example, we will make use of `MemoryFile`s in a `MemoryFileSystem`.
    memory_fs = MemoryFileSystem()

    # Usually, the data already resides in memory.
    # For explanatory reasons, we will load the files found with glob in memory,
    #  and load the scene with FSFiles.
    filenames = glob.glob('data/H-000-MSG4__-MSG4________-*201903011200*')
    fs_files = []
    for fn in filenames:
        with open(fn, 'rb') as fh:
            fs_files.append(MemoryFile(
                fs=memory_fs,
                path="{}{}".format(memory_fs.root_marker, fn),
                data=fh.read()
            ))
            fs_files[-1].commit()  # commit the file to the filesystem
    fs_files = [FSFile(open_file) for open_file in filenames]  # wrap MemoryFiles as FSFiles
    # similar to the example above, we pass a list of FSFiles to the `Scene`
    scn = Scene(filenames=fs_files, reader='seviri_l1b_hrit')
    scn.load(['VIS006', 'IR_108'])
    print(scn['IR_108'])


Output:

.. code-block:: none

    <xarray.DataArray (y: 3712, x: 3712)>
    dask.array<shape=(3712, 3712), dtype=float32, chunksize=(464, 3712)>
    Coordinates:
        acq_time  (y) datetime64[ns] NaT NaT NaT NaT NaT NaT ... NaT NaT NaT NaT NaT
      * x         (x) float64 5.566e+06 5.563e+06 5.56e+06 ... -5.566e+06 -5.569e+06
      * y         (y) float64 -5.566e+06 -5.563e+06 ... 5.566e+06 5.569e+06
    Attributes:
        orbital_parameters:       {'projection_longitude': 0.0, 'projection_latit...
        platform_name:            Meteosat-11
        georef_offset_corrected:  True
        standard_name:            brightness_temperature
        raw_metadata:             {'file_type': 0, 'total_header_length': 6198, '...
        wavelength:               (9.8, 10.8, 11.8)
        units:                    K
        sensor:                   seviri
        platform_name:            Meteosat-11
        start_time:               2019-03-01 12:00:09.716000
        end_time:                 2019-03-01 12:12:42.946000
        area:                     Area ID: some_area_name\\nDescription: On-the-fl...
        name:                     IR_108
        resolution:               3000.403165817
        calibration:              brightness_temperature
        polarization:             None
        level:                    None
        modifiers:                ()
        ancillary_variables:      []


References:
    - `MSG Level 1.5 Image Data Format Description`_

.. _MSG Level 1.5 Image Data Format Description:
    https://www-cdn.eumetsat.int/files/2020-05/pdf_ten_05105_msg_img_data.pdf
.. _fsspec:
    https://filesystem-spec.readthedocs.io
"""

from __future__ import division

import copy
import logging
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr
from pyresample import geometry

import satpy.readers.utils as utils
from satpy import CHUNK_SIZE
from satpy._compat import cached_property
from satpy.readers._geos_area import get_area_definition, get_area_extent, get_geos_area_naming
from satpy.readers.eum_base import get_service_mode, recarray2dict, time_cds_short
from satpy.readers.hrit_base import (
    HRITFileHandler,
    ancillary_text,
    annotation_header,
    base_hdr_map,
    image_data_function,
)
from satpy.readers.seviri_base import (
    CHANNEL_NAMES,
    HRV_NUM_COLUMNS,
    SATNUM,
    NoValidOrbitParams,
    OrbitPolynomialFinder,
    SEVIRICalibrationHandler,
    add_scanline_acq_time,
    create_coef_dict,
    get_cds_time,
    get_satpos,
    mask_bad_quality,
    pad_data_horizontally,
)
from satpy.readers.seviri_l1b_native_hdr import hrit_epilogue, hrit_prologue, impf_configuration

logger = logging.getLogger('hrit_msg')

# MSG implementation:
key_header = np.dtype([('key_number', 'u1'),
                       ('seed', '>f8')])

segment_identification = np.dtype([('GP_SC_ID', '>i2'),
                                   ('spectral_channel_id', '>i1'),
                                   ('segment_sequence_number', '>u2'),
                                   ('planned_start_segment_number', '>u2'),
                                   ('planned_end_segment_number', '>u2'),
                                   ('data_field_representation', '>i1')])

image_segment_line_quality = np.dtype([('line_number_in_grid', '>i4'),
                                       ('line_mean_acquisition',
                                        [('days', '>u2'),
                                         ('milliseconds', '>u4')]),
                                       ('line_validity', 'u1'),
                                       ('line_radiometric_quality', 'u1'),
                                       ('line_geometric_quality', 'u1')])

msg_variable_length_headers = {
    image_segment_line_quality: 'image_segment_line_quality'}

msg_text_headers = {image_data_function: 'image_data_function',
                    annotation_header: 'annotation_header',
                    ancillary_text: 'ancillary_text'}

msg_hdr_map = base_hdr_map.copy()
msg_hdr_map.update({7: key_header,
                    128: segment_identification,
                    129: image_segment_line_quality
                    })


orbit_coef = np.dtype([('StartTime', time_cds_short),
                       ('EndTime', time_cds_short),
                       ('X', '>f8', (8, )),
                       ('Y', '>f8', (8, )),
                       ('Z', '>f8', (8, )),
                       ('VX', '>f8', (8, )),
                       ('VY', '>f8', (8, )),
                       ('VZ', '>f8', (8, ))])

attitude_coef = np.dtype([('StartTime', time_cds_short),
                          ('EndTime', time_cds_short),
                          ('XofSpinAxis', '>f8', (8, )),
                          ('YofSpinAxis', '>f8', (8, )),
                          ('ZofSpinAxis', '>f8', (8, ))])

cuc_time = np.dtype([('coarse', 'u1', (4, )),
                     ('fine', 'u1', (3, ))])


class HRITMSGPrologueEpilogueBase(HRITFileHandler):
    """Base reader for prologue and epilogue files."""

    def __init__(self, filename, filename_info, filetype_info, hdr_info):
        """Initialize the file handler for prologue and epilogue files."""
        super(HRITMSGPrologueEpilogueBase, self).__init__(filename, filename_info, filetype_info, hdr_info)
        self._reduced = None

    def _reduce(self, mda, max_size):
        """Reduce the metadata."""
        if self._reduced is None:
            self._reduced = utils.reduce_mda(mda, max_size=max_size)
        return self._reduced

    def reduce(self, max_size):
        """Reduce the metadata (placeholder)."""
        raise NotImplementedError


class HRITMSGPrologueFileHandler(HRITMSGPrologueEpilogueBase):
    """SEVIRI HRIT prologue reader."""

    def __init__(self, filename, filename_info, filetype_info, calib_mode='nominal',
                 ext_calib_coefs=None, include_raw_metadata=False,
                 mda_max_array_size=None, fill_hrv=None, mask_bad_quality_scan_lines=None):
        """Initialize the reader."""
        super(HRITMSGPrologueFileHandler, self).__init__(filename, filename_info,
                                                         filetype_info,
                                                         (msg_hdr_map,
                                                          msg_variable_length_headers,
                                                          msg_text_headers))
        self.prologue = {}
        self.read_prologue()

        service = filename_info['service']
        if service == '':
            self.mda['service'] = '0DEG'
        else:
            self.mda['service'] = service

    def read_prologue(self):
        """Read the prologue metadata."""
        with utils.generic_open(self.filename, mode="rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.frombuffer(fp_.read(hrit_prologue.itemsize), dtype=hrit_prologue, count=1)
            self.prologue.update(recarray2dict(data))
            try:
                impf = np.frombuffer(fp_.read(impf_configuration.itemsize), dtype=impf_configuration, count=1)[0]
            except ValueError:
                logger.info('No IMPF configuration field found in prologue.')
            else:
                self.prologue.update(recarray2dict(impf))

    @cached_property
    def satpos(self):
        """Get actual satellite position in geodetic coordinates (WGS-84).

        Evaluate orbit polynomials at the start time of the scan.

        Returns: Longitude [deg east], Latitude [deg north] and Altitude [m]
        """
        a, b = self.get_earth_radii()
        start_time = self.prologue['ImageAcquisition'][
            'PlannedAcquisitionTime']['TrueRepeatCycleStart']
        poly_finder = OrbitPolynomialFinder(self.prologue['SatelliteStatus'][
            'Orbit']['OrbitPolynomial'])
        orbit_polynomial = poly_finder.get_orbit_polynomial(start_time)
        return get_satpos(
            orbit_polynomial=orbit_polynomial,
            time=start_time,
            semi_major_axis=a,
            semi_minor_axis=b,
        )

    def get_earth_radii(self):
        """Get earth radii from prologue.

        Returns:
            Equatorial radius, polar radius [m]

        """
        earth_model = self.prologue['GeometricProcessing']['EarthModel']
        a = earth_model['EquatorialRadius'] * 1000
        b = (earth_model['NorthPolarRadius'] +
             earth_model['SouthPolarRadius']) / 2.0 * 1000
        return a, b

    def reduce(self, max_size):
        """Reduce the prologue metadata."""
        return self._reduce(self.prologue, max_size=max_size)


class HRITMSGEpilogueFileHandler(HRITMSGPrologueEpilogueBase):
    """SEVIRI HRIT epilogue reader."""

    def __init__(self, filename, filename_info, filetype_info, calib_mode='nominal',
                 ext_calib_coefs=None, include_raw_metadata=False,
                 mda_max_array_size=None, fill_hrv=None, mask_bad_quality_scan_lines=None):
        """Initialize the reader."""
        super(HRITMSGEpilogueFileHandler, self).__init__(filename, filename_info,
                                                         filetype_info,
                                                         (msg_hdr_map,
                                                          msg_variable_length_headers,
                                                          msg_text_headers))
        self.epilogue = {}
        self.read_epilogue()

        service = filename_info['service']
        if service == '':
            self.mda['service'] = '0DEG'
        else:
            self.mda['service'] = service

    def read_epilogue(self):
        """Read the epilogue metadata."""
        with utils.generic_open(self.filename, mode="rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.frombuffer(fp_.read(hrit_epilogue.itemsize), dtype=hrit_epilogue, count=1)
            self.epilogue.update(recarray2dict(data))

    def reduce(self, max_size):
        """Reduce the epilogue metadata."""
        return self._reduce(self.epilogue, max_size=max_size)


class HRITMSGFileHandler(HRITFileHandler):
    """SEVIRI HRIT format reader.

    **Calibration**

    See :mod:`satpy.readers.seviri_base`.


    **Padding of the HRV channel**

    By default, the HRV channel is loaded padded with no-data, that is it is
    returned as a full-disk dataset. If you want the original, unpadded, data,
    just provide the `fill_hrv` as False in the `reader_kwargs`::

        scene = satpy.Scene(filenames,
                            reader='seviri_l1b_hrit',
                            reader_kwargs={'fill_hrv': False})

    **Metadata**

    See :mod:`satpy.readers.seviri_base`.

    """

    def __init__(self, filename, filename_info, filetype_info,
                 prologue, epilogue, calib_mode='nominal',
                 ext_calib_coefs=None, include_raw_metadata=False,
                 mda_max_array_size=100, fill_hrv=True,
                 mask_bad_quality_scan_lines=True):
        """Initialize the reader."""
        super(HRITMSGFileHandler, self).__init__(filename, filename_info,
                                                 filetype_info,
                                                 (msg_hdr_map,
                                                  msg_variable_length_headers,
                                                  msg_text_headers))

        self.prologue_ = prologue
        self.epilogue_ = epilogue
        self.prologue = prologue.prologue
        self.epilogue = epilogue.epilogue
        self._filename_info = filename_info
        self.include_raw_metadata = include_raw_metadata
        self.mda_max_array_size = mda_max_array_size
        self.fill_hrv = fill_hrv
        self.calib_mode = calib_mode
        self.ext_calib_coefs = ext_calib_coefs or {}
        self.mask_bad_quality_scan_lines = mask_bad_quality_scan_lines

        self._get_header()

    def _get_header(self):
        """Read the header info, and fill the metadata dictionary."""
        earth_model = self.prologue['GeometricProcessing']['EarthModel']
        self.mda['offset_corrected'] = earth_model['TypeOfEarthModel'] == 2

        # Projection
        a, b = self.prologue_.get_earth_radii()
        self.mda['projection_parameters']['a'] = a
        self.mda['projection_parameters']['b'] = b
        ssp = self.prologue['ImageDescription'][
            'ProjectionDescription']['LongitudeOfSSP']
        self.mda['projection_parameters']['SSP_longitude'] = ssp
        self.mda['projection_parameters']['SSP_latitude'] = 0.0

        # Orbital parameters
        self.mda['orbital_parameters']['satellite_nominal_longitude'] = self.prologue['SatelliteStatus'][
            'SatelliteDefinition']['NominalLongitude']
        self.mda['orbital_parameters']['satellite_nominal_latitude'] = 0.0
        try:
            actual_lon, actual_lat, actual_alt = self.prologue_.satpos
            self.mda['orbital_parameters']['satellite_actual_longitude'] = actual_lon
            self.mda['orbital_parameters']['satellite_actual_latitude'] = actual_lat
            self.mda['orbital_parameters']['satellite_actual_altitude'] = actual_alt
        except NoValidOrbitParams as err:
            logger.warning(err)

        # Misc
        self.platform_id = self.prologue["SatelliteStatus"][
            "SatelliteDefinition"]["SatelliteId"]
        self.platform_name = "Meteosat-" + SATNUM[self.platform_id]
        self.mda['platform_name'] = self.platform_name
        service = self._filename_info['service']
        if service == '':
            self.mda['service'] = '0DEG'
        else:
            self.mda['service'] = service
        self.channel_name = CHANNEL_NAMES[self.mda['spectral_channel_id']]

    @property
    def nominal_start_time(self):
        """Get the start time."""
        return self.prologue['ImageAcquisition'][
            'PlannedAcquisitionTime']['TrueRepeatCycleStart']

    @property
    def nominal_end_time(self):
        """Get the end time."""
        return self.prologue['ImageAcquisition'][
            'PlannedAcquisitionTime']['PlannedRepeatCycleEnd']

    @property
    def start_time(self):
        """Get the start time."""
        return self.epilogue['ImageProductionStats'][
            'ActualScanningSummary']['ForwardScanStart']

    @property
    def end_time(self):
        """Get the end time."""
        return self.epilogue['ImageProductionStats'][
            'ActualScanningSummary']['ForwardScanEnd']

    def _get_area_extent(self, pdict):
        """Get the area extent of the file.

        Until December 2017, the data is shifted by 1.5km SSP North and West against the nominal GEOS projection. Since
        December 2017 this offset has been corrected. A flag in the data indicates if the correction has been applied.
        If no correction was applied, adjust the area extent to match the shifted data.

        For more information see Section 3.1.4.2 in the MSG Level 1.5 Image Data Format Description. The correction
        of the area extent is documented in a `developer's memo <https://github.com/pytroll/satpy/wiki/
        SEVIRI-georeferencing-offset-correction>`_.
        """
        aex = get_area_extent(pdict)

        if not self.mda['offset_corrected']:
            # Geo-referencing offset present. Adjust area extent to match the shifted data. Note that we have to adjust
            # the corners in the *opposite* direction, i.e. S-E. Think of it as if the coastlines were fixed and you
            # dragged the image to S-E until coastlines and data area aligned correctly.
            #
            # Although the image is flipped upside-down and left-right, the projection coordinates retain their
            # properties, i.e. positive x/y is East/North, respectively.
            xadj = 1500
            yadj = -1500
            aex = (aex[0] + xadj, aex[1] + yadj,
                   aex[2] + xadj, aex[3] + yadj)

        return aex

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        # Common parameters for both HRV and other channels
        nlines = int(self.mda['number_of_lines'])
        loff = np.float32(self.mda['loff'])
        pdict = {}
        pdict['cfac'] = np.int32(self.mda['cfac'])
        pdict['lfac'] = np.int32(self.mda['lfac'])
        pdict['coff'] = np.float32(self.mda['coff'])

        pdict['a'] = self.mda['projection_parameters']['a']
        pdict['b'] = self.mda['projection_parameters']['b']
        pdict['h'] = self.mda['projection_parameters']['h']
        pdict['ssp_lon'] = self.mda['projection_parameters']['SSP_longitude']

        pdict['nlines'] = nlines
        pdict['ncols'] = int(self.mda['number_of_columns'])
        if (self.prologue['ImageDescription']['Level15ImageProduction']
                         ['ImageProcDirection'] == 0):
            pdict['scandir'] = 'N2S'
        else:
            pdict['scandir'] = 'S2N'

        area_naming_input_dict = {'platform_name': 'msg',
                                  'instrument_name': 'seviri',
                                  'resolution': int(dsid['resolution'])
                                  }
        area_naming = get_geos_area_naming({**area_naming_input_dict,
                                            **get_service_mode('seviri', pdict['ssp_lon'])})

        # Compute area definition for non-HRV channels:
        if dsid['name'] != 'HRV':
            pdict['loff'] = loff - nlines
            aex = self._get_area_extent(pdict)
            pdict['a_name'] = area_naming['area_id']
            pdict['a_desc'] = area_naming['description']
            pdict['p_id'] = ""
            area = get_area_definition(pdict, aex)
            self.area = area
            return self.area

        segment_number = self.mda['segment_sequence_number']

        current_first_line = ((segment_number -
                               self.mda['planned_start_segment_number'])
                              * pdict['nlines'])

        # Or, if we are processing HRV:
        pdict['a_name'] = area_naming['area_id']
        pdict['p_id'] = ""
        bounds = self.epilogue['ImageProductionStats']['ActualL15CoverageHRV'].copy()
        if self.fill_hrv:
            bounds['UpperEastColumnActual'] = 1
            bounds['UpperWestColumnActual'] = HRV_NUM_COLUMNS
            bounds['LowerEastColumnActual'] = 1
            bounds['LowerWestColumnActual'] = HRV_NUM_COLUMNS
            pdict['ncols'] = HRV_NUM_COLUMNS

        upper_south_line = bounds[
            'LowerNorthLineActual'] - current_first_line - 1
        upper_south_line = min(max(upper_south_line, 0), pdict['nlines'])
        lower_coff = (5566 - bounds['LowerEastColumnActual'] + 1)
        upper_coff = (5566 - bounds['UpperEastColumnActual'] + 1)

        # First we look at the lower window
        pdict['nlines'] = upper_south_line
        pdict['loff'] = loff - upper_south_line
        pdict['coff'] = lower_coff
        pdict['a_desc'] = area_naming['description']
        lower_area_extent = self._get_area_extent(pdict)
        lower_area = get_area_definition(pdict, lower_area_extent)

        # Now the upper window
        pdict['nlines'] = nlines - upper_south_line
        pdict['loff'] = loff - pdict['nlines'] - upper_south_line
        pdict['coff'] = upper_coff
        pdict['a_desc'] = area_naming['description']
        upper_area_extent = self._get_area_extent(pdict)
        upper_area = get_area_definition(pdict, upper_area_extent)

        area = geometry.StackedAreaDefinition(lower_area, upper_area)

        self.area = area.squeeze()
        return self.area

    def get_dataset(self, key, info):
        """Get the dataset."""
        res = super(HRITMSGFileHandler, self).get_dataset(key, info)
        res = self.calibrate(res, key['calibration'])

        is_calibration = key['calibration'] in ['radiance', 'reflectance', 'brightness_temperature']
        if (is_calibration and self.mask_bad_quality_scan_lines):  # noqa: E129
            res = self._mask_bad_quality(res)

        if key['name'] == 'HRV' and self.fill_hrv:
            res = self.pad_hrv_data(res)
        self._update_attrs(res, info)
        self._add_scanline_acq_time(res)
        return res

    def pad_hrv_data(self, res):
        """Add empty pixels around the HRV."""
        logger.debug('Padding HRV data to full disk')
        nlines = int(self.mda['number_of_lines'])

        segment_number = self.mda['segment_sequence_number']

        current_first_line = (segment_number
                              - self.mda['planned_start_segment_number']) * nlines
        bounds = self.epilogue['ImageProductionStats']['ActualL15CoverageHRV']

        upper_south_line = bounds[
          'LowerNorthLineActual'] - current_first_line - 1
        upper_south_line = min(max(upper_south_line, 0), nlines)

        data_list = list()
        if upper_south_line > 0:
            # we have some of the lower window
            data_lower = pad_data_horizontally(res[:upper_south_line, :].data,
                                               (upper_south_line, HRV_NUM_COLUMNS),
                                               bounds['LowerEastColumnActual'],
                                               bounds['LowerWestColumnActual'])
            data_list.append(data_lower)

        if upper_south_line < nlines:
            # we have some of the upper window
            data_upper = pad_data_horizontally(res[upper_south_line:, :].data,
                                               (nlines - upper_south_line, HRV_NUM_COLUMNS),
                                               bounds['UpperEastColumnActual'],
                                               bounds['UpperWestColumnActual'])
            data_list.append(data_upper)
        return xr.DataArray(da.vstack(data_list), dims=('y', 'x'), attrs=res.attrs.copy())

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()
        calib = SEVIRICalibrationHandler(
            platform_id=self.platform_id,
            channel_name=self.channel_name,
            coefs=self._get_calib_coefs(self.channel_name),
            calib_mode=self.calib_mode,
            scan_time=self.start_time
        )
        res = calib.calibrate(data, calibration)
        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res

    def _mask_bad_quality(self, data):
        """Mask scanlines with bad quality."""
        line_validity = self.mda['image_segment_line_quality']['line_validity']
        line_radiometric_quality = self.mda['image_segment_line_quality']['line_radiometric_quality']
        line_geometric_quality = self.mda['image_segment_line_quality']['line_geometric_quality']
        data = mask_bad_quality(data, line_validity, line_geometric_quality, line_radiometric_quality)
        return data

    def _get_raw_mda(self):
        """Compile raw metadata to be included in the dataset attributes."""
        # Metadata from segment header (excluding items which vary among the different segments)
        raw_mda = copy.deepcopy(self.mda)
        for key in ('image_segment_line_quality', 'segment_sequence_number', 'annotation_header', 'loff'):
            raw_mda.pop(key, None)

        # Metadata from prologue and epilogue (large arrays removed)
        raw_mda.update(self.prologue_.reduce(self.mda_max_array_size))
        raw_mda.update(self.epilogue_.reduce(self.mda_max_array_size))

        return raw_mda

    def _add_scanline_acq_time(self, dataset):
        """Add scanline acquisition time to the given dataset."""
        tline = self.mda['image_segment_line_quality']['line_mean_acquisition']
        acq_time = get_cds_time(days=tline['days'], msecs=tline['milliseconds'])
        add_scanline_acq_time(dataset, acq_time)

    def _update_attrs(self, res, info):
        """Update dataset attributes."""
        res.attrs['units'] = info['units']
        res.attrs['wavelength'] = info['wavelength']
        res.attrs['standard_name'] = info['standard_name']
        res.attrs['platform_name'] = self.platform_name
        res.attrs['sensor'] = 'seviri'
        res.attrs['nominal_start_time'] = self.nominal_start_time
        res.attrs['nominal_end_time'] = self.nominal_end_time
        res.attrs['orbital_parameters'] = {
            'projection_longitude': self.mda['projection_parameters']['SSP_longitude'],
            'projection_latitude': self.mda['projection_parameters']['SSP_latitude'],
            'projection_altitude': self.mda['projection_parameters']['h']}
        res.attrs['orbital_parameters'].update(self.mda['orbital_parameters'])
        res.attrs['georef_offset_corrected'] = self.mda['offset_corrected']
        if self.include_raw_metadata:
            res.attrs['raw_metadata'] = self._get_raw_mda()

    def _get_calib_coefs(self, channel_name):
        """Get coefficients for calibration from counts to radiance."""
        band_idx = self.mda['spectral_channel_id'] - 1
        coefs_nominal = self.prologue["RadiometricProcessing"][
            "Level15ImageCalibration"]
        coefs_gsics = self.prologue["RadiometricProcessing"]['MPEFCalFeedback']
        radiance_types = self.prologue['ImageDescription'][
                'Level15ImageProduction']['PlannedChanProcessing']
        return create_coef_dict(
            coefs_nominal=(
                coefs_nominal['CalSlope'][band_idx],
                coefs_nominal['CalOffset'][band_idx]
            ),
            coefs_gsics=(
                coefs_gsics['GSICSCalCoeff'][band_idx],
                coefs_gsics['GSICSOffsetCount'][band_idx]
            ),
            ext_coefs=self.ext_calib_coefs.get(channel_name, {}),
            radiance_type=radiance_types[band_idx]
        )


def pad_data(data, final_size, east_bound, west_bound):
    """Pad the data given east and west bounds and the desired size."""
    nlines = final_size[0]
    if west_bound - east_bound != data.shape[1] - 1:
        raise IndexError('East and west bounds do not match data shape')
    padding_east = da.zeros((nlines, east_bound - 1),
                            dtype=data.dtype, chunks=CHUNK_SIZE)
    padding_west = da.zeros((nlines, (final_size[1] - west_bound)),
                            dtype=data.dtype, chunks=CHUNK_SIZE)
    if np.issubdtype(data.dtype, np.floating):
        padding_east = padding_east * np.nan
        padding_west = padding_west * np.nan
    return np.hstack((padding_east, data, padding_west))
