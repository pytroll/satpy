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
in the `MSG Level 1.5 Image Format Description`_. The files are usually named as
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
visible (VIS) and infrared (IR) channels. Additionally there is one prologue and one epilogue file for the entire scan
which contain global metadata valid for all channels.

Reader Arguments
----------------
Some arguments can be provided to the reader to change it's behaviour. These are
provided through the `Scene` instantiation, eg::

  Scene(reader="seviri_l1b_hrit", filenames=fnames, reader_kwargs={'fill_hrv': False})

To see the full list of arguments that can be provided, look into the documentation
of :class:`HRITMSGFileHandler`.

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
        satellite_longitude:      0.0
        satellite_latitude:       0.0
        satellite_altitude:       35785831.0
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


* The ``orbital_parameters`` attribute provides the nominal and actual satellite position, as well as the projection
  centre.
* You can choose between nominal and GSICS calibration coefficients or even specify your own coefficients, see
  :class:`HRITMSGFileHandler`.
* The ``raw_metadata`` attribute provides raw metadata from the prologue, epilogue and segment header. By default,
  arrays with more than 100 elements are excluded in order to limit memory usage. This threshold can be adjusted,
  see :class:`HRITMSGFileHandler`.
* The ``acq_time`` coordinate provides the acquisition time for each scanline. Use a ``MultiIndex`` to enable selection
  by acquisition time:

  .. code-block:: python

      import pandas as pd
      mi = pd.MultiIndex.from_arrays([scn['IR_108']['y'].data, scn['IR_108']['acq_time'].data],
                                     names=('y_coord', 'time'))
      scn['IR_108']['y'] = mi
      scn['IR_108'].sel(time=np.datetime64('2019-03-01T12:06:13.052000000'))


References:
    - `MSG Level 1.5 Image Format Description`_
    - `Radiometric Calibration of MSG SEVIRI Level 1.5 Image Data in Equivalent Spectral Blackbody Radiance`_

.. _MSG Level 1.5 Image Format Description: http://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=
    PDF_TEN_05105_MSG_IMG_DATA&RevisionSelectionMethod=LatestReleased&Rendition=Web

.. _Radiometric Calibration of MSG SEVIRI Level 1.5 Image Data in Equivalent Spectral Blackbody Radiance:
    https://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_TEN_MSG_SEVIRI_RAD_CALIB&
    RevisionSelectionMethod=LatestReleased&Rendition=Web

"""

from __future__ import division

import copy
import logging
from datetime import datetime

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

import satpy.readers.utils as utils
from pyresample import geometry
from satpy import CHUNK_SIZE
from satpy.readers.eum_base import recarray2dict, time_cds_short
from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function)
from satpy.readers.seviri_base import (CALIB, CHANNEL_NAMES, SATNUM,
                                       VIS_CHANNELS, SEVIRICalibrationHandler,
                                       chebyshev, get_cds_time)
from satpy.readers.seviri_l1b_native_hdr import (hrit_epilogue, hrit_prologue,
                                                 impf_configuration)
from satpy.readers._geos_area import get_area_extent, get_area_definition


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


class NoValidOrbitParams(Exception):
    """Exception when validOrbitParameters are missing."""

    pass


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
                 ext_calib_coefs=None, mda_max_array_size=None, fill_hrv=None):
        """Initialize the reader."""
        super(HRITMSGPrologueFileHandler, self).__init__(filename, filename_info,
                                                         filetype_info,
                                                         (msg_hdr_map,
                                                          msg_variable_length_headers,
                                                          msg_text_headers))
        self.prologue = {}
        self.read_prologue()
        self.satpos = None

        service = filename_info['service']
        if service == '':
            self.mda['service'] = '0DEG'
        else:
            self.mda['service'] = service

    def read_prologue(self):
        """Read the prologue metadata."""
        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.fromfile(fp_, dtype=hrit_prologue, count=1)
            self.prologue.update(recarray2dict(data))
            try:
                impf = np.fromfile(fp_, dtype=impf_configuration, count=1)[0]
            except IndexError:
                logger.info('No IMPF configuration field found in prologue.')
            else:
                self.prologue.update(recarray2dict(impf))

    def get_satpos(self):
        """Get actual satellite position in geodetic coordinates (WGS-84).

        Returns: Longitude [deg east], Latitude [deg north] and Altitude [m]
        """
        if self.satpos is None:
            logger.debug("Computing actual satellite position")

            try:
                # Get satellite position in cartesian coordinates
                x, y, z = self._get_satpos_cart()

                # Transform to geodetic coordinates
                geocent = pyproj.Proj(proj='geocent')
                a, b = self.get_earth_radii()
                latlong = pyproj.Proj(proj='latlong', a=a, b=b, units='m')
                lon, lat, alt = pyproj.transform(geocent, latlong, x, y, z)
            except NoValidOrbitParams as err:
                logger.warning(err)
                lon = lat = alt = None

            # Cache results
            self.satpos = lon, lat, alt

        return self.satpos

    def _get_satpos_cart(self):
        """Determine satellite position in earth-centered cartesion coordinates.

        The coordinates as a function of time are encoded in the coefficients of an 8th-order Chebyshev polynomial.
        In the prologue there is one set of coefficients for each coordinate (x, y, z). The coordinates are obtained by
        evalutaing the polynomials at the start time of the scan.

        Returns: x, y, z [m]
        """
        orbit_polynomial = self.prologue['SatelliteStatus']['Orbit']['OrbitPolynomial']

        # Find Chebyshev coefficients for the start time of the scan
        coef_idx = self._find_orbit_coefs()
        tstart = orbit_polynomial['StartTime'][0, coef_idx]
        tend = orbit_polynomial['EndTime'][0, coef_idx]

        # Obtain cartesian coordinates (x, y, z) of the satellite by evaluating the Chebyshev polynomial at the
        # start time of the scan. Express timestamps in microseconds since 1970-01-01 00:00.
        time = self.prologue['ImageAcquisition']['PlannedAcquisitionTime']['TrueRepeatCycleStart']
        time64 = np.datetime64(time).astype('int64')
        domain = [np.datetime64(tstart).astype('int64'),
                  np.datetime64(tend).astype('int64')]
        x = chebyshev(coefs=orbit_polynomial['X'][coef_idx], time=time64, domain=domain)
        y = chebyshev(coefs=orbit_polynomial['Y'][coef_idx], time=time64, domain=domain)
        z = chebyshev(coefs=orbit_polynomial['Z'][coef_idx], time=time64, domain=domain)

        return x*1000, y*1000, z*1000  # km -> m

    def _find_orbit_coefs(self):
        """Find orbit coefficients for the start time of the scan.

        The header entry SatelliteStatus/Orbit/OrbitPolynomial contains multiple coefficients, each
        of them valid for a certain time interval. Find the coefficients which are valid for the
        start time of the scan.

        A manoeuvre is a discontinuity in the orbit parameters. The flight dynamic algorithms are
        not made to interpolate over the time-span of the manoeuvre; hence we have elements
        describing the orbit before a manoeuvre and a new set of elements describing the orbit after
        the manoeuvre. The flight dynamic products are created so that there is an intentional gap
        at the time of the manoeuvre. Also the two pre-manoeuvre elements may overlap. But the
        overlap is not of an issue as both sets of elements describe the same pre-manoeuvre orbit
        (with negligible variations).

        Returns: Corresponding index in the coefficient list.

        """
        time = np.datetime64(self.prologue['ImageAcquisition']['PlannedAcquisitionTime']['TrueRepeatCycleStart'])
        intervals_tstart = self.prologue['SatelliteStatus']['Orbit']['OrbitPolynomial']['StartTime'][0].astype(
            'datetime64[us]')
        intervals_tend = self.prologue['SatelliteStatus']['Orbit']['OrbitPolynomial']['EndTime'][0].astype(
            'datetime64[us]')
        try:
            # Find index of interval enclosing the nominal timestamp of the scan. If there are
            # multiple enclosing intervals, use the most recent one.
            enclosing = np.where(np.logical_and(time >= intervals_tstart, time < intervals_tend))[0]
            most_recent = np.argmax(intervals_tstart[enclosing])
            return enclosing[most_recent]
        except ValueError:
            # No enclosing interval. Instead, find the interval whose centre is closest to the scan's timestamp
            # (but not more than 6 hours apart)
            intervals_centre = intervals_tstart + 0.5 * (intervals_tend - intervals_tstart)
            diffs_us = (time - intervals_centre).astype('i8')
            closest_match = np.argmin(np.fabs(diffs_us))
            if abs(intervals_centre[closest_match] - time) < np.timedelta64(6, 'h'):
                logger.warning('No orbit coefficients valid for {}. Using closest match.'.format(time))
                return closest_match
            else:
                raise NoValidOrbitParams('Unable to find orbit coefficients valid for {}'.format(time))

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
                 ext_calib_coefs=None, mda_max_array_size=None, fill_hrv=None):
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
        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.fromfile(fp_, dtype=hrit_epilogue, count=1)
            self.epilogue.update(recarray2dict(data))

    def reduce(self, max_size):
        """Reduce the epilogue metadata."""
        return self._reduce(self.epilogue, max_size=max_size)


class HRITMSGFileHandler(HRITFileHandler, SEVIRICalibrationHandler):
    """SEVIRI HRIT format reader.

    **Calibration**

    It is possible to choose between two file-internal calibration coefficients for the conversion
    from counts to radiances:

        - Nominal for all channels (default)
        - GSICS for IR channels and nominal for VIS channels

    In order to change the default behaviour, use the ``reader_kwargs`` upon Scene creation::

        import satpy
        import glob

        filenames = glob.glob('H-000-MSG3*')
        scene = satpy.Scene(filenames,
                            reader='seviri_l1b_hrit',
                            reader_kwargs={'calib_mode': 'GSICS'})
        scene.load(['VIS006', 'IR_108'])

    Furthermore, it is possible to specify external calibration coefficients for the conversion from
    counts to radiances. They must be specified in [mW m-2 sr-1 (cm-1)-1]. External coefficients
    take precedence over internal coefficients. If external calibration coefficients are specified
    for only a subset of channels, the remaining channels will be calibrated using the chosen
    file-internal coefficients (nominal or GSICS).

    In the following example we use external calibration coefficients for the ``VIS006`` &
    ``IR_108`` channels, and nominal coefficients for the remaining channels::

        coefs = {'VIS006': {'gain': 0.0236, 'offset': -1.20},
                 'IR_108': {'gain': 0.2156, 'offset': -10.4}}
        scene = satpy.Scene(filenames,
                            reader='seviri_l1b_hrit',
                            reader_kwargs={'ext_calib_coefs': coefs})
        scene.load(['VIS006', 'VIS008', 'IR_108', 'IR_120'])

    In the next example we use we use external calibration coefficients for the ``VIS006`` &
    ``IR_108`` channels, nominal coefficients for the remaining VIS channels and GSICS coefficients
    for the remaining IR channels::

        coefs = {'VIS006': {'gain': 0.0236, 'offset': -1.20},
                 'IR_108': {'gain': 0.2156, 'offset': -10.4}}
        scene = satpy.Scene(filenames,
                            reader='seviri_l1b_hrit',
                            reader_kwargs={'calib_mode': 'GSICS',
                                           'ext_calib_coefs': coefs})
        scene.load(['VIS006', 'VIS008', 'IR_108', 'IR_120'])

    **Raw Metadata**

    By default, arrays with more than 100 elements are excluded from the raw reader metadata to
    limit memory usage. This threshold can be adjusted using the `mda_max_array_size` keyword
    argument::

        scene = satpy.Scene(filenames,
                            reader='seviri_l1b_hrit',
                            reader_kwargs={'mda_max_array_size': 1000})

    **Padding of the HRV channel**

    By default, the HRV channel is loaded padded with no-data, that is it is
    returned as a full-disk dataset. If you want the original, unpadded, data,
    just provide the `fill_hrv` as False in the `reader_kwargs`::

        scene = satpy.Scene(filenames,
                            reader='seviri_l1b_hrit',
                            reader_kwargs={'fill_hrv': False})

    """

    def __init__(self, filename, filename_info, filetype_info,
                 prologue, epilogue, calib_mode='nominal',
                 ext_calib_coefs=None, mda_max_array_size=100, fill_hrv=True):
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
        self.ext_calib_coefs = ext_calib_coefs if ext_calib_coefs is not None else {}
        self.mda_max_array_size = mda_max_array_size
        self.fill_hrv = fill_hrv
        calib_mode_choices = ('NOMINAL', 'GSICS')
        if calib_mode.upper() not in calib_mode_choices:
            raise ValueError('Invalid calibration mode: {}. Choose one of {}'.format(
                calib_mode, calib_mode_choices))
        self.calib_mode = calib_mode.upper()

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
        actual_lon, actual_lat, actual_alt = self.prologue_.get_satpos()
        self.mda['orbital_parameters']['satellite_nominal_longitude'] = self.prologue['SatelliteStatus'][
            'SatelliteDefinition']['NominalLongitude']
        self.mda['orbital_parameters']['satellite_nominal_latitude'] = 0.0
        if actual_lon is not None:
            self.mda['orbital_parameters']['satellite_actual_longitude'] = actual_lon
            self.mda['orbital_parameters']['satellite_actual_latitude'] = actual_lat
            self.mda['orbital_parameters']['satellite_actual_altitude'] = actual_alt

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

        # Compute area definition for non-HRV channels:
        if dsid.name != 'HRV':
            pdict['loff'] = loff - nlines
            aex = self._get_area_extent(pdict)
            pdict['a_name'] = 'geosmsg'
            pdict['a_desc'] = 'MSG/SEVIRI low resolution channel area'
            pdict['p_id'] = 'msg_lowres'
            area = get_area_definition(pdict, aex)
            self.area = area
            return self.area

        segment_number = self.mda['segment_sequence_number']

        current_first_line = ((segment_number -
                               self.mda['planned_start_segment_number'])
                              * pdict['nlines'])

        # Or, if we are processing HRV:
        pdict['a_name'] = 'geosmsg_hrv'
        pdict['p_id'] = 'msg_hires'
        bounds = self.epilogue['ImageProductionStats']['ActualL15CoverageHRV'].copy()
        if self.fill_hrv:
            bounds['UpperEastColumnActual'] = 1
            bounds['UpperWestColumnActual'] = 11136
            bounds['LowerEastColumnActual'] = 1
            bounds['LowerWestColumnActual'] = 11136
            pdict['ncols'] = 11136

        upper_south_line = bounds[
            'LowerNorthLineActual'] - current_first_line - 1
        upper_south_line = min(max(upper_south_line, 0), pdict['nlines'])
        lower_coff = (5566 - bounds['LowerEastColumnActual'] + 1)
        upper_coff = (5566 - bounds['UpperEastColumnActual'] + 1)

        # First we look at the lower window
        pdict['nlines'] = upper_south_line
        pdict['loff'] = loff - upper_south_line
        pdict['coff'] = lower_coff
        pdict['a_desc'] = 'MSG/SEVIRI high resolution channel, lower window'
        lower_area_extent = self._get_area_extent(pdict)
        lower_area = get_area_definition(pdict, lower_area_extent)

        # Now the upper window
        pdict['nlines'] = nlines - upper_south_line
        pdict['loff'] = loff - pdict['nlines'] - upper_south_line
        pdict['coff'] = upper_coff
        pdict['a_desc'] = 'MSG/SEVIRI high resolution channel, upper window'
        upper_area_extent = self._get_area_extent(pdict)
        upper_area = get_area_definition(pdict, upper_area_extent)

        area = geometry.StackedAreaDefinition(lower_area, upper_area)

        self.area = area.squeeze()
        return self.area

    def get_dataset(self, key, info):
        """Get the dataset."""
        res = super(HRITMSGFileHandler, self).get_dataset(key, info)
        res = self.calibrate(res, key.calibration)
        if key.name == 'HRV' and self.fill_hrv:
            res = self.pad_hrv_data(res)

        res.attrs['units'] = info['units']
        res.attrs['wavelength'] = info['wavelength']
        res.attrs['standard_name'] = info['standard_name']
        res.attrs['platform_name'] = self.platform_name
        res.attrs['sensor'] = 'seviri'
        res.attrs['satellite_longitude'] = self.mda[
            'projection_parameters']['SSP_longitude']
        res.attrs['satellite_latitude'] = self.mda[
            'projection_parameters']['SSP_latitude']
        res.attrs['satellite_altitude'] = self.mda['projection_parameters']['h']
        res.attrs['orbital_parameters'] = {
            'projection_longitude': self.mda['projection_parameters']['SSP_longitude'],
            'projection_latitude': self.mda['projection_parameters']['SSP_latitude'],
            'projection_altitude': self.mda['projection_parameters']['h']}
        res.attrs['orbital_parameters'].update(self.mda['orbital_parameters'])
        res.attrs['georef_offset_corrected'] = self.mda['offset_corrected']
        res.attrs['raw_metadata'] = self._get_raw_mda()

        # Add scanline timestamps as additional y-coordinate
        res.coords['acq_time'] = ('y', self._get_timestamps())
        res.coords['acq_time'].attrs['long_name'] = 'Mean scanline acquisition time'

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
            data_lower = pad_data(res[:upper_south_line, :].data,
                                  (upper_south_line, 11136),
                                  bounds['LowerEastColumnActual'],
                                  bounds['LowerWestColumnActual'])
            data_list.append(data_lower)

        if upper_south_line < nlines:
            # we have some of the upper window
            data_upper = pad_data(res[upper_south_line:, :].data,
                                  (nlines - upper_south_line, 11136),
                                  bounds['UpperEastColumnActual'],
                                  bounds['UpperWestColumnActual'])
            data_list.append(data_upper)
        return xr.DataArray(da.vstack(data_list), dims=('y', 'x'))

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()
        channel_name = self.channel_name

        if calibration == 'counts':
            res = data
        elif calibration in ['radiance', 'reflectance', 'brightness_temperature']:
            # Choose calibration coefficients
            # a) Internal: Nominal or GSICS?
            band_idx = self.mda['spectral_channel_id'] - 1
            if self.calib_mode != 'GSICS' or self.channel_name in VIS_CHANNELS:
                # you cant apply GSICS values to the VIS channels
                coefs = self.prologue["RadiometricProcessing"]["Level15ImageCalibration"]
                int_gain = coefs['CalSlope'][band_idx]
                int_offset = coefs['CalOffset'][band_idx]
            else:
                coefs = self.prologue["RadiometricProcessing"]['MPEFCalFeedback']
                int_gain = coefs['GSICSCalCoeff'][band_idx]
                int_offset = coefs['GSICSOffsetCount'][band_idx]

            # b) Internal or external? External takes precedence.
            gain = self.ext_calib_coefs.get(self.channel_name, {}).get('gain', int_gain)
            offset = self.ext_calib_coefs.get(self.channel_name, {}).get('offset', int_offset)

            # Convert to radiance
            data = data.where(data > 0)
            res = self._convert_to_radiance(data.astype(np.float32), gain, offset)
            line_mask = self.mda['image_segment_line_quality']['line_validity'] >= 2
            line_mask &= self.mda['image_segment_line_quality']['line_validity'] <= 3
            line_mask &= self.mda['image_segment_line_quality']['line_radiometric_quality'] == 4
            line_mask &= self.mda['image_segment_line_quality']['line_geometric_quality'] == 4
            res *= np.choose(line_mask, [1, np.nan])[:, np.newaxis].astype(np.float32)

        if calibration == 'reflectance':
            solar_irradiance = CALIB[self.platform_id][channel_name]["F"]
            res = self._vis_calibrate(res, solar_irradiance)

        elif calibration == 'brightness_temperature':
            cal_type = self.prologue['ImageDescription'][
                'Level15ImageProduction']['PlannedChanProcessing'][self.mda['spectral_channel_id']]
            res = self._ir_calibrate(res, channel_name, cal_type)

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res

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

    def _get_timestamps(self):
        """Read scanline timestamps from the segment header."""
        tline = self.mda['image_segment_line_quality']['line_mean_acquisition']
        return get_cds_time(days=tline['days'], msecs=tline['milliseconds'])


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
