#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2012-2021 Satpy developers
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
"""Reader for aapp level 1b data.

Options for loading:

 - pre_launch_coeffs (False): use pre-launch coefficients if True, operational
   otherwise (if available).

https://nwp-saf.eumetsat.int/site/download/documentation/aapp/NWPSAF-MF-UD-003_Formats_v8.0.pdf
"""
import functools
import logging
from datetime import datetime, timedelta

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed

from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import get_chunk_size_limit

CHANNEL_DTYPE = np.float64


def get_avhrr_lac_chunks(shape, dtype):
    """Get chunks from a given shape adapted for full-resolution AVHRR data."""
    limit = get_chunk_size_limit(dtype)
    return da.core.normalize_chunks(("auto", 2048), shape=shape, limit=limit, dtype=dtype)


def get_aapp_chunks(shape):
    """Get chunks from a given shape adapted for AAPP data."""
    return get_avhrr_lac_chunks(shape, dtype=CHANNEL_DTYPE)


logger = logging.getLogger(__name__)

AVHRR_CHANNEL_NAMES = ["1", "2", "3a", "3b", "4", "5"]

AVHRR_ANGLE_NAMES = ['sensor_zenith_angle',
                     'solar_zenith_angle',
                     'sun_sensor_azimuth_difference_angle']

AVHRR_PLATFORM_IDS2NAMES = {4: 'NOAA-15',
                            2: 'NOAA-16',
                            6: 'NOAA-17',
                            7: 'NOAA-18',
                            8: 'NOAA-19',
                            11: 'Metop-B',
                            12: 'Metop-A',
                            13: 'Metop-C',
                            14: 'Metop simulator'}


def create_xarray(arr):
    """Create an `xarray.DataArray`."""
    res = xr.DataArray(arr, dims=['y', 'x'])
    return res


class AAPPL1BaseFileHandler(BaseFileHandler):
    """A base file handler for the AAPP level-1 formats."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize AAPP level-1 file handler object."""
        super().__init__(filename, filename_info, filetype_info)

        self.channels = None
        self.units = None
        self.sensor = "unknown"

        self._data = None
        self._header = None
        self.area = None

        self._channel_names = []
        self._angle_names = []

    def _set_filedata_layout(self):
        """Set the file data type/layout."""
        self._header_offset = 0
        self._scan_type = np.dtype([("siteid", "<i2")])
        self._header_type = np.dtype([("siteid", "<i2")])

    @property
    def start_time(self):
        """Get the time of the first observation."""
        return datetime(self._data['scnlinyr'][0], 1, 1) + timedelta(
            days=int(self._data['scnlindy'][0]) - 1,
            milliseconds=int(self._data['scnlintime'][0]))

    @property
    def end_time(self):
        """Get the time of the final observation."""
        return datetime(self._data['scnlinyr'][-1], 1, 1) + timedelta(
            days=int(self._data['scnlindy'][-1]) - 1,
            milliseconds=int(self._data['scnlintime'][-1]))

    def _update_dataset_attributes(self, dataset, key, info):
        dataset.attrs.update({'platform_name': self.platform_name,
                              'sensor': self.sensor})
        dataset.attrs.update(key.to_dict())
        for meta_key in ('standard_name', 'units'):
            if meta_key in info:
                dataset.attrs.setdefault(meta_key, info[meta_key])

    def _get_platform_name(self, platform_names_lookup):
        """Get the platform name from the file header."""
        self.platform_name = platform_names_lookup.get(self._header['satid'][0], None)
        if self.platform_name is None:
            raise ValueError("Unsupported platform ID: %d" % self.header['satid'])

    def read(self):
        """Read the data."""
        tic = datetime.now()
        header = np.memmap(self.filename, dtype=self._header_type, mode="r", shape=(1, ))
        data = np.memmap(self.filename, dtype=self._scan_type, offset=self._header_offset, mode="r")
        logger.debug("Reading time %s", str(datetime.now() - tic))

        self._header = header
        self._data = data

    def _calibrate_active_channel_data(self, key):
        """Calibrate active channel data only."""
        raise NotImplementedError("This should be implemented in the sub class")

    def get_dataset(self, key, info):
        """Get a dataset from the file."""
        if key['name'] in self._channel_names:
            dataset = self._calibrate_active_channel_data(key)
            if dataset is None:
                return None
        elif key['name'] in ['longitude', 'latitude']:
            dataset = self.navigate(key['name'])
            dataset.attrs = info
        elif key['name'] in self._angle_names:
            dataset = self.get_angles(key['name'])
        else:
            raise ValueError("Not a supported dataset: %s", key['name'])

        self._update_dataset_attributes(dataset, key, info)
        return dataset


class AVHRRAAPPL1BFile(AAPPL1BaseFileHandler):
    """Reader for AVHRR L1B files created from the AAPP software."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize object information by reading the input file."""
        super(AVHRRAAPPL1BFile, self).__init__(filename, filename_info,
                                               filetype_info)

        self.channels = {i: None for i in AVHRR_CHANNEL_NAMES}
        self.units = {i: 'counts' for i in AVHRR_CHANNEL_NAMES}

        self._is3b = None
        self._is3a = None
        self._channel_names = AVHRR_CHANNEL_NAMES
        self._angle_names = AVHRR_ANGLE_NAMES

        self._set_filedata_layout()
        self.read()

        self.active_channels = self._get_active_channels()

        self._get_platform_name(AVHRR_PLATFORM_IDS2NAMES)
        self.sensor = 'avhrr-3'

        self._get_all_interpolated_angles = functools.lru_cache(maxsize=10)(
            self._get_all_interpolated_angles_uncached
        )
        self._get_all_interpolated_coordinates = functools.lru_cache(maxsize=10)(
            self._get_all_interpolated_coordinates_uncached
        )

    def _set_filedata_layout(self):
        """Set the file data type/layout."""
        self._header_offset = 22016
        self._scan_type = _SCANTYPE
        self._header_type = _HEADERTYPE

    def _get_active_channels(self):
        status = self._get_channel_binary_status_from_header()
        return self._convert_binary_channel_status_to_activation_dict(status)

    def _calibrate_active_channel_data(self, key):
        """Calibrate active channel data only."""
        if self.active_channels[key['name']]:
            return self.calibrate(key)
        return None

    def _get_channel_binary_status_from_header(self):
        status = self._header['inststat1'].item()
        change_line = self._header['statchrecnb']
        if change_line > 0:
            status |= self._header['inststat2'].item()
        return status

    @staticmethod
    def _convert_binary_channel_status_to_activation_dict(status):
        bits_channels = ((13, '1'),
                         (12, '2'),
                         (11, '3a'),
                         (10, '3b'),
                         (9, '4'),
                         (8, '5'))
        activated = dict()
        for bit, channel_name in bits_channels:
            activated[channel_name] = bool(status >> bit & 1)
        return activated

    def available_datasets(self, configured_datasets=None):
        """Get the available datasets."""
        for _, mda in configured_datasets:
            if mda['name'] in self._channel_names:
                yield self.active_channels[mda['name']], mda
            else:
                yield True, mda

    def get_angles(self, angle_id):
        """Get sun-satellite viewing angles."""
        sunz, satz, azidiff = self._get_all_interpolated_angles()

        name_to_variable = dict(zip(self._angle_names, (satz, sunz, azidiff)))
        return create_xarray(name_to_variable[angle_id])

    def _get_all_interpolated_angles_uncached(self):
        sunz40km, satz40km, azidiff40km = self._get_tiepoint_angles_in_degrees()
        return self._interpolate_arrays(sunz40km, satz40km, azidiff40km)

    def _get_tiepoint_angles_in_degrees(self):
        sunz40km = self._data["ang"][:, :, 0] * 1e-2
        satz40km = self._data["ang"][:, :, 1] * 1e-2
        azidiff40km = self._data["ang"][:, :, 2] * 1e-2
        return sunz40km, satz40km, azidiff40km

    def _interpolate_arrays(self, *input_arrays, geolocation=False):
        lines = input_arrays[0].shape[0]
        try:
            interpolator = self._create_40km_interpolator(lines, *input_arrays, geolocation=geolocation)
        except ImportError:
            logger.warning("Could not interpolate, python-geotiepoints missing.")
            output_arrays = input_arrays
        else:
            output_delayed = delayed(interpolator.interpolate, nout=3)()
            output_arrays = [da.from_delayed(out_array, (lines, 2048), in_array.dtype)
                             for in_array, out_array in zip(input_arrays, output_delayed)]
        return output_arrays

    @staticmethod
    def _create_40km_interpolator(lines, *arrays_40km, geolocation=False):
        if geolocation:
            # Slower but accurate at datum line
            from geotiepoints.geointerpolator import GeoInterpolator as Interpolator
        else:
            from geotiepoints.interpolator import Interpolator
        cols40km = np.arange(24, 2048, 40)
        cols1km = np.arange(2048)
        rows40km = np.arange(lines)
        rows1km = np.arange(lines)
        along_track_order = 1
        cross_track_order = 3
        satint = Interpolator(
            arrays_40km, (rows40km, cols40km),
            (rows1km, cols1km), along_track_order, cross_track_order)
        return satint

    def navigate(self, coordinate_id):
        """Get the longitudes and latitudes of the scene."""
        lons, lats = self._get_all_interpolated_coordinates()
        if coordinate_id == 'longitude':
            return create_xarray(lons)
        if coordinate_id == 'latitude':
            return create_xarray(lats)

        raise KeyError("Coordinate {} unknown.".format(coordinate_id))

    def _get_all_interpolated_coordinates_uncached(self):
        lons40km, lats40km = self._get_coordinates_in_degrees()
        return self._interpolate_arrays(lons40km, lats40km, geolocation=True)

    def _get_coordinates_in_degrees(self):
        lons40km = self._data["pos"][:, :, 1] * 1e-4
        lats40km = self._data["pos"][:, :, 0] * 1e-4
        return lons40km, lats40km

    def calibrate(self,
                  dataset_id,
                  pre_launch_coeffs=False,
                  calib_coeffs=None):
        """Calibrate the data."""
        if calib_coeffs is None:
            calib_coeffs = {}

        units = {'reflectance': '%',
                 'brightness_temperature': 'K',
                 'counts': '',
                 'radiance': 'W*m-2*sr-1*cm ?'}

        if dataset_id['name'] in ("3a", "3b") and self._is3b is None:
            # Is it 3a or 3b:
            line_chunks = get_aapp_chunks((self._data.shape[0], 2048))[0]
            self._is3a = da.bitwise_and(da.from_array(self._data['scnlinbit'],
                                                      chunks=line_chunks), 3) == 0
            self._is3b = da.bitwise_and(da.from_array(self._data['scnlinbit'],
                                                      chunks=line_chunks), 3) == 1

        try:
            vis_idx = ['1', '2', '3a'].index(dataset_id['name'])
            ir_idx = None
        except ValueError:
            vis_idx = None
            ir_idx = ['3b', '4', '5'].index(dataset_id['name'])

        mask = True
        if vis_idx is not None:
            coeffs = calib_coeffs.get('ch' + dataset_id['name'])
            if dataset_id['name'] == '3a':
                mask = self._is3a[:, None]
            ds = create_xarray(
                _vis_calibrate(self._data,
                               vis_idx,
                               dataset_id['calibration'],
                               pre_launch_coeffs,
                               coeffs,
                               mask=mask))
        else:
            if dataset_id['name'] == '3b':
                mask = self._is3b[:, None]
            ds = create_xarray(
                _ir_calibrate(self._header,
                              self._data,
                              ir_idx,
                              dataset_id['calibration'],
                              mask=mask))

        ds.attrs['units'] = units[dataset_id['calibration']]
        ds.attrs.update(dataset_id._asdict())
        return ds


# AAPP 1b header

_HEADERTYPE = np.dtype([("siteid", "S3"),
                        ("blank", "S1"),
                        ("l1bversnb", "<i2"),
                        ("l1bversyr", "<i2"),
                        ("l1bversdy", "<i2"),
                        ("reclg", "<i2"),
                        ("blksz", "<i2"),
                        ("hdrcnt", "<i2"),
                        ("filler0", "S6"),
                        ("dataname", "S42"),
                        ("prblkid", "S8"),
                        ("satid", "<i2"),
                        ("instid", "<i2"),
                        ("datatype", "<i2"),
                        ("tipsrc", "<i2"),
                        ("startdatajd", "<i4"),
                        ("startdatayr", "<i2"),
                        ("startdatady", "<i2"),
                        ("startdatatime", "<i4"),
                        ("enddatajd", "<i4"),
                        ("enddatayr", "<i2"),
                        ("enddatady", "<i2"),
                        ("enddatatime", "<i4"),
                        ("cpidsyr", "<i2"),
                        ("cpidsdy", "<i2"),
                        ("filler1", "S8"),
                        # data set quality indicators
                        ("inststat1", "<i4"),
                        ("filler2", "S2"),
                        ("statchrecnb", "<i2"),
                        ("inststat2", "<i4"),
                        ("scnlin", "<i2"),
                        ("callocscnlin", "<i2"),
                        ("misscnlin", "<i2"),
                        ("datagaps", "<i2"),
                        ("okdatafr", "<i2"),
                        ("pacsparityerr", "<i2"),
                        ("auxsyncerrsum", "<i2"),
                        ("timeseqerr", "<i2"),
                        ("timeseqerrcode", "<i2"),
                        ("socclockupind", "<i2"),
                        ("locerrind", "<i2"),
                        ("locerrcode", "<i2"),
                        ("pacsstatfield", "<i2"),
                        ("pacsdatasrc", "<i2"),
                        ("filler3", "S4"),
                        ("spare1", "S8"),
                        ("spare2", "S8"),
                        ("filler4", "S10"),
                        # Calibration
                        ("racalind", "<i2"),
                        ("solarcalyr", "<i2"),
                        ("solarcaldy", "<i2"),
                        ("pcalalgind", "<i2"),
                        ("pcalalgopt", "<i2"),
                        ("scalalgind", "<i2"),
                        ("scalalgopt", "<i2"),
                        ("irttcoef", "<i2", (4, 6)),
                        ("filler5", "<i4", (2, )),
                        # radiance to temperature conversion
                        ("albcnv", "<i4", (2, 3)),
                        ("radtempcnv", "<i4", (3, 3)),
                        ("filler6", "<i4", (3, )),
                        # Navigation
                        ("modelid", "S8"),
                        ("nadloctol", "<i2"),
                        ("locbit", "<i2"),
                        ("filler7", "S2"),
                        ("rollerr", "<i2"),
                        ("pitcherr", "<i2"),
                        ("yawerr", "<i2"),
                        ("epoyr", "<i2"),
                        ("epody", "<i2"),
                        ("epotime", "<i4"),
                        ("smaxis", "<i4"),
                        ("eccen", "<i4"),
                        ("incli", "<i4"),
                        ("argper", "<i4"),
                        ("rascnod", "<i4"),
                        ("manom", "<i4"),
                        ("xpos", "<i4"),
                        ("ypos", "<i4"),
                        ("zpos", "<i4"),
                        ("xvel", "<i4"),
                        ("yvel", "<i4"),
                        ("zvel", "<i4"),
                        ("earthsun", "<i4"),
                        ("filler8", "S16"),
                        # analog telemetry conversion
                        ("pchtemp", "<i2", (5, )),
                        ("reserved1", "<i2"),
                        ("pchtempext", "<i2", (5, )),
                        ("reserved2", "<i2"),
                        ("pchpow", "<i2", (5, )),
                        ("reserved3", "<i2"),
                        ("rdtemp", "<i2", (5, )),
                        ("reserved4", "<i2"),
                        ("bbtemp1", "<i2", (5, )),
                        ("reserved5", "<i2"),
                        ("bbtemp2", "<i2", (5, )),
                        ("reserved6", "<i2"),
                        ("bbtemp3", "<i2", (5, )),
                        ("reserved7", "<i2"),
                        ("bbtemp4", "<i2", (5, )),
                        ("reserved8", "<i2"),
                        ("eleccur", "<i2", (5, )),
                        ("reserved9", "<i2"),
                        ("motorcur", "<i2", (5, )),
                        ("reserved10", "<i2"),
                        ("earthpos", "<i2", (5, )),
                        ("reserved11", "<i2"),
                        ("electemp", "<i2", (5, )),
                        ("reserved12", "<i2"),
                        ("chtemp", "<i2", (5, )),
                        ("reserved13", "<i2"),
                        ("bptemp", "<i2", (5, )),
                        ("reserved14", "<i2"),
                        ("mhtemp", "<i2", (5, )),
                        ("reserved15", "<i2"),
                        ("adcontemp", "<i2", (5, )),
                        ("reserved16", "<i2"),
                        ("d4bvolt", "<i2", (5, )),
                        ("reserved17", "<i2"),
                        ("d5bvolt", "<i2", (5, )),
                        ("reserved18", "<i2"),
                        ("bbtempchn3b", "<i2", (5, )),
                        ("reserved19", "<i2"),
                        ("bbtempchn4", "<i2", (5, )),
                        ("reserved20", "<i2"),
                        ("bbtempchn5", "<i2", (5, )),
                        ("reserved21", "<i2"),
                        ("refvolt", "<i2", (5, )),
                        ("reserved22", "<i2"), ])

# AAPP 1b scanline

_SCANTYPE = np.dtype([("scnlin", "<i2"),
                      ("scnlinyr", "<i2"),
                      ("scnlindy", "<i2"),
                      ("clockdrift", "<i2"),
                      ("scnlintime", "<i4"),
                      ("scnlinbit", "<i2"),
                      ("filler0", "S10"),
                      ("qualind", "<i4"),
                      ("scnlinqual", "<i4"),
                      ("calqual", "<i2", (3, )),
                      ("cbiterr", "<i2"),
                      ("filler1", "S8"),
                      # Calibration
                      ("calvis", "<i4", (3, 3, 5)),
                      ("calir", "<i4", (3, 2, 3)),
                      ("filler2", "<i4", (3, )),
                      # Navigation
                      ("navstat", "<i4"),
                      ("attangtime", "<i4"),
                      ("rollang", "<i2"),
                      ("pitchang", "<i2"),
                      ("yawang", "<i2"),
                      ("scalti", "<i2"),
                      ("ang", "<i2", (51, 3)),
                      ("filler3", "<i2", (3, )),
                      ("pos", "<i4", (51, 2)),
                      ("filler4", "<i4", (2, )),
                      ("telem", "<i2", (103, )),
                      ("filler5", "<i2"),
                      ("hrpt", "<i2", (2048, 5)),
                      ("filler6", "<i4", (2, )),
                      # tip minor frame header
                      ("tipmfhd", "<i2", (7, 5)),
                      # cpu telemetry
                      ("cputel", "S6", (2, 5)),
                      ("filler7", "<i2", (67, )), ])


def _vis_calibrate(data,
                   chn,
                   calib_type,
                   pre_launch_coeffs=False,
                   calib_coeffs=None,
                   mask=True):
    """Calibrate visible channel data.

    *calib_type* in count, reflectance, radiance.

    """
    # Calibration count to albedo, the calibration is performed separately for
    # two value ranges.
    if calib_type not in ['counts', 'radiance', 'reflectance']:
        raise ValueError('Calibration ' + calib_type + ' unknown!')

    channel_data = data["hrpt"][:, :, chn]
    chunks = get_aapp_chunks(channel_data.shape)
    line_chunks = chunks[0]
    channel = da.from_array(channel_data, chunks=chunks)
    mask &= channel != 0

    if calib_type == 'counts':
        return channel

    channel = channel.astype(CHANNEL_DTYPE)

    if calib_type == 'radiance':
        logger.info("Radiances are not yet supported for " +
                    "the VIS/NIR channels!")

    if pre_launch_coeffs:
        coeff_idx = 2
    else:
        # check that coeffs are valid
        if np.all(data["calvis"][:, chn, 0, 4] == 0):
            logger.info(
                "No valid operational coefficients, fall back to pre-launch")
            coeff_idx = 2
        else:
            coeff_idx = 0

    intersection = da.from_array(data["calvis"][:, chn, coeff_idx, 4],
                                 chunks=line_chunks)

    if calib_coeffs is not None:
        logger.info("Updating from external calibration coefficients.")
        slope1 = da.from_array(calib_coeffs[0], chunks=line_chunks)
        intercept1 = da.from_array(calib_coeffs[1], chunks=line_chunks)
        slope2 = da.from_array(calib_coeffs[2], chunks=line_chunks)
        intercept2 = da.from_array(calib_coeffs[3], chunks=line_chunks)
    else:
        slope1 = da.from_array(data["calvis"][:, chn, coeff_idx, 0],
                               chunks=line_chunks) * 1e-10
        intercept1 = da.from_array(data["calvis"][:, chn, coeff_idx, 1],
                                   chunks=line_chunks) * 1e-7
        slope2 = da.from_array(data["calvis"][:, chn, coeff_idx, 2],
                               chunks=line_chunks) * 1e-10
        intercept2 = da.from_array(data["calvis"][:, chn, coeff_idx, 3],
                                   chunks=line_chunks) * 1e-7

        # In the level 1b file, the visible coefficients are stored as 4-byte integers. Scaling factors then convert
        # them to real numbers which are applied to the measured counts. The coefficient is different depending on
        # whether the counts are less than or greater than the high-gain/low-gain transition value (nominally 500).
        # The slope for visible channels should always be positive (reflectance increases with count). With the
        # pre-launch coefficients the channel 2, 3a slope is always positive but with the operational coefs the stored
        # number in the high-reflectance regime overflows the maximum 2147483647, i.e. it is negative when
        # interpreted as a signed integer. So you have to modify it. Also chanel 1 is treated the same way in AAPP.
        slope2 = da.where(slope2 < 0, slope2 + 0.4294967296, slope2)

    channel = da.where(channel <= intersection[:, None],
                       channel * slope1[:, None] + intercept1[:, None],
                       channel * slope2[:, None] + intercept2[:, None])

    channel = channel.clip(min=0)

    return da.where(mask, channel, np.nan)


def _ir_calibrate(header, data, irchn, calib_type, mask=True):
    """Calibrate for IR bands.

    *calib_type* in brightness_temperature, radiance, count

    """
    channel_data = data["hrpt"][:, :, irchn + 2]
    chunks = get_aapp_chunks(channel_data.shape)
    line_chunks = chunks[0]

    count = da.from_array(channel_data, chunks=chunks)

    if calib_type == 0:
        return count

    # Mask unnaturally low values
    mask &= count != 0
    count = count.astype(CHANNEL_DTYPE)

    k1_ = da.from_array(data['calir'][:, irchn, 0, 0], chunks=line_chunks) / 1.0e9
    k2_ = da.from_array(data['calir'][:, irchn, 0, 1], chunks=line_chunks) / 1.0e6
    k3_ = da.from_array(data['calir'][:, irchn, 0, 2], chunks=line_chunks) / 1.0e6

    # Count to radiance conversion:
    rad = k1_[:, None] * count * count + k2_[:, None] * count + k3_[:, None]

    # Suspicious lines
    mask &= ((k1_ != 0) | (k2_ != 0) | (k3_ != 0))[:, None]

    if calib_type == 2:
        mask &= rad > 0.0
        return da.where(mask, rad, np.nan)

    # Central wavenumber:
    cwnum = header['radtempcnv'][0, irchn, 0]
    if irchn == 0:
        cwnum = cwnum / 1.0e2
    else:
        cwnum = cwnum / 1.0e3

    bandcor_2 = header['radtempcnv'][0, irchn, 1] / 1e5
    bandcor_3 = header['radtempcnv'][0, irchn, 2] / 1e6

    ir_const_1 = 1.1910659e-5
    ir_const_2 = 1.438833

    t_planck = (ir_const_2 * cwnum) / \
        np.log(1 + ir_const_1 * cwnum * cwnum * cwnum / rad)

    # Band corrections applied to t_planck to get correct
    # brightness temperature for channel:
    if bandcor_2 < 0:  # Post AAPP-v4
        tb_ = bandcor_2 + bandcor_3 * t_planck
    else:  # AAPP 1 to 4
        tb_ = (t_planck - bandcor_2) / bandcor_3

    # Mask unnaturally low values

    return da.where(mask, tb_, np.nan)
