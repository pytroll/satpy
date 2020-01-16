#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
r"""Interface to SEVIRI L1B data from ICARE (Lille).

Introduction
------------

The ``seviri_l1b_icare`` reader reads MSG-SEVIRI L1.5 image data in HDF format
that has been produced by the ICARE Data and Services Center
Data can be accessed via: http://www.icare.univ-lille1.fr

Each SEVIRI timeslot comes as 12 HDF files, one per band. Only those bands
that are of interest need to be passed to the reader. Others can be ignored.
Filenames follow the format: GEO_L1B-MSG1_YYYY-MM-DDTHH-MM-SS_G_CHANN_VX-XX.hdf
Where:
YYYY, MM, DD, HH, MM, SS specify the timeslot starting time.
CHANN is the channel (i.e: HRV, IR016, WV073, etc)
VX-XX is the processing version number

Example
-------
Here is an example how to read the data in satpy:

.. code-block:: python

    from satpy import Scene
    import glob

    filenames = glob.glob('data/*2019-03-01T12-00-00*.hdf')
    scn = Scene(filenames=filenames, reader='seviri_l1b_icare')
    scn.load(['VIS006', 'IR_108'])
    print(scn['IR_108'])

Output:

.. code-block:: none

    <xarray.DataArray 'array-a1d52b7e19ec5a875e2f038df5b60d7e' (y: 3712, x: 3712)>
    dask.array<add, shape=(3712, 3712), dtype=float32, chunksize=(1024, 1024), chunktype=numpy.ndarray>
    Coordinates:
        crs      object +proj=geos +a=6378169.0 +b=6356583.8 +lon_0=0.0 +h=35785831.0 +units=m +type=crs
      * y        (y) float64 5.566e+06 5.563e+06 5.56e+06 ... -5.566e+06 -5.569e+06
      * x        (x) float64 -5.566e+06 -5.563e+06 -5.56e+06 ... 5.566e+06 5.569e+06
    Attributes:
        start_time:           2004-12-29 12:15:00
        end_time:             2004-12-29 12:27:44
        area:                 Area ID: geosmsg\nDescription: MSG/SEVIRI low resol...
        name:                 IR_108
        resolution:           3000.403165817
        calibration:          brightness_temperature
        polarization:         None
        level:                None
        modifiers:            ()
        ancillary_variables:  []

"""
from satpy.readers._geos_area import get_area_extent, get_area_definition
from satpy.readers.hdf4_utils import HDF4FileHandler
from datetime import datetime
import numpy as np


class SEVIRI_ICARE(HDF4FileHandler):
    """SEVIRI L1B handler for HDF4 files."""
    def __init__(self, filename, filename_info, filetype_info):
        super(SEVIRI_ICARE, self).__init__(filename,
                                           filename_info,
                                           filetype_info)
        # These are VIS bands
        self.ref_bands = ['HRV', 'VIS006', 'VIS008', 'IR_016']
        # And these are IR bands
        self.bt_bands = ['IR_039', 'IR_062', 'IR_073',
                         'IR_087', 'IR_097', 'IR_108',
                         'IR_120', 'IR_134',
                         'WV_062', 'WV_073']

    @property
    def sensor_name(self):
        # the sensor and platform names are stored together, eg: MSG1/SEVIRI
        attr = self['/attr/Sensors']
        if isinstance(attr, np.ndarray):
            attr = str(attr.astype(str)).lower()
        else:
            attr = attr.lower()
        plat = attr[0:4]
        sens = attr[5:]
        # icare uses non-standard platform names
        if plat == 'msg1':
            plat = 'Meteosat-08'
        elif plat == 'msg2':
            plat = 'Meteosat-09'
        elif plat == 'msg3':
            plat = 'Meteosat-10'
        elif plat == 'msg4':
            plat = 'Meteosat-11'
        else:
            raise NameError("Unsupported satellite platform:"+plat)
        return [plat, sens]

    @property
    def satlon(self):
        attr = self['/attr/Sub_Satellite_Longitude']
        if isinstance(attr, np.ndarray):
            attr = float(attr.astype(str))
        return attr

    @property
    def projlon(self):
        attr = self['/attr/Projection_Longitude']
        if isinstance(attr, np.ndarray):
            attr = float(attr.astype(str))
        return attr

    @property
    def projection(self):
        attr = self['/attr/Geographic_Projection']
        if isinstance(attr, np.ndarray):
            attr = str(attr.astype(str))
        attr = attr.lower()
        if attr != 'geos':
            raise NotImplementedError("Only the GEOS projection is supported.\
                                        This is:", attr)
        return attr

    @property
    def zone(self):
        attr = self['/attr/Zone']
        if isinstance(attr, np.ndarray):
            attr = str(attr.astype(str)).lower()
        return attr

    @property
    def res(self):
        attr = self['/attr/Nadir_Pixel_Size']
        if isinstance(attr, np.ndarray):
            attr = str(attr.astype(str)).lower()
        return float(attr)

    @property
    def end_time(self):
        attr = self['/attr/End_Acquisition_Date']
        if isinstance(attr, np.ndarray):
            attr = str(attr.astype(str))
        # In some versions milliseconds are present, sometimes not.
        try:
            endacq = datetime.strptime(attr, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            endacq = datetime.strptime(attr, "%Y-%m-%dT%H:%M:%S.%fZ")
        return endacq

    @property
    def start_time(self):
        attr = self['/attr/Beginning_Acquisition_Date']
        if isinstance(attr, np.ndarray):
            attr = str(attr.astype(str))
        # In some versions milliseconds are present, sometimes not.
        try:
            stacq = datetime.strptime(attr, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            stacq = datetime.strptime(attr, "%Y-%m-%dT%H:%M:%S.%fZ")
        return stacq

    @property
    def alt(self):
        attr = self['/attr/Altitude']
        if isinstance(attr, np.ndarray):
            attr = attr.astype(str)
        attr = float(attr)
        # This is stored in km, convert to m
        attr = attr * 1000.
        return attr

    @property
    def geoloc(self):
        attr = self['/attr/Geolocation']
        if isinstance(attr, np.ndarray):
            attr = attr.astype(str)
        cfac = float(attr[0])
        coff = float(attr[1])
        lfac = float(attr[2])
        loff = float(attr[3])
        return [cfac, lfac, coff, loff]

    def get_metadata(self, data, ds_info):
        mda = {}
        mda.update(data.attrs)
        mda.update(ds_info)
        geoloc = self.geoloc
        mda.update({
                    'start_time': self.start_time,
                    'end_time': self.end_time,
                    'platform_name': self.sensor_name[0],
                    'sensor': self.sensor_name[1],
                    'zone': self.zone,
                    'projection_altitude': self.alt,
                    'cfac': geoloc[0],
                    'lfac': geoloc[1],
                    'coff': geoloc[2],
                    'loff': geoloc[3],
                    'resolution': self.res,
                    'satellite_actual_longitude': self.satlon,
                    'projection_longitude': self.projlon,
                    'projection_type': self.projection
        })

        return mda

    def _get_dsname(self, ds_id):
        """Returns the correct dataset name based on requested band."""
        if ds_id.name in self.ref_bands:
            ds_get_name = 'Normalized_Radiance'
        elif ds_id.name in self.bt_bands:
            ds_get_name = 'Brightness_Temperature'
        else:
            raise NameError("Datset type "+ds_id.name+" is not supported.")
        return ds_get_name

    def get_dataset(self, ds_id, ds_info):
        ds_get_name = self._get_dsname(ds_id)
        data = self[ds_get_name]
        data.attrs = self.get_metadata(data, ds_info)
        fill = data.attrs.pop('_FillValue')
        offset = data.attrs.get('add_offset')
        scale_factor = data.attrs.get('scale_factor')
        data = data.where(data != fill)
        data.values = data.values.astype(np.float32)
        if scale_factor is not None and offset is not None:
            data.values *= scale_factor
            data.values += offset
            # Now we correct range from 0-1 to 0-100 for VIS:
            if ds_id.name in self.ref_bands:
                data.values *= 100.
        return data

    def get_area_def(self, ds_id):
        ds_get_name = self._get_dsname(ds_id)
        ds_shape = self[ds_get_name + '/shape']
        geoloc = self.geoloc

        pdict = {}
        pdict['cfac'] = np.int32(geoloc[0])
        pdict['lfac'] = np.int32(geoloc[1])
        pdict['coff'] = np.float32(geoloc[2])
        pdict['loff'] = -np.float32(geoloc[3])

        # Unfortunately this dataset does not store a, b or h.
        # We assume a and b here, and calculate h from altitude
        # a and b are from SEVIRI data HRIT header (201912101300)
        pdict['a'] = 6378169
        pdict['b'] = 6356583.8
        pdict['h'] = self.alt - pdict['a']
        pdict['ssp_lon'] = self.projlon
        pdict['ncols'] = int(ds_shape[0])
        pdict['nlines'] = int(ds_shape[1])

        # Force scandir to SEVIRI default, not known from file
        pdict['scandir'] = 'S2N'
        pdict['a_name'] = 'geosmsg'
        if ds_id.name == 'HRV':
            pdict['a_desc'] = 'MSG/SEVIRI HRV channel area'
            pdict['p_id'] = 'msg_hires'
        else:
            pdict['a_desc'] = 'MSG/SEVIRI low resolution channel area'
            pdict['p_id'] = 'msg_lowres'

        aex = get_area_extent(pdict)
        area = get_area_definition(pdict, aex)

        return area
