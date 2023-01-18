#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2020 Satpy developers
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
"""SAFE MSI L1C reader.

The MSI data has a special value for saturated pixels. By default, these
pixels are set to np.inf, but for some applications it might be desirable
to have these pixels left untouched.
For this case, the `mask_saturated` flag is available in the reader, and can be
toggled with ``reader_kwargs`` upon Scene creation::

    scene = satpy.Scene(filenames,
                        reader='msi_safe',
                        reader_kwargs={'mask_saturated': False})
    scene.load(['B01'])

L1B format description for the files read here:

  https://sentinels.copernicus.eu/documents/247904/0/Sentinel-2-product-specifications-document-V14-9.pdf/

"""

import logging
import xml.etree.ElementTree as ET

import dask.array as da
import numpy as np
import rioxarray
from pyresample import geometry
from xarray import DataArray

from satpy import CHUNK_SIZE
from satpy._compat import cached_property
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


PLATFORMS = {'S2A': "Sentinel-2A",
             'S2B': "Sentinel-2B",
             'S2C': "Sentinel-2C",
             'S2D': "Sentinel-2D"}


class SAFEMSIL1C(BaseFileHandler):
    """File handler for SAFE MSI files (jp2)."""

    def __init__(self, filename, filename_info, filetype_info, mda, tile_mda, mask_saturated=True):
        """Initialize the reader."""
        super(SAFEMSIL1C, self).__init__(filename, filename_info,
                                         filetype_info)
        del mask_saturated
        self._start_time = filename_info['observation_time']
        self._end_time = filename_info['observation_time']
        self._channel = filename_info['band_name']
        self._tile_mda = tile_mda
        self._mda = mda
        self.platform_name = PLATFORMS[filename_info['fmission_id']]

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._channel != key['name']:
            return

        logger.debug('Reading %s.', key['name'])
        proj = self._read_from_file(key)
        proj.attrs = info.copy()
        proj.attrs['units'] = '%'
        proj.attrs['platform_name'] = self.platform_name
        return proj

    def _read_from_file(self, key):
        proj = rioxarray.open_rasterio(self.filename, chunks=CHUNK_SIZE)
        proj = proj.squeeze("band")
        if key["calibration"] == "reflectance":
            return self._mda.calibrate_to_reflectances(proj, self._channel)
        if key["calibration"] == "radiance":
            return self._mda.calibrate_to_radiances(proj, self._channel)

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._start_time

    def get_area_def(self, dsid):
        """Get the area def."""
        if self._channel != dsid['name']:
            return
        return self._tile_mda.get_area_def(dsid)


class SAFEMSIXMLMetadata(BaseFileHandler):
    """Base class for SAFE MSI XML metadata filehandlers."""

    def __init__(self, filename, filename_info, filetype_info, mask_saturated=True):
        """Init the reader."""
        super().__init__(filename, filename_info, filetype_info)
        self._start_time = filename_info['observation_time']
        self._end_time = filename_info['observation_time']
        self.root = ET.parse(self.filename)
        self.tile = filename_info['dtile_number']
        self.platform_name = PLATFORMS[filename_info['fmission_id']]
        self.mask_saturated = mask_saturated
        import bottleneck  # noqa
        import geotiepoints  # noqa

    @property
    def end_time(self):
        """Get end time."""
        return self._start_time

    @property
    def start_time(self):
        """Get start time."""
        return self._start_time


class SAFEMSIMDXML(SAFEMSIXMLMetadata):
    """File handle for sentinel 2 safe XML generic metadata."""

    def calibrate_to_reflectances(self, data, band_name):
        """Calibrate *data* using the radiometric information for the metadata."""
        quantification = int(self.root.find('.//QUANTIFICATION_VALUE').text)
        data = self._sanitize_data(data)
        return (data + self.band_offset(band_name)) / quantification * 100

    def _sanitize_data(self, data):
        data = data.where(data != self.no_data)
        if self.mask_saturated:
            data = data.where(data != self.saturated, np.inf)
        return data

    def band_offset(self, band):
        """Get the band offset for *band*."""
        band_index = self._band_index(band)
        return self.band_offsets.get(band_index, 0)

    def _band_index(self, band):
        band_indices = self.band_indices
        band_conversions = {"B01": "B1", "B02": "B2", "B03": "B3", "B04": "B4", "B05": "B5", "B06": "B6", "B07": "B7",
                            "B08": "B8", "B8A": "B8A", "B09": "B9", "B10": "B10", "B11": "B11", "B12": "B12"}
        band_index = band_indices[band_conversions[band]]
        return band_index

    @cached_property
    def band_indices(self):
        """Get the band indices from the metadata."""
        spectral_info = self.root.findall('.//Spectral_Information')
        band_indices = {spec.attrib["physicalBand"]: int(spec.attrib["bandId"]) for spec in spectral_info}
        return band_indices

    @cached_property
    def band_offsets(self):
        """Get the band offsets from the metadata."""
        offsets = self.root.find('.//Radiometric_Offset_List')
        if offsets is not None:
            band_offsets = {int(off.attrib["band_id"]): float(off.text) for off in offsets}
        else:
            band_offsets = {}
        return band_offsets

    @cached_property
    def special_values(self):
        """Get the special values from the metadata."""
        special_values = self.root.findall('.//Special_Values')
        special_values_dict = {value[0].text: float(value[1].text) for value in special_values}
        return special_values_dict

    @property
    def no_data(self):
        """Get the nodata value from the metadata."""
        return self.special_values["NODATA"]

    @property
    def saturated(self):
        """Get the saturated value from the metadata."""
        return self.special_values["SATURATED"]

    def calibrate_to_radiances(self, data, band_name):
        """Calibrate *data* to radiance using the radiometric information for the metadata."""
        physical_gain = self.physical_gain(band_name)
        data = self._sanitize_data(data)
        return (data + self.band_offset(band_name)) / physical_gain

    def physical_gain(self, band_name):
        """Get the physical gain for a given *band_name*."""
        band_index = self._band_index(band_name)
        return self.physical_gains[band_index]

    @cached_property
    def physical_gains(self):
        """Get the physical gains dictionary."""
        physical_gains = {int(elt.attrib["bandId"]): float(elt.text) for elt in self.root.findall(".//PHYSICAL_GAINS")}
        return physical_gains


def _fill_swath_edges(angles):
    """Fill gaps at edges of swath."""
    darr = DataArray(angles, dims=['y', 'x'])
    darr = darr.bfill('x')
    darr = darr.ffill('x')
    darr = darr.bfill('y')
    darr = darr.ffill('y')
    angles = darr.data
    return angles


class SAFEMSITileMDXML(SAFEMSIXMLMetadata):
    """File handle for sentinel 2 safe XML tile metadata."""

    def __init__(self, filename, filename_info, filetype_info, mask_saturated=True):
        """Init the reader."""
        super().__init__(filename, filename_info, filetype_info, mask_saturated)
        self.geocoding = self.root.find('.//Tile_Geocoding')

    def get_area_def(self, dsid):
        """Get the area definition of the dataset."""
        area_extent = self._area_extent(dsid['resolution'])
        cols, rows = self._shape(dsid['resolution'])
        area = geometry.AreaDefinition(
            self.tile,
            "On-the-fly area",
            self.tile,
            self.projection,
            cols,
            rows,
            area_extent)
        return area

    @cached_property
    def projection(self):
        """Get the geographic projection."""
        from pyproj import CRS
        epsg = self.geocoding.find('HORIZONTAL_CS_CODE').text
        return CRS(epsg)

    def _area_extent(self, resolution):
        cols, rows = self._shape(resolution)
        geoposition = self.geocoding.find('Geoposition[@resolution="' + str(resolution) + '"]')
        ulx = float(geoposition.find('ULX').text)
        uly = float(geoposition.find('ULY').text)
        xdim = float(geoposition.find('XDIM').text)
        ydim = float(geoposition.find('YDIM').text)
        area_extent = (ulx, uly + rows * ydim, ulx + cols * xdim, uly)
        return area_extent

    def _shape(self, resolution):
        rows = int(self.geocoding.find('Size[@resolution="' + str(resolution) + '"]/NROWS').text)
        cols = int(self.geocoding.find('Size[@resolution="' + str(resolution) + '"]/NCOLS').text)
        return cols, rows

    @staticmethod
    def _do_interp(minterp, xcoord, ycoord):
        interp_points2 = np.vstack((ycoord.ravel(), xcoord.ravel()))
        res = minterp(interp_points2)
        return res.reshape(xcoord.shape)

    def interpolate_angles(self, angles, resolution):
        """Interpolate the angles."""
        from geotiepoints.multilinear import MultilinearInterpolator

        cols, rows = self._shape(resolution)

        smin = [0, 0]
        smax = np.array(angles.shape) - 1
        orders = angles.shape
        minterp = MultilinearInterpolator(smin, smax, orders)
        minterp.set_values(da.atleast_2d(angles.ravel()))

        y = da.arange(rows, dtype=angles.dtype, chunks=CHUNK_SIZE) / (rows-1) * (angles.shape[0] - 1)
        x = da.arange(cols, dtype=angles.dtype, chunks=CHUNK_SIZE) / (cols-1) * (angles.shape[1] - 1)
        xcoord, ycoord = da.meshgrid(x, y)
        return da.map_blocks(self._do_interp, minterp, xcoord, ycoord, dtype=angles.dtype, chunks=xcoord.chunks)

    def _get_coarse_dataset(self, key, info):
        """Get the coarse dataset refered to by `key` from the XML data."""
        angles = self.root.find('.//Tile_Angles')
        if key['name'] in ['solar_zenith_angle', 'solar_azimuth_angle']:
            angles = self._get_solar_angles(angles, info)
        elif key['name'] in ['satellite_zenith_angle', 'satellite_azimuth_angle']:
            angles = self._get_satellite_angles(angles, info)
        else:
            angles = None
        return angles

    def _get_solar_angles(self, angles, info):
        angles = self._get_values_from_tag(angles, info['xml_tag'])
        return angles

    @staticmethod
    def _get_values_from_tag(xml_tree, xml_tag):
        elts = xml_tree.findall(xml_tag + '/Values_List/VALUES')
        return np.array([[val for val in elt.text.split()] for elt in elts],
                        dtype=np.float64)

    def _get_satellite_angles(self, angles, info):
        arrays = []
        elts = angles.findall(info['xml_tag'] + '[@bandId="1"]')
        for elt in elts:
            arrays.append(self._get_values_from_tag(elt, info['xml_item']))
        angles = np.nanmean(np.dstack(arrays), -1)
        return angles

    def get_dataset(self, key, info):
        """Get the dataset referred to by `key`."""
        angles = self._get_coarse_dataset(key, info)
        if angles is None:
            return None

        angles = _fill_swath_edges(angles)

        res = self.interpolate_angles(angles, key['resolution'])

        proj = DataArray(res, dims=['y', 'x'])
        proj.attrs = info.copy()
        proj.attrs['units'] = 'degrees'
        proj.attrs['platform_name'] = self.platform_name
        return proj
