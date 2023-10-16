#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022-2023 Satpy developers
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

"""Reader for the EnMAP L1B ZIP data."""

import logging
import math
import os
import shutil
import tempfile
import zipfile
from glob import glob

import numpy as np
import xarray as xr
from enpt.model.images.images_sensorgeo import EnMAP_SWIR_SensorGeo, EnMAP_VNIR_SensorGeo
from enpt.model.metadata import EnMAP_Metadata_L1B_SensorGeo
from enpt.options.config import EnPTConfig

from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class HSIBaseFileHandler(BaseFileHandler):
    """File handler for EnMAP L1B ZIP files."""

    def __init__(self, filename, filename_info, filetype_info,):
        """Prepare the class for dataset reading."""
        super(HSIBaseFileHandler, self).__init__(filename, filename_info, filetype_info)

        # set config for enpt
        config_minimal = dict(path_l1b_enmap_image=filename, drop_bad_bands=False)

        self.cfg = EnPTConfig(**config_minimal)

        # unzip L1B file
        self._unzip_file(filename)

        # read meta data
        self._meta = None
        self.meta = EnMAP_Metadata_L1B_SensorGeo(glob(os.path.join(self.root_dir, "*METADATA.XML"))[0],
                                                 config=self.cfg, logger=logger)
        self.meta.read_metadata()
        self.meta.earthSunDist = self.meta.get_earth_sun_distance(self.meta.observation_datetime)

        # load bands info for dims
        self._load_bands()

    def _unzip_file(self, filename):
        """Unzip L1B file."""
        self.root_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(filename, "r") as zf:
            logger.debug(f'Unzip {filename} to {self.root_dir}')
            zf.extractall(self.root_dir)

        # move the data one level up in case they are within a sub-folder in the zip file
        content = glob(os.path.join(self.root_dir, '*'))

        if len(content) == 1 and os.path.isdir(content[0]):
            for fp in glob(os.path.join(self.root_dir, '**', '*')):
                shutil.move(fp, self.root_dir)

    def _load_bands(self):
        # read wavelength which is the dim for other variables
        self.bands_vnir = xr.DataArray(self.meta.vnir.wvl_center, dims='bands_vnir').rename('bands_vnir')
        self.bands_swir = xr.DataArray(self.meta.swir.wvl_center, dims='bands_swir').rename('bands_swir')
        self.bands_vnir.attrs['units'] = 'nm'
        self.bands_swir.attrs['units'] = 'nm'

        # read fwhm which is the anciilary variable of bands
        self.fwhm_vnir = xr.DataArray(self.meta.vnir.fwhm, dims='bands_vnir').rename(f'fwhm_vnir')
        self.fwhm_swir = xr.DataArray(self.meta.swir.fwhm, dims='bands_swir').rename(f'fwhm_swir')
        self.fwhm_vnir.attrs['units'] = 'nm'
        self.fwhm_swir.attrs['units'] = 'nm'
        self.fwhm_vnir.attrs['standard_name'] = 'full width at half maximum'
        self.fwhm_swir.attrs['standard_name'] = 'full width at half maximum'

        # assign fwhm as bands coords
        self.bands_vnir.coords['fwhm_vnir'] = self.fwhm_vnir
        self.bands_swir.coords['fwhm_swir'] = self.fwhm_swir

    def __del__(self):
        """Delete the object."""
        if self.root_dir:
            shutil.rmtree(self.root_dir)

    @property
    def get_metadata(self,):
        """Derive metadata."""
        # Use buffered data if available
        if self._meta is None:
            self._meta = {'time': self.meta.observation_datetime,
                          'vza': self.meta.geom_view_zenith,
                          'vaa': self.meta.geom_view_azimuth,
                          'sza': self.meta.geom_sun_zenith,
                          'saa': self.meta.geom_sun_azimuth,
                          'earthSunDist': self.meta.earthSunDist,
                          'aot': self.meta.aot,
                          'granule_id': self.meta.vnir.scene_basename,
                          }

        return self._meta

    def calibrate(self, data, yaml_info, band_dimname):
        """Calibrate data."""
        # get calibration method and detector name
        calibration = yaml_info['calibration']
        detector_meta = getattr(self.meta, self.name)

        logger.debug('Calibrating %s to %s', self.name, calibration)

        if calibration == 'counts':
            # original DN values
            calibrated_data = data
        elif calibration == 'radiance':
            # Lλ = QCAL * GAIN + OFFSET
            # NOTE: - DLR provides gains between 2000 and 10000, so we have to DEVIDE by gains
            #       - DLR gains / offsets are provided in W/m2/sr/nm, so we have to multiply by 1000 to get
            #         mW/m2/sr/nm as needed later
            calibrated_data = 1000 * (data.transpose(..., band_dimname) * detector_meta.gains + detector_meta.offsets)
        elif calibration == 'reflectance':
            radiance = 1000 * (data.transpose(..., band_dimname) * detector_meta.gains + detector_meta.offsets)
            constant = self.cfg.scale_factor_toa_ref * math.pi * self.meta.earthSunDist ** 2 / \
                (math.cos(math.radians(self.meta.geom_sun_zenith)))
            solIrr = detector_meta.solar_irrad.reshape(1, 1, data.sizes[band_dimname])
            calibrated_data = (constant * radiance / solIrr).astype(np.int16)
            calibrated_data.attrs['scale_factor'] = self.cfg.scale_factor_toa_ref
        else:
            raise ValueError("Unknown calibration %s for dataset %s" % (calibration, self.name))

        # add units
        calibrated_data.attrs['units'] = yaml_info['units']

        return calibrated_data.transpose(band_dimname, ...)

    def get_dataset(self, key, yaml_info):
        """Get dataset using file_key in yaml_info."""
        # get yaml info
        self.name = yaml_info['name']
        detector = yaml_info.get('detector', '')
        band_dimname = 'bands'+'_'+detector

        logger.debug('Reading in file to get dataset with name %s.', self.name)

        # load VNIR or SWIR data
        if self.name in ['vnir', 'swir']:
            # load sensor
            if self.name == 'vnir':
                sensor = EnMAP_VNIR_SensorGeo(self.root_dir, config=self.cfg,
                                              meta=getattr(self.meta, self.name), logger=logger)
            elif self.name == 'swir':
                sensor = EnMAP_SWIR_SensorGeo(self.root_dir, config=self.cfg,
                                              meta=getattr(self.meta, self.name), logger=logger)

            # read Tiff file
            data = xr.open_dataset(sensor.get_paths().data)['band_data'].rename({'band': band_dimname})

            # apply calibration
            data = self.calibrate(data, yaml_info, band_dimname)

            # set name and coords
            data = data.rename(self.name)
            data.coords[band_dimname] = getattr(self, band_dimname)

        # load longitude/latitude data
        if self.name in ['longitude_vnir', 'latitude_vnir', 'longitude_swir', 'latitude_swir']:
            lons, lats = getattr(self.meta, detector).compute_geolayer_for_cube()
            if self.name.startswith('longitude'):
                data = lons
            elif self.name.startswith('latitude'):
                data = lats

            # save to DataArray
            data = xr.DataArray(data, dims=['y', 'x', band_dimname]).rename(self.name).transpose(band_dimname, ...)
            data.attrs['units'] = yaml_info['units']
            data.attrs['standrad_name'] = yaml_info['standrad_name']

        # load mask TIFF data
        if 'mask' in self.name or self.name in ['deadpixelmap', 'quicklook']:
            # read TIFF file
            data = xr.open_dataset(os.path.join(self.root_dir, getattr(self.meta.vnir, yaml_info['file_key'])))\
                     .isel(band=0)['band_data'].rename(self.name)
            if yaml_info['description']:
                data.attrs['description'] = yaml_info['description']

        # load smile_coef
        if 'smile_coef' in self.name:
            # the smile_coef should be corrected: https://github.com/GFZ/enpt/pull/6
            data = xr.DataArray(getattr(self.meta, detector).smile, dims=['x', band_dimname]).rename(self.name)

        # add metadata
        data.attrs.update(self.get_metadata)

        return data
