#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018.

# Author(s):

#   Trygve Aspenes <trygveas@met.no>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""MITIFF writer objects for creating MITIFF files from `Dataset` objects.

"""

import logging

import numpy as np

from satpy.writers import ImageWriter

from satpy.writers import get_enhanced_image
from satpy.dataset import DatasetID

import dask

IMAGEDESCRIPTION = 270

LOG = logging.getLogger(__name__)

KELVIN_TO_CELSIUS = -273.15


class MITIFFWriter(ImageWriter):

    def __init__(self, name=None, filename=None, enhancement_config=None, base_dir=None, tags=None, **kwargs):
        ImageWriter.__init__(self,
                             name,
                             filename,
                             enhancement_config,
                             base_dir,
                             default_config_filename="writers/mitiff.yaml",
                             **kwargs)

        self.tags = self.info.get("tags",
                                  None) if tags is None else tags
        if self.tags is None:
            self.tags = {}
        elif not isinstance(self.tags, dict):
            # if it's coming from a config file
            self.tags = dict(tuple(x.split("=")) for x in self.tags.split(","))

        self.mitiff_config = {}
        self.translate_channel_name = {}
        self.channel_order = {}

    def save_image(self):
        raise NotImplementedError("save_image mitiff is not implemented.")

    def save_dataset(self, dataset, filename=None, fill_value=None,
                     compute=True, **kwargs):
        LOG.debug("Starting in mitiff save_dataset ... ")

        def _delayed_create(create_opts, dataset):
            try:
                if 'platform_name' not in kwargs:
                    kwargs['platform_name'] = dataset.attrs['platform_name']
                if 'name' not in kwargs:
                    kwargs['name'] = dataset.attrs['name']
                if 'start_time' not in kwargs:
                    kwargs['start_time'] = dataset.attrs['start_time']
                if 'sensor' not in kwargs:
                    kwargs['sensor'] = dataset.attrs['sensor']

                try:
                    self.mitiff_config[kwargs['sensor']] = dataset.attrs['metadata_requirements']['config']
                    self.channel_order[kwargs['sensor']] = dataset.attrs['metadata_requirements']['order']
                    self.file_pattern = dataset.attrs['metadata_requirements']['file_pattern']
                except KeyError:
                    # For some mitiff products this info is needed, for others not.
                    # If needed you should know how to fix this
                    pass

                try:
                    self.translate_channel_name[kwargs['sensor']] = \
                        dataset.attrs['metadata_requirements']['translate']
                except KeyError:
                    # For some mitiff products this info is needed, for others not.
                    # If needed you should know how to fix this
                    pass

                image_description = self._make_image_description(dataset, **kwargs)
                gen_filename = filename or self.get_filename(**dataset.attrs)
                LOG.info("Saving mitiff to: %s ...", gen_filename)
                self._save_datasets_as_mitiff(dataset, image_description,
                                              gen_filename, **kwargs)
            except (KeyError, ValueError, RuntimeError):
                raise

        create_opts = ()
        delayed = dask.delayed(_delayed_create)(create_opts, dataset)

        if compute:
            return delayed.compute()
        return delayed

    def save_datasets(self, datasets, filename=None, fill_value=None,
                      compute=True, **kwargs):
        """Save all datasets to one or more files.
        """
        LOG.debug("Starting in mitiff save_datasets ... ")

        def _delayed_create(create_opts, datasets):
            LOG.debug("create_opts: %s", create_opts)
            try:
                if 'platform_name' not in kwargs:
                    kwargs['platform_name'] = datasets.attrs['platform_name']
                if 'name' not in kwargs:
                    kwargs['name'] = datasets[0].attrs['name']
                if 'start_time' not in kwargs:
                    kwargs['start_time'] = datasets[0].attrs['start_time']
                if 'sensor' not in kwargs:
                    kwargs['sensor'] = datasets[0].attrs['sensor']

                try:
                    self.mitiff_config[kwargs['sensor']] = datasets['metadata_requirements']['config']
                    self.translate_channel_name[kwargs['sensor']] = datasets['metadata_requirements']['translate']
                    self.channel_order[kwargs['sensor']] = datasets['metadata_requirements']['order']
                    self.file_pattern = datasets['metadata_requirements']['file_pattern']
                except KeyError:
                    # For some mitiff products this info is needed, for others not.
                    # If needed you should know how to fix this
                    pass

                image_description = self._make_image_description(datasets, **kwargs)
                LOG.debug("File pattern %s", self.file_pattern)
                if isinstance(datasets, list):
                    kwargs['start_time'] = datasets[0].attrs['start_time']
                else:
                    kwargs['start_time'] = datasets.attrs['start_time']
                gen_filename = filename or self.get_filename(**kwargs)
                LOG.info("Saving mitiff to: %s ...", gen_filename)
                self._save_datasets_as_mitiff(datasets, image_description, gen_filename, **kwargs)
            except (KeyError, ValueError, RuntimeError):
                raise

        create_opts = ()
        delayed = dask.delayed(_delayed_create)(create_opts, datasets)
        LOG.debug("About to call delayed compute ...")
        if compute:
            return delayed.compute()
        return delayed

    def _make_channel_list(self, datasets, **kwargs):
        channels = []
        try:
            if self.channel_order:
                for cn in self.channel_order[kwargs['sensor']]:
                    for ch, ds in enumerate(datasets):
                        if ds.attrs['prerequisites'][ch][0] == cn:
                            channels.append(
                                ds.attrs['prerequisites'][ch][0])
                            break
            else:
                for ch, ds in enumerate(datasets):
                    channels.append(ch + 1)
        except KeyError:
            for ch, ds in enumerate(datasets):
                channels.append(ch + 1)
        return channels

    def _channel_names(self, channels, cns, **kwargs):
        _image_description = ""
        for ch in channels:
            try:
                _image_description += str(
                    self.mitiff_config[kwargs['sensor']][cns.get(ch, ch)]['alias'])
            except KeyError:
                _image_description += str(ch)
            _image_description += ' '

        # Replace last char(space) with \n
        _image_description = _image_description[:-1]
        _image_description += '\n'

        return _image_description

    def _add_sizes(self, datasets, first_dataset):
        _image_description = ' Xsize: '
        if isinstance(datasets, list):
            _image_description += str(first_dataset.sizes['x']) + '\n'
        else:
            _image_description += str(datasets.sizes['x']) + '\n'

        _image_description += ' Ysize: '
        if isinstance(datasets, list):
            _image_description += str(first_dataset.sizes['y']) + '\n'
        else:
            _image_description += str(datasets.sizes['y']) + '\n'

        return _image_description

    def _add_proj4_string(self, datasets, first_dataset):
        proj4_string = " Proj string: "

        if isinstance(datasets, list):
            proj4_string += first_dataset.attrs['area'].proj4_string
        else:
            proj4_string += datasets.attrs['area'].proj4_string

        x_0 = 0
        y_0 = 0
        if 'EPSG:32631' in proj4_string:
            proj4_string = proj4_string.replace("+init=EPSG:32631",
                                                "+proj=etmerc +lat_0=0 +lon_0=3 +k=0.9996 +ellps=WGS84 +datum=WGS84")
            x_0 = 500000
        elif 'EPSG:32632' in proj4_string:
            proj4_string = proj4_string.replace("+init=EPSG:32632",
                                                "+proj=etmerc +lat_0=0 +lon_0=9 +k=0.9996 +ellps=WGS84 +datum=WGS84")
            x_0 = 500000
        elif 'EPSG:32633' in proj4_string:
            proj4_string = proj4_string.replace("+init=EPSG:32633",
                                                "+proj=etmerc +lat_0=0 +lon_0=15 +k=0.9996 +ellps=WGS84 +datum=WGS84")
            x_0 = 500000
        elif 'EPSG:32634' in proj4_string:
            proj4_string = proj4_string.replace("+init=EPSG:32634",
                                                "+proj=etmerc +lat_0=0 +lon_0=21 +k=0.9996 +ellps=WGS84 +datum=WGS84")
            x_0 = 500000
        elif 'EPSG:32635' in proj4_string:
            proj4_string = proj4_string.replace("+init=EPSG:32635",
                                                "+proj=etmerc +lat_0=0 +lon_0=27 +k=0.9996 +ellps=WGS84 +datum=WGS84")
            x_0 = 500000
        elif 'EPSG' in proj4_string:
            LOG.warning("EPSG used in proj string but not converted. Please add this in code")

        if 'geos' in proj4_string:
            proj4_string = proj4_string.replace("+sweep=x ", "")
            if '+a=6378137.0 +b=6356752.31414' in proj4_string:
                proj4_string = proj4_string.replace("+a=6378137.0 +b=6356752.31414",
                                                    "+ellps=WGS84")
            if '+units=m' in proj4_string:
                proj4_string = proj4_string.replace("+units=m", "+units=km")

        if not any(datum in proj4_string for datum in ['datum', 'towgs84']):
            proj4_string += ' +towgs84=0,0,0'

        if 'units' not in proj4_string:
            proj4_string += ' +units=km'

        if isinstance(datasets, list):
            proj4_string += ' +x_0=%.6f' % (
                (-first_dataset.attrs['area'].area_extent[0] +
                 first_dataset.attrs['area'].pixel_size_x) + x_0)
            proj4_string += ' +y_0=%.6f' % (
                (-first_dataset.attrs['area'].area_extent[1] +
                 first_dataset.attrs['area'].pixel_size_y) + y_0)
        else:
            proj4_string += ' +x_0=%.6f' % (
                (-datasets.attrs['area'].area_extent[0] +
                 datasets.attrs['area'].pixel_size_x) + x_0)
            proj4_string += ' +y_0=%.6f' % (
                (-datasets.attrs['area'].area_extent[1] +
                 datasets.attrs['area'].pixel_size_y) + y_0)

        LOG.debug("proj4_string: %s", proj4_string)
        proj4_string += '\n'

        return proj4_string

    def _add_pixel_sizes(self, datasets, first_dataset):
        _image_description = ""
        if isinstance(datasets, list):
            _image_description += ' Ax: %.6f' % (
                first_dataset.attrs['area'].pixel_size_x / 1000.)
            _image_description += ' Ay: %.6f' % (
                first_dataset.attrs['area'].pixel_size_y / 1000.)
        else:
            _image_description += ' Ax: %.6f' % (
                datasets.attrs['area'].pixel_size_x / 1000.)
            _image_description += ' Ay: %.6f' % (
                datasets.attrs['area'].pixel_size_y / 1000.)

        return _image_description

    def _add_corners(self, datasets, first_dataset):
        # But this ads up to upper left corner of upper left pixel.
        # But need to use the center of the pixel.
        # Therefor use the center of the upper left pixel.
        _image_description = ""
        if isinstance(datasets, list):
            _image_description += ' Bx: %.6f' % (
                first_dataset.attrs['area'].area_extent[0] / 1000. +
                first_dataset.attrs['area'].pixel_size_x / 1000. / 2.)  # LL_x
            _image_description += ' By: %.6f' % (
                first_dataset.attrs['area'].area_extent[3] / 1000. -
                first_dataset.attrs['area'].pixel_size_y / 1000. / 2.)  # UR_y
        else:
            _image_description += ' Bx: %.6f' % (
                datasets.attrs['area'].area_extent[0] / 1000. +
                datasets.attrs['area'].pixel_size_x / 1000. / 2.)  # LL_x
            _image_description += ' By: %.6f' % (
                datasets.attrs['area'].area_extent[3] / 1000. -
                datasets.attrs['area'].pixel_size_y / 1000. / 2.)  # UR_y

        _image_description += '\n'
        return _image_description

    def _add_calibration_datasets(self, ch, datasets, reverse_offset, reverse_scale, decimals):
        _reverse_offset = reverse_offset
        _reverse_scale = reverse_scale
        _decimals = decimals
        _table_calibration = ""
        found_calibration = False
        skip_calibration = False
        for i, ds in enumerate(datasets):
            if 'prerequisites' in ds.attrs and isinstance(ds.attrs['prerequisites'][i], DatasetID):
                if ds.attrs['prerequisites'][i][0] == ch:
                    if ds.attrs['prerequisites'][i][4] == 'RADIANCE':
                        raise NotImplementedError(
                            "Mitiff radiance calibration not implemented.")
                    # _table_calibration += ', Radiance, '
                    # _table_calibration += '[W/m²/µm/sr]'
                    # _decimals = 8
                    elif ds.attrs['prerequisites'][i][4] == 'brightness_temperature':
                        found_calibration = True
                        _table_calibration += ', BT, '
                        _table_calibration += u'\u00B0'  # '\u2103'
                        _table_calibration += u'[C]'

                        _reverse_offset = 255.
                        _reverse_scale = -1.
                        _decimals = 2
                    elif ds.attrs['prerequisites'][i][4] == 'reflectance':
                        found_calibration = True
                        _table_calibration += ', Reflectance(Albedo), '
                        _table_calibration += '[%]'
                        _decimals = 2
                    else:
                        LOG.warning("Unknown calib type. Must be Radiance, Reflectance or BT.")

                        break
                else:
                    continue
            else:
                _table_calibration = ""
                skip_calibration = True
                break
        if not found_calibration:
            _table_calibration = ""
            skip_calibration = True
            # How to format string by passing the format
            # http://stackoverflow.com/questions/1598579/rounding-decimals-with-new-python-format-function
        return skip_calibration, _table_calibration, _reverse_offset, _reverse_scale, _decimals

    def _add_calibration(self, channels, cns, datasets, **kwargs):

        _table_calibration = ""
        skip_calibration = False
        for ch in channels:

            palette = False
            # Make calibration.
            if palette:
                raise NotImplementedError("Mitiff palette saving is not implemented.")
            else:
                _table_calibration += 'Table_calibration: '
                try:
                    _table_calibration += str(
                        self.mitiff_config[kwargs['sensor']][cns.get(ch, ch)]['alias'])
                except KeyError:
                    _table_calibration += str(ch)

                _reverse_offset = 0.
                _reverse_scale = 1.
                _decimals = 2

                skip_calibration, __table_calibration, _reverse_offset, _reverse_scale, _decimals = \
                    self._add_calibration_datasets(ch, datasets, _reverse_offset, _reverse_scale, _decimals)
                _table_calibration += __table_calibration

                if not skip_calibration:
                    _table_calibration += ', 8, [ '
                    for val in range(0, 256):
                        # Comma separated list of values
                        _table_calibration += '{0:.{1}f} '.format((float(self.mitiff_config[
                            kwargs['sensor']][cns.get(ch, ch)]['min-val']) +
                            ((_reverse_offset + _reverse_scale * val) *
                             (float(self.mitiff_config[kwargs['sensor']][cns.get(ch, ch)]['max-val']) -
                             float(self.mitiff_config[kwargs['sensor']][cns.get(ch, ch)]['min-val']))) / 255.),
                            _decimals)
                        # _table_calibration += '0.00000000 '

                    _table_calibration += ']\n\n'
                else:
                    _table_calibration = ""

        return _table_calibration

    def _make_image_description(self, datasets, **kwargs):
        """
        generate image description for mitiff.

        Satellite: NOAA 18
        Date and Time: 06:58 31/05-2016
        SatDir: 0
        Channels:   6 In this file: 1-VIS0.63 2-VIS0.86 3(3B)-IR3.7
        4-IR10.8 5-IR11.5 6(3A)-VIS1.6
        Xsize:  4720
        Ysize:  5544
        Map projection: Stereographic
        Proj string: +proj=stere +lon_0=0 +lat_0=90 +lat_ts=60
        +ellps=WGS84 +towgs84=0,0,0 +units=km
        +x_0=2526000.000000 +y_0=5806000.000000
        TrueLat: 60 N
        GridRot: 0
        Xunit:1000 m Yunit: 1000 m
        NPX: 0.000000 NPY: 0.000000
        Ax: 1.000000 Ay: 1.000000 Bx: -2526.000000 By: -262.000000

        Satellite: <satellite name>
        Date and Time: <HH:MM dd/mm-yyyy>
        SatDir: 0
        Channels:   <number of chanels> In this file: <channels names in order>
        Xsize:  <number of pixels x>
        Ysize:  <number of pixels y>
        Map projection: Stereographic
        Proj string: <proj4 string with +x_0 and +y_0 which is the positive
        distance from proj origo
        to the lower left corner of the image data>
        TrueLat: 60 N
        GridRot: 0
        Xunit:1000 m Yunit: 1000 m
        NPX: 0.000000 NPY: 0.000000
        Ax: <pixels size x in km> Ay: <pixel size y in km> Bx: <left corner of
        upper right pixel in km>
        By: <upper corner of upper right pixel in km>


        if palette image write special palette
        if normal channel write table calibration:
        Table_calibration: <channel name>, <calibration type>, [<unit>],
        <no of bits of data>,
        [<calibration values space separated>]\n\n
        """

        translate_platform_name = {'metop01': 'Metop-B',
                                   'metop02': 'Metop-A',
                                   'metop03': 'Metop-C',
                                   'noaa15': 'NOAA-15',
                                   'noaa16': 'NOAA-16',
                                   'noaa17': 'NOAA-17',
                                   'noaa18': 'NOAA-18',
                                   'noaa19': 'NOAA-19'}

        first_dataset = datasets
        if isinstance(datasets, list):
            LOG.debug("Datasets is a list of dataset")
            first_dataset = datasets[0]

        if 'platform_name' in first_dataset.attrs:
            _platform_name = translate_platform_name.get(
                first_dataset.attrs['platform_name'],
                first_dataset.attrs['platform_name'])
        elif 'platform_name' in kwargs:
            _platform_name = translate_platform_name.get(
                kwargs['platform_name'], kwargs['platform_name'])
        else:
            _platform_name = None

        _image_description = ''
        _image_description.encode('utf-8')

        _image_description += ' Satellite: '
        if _platform_name is not None:
            _image_description += _platform_name

        _image_description += '\n'

        _image_description += ' Date and Time: '
        # Select earliest start_time
        first = True
        earliest = 0
        for dataset in datasets:
            if first:
                earliest = dataset.attrs['start_time']
            else:
                if dataset.attrs['start_time'] < earliest:
                    earliest = dataset.attrs['start_time']
            first = False
        LOG.debug("earliest start_time: %s", earliest)
        _image_description += earliest.strftime("%H:%M %d/%m-%Y\n")

        _image_description += ' SatDir: 0\n'

        _image_description += ' Channels: '

        if isinstance(datasets, list):
            LOG.debug("len datasets: %s", len(datasets))
            _image_description += str(len(datasets))
        elif 'bands' in datasets.sizes:
            LOG.debug("len datasets: %s", datasets.sizes['bands'])
            _image_description += str(datasets.sizes['bands'])
        elif len(datasets.sizes) == 2:
            LOG.debug("len datasets: 1")
            _image_description += '1'

        _image_description += ' In this file: '

        channels = self._make_channel_list(datasets, **kwargs)

        try:
            cns = self.translate_channel_name.get(kwargs['sensor'], {})
        except KeyError:
            pass

        _image_description += self._channel_names(channels, cns, **kwargs)

        _image_description += self._add_sizes(datasets, first_dataset)

        _image_description += ' Map projection: Stereographic\n'

        _image_description += self._add_proj4_string(datasets, first_dataset)

        _image_description += ' TrueLat: 60N\n'
        _image_description += ' GridRot: 0\n'

        _image_description += ' Xunit:1000 m Yunit: 1000 m\n'

        _image_description += ' NPX: %.6f' % (0)
        _image_description += ' NPY: %.6f' % (0) + '\n'

        _image_description += self._add_pixel_sizes(datasets, first_dataset)
        _image_description += self._add_corners(datasets, first_dataset)

        if isinstance(datasets, list):
            LOG.debug("Area extent: %s", first_dataset.attrs['area'].area_extent)
        else:
            LOG.debug("Area extent: %s", datasets.attrs['area'].area_extent)

        _image_description += self._add_calibration(channels, cns, datasets, **kwargs)

        return _image_description

    def _save_datasets_as_mitiff(self, datasets, image_description,
                                 gen_filename, **kwargs):
        """Put all togehter and save as a tiff file with the special tag
           making it a mitiff file.
        """
        from libtiff import TIFF

        tif = TIFF.open(gen_filename, mode='w')

        tif.SetField(IMAGEDESCRIPTION, (image_description).encode('utf-8'))

        cns = self.translate_channel_name.get(kwargs['sensor'], {})
        if isinstance(datasets, list):
            LOG.debug("Saving datasets as list")

            for _cn in self.channel_order[kwargs['sensor']]:
                for dataset in datasets:
                    if dataset.attrs['name'] == _cn:
                        reverse_offset = 0.
                        reverse_scale = 1.
                        if dataset.attrs['calibration'] == 'brightness_temperature':
                            reverse_offset = 255.
                            reverse_scale = -1.
                            dataset.data += KELVIN_TO_CELSIUS

                        # Need to possible translate channels names from satpy to mitiff
                        cn = cns.get(dataset.attrs['name'], dataset.attrs['name'])
                        _data = reverse_offset + reverse_scale * (((dataset.data - float(self.mitiff_config[
                            kwargs['sensor']][cn]['min-val'])) /
                            (float(self.mitiff_config[kwargs['sensor']][cn]['max-val']) -
                             float(self.mitiff_config[kwargs['sensor']][cn]['min-val']))) * 255.)
                        data = _data.clip(0, 255)

                        tif.write_image(data.astype(np.uint8), compression='deflate')
                        break
        elif 'dataset' in datasets.attrs['name']:
            LOG.debug("Saving %s as a dataset.", datasets.attrs['name'])
            for _cn in self.channel_order[kwargs['sensor']]:
                for i, band in enumerate(datasets['bands']):
                    if band == _cn:
                        chn = datasets.sel(bands=band)
                        reverse_offset = 0.
                        reverse_scale = 1.
                        if chn.attrs['prerequisites'][i][4] == 'brightness_temperature':
                            reverse_offset = 255.
                            reverse_scale = -1.
                            chn.data += KELVIN_TO_CELSIUS

                        # Need to possible translate channels names from satpy to mitiff
                        cn = cns.get(chn.attrs['prerequisites'][i][0],
                                     chn.attrs['prerequisites'][i][0])
                        _data = reverse_offset + reverse_scale * (((chn.data - float(self.mitiff_config[
                            kwargs['sensor']][cn]['min-val'])) /
                            (float(self.mitiff_config[kwargs['sensor']][cn]['max-val']) -
                             float(self.mitiff_config[kwargs['sensor']][cn]['min-val']))) * 255.)
                        data = _data.clip(0, 255)

                        tif.write_image(data.astype(np.uint8), compression='deflate')
                        break

        else:
            LOG.debug("Saving datasets as enhanced image")
            img = get_enhanced_image(datasets.squeeze(), enhance=self.enhancer)
            for i, band in enumerate(img.data['bands']):
                chn = img.data.sel(bands=band)
                data = chn.values * 254. + 1
                data = data.clip(0, 255)
                tif.write_image(data.astype(np.uint8), compression='deflate')

        del tif
