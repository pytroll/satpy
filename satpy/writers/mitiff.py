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

from satpy.utils import ensure_dir
from satpy.writers import ImageWriter

from satpy.config import get_environ_config_dir
from satpy.composites import CompositorLoader
from satpy.writers import get_enhanced_image

import os

import dask

import yaml

IMAGEDESCRIPTION = 270

LOG = logging.getLogger(__name__)

KELVIN_TO_CELSIUS = -273.15

class MITIFFWriter(ImageWriter):

    def __init__(self, floating_point=False, tags=None, **kwargs):
        ImageWriter.__init__(self,
                             default_config_filename="writers/mitiff.yaml",
                             **kwargs)

        self.tags = self.info.get("tags",
                                  None) if tags is None else tags
        if self.tags is None:
            self.tags = {}
        elif not isinstance(self.tags, dict):
            # if it's coming from a config file
            self.tags = dict(tuple(x.split("=")) for x in self.tags.split(","))

        print "kwargs: {}".format(kwargs)
        #self.base_dir = kwargs['base_dir']
        LOG.debug("1self.config: {}".format(self.config))
        self.config = {}
        LOG.debug("2self.config: {}".format(self.config))

        self.mitiff_config = { 'avhrr-3' : { '1' : { 'alias': '1-VIS0.63', 'calibration': 'reflectance', 'min-val': '0', 'max-val': '100'},
                                             '2' : { 'alias': '2-VIS0.86', 'calibration': 'reflectance', 'min-val': '0', 'max-val': '100'},
                                             '3' : { 'alias': '3(3B)-IR3.7', 'calibration': 'brightness_temperature', 'min-val': '-150', 'max-val': '50'},
                                             '4' : { 'alias': '4-IR10.8', 'calibration': 'brightness_temperature', 'min-val': '-150', 'max-val': '50'},
                                             '5' : { 'alias': '5-IR11.5', 'calibration': 'brightness_temperature', 'min-val': '-150', 'max-val': '50'},
                                             '6' : { 'alias': '6(3A)-VIS1.6', 'calibration': 'reflectance', 'min-val':  '0', 'max-val': '100'}
        }}
        self.translate_channel_name = {'avhrr-3' : {'1':'1',
                                                    '2':'2',
                                                    '3a':'6',
                                                    '4':'4',
                                                    '5':'5',
                                                    '3b':'3'}
        }
        self.channel_order = {'avhrr-3': ['1', '2', '3b', '4', '5', '3a']}
        self.lc = {}
        
    def save_dataset(self, dataset, filename=None, fill_value=None,
                     compute=True, base_dir=None, **kwargs):
        LOG.debug("Starting in mitiff save_dataset ... ")
        def _delayed_create(create_opts, dataset):#, area, start_time, tags):
            try:
                #if type(kwargs["sensor"]) not in (tuple, list, set):
                #    kwargs['sensor'] = kwargs['sensor'].replace("/","-")
                #if kwargs['sensor'] not in self.config:
                #    LOG.error("Sensor {} not defined in config. Go fix your config!".format(kwargs['sensor']))
                #    return False
                #else:
                #    for i,sensor in enumerate(kwargs["sensor"]):
                #        kwargs["sensor"][i] = sensor.replace("/","-")
                #if sensor not in self.config:
                #    LOG.error("Sensor {} not defined in config. Go fix your config!".format(sensor))
                #    return False
                
                if 'platform_name' not in kwargs:
                    kwargs['platform_name'] = dataset.attrs['platform_name']
                if 'name' not in kwargs:
                    kwargs['name'] = dataset.attrs['name']
                if 'start_time' not in kwargs:
                    kwargs['start_time'] = dataset.attrs['start_time']
                if 'sensor' not in kwargs:
                    kwargs['sensor'] = dataset.attrs['sensor']
                
                image_description = self._make_image_description(dataset, **kwargs)
                print "File pattern {}".format(self.file_pattern)
                print "kwargs {}".format(kwargs)
                LOG.info("Saving mitiff to: {} ...".format(self.get_filename(**kwargs)))
                gen_filename = self.get_filename(**kwargs)
                self._save_datasets_as_mitiff(dataset, image_description, gen_filename, **kwargs)
            except:
                raise

        #return gen_filename
        create_opts=()
        delayed = dask.delayed(_delayed_create)(create_opts, dataset)
        delayed.compute()
        return delayed
                     
    def save_datasets(self, datasets, **kwargs):
        """Save all datasets to one or more files.
        """
        LOG.debug("Starting in mitiff save_datasetsssssssssssssssss ... ")
        LOG.debug("kwargs: {}".format(kwargs))

        def _delayed_create(create_opts, datasets):#, area, start_time, tags):
            try:
                #if type(kwargs["sensor"]) not in (tuple, list, set):
                #    kwargs['sensor'] = kwargs['sensor'].replace("/","-")
                #if kwargs['sensor'] not in self.config:
                #    LOG.error("Sensor {} not defined in config. Go fix your config!".format(kwargs['sensor']))
                #    return False
                #else:
                #    for i,sensor in enumerate(kwargs["sensor"]):
                #        kwargs["sensor"][i] = sensor.replace("/","-")
                #if sensor not in self.config:
                #    LOG.error("Sensor {} not defined in config. Go fix your config!".format(sensor))
                #    return False
                if 'mitiff_config_file' in kwargs:
                    print "Using provided mitiff config file: ",kwargs['mitiff_config_file']
                    config_file = kwargs['mitiff_config_file']
                    with open(config_file) as fd:
                        self.lc = yaml.load(fd)
                        if not isinstance(self.lc, dict):
                            raise ValueError("YAML file doesn't exist or string is not YAML dict: {}".format(config_file))
                        print "Mitiff config is: ",self.lc
                if 'platform_name' not in kwargs:
                    kwargs['platform_name'] = datasets.attrs['platform_name']
                if 'name' not in kwargs:
                    kwargs['name'] = datasets[0].attrs['name']
                if 'start_time' not in kwargs:
                    kwargs['start_time'] = datasets[0].attrs['start_time']
                if 'sensor' not in kwargs:
                    kwargs['sensor'] = datasets[0].attrs['sensor']
                
                for mitiff_product in self.lc:
                    self.mitiff_config[kwargs['sensor']] = self.lc[mitiff_product]['channels']
                    self.translate_channel_name[kwargs['sensor']] = self.lc[mitiff_product]['translate_channel_name']
                    self.channel_order[kwargs['sensor']] = self.lc[mitiff_product]['channel_order']
                    self.file_pattern = self.lc[mitiff_product]['file_pattern']
                    image_description = self._make_image_description(datasets, **kwargs)
                    print "File pattern {}".format(self.file_pattern)
                    #kwargs['name']  ="shallalal"
                    if type(datasets) in (list,):
                        kwargs['start_time'] = datasets[0].attrs['start_time']
                    else:
                        kwargs['start_time'] = datasets.attrs['start_time']
                    print kwargs
                    self.filename_parser = self.create_filename_parser(kwargs['mitiff_dir'])
                    LOG.info("Saving mitiff to: {} ...".format(self.get_filename(**kwargs)))
                    print "Saving mitiff to: {} ...".format(self.get_filename(**kwargs))
                    gen_filename = self.get_filename(**kwargs)
                    self._save_datasets_as_mitiff(datasets, image_description, gen_filename, **kwargs)
            except:
                raise

        #return gen_filename
        create_opts=()
        delayed = dask.delayed(_delayed_create)(create_opts, datasets)
        print "About to call delayed compute ..."
        delayed.compute()
        return delayed
            
#    def save_dataset(self, dataset, filename=None, fill_value=None, overlay=None, **kwargs):
#        """Saves the *dataset* to a given *filename*.
#        """
#        LOG.debug("Starting in save_dataset ... ")
#        with tifffile.TiffWriter(output_fn, **tifargs) as tif:
#            tif.save(image_data, **args)


#    def save_image(self, img, filename=None, floating_point=False, **kwargs):
#        """Save the image to the given *filename* in mitiff format.
#        `floating_point` allows the saving of
#        'L' mode images in floating point format if set to True.
#        """

#        LOG.debug("About to start save_image in mitiff")
#        filename = filename or self.get_filename(**img.info)
#        mt.save(img, filename, **kwargs)
#        LOG.debug("save_image complete in mitiff")

#    def _make_calibration_string(self, datasets)
#        """Make the calibration string to insert in as a tifftag description.
#        """

    def _make_image_description(self, datasets, **kwargs):
        #generate image desdcription for mitiff.
        """
        Satellite: NOAA 18
        Date and Time: 06:58 31/05-2016
        SatDir: 0
        Channels:   6 In this file: 1-VIS0.63 2-VIS0.86 3(3B)-IR3.7 4-IR10.8 5-IR11.5 6(3A)-VIS1.6
        Xsize:  4720
        Ysize:  5544
        Map projection: Stereographic
        Proj string: +proj=stere +lon_0=0 +lat_0=90 +lat_ts=60 +ellps=WGS84 +towgs84=0,0,0 +units=km +x_0=2526000.000000 +y_0=5806000.000000
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
        Proj string: <proj4 string with +x_0 and +y_0 which is the positive distance from proj origo to the lower left corner of the image data>
        TrueLat: 60 N
        GridRot: 0
        Xunit:1000 m Yunit: 1000 m
        NPX: 0.000000 NPY: 0.000000
        Ax: <pixels size x in km> Ay: <pixel size y in km> Bx: <left corner of upper right pixel in km> By: <upper corner of upper right pixel in km>
     
        
        if palette image write special palette
        if normal channel write table calibration:
        Table_calibration: <channel name>, <calibration type>, [<unit>], <no of bits of data>, [<calibration values space separated>]\n\n
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
        if type(datasets) in (list,):
            print "Datasets is a list og datasets"
            first_dataset = datasets[0]
            
        if 'platform_name' in first_dataset.attrs:
            _platform_name = translate_platform_name.get(first_dataset.attrs['platform_name'],first_dataset.attrs['platform_name'])
        elif 'platform_name' in kwargs:
            _platform_name = translate_platform_name.get(kwargs['platform_name'],kwargs['platform_name'])
        else:
            _platform_name = None
            
        _image_description = ''
        _image_description.encode('utf-8')

        _image_description += ' Satellite: '
        if ( _platform_name != None ):
            _image_description += _platform_name
    
        _image_description += '\n'
        
        _image_description += ' Date and Time: '
        #Select earliest start_time
        first = True
        earliest = 0
        for dataset in datasets:
            if first:
                earliest = dataset.attrs['start_time']
            else:
                if dataset.attrs['start_time'] < earliest:
                    earliest = dataset.attrs['start_time']
            first=False
        print "earliest: ",earliest
        print "---------------------------------------------------------------------------"
        _image_description += earliest.strftime("%H:%M %d/%m-%Y\n")
       
        _image_description += ' SatDir: 0\n'
    
        _image_description += ' Channels: '

        #print "datasets in make_image_desc: {}".format(datasets)
        print "---------------------------------------------------------------------------"

        print "first_dataset.attrs: ",first_dataset.attrs
        print "first_dataset: ",first_dataset
        if type(datasets) in (list,):
            print "len datasets: {}".format(len(datasets))
            _image_description += str(len(datasets))
        else:
            print "len datasets: {}".format(datasets.sizes['bands'])
            print "---------------------------------------------------------------------------"
            _image_description += str(datasets.sizes['bands'])

        _image_description += ' In this file: '
        #tcn = translate_channel_name.get(kwargs['sensor'][0])

        channels = []
        print datasets[0].attrs['name']
        print "---------------------------------------------------------------------------"
        for cn in self.channel_order[kwargs['sensor']]:#
            for ch in xrange(len(datasets)):
                if datasets[ch].attrs['name'] == cn:
                    channels.append(datasets[ch].attrs['name'])
                    break

        print "channels: ",channels
        
        cns = self.translate_channel_name.get(kwargs['sensor'],{})
        for ch in channels:
            print ch
            #print datasets.attrs['metadata_requirements']
            #print datasets[0].info['metadata_requirements'][ch]
            #try:
            #    _image_description += datasets[0].info['metadata_requirements'][ch]['alias']
            #except KeyError:
            try:
                _image_description += str(self.mitiff_config[kwargs['sensor']][cns.get(ch,ch)]['alias'])
            except KeyError:
                _image_description += str(ch)
            _image_description += ' '
        
        #Replace last char(space) with \n
        _image_description = _image_description[:-1]
        _image_description += '\n'
       
        _image_description += ' Xsize: '
        if type(datasets) in (list,):
            _image_description += str(first_dataset.sizes['x']) + '\n'
        else:
            _image_description += str(datasets.sizes['x']) + '\n'
    
        _image_description += ' Ysize: '
        if type(datasets) in (list,):
            _image_description += str(first_dataset.sizes['y']) + '\n'
        else:
            _image_description += str(datasets.sizes['y']) + '\n'
            
        _image_description += ' Map projection: Stereographic\n'
        if type(datasets) in (list,):
            print first_dataset.attrs['area'].proj4_string
            proj4_string = first_dataset.attrs['area'].proj4_string
        else:
            print datasets.attrs['area'].proj4_string
            proj4_string = datasets.attrs['area'].proj4_string

        if 'geos' in proj4_string:
            proj4_string = proj4_string.replace("+sweep=x ","")
            if '+a=6378137.0 +b=6356752.31414' in proj4_string:
                proj4_string = proj4_string.replace("+a=6378137.0 +b=6356752.31414","+ellps=WGS84")
            if '+units=m' in proj4_string:
                proj4_string = proj4_string.replace("+units=m","+units=km")
                
        _image_description += ' Proj string: ' + proj4_string
        if not all( datum in proj4_string for datum in ['datum','towgs84']):
            _image_description += ' +towgs84=0,0,0'

        if not 'units' in proj4_string:
            _image_description += ' +units=km'
        
        #Need to use center of lower left pixel. Subtract half a pixel size
        #image_description += ' +x_0=%.6f' % (-datasets[0].info['area'].area_extent[0]-datasets[0].info['area'].pixel_size_x/2.)
        #image_description += ' +y_0=%.6f' % (-datasets[0].info['area'].area_extent[1]-datasets[0].info['area'].pixel_size_y/2.)
        if type(datasets) in (list,):
            _image_description += ' +x_0=%.6f' % (-first_dataset.attrs['area'].area_extent[0]+first_dataset.attrs['area'].pixel_size_x)
            _image_description += ' +y_0=%.6f' % (-first_dataset.attrs['area'].area_extent[1]+first_dataset.attrs['area'].pixel_size_y)
        else:
            _image_description += ' +x_0=%.6f' % (-datasets.attrs['area'].area_extent[0]+datasets.attrs['area'].pixel_size_x)
            _image_description += ' +y_0=%.6f' % (-datasets.attrs['area'].area_extent[1]+datasets.attrs['area'].pixel_size_y)

        _image_description += '\n'
        _image_description += ' TrueLat: 60N\n'
        _image_description += ' GridRot: 0\n'
    
        _image_description += ' Xunit:1000 m Yunit: 1000 m\n'

        _image_description += ' NPX: %.6f' % (0)
        _image_description += ' NPY: %.6f' % (0) + '\n'

        if type(datasets) in (list,):
            _image_description += ' Ax: %.6f' % (first_dataset.attrs['area'].pixel_size_x/1000.)
            _image_description += ' Ay: %.6f' % (first_dataset.attrs['area'].pixel_size_y/1000.)
        else:
            _image_description += ' Ax: %.6f' % (datasets.attrs['area'].pixel_size_x/1000.)
            _image_description += ' Ay: %.6f' % (datasets.attrs['area'].pixel_size_y/1000.)

            #But this ads up to upper left corner of upper left pixel.
        #But need to use the center of the pixel. Therefor use the center of the upper left pixel.
        if type(datasets) in (list,):
            _image_description += ' Bx: %.6f' % (first_dataset.attrs['area'].area_extent[0]/1000. + first_dataset.attrs['area'].pixel_size_x/1000./2.) #LL_x
            _image_description += ' By: %.6f' % (first_dataset.attrs['area'].area_extent[3]/1000. - first_dataset.attrs['area'].pixel_size_y/1000./2.) #UR_y
        else:
            _image_description += ' Bx: %.6f' % (datasets.attrs['area'].area_extent[0]/1000. + datasets.attrs['area'].pixel_size_x/1000./2.) #LL_x
            _image_description += ' By: %.6f' % (datasets.attrs['area'].area_extent[3]/1000. - datasets.attrs['area'].pixel_size_y/1000./2.) #UR_y

        _image_description += '\n'
    
        if type(datasets) in (list,):
            LOG.debug("Area extent: {}".format(first_dataset.attrs['area'].area_extent))
        else:
            LOG.debug("Area extent: {}".format(datasets.attrs['area'].area_extent))

        for ch in channels:
            found_channel = False
            print ch
                    
            palette=False
            #Make calibration.
            if palette:
                raise NotImplementedError("Mitiff palette saving is not implemented.")
            else:
                _image_description += 'Table_calibration: '
                #try:
                #    _image_description += datasets[0].info['metadata_requirements'][ch]['alias']
                #except KeyError:
                try:
                    _image_description += str(self.mitiff_config[kwargs['sensor']][cns.get(ch,ch)]['alias'])
                except KeyError:
                    _image_description += str(ch)
                    #_image_description += str(ch.name)

                _reverse_offset = 0.;
                _reverse_scale = 1.;

                #FIXME need to correlate the configured calibration and the calibration for the dataset.
                try:
                    if ch.calibration == 'RADIANCE':
                        raise NotImplementedError("Mitiff radiance calibration not implemented.")
                    #_image_description += ', Radiance, '
                    #_image_description += '[W/m²/µm/sr]'
                    #_decimals = 8
                    elif ch.calibration == 'brightness_temperature':
                        _image_description += ', BT, '
                        _image_description += u'\u00B0'#'\u2103'
                        _image_description += u'[C]'
                        
                        _reverse_offset = 255.;
                        _reverse_scale = -1.;
                        _decimals = 2
                    elif ch.calibration == 'reflectance':
                        _image_description += ', Reflectance(Albedo), '
                        _image_description += '[%]'
                        _decimals = 2
                    else:
                        LOG.warning("Unknown calib type. Must be Radiance, Reflectance or BT.")
                except AttributeError:
                    for ds in datasets:
                        if ds.attrs['name'] == ch:
                            print ch
                            if ds.attrs['calibration'] == 'RADIANCE':
                                raise NotImplementedError("Mitiff radiance calibration not implemented.")
                            #_image_description += ', Radiance, '
                            #_image_description += '[W/m²/µm/sr]'
                            #_decimals = 8
                            elif ds.attrs['calibration'] == 'brightness_temperature':
                                _image_description += ', BT, '
                                _image_description += u'\u00B0'#'\u2103'
                                _image_description += u'[C]'
                        
                                _reverse_offset = 255.;
                                _reverse_scale = -1.;
                                _decimals = 2
                            elif ds.attrs['calibration'] == 'reflectance':
                                _image_description += ', Reflectance(Albedo), '
                                _image_description += '[%]'
                                _decimals = 2
                            else:
                                LOG.warning("Unknown calib type. Must be Radiance, Reflectance or BT.")
                            
                            break;
                        else:
                            continue
                            #How to format string by passing the format
                #http://stackoverflow.com/questions/1598579/rounding-decimals-with-new-python-format-function
            
                _image_description += ', 8, [ '
                for val in range(0,256):
                    #Comma separated list of values
                    #calib.append(boost::str(boost::format("%.8f ") % (prod_chan_it->min_val + (val * (prod_chan_it->max_val - prod_chan_it->min_val)) / 255.)));
                    _image_description += '{0:.{1}f} '.format((float(self.mitiff_config[kwargs['sensor']][cns.get(ch,ch)]['min-val']) + ( (_reverse_offset + _reverse_scale*val) * ( float(self.mitiff_config[kwargs['sensor']][cns.get(ch,ch)]['max-val']) - float(self.mitiff_config[kwargs['sensor']][cns.get(ch,ch)]['min-val'])))/255.),_decimals)
                    #_image_description += '0.00000000 '
                
                _image_description += ']\n\n'
                    
        return _image_description

    def _save_datasets_as_mitiff(self, datasets, image_description, gen_filename, **kwargs):
        """Put all togehter and save as a tiff file with the special tag making it a 
           mitiff file.
        """
        from libtiff import TIFF

        tif = TIFF.open(gen_filename, mode ='w')
        
        tif.SetField(IMAGEDESCRIPTION, (image_description).encode('utf-8'))
        
        cns = self.translate_channel_name.get(kwargs['sensor'],{})
        if type(datasets) in (list,):
            for _cn in self.channel_order[kwargs['sensor']]:#
                for dataset in datasets:
                    if dataset.attrs['name'] == _cn:
                        print "Doing: ",dataset.attrs['name']
                        reverse_offset = 0.
                        reverse_scale = 1.
                        if dataset.attrs['calibration'] == 'brightness_temperature':
                            reverse_offset = 255.
                            reverse_scale = -1.
                            dataset.data += KELVIN_TO_CELSIUS
                            print "BT"
                    
                        #Need to possible translate channels names from satpy to mitiff
                        cn = cns.get(dataset.attrs['name'],dataset.attrs['name'])
                        _data=reverse_offset + reverse_scale*(((dataset.data-float(self.mitiff_config[kwargs['sensor']][cn]['min-val']))/(float(self.mitiff_config[kwargs['sensor']][cn]['max-val']) - float(self.mitiff_config[kwargs['sensor']][cn]['min-val'])))*255.)
                        data = _data.clip(0,255)

                        tif.write_image(data.astype(np.uint8), compression='deflate')
                        break
        else:
            img = get_enhanced_image( datasets.squeeze(), self.enhancer)
            print img
            for i,band in enumerate(img.data['bands']):
                print band
                chn = img.data.sel(bands=band)
                data = chn.values*254. + 1
                data = data.clip(0,255)
                tif.write_image(data.astype(np.uint8), compression='deflate')
            
        tif.close
