#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Simple command line too that converts an image into a NinJo Tiff file.

NinJo Tiff metadata can be passed as command line input or
through a config file (an example is given in the ninjo-cmd.yaml
file in the etc directory).

The area matching the input image shall be defined in the
areas configuration file (located in $PPP_CONFIG_DIR).

"""

import argparse
import os

import yaml
from yaml import UnsafeLoader

from satpy import Scene
from satpy.pyresample import get_area_def
from satpy.utils import debug_on

debug_on()

parser = argparse.ArgumentParser(description='Turn an image into a NinjoTiff.')
parser.add_argument('--cfg', dest='cfg', action="store",
                    help="YAML configuration as an alternative to the command line input for NinJo metadata.")
parser.add_argument('--input_dir', dest='input_dir', action="store",
                    help="Directory with input data, that must contain a timestamp in the filename.")
parser.add_argument('--chan_id', dest='chan_id', action="store", help="Channel ID", default="9999")
parser.add_argument('--sat_id', dest='sat_id', action="store", help="Satellite ID", default="8888")
parser.add_argument('--data_cat', dest='data_cat', action="store",
                    help="Category of data (one of GORN, GPRN, PORN)", default="GORN")
parser.add_argument('--area', dest='areadef', action="store",
                    help="Area name, the definition must exist in your areas configuration file",
                    default="nrEURO1km_NPOL_COALeqc")
parser.add_argument('--ph_unit', dest='ph_unit', action="store", help="Physical unit", default="CELSIUS")
parser.add_argument('--data_src', dest='data_src', action="store", help="Data source", default="EUMETCAST")
args = parser.parse_args()

if (args.input_dir is not None):
    os.chdir(args.input_dir)

cfg = vars(args)
if (args.cfg is not None):
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=UnsafeLoader)

narea = get_area_def(args.areadef)
global_data = Scene(reader="generic_image")
global_data.load(['image'])

global_data['image'].info['area'] = narea
fname = global_data['image'].info['filename']
ofname = fname[:-3] + "tif"

# global_data.save_dataset('image', filename="out.png", writer="simple_image")
global_data.save_dataset('image', filename=ofname, writer="ninjotiff",
                         sat_id=cfg['sat_id'],
                         chan_id=cfg['chan_id'],
                         data_cat=cfg['data_cat'],
                         data_source=cfg['data_src'],
                         physic_unit=cfg['ph_unit'])
