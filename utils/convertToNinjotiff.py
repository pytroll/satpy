import os, sys
from satpy.utils import debug_on
import pyninjotiff
debug_on()
from satpy import Scene
import argparse

parser = argparse.ArgumentParser(description='Turn an image into a NinjoTiff.')
parser.add_argument('--input_dir', dest='input_dir', action="store", help="Directory with input data, that must contain a timestamp in the filename.")
parser.add_argument('--chan_id', dest='chan_id', action="store", help="Channel ID", default="9999")
parser.add_argument('--sat_id', dest='sat_id', action="store", help="Satellite ID", default="8888")
parser.add_argument('--data_cat', dest='data_cat', action="store", help="Category of data (one of GORN, GPRN, PORN)", default="GORN")
parser.add_argument('--area', dest='area', action="store", help="Area name, the definition must exist in your areas.def configuration file.", default="nrEURO3km")
parser.add_argument('--ph_unit', dest='ph_unit', action="store", help="Physical unit.", default="CELSIUS")
args = parser.parse_args()

if (args.input_dir != None):
    os.chdir(args.input_dir)

global_data = Scene(sensor="images", reader="generic_image")
global_data.load(['image'])
#print(global_data)
#global_data.save_dataset('image', filename="out.png", writer="simple_image")
global_data.save_dataset('image', filename="out.tif", writer="ninjotiff",
                      sat_id=args.sat_id,
                      chan_id=args.chan_id,
                      data_cat=args.data_cat,
                      data_source='EUMCAST',
                      physic_unit=args.ph_unit)

