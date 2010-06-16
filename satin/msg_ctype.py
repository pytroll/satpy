#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of the mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Module to handle cloudtype data.
"""
import pps_gisdata # From NWCSAF/PPS - ACPG

import epshdf # From NWCSAF/PPS - ACPG
import area
from pp.channel import GenericChannel
import numpy

import logging
import ConfigParser
import glob
from satin import CONFIG_PATH
import os.path

CONF = ConfigParser.ConfigParser()
CONF.read(os.path.join(CONFIG_PATH, "meteosat09.cfg"))

MODE = os.getenv("SMHI_MODE")
if MODE is None:
    MODE = "offline"

CTYPE_DIR = CONF.get(MODE, 'ctype_dir')

MSG_PGE_EXTENTIONS = ["PLAX.CTTH.0.h5", "PLAX.CLIM.0.h5", "h5"]

LOG = logging.getLogger('satin')


# ----------------------------------------
class MsgCloudTypeData(object):
    """NWCSAF/MSG Cloud Type data layer
    """    
    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""
        
class MsgCloudType(GenericChannel):
    """NWCSAF/MSG Cloud Type data structure as retrieved from HDF5
    file. Resolution sets the nominal resolution of the data.
    """
    def __init__(self, resolution = 0):
        self.filled = False
        self.name = "CloudType"
        self.resolution = resolution
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
        self.pcs_def = ""
        self.xscale = 0
        self.yscale = 0
        self.ll_lon = 0.0
        self.ll_lat = 0.0
        self.ur_lon = 0.0
        self.ur_lat = 0.0
        self.region_name = ""
        self.cfac = 0
        self.lfac = 0
        self.coff = 0
        self.loff = 0
        self.nb_param = 0
        self.gp_sc_id = 0
        self.image_acquisition_time = 0
        self.spectral_channel_id = 0
        self.nominal_product_time = 0
        self.sgs_product_quality = 0
        self.sgs_product_completeness = 0
        self.product_algorithm_version = ""
        self.cloudtype = None
        self.processing_flags = None
        self.cloudphase = None

    def __repr__(self):
        return ("'%s: shape %s, resolution %sm'"%
                (self.name, 
                 self.cloudtype.data.shape, 
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled
        
# ------------------------------------------------------------------
    def read_msg_ctype(self, filename):
        """Reader for the NWCSAF/MSG cloudtype. Use *filename* to read data.
        """
        import _pyhl
        node_list = _pyhl.read_nodelist(filename)
        node_list.selectAll()
        node_list.fetch()
        
        self.cloudtype = MsgCloudTypeData()
        self.processing_flags = MsgCloudTypeData()
        self.cloudphase = MsgCloudTypeData()
        
        # The header
        node = node_list.getNode("/PACKAGE")
        self.package = node.data()
        node = node_list.getNode("/SAF")
        self.saf = node.data()
        node = node_list.getNode("/PRODUCT_NAME")
        self.product_name = node.data()
        node = node_list.getNode("/NC")
        self.num_of_columns = node.data()
        node = node_list.getNode("/NL")
        self.num_of_lines = node.data()
        node = node_list.getNode("/PROJECTION_NAME")
        self.projection_name = node.data()
        node = node_list.getNode("/REGION_NAME")
        self.region_name = node.data()
        node = node_list.getNode("/CFAC")
        self.cfac = node.data()
        node = node_list.getNode("/LFAC")
        self.lfac = node.data()
        node = node_list.getNode("/COFF")
        self.coff = node.data()
        node = node_list.getNode("/LOFF")
        self.loff = node.data()
        node = node_list.getNode("/NB_PARAMETERS")
        self.nb_param = node.data()
        node = node_list.getNode("/GP_SC_ID")
        self.gp_sc_id = node.data()
        node = node_list.getNode("/IMAGE_ACQUISITION_TIME")
        self.image_acquisition_time = node.data()
        node = node_list.getNode("/SPECTRAL_CHANNEL_ID")
        self.spectral_channel_id = node.data()
        node = node_list.getNode("/NOMINAL_PRODUCT_TIME")
        self.nominal_product_time = node.data()
        node = node_list.getNode("/SGS_PRODUCT_QUALITY")
        self.sgs_product_quality = node.data()
        node = node_list.getNode("/SGS_PRODUCT_COMPLETENESS")
        self.sgs_product_completeness = node.data()
        node = node_list.getNode("/PRODUCT_ALGORITHM_VERSION")
        self.product_algorithm_version = node.data()    
        # ------------------------
    
        # The cloudtype data
        node = node_list.getNode("/CT")
        self.cloudtype.data = node.data()
        node = node_list.getNode("/CT/SCALING_FACTOR")
        self.cloudtype.scaling_factor = node.data()
        node = node_list.getNode("/CT/OFFSET")
        self.cloudtype.offset = node.data()
        node = node_list.getNode("/CT/N_LINES")
        self.cloudtype.num_of_lines = node.data()
        node = node_list.getNode("/CT/N_COLS")
        self.cloudtype.num_of_columns = node.data()
        node = node_list.getNode("/CT/PRODUCT")
        self.cloudtype.product = node.data()
        node = node_list.getNode("/CT/ID")
        self.cloudtype.id = node.data()
        # ------------------------
    
        # The cloud phase data
        node = node_list.getNode("/CT_PHASE")
        self.cloudphase.data = node.data()
        node = node_list.getNode("/CT_PHASE/SCALING_FACTOR")
        self.cloudphase.scaling_factor = node.data()
        node = node_list.getNode("/CT_PHASE/OFFSET")
        self.cloudphase.offset = node.data()
        node = node_list.getNode("/CT_PHASE/N_LINES")
        self.cloudphase.num_of_lines = node.data()
        node = node_list.getNode("/CT_PHASE/N_COLS")
        self.cloudphase.num_of_columns = node.data()
        node = node_list.getNode("/CT_PHASE/PRODUCT")
        self.cloudphase.product = node.data()
        node = node_list.getNode("/CT_PHASE/ID")
        self.cloudphase.id = node.data()
        # ------------------------
    
        # The cloudtype processing/quality flags
        node = node_list.getNode("/CT_QUALITY")
        self.processing_flags.data = node.data()
        node = node_list.getNode("/CT_QUALITY/SCALING_FACTOR")
        self.processing_flags.scaling_factor = node.data()
        node = node_list.getNode("/CT_QUALITY/OFFSET")
        self.processing_flags.offset = node.data()
        node = node_list.getNode("/CT_QUALITY/N_LINES")
        self.processing_flags.num_of_lines = node.data()
        node = node_list.getNode("/CT_QUALITY/N_COLS")
        self.processing_flags.num_of_columns = node.data()
        node = node_list.getNode("/CT_QUALITY/PRODUCT")
        self.processing_flags.product = node.data()
        node = node_list.getNode("/CT_QUALITY/ID")
        self.processing_flags.id = node.data()
        # ------------------------

        self.filled = True

    def save(self, filename):
        """Save the current cloudtype object to hdf *filename*, in pps format.
        """
        ctype = msg_ctype2ppsformat(self)
        LOG.info("Saving CType hdf file...")
        epshdf.write_cloudtype(filename, ctype, 6)
        LOG.info("Saving CType hdf file done !")

    
    def project(self, coverage):
        """Remaps the NWCSAF/MSG Cloud Type to cartographic map-projection on
        area give by a pre-registered area-id. Faster version of msg_remap!
        """
        LOG.info("Projecting channel %s..."%(self.name))
        
        region = coverage.out_area
        dest_area = region.area_id

        retv = MsgCloudType()
        
        retv.package = self.package
        retv.saf = self.saf
        retv.product_name = self.product_name
        retv.region_name = dest_area
        retv.cfac = self.cfac
        retv.lfac = self.lfac
        retv.coff = self.coff
        retv.loff = self.loff
        retv.nb_param = self.nb_param
        retv.gp_sc_id = self.gp_sc_id
        retv.image_acquisition_time = self.image_acquisition_time
        retv.spectral_channel_id = self.spectral_channel_id
        retv.nominal_product_time = self.nominal_product_time
        retv.sgs_product_quality = self.sgs_product_quality
        retv.sgs_product_completeness = self.sgs_product_completeness
        retv.product_algorithm_version = self.product_algorithm_version
        
        retv.cloudtype = MsgCloudTypeData()
        retv.processing_flags = MsgCloudTypeData()
        retv.cloudphase = MsgCloudTypeData()

        retv.cloudtype.data = coverage.project_array(self.cloudtype.data)
        
        retv.cloudphase.data = coverage.project_array(self.cloudphase.data)
        retv.processing_flags.data = \
            coverage.project_array(self.processing_flags.data)
        
        retv.region_name = dest_area
        retv.projection_name = region.proj_id
        retv.pcs_def = region.proj4_string
        
        retv.num_of_columns = region.x_size
        retv.num_of_lines = region.y_size
        retv.xscale = region.pixel_size_x
        retv.yscale = region.pixel_size_y
        ll_lonlat = pps_gisdata.xy2lonlat(dest_area, 0, region.y_size)
        ur_lonlat = pps_gisdata.xy2lonlat(dest_area, region.x_size, 0)
        retv.ll_lon = ll_lonlat[0]
        retv.ll_lat = ll_lonlat[1]
        retv.ur_lon = ur_lonlat[0]
        retv.ur_lat = ur_lonlat[1]
        
        
        retv.cloudtype.offset = self.cloudtype.offset    
        retv.cloudtype.scaling_factor = self.cloudtype.scaling_factor    
        retv.cloudtype.id = self.cloudtype.id
        retv.cloudtype.product = self.cloudtype.product
        retv.cloudtype.num_of_columns = region.x_size
        retv.cloudtype.num_of_lines = region.y_size
            
        retv.cloudphase.offset = self.cloudphase.offset    
        retv.cloudphase.scaling_factor = self.cloudphase.scaling_factor    
        retv.cloudphase.id = self.cloudphase.id
        retv.cloudphase.product = self.cloudphase.product
        retv.cloudphase.num_of_columns = region.x_size
        retv.cloudphase.num_of_lines = region.y_size
        
        retv.cloudtype.offset = self.cloudtype.offset    
        retv.cloudtype.scaling_factor = self.cloudtype.scaling_factor    
        retv.cloudtype.id = self.cloudtype.id
        retv.cloudtype.product = self.cloudtype.product
        retv.processing_flags.num_of_columns = region.x_size
        retv.processing_flags.num_of_lines = region.y_size    
        
        retv.filled = True
        retv.resolution = self.resolution
        
        return retv

def get_bit_from_flags(arr, nbit):
    """I don't know what this function does.
    """
    res = numpy.bitwise_and(numpy.right_shift(arr, nbit), 1)
    return res.astype('b')

# ------------------------------------------------------------------
def msg_ctype2ppsformat(msgctype, satid="Meteosat 9"):
    """Converts the NWCSAF/MSG Cloud Type to the PPS format,
    in order to have consistency in output format between PPS and MSG.
    """
    retv = epshdf.CloudType()
    retv.region = epshdf.SafRegion()
    retv.region.xsize = msgctype.num_of_columns
    retv.region.ysize = msgctype.num_of_lines
    retv.region.id = msgctype.region_name
    retv.region.pcs_id = msgctype.projection_name
    pcsdef = ' '.join(area.area(msgctype.region_name).pcs.definition)
    retv.region.pcs_def = pcsdef
    retv.region.area_extent = area.area(msgctype.region_name).extent
    retv.satellite_id = satid

    luts = pps_luts()
    retv.cloudtype_lut = luts[0]
    retv.phaseflag_lut = []
    retv.qualityflag_lut = []
    retv.cloudtype_des = "MSG SEVIRI Cloud Type"
    retv.qualityflag_des = 'MSG SEVIRI bitwise quality/processing flags'
    retv.phaseflag_des = 'MSG SEVIRI Cloud phase flags'
    
    retv.cloudtype = msgctype.cloudtype.data
    retv.cloudphase = msgctype.cloudphase.data
    retv.qualityflag = convert_procflags2pps(msgctype.processing_flags.data)
    
    return retv

# ------------------------------------------------------------------
def convert_procflags2pps(data):
    """Converting cloud type processing flags to
    the PPS format, in order to have consistency between
    PPS and MSG cloud type contents.
    """
    
    ones = numpy.ones(data.shape,"h")

    # msg illumination bit 0,1,2 (undefined,night,twilight,day,sunglint) maps
    # to pps bits 2, 3 and 4:
    is_bit0_set = get_bit_from_flags(data, 0)    
    is_bit1_set = get_bit_from_flags(data, 1)    
    is_bit2_set = get_bit_from_flags(data, 2)
    illum = is_bit0_set * numpy.left_shift(ones, 0) + \
            is_bit1_set * numpy.left_shift(ones, 1) + \
            is_bit2_set * numpy.left_shift(ones, 2)
    del is_bit0_set
    del is_bit1_set
    del is_bit2_set
    # Night?
    # If night in msg then set pps night bit and nothing else.
    # If twilight in msg then set pps twilight bit and nothing else.
    # If day in msg then unset both the pps night and twilight bits.
    # If sunglint in msg unset both the pps night and twilight bits and set the
    # pps sunglint bit.
    arr = numpy.where(numpy.equal(illum, 1), numpy.left_shift(ones, 2), 0)
    arr = numpy.where(numpy.equal(illum, 2), numpy.left_shift(ones, 3), arr)
    arr = numpy.where(numpy.equal(illum, 3), 0, arr)
    arr = numpy.where(numpy.equal(illum, 4), numpy.left_shift(ones, 4), arr)
    retv = numpy.array(arr)
    del illum
    
    # msg nwp-input bit 3 (nwp present?) maps to pps bit 7:
    # msg nwp-input bit 4 (low level inversion?) maps to pps bit 6:
    is_bit3_set = get_bit_from_flags(data, 3)
    is_bit4_set = get_bit_from_flags(data, 4)
    nwp = (is_bit3_set * numpy.left_shift(ones, 0) + 
           is_bit4_set * numpy.left_shift(ones, 1))
    del is_bit3_set
    del is_bit4_set

    arr = numpy.where(numpy.equal(nwp, 1), numpy.left_shift(ones, 7), 0)
    arr = numpy.where(numpy.equal(nwp, 2), numpy.left_shift(ones, 7) +
                      numpy.left_shift(ones, 6), arr)
    arr = numpy.where(numpy.equal(nwp, 3), 0, arr)
    retv = numpy.add(arr, retv)
    del nwp
    
    # msg seviri-input bits 5&6 maps to pps bit 8:
    is_bit5_set = get_bit_from_flags(data, 5)
    is_bit6_set = get_bit_from_flags(data, 6)
    seviri = (is_bit5_set * numpy.left_shift(ones, 0) +
              is_bit6_set * numpy.left_shift(ones, 1))
    del is_bit5_set
    del is_bit6_set

    retv = numpy.add(retv,
                     numpy.where(numpy.logical_or(numpy.equal(seviri, 2),
                                                  numpy.equal(seviri, 3)),
                                 numpy.left_shift(ones, 8), 0))
    del seviri
    
    # msg quality bits 7&8 maps to pps bit 9&10:
    is_bit7_set = get_bit_from_flags(data, 7)
    is_bit8_set = get_bit_from_flags(data, 8)
    quality = (is_bit7_set * numpy.left_shift(ones, 0) +
               is_bit8_set * numpy.left_shift(ones,1))
    del is_bit7_set
    del is_bit8_set

    arr = numpy.where(numpy.equal(quality, 2), numpy.left_shift(ones, 9), 0)
    arr = numpy.where(numpy.equal(quality, 3), numpy.left_shift(ones, 10), arr)
    retv = numpy.add(arr, retv)
    del quality
    
    # msg bit 9 (stratiform-cumuliform distinction?) maps to pps bit 11:
    is_bit9_set = get_bit_from_flags(data, 9)
    retv = numpy.add(retv,
                     numpy.where(is_bit9_set,
                                 numpy.left_shift(ones, 11),
                                 0))
    del is_bit9_set
    
    return retv.astype('h')

# ------------------------------------------------------------------
def pps_luts():
    """Gets the LUTs for the PPS Cloud Type data fields.
    Returns a tuple with Cloud Type lut, Cloud Phase lut, Processing flags lut
    """
    ctype_lut = ['0: Not processed',
                 '1: Cloud free land',
                 '2: Cloud free sea',
                 '3: Snow/ice contaminated land',
                 '4: Snow/ice contaminated sea',
                 '5: Very low cumiliform cloud',
                 '6: Very low stratiform cloud',
                 '7: Low cumiliform cloud',
                 '8: Low stratiform cloud',
                 '9: Medium level cumiliform cloud',
                 '10: Medium level stratiform cloud',
                 '11: High and opaque cumiliform cloud',
                 '12: High and opaque stratiform cloud',
                 '13:Very high and opaque cumiliform cloud',
                 '14: Very high and opaque stratiform cloud',
                 '15: Very thin cirrus cloud',
                 '16: Thin cirrus cloud',
                 '17: Thick cirrus cloud',
                 '18: Cirrus above low or medium level cloud',
                 '19: Fractional or sub-pixel cloud',
                 '20: Undefined']
    phase_lut = ['1: Not processed or undefined',
                 '2: Water',
                 '4: Ice',
                 '8: Tb11 below 260K',
                 '16: value not defined',
                 '32: value not defined',
                 '64: value not defined',
                 '128: value not defined']
    quality_lut = ['1: Land',
                   '2: Coast',
                   '4: Night',
                   '8: Twilight',
                   '16: Sunglint',
                   '32: High terrain',
                   '64: Low level inversion',
                   '128: Nwp data present',
                   '256: Avhrr channel missing',
                   '512: Low quality',
                   '1024: Reclassified after spatial smoothing',
                   '2048: Stratiform-Cumuliform Distinction performed',
                   '4096: bit not defined',
                   '8192: bit not defined',
                   '16384: bit not defined',
                   '32768: bit not defined']

    return ctype_lut, phase_lut, quality_lut

def cloudtype_channel(time_slot, area_name):
    """Create and return a cloudtype channel.
    """
    time_string = time_slot.strftime("%Y%m%d%H%M")

    prefix = ("SAFNWC_MSG?_CT___%s_%s"%(time_string, area_name))

    msgctype_filename = None

    for ext in MSG_PGE_EXTENTIONS:
        # Look for a ctype file
        match_str = ("%s/%s.%s"%(CTYPE_DIR, prefix, ext))
        LOG.info("file-match: %s"%match_str)
        flist = glob.glob(match_str)

        if len(flist) > 1:
            LOG.error("More than one matching input file: N = %d"
                      %len(flist))
            raise RuntimeError("Cannot choose input CType file.")
        elif len(flist) == 0:
            LOG.warning("No matching .%s input cloud type file."
                        %ext)
        else:
            # File found:
            LOG.info("MSG CT file found: %s"%flist[0])
            msgctype_filename = flist[0]
            break

    if msgctype_filename is not None:
        LOG.info("Read MSG CT file: %s"%msgctype_filename)
        ctype = MsgCloudType(resolution = 3000)
        ctype.read_msg_ctype(msgctype_filename)
        LOG.debug("MSG CT area: %s"%ctype.region_name)
        return ctype
    else:
        LOG.error("No MSG CT input file found!")
        raise RuntimeError("No input CType file.")

COMPRESS_LVL = 6

class NordRadCType(object):
    """Wrapper aroud the msg_ctype channel.
    """

    def __init__(self, ctype_instance, datestr):
        self.ctype = ctype_instance
        self.datestr = datestr
    

# ------------------------------------------------------------------
    def save(self, filename):
        """Save the current instance to nordrad hdf format.
        """
        import _pyhl
        status = 1

        msgctype = self.ctype

        node_list = _pyhl.nodelist()

        # What
        node = _pyhl.node(_pyhl.GROUP_ID, "/what")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/object")
        node.setScalarValue(-1, "IMAGE", "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/sets")
        node.setScalarValue(-1, 1, "int", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/version")
        node.setScalarValue(-1, "H5rad 1.2", "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/date")
        yyyymmdd = self.datestr[0:8]
        hourminsec = self.datestr[8:12]+'00'
        node.setScalarValue(-1, yyyymmdd, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/time")
        node.setScalarValue(-1, hourminsec, "string", -1)
        node_list.addNode(node)    

        # Where
        node = _pyhl.node(_pyhl.GROUP_ID, "/where")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/projdef")
        node.setScalarValue(-1, msgctype.pcs_def, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/xsize")
        node.setScalarValue(-1, msgctype.num_of_columns, "int", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/ysize")
        node.setScalarValue(-1, msgctype.num_of_lines, "int", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/xscale")
        node.setScalarValue(-1, msgctype.xscale, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/yscale")
        node.setScalarValue(-1, msgctype.yscale, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/LL_lon")
        node.setScalarValue(-1, msgctype.ll_lon, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/LL_lat")
        node.setScalarValue(-1, msgctype.ll_lat, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/UR_lon")
        node.setScalarValue(-1, msgctype.ur_lon, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/UR_lat")
        node.setScalarValue(-1, msgctype.ur_lat, "float", -1)
        node_list.addNode(node)

        # How
        node = _pyhl.node(_pyhl.GROUP_ID, "/how")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/how/area")
        node.setScalarValue(-1, msgctype.region_name, "string", -1)
        node_list.addNode(node)

        # image1
        node = _pyhl.node(_pyhl.GROUP_ID, "/image1")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.DATASET_ID, "/image1/data")
        node.setArrayValue(1, [msgctype.num_of_columns, msgctype.num_of_lines],
                           msgctype.cloudtype.data, "uchar", -1)
        node_list.addNode(node)

        node = _pyhl.node(_pyhl.GROUP_ID, "/image1/what")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/product")
        #We should eventually try to use the msg-parameters "package",
        #"product_algorithm_version", and "product_name":
        node.setScalarValue(1, 'MSGCT', "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/prodpar")
        node.setScalarValue(1, 0.0, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/quantity")
        node.setScalarValue(1, "ct", "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/startdate")
        node.setScalarValue(-1, yyyymmdd, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/starttime")
        node.setScalarValue(-1, hourminsec, "string", -1)
        node_list.addNode(node)    
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/enddate")
        node.setScalarValue(-1, yyyymmdd, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/endtime")
        node.setScalarValue(-1, hourminsec, "string", -1)
        node_list.addNode(node)    
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/gain")
        node.setScalarValue(-1, 1.0, "float", -1)
        node_list.addNode(node)    
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/offset")
        node.setScalarValue(-1, 0.0, "float", -1)
        node_list.addNode(node)    
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/nodata")
        node.setScalarValue(-1, 0.0, "float", -1)
        node_list.addNode(node)
        # What we call missingdata in PPS:
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/undetect")
        node.setScalarValue(-1, 20.0, "float", -1)
        node_list.addNode(node)    

        node_list.write(filename, COMPRESS_LVL)

#         if status:
#             for tup in N2SERVERS_AND_PORTS:
#                 cmdstr = "%s %s:%d %s"%(N2INJECT,tup[0],tup[1],filename)
#                 LOG.info("Command: %s"%(cmdstr))
#                 os.system(cmdstr)
#             else:
#                 LOG.error("Failed writing cloudtype product for Nordrad!")
#                 LOG.info("Filename = %s"%(filename))
                
        
        return status

