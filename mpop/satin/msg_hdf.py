#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2012.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Plugin for reading PPS's cloud products hdf files.
"""

import ConfigParser
import os.path
from mpop import CONFIG_PATH
import mpop.channel
import numpy as np

import glob
from mpop.utils import get_logger
from mpop.projector import get_area_def

LOG = get_logger('satin/msg_hdf')
COMPRESS_LVL = 6

def pcs_def_from_region(region):
    items = region.proj_dict.items()
    return ' '.join([ t[0] + '=' + t[1] for t in items])   

def _get_area_extent(cfac, lfac, coff, loff, numcols, numlines):
    """Get the area extent from msg parameters.
    """

    xur = (numcols - coff) * 2**16 / (cfac * 1.0)
    xur = np.deg2rad(xur) * 35785831.0
    xll = (-1 - coff) * 2**16 / (cfac * 1.0)
    xll = np.deg2rad(xll) * 35785831.0
    xres = (xur - xll) / numcols
    xur, xll = xur - xres/2, xll + xres/2
    yll = (numlines - loff) * 2**16 / (-lfac * 1.0)
    yll = np.deg2rad(yll) * 35785831.0
    yur = (-1 - loff) * 2**16 / (-lfac * 1.0)
    yur = np.deg2rad(yur) * 35785831.0
    yres = (yur - yll) / numlines
    yll, yur = yll + yres/2, yur - yres/2
    return xll, yll, xur, yur

def get_area_extent(filename):
    """Get the area extent of the data in *filename*.
    """
    import tables
    h5f = tables.openFile(filename)
    aex = _get_area_extent(h5f.root._v_attrs["CFAC"],
                           h5f.root._v_attrs["LFAC"],
                           h5f.root._v_attrs["COFF"],
                           h5f.root._v_attrs["LOFF"],
                           h5f.root._v_attrs["NC"],
                           h5f.root._v_attrs["NL"])
    h5f.close()
    return aex


class PpsCloudType(mpop.channel.GenericChannel):
    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "CloudType")
        self.region = None
        self.des = ""
        self.cloudtype_des = ""
        self.qualityflag_des = ""
        self.phaseflag_des = ""
        self.sec_1970 = 0
        self.satellite_id = ""
        self.cloudtype_lut = []
        self.qualityflag_lut = []
        self.phaseflag_lut = []
        self.cloudtype = None
        self.qualityflag = None
        self.phaseflag = None

    def save(self, filename):
        """Save to *filename*.
        """
        import epshdf
        epshdf.write_cloudtype(filename, self, COMPRESS_LVL)

class PpsCTTH(mpop.channel.GenericChannel):
    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "CTTH")
        self.region = None
        self.des = ""
        self.ctt_des = ""
        self.cth_des = ""
        self.ctp_des = ""
        self.cloudiness_des = ""
        self.processingflag_des = ""
        self.sec_1970 = 0
        self.satellite_id = ""
        self.processingflag_lut = []

        self.temperature = None
        self.t_gain = 1.0
        self.t_intercept = 0.0
        self.t_nodata = 255

        self.pressure = None
        self.p_gain = 1.0
        self.p_intercept = 0.0
        self.p_nodata = 255

        self.height = None
        self.h_gain = 1.0
        self.h_intercept = 0.0
        self.h_nodata = 255

        self.cloudiness = None
        self.c_nodata = 255
        self.processingflag = None

    def save(self, filename):
        """Save to *filename*.
        """
        import epshdf
        epshdf.write_cloudtop(filename, self, COMPRESS_LVL)

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
        
class MsgCloudType(mpop.channel.GenericChannel):
    """NWCSAF/MSG Cloud Type data structure as retrieved from HDF5
    file. Resolution sets the nominal resolution of the data.
    """
    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "CloudType")
        self.filled = False
        self.name = "CloudType"
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
        self.shape = None
        self.satid = ""
        self.qc_straylight = -1
        
    def __str__(self):
        return ("'%s: shape %s, resolution %sm'"%
                (self.name, 
                 self.cloudtype.shape, 
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

# ------------------------------------------------------------------
    def read(self, filename):
        """Reader for the NWCSAF/MSG cloudtype. Use *filename* to read data.
        """
        import tables
        
        self.cloudtype = MsgCloudTypeData()
        self.processing_flags = MsgCloudTypeData()
        self.cloudphase = MsgCloudTypeData()


        h5f = tables.openFile(filename)
        # pylint: disable-msg=W0212
        self.package = h5f.root._v_attrs["PACKAGE"]
        self.saf = h5f.root._v_attrs["SAF"]
        self.product_name = h5f.root._v_attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.root._v_attrs["NC"]
        self.num_of_lines = h5f.root._v_attrs["NL"]
        self.projection_name = h5f.root._v_attrs["PROJECTION_NAME"]
        self.region_name = h5f.root._v_attrs["REGION_NAME"]
        self.cfac = h5f.root._v_attrs["CFAC"]
        self.lfac = h5f.root._v_attrs["LFAC"]
        self.coff = h5f.root._v_attrs["COFF"]
        self.loff = h5f.root._v_attrs["LOFF"]
        self.nb_param = h5f.root._v_attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.root._v_attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.root._v_attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.root._v_attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.root._v_attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.root._v_attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.root._v_attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.root._v_attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------
    
        # The cloudtype data
        self.cloudtype.data = h5f.root.CT[:, :]
        self.cloudtype.scaling_factor = h5f.root.CT.attrs["SCALING_FACTOR"]
        self.cloudtype.offset = h5f.root.CT.attrs["OFFSET"]
        self.cloudtype.num_of_lines = h5f.root.CT.attrs["N_LINES"]
        self.cloudtype.num_of_columns = h5f.root.CT.attrs["N_COLS"]
        self.shape = (self.cloudtype.num_of_lines,
                      self.cloudtype.num_of_columns)
        self.cloudtype.product = h5f.root.CT.attrs["PRODUCT"]
        self.cloudtype.id = h5f.root.CT.attrs["ID"]
        # ------------------------
    
        # The cloud phase data
        self.cloudphase.data = h5f.root.CT_PHASE[:, :]
        self.cloudphase.scaling_factor = h5f.root.CT_PHASE.attrs["SCALING_FACTOR"]
        self.cloudphase.offset = h5f.root.CT_PHASE.attrs["OFFSET"]
        self.cloudphase.num_of_lines = h5f.root.CT_PHASE.attrs["N_LINES"]
        self.cloudphase.num_of_columns = h5f.root.CT_PHASE.attrs["N_COLS"]
        self.cloudphase.product = h5f.root.CT_PHASE.attrs["PRODUCT"]
        self.cloudphase.id = h5f.root.CT_PHASE.attrs["ID"]
        # ------------------------
    
        # The cloudtype processing/quality flags
        self.processing_flags.data = h5f.root.CT_QUALITY[:, :]
        self.processing_flags.scaling_factor = \
                          h5f.root.CT_QUALITY.attrs["SCALING_FACTOR"]
        self.processing_flags.offset = h5f.root.CT_QUALITY.attrs["OFFSET"]
        self.processing_flags.num_of_lines = h5f.root.CT_QUALITY.attrs["N_LINES"]
        self.processing_flags.num_of_columns = h5f.root.CT_QUALITY.attrs["N_COLS"]
        self.processing_flags.product = h5f.root.CT_QUALITY.attrs["PRODUCT"]
        self.processing_flags.id = h5f.root.CT_QUALITY.attrs["ID"]
        # ------------------------
        h5f.close()
        
        self.cloudtype = (self.cloudtype.data * self.cloudtype.scaling_factor
                          + self.cloudtype.offset)
        self.cloudphase = (self.cloudphase.data * self.cloudphase.scaling_factor
                          + self.cloudphase.offset)
        self.processing_flags = self.processing_flags.data

        self.area = get_area_def(self.region_name)
        
        self.filled = True
        

    def save(self, filename):
        """Save the current cloudtype object to hdf *filename*, in pps format.
        """
        import tables
        ctype = self.convert2pps()
        LOG.info("Saving CType hdf file...")
        ctype.save(filename)
        h5f = tables.openFile(filename, mode="a")
        h5f.root._v_attrs["straylight_contaminated"] = self.qc_straylight
        h5f.close()
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
        

        retv.cloudtype = coverage.project_array(self.cloudtype)
        
        retv.cloudphase = coverage.project_array(self.cloudphase)
        retv.processing_flags = \
            coverage.project_array(self.processing_flags)
        
        retv.qc_straylight = self.qc_straylight
        retv.region_name = dest_area
        retv.area = region
        retv.projection_name = region.proj_id
        
        retv.pcs_def = pcs_def_from_region(region)
        
        retv.num_of_columns = region.x_size
        retv.num_of_lines = region.y_size
        retv.xscale = region.pixel_size_x
        retv.yscale = region.pixel_size_y

        import pyproj
        prj = pyproj.Proj(region.proj4_string)
        aex = region.area_extent
        lonur, latur = prj(aex[2], aex[3], inverse=True)
        lonll, latll = prj(aex[0], aex[1], inverse=True)
        retv.ll_lon = lonll
        retv.ll_lat = latll
        retv.ur_lon = lonur
        retv.ur_lat = latur
        
        self.shape = region.shape

        retv.filled = True
        retv.resolution = self.resolution
        
        return retv

    def convert2pps(self):
        """Converts the NWCSAF/MSG Cloud Type to the PPS format,
        in order to have consistency in output format between PPS and MSG.
        """
        import epshdf
        retv = PpsCloudType()
        retv.region = epshdf.SafRegion()
        retv.region.xsize = self.num_of_columns
        retv.region.ysize = self.num_of_lines
        retv.region.id = self.region_name
        retv.region.pcs_id = self.projection_name
        
        retv.region.pcs_def = pcs_def_from_region(self.area)
        retv.region.area_extent = self.area.area_extent
        retv.satellite_id = self.satid

        luts = pps_luts()
        retv.cloudtype_lut = luts[0]
        retv.phaseflag_lut = []
        retv.qualityflag_lut = []
        retv.cloudtype_des = "MSG SEVIRI Cloud Type"
        retv.qualityflag_des = 'MSG SEVIRI bitwise quality/processing flags'
        retv.phaseflag_des = 'MSG SEVIRI Cloud phase flags'

        retv.cloudtype = self.cloudtype.astype('B')
        retv.phaseflag = self.cloudphase.astype('B')
        retv.qualityflag = ctype_procflags2pps(self.processing_flags)

        return retv

    def convert2nordrad(self):
        return NordRadCType(self)

class MsgCTTHData(object):
    """CTTH data object.
    """
    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""
        
class MsgCTTH(mpop.channel.GenericChannel):
    """CTTH channel.
    """
    def __init__(self, resolution = None):
        mpop.channel.GenericChannel.__init__(self, "CTTH")
        self.filled = False
        self.name = "CTTH"
        self.resolution = resolution
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
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
        self.cloudiness = None # Effective cloudiness
        self.processing_flags = None
        self.height = None
        self.temperature = None
        self.pressure = None
        self.satid = ""
        
    def __str__(self):
        return ("'%s: shape %s, resolution %sm'"%
                (self.name, 
                 self.shape, 
                 self.resolution))   

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

    def read(self, filename):
        import tables
        
        self.cloudiness = MsgCTTHData() # Effective cloudiness
        self.temperature = MsgCTTHData()
        self.height = MsgCTTHData()
        self.pressure = MsgCTTHData()
        self.processing_flags = MsgCTTHData()

        h5f = tables.openFile(filename)
        
        # The header
        # pylint: disable-msg=W0212
        self.package = h5f.root._v_attrs["PACKAGE"]
        self.saf = h5f.root._v_attrs["SAF"]
        self.product_name = h5f.root._v_attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.root._v_attrs["NC"]
        self.num_of_lines = h5f.root._v_attrs["NL"]
        self.projection_name = h5f.root._v_attrs["PROJECTION_NAME"]
        self.region_name = h5f.root._v_attrs["REGION_NAME"]
        self.cfac = h5f.root._v_attrs["CFAC"]
        self.lfac = h5f.root._v_attrs["LFAC"]
        self.coff = h5f.root._v_attrs["COFF"]
        self.loff = h5f.root._v_attrs["LOFF"]
        self.nb_param = h5f.root._v_attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.root._v_attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.root._v_attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.root._v_attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.root._v_attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.root._v_attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.root._v_attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.root._v_attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------
    
        # The CTTH cloudiness data
        self.cloudiness.data = h5f.root.CTTH_EFFECT[:, :]
        self.cloudiness.scaling_factor = \
                             h5f.root.CTTH_EFFECT.attrs["SCALING_FACTOR"]
        self.cloudiness.offset = h5f.root.CTTH_EFFECT.attrs["OFFSET"]
        self.cloudiness.num_of_lines = h5f.root.CTTH_EFFECT.attrs["N_LINES"]
        self.cloudiness.num_of_columns = h5f.root.CTTH_EFFECT.attrs["N_COLS"]
        self.cloudiness.product = h5f.root.CTTH_EFFECT.attrs["PRODUCT"]
        self.cloudiness.id = h5f.root.CTTH_EFFECT.attrs["ID"]

        self.cloudiness.data = np.ma.masked_equal(self.cloudiness.data, 255)
        self.cloudiness = np.ma.masked_equal(self.cloudiness.data, 0)
        
        # ------------------------
    
        # The CTTH temperature data
        self.temperature.data = h5f.root.CTTH_TEMPER[:, :]
        self.temperature.scaling_factor = \
                               h5f.root.CTTH_TEMPER.attrs["SCALING_FACTOR"]
        self.temperature.offset = h5f.root.CTTH_TEMPER.attrs["OFFSET"]
        self.temperature.num_of_lines = h5f.root.CTTH_TEMPER.attrs["N_LINES"]
        self.shape = (self.temperature.num_of_lines,
                      self.temperature.num_of_columns)
        self.temperature.num_of_columns = h5f.root.CTTH_TEMPER.attrs["N_COLS"]
        self.temperature.product = h5f.root.CTTH_TEMPER.attrs["PRODUCT"]
        self.temperature.id = h5f.root.CTTH_TEMPER.attrs["ID"]
        
        self.temperature = (np.ma.masked_equal(self.temperature.data, 0) *
                            self.temperature.scaling_factor +
                            self.temperature.offset)

        # ------------------------
    
        # The CTTH pressure data
        self.pressure.data = h5f.root.CTTH_PRESS[:, :]
        self.pressure.scaling_factor = \
                                     h5f.root.CTTH_PRESS.attrs["SCALING_FACTOR"]
        self.pressure.offset = h5f.root.CTTH_PRESS.attrs["OFFSET"]
        self.pressure.num_of_lines = h5f.root.CTTH_PRESS.attrs["N_LINES"]
        self.pressure.num_of_columns = h5f.root.CTTH_PRESS.attrs["N_COLS"]
        self.pressure.product = h5f.root.CTTH_PRESS.attrs["PRODUCT"]
        self.pressure.id = h5f.root.CTTH_PRESS.attrs["ID"]
        
        self.pressure.data = np.ma.masked_equal(self.pressure.data, 255)
        self.pressure = (np.ma.masked_equal(self.pressure.data, 0) *
                         self.pressure.scaling_factor +
                         self.pressure.offset)

        # ------------------------
    
        # The CTTH height data
        self.height.data = h5f.root.CTTH_HEIGHT[:, :]
        self.height.scaling_factor = \
                                   h5f.root.CTTH_HEIGHT.attrs["SCALING_FACTOR"]
        self.height.offset = h5f.root.CTTH_HEIGHT.attrs["OFFSET"]
        self.height.num_of_lines = h5f.root.CTTH_HEIGHT.attrs["N_LINES"]
        self.height.num_of_columns = h5f.root.CTTH_HEIGHT.attrs["N_COLS"]
        self.height.product = h5f.root.CTTH_HEIGHT.attrs["PRODUCT"]
        self.height.id = h5f.root.CTTH_HEIGHT.attrs["ID"]
        
        self.height.data = np.ma.masked_equal(self.height.data, 255)
        self.height = (np.ma.masked_equal(self.height.data, 0) *
                       self.height.scaling_factor +
                       self.height.offset)

        
        # ------------------------
    
        # The CTTH processing/quality flags
        self.processing_flags.data = h5f.root.CTTH_QUALITY[:, :]
        self.processing_flags.scaling_factor = \
                                h5f.root.CTTH_QUALITY.attrs["SCALING_FACTOR"]
        self.processing_flags.offset = h5f.root.CTTH_QUALITY.attrs["OFFSET"]
        self.processing_flags.num_of_lines = \
                                h5f.root.CTTH_QUALITY.attrs["N_LINES"]
        self.processing_flags.num_of_columns = \
                                h5f.root.CTTH_QUALITY.attrs["N_COLS"]
        self.processing_flags.product = h5f.root.CTTH_QUALITY.attrs["PRODUCT"]
        self.processing_flags.id = h5f.root.CTTH_QUALITY.attrs["ID"]

        self.processing_flags = \
             np.ma.masked_equal(self.processing_flags.data, 0)

        h5f.close()
        
        self.shape = self.height.shape

        self.area = get_area_def(self.region_name)

        self.filled = True


    def save(self, filename):
        """Save the current CTTH channel to HDF5 format.
        """
        ctth = self.convert2pps()
        LOG.info("Saving CTTH hdf file...")
        ctth.save(filename)
        LOG.info("Saving CTTH hdf file done !")

    def project(self, coverage):
        """Project the current CTTH channel along the *coverage*
        """
        dest_area = coverage.out_area
        dest_area_id = dest_area.area_id
        

        retv = MsgCTTH()

        retv.temperature = coverage.project_array(self.temperature)
        retv.height = coverage.project_array(self.height)
        retv.pressure = coverage.project_array(self.pressure)
        retv.cloudiness = coverage.project_array(self.cloudiness)
        retv.processing_flags = \
            coverage.project_array(self.processing_flags)

        retv.area = dest_area
        retv.region_name = dest_area_id
        retv.projection_name = dest_area.proj_id
        retv.num_of_columns = dest_area.x_size
        retv.num_of_lines = dest_area.y_size
        
        retv.shape = dest_area.shape

        retv.name = self.name
        retv.resolution = self.resolution
        retv.filled = True

        return retv

# ------------------------------------------------------------------
    def convert2pps(self):
        """Convert the current CTTH channel to pps format.
        """
        import epshdf
        retv = PpsCTTH()
        retv.region = epshdf.SafRegion()
        retv.region.xsize = self.num_of_columns
        retv.region.ysize = self.num_of_lines
        retv.region.id = self.region_name
        retv.region.pcs_id = self.projection_name
        retv.region.pcs_def = pcs_def_from_region(self.area)
        retv.region.area_extent = self.area.area_extent
        retv.satellite_id = self.satid

        retv.processingflag_lut = []
        retv.des = "MSG SEVIRI Cloud Top Temperature & Height"
        retv.ctt_des = "MSG SEVIRI cloud top temperature (K)"
        retv.ctp_des = "MSG SEVIRI cloud top pressure (hPa)"
        retv.ctp_des = "MSG SEVIRI cloud top height (m)"
        retv.cloudiness_des = "MSG SEVIRI effective cloudiness (%)"
        retv.processingflag_des = 'MSG SEVIRI bitwise quality/processing flags'

        retv.t_gain = 1.0
        retv.t_intercept = 100.0
        retv.t_nodata = 255

        retv.temperature = ((self.temperature - retv.t_intercept) /
                            retv.t_gain).filled(retv.t_nodata).astype('B')

        retv.h_gain = 200.0
        retv.h_intercept = 0.0
        retv.h_nodata = 255

        retv.height = ((self.height - retv.h_intercept) /
                       retv.h_gain).filled(retv.h_nodata).astype('B')
        
        retv.p_gain = 25.0
        retv.p_intercept = 0.0
        retv.p_nodata = 255

        retv.pressure = ((self.pressure - retv.p_intercept) /
                         retv.p_gain).filled(retv.p_nodata).astype('B')

        retv.cloudiness = self.cloudiness.astype('B')
        retv.c_nodata = 255 # Is this correct? FIXME

        retv.processingflag = ctth_procflags2pps(self.processing_flags)

        return retv

# ------------------------------------------------------------------ 


def get_bit_from_flags(arr, nbit):
    """I don't know what this function does.
    """
    res = np.bitwise_and(np.right_shift(arr, nbit), 1)
    return res.astype('b')


def ctth_procflags2pps(data):
    """Convert ctth processing flags from MSG to PPS format.
    """

    ones = np.ones(data.shape,"h")

    # 2 bits to define processing status
    # (maps to pps bits 0 and 1:)
    is_bit0_set = get_bit_from_flags(data, 0)    
    is_bit1_set = get_bit_from_flags(data, 1)
    proc = (is_bit0_set * np.left_shift(ones, 0) +
            is_bit1_set * np.left_shift(ones, 1))
    del is_bit0_set
    del is_bit1_set

    # Non-processed?
    # If non-processed in msg (0) then set pps bit 0 and nothing else.
    # If non-processed in msg due to FOV is cloud free (1) then do not set any
    # pps bits.
    # If processed (because cloudy) with/without result in msg (2&3) then set
    # pps bit 1.

    arr = np.where(np.equal(proc, 0), np.left_shift(ones, 0), 0)
    arr = np.where(np.equal(proc, 2), np.left_shift(ones, 1), 0)
    arr = np.where(np.equal(proc, 3), np.left_shift(ones, 1), 0)
    retv = np.array(arr)
    del proc


    # 1 bit to define if RTTOV-simulations are available?
    # (maps to pps bit 3:)
    is_bit2_set = get_bit_from_flags(data, 2)    
    proc = is_bit2_set

    # RTTOV-simulations available?

    arr = np.where(np.equal(proc, 1), np.left_shift(ones, 3), 0)
    retv = np.add(retv, arr)
    del is_bit2_set
    
    # 3 bits to describe NWP input data
    # (maps to pps bits 4&5:)
    is_bit3_set = get_bit_from_flags(data, 3)
    is_bit4_set = get_bit_from_flags(data, 4)
    is_bit5_set = get_bit_from_flags(data, 5)    
    # Put together the three bits into a nwp-flag:
    nwp_bits = (is_bit3_set * np.left_shift(ones, 0) +
                is_bit4_set * np.left_shift(ones, 1) +
                is_bit5_set * np.left_shift(ones, 2))
    arr = np.where(np.logical_and(np.greater_equal(nwp_bits, 3),
                                        np.less_equal(nwp_bits, 5)), 
                      np.left_shift(ones, 4),
                      0)
    arr = np.add(arr, np.where(np.logical_or(np.equal(nwp_bits, 2),
                                                      np.equal(nwp_bits, 4)),
                                     np.left_shift(ones, 5),
                                     0))

    retv = np.add(retv, arr)
    del is_bit3_set
    del is_bit4_set
    del is_bit5_set

    # 2 bits to describe SEVIRI input data
    # (maps to pps bits 6:)
    is_bit6_set = get_bit_from_flags(data, 6)
    is_bit7_set = get_bit_from_flags(data, 7)
    # Put together the two bits into a seviri-flag:
    seviri_bits = (is_bit6_set * np.left_shift(ones, 0) +
                   is_bit7_set * np.left_shift(ones, 1))
    arr = np.where(np.greater_equal(seviri_bits, 2),
                      np.left_shift(ones, 6), 0)

    retv = np.add(retv, arr)
    del is_bit6_set
    del is_bit7_set
    
    # 4 bits to describe which method has been used
    # (maps to pps bits 7&8 and bit 2:)
    is_bit8_set = get_bit_from_flags(data, 8)
    is_bit9_set = get_bit_from_flags(data, 9)
    is_bit10_set = get_bit_from_flags(data, 10)
    is_bit11_set = get_bit_from_flags(data, 11)
    # Put together the four bits into a method-flag:
    method_bits = (is_bit8_set * np.left_shift(ones, 0) +
                   is_bit9_set * np.left_shift(ones, 1) +
                   is_bit10_set * np.left_shift(ones, 2) +
                   is_bit11_set * np.left_shift(ones, 3))
    arr = np.where(np.logical_or(
        np.logical_and(np.greater_equal(method_bits, 1),
                          np.less_equal(method_bits, 2)), 
        np.equal(method_bits, 13)), 
                      np.left_shift(ones, 2),
                      0)
    arr = np.add(arr, 
                    np.where(np.equal(method_bits, 1),
                                np.left_shift(ones, 7),
                                0))
    arr = np.add(arr, 
                    np.where(np.logical_and(
                        np.greater_equal(method_bits, 3), 
                        np.less_equal(method_bits, 12)), 
                                np.left_shift(ones, 8),
                                0))

    # (Maps directly - as well - to the spare bits 9-12) 
    arr = np.add(arr, np.where(is_bit8_set, np.left_shift(ones, 9), 0))
    arr = np.add(arr, np.where(is_bit9_set,
                               np.left_shift(ones, 10),
                               0))
    arr = np.add(arr, np.where(is_bit10_set,
                               np.left_shift(ones, 11),
                               0))
    arr = np.add(arr, np.where(is_bit11_set,
                               np.left_shift(ones, 12),
                               0))   
    retv = np.add(retv, arr)
    del is_bit8_set
    del is_bit9_set
    del is_bit10_set
    del is_bit11_set

    # 2 bits to describe the quality of the processing itself
    # (maps to pps bits 14&15:)
    is_bit12_set = get_bit_from_flags(data, 12)
    is_bit13_set = get_bit_from_flags(data, 13)
    # Put together the two bits into a quality-flag:
    qual_bits = (is_bit12_set * np.left_shift(ones, 0) +
                 is_bit13_set * np.left_shift(ones, 1))
    arr = np.where(np.logical_and(np.greater_equal(qual_bits, 1), 
                                  np.less_equal(qual_bits, 2)), 
                   np.left_shift(ones, 14), 0)
    arr = np.add(arr, 
                 np.where(np.equal(qual_bits, 2),
                          np.left_shift(ones, 15),
                          0))

    retv = np.add(retv, arr)
    del is_bit12_set
    del is_bit13_set    
    
    return retv.astype('h')


def ctype_procflags2pps(data):
    """Converting cloud type processing flags to
    the PPS format, in order to have consistency between
    PPS and MSG cloud type contents.
    """
    
    ones = np.ones(data.shape,"h")

    # msg illumination bit 0,1,2 (undefined,night,twilight,day,sunglint) maps
    # to pps bits 2, 3 and 4:
    is_bit0_set = get_bit_from_flags(data, 0)    
    is_bit1_set = get_bit_from_flags(data, 1)    
    is_bit2_set = get_bit_from_flags(data, 2)
    illum = is_bit0_set * np.left_shift(ones, 0) + \
            is_bit1_set * np.left_shift(ones, 1) + \
            is_bit2_set * np.left_shift(ones, 2)
    del is_bit0_set
    del is_bit1_set
    del is_bit2_set
    # Night?
    # If night in msg then set pps night bit and nothing else.
    # If twilight in msg then set pps twilight bit and nothing else.
    # If day in msg then unset both the pps night and twilight bits.
    # If sunglint in msg unset both the pps night and twilight bits and set the
    # pps sunglint bit.
    arr = np.where(np.equal(illum, 1), np.left_shift(ones, 2), 0)
    arr = np.where(np.equal(illum, 2), np.left_shift(ones, 3), arr)
    arr = np.where(np.equal(illum, 3), 0, arr)
    arr = np.where(np.equal(illum, 4), np.left_shift(ones, 4), arr)
    retv = np.array(arr)
    del illum
    
    # msg nwp-input bit 3 (nwp present?) maps to pps bit 7:
    # msg nwp-input bit 4 (low level inversion?) maps to pps bit 6:
    is_bit3_set = get_bit_from_flags(data, 3)
    is_bit4_set = get_bit_from_flags(data, 4)
    nwp = (is_bit3_set * np.left_shift(ones, 0) + 
           is_bit4_set * np.left_shift(ones, 1))
    del is_bit3_set
    del is_bit4_set

    arr = np.where(np.equal(nwp, 1), np.left_shift(ones, 7), 0)
    arr = np.where(np.equal(nwp, 2), np.left_shift(ones, 7) +
                      np.left_shift(ones, 6), arr)
    arr = np.where(np.equal(nwp, 3), 0, arr)
    retv = np.add(arr, retv)
    del nwp
    
    # msg seviri-input bits 5&6 maps to pps bit 8:
    is_bit5_set = get_bit_from_flags(data, 5)
    is_bit6_set = get_bit_from_flags(data, 6)
    seviri = (is_bit5_set * np.left_shift(ones, 0) +
              is_bit6_set * np.left_shift(ones, 1))
    del is_bit5_set
    del is_bit6_set

    retv = np.add(retv,
                     np.where(np.logical_or(np.equal(seviri, 2),
                                                  np.equal(seviri, 3)),
                                 np.left_shift(ones, 8), 0))
    del seviri
    
    # msg quality bits 7&8 maps to pps bit 9&10:
    is_bit7_set = get_bit_from_flags(data, 7)
    is_bit8_set = get_bit_from_flags(data, 8)
    quality = (is_bit7_set * np.left_shift(ones, 0) +
               is_bit8_set * np.left_shift(ones,1))
    del is_bit7_set
    del is_bit8_set

    arr = np.where(np.equal(quality, 2), np.left_shift(ones, 9), 0)
    arr = np.where(np.equal(quality, 3), np.left_shift(ones, 10), arr)
    retv = np.add(arr, retv)
    del quality
    
    # msg bit 9 (stratiform-cumuliform distinction?) maps to pps bit 11:
    is_bit9_set = get_bit_from_flags(data, 9)
    retv = np.add(retv,
                     np.where(is_bit9_set,
                                 np.left_shift(ones, 11),
                                 0))
    del is_bit9_set
    
    return retv.astype('h')


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

class NordRadCType(object):
    """Wrapper aroud the msg_ctype channel.
    """

    def __init__(self, ctype_instance):
        self.ctype = ctype_instance
        self.datestr = ctype_instance.image_acquisition_time
    

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
        node.setScalarValue(-1, msgctype.area.proj4_string, "string", -1)
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
                           msgctype.cloudtype.astype('B'), "uchar", -1)
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

        return status

MSG_PGE_EXTENTIONS = ["PLAX.CTTH.0.h5", "PLAX.CLIM.0.h5", "h5"]

def get_best_product(filename, area_extent):
    """Get the best of the available products for the *filename* template.
    """

    for ext in MSG_PGE_EXTENTIONS:
        match_str = filename + "." + ext
        flist = glob.glob(match_str)
        if len(flist) == 0:
            LOG.warning("No matching .%s input MSG file."
                        %ext)
        else:
            # File found:
            if area_extent is None:
                LOG.warning("Didn't specify an area, taking " + flist[0])
                return flist[0]
            for fname in flist:
                aex = get_area_extent(fname)
                if np.all(np.max(np.abs(np.array(aex) -
                                        np.array(area_extent))) < 1000):
                    LOG.info("MSG CT file found: %s"%fname)
                    return fname


    
def load(scene, **kwargs):
    """Load data into the *channels*. *Channels* is a list or a tuple
    containing channels we will load data into. If None, all channels are
    loaded.
    """

    area_extent = kwargs.get("area_extent")

    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, scene.fullname+".cfg"))
    directory = conf.get(scene.instrument_name+"-level3", "dir",
                         raw=True)
    filename = conf.get(scene.instrument_name+"-level3", "filename",
                        raw=True)
    pathname = os.path.join(directory, filename)
    
    if "CTTH" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    %{"number": "03",
                      "product": "CTTH_"})
        ct_chan = MsgCTTH()
        ct_chan.read(get_best_product(filename, area_extent))
        ct_chan.satid = (scene.satname.capitalize() +
                         str(int(scene.number)).rjust(2))
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)

    if "CloudType" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    %{"number": "02",
                      "product": "CT___"})
        ct_chan = MsgCloudType()
        ct_chan.read(get_best_product(filename, area_extent))
        ct_chan.satid = (scene.satname.capitalize() +
                         str(int(scene.number)).rjust(2))
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)

    LOG.info("Loading channels done.")
