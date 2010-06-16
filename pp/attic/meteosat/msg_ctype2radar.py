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

"""Ctype2radar allow saving for nordrad format
"""

import logging

LOG = logging.getLogger("pp.meteosat")

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

