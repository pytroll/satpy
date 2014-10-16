#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2014.

# Author(s):

#   Kristian Rune Larssen <krl@dmi.dk>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>

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


"""mpop netcdf4 writer interface.
"""

__revision__ = 0.1

import numpy as np
import logging

logger = logging.getLogger(__name__)


def save(scene, filename, compression=True, dtype=np.int16, band_axis=2):
    """Saves the scene as a NetCDF4 file, with CF conventions.

    *band_axis* gives the which axis to use for the band dimension. For
     example, use band_axis=0 to get dimensions like (band, y, x).
    """
    from mpop.satout.cfscene import CFScene

    scene.add_to_history("Saved as netcdf4/cf by pytroll/mpop.")
    return netcdf_cf_writer(filename, CFScene(scene, dtype, band_axis),
                            compression=compression)


class WriterDimensionError(Exception):

    """ Basic writer exception """
    pass


def attribute_dispenser(info):
    """ returns valid attribute key value pairs
    (for cosmetic reasons, sorted is better than random)"""
    for k, v in sorted(info.iteritems()):
        if k.startswith('var_'):
            continue
        yield (k, v)


def variable_dispenser(root_object, object_list):
    """ Assembles a list of meta info objects """
    # Handle members with info objects
    for v in dir(root_object):
        obj = getattr(root_object, v)
        if callable(obj):
            continue

        # Try to find members with 'info' attribute defined
        # if in list members search through list to find
        # elements with the 'info' attribute defined
        try:
            # test for info attribute on list elements
            for under_obj in obj:
                try:
                    under_obj.info
                    variable_dispenser(under_obj, object_list)
                except AttributeError:
                    pass
        except TypeError:
            try:
                # test for info attribute scalar members
                obj.info
                variable_dispenser(obj, object_list)

            except AttributeError:
                pass

    # Handle local info objects
    try:
        # handle output of member variables without info attribute
        if 'var_children' in root_object.info:
            object_list.extend(root_object.info['var_children'])

        # handle object with info attribute
        object_list.append(root_object.info)
    except AttributeError:
        pass


def find_tag(info_list, tag):
    """ 
        Iterates through info objects to find specific tag. 
        Returns list of matching values.
    """
    tag_data = []
    for info in info_list:
        try:
            tag_data.append(info[tag])
        except KeyError:
            pass
    return tag_data


def find_FillValue_tags(info_list):
    """ 
        Iterates through info objects to find _FillValue tags for var_names
    """
    fill_value_dict = {}
    for info in info_list:
        try:
            fill_value_dict[info['var_name']] = info['_FillValue']
        except KeyError:
            pass
            try:
                fill_value_dict[info['var_name']] = None
            except KeyError:
                pass
    return fill_value_dict


def find_info(info_list, tag):
    """ 
        Iterates through info objects to find specific tag.
        Return list of matching info objects.
    """
    tag_info_objects = []
    for info in info_list:
        if tag in info:
            tag_info_objects.append(info)
    return tag_info_objects


def dtype(element):
    """
        Return the dtype of an array or the type of the element.
    """

    if hasattr(element, "dtype"):
        return element.dtype
    else:
        return type(element)


def shape(element):
    """
        Return the shape of an array or empty tuple if not an array.
    """

    if hasattr(element, "shape"):
        return element.shape
    else:
        return ()


def netcdf_cf_writer(filename, root_object, compression=True):
    """ Write data to file to netcdf file. """
    from netCDF4 import Dataset

    rootgrp = Dataset(filename, 'w')
    try:
        info_list = []
        variable_dispenser(root_object, info_list)

        # find available dimensions
        dim_names = find_tag(info_list, 'var_dim_names')

        # go through all cases of 'var_callback' and create objects which are
        # linked to by the 'var_data' keyword. This ensures that data are only
        # read in when needed.

        cb_infos = find_info(info_list, 'var_callback')

        for info in cb_infos:
            # execute the callback functors
            info['var_data'] = info['var_callback']()

        var_data = find_tag(info_list, 'var_data')

        # create dimensions in NetCDF file, dimension lengths are based on
        # array sizes
        used_dim_names = {}
        for names, values in zip(dim_names, [shape(v) for v in var_data]):

            # case of a scalar
            if len(names) == 0:
                continue
            for dim_name, dim_size in zip(names, values):

                # ensure unique dimension names
                if dim_name in used_dim_names:
                    if dim_size != used_dim_names[dim_name]:
                        raise WriterDimensionError("Dimension name "
                                                   + dim_name +
                                                   " already in use")
                    else:
                        continue

                rootgrp.createDimension(dim_name, dim_size)
                used_dim_names[dim_name] = dim_size

        # create variables

        var_names = find_tag(info_list, 'var_name')

        nc_vars = []

        fill_value_dict = find_FillValue_tags(info_list)
        for name, vtype, dim_name in zip(var_names,
                                         [dtype(vt) for vt in var_data],
                                         dim_names):

            # in the case of arrays containing strings:
            if str(vtype) == "object":
                vtype = str
            nc_vars.append(rootgrp.createVariable(
                name, vtype, dim_name,
                zlib=compression,
                fill_value=fill_value_dict[name]))

        # insert attributes, search through info objects and create global
        # attributes and attributes for each variable.
        for info in info_list:
            if 'var_name' in info:
                # handle variable attributes
                nc_var = rootgrp.variables[info['var_name']]
                nc_var.set_auto_maskandscale(False)
                for j, k in attribute_dispenser(info):
                    if j not in ["_FillValue"]:
                        setattr(nc_var, j, k)
            else:
                # handle global attributes
                for j, k in attribute_dispenser(info):
                    try:
                        setattr(rootgrp, j, k)
                    except TypeError as err:
                        logger.warning("Not saving %s with value %s because %s",
                                       str(j), str(k), str(err))

        # insert data

        for name, vname, vdata in zip(var_names, nc_vars, var_data):
            vname[:] = vdata
    finally:
        rootgrp.close()


if __name__ == '__main__':
    from mpop.satellites.meteosat09 import Meteosat09SeviriScene
    import datetime

    TIME = datetime.datetime(2009, 10, 8, 14, 30)
    GLOB = Meteosat09SeviriScene(area_id="EuropeCanary", time_slot=TIME)
    GLOB.load([0.6, 10.8])

    save(GLOB, 'tester.nc')
