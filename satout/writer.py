#!/usr/bin/env python
#
#
#
#


__revision__ = 0.1 

""" mpop writer interface """



class WriterDimensionError(Exception):
    """ Basic writer exception """
    pass


def attribute_dispenser( info ):
    """ returns valid attribute key value pairs"""
    for k, v in info.iteritems():
        if k.startswith('var_'):
            continue
        yield (k,v)


def variable_dispenser( root_object, object_list ):
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
            for o in obj:
                try:
                    o.info
                    variable_dispenser(o, object_list)
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

def find_tag(info_list , tag):
    """ 
        Iterates through info objects to find specific tag. 
        Returns list of matching values.
    """
    tag_data = []
    for m in info_list: 
        try:
            tag_data.append(m[tag])
        except KeyError:
            pass
    return tag_data

def find_info(info_list, tag):
    """ 
        Iterates through info objects to find specific tag
        Return list of matching info objects
    """
    tag_info_objects = []
    for m in info_list: 
        if 'var_callback' in m:
            tag_info_objects.append(m)
    return tag_info_objects


def netcdf_cf_writer( filename, root_object ):
    """ Write data to file to netcdf file. """
    import netCDF4
    from netCDF4 import Dataset

    rootgrp = Dataset(filename, 'w', format='NETCDF4_CLASSIC')
    info_list = []
    variable_dispenser( root_object, info_list )
    
    # find available dimensions
    ##dims = find_tag( info_list , 'var_dims' )
    dim_names = find_tag( info_list , 'var_dim_names' )

    # go through all cases of 'var_callback' and create objects which are
    # linked to by the 'var_data' keyword. This ensures that data are only read
    # in when needed.

    cb_infos = find_info(info_list, 'var_callback')
    
    for info in cb_infos:
        # execute the callback functors
        info['var_data'] = info['var_callback']()

    var_data = find_tag(info_list , 'var_data')
    
    # create dimensions in NetCDF file, dimension lenghts are base on array
    # sizes
    used_dim_names = {}
    for names, values in zip(dim_names, [ v.shape for v in var_data ] ):
        for dim_name, dim_size in zip(names, values):

            # ensure unique dimension names
            if dim_name in used_dim_names:
                if dim_size != used_dim_names[dim_name]:
                    raise WriterDimensionError("Dimension name already in use")
                else:
                    continue

            rootgrp.createDimension(dim_name, dim_size)
            used_dim_names[dim_name] = dim_size
   
    # create variables


    var_names = find_tag(info_list , 'var_name')

    nc_vars = []
    for name, vtype, dim_name in zip(var_names, [vt.dtype for vt in var_data ], dim_names ):
        nc_vars.append(rootgrp.createVariable(name, vtype, dim_name))

    # insert attributes, search through info objects and create global
    # attributes and attributes for each variable.
    for mi in info_list:
        if 'var_name' in mi:
            # handle variable attributes
            nc_var = rootgrp.variables[mi['var_name']]
            for k, v in attribute_dispenser(mi):
                setattr( nc_var, k, v )
        else:
            # handle global attributes
            for k, v in attribute_dispenser(mi):
                setattr( rootgrp, k, v )


    # insert data 

    for vname, vdata in zip(nc_vars, var_data):
        vname[:] = vdata

    rootgrp.close()
    

if __name__ == '__main__':
    from pp.satellites.meteosat09 import Meteosat09SeviriScene
    import datetime

    t = datetime.datetime(2009, 10, 8, 14, 30)
    g = Meteosat09SeviriScene(area_id="EuropeCanary", time_slot=t)
    g.load([0.6, 10.8])

    rootgrp = netcdf_cf_writer( 'tester.nc', g )

