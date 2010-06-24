"""Very simple netcdf reader for mpop.
"""
from netCDF4 import Dataset
import datetime
import saturn.runner

def load_from_nc4(filename):
    """Load data from a netcdf4 file.
    """
    rootgrp = Dataset(filename, 'r')

    klass = saturn.runner.get_class(rootgrp.Platform,
                                    rootgrp.Number,
                                    rootgrp.Service)
    time_slot = datetime.datetime.strptime(rootgrp.Time,
                                           "%Y-%m-%d %H:%M:%S UTC")
    area_id = rootgrp.Area_Name

    scene = klass(time_slot=time_slot, area_id=area_id)

    for var_name in rootgrp.variables:
        
        var = rootgrp.variables[var_name]

        scene[var.short_name] = var[:,:].astype(var.dtype)

        if not hasattr(scene[var.short_name], 'info'):
            scene[var.short_name].info = {}
        scene[var.short_name].info['var_data'] = scene[var.short_name].data
        scene[var.short_name].info['var_name'] = var_name
        scene[var.short_name].info['var_dim_names'] = var.dimensions
        for attr in var.ncattrs():
            scene[var.short_name].info[attr] = getattr(var, attr)

    
    if not hasattr(scene, 'info'):
        scene.info = {}
        
    for attr in rootgrp.ncattrs():
        scene.info[attr] = getattr(rootgrp, attr)

    scene.info['var_children'] = []
    rootgrp.close()

    return scene


if __name__ == '__main__':
    
    g = load_from_nc4('tester.nc')
    l = g.project("euro4")
    l.vis06().show()
