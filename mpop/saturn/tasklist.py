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

"""Tasklist class and helper function.
"""

from mpop.saturn.filelist import FileList
from mpop import CONFIG_PATH

class TaskList(dict):
    """Defines a tasklist.
    """
    
    def __init__(self, product_file=None):
        dict.__init__(self)

        if product_file is not None:
            global_vars = {"__file__":product_file}
            local_vars = {}
            execfile(product_file, global_vars, local_vars)
            tasks = local_vars["PRODUCTS"]
            for area in tasks:
                self[area] = {}
                for product in tasks[area]:
                    self[area][product] = FileList(tasks[area][product])

    
    def split(self, *keys):
        """Split the tasklist along the *keys* parameter: keys in the first
        part, non keys in the second.
        """
        tl1 = TaskList()
        tl2 = TaskList()
        for key, item in self.items():
            if key in keys:
                tl1[key] = item
            else:
                tl2[key] = item
        return tl1, tl2

    def get_prerequisites(self, klass, area_id=None):
        """Get the channels we need to load to fulfill the tasklist according
        to methods defined in *klass*. If area is provided, account only for
        tasks on this area.
        """
        if area_id is None:
            areas = self.keys()
        elif(isinstance(area_id, (list, tuple, set))):
            areas = list(area_id)
        else:
            areas = [area_id]
        prerequisites = set()
        for area in areas:
            productlist = self.get(area, {})
            for product in productlist:
                if(hasattr(klass, product) and
                   hasattr(getattr(klass, product), "prerequisites")):
                    prerequisites |= getattr(getattr(klass, product),
                                             "prerequisites")
        return prerequisites

    def shape(self,
              klass,
              mode=set(),
              original_areas=set(),
              specific_composites=set()):
        """Shape the given the tasklist according to the options.
        """

        composites = set()

        if len(original_areas) == 0:
            original_areas = set(self.keys())

        new_tasklist = TaskList()

        if len(specific_composites) == 0 and len(mode) ==0:
            for area in original_areas:
                new_tasklist[area] = self.get(area, {})

        for i in dir(klass):
            if hasattr(getattr(klass, i), "prerequisites"):
                if("pge" in mode and
                   ("CloudType" in getattr(getattr(klass, i),
                                           "prerequisites") or
                    "CTTH" in getattr(getattr(klass, i),
                                      "prerequisites"))):
                    composites |= set([i])
                elif("rgb" in mode and
                   ("CloudType" not in getattr(getattr(klass, i),
                                               "prerequisites") or
                    "CTTH" not in getattr(getattr(klass, i),
                                          "prerequisites"))):
                    composites |= set([i])

        for area in original_areas:
            new_tasklist.setdefault(area, {})
            for product in specific_composites:
                filelist = self.get(area, {}).get(product,
                                                  [area+"_"+product+".png"])
                new_tasklist[area].setdefault(product, FileList())
                new_tasklist[area][product].extend(filelist)
            for product in composites:
                filelist = self.get(area, {}).get(product, FileList())
                if len(filelist) > 0:
                    new_tasklist[area].setdefault(product, FileList())
                    new_tasklist[area][product].extend(filelist)


        return new_tasklist

def get_product_list(satscene):
    """Returns the tasklist corresponding to the satellite described in
    *satscene*, which can be a scene object, a list or a tuple. If the
    corresponding file could not be found, the function returns more generic
    tasklists (variant and name based, then only variant based), or None if no
    file can be found.

    NB: the product files are looked for in the CONFIG_PATH directory.
    """
    
    if isinstance(satscene, (list, tuple)):
        if len(satscene) != 3:
            raise ValueError("Satscene must be a triplet (variant, name, "
                             "number) or a scene object.")
        components = satscene
    else:           
        components = [satscene.variant, satscene.satname, satscene.number]

    import os.path

    for i in range(len(components)):
        pathname = os.path.join(CONFIG_PATH,
                                "".join(components[:len(components)-i]))
        if os.path.exists(pathname+"_products.py"):
            return TaskList(pathname+"_products.py")

    return None
