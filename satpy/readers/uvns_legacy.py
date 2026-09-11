#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version #
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to Sentinels 4,5, UVN and 5P Tropomi L1 & 2 netCDF Reader.

The UVN and UVNS are the satellite instruments on board the Copernicus Sentinel-4 and 5 satellites. 
The TROPOspheric Monitoring Instrument (TROPOMI) is on board the  Sentinel 5 Precursor (S5P) satellite.
These sensors measure key atmospheric trace gasses such as ozone, nitrogen oxides, sulfur dioxide,
carbon monoxide, methane, and formaldehyde.
Sentinel 5P data are available from the 'Sentinel Ecosystem': https://dataspace.copernicus.eu/

"""
import contextlib 
from   satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4
import logging
import numpy as np
import xarray as xr
import dask.array as da
import re
from   datetime import datetime as d_t
import dateutil.parser as dparser
import string
import itertools
from   collections import Counter
from   netCDF4 import Dataset
import h5py
import glob
from   pyresample.geometry import SwathDefinition
##from   satpy.utils import debug_on
##debug_on()
 
logger = logging.getLogger(__name__)

class EmptyFileError(Exception):
    "Raised when a netCDF4 file is empty."
    pass


#==========================================================================
# Start of netCDF file traversal classes and methods.
#==========================================================================

class NC_Group_Walker():
    '''Base class to traverse internal netCDF paths.'''
    #--------------------------------------------------------------------------

    @classmethod
    def Walk_Back( cls, current_path):
        """Traverse an internal netcdf path backwards generating a sequence of initial and parent paths.

        :param cls: class method parameter
        :param current_path: Current internal netCDF path from which parent paths are to be found.
        :returns: yields current path(s) recursively from leaf to root.
        :rtype: string

        """
        yield current_path.lstrip( '/')
        if current_path:
            next_path = current_path.rsplit('/', 1)[0]
            if current_path != next_path:
                yield from cls.Walk_Back( next_path)
            else:
                yield ''

    # Walk down the tree - trunk to leaf.  
    @classmethod
    def Walk_Forward( cls, current_group, group_path=''):
        """Traverse an internal netcdf path from root to leaf, through the groups,

        :param cls: class method parameter
        :param current_group: name of current group node. 
        :param group_path: internal netCDF path of current_group.
        :returns: yields current (all) group path(s) recursively from root to leaf.
        :rtype: string

        """
        if group_path:
            yield group_path.lstrip( '/')
        for sub_group in current_group.groups.keys():
            yield from cls.Walk_Forward( current_group.groups[sub_group], group_path + '/' + sub_group)
            

class NC_Walker( NC_Group_Walker):
    '''Add handlers for attributes and variables during traversal of a netCDF file.

    :base: NC_Group_Walker

    '''
    
    @classmethod
    def Attributes( cls, current_group):
        """Given an internal netCDF path, find the names of the attributes.

        :param cls:  class method parameter
        :param current_group: netCDF4 group
        :returns: list of attribute names.
        :rtype: list

        """
        return current_group.ncattrs()
        
    @classmethod
    def Variables( cls, current_group):
        """Given an internal netCDF path, find the names of the variables.

        :param cls:  class method parameter
        :param current_group: netCDF4 group
        :returns: list of variable names.
        :rtype: list

        """
        return current_group.variables.keys()
    
    @classmethod
    def Dimensions( cls, current_group):
        """Given an internal netCDF path, find the names of the dimensions.

        :param cls:          class method parameter
        :param current_group: netCDF4 variable/group
        :returns:            list of dimension names.
        :rtype:              list

        """
        return current_group.dimensions.keys()    
    #--------------------------------------------------------------------------

    @classmethod
    def Walk_Object(cls, walk_func, obj_func, root_group):
        """Traverse a netCDF file in direction defined by walk_func returning
           netCDF internal path to object (attribute, variable) retrived by obj_func.

        :param cls:       class method parameter
        :param walk_func:  traversal function and associated args.
        :param obj_func:   function to retieve netCDF element (variable, attribute)
        :param root_group: netCDF group entry point (e.g. top level).
        :returns:         generates paths to netCDF elements found. 
        :rtype:           string

        """
        for current_path in getattr(cls, walk_func['Name'])(*walk_func['Args']):
            try:
                for obj_name in getattr(cls, obj_func)(root_group[current_path]):
                    yield '/'.join([current_path, obj_name])
            except (IndexError, AttributeError):
                pass
            
    @classmethod
    def Walk_Back_Object( cls, obj_func, root_group, leaf_path):
        """Report the internal netCDF path of elements (attributes, variables) identified
           by the obj_func and found from a reverse search from leaf (leaf_path) to root.

        :param cls:       class method parameter
        :param obj_func:   method returning netCDF element to report e.g. Attributes
        :param root_group: netCDF4 Dataset root group.
        :param lea_path:   initial netCDF internal path from where to start search.
        :returns:         generates internal netCDF path(s) to found element(s)
        :rtype:           string

        """
        yield from cls.Walk_Object({'Name':'Walk_Back', 'Args':[leaf_path]}, obj_func, root_group)
            
    @classmethod
    def Walk_Back_Group_Attributes(cls, root_group, leaf_path):
        """Report the internal netCDF path of netCDF attributes found from a
           reverse search from leaf (leaf_path) to root.

        :param cls:       class method parameter
        :param root_group: netCDF4 Dataset root group.
        :param leaf_path:  initial netCDF internal path from where to start search.
        :returns:         generates internal netCDF path(s) to attribute(s).
        :rtype:           string

        """
        yield from cls.Walk_Back_Object('Attributes', root_group, leaf_path)

    @classmethod
    def Walk_Back_Variables(cls, root_group, leaf_path):
        """Report the internal netCDF path of netCDF variables found from a
           reverse search from leaf (leaf_path) to root.

        :param cls:       class method parameter
        :param root_group: netCDF4 Dataset root group.
        :param leaf_path:  initial netCDF internal path from where to start search.
        :returns:         generates internal netCDF path(s) to variable(s).
        :rtype: 

        """
        yield from cls.Walk_Back_Object('Variables', root_group, leaf_path)
            
    #--------------------------------------------------------------------------
    
    @classmethod
    def Walk_Forward_Object(cls, obj_func, root_group):
        """Report the internal netCDF path of elements identified by the obj_func while
           traversing the netCDF file.

        :param cls:       class method parameter
        :param obj_func:   method returning netCDF element to report e.g. Attributes
        :param root_group: netCDF4 Dataset root group.
        :returns:         generates internal netCDF path(s) to found element(s)
        :rtype:           string

        """
        yield from cls.Walk_Object({'Name':'Walk_Forward', 'Args':[root_group]}, obj_func, root_group)

    @classmethod
    def Walk_Group_Attributes(cls, root_group):
        """"Report the internal netCDF path of netCDF attributes found from a
           search from root to leaf.

        :param cls:       class method parameter
        :param root_group: netCDF4 Dataset root group or starting point.
        :returns:         generates internal netCDF path(s) to attribute(s).
        :rtype:           string

        """
        yield from cls.Walk_Forward_Object('Attributes', root_group)

    @classmethod
    def Walk_Variables(cls, root_group):
        """Report the internal netCDF path of netCDF variables found from a
           search from root to leaf.

        :param cls:       class method parameter
        :param root_group: netCDF4 Dataset root group.
        :returns:         generates internal netCDF path(s) to variable(s).
        :rtype:           string

        """
        yield from cls.Walk_Forward_Object('Variables', root_group)
                                 
    #==========================================================================
    #==========================================================================
    
class NC_Finder( NC_Walker):
    '''Adds capability to find and return internal netCDF4 paths which match
    a given (variable or attribute) name.

    :base: NC_Walker

    '''
    #--------------------------------------------------------------------------

    @classmethod
    def Match(cls, name0, name1):
        """Given two names or internal netCDF paths, see if the last element (name) matches.

        :param cls:   class method parameter
        :param name0: name or internal netCDF path
        :param name1: name or internal netCDF path
        :returns:     True if name0 == name1, False if not.
        :rtype:       boolean

        """
        return name0.rsplit('/', 1)[-1] == name1.rsplit('/', 1)[-1]

    @classmethod
    def Find_Object(cls, name_query, walk_object):
        """Find all internal netCDF4 paths for objects found when traversing a netCDF file.

        :param cls:        class method parameter
        :param name_query:  name or identifier
        :param walk_object: dict providing NC_Walker method name and args.
        :returns:          netCDF internal path(s) to matched element(s)
        :rtype:            string

        """
        for candidate_name in getattr(cls, walk_object['Name'])(*walk_object['Args']):
            if cls.Match(name_query, candidate_name):
                yield candidate_name
                
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    @classmethod
    def Find_Ancestor_Objects(cls, name_query, obj_func, root_group, nc_path):
        """Find all internal netCDF4 paths for required objects found when traversing
           a netCDF file in the reverse direction (leaf to root).
        
        :param cls:       class method parameter
        :param name_query: identifier to find
        :param obj_func:   function to provide element (e.g. variable, or attribute). 
        :param root_group: netCDF4 starting entry point group.
        :param nc_path:    path to leaf starting point (relative to root_group).
        :returns:         netCDF internal path(s) to matched element(s)
        :rtype:           string

        """
        yield from cls.Find_Object(name_query, {'Name':"Walk_Back_Object", 'Args':[obj_func, root_group, nc_path]})

    @classmethod
    def Find_Ancestor_Group_Attributes(cls, name_query, root_group, nc_path):
        """Find all internal netCDF4 paths for attributes found when traversing
           a netCDF file in the reverse direction (leaf to root).
        
        :param cls:       class method parameter
        :param name_query: identifier to find
        :param root_group: netCDF4 starting entry point group.
        :param nc_path:    path to leaf starting point (relative to root_group).
        :returns:         netCDF internal path(s) to matched element(s)
        :rtype:           string

        """
        yield from cls.Find_Ancestor_Objects(name_query, 'Attributes', root_group, nc_path)
            
    @classmethod
    def Find_Ancestor_Variables(cls, name_query, root_group, nc_path):
        """Find all internal netCDF4 paths for variables found when traversing
           a netCDF file in the reverse direction (leaf to root).
        
        :param cls:       class method parameter
        :param name_query: identifier to find
        :param root_group: netCDF4 starting entry point group.
        :param nc_path:    path to leaf starting point (relative to root_group).
        :returns:         netCDF internal path(s) to matched element(s)
        :rtype:           string

        """        
        yield from cls.Find_Ancestor_Objects(name_query, 'Variables', root_group, nc_path)
        
    #--------------------------------------------------------------------------    
    #--------------------------------------------------------------------------
        
    @classmethod
    def Find_Descendant_Objects(cls, name_query, obj_func, root_group):
        """Find all internal netCDF4 paths for required objects found when traversing
            a netCDF file in the forward direction (root to leaf).
        
        :param cls:       class method parameter
        :param name_query: identifier to find
        :param obj_func:   function to provide element (e.g. variable, or attribute). 
        :param root_group: netCDF4 starting entry point group.
        :param nc_path:    path to leaf starting point (relative to root_group).
        :returns:         netCDF internal path(s) to matched element(s)
        :rtype:           string

        """
        yield from cls.Find_Object(name_query, {'Name':"Walk_Forward_Object", 'Args':[obj_func, root_group]})

    @classmethod
    def Find_Descendant_Attributes(cls, name_query, root_group):
        """Find all internal netCDF4 paths for attributes found when traversing
           a netCDF file in the forward direction (root to leaf).

        :param cls:       class method parameter
        :param name_query: identifier to find
        :param root_group: netCDF4 starting entry point group.
        :returns:         netCDF internal path(s) to matched element(s)
        :rtype:           string

        """
        yield from cls.Find_Descendant_Objects(name_query, 'Attributes', root_group)
        
    @classmethod
    def Find_Descendant_Variables(cls, name_query, root_group):
        """Find all internal netCDF4 paths for variables found when traversing
           a netCDF file in the forward direction (root to leaf).

        :param cls:       class method parameter
        :param name_query: identifier to find
        :param root_group: netCDF4 starting entry point group.
        :returns:         netCDF internal path(s) to matched element(s)
        :rtype:           string

        """
        yield from cls.Find_Descendant_Objects(name_query, 'Variables', root_group)
   
#==========================================================================
# End of netCDF file traversal classes and methods.
#==========================================================================

#==========================================================================
# Start of classes with methods to handle string matching.
#==========================================================================
class Match_Query():
    """Class containing methods to find strings which match a pattern best. 

    """
    @classmethod
    def Match( cls, query, reference, con_sym='*'):
        """See if an incomplete string with missing sections matches a complete reference string
           (case insensitive).

        :param cls:       class method parameter
        :param query:     a query string with optional gaps filled with 'con_sym'
                          e.g. 'sensing*start*time*'
        :param reference: complete reference string for comparison.
        :param con_sym:   symbol for gaps of 0 or more characters.
        :returns:         True if query matches, else False
        :rtype:           boolean

        """
        
        current_reference = reference.lower()
        current_query     = query.lower()

        if not con_sym in current_query :
            # A simple match if no gaps.
            return current_query == current_reference
       
        star_parts = current_query.split(con_sym)
        # Align with left
        star_part = star_parts[0]        
        if star_part:
            if current_reference.startswith(star_part):            
                len_parts = len(star_parts)
                if len_parts == 1 or (len_parts == 2 and not star_parts[1]):
                    return True
            else:
                return False
            #Matches but more to process
            current_reference = current_reference[len(star_part):]
            
        star_parts = star_parts[1:]
        # Align with right
        star_part  = star_parts[-1]
        if star_part :
            if current_reference.endswith(star_part):
                len_parts = len(star_parts)
                if len_parts == 1 or (len_parts == 2 and not star_parts[0]) :
                    return True
            else:
                 return False
            #Matches but more to process
            current_reference = current_reference[:(len(current_reference) - len(star_part))]          
        star_parts = star_parts[:-1]

        # If necessary, Finish by breakdown.
        for star_part in star_parts:
            try:
                current_index = current_reference.index(star_part)
            except:
                return False
            current_reference = current_reference[(current_index + len(star_part)):]
        return True

    #-------------------------------------------------------------------------------    
    @classmethod
    def Matches(cls, queries, references):
        """Find queries from a query string or list of query strings which are in a list of reference strings.
           The queries can be partial queries (see 'Match').
           e.g. queries = ['scan*line', 'ground_pixel'],
           e.g. references = ['wavelength', 'scanline', 'ground_pixel', 'corners']

        :param cls:        class method parameter
        :param queries:    a single string or list of query strings
        :param references: a single string or list of reference (complete) strings.
        :returns:          the query string and its match
        :rtype:            triplet of 2 strings

        """

        #If parameters are strings, convert to list of one string
        if not isinstance(queries, list):
            query_list = [queries]
        else:
            query_list = queries

        if not isinstance(references, list):
            reference_list = [references]
        else:
            reference_list = references

        for current_query in query_list:
            for current_reference in reference_list:
                if cls.Match(current_query, current_reference):
                    yield (current_query, current_reference)
                    
    #--------------------------------------------------------------------------------

# Extends Match_Query to use a list of queries found in a dictionary and parses the list
# according to the selected dict keys.
class Match_ID(Match_Query):
    """Class containing methods to find the closest matches between netCDF4 elements
       and expected/desired ones and retrieve them.
       Maps names used in the program/satpy with names found in the netCDF file.  

       :base: Match_Query
    """

    @classmethod
    def Clean_HDF5_String(cls, dirty_string):
        """netCDF4 doesn't handle HDF5 strings correctly, a combination of bytearrays
        and encoding, this method attempts to clean them.

        :param cls:         class method paramter
        :param dirty_string: hdf5 string
        :returns:           'cleaned' string
        :rtype:             string

        """
        clean_string = ''
        import re
        #clean_string = re.sub(r'[^x00-x7_f]+',' ', dirty_string)
        if dirty_string.endswith('\x7f'):
            dirty_string = dirty_string[:-2]
    
        for character in dirty_string :
            if ord(character) < 20 or ord(character) > 127 :
                break
            clean_string += character
        return clean_string


    @classmethod
    def Forward(cls, object_type, root_group, leaf_path=''):
        """Convenience method to instigate forward traversal of netCDF4 file.

        :param cls:        class method paramter
        :param object_type: type of object to retrieve
        :param root_group:  netCDF4 Dataset root or starting position.
        :param leaf_path:   ignored
        :returns:          generates paths to required objects
        :rtype:            string

        """
        
        yield from NC_Finder.Walk_Forward_Object(object_type, root_group)
        
    @classmethod
    def Reverse(cls, object_type, root_group, leaf_path):
        """Convenience method to instigate reverse traversal of netCDF4 file.

        :param cls:        class method paramter
        :param object_type: type of object to retrieve
        :param root_group:  netCDF4 Dataset position.
        :param leaf_path:   internal netCDF4 path to start from, relative to root_group.
        :returns:          generates paths to required objects
        :rtype:            string

        """
        yield from NC_Finder.Walk_Back_Object(object_type, root_group, leaf_path)

    @classmethod
    def Get_Attributes(cls, att_name, nc_path):
        """Convenience method to retrieve named attribute.

        :param cls:     class method paramter
        :param att_name: name of attribute to be retrived
        :param nc_path:  netCDF4 Dataset element where attribute to be found.
        :returns:       netCDF4 attribute
        :rtype:         attribute type

        """
        return nc_path.getncattr(att_name)

    @classmethod
    def Get_Variables(cls, var_name, nc_path):
        """Convenience method to retrieve named variable.

        :param cls:     class method paramter
        :param var_name: name of attribute to be retrived
        :param nc_path:  netCDF4 Dataset variables 'group'.
        :returns:       netCDF4 Dataset variable
        :rtype:         netCDF4 Dataset variable type

        """
        return nc_path.variables[var_name]
    
    @classmethod
    def Get_Dimensions(cls, dim_name, nc_path):
        """Convenience method to retrieve named dimension.

        :param cls:     class method paramter
        :param var_name: name of attribute to be retrived
        :param nc_path:  netCDF4 Dataset dimensions 'group'.
        :returns:       netCDF4 Dataset dimension
        :rtype:         netCDF4 Dataset dimension type

        """
        return nc_path.dimensions[dim_name]
    
    @classmethod
    def Find_IDs(cls, object_type, metadata_templates, root_group, leaf_path='', all=False):
        """Find netCDF4 file elements which match the (partial) names in a list.
           Maps the found names with reference names.
           If the list is ordered by 'desirability' and the parameter 'all' is True,
           the 'best' match will be found.

        :param cls:         class method paramter
        :param object_type:  netCDF4 element type (e.g. attribute, etc.)
        :param metadata_templates: a dict of reference names (keys) and lists of fuzzy search names (values).
        :param root_group:   root or starting netCDF4 Dataset group/element 
        :param leaf_path:    starting path, relative to root_group, for reverse traversal.
        :param all:         if True return all found, if False return the first.
        :returns:           the netCDF4 reference name and the internal netCDF path of the matched element.
        :rtype:             triplet of 2 strings.

        """
        for ID in metadata_templates.keys():
            for query in metadata_templates[ID]:
                if leaf_path :
                    walk_direction_function = 'Reverse'
                else:
                    walk_direction_function = 'Forward'
                    
                for o_name in getattr( cls, walk_direction_function)( object_type, root_group, leaf_path) : 
                    for i, j in cls.Matches( query, o_name.rsplit('/')[-1]) :
                        yield ID, o_name
                        if not all:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
                
        #return current_status
        
    @classmethod
    def Get_Objects(cls, obj_type, obj_templates, root_group, leaf_path='', hdf5=False):
        """Find the file elements which best match the names used/expected by satpy,
           remap the names and return the matched netCDF4 Dataset elements.

        :param cls:      class method paramter
        :param obj_type:  netCDF4 element type (e.g. attribute, etc.)
        :param obj_templates: a dict of reference names (keys) and lists of fuzzy search names (values).
        :param root_group:   root or starting netCDF4 Dataset group/element 
        :param leaf_path: starting path, relative to root_group, for reverse traversal.
        :param hdf5:     working on an HDF5 file?
        :returns:        a dict of reference names (keys) netCDF4 Dataset loaded objects (values).
        :rtype:          dict

        """
        current_status = {}
        for key, obj_path in cls.Find_IDs(obj_type, obj_templates, root_group, leaf_path):
            obj_parts = obj_path.rsplit('/', 1)

            # Doesn't like using empty string on root_group.
            if obj_parts[0] :
                curr_path = root_group[obj_parts[0]]
            else:
                curr_path = root_group
            nc_object = getattr(cls, 'Get_' + obj_type)(obj_parts[1],  curr_path)

            # Hate using isinstance ...
            if isinstance(nc_object, str) and hdf5 :
                nc_object = cls.Clean_HDF5_String(nc_object)
            current_status[key] = nc_object              
        return current_status

###########################################################################################
###########################################################################################
class Coordinate_Base():

    @classmethod
    def Test(cls, values, alt_values, primary_reference, secondary_reference, complete=True):
        """Check that all entries in a reference list occur in a given list.
        Remap the given list to alternative values.

        :param values: list to be compared (dimension names or shape)
        :param alt_values: list of replacement elements (shape or dimension names)
        :param reference_dimensions: reference list (dimension names or shape)
        :returns: ordered list of renamed elements or empty list if incomplete match.
        :rtype: list of strings or numbers.

        """
        a_v = []

        ##logger.debug('TEST')
        ##logger.debug(f'values: {values}')
        ##logger.debug(f'alt_values: {alt_values}')
        ##logger.debug(f'primary_reference: {primary_reference}')
        ##logger.debug(f'secondary_reference: {secondary_reference}')
        ##logger.debug('bbbbbbbbbbbbbbbbbbbbbbbb')
        
        for ref0, ref1 in list(zip(primary_reference, secondary_reference)):
            if not ref0 in values :
                return []
            else:
                ##Problem of named dimension having different size, but does this work for shaped dimensions?
                ##This means when we check with the size, the dimension names must agree!?
                ##I think we want to allow different names but not different sizes?
                alt_value = alt_values[values.index(ref0)]
                if complete and alt_value != ref1:
                    return []
                a_v.append(alt_value)
        return a_v

class Geographical_Coordinates(Coordinate_Base):
    """satpy geocoordinate interface with netCDF4 to find, name, and manage geocoordinate information in a netCDF4 file.

    """
    #    dims = variable.dimensions
    #    dims = variable.shape
            
    def Valid( self, nc_variable):
        """Check whether the dimensions of a nc_variable correspond to a preselected set of dimensions.
        Selection can be based on dimension names or dimension lengths.
        If selection is based on dimension names, determine associated dimension lengths, and vice versa.

        :param nc_variable: nc_variable dict containing shape and dimension information.
        :returns: dict containing lists of dimension names and associated lengths (shape). None if incomplete.
        :rtype: dict or None

        """        
        # Match the data arrays on the basis of common shapes.
        shape = self.coordinate_information['Shape']
        names = self.coordinate_information['Names']

        if 'Shape' in self.coordinate_information['Basis']:
            a_v = self.Test(nc_variable.shape, nc_variable.dimensions, shape, names)
            if a_v :
                return {self.names_name: a_v, self.shape_name: shape}
        # Match the data arrays using dimension names.
        else:
            a_v = self.Test(nc_variable.dimensions, nc_variable.shape, names, shape)
            if a_v :
                return {self.names_name: names, self.shape_name: a_v}
            
        return None
                
    #-----------------------------------------------------------------
    
    def Shapes( self, root_group):
        """Find all the array shapes for variables below the given root.

        :param root_group: netCDF4 Dataset node (group) from which to start search.
        :returns:         generates numpy shape triplets.
        :rtype:           triplet of integers

        """
        nc_walker = NC_Walker()
        for variable in nc_walker.Walk_Variables( root_group):
            yield root_group[variable].shape
            
    def Sort1( self, shape_list):
        """Given a list of triplets (array shapes) count the frequency of each value (1-D sort).

        :param shape_list: list of triplets or list
        :returns: lists of 'frequency' dicts containing individual values and their frequency. 
        :rtype: list of 1-D 'frequency' dicts

        """
        counted = Counter([item for sublist in shape_list for item in sublist])
        ix = []
        for k, v in counted.items() :
            ix.append( {'n':v, 'y':k})
        return ix
            
    def Sort2( self, shape_list):
        """Given a list of triplets (2-D array shapes) count the frequency of each value (2-D sort).

        :param shape_list: list of triplets, or of list, of 2-D 'shape' information.
        :returns: lists of 'frequency' dicts containing value pairs and their frequency. 
        :rtype: list of 2-D 'frequency' dicts

        """
        sort_shape = sorted(shape_list, key=lambda x: (x[0], x[1]))
        i = 0
        k = i
        ix = [{'n':0, 'x':sort_shape[k][1], 'y':sort_shape[k][0]}]
        while i < len( sort_shape):
            if sort_shape[k][0] == sort_shape[i][0] and sort_shape[k][1] == sort_shape[i][1]:
               ix[-1]['n'] += 1
               i           += 1
            else:
               k = i
               ix.append( {'n':0, 'x':sort_shape[k][1], 'y':sort_shape[k][0]})
        return ix

    def Shape_Max( self, hist_list):
        """Given a list of 'frequency' dicts, find the dict with maximum frequency.

        :param hist_list: list of 'frequency' dicts
        :returns: 'frequency' dict with maximum value or empty dict if none found.
        :rtype: dict.

        """
        max_num = 0
        for i in range( len( hist_list)):
            if max_num < hist_list[i]['n'] :
                max_num = hist_list[i]['n']
                max_ix  = i
        if max_num :
            return hist_list[max_ix]
        return {}
        
    def Shape_Deconstruct(self, root_group):
        """Deduce the y [and x] dimensions from the frequency of occurence of the netCDF4 variables' dimension sizes.

        :param root_group: netCDF4 Dataset node (group) from which to start search.
        :returns: shape as list or empty list.
        :rtype: list

        """
        shape_list2 = []
        shape_list1 = []
        for shape in self.Shapes(root_group):
            if len( shape)   == 1:
                shape_list1.append(list(shape))
            elif len( shape) == 2 :
                shape_list2.append(list(shape))

        hist_list = self.Sort2(shape_list2)
        max2 = self.Shape_Max(hist_list)

        hist_list = self.Sort1(shape_list1)
        max1 = self.Shape_Max(hist_list)

        #Data dominated by 1 dimension.
        try:
            if max1['n'] >  max2['n'] * 4 :
                return [max1['y']]
        except KeyError:
            pass
        
        if max2:
            return [max2['y'], max2['x']]
        elif max1:
            return [max1['y']]

        return []
                        
    #-----------------------------------------------------------------
    def Dimensions( self, root_group, hdf5=False):
        '''Examine the dimension information in a netCDF file to identify the 'y','x' dimensions required by satpy.
        Implemented are 3 ways to find the geocoordinates, represented in satpy by (y, x), for the xarray.
        The first method is to find any dimension names corresponding to a template pattern match.
        The second relies on finding a latitude data variable and using that for the selected dimension names or shape.
        The third examines the shapes in the file, and assumes the shape which occurs most is the
        default data shape.
 
        :param root_group: netCDF4 Dataset node (group) from which to start search.
        :returns: a 'coordinate information' dict containing the dimension names associated with the 'y' and 'x' dimensions, their sizes,
                  and which method was used for the identification.
        :rtype: (coordinate information) dict

        '''
        
        decision   = 'Dimension Names'
        key        = 'Dimensions'
        dimensions = Match_ID.Get_Objects( key, self.metadata_templates[key], root_group, self.clean_string)
        
        if dimensions :
            dim_names = []
            dim_sizes = []
            for w in ['y','x']:
                if w in dimensions.keys():
                    dim_names.append( dimensions[w].name)
                    dim_sizes.append( dimensions[w].size)
        else:
            # If the named dimensions do not exist, we look at the latitude file to see if the dimensions are named.
            # It is possible the latitude dimension names are different to the data dimensions ....
            key       = 'Variables'
            sub_key    = 'latitude'
            geo_coords = Match_ID.Get_Objects( key, {sub_key: self.metadata_templates[key][sub_key]}, root_group, self.clean_string)
            if geo_coords :
                # Take dimensions/shape from latitude variable.
                decision = 'Latitude Names'
                dim_names = list( geo_coords[sub_key].dimensions)
                dim_sizes = list( geo_coords[sub_key].shape)
                # If the latitude dimensions are phony, we need to look for data with the same shape as the latitude file.
                if dim_names[0].startswith( self.phony_dimension_prefix):
                    decision = 'Latitude Shape'
                    dim_names = []                   
            else:
                # Take dimensions/shape from variable shapes.
                decision = 'Shape'
                dim_names = []
                dim_sizes = self.Shape_Deconstruct( root_group)

        self.coordinate_information = {'Names': dim_names, 'Shape': dim_sizes, 'Basis': decision}
                
    def __init__( self, metadata_templates, is_hDF=False):
        """Initialise dimensions manager

        :param metadata_templates: dict containing dimension search strings.
        :param is_hDF: whether we ar dealing with an HDF file.
        :returns: None
        :rtype: None

        """
        super().__init__()
        self.metadata_templates = {'Dimensions' : metadata_templates['Dimensions']}
        self.metadata_templates['Variables'] = {}
        for key in ['latitude', 'longitude', 'latitude_boundaries', 'longitude_boundaries'] :
            try:
                self.metadata_templates['Variables'][key] =  metadata_templates['Variables'][key]
            except KeyError:
                pass
                
        # pseudo dimension added by netCDF4 when opening a dimensionless hdf5 file.
        self.phony_dimension_prefix = 'phony_dim'
        self.names_name = 'geocoordinate_names'
        self.shape_name = 'geocoordinate_shape'
        self.clean_string = is_hDF

class Name_Manager():
    ''' Satpy allows only unique variable names.  This manager addresses the issue that
    variables with the same name exist but are found under different internal netCDF paths
    such as '*band*' or '*channel*' subgroups.

    '''
 
    @classmethod
    def Variable_Name_Mapping(cls, variable_paths):
        var_name_map = {}
        for varpath in variable_paths:
    	     name = varpath.rsplit('/',1)[-1]
    	     try:
    	         var_name_map[name].append(varpath)
    	     except KeyError:
    	         var_name_map[name] = [varpath]
        return var_name_map

    @classmethod
    def Variable_Name_Duplicates(cls, variable_paths):
        var_name_map = cls.Variable_Name_Mapping(variable_paths)
        unique_name_map = {}
        for name, paths in var_name_map.items():
            if len(paths) > 1:
                parts_list = []
                for path in paths:
                    parts_list.append(path.split('/'))
                common_parts = set.intersection(*[set(parts[:-1]) for parts in parts_list])
                for path, parts in list(zip(paths,parts_list)):
                    parts.reverse()
                    name_list = [pname for pname in parts if pname not in common_parts]
                    unique_name_map[path] = '-'.join(name_list)
            else:
                unique_name_map[paths[0]] = name
        return unique_name_map

    @classmethod
    def Unique_Names(cls, variable_paths):
        return cls.Variable_Name_Duplicates(variable_paths)

class Rename_Manager():
    """Manager to handle x_array coordinate renaming.

    """

    @classmethod
    def _Get_Name(cls, geo_names, geo_shape, new_names):
        """Generate the association between original dimension names and shapes with
        required x_array coordinate dimensions.

        :param cls: class method parameter
        :param geo_names: list of original dimension names
        :param geo_shape: shape of dimensions (list/triplet of dimension sizes).
        :param new_names: list of required new names
        :returns: list of old name, size, new name
        :rtype: list

        """
        for i in range(len(geo_names)):
            yield geo_names[i], geo_shape[i], new_names[i]

    @classmethod
    def Coordinates(cls, geo_names, geo_shape, var_names, var_shape):
        """Create a new geocoordinates dimensions list with required fields renamed
        to satpy's requirement.

        :param cls:  class method parameter
        :param geo_names: identified geocoordinates
        :param geo_shape: identified geocoordinate sizes
        :param var_names: current list of dimensions
        :param var_shape: sizes of current dimension list
        :returns: new list of current dimensions with required dimensions renamed.
        :rtype: list of strings

        """
        renamed_list   = list(var_names).copy()
        for name, size, new_name in cls._Get_Name(geo_names, geo_shape, ['y', 'x']) :
            try:
                renamed_list[var_names.index(name)] = new_name
            except:
                try:
                    renamed_list[var_shape.index(size)] = new_name
                except:
                    pass
        return renamed_list

    @classmethod
    def Default_Coordinates(cls, variable_dimensions, variable_shape):
        rank = len(variable_dimensions)
        if rank < 2:
            return ['y']
        elif rank == 2:
            return ['y','x']

        new_names = []
        #Intentionally reversed because of pop.
        coordinates = ['x','y']
        if variable_dimensions[0] == 'time' and variable_shape[0] > 1:
            coordinates.append('time')
                
        for name, size in list(zip(variable_dimensions,variable_shape)):
            #We need to accomodate size == 0, whatever that means!
            if size < 2:
                new_names.append(name)
            else:
                try:
                    new_names.append(coordinates.pop())
                except:
                    new_names.append(name)
        assert 'y' in new_names
        return new_names

    @classmethod
    def Dimensions(cls, ds_info, variable_dimensions, variable_shape):
        """FIXME! briefly describe function

        :param cls: 
        :param ds_info: 
        :param variable_dimensions: 
        :param variable_shape: 
        :returns: 
        :rtype: 

        """
        try:
            new_names = Rename_Manager.Coordinates(ds_info['geocoordinate_names'], ds_info['geocoordinate_shape'],
                                                  variable_dimensions, variable_shape)
        except KeyError:
            new_names = Rename_Manager.Default_Coordinates(variable_dimensions, variable_shape)
        return new_names

class Information_Manager():
    """ Common utilities Base for Coordinates and Dimensions managers.

    :base: None

    """

    @classmethod
    def Valid_Dicts(cls, options, reference):
        """See if two dicts are equivalent

        :param options: dict for comparison
        :param reference: reference dict
        :returns: True if dicts are equivalent otherwise False
        :rtype: boolean (True/False)

        """
        for key, value in reference.items() :
            try:
                if options[key] != value :
                    return False
            except KeyError :
                return False
        return True
    
    @classmethod
    def Valid_Lists(cls, options, reference):
        """See if two lists contain equivalent items.

        :param options: list for comparison
        :param reference: reference list
        :returns: True if lists are equivalent otherwise False
        :rtype: boolean (True/False)

        """
        for current in reference :
            if current not in options :
                return False
        return True

    @classmethod
    def Valid(cls, options, reference):
        """Establish whether two lists or dicts are equivalent.

        :param cls: 
        :param options: 
        :param reference: 
        :returns: 
        :rtype: 

        """
        if isinstance(reference, dict):
            return cls.Valid_Dicts(options, reference)
        return cls.Valid_Lists(options, reference)
        
    @classmethod
    def Dimensions(cls, current_variable):
        """Convenience method to retrieve the dimensions from a netCDF4 Dataset
        variable node.

        :param cls: class method parameter
        :param current_coordinate: netCDF4 Dataset variable node
        :returns: dimensions of variable
        :rtype: triplet of dimension names

        """
        return current_variable.dimensions
        
    @classmethod
    def Shape(cls, current_variable):
        """Convenience method to retrieve the shape of a netCDF4 Dataset
        variable.

        :param cls: class method parameter
        :param current_coordinate: netCDF4 Dataset variable node
        :returns: shape of variable
        :rtype: triplet of dimension sizes (numerical)

        """
        return current_variable.shape

class Coordinates_Manager(Information_Manager):
    """Manage coordinate identification and match variables to appropriate coordinates.

    :base: Information_Manager

    """

    # Try to find a latitude/longitude file which matches the variable's dimensions
    def Find(self, variable_path, root_group, longitude_name=''):
        """Find the geocoordinates associated with a variable.

        :param variable_path: Internal netCDF4 file internal path to variable.
        :param root_group: root Dataset node (relative to variable path)
        :param longitude_name: name of geocoordinate to search for.
        :returns: internal netCDF4 file path to geocoordinate 
        :rtype: string

        """
        
        key       = 'Variables'
        sub_key    = 'longitude'
        leaf_path  = variable_path.rsplit('/', 1)[0]
        
        # Find any suitable paths using backward lookup.
        geos = [*Match_ID.Find_IDs(key, {sub_key: self.metadata_templates[sub_key]}, root_group, leaf_path, True)]
        
        #  Didn't find it on the backward look, try forward from the top.
        if not geos :
            geos = [*Match_ID.Find_IDs(key, {sub_key: self.metadata_templates[sub_key]}, root_group, False)]
        # Oooops, no suitable geo_coordinate path found.
        if not geos :
            return None
       
        # if this is a geocordinate variable, then the latitude or longitude file refers to itself.
        # We want to retrieve the same one from the list using the full path.
        for r_key, lon_path in geos :
            lat_path = lon_path.replace('ongitude', 'atitude')
            if lon_path == variable_path or lat_path == variable_path :
                return lon_path

        # Maybe we don't have the full geo_coordinate path, but just the geo_coordinate name.
        if longitude_name :
            for r_key, lon_path in geos :
                if lon_path.rsplit('/', 1)[-1] == longitude_name :
                    return lon_path

        # Having failed with the above, match on dimension names and associated sizes - best way remaining.
        current_variable  = root_group[variable_path]
        variable_dict     = dict(zip(current_variable.dimensions, current_variable.shape))
        for r_key, obj_path in geos :
            current_coordinate = root_group[obj_path]
            coordinate_dict    = dict(zip(current_coordinate.dimensions, current_coordinate.shape))
            if self.Valid(variable_dict, dict(zip(current_coordinate.dimensions, current_coordinate.shape))):
                return obj_path

        # Well, that failed, try on just sizes then = error prone.
        for r_key, obj_path in geos :
            if self.Valid(current_variable.shape, root_group[obj_path].shape):
                return obj_path

        # Oh dear, haven't found a geo_coordinate to match - handle upstream.
        return None

    def _NP(self, name, path0, r0, r1):
        """Replace a substring with another in a string or the at the end of a pathname. 

        :param name: original name string
        :param path0: original path string
        :param r0: substring to replace
        :param r1: replacement string
        :returns: original string, original path, modified string, modified path
        :rtype: string, string, string, string

        """
        path1 = path0.rsplit('/', 1)
        path1[-1] = path1[-1].replace(r0, r1)
        path1 = '/'.join(path1)
        return name, path0, name.replace(r0, r1), path1

    def _Standard_Name(self, path_name, coordinates):
        """Identify and/or prepare geocoordinate to have 'standard_name' attribute as required by satpy. 

        :param path_name: netCDF4 internal file path of variable.
        :param coordinates: coordinates list.
        :returns: (attribute) dict for standard_name of geocoordinate or empty dict.
        :rtype: (attribute) dict

        """
        if path_name in coordinates:
            name = path_name.rsplit('/', 1)[-1]
            if 'atitude' in name:
                return {'standard_name': 'latitude'}
            if 'ongitude' in name:
                return {'standard_name': 'longitude'}
        return {}

    def _Mash_Names(self, name, path):
        """Create compatible latitude variable name and netCDF variable path from given longitude name and path
        or vice versa. Order into longitude, latitude sequence.
        
        :param name: geocoordinate name
        :param path: internal netCDF4 path to geocoordinate
        :returns: completed set of gecooordinate names and paths in dict.
        :rtype: (coordinates) dict

        """
        if 'ongitude' in name :
            n0, p0, n1, p1 = self._NP(name, path, 'ongitude', 'atitude')
        else :
            n1, p1, n0, p0 = self._NP(name, path, 'atitude', 'ongitude')
        return {'coordinates':  {n0:p0, n1:p1}}

    def Find_Coordinates(self, variable_path, root_group, longitude_name=''):
        """Search for (the best) geocoordinates in a UVNS netCDF4 file for a
        specific variable.

        :param variable_path: netCDF4 Dataset internal path to current variable.
        :param root_group: netCDF4 Dataset root node
        :param longitude_name: the name of the longitude variable to discover.
        :returns: netCDF4 Dataset internal path to discovered longitude and latitude.
        :rtype: list of two strings.

        """
        longitude_path  = self.Find(variable_path, root_group, longitude_name)
        if not longitude_path :
            return None
        longitude_parts = longitude_path.rsplit('/', 1)
        longitude_parts[-1] = longitude_parts[-1].replace('ongitude', 'atitude')
        latitude_path   = '/'.join(longitude_parts)
        coordinates    = [longitude_path, latitude_path]
        return coordinates

    def _Clean_and_Check_Path(self, coordinates):
        """Ensure netCDF4 internal path of the geocoordinates does not have leading '/'
        and check it is a path. Aborts on path error.

        :param coordinates: list of geocoordinat netCDF4 Dataset internal paths.
        :returns: Nothing, parameter changed in-situ.
        :rtype: None.

        """
        for i in range(len(coordinates)):
            #Remove leading '/' if it exists (netCDF4 barfs otherwise). 
            coordinates[i] = coordinates[i].lstrip('/')
            #Check we have an internal path.
            #Rudimentory path check.
            if '/' not in coordinates[i] :
                logger.error(f'Incorrect coordinates path for {variable_name} at {variable_path}:')
                logger.error(f'{coordinates}')
                assert False
 
    def Get_Coordinates(self, ds_info, root_group):
        """Find the geocordinates for a particular netCDF Dataset variable.

        :param ds_info: 'ds_info' information dict for current variable
        :param root_group: netCDF4 Dataset root node.
        :returns: list of internal netCDF4 Dataset paths for required geocoordinates
        :rtype: list of strings

        """
        #Get the internal netCDF Dataset path to the current variable.
        variable_path = ds_info['file_key']
        try:
            # Get the coordinates attribute if it already exists.
            coordinates = ds_info['coordinates']
            # But is it a path or just a name?
            try:
                root_group[coordinates[0]]
                #Path to longitude found therefore everything O.K.
            except IndexError:
                #Well it doesn't exist, so assume a name.
                #Go away and find the best geocordinates using the longitude name.
                coordinates = self.Find_Coordinates(variable_path , root_group, coordinates[0])
        except KeyError:
            #There can't have been a coordinates.
            #Go away and find the best geocordinates.
            coordinates = self.Find_Coordinates(variable_path, root_group)
        return coordinates
    
    def _Identify_Geocoordinates(self, coordinates, variable_dB):
        """Find/deduce the coordinates in the satpy ds_info 'database'.  

        :param coordinates: list of geocoordinate strings
        :param variable_dB: dict of ds_info
        :returns: dict of satpy geocoordinate names (as keys) and netCDF4 paths as values.
        :rtype: dict

        """
        for key in variable_dB.keys():
            if variable_dB[key]['file_key'] in coordinates:
                #We found a geocoordinate in database.
                #From this, find the other one and return a dict containing both.
                return self._Mash_Names(variable_dB[key]['name'], variable_dB[key]['file_key'])
        #Didn't find geocoordinates in database.
        return {}
 
    # Primarily we need to change the coordinates from a direct nc file path to a satpy id.
    # For the coordinate variables themselves, we need to ensure they have a standard_name
    # attribute which corresponds to latitude or longitude.
    def Update(self, variable_name, variable_dB, root_group):
        """For the given variable find the best geocordinates in the netCDF4 file and
        relate that to the appropriate satpy 'ds_info'

        :param variable_name: netCDF4 Dataset internal path to the current variable.
        :param variable_dB: dict of 'ds_info's
        :param root_group: netCDF4 Dataset root node.
        :returns: coords dict containing 'standard_name' attributes set up, satpy name for
                  the geocoordinate, internal netCDF4 Dataset internal path to the
                  geocoordinate variables.
        :rtype: dict

        """
        
        variable_info = variable_dB[variable_name]
        #Get the netCDF4 internal paths of the geocoordinates.
        coordinates = self.Get_Coordinates(variable_info, root_group)
        if not coordinates :
            #No coordinates found so we are stumped, for now.
            #Possibly a variable without geocoordinates.
            return {} 

        #Check we have a proper netCDF4 internal path                   
        self._Clean_and_Check_Path(coordinates)

        #Find the coordinates in the satpy 'database'
        coords_names = self._Identify_Geocoordinates(coordinates, variable_dB)
        if not coords_names:
            #Didn't find them.
            return {}
        
        for coordinate, file_key in coords_names['coordinates'].items():
            coordinate_info = variable_dB[coordinate]
            assert file_key == coordinate_info['file_key'], f'Wrong coordinate variable found!'
            if not Coordinate_Base.Test(variable_info['dimensions'], variable_info['shape'],
                                        coordinate_info['dimensions'], coordinate_info['shape']):
                return {}
        
        #Sat_py Won't process area unless latitude or longitude data have a 'standard_name' defined as latitude or longitude!!!!
        #Set up standard_name attribute
        coords_names.update(self._Standard_Name(variable_dB[variable_name]['file_key'], coordinates))
        return coords_names
        
               
    def __init__(self, metadata_templates):
        """Initialise Coordinates_Manager

        :param metadata_templates:  dict containing geocoordinate name search strings.
        :returns: Nothing
        :rtype: None

        """
        super().__init__()
        self.metadata_templates = {'latitude': metadata_templates['Variables']['latitude'],
                                   'longitude': metadata_templates['Variables']['longitude']}

class Dimensions_Manager(Information_Manager):
    """ Manage loading, naming, and processing of variable dimensions coordinates
    from netCDF4 file to x_array.

    :base: Information_Manager

    """
            
    def Update(self, info, root_group):
        """Find the dimension (names) and the dimenson variable datasets (netCDF paths)
        for the current variable described by its satpy 'ds_info'.

        :param info: satpy ds_info for current variable
        :param root_group: netCDF4 Dataset root node.
        :returns: a 'dimensions' dict with dimension names as keys and dimension netCDF patths as values.
        :rtype: 'dimensions' dict.

        """
        #Get dimension names from ds_info dict
        dimensions = info['dimensions']
        #Path to current variable
        nc_path     = info['file_key'].rsplit('/', 1)[0]
        dim_paths   = []
        #Shape of current variable
        shape      = root_group[info['file_key']].shape
        counter = 0
        for dim in dimensions:
            var_path = ''
            #Look for dimension variable data in netCDF4 file.
            #Assumes closest in direct hierarchy is most relevant.
            try:
                #Search upward, from the path of the variable, to find variables with dimension name
                var_path = list(NC_Finder.Find_Ancestor_Variables(dim, root_group, nc_path))
            except IndexError:
                var_path = []

            if not var_path:
                try:
                    #Search downwards from root to find variables with dimension name
                    var_path += list(NC_Finder.Find_Descendant_Variables(dim, root_group))
                except IndexError:
                    pass

            #Ensure the correct dimension has been found. 
            found_path = ''
            if var_path :
                current_dim_size = shape[counter]
                for path in var_path:
                    found_shape = root_group[path].shape
                    if len(found_shape) == 1 and current_dim_size == found_shape[0] :
                        found_path = path
                        break
            #Add the netCDF4 dimension variable path to the path list. 
            dim_paths.append(found_path)
            counter += 1
            
        try:
            #Assemble and return a 'dimensions' dictionary containing a dictionary
            #of dimension names as keys and their paths as values.
            return {'dimensions': dict(zip(dimensions, dim_paths))}
        except:
            return {}
        
    # Now, we have another problem.  Because coordinates is a dict, if there
    # are two dimensions with the same name, then only one entry is made in coordinates.
    # Superficially this passes, but crashes xarray later.
    def Load_Dimensions(self, ds_info, root_group):
        """Manage loading of a netCDF4 Dataset variable's dimension data:
        If not previously loaded, load and store dimension data, as numpy array,
        from a netCDF4 file for a particular netCDF4 variable, using variable
        information held in ds_info dict otherwise retrieve dimension data from store.
        Uniquely identify a variable's dimensions if there are duplicate names.
        Handle dimensions with names but no variable data.
        Rename (geo)coordinate dimensions according to satpy policy e.g. ('y','x').
        
        :param ds_info: ds_info augmented dict
        :param root_group: netCDF4 Dataset root node.
        :returns: dict with keys of (unique) dimension name and values of dimension data.
        :rtype: x_array compatible coordinates dict.
        """

        # Find dimension names of a netCDF4 variable and it's shape
        variable_dimensions  = root_group[ds_info['file_key']].dimensions
        variable_shape       = root_group[ds_info['file_key']].shape
        coords              = {}
        duplicate_dimensions = {}

        # Change appropriate geocoordinate dimensions to satpy y,x form.
        new_names = Rename_Manager.Dimensions(ds_info, variable_dimensions, variable_shape)
        
        for dim, size in  list(zip(variable_dimensions, variable_shape)):
            # Manage/set up catch for duplicated dimensions and catch if necessary.
            try :
                duplicate_dimensions[dim]['Count'] += 1
            except KeyError:       
                if variable_dimensions.count(dim) > 1 :
                    duplicate_dimensions[dim] = {'Count': 0}

            # Current dimension is a duplicate (already visited)?
            try:
                data = duplicate_dimensions[dim]['Data']
            except KeyError:
                # First time for this dimension.
                # info['dimensions'] contains preprocessed dimension links.
                dimension_path = ds_info['dimensions'][dim]
                # Have we already stored the dimension with its full path?
                try:
                    # Has it been stored by a previous variable encounter?
                    data = self.dimension[dimension_path]
                except KeyError:
                    # Not previously encountered.
                    #
                    # Possibly a stored dimension which does not have associated variable data.
                    # If created locally, might be of different sizes though!
                    # We need to store all the previously created sizes of the dimension.
                    try:
                        # Check for previous dims with same name
                        for data in self.dimension[dim]:
                            if len(data) == size:
                                # Found
                                break
                        if len(data) != size:
                            # Not found in dimension name list 
                            raise KeyError
                    except KeyError:
                        #O.K. look for dimension variable data.
                        try:
                            dim_var = root_group[dimension_path]
                            # It does not make sense to mask a dimension ?
                            dim_var.set_auto_mask(False)
                            # Load data
                            ##To_do: Try not loading data!!!
                            data = dim_var[:]
                            # Attach any found dimension variable to the dataset.
                            # Check for badly assembled dimension.
                            # satpy/xarray barfs if dimension not positive incremental
                            if not np.all(data[1:] > data[:-1]):
                                logger.warning(f'Malformed dimension {dim_var.name} reindexing.')
                                data = np.arange(len(data))
                            # Store data according to its netCDF4 internal path
                            self.dimension[dimension_path] = data
                        except IndexError:
                            # No variable found that is associated with dimension.
                            # Create simple index data and store under dimension name, in a list.
                            data = np.arange(size)
                            try:
                                self.dimension[dim].append(data)
                            except KeyError:
                                self.dimension[dim] = [data]
                try:
                    duplicate_dimensions[dim]['Data'] = data
                except KeyError:
                    pass
            try:
                suffix = ':' + str(duplicate_dimensions[dim]['Count'])
            except KeyError:
                suffix = ''
            coords[new_names.pop(0) + suffix] =  data
        return coords

    def Load_Coordinates(self, info, root_group):
        """If (geo)coordinate information has previously been discovered for a netCDF4 variable,
        this method manages/handles the (geo)coordinate data:
        If not previously encountered, load the (geo)coordinate data from a netCDF4 file and
        store as x_arrays, otherwise retrieve previously stored x_arrays from store.
        
        :param info: ds_info dict
        :param root_group: netCDF4 Dataset root node.
        :returns: dict with keys of coordinate names and values of coordinate data in x_array.
        :rtype: dict

        """
        coords = {}
        ##logger.debug('Load_Coordinates')
        try:
            for name in info['coordinates']:
                # Have previously discovered geocoordinate information
                try:
                    use_name = [name for name in ['latitude', 'longitude'] if name in name.lower()][0]
                except IndexError:
                    logger.info("Unexpected name in 'coordinates' Dimensions_Manager::Load_Coordinates()")
                    logger.info(f"{info['coordinates']}")
                    assert False

                path = info['coordinates'][name]
                try:
                    # Already 'visited'
                    coords[name] = self.coordinates[path]
                except:
                    # Not 'visited'
                    variable   = root_group[path]
                    # Make dimensions 'y','x' compliant
                    dimensions = Rename_Manager.Coordinates(info['geocoordinate_names'], info['geocoordinate_shape'], variable.dimensions, variable.shape)
                    # Make and store geocoordinate data x_array.
                    coords[name] = xr.DataArray(da.from_array(variable), dims=dimensions)
                    try:
                        # Uniquely identify coordinate by internal netCDF4 path.
                        self.coordinates[path] = coords[name]
                    except AttributeError:
                        # First entry into store.
                        self.coordinates = {path: coords[name]}
        except KeyError :
            # No specific geocoordinate information
            coords = {}
        return coords     
    
    def Load(self, info, root_group):
        """Load dimensions and geocoordinates/coordinates for a particulart netCDF4 variable. 

        :param info: augmented satpy ds_info dict
        :param root_group: netCDF4 Dataset root node.
        :returns: x_array compatible 'coords' dict and dimension list
        :rtype: dict, list

        """
        #Load dimensions for variable
        coords     = self.Load_Dimensions(info, root_group )
        #Extract dimension names
        dimensions = list(coords.keys())
        #Update geocoordinate data if possible/necessary.
        coords.update(self.Load_Coordinates(info, root_group ))

        return coords, dimensions

    def __init__(self):
        self.dimension     = {}

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
    
class Variable_Attribute_Manager():
    """Class to manage (uvns) netCDF4 variable attributes

    :base: None
    
    """

    @classmethod
    def coordinates(cls, coordinates):
        """Some UVNS files list variable coordinates in a space separated list in a string.
        This method breaks the coordinates string to make it satpy friendly, if possible.

        :param cls: class object parameter
        :param coordinates: (possible) coordinates string
        :returns: dict containing coordinates strings
        :rtype: dict

        """
        try:
            try:
                return {'coordinates': [l for l in coordinates.split(' ')]}
            except:
                return {'coordinates': coordinates}
        except AttributeError:
            pass
        return {}

    @classmethod
    def Update(cls, variable, metadata_templates, clean_string=False):
        """Identify, load, and, if necessary rename, netCDF4 (variable) attributes.

        :param cls: class method parameter
        :param variable: netCDF4 Dataset variable node
        :param metadata_templates: dictionary of attribute keywords/templates to find/match.
        :param clean_string: whether the string is likely to contain badly decoded features and needs cleaning.
        :returns: dict of found attributes.
        :rtype: dict
        """
        res = NetCDF4FileHandler._get_object_attrs(variable)
        res_keys = list(res.keys())
        info = {}
        #Search for specific attributes.
        for key, metadata_template in metadata_templates.items() :
            if metadata_template:
                matches = list(Match_ID.Matches(metadata_template, res_keys))
                if matches:
                    matches = matches[0][1]
                    if key != matches:
                        res[key] = res[matches]
                        del res[matches]
                    if isinstance(res[key], str) and clean_string :
                        res[key] = Match_ID.Clean_HDF5_String(res[key])
                        
        #Handle specific issues.
        for key, data in res.items():
            try:
                res.update(getattr(cls, key)(data))
            except AttributeError:
                pass
        return res
        
    @classmethod
    def Load(cls, variable, metadata_templates, key='Variable Attributes', clean_string=False, file_type='uvn' ):
        """Initialise dict containing attributes and dimension names and load other attributes from a netCDF4
        Dataset variable.. 

        :param cls: class method parameter
        :param variable: netCDF Dataset variable node.
        :param metadata_templates: dictionary of attribute keywords/templates to find/match. 
        :param key: sub group of metadata_templates for variable attribute matching
        :param clean_string: whether the string is likely to contain badly decoded features and needs cleaning.
        :param file_type: required by satpy,  set to 'uvn'
        :returns: dict of attributes
        :rtype:   dict

        """
        res = {'standard_name':variable.name, 'file_type':file_type,'dimensions':  variable.dimensions,
               'shape':variable.shape, 'size':variable.size}
        res.update(cls.Update(variable, metadata_templates[key], clean_string))
        if not res['dimensions'] and res['size'] == 1:
            if variable.dtype == str :
                res['dimensions'] = ('scalar_string')
                logging.debug(f'{variable.name}\t{variable.dtype}\t{res["dimensions"]}\t{variable[:]}')
            else:
                res['dimensions'] = ('scalar')  
        return res
        
 
class Global_Attribute_Manager():
    """ Handle/manage netCDF4 global attributes.
    
    :base: None

    """
    
    @classmethod
    def Time(cls, what_time):
        """Try and decode given time string. 

        :param cls: class object parameter.
        :param what_time: string of time. (Be wary of ambiguous dates!) 
        :returns: intitialised datetime object
        :rtype: datetime.datetime instance

        """

        #Already a datetime object
        if isinstance(what_time, d_t):
            return what_time
        
        try:
            # Proper iso time.
            return  d_t.fromisoformat(what_time)
        except:
            try:
                # Try dateutil's time interpreter.
                return dparser.parse(what_time)
            except dparser.ParserError:
                # Unrecognised connectives sometimes found.
                for component in [':', '_']:
                    time_format = "".join(['%Y%m%d', component, '%H%M%S'])
                    for suffix in ['', 'z']:
                        time_format += suffix
                        try:
                            return d_t.strptime(what_time.lower(), time_format)
                        except:
                            pass
        return None

    @classmethod
    def start_time(cls, what_time):
        """Convenience method wrapper to be called with getattr.

        :param cls:  class object parameter.
        :param what_time: string containing (start) time
        :returns: intitialised datetime object
        :rtype: datetime.datetime instance

        """
        return cls.Time(what_time)

    @classmethod
    def end_time(cls, what_time):
        """Convenience method wrapper to be called with getattr.

        :param cls:  class object parameter.
        :param what_time: string containing (end) time
        :returns: intitialised datetime object
        :rtype: datetime.datetime instance

        """
        return cls.Time(what_time)
    
    @classmethod
    def Load(cls, nc_path, metadata_templates, key='Attributes', clean_string=False):
        """Load and if necessary remap netCDF4 (global) attribuites.

        :param cls:   class object parameter.
        :param nc_path: root of netCDF4 path from which to search for.
        :param metadata_templates: dictionary of lists of names to search for.
        :param key: netCDF4 type to search for (e.g. Attributes).
        :param clean_string: attempt to tidy up bad encoding of strings e.g. hdf5.
        :returns: dict of found attributes
        :rtype: dict

        """
        res = Match_ID.Get_Objects(key, metadata_templates[key], nc_path, clean_string)
        for current_key, current_value in res.items():
            try:
                res[current_key] = getattr(cls, current_key)(current_value)
            except:
                pass
        return res


class Image_Manager():
    """Class to assemble and preprocess a variable dataset from a UVNS netcdf4 file creating an xarray data_array."""
    
    def Clean_Info(self, info, blacklist):
        """Remove (delete) entries in a dict according to a blacklist.

        :param self: instance parameter.
        :param info: dict to be checked.
        :param blacklist: list of entries to delete if found.
        :returns: a copy of the dict with listed entries removed.
        :rtype: dict

        """
        current_info = info.copy()
        for key in blacklist:
            try:
                del current_info[key]
            except KeyError:
                pass
        return current_info
        
    def Combine_Metadata(self, data, ds_info):
        """Combine xarray data_array attributes with a dict (of attributes).

        :param self: instance parameter.
        :param data: xarray data_array.
        :param ds_info: dict of attributes to combine.
        :returns: a new view of the combined attribute dicts.
        :rtype: dict

        """
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info)        
        return metadata

    def Default_Fill_Value(self, d_type):
        """Find the (netCDF4) default fill values for the specified data type.

        """

        for key, value in netCDF4.default_fillvals.items():
            if d_type == np.dtype(key):
                return value
        return None
    
    def Mask_and_Scale(self, data):
        """Mask and scale a numpy array using information from attributes or default values.

        :param data: xarray (with attributes)
        :returns: xarray with data masked with fill value.
        :rtype: xarray data_array view

        """
        
        # As numpy doesn't have integer Nan, we must assume that, if the data is integer with a fill value,
        # it must have an appropriate '_Fill_value' attribute.        
            
        fill_value = data.attrs.get('_Fill_value')
        if fill_value == None :
            fill_value = self.Default_Fill_Value(data.dtype)
                     
        # Mhhhm, this is suspect for floats - we hope no error has been introduced.
        good_mask = data != fill_value

        try:
            data = data * data.attrs.get('scale_factor') + data.attrs.get('add_offset')
        except TypeError:
            pass
        
        if np.issubdtype(data.dtype, np.integer) :
            final_fill_value = self.Default_Fill_Value(data.dtype)
        else:
            final_fill_value = np.nan
        
        if final_fill_value is np.nan :
            data.attrs.pop('_Fill_value', None)
        else:
            data.attrs['_Fill_value'] = final_fill_value
            
        try:
            data = data.where( good_mask, final_fill_value)
        except TypeError:
            pass
        
        return data   

    def Drop_Coordinates(self, data):
        """Drop any coordinates from x_array with blacklisted names.

        :param data: xarray data_array
        :returns: data_array with updated coordinates.
        :rtype: view of updated data_array.

        """
        # drop coords.
        for coord in self.drop_coordinates:
            if coord in data.coords :
                data = data.drop_vars(coord)
        return data

    def Add_Time_Dimension(self, data, ds_info):
        """Add time dimension to x_array coordinates.

        :param data: xarray
        :param time: datetime vector or scalar.
        :returns: data with updated coordinates.
        :rtype: view of updated data.

        """
        
        try:
            if 'time' in data.dims or 'time' in ds_info['dimensions']:
                return data
        except KeyError:
            pass
        
        time = ds_info['start_time']
        data = data.expand_dims('time')
        # Buuurgh. Numpy becomes scalar for one element - not liked by xarray.
        try:
            data = data.assign_coords({"time": ("time", np.array(time))})
        except:
            data = data.assign_coords({"time": ("time", np.array([time]))})
        return data

    def Prepare_Geo(self, bounds_data, name):
        """FIXME! briefly describe function

        :param bounds_data: 
        :param name: 
        :returns: 
        :rtype: 

        """
        if name not in self.prepare_geo :
            return bounds_data
    
        """
        From whomsoever wrote the tropomi reader, with thanks.
        Assume it is correct.
        Prepare lat/lon bounds for pcolormesh.
        lat/lon bounds are ordered in the following way::

        3----2
        |    |
        0----1

        Extend longitudes and latitudes with one element to support "pcolormesh":

            (X[i+1, j], Y[i+1, j])         (X[i+1, j+1], Y[i+1, j+1])
                                  +--------+
                                  | C[i,j] |
                                  +--------+
                 (X[i, j], Y[i, j])        (X[i, j+1], Y[i, j+1])

        """
        # Create the left array
        left = np.vstack([bounds_data[:, :, 0], bounds_data[-1:, :, 3]])
        # Create the right array
        right = np.vstack([bounds_data[:, -1:, 1], bounds_data[-1:, -1:, 2]])
        # Stack horizontally
        dest = np.hstack([left, right])
        # Convert to DataArray
        dask_dest = da.from_array(dest, chunks=CHUNK_SIZE)
        dest = xr.DataArray(dask_dest, dims=('y_bounds', 'x_bounds'), attrs=bounds_data.attrs)
        return dest       

    def Get(self,  ds_info, root_group):
        """Assemble an x_array data array using information provided and from
        netCDF4 file.

        :param ds_info: (satpy) information dict.
        :param root_group: netCDF4 root node
        :returns: required xarray data_array 
        :rtype: xarray data_array 

        """
        name     = ds_info['name']

        coords, dimensions = self.dimensions_manager.Load(ds_info, root_group)

        
        init_dict = self.Clean_Info(ds_info, self.drop_info)
        variable = root_group[ds_info.get('file_key', name)]
        chunks = np.full(len(dimensions), 'auto')
        #Some S5 even have dims of size 0!
        logger.debug(f'Creating x arrays.')

        try: 
            data = xr.DataArray(da.from_array(variable, chunks=chunks), dims=dimensions, coords=coords, attrs=init_dict, name=name)
        except (ValueError, ZeroDivisionError):
            #Chunking failure.
            #Don't chunk.
            logger.debug('Chunking failure, reverting to no chunking.')
            chunks = np.full(len(dimensions), -1)
            data = xr.DataArray(da.from_array(variable, chunks=chunks), dims=dimensions, coords=coords, attrs=init_dict, name=name)
            
        logger.debug(f'Created x arrays.')
        
        if ds_info['standard_name'] not in ['latitude', 'longitude']:
            data = self.Add_Time_Dimension(data, ds_info)
        else:
            # satpy doesn't like 3_d geocoordinates.
            data = data.squeeze()
            
        data = self.Mask_and_Scale(data)

        return data

    def __init__(self, dimensions_manager):
        """Initialise 'Image Manager'

        :param dimensions_manager: 
        :returns: None
        :rtype: None

        """
        self.dimensions_manager        = dimensions_manager
        #self.drop_coordinates         = ['y', 'x', 'layer', 'vertices']
        ##self.drop_info                 = ['dimensions', 'coordinates', 'geocoordinate_names',  'geocoordinate_shape']
        self.drop_info                 = ['coordinates', 'geocoordinate_names',  'geocoordinate_shape']
        self.prepare_geo               = ['assembled_lat_bounds', 'assembled_lon_bounds']

class UVNS_NetCDF4FileHandler(NetCDF4FileHandler):
    """Utility interface providing basic functionality between uvns reader and
       NetCDF4FileHandler.
    
    :base: NetCDF4_file_handler

    """

    def Find_Paths(self, nc_path):
        for key in self.file_content.keys():
            if not nc_path or nc_path in key:
                yield key

    def Variable_Paths(self, nc_path=''):
        for key in self.Find_Paths(nc_path):
            if isinstance(self.file_content[key], netCDF4._netCDF4.Variable):
               yield key

    def Shape_Paths(self, nc_path=''):
        for key in self.Find_Paths(nc_path):
            if key.endswith('shape/'):
                yield key
                

    def Shapes(self, nc_path=''):
        """Find all the array shapes for variables below the given root.

        :param root_group: netCDF4 Dataset node (group) from which to start search.
        :returns:         generates numpy shape triplets.
        :rtype:           triplet of integers

        """
        for key in self.Shape_Paths(nc_path):
            yield self.file_content[key]
        

    def __init__(self, filename, filename_info, filetype_info, cache_handle=True):
        super().__init__(filename, filename_info, filetype_info, cache_handle=True)

    
class UVNS_Base(UVNS_NetCDF4FileHandler):
    """File handler/manager for UVNS netCDF files.
    
    :base: NetCDF4_file_handler

    """
    
    def Open(self):
        """Create netcdf file handle or return existing one.

        :returns: None 
        :rtype: None

        """
        try:
            return self.root_group
        except AttributeError:
            ####logger.debug('Opening file')
            self.root_group = netCDF4.Dataset(self.filename, 'r')
        
            
    def Close(self):
        """Close netcdf4 file and delete handle.

        :returns: None 
        :rtype: None

        """
        ####logger.debug('Closing file')
        try:
            self.root_group.close()
            del self.root_group
        except RuntimeError:
            del self.root_group
        except (NameError, AttributeError):
            pass
             
    def Type(self, file_structure):
        """Distinguish  between hdf5 and netcdf file types.

        :param file_structure: dict holding file structure information.
        :returns: file type.
        :rtype: string

        """
        ext = file_structure.lower()
        if 'h' in ext and '5' in ext:
            return 'hdf5'
        return 'netCDF'
       
    def _Dynamic_Datasets(self):
        """Generates information about loadable datasets:
        Automatically determine datasets in the current file; uniquely identifying them
        and their coordinates; and loading attributes/metadata.

        :returns: generates fully initialised satpy ds_info dict for each available dataset
        :rtype: satpy ds_info dict

        """
        ##logger.debug("Available_datasets begin...")
        root_group = self.Open()
        if not root_group.variables and not root_group.groups :
            self.Close()
            logger.error(f"File exists but is empty! {self.filename}")
            raise EmptyFileError("File exists but is empty!")
        
        is_hDF = self.file_format == 'hdf5'
        geo_coordinates = Geographical_Coordinates(self.metadata_templates, is_hDF)
        geo_coordinates.Dimensions(root_group)

        global_attributes         = Global_Attribute_Manager.Load(root_group, self.metadata_templates, clean_string=is_hDF)
        global_attributes['file_structure'] = self.file_format
        global_attributes['filename']    = self.filename
        self.file_info.update(global_attributes)
        file_type = self.filetype_info['file_type']
        information_manager       = Information_Manager()

        unique_filenames = Name_Manager.Unique_Names(list(self.Variable_Paths()))  
        valid_info = {}
        for variable in NC_Finder.Walk_Variables(root_group) :
            ###logger.debug(f"{variable}")
            
            current_variable = root_group[variable]            
            # Avoid the whole array of strings debate ...
            # Fix this later.
            if current_variable.dtype == str and is_hDF:
                continue
            
            info = self.file_info.copy()
            info.update(Variable_Attribute_Manager.Load(current_variable, self.metadata_templates, clean_string=is_hDF, file_type=file_type))
            info.update(global_attributes)
            info['name'] = unique_filenames[variable]
            info['file_key'] = variable
            
            
            valid = geo_coordinates.Valid(current_variable)
                
            if valid:
                info.update(valid)

            if valid or info['dimensions']:
                valid_info[info['name']] = info
            else:
                logger.debug(f'!!! INVALID: {info}')
                assert False


        coordinates_manager = Coordinates_Manager(self.metadata_templates)
        for variable_name in valid_info.keys():
            valid_info[variable_name].update(coordinates_manager.Update( variable_name, valid_info, root_group))            
            valid_info[variable_name].update(self.dimensions_manager.Update( valid_info[variable_name], root_group))
            yield True, valid_info[variable_name]

    def _Existing_Datasets(self, configured_datasets=None):
        """Add information from existing datasets."""
        for is_avail, ds_info in (configured_datasets or []):
            yield is_avail, ds_info

    def Load_Time(self, filename_info):
        """ Load start, end, and creation time from global attributes, if they exist.
        
        :param filename_info: satpy filename information
        :returns: view of file information dict updated with time information if found.
        :rtype: view of dict.

        """
        # Find start, end and creation times if in global attributes.
        for current_time in ['start_time', 'end_time', 'creation_time'] :
            try:
                #filename_info[current_time] = Global_Attribute_Manager.Time(Global_Attribute_Manager, filename_info[current_time])
                filename_info[current_time] = Global_Attribute_Manager.Time(filename_info[current_time])
            except KeyError:
                pass
        return filename_info


    def __init__(self, filename, filename_info, filetype_info, cache_handle=True, band_identifier=['_BAND']):
        """Primary initialisation of the reader.  Includes setting up name requirements and mappings.
        
        :param filename: path including name of file.
        :param filename_info: information from filename breakdown.
        :param filetype_info: file type (e.g. 'uvn')
        :param cache_handle: whether to keep file handle open.
        :param band_identifier: string to append to variable name if more than one bands or channels are identified.
        :returns: None
        :rtype: None

        """
        
        if filename :
            # Initialise the reader base.
            ####logger.debug(f'File handle: {self.file_handle}')
            ####logger.debug(f'File name: {filename}')
            ####logger.debug(f'File name Info: {filename_info}')
            ####logger.debug(f'File type info: {filetype_info}')
            super().__init__(filename, filename_info, filetype_info, cache_handle=cache_handle)
            self.Open()
        self.engine = None

        self.file_info = self.Load_Time(filename_info)

        # Variable/parameter mapping and search templates in order of search. 
        group_attributes   = {'start_time'           : ['time*coverage*start', 'sensing*start*time*', 'start*time',  'start'],
                             'end_time'             : ['time*coverage*end',   'sensing*end*time*',   'end*time',   'end'],
                             'yx'                   : [['y', 'x'], ['scanline', 'ground*pixel']],
                             'product'              : ['shortproductname', 'productname', '*product*name*', 'type', 'title'],
                             'platform_shortname'   : ['satellite*id', 'spacecraft'],
                             'sensor'               : ['instrument*id', 'instrument', 'data_source'],
                             'title'                : ['title']}
        
        # See: https://cfconventions.org/Data/cf-documents/requirements-recommendations/conformance-1.8.html
        # See: http://cfconventions.org/cf-conventions/cf-conventions.html
        # WARNING: LOWER CASE values!
        variable_attributes = {'_Fill_value'   : ['*fill*value'],
                              'units'        : ['unit*'],        
                              'standard_name': [],        
                              'long_name'    : [],        
                              'reference'    : [],        
                              'coordinates'  : [],        
                              'bounds'       : [],        
                              'missing_value': [],        
                              'valid_range'  : ['valid*range'],        
                              'valid_min'    : ['valuerangemin', 'valid*min'],        
                              'valid_max'    : ['valuerangemax', 'valid*max'],        
                              'actual_range' : [],        
                              'scale_factor' : [],        
                              'add_offset'   : [],        
                              'flag_values'  : [],        
                              'flag_meanings': [],        
                              'flag_masks'   : [],        
                              'title'        : ['title'],        
                              'axis'         : []}
        
        #coordinates attribute is a string whose value is a blank separated list of variable names. All specified variable names must exist in the file.
        #All horizontal coordinate variables (in the Unidata sense) should have an axis attribute.
        #The type of the bounds attribute is a string whose value is a single variable name.
        group_names         = {'Band' : ['*band*', '*channel*']}
 
        variables          = {'latitude'             : ['latitude*cent*',    'latitude',    '*latitude*'],
                              'longitude'            : ['longitude*cent*',   'longitude',   '*longitude*'],
                              'latitude_boundaries'  : ['latitude*corner*',  'latitude*bound*'],
                              'longitude_boundaries' : ['longitude*corner*', 'longitude*bound*']}
        
        #dimensions         = {'x'                    : ['x', 'ground*pixel', 'pixel'],
        #                      'y'                    : ['y', 'scanline', 'time']}
        dimensions         = {'x'                    : ['x', 'ground*pixel', 'pixel'],
                              'y'                    : ['y', 'scanline', 'time']}

        self.metadata_templates = {'Attributes'         : group_attributes,
                                  'Variable Attributes': variable_attributes,
                                  'Groups'             : group_names,
                                  'Variables'          : variables,
                                  'Dimensions'         : dimensions,
                                  'Bands'              : ['uvvis','nir']}
        
        try:
            file_struct = filename_info['file_structure']
        except:
            file_struct = ''

        # Actual file type.
        self.file_format  = self.Type(file_struct)
        self.dimensions_manager = Dimensions_Manager()

class UVNS_Reader(UVNS_Base):
    '''Satpy interface to UVNS reader. Provides obligatory methods.
       :base: UVNS_Base
    
    '''

    def available_datasets(self, configured_datasets=None):
        """Satpy interface (required): gives information about available datasets.

        :param configured_datasets: datasets already identified.
        :returns: generates triplets of (True, variable information dict) for available datasets.
        :rtype:   triplet of boolean, dict

        """
        existing    = self._Existing_Datasets(configured_datasets=configured_datasets)
        dynamic     = self._Dynamic_Datasets()

        for dataset_available, dataset_info in itertools.chain(existing, dynamic):
            yield dataset_available, dataset_info
        

    def get_dataset(self, ds_id, ds_info):
        """Satpy interface (required): load previously identified satpy dataset.

        :param ds_id:   satpy dataset identifier
        :param ds_info: satpy datset information for ds_id
        :returns:       loaded data for ds_id
        :rtype:         x_array

        """
        logger.debug("Getting data for: %s", ds_id['name'])
        root_group    = self.Open()
        image_manager = Image_Manager(self.dimensions_manager)
        z = image_manager.Get(ds_info, root_group)
        logger.debug(f"Finished getting data for: {ds_id['name']} {z.shape}")
        logger.debug('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh')
        #import ipdb; ipdb.set_trace()
        return z

    #Warning: on testing, destructor only called after any wrapper programs exited:
    # not when satpy Scene goes out of scope! becomes netCDF4 error (hangs).
    # Therefore should close/delete manually if creating succesive scenes.
    def __del__(self):
        """Tidy up on delete.

        """
        self.Close()
        ##logger.debug('Destructor called.')

    def __init__(self, filename, filename_info, filetype_info):
        """Initialises base class.

        :param filename:      current filename string
        :param filename_info: information decoded from filename
        :param filetype_info: satpy filetype information.

        """
        #Do not cache.
        super().__init__(filename, filename_info, filetype_info, False)

