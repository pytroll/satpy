#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2012, 2013, 2015.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

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

"""Plugin for reading PPS's cloud products hdf files, in the fileformat used
in PPS v2012, and before.
"""
import ConfigParser
from ConfigParser import NoOptionError

from datetime import datetime, timedelta
import os.path

import mpop.channel
from mpop import CONFIG_PATH
from mpop.utils import get_logger
import numpy as np

import h5py

LOG = get_logger('satin/nwcsaf_pps')


class InfoObject(object):

    """Simple data and info container.
    """

    def __init__(self):
        self.info = {}
        self.data = None


def pack_signed(data, data_type):
    bits = np.iinfo(data_type).bits
    scale_factor = (data.max() - data.min()) / (2**bits - 2)
    add_offset = (data.max() - data.min()) / 2
    no_data = - 2**(bits - 1)
    pack = ((data - add_offset) / scale_factor).astype(data_type)
    return pack, scale_factor, add_offset, no_data


class NwcSafPpsChannel(mpop.channel.GenericChannel):

    def __init__(self, filename=None):
        mpop.channel.GenericChannel.__init__(self)
        self._md = {}
        self._projectables = []
        self._keys = []
        self._refs = {}
        self.shape = None
        if filename:
            self.read(filename)

    def read(self, filename, load_lonlat=True):
        """Read product in hdf format from *filename*
        """
        LOG.debug("Filename: %s" % filename)

        is_temp = False
        if not h5py.is_hdf5(filename):
            # Try see if it is bzipped:
            import bz2
            bz2file = bz2.BZ2File(filename)
            import tempfile
            tmpfilename = tempfile.mktemp()
            try:
                ofpt = open(tmpfilename, 'wb')
                ofpt.write(bz2file.read())
                ofpt.close()
                is_temp = True
            except IOError:
                import traceback
                traceback.print_exc()
                raise IOError("Failed to read the file %s" % filename)

            filename = tmpfilename

        if not h5py.is_hdf5(filename):
            if is_temp:
                os.remove(filename)
            raise IOError("File is not a hdf5 file!" % filename)

        h5f = h5py.File(filename, "r")

        # Read the global attributes

        self._md = dict(h5f.attrs)
        self._md["satellite"] = h5f.attrs['satellite_id']
        self._md["orbit"] = h5f.attrs['orbit_number']
        self._md["time_slot"] = (timedelta(seconds=long(h5f.attrs['sec_1970']))
                                 + datetime(1970, 1, 1, 0, 0))

        # Read the data and attributes
        #   This covers only one level of data. This could be made recursive.
        for key, dataset in h5f.iteritems():
            setattr(self, key, InfoObject())
            getattr(self, key).info = dict(dataset.attrs)
            for skey, value in dataset.attrs.iteritems():
                if isinstance(value, h5py.h5r.Reference):
                    self._refs[(key, skey)] = h5f[value].name.split("/")[1]

            if type(dataset.id) is h5py.h5g.GroupID:
                LOG.warning("Format reader does not support groups")
                continue

            try:
                getattr(self, key).data = dataset[:]
                is_palette = (dataset.attrs.get("CLASS", None) == "PALETTE")
                if(len(dataset.shape) > 1 and
                   not is_palette and
                   key not in ["lon", "lat",
                               "row_indices", "column_indices"]):
                    self._projectables.append(key)
                    if self.shape is None:
                        self.shape = dataset.shape
                    elif self.shape != dataset.shape:
                        raise ValueError("Different variable shapes !")
                else:
                    self._keys.append(key)
            except TypeError:
                setattr(self, key, np.dtype(dataset))
                self._keys.append(key)

        h5f.close()

        if is_temp:
            os.remove(filename)

        if not load_lonlat:
            return

        # Setup geolocation
        # We need a no-data mask from one of the projectables to
        # mask out bow-tie deletion pixels from the geolocation array
        # So far only relevant for VIIRS.
        # Preferably the lon-lat data in the PPS VIIRS geolocation
        # file should already be masked.
        # The no-data values in the products are not only where geo-location is absent
        # Only the Cloud Type can be used as a proxy so far.
        # Adam Dybbroe, 2012-08-31
        nodata_mask = False  # np.ma.masked_equal(np.ones(self.shape), 0).mask
        for key in self._projectables:
            projectable = getattr(self,  key)
            if key in ['cloudtype']:
                nodata_array = np.ma.array(projectable.data)
                nodata_mask = np.ma.masked_equal(nodata_array, 0).mask
                break

        try:
            from pyresample import geometry
        except ImportError:
            return

        tiepoint_grid = False
        if hasattr(self, "row_indices") and hasattr(self, "column_indices"):
            column_indices = self.column_indices.data
            row_indices = self.row_indices.data
            tiepoint_grid = True

        interpolate = False
        if hasattr(self, "lon") and hasattr(self, "lat"):
            if 'intercept' in self.lon.info:
                offset_lon = self.lon.info["intercept"]
            elif 'offset' in self.lon.info:
                offset_lon = self.lon.info["offset"]
            if 'gain' in self.lon.info:
                gain_lon = self.lon.info["gain"]
            lons = self.lon.data * gain_lon + offset_lon

            if 'intercept' in self.lat.info:
                offset_lat = self.lat.info["intercept"]
            elif 'offset' in self.lat.info:
                offset_lat = self.lat.info["offset"]
            if 'gain' in self.lat.info:
                gain_lat = self.lat.info["gain"]
            lats = self.lat.data * gain_lat + offset_lat

            if lons.shape != self.shape or lats.shape != self.shape:
                # Data on tiepoint grid:
                interpolate = True
                if not tiepoint_grid:
                    errmsg = ("Interpolation needed but insufficient" +
                              "information on the tiepoint grid")
                    raise IOError(errmsg)
            else:
                # Geolocation available on the full grid:
                # We neeed to mask out nodata (VIIRS Bow-tie deletion...)
                # We do it for all instruments, checking only against the
                # nodata
                lons = np.ma.masked_array(lons, nodata_mask)
                lats = np.ma.masked_array(lats, nodata_mask)

                self.area = geometry.SwathDefinition(lons=lons, lats=lats)

        elif hasattr(self, "region") and self.region.data["area_extent"].any():
            region = self.region.data
            proj_dict = dict([elt.split('=')
                              for elt in region["pcs_def"].split(',')])
            self.area = geometry.AreaDefinition(region["id"],
                                                region["name"],
                                                region["proj_id"],
                                                proj_dict,
                                                region["xsize"],
                                                region["ysize"],
                                                region["area_extent"])

        if interpolate:
            from geotiepoints import SatelliteInterpolator

            cols_full = np.arange(self.shape[1])
            rows_full = np.arange(self.shape[0])

            satint = SatelliteInterpolator((lons, lats),
                                           (row_indices,
                                            column_indices),
                                           (rows_full, cols_full))
            #satint.fill_borders("y", "x")
            lons, lats = satint.interpolate()

            self.area = geometry.SwathDefinition(lons=lons, lats=lats)

    def project(self, coverage):
        """Project what can be projected in the product.
        """

        import copy
        res = copy.copy(self)

        # Project the data
        for var in self._projectables:
            LOG.info("Projecting " + str(var))
            res.__dict__[var] = copy.copy(self.__dict__[var])
            res.__dict__[var].data = coverage.project_array(
                self.__dict__[var].data)

        # Take care of geolocation

        res.region = copy.copy(self.region)

        region = copy.copy(res.region.data)
        area = coverage.out_area
        try:
            # It's an area
            region["area_extent"] = np.array(area.area_extent)
            region["xsize"] = area.x_size
            region["ysize"] = area.y_size
            region["xscale"] = area.pixel_size_x
            region["yscale"] = area.pixel_size_y
            region["lon_0"] = area.proj_dict.get("lon_0", 0)
            region["lat_0"] = area.proj_dict.get("lat_0", 0)
            region["lat_ts"] = area.proj_dict.get("lat_ts", 0)
            region["name"] = area.name
            region["id"] = area.area_id
            region["pcs_id"] = area.proj_id
            pcs_def = ",".join([key + "=" + val
                                for key, val in area.proj_dict.iteritems()])
            region["pcs_def"] = pcs_def
            res.region.data = region

            # If switching to area representation, try removing lon and lat
            try:
                delattr(res, "lon")
                res._keys.remove("lon")
                delattr(res, "lat")
                res._keys.remove("lat")
            except AttributeError:
                pass

        except AttributeError:
            # It's a swath
            lons, scale_factor, add_offset, no_data = \
                pack_signed(area.lons[:], np.int16)
            res.lon = InfoObject()
            res.lon.data = lons
            res.lon.info["description"] = "geographic longitude (deg)"
            res.lon.info["intercept"] = add_offset
            res.lon.info["gain"] = scale_factor
            res.lon.info["no_data_value"] = no_data
            if "lon" not in res._keys:
                res._keys.append("lon")

            lats, scale_factor, add_offset, no_data = \
                pack_signed(area.lats[:], np.int16)
            res.lat = InfoObject()
            res.lat.data = lats
            res.lat.info["description"] = "geographic latitude (deg)"
            res.lat.info["intercept"] = add_offset
            res.lat.info["gain"] = scale_factor
            res.lat.info["no_data_value"] = no_data
            if "lat" not in res._keys:
                res._keys.append("lat")
            # Remove region parameters if switching from area
            region["area_extent"] = np.zeros(4)
            region["xsize"] = 0
            region["ysize"] = 0
            region["xscale"] = 0
            region["yscale"] = 0
            region["lon_0"] = 0
            region["lat_0"] = 0
            region["lat_ts"] = 0
            region["name"] = ""
            region["id"] = ""
            region["pcs_id"] = ""
            region["pcs_def"] = ""
            res.region.data = region
        return res

    def write(self, filename, **kwargs):
        """Write product in hdf format to *filename*
        """

        LOG.debug("Writing to " + filename)
        h5f = h5py.File(filename, "w")

        for dataset in self._projectables:
            dset = h5f.create_dataset(dataset, data=getattr(self, dataset).data,
                                      compression='gzip', compression_opts=6)
            for key, value in getattr(self, dataset).info.iteritems():
                dset.attrs[key] = value

        for thing in self._keys:
            try:
                dset = h5f.create_dataset(thing, data=getattr(self, thing).data,
                                          compression='gzip', compression_opts=6)
                for key, value in getattr(self, thing).info.iteritems():
                    dset.attrs[key] = value
            except AttributeError:
                h5f[thing] = getattr(self, thing)

        for key, value in self._md.iteritems():
            if key in ["time_slot", "satellite"]:
                continue
            h5f.attrs[key] = value

        for (key, skey), value in self._refs.iteritems():
            h5f[key].attrs[skey] = h5f[value].ref

        h5f.close()

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return len(self._projectables) > 0


class CloudType(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "CloudType"


class CloudTopTemperatureHeight(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "CTTH"


class CloudMask(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "CMa"


class PrecipitationClouds(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "PC"


class CloudPhysicalProperties(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "CPP"


def load(scene, geofilename=None, **kwargs):
    del kwargs

    import glob

    lonlat_is_loaded = False

    products = []
    if "CTTH" in scene.channels_to_load:
        products.append("ctth")
    if "CloudType" in scene.channels_to_load:
        products.append("cloudtype")
    if "CMa" in scene.channels_to_load:
        products.append("cloudmask")
    if "PC" in scene.channels_to_load:
        products.append("precipclouds")
    if "CPP" in scene.channels_to_load:
        products.append("cpp")

    if len(products) == 0:
        return

    try:
        area_name = scene.area_id or scene.area.area_id
    except AttributeError:
        area_name = "satproj_?????_?????"

    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, scene.fullname + ".cfg"))
    directory = conf.get(scene.instrument_name + "-level3", "dir")
    try:
        geodir = conf.get(scene.instrument_name + "-level3", "geodir")
    except NoOptionError:
        LOG.warning("No option 'geodir' in level3 section")
        geodir = None

    filename = conf.get(scene.instrument_name + "-level3", "filename",
                        raw=True)
    pathname_tmpl = os.path.join(directory, filename)

    if not geofilename and geodir:
        # Load geo file from config file:
        try:
            if not scene.orbit:
                orbit = ""
            else:
                orbit = scene.orbit
            geoname_tmpl = conf.get(scene.instrument_name + "-level3",
                                    "geofilename", raw=True)
            filename_tmpl = (scene.time_slot.strftime(geoname_tmpl)
                             % {"orbit": orbit.zfill(5) or "*",
                                "area": area_name,
                                "satellite": scene.satname + scene.number})

            file_list = glob.glob(os.path.join(geodir, filename_tmpl))
            if len(file_list) > 1:
                LOG.warning("More than 1 file matching for geoloaction: "
                            + str(file_list))
            elif len(file_list) == 0:
                LOG.warning("No geolocation file matching!: " + filename_tmpl)
            else:
                geofilename = file_list[0]
        except NoOptionError:
            geofilename = None

    classes = {"ctth": CloudTopTemperatureHeight,
               "cloudtype": CloudType,
               "cloudmask": CloudMask,
               "precipclouds": PrecipitationClouds,
               "cpp": CloudPhysicalProperties
               }

    nodata_mask = False

    chn = None
    for product in products:
        LOG.debug("Loading " + product)
        if not scene.orbit:
            orbit = ""
        else:
            orbit = scene.orbit
        filename_tmpl = (scene.time_slot.strftime(pathname_tmpl)
                         % {"orbit": orbit.zfill(5) or "*",
                            "area": area_name,
                            "satellite": scene.satname + scene.number,
                            "product": product})

        file_list = glob.glob(filename_tmpl)
        if len(file_list) > 1:
            LOG.warning("More than 1 file matching for " + product + "! "
                        + str(file_list))
            continue
        elif len(file_list) == 0:
            LOG.warning("No " + product + " matching!: " + filename_tmpl)
            continue
        else:
            filename = file_list[0]

            chn = classes[product]()
            chn.read(filename, lonlat_is_loaded == False)
            scene.channels.append(chn)

        # Setup geolocation
        # We need a no-data mask from one of the projectables to
        # mask out bow-tie deletion pixels from the geolocation array
        # So far only relevant for VIIRS.
        # Preferably the lon-lat data in the PPS VIIRS geolocation
        # file should already be masked.
        # The no-data values in the products are not only where geo-location is absent
        # Only the Cloud Type can be used as a proxy so far.
        # Adam Dybbroe, 2012-08-31
        if hasattr(chn, '_projectables'):
            for key in chn._projectables:
                projectable = getattr(chn,  key)
                if key in ['cloudtype']:
                    nodata_array = np.ma.array(projectable.data)
                    nodata_mask = np.ma.masked_equal(nodata_array, 0).mask
                    break
        else:
            LOG.warning("Channel has no '_projectables' member." +
                        " No nodata-mask set...")

    if chn is None:
        return

    # Is this safe!? AD 2012-08-25
    shape = chn.shape

    interpolate = False
    if geofilename:
        geodict = get_lonlat(geofilename)
        lons, lats = geodict['lon'], geodict['lat']
        if lons.shape != shape or lats.shape != shape:
            interpolate = True
            row_indices = geodict['row_indices']
            column_indices = geodict['column_indices']

        lonlat_is_loaded = True
    else:
        LOG.warning("No Geo file specified: " +
                    "Geolocation will be loaded from product")

    if lonlat_is_loaded:
        if interpolate:
            from geotiepoints import SatelliteInterpolator

            cols_full = np.arange(shape[1])
            rows_full = np.arange(shape[0])

            satint = SatelliteInterpolator((lons, lats),
                                           (row_indices,
                                            column_indices),
                                           (rows_full, cols_full))
            #satint.fill_borders("y", "x")
            lons, lats = satint.interpolate()

        try:
            from pyresample import geometry
            lons = np.ma.masked_array(lons, nodata_mask)
            lats = np.ma.masked_array(lats, nodata_mask)
            scene.area = geometry.SwathDefinition(lons=lons,
                                                  lats=lats)
        except ImportError:
            scene.area = None
            scene.lat = lats
            scene.lon = lons

    LOG.info("Loading PPS parameters done.")


def get_lonlat(filename):
    """Read lon,lat from hdf5 file"""
    import h5py
    LOG.debug("Geo File = " + filename)

    h5f = h5py.File(filename, 'r')

    # We neeed to mask out nodata (VIIRS Bow-tie deletion...)
    # We do it for all instruments, checking only against the nodata
    nodata = h5f['where']['lon']['what'].attrs['nodata']
    gain = h5f['where']['lon']['what'].attrs['gain']
    offset = h5f['where']['lon']['what'].attrs['offset']

    longitudes = np.ma.array(h5f['where']['lon']['data'].value)
    lons = np.ma.masked_equal(longitudes, nodata) * gain + offset
    latitudes = np.ma.array(h5f['where']['lat']['data'].value)
    lats = np.ma.masked_equal(latitudes, nodata) * gain + offset

    col_indices = None
    row_indices = None
    if "column_indices" in h5f["where"].keys():
        col_indices = h5f['/where/column_indices'].value
    if "row_indices" in h5f["where"].keys():
        row_indices = h5f['/where/row_indices'].value

    h5f.close()
    return {'lon': lons,
            'lat': lats,
            'col_indices': col_indices, 'row_indices': row_indices}
    # return lons, lats
