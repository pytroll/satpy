#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <adam.dybbroe@smhi.se>
#   Martin.Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""PPS netcdf cloud product reader
"""

import os.path
from ConfigParser import ConfigParser
from ConfigParser import NoOptionError

from datetime import datetime
import glob
import numpy as np

import mpop.channel
from mpop import CONFIG_PATH
from mpop.plugin_base import Reader

import logging
LOG = logging.getLogger(__name__)

NEW_PRODNAMES = {'cloudtype': 'CT',
                 'cloudmask': 'CMA',
                 'precipclouds': 'PC',
                 'cpp': 'CPP',
                 'ctth': 'CTTH'}

PPS_DATASETS = ['Cloud Type',
                'Multilayer Cloud Detection',
                "SAFNWC PPS PC likelihood of intense precipitation",
                "SAFNWC PPS PC likelihood of moderate precipitation",
                "SAFNWC PPS PC likelihood of light precipitation",
                ]


class InfoObject(object):

    """Simple data and info container.
    """

    def __init__(self):
        self.info = {}
        self.data = None


class NwcSafPpsChannel(mpop.channel.GenericChannel):

    def __init__(self, filename=None):
        mpop.channel.GenericChannel.__init__(self)
        self.mda = {}
        self._projectables = []
        self._keys = []
        self._refs = {}
        self.shape = None
        if filename:
            self.read(filename)

    def read(self, filename, load_lonlat=True):
        """Read the PPS v2014 formatet data"""
        LOG.debug("New netCDF CF file format!")

        import h5py

        h5f = h5py.File(filename, 'r')
        self.mda.update(h5f.attrs.items())

        self.mda["satellite"] = h5f.attrs['platform']
        self.mda["orbit"] = h5f.attrs['orbit_number']
        try:
            self.mda["time_slot"] = datetime.strptime(h5f.attrs['time_coverage_start'][:-2],
                                                      "%Y%m%dT%H%M%S")
        except AttributeError:
            LOG.debug("No time information in product file!")

        variables = {}

        for key, item in h5f.items():
            if item.attrs.get("CLASS") != 'DIMENSION_SCALE':
                variables[key] = item

        # processed variables
        processed = set()

        non_processed = set(variables.keys()) - processed

        for var_name in non_processed:
            if var_name in ['lon', 'lat']:
                continue

            var = variables[var_name]
            if ("standard_name" not in var.attrs.keys() and
                    "long_name" not in var.attrs.keys() and
                    var.attrs.get("CLASS") != "PALETTE"):
                LOG.info("Delayed processing of " + var_name)
                continue

            # Don't know how to unambiguously decide if the array is really a
            # data array or a palette or something else!
            # FIXME!
            if "standard_name" in var.attrs.keys():
                self._projectables.append(var_name)
            elif "long_name" in var.attrs.keys():
                dset_found = False
                for item in PPS_DATASETS:
                    if var.attrs['long_name'].find(item) >= 0:
                        self._projectables.append(var_name)
                        dset_found = True
                        break
                if not dset_found:
                    self.mda[var_name] = var[:]
                    # try:
                    #     self.mda[var_name] = var[:].filled(0)
                    # except AttributeError:
                    #     self.mda[var_name] = var[:]
                    continue
            elif var.attrs.get("CLASS") == "PALETTE":
                self.mda[var_name] = var[:]
                continue

            setattr(self, var_name, InfoObject())
            for key, item in var.attrs.items():
                if key != "DIMENSION_LIST":
                    getattr(self, var_name).info[key] = item

            data = var[:]
            if 'valid_range' in var.attrs.keys():
                data = np.ma.masked_outside(data, *var.attrs['valid_range'])
            elif '_FillValue' in var.attrs.keys():
                data = np.ma.masked_where(data, var.attrs['_FillValue'])
            dataset = (data * var.attrs.get("scale_factor", 1)
                       + var.attrs.get("add_offset", 0))

            getattr(self, var_name).data = dataset

            LOG.debug("long_name: " + str(var.attrs['long_name']))
            LOG.debug("Var=" + str(var_name) + " shape=" + str(dataset.shape))

            if self.shape is None:
                self.shape = dataset.shape
            elif self.shape != dataset.shape:
                LOG.debug("Shape=" + str(dataset.shape) +
                          " Not the same shape as previous field...")
                # raise ValueError("Different variable shapes !")

            # dims = var.dimensions
            # dim = dims[0]

            processed |= set([var_name])

        non_processed = set(variables.keys()) - processed
        if len(non_processed) > 0:
            LOG.warning(
                "Remaining non-processed variables: " + str(non_processed))

        # Get lon,lat:
        # from pyresample import geometry
        # area = geometry.SwathDefinition(lons=lon, lats=lat)

        # close the h5 file.
        h5f.close()
        del h5f

        return

    def project(self, coverage):
        """Project the data"""
        LOG.debug("Projecting channel %s..." % (self.name))
        import copy
        res = copy.copy(self)

        # Project the data
        for var in self._projectables:
            LOG.info("Projecting " + str(var))
            res.__dict__[var] = copy.copy(self.__dict__[var])
            res.__dict__[var].data = coverage.project_array(
                self.__dict__[var].data)

        res.name = self.name
        res.resolution = self.resolution
        res.filled = True
        res.area = coverage.out_area
        return res

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return True
        # return len(self._projectables) > 0

    def save(self, filename, old=True, **kwargs):
        del kwargs
        if old:
            from nwcsaf_formats.ppsv2014_to_oldformat import write_product
            write_product(self, filename)

        else:
            raise NotImplementedError("Can't save to new pps format yet.")


class PPSReader(Reader):

    pformat = "h5_pps_l2"

    def load(self, satscene, *args, **kwargs):
        """Read data from file and load it into *satscene*.
        """
        lonlat_is_loaded = False

        geofilename = kwargs.get('geofilename')
        prodfilename = kwargs.get('filename')

        products = []
        if "CTTH" in satscene.channels_to_load:
            products.append("ctth")
        if "CT" in satscene.channels_to_load:
            products.append("cloudtype")
        if "CMA" in satscene.channels_to_load:
            products.append("cloudmask")
        if "PC" in satscene.channels_to_load:
            products.append("precipclouds")
        if "CPP" in satscene.channels_to_load:
            products.append("cpp")

        if len(products) == 0:
            return

        try:
            area_name = satscene.area_id or satscene.area.area_id
        except AttributeError:
            area_name = "satproj_?????_?????"

        # Looking for geolocation file

        conf = ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))

        try:
            geodir = conf.get(satscene.instrument_name + "-level3",
                              "cloud_product_geodir",
                              vars=os.environ)
        except NoOptionError:
            LOG.warning("No option 'geodir' in level3 section")
            geodir = None

        if not geofilename and geodir:
            # Load geo file from config file:
            try:
                if not satscene.orbit:
                    orbit = ""
                else:
                    orbit = satscene.orbit
                geoname_tmpl = conf.get(satscene.instrument_name + "-level3",
                                        "cloud_product_geofilename",
                                        raw=True,
                                        vars=os.environ)
                filename_tmpl = (satscene.time_slot.strftime(geoname_tmpl)
                                 % {"orbit": str(orbit).zfill(5) or "*",
                                    "area": area_name,
                                    "satellite": satscene.satname + satscene.number})

                file_list = glob.glob(os.path.join(geodir, filename_tmpl))
                if len(file_list) > 1:
                    LOG.warning("More than 1 file matching for geoloaction: "
                                + str(file_list))
                elif len(file_list) == 0:
                    LOG.warning(
                        "No geolocation file matching!: "
                        + os.path.join(geodir, filename_tmpl))
                else:
                    geofilename = file_list[0]
            except NoOptionError:
                geofilename = None

        # Reading the products

        classes = {"ctth": CloudTopTemperatureHeight,
                   "cloudtype": CloudType,
                   "cloudmask": CloudMask,
                   "precipclouds": PrecipitationClouds,
                   "cpp": CloudPhysicalProperties
                   }

        nodata_mask = False

        area = None
        lons = None
        lats = None
        chn = None
        shape = None
        read_external_geo = {}
        for product in products:
            LOG.debug("Loading " + product)

            if isinstance(prodfilename, (list, tuple, set)):
                for fname in prodfilename:
                    kwargs['filename'] = fname
                    self.load(satscene, *args, **kwargs)
                return
            elif (prodfilename and
                  os.path.basename(prodfilename).startswith('S_NWC')):
                if os.path.basename(prodfilename).split("_")[2] == NEW_PRODNAMES[product]:
                    filename = prodfilename
                else:
                    continue
            else:
                filename = conf.get(satscene.instrument_name + "-level3",
                                    "cloud_product_filename",
                                    raw=True,
                                    vars=os.environ)
                directory = conf.get(satscene.instrument_name + "-level3",
                                     "cloud_product_dir",
                                     vars=os.environ)
                pathname_tmpl = os.path.join(directory, filename)
                LOG.debug("Path = " + str(pathname_tmpl))

                if not satscene.orbit:
                    orbit = ""
                else:
                    orbit = satscene.orbit

                filename_tmpl = (satscene.time_slot.strftime(pathname_tmpl)
                                 % {"orbit": str(orbit).zfill(5) or "*",
                                    "area": area_name,
                                    "satellite": satscene.satname + satscene.number,
                                    "product": product})

                file_list = glob.glob(filename_tmpl)
                if len(file_list) == 0:
                    product_name = NEW_PRODNAMES.get(product, product)
                    LOG.info("No " + str(product) +
                             " product in old format matching")
                    filename_tmpl = (satscene.time_slot.strftime(pathname_tmpl)
                                     % {"orbit": str(orbit).zfill(5) or "*",
                                        "area": area_name,
                                        "satellite": satscene.satname + satscene.number,
                                        "product": product_name})

                    file_list = glob.glob(filename_tmpl)

                if len(file_list) > 1:
                    LOG.warning("More than 1 file matching for " + product + "! "
                                + str(file_list))
                    continue
                elif len(file_list) == 0:
                    LOG.warning(
                        "No " + product + " matching!: " + filename_tmpl)
                    continue
                else:
                    filename = file_list[0]

            chn = classes[product]()
            chn.read(filename, lonlat_is_loaded == False)
            satscene.channels.append(chn)
            # Check if geolocation is loaded:
            if not chn.area:
                read_external_geo[product] = chn
                shape = chn.shape

        # Check if some 'channel'/product needs geolocation. If some product does
        # not have geolocation, get it from the geofilename:
        if not read_external_geo:
            LOG.info("Loading PPS parameters done.")
            return

        # Load geolocation
        interpolate = False
        if geofilename:
            geodict = get_lonlat(geofilename)
            lons, lats = geodict['lon'], geodict['lat']
            if lons.shape != shape or lats.shape != shape:
                interpolate = True
                row_indices = geodict['row_indices']
                column_indices = geodict['col_indices']

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
                # satint.fill_borders("y", "x")
                lons, lats = satint.interpolate()

            try:
                from pyresample import geometry
                lons = np.ma.masked_array(lons, nodata_mask)
                lats = np.ma.masked_array(lats, nodata_mask)
                area = geometry.SwathDefinition(lons=lons,
                                                lats=lats)
            except ImportError:
                area = None

        for chn in read_external_geo.values():
            if area:
                chn.area = area
            else:
                chn.lat = lats
                chn.lon = lons

        LOG.info("Loading PPS parameters done.")

        return


class CloudType(NwcSafPpsChannel):

    def __init__(self, filename=None):
        NwcSafPpsChannel.__init__(self, filename)
        self.name = "CT"


class CloudTopTemperatureHeight(NwcSafPpsChannel):

    def __init__(self, filename=None):
        NwcSafPpsChannel.__init__(self, filename)
        self.name = "CTTH"


class CloudMask(NwcSafPpsChannel):

    def __init__(self, filename=None):
        NwcSafPpsChannel.__init__(self, filename)
        self.name = "CMA"


class PrecipitationClouds(NwcSafPpsChannel):

    def __init__(self, filename=None):
        NwcSafPpsChannel.__init__(self, filename)
        self.name = "PC"


class CloudPhysicalProperties(NwcSafPpsChannel):

    def __init__(self, filename=None):
        NwcSafPpsChannel.__init__(self, filename)
        self.name = "CPP"


def get_lonlat(filename):
    """Read lon,lat from netCDF4 CF file"""
    import h5py

    col_indices = None
    row_indices = None

    LOG.debug("Geo File = " + filename)

    h5f = h5py.File(filename, 'r')

    lon = h5f["where"]["lon"]["data"]
    no_data = h5f["where"]["lon"]["what"].attrs["nodata"]
    missing_data = h5f["where"]["lon"]["what"].attrs["missingdata"]

    lons = np.ma.masked_equal(lon[:], no_data)
    lons = np.ma.masked_equal(lons, missing_data)

    scale_factor = h5f["where"]["lon"]["what"].attrs["gain"]
    add_offset = h5f["where"]["lon"]["what"].attrs["offset"]

    lons = lons * scale_factor + add_offset

    lat = h5f["where"]["lat"]["data"]
    no_data = h5f["where"]["lat"]["what"].attrs["nodata"]
    missing_data = h5f["where"]["lat"]["what"].attrs["missingdata"]

    lats = np.ma.masked_equal(lat[:], no_data)
    lats = np.ma.masked_equal(lats, missing_data)

    scale_factor = h5f["where"]["lat"]["what"].attrs["gain"]
    add_offset = h5f["where"]["lat"]["what"].attrs["offset"]

    lats = lats * scale_factor + add_offset

    # FIXME: this is to mask out the npp bowtie deleted pixels...
    if h5f["how"].attrs['platform'] == "npp":

        new_mask = np.zeros((16, 3200), dtype=bool)
        new_mask[0, :1008] = True
        new_mask[1, :640] = True
        new_mask[14, :640] = True
        new_mask[15, :1008] = True
        new_mask[14, 2560:] = True
        new_mask[1, 2560:] = True
        new_mask[0, 2192:] = True
        new_mask[15, 2192:] = True
        new_mask = np.tile(new_mask, (lons.shape[0] / 16, 1))
        lons = np.ma.masked_where(new_mask, lons)
        lats = np.ma.masked_where(new_mask, lats)

    # close the h5 file.
    h5f.close()
    del h5f

    return {'lon': lons,
            'lat': lats,
            'col_indices': col_indices, 'row_indices': row_indices}


if __name__ == '__main__':

    from mpop.utils import debug_on
    debug_on()

    # cpp = CloudPhysicalProperties(
    #    "/data/proj/safutv/data/polar_out/direct_readout/S_NWC_CPP_eos2_66743_20141120T1143140Z_20141120T1156542Z.nc")
    ct = CloudType(
        "S_NWC_CT_noaa19_30165_20141216T0135195Z_20141216T0151114Z.h5")

    res = get_lonlat(
        "S_NWC_avhrr_noaa19_30165_20141216T0135195Z_20141216T0151114Z.h5")
