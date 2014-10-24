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

"""The :mod:`satout.cfscene` module provide a proxy class and utilites for
conversion of mpop scene to cf conventions.
"""

import numpy as np
import numpy.ma as ma
from netCDF4 import date2num

from mpop.channel import Channel
from mpop.satout.netcdf4 import netcdf_cf_writer

import logging
LOG = logging.getLogger('cfscene')


#CF_DATA_TYPE = np.int16
CF_FLOAT_TYPE = np.float64
TIME_UNITS = "seconds since 1970-01-01 00:00:00"


class InfoObject(object):

    """Simple data and info container.
    """

    def __init__(self):
        self.info = {}
        self.data = None


class CFScene(object):

    """Scene proxy class for cf conventions. The constructor should be called
    with the *scene* to transform as argument.
    """

    def __init__(self, scene, dtype=np.int16, band_axis=2):
        if not issubclass(dtype, np.integer):
            raise TypeError('Only integer saving allowed for CF data')

        self.info = scene.info.copy()
        if "time" in self.info:
            del self.info["time"]
        CF_DATA_TYPE = dtype

        # Other global attributes
        self.info["Conventions"] = "CF-1.5"
        self.info["platform"] = scene.satname + "-" + str(scene.number)
        self.info["instrument"] = scene.instrument_name
        if scene.variant:
            self.info["service"] = scene.variant
        if scene.orbit:
            self.info["orbit"] = scene.orbit

        self.time = InfoObject()
        self.time.data = date2num(scene.time_slot,
                                  TIME_UNITS)
        self.time.info = {"var_name": "time",
                          "var_data": self.time.data,
                          "var_dim_names": (),
                          "long_name": "Nominal time of the image",
                          "standard_name": "time",
                          "units": TIME_UNITS}

        grid_mappings = []
        areas = []
        area = None
        area_units = []
        counter = 0
        gm_counter = 0
        area_counter = 0

        for chn in scene:

            if not chn.is_loaded():
                continue

            if not isinstance(chn, Channel):
                setattr(self, chn.name, chn)
                continue

            fill_value = np.iinfo(CF_DATA_TYPE).min
            if ma.count_masked(chn.data) == chn.data.size:
                # All data is masked
                data = np.ones(chn.data.shape, dtype=CF_DATA_TYPE) * fill_value
                scale = 1
                offset = 0
            else:
                chn_max = chn.data.max()
                chn_min = chn.data.min()

                scale = ((chn_max - chn_min) /
                         (2 ** np.iinfo(CF_DATA_TYPE).bits - 2.0))
                # Handle the case where all data has the same value.
                if scale == 0:
                    scale = 1
                if np.iinfo(CF_DATA_TYPE).kind == 'i':
                    # Signed data type
                    offset = (chn_max + chn_min) / 2.0
                else:  # Unsigned data type
                    offset = chn_min - scale

                if isinstance(chn.data, np.ma.MaskedArray):
                    data = (
                        (chn.data.data - offset) / scale).astype(CF_DATA_TYPE)
                    data[chn.data.mask] = fill_value
                else:
                    data = ((chn.data - offset) / scale).astype(CF_DATA_TYPE)

            data = np.ma.expand_dims(data, band_axis)

            # it's a grid mapping
            try:
                if chn.area.proj_dict not in grid_mappings:
                    # create new grid mapping
                    grid_mappings.append(chn.area.proj_dict)
                    area = InfoObject()
                    area.data = 0
                    area.info = {"var_name": "grid_mapping_" + str(gm_counter),
                                 "var_data": area.data,
                                 "var_dim_names": ()}
                    area.info.update(proj2cf(chn.area.proj_dict))
                    area.info.setdefault("units", "m")
                    setattr(self, area.info["var_name"], area)
                    gm_counter += 1
                else:
                    # use an existing grid mapping
                    str_gmc = str(grid_mappings.index(chn.area.proj_dict))
                    area = InfoObject()
                    area.data = 0
                    area.info = {"var_name": "grid_mapping_" + str_gmc,
                                 "var_data": area.data,
                                 "var_dim_names": ()}
                    area.info.update(proj2cf(chn.area.proj_dict))
                    area.info.setdefault("units", "m")

                if(chn.area in areas):
                    str_arc = str(areas.index(chn.area))
                    xy_names = ["y" + str_arc, "x" + str_arc]
                else:
                    areas.append(chn.area)
                    str_arc = str(area_counter)
                    area_counter += 1
                    x__ = InfoObject()
                    chn.area.get_proj_coords(cache=True)
                    x__.data = chn.area.projection_x_coords[0, :]
                    x__.info = {"var_name": "x" + str_arc,
                                "var_data": x__.data,
                                "var_dim_names": ("x" + str_arc,),
                                "units": "rad",
                                "standard_name": "projection_x_coordinate",
                                "long_name": "x coordinate of projection"}
                    if area.info["grid_mapping_name"] == "geostationary":
                        x__.data /= float(
                            area.info["perspective_point_height"])
                        xpix = np.arange(len(x__.data), dtype=np.uint16)
                        xsca = ((x__.data[-1] - x__.data[0]) /
                                (xpix[-1] + xpix[0]))
                        xoff = x__.data[0] - xpix[0] * xsca
                        x__.data = xpix
                        x__.info["var_data"] = xpix
                        x__.info["scale_factor"] = xsca
                        x__.info["add_offset"] = xoff
                    setattr(self, x__.info["var_name"], x__)

                    y__ = InfoObject()
                    y__.data = chn.area.projection_y_coords[:, 0]
                    y__.info = {"var_name": "y" + str_arc,
                                "var_data": y__.data,
                                "var_dim_names": ("y" + str_arc,),
                                "units": "rad",
                                "standard_name": "projection_y_coordinate",
                                "long_name": "y coordinate of projection"}
                    if area.info["grid_mapping_name"] == "geostationary":
                        y__.data /= float(
                            area.info["perspective_point_height"])
                        ypix = np.arange(len(y__.data), dtype=np.uint16)
                        ysca = ((y__.data[-1] - y__.data[0]) /
                                (ypix[-1] + ypix[0]))
                        yoff = y__.data[0] - ypix[0] * ysca
                        y__.data = ypix
                        y__.info["var_data"] = ypix
                        y__.info["scale_factor"] = ysca
                        y__.info["add_offset"] = yoff
                    setattr(self, y__.info["var_name"], y__)

                    xy_names = [y__.info["var_name"], x__.info["var_name"]]

            # It's not a grid mapping, go for lons and lats
            except AttributeError:
                area = None
                if(chn.area in areas):
                    str_arc = str(areas.index(chn.area))
                    coordinates = ("lat" + str_arc + " " + "lon" + str_arc)
                else:
                    areas.append(chn.area)
                    str_arc = str(area_counter)
                    area_counter += 1
                    lons = InfoObject()
                    try:
                        lons.data = chn.area.lons[:]
                    except AttributeError:
                        lons.data = scene.area.lons[:]

                    lons.info = {"var_name": "lon" + str_arc,
                                 "var_data": lons.data,
                                 "var_dim_names": ("y" + str_arc,
                                                   "x" + str_arc),
                                 "units": "degrees east",
                                 "long_name": "longitude coordinate",
                                 "standard_name": "longitude"}
                    if lons.data is not None:
                        setattr(self, lons.info["var_name"], lons)

                    lats = InfoObject()
                    try:
                        lats.data = chn.area.lats[:]
                    except AttributeError:
                        lats.data = scene.area.lats[:]

                    lats.info = {"var_name": "lat" + str_arc,
                                 "var_data": lats.data,
                                 "var_dim_names": ("y" + str_arc,
                                                   "x" + str_arc),
                                 "units": "degrees north",
                                 "long_name": "latitude coordinate",
                                 "standard_name": "latitude"}
                    if lats.data is not None:
                        setattr(self, lats.info["var_name"], lats)

                    if lats.data is not None and lons.data is not None:
                        coordinates = (lats.info["var_name"] + " " +
                                       lons.info["var_name"])
                xy_names = ["y" + str_arc, "x" + str_arc]

            if (chn.area, chn.info['units']) in area_units:
                str_cnt = str(area_units.index((chn.area, chn.info['units'])))
                # area has been used before
                band = getattr(self, "band" + str_cnt)

                # data
                band.data = np.concatenate((band.data, data), axis=band_axis)
                band.info["var_data"] = band.data

                # bandname
                bandname = getattr(self, "bandname" + str_cnt)
                bandname.data = np.concatenate((bandname.data,
                                                np.array([chn.name])))
                bandname.info["var_data"] = bandname.data

                # offset
                off_attr = np.concatenate((off_attr,
                                           np.array([offset])))
                band.info["add_offset"] = off_attr

                # scale
                sca_attr = np.concatenate((sca_attr,
                                           np.array([scale])))
                band.info["scale_factor"] = sca_attr

                # wavelength bounds
                bwl = getattr(self, "wl_bnds" + str_cnt)
                bwl.data = np.vstack((bwl.data,
                                      np.array([chn.wavelength_range[0],
                                                chn.wavelength_range[2]])))
                bwl.info["var_data"] = bwl.data

                # nominal_wavelength
                nwl = getattr(self, "nominal_wavelength" + str_cnt)
                nwl.data = np.concatenate((nwl.data,
                                           np.array([chn.wavelength_range[1]])))
                nwl.info["var_data"] = nwl.data

            else:
                # first encounter of this area and unit
                str_cnt = str(counter)
                counter += 1
                area_units.append((chn.area, chn.info["units"]))

                # data

                band = InfoObject()
                band.data = data
                dim_names = xy_names
                dim_names.insert(band_axis, 'band' + str_cnt)
                band.info = {"var_name": "Image" + str_cnt,
                             "var_data": band.data,
                             'var_dim_names': dim_names,
                             "_FillValue": fill_value,
                             "long_name": "Band data",
                             "units": chn.info["units"],
                             "resolution": chn.resolution}

                # bandname

                bandname = InfoObject()
                bandname.data = np.array([chn.name], 'O')
                bandname.info = {"var_name": "band" + str_cnt,
                                 "var_data": bandname.data,
                                 "var_dim_names": ("band" + str_cnt,),
                                 "standard_name": "band_name"}
                setattr(self, "bandname" + str_cnt, bandname)

                # offset
                off_attr = np.array([offset])
                band.info["add_offset"] = off_attr

                # scale
                sca_attr = np.array([scale])
                band.info["scale_factor"] = sca_attr

                # wavelength bounds
                wlbnds = InfoObject()
                wlbnds.data = np.array([[chn.wavelength_range[0],
                                         chn.wavelength_range[2]]])
                wlbnds.info = {"var_name": "wl_bnds" + str_cnt,
                               "var_data": wlbnds.data,
                               "var_dim_names": ("band" + str_cnt, "nv")}
                setattr(self, wlbnds.info["var_name"], wlbnds)

                # nominal_wavelength
                nomwl = InfoObject()
                nomwl.data = np.array([chn.wavelength_range[1]])
                nomwl.info = {"var_name": "nominal_wavelength" + str_cnt,
                              "var_data": nomwl.data,
                              "var_dim_names": ("band" + str_cnt,),
                              "standard_name": "radiation_wavelength",
                              "units": "um",
                              "bounds": wlbnds.info["var_name"]}
                setattr(self, "nominal_wavelength" + str_cnt, nomwl)

                # grid mapping or lon lats
                if area is not None:
                    band.info["grid_mapping"] = area.info["var_name"]
                else:
                    band.info["coordinates"] = coordinates

                setattr(self, "band" + str_cnt, band)

        for i, area_unit in enumerate(area_units):
            # compute data reduction
            fill_value = np.iinfo(CF_DATA_TYPE).min
            band = getattr(self, "band" + str(i))
            # band.info["valid_range"] = np.array([valid_min, valid_max]),

    def save(self, filename, *args, **kwargs):
        return netcdf_cf_writer(filename, self, kwargs.get("compression", True))


def proj2cf(proj_dict):
    """Return the cf grid mapping from a proj dict.

    Description of the cf grid mapping:
    http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/ch05s06.html

    Table of the available grid mappings:
    http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/apf.html
    """

    cases = {"geos": geos2cf,
             "stere": stere2cf,
             "merc": merc2cf,
             "aea": aea2cf,
             "laea": laea2cf,
             "ob_tran": obtran2cf,
             "eqc": eqc2cf, }

    return cases[proj_dict["proj"]](proj_dict)


def geos2cf(proj_dict):
    """Return the cf grid mapping from a geos proj dict.
    """

    return {"grid_mapping_name": "geostationary",
            "latitude_of_projection_origin": 0.0,
            "longitude_of_projection_origin": eval(proj_dict["lon_0"]),
            "semi_major_axis": eval(proj_dict["a"]),
            "semi_minor_axis": eval(proj_dict["b"]),
            "perspective_point_height": eval(proj_dict["h"])
            }


def eqc2cf(proj_dict):
    """Return the cf grid mapping from a eqc proj dict. However, please be
    aware that this is not an official CF projection. See
    http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/apf.html.
    """

    return {"grid_mapping_name": "equirectangular",
            "latitude_of_true_scale": eval(proj_dict.get("lat_ts", "0")),
            "latitude_of_projection_origin": eval(proj_dict["lat_0"]),
            "longitude_of_projection_origin": eval(proj_dict["lon_0"]),
            "false_easting": eval(proj_dict.get("x_0", "0")),
            "false_northing": eval(proj_dict.get("y_0", "0"))
            }


def stere2cf(proj_dict):
    """Return the cf grid mapping from a stereographic proj dict.
    """

    return {"grid_mapping_name": "stereographic",
            "latitude_of_projection_origin": eval(proj_dict["lat_0"]),
            "longitude_of_projection_origin": eval(proj_dict["lon_0"]),
            "scale_factor_at_projection_origin": eval(
                proj_dict.get("x_0", "1.0")),
            "false_easting": eval(proj_dict.get("x_0", "0")),
            "false_northing": eval(proj_dict.get("y_0", "0"))
            }


def merc2cf(proj_dict):
    """Return the cf grid mapping from a mercator proj dict.
    """

    raise NotImplementedError(
        "CF grid mapping from a PROJ.4 mercator projection is not implemented")


def aea2cf(proj_dict):
    """Return the cf grid mapping from a Albers Equal Area proj dict.
    """

    #standard_parallels = []
    # for item in ['lat_1', 'lat_2']:
    #    if item in proj_dict:
    #        standard_parallels.append(eval(proj_dict[item]))
    if 'lat_2' in proj_dict:
        standard_parallel = [eval(proj_dict['lat_1']),
                             eval(proj_dict['lat_2'])]
    else:
        standard_parallel = [eval(proj_dict['lat_1'])]

    lat_0 = 0.0
    if 'lat_0' in proj_dict:
        lat_0 = eval(proj_dict['lat_0'])

    x_0 = 0.0
    if 'x_0' in proj_dict:
        x_0 = eval(proj_dict['x_0'])

    y_0 = 0.0
    if 'y_0' in proj_dict:
        y_0 = eval(proj_dict['y_0'])

    retv = {"grid_mapping_name": "albers_conical_equal_area",
            "standard_parallel": standard_parallel,
            "latitude_of_projection_origin": lat_0,
            "longitude_of_central_meridian": eval(proj_dict["lon_0"]),
            "false_easting": x_0,
            "false_northing": y_0
            }

    retv = build_dict("albers_conical_equal_area",
                      proj_dict,
                      standard_parallel=["lat_1", "lat_2"],
                      latitude_of_projection_origin="lat_0",
                      longitude_of_central_meridian="lon_0",
                      false_easting="x_0",
                      false_northing="y_0")

    return retv


def laea2cf(proj_dict):
    """Return the cf grid mapping from a Lambert azimuthal equal-area proj dict.
    http://trac.osgeo.org/gdal/wiki/NetCDF_ProjectionTestingStatus
    """
    x_0 = eval(proj_dict.get('x_0', '0.0'))
    y_0 = eval(proj_dict.get('y_0', '0.0'))

    retv = {"grid_mapping_name": "lambert_azimuthal_equal_area",
            "longitude_of_projection_origin": eval(proj_dict["lon_0"]),
            "latitude_of_projection_origin": eval(proj_dict["lat_0"]),
            "false_easting": x_0,
            "false_northing": y_0
            }

    retv = build_dict("lambert_azimuthal_equal_area",
                      proj_dict,
                      longitude_of_projection_origin="lon_0",
                      latitude_of_projection_origin="lat_0",
                      false_easting="x_0",
                      false_northing="y_0")

    return retv


def obtran2cf(proj_dict):
    """Return a grid mapping from a rotated pole grid (General Oblique
    Transformation projection) proj dict.

    Please be aware this is not yet supported by CF!
    """
    LOG.warning("The General Oblique Transformation " +
                "projection is not CF compatible yet...")
    x_0 = eval(proj_dict.get('x_0', '0.0'))
    y_0 = eval(proj_dict.get('y_0', '0.0'))

    retv = {"grid_mapping_name": "general_oblique_transformation",
            "longitude_of_projection_origin": eval(proj_dict["lon_0"]),
            "grid_north_pole_latitude": eval(proj_dict["o_lat_p"]),
            "grid_north_pole_longitude": eval(proj_dict["o_lon_p"]),
            "false_easting": x_0,
            "false_northing": y_0
            }

    retv = build_dict("general_oblique_transformation",
                      proj_dict,
                      longitude_of_projection_origin="lon_0",
                      grid_north_pole_latitude="o_lat_p",
                      grid_north_pole_longitude="o_lon_p",
                      false_easting="x_0",
                      false_northing="y_0")

    return retv


def build_dict(proj_name, proj_dict, **kwargs):
    new_dict = {}
    new_dict["grid_mapping_name"] = proj_name
    for key, val in kwargs.items():
        if isinstance(val, (list, tuple)):
            new_dict[key] = [eval(proj_dict[x]) for x in val if x in proj_dict]
        elif val in proj_dict:
            new_dict[key] = eval(proj_dict[val])
    # add a, b, rf and/or ellps
    if "a" in proj_dict:
        new_dict["semi_major_axis"] = eval(proj_dict["a"])
    if "b" in proj_dict:
        new_dict["semi_minor_axis"] = eval(proj_dict["b"])
    if "rf" in proj_dict:
        new_dict["inverse_flattening"] = eval(proj_dict["rf"])
    if "ellps" in proj_dict:
        new_dict["ellipsoid"] = proj_dict["ellps"]

    return new_dict


def aeqd2cf(proj_dict):
    return build_dict("azimuthal_equidistant",
                      proj_dict,
                      standard_parallel=["lat_1", "lat_2"],
                      latitude_of_projection_origin="lat_0",
                      longitude_of_central_meridian="lon_0",
                      false_easting="x_0",
                      false_northing="y_0")
