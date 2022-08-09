#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2022 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Html formatting function for Scene representation in notebooks."""

from functools import lru_cache
from html import escape
from importlib.resources import read_binary

import toolz
import xarray as xr
from pyresample._formatting_html import _icon, area_repr, collapsible_section  # , STATIC_FILES
from xarray.core.formatting_html import _mapping_section, summarize_vars  # , datavar_section

STATIC_FILES = {"html": [("xarray.static.html", "icons-svg-inline.html")],
                "css": [("pyresample.static.css", "style.css"), ("xarray.static.css", "style.css")]
                }


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed."""
    css = "\n".join([read_binary(package, resource).decode("utf-8") for package, resource in STATIC_FILES["css"]])

    html = "\n".join([read_binary(package, resource).decode("utf-8") for package, resource in STATIC_FILES["html"]])

    return [html, css]


@toolz.curry
def attr(attr_name, ds):
    """Get attribute."""
    return ds.attrs.get(attr_name)


def sensor_section(platform_name, sensor_name, datasets):
    """Generate sensor section."""
    by_area = toolz.groupby(lambda x: x.attrs.get("area").proj_dict.get("proj"), datasets)

    sensor_name = sensor_name.upper()
    icon = _icon("icon-satellite")

    section_name = (f"<div>"
                    f"<a href='https://space.oscar.wmo.int/satellites/view/{platform_name}'>{platform_name}</a>/"
                    f"<a href='https://space.oscar.wmo.int/instruments/view/{sensor_name}'>{sensor_name}</a>"
                    "</div>")

    section_details = ""
    for proj, ds in by_area.items():
        section_details += resolution_section(proj, ds)

    html = collapsible_section(section_name, details=section_details, icon=icon)
    # html += "</div>"
    return html


def resolution_section(projection, datasets):
    """Generate resolution section."""
    def resolution(dataset):
        area = dataset.attrs.get("area")
        resolution_str = "/".join([str(round(x, 1)) for x in area.resolution])
        return resolution_str

    by_resolution = toolz.groupby(resolution, datasets)

    # html = f"<div>{projection}</div>"
    # modify area representation
    html = area_repr(datasets[0].attrs.get("area"), include_header=False)
    #
    for res, ds in by_resolution.items():
        ds_dict = {i.attrs['name']: i.rename(i.attrs['name']) for i in ds if i.attrs.get('area') is not None}
        dss = xr.merge(ds_dict.values())
        html += xarray_dataset_repr(dss, "Resolution (x/y): {}".format(res))

    html += "</div>"
    return html


def scene_repr(scene):
    """Html representation of Scene.

    Args:
        scene (:class:`~satpy.scene.Scene`): Satpy scene.

    Returns:
        str: Html str

    TODO:
        - streamline loading and combining of css styles. Move _load_static_files function into pyresample
        - display combined numer of datasets, area name, projection, extent, sensor, start/end time after object type?
        - drop "unecessary" attributes from the datasets?
        - make the area show as tabbed display (attribtues/ map each in tab)?
        - only show resolution and dimensions (number of pixels) for each section if the area definition extent,
          projection (and name) is the same?
        - for the data variables list not only display channel (dataarray) name but also other DataId info
          (like spectral range)?
        - what about composites?
    """
    icons_svg, css_style = _load_static_files()

    obj_type = f"satpy.scene.{type(scene).__name__}"
    header = ("<div class='pyresample-header'>"
              "<div class='pyresample-obj-type'>"
              f"{escape(obj_type)}"
              "</div>"
              "</div>"
              )
    # insert number of different sensors, area defs (projections), resolutions, area_extents? after object type

    html = (
           f"{icons_svg}<style>{css_style}</style>"
           f"<pre class='pyresample-text-repr-fallback'>{escape(repr(scene))}</pre>"
           "<div class='pyresample-wrap' style='display:none'>"
           )

    html += f"{header}"

    dslist = list(scene._datasets.values())
    # scn_by_sensor = toolz.groupby(attr("sensor"), dslist)
    scn_by_sensor = toolz.groupby(lambda x: (x.attrs.get("platform_name"), x.attrs.get("sensor")), dslist)

    # when more than one platform/sensor collapse section

    for (platform, sensor), dss in scn_by_sensor.items():
        html += sensor_section(platform, sensor, dss)  # simple(scene)

    html += "</div>"

    return html


def xarray_dataset_repr(dataset, ds_name):
    """Wrap xarray dataset representation html."""
    data_variables = _mapping_section(mapping=dataset, name=ds_name, details_func=summarize_vars,
                                      max_items_collapse=15, expand_option_name="display_expand_data_vars")

    ds_list = ("<div class='xr-wrap' style='display:none'>"
               f"<ul class='xr-sections'>"
               f"<li class='xr-section-item'>{data_variables}</li>"
               "</ul></div>")

    return ds_list


def simple(scene):
    """Generate old scene repr."""
    html = ""
    html += "<div class='pyresample-area-sections'>"

    for _, dss in scene.iter_by_area():
        area = scene[dss[0]].attrs.get("area")
        html += area_repr(area, include_header=False)

        ds_names = [x.name for x in dss]
        scn_ds = scene.to_xarray_dataset(ds_names)
        html += xarray_dataset_repr(scn_ds, "Data Variables")
        # collapsible_section("Channels", details=ds_list, icon=icon)
        html += "<hr>"

    return html
