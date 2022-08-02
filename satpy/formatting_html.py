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

from pyresample.formatting_html import area_repr  # , _icon, collapsible_section, STATIC_FILES
from xarray.core.formatting_html import datavar_section

STATIC_FILES = {"html": [("xarray.static.html", "icons-svg-inline.html")],
                "css": [("pyresample.static.css", "style.css"), ("xarray.static.css", "style.css")]
                }


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed."""
    css = "\n".join([read_binary(package, resource).decode("utf-8") for package, resource in STATIC_FILES["css"]])

    html = "\n".join([read_binary(package, resource).decode("utf-8") for package, resource in STATIC_FILES["html"]])

    return [html, css]


def scene_repr(scene):
    """Html representation of Scene.

    Args:
        scene (:class:`~satpy.scene.Scene`): Satpy scene.

    Returns:
        str: Html str

    TODO:
        - streamline loading and combining of css styles. Move _load_static_files function into pyresample
        - display combined numer of datasets, area name, projection, extent, sensor after object type?
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

    html = (
           f"{icons_svg}<style>{css_style}</style>"
           f"<pre class='pyresample-text-repr-fallback'>{escape(repr(scene))}</pre>"
           "<div class='pyresample-wrap' style='display:none'>"
           )

    html += f"{header}"

    html += "<div class='pyresample-area-sections'>"

    for _, dss in scene.iter_by_area():
        area = scene[dss[0]].attrs.get("area")
        html += area_repr(area, include_header=False)

        ds_names = [x.name for x in dss]
        scn_ds = scene.to_xarray_dataset(ds_names)
        ds_list = ("<div class='xr-wrap' style='display:none'>"
                   f"<ul class='xr-sections'>"
                   f"<li class='xr-section-item'>{datavar_section(scn_ds.data_vars)}</li>"
                   "</ul></div>")

        html += ds_list  # collapsible_section("Channels", details=ds_list, icon=icon)
        html += "<hr>"

    html += "</div>"

    return html
