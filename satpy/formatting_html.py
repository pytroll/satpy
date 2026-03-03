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

import uuid
from functools import lru_cache
from html import escape
from importlib.resources import read_binary

import toolz
import xarray as xr

try:
    from pyresample._formatting_html import _icon, plot_area_def
except ModuleNotFoundError:
    cartopy = False

from xarray.core.formatting_html import _mapping_section, summarize_vars  # , datavar_section

STATIC_FILES = {"html": [("pyresample.static.html", "icons_svg_inline.html")],
                "css": [("pyresample.static.css", "style.css")]
                }

css = """
:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.satpy-scene-sections {
  padding: 0 !important;
  display: grid;
  grid-template-columns: 20px 20px 150px auto 20px 20px;
  width: 1000px;
  margin-top: 0px;
}

.satpy-section-name {
  grid-column: 2 / 3;
}

.satpy-scene-section-item {
  display: contents;
}

.satpy-scene-section-item input {
  display: none;
}

.satpy-scene-section-item input:enabled + label {
  cursor: pointer;
}

.satpy-scene-section-summary {
  grid-column: 1 / 4;
  padding-top: 4px;
  padding-bottom: 0px;
}

.satpy-scene-section-summary > span {
  float: right;
}

.satpy-scene-section-in:checked + label > span {
  display: none;
}

.satpy-scene-section-inline-preview {
  grid-column: 4 / -1;
  padding-left: 3px;
  padding-top: 4px;
  padding-bottom: 0px;
}

.satpy-scene-section-in:checked ~ .satpy-scene-section-inline-preview {
  display: none;
}

.satpy-scene-section-details,
.satpy-scene-section-in:checked ~ .satpy-scene-section-preview {
  display: none;
}

.satpy-scene-section-in:checked ~ .satpy-scene-section-details,
.satpy-scene-section-preview {
  display: contents;
  padding: 0;
  grid-column: 3 / -1;
}

.satpy-scene-section-area {
  display: contents;
}

.satpy-area-name {
  grid-column: 3;
  font-weight: bold;
}

.satpy-area-details {
  grid-column: 4;
}

/*show hide css for area def section */
.satpy-area-attrs {
  margin-bottom: 5px;
}

.satpy-area-attrs,
.satpy-area-map {
  display: none;
}

.satpy-area-attrs-in:checked ~ .satpy-area-attrs,
.satpy-area-map-in:checked ~ .satpy-area-map {
  display: block;
  grid-column: 3 / -1;
}

.satpy-area-attrs dt {
  width: 190px;
  float: left;
  font-weight: bold;
  margin-right: 0.5em;
}

.satpy-area-attrs dt:after {
  content: ": ";
}

.satpy-area-attrs dd {
  all: initial;
}

.satpy-area-attrs dd:after {
  clear: left;
  content: " ";
  display: block;
}

.satpy-scene-section-datasets {
  grid-column: 3 / -1;
}

.xr-wrap {
  all: initial;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 1000px;
}
 .xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 250px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  margin-bottom: 5px;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
"""


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


def collapsible_section_satpy(name, inline_details="", details="", n_items=None, enabled=True, collapsed=False,
                              icon=None):
    """Create a collapsible section.

    Args:
      name (str): Name of the section
      inline_details (str): Information to show when section is collapsed. Default nothing.
      details (str): Details to show when section is expanded.
      n_items (int): Number of items in this section.
      enabled (boolean): Is collapsing enabled. Default True.
      collapsed (boolean): Is the section collapsed on first show. Default False.
      icon (str): Name of the icon to use for the section.

    Returns:
      str: Html div structure for collapsible section.

    """
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())

    has_items = n_items is not None and n_items
    n_items_span = "" if n_items is None else f" <span>{n_items}</span>"
    enabled = "" if enabled and has_items else "disabled"
    collapsed = "" if collapsed or not has_items else "checked"
    tip = " title='Expand/collapse section'" if enabled else ""

    if icon is None:
        icon = _icon("icon-database")

    return ("<div class='satpy-scene-section-item'>"
            f"<input id='{data_id}' class='satpy-scene-section-in' "
            f"type='checkbox' {enabled} {collapsed}>"
            f"<label for='{data_id}' class='satpy-scene-section-summary' {tip}>{icon} {name}: {n_items_span}</label>"
            f"<div class='satpy-scene-section-inline-preview'>{inline_details}</div>"
            f"<div class='satpy-scene-section-details'>{details}</div>"
            "</div>"
            )


def sensor_section(platform_name, sensor_name, datasets):
    """Generate sensor section."""
    by_area = toolz.groupby(lambda x: x.attrs.get("area").proj_dict.get("proj"), datasets)
    n_areas = len(list(by_area.keys()))
    n_datasets = len(datasets)
    inline_details = f"Area(s) with {n_datasets} channels"

    by_area_type = toolz.groupby(lambda x: type(x.attrs.get("area")).__name__, datasets)

    sensor_name = sensor_name.upper()
    icon = _icon("icon-satellite")

    section_name = (f"<a href='https://space.oscar.wmo.int/satellites/view/{platform_name}'>{platform_name}</a> / "
                    f"<a href='https://space.oscar.wmo.int/instruments/view/{sensor_name}'>{sensor_name}</a>"
                    )

    if "AreaDefinition" in by_area_type.keys():
        by_area = toolz.groupby(lambda x: x.attrs.get("area").proj_dict.get("proj"), by_area_type["AreaDefinition"])
        section_details = ""
        for proj, ds in by_area.items():
            section_details += resolution_section(proj, ds)

    # html = collapsible_section_satpy(section_name, details=section_details,
                                     # inline_details=inline_details, n_items=n_areas, icon=icon)
        html = collapsible_section_satpy(section_name, details=section_details, n_items=number_of_platforms, icon=icon)

    if "SwathDefinition" in by_area_type.keys():
        from pyresample._formatting_html import swath_area_attrs_section

        from satpy import Scene

        swathlist = Scene._compare_swath_defs(max, [ds.area for ds in by_area_type["SwathDefinition"]])

        # by_area = toolz.groupby(lambda x: x.attrs.get("area").proj_dict.get("proj"), by_area_type["AreaDefinition"])

        html = collapsible_section_satpy(section_name, details=swath_area_attrs_section(swathlist),
                                         n_items=number_of_platforms, icon=icon)

    return html


def resolution_section(projection, datasets):
    """Generate resolution section."""
    def resolution(dataset):
        area = dataset.attrs.get("area")
        resolution_str = "/".join([str(round(x, 1)) for x in area.resolution])
        return resolution_str

    by_resolution = toolz.groupby(resolution, datasets)

    areadefinition = datasets[0].attrs.get("area")
    proj_dict = areadefinition.proj_dict
    proj_str = "{{{}}}".format(", ".join(["'%s': '%s'" % (str(k), str(proj_dict[k])) for k in
                                          sorted(proj_dict.keys())]))

    area_attrs = ("<dl>"
                  f"<dt>Description</dt><dd>{areadefinition.description}</dd>"
                  f"<dt>Projection</dt><dd>{proj_str}</dd>"
                  f"<dt>Extent (ll_x, ll_y, ur_x, ur_y)</dt>"
                  f"<dd>{tuple(round(x, 4) for x in areadefinition.area_extent)}</dd>"
                  "</dl>"
                  )

    area_map = plot_area_def(areadefinition, fmt="svg")

    attrs_id = "attrs-" + str(uuid.uuid4())
    map_id = "map-" + str(uuid.uuid4())
    attrs_icon = _icon("icon-file-text2")
    map_icon = _icon("icon-globe")

    html = ("<div class='satpy-scene-section-area'>"
            f"<div class='satpy-area-name'>{areadefinition.area_id}</div>"
            f"<div class='satpy-area-details'></div>"
            f"<input id='{attrs_id}' class='satpy-area-attrs-in' type='checkbox'>"
            f"<label for='{attrs_id}' title='Show/Hide properties'>{attrs_icon}</label>"
            f"<input id='{map_id}' class='satpy-area-map-in' type='checkbox'>"
            f"<label for='{map_id}' title='Show/Hide map'>{map_icon}</label>"
            f"<div class='satpy-area-attrs'>{area_attrs}</div>"
            f"<div class='satpy-area-map'>{area_map}</div>"
            "</div>"
            )

    for res, ds in by_resolution.items():
        ds_dict = {i.attrs["name"]: i.rename(i.attrs["name"]) for i in ds if i.attrs.get("area") is not None}
        dss = xr.merge(ds_dict.values(), compat="override")
        html += xarray_dataset_repr(dss, "Resolution (x/y): {}".format(res))

    return html


def scene_repr(scene):
    """Html representation of Scene.

    Args:
        scene (:class:`~satpy.scene.Scene`): Satpy scene.

    Returns:
        str: Html str

    Todo:
        - streamline loading and combining of css styles. Move _load_static_files function into pyresample
        - display combined numer of datasets, area name, projection, extent, sensor, start/end time after object type?
        - drop "unecessary" attributes from the datasets?
        - only show resolution and dimensions (number of pixels) for each section if the area definition extent,
          projection (and name) is the same?
        - for the data variables list not only display channel (dataarray) name but also other DataId info
          (like spectral range)?
        - what about composites?
    """
    icons_svg, css_style = _load_static_files()
    css_style = "\n".join([css, css_style])

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
    html += "<ul class='satpy-scene-sections'>"

    dslist = list(scene._datasets.values())
    scn_by_sensor = toolz.groupby(lambda x: (x.attrs.get("platform_name"), x.attrs.get("sensor")), dslist)

    # when more than one platform/sensor collapse section
    for (platform, sensor), dss in scn_by_sensor.items():
        html += sensor_section(platform, sensor, dss)

    html += "</ul></div>"

    return html


def xarray_dataset_repr(dataset, ds_name):
    """Wrap xarray dataset representation html."""
    data_variables = _mapping_section(mapping=dataset, name=ds_name, details_func=summarize_vars,
                                      max_items_collapse=15, expand_option_name="display_expand_data_vars")

    ds_list = ("<div class='satpy-scene-section-datasets'>"
               # "<div class='xr-wrap' style='display:none'>"
               f"<ul class='xr-sections'>"
               f"<li class='xr-section-item'>{data_variables}</li>"
               "</ul>"
               "</div>"
               # "</div>"
               )

    return ds_list
