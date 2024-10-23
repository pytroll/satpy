"""Generate the area definition list restructuredtext document.

This should be run once before generating the sphinx documentation to
produce the ``area_def_list.rst`` file referenced by ``satpy/resample.py``.

"""
import logging
import pathlib
import sys
from datetime import datetime

import bokeh
import geoviews as gv
import geoviews.feature as gf
from bokeh.embed import components
from jinja2 import Template
from pyresample._formatting_html import _load_static_files
from pyresample.area_config import area_repr, load_area
from pyresample.utils.proj4 import ignore_pyproj_proj_warnings
from reader_table import rst_table_header, rst_table_row

from satpy.resample import get_area_file

logger = logging.getLogger(__name__)

gv.extension("bokeh")


TEMPLATE = '''

{{ table_header }}
{% for area_name, area_def in areas.items() if area_def._repr_html_ is defined %}
{{ create_table_row(area_name, area_def) }}
{% endfor %}


.. raw:: html

    {{ resources }}
    {{ pyr_icons_svg | indent(5) }}
    <style>
    {{ pyr_css_style | indent(5) }}
    </style>
    {{ script | indent(5)}}

{% for area_name, area_div in area_divs_dict.items() %}

{{ area_name }}
{{ rst_underline('^', area_name|length) }}

.. raw:: html

    {{ area_repr(areas[area_name], map_content=area_div, include_header=False, include_static_files=False) |
       indent(5) }}
    <br>

{% endfor %}
'''  # noqa: Q001


def main():
    """Parse CLI arguments and generate area definition list file."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate restructuredtext area definition list for sphinx documentation")
    parser.add_argument("--area-file",
                        help="Input area YAML file to read")
    parser.add_argument("-o", "--output-file",
                        type=pathlib.Path,
                        help="HTML or restructuretext filename to create. "
                             "Defaults to 'area_def_list.rst' in the "
                             "documentation source directory.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.output_file is None:
        args.output_file = str(pathlib.Path(__file__).resolve().parent / "area_def_list.rst")
    area_file = args.area_file
    if area_file is None:
        area_file = get_area_file()[0]

    area_list = load_area(area_file)
    areas_dict = {_area_name(area): area for area in area_list}
    logger.info(f"Generating bokeh plots ({datetime.now()})...")
    script, divs_dict = _generate_html_map_divs(areas_dict)
    logger.info(f"Done generating bokeh plots ({datetime.now()})")

    def rst_underline(ch, num_chars):
        return ch * num_chars

    template = Template(TEMPLATE)
    icons_svg, css_style = _load_static_files()
    logger.info(f"Rendering document ({datetime.now()})...")
    res = template.render(
        resources=bokeh.resources.CDN.render(),
        script=script,
        area_divs_dict=divs_dict,
        areas=areas_dict,
        rst_underline=rst_underline,
        area_repr=area_repr,
        pyr_icons_svg=icons_svg,
        pyr_css_style=css_style,
        table_header=rst_table_header("Area Definitions", header=["Name", "Description", "Projection"],
                                      widths="auto", class_name="area-table"),
        create_table_row=_area_table_row,
    )
    logger.info(f"Done rendering document ({datetime.now()})")

    with open(args.output_file, mode="w") as f:
        f.write(res)


def _area_name(area_def) -> str:
    if hasattr(area_def, "attrs"):
        # pyresample 2
        return area_def.attrs["name"]
    # pyresample 1
    return area_def.area_id


def _area_table_row(area_name, area_def):
    with ignore_pyproj_proj_warnings():
        area_proj = area_def.proj_dict.get("proj")
    return rst_table_row([f"`{area_name}`_", area_def.description, area_proj])


def _generate_html_map_divs(areas_dict: dict) -> tuple[str, dict]:
    areas_bokeh_models = {}
    for area_name, area_def in areas_dict.items():
        if not hasattr(area_def, "to_cartopy_crs"):
            logger.info(f"Skipping {area_name} because it can't be converted to cartopy CRS")
            continue
        crs = area_def.to_cartopy_crs()

        features = gv.Overlay([gf.ocean, gf.land, gf.borders, gf.coastline])
        f = gv.render(
            features.opts(
                toolbar=None,
                default_tools=[],
                projection=crs,
                xlim=crs.bounds[:2],
                ylim=crs.bounds[2:],
            ),
            backend="bokeh")
        areas_bokeh_models[area_name] = f

    script, divs_dict = components(areas_bokeh_models)
    return script, divs_dict


if __name__ == "__main__":
    sys.exit(main())
