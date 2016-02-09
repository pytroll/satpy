#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

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

"""Base classes for composite objects.
"""

from satpy.projectable import InfoObject, Projectable
import numpy as np
import logging
from satpy.tools import sunzen_corr_cos
import six
from satpy import CONFIG_PATH, config_search_paths, runtime_import
from satpy.readers import DatasetID
import os
try:
    import configparser
except:
    from six.moves import configparser

LOG = logging.getLogger(__name__)


class IncompatibleAreas(Exception):
    """
    Error raised upon compositing things of different shapes.
    """
    pass

class CompositeReader(object):
    """Read composites using the configuration files on disk.
    """
    def __init__(self, composite_names, sensor_names=None):
        pass

# FIXME: is kwargs really used here ? what for ? This should be made explicit probably
def _get_compositors_from_configs(composite_configs, composite_names, sensor_names=None, **kwargs):
    """Use the *composite_configs* to return the compositors for requested *composite_names*.

    :param composite_configs:
    :param composite_names:
    :return: A list of loaded compositors
    """

    conf = configparser.ConfigParser()
    conf.read(composite_configs)
    compositors = {}
    for section_name in conf.sections():
        if section_name.startswith("composite:"):
            options = dict(conf.items(section_name))
            options["sensor"] = options.setdefault("sensor", None)
            if options.get("sensor", None):
                options["sensor"] = set(options["sensor"].split(","))
            comp_cls = options.pop("compositor", None)
            if not comp_cls:
                raise ValueError("'compositor' missing or empty in config files: %s" % (composite_configs,))

            # Check if the caller only wants composites for a certain sensor
            if (options["sensor"] is not None and
                        sensor_names is not None and
                        not (set(sensor_names) & set(options["sensor"]))):
                continue
            # Check if the caller only wants composites with certain names
            if composite_names is not None and options["name"] not in composite_names:
                continue

            if options["name"] in compositors:
                LOG.debug("Duplicate composite found, previous composite '%s' will be overwritten",
                            options["name"])

            # Get other identifiers that could be used to filter the prerequisites
            other_identifiers = {}
            for o_id in ["resolution", "calibration", "polarization"]:
                if o_id in options:
                    other_identifiers[o_id] = options[o_id].split(",")
            optional_other_identifiers = {}
            for o_id in ["resolution", "calibration", "polarization"]:
                if "optional_" + o_id in options:
                    optional_other_identifiers[o_id] = options["optional_" + o_id].split(",")

            def _normalize_prereqs(prereqs, other_identifiers):
                # Pull out prerequisites
                prerequisites = prereqs.split(",")
                prereqs = []
                for idx, prerequisite in enumerate(prerequisites):
                    ds_id = {"name": None, "wavelength": None}
                    # convert the prerequisite
                    try:
                        # prereqs can be wavelengths
                        ds_id["wavelength"] = float(prerequisite)
                    except ValueError:
                        # or names
                        ds_id["name"] = prerequisite

                    # add further filtering to the prerequisites via the DatasetID namedtuple
                    for o_id, vals in other_identifiers.items():
                        ds_id[o_id] = vals[idx]

                    prereqs.append(DatasetID(**ds_id))

                return prereqs

            if "prerequisites" in options:
                options["prerequisites"] = _normalize_prereqs(options["prerequisites"], other_identifiers)

            if "optional_prerequisites" in options:
                options["optional_prerequisites"] = _normalize_prereqs(options["optional_prerequisites"],
                                                                       optional_other_identifiers)

            if "metadata_requirements" in options:
                options["metadata_requirements"] = options["metadata_requirements"].split(",")

            try:
                loader = runtime_import(comp_cls)
            except ImportError:
                LOG.warning("Could not import composite class '%s' for"
                            " compositor '%s'", comp_cls, options["name"])
                continue

            options.update(**kwargs)
            comp = loader(**options)
            compositors[options["name"]] = comp
    return compositors



def load_compositors(composite_names=None, sensor_names=None, ppp_config_dir=CONFIG_PATH, **kwargs):
    """Load the requested *composite_names*.

    :param composite_names: The name of the desired composites
    :param sensor_names:  The name of the desired sensors to load composites for
    :param ppp_config_dir: The config directory
    :return: A list of loaded compositors
    """
    if sensor_names is None:
        sensor_names = []
    config_filenames = ["generic.cfg"] + [sensor_name + ".cfg" for sensor_name in sensor_names]
    compositors = dict()
    for config_filename in config_filenames:
        LOG.debug("Looking for composites config file %s", config_filename)
        composite_configs = config_search_paths(os.path.join("composites", config_filename), ppp_config_dir)
        if not composite_configs:
            LOG.debug("No composite config found called %s", config_filename)
            continue
        composite_configs.reverse()
        compositors.update(_get_compositors_from_configs(composite_configs, composite_names, sensor_names, **kwargs))
    return compositors


class CompositeBase(InfoObject):
    def __init__(self, name, prerequisites=[], optional_prerequisites=[], metadata_requirements=[], **kwargs):
        # Required info
        kwargs["name"] = name
        kwargs["prerequisites"] = prerequisites
        kwargs["optional_prerequisites"] = optional_prerequisites
        kwargs["metadata_requirements"] = metadata_requirements
        super(CompositeBase, self).__init__(**kwargs)

    def __call__(self, datasets, optional_datasets=None, **info):
        raise NotImplementedError()

    def __str__(self):
        from pprint import pformat
        return pformat(self.info)

    def __repr__(self):
        from pprint import pformat
        return pformat(self.info)

class SunZenithNormalize(object):
    # FIXME: the cache should be cleaned up
    coszen = {}

    def __call__(self, projectable,  *args, **kwargs):
        from pyorbital.astronomy import cos_zen
        key = (projectable.info["start_time"], projectable.info["area"].name)
        if key not in self.coszen:
            LOG.debug("Computing sun zenith angles.")
            self.coszen[key] = np.ma.masked_outside(cos_zen(projectable.info["start_time"],
                                                    *projectable.info["area"].get_lonlats()),
                                                    0.035, # about 88 degrees.
                                                    1,
                                                    copy=False)
        return sunzen_corr_cos(projectable, self.coszen[key])


class RGBCompositor(CompositeBase):
    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(projectables),))
        the_data = np.rollaxis(np.ma.dstack([projectable for projectable in projectables]), axis=2)
        info = projectables[0].info.copy()
        info.update(projectables[1].info)
        info.update(projectables[2].info)
        info.update(self.info)
        # FIXME: should this be done here ?
        info["wavelength_range"] = None
        info.pop("units", None)
        sensor = set()
        for projectable in projectables:
            current_sensor = projectable.info.get("sensor", None)
            if current_sensor:
                if isinstance(current_sensor, (str, bytes, six.text_type)):
                    sensor.add(current_sensor)
                else:
                    sensor |= current_sensor
        if len(sensor) == 0:
            sensor = None
        elif len(sensor) == 1:
            sensor = list(sensor)[0]
        info["sensor"] = sensor
        info["mode"] = "RGB"
        return Projectable(data=the_data, **info)


class SunCorrectedRGB(RGBCompositor):
    def __call__(self, projectables, *args, **kwargs):
        suncorrector = SunZenithNormalize()
        for i, projectable in enumerate(projectables):
            # FIXME: check the wavelength instead, so radiances can be corrected too.
            if projectable.info.get("units") == "%":
                projectables[i] = suncorrector(projectable)
        res = RGBCompositor.__call__(self,
                                     projectables,
                                     *args, **kwargs)
        return res


class Airmass(RGBCompositor):
    def __call__(self, projectables, *args, **kwargs):
        """Make an airmass RGB image composite.

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | WV6.2 - WV7.3      |     -25 to 0 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR9.7 - IR10.8     |     -40 to 5 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | WV6.2              |   243 to 208 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        """
        res = RGBCompositor.__call__(self,
                                     (projectables[0] - projectables[1],
                                      projectables[2] - projectables[3],
                                      projectables[0]),
                                     *args, **kwargs)
        return res


class Convection(RGBCompositor):
    def __call__(self, projectables, *args, **kwargs):
        """Make a Severe Convection RGB image composite.

        +--------------------+--------------------+--------------------+
        | Channels           | Span               | Gamma              |
        +====================+====================+====================+
        | WV6.2 - WV7.3      |     -30 to 0 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR3.9 - IR10.8     |      0 to 55 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR1.6 - VIS0.6     |    -70 to 20 %     | gamma 1            |
        +--------------------+--------------------+--------------------+
        """
        res = RGBCompositor.__call__(self,
                                     (projectables[3] - projectables[4],
                                      projectables[2] - projectables[5],
                                      projectables[1] - projectables[0]),
                                     *args, **kwargs)
        return res


class Dust(RGBCompositor):
    def __call__(self, projectables, *args, **kwargs):
        """Make a Dust RGB image composite.

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | IR12.0 - IR10.8    |     -4 to 2 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR10.8 - IR8.7     |     0 to 15 K      | gamma 2.5          |
        +--------------------+--------------------+--------------------+
        | IR10.8             |   261 to 289 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        """
        res = RGBCompositor.__call__(self,
                                     (projectables[2] - projectables[1],
                                      projectables[1] - projectables[0],
                                      projectables[1]),
                                     *args, **kwargs)
        return res
