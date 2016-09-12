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

import logging
import os

import numpy as np
import six
import yaml

from satpy.config import CONFIG_PATH, config_search_paths, runtime_import
from satpy.projectable import InfoObject, Projectable, combine_info
from satpy.readers import DatasetID
from satpy.tools import sunzen_corr_cos

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

# FIXME: is kwargs really used here ? what for ? This should be made
# explicit probably


def _get_compositors_from_configs(composite_configs, composite_names, sensor_names=None, **kwargs):
    """Use the *composite_configs* to return the compositors for requested *composite_names*.

    Args:
        composite_configs: the config files to load from
        composite_names: get compositors to fetch

    Returns:
        A list of loaded compositors
    """

    conf = {}
    files_processed = []
    for composite_config in composite_configs:
        if composite_config in files_processed:
            continue
        files_processed.append(composite_config)
        with open(composite_config) as conf_file:
            conf.update(yaml.load(conf_file))

    modifiers = {}

    for modifier_name, options in conf['modifiers'].items():

        try:
            loader = options.pop('compositor')
        except KeyError:
            raise ValueError(
                "'modifier' missing or empty in config files: %s" % (composite_configs,))
        options['name'] = modifier_name
        # Check if the caller only wants composites for a certain sensor
        if ('sensor' in options and
                sensor_names is not None and
                not (set(sensor_names) & set(options["sensor"]))):
            continue
        # Check if the caller only wants composites with certain names
        if composite_names is not None and modifier_name not in composite_names:
            continue

        # fix prerequisites in case of modifiers
        prereqs = []
        for item in options['prerequisites']:
            if isinstance(item, dict):
                # look into modifiers for matches and adapt the prerequisites
                # accordingly

                prereqs.append(item.keys()[0])
            else:
                prereqs.append(item)
        options['prerequisites'] = prereqs

        options.update(**kwargs)
        modifiers[modifier_name] = loader, options

    compositors = {}

    for composite_name, options in conf['composites'].items():

        try:
            loader = options.pop('compositor')
        except KeyError:
            raise ValueError(
                "'compositor' missing or empty in config files: %s" % (composite_configs,))
        options['name'] = composite_name
        # Check if the caller only wants composites for a certain sensor
        if ('sensor' in options and
                sensor_names is not None and
                not (set(sensor_names) & set(options["sensor"]))):
            continue
        # Check if the caller only wants composites with certain names
        if composite_names is not None and composite_name not in composite_names:
            continue

        # fix prerequisites in case of modifiers
        prereqs = []
        print options['prerequisites']
        for item in options['prerequisites']:
            if isinstance(item, dict):
                prereqs.append(item.keys()[0])
            else:
                prereqs.append(item)
        options['prerequisites'] = prereqs

        options.update(**kwargs)
        comp = loader(**options)
        compositors[composite_name] = comp
    print compositors
    return compositors, modifiers


def load_compositors(composite_names=None, sensor_names=None,
                     ppp_config_dir=CONFIG_PATH, **kwargs):
    """Load the requested *composite_names*.

    :param composite_names: The name of the desired composites
    :param sensor_names:  The name of the desired sensors to load composites for
    :param ppp_config_dir: The config directory
    :return: A list of loaded compositors
    """
    if sensor_names is None:
        sensor_names = []
    config_filenames = ["generic.yaml"] + \
        [sensor_name + ".yaml" for sensor_name in sensor_names]
    compositors = dict()
    modifiers = dict()
    for config_filename in config_filenames:
        LOG.debug("Looking for composites config file %s", config_filename)
        composite_configs = config_search_paths(os.path.join("composites",
                                                             config_filename),
                                                ppp_config_dir)
        if not composite_configs:
            LOG.debug("No composite config found called %s", config_filename)
            continue
        new_compositors, new_modifiers = _get_compositors_from_configs(composite_configs,
                                                                       composite_names,
                                                                       sensor_names,
                                                                       **kwargs)
        compositors.update(new_compositors)
        modifiers.update(new_modifiers)
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
            self.coszen[key] = np.ma.masked_outside(cos_zen(np.datetime64(projectable.info["start_time"]),
                                                            *projectable.info["area"].get_lonlats()),
                                                    0.035,  # about 88 degrees.
                                                    1,
                                                    copy=False)
        return sunzen_corr_cos(projectable, self.coszen[key])


class CO2Corrector(CompositeBase):

    def __call__(self, (ir_039, ir_108, ir_134), optional_datasets=None, **info):
        """CO2 correction of the brightness temperature of the MSG 3.9um
        channel.

        .. math::

          T4_CO2corr = (BT(IR3.9)^4 + Rcorr)^0.25
          Rcorr = BT(IR10.8)^4 - (BT(IR10.8)-dt_CO2)^4
          dt_CO2 = (BT(IR10.8)-BT(IR13.4))/4.0
        """

        dt_co2 = (ir_108 - ir_134) / 4.0
        rcorr = ir_108 ** 4 - (ir_108 - dt_co2) ** 4
        t4_co2corr = ir_039 ** 4 + rcorr
        t4_co2corr.data = np.ma.where(t4_co2corr > 0.0, t4_co2corr, 0)
        t4_co2corr = t4_co2corr ** 0.25

        t4_co2corr.info = ir_039.info.copy()
        t4_co2corr.info.setdefault('modifiers', []).append('co2_correction')

        return t4_co2corr


class RGBCompositor(CompositeBase):

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" %
                             (len(projectables),))
        the_data = np.rollaxis(np.ma.dstack(
            [projectable for projectable in projectables]), axis=2)
        #info = projectables[0].info.copy()
        # info.update(projectables[1].info)
        # info.update(projectables[2].info)
        info = combine_info(*projectables)
        info.update(self.info)
        info['id'] = DatasetID(self.info['name'])
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
            # FIXME: check the wavelength instead, so radiances can be
            # corrected too.
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
