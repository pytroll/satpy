#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016

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

from satpy.config import CONFIG_PATH, config_search_paths, glob_config
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


class CompositorLoader(object):
    """Read composites using the configuration files on disk.
    """

    def __init__(self, ppp_config_dir=CONFIG_PATH):
        from satpy.config import glob_config

        self.modifiers = {}
        self.compositors = {}
        self.ppp_config_dir = ppp_config_dir

    def load_sensor_composites(self, sensor_names):
        config_filenames = [sensor_name + ".yaml"
                            for sensor_name in sensor_names]
        for config_filename in config_filenames:
            LOG.debug("Looking for composites config file %s", config_filename)
            composite_configs = config_search_paths(
                os.path.join("composites", config_filename),
                self.ppp_config_dir)
            if not composite_configs:
                LOG.debug("No composite config found called %s",
                          config_filename)
                continue
            for composite_config in composite_configs:
                self._load_config(composite_config)

    def load_compositor(self, key, sensor_names):
        for sensor_name in sensor_names:
            if sensor_name not in self.compositors:
                self.load_sensor_composites([sensor_name])
            try:
                return self.compositors[sensor_name][key]
            except KeyError:
                continue
        raise KeyError

    def load_compositors(self, sensor_names):
        res = {}
        for sensor_name in sensor_names:
            if sensor_name not in self.compositors:
                self.load_sensor_composites([sensor_name])

            res.update(self.compositors[sensor_name])
        return res

    def _load_config(self, composite_config, **kwargs):
        with open(composite_config) as conf_file:
            conf = yaml.load(conf_file)

        try:
            sensor_name = conf['sensor_name']
        except KeyError:
            LOG.debug('No "sensor_name" tag found in %s, skipping.',
                      composite_config)
            return

        sensor_id = sensor_name.split('/')[-1]
        sensor_deps = sensor_name.split('/')[:-1]

        compositors = self.compositors.setdefault(sensor_id, {})
        modifiers = self.modifiers.setdefault(sensor_id, {})

        for sensor_dep in reversed(sensor_deps):
            if sensor_dep not in self.compositors or sensor_dep not in self.modifiers:
                self.load_sensor_composites([sensor_dep])

        try:
            compositors.update(self.compositors[sensor_deps[-1]])
            modifiers.update(self.modifiers[sensor_deps[-1]])
        except IndexError:
            # No deps, so no updating is needed
            pass

        for i in ['modifiers', 'composites']:
            if i not in conf:
                continue
            for composite_name, options in conf[i].items():
                try:
                    loader = options.pop('compositor')
                except KeyError:
                    raise ValueError("'compositor' missing or empty in %s" %
                                     composite_config)
                options['name'] = composite_name

                # fix prerequisites in case of modifiers
                prereqs = []
                for item in options.get('prerequisites', []):

                    if isinstance(item, dict):
                        #prereqs.append(item.keys()[0])
                        if len(item.keys()) > 1:
                            raise RuntimeError('Wrong prerequisite definition')
                        key = item.keys()[0]
                        mods = item.values()[0]
                        comp_name = key
                        for modifier in mods:
                            prev_comp_name = comp_name
                            comp_name = '_'.join((str(comp_name), modifier))

                            mloader, moptions = modifiers[modifier]
                            moptions = moptions.copy()
                            moptions.update(**kwargs)
                            moptions['name'] = comp_name
                            moptions['prerequisites'] = (
                                [prev_comp_name] + moptions['prerequisites'])
                            compositors[comp_name] = mloader(**moptions)
                        prereqs.append(comp_name)
                    else:
                        prereqs.append(item)
                options['prerequisites'] = prereqs

                if i == 'composites':
                    options.update(**kwargs)
                    comp = loader(**options)
                    compositors[composite_name] = comp
                elif i == 'modifiers':
                    modifiers[composite_name] = loader, options

        return conf


class CompositeBase(InfoObject):
    def __init__(self,
                 name,
                 prerequisites=[],
                 optional_prerequisites=[],
                 metadata_requirements=[],
                 **kwargs):
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


class SunZenithCorrector(CompositeBase):
    # FIXME: the cache should be cleaned up
    coszen = {}

    def __call__(self, projectables, **info):
        key = (projectables[0].info["start_time"],
               projectables[0].info["area"].name)
        LOG.debug("Applying sun zen correction")
        if len(projectables) == 1:
            if key not in self.coszen:
                from pyorbital.astronomy import cos_zen
                LOG.debug("Computing sun zenith angles.")
                self.coszen[key] = np.ma.masked_outside(cos_zen(np.datetime64(projectables[0].info["start_time"]),
                                                                *projectables[0].info["area"].get_lonlats()),
                                                        0.035,  # about 88 degrees.
                                                        1,
                                                        copy=False)
            coszen = self.coszen[key]
        else:
            coszen = np.cos(np.deg2rad(projectables[1]))
        return sunzen_corr_cos(projectables[0], coszen)


class CO2Corrector(CompositeBase):
    def __call__(self, projectables, optional_datasets=None, **info):
        """CO2 correction of the brightness temperature of the MSG 3.9um
        channel.

        .. math::

          T4_CO2corr = (BT(IR3.9)^4 + Rcorr)^0.25
          Rcorr = BT(IR10.8)^4 - (BT(IR10.8)-dt_CO2)^4
          dt_CO2 = (BT(IR10.8)-BT(IR13.4))/4.0
        """
        (ir_039, ir_108, ir_134) = projectables
        LOG.info('Applying CO2 correction')
        dt_co2 = (ir_108 - ir_134) / 4.0
        rcorr = ir_108**4 - (ir_108 - dt_co2)**4
        t4_co2corr = ir_039**4 + rcorr
        t4_co2corr = np.ma.where(t4_co2corr > 0.0, t4_co2corr, 0)
        t4_co2corr = t4_co2corr**0.25

        info = ir_039.info.copy()
        info.setdefault('modifiers', []).append('co2_correction')

        return Projectable(t4_co2corr, mask=t4_co2corr.mask, **info)


class RGBCompositor(CompositeBase):

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" %
                             (len(projectables), ))
        try:
            the_data = np.rollaxis(
                np.ma.dstack([projectable for projectable in projectables]),
                axis=2)
        except ValueError:
            raise IncompatibleAreas
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
        res = RGBCompositor.__call__(self, (projectables[0] - projectables[1],
                                            projectables[2] - projectables[3],
                                            projectables[0]), *args, **kwargs)
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
        res = RGBCompositor.__call__(self, (projectables[3] - projectables[4],
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
        res = RGBCompositor.__call__(self, (projectables[2] - projectables[1],
                                            projectables[1] - projectables[0],
                                            projectables[1]), *args, **kwargs)
        return res
