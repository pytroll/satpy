#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2017

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
import time

import numpy as np
import six
import yaml

from satpy.config import (CONFIG_PATH, config_search_paths,
                          recursive_dict_update)
from satpy.dataset import (DATASET_KEYS, Dataset, DatasetID, InfoObject,
                           combine_info)
from satpy.readers import DatasetDict
from satpy.tools import sunzen_corr_cos
from satpy.tools import atmospheric_path_length_correction
from satpy.writers import get_enhanced_image

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
        self.modifiers = {}
        self.compositors = {}
        self.ppp_config_dir = ppp_config_dir

    def load_sensor_composites(self, sensor_name):
        """Load all compositor configs for the provided sensor."""
        config_filename = sensor_name + ".yaml"
        LOG.debug("Looking for composites config file %s", config_filename)
        composite_configs = config_search_paths(
            os.path.join("composites", config_filename),
            self.ppp_config_dir, check_exists=True)
        if not composite_configs:
            LOG.debug("No composite config found called {}".format(
                config_filename))
            return
        self._load_config(composite_configs)

    def get_compositor(self, key, sensor_names):
        for sensor_name in sensor_names:
            try:
                return self.compositors[sensor_name][key]
            except KeyError:
                continue
        raise KeyError("Could not find compositor '{}'".format(key))

    def get_modifier(self, key, sensor_names):
        for sensor_name in sensor_names:
            try:
                return self.modifiers[sensor_name][key]
            except KeyError:
                continue
        raise KeyError("Could not find modifier '{}'".format(key))

    def load_compositors(self, sensor_names):
        """Load all compositor configs for the provided sensors.

        Args:
            sensor_names (list of strings): Sensor names that have matching
                                            ``sensor_name.yaml`` config files.

        Returns:
            (comps, mods): Where `comps` is a dictionary:

                    sensor_name -> composite ID -> compositor object

                And `mods` is a dictionary:

                    sensor_name -> modifier name -> (modifier class,
                    modifiers options)

                Note that these dictionaries are copies of those cached in
                this object.
        """
        comps = {}
        mods = {}
        for sensor_name in sensor_names:
            if sensor_name not in self.compositors:
                self.load_sensor_composites(sensor_name)
            if sensor_name in self.compositors:
                comps[sensor_name] = DatasetDict(
                    self.compositors[sensor_name].copy())
                mods[sensor_name] = self.modifiers[sensor_name].copy()
        return comps, mods

    def _process_composite_config(self, composite_name, conf,
                                  composite_type, sensor_id, composite_config, **kwargs):

        compositors = self.compositors[sensor_id]
        modifiers = self.modifiers[sensor_id]

        try:
            options = conf[composite_type][composite_name]
            loader = options.pop('compositor')
        except KeyError:
            if composite_name in compositors or composite_name in modifiers:
                return conf
            raise ValueError("'compositor' missing or empty in {0}. Option keys = {1}".format(
                composite_config, str(options.keys())))

        options['name'] = composite_name
        for prereq_type in ['prerequisites', 'optional_prerequisites']:
            prereqs = []
            for item in options.get(prereq_type, []):
                if isinstance(item, dict):
                    # we want this prerequisite to act as a query with
                    # 'modifiers' being None otherwise it will be an empty
                    # tuple
                    item.setdefault('modifiers', None)
                    key = DatasetID.from_dict(item)
                    prereqs.append(key)
                else:
                    prereqs.append(item)
            options[prereq_type] = prereqs

        if composite_type == 'composites':
            options.update(**kwargs)
            key = DatasetID.from_dict(options)
            comp = loader(**options)
            compositors[key] = comp
        elif composite_type == 'modifiers':
            modifiers[composite_name] = loader, options

    def _load_config(self, composite_configs, **kwargs):
        if not isinstance(composite_configs, (list, tuple)):
            composite_configs = [composite_configs]

        conf = {}
        for composite_config in composite_configs:
            with open(composite_config) as conf_file:
                conf = recursive_dict_update(conf, yaml.load(conf_file))
        try:
            sensor_name = conf['sensor_name']
        except KeyError:
            LOG.debug('No "sensor_name" tag found in %s, skipping.',
                      composite_config)
            return

        sensor_id = sensor_name.split('/')[-1]
        sensor_deps = sensor_name.split('/')[:-1]

        compositors = self.compositors.setdefault(sensor_id, DatasetDict())
        modifiers = self.modifiers.setdefault(sensor_id, {})

        for sensor_dep in reversed(sensor_deps):
            if sensor_dep not in self.compositors or sensor_dep not in self.modifiers:
                self.load_sensor_composites(sensor_dep)

        if sensor_deps:
            compositors.update(self.compositors[sensor_deps[-1]])
            modifiers.update(self.modifiers[sensor_deps[-1]])

        for composite_type in ['modifiers', 'composites']:
            if composite_type not in conf:
                continue
            for composite_name in conf[composite_type]:
                self._process_composite_config(composite_name, conf,
                                               composite_type, sensor_id, composite_config, **kwargs)


class CompositeBase(InfoObject):

    def __init__(self,
                 name,
                 prerequisites=None,
                 optional_prerequisites=None,
                 metadata_requirements=None,
                 **kwargs):
        # Required info
        kwargs["name"] = name
        kwargs["prerequisites"] = prerequisites or []
        kwargs["optional_prerequisites"] = optional_prerequisites or []
        kwargs["metadata_requirements"] = metadata_requirements or []
        super(CompositeBase, self).__init__(**kwargs)

    def __call__(self, datasets, optional_datasets=None, **info):
        raise NotImplementedError()

    def __str__(self):
        from pprint import pformat
        return pformat(self.info)

    def __repr__(self):
        from pprint import pformat
        return pformat(self.info)

    def apply_modifier_info(self, origin, destination):
        o = getattr(origin, 'info', origin)
        d = getattr(destination, 'info', destination)
        for k in DATASET_KEYS:
            if k == 'modifiers':
                d[k] = self.info[k]
            elif d.get(k) is None:
                if self.info.get(k) is not None:
                    d[k] = self.info[k]
                elif o.get(k) is not None:
                    d[k] = o[k]


class SunZenithCorrectorBase(CompositeBase):

    """Base class for sun zenith correction"""

    # FIXME: the cache should be cleaned up
    coszen = {}

    def __call__(self, projectables, **info):
        vis = projectables[0]
        if vis.info.get("sunz_corrected"):
            LOG.debug("Sun zen correction already applied")
            return vis

        if hasattr(vis.info["area"], 'name'):
            area_name = vis.info["area"].name
        else:
            area_name = 'swath' + str(vis.shape)
        key = (vis.info["start_time"], area_name)
        tic = time.time()
        LOG.debug("Applying sun zen correction")
        if len(projectables) == 1:
            if key not in self.coszen:
                from pyorbital.astronomy import cos_zen
                LOG.debug("Computing sun zenith angles.")
                self.coszen[key] = np.ma.masked_outside(cos_zen(vis.info["start_time"],
                                                                *vis.info["area"].get_lonlats()),
                                                        # about 88 degrees.
                                                        0.035,
                                                        1,
                                                        copy=False)
            coszen = self.coszen[key]
        else:
            coszen = np.cos(np.deg2rad(projectables[1]))

        if vis.shape != coszen.shape:
            # assume we were given lower resolution szen data than band data
            LOG.debug(
                "Interpolating coszen calculations for higher resolution band")
            factor = int(vis.shape[1] / coszen.shape[1])
            coszen = np.repeat(
                np.repeat(coszen, factor, axis=0), factor, axis=1)

        # sunz correction will be in place so we need a copy
        proj = vis.copy()
        proj = self._apply_correction(proj, coszen)
        vis.mask[coszen < 0] = True
        self.apply_modifier_info(vis, proj)
        LOG.debug(
            "Sun-zenith correction applied. Computation time: %5.1f (sec)", time.time() - tic)
        return proj

    def _apply_correction(self, proj, coszen):
        raise NotImplementedError("Correction method shall be defined!")


class SunZenithCorrector(SunZenithCorrectorBase):

    """Standard sun zenith correction, 1/cos(sunz)"""

    def _apply_correction(self, proj, coszen):
        LOG.debug("Apply the standard sun-zenith correction [1/cos(sunz)]")
        return sunzen_corr_cos(proj, coszen)


class EffectiveSolarPathLengthCorrector(SunZenithCorrectorBase):

    """Special sun zenith correction with the method proposed by Li and Shibata
    (2006): https://doi.org/10.1175/JAS3682.1
    """

    def _apply_correction(self, proj, coszen):
        LOG.debug(
            "Apply the effective solar atmospheric path length correction method by Li and Shibata")
        return atmospheric_path_length_correction(proj, coszen)


def show(data, filename=None):
    """Show the stretched data.
    """
    from PIL import Image as pil
    img = pil.fromarray(((data - data.min()) * 255.0 /
                         (data.max() - data.min())).astype(np.uint8))
    if filename is None:
        img.show()
    else:
        img.save(filename)


class PSPRayleighReflectance(CompositeBase):

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the corrected reflectance when removing Rayleigh scattering. Uses
        pyspectral.
        """
        from pyspectral.rayleigh import Rayleigh

        (vis, blue) = projectables
        if vis.shape != blue.shape:
            raise IncompatibleAreas
        try:
            (sata, satz, suna, sunz) = optional_datasets
        except ValueError:
            from pyorbital.astronomy import get_alt_az, sun_zenith_angle
            from pyorbital.orbital import get_observer_look
            sunalt, suna = get_alt_az(
                vis.info['start_time'], *vis.info['area'].get_lonlats())
            suna = np.rad2deg(suna)
            sunz = sun_zenith_angle(
                vis.info['start_time'], *vis.info['area'].get_lonlats())
            lons, lats = vis.info['area'].get_lonlats()
            sata, satel = get_observer_look(vis.info['satellite_longitude'],
                                            vis.info['satellite_latitude'],
                                            vis.info['satellite_altitude'],
                                            vis.info['start_time'],
                                            lons, lats, 0)
            satz = 90 - satel
            del satel
        LOG.info('Removing Rayleigh scattering and aerosol absorption')

        ssadiff = np.abs(suna - sata)
        ssadiff = np.where(ssadiff > 180, 360 - ssadiff, ssadiff)
        del sata, suna

        atmosphere = self.info.get('atmosphere', 'us-standard')
        aerosol_type = self.info.get('aerosol_type', 'marine_clean_aerosol')

        corrector = Rayleigh(vis.info['platform_name'], vis.info['sensor'],
                             atmosphere=atmosphere,
                             aerosol_type=aerosol_type)

        refl_cor_band = corrector.get_reflectance(
            sunz, satz, ssadiff, vis.id.wavelength[1], blue)

        proj = Dataset(vis - refl_cor_band,
                       copy=False,
                       **vis.info)
        self.apply_modifier_info(vis, proj)

        return proj


class NIRReflectance(CompositeBase):

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the reflectance part of an NIR channel. Not supposed to be used
        for wavelength outside [3, 4] µm.
        """
        try:
            from pyspectral.near_infrared_reflectance import Calculator
        except ImportError:
            LOG.info("Couldn't load pyspectral")
            raise

        nir, tb11 = projectables
        LOG.info('Getting reflective part of %s', nir.info['name'])

        sun_zenith = None
        tb13_4 = None

        for dataset in optional_datasets:
            if (dataset.info['units'] == 'K' and
                    "wavelengh" in dataset.info and
                    dataset.info["wavelength"][0] <= 13.4 <= dataset.info["wavelength"][2]):
                tb13_4 = dataset
            elif dataset.info["standard_name"] == "solar_zenith_angle":
                sun_zenith = dataset

        # Check if the sun-zenith angle was provided:
        if sun_zenith is None:
            from pyorbital.astronomy import sun_zenith_angle as sza
            lons, lats = nir.info["area"].get_lonlats()
            sun_zenith = sza(nir.info['start_time'], lons, lats)

        refl39 = Calculator(nir.info['platform_name'],
                            nir.info['sensor'], nir.id.wavelength[1])

        proj = Dataset(refl39.reflectance_from_tbs(sun_zenith, nir,
                                                   tb11, tb13_4) * 100,
                       **nir.info)
        proj.info['units'] = '%'
        self.apply_modifier_info(nir, proj)

        return proj


class PSPAtmosphericalCorrection(CompositeBase):

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the atmospherical correction. Uses pyspectral.
        """
        from pyspectral.atm_correction_ir import AtmosphericalCorrection

        band = projectables[0]

        if optional_datasets:
            satz = optional_datasets[0]
        else:
            from pyorbital.orbital import get_observer_look
            lons, lats = band.info['area'].get_lonlats()

            try:
                dummy, satel = get_observer_look(band.info['satellite_longitude'],
                                                 band.info[
                                                     'satellite_latitude'],
                                                 band.info[
                                                     'satellite_altitude'],
                                                 band.info['start_time'],
                                                 lons, lats, 0)
            except KeyError:
                raise KeyError(
                    'Band info is missing some meta data!')
            satz = 90 - satel
            del satel

        LOG.info('Correction for limb cooling')
        corrector = AtmosphericalCorrection(band.info['platform_name'],
                                            band.info['sensor'])

        atm_corr = corrector.get_correction(satz, band.info['name'], band)

        proj = Dataset(atm_corr,
                       copy=False,
                       **band.info)
        self.apply_modifier_info(band, proj)

        return proj


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

        proj = Dataset(t4_co2corr, mask=t4_co2corr.mask, **info)

        self.apply_modifier_info(ir_039, proj)

        return proj


class DifferenceCompositor(CompositeBase):

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))
        info = combine_info(*projectables)
        info['name'] = self.info['name']

        return Dataset(projectables[0] - projectables[1], **info)


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
        else:
            areas = [projectable.info.get('area', None)
                     for projectable in projectables]
            areas = [area for area in areas if area is not None]
            if areas and areas.count(areas[0]) != len(areas):
                raise IncompatibleAreas

        info = combine_info(*projectables)
        info.update(self.info)
        # FIXME: should this be done here ?
        info["wavelength"] = None
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
        return Dataset(data=the_data, **info)


class BWCompositor(CompositeBase):

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 1:
            raise ValueError("Expected 1 dataset, got %d" %
                             (len(projectables), ))

        info = combine_info(*projectables)
        info['name'] = self.info['name']
        info['standard_name'] = self.info['standard_name']

        return Dataset(projectables[0], **info)


class ColormapCompositor(RGBCompositor):

    """A compositor that uses colormaps."""
    @staticmethod
    def build_colormap(palette, dtype, info):
        """Create the colormap from the `raw_palette` and the valid_range."""

        from trollimage.colormap import Colormap
        if dtype == np.dtype('uint8'):
            tups = [(val, tuple(tup))
                    for (val, tup) in enumerate(palette[:-1])]
            colormap = Colormap(*tups)

        elif 'valid_range' in info:
            tups = [(val, tuple(tup))
                    for (val, tup) in enumerate(palette[:-1])]
            colormap = Colormap(*tups)

            sf = info['scale_factor']
            colormap.set_range(
                *info['valid_range'] * sf + info['add_offset'])

        return colormap


class ColorizeCompositor(ColormapCompositor):

    """A compositor colorizing the data, interpolating the palette colors when
    needed.
    """

    def __call__(self, projectables, **info):
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))

        # TODO: support datasets with palette to delegate this to the image
        # writer.

        data, palette = projectables
        colormap = self.build_colormap(palette / 255.0, data.dtype, data.info)

        r, g, b = colormap.colorize(data)
        r[data.mask] = palette[-1][0]
        g[data.mask] = palette[-1][1]
        b[data.mask] = palette[-1][2]
        r = Dataset(r, copy=False, mask=data.mask, **data.info)
        g = Dataset(g, copy=False, mask=data.mask, **data.info)
        b = Dataset(b, copy=False, mask=data.mask, **data.info)

        return super(ColorizeCompositor, self).__call__((r, g, b), **data.info)


class PaletteCompositor(ColormapCompositor):

    """A compositor colorizing the data, not interpolating the palette colors.
    """

    def __call__(self, projectables, **info):
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))

        # TODO: support datasets with palette to delegate this to the image
        # writer.

        data, palette = projectables
        palette = palette / 255.0
        colormap = self.build_colormap(palette, data.dtype, data.info)

        channels, colors = colormap.palettize(data)
        channels = palette[channels]

        r = Dataset(channels[:, :, 0], copy=False, mask=data.mask, **data.info)
        g = Dataset(channels[:, :, 1], copy=False, mask=data.mask, **data.info)
        b = Dataset(channels[:, :, 2], copy=False, mask=data.mask, **data.info)

        return super(PaletteCompositor, self).__call__((r, g, b), **data.info)


class DayNightCompositor(RGBCompositor):

    """A compositor that takes one composite on the night side, another on day
    side, and then blends them together."""

    def __call__(self, projectables, lim_low=85., lim_high=95., *args,
                 **kwargs):
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" %
                             (len(projectables), ))
        try:
            day_data = projectables[0].copy()
            night_data = projectables[1].copy()
            coszen = np.cos(np.deg2rad(projectables[2]))

            coszen -= min(np.cos(np.deg2rad(lim_high)),
                          np.cos(np.deg2rad(lim_low)))
            coszen /= np.abs(np.cos(np.deg2rad(lim_low)) -
                             np.cos(np.deg2rad(lim_high)))
            coszen = np.clip(coszen, 0, 1)

            full_data = []

            # Apply enhancements
            day_data = enhance2dataset(day_data)
            night_data = enhance2dataset(night_data)

            # Match dimensions to the data with more channels
            # There are only 1-channel and 3-channel composites
            if day_data.shape[0] > night_data.shape[0]:
                night_data = np.ma.repeat(night_data, 3, 0)
            elif day_data.shape[0] < night_data.shape[0]:
                day_data = np.ma.repeat(day_data, 3, 0)

            for i in range(day_data.shape[0]):
                day = day_data[i, :, :]
                night = night_data[i, :, :]

                data = (1 - coszen) * np.ma.masked_invalid(night).filled(0) + \
                    coszen * np.ma.masked_invalid(day).filled(0)
                data = np.ma.array(data, mask=np.logical_and(night.mask,
                                                             day.mask),
                                   copy=False)
                data = Dataset(np.ma.masked_invalid(data),
                               copy=True,
                               **projectables[0].info)
                full_data.append(data)

            res = RGBCompositor.__call__(self, (full_data[0],
                                                full_data[1],
                                                full_data[2]),
                                         *args, **kwargs)

        except ValueError:
            raise IncompatibleAreas

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
        try:
            res = RGBCompositor.__call__(self, (projectables[0] - projectables[1],
                                                projectables[2] -
                                                projectables[3],
                                                projectables[0]), *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
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
        try:
            res = RGBCompositor.__call__(self, (projectables[3] - projectables[4],
                                                projectables[2] -
                                                projectables[5],
                                                projectables[1] - projectables[0]),
                                         *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res


class Dust(RGBCompositor):

    def __call__(self, projectables, *args, **kwargs):
        """Make a dust (or fog or night_fog) RGB image composite.

        Fog:
        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | IR12.0 - IR10.8    |     -4 to 2 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR10.8 - IR8.7     |      0 to 6 K      | gamma 2.0          |
        +--------------------+--------------------+--------------------+
        | IR10.8             |   243 to 283 K     | gamma 1            |
        +--------------------+--------------------+--------------------+

        Dust:
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
        try:
            res = RGBCompositor.__call__(self, (projectables[2] - projectables[1],
                                                projectables[1] -
                                                projectables[0],
                                                projectables[1]), *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas

        return res


class RealisticColors(RGBCompositor):

    def __call__(self, projectables, *args, **kwargs):
        try:

            vis06 = projectables[0]
            vis08 = projectables[1]
            hrv = projectables[2]

            ndvi = (vis08 - vis06) / (vis08 + vis06)
            ndvi = np.where(ndvi < 0, 0, ndvi)

            # info = combine_info(*projectables)
            # info['name'] = self.info['name']
            # info['standard_name'] = self.info['standard_name']

            ch1 = Dataset(ndvi * vis06 + (1 - ndvi) * vis08,
                          copy=False,
                          **vis06.info)
            ch2 = Dataset(ndvi * vis08 + (1 - ndvi) * vis06,
                          copy=False,
                          **vis08.info)
            ch3 = Dataset(3 * hrv - vis06 - vis08,
                          copy=False,
                          **hrv.info)

            res = RGBCompositor.__call__(self, (ch1, ch2, ch3),
                                         *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res


def enhance2dataset(dset):
    """Apply enhancements to dataset *dset* and convert the image data
    back to Dataset object."""
    img = get_enhanced_image(dset)

    data = np.rollaxis(np.dstack(img.channels), axis=2)
    mask = dset.mask
    if mask.ndim < data.ndim:
        mask = np.expand_dims(mask, 0)
        mask = np.repeat(mask, 3, 0)
    elif mask.ndim > data.ndim:
        mask = mask[0, :, :]
    data = Dataset(np.ma.masked_array(data, mask=mask),
                   copy=False,
                   **dset.info)
    return data
