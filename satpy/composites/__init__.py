#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2018 PyTroll developers

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

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
from weakref import WeakValueDictionary

import numpy as np
import six
import xarray as xr
import xarray.ufuncs as xu
import yaml

from satpy.config import (CONFIG_PATH, config_search_paths,
                          recursive_dict_update)
from satpy.dataset import (DATASET_KEYS, Dataset, DatasetID, MetadataObject,
                           combine_metadata)
from satpy.readers import DatasetDict
from satpy.utils import sunzen_corr_cos, atmospheric_path_length_correction
from satpy.writers import get_enhanced_image
from satpy import CHUNK_SIZE

LOG = logging.getLogger(__name__)


class IncompatibleAreas(Exception):

    """
    Error raised upon compositing things of different shapes.
    """
    pass


class IncompatibleTimes(Exception):

    """
    Error raised upon compositing things from different times.
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


class CompositeBase(MetadataObject):

    def __init__(self,
                 name,
                 prerequisites=None,
                 optional_prerequisites=None,
                 **kwargs):
        # Required info
        kwargs["name"] = name
        kwargs["prerequisites"] = prerequisites or []
        kwargs["optional_prerequisites"] = optional_prerequisites or []
        super(CompositeBase, self).__init__(**kwargs)

    def __call__(self, datasets, optional_datasets=None, **info):
        raise NotImplementedError()

    def __str__(self):
        from pprint import pformat
        return pformat(self.attrs)

    def __repr__(self):
        from pprint import pformat
        return pformat(self.attrs)

    def apply_modifier_info(self, origin, destination):
        o = getattr(origin, 'attrs', origin)
        d = getattr(destination, 'attrs', destination)
        for k in DATASET_KEYS:
            if k == 'modifiers':
                d[k] = self.attrs[k]
            elif d.get(k) is None:
                if self.attrs.get(k) is not None:
                    d[k] = self.attrs[k]
                elif o.get(k) is not None:
                    d[k] = o[k]


class SunZenithCorrectorBase(CompositeBase):

    """Base class for sun zenith correction"""

    coszen = WeakValueDictionary()

    def __call__(self, projectables, **info):
        vis = projectables[0]
        if vis.attrs.get("sunz_corrected"):
            LOG.debug("Sun zen correction already applied")
            return vis

        if hasattr(vis.attrs["area"], 'name'):
            area_name = vis.attrs["area"].name
        else:
            area_name = 'swath' + str(vis.shape)
        key = (vis.attrs["start_time"], area_name)
        tic = time.time()
        LOG.debug("Applying sun zen correction")
        if len(projectables) == 1:
            coszen = self.coszen.get(key)
            if coszen is None:
                from pyorbital.astronomy import cos_zen
                LOG.debug("Computing sun zenith angles.")
                lons, lats = vis.attrs["area"].get_lonlats_dask(CHUNK_SIZE)

                coszen = xr.DataArray(cos_zen(vis.attrs["start_time"],
                                              lons, lats),
                                      dims=['y', 'x'],
                                      coords=[vis['y'], vis['x']])
                coszen = coszen.where((coszen > 0.035) & (coszen < 1))
                self.coszen[key] = coszen
        else:
            coszen = np.cos(np.deg2rad(projectables[1]))
            self.coszen[key] = coszen

        if vis.shape != coszen.shape:
            # assume we were given lower resolution szen data than band data
            LOG.debug(
                "Interpolating coszen calculations for higher resolution band")
            factor = int(vis.shape[1] / coszen.shape[1])
            coszen = np.repeat(
                np.repeat(coszen, factor, axis=0), factor, axis=1)

        proj = self._apply_correction(vis, coszen)
        proj.attrs = vis.attrs.copy()
        self.apply_modifier_info(vis, proj)
        LOG.debug(
            "Sun-zenith correction applied. Computation time: %5.1f (sec)",
            time.time() - tic)
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


class PSPRayleighReflectance(CompositeBase):

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the corrected reflectance when removing Rayleigh scattering.

        Uses pyspectral.
        """
        from pyspectral.rayleigh import Rayleigh

        (vis, red) = projectables
        if vis.shape != red.shape:
            raise IncompatibleAreas
        try:
            (sata, satz, suna, sunz) = optional_datasets
        except ValueError:
            from pyorbital.astronomy import get_alt_az, sun_zenith_angle
            from pyorbital.orbital import get_observer_look
            lons, lats = vis.attrs['area'].get_lonlats_dask(CHUNK_SIZE)
            sunalt, suna = get_alt_az(vis.attrs['start_time'], lons, lats)
            suna = np.rad2deg(suna)
            sunz = sun_zenith_angle(vis.attrs['start_time'], lons, lats)
            sata, satel = get_observer_look(vis.attrs['satellite_longitude'],
                                            vis.attrs['satellite_latitude'],
                                            vis.attrs['satellite_altitude'],
                                            vis.attrs['start_time'],
                                            lons, lats, 0)
            satz = 90 - satel
            del satel
        LOG.info('Removing Rayleigh scattering and aerosol absorption')

        # First make sure the two azimuth angles are in the range 0-360:
        sata = sata % 360.
        suna = suna % 360.
        ssadiff = abs(suna - sata)
        ssadiff = xu.minimum(ssadiff, 360 - ssadiff)
        del sata, suna

        atmosphere = self.attrs.get('atmosphere', 'us-standard')
        aerosol_type = self.attrs.get('aerosol_type', 'marine_clean_aerosol')

        corrector = Rayleigh(vis.attrs['platform_name'], vis.attrs['sensor'],
                             atmosphere=atmosphere,
                             aerosol_type=aerosol_type)

        try:
            refl_cor_band = corrector.get_reflectance(sunz.load().values,
                                                      satz.load().values,
                                                      ssadiff.load().values,
                                                      vis.attrs['name'],
                                                      red.values)
        except KeyError:
            LOG.warning("Could not get the reflectance correction using band name: %s", vis.attrs['name'])
            LOG.warning("Will try use the wavelength, however, this may be ambiguous!")
            refl_cor_band = corrector.get_reflectance(sunz.load().values,
                                                      satz.load().values,
                                                      ssadiff.load().values,
                                                      vis.attrs['wavelength'][1],
                                                      red.values)

        proj = vis - refl_cor_band
        proj.attrs = vis.attrs
        self.apply_modifier_info(vis, proj)
        return proj


class NIRReflectance(CompositeBase):

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the reflectance part of an NIR channel. Not supposed to be used
        for wavelength outside [3, 4] µm.
        """
        self._init_refl3x(projectables)
        _nir, _ = projectables
        proj = Dataset(self._get_reflectance(projectables, optional_datasets) * 100, **_nir.info)

        proj.info['units'] = '%'
        self.apply_modifier_info(_nir, proj)

        return proj

    def _init_refl3x(self, projectables):
        """Initiate the 3.x reflectance derivations
        """
        try:
            from pyspectral.near_infrared_reflectance import Calculator
        except ImportError:
            LOG.info("Couldn't load pyspectral")
            raise

        _nir, _tb11 = projectables
        self._refl3x = Calculator(_nir.attrs['platform_name'], _nir.attrs['sensor'], _nir.attrs['name'])

    def _get_reflectance(self, projectables, optional_datasets):
        """Calculate 3.x reflectance with pyspectral"""
        _nir, _tb11 = projectables
        LOG.info('Getting reflective part of %s', _nir.attrs['name'])

        sun_zenith = None
        tb13_4 = None

        for dataset in optional_datasets:
            if (dataset.attrs['units'] == 'K' and
                    "wavelengh" in dataset.attrs and
                    dataset.attrs["wavelength"][0] <= 13.4 <= dataset.attrs["wavelength"][2]):
                tb13_4 = dataset
            elif dataset.attrs["standard_name"] == "solar_zenith_angle":
                sun_zenith = dataset

        # Check if the sun-zenith angle was provided:
        if sun_zenith is None:
            from pyorbital.astronomy import sun_zenith_angle as sza
            lons, lats = _nir.attrs["area"].get_lonlats_dask(CHUNK_SIZE)
            sun_zenith = sza(_nir.attrs['start_time'], lons, lats)

        return self._refl3x.reflectance_from_tbs(sun_zenith, _nir, _tb11, tb_ir_co2=tb13_4)


class NIREmissivePartFromReflectance(NIRReflectance):

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the emissive part an NIR channel after having derived the reflectance.
        Not supposed to be used for wavelength outside [3, 4] µm.
        """
        self._init_refl3x(projectables)
        # Derive the sun-zenith angles, and use the nir and thermal ir
        # brightness tempertures and derive the reflectance using
        # PySpectral. The reflectance is stored internally in PySpectral and
        # needs to be derived first in order to get the emissive part.
        _ = self._get_reflectance(projectables, optional_datasets)
        _nir, _ = projectables
        proj = Dataset(self._refl3x.emissive_part_3x(), **_nir.attrs)

        proj.attrs['units'] = 'K'
        self.apply_modifier_info(_nir, proj)

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
            lons, lats = band.attrs['area'].get_lonlats_dask(CHUNK_SIZE)

            try:
                dummy, satel = get_observer_look(band.attrs['satellite_longitude'],
                                                 band.attrs[
                                                     'satellite_latitude'],
                                                 band.attrs[
                                                     'satellite_altitude'],
                                                 band.attrs['start_time'],
                                                 lons, lats, 0)
            except KeyError:
                raise KeyError(
                    'Band info is missing some meta data!')
            satz = 90 - satel
            del satel

        LOG.info('Correction for limb cooling')
        corrector = AtmosphericalCorrection(band.attrs['platform_name'],
                                            band.attrs['sensor'])

        atm_corr = corrector.get_correction(satz, band.attrs['name'], band)
        proj = band - atm_corr
        proj.attrs = band.attrs
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
        t4_co2corr = (ir_039**4 + rcorr).clip(0.0) ** 0.25

        t4_co2corr.attrs = ir_039.attrs.copy()

        self.apply_modifier_info(ir_039, t4_co2corr)

        return t4_co2corr


class DifferenceCompositor(CompositeBase):

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))
        info = combine_metadata(*projectables)
        info['name'] = self.attrs['name']

        return Dataset(projectables[0] - projectables[1], **info)


class GenericCompositor(CompositeBase):

    modes = {1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA'}

    def _concat_datasets(self, projectables, mode):
        try:
            data = xr.concat(projectables, 'bands')
            data['bands'] = list(mode)
        except ValueError:
            raise IncompatibleAreas
        else:
            areas = [projectable.attrs.get('area', None)
                     for projectable in projectables]
            areas = [area for area in areas if area is not None]
            if areas and areas.count(areas[0]) != len(areas):
                raise IncompatibleAreas

        return data

    def _get_sensors(self, projectables):
        sensor = set()
        for projectable in projectables:
            current_sensor = projectable.attrs.get("sensor", None)
            if current_sensor:
                if isinstance(current_sensor, (str, bytes, six.text_type)):
                    sensor.add(current_sensor)
                else:
                    sensor |= current_sensor
        if len(sensor) == 0:
            sensor = None
        elif len(sensor) == 1:
            sensor = list(sensor)[0]
        return sensor

    def _get_times(self, projectables):
        try:
            times = [proj['time'][0].values for proj in projectables]
        except KeyError:
            pass
        else:
            # Is there a more gracious way to handle this ?
            if np.max(times) - np.min(times) > np.timedelta64(1, 's'):
                raise IncompatibleTimes
            else:
                mid_time = (np.max(times) - np.min(times)) / 2 + np.min(times)
            return mid_time

    def __call__(self, projectables, nonprojectables=None, **attrs):

        num = len(projectables)
        mode = self.modes[num]
        if len(projectables) > 1:
            data = self._concat_datasets(projectables, mode)
        else:
            data = projectables[0]

        # if inputs have a time coordinate that may differ slightly between
        # themselves then find the mid time and use that as the single
        # time coordinate value
        time = self._get_times(projectables)
        if time is not None:
            data['time'] = [time]

        new_attrs = combine_metadata(*projectables)
        # remove metadata that shouldn't make sense in a composite
        new_attrs["wavelength"] = None
        new_attrs.pop("units", None)
        new_attrs.pop('calibration', None)
        new_attrs.pop('modifiers', None)

        new_attrs.update({key: val
                          for (key, val) in attrs.items()
                          if val is not None})
        new_attrs.update(self.attrs)
        new_attrs["sensor"] = self._get_sensors(projectables)
        new_attrs["mode"] = mode
        return xr.DataArray(data=data, attrs=new_attrs)


class RGBCompositor(GenericCompositor):

    def __call__(self, projectables, nonprojectables=None, **info):

        import warnings
        warnings.warn("RGBCompositor is deprecated, use GenericCompositor "
                      "instead.", DeprecationWarning)

        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" %
                             (len(projectables), ))
        return super(RGBCompositor, self).__call__(projectables, **info)


class BWCompositor(GenericCompositor):

    def __call__(self, projectables, nonprojectables=None, **info):

        import warnings
        warnings.warn("BWCompositor is deprecated, use GenericCompositor "
                      "instead.", DeprecationWarning)

        return super(BWCompositor, self).__call__(projectables, **info)


class ColormapCompositor(GenericCompositor):

    """A compositor that uses colormaps."""
    @staticmethod
    def build_colormap(palette, dtype, info):
        """Create the colormap from the `raw_palette` and the valid_range."""

        from trollimage.colormap import Colormap

        palette = np.asanyarray(palette).squeeze()
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
        palette = np.asanyarray(palette).squeeze()
        colormap = self.build_colormap(palette / 255.0, data.dtype, data.attrs)

        r, g, b = colormap.colorize(np.asanyarray(data))
        r[data.mask] = palette[-1][0]
        g[data.mask] = palette[-1][1]
        b[data.mask] = palette[-1][2]
        r = Dataset(r, copy=False, mask=data.mask, **data.attrs)
        g = Dataset(g, copy=False, mask=data.mask, **data.attrs)
        b = Dataset(b, copy=False, mask=data.mask, **data.attrs)

        return super(ColorizeCompositor, self).__call__((r, g, b), **data.attrs)


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
        palette = np.asanyarray(palette).squeeze() / 255.0
        colormap = self.build_colormap(palette, data.dtype, data.attrs)

        channels, colors = colormap.palettize(np.asanyarray(data.squeeze()))
        channels = palette[channels]

        r = xr.DataArray(channels[:, :, 0].reshape(data.shape),
                         dims=data.dims, coords=data.coords)
        g = xr.DataArray(channels[:, :, 1].reshape(data.shape),
                         dims=data.dims, coords=data.coords)
        b = xr.DataArray(channels[:, :, 2].reshape(data.shape),
                         dims=data.dims, coords=data.coords)

        return super(PaletteCompositor, self).__call__((r, g, b), **data.attrs)


class DayNightCompositor(GenericCompositor):

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

            res = super(DayNightCompositor, self).__call__((full_data[0],
                                                            full_data[1],
                                                            full_data[2]),
                                                           *args, **kwargs)

        except ValueError:
            raise IncompatibleAreas

        return res


def sub_arrays(proj1, proj2):
    """Substract two DataArrays and combine their attrs."""
    res = proj1 - proj2
    res.attrs = combine_metadata(proj1.attrs, proj2.attrs)
    if (res.attrs.get('area') is None and
            proj1.attrs.get('area') is not None and
            proj2.attrs.get('area') is not None):
        raise IncompatibleAreas
    return res


class Airmass(GenericCompositor):

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
            ch1 = sub_arrays(projectables[0], projectables[1])
            ch2 = sub_arrays(projectables[2], projectables[3])
            res = super(Airmass, self).__call__((ch1, ch2,
                                                projectables[0]),
                                                *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res


class Convection(GenericCompositor):

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
            ch1 = sub_arrays(projectables[3], projectables[4])
            ch2 = sub_arrays(projectables[2], projectables[5])
            ch3 = sub_arrays(projectables[1], projectables[0])
            res = super(Convection, self).__call__((ch1, ch2, ch3),
                                                   *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas
        return res


class Dust(GenericCompositor):

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

            ch1 = sub_arrays(projectables[2], projectables[1])
            ch2 = sub_arrays(projectables[1], projectables[0])
            res = super(Dust, self).__call__((ch1, ch2,
                                              projectables[1]),
                                             *args, **kwargs)
        except ValueError:
            raise IncompatibleAreas

        return res


class RealisticColors(GenericCompositor):

    def __call__(self, projectables, *args, **kwargs):
        try:

            vis06 = projectables[0]
            vis08 = projectables[1]
            hrv = projectables[2]

            try:
                ch3 = 3 * hrv - vis06 - vis08
                ch3.attrs = hrv.attrs
            except ValueError as err:
                raise IncompatibleAreas

            ndvi = (vis08 - vis06) / (vis08 + vis06)
            ndvi = np.where(ndvi < 0, 0, ndvi)

            ch1 = ndvi * vis06 + (1 - ndvi) * vis08
            ch1.attrs = vis06.attrs
            ch2 = ndvi * vis08 + (1 - ndvi) * vis06
            ch2.attrs = vis08.attrs

            res = super(RealisticColors, self).__call__((ch1, ch2, ch3),
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
