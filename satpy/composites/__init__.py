#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2019 PyTroll developers

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
import warnings
from weakref import WeakValueDictionary

import dask.array as da
import numpy as np
import six
import xarray as xr
import xarray.ufuncs as xu
import yaml

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader

from satpy.config import CONFIG_PATH, config_search_paths, recursive_dict_update
from satpy.dataset import DATASET_KEYS, DatasetID, MetadataObject, combine_metadata
from satpy.readers import DatasetDict
from satpy.utils import sunzen_corr_cos, atmospheric_path_length_correction
from satpy.writers import get_enhanced_image
from satpy import CHUNK_SIZE

try:
    from pyspectral.near_infrared_reflectance import Calculator
except ImportError:
    Calculator = None
try:
    from pyorbital.astronomy import sun_zenith_angle
except ImportError:
    sun_zenith_angle = None


LOG = logging.getLogger(__name__)


class IncompatibleAreas(Exception):
    """Error raised upon compositing things of different shapes."""
    pass


class IncompatibleTimes(Exception):
    """Error raised upon compositing things from different times."""
    pass


class CompositorLoader(object):
    """Read composites using the configuration files on disk."""

    def __init__(self, ppp_config_dir=None):
        if ppp_config_dir is None:
            ppp_config_dir = CONFIG_PATH
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
            dep_num = 0
            for item in options.get(prereq_type, []):
                if isinstance(item, dict):
                    # Handle in-line composites
                    if 'compositor' in item:
                        # Create an unique temporary name for the composite
                        sub_comp_name = '_' + composite_name + '_dep_{}'.format(dep_num)
                        dep_num += 1
                        # Minimal composite config
                        sub_conf = {composite_type: {sub_comp_name: item}}
                        self._process_composite_config(
                            sub_comp_name, sub_conf, composite_type, sensor_id,
                            composite_config, **kwargs)
                    else:
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
                conf = recursive_dict_update(conf, yaml.load(conf_file, Loader=UnsafeLoader))
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


def check_times(projectables):
    times = []
    for proj in projectables:
        try:
            if proj['time'].size and proj['time'][0] != 0:
                times.append(proj['time'][0].values)
            else:
                break  # right?
        except KeyError:
            # the datasets don't have times
            break
        except IndexError:
            # time is a scalar
            if proj['time'].values != 0:
                times.append(proj['time'].values)
            else:
                break
    else:
        # Is there a more gracious way to handle this ?
        if np.max(times) - np.min(times) > np.timedelta64(1, 's'):
            raise IncompatibleTimes
        else:
            mid_time = (np.max(times) - np.min(times)) / 2 + np.min(times)
        return mid_time


def sub_arrays(proj1, proj2):
    """Substract two DataArrays and combine their attrs."""
    attrs = combine_metadata(proj1.attrs, proj2.attrs)
    if (attrs.get('area') is None and
            proj1.attrs.get('area') is not None and
            proj2.attrs.get('area') is not None):
        raise IncompatibleAreas
    res = proj1 - proj2
    res.attrs = attrs
    return res


class CompositeBase(MetadataObject):

    def __init__(self, name, prerequisites=None, optional_prerequisites=None, **kwargs):
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

    def check_areas(self, data_arrays):
        if len(data_arrays) == 1:
            return data_arrays

        if 'x' in data_arrays[0].dims and \
                not all(x.sizes['x'] == data_arrays[0].sizes['x']
                        for x in data_arrays[1:]):
            raise IncompatibleAreas("X dimension has different sizes")
        if 'y' in data_arrays[0].dims and \
                not all(x.sizes['y'] == data_arrays[0].sizes['y']
                        for x in data_arrays[1:]):
            raise IncompatibleAreas("Y dimension has different sizes")

        areas = [ds.attrs.get('area') for ds in data_arrays]
        if all(a is None for a in areas):
            return data_arrays
        elif any(a is None for a in areas):
            raise ValueError("Missing 'area' attribute")

        if not all(areas[0] == x for x in areas[1:]):
            LOG.debug("Not all areas are the same in "
                      "'{}'".format(self.attrs['name']))
            raise IncompatibleAreas("Areas are different")

        return data_arrays


class SunZenithCorrectorBase(CompositeBase):
    """Base class for sun zenith correction."""

    coszen = WeakValueDictionary()

    def __init__(self, max_sza=95.0, **kwargs):
        """Collect custom configuration values.

        Args:
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.max_sza = max_sza
        self.max_sza_cos = np.cos(np.deg2rad(max_sza)) if max_sza is not None else None
        super(SunZenithCorrectorBase, self).__init__(**kwargs)

    def __call__(self, projectables, **info):
        projectables = self.check_areas(projectables)
        vis = projectables[0]
        if vis.attrs.get("sunz_corrected"):
            LOG.debug("Sun zen correction already applied")
            return vis

        area_name = hash(vis.attrs['area'])
        key = (vis.attrs["start_time"], area_name)
        tic = time.time()
        LOG.debug("Applying sun zen correction")
        coszen = self.coszen.get(key)
        if coszen is None and len(projectables) == 1:
            # we were not given SZA, generate SZA then calculate cos(SZA)
            from pyorbital.astronomy import cos_zen
            LOG.debug("Computing sun zenith angles.")
            lons, lats = vis.attrs["area"].get_lonlats_dask(CHUNK_SIZE)

            coords = {}
            if 'y' in vis.coords and 'x' in vis.coords:
                coords['y'] = vis['y']
                coords['x'] = vis['x']
            coszen = xr.DataArray(cos_zen(vis.attrs["start_time"], lons, lats),
                                  dims=['y', 'x'], coords=coords)
            if self.max_sza is not None:
                coszen = coszen.where(coszen >= self.max_sza_cos)
            self.coszen[key] = coszen
        elif coszen is None:
            # we were given the SZA, calculate the cos(SZA)
            coszen = np.cos(np.deg2rad(projectables[1]))
            self.coszen[key] = coszen

        proj = self._apply_correction(vis, coszen)
        proj.attrs = vis.attrs.copy()
        self.apply_modifier_info(vis, proj)
        LOG.debug("Sun-zenith correction applied. Computation time: %5.1f (sec)", time.time() - tic)
        return proj

    def _apply_correction(self, proj, coszen):
        raise NotImplementedError("Correction method shall be defined!")


class SunZenithCorrector(SunZenithCorrectorBase):
    """Standard sun zenith correction using ``1 / cos(sunz)``.

    In addition to adjusting the provided reflectances by the cosine of the
    solar zenith angle, this modifier forces all reflectances beyond a
    solar zenith angle of ``max_sza`` to 0. It also gradually reduces the
    amount of correction done between ``correction_limit`` and ``max_sza``. If
    ``max_sza`` is ``None`` then a constant correction is applied to zenith
    angles beyond ``correction_limit``.

    To set ``max_sza`` to ``None`` in a YAML configuration file use:

    .. code-block:: yaml

      sunz_corrected:
        compositor: !!python/name:satpy.composites.SunZenithCorrector
        max_sza: !!null
        optional_prerequisites:
        - solar_zenith_angle

    """

    def __init__(self, correction_limit=88., **kwargs):
        """Collect custom configuration values.

        Args:
            correction_limit (float): Maximum solar zenith angle to apply the
                correction in degrees. Pixels beyond this limit have a
                constant correction applied. Default 88.
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.correction_limit = correction_limit
        super(SunZenithCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        LOG.debug("Apply the standard sun-zenith correction [1/cos(sunz)]")
        return sunzen_corr_cos(proj, coszen, limit=self.correction_limit, max_sza=self.max_sza)


class EffectiveSolarPathLengthCorrector(SunZenithCorrectorBase):
    """Special sun zenith correction with the method proposed by Li and Shibata.

    (2006): https://doi.org/10.1175/JAS3682.1

    In addition to adjusting the provided reflectances by the cosine of the
    solar zenith angle, this modifier forces all reflectances beyond a
    solar zenith angle of `max_sza` to 0 to reduce noise in the final data.
    It also gradually reduces the amount of correction done between
    ``correction_limit`` and ``max_sza``. If ``max_sza`` is ``None`` then a
    constant correction is applied to zenith angles beyond
    ``correction_limit``.

    To set ``max_sza`` to ``None`` in a YAML configuration file use:

    .. code-block:: yaml

      effective_solar_pathlength_corrected:
        compositor: !!python/name:satpy.composites.EffectiveSolarPathLengthCorrector
        max_sza: !!null
        optional_prerequisites:
        - solar_zenith_angle

    """

    def __init__(self, correction_limit=88., **kwargs):
        """Collect custom configuration values.

        Args:
            correction_limit (float): Maximum solar zenith angle to apply the
                correction in degrees. Pixels beyond this limit have a
                constant correction applied. Default 88.
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.correction_limit = correction_limit
        super(EffectiveSolarPathLengthCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        LOG.debug("Apply the effective solar atmospheric path length correction method by Li and Shibata")
        return atmospheric_path_length_correction(proj, coszen, limit=self.correction_limit, max_sza=self.max_sza)


class PSPRayleighReflectance(CompositeBase):

    _rayleigh_cache = WeakValueDictionary()

    def get_angles(self, vis):
        from pyorbital.astronomy import get_alt_az, sun_zenith_angle
        from pyorbital.orbital import get_observer_look

        lons, lats = vis.attrs['area'].get_lonlats_dask(
            chunks=vis.data.chunks)

        sunalt, suna = get_alt_az(vis.attrs['start_time'], lons, lats)
        suna = xu.rad2deg(suna)
        sunz = sun_zenith_angle(vis.attrs['start_time'], lons, lats)
        sata, satel = get_observer_look(
            vis.attrs['satellite_longitude'],
            vis.attrs['satellite_latitude'],
            vis.attrs['satellite_altitude'],
            vis.attrs['start_time'],
            lons, lats, 0)
        satz = 90 - satel
        return sata, satz, suna, sunz

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the corrected reflectance when removing Rayleigh scattering.

        Uses pyspectral.
        """
        from pyspectral.rayleigh import Rayleigh
        if not optional_datasets or len(optional_datasets) != 4:
            vis, red = self.check_areas(projectables)
            sata, satz, suna, sunz = self.get_angles(vis)
            red.data = da.rechunk(red.data, vis.data.chunks)
        else:
            vis, red, sata, satz, suna, sunz = self.check_areas(
                projectables + optional_datasets)
            sata, satz, suna, sunz = optional_datasets
            # get the dask array underneath
            sata = sata.data
            satz = satz.data
            suna = suna.data
            sunz = sunz.data

        # First make sure the two azimuth angles are in the range 0-360:
        sata = sata % 360.
        suna = suna % 360.
        ssadiff = da.absolute(suna - sata)
        ssadiff = da.minimum(ssadiff, 360 - ssadiff)
        del sata, suna

        atmosphere = self.attrs.get('atmosphere', 'us-standard')
        aerosol_type = self.attrs.get('aerosol_type', 'marine_clean_aerosol')
        rayleigh_key = (vis.attrs['platform_name'],
                        vis.attrs['sensor'], atmosphere, aerosol_type)
        LOG.info("Removing Rayleigh scattering with atmosphere '{}' and aerosol type '{}' for '{}'".format(
            atmosphere, aerosol_type, vis.attrs['name']))
        if rayleigh_key not in self._rayleigh_cache:
            corrector = Rayleigh(vis.attrs['platform_name'], vis.attrs['sensor'],
                                 atmosphere=atmosphere,
                                 aerosol_type=aerosol_type)
            self._rayleigh_cache[rayleigh_key] = corrector
        else:
            corrector = self._rayleigh_cache[rayleigh_key]

        try:
            refl_cor_band = corrector.get_reflectance(sunz, satz, ssadiff,
                                                      vis.attrs['name'],
                                                      red.data)
        except (KeyError, IOError):
            LOG.warning("Could not get the reflectance correction using band name: %s", vis.attrs['name'])
            LOG.warning("Will try use the wavelength, however, this may be ambiguous!")
            refl_cor_band = corrector.get_reflectance(sunz, satz, ssadiff,
                                                      vis.attrs['wavelength'][1],
                                                      red.data)
        proj = vis - refl_cor_band
        proj.attrs = vis.attrs
        self.apply_modifier_info(vis, proj)
        return proj


class NIRReflectance(CompositeBase):

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the reflectance part of an NIR channel.

        Not supposed to be used for wavelength outside [3, 4] µm.

        """
        self._init_refl3x(projectables)
        _nir, _ = projectables
        refl = self._get_reflectance(projectables, optional_datasets) * 100
        proj = xr.DataArray(refl, dims=_nir.dims,
                            coords=_nir.coords, attrs=_nir.attrs)

        proj.attrs['units'] = '%'
        self.apply_modifier_info(_nir, proj)

        return proj

    def _init_refl3x(self, projectables):
        """Initiate the 3.x reflectance derivations."""
        if not Calculator:
            LOG.info("Couldn't load pyspectral")
            raise ImportError("No module named pyspectral.near_infrared_reflectance")
        _nir, _tb11 = projectables
        self._refl3x = Calculator(_nir.attrs['platform_name'], _nir.attrs['sensor'], _nir.attrs['name'])

    def _get_reflectance(self, projectables, optional_datasets):
        """Calculate 3.x reflectance with pyspectral."""
        _nir, _tb11 = projectables
        LOG.info('Getting reflective part of %s', _nir.attrs['name'])

        sun_zenith = None
        tb13_4 = None

        for dataset in optional_datasets:
            wavelengths = dataset.attrs.get('wavelength', [100., 0, 0])
            if (dataset.attrs.get('units') == 'K' and
                    wavelengths[0] <= 13.4 <= wavelengths[2]):
                tb13_4 = dataset
            elif ("standard_name" in dataset.attrs and
                  dataset.attrs["standard_name"] == "solar_zenith_angle"):
                sun_zenith = dataset

        # Check if the sun-zenith angle was provided:
        if sun_zenith is None:
            if sun_zenith_angle is None:
                raise ImportError("No module named pyorbital.astronomy")
            lons, lats = _nir.attrs["area"].get_lonlats_dask(CHUNK_SIZE)
            sun_zenith = sun_zenith_angle(_nir.attrs['start_time'], lons, lats)

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
        proj = xr.DataArray(self._refl3x.emissive_part_3x(), attrs=_nir.attrs,
                            dims=_nir.dims, coords=_nir.coords)

        proj.attrs['units'] = 'K'
        self.apply_modifier_info(_nir, proj)

        return proj


class PSPAtmosphericalCorrection(CompositeBase):

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the atmospherical correction. Uses pyspectral."""
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
        """CO2 correction of the brightness temperature of the MSG 3.9um channel.

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
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.check_areas(projectables)
        info = combine_metadata(*projectables)
        info['name'] = self.attrs['name']

        proj = projectables[0] - projectables[1]
        proj.attrs = info
        return proj


class GenericCompositor(CompositeBase):

    modes = {1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA'}

    def __init__(self, name, common_channel_mask=True, **kwargs):
        """Collect custom configuration values.

        Args:
            common_channel_mask (bool): If True, mask all the channels with
                a mask that combines all the invalid areas of the given data.
        """
        self.common_channel_mask = common_channel_mask
        super(GenericCompositor, self).__init__(name, **kwargs)

    def _concat_datasets(self, projectables, mode):
        try:
            data = xr.concat(projectables, 'bands', coords='minimal')
            data['bands'] = list(mode)
        except ValueError as e:
            LOG.debug("Original exception for incompatible areas: {}".format(str(e)))
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

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """Build the composite."""
        num = len(projectables)
        mode = attrs.get('mode')
        if mode is None:
            # num may not be in `self.modes` so only check if we need to
            mode = self.modes[num]
        if len(projectables) > 1:
            projectables = self.check_areas(projectables)
            data = self._concat_datasets(projectables, mode)
            # Skip masking if user wants it or a specific alpha channel is given.
            if self.common_channel_mask and mode[-1] != 'A':
                data = data.where(data.notnull().all(dim='bands'))
        else:
            data = projectables[0]

        # if inputs have a time coordinate that may differ slightly between
        # themselves then find the mid time and use that as the single
        # time coordinate value
        if len(projectables) > 1:
            time = check_times(projectables)
            if time is not None and 'time' in data.dims:
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

        return xr.DataArray(data=data.data, attrs=new_attrs,
                            dims=data.dims, coords=data.coords)


class FillingCompositor(GenericCompositor):
    """Make a regular RGB, filling the RGB bands with the first provided dataset's values."""

    def __call__(self, projectables, nonprojectables=None, **info):
        projectables[1] = projectables[1].fillna(projectables[0])
        projectables[2] = projectables[2].fillna(projectables[0])
        projectables[3] = projectables[3].fillna(projectables[0])
        return super(FillingCompositor, self).__call__(projectables[1:], **info)


class RGBCompositor(GenericCompositor):

    def __call__(self, projectables, nonprojectables=None, **info):
        warnings.warn("RGBCompositor is deprecated, use GenericCompositor instead.", DeprecationWarning)
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(projectables),))
        return super(RGBCompositor, self).__call__(projectables, **info)


class BWCompositor(GenericCompositor):

    def __call__(self, projectables, nonprojectables=None, **info):
        warnings.warn("BWCompositor is deprecated, use GenericCompositor instead.", DeprecationWarning)
        return super(BWCompositor, self).__call__(projectables, **info)


class ColormapCompositor(GenericCompositor):
    """A compositor that uses colormaps."""

    @staticmethod
    def build_colormap(palette, dtype, info):
        """Create the colormap from the `raw_palette` and the valid_range."""
        from trollimage.colormap import Colormap
        sqpalette = np.asanyarray(palette).squeeze() / 255.0
        if hasattr(palette, 'attrs') and 'palette_meanings' in palette.attrs:
            meanings = palette.attrs['palette_meanings']
            iterator = zip(meanings, sqpalette)
        else:
            iterator = enumerate(sqpalette[:-1])

        if dtype == np.dtype('uint8'):
            tups = [(val, tuple(tup))
                    for (val, tup) in iterator]
            colormap = Colormap(*tups)

        elif 'valid_range' in info:
            tups = [(val, tuple(tup))
                    for (val, tup) in iterator]
            colormap = Colormap(*tups)

            sf = info.get('scale_factor', np.array(1))
            colormap.set_range(
                *info['valid_range'] * sf + info.get('add_offset', 0))

        return colormap, sqpalette


class ColorizeCompositor(ColormapCompositor):
    """A compositor colorizing the data, interpolating the palette colors when needed."""

    def __call__(self, projectables, **info):
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))

        # TODO: support datasets with palette to delegate this to the image
        # writer.

        data, palette = projectables
        colormap, palette = self.build_colormap(palette, data.dtype, data.attrs)

        r, g, b = colormap.colorize(np.asanyarray(data))
        r[data.mask] = palette[-1][0]
        g[data.mask] = palette[-1][1]
        b[data.mask] = palette[-1][2]
        raise NotImplementedError("This compositor wasn't fully converted to dask yet.")

        # r = Dataset(r, copy=False, mask=data.mask, **data.attrs)
        # g = Dataset(g, copy=False, mask=data.mask, **data.attrs)
        # b = Dataset(b, copy=False, mask=data.mask, **data.attrs)
        #
        # return super(ColorizeCompositor, self).__call__((r, g, b), **data.attrs)


class PaletteCompositor(ColormapCompositor):
    """A compositor colorizing the data, not interpolating the palette colors."""

    def __call__(self, projectables, **info):
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))

        # TODO: support datasets with palette to delegate this to the image
        # writer.

        data, palette = projectables
        colormap, palette = self.build_colormap(palette, data.dtype, data.attrs)

        channels, colors = colormap.palettize(np.asanyarray(data.squeeze()))
        channels = palette[channels]
        fill_value = data.attrs.get('_FillValue', np.nan)
        if np.isnan(fill_value):
            mask = data.notnull()
        else:
            mask = data != data.attrs['_FillValue']
        r = xr.DataArray(channels[:, :, 0].reshape(data.shape),
                         dims=data.dims, coords=data.coords,
                         attrs=data.attrs).where(mask)
        g = xr.DataArray(channels[:, :, 1].reshape(data.shape),
                         dims=data.dims, coords=data.coords,
                         attrs=data.attrs).where(mask)
        b = xr.DataArray(channels[:, :, 2].reshape(data.shape),
                         dims=data.dims, coords=data.coords,
                         attrs=data.attrs).where(mask)

        res = super(PaletteCompositor, self).__call__((r, g, b), **data.attrs)
        res.attrs['_FillValue'] = np.nan
        return res


class DayNightCompositor(GenericCompositor):
    """A compositor that blends a day data with night data."""

    def __init__(self, name, lim_low=85., lim_high=95., **kwargs):
        """Collect custom configuration values.

        Args:
            lim_low (float): lower limit of Sun zenith angle for the
                             blending of the given channels
            lim_high (float): upper limit of Sun zenith angle for the
                             blending of the given channels
        """
        self.lim_low = lim_low
        self.lim_high = lim_high
        super(DayNightCompositor, self).__init__(name, **kwargs)

    def __call__(self, projectables, **kwargs):
        projectables = self.check_areas(projectables)

        day_data = projectables[0]
        night_data = projectables[1]

        lim_low = np.cos(np.deg2rad(self.lim_low))
        lim_high = np.cos(np.deg2rad(self.lim_high))
        try:
            coszen = np.cos(np.deg2rad(projectables[2]))
        except IndexError:
            from pyorbital.astronomy import cos_zen
            LOG.debug("Computing sun zenith angles.")
            # Get chunking that matches the data
            try:
                chunks = day_data.sel(bands=day_data['bands'][0]).chunks
            except KeyError:
                chunks = day_data.chunks
            lons, lats = day_data.attrs["area"].get_lonlats_dask(chunks)
            coszen = xr.DataArray(cos_zen(day_data.attrs["start_time"],
                                          lons, lats),
                                  dims=['y', 'x'],
                                  coords=[day_data['y'], day_data['x']])
        # Calculate blending weights
        coszen -= np.min((lim_high, lim_low))
        coszen /= np.abs(lim_low - lim_high)
        coszen = coszen.clip(0, 1)

        # Apply enhancements to get images
        day_data = enhance2dataset(day_data)
        night_data = enhance2dataset(night_data)

        # Adjust bands so that they match
        # L/RGB -> RGB/RGB
        # LA/RGB -> RGBA/RGBA
        # RGB/RGBA -> RGBA/RGBA
        day_data = add_bands(day_data, night_data['bands'])
        night_data = add_bands(night_data, day_data['bands'])

        # Replace missing channel data with zeros
        day_data = zero_missing_data(day_data, night_data)
        night_data = zero_missing_data(night_data, day_data)

        # Get merged metadata
        attrs = combine_metadata(day_data, night_data)

        # Blend the two images together
        data = (1 - coszen) * night_data + coszen * day_data
        data.attrs = attrs

        # Split to separate bands so the mode is correct
        data = [data.sel(bands=b) for b in data['bands']]

        return super(DayNightCompositor, self).__call__(data, **kwargs)


def enhance2dataset(dset):
    """Apply enhancements to dataset *dset* and return the resulting data
    array of the image."""
    attrs = dset.attrs
    img = get_enhanced_image(dset)
    # Clip image data to interval [0.0, 1.0]
    data = img.data.clip(0.0, 1.0)
    data.attrs = attrs

    return data


def add_bands(data, bands):
    """Add bands so that they match *bands*"""
    # Add R, G and B bands, remove L band
    if 'L' in data['bands'].data and 'R' in bands.data:
        lum = data.sel(bands='L')
        new_data = xr.concat((lum, lum, lum), dim='bands')
        new_data['bands'] = ['R', 'G', 'B']
        data = new_data
    # Add alpha band
    if 'A' not in data['bands'].data and 'A' in bands.data:
        new_data = [data.sel(bands=band) for band in data['bands'].data]
        # Create alpha band based on a copy of the first "real" band
        alpha = new_data[0].copy()
        alpha.data = da.ones((data.sizes['y'],
                              data.sizes['x']),
                             chunks=new_data[0].chunks)
        # Rename band to indicate it's alpha
        alpha['bands'] = 'A'
        new_data.append(alpha)
        new_data = xr.concat(new_data, dim='bands')
        data = new_data

    return data


def zero_missing_data(data1, data2):
    """Replace NaN values with zeros in data1 if the data is valid in data2."""
    nans = xu.logical_and(xu.isnan(data1), xu.logical_not(xu.isnan(data2)))
    return data1.where(~nans, 0)


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
        warnings.warn("Airmass compositor is deprecated, use GenericCompositor "
                      "with DifferenceCompositor instead.", DeprecationWarning)
        ch1 = sub_arrays(projectables[0], projectables[1])
        ch2 = sub_arrays(projectables[2], projectables[3])
        res = super(Airmass, self).__call__((ch1, ch2,
                                             projectables[0]),
                                            *args, **kwargs)
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
        warnings.warn("Convection ompositor is deprecated, use GenericCompositor "
                      "with DifferenceCompositor instead.", DeprecationWarning)

        ch1 = sub_arrays(projectables[3], projectables[4])
        ch2 = sub_arrays(projectables[2], projectables[5])
        ch3 = sub_arrays(projectables[1], projectables[0])
        res = super(Convection, self).__call__((ch1, ch2, ch3),
                                               *args, **kwargs)
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
        warnings.warn("Dust compositor is deprecated, use GenericCompositor "
                      "with DifferenceCompositor instead.", DeprecationWarning)

        ch1 = sub_arrays(projectables[2], projectables[1])
        ch2 = sub_arrays(projectables[1], projectables[0])
        res = super(Dust, self).__call__((ch1, ch2,
                                          projectables[1]),
                                         *args, **kwargs)
        return res


class RealisticColors(GenericCompositor):

    def __call__(self, projectables, *args, **kwargs):
        projectables = self.check_areas(projectables)
        vis06 = projectables[0]
        vis08 = projectables[1]
        hrv = projectables[2]

        try:
            ch3 = 3 * hrv - vis06 - vis08
            ch3.attrs = hrv.attrs
        except ValueError:
            raise IncompatibleAreas

        ndvi = (vis08 - vis06) / (vis08 + vis06)
        ndvi = np.where(ndvi < 0, 0, ndvi)

        ch1 = ndvi * vis06 + (1 - ndvi) * vis08
        ch1.attrs = vis06.attrs
        ch2 = ndvi * vis08 + (1 - ndvi) * vis06
        ch2.attrs = vis08.attrs

        res = super(RealisticColors, self).__call__((ch1, ch2, ch3),
                                                    *args, **kwargs)
        return res


class CloudCompositor(GenericCompositor):

    def __init__(self, name, transition_min=258.15, transition_max=298.15,
                 transition_gamma=3.0, **kwargs):
        """Collect custom configuration values.

        Args:
            transition_min (float): Values below or equal to this are
                                    clouds -> opaque white
            transition_max (float): Values above this are
                                    cloud free -> transparent
            transition_gamma (float): Gamma correction to apply at the end

        """
        self.transition_min = transition_min
        self.transition_max = transition_max
        self.transition_gamma = transition_gamma
        super(CloudCompositor, self).__init__(name, **kwargs)

    def __call__(self, projectables, **kwargs):

        data = projectables[0]

        # Default to rough IR thresholds
        # Values below or equal to this are clouds -> opaque white
        tr_min = self.transition_min
        # Values above this are cloud free -> transparent
        tr_max = self.transition_max
        # Gamma correction
        gamma = self.transition_gamma

        slope = 1 / (tr_min - tr_max)
        offset = 1 - slope * tr_min

        alpha = data.where(data > tr_min, 1.)
        alpha = alpha.where(data <= tr_max, 0.)
        alpha = alpha.where((data <= tr_min) | (data > tr_max), slope * data + offset)

        # gamma adjustment
        alpha **= gamma
        res = super(CloudCompositor, self).__call__((data, alpha), **kwargs)
        return res


class RatioSharpenedRGB(GenericCompositor):
    """Sharpen RGB bands with ratio of a high resolution band to a lower resolution version.

    Any pixels where the ratio is computed to be negative or infinity, it is
    reset to 1. Additionally, the ratio is limited to 1.5 on the high end to
    avoid high changes due to small discrepancies in instrument detector
    footprint. Note that the input data to this compositor must already be
    resampled so all data arrays are the same shape.

    Example:

        R_lo -  1000m resolution - shape=(2000, 2000)
        G - 1000m resolution - shape=(2000, 2000)
        B - 1000m resolution - shape=(2000, 2000)
        R_hi -  500m resolution - shape=(4000, 4000)

        ratio = R_hi / R_lo
        new_R = R_hi
        new_G = G * ratio
        new_B = B * ratio

    """

    def __init__(self, *args, **kwargs):
        self.high_resolution_band = kwargs.pop("high_resolution_band", "red")
        if self.high_resolution_band not in ['red', 'green', 'blue', None]:
            raise ValueError("RatioSharpenedRGB.high_resolution_band must "
                             "be one of ['red', 'green', 'blue', None]. Not "
                             "'{}'".format(self.high_resolution_band))
        kwargs.setdefault('common_channel_mask', False)
        super(RatioSharpenedRGB, self).__init__(*args, **kwargs)

    def _get_band(self, high_res, low_res, color, ratio):
        """Figure out what data should represent this color."""
        if self.high_resolution_band == color:
            ret = high_res
        else:
            ret = low_res * ratio
            ret.attrs = low_res.attrs.copy()
        return ret

    def __call__(self, datasets, optional_datasets=None, **info):
        """Sharpen low resolution datasets by multiplying by the ratio of ``high_res / low_res``."""
        if len(datasets) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(datasets), ))
        if not all(x.shape == datasets[0].shape for x in datasets[1:]) or \
                (optional_datasets and
                 optional_datasets[0].shape != datasets[0].shape):
            raise IncompatibleAreas('RatioSharpening requires datasets of '
                                    'the same size. Must resample first.')

        new_attrs = {}
        if optional_datasets:
            datasets = self.check_areas(datasets + optional_datasets)
            high_res = datasets[-1]
            p1, p2, p3 = datasets[:3]
            if 'rows_per_scan' in high_res.attrs:
                new_attrs.setdefault('rows_per_scan', high_res.attrs['rows_per_scan'])
            new_attrs.setdefault('resolution', high_res.attrs['resolution'])
            colors = ['red', 'green', 'blue']

            if self.high_resolution_band in colors:
                LOG.debug("Sharpening image with high resolution {} band".format(self.high_resolution_band))
                low_res = datasets[:3][colors.index(self.high_resolution_band)]
                ratio = high_res / low_res
                # make ratio a no-op (multiply by 1) where the ratio is NaN or
                # infinity or it is negative.
                ratio = ratio.where(np.isfinite(ratio) & (ratio >= 0), 1.)
                # we don't need ridiculously high ratios, they just make bright pixels
                ratio = ratio.clip(0, 1.5)
            else:
                LOG.debug("No sharpening band specified for ratio sharpening")
                high_res = None
                ratio = 1.

            r = self._get_band(high_res, p1, 'red', ratio)
            g = self._get_band(high_res, p2, 'green', ratio)
            b = self._get_band(high_res, p3, 'blue', ratio)
        else:
            datasets = self.check_areas(datasets)
            r, g, b = datasets[:3]

        # combine the masks
        mask = ~(r.isnull() | g.isnull() | b.isnull())
        r = r.where(mask)
        g = g.where(mask)
        b = b.where(mask)

        # Collect information that is the same between the projectables
        # we want to use the metadata from the original datasets since the
        # new r, g, b arrays may have lost their metadata during calculations
        info = combine_metadata(*datasets)
        info.update(new_attrs)
        # Update that information with configured information (including name)
        info.update(self.attrs)
        # Force certain pieces of metadata that we *know* to be true
        info.setdefault("standard_name", "true_color")
        return super(RatioSharpenedRGB, self).__call__((r, g, b), **info)


def _mean4(data, offset=(0, 0), block_id=None):
    rows, cols = data.shape
    # we assume that the chunks except the first ones are aligned
    if block_id[0] == 0:
        row_offset = offset[0] % 2
    else:
        row_offset = 0
    if block_id[1] == 0:
        col_offset = offset[1] % 2
    else:
        col_offset = 0
    row_after = (row_offset + rows) % 2
    col_after = (col_offset + cols) % 2
    pad = ((row_offset, row_after), (col_offset, col_after))

    rows2 = rows + row_offset + row_after
    cols2 = cols + col_offset + col_after

    av_data = np.pad(data, pad, 'edge')
    new_shape = (int(rows2 / 2.), 2, int(cols2 / 2.), 2)
    data_mean = np.nanmean(av_data.reshape(new_shape), axis=(1, 3))
    data_mean = np.repeat(np.repeat(data_mean, 2, axis=0), 2, axis=1)
    data_mean = data_mean[row_offset:row_offset + rows, col_offset:col_offset + cols]
    return data_mean


class SelfSharpenedRGB(RatioSharpenedRGB):
    """Sharpen RGB with ratio of a band with a strided-version of itself.

    Example:

        R -  500m resolution - shape=(4000, 4000)
        G - 1000m resolution - shape=(2000, 2000)
        B - 1000m resolution - shape=(2000, 2000)

        ratio = R / four_element_average(R)
        new_R = R
        new_G = G * ratio
        new_B = B * ratio

    """

    @staticmethod
    def four_element_average_dask(d):
        """Average every 4 elements (2x2) in a 2D array"""
        try:
            offset = d.attrs['area'].crop_offset
        except (KeyError, AttributeError):
            offset = (0, 0)

        res = d.data.map_blocks(_mean4, offset=offset, dtype=d.dtype)
        return xr.DataArray(res, attrs=d.attrs, dims=d.dims, coords=d.coords)

    def __call__(self, datasets, optional_datasets=None, **attrs):
        colors = ['red', 'green', 'blue']
        if self.high_resolution_band not in colors:
            raise ValueError("SelfSharpenedRGB requires at least one high resolution band, not "
                             "'{}'".format(self.high_resolution_band))

        high_res = datasets[colors.index(self.high_resolution_band)]
        high_mean = self.four_element_average_dask(high_res)
        red = high_mean if self.high_resolution_band == 'red' else datasets[0]
        green = high_mean if self.high_resolution_band == 'green' else datasets[1]
        blue = high_mean if self.high_resolution_band == 'blue' else datasets[2]
        return super(SelfSharpenedRGB, self).__call__((red, green, blue), optional_datasets=(high_res,), **attrs)


class LuminanceSharpeningCompositor(GenericCompositor):

    def __call__(self, projectables, *args, **kwargs):
        from trollimage.image import rgb2ycbcr, ycbcr2rgb
        projectables = self.check_areas(projectables)
        luminance = projectables[0].copy()
        luminance /= 100.
        # Limit between min(luminance) ... 1.0
        luminance = da.where(luminance > 1., 1., luminance)

        # Get the enhanced version of the composite to be sharpened
        rgb_img = enhance2dataset(projectables[1])

        # This all will be eventually replaced with trollimage convert() method
        # ycbcr_img = rgb_img.convert('YCbCr')
        # ycbcr_img.data[0, :, :] = luminance
        # rgb_img = ycbcr_img.convert('RGB')

        # Replace luminance of the IR composite
        y__, cb_, cr_ = rgb2ycbcr(rgb_img.data[0, :, :],
                                  rgb_img.data[1, :, :],
                                  rgb_img.data[2, :, :])

        r__, g__, b__ = ycbcr2rgb(luminance, cb_, cr_)
        y_size, x_size = r__.shape
        r__ = da.reshape(r__, (1, y_size, x_size))
        g__ = da.reshape(g__, (1, y_size, x_size))
        b__ = da.reshape(b__, (1, y_size, x_size))

        rgb_img.data = da.vstack((r__, g__, b__))
        return super(LuminanceSharpeningCompositor, self).__call__(rgb_img, *args, **kwargs)


class SandwichCompositor(GenericCompositor):

    def __call__(self, projectables, *args, **kwargs):
        projectables = self.check_areas(projectables)
        luminance = projectables[0]
        luminance /= 100.
        # Limit between min(luminance) ... 1.0
        luminance = luminance.clip(max=1.)

        # Get the enhanced version of the RGB composite to be sharpened
        rgb_img = enhance2dataset(projectables[1])
        rgb_img *= luminance
        return super(SandwichCompositor, self).__call__(rgb_img, *args, **kwargs)
