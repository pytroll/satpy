#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020 Satpy developers
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
"""Base classes for composite objects."""

import logging
import os
import time
import warnings
from weakref import WeakValueDictionary

import dask.array as da
import numpy as np
import xarray as xr
import yaml

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader

from satpy.config import CONFIG_PATH, config_search_paths, recursive_dict_update
from satpy.config import get_environ_ancpath, get_entry_points_config_dirs
from satpy.dataset import DATASET_KEYS, DatasetID, MetadataObject, combine_metadata
from satpy.readers import DatasetDict
from satpy.utils import sunzen_corr_cos, atmospheric_path_length_correction, get_satpos
from satpy.writers import get_enhanced_image

try:
    from pyspectral.near_infrared_reflectance import Calculator
except ImportError:
    Calculator = None
try:
    from pyorbital.astronomy import sun_zenith_angle
except ImportError:
    sun_zenith_angle = None


LOG = logging.getLogger(__name__)

NEGLIBLE_COORDS = ['time']
"""Keywords identifying non-dimensional coordinates to be ignored during composite generation."""

MASKING_COMPOSITOR_METHODS = ['less', 'less_equal', 'equal', 'greater_equal',
                              'greater', 'not_equal', 'isnan', 'isfinite',
                              'isneginf', 'isposinf']


class IncompatibleAreas(Exception):
    """Error raised upon compositing things of different shapes."""

    pass


class IncompatibleTimes(Exception):
    """Error raised upon compositing things from different times."""

    pass


class CompositorLoader(object):
    """Read composites using the configuration files on disk."""

    def __init__(self, ppp_config_dir=None):
        """Initialize the compositor loader."""
        if ppp_config_dir is None:
            ppp_config_dir = CONFIG_PATH
        self.modifiers = {}
        self.compositors = {}
        self.ppp_config_dir = ppp_config_dir

    def load_sensor_composites(self, sensor_name):
        """Load all compositor configs for the provided sensor."""
        config_filename = sensor_name + ".yaml"
        LOG.debug("Looking for composites config file %s", config_filename)
        paths = get_entry_points_config_dirs('satpy.composites')
        paths.append(self.ppp_config_dir)
        composite_configs = config_search_paths(
            os.path.join("composites", config_filename),
            *paths, check_exists=True)
        if not composite_configs:
            LOG.debug("No composite config found called {}".format(
                config_filename))
            return
        self._load_config(composite_configs)

    def get_compositor(self, key, sensor_names):
        """Get the modifier for *sensor_names*."""
        for sensor_name in sensor_names:
            try:
                return self.compositors[sensor_name][key]
            except KeyError:
                continue
        raise KeyError("Could not find compositor '{}'".format(key))

    def get_modifier(self, key, sensor_names):
        """Get the modifier for *sensor_names*."""
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
    """Check that *projectables* have compatible times."""
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
    if (attrs.get('area') is None
            and proj1.attrs.get('area') is not None
            and proj2.attrs.get('area') is not None):
        raise IncompatibleAreas
    res = proj1 - proj2
    res.attrs = attrs
    return res


class CompositeBase(MetadataObject):
    """Base class for all compositors and modifiers."""

    def __init__(self, name, prerequisites=None, optional_prerequisites=None, **kwargs):
        """Initialise the compositor."""
        # Required info
        kwargs["name"] = name
        kwargs["prerequisites"] = prerequisites or []
        kwargs["optional_prerequisites"] = optional_prerequisites or []
        super(CompositeBase, self).__init__(**kwargs)

    def __call__(self, datasets, optional_datasets=None, **info):
        """Generate a composite."""
        raise NotImplementedError()

    def __str__(self):
        """Stringify the object."""
        from pprint import pformat
        return pformat(self.attrs)

    def __repr__(self):
        """Represent the object."""
        from pprint import pformat
        return pformat(self.attrs)

    def apply_modifier_info(self, origin, destination):
        """Apply the modifier info from *origin* to *destination*."""
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

    def match_data_arrays(self, data_arrays):
        """Match data arrays so that they can be used together in a composite."""
        self.check_geolocation(data_arrays)
        return self.drop_coordinates(data_arrays)

    def drop_coordinates(self, data_arrays):
        """Drop neglible non-dimensional coordinates."""
        new_arrays = []
        for ds in data_arrays:
            drop = [coord for coord in ds.coords
                    if coord not in ds.dims and any([neglible in coord for neglible in NEGLIBLE_COORDS])]
            if drop:
                new_arrays.append(ds.drop(drop))
            else:
                new_arrays.append(ds)

        return new_arrays

    def check_geolocation(self, data_arrays):
        """Check that the geolocations of the *data_arrays* are compatible."""
        if len(data_arrays) == 1:
            return

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
            return
        elif any(a is None for a in areas):
            raise ValueError("Missing 'area' attribute")

        if not all(areas[0] == x for x in areas[1:]):
            LOG.debug("Not all areas are the same in "
                      "'{}'".format(self.attrs['name']))
            raise IncompatibleAreas("Areas are different")

    def check_areas(self, data_arrays):
        """Check that the areas of the *data_arrays* are compatible."""
        warnings.warn('satpy.composites.CompositeBase.check_areas is deprecated, use '
                      'satpy.composites.CompositeBase.match_data_arrays instead')
        return self.match_data_arrays(data_arrays)


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
        """Generate the composite."""
        projectables = self.match_data_arrays(list(projectables) + list(info.get('optional_datasets', [])))
        vis = projectables[0]
        if vis.attrs.get("sunz_corrected"):
            LOG.debug("Sun zen correction already applied")
            return vis

        area_name = hash(vis.attrs['area'])
        key = (vis.attrs["start_time"], area_name)
        tic = time.time()
        LOG.debug("Applying sun zen correction")
        coszen = self.coszen.get(key)
        if coszen is None and not info.get('optional_datasets'):
            # we were not given SZA, generate SZA then calculate cos(SZA)
            from pyorbital.astronomy import cos_zen
            LOG.debug("Computing sun zenith angles.")
            lons, lats = vis.attrs["area"].get_lonlats(chunks=vis.data.chunks)

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
    """Pyspectral-based rayleigh corrector for visible channels."""

    _rayleigh_cache = WeakValueDictionary()

    def get_angles(self, vis):
        """Get the sun and satellite angles from the current dataarray."""
        from pyorbital.astronomy import get_alt_az, sun_zenith_angle
        from pyorbital.orbital import get_observer_look

        lons, lats = vis.attrs['area'].get_lonlats(chunks=vis.data.chunks)
        lons = da.where(lons >= 1e30, np.nan, lons)
        lats = da.where(lats >= 1e30, np.nan, lats)
        sunalt, suna = get_alt_az(vis.attrs['start_time'], lons, lats)
        suna = np.rad2deg(suna)
        sunz = sun_zenith_angle(vis.attrs['start_time'], lons, lats)

        sat_lon, sat_lat, sat_alt = get_satpos(vis)
        sata, satel = get_observer_look(
            sat_lon,
            sat_lat,
            sat_alt / 1000.0,  # km
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
            vis, red = self.match_data_arrays(projectables)
            sata, satz, suna, sunz = self.get_angles(vis)
            red.data = da.rechunk(red.data, vis.data.chunks)
        else:
            vis, red, sata, satz, suna, sunz = self.match_data_arrays(
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
    """Get the reflective part of NIR bands."""

    def __init__(self, sunz_threshold=None, **kwargs):
        """Collect custom configuration values.

        Args:
            sunz_threshold: The threshold sun zenith angle used when deriving
                the near infrared reflectance. Above this angle the derivation
                will assume this sun-zenith everywhere. Default None, in which
                case the default threshold defined in Pyspectral will be used.

        """
        self.sunz_threshold = sunz_threshold
        super(NIRReflectance, self).__init__(sunz_threshold=sunz_threshold, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the reflectance part of an NIR channel.

        Not supposed to be used for wavelength outside [3, 4] µm.

        """
        self._init_refl3x(projectables)
        _nir, _ = projectables
        projectables = self.match_data_arrays(projectables)

        refl = self._get_reflectance(projectables, optional_datasets) * 100
        proj = xr.DataArray(refl, dims=_nir.dims,
                            coords=_nir.coords, attrs=_nir.attrs)

        proj.attrs['units'] = '%'
        proj.attrs['sunz_threshold'] = self.sunz_threshold
        self.apply_modifier_info(_nir, proj)
        return proj

    def _init_refl3x(self, projectables):
        """Initialize the 3.x reflectance derivations."""
        if not Calculator:
            LOG.info("Couldn't load pyspectral")
            raise ImportError("No module named pyspectral.near_infrared_reflectance")
        _nir, _tb11 = projectables
        self._refl3x = Calculator(_nir.attrs['platform_name'], _nir.attrs['sensor'], _nir.attrs['name'],
                                  sunz_threshold=self.sunz_threshold)

    def _get_reflectance(self, projectables, optional_datasets):
        """Calculate 3.x reflectance with pyspectral."""
        _nir, _tb11 = projectables
        LOG.info('Getting reflective part of %s', _nir.attrs['name'])
        da_nir = _nir.data
        da_tb11 = _tb11.data
        sun_zenith = None
        tb13_4 = None

        for dataset in optional_datasets:
            wavelengths = dataset.attrs.get('wavelength', [100., 0, 0])
            if (dataset.attrs.get('units') == 'K' and
                    wavelengths[0] <= 13.4 <= wavelengths[2]):
                tb13_4 = dataset.data
            elif ("standard_name" in dataset.attrs and
                  dataset.attrs["standard_name"] == "solar_zenith_angle"):
                sun_zenith = dataset.data

        # Check if the sun-zenith angle was provided:
        if sun_zenith is None:
            if sun_zenith_angle is None:
                raise ImportError("No module named pyorbital.astronomy")
            lons, lats = _nir.attrs["area"].get_lonlats(chunks=_nir.data.chunks)
            sun_zenith = sun_zenith_angle(_nir.attrs['start_time'], lons, lats)

        return self._refl3x.reflectance_from_tbs(sun_zenith, da_nir, da_tb11, tb_ir_co2=tb13_4)


class NIREmissivePartFromReflectance(NIRReflectance):
    """Get the emissive part of NIR bands."""

    def __init__(self, sunz_threshold=None, **kwargs):
        """Collect custom configuration values.

        Args:
            sunz_threshold: The threshold sun zenith angle used when deriving
                the near infrared reflectance. Above this angle the derivation
                will assume this sun-zenith everywhere. Default None, in which
                case the default threshold defined in Pyspectral will be used.
        """
        self.sunz_threshold = sunz_threshold
        super(NIREmissivePartFromReflectance, self).__init__(sunz_threshold=sunz_threshold, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the emissive part an NIR channel after having derived the reflectance.

        Not supposed to be used for wavelength outside [3, 4] µm.

        """
        projectables = self.match_data_arrays(projectables)
        self._init_refl3x(projectables)
        # Derive the sun-zenith angles, and use the nir and thermal ir
        # brightness tempertures and derive the reflectance using
        # PySpectral. The reflectance is stored internally in PySpectral and
        # needs to be derived first in order to get the emissive part.
        _ = self._get_reflectance(projectables, optional_datasets)
        _nir, _ = projectables

        emis = self._refl3x.emissive_part_3x()
        proj = xr.DataArray(emis, attrs=_nir.attrs, dims=_nir.dims, coords=_nir.coords)

        proj.attrs['units'] = 'K'
        proj.attrs['sunz_threshold'] = self.sunz_threshold
        self.apply_modifier_info(_nir, proj)
        return proj


class PSPAtmosphericalCorrection(CompositeBase):
    """Correct for atmospheric effects."""

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the atmospherical correction.

        Uses pyspectral.
        """
        from pyspectral.atm_correction_ir import AtmosphericalCorrection

        band = projectables[0]

        if optional_datasets:
            satz = optional_datasets[0]
        else:
            from pyorbital.orbital import get_observer_look
            lons, lats = band.attrs['area'].get_lonlats(chunks=band.data.chunks)
            sat_lon, sat_lat, sat_alt = get_satpos(band)
            try:
                dummy, satel = get_observer_look(sat_lon,
                                                 sat_lat,
                                                 sat_alt / 1000.0,  # km
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
    """Correct for CO2."""

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
    """Make the difference of two data arrays."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info['name'] = self.attrs['name']

        proj = projectables[0] - projectables[1]
        proj.attrs = info
        return proj


class SingleBandCompositor(CompositeBase):
    """Basic single-band composite builder.

    This preserves all the attributes of the dataset it is derived from.
    """

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """Build the composite."""
        if len(projectables) != 1:
            raise ValueError("Can't have more than one band in a single-band composite")

        data = projectables[0]
        new_attrs = data.attrs.copy()

        new_attrs.update({key: val
                          for (key, val) in attrs.items()
                          if val is not None})
        resolution = new_attrs.get('resolution', None)
        new_attrs.update(self.attrs)
        if resolution is not None:
            new_attrs['resolution'] = resolution

        return xr.DataArray(data=data.data, attrs=new_attrs,
                            dims=data.dims, coords=data.coords)


class GenericCompositor(CompositeBase):
    """Basic colored composite builder."""

    modes = {1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA'}

    def __init__(self, name, common_channel_mask=True, **kwargs):
        """Collect custom configuration values.

        Args:
            common_channel_mask (bool): If True, mask all the channels with
                a mask that combines all the invalid areas of the given data.

        """
        self.common_channel_mask = common_channel_mask
        super(GenericCompositor, self).__init__(name, **kwargs)

    @classmethod
    def infer_mode(cls, data_arr):
        """Guess at the mode for a particular DataArray."""
        if 'mode' in data_arr.attrs:
            return data_arr.attrs['mode']
        if 'bands' not in data_arr.dims:
            return cls.modes[1]
        if 'bands' in data_arr.coords and isinstance(data_arr.coords['bands'][0], str):
            return ''.join(data_arr.coords['bands'].values)
        return cls.modes[data_arr.sizes['bands']]

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
                if isinstance(current_sensor, (str, bytes)):
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
            projectables = self.match_data_arrays(projectables)
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
        resolution = new_attrs.get('resolution', None)
        new_attrs.update(self.attrs)
        if resolution is not None:
            new_attrs['resolution'] = resolution
        new_attrs["sensor"] = self._get_sensors(projectables)
        new_attrs["mode"] = mode

        return xr.DataArray(data=data.data, attrs=new_attrs,
                            dims=data.dims, coords=data.coords)


class FillingCompositor(GenericCompositor):
    """Make a regular RGB, filling the RGB bands with the first provided dataset's values."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        projectables[1] = projectables[1].fillna(projectables[0])
        projectables[2] = projectables[2].fillna(projectables[0])
        projectables[3] = projectables[3].fillna(projectables[0])
        return super(FillingCompositor, self).__call__(projectables[1:], **info)


class Filler(GenericCompositor):
    """Fix holes in projectable 1 with data from projectable 2."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        filled_projectable = projectables[0].fillna(projectables[1])
        return super(Filler, self).__call__([filled_projectable], **info)


class RGBCompositor(GenericCompositor):
    """Make a composite from three color bands (deprecated)."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        warnings.warn("RGBCompositor is deprecated, use GenericCompositor instead.", DeprecationWarning)
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(projectables),))
        return super(RGBCompositor, self).__call__(projectables, **info)


class ColormapCompositor(GenericCompositor):
    """A compositor that uses colormaps."""

    @staticmethod
    def build_colormap(palette, dtype, info):
        """Create the colormap from the `raw_palette` and the valid_range.

        Colormaps come in different forms, but they are all supposed to have
        color values between 0 and 255. The following cases are considered:
        - Palettes comprised of only a list on colors. If *dtype* is uint8,
          the values of the colormap are the enumaration of the colors.
          Otherwise, the colormap values will be spread evenly from the min
          to the max of the valid_range provided in `info`.
        - Palettes that have a palette_meanings attribute. The palette meanings
          will be used as values of the colormap.

        """
        from trollimage.colormap import Colormap
        sqpalette = np.asanyarray(palette).squeeze() / 255.0
        set_range = True
        if hasattr(palette, 'attrs') and 'palette_meanings' in palette.attrs:
            set_range = False
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

            if set_range:
                sf = info.get('scale_factor', np.array(1))
                colormap.set_range(
                    *(np.array(info['valid_range']) * sf
                      + info.get('add_offset', 0)))
        else:
            raise AttributeError("Data needs to have either a valid_range or be of type uint8" +
                                 " in order to be displayable with an attached color-palette!")

        return colormap, sqpalette


class ColorizeCompositor(ColormapCompositor):
    """A compositor colorizing the data, interpolating the palette colors when needed."""

    def __call__(self, projectables, **info):
        """Generate the composite."""
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
        """Generate the composite."""
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

    def __init__(self, name, lim_low=85., lim_high=88., **kwargs):
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
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)

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
            lons, lats = day_data.attrs["area"].get_lonlats(chunks=chunks)
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
    """Return the enhancemened to dataset *dset* as an array."""
    attrs = dset.attrs
    img = get_enhanced_image(dset)
    # Clip image data to interval [0.0, 1.0]
    data = img.data.clip(0.0, 1.0)
    data.attrs = attrs
    # remove 'mode' if it is specified since it may have been updated
    data.attrs.pop('mode', None)
    # update mode since it may have changed (colorized/palettize)
    data.attrs['mode'] = GenericCompositor.infer_mode(data)
    return data


def add_bands(data, bands):
    """Add bands so that they match *bands*."""
    # Add R, G and B bands, remove L band
    if 'L' in data['bands'].data and 'R' in bands.data:
        lum = data.sel(bands='L')
        # Keep 'A' if it was present
        if 'A' in data['bands']:
            alpha = data.sel(bands='A')
            new_data = (lum, lum, lum, alpha)
            new_bands = ['R', 'G', 'B', 'A']
            mode = 'RGBA'
        else:
            new_data = (lum, lum, lum)
            new_bands = ['R', 'G', 'B']
            mode = 'RGB'
        data = xr.concat(new_data, dim='bands', coords={'bands': new_bands})
        data['bands'] = new_bands
        data.attrs['mode'] = mode
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
        new_data.attrs['mode'] = data.attrs['mode'] + 'A'
        data = new_data

    return data


def zero_missing_data(data1, data2):
    """Replace NaN values with zeros in data1 if the data is valid in data2."""
    nans = np.logical_and(np.isnan(data1), np.logical_not(np.isnan(data2)))
    return data1.where(~nans, 0)


class RealisticColors(GenericCompositor):
    """Create a realistic colours composite for SEVIRI."""

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
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
    """Detect clouds based on thresholding and use it as a mask for compositing."""

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
        """Generate the composite."""
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

    Example::

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
        """Instanciate the ration sharpener."""
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
            datasets = self.match_data_arrays(datasets + optional_datasets)
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
            datasets = self.match_data_arrays(datasets)
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

    Example::

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
        """Average every 4 elements (2x2) in a 2D array."""
        try:
            offset = d.attrs['area'].crop_offset
        except (KeyError, AttributeError):
            offset = (0, 0)

        res = d.data.map_blocks(_mean4, offset=offset, dtype=d.dtype)
        return xr.DataArray(res, attrs=d.attrs, dims=d.dims, coords=d.coords)

    def __call__(self, datasets, optional_datasets=None, **attrs):
        """Generate the composite."""
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
    """Create a high resolution composite by sharpening a low resolution using high resolution luminance.

    This is done by converting to YCbCr colorspace, replacing Y, and convertin back to RGB.
    """

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        from trollimage.image import rgb2ycbcr, ycbcr2rgb
        projectables = self.match_data_arrays(projectables)
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
    """Make a sandwich product."""

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        luminance = projectables[0]
        luminance /= 100.
        # Limit between min(luminance) ... 1.0
        luminance = luminance.clip(max=1.)

        # Get the enhanced version of the RGB composite to be sharpened
        rgb_img = enhance2dataset(projectables[1])
        rgb_img *= luminance
        return super(SandwichCompositor, self).__call__(rgb_img, *args, **kwargs)


class NaturalEnh(GenericCompositor):
    """Enhanced version of natural color composite by Simon Proud.

    Args:
        ch16_w (float): weight for red channel (1.6 um). Default: 1.3
        ch08_w (float): weight for green channel (0.8 um). Default: 2.5
        ch06_w (float): weight for blue channel (0.6 um). Default: 2.2

    """

    def __init__(self, name, ch16_w=1.3, ch08_w=2.5, ch06_w=2.2,
                 *args, **kwargs):
        """Initialize the class."""
        self.ch06_w = ch06_w
        self.ch08_w = ch08_w
        self.ch16_w = ch16_w
        super(NaturalEnh, self).__init__(name, *args, **kwargs)

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        ch16 = projectables[0]
        ch08 = projectables[1]
        ch06 = projectables[2]

        ch1 = self.ch16_w * ch16 + self.ch08_w * ch08 + self.ch06_w * ch06
        ch1.attrs = ch16.attrs
        ch2 = ch08
        ch3 = ch06

        return super(NaturalEnh, self).__call__((ch1, ch2, ch3),
                                                *args, **kwargs)


class StaticImageCompositor(GenericCompositor):
    """A compositor that loads a static image from disk.

    If the filename passed to this compositor is not valid then
    the SATPY_ANCPATH environment variable will be checked to see
    if the image is located there
    """

    def __init__(self, name, filename=None, area=None, **kwargs):
        """Collect custom configuration values.

        Args:
            filename (str): Filename of the image to load
            area (str): Name of area definition for the image.  Optional
                        for images with built-in area definitions (geotiff)

        """
        if filename is None:
            raise ValueError("No image configured for static image compositor")
        self.filename = filename
        self.area = None
        if area is not None:
            from satpy.resample import get_area_def
            self.area = get_area_def(area)

        super(StaticImageCompositor, self).__init__(name, **kwargs)

    def __call__(self, *args, **kwargs):
        """Call the compositor."""
        from satpy import Scene
        # Check if filename exists, if not then try from SATPY_ANCPATH
        if not os.path.isfile(self.filename):
            tmp_filename = os.path.join(get_environ_ancpath(), self.filename)
            if os.path.isfile(tmp_filename):
                self.filename = tmp_filename
        scn = Scene(reader='generic_image', filenames=[self.filename])
        scn.load(['image'])
        img = scn['image']
        # use compositor parameters as extra metadata
        # most important: set 'name' of the image
        img.attrs.update(self.attrs)
        # Check for proper area definition.  Non-georeferenced images
        # do not have `area` in the attributes
        if 'area' not in img.attrs:
            if self.area is None:
                raise AttributeError("Area definition needs to be configured")
            img.attrs['area'] = self.area
        img.attrs['sensor'] = None
        img.attrs['mode'] = ''.join(img.bands.data)
        img.attrs.pop('modifiers', None)
        img.attrs.pop('calibration', None)
        # Add start time if not present in the filename
        if 'start_time' not in img.attrs or not img.attrs['start_time']:
            import datetime as dt
            img.attrs['start_time'] = dt.datetime.utcnow()
        if 'end_time' not in img.attrs or not img.attrs['end_time']:
            import datetime as dt
            img.attrs['end_time'] = dt.datetime.utcnow()

        return img


class BackgroundCompositor(GenericCompositor):
    """A compositor that overlays one composite on top of another."""

    def __call__(self, projectables, *args, **kwargs):
        """Call the compositor."""
        projectables = self.match_data_arrays(projectables)

        # Get enhanced datasets
        foreground = enhance2dataset(projectables[0])
        background = enhance2dataset(projectables[1])

        # Adjust bands so that they match
        # L/RGB -> RGB/RGB
        # LA/RGB -> RGBA/RGBA
        # RGB/RGBA -> RGBA/RGBA
        foreground = add_bands(foreground, background['bands'])
        background = add_bands(background, foreground['bands'])

        # Get merged metadata
        attrs = combine_metadata(foreground, background)
        if attrs.get('sensor') is None:
            # sensor can be a set
            attrs['sensor'] = self._get_sensors(projectables)

        # Stack the images
        if 'A' in foreground.attrs['mode']:
            # Use alpha channel as weight and blend the two composites
            alpha = foreground.sel(bands='A')
            data = []
            # NOTE: there's no alpha band in the output image, it will
            # be added by the data writer
            for band in foreground.mode[:-1]:
                fg_band = foreground.sel(bands=band)
                bg_band = background.sel(bands=band)
                chan = (fg_band * alpha + bg_band * (1 - alpha))
                chan = xr.where(chan.isnull(), bg_band, chan)
                data.append(chan)
        else:
            data = xr.where(foreground.isnull(), background, foreground)
            # Split to separate bands so the mode is correct
            data = [data.sel(bands=b) for b in data['bands']]

        res = super(BackgroundCompositor, self).__call__(data, **kwargs)
        res.attrs.update(attrs)
        return res


class MaskingCompositor(GenericCompositor):
    """A compositor that masks e.g. IR 10.8 channel data using cloud products from NWC SAF."""

    def __init__(self, name, transparency=None, conditions=None, **kwargs):
        """Collect custom configuration values.

        Kwargs:
            transparency (dict): transparency for each cloud type as
                                 key-value pairs in a dictionary.
                                 Will be converted to `conditions`.
                                 DEPRECATED.
            conditions (list): list of three items determining the masking
                               settings.

        Each condition in *conditions* consists of of three items:
        - `method`: Numpy method name.  The following are supported
           operations: `less`, `less_equal`, `equal`, `greater_equal`,
           `greater`, `not_equal`, `isnan`, `isfinite`, `isinf`,
          `isneginf`, `isposinf`
        - `value`: threshold value of the *mask* applied with the
          operator.  Can be a string, in which case the corresponding
          value will be determined from `flag_meanings` and
          `flag_values` attributes of the mask.
          NOTE: the `value` should not be given to 'is*` methods.
        - `transparency`: transparency from interval [0 ... 100] used
          for the method/threshold. Value of 100 is fully transparent.

        Example::

          >>> conditions = [{'method': 'greater_equal', 'value': 0,
                             'transparency': 100},
                            {'method': 'greater_equal', 'value': 1,
                             'transparency': 80},
                            {'method': 'greater_equal', 'value': 2,
                             'transparency': 0},
                            {'method': 'isnan',
                             'transparency': 100}]
          >>> compositor = MaskingCompositor("masking compositor",
                                             transparency=transparency)
          >>> result = compositor([data, mask])

        This will set transparency of `data` based on the values in
        the `mask` dataset.  Locations where `mask` has values of `0`
        will be fully transparent, locations with `1` will be
        semi-transparent and locations with `2` will be fully visible
        in the resulting image.  In the end all `NaN` areas in the mask are
        set to full transparency.  All the unlisted locations will be
        visible.

        The transparency is implemented by adding an alpha layer to
        the composite.  The locations with transparency of `100` will
        be set to NaN in the data.  If the input `data` contains an
        alpha channel, it will be discarded.

        """
        if transparency:
            LOG.warning("Using 'transparency' is deprecated in "
                        "MaskingCompositor, use 'conditions' instead.")
            self.conditions = []
            for key, transp in transparency.items():
                self.conditions.append({'method': 'equal',
                                        'value': key,
                                        'transparency': transp})
            LOG.info("Converted 'transparency' to 'conditions': %s",
                     str(self.conditions))
        else:
            self.conditions = conditions
        if self.conditions is None:
            raise ValueError("Masking conditions not defined.")

        super(MaskingCompositor, self).__init__(name, **kwargs)

    def __call__(self, projectables, *args, **kwargs):
        """Call the compositor."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        data_in = projectables[0]
        mask_in = projectables[1]
        mask_data = mask_in.data

        alpha_attrs = data_in.attrs.copy()
        if 'bands' in data_in.dims:
            data = [data_in.sel(bands=b) for b in data_in['bands'] if b != 'A']
        else:
            data = [data_in]

        # Create alpha band
        alpha = da.ones((data[0].sizes['y'],
                         data[0].sizes['x']),
                        chunks=data[0].chunks)

        for condition in self.conditions:
            method = condition['method']
            value = condition.get('value', None)
            if isinstance(value, str):
                value = _get_flag_value(mask_in, value)
            transparency = condition['transparency']
            mask = self._get_mask(method, value, mask_data)

            if transparency == 100.0:
                data = self._set_data_nans(data, mask, alpha_attrs)
            alpha_val = 1. - transparency / 100.
            alpha = da.where(mask, alpha_val, alpha)

        alpha = xr.DataArray(data=alpha, attrs=alpha_attrs,
                             dims=data[0].dims, coords=data[0].coords)
        data.append(alpha)

        res = super(MaskingCompositor, self).__call__(data, **kwargs)
        return res

    def _get_mask(self, method, value, mask_data):
        """Get mask array from *mask_data* using *method* and threshold *value*.

        The *method* is the name of a numpy function.

        """
        if method not in MASKING_COMPOSITOR_METHODS:
            raise AttributeError("Unsupported Numpy method %s, use one of %s",
                                 method, str(MASKING_COMPOSITOR_METHODS))

        func = getattr(np, method)

        if value is None:
            return func(mask_data)
        return func(mask_data, value)

    def _set_data_nans(self, data, mask, attrs):
        """Set *data* to nans where *mask* is True.

        The attributes *attrs** will be written to each band in *data*.

        """
        for i, dat in enumerate(data):
            data[i] = xr.where(mask, np.nan, dat)
            data[i].attrs = attrs

        return data


def _get_flag_value(mask, val):
    """Get a numerical value of the named flag.

    This function assumes the naming used in product generated with
    NWC SAF GEO/PPS softwares.

    """
    flag_meanings = mask.attrs['flag_meanings']
    flag_values = mask.attrs['flag_values']
    if isinstance(flag_meanings, str):
        flag_meanings = flag_meanings.split()

    index = flag_meanings.index(val)

    return flag_values[index]
