# Copyright (c) 2015-2025 Satpy developers
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

"""Composites."""

from __future__ import annotations

import logging

import dask.array as da
import numpy as np
import xarray as xr

from satpy.dataset import combine_metadata

from .core import CompositeBase, GenericCompositor, SingleBandCompositor

LOG = logging.getLogger(__name__)

MASKING_COMPOSITOR_METHODS = ["less", "less_equal", "equal", "greater_equal",
                              "greater", "not_equal", "isnan", "isfinite",
                              "isneginf", "isposinf"]


class MaskingCompositor(GenericCompositor):
    """A compositor that masks e.g. IR 10.8 channel data using cloud products from NWC SAF."""

    _supported_modes = {"LA", "RGBA"}

    def __init__(self, name, transparency=None, conditions=None, mode="LA",
                 **kwargs):
        """Collect custom configuration values.

        Kwargs:
            transparency (dict): transparency for each cloud type as
                                 key-value pairs in a dictionary.
                                 Will be converted to `conditions`.
                                 DEPRECATED.
            conditions (list): list of three items determining the masking
                               settings.
            mode (str, optional): Image mode to return.  For single-band input,
                                  this shall be "LA" (default) or "RGBA".  For
                                  multi-band input, this argument is ignored
                                  as the result is always RGBA.

        Each condition in *conditions* consists of three items:

        - `method`: Numpy method name.  The following are supported
            operations: `less`, `less_equal`, `equal`, `greater_equal`,
            `greater`, `not_equal`, `isnan`, `isfinite`, `isinf`,
            `isneginf`, or `isposinf`.
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
                self.conditions.append({"method": "equal",
                                        "value": key,
                                        "transparency": transp})
            LOG.info("Converted 'transparency' to 'conditions': %s",
                     str(self.conditions))
        else:
            self.conditions = conditions
        if self.conditions is None:
            raise ValueError("Masking conditions not defined.")
        if mode not in self._supported_modes:
            raise ValueError(f"Invalid mode {mode!s}.  Supported modes: " +
                             ", ".join(self._supported_modes))
        self.mode = mode

        super(MaskingCompositor, self).__init__(name, **kwargs)

    def __call__(self, projectables, *args, **kwargs):
        """Call the compositor."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        data_in = projectables[0]
        mask_in = projectables[1]

        # remove "bands" dimension for single band masks (ex. "L")
        mask_in = mask_in.squeeze(drop=True)

        alpha_attrs = data_in.attrs.copy()
        data = self._select_data_bands(data_in)

        alpha = self._get_alpha_bands(data, mask_in, alpha_attrs)
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

    def _select_data_bands(self, data_in):
        """Select data to be composited from input data.

        From input data, select the bands that need to have masking applied.
        """
        if "bands" in data_in.dims:
            return [data_in.sel(bands=b) for b in data_in["bands"] if b != "A"]
        if self.mode == "RGBA":
            return [data_in, data_in, data_in]
        return [data_in]

    def _get_alpha_bands(self, data, mask_in, alpha_attrs):
        """Get alpha bands.

        From input data, masks, and attributes, get alpha band.
        """
        # Create alpha band
        mask_data = mask_in.data
        alpha = da.ones((data[0].sizes["y"],
                         data[0].sizes["x"]),
                        chunks=data[0].chunks)

        for condition in self.conditions:
            method = condition["method"]
            value = condition.get("value", None)
            if isinstance(value, str):
                value = _get_flag_value(mask_in, value)
            transparency = condition["transparency"]
            mask = self._get_mask(method, value, mask_data)

            if transparency == 100.0:
                data = self._set_data_nans(data, mask, alpha_attrs)
            alpha_val = 1. - transparency / 100.
            alpha = da.where(mask, alpha_val, alpha)

        return xr.DataArray(data=alpha, attrs=alpha_attrs,
                            dims=data[0].dims, coords=data[0].coords)


def _get_flag_value(mask, val):
    """Get a numerical value of the named flag.

    This function assumes the naming used in product generated with
    NWC SAF GEO/PPS softwares.

    """
    flag_meanings = mask.attrs["flag_meanings"]
    flag_values = mask.attrs["flag_values"]
    if isinstance(flag_meanings, str):
        flag_meanings = flag_meanings.split()

    index = flag_meanings.index(val)

    return flag_values[index]


class LongitudeMaskingCompositor(SingleBandCompositor):
    """Masks areas outside defined longitudes."""

    def __init__(self, name, lon_min=None, lon_max=None, **kwargs):  # noqa: D417
        """Collect custom configuration values.

        Args:
            lon_min (float): lower longitude limit
            lon_max (float): upper longitude limit
        """
        self.lon_min = lon_min
        self.lon_max = lon_max
        if self.lon_min is None and self.lon_max is None:
            raise ValueError("Masking conditions not defined. \
                At least lon_min or lon_max has to be specified.")
        if not self.lon_min:
            self.lon_min = -180.
        if not self.lon_max:
            self.lon_max = 180.
        super().__init__(name, **kwargs)

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectable = projectables[0]
        lons, lats = projectable.attrs["area"].get_lonlats()

        if self.lon_max > self.lon_min:
            lon_min_max = np.logical_and(lons >= self.lon_min, lons <= self.lon_max)
        else:
            lon_min_max = np.logical_or(lons >= self.lon_min, lons <= self.lon_max)

        masked_projectable = projectable.where(lon_min_max)
        return super().__call__([masked_projectable], **info)


class SimpleFireMaskCompositor(CompositeBase):
    """Class for a simple fire detection compositor."""

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """Compute a simple fire detection to create a boolean mask to be used in "flames" composites.

        Expects 4 channel inputs, calibrated to BT/reflectances, in this order [µm]: 10.x, 3.x, 2.x, 0.6.

        It applies 4 spectral tests, for which the thresholds must be provided in the yaml as "test_thresholds":
        - Test 0: 10.x > thr0 (clouds filter)
        - Test 1: 3.x-10.x > thr1 (hotspot)
        - Test 2: 0.6 > thr2 (clouds, sunglint filter)
        - Test 3: 3.x+2.x > thr3 (hotspot)

        .. warning::
            This fire detection algorithm is extremely simple, so it is prone to false alarms and missed detections.
            It is intended only for PR-like visualisation of large fires, not for any other use.
            The tests have been designed for MTG-FCI.

        """
        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info["name"] = self.attrs["name"]
        info.update(self.attrs)

        # fire spectral tests

        # test 0: # window channel should be warm (no clouds)
        ir_105_temp = projectables[0] > self.attrs["test_thresholds"][0]
        # test 1: # 3.8-10.5µm should be high (hotspot)
        temp_diff = projectables[1] - projectables[0] > self.attrs["test_thresholds"][1]
        # test 2: vis_06 should be low (no clouds, no sunglint)
        vis_06_bright = projectables[3] < self.attrs["test_thresholds"][2]
        # test 3: 3.8+2.2µm should be high (hotspot)
        ir38_plus_nir22 = projectables[1] + projectables[2] >= self.attrs["test_thresholds"][3]

        res = ir_105_temp & temp_diff & vis_06_bright & ir38_plus_nir22  # combine all tests

        res.attrs = info
        return res


class CloudCompositor(GenericCompositor):
    """Detect clouds based on thresholding and use it as a mask for compositing."""

    def __init__(self, name, transition_min=258.15, transition_max=298.15,  # noqa: D417
                 transition_gamma=3.0, invert_alpha=False, **kwargs):
        """Collect custom configuration values.

        Args:
            transition_min (float): Values below or equal to this are
                                    clouds -> opaque white
            transition_max (float): Values above this are
                                    cloud free -> transparent
            transition_gamma (float): Gamma correction to apply at the end
            invert_alpha (bool): Invert the alpha channel to make low data values transparent
                                 and high data values opaque.

        """
        self.transition_min = transition_min
        self.transition_max = transition_max
        self.transition_gamma = transition_gamma
        self.invert_alpha = invert_alpha
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

        if self.invert_alpha:
            alpha.data = 1.0 - alpha.data

        # gamma adjustment
        alpha **= gamma
        res = super(CloudCompositor, self).__call__((data, alpha), **kwargs)
        return res


class HighCloudCompositor(CloudCompositor):
    """Detect high clouds based on latitude-dependent thresholding and use it as a mask for compositing.

    This compositor aims at identifying high clouds and assigning them a transparency based on the brightness
    temperature (cloud opacity). In contrast to the `CloudCompositor`, the brightness temperature threshold at
    the lower end, used to identify high opaque clouds, is made a function of the latitude in order to have
    tropopause level clouds appear opaque at both high and low latitudes. This follows the Geocolor
    implementation of high clouds in Miller et al. (2020, :doi:`10.1175/JTECH-D-19-0134.1`), but
    with some adjustments to the thresholds based on recent developments and feedback from CIRA.

    The two brightness temperature thresholds in `transition_min` are used together with the corresponding
    latitude limits in `latitude_min` to compute a modified version of `transition_min` that is later used
    when calling `CloudCompositor`. The modified version of `transition_min` will be an array with the same
    shape as the input projectable dataset, where the actual values of threshold_min are a function of the
    dataset `latitude`:

      - transition_min = transition_min[0] where abs(latitude) < latitude_min(0)
      - transition_min = transition_min[1] where abs(latitude) > latitude_min(0)
      - transition_min = linear interpolation between transition_min[0] and transition_min[1] as a function
                         of where abs(latitude).
    """

    def __init__(self, name, transition_min_limits=(210., 230.), latitude_min_limits=(30., 60.),  # noqa: D417
                 transition_max=300, transition_gamma=1.0, **kwargs):
        """Collect custom configuration values.

        Args:
            transition_min_limits (tuple): Brightness temperature values used to identify opaque white
                                           clouds at different latitudes
            transition_max (float): Brightness temperatures above this value are not considered to
                                    be high clouds -> transparent
            latitude_min_limits (tuple): Latitude values defining the intervals for computing latitude-dependent
                                         `transition_min` values from `transition_min_limits`.
            transition_gamma (float): Gamma correction to apply to the alpha channel within the brightness
                                      temperature range (`transition_min` to `transition_max`).

        """
        if len(transition_min_limits) != 2:
            raise ValueError(f"Expected 2 `transition_min_limits` values, got {len(transition_min_limits)}")
        if len(latitude_min_limits) != 2:
            raise ValueError(f"Expected 2 `latitude_min_limits` values, got {len(latitude_min_limits)}")
        if type(transition_max) in [list, tuple]:
            raise ValueError(f"Expected `transition_max` to be of type float, is of type {type(transition_max)}")

        self.transition_min_limits = transition_min_limits
        self.latitude_min_limits = latitude_min_limits
        super().__init__(name, transition_min=None, transition_max=transition_max,
                         transition_gamma=transition_gamma, **kwargs)

    def __call__(self, projectables, **kwargs):
        """Generate the composite.

        `projectables` is expected to be a list or tuple with a single element:
          - index 0: Brightness temperature of a thermal infrared window channel (e.g. 10.5 microns).
        """
        if len(projectables) != 1:
            raise ValueError(f"Expected 1 dataset, got {len(projectables)}")

        data = projectables[0]
        _, lats = data.attrs["area"].get_lonlats(chunks=data.chunks, dtype=data.dtype)
        lats = np.abs(lats)

        slope = (self.transition_min_limits[1] - self.transition_min_limits[0]) / \
                (self.latitude_min_limits[1] - self.latitude_min_limits[0])
        offset = self.transition_min_limits[0] - slope * self.latitude_min_limits[0]

        # Compute pixel-level latitude dependent transition_min values and pass to parent CloudCompositor class
        transition_min = xr.DataArray(name="transition_min", coords=data.coords, dims=data.dims).astype(data.dtype)
        transition_min = transition_min.where(lats >= self.latitude_min_limits[0], self.transition_min_limits[0])
        transition_min = transition_min.where(lats <= self.latitude_min_limits[1], self.transition_min_limits[1])
        transition_min = transition_min.where((lats < self.latitude_min_limits[0]) |
                                              (lats > self.latitude_min_limits[1]), slope * lats + offset)
        self.transition_min = transition_min

        return super().__call__(projectables, **kwargs)


class LowCloudCompositor(CloudCompositor):
    """Detect low-level clouds based on thresholding and use it as a mask for compositing during night-time.

    This compositor computes the brightness temperature difference between a window channel (e.g. 10.5 micron)
    and the near-infrared channel e.g. (3.8 micron) and uses this brightness temperature difference, `BTD`, to
    create a partially transparent mask for compositing.

    Pixels with `BTD` values below a given threshold  will be transparent, whereas pixels with `BTD` values
    above another threshold will be opaque. The transparency of all other `BTD` values will be a linear
    function of the `BTD` value itself. Two sets of thresholds are used, one set for land surface types
    (`range_land`) and another one for water surface types (`range_water`), respectively. Hence,
    this compositor requires a land-water-mask as a prerequisite input. This follows the GeoColor
    implementation of night-time low-level clouds in Miller et al. (2020, :doi:`10.1175/JTECH-D-19-0134.1`), but
    with some adjustments to the thresholds based on recent developments and feedback from CIRA.

    Please note that the spectral test and thus the output of the compositor (using the expected input data) is
    only applicable during night-time.
    """

    def __init__(self, name, values_land=(1,), values_water=(0,),  # noqa: D417
                 range_land=(0.0, 4.0),
                 range_water=(0.0, 4.0),
                 transition_gamma=1.0,
                 invert_alpha=True, **kwargs):
        """Init info.

        Collect custom configuration values.

        Args:
            values_land (list): List of values used to identify land surface pixels in the land-water-mask.
            values_water (list): List of values used to identify water surface pixels in the land-water-mask.
            range_land (tuple): Threshold values used for masking low-level clouds from the brightness temperature
                                difference over land surface types.
            range_water (tuple): Threshold values used for masking low-level clouds from the brightness temperature
                                 difference over water.
            transition_gamma (float): Gamma correction to apply to the alpha channel within the brightness
                                      temperature difference range.
            invert_alpha (bool): Invert the alpha channel to make low data values transparent
                                 and high data values opaque.
        """
        if len(range_land) != 2:
            raise ValueError(f"Expected 2 `range_land` values, got {len(range_land)}")
        if len(range_water) != 2:
            raise ValueError(f"Expected 2 `range_water` values, got {len(range_water)}")

        self.values_land = values_land if type(values_land) in [list, tuple] else [values_land]
        self.values_water = values_water if type(values_water) in [list, tuple] else [values_water]
        self.range_land = range_land
        self.range_water = range_water
        super().__init__(name, transition_min=None, transition_max=None,
                         transition_gamma=transition_gamma, invert_alpha=invert_alpha, **kwargs)

    def __call__(self, projectables, **kwargs):
        """Generate the composite.

        `projectables` is expected to be a list or tuple with the following three elements:
          - index 0: Brightness temperature difference between a window channel (e.g. 10.5 micron) and a
                     near-infrared channel e.g. (3.8 micron).
          - index 1. Brightness temperature of the window channel (used to filter out noise-induced false alarms).
          - index 2: Land-Sea-Mask.
        """
        if len(projectables) != 3:
            raise ValueError(f"Expected 3 datasets, got {len(projectables)}")

        projectables = self.match_data_arrays(projectables)
        btd, bt_win, lsm = projectables
        lsm = lsm.squeeze(drop=True)
        lsm = lsm.round()  # Make sure to have whole numbers in case of smearing from resampling

        # Call CloudCompositor for land surface pixels
        self.transition_min, self.transition_max = self.range_land
        res = super().__call__([btd.where(lsm.isin(self.values_land))], **kwargs)

        # Call CloudCompositor for /water surface pixels
        self.transition_min, self.transition_max = self.range_water
        res_water = super().__call__([btd.where(lsm.isin(self.values_water))], **kwargs)

        # Compine resutls for land and water surface pixels
        res = res.where(lsm.isin(self.values_land), res_water)

        # Make pixels with cold window channel brightness temperatures transparent to avoid spurious false
        # alarms caused by noise in the 3.9um channel that can occur for very cold cloud tops
        res.loc["A"] = res.sel(bands="A").where(bt_win >= 230, 0.0)

        return res
