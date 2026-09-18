
"""Composites."""

from __future__ import annotations

import logging
import warnings

import dask.array as da
import numpy as np
import xarray as xr

from satpy.dataset import combine_metadata
from satpy.modifiers.angles import get_satellite_zenith_angle

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

    def __init__(self, name, transition_min_limits=(190., 220.), latitude_min_limits=(30., 60.),  # noqa: D417
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
    """Detect low-level clouds based on thresholding for night-time compositing.

    This compositor takes the brightness temperature difference (BTD) between a window channel around 10.5 micron
    and a near-infrared channel around 3.8 micron and uses thresholding to detect low-level clouds and fog. Separate
    detection thresholds are used for land and water surface types, requiring a land-sea mask as an input. A
    window-channel brightness temperature is also used to filter out very cold pixels, which can otherwise produce
    noise-induced false alarms.

    Pixels with a BTD above the relevant surface-dependent threshold are considered cloudy and the resulting cloud
    mask is passed to :class:`CloudCompositor`, which computes the alpha channel as a function of the BTD and the
    alpha limits `transition_min` and `transition_max` to create a semi-transparent low cloud cloud composite.

    An optional second split-window difference (using two window channels around 10.5 and 8.7 microns) can be used to
    identify bare soil and remove associated false cloud detections. If this input is provided, the satellite zenith
    angle is also used to account for the increased split-window difference towards larger viewing angles. If the
    satellite zenith angle is not provided as an optional input, it will be computed on the fly.

    This follows the GeoColor implementation of night-time low-level clouds in Miller et al. (2020,
    :doi:`10.1175/JTECH-D-19-0134.1`) including some modifications described in Strandgren et al. (2026, in
    preparation).

    The spectral tests and thus the output of this compositor (using the expected input data) is only applicable during
    night-time.

    """
    def __init__(self, name, values_land=(1,), values_water=(0,),  # noqa: D417
                 threshold_land=1.5,
                 threshold_water=0.0,
                 thresholds_bare_soil=(4.1, 1.5),
                 transition_min=0,
                 transition_max=5.0,
                 transition_gamma=1.0,
                 range_land=None,
                 range_water=None,
                 invert_alpha=True, **kwargs):
        """Initialize the compositor.

        Args:
            name: Name of the compositor.
            values_land (list): Values in the land-sea mask identifying land surface pixels.
            values_water (list): Values in the land-sea mask identifying water surface pixels.
            threshold_land (float): BTD threshold above which low-level clouds are detected over land.
            threshold_water (float): BTD threshold above which low-level clouds are detected over water.
            thresholds_bare_soil (tuple): Two coefficients used to identify bare soil from the optional second
                                          split-window difference. The first value is the base threshold and
                                          the second value describes its dependence on satellite zenith angle.
            transition_min (float): Lower limit of the BTD-to-alpha computation (fully transparent).
            transition_max (float): Upper limit of the BTD-to-alpha computation (fully opaque).
            transition_gamma (float): Gamma correction applied to the alpha channel within the transition range.
            range_land (tuple): Deprecated. Previously used to specify the land threshold and alpha range.
                                If provided, a warning is issued and the value is ignored.
            range_water (tuple): Deprecated. Previously used to specify the water threshold and alpha range.
                                 If provided, a warning is issued and the value is ignored.
            invert_alpha (bool): Invert the alpha channel so that low data values are transparent and high data values
                                are opaque.
            **kwargs: Additional arguments passed to :class:`CloudCompositor`.

        """
        if range_land is not None:
            warnings.warn(
                "'range_land' is deprecated and will be removed in a future version. "
                "Please use 'limit_land' instead as low-level cloud detection threshold over land.",
                UserWarning,
                stacklevel=2,
            )

        if range_water is not None:
            warnings.warn(
                "'range_water' is deprecated and will be removed in a future version. "
                "Please use 'limit_water' instead as low-level cloud detection threshold over water.",
                UserWarning,
                stacklevel=2,
            )

        self.values_land = self._normalize_surface_type_values(values_land)
        self.values_water = self._normalize_surface_type_values(values_water)
        self.threshold_land = threshold_land
        self.threshold_water = threshold_water
        self.thresholds_bare_soil = thresholds_bare_soil

        super().__init__(name, transition_min=transition_min, transition_max=transition_max,
                         transition_gamma=transition_gamma, invert_alpha=invert_alpha, **kwargs)

    @staticmethod
    def _normalize_surface_type_values(values):
        """Convert a single surface value to a list."""
        return values if isinstance(values, (list, tuple)) else [values]

    def _get_low_cloud_mask(self, split_window_low_clouds, is_land, is_water):
        """Determine low-level cloud and fog pixels from the IR10.5-IR3.8 split-window difference."""
        cloud_over_land = is_land & (split_window_low_clouds >= self.threshold_land)
        cloud_over_water = is_water & (split_window_low_clouds >= self.threshold_water)

        return (cloud_over_land | cloud_over_water)

    def _remove_noise(self, low_cloud_mask, window):
        """Exclude very cold pixels which may contain noisy false alarms in the IR10.5-IR3.8 split-window difference."""
        possible_noise = window < 230

        return low_cloud_mask & ~possible_noise

    def _remove_bare_soil(self, low_cloud_mask, split_window_bare_soil, window, satz, is_land):
        """Remove bare-soil false alarms using the IR10.5-IR8.7 split-window difference."""
        if split_window_bare_soil is None:
            LOG.debug(
                "No IR10.5-IR8.7 split-window difference data were provided. Low-level cloud false alarms are likely "
                "to appear over arid surface types."
            )
            return low_cloud_mask

        if satz is None:
            LOG.debug("Computing satellite zenith angle")
            satz = get_satellite_zenith_angle(window)

        sec = 1. / np.cos(np.deg2rad(satz))

        threshold_base, threshold_sec = self.thresholds_bare_soil

        bare_soil = is_land & (
            split_window_bare_soil > threshold_base + threshold_sec * (sec - 1)
        )

        return low_cloud_mask & ~bare_soil

    def __call__(self, projectables, optional_datasets=[], **kwargs):
        """Generate the low-level cloud composite.

        Args:
            projectables: Three datasets containing:
                0. Brightness temperature difference between a window channel around 10.5 micron and a
                   near-infrared channel around 3.8 micron).
                1. Brightness temperature of the window channel, used to filter noise-induced false alarms.
                2. Land-sea mask used to distinguish between land and water detection thresholds.
            optional_datasets: Optional datasets containing:
                0. Brightness temperature difference between the window channel and a second infrared channel
                   around 8.7 microns, used to identify and remove bare-soil false alarms.
                1. Satellite zenith angle, used to increase the bare soil with increasing atmospheric path length.
                   If not provided, it is computed from the window-channel data when the optional bare-soil
                   split-window difference is available.
            **kwargs: Additional arguments passed to :class:`CloudCompositor`.

        Returns:
            The composited low-level cloud mask.
        """
        LOG.debug("Applying detection scheme for low-level clouds and fog (night-time only)")

        if len(projectables) != 3:
            raise ValueError(f"Expected 3 datasets, got {len(projectables)}")

        datasets = self.match_data_arrays(projectables + optional_datasets)
        split_window_low_clouds, window, lsm = datasets[:3]
        split_window_bare_soil = datasets[3] if len(datasets) > 3 else None
        satz = datasets[4] if len(datasets) > 4 else None

        lsm = lsm.squeeze(drop=True)
        lsm = lsm.round()  # Make sure to have whole numbers in case of smearing from resampling
        is_land = lsm.isin(self.values_land)
        is_water = lsm.isin(self.values_water)

        low_cloud_mask = self._get_low_cloud_mask(split_window_low_clouds, is_land, is_water)
        low_cloud_mask = self._remove_noise(low_cloud_mask, window)
        low_cloud_mask = self._remove_bare_soil(low_cloud_mask, split_window_bare_soil, window, satz, is_land)

        return super().__call__([split_window_low_clouds.where(low_cloud_mask)], **kwargs)
