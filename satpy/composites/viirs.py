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
"""Composite classes for the VIIRS instrument.
"""

import logging
import os

import numpy as np

from satpy.composites import CompositeBase, IncompatibleAreas
from satpy.config import get_environ_ancpath
from satpy.dataset import Dataset, combine_info

LOG = logging.getLogger(__name__)


class VIIRSFog(CompositeBase):

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))

        p1, p2 = projectables
        fog = p1 - p2
        fog.info.update(self.info)
        fog.info["area"] = p1.info["area"]
        fog.info["start_time"] = p1.info["start_time"]
        fog.info["end_time"] = p1.info["end_time"]
        fog.info["name"] = self.info["name"]
        fog.info["wavelength"] = None
        fog.info.setdefault("mode", "L")
        return fog


class VIIRSTrueColor(CompositeBase):

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" %
                             (len(projectables), ))

        # Collect information that is the same between the projectables
        info = combine_info(*projectables)
        # Update that information with configured information (including name)
        info.update(self.info)
        # Force certain pieces of metadata that we *know* to be true
        info["wavelength"] = None
        info["mode"] = self.info.get("mode", "RGB")
        return Dataset(data=np.rollaxis(
            np.ma.dstack([projectable for projectable in projectables]),
            axis=2),
            **info)


class RatioSharpenedRGB(CompositeBase):

    def __init__(self, *args, **kwargs):
        self.high_resolution_band = kwargs.pop("high_resolution_band", "red")
        super(RatioSharpenedRGB, self).__init__(*args, **kwargs)

    def __call__(self, datasets, optional_datasets=None, **info):
        if len(datasets) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(datasets), ))

        area = None
        n = {}
        p1, p2, p3 = datasets
        if optional_datasets:
            high_res = optional_datasets[0]
            low_res = datasets[["red", "green", "blue"].index(
                self.high_resolution_band)]
            if high_res.info["area"] != low_res.info["area"]:
                if np.mod(high_res.shape[0], low_res.shape[0]) or \
                        np.mod(high_res.shape[1], low_res.shape[1]):
                    raise IncompatibleAreas(
                        "High resolution band is not mapped the same area as the low resolution bands")
                else:
                    f0 = high_res.shape[0] / low_res.shape[0]
                    f1 = high_res.shape[1] / low_res.shape[1]
                    if p1.shape != high_res.shape:
                        p1 = np.ma.repeat(np.ma.repeat(
                            p1, f0, axis=0), f1, axis=1)
                        p1.info["area"] = high_res.info["area"]
                    if p2.shape != high_res.shape:
                        p2 = np.ma.repeat(np.ma.repeat(
                            p2, f0, axis=0), f1, axis=1)
                        p2.info["area"] = high_res.info["area"]
                    if p3.shape != high_res.shape:
                        p3 = np.ma.repeat(np.ma.repeat(
                            p3, f0, axis=0), f1, axis=1)
                        p3.info["area"] = high_res.info["area"]
                    area = high_res.info["area"]
            if 'rows_per_scan' in high_res.info:
                n.setdefault('rows_per_scan', high_res.info['rows_per_scan'])
            n.setdefault('resolution', high_res.info['resolution'])
            if self.high_resolution_band == "red":
                LOG.debug("Sharpening image with high resolution red band")
                ratio = high_res.data / p1.data
                r = high_res.data
                g = p2.data * ratio
                b = p3.data * ratio
            elif self.high_resolution_band == "green":
                LOG.debug("Sharpening image with high resolution green band")
                ratio = high_res.data / p2.data
                r = p1.data * ratio
                g = high_res.data
                b = p3.data * ratio
            elif self.high_resolution_band == "blue":
                LOG.debug("Sharpening image with high resolution blue band")
                ratio = high_res.data / p3.data
                r = p1.data * ratio
                g = p2.data * ratio
                b = high_res.data
            else:
                # no sharpening
                r = p1.data
                g = p2.data
                b = p3.data
            mask = p1.mask | p2.mask | p3.mask | high_res.mask
        else:
            r, g, b = p1.data, p2.data, p3.data
            mask = p1.mask | p2.mask | p3.mask

        # Collect information that is the same between the projectables
        info = combine_info(*datasets)
        info.update(n)
        # Update that information with configured information (including name)
        info.update(self.info)
        # Force certain pieces of metadata that we *know* to be true
        info["wavelength"] = None
        info.setdefault("standard_name", "true_color")
        info["mode"] = self.info.get("mode", "RGB")
        if area is not None:
            info['area'] = area
        return Dataset(data=np.concatenate(
            ([r], [g], [b]), axis=0),
            mask=np.array([[mask, mask, mask]]),
            **info)


class ReflectanceCorrector(CompositeBase):
    """CREFL modifier

    Uses a python rewrite of the C CREFL code written for VIIRS and MODIS.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the compositor with values from the user or from the configuration file.

        If `dem_filename` can't be found or opened then correction is done
        assuming TOA or sealevel options.

        :param dem_filename: path to the ancillary 'averaged heights' file
                             default: CMGDEM.hdf
                             environment override: os.path.join(<SATPY_ANCPATH>, <CREFL_ANCFILENAME>)
        :param dem_sds: variable name to load from the ancillary file
        """
        dem_filename = kwargs.pop("dem_filename",
                                  os.environ.get("CREFL_ANCFILENAME",
                                                 "CMGDEM.hdf"))
        if os.path.exists(dem_filename):
            self.dem_file = dem_filename
        else:
            self.dem_file = os.path.join(get_environ_ancpath(), dem_filename)
        self.dem_sds = kwargs.pop("dem_sds", "averaged elevation")
        super(ReflectanceCorrector, self).__init__(*args, **kwargs)

    def __call__(self, datasets, **info):
        refl_data, sensor_aa, sensor_za, solar_aa, solar_za = datasets
        if refl_data.info.get("rayleigh_corrected"):
            return refl_data

        if os.path.isfile(self.dem_file):
            LOG.debug("Loading CREFL averaged elevation information from: %s",
                      self.dem_file)
            from netCDF4 import Dataset as NCDataset
            # HDF4 file, NetCDF library needs to be compiled with HDF4 support
            nc = NCDataset(self.dem_file, "r")
            avg_elevation = nc.variables[self.dem_sds][:]
        else:
            avg_elevation = None

        from satpy.composites.crefl_utils import run_crefl, get_coefficients

        percent = refl_data.info["units"] == "%"

        coefficients = get_coefficients(refl_data.info["sensor"],
                                        refl_data.info["wavelength"],
                                        refl_data.info["resolution"])

        results = run_crefl(refl_data,
                            coefficients,
                            sensor_aa.info["area"].lons,
                            sensor_aa.info["area"].lats,
                            sensor_aa,
                            sensor_za,
                            solar_aa,
                            solar_za,
                            avg_elevation=avg_elevation,
                            percent=percent, )

        info.update(refl_data.info)
        info["rayleigh_corrected"] = True
        factor = 100. if percent else 1.
        proj = Dataset(data=results.data * factor,
                       mask=results.mask,
                       dtype=results.dtype,
                       **info)

        self.apply_modifier_info(refl_data, proj)

        return proj


class HistogramDNB(CompositeBase):
    """Histogram equalized DNB composite.

    The logic for this code was taken from Polar2Grid and was originally developed by Eva Schiffer (SSEC).

    This composite separates the DNB data in to 3 main regions: Day, Night, and Mixed. Each region is
    equalized separately to bring out the most information from the region due to the high dynamic range
    of the DNB data. Optionally, the mixed region can be separated in to multiple smaller regions by
    using the `mixed_degree_step` keyword.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the compositor with values from the user or from the configuration file.

        :param high_angle_cutoff: solar zenith angle threshold in degrees, values above this are considered "night"
        :param low_angle_cutoff: solar zenith angle threshold in degrees, values below this are considered "day"
        :param mixed_degree_step: Step interval to separate "mixed" region in to multiple parts
                                  by default does whole mixed region
        """
        self.high_angle_cutoff = int(kwargs.pop("high_angle_cutoff", 100))
        self.low_angle_cutoff = int(kwargs.pop("low_angle_cutoff", 88))
        self.mixed_degree_step = int(kwargs.pop(
            "mixed_degree_step")) if "mixed_degree_step" in kwargs else None
        super(HistogramDNB, self).__init__(*args, **kwargs)

    def __call__(self, datasets, **info):
        """Create the composite by scaling the DNB data using a histogram equalization method.

        :param datasets: 2-element tuple (Day/Night Band data, Solar Zenith Angle data)
        :param **info: Miscellaneous metadata for the newly produced composite
        """
        if len(datasets) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(datasets), ))

        dnb_data = datasets[0]
        sza_data = datasets[1]
        good_mask = ~(dnb_data.mask | sza_data.mask)
        output_dataset = dnb_data.copy()
        output_dataset.mask = ~good_mask
        day_mask, mixed_mask, night_mask = make_day_night_masks(
            sza_data,
            good_mask,
            self.high_angle_cutoff,
            self.low_angle_cutoff,
            stepsDegrees=self.mixed_degree_step)

        did_equalize = False
        if day_mask.any():
            LOG.debug("Histogram equalizing DNB day data...")
            histogram_equalization(dnb_data.data, day_mask, out=output_dataset)
            did_equalize = True
        if mixed_mask:
            for mask in mixed_mask:
                if mask.any():
                    LOG.debug("Histogram equalizing DNB mixed data...")
                    histogram_equalization(dnb_data.data,
                                           mask,
                                           out=output_dataset)
                    did_equalize = True
        if night_mask.any():
            LOG.debug("Histogram equalizing DNB night data...")
            histogram_equalization(dnb_data.data,
                                   night_mask,
                                   out=output_dataset)
            did_equalize = True

        if not did_equalize:
            raise RuntimeError("No valid data found to histogram equalize")

        info = dnb_data.info.copy()
        info.update(self.info)
        info["standard_name"] = "equalized_radiance"
        info["mode"] = "L"
        output_dataset.info = info
        return output_dataset


class AdaptiveDNB(HistogramDNB):
    """Adaptive histogram equalized DNB composite.

    The logic for this code was taken from Polar2Grid and was originally developed by Eva Schiffer (SSEC).

    This composite separates the DNB data in to 3 main regions: Day, Night, and Mixed. Each region is
    equalized separately to bring out the most information from the region due to the high dynamic range
    of the DNB data. Optionally, the mixed region can be separated in to multiple smaller regions by
    using the `mixed_degree_step` keyword.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the compositor with values from the user or from the configuration file.

        Adaptive histogram equalization and regular histogram equalization can be configured
        independently for each region: day, night, or mixed.
        A region can be set to use adaptive equalization "always", or "never", or only when
        there are multiple regions in a single scene "multiple" via the `adaptive_X` keyword
        arguments (see below).

        :param adaptive_day: one of ("always", "multiple", "never") meaning when adaptive equalization is used.
        :param adaptive_mixed: one of ("always", "multiple", "never") meaning when adaptive equalization is used.
        :param adaptive_night: one of ("always", "multiple", "never") meaning when adaptive equalization is used.
        """
        self.adaptive_day = kwargs.pop("adaptive_day", "always")
        self.adaptive_mixed = kwargs.pop("adaptive_mixed", "always")
        self.adaptive_night = kwargs.pop("adaptive_night", "always")
        self.day_radius_pixels = int(kwargs.pop("day_radius_pixels", 400))
        self.mixed_radius_pixels = int(kwargs.pop("mixed_radius_pixels", 100))
        self.night_radius_pixels = int(kwargs.pop("night_radius_pixels", 400))

        super(AdaptiveDNB, self).__init__(*args, **kwargs)

    def __call__(self, datasets, **info):
        """Create the composite by scaling the DNB data using an adaptive histogram equalization method.

        :param datasets: 2-element tuple (Day/Night Band data, Solar Zenith Angle data)
        :param **info: Miscellaneous metadata for the newly produced composite
        """
        if len(datasets) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(datasets), ))

        dnb_data = datasets[0]
        sza_data = datasets[1]
        good_mask = ~(dnb_data.mask | sza_data.mask)
        output_dataset = dnb_data.copy()
        output_dataset.mask = ~good_mask
        day_mask, mixed_mask, night_mask = make_day_night_masks(
            sza_data,
            good_mask,
            self.high_angle_cutoff,
            self.low_angle_cutoff,
            stepsDegrees=self.mixed_degree_step)

        did_equalize = False
        has_multi_times = len(mixed_mask) > 0
        if day_mask.any():
            did_equalize = True
            if self.adaptive_day == "always" or (
                    has_multi_times and self.adaptive_day == "multiple"):
                LOG.debug("Adaptive histogram equalizing DNB day data...")
                local_histogram_equalization(
                    dnb_data.data,
                    day_mask,
                    valid_data_mask=good_mask,
                    local_radius_px=self.day_radius_pixels,
                    out=output_dataset)
            else:
                LOG.debug("Histogram equalizing DNB day data...")
                histogram_equalization(dnb_data.data,
                                       day_mask,
                                       out=output_dataset)
        if mixed_mask:
            for mask in mixed_mask:
                if mask.any():
                    did_equalize = True
                    if self.adaptive_mixed == "always" or (
                            has_multi_times and
                            self.adaptive_mixed == "multiple"):
                        LOG.debug(
                            "Adaptive histogram equalizing DNB mixed data...")
                        local_histogram_equalization(
                            dnb_data.data,
                            mask,
                            valid_data_mask=good_mask,
                            local_radius_px=self.mixed_radius_pixels,
                            out=output_dataset)
                    else:
                        LOG.debug("Histogram equalizing DNB mixed data...")
                        histogram_equalization(dnb_data.data,
                                               day_mask,
                                               out=output_dataset)
        if night_mask.any():
            did_equalize = True
            if self.adaptive_night == "always" or (
                    has_multi_times and self.adaptive_night == "multiple"):
                LOG.debug("Adaptive histogram equalizing DNB night data...")
                local_histogram_equalization(
                    dnb_data.data,
                    night_mask,
                    valid_data_mask=good_mask,
                    local_radius_px=self.night_radius_pixels,
                    out=output_dataset)
            else:
                LOG.debug("Histogram equalizing DNB night data...")
                histogram_equalization(dnb_data.data,
                                       night_mask,
                                       out=output_dataset)

        if not did_equalize:
            raise RuntimeError("No valid data found to histogram equalize")

        info = dnb_data.info.copy()
        info.update(self.info)
        info["standard_name"] = "equalized_radiance"
        info["mode"] = "L"
        output_dataset.info = info
        return output_dataset


class ERFDNB(CompositeBase):
    """Equalized DNB composite using the error function (erf).

    The logic for this code was taken from Polar2Grid and was originally developed by Curtis Seaman and Steve Miller.
    The original code was written in IDL and is included as comments in the code below.
    """

    def __init__(self, *args, **kwargs):
        self.saturation_correction = kwargs.pop(
            "saturation_correction", False) in [True, "True", "true"]
        super(ERFDNB, self).__init__(*args, **kwargs)

    def __call__(self, datasets, **info):
        if len(datasets) != 4:
            raise ValueError("Expected 4 datasets, got %d" % (len(datasets), ))

        from scipy.special import erf
        dnb_data = datasets[0]
        sza_data = datasets[1]
        lza_data = datasets[2]
        good_mask = ~(dnb_data.mask | sza_data.mask)
        output_dataset = dnb_data.copy()
        output_dataset.mask = ~good_mask
        # this algorithm assumes units of "W cm-2 sr-1" so if there are other
        # units we need to adjust for that
        if dnb_data.info.get("units", "W m-2 sr-1") == "W m-2 sr-1":
            unit_factor = 10000.
        else:
            unit_factor = 1.

        # convert to decimal instead of %
        moon_illum_fraction = np.mean(datasets[3]) * 0.01

        # From Steve Miller and Curtis Seaman
        # maxval = 10.^(-1.7 - (((2.65+moon_factor1+moon_factor2))*(1+erf((solar_zenith-95.)/(5.*sqrt(2.0))))))
        # minval = 10.^(-4. - ((2.95+moon_factor2)*(1+erf((solar_zenith-95.)/(5.*sqrt(2.0))))))
        # scaled_radiance = (radiance - minval) / (maxval - minval)
        # radiance = sqrt(scaled_radiance)

        # Version 2: Update from Curtis Seaman
        # maxval = 10.^(-1.7 - (((2.65+moon_factor1+moon_factor2))*(1+erf((solar_zenith-95.)/(5.*sqrt(2.0))))))
        # minval = 10.^(-4. - ((2.95+moon_factor2)*(1+erf((solar_zenith-95.)/(5.*sqrt(2.0))))))
        # saturated_pixels = where(radiance gt maxval, nsatpx)
        # saturation_pct = float(nsatpx)/float(n_elements(radiance))
        # print, 'Saturation (%) = ', saturation_pct
        #
        # while saturation_pct gt 0.005 do begin
        #   maxval = maxval*1.1
        #   saturated_pixels = where(radiance gt maxval, nsatpx)
        #   saturation_pct = float(nsatpx)/float(n_elements(radiance))
        #   print, saturation_pct
        # endwhile
        #
        # scaled_radiance = (radiance - minval) / (maxval - minval)
        # radiance = sqrt(scaled_radiance)

        moon_factor1 = 0.7 * (1.0 - moon_illum_fraction)
        moon_factor2 = 0.0022 * lza_data
        erf_portion = 1 + erf((sza_data - 95.0) / (5.0 * np.sqrt(2.0)))
        max_val = np.power(
            10, -1.7 -
            (2.65 + moon_factor1 + moon_factor2) * erf_portion) * unit_factor
        min_val = np.power(10, -4.0 -
                           (2.95 + moon_factor2) * erf_portion) * unit_factor

        # Update from Curtis Seaman, increase max radiance curve until less
        # than 0.5% is saturated
        if self.saturation_correction:
            saturation_pct = float(np.count_nonzero(dnb_data >
                                                    max_val)) / dnb_data.size
            LOG.debug("Dynamic DNB saturation percentage: %f", saturation_pct)
            while saturation_pct > 0.005:
                max_val *= 1.1 * unit_factor
                saturation_pct = float(np.count_nonzero(
                    dnb_data > max_val)) / dnb_data.size
                LOG.debug("Dynamic DNB saturation percentage: %f",
                          saturation_pct)

        inner_sqrt = (dnb_data - min_val) / (max_val - min_val)
        # clip negative values to 0 before the sqrt
        inner_sqrt[inner_sqrt < 0] = 0
        np.sqrt(inner_sqrt, out=output_dataset)

        info = dnb_data.info.copy()
        info.update(self.info)
        info["standard_name"] = "equalized_radiance"
        info["mode"] = "L"
        output_dataset.info = info
        return output_dataset


def make_day_night_masks(solarZenithAngle,
                         good_mask,
                         highAngleCutoff,
                         lowAngleCutoff,
                         stepsDegrees=None):
    """
    given information on the solarZenithAngle for each point,
    generate masks defining where the day, night, and mixed regions are

    optionally provide the highAngleCutoff and lowAngleCutoff that define
    the limits of the terminator region (if no cutoffs are given the
    DEFAULT_HIGH_ANGLE and DEFAULT_LOW_ANGLE will be used)

    optionally provide the stepsDegrees that define how many degrees each
    "mixed" mask in the terminator region should be (if no stepsDegrees is
    given, the whole terminator region will be one mask)
    """
    # if the caller passes None, we're only doing one step
    stepsDegrees = highAngleCutoff - \
        lowAngleCutoff if stepsDegrees is None else stepsDegrees

    night_mask = (solarZenithAngle > highAngleCutoff) & good_mask
    day_mask = (solarZenithAngle <= lowAngleCutoff) & good_mask
    mixed_mask = []
    steps = range(lowAngleCutoff, highAngleCutoff + 1, stepsDegrees)
    if steps[-1] >= highAngleCutoff:
        steps[-1] = highAngleCutoff
    steps = zip(steps, steps[1:])
    for i, j in steps:
        LOG.debug("Processing step %d to %d" % (i, j))
        tmp = (solarZenithAngle > i) & (solarZenithAngle <= j) & good_mask
        if tmp.any():
            LOG.debug("Adding step %d to %d" % (i, j))
            # log.debug("Points to process in this range: " + str(np.sum(tmp)))
            mixed_mask.append(tmp)
        del tmp

    return day_mask, mixed_mask, night_mask


def histogram_equalization(
        data,
        mask_to_equalize,
        number_of_bins=1000,
        std_mult_cutoff=4.0,
        do_zerotoone_normalization=True,
        valid_data_mask=None,

        # these are theoretically hooked up, but not useful with only one
        # equalization
        clip_limit=None,
        slope_limit=None,

        # these parameters don't do anything, they're just here to mirror those
        # in the other call
        do_log_scale=False,
        log_offset=None,
        local_radius_px=None,
        out=None):
    """
    Perform a histogram equalization on the data selected by mask_to_equalize.
    The data will be separated into number_of_bins levels for equalization and
    outliers beyond +/- std_mult_cutoff*std will be ignored.

    If do_zerotoone_normalization is True the data selected by mask_to_equalize
    will be returned in the 0 to 1 range. Otherwise the data selected by
    mask_to_equalize will be returned in the 0 to number_of_bins range.

    Note: the data will be changed in place.
    """

    out = out if out is not None else data.copy()
    mask_to_use = mask_to_equalize if valid_data_mask is None else valid_data_mask

    LOG.debug("determining DNB data range for histogram equalization")
    avg = np.mean(data[mask_to_use])
    std = np.std(data[mask_to_use])
    # limit our range to +/- std_mult_cutoff*std; e.g. the default
    # std_mult_cutoff is 4.0 so about 99.8% of the data
    concervative_mask = (data < (avg + std * std_mult_cutoff)) & (
        data > (avg - std * std_mult_cutoff)) & mask_to_use

    LOG.debug("running histogram equalization")
    cumulative_dist_function, temp_bins = _histogram_equalization_helper(
        data[concervative_mask],
        number_of_bins,
        clip_limit=clip_limit,
        slope_limit=slope_limit)

    # linearly interpolate using the distribution function to get the new
    # values
    out[mask_to_equalize] = np.interp(data[mask_to_equalize], temp_bins[:-1],
                                      cumulative_dist_function)

    # if we were asked to, normalize our data to be between zero and one,
    # rather than zero and number_of_bins
    if do_zerotoone_normalization:
        _linear_normalization_from_0to1(out, mask_to_equalize, number_of_bins)

    return out


def local_histogram_equalization(data, mask_to_equalize, valid_data_mask=None, number_of_bins=1000,
                                 std_mult_cutoff=3.0,
                                 do_zerotoone_normalization=True,
                                 local_radius_px=300,
                                 clip_limit=60.0,  # 20.0,
                                 slope_limit=3.0,  # 0.5,
                                 do_log_scale=True,
                                 # can't take the log of zero, so the offset
                                 # may be needed; pass 0.0 if your data doesn't
                                 # need it
                                 log_offset=0.00001,
                                 out=None
                                 ):
    """
    equalize the provided data (in the mask_to_equalize) using adaptive histogram equalization
    tiles of width/height (2 * local_radius_px + 1) will be calculated and results for each pixel will be bilinerarly interpolated from the nearest 4 tiles
    when pixels fall near the edge of the image (there is no adjacent tile) the resultant interpolated sum from the available tiles will be multipled to
    account for the weight of any missing tiles (pixel total interpolated value = pixel available interpolated value / (1 - missing interpolation weight))

    if do_zerotoone_normalization is True the data will be scaled so that all data in the mask_to_equalize falls between 0 and 1; otherwise the data
    in mask_to_equalize will all fall between 0 and number_of_bins

    returns the equalized data
    """

    out = out if out is not None else np.ma.zeros_like(data)
    # if we don't have a valid mask, use the mask of what we should be
    # equalizing
    if valid_data_mask is None:
        valid_data_mask = mask_to_equalize

    # calculate some useful numbers for our tile math
    total_rows = data.shape[0]
    total_cols = data.shape[1]
    tile_size = int((local_radius_px * 2.0) + 1.0)
    row_tiles = int(total_rows / tile_size) if (
        total_rows % tile_size is 0) else int(total_rows / tile_size) + 1
    col_tiles = int(total_cols / tile_size) if (
        total_cols % tile_size is 0) else int(total_cols / tile_size) + 1

    # an array of our distribution functions for equalization
    all_cumulative_dist_functions = [[]]
    # an array of our bin information for equalization
    all_bin_information = [[]]

    # loop through our tiles and create the histogram equalizations for each
    # one
    for num_row_tile in range(row_tiles):

        # make sure we have enough rows available to store info on this next
        # row of tiles
        if len(all_cumulative_dist_functions) <= num_row_tile:
            all_cumulative_dist_functions.append([])
        if len(all_bin_information) <= num_row_tile:
            all_bin_information.append([])

        # go through each tile in this row and calculate the equalization
        for num_col_tile in range(col_tiles):

            # calculate the range for this tile (min is inclusive, max is
            # exclusive)
            min_row = num_row_tile * tile_size
            max_row = min_row + tile_size
            min_col = num_col_tile * tile_size
            max_col = min_col + tile_size

            # for speed of calculation, pull out the mask of pixels that should
            # be used to calculate the histogram
            mask_valid_data_in_tile = valid_data_mask[min_row:max_row, min_col:
                                                      max_col]

            # if we have any valid data in this tile, calculate a histogram equalization for this tile
            # (note: even if this tile does no fall in the mask_to_equalize, it's histogram may be used by other tiles)
            cumulative_dist_function, temp_bins = None, None
            if mask_valid_data_in_tile.any():

                # use all valid data in the tile, so separate sections will
                # blend cleanly
                temp_valid_data = data[min_row:max_row, min_col:max_col][
                    mask_valid_data_in_tile]
                temp_valid_data = temp_valid_data[
                    temp_valid_data >= 0
                ]  # TEMP, testing to see if negative data is messing everything up
                # limit the contrast by only considering data within a certain
                # range of the average
                if std_mult_cutoff is not None:
                    avg = np.mean(temp_valid_data)
                    std = np.std(temp_valid_data)
                    # limit our range to avg +/- std_mult_cutoff*std; e.g. the
                    # default std_mult_cutoff is 4.0 so about 99.8% of the data
                    concervative_mask = (
                        temp_valid_data < (avg + std * std_mult_cutoff)) & (
                            temp_valid_data > (avg - std * std_mult_cutoff))
                    temp_valid_data = temp_valid_data[concervative_mask]

                # if we are taking the log of our data, do so now
                if do_log_scale:
                    temp_valid_data = np.log(temp_valid_data + log_offset)

                # do the histogram equalization and get the resulting
                # distribution function and bin information
                if temp_valid_data.size > 0:
                    cumulative_dist_function, temp_bins = _histogram_equalization_helper(
                        temp_valid_data,
                        number_of_bins,
                        clip_limit=clip_limit,
                        slope_limit=slope_limit)

            # hang on to our equalization related information for use later
            all_cumulative_dist_functions[num_row_tile].append(
                cumulative_dist_function)
            all_bin_information[num_row_tile].append(temp_bins)

    # get the tile weight array so we can use it to interpolate our data
    tile_weights = _calculate_weights(tile_size)

    # now loop through our tiles and linearly interpolate the equalized
    # versions of the data
    for num_row_tile in range(row_tiles):
        for num_col_tile in range(col_tiles):

            # calculate the range for this tile (min is inclusive, max is
            # exclusive)
            min_row = num_row_tile * tile_size
            max_row = min_row + tile_size
            min_col = num_col_tile * tile_size
            max_col = min_col + tile_size

            # for convenience, pull some of these tile sized chunks out
            temp_all_data = data[min_row:max_row, min_col:max_col].copy()
            temp_mask_to_equalize = mask_to_equalize[min_row:max_row, min_col:
                                                     max_col]
            temp_all_valid_data_mask = valid_data_mask[min_row:max_row,
                                                       min_col:max_col]

            # if we have any data in this tile, calculate our weighted sum
            if temp_mask_to_equalize.any():
                if do_log_scale:
                    temp_all_data[temp_all_valid_data_mask] = np.log(
                        temp_all_data[temp_all_valid_data_mask] + log_offset)
                temp_data_to_equalize = temp_all_data[temp_mask_to_equalize]
                temp_all_valid_data = temp_all_data[temp_all_valid_data_mask]

                # a place to hold our weighted sum that represents the interpolated contributions
                # of the histogram equalizations from the surrounding tiles
                temp_sum = np.zeros_like(temp_data_to_equalize)

                # how much weight were we unable to use because those tiles
                # fell off the edge of the image?
                unused_weight = np.zeros(temp_data_to_equalize.shape,
                                         dtype=tile_weights.dtype)

                # loop through all the surrounding tiles and process their
                # contributions to this tile
                for weight_row in range(3):
                    for weight_col in range(3):
                        # figure out which adjacent tile we're processing (in
                        # overall tile coordinates instead of relative to our
                        # current tile)
                        calculated_row = num_row_tile - 1 + weight_row
                        calculated_col = num_col_tile - 1 + weight_col
                        tmp_tile_weights = tile_weights[
                            weight_row, weight_col][np.where(
                                temp_mask_to_equalize)]

                        # if we're inside the tile array and the tile we're
                        # processing has a histogram equalization for us to
                        # use, process it
                        if ((calculated_row >= 0) and
                            (calculated_row < row_tiles) and
                            (calculated_col >= 0) and
                            (calculated_col < col_tiles) and (
                                all_bin_information[calculated_row][
                                    calculated_col] is not None) and
                            (all_cumulative_dist_functions[calculated_row][
                                calculated_col] is not None)):

                            # equalize our current tile using the histogram
                            # equalization from the tile we're processing
                            temp_equalized_data = np.interp(
                                temp_all_valid_data, all_bin_information[
                                    calculated_row][calculated_col][:-1],
                                all_cumulative_dist_functions[calculated_row][
                                    calculated_col])
                            temp_equalized_data = temp_equalized_data[np.where(
                                temp_mask_to_equalize[
                                    temp_all_valid_data_mask])]

                            # add the contribution for the tile we're
                            # processing to our weighted sum
                            temp_sum += (temp_equalized_data *
                                         tmp_tile_weights)

                        else:  # if the tile we're processing doesn't exist, hang onto the weight we would have used for it so we can correct that later
                            unused_weight -= tmp_tile_weights

                # if we have unused weights, scale our values to correct for
                # that
                if unused_weight.any():
                    # TODO, if the mask masks everything out this will be a
                    # zero!
                    temp_sum /= unused_weight + 1

                # now that we've calculated the weighted sum for this tile, set
                # it in our data array
                out.data[min_row:max_row, min_col:max_col][
                    temp_mask_to_equalize] = temp_sum
                """
                # TEMP, test without using weights
                data[min_row:max_row, min_col:max_col][temp_mask_to_equalize] = np.interp(temp_data_to_equalize,
                                                                                             all_bin_information          [num_row_tile  ][num_col_tile][:-1],
                                                                                             all_cumulative_dist_functions[num_row_tile  ][num_col_tile])
                """

    # if we were asked to, normalize our data to be between zero and one,
    # rather than zero and number_of_bins
    if do_zerotoone_normalization:
        _linear_normalization_from_0to1(out, mask_to_equalize, number_of_bins)

    return out


def _histogram_equalization_helper(valid_data,
                                   number_of_bins,
                                   clip_limit=None,
                                   slope_limit=None):
    """
    calculate the simplest possible histogram equalization, using only valid data

    returns the cumulative distribution function and bin information
    """

    # bucket all the selected data using np's histogram function
    temp_histogram, temp_bins = np.histogram(valid_data, number_of_bins)

    # if we have a clip limit and we should do our clipping before building
    # the cumulative distribution function, clip off our histogram
    if (clip_limit is not None):

        # clip our histogram and remember how much we removed
        pixels_to_clip_at = int(clip_limit *
                                (valid_data.size / float(number_of_bins)))
        mask_to_clip = temp_histogram > clip_limit
        num_bins_clipped = sum(mask_to_clip)
        num_pixels_clipped = sum(temp_histogram[mask_to_clip]) - (
            num_bins_clipped * pixels_to_clip_at)
        temp_histogram[mask_to_clip] = pixels_to_clip_at

    # calculate the cumulative distribution function
    cumulative_dist_function = temp_histogram.cumsum()

    # if we have a clip limit and we should do our clipping after building the
    # cumulative distribution function, clip off our cdf
    if (slope_limit is not None):

        # clip our cdf and remember how much we removed
        pixel_height_limit = int(slope_limit *
                                 (valid_data.size / float(number_of_bins)))
        cumulative_excess_height = 0
        num_clipped_pixels = 0
        weight_metric = np.zeros(cumulative_dist_function.shape, dtype=float)

        for pixel_index in range(1, cumulative_dist_function.size):

            current_pixel_count = cumulative_dist_function[pixel_index]

            diff_from_acceptable = (
                current_pixel_count - cumulative_dist_function[pixel_index - 1]
                - pixel_height_limit - cumulative_excess_height)
            if diff_from_acceptable < 0:
                weight_metric[pixel_index] = abs(diff_from_acceptable)
            cumulative_excess_height += max(diff_from_acceptable, 0)
            cumulative_dist_function[
                pixel_index] = current_pixel_count - cumulative_excess_height
            num_clipped_pixels = num_clipped_pixels + cumulative_excess_height

    # now normalize the overall distribution function
    cumulative_dist_function = (
        number_of_bins -
        1) * cumulative_dist_function / cumulative_dist_function[-1]

    # return what someone else will need in order to apply the equalization
    # later
    return cumulative_dist_function, temp_bins


def _calculate_weights(tile_size):
    """
    calculate a weight array that will be used to quickly bilinearly-interpolate the histogram equalizations
    tile size should be the width and height of a tile in pixels

    returns a 4D weight array, where the first 2 dimensions correspond to the grid of where the tiles are
    relative to the tile being interpolated
    """

    # we are essentially making a set of weight masks for an ideal center tile
    # that has all 8 surrounding tiles available

    # create our empty template tiles
    template_tile = np.zeros((3, 3, tile_size, tile_size), dtype=np.float32)
    """
    # TEMP FOR TESTING, create a weight tile that does no interpolation
    template_tile[1,1] = template_tile[1,1] + 1.0
    """

    # for ease of calculation, figure out the index of the center pixel in a tile
    # and how far that pixel is from the edge of the tile (in pixel units)
    center_index = int(tile_size / 2)
    center_dist = tile_size / 2.0

    # loop through each pixel in the tile and calculate the 9 weights for that pixel
    # were weights for a pixel are 0.0 they are not set (since the template_tile
    # starts out as all zeros)
    for row in range(tile_size):
        for col in range(tile_size):

            vertical_dist = abs(
                center_dist - row
            )  # the distance from our pixel to the center of our tile, vertically
            horizontal_dist = abs(
                center_dist - col
            )  # the distance from our pixel to the center of our tile, horizontally

            # pre-calculate which 3 adjacent tiles will affect our tile
            # (note: these calculations aren't quite right if center_index equals the row or col)
            horizontal_index = 0 if col < center_index else 2
            vertical_index = 0 if row < center_index else 2

            # if this is the center pixel, we only need to use it's own tile
            # for it
            if (row is center_index) and (col is center_index):

                # all of the weight for this pixel comes from it's own tile
                template_tile[1, 1][row, col] = 1.0

            # if this pixel is in the center row, but is not the center pixel
            # we're going to need to linearly interpolate it's tile and the
            # tile that is horizontally nearest to it
            elif (row is center_index) and (col is not center_index):

                # linear interp horizontally

                beside_weight = horizontal_dist / tile_size  # the weight from the adjacent tile
                local_weight = (
                    tile_size -
                    horizontal_dist) / tile_size  # the weight from this tile

                # set the weights for the two relevant tiles
                template_tile[1, 1][row, col] = local_weight
                template_tile[1, horizontal_index][row, col] = beside_weight

            # if this pixel is in the center column, but is not the center pixel
            # we're going to need to linearly interpolate it's tile and the
            # tile that is vertically nearest to it
            elif (row is not center_index) and (col is center_index):

                # linear interp vertical

                beside_weight = vertical_dist / tile_size  # the weight from the adjacent tile
                local_weight = (
                    tile_size -
                    vertical_dist) / tile_size  # the weight from this tile

                # set the weights for the two relevant tiles
                template_tile[1, 1][row, col] = local_weight
                template_tile[vertical_index, 1][row, col] = beside_weight

            # if the pixel is in one of the four quadrants that are above or below the center
            # row and column, we need to bilinearly interpolate it between the
            # nearest four tiles
            else:

                # bilinear interpolation

                local_weight = ((tile_size - vertical_dist) / tile_size) * (
                    (tile_size - horizontal_dist) /
                    tile_size)  # the weight from this tile
                vertical_weight = ((vertical_dist) / tile_size) * (
                    (tile_size - horizontal_dist) / tile_size
                )  # the weight from the vertically   adjacent tile
                horizontal_weight = (
                    (tile_size - vertical_dist) / tile_size) * (
                        (horizontal_dist) / tile_size
                )  # the weight from the horizontally adjacent tile
                diagonal_weight = ((vertical_dist) / tile_size) * (
                    (horizontal_dist) / tile_size
                )  # the weight from the diagonally   adjacent tile

                # set the weights for the four relevant tiles
                template_tile[1, 1, row, col] = local_weight
                template_tile[vertical_index, 1, row, col] = vertical_weight
                template_tile[1, horizontal_index, row,
                              col] = horizontal_weight
                template_tile[vertical_index, horizontal_index, row,
                              col] = diagonal_weight

    # return the weights for an ideal center tile
    return template_tile


def _linear_normalization_from_0to1(
        data,
        mask,
        theoretical_max,
        theoretical_min=0,
        message="normalizing equalized data to fit in 0 to 1 range"):
    """Do a linear normalization so all data is in the 0 to 1 range.

    This is a sloppy but fast calculation that relies on parameters giving it
    the correct theoretical current max and min so it can scale the data
    accordingly.
    """

    LOG.debug(message)
    if theoretical_min is not 0:
        data[mask] = data[mask] - theoretical_min
        theoretical_max = theoretical_max - theoretical_min
    data[mask] = data[mask] / theoretical_max


class NCCZinke(CompositeBase):
    """Equalized DNB composite using the Zinke algorithm.

    http://www.tandfonline.com/doi/full/10.1080/01431161.2017.1338838
    DOI: 10.1080/01431161.2017.1338838

    """

    def __call__(self, datasets, **info):
        if len(datasets) != 4:
            raise ValueError("Expected 4 datasets, got %d" % (len(datasets), ))

        dnb_data = datasets[0]
        sza_data = datasets[1]
        lza_data = datasets[2]
        #good_mask = ~(dnb_data.mask | sza_data.mask)
        #output_dataset = dnb_data.copy()
        #output_dataset.mask = ~good_mask
        # this algorithm assumes units of "W cm-2 sr-1" so if there are other
        # units we need to adjust for that
        if dnb_data.info.get("units", "W m-2 sr-1") == "W m-2 sr-1":
            unit_factor = 10000.
        else:
            unit_factor = 1.

        mda = dnb_data.info.copy()

        dnb_data = dnb_data / unit_factor

        # convert to decimal instead of %
        moon_illum_fraction = np.mean(datasets[3]) * 0.01

        phi = np.rad2deg(np.arccos(2. * moon_illum_fraction - 1))

        vfl = 0.026 * phi + 4.0e-9 * (phi ** 4.)

        m_fullmoon = -12.74
        m_sun = -26.74
        m_moon = vfl + m_fullmoon

        gs_ = self.gain_factor(sza_data)

        r_sun_moon = 10.**((m_sun - m_moon) / -2.5)
        gl_ = r_sun_moon * self.gain_factor(lza_data)
        gtot = 1. / (1. / gs_ + 1. / gl_)

        dnb_data += 2.6e-10
        dnb_data *= gtot

        mda['name'] = self.info['name']
        mda.pop('calibration')
        mda.pop('wavelength')
        mda['standard_name'] = 'ncc_radiance'

        return Dataset(dnb_data, copy=False, **mda)

    @staticmethod
    def gain_factor(theta):
        gain = np.empty_like(theta)

        mask = theta <= 87.541
        gain[mask] = (58 + 4 / np.cos(np.deg2rad(theta[mask]))) / 5

        mask = np.logical_and(theta <= 96, 87.541 < theta)
        gain[mask] = (123 * np.exp(1.06 * (theta[mask] - 89.589)) *
                      ((theta[mask] - 93)**2 / 18 + 0.5))

        mask = np.logical_and(96 < theta, theta <= 101)
        gain[mask] = 123 * np.exp(1.06 * (theta[mask] - 89.589))

        mask = np.logical_and(101 < theta, theta <= 103.49)
        gain[mask] = (123 * np.exp(1.06 * (101 - 89.589)) *
                      np.log(theta[mask] - (101 - np.e)) ** 2)

        gain[theta > 103.49] = 6.0e7

        return gain



class SnowAge(CompositeBase):
    """Returns RGB snow product based on method presented at the second 
    CSPP/IMAPP users' meeting at Eumetsat in Darmstadt on 14-16 April 2015
    """
    # Bernard Bellec snow Look-Up Tables V 1.0 (c) Meteo-France
    # These Look-up Tables allow you to create the RGB snow product
    # for SUOMI-NPP VIIRS Imager according to the algorithm
    # presented at the second CSPP/IMAPP users' meeting at Eumetsat
    # in Darmstadt on 14-16 April 2015
    # The algorithm and the product are described in this
    # presentation :
    # http://www.ssec.wisc.edu/meetings/cspp/2015/Agenda%20PDF/Wednesday/Roquet_snow_product_cspp2015.pdf
    # For further information you may contact
    # Bernard Bellec at Bernard.Bellec@meteo.fr
    # or
    # Pascale Roquet at Pascale.Roquet@meteo.fr

    def __call__(self, projectables, nonprojectables=None, **info):
        if len(projectables) != 5:
            raise ValueError("Expected 5 datasets, got %d" %
                             (len(projectables), ))

        # Collect information that is the same between the projectables
        info = combine_info(*projectables)
        # Update that information with configured information (including name)
        info.update(self.info)
        # Force certain pieces of metadata that we *know* to be true
        info["wavelength"] = None
        info["mode"] = self.info.get("mode", "RGB")

        m07 = projectables[0]*255./160.
        m08 = projectables[1]*255./160.
        m09 = projectables[2]*255./160.
        m10 = projectables[3]*255./160.
        m11 = projectables[4]*255./160.
        refcu = m11 - m10
        refcu[refcu < 0] = 0

        ch1 = m07 - refcu / 2. - m09 / 4.
        ch2 = m08 + refcu / 4. + m09 / 4.
        ch3 = m11 + m09

        return Dataset(data=[ch1, ch2, ch3], **info)

