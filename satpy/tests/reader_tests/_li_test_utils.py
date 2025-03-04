# Copyright (c) 2022 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Common utility modules used for LI mock-oriented unit tests."""

import datetime as dt

import numpy as np
import xarray as xr

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.tests.utils import RANDOM_GEN

# mapping of netcdf type code to numpy data type:
TYPE_MAP = {
    "i1": np.int8,
    "i2": np.int16,
    "i4": np.int32,
    "i8": np.int64,
    "u1": np.uint8,
    "u2": np.uint16,
    "u4": np.uint32,
    "u8": np.uint64,
    "f4": np.float32,
    "f8": np.float64,
}

def rand_type(num, dtype):
        return RANDOM_GEN.integers(low=np.iinfo(dtype).min, high=np.iinfo(dtype).max - 1, size=num, dtype=dtype)

def l2_le_schema(settings=None):
    """Define schema for LI L2 LE product."""
    settings = settings or {}
    nobs = settings.get("num_obs", 123)
    nchunks = settings.get("num_chunks", 23)
    nfilters = settings.get("num_filters", 2)

    return {
        "providers": settings.get("providers", {}),
        "variable_path": settings.get("variable_path", "data/"),
        "dimensions": {
            "unfiltered_events": nobs,
            "l1b_chunks": nchunks,
            "l1b_offsets": nchunks,
            "filters": nfilters,
            "scalar": 1,
        },
        "variables": {},
        "sector_variables": {
            "event_id": {
                "format": "u4",
                "shape": ("unfiltered_events",),
                "long_name": "ID of LI L2 Event",
                "default_data": lambda: rand_type(nobs, np.uint32)
            },
            "group_id": {
                "format": "u4",
                "shape": ("unfiltered_events",),
                "long_name": "ID of associated LI L2 Group object",
                "default_data": lambda: rand_type(nobs, np.uint32)
            },
            "l1b_chunk_ids": {
                "format": "u4",
                "shape": ("l1b_chunks",),
                "long_name": "Array of L1b event chunk IDs",
                "default_data": lambda: rand_type(nchunks, np.uint32)
            },
            "l1b_chunk_offsets": {
                "format": "u4",
                "shape": ("l1b_offsets",),
                "long_name": "Array offset for L1b event chunk boundaries",
                "default_data": lambda: np.arange(nchunks)
            },
            "l1b_window": {
                "format": "u4",
                "shape": ("unfiltered_events",),
                "long_name": "window index of associated L1b event",
                "default_data": lambda: rand_type(nobs, np.uint32)
            },
            "filter_values": {
                "format": "u1",
                "shape": ("unfiltered_events", "filters",),
                "fill_value": 255,
                "scale_factor": 0.004,
                "add_offset": 0.0,
                "long_name": "L2 filter results",
                "default_data": lambda: rand_type((nobs, nfilters), np.uint8)
            },
            "epoch_time": {
                "format": "f8",
                "shape": ("scalar",),
                "long_name": "Start time of integration frame",
                "default_data": lambda: 1.234,
                "precision": "1 millisecond",
                "time_standard": "UTC",
                "standard_name": "time",
                "units": "seconds since 2000-01-01 00:00:00.0",
            },
            "time_offset": {
                "format": "f4",
                "shape": ("unfiltered_events",),
                "fill_value": 9.96921e36,
                "long_name": "Time offset from epoch time",
                "default_data": lambda: np.linspace(0.0, 1000.0, nobs),
                "units": "seconds",
            },
            "detector": {
                "format": "u1",
                "fill_value": 255,
                "shape": ("scalar",),
                "long_name": "ID of detector for this group",
                "default_data": lambda: 1,
                "meanings": "1 = detector_1, 2 = detector_2, 3 = detector_3, 4 = detector_4",
            },
            "l1b_filter_qa": {
                "format": "u1",
                "fill_value": 255,
                "shape": ("unfiltered_events",),
                "add_offset" : 0.0,
                "scale_offset": 0.004,
                "long_name": "L1b event confidence",
                "default_data": lambda : rand_type(nobs, np.uint8),
            },
            "l2_group_filter_qa": {
                "format": "u1",
                "fill_value": 255,
                "shape": ("unfiltered_events",),
                "add_offset" : 0.0,
                "scale_offset": 0.004,
                "long_name": "L2 group confidence",
                "default_data": lambda: (np.arange(nobs) + 10000),
            },
        }
    }


def l2_lef_schema(settings=None):
    """Define schema for LI L2 LEF product."""
    epoch_ts = dt.datetime(2000, 1, 1, 0, 0, 0, 0)
    start_time = dt.datetime.now()
    start_ts = (start_time - epoch_ts).total_seconds()

    settings = settings or {}

    nobs = settings.get("num_obs", 123)

    return {
        "providers": settings.get("providers", {}),
        "variable_path": settings.get("variable_path", "data/"),
        "dimensions": {
            "events": nobs,
            "scalar": 1,
        },
        "variables": {
            "l1b_geolocation_warning": {
                "format": "i1",
                "shape": ("scalar",),  # test explicitly the scalar case
                "long_name": "L1b event geolocation warning",
                "default_data": lambda: 0
            },
            "l1b_missing_warning": {
                "format": "i1",
                "shape": ("scalar",),
                "long_name": "Expected L1b inputs missing",
                "default_data": lambda: 0
            },
            "l1b_radiometric_warning": {
                "format": "i1",
                "shape": ("scalar",),
                "long_name": "L1b event radiometric warning",
                "default_data": lambda: 0
            },
        },
        "sector_variables": {
            "event_id": {
                "format": "u4",
                "shape": ("events",),
                "long_name": "ID of LI L2 Event",
                "default_data": lambda: np.arange(1, nobs + 1)
            },
            "group_id": {
                "format": "u4",
                "shape": ("events",),
                "long_name": "ID of associated LI L2 Group object",
                "default_data": lambda: np.arange(1, nobs + 1)
            },
            "flash_id": {
                "format": "u4",
                "shape": ("events",),
                "long_name": "ID of associated LI L2 Flash object",
                "default_data": lambda: np.arange(1, nobs + 1)
            },
            "detector": {
                "format": "u1",
                "shape": ("scalar",),
                "fill_value": 255,
                "long_name": "ID of detector for this group",
                "meaning": "1 = detector_1, 2 = detector_2, 3 = detector_3, 4 = detector_4",
                "default_data": lambda: 1
            },
            "latitude": {
                "format": "i2",
                "shape": ("events",),
                "fill_value": -32767,
                "long_name": "Latitude of group",
                "units": "degrees_north",
                "standard_name": "latitude",
                "default_data": lambda: np.linspace(-90, 90, nobs)
            },
            "longitude": {
                "format": "i2",
                "shape": ("events",),
                "fill_value": -32767,
                "long_name": "Longitude of group",
                "units": "degrees_east",
                "standard_name": "longitude",
                "default_data": lambda: np.linspace(-180, 80, nobs)
            },
            "radiance": {
                "format": "u2",
                "shape": ("events",),
                "long_name": "Radiance of Flash",
                "standard_name": "radiance",
                "units": "mW.m-2.sr-1",
                "default_data": lambda: np.clip(np.round(RANDOM_GEN.normal(500, 100, nobs)), 1, 2 ** 16 - 1)
            },
            "event_filter_qa": {
                "format": "u2",
                "shape": ("events",),
                "fill_value": 255,
                "long_name": "L2 event pre-filtering quality assurance value",
                "default_data": lambda: RANDOM_GEN.integers(1, 2 ** 8 - 1, nobs)
            },
            "epoch_time": {
                "format": "f8",
                "shape": ("scalar",),
                "long_name": "Start time of integration frame",
                "units": "seconds since 2000-01-01 00:00:00.0",
                "default_data": lambda: start_ts
            },
            "time_offset": {
                "format": "f4",
                "shape": ("events",),
                "long_name": "Time offset from epoch time",
                "units": "seconds",
                "default_data": lambda: RANDOM_GEN.uniform(1, 2 ** 31 - 1, nobs)
            },
            "detector_row": {
                "format": "u2",
                "shape": ("events",),
                "fill_value": 65535,
                "long_name": "Detector row position of event pixel",
                "units": "1",
                "default_data": lambda: RANDOM_GEN.integers(1, 1000, nobs)
            },
            "detector_column": {
                "format": "u2",
                "shape": ("events",),
                "fill_value": 65535,
                "long_name": "Detector column position of event pixel",
                "units": "1",
                "default_data": lambda: RANDOM_GEN.integers(1, 1000, nobs)
            },
        }
    }


def l2_lgr_schema(settings=None):
    """Define schema for LI L2 LGR product."""
    settings = settings or {}
    ngrps = settings.get("num_groups", 120)

    return {
        "providers": settings.get("providers", {}),
        "variable_path": settings.get("variable_path", ""),
        "dimensions": {
            "groups": ngrps,
            "scalar": 1,
        },
        "variables": {
            "group_time": {
                "format": "f8",
                "shape": ("groups",),
                "long_name": "Start time of integration frame",
                "standard_name": "time",
                "units": "seconds since 2000-01-01 00:00:00.0",
                "precision": "0.001",
                "time_standard": "UTC",
                "default_data": lambda: np.linspace(-90, 90, ngrps)
            },
            "latitude": {
                "format": "i2",
                "shape": ("groups",),
                "long_name": "Latitude of group",
                "units": "degrees_north",
                "fill_value": -32767,
                "default_data": lambda: np.linspace(-90, 90, ngrps)
            },
            "longitude": {
                "format": "i2",
                "shape": ("groups",),
                "long_name": "Longitude of group",
                "fill_value": -32767,
                "units": "degrees_east",
                "default_data": lambda: np.linspace(-180, 80, ngrps)
            },
            "radiance": {
                "format": "u2",
                "shape": ("groups",),
                "long_name": "Radiance of group",
                "fill_value": 65535,
                "units": "mW.m-2.sr-1",
                "scale_factor": 0.5,
                "add_offset": 0.0,
                "default_data": lambda: rand_type(ngrps, np.uint16)
            },
           "group_id": {
                "format": "u4",
                "shape": ("groups",),
                "long_name": "LI L2 group IDs",
                "default_data": lambda: np.linspace(-180, 80, ngrps)
            },
           "flash_id": {
                "format": "u4",
                "shape": ("groups",),
                "long_name": "ID of associated LI L2 Flash object with each group",
                "default_data": lambda: np.linspace(-180, 80, ngrps)
            },
           "number_of_events": {
                "format": "u2",
                "shape": ("groups",),
                "long_name": "Number of events in each group",
                "default_data": lambda: np.linspace(-180, 80, ngrps)
            },
           "group_filter_qa": {
                "format": "u1",
                "shape": ("groups",),
                "fill_value": 255,
                "long_name": "L2 filtered group quality assurance value",
                "add_offset": 0.0,
                "scale_factor": 0.004,
                "default_data": lambda: np.linspace(-180, 80, ngrps)
            },
            "l1b_geolocation_warning": {
                "format": "i1",
                "shape": ("scalar",),  # test explicitly the scalar case
                "long_name": "L1b event geolocation warning",
                "default_data": lambda: 0
            },
            "l1b_radiometric_warning": {
                "format": "i1",
                "shape": ("scalar",),
                "long_name": "L1b event radiometric warning",
                "default_data": lambda: 0
            },
        }
    }


def l2_lfl_schema(settings=None):
    """Define schema for LI L2 LFL product."""
    settings = settings or {}

    nobs = settings.get("num_obs", 1234)
    epoch = dt.datetime(2000, 1, 1)
    stime = (dt.datetime(2019, 1, 1) - epoch).total_seconds()
    etime = (dt.datetime(2019, 1, 2) - epoch).total_seconds()

    return {
        "providers": settings.get("providers", {}),
        "variable_path": settings.get("variable_path", ""),
        "dimensions": {
            "flashes": nobs,
            "scalar": 1,
        },
        "variables": {
            "latitude": {
                "format": "i2",
                "shape": ("flashes",),
                "long_name": "Latitude of Flash",
                "standard_name": "latitude",
                "units": "degrees_north",
                "fill_value": -32767,
                "add_offset": 0.0,
                "scale_factor": 0.0027,
                # Note: using a default range of [-88.3deg, 88.3deg] to stay in
                # the available type range [-32727,32727] with scaling:
                "default_data": lambda: np.linspace(-88.3 / 0.0027, 88.3 / 0.0027, nobs)
            },
            "longitude": {
                "format": "i2",
                "shape": ("flashes",),
                "long_name": "Longitude of Flash",
                "standard_name": "longitude",
                "units": "degrees_east",
                "fill_value": -32767,
                "add_offset": 0.0,
                "scale_factor": 0.0027,
                # Note: using a default range of [-88.3deg, 88.3deg] to stay in
                # the available type range [-32727,32727] with scaling:
                "default_data": lambda: np.linspace(-88.3 / 0.0027, 88.3 / 0.0027, nobs)
            },
            "radiance": {
                "format": "u2",
                "shape": ("flashes",),
                "fill_value" : 65535,
                "long_name": "Radiance of Flash",
                "standard_name": "radiance",
                "units": "mW.m-2.sr-1",
                "default_data": lambda: np.round(RANDOM_GEN.normal(500, 100, nobs))
            },
            "flash_duration": {
                "format": "u2",
                "shape": ("flashes",),
                "long_name": "Flash duration",
                "standard_name": "flash_duration",
                "units": "ms",
                "default_data": lambda: np.linspace(0, 1000, nobs)
            },
            "flash_filter_confidence": {
                "format": "u1",
                "shape": ("flashes",),
                "fill_value": 255,
                "long_name": "L2 filtered flash confidence",
                "standard_name": "flash_filter_confidence",
                "default_data": lambda: np.clip(np.round(RANDOM_GEN.normal(20, 10, nobs)), 1, 2 ** 7 - 1)
            },
            "flash_footprint": {
                "format": "u2",
                "shape": ("flashes",),
                "long_name": "Flash footprint size",
                "standard_name": "flash_footprint",
                "units": "L1 grid pixels",
                "default_data": lambda: np.maximum(1, np.round(RANDOM_GEN.normal(5, 3, nobs)))
            },
            "flash_id": {
                "format": "u4",
                "shape": ("flashes",),
                "long_name": "Flash footprint size",
                "standard_name": "flash_id",
                "default_data": lambda: np.arange(1, nobs + 1)
            },
            "flash_time": {
                "format": "f8",
                "shape": ("flashes",),
                "long_name": "Nominal flash time",
                "units": "seconds since 2000-01-01 00:00:00.0",
                "standard_name": "time",
                "precision": "1 millisecond",
                "default_data": lambda: RANDOM_GEN.uniform(stime, etime, nobs)
            },
            "l1b_geolocation_warning": {
                "format": "i1",
                "shape": ("scalar",),
                "long_name": "L1b geolocation warning",
                "default_data": lambda: -127
            },
            "l1b_radiometric_warning": {
                "format": "i1",
                "shape": ("scalar",),
                "long_name": "L1b radiometric warning",
                "default_data": lambda: -127
            },
            "number_of_events": {
                "format": "u2",
                "shape": ("flashes",),
                "long_name": "Number of events in each flash",
                "default_data": lambda: rand_type(nobs, np.uint16)
            },
            "number_of_groups": {
                "format": "u2",
                "shape": ("flashes",),
                "long_name": "Number of flashes in each flash",
                "default_data": lambda: rand_type(nobs, np.uint16)
            },
        }
    }


def l2_af_schema(settings=None):
    """Define schema for LI L2 AF product."""
    settings = settings or {}
    nacc = settings.get("num_accumulations", 1)
    npix = settings.get("num_pixels", 1234)
    return {
        "providers": settings.get("providers", {}),
        "variable_path": settings.get("variable_path", ""),
        "dimensions": accumulation_dimensions(nacc, npix),
        "variables": {
            "accumulation_offsets": {
                "format": "u4",
                "shape": ("accumulations",),
                "default_data": lambda: rand_type(nacc, np.uint32)
            },
            "accumulation_start_times": {
                "format": "f8",
                "shape": ("accumulations",),
                "long_name": "Accumulation start time",
                "units": "seconds since 2000-01-01 00:00:00.0",
                "precision": "0.001",
                "default_data": lambda: np.linspace(0.0, 1000.0, nacc)
            },
            "l1b_geolocation_warning": {
                "format": "i1",
                "shape": ("accumulations",),
                "long_name": "L1b geolocation warning",
                "default_data": lambda: rand_type(nacc, np.int8)
            },
            "l1b_radiometric_warning": {
                "format": "i1",
                "shape": ("accumulations",),
                "long_name": "L1b radiometric warning",
                "default_data": lambda: rand_type(nacc, np.int8)
            },
            "average_flash_qa": {
                "format": "u1",
                "shape": ("accumulations",),
                "default_data": lambda: rand_type(nacc, np.uint8),
                "fill_value": 255,
                "scale_factor": 0.004,
                "add_offset": 0.0,
                "long_name": "average flash confidence value",

            },
            "flash_accumulation": {
                "format": "u2",
                "shape": ("pixels",),
                "fill_value": 65535,
                "scale_factor": 0.001,
                "long_name": "Per area accumulation of flashes",
                "grid_mapping": "mtg_geos_projection",
                "units": "flashes/pixel",
                "coordinate": "sparse: x y" ,
                "default_data": lambda: np.clip(np.round(RANDOM_GEN.normal(1, 2, npix)), 1, 2 ** 16 - 1)
            },
            "mtg_geos_projection": mtg_geos_projection(),
            "x": fci_grid_definition("X", npix),
            "y": fci_grid_definition("Y", npix),
        }
    }

def l2_afr_schema(settings=None):
    """Define schema for LI L2 AFR product."""
    settings = settings or {}
    nacc = settings.get("num_accumulations", 1)
    npix = settings.get("num_pixels", 1234)

    return {
        "providers": settings.get("providers", {}),
        "variable_path": settings.get("variable_path", ""),
        "dimensions": accumulation_dimensions(nacc, npix),
        "variables": {
            "accumulation_offsets": {
                "format": "u4",
                "shape": ("accumulations",),
                "default_data": lambda: rand_type(nacc, np.uint32)
            },
            "accumulation_start_times": {
                "format": "f8",
                "shape": ("accumulations",),
                "long_name": "Accumulation start time",
                "units": "seconds since 2000-01-01 00:00:00.0",
                "precision" : "0.001",
                "default_data": lambda: np.linspace(0.0, 1000.0, nacc)
            },
            "l1b_geolocation_warning": {
                "format": "i1",
                "shape": ("accumulations",),
                "long_name": "L1b geolocation warning",
                "default_data": lambda: rand_type(nacc, np.int8)
            },
            "l1b_radiometric_warning": {
                "format": "i1",
                "shape": ("accumulations",),
                "long_name": "L1b radiometric warning",
                "default_data": lambda: rand_type(nacc, np.int8)
            },
            "average_flash_qa": {
                "format": "u1",
                "shape": ("accumulations",),
                "default_data": lambda: rand_type(nacc, np.uint8),
                "fill_value": 255,
                "scale_factor": 0.004,
                "add_offset": 0.0,
                "long_name":"average flash confidence value",

            },
            "flash_radiance": {
                "format": "u2",
                "shape": ("pixels",),
                "fill_value": 65535,
                "scale_factor": 1.0,
                "add_offset": 0.0,
                "long_name": "Area averaged flash radiance accumulation",
                "grid_mapping": "mtg_geos_projection",
                "units": "mW.m-2.sr-1",
                "coordinate": "sparse: x y" ,
                "default_data": lambda: RANDOM_GEN.integers(low=1, high=6548, size=(npix), dtype=np.int16)
            },
            "mtg_geos_projection": mtg_geos_projection(),
            "x": fci_grid_definition("X", npix),
            "y": fci_grid_definition("Y", npix),
        }
    }

def l2_afa_schema(settings=None):
    """Define schema for LI L2 AFA product."""
    settings = settings or {}
    nacc = settings.get("num_accumulations", 1)
    npix = settings.get("num_pixels", 1234)

    return {
        "providers": settings.get("providers", {}),
        "variable_path": settings.get("variable_path", ""),
        "dimensions": accumulation_dimensions(nacc, npix),
        "variables": {
            "accumulation_offsets": {
                "format": "u4",
                "shape": ("accumulations",),
                "default_data": lambda: rand_type(nacc, np.uint32)
            },
            "accumulation_start_times": {
                "format": "f8",
                "shape": ("accumulations",),
                "long_name": "Accumulation start time",
                "units": "seconds since 2000-01-01 00:00:00.0",
                "precision" : "0.001",
                "default_data": lambda: np.linspace(0.0, 1000.0, nacc)
            },
            "l1b_geolocation_warning": {
                "format": "i1",
                "shape": ("accumulations",),
                "long_name": "L1b geolocation warning",
                "default_data": lambda: rand_type(nacc, np.int8)
            },
            "l1b_radiometric_warning": {
                "format": "i1",
                "shape": ("accumulations",),
                "long_name": "L1b radiometric warning",
                "default_data": lambda: rand_type(nacc, np.int8)
            },
            "average_flash_qa": {
                "format": "u1",
                "shape": ("accumulations",),
                "default_data": lambda: rand_type(nacc, np.uint8),
                "fill_value": 255,
                "scale_factor": 0.004,
                "add_offset": 0.0,
                "long_name":"average flash confidence value",

            },
            "accumulated_flash_area": {
                "format": "u4",
                "shape": ("pixels",),
                "long_name": "Number of contributing unique flashes to each pixel",
                "grid_mapping": "mtg_geos_projection",
                "coordinate": "sparse: x y" ,
                "default_data": lambda: np.mod(np.arange(npix), 10) + 1
            },
            "mtg_geos_projection": mtg_geos_projection(),
            "x": fci_grid_definition("X", npix),
            "y": fci_grid_definition("Y", npix),
        }
    }


def accumulation_dimensions(nacc, nobs):
    """Set dimensions for the accumulated products."""
    return {
        "accumulations": nacc,
        "pixels": nobs,
    }


def fci_grid_definition(axis, nobs):
    """FCI grid definition on X or Y axis."""
    scale_factor = 5.58871526031607e-5
    add_offset = -0.15561777642350116
    if axis == "X":
        long_name = "azimuth angle encoded as column"
        standard_name = "projection_x_coordinate"
        scale_factor *= -1
        add_offset *= -1
    else:
        long_name = "zenith angle encoded as row"
        standard_name = "projection_y_coordinate"

    return {
        "format": "i2",
        "shape": ("pixels",),
        "add_offset": add_offset,
        "axis": axis,
        "long_name": long_name,
        "scale_factor": scale_factor,
        "standard_name": standard_name,
        "units": "radian",
        "valid_range": np.asarray([1, 5568]),
        "default_data": lambda: np.clip(np.round(RANDOM_GEN.normal(2000, 500, nobs)), 1, 2 ** 16 - 1)
    }


def mtg_geos_projection():
    """MTG geos projection definition."""
    return {
        "format": "i4",
        "shape": ("accumulations",),
        "grid_mapping_name": "geostationary",
        "inverse_flattening": 298.257223563,
        "latitude_of_projection_origin": 0,
        "longitude_of_projection_origin": 0,
        "perspective_point_height": 3.57864e7,
        "semi_major_axis": 6378137.0,
        "semi_minor_axis": 6356752.31424518,
        "sweep_angle_axis": "y",
        "long_name": "MTG geostationary projection",
        "default_data": lambda: -2147483647
    }

#Dict containing the expecteded dtype output for each variable
expected_product_dtype = {
    "2-LE": {
        "event_id": np.uint32,
        "group_id": np.uint32,
        "l1b_chunk_ids": np.uint32,
        "l1b_chunk_offsets": np.uint32,
        "l1b_window": np.uint32,
        "filter_values": np.float32,
        "flash_id": np.uint32,
        "time_offset": np.dtype("timedelta64[ns]"),
        "epoch_time": np.dtype("datetime64[ns]"),
        "detector": np.float32,
        "l1b_filter_qa": np.float32,
        "l2_group_filter_qa": np.float32,
    },
    "2-LEF": {
        "l1b_geolocation_warning": np.int8,
        "l1b_radiometric_warning": np.int8,
        "l1b_missing_warning": np.int8,
        "event_id": np.uint32,
        "group_id": np.uint32,
        "flash_id": np.uint32,
        "detector": np.float32,
        "latitude": np.float32,
        "longitude": np.float32,
        "radiance": np.uint16,
        "event_filter_qa": np.float32,
        "epoch_time": np.dtype("datetime64[ns]"),
        "time_offset": np.dtype("timedelta64[ns]"),
        "detector_row": np.float32,
        "detector_column": np.float32,
    },
    "2-LGR": {
        "group_time": np.dtype("datetime64[ns]"),
        "l1b_geolocation_warning": np.int8,
        "l1b_radiometric_warning": np.int8,
        "latitude": np.float32,
        "longitude": np.float32,
        "radiance": np.float32,
        "group_id": np.uint32,
        "flash_id": np.uint32,
        "number_of_events": np.uint16,
        "group_filter_qa": np.float32,
    },
    "2-LFL": {
        "latitude": np.float32,
        "longitude": np.float32,
        "radiance": np.float32,
        "flash_duration": np.dtype("timedelta64[ns]"),
        "flash_filter_confidence": np.float32,
        "flash_footprint": np.uint16,
        "flash_id": np.uint32,
        "flash_time": np.dtype("datetime64[ns]"),
        "l1b_geolocation_warning": np.int8,
        "l1b_radiometric_warning": np.int8,
        "l1b_missing_warning": np.int8,
        "number_of_events": np.uint16,
        "number_of_groups": np.uint16,
    },
    "2-AF": {
        "l1b_geolocation_warning": np.int8,
        "l1b_radiometric_warning": np.int8,
        "accumulation_offsets": np.uint32,
        "accumulation_start_times": np.dtype("datetime64[ns]"),
        "average_flash_qa": np.float32,
        "mtg_geos_projection": np.int32,
        "latitude": np.float32,
        "longitude": np.float32,
        "x": np.float64,
        "y": np.float64,
        "flash_accumulation": np.float32,
    },
    "2-AFA": {
        "l1b_geolocation_warning": np.int8,
        "l1b_radiometric_warning": np.int8,
        "accumulation_offsets": np.uint32,
        "accumulation_start_times": np.dtype("datetime64[ns]"),
        "average_flash_qa": np.float32,
        "mtg_geos_projection": np.int32,
        "latitude": np.float32,
        "longitude": np.float32,
        "x": np.float64,
        "y": np.float64,
        "accumulated_flash_area": np.uint32,
    },
    "2-AFR": {
        "l1b_geolocation_warning": np.int8,
        "l1b_radiometric_warning": np.int8,
        "l1b_missing_warning": np.int8,
        "accumulation_offsets": np.uint32,
        "accumulation_start_times": np.dtype("datetime64[ns]"),
        "latitude": np.float32,
        "longitude": np.float32,
        "average_flash_qa": np.float32,
        "mtg_geos_projection": np.int32,
        "x": np.float64,
        "y": np.float64,
        "flash_radiance": np.float32,
    },
}


products_dict = {
    "2-LE": {"ftype": "li_l2_le_nc", "schema": l2_le_schema},
    "2-LEF": {"ftype": "li_l2_lef_nc", "schema": l2_lef_schema},
    "2-LGR": {"ftype": "li_l2_lgr_nc", "schema": l2_lgr_schema},
    "2-LFL": {"ftype": "li_l2_lfl_nc", "schema": l2_lfl_schema},
    "2-AF": {"ftype": "li_l2_af_nc", "schema": l2_af_schema},
    "2-AFA": {"ftype": "li_l2_afa_nc", "schema": l2_afa_schema},
    "2-AFR": {"ftype": "li_l2_afr_nc", "schema": l2_afr_schema},
}


def get_product_schema(pname, settings=None):
    """Retrieve an LI product schema given its name."""
    return products_dict[pname]["schema"](settings)


def extract_filetype_info(filetype_infos, filetype):
    """Extract Satpy-conform filetype_info from filetype_infos fixture."""
    ftype_info = filetype_infos[filetype]
    ftype_info["file_type"] = filetype
    return ftype_info


def set_variable_path(var_path, desc, sname):
    """Replace variable default path if applicable and ensure trailing separator."""
    vpath = desc.get("path", var_path)
    # Ensure we have a trailing separator:
    if vpath != "" and vpath[-1] != "/":
        vpath += "/"
    if sname != "":
        vpath += sname + "/"
    return vpath


def populate_dummy_data(data, names, details):
    """Populate variable with dummy data."""
    vname, sname = names
    desc, providers, settings = details
    if vname in providers:
        prov = providers[vname]
        # prov might be a function or directly an array that we assume will be of the correct shape:
        data[:] = prov(vname, sname, settings) if callable(prov) else prov
    else:
        # Otherwise we write the default data:
        if data.shape == ():
            # scalar case
            data = desc["default_data"]()
        else:
            data[:] = desc["default_data"]()


def add_attributes(attribs, ignored_attrs, desc):
    """Add all the custom properties directly as attributes."""
    for key, val in desc.items():
        if key not in ignored_attrs:
            attribs[key] = val


# Note: the helper class below has some missing abstract class implementation,
# but that is not critical to us, so ignoring them for now.
class FakeLIFileHandlerBase(FakeNetCDF4FileHandler):  # pylint: disable=abstract-method
    """Class for faking the NetCDF4 Filehandler."""

    # Optional parameter that may be provided at the time of the creation of this file handler
    # to customize the generated content. This may be either a simple dictionary or a callable
    # if a callable is provided it will be called to retrieve the actual parameter to be used:
    schema_parameters = None

    def get_variable_writer(self, dset, settings):
        """Get a variable writer."""
        # use a variable path prefix:
        var_path = settings.get("variable_path", "")

        # Also keep track of the potential providers:
        providers = settings.get("providers", {})

        # list of ignored attribute names:
        ignored_attrs = ["path", "format", "shape", "default_data", "fill_value"]

        # dictionary of dimensions:
        dims = settings.get("dimensions", {})

        def write_variable(vname, desc, sname=""):
            """Write a variable in our dataset."""
            # get numeric shape:
            shape_str = desc["shape"]
            shape = tuple([dims[dname] for dname in shape_str])

            # Get the desired data type:
            dtype = TYPE_MAP[desc["format"]]

            # Prepare a numpy array with the appropriate shape and type:
            data = np.zeros(shape, dtype=dtype)

            # Replace variable default path if applicable:
            vpath = set_variable_path(var_path, desc, sname)

            # Variable full name:
            full_name = f"{vpath}{vname}"

            # Add all the custom properties directly as attributes:
            attribs = {}
            add_attributes(attribs, ignored_attrs, desc)

            # Rename the fill value attribute:
            if "fill_value" in desc:
                attribs["_FillValue"] = desc["fill_value"]

            names = [vname, sname]
            details = [desc, providers, settings]
            populate_dummy_data(data, names, details)

            # Now we assign that data array:
            dset[full_name] = xr.DataArray(data, dims=shape_str, attrs=attribs)

            # Write the copy of the content:
            self.content[full_name] = data

        return write_variable

    def get_test_content(self, filename, filename_info, filetype_info):
        """Get the content of the test data.

        Here we generate the default content we want to provide depending
        on the provided filename infos.
        """
        # Retrieve the correct schema to write with potential customization parameters:
        params = FakeLIFileHandlerBase.schema_parameters
        if callable(params):
            # Note: params *IS* callable below:
            params = params(filename, filename_info, filetype_info)  # pylint: disable=not-callable

        settings = get_product_schema(filetype_info["file_desc"]["product_type"], params)

        # Resulting dataset:
        dset = {}

        # Also keep a copy of the written content:
        self.content = {}

        # Retrieve the variable writer function
        write_variable = self.get_variable_writer(dset, settings)

        # Write all the raw (i.e not in sectors) variables:
        self.write_variables(settings, write_variable)

        # Write the sector variables:
        self.write_sector_variables(settings, write_variable)

        return dset

    def write_variables(self, settings, write_variable):
        """Write raw (i.e. not in sectors) variables."""
        if "variables" in settings:
            variables = settings.get("variables")
            for vname, desc in variables.items():
                write_variable(vname, desc)

    def write_sector_variables(self, settings, write_variable):
        """Write the sector variables."""
        if "sector_variables" in settings:
            sector_vars = settings.get("sector_variables")
            sectors = settings.get("sectors", ["north", "east", "south", "west"])

            for sname in sectors:
                for vname, desc in sector_vars.items():
                    write_variable(vname, desc, sname)
