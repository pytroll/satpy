"""Reader for GMS-1-4 VISSR Level 1B data.

Introduction
------------
The ``gms4_vissr_l1b`` reader can decode, navigate and calibrate Level 1B data
from the Visible and Infrared Spin Scan Radiometer (VISSR) in `VISSR
archive format`. Corresponding platforms are GMS-1 to GMS-4
(Japanese Geostationary Meteorological Satellite).

Unlike GMS-5, GMS-1-4 VISSR only has two channels, each stored in a separate file:

.. code-block:: none

    VS901110.Z23
    IR901110.Z23

This is how to read them with Satpy:

.. code-block:: python

    from satpy import Scene
    import glob

    filenames = glob.glob("/data/VS*")
    scene = Scene(filenames, reader="gms4-vissr_l1b")
    scene.load(["VIS"])


References:
~~~~~~~~~~~

Details about platform, instrument and data format can be found in the
following references:

    - `VISSR Format Description`_
    - `GMS User Guide`_

.. _VISSR Format Description:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/VISSR_FORMAT_GMS-4.pdf
.. _GMS User Guide:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/GMS_Users_Guide_3rd_Edition_Rev1.pdf


Compression
-----------

Gzip-compressed VISSR files can be decompressed on the fly using
:class:`~satpy.readers.core.remote.FSFile`:

.. code-block:: python

    import fsspec
    from satpy import Scene
    from satpy.readers.core.remote import FSFile

    filename = "IR901110.Z23.gz"
    open_file = fsspec.open(filename, compression="gzip")
    fs_file = FSFile(open_file)
    scene = Scene([fs_file], reader="gms4-vissr_l1b")
    scene.load(["IR"])


Calibration
-----------

Sensor counts are calibrated by looking up reflectance/temperature values in the
calibration tables included in each file. See section 2.2 in the VISSR user
guide.


Navigation
----------

VISSR images are oversampled and not rectified.

Some older GMS-1/2/3 archive files don't populate the mode block's
``spin_rate`` telemetry field (it reads back as 0, which is physically
impossible for a spin-scan imager). When that happens, this reader falls
back to the coordinate conversion parameters segment's own
``daily_mean_spin_rate`` field -- real, file-specific telemetry that was
confirmed populated on multiple satellites (GMS-1 and GMS-3) even when
the mode block's own value wasn't.


Oversampling
~~~~~~~~~~~~
Like other VISSR archive formats, GMS-1..4 VISSR oversamples the viewed scene
in the E-W (pixel) direction: each channel's stepping angle (line-to-line)
and sampling angle (pixel-to-pixel) are stored as telemetry in every file's
coordinate transformation parameters block, and the sampling angle is
smaller than the pixel's own angular size -- so consecutive pixels overlap
on the ground rather than tiling it exactly. Unlike GMS-5, the JMA VISSR
format spec does not publish fixed IR/VIS angle values for GMS-1..4 (they
vary slightly file to file), so this reader always derives the actual
oversampling ratio from each file's own ``sampling_angle_ir``/
``sampling_angle_vis`` and ``stepping_angle_ir``/``stepping_angle_vis``
fields rather than assuming a constant. Nominal nadir resolution is
~1.25 km for VIS and ~5 km for IR.

This cannot be represented by a pyresample area definition, so each dataset
is accompanied by 2-dimensional longitude and latitude coordinates. For
resampling purpose a full disc area definition with uniform sampling is provided
via

.. code-block:: python

    scene[dataset].attrs["area_def_uniform_sampling"]


Rectification
~~~~~~~~~~~~~

VISSR images are not rectified. That means lon/lat coordinates are different

1) for all channels of the same repeat cycle, even if their spatial resolution
   is identical
2) for different repeat cycles, even if the channel is identical

However, the above area definition is using the nominal subsatellite point as
projection center. As this rarely changes, the area definition is pretty
constant.


Performance
~~~~~~~~~~~

Navigation of VISSR images is computationally expensive, because for each pixel
the view vector of the (rotating) instrument needs to be intersected with the
earth, including interpolation of attitude and orbit prediction. For IR channels
this takes about 10 seconds, for VIS channels about 160 seconds.


Space Pixels
------------

VISSR produces data for pixels outside the Earth disk (i.e. atmospheric limb or
deep space pixels). By default, these pixels are masked out as they contain
data of limited or no value, but some applications do require these pixels.
To turn off masking, set ``mask_space=False`` upon scene creation:

.. code-block:: python

    import satpy
    import glob

    filenames = glob.glob("VS*")
    scene = satpy.Scene(filenames,
                        reader="gms4-vissr_l1b",
                        reader_kwargs={"mask_space": False})
    scene.load(["VIS"])


Partial Scans
-------------

On demand a special Typhoon schedule would be activated between
03:00 and 05:00 UTC.

"""

import functools
import os

import dask.array as da
import numpy as np
import xarray as xr

import satpy.readers.core._geos_area as geos_area
import satpy.readers.gms.gms4_vissr_format as fmt
import satpy.readers.gms.gms_vissr_navigation as nav_shared
from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.utils import generic_open
from satpy.readers.hrit_jma import mjd2datetime64
from satpy.utils import datetime64_to_pydatetime


def _mjd_to_datetime(mjd):
    return datetime64_to_pydatetime(mjd2datetime64(np.array(mjd)))


class GmsVissrFileHandler(BaseFileHandler):
    """File handler for GMS-1..4 native VISSR archive files."""

    def __init__(self, filename, filename_info, filetype_info, mask_space=True):
        """Open *filename* and parse its header blocks (mode, coordinate conversion, calibration, etc.)."""
        super().__init__(filename, filename_info, filetype_info)
        self._l1b = GmsVissrL1bFile(filename)
        self._mask_space = mask_space

    @property
    def start_time(self):
        """Nominal start time of the scan, from the file's own scheduled_observation_time telemetry."""
        return _mjd_to_datetime(float(self._l1b.coord["scheduled_observation_time"]))

    @property
    def end_time(self):
        """Nominal end time of the scan, from the last valid per-line scan_time in the LCW."""
        if self._l1b.scan_times.size:
            last_valid = self._l1b.scan_times[np.isfinite(self._l1b.scan_times)]
            if last_valid.size:
                return _mjd_to_datetime(float(last_valid.max()))
        return self.start_time

    @property
    def sensor_names(self):
        """Set of sensor names this file handler provides data for."""
        return {"VISSR"}

    def combine_info(self, all_infos):
        """Combine per-file info dicts; GMS-1..4 archives are always a single VIS+IR file pair, never segmented."""
        if len(all_infos) == 1:
            return all_infos[0]
        return super().combine_info(all_infos)

    def get_dataset(self, dataset_id, ds_info):
        """Return the calibrated DataArray for *dataset_id*, or None if this file doesn't hold that channel."""
        requested_channel = ds_info.get("name", getattr(dataset_id, "name", None))
        if requested_channel != self._l1b.channel:
            return None

        data_array = self._l1b.get_dataset(mask_space=self._mask_space)
        data_array.name = ds_info.get("name", self._l1b.channel)
        data_array.attrs.update(ds_info)
        data_array.attrs["start_time"] = self.start_time
        data_array.attrs["end_time"] = self.end_time
        data_array.attrs["platform_name"] = self._platform_name()
        data_array.attrs["sensor"] = "VISSR"

        nadir_resolution = (
            float(self._l1b.coord[f"stepping_angle_{self._l1b.channel.lower()}"])
            * float(self._l1b.mode["satellite_height"])
        )
        data_array.coords["longitude"].attrs["resolution"] = nadir_resolution
        data_array.coords["latitude"].attrs["resolution"] = nadir_resolution

        data_array.coords["longitude"].attrs["standard_name"] = "longitude"
        data_array.coords["latitude"].attrs["standard_name"] = "latitude"

        try:
            data_array.attrs["area_def_uniform_sampling"] = self._get_area_def_uniform_sampling(dataset_id)
        except (KeyError, ValueError, ZeroDivisionError) as e:
            data_array.attrs["area_def_uniform_sampling"] = None
            data_array.attrs["area_def_uniform_sampling_error"] = str(e)

        return data_array

    def _get_area_def_uniform_sampling(self, dataset_id):
        estimator = AreaDefEstimator(self._l1b, self._platform_name())
        return estimator.get_area_def_uniform_sampling(dataset_id)

    def _platform_name(self):
        name = bytes(self._l1b.mode["satellite_name"]).split(b"\x00")[0].decode("ascii", "replace").strip()
        if name:
            return name
        return "GMS (satellite unknown -- mode block unavailable/unparsed)"


def _lookup_calibration_value(block, lut, mask):
    """Look up calibrated values for a raw-counts block via the file's own calibration LUT."""
    return lut[block.astype(np.int64) & mask]


class Calibrator:
    """Calibrate GMS-1..4 VISSR counts to reflectance (%) or brightness temperature (K)."""

    def __init__(self, calib_table, channel):
        """Store the file's own calibration LUT and the channel it applies to."""
        self._calib_table = calib_table
        self._channel = channel
        self._mask = 0xFF if channel == fmt.IR_CHANNEL else 0x3F

    def calibrate(self, counts, calibration):
        """Transform counts (a dask array of raw pixel values) to the given calibration level.

        *calibration* is one of "counts" (pass through unchanged),
        "reflectance" (VIS, % 0-100), or "brightness_temperature" (IR, K).
        """
        if calibration == "counts":
            return counts
        res = self._calibrate(counts)
        res = self._postproc(res, calibration)
        return res

    def _calibrate(self, counts):
        return counts.map_blocks(
            _lookup_calibration_value,
            lut=self._calib_table,
            mask=self._mask,
            dtype=np.float32,
            meta=np.array((), dtype=np.float32),
        )

    def _postproc(self, res, calibration):
        if calibration == "reflectance":
            # convert to percent
            return res * 100
        return res


def _read_struct(raw, offset, dtype):
    return np.frombuffer(raw[offset:offset + dtype.itemsize], dtype=dtype, count=1)[0]


class GmsVissrL1bFile:
    """Load a single IR or VIS GMS-1..4 archive file."""

    def __init__(self, path, line_chunks=64):
        """Detect the file's channel (VIS/IR) and parse its header blocks."""
        name = os.path.basename(os.fspath(path)).upper()
        if name.startswith("VS"):
            self.channel = fmt.VIS_CHANNEL
        elif name.startswith("IR"):
            self.channel = fmt.IR_CHANNEL
        else:
            try:
                size = os.path.getsize(path)
            except (TypeError, OSError):
                with generic_open(path, "rb") as f:
                    size = len(f.read())
            self.channel = (fmt.VIS_CHANNEL
                             if size % fmt.VIS_BLOCK_LEN == 0
                             else fmt.IR_CHANNEL)

        self.path = path
        with generic_open(path, "rb") as f:
            self._raw = f.read()

        spec = fmt.IMAGE_DATA[self.channel]
        params = spec["params"]

        self.mode = _read_struct(self._raw, params["mode"]["offset"], params["mode"]["dtype"])
        self.coord = _read_struct(self._raw, params["coordinate_conversion"]["offset"],
                                   params["coordinate_conversion"]["dtype"])
        self.attitude = _read_struct(self._raw, params["attitude_prediction"]["offset"],
                                      params["attitude_prediction"]["dtype"])
        self.orbit1 = _read_struct(self._raw, params["orbit_prediction_1"]["offset"],
                                    params["orbit_prediction_1"]["dtype"])
        self.orbit2 = _read_struct(self._raw, params["orbit_prediction_2"]["offset"],
                                    params["orbit_prediction_2"]["dtype"])

        cal_key = "ir_calibration" if self.channel == fmt.IR_CHANNEL else "vis_calibration"
        self.calibration = _read_struct(self._raw, params[cal_key]["offset"],
                                         params[cal_key]["dtype"])

        self._line_chunks = line_chunks
        self._parse_image_data(spec)

    def _parse_image_data(self, spec):
        data_dtype = spec["dtype"]
        offset = spec["offset"]
        pair_bytes = data_dtype.itemsize * 2  # 2 lines per raw block
        n_pairs = (len(self._raw) - offset) // pair_bytes

        arr = np.frombuffer(
            self._raw[offset:offset + n_pairs * pair_bytes],
            dtype=data_dtype, count=n_pairs * 2,
        )

        self.line_numbers = arr["LCW"]["line_number"].astype(np.int64)
        self.scan_times = arr["LCW"]["scan_time"].astype(np.float64)
        self.west_earth_edges = arr["LCW"]["west_side_earth_edge"].astype(np.int32)
        self.east_earth_edges = arr["LCW"]["east_side_earth_edge"].astype(np.int32)
        self._pixels_np = arr["image_data"]  # (nlines, npix) uint8
        self.n_lines, self.n_pixels = self._pixels_np.shape

    def pixel_counts_dask(self):
        """Return raw 0-255 (IR) / 0-63 (VIS) pixel counts as a dask array."""
        return da.from_array(self._pixels_np, chunks=(self._line_chunks, self.n_pixels))

    def calibration_lut(self):
        """Return this file's own calibration lookup table for its channel."""
        if self.channel == fmt.IR_CHANNEL:
            return self.calibration["conversion_table_of_equivalent_black_body_temperature"]
        else:
            return self.calibration["vis1_calibration_table"]["brightness_albedo_conversion_table"]

    def get_earth_mask(self):
        """Return a boolean mask, True = earth disk, False = space, per scan line."""
        fill_value = -1
        west, east = self._earth_edges_for_mask(fill_value)

        valid = (west != fill_value) & (east != fill_value)
        w = np.maximum(west, 0)
        e = np.minimum(east, self.n_pixels - 1)
        valid &= (w <= e)

        pixel_idx = np.arange(self.n_pixels)
        return valid[:, None] & (pixel_idx[None, :] >= w[:, None]) & (pixel_idx[None, :] <= e[:, None])

    def _earth_edges_for_mask(self, fill_value):
        """Return west/east earth-edge arrays, oversampling-corrected for VIS."""
        west = self.west_earth_edges.copy()
        east = self.east_earth_edges.copy()
        if self.channel != fmt.VIS_CHANNEL:
            return west, east

        sampling_angle_ir = float(self.coord["sampling_angle_ir"])
        sampling_angle_vis = float(self.coord["sampling_angle_vis"])
        ratio = sampling_angle_ir / sampling_angle_vis if sampling_angle_vis > 0 else 2.0
        west = np.where(west != fill_value, (west * ratio).astype(np.int32), west)
        east = np.where(east != fill_value, (east * ratio).astype(np.int32), east)
        return west, east

    def calibrated_dask(self):
        """Return calibrated physical values as a lazy dask array."""
        calibrator = Calibrator(self.calibration_lut(), self.channel)
        counts = self.pixel_counts_dask()
        calibration_level = "brightness_temperature" if self.channel == fmt.IR_CHANNEL else "reflectance"
        return calibrator.calibrate(counts, calibration_level)

    def _build_navigation_parameters(self, channel=None, solar=False):
        channel = channel or self.channel
        suffix = f"{'ir' if channel == 'IR' else 'vis'}{'_solar' if solar else ''}"

        spinning_rate = float(self.mode["spin_rate"])
        if spinning_rate <= 0:
            # mode["spin_rate"] isn't populated in some older GMS-1/2/3
            # archives. The coordinate conversion parameters segment carries
            # its own per-file "daily mean spin rate" telemetry (word 130).
            spinning_rate = float(self.coord["daily_mean_spin_rate"])
        if spinning_rate <= 0:
            raise ValueError(
                "Neither mode['spin_rate'] nor coord['daily_mean_spin_rate'] "
                "is populated in this file -- can't navigate without a real "
                "spin rate."
            )

        scan_params = nav_shared.ScanningParameters(
            start_time_of_scan=float(self.coord["scheduled_observation_time"]),
            spinning_rate=spinning_rate,
            num_sensors=float(self.coord[f"num_sensors_{suffix}"]),
            sampling_angle=float(self.coord[f"sampling_angle_{suffix}"]),
        )

        misalignment = np.ascontiguousarray(
            np.asarray(self.coord["matrix_of_misalignment"], dtype=np.float64)
            .reshape(3, 3, order="F")
        )
        scanning_angles = nav_shared.ScanningAngles(
            stepping_angle=float(self.coord[f"stepping_angle_{suffix}"]),
            sampling_angle=float(self.coord[f"sampling_angle_{suffix}"]),
            misalignment=misalignment,
        )

        image_offset = nav_shared.ImageOffset(
            line_offset=float(self.coord[f"central_line_{suffix}"]),
            pixel_offset=float(self.coord[f"central_pixel_{suffix}"]),
        )

        earth_ellipsoid = nav_shared.EarthEllipsoid(
            flattening=nav_shared.EARTH_FLATTENING,
            equatorial_radius=nav_shared.EARTH_EQUATORIAL_RADIUS,
        )

        proj_params = nav_shared.ProjectionParameters(
            image_offset=image_offset,
            scanning_angles=scanning_angles,
            earth_ellipsoid=earth_ellipsoid,
        )

        static = nav_shared.StaticNavigationParameters(proj_params=proj_params, scan_params=scan_params)
        predicted = self._build_predicted_navigation_params()
        return nav_shared.ImageNavigationParameters(static=static, predicted=predicted)

    def _build_predicted_navigation_params(self):
        at = self.attitude["data"]
        attitudes = nav_shared.Attitude(
            angle_between_earth_and_sun=at["beta_angle"].astype(np.float64),
            angle_between_sat_spin_and_z_axis=at["angle_between_z_axis_and_spin_axis"].astype(np.float64),
            angle_between_sat_spin_and_yz_plane=at["angle_between_spin_axis_and_yz_plane"].astype(np.float64),
        )
        attitude_prediction = nav_shared.AttitudePrediction(
            prediction_times=at["prediction_time_mjd"].astype(np.float64),
            attitude=attitudes,
        )

        o1, o2 = self.orbit1["data"], self.orbit2["data"]
        combined = np.concatenate([o1[:8], o2])

        orbit_angles = nav_shared.OrbitAngles(
            greenwich_sidereal_time=np.deg2rad(combined["greenwich_sidereal_time"].astype(np.float64)),
            declination_from_sat_to_sun=np.deg2rad(combined["declination_sat_to_sun"].astype(np.float64)),
            right_ascension_from_sat_to_sun=np.deg2rad(combined["right_ascension_sat_to_sun"].astype(np.float64)),
        )
        sat_pos_arr = combined["satellite_position_earth_fixed"]
        sat_position = nav_shared.Satpos(
            x=sat_pos_arr[:, 0].astype(np.float64),
            y=sat_pos_arr[:, 1].astype(np.float64),
            z=sat_pos_arr[:, 2].astype(np.float64),
        )
        npa = combined["npa_matrix"].reshape(-1, 3, 3).transpose(0, 2, 1)
        orbit_prediction = nav_shared.OrbitPrediction(
            prediction_times=combined["prediction_time_mjd"].astype(np.float64),
            angles=orbit_angles,
            sat_position=sat_position,
            nutation_precession=np.ascontiguousarray(npa),
        )
        return nav_shared.PredictedNavigationParameters(attitude=attitude_prediction, orbit=orbit_prediction)

    def navigate_dask(self):
        """Return lat/lon as dask arrays, via the navigation module."""
        nav_params = self._build_navigation_parameters()
        lines = self.line_numbers.astype(np.float64) - 1.0  # see _build_navigation_parameters note on +1 convention
        pixels = np.arange(self.n_pixels, dtype=np.float64)
        lons, lats = nav_shared.get_lons_lats(lines, pixels, nav_params)

        chunks = (self._line_chunks, self.n_pixels)
        lats = lats.rechunk(chunks) if hasattr(lats, "rechunk") else da.from_array(lats, chunks=chunks)
        lons = lons.rechunk(chunks) if hasattr(lons, "rechunk") else da.from_array(lons, chunks=chunks)
        return lats, lons

    def get_dataset(self, mask_space=True):
        """Return an xarray.DataArray of calibrated values, space-masked."""
        data = self.calibrated_dask()
        if mask_space:
            earth_mask = da.from_array(self.get_earth_mask(), chunks=(self._line_chunks, self.n_pixels))
            data = da.where(earth_mask, data, np.nan)
        lat, lon = self.navigate_dask()
        da_out = xr.DataArray(
            data, dims=("y", "x"),
            coords={
                "longitude": (("y", "x"), lon),
                "latitude": (("y", "x"), lat),
            },
            attrs={
                "platform": "GMS (1-4 family, satellite ID TBD from mode block)",
                "sensor": "VISSR",
                "channel": self.channel,
                "units": "K" if self.channel == fmt.IR_CHANNEL else "albedo",
                "start_time": self.coord["scheduled_observation_time"],
            },
        )
        return da_out


class AreaDefEstimator:
    """Estimate a uniform-sampling AreaDefinition for GMS-1..4 VISSR images."""

    def __init__(self, l1b_file, platform_name):
        """Store the parsed L1b file and platform name used for area naming."""
        self.l1b = l1b_file
        self.platform_name = platform_name

    def get_area_def_uniform_sampling(self, dataset_id):
        """Build and return the uniform-sampling AreaDefinition for *dataset_id*."""
        proj_dict = self._get_proj_dict(dataset_id)
        extent = geos_area.get_area_extent(proj_dict)
        return geos_area.get_area_definition(proj_dict, extent)

    def _get_proj_dict(self, dataset_id):
        proj_dict = {}
        proj_dict.update(self._get_name_dict(dataset_id))
        proj_dict.update(self._get_proj4_dict())
        proj_dict.update(self._get_shape_dict(dataset_id))
        return proj_dict

    def _get_name_dict(self, dataset_id):
        if hasattr(dataset_id, "get"):
            resolution = dataset_id.get("resolution")
        else:
            resolution = getattr(dataset_id, "resolution", None)
        name_dict = geos_area.get_geos_area_naming({
            "platform_name": self.platform_name,
            "instrument_name": "VISSR",
            "service_name": "western-pacific",
            "service_desc": "Western Pacific",
            "resolution": resolution,
        })
        return {
            "a_name": name_dict["area_id"],
            "p_id": name_dict["area_id"],
            "a_desc": name_dict["description"],
        }

    def _get_proj4_dict(self):
        return {
            "ssp_lon": float(self.l1b.mode["ssp_longitude"]),
            "a": nav_shared.EARTH_EQUATORIAL_RADIUS,
            "b": nav_shared.EARTH_POLAR_RADIUS,
            "h": float(self.l1b.mode["satellite_height"]),
        }

    def _get_shape_dict(self, dataset_id):
        size = self.l1b.n_lines
        if size <= 1:
            raise ValueError(
                f"Implausible dataset shape (n_lines={size}) -- can't "
                f"build a uniform-sampling area definition from this."
            )

        suffix = "ir" if self.l1b.channel == "IR" else "vis"
        stepping_angle = float(self.l1b.coord[f"stepping_angle_{suffix}"])

        line_pixel_offset = 0.5 * size
        lfac_cfac = geos_area.sampling_to_lfac_cfac(stepping_angle)
        return {
            "nlines": size,
            "ncols": size,
            "lfac": lfac_cfac,
            "cfac": lfac_cfac,
            "coff": line_pixel_offset,
            "loff": line_pixel_offset,
            "scandir": "N2S",
        }
