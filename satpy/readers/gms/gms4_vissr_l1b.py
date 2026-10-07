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

Sensor counts are calibrated by looking up unnormalized reflectance/temperature
values in the calibration tables included in each file. See section 2.2 in the
VISSR user guide.


Navigation
----------

VISSR images are oversampled and not rectified.

Some older GMS-1/2/3 archive files don't populate the mode block's
``spin_rate`` telemetry field (it reads back as 0, which is physically
impossible for a spin-scan imager). When that happens, this reader falls
back to the coordinate conversion parameters segment's own
``daily_mean_spin_rate`` field.


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

import os

import dask.array as da
import numpy as np
import xarray as xr

import satpy.readers.gms.gms4_vissr_format as fmt
import satpy.readers.gms.gms_vissr_common as common
import satpy.readers.gms.gms_vissr_navigation as nav
from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.utils import generic_open
from satpy.readers.hrit_jma import mjd2datetime64
from satpy.utils import datetime64_to_pydatetime, get_legacy_chunk_size

CHUNK_SIZE = get_legacy_chunk_size()
INSTRUMENT = "VISSR"


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
        """Set of sensor names this file handler provides data for (as in the YAML definition)."""
        return {"gms4-vissr"}

    def combine_info(self, all_infos):
        """Combine per-file info dicts; GMS-1..4 archives are always a single VIS+IR file pair, never segmented."""
        if len(all_infos) == 1:
            return all_infos[0]
        return super().combine_info(all_infos)

    def get_dataset(self, dataset_id, ds_info):
        """Return the DataArray for *dataset_id*, or None if this file doesn't hold that channel."""
        requested_channel = ds_info.get("name", getattr(dataset_id, "name", None))
        if requested_channel != self._l1b.channel:
            return None

        data_array = self._l1b.get_dataset(dataset_id["calibration"], mask_space=self._mask_space)
        data_array.name = ds_info.get("name", self._l1b.channel)
        data_array.attrs.update(ds_info)
        data_array.attrs["platform_name"] = self._platform_name()

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
        size = self._l1b.n_lines
        if size <= 1:
            raise ValueError(
                f"Implausible dataset shape (n_lines={size}) -- can't "
                f"build a uniform-sampling area definition from this."
            )
        suffix = "ir" if self._l1b.channel == "IR" else "vis"
        estimator = common.AreaDefEstimator(
            platform_name=self._platform_name(),
            sensor_name=INSTRUMENT,
            ssp_lon=float(self._l1b.mode["ssp_longitude"]),
            satellite_height=float(self._l1b.mode["satellite_height"]),
        )
        return estimator.get_area_def_uniform_sampling(
            dataset_id,
            size=size,
            stepping_angle=float(self._l1b.coord[f"stepping_angle_{suffix}"]),
        )

    def _platform_name(self):
        name = bytes(self._l1b.mode["satellite_name"]).split(b"\x00")[0].decode("ascii", "replace").strip()
        if name:
            return name
        return "GMS (satellite unknown -- mode block unavailable/unparsed)"


def _read_struct(raw, offset, dtype):
    return np.frombuffer(raw[offset:offset + dtype.itemsize], dtype=dtype, count=1)[0]


class GmsVissrL1bFile:
    """Load a single IR or VIS GMS-1..4 archive file."""

    def __init__(self, path):
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
        # Chunks of whole lines, holding about as many pixels as a Satpy chunk.
        self._chunks = (max(1, CHUNK_SIZE * CHUNK_SIZE // self.n_pixels), self.n_pixels)

    def pixel_counts_dask(self):
        """Return raw 0-255 (IR) / 0-63 (VIS) pixel counts as a dask array."""
        return da.from_array(self._pixels_np, chunks=self._chunks)

    def calibration_lut(self):
        """Return this file's own calibration lookup table for its channel."""
        if self.channel == fmt.IR_CHANNEL:
            return self.calibration["conversion_table_of_equivalent_black_body_temperature"]
        else:
            return self.calibration["vis1_calibration_table"]["brightness_albedo_conversion_table"]

    def get_earth_mask(self):
        """Return a mask where 1 is the earth disk and 0 is space."""
        return common.get_earth_mask((self.n_lines, self.n_pixels), self._earth_edges_for_mask())

    def _earth_edges_for_mask(self):
        """Return west/east earth-edge arrays, oversampling-corrected for VIS."""
        west = self.west_earth_edges.copy()
        east = self.east_earth_edges.copy()
        if self.channel != fmt.VIS_CHANNEL:
            return west, east

        # VIS data contains earth edges of the IR channel.
        sampling_angle_ir = float(self.coord["sampling_angle_ir"])
        sampling_angle_vis = float(self.coord["sampling_angle_vis"])
        ratio = sampling_angle_ir / sampling_angle_vis if sampling_angle_vis > 0 else 2.0
        return common.scale_earth_edges(west, ratio), common.scale_earth_edges(east, ratio)

    def calibrate(self, calibration):
        """Return the counts or the calibrated values as a lazy dask array.

        Args:
            calibration: "counts", "brightness_temperature" (IR) or "unnormalized_reflectance" (VIS)
        """
        calibrator = common.Calibrator(self.calibration_lut())
        return calibrator.calibrate(self.pixel_counts_dask(), calibration)

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

        scan_params = nav.ScanningParameters(
            start_time_of_scan=float(self.coord["scheduled_observation_time"]),
            spinning_rate=spinning_rate,
            num_sensors=float(self.coord[f"num_sensors_{suffix}"]),
            sampling_angle=float(self.coord[f"sampling_angle_{suffix}"]),
        )

        misalignment = np.ascontiguousarray(
            np.asarray(self.coord["matrix_of_misalignment"], dtype=np.float64)
            .reshape(3, 3, order="F")
        )
        scanning_angles = nav.ScanningAngles(
            stepping_angle=float(self.coord[f"stepping_angle_{suffix}"]),
            sampling_angle=float(self.coord[f"sampling_angle_{suffix}"]),
            misalignment=misalignment,
        )

        image_offset = nav.ImageOffset(
            line_offset=float(self.coord[f"central_line_{suffix}"]),
            pixel_offset=float(self.coord[f"central_pixel_{suffix}"]),
        )

        earth_ellipsoid = nav.EarthEllipsoid(
            flattening=nav.EARTH_FLATTENING,
            equatorial_radius=nav.EARTH_EQUATORIAL_RADIUS,
        )

        proj_params = nav.ProjectionParameters(
            image_offset=image_offset,
            scanning_angles=scanning_angles,
            earth_ellipsoid=earth_ellipsoid,
        )

        static = nav.StaticNavigationParameters(proj_params=proj_params, scan_params=scan_params)
        predicted = self._build_predicted_navigation_params()
        return nav.ImageNavigationParameters(static=static, predicted=predicted)

    def _build_predicted_navigation_params(self):
        at = self.attitude["data"]
        attitudes = nav.Attitude(
            angle_between_earth_and_sun=at["beta_angle"].astype(np.float64),
            angle_between_sat_spin_and_z_axis=at["angle_between_z_axis_and_spin_axis"].astype(np.float64),
            angle_between_sat_spin_and_yz_plane=at["angle_between_spin_axis_and_yz_plane"].astype(np.float64),
        )
        attitude_prediction = nav.AttitudePrediction(
            prediction_times=at["prediction_time_mjd"].astype(np.float64),
            attitude=attitudes,
        )

        o1, o2 = self.orbit1["data"], self.orbit2["data"]
        combined = np.concatenate([o1[:8], o2])

        orbit_angles = nav.OrbitAngles(
            greenwich_sidereal_time=np.deg2rad(combined["greenwich_sidereal_time"].astype(np.float64)),
            declination_from_sat_to_sun=np.deg2rad(combined["declination_sat_to_sun"].astype(np.float64)),
            right_ascension_from_sat_to_sun=np.deg2rad(combined["right_ascension_sat_to_sun"].astype(np.float64)),
        )
        sat_pos_arr = combined["satellite_position_earth_fixed"]
        sat_position = nav.Satpos(
            x=sat_pos_arr[:, 0].astype(np.float64),
            y=sat_pos_arr[:, 1].astype(np.float64),
            z=sat_pos_arr[:, 2].astype(np.float64),
        )
        npa = combined["npa_matrix"].reshape(-1, 3, 3).transpose(0, 2, 1)
        orbit_prediction = nav.OrbitPrediction(
            prediction_times=combined["prediction_time_mjd"].astype(np.float64),
            angles=orbit_angles,
            sat_position=sat_position,
            nutation_precession=np.ascontiguousarray(npa),
        )
        return nav.PredictedNavigationParameters(attitude=attitude_prediction, orbit=orbit_prediction)

    def navigate_dask(self):
        """Return lon/lat as dask arrays, via the navigation module."""
        nav_params = self._build_navigation_parameters()
        line_chunks, pixel_chunks = self._chunks
        # Line numbers are 1-based, the navigation module expects 0-based lines.
        lines = da.from_array(self.line_numbers.astype(np.float64) - 1.0, chunks=line_chunks)
        pixels = da.from_array(np.arange(self.n_pixels, dtype=np.float64), chunks=pixel_chunks)
        return nav.get_lons_lats(lines, pixels, nav_params)

    def get_dataset(self, calibration, mask_space=True):
        """Return an xarray.DataArray of counts or calibrated values, with lon/lat coordinates."""
        data = self.calibrate(calibration)
        if mask_space:
            earth_mask = da.from_array(self.get_earth_mask(), chunks=self._chunks)
            data = da.where(earth_mask, data, np.float32(np.nan))
        lon, lat = self.navigate_dask()
        return xr.DataArray(
            data,
            dims=("y", "x"),
            coords={
                "longitude": (("y", "x"), lon),
                "latitude": (("y", "x"), lat),
            },
        )
