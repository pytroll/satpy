"""Reader for GMS-5 VISSR Level 1B data.

Introduction
------------
The ``gms5_vissr_l1b`` reader can decode, navigate and calibrate Level 1B data
from the Visible and Infrared Spin Scan Radiometer (VISSR) in `VISSR
archive format`. Corresponding platforms are GMS-5 (Japanese Geostationary
Meteorological Satellite) and GOES-09 (2003-2006 backup after MTSAT-1 launch
failure).

VISSR has four channels, each stored in a separate file:

.. code-block:: none

    VISSR_20020101_0031_IR1.A.IMG
    VISSR_20020101_0031_IR2.A.IMG
    VISSR_20020101_0031_IR3.A.IMG
    VISSR_20020101_0031_VIS.A.IMG

This is how to read them with Satpy:

.. code-block:: python

    from satpy import Scene
    import glob

    filenames = glob.glob(""/data/VISSR*")
    scene = Scene(filenames, reader="gms5-vissr_l1b")
    scene.load(["VIS", "IR1"])


References
~~~~~~~~~~

Details about platform, instrument and data format can be found in the
following references:

    - `VISSR Format Description`_
    - `GMS User Guide`_

.. _VISSR Format Description:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/VISSR_FORMAT_GMS-5.pdf
.. _GMS User Guide:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/GMS_Users_Guide_3rd_Edition_Rev1.pdf


Compression
-----------

Gzip-compressed VISSR files can be decompressed on the fly using
:class:`~satpy.readers.FSFile`:

.. code-block:: python

    import fsspec
    from satpy import Scene
    from satpy.readers import FSFile

    filename = "VISSR_19960217_2331_IR1.A.IMG.gz"
    open_file = fsspec.open(filename, compression="gzip")
    fs_file = FSFile(open_file)
    scene = Scene([fs_file], reader="gms5-vissr_l1b")
    scene.load(["IR1"])


Calibration
-----------

Sensor counts are calibrated by looking up reflectance/temperature values in the
calibration tables included in each file. See section 2.2 in the VISSR user
guide.


Navigation
----------

VISSR images are oversampled and not rectified.


Oversampling
~~~~~~~~~~~~
VISSR oversamples the viewed scene in E-W direction by a factor of ~1.46:
IR/VIS pixels are 14/3.5 urad on a side, but the instrument samples every
9.57/2.39 urad in E-W direction. That means pixels are actually overlapping on
the ground.

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
   is identical (IR channels)
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

    filenames = glob.glob("VISSR*.IMG")
    scene = satpy.Scene(filenames,
                        reader="gms5-vissr_l1b",
                        reader_kwargs={"mask_space": False})
    scene.load(["VIS", "IR1])


Metadata
--------

Dataset attributes include metadata such as time and orbital parameters,
see :ref:`dataset_metadata`.

Partial Scans
-------------

Between 2001 and 2003 VISSR also recorded partial scans of the northern
hemisphere. On demand a special Typhoon schedule would be activated between
03:00 and 05:00 UTC.
"""

import datetime as dt

import dask.array as da
import numba
import numpy as np
import xarray as xr

import satpy.readers._geos_area as geos_area
import satpy.readers.gms.gms5_vissr_format as fmt
import satpy.readers.gms.gms5_vissr_navigation as nav
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hrit_jma import mjd2datetime64
from satpy.readers.utils import generic_open
from satpy.utils import get_legacy_chunk_size

CHUNK_SIZE = get_legacy_chunk_size()


def _recarr2dict(arr, preserve=None):
    if not preserve:
        preserve = []
    res = {}
    for key, value in zip(arr.dtype.names, arr):
        if key.startswith("reserved"):
            continue
        if value.dtype.names and key not in preserve:
            # Nested record array
            res[key] = _recarr2dict(value)
        else:
            # Scalar or record array that shall be preserved
            res[key] = value
    return res


class GMS5VISSRFileHandler(BaseFileHandler):
    """File handler for GMS-5 VISSR data in VISSR archive format."""

    def __init__(self, filename, filename_info, filetype_info, mask_space=True):
        """Initialize the file handler.

        Args:
            filename: Name of file to be read
            filename_info: Information obtained from filename
            filetype_info: Information about file type
            mask_space: Mask space pixels.
        """
        super(GMS5VISSRFileHandler, self).__init__(
            filename, filename_info, filetype_info
        )
        self._filename = filename
        self._filename_info = filename_info
        self._header, self._channel_type = self._read_header(filename)
        self._mda = self._get_mda()
        self._mask_space = mask_space

    def _read_header(self, filename):
        header = {}
        with generic_open(filename, mode="rb") as file_obj:
            header["control_block"] = self._read_control_block(file_obj)
            channel_type = self._get_channel_type(
                header["control_block"]["parameter_block_size"]
            )
            header["image_parameters"] = self._read_image_params(file_obj, channel_type)
        return header, channel_type

    @staticmethod
    def _get_channel_type(parameter_block_size):
        if parameter_block_size == 4:
            return fmt.VIS_CHANNEL
        elif parameter_block_size == 16:
            return fmt.IR_CHANNEL
        raise ValueError(
            f"Cannot determine channel type, possibly corrupt file "
            f"(unknown parameter block size: {parameter_block_size})"
        )

    def _read_control_block(self, file_obj):
        ctrl_block = read_from_file_obj(file_obj, dtype=fmt.CONTROL_BLOCK, count=1)
        return _recarr2dict(ctrl_block[0])

    def _read_image_params(self, file_obj, channel_type):
        """Read image parameters from the header."""
        image_params = {}
        for name, param in fmt.IMAGE_PARAMS.items():
            image_params[name] = self._read_image_param(file_obj, param, channel_type)

        image_params["orbit_prediction"] = self._concat_orbit_prediction(
            image_params.pop("orbit_prediction_1"),
            image_params.pop("orbit_prediction_2"),
        )
        return image_params

    @staticmethod
    def _read_image_param(file_obj, param, channel_type):
        """Read a single image parameter block from the header."""
        image_params = read_from_file_obj(
            file_obj,
            dtype=param["dtype"],
            count=1,
            offset=param["offset"][channel_type],
        )
        return _recarr2dict(image_params[0], preserve=param.get("preserve"))

    @staticmethod
    def _concat_orbit_prediction(orb_pred_1, orb_pred_2):
        """Concatenate orbit prediction data.

        It is split over two image parameter blocks in the header.
        """
        orb_pred = orb_pred_1
        orb_pred["data"] = np.concatenate([orb_pred_1["data"], orb_pred_2["data"]])
        return orb_pred

    def _get_frame_parameters_key(self):
        if self._channel_type == fmt.VIS_CHANNEL:
            return "vis_frame_parameters"
        return "ir_frame_parameters"

    def _get_actual_shape(self):
        actual_num_lines = self._header["control_block"][
            "available_block_size_of_image_data"
        ]
        _, nominal_num_pixels = self._get_nominal_shape()
        return actual_num_lines, nominal_num_pixels

    def _get_nominal_shape(self):
        frame_params = self._header["image_parameters"]["mode"][
            self._get_frame_parameters_key()
        ]
        return frame_params["number_of_lines"], frame_params["number_of_pixels"]

    def _get_mda(self):
        return {
            "platform": self._mode_block["satellite_name"].decode().strip().upper(),
            "sensor": "VISSR",
            "time_parameters": self._get_time_parameters(),
            "orbital_parameters": self._get_orbital_parameters(),
        }

    def _get_orbital_parameters(self):
        # Note: SSP longitude in simple coordinate conversion table seems to be
        # incorrect (80 deg instead of 140 deg). Use orbital parameters instead.
        im_params = self._header["image_parameters"]
        mode = im_params["mode"]
        simple_coord = im_params["simple_coordinate_conversion_table"]
        orb_params = im_params["coordinate_conversion"]["orbital_parameters"]
        return {
            "satellite_nominal_longitude": mode["ssp_longitude"],
            "satellite_nominal_latitude": 0.0,
            "satellite_nominal_altitude": mode["satellite_height"],
            "satellite_actual_longitude": orb_params["longitude_of_ssp"],
            "satellite_actual_latitude": orb_params["latitude_of_ssp"],
            "satellite_actual_altitude": simple_coord["satellite_height"],
        }

    def _get_time_parameters(self):
        start_time = mjd2datetime64(self._mode_block["observation_time_mjd"])
        start_time = start_time.astype(dt.datetime).replace(second=0, microsecond=0)
        end_time = start_time + dt.timedelta(
            minutes=25
        )  # Source: GMS User Guide, section 3.3.1
        return {
            "nominal_start_time": start_time,
            "nominal_end_time": end_time,
        }

    def get_dataset(self, dataset_id, ds_info):
        """Get dataset from file."""
        image_data = self._get_image_data()
        counts = self._get_counts(image_data)
        dataset = self._calibrate(counts, dataset_id)
        space_masker = SpaceMasker(image_data, dataset_id["name"])
        dataset = self._mask_space_pixels(dataset, space_masker)
        self._attach_lons_lats(dataset, dataset_id)
        self._update_attrs(dataset, dataset_id, ds_info)
        return dataset

    def _get_image_data(self):
        image_data = self._read_image_data()
        return da.from_array(image_data, chunks=(CHUNK_SIZE,))

    def _read_image_data(self):
        num_lines, _ = self._get_actual_shape()
        specs = self._get_image_data_type_specs()
        with generic_open(self._filename, "rb") as file_obj:
            return read_from_file_obj(
                file_obj, dtype=specs["dtype"], count=num_lines, offset=specs["offset"]
            )

    def _get_image_data_type_specs(self):
        return fmt.IMAGE_DATA[self._channel_type]

    def _get_counts(self, image_data):
        return self._make_counts_data_array(image_data)

    def _make_counts_data_array(self, image_data):
        return xr.DataArray(
            image_data["image_data"],
            dims=("y", "x"),
            coords={
                "acq_time": ("y", self._get_acq_time(image_data)),
                "line_number": ("y", self._get_line_number(image_data)),
            },
        )

    def _get_acq_time(self, dask_array):
        acq_time = dask_array["LCW"]["scan_time"].compute()
        return mjd2datetime64(acq_time)

    def _get_line_number(self, dask_array):
        return dask_array["LCW"]["line_number"].compute()

    def _calibrate(self, counts, dataset_id):
        table = self._get_calibration_table(dataset_id)
        cal = Calibrator(table)
        return cal.calibrate(counts, dataset_id["calibration"])

    def _get_calibration_table(self, dataset_id):
        tables = {
            "VIS": self._header["image_parameters"]["vis_calibration"][
                "vis1_calibration_table"
            ]["brightness_albedo_conversion_table"],
            "IR1": self._header["image_parameters"]["ir1_calibration"][
                "conversion_table_of_equivalent_black_body_temperature"
            ],
            "IR2": self._header["image_parameters"]["ir2_calibration"][
                "conversion_table_of_equivalent_black_body_temperature"
            ],
            "IR3": self._header["image_parameters"]["wv_calibration"][
                "conversion_table_of_equivalent_black_body_temperature"
            ],
        }
        return tables[dataset_id["name"]]

    def _get_area_def_uniform_sampling(self, dataset_id):
        a = AreaDefEstimator(
            coord_conv_params=self._header["image_parameters"]["coordinate_conversion"],
            metadata=self._mda,
        )
        return a.get_area_def_uniform_sampling(dataset_id)

    def _mask_space_pixels(self, dataset, space_masker):
        if self._mask_space:
            return space_masker.mask_space(dataset)
        return dataset

    def _attach_lons_lats(self, dataset, dataset_id):
        lons, lats = self._get_lons_lats(dataset, dataset_id)
        dataset.coords["lon"] = lons
        dataset.coords["lat"] = lats

    def _get_lons_lats(self, dataset, dataset_id):
        lines, pixels = self._get_image_coords(dataset)
        nav_params = self._get_navigation_parameters(dataset_id)
        lons, lats = nav.get_lons_lats(lines, pixels, nav_params)
        return self._make_lons_lats_data_array(lons, lats)

    def _get_image_coords(self, data):
        lines = data.coords["line_number"].values
        pixels = np.arange(data.shape[1])
        return lines.astype(np.float64), pixels.astype(np.float64)

    def _get_navigation_parameters(self, dataset_id):
        return nav.ImageNavigationParameters(
            static=self._get_static_navigation_params(dataset_id),
            predicted=self._get_predicted_navigation_params()
        )

    def _get_static_navigation_params(self, dataset_id):
        """Get static navigation parameters.

        Note that, "central_line_number_of_vissr_frame" is different for each
        channel, even if their spatial resolution is identical. For example:

        VIS: 5513.0
        IR1: 1378.5
        IR2: 1378.7
        IR3: 1379.1001
        """
        alt_ch_name = _get_alternative_channel_name(dataset_id)
        scan_params = nav.ScanningParameters(
            start_time_of_scan=self._coord_conv["scheduled_observation_time"],
            spinning_rate=self._mode_block["spin_rate"],
            num_sensors=self._coord_conv["number_of_sensor_elements"][alt_ch_name],
            sampling_angle=self._coord_conv["sampling_angle_along_pixel"][alt_ch_name],
        )
        proj_params = self._get_proj_params(dataset_id)
        return nav.StaticNavigationParameters(
            proj_params=proj_params,
            scan_params=scan_params
        )

    def _get_proj_params(self, dataset_id):
        proj_params = nav.ProjectionParameters(
            image_offset=self._get_image_offset(dataset_id),
            scanning_angles=self._get_scanning_angles(dataset_id),
            earth_ellipsoid=self._get_earth_ellipsoid()
        )
        return proj_params

    def _get_earth_ellipsoid(self):
        # Use earth radius and flattening from JMA's Msial library, because
        # the values in the data seem to be pretty old. For example the
        # equatorial radius is from the Bessel Ellipsoid (1841).
        return nav.EarthEllipsoid(
            flattening=nav.EARTH_FLATTENING,
            equatorial_radius=nav.EARTH_EQUATORIAL_RADIUS,
        )

    def _get_scanning_angles(self, dataset_id):
        alt_ch_name = _get_alternative_channel_name(dataset_id)
        misalignment = np.ascontiguousarray(
            self._coord_conv["matrix_of_misalignment"].transpose().astype(np.float64)
        )
        return nav.ScanningAngles(
            stepping_angle=self._coord_conv["stepping_angle_along_line"][alt_ch_name],
            sampling_angle=self._coord_conv["sampling_angle_along_pixel"][
                alt_ch_name],
            misalignment=misalignment
        )

    def _get_image_offset(self, dataset_id):
        alt_ch_name = _get_alternative_channel_name(dataset_id)
        center_line_vissr_frame = self._coord_conv["central_line_number_of_vissr_frame"][
            alt_ch_name
        ]
        center_pixel_vissr_frame = self._coord_conv["central_pixel_number_of_vissr_frame"][
            alt_ch_name
        ]
        pixel_offset = self._coord_conv[
            "pixel_difference_of_vissr_center_from_normal_position"
        ][alt_ch_name]
        return nav.ImageOffset(
            line_offset=center_line_vissr_frame,
            pixel_offset=center_pixel_vissr_frame + pixel_offset,
        )

    def _get_predicted_navigation_params(self):
        """Get predictions of time-dependent navigation parameters."""
        attitude_prediction = self._get_attitude_prediction()
        orbit_prediction = self._get_orbit_prediction()
        return nav.PredictedNavigationParameters(
            attitude=attitude_prediction,
            orbit=orbit_prediction
        )

    def _get_attitude_prediction(self):
        att_pred = self._header["image_parameters"]["attitude_prediction"]["data"]
        attitudes = nav.Attitude(
            angle_between_earth_and_sun=att_pred["sun_earth_angle"].astype(
                np.float64),
            angle_between_sat_spin_and_z_axis=att_pred[
                "right_ascension_of_attitude"
            ].astype(np.float64),
            angle_between_sat_spin_and_yz_plane=att_pred[
                "declination_of_attitude"
            ].astype(np.float64),
        )
        attitude_prediction = nav.AttitudePrediction(
            prediction_times=att_pred["prediction_time_mjd"].astype(np.float64),
            attitude=attitudes
        )
        return attitude_prediction

    def _get_orbit_prediction(self):
        orb_pred = self._header["image_parameters"]["orbit_prediction"]["data"]
        orbit_angles = nav.OrbitAngles(
            greenwich_sidereal_time=np.deg2rad(
                orb_pred["greenwich_sidereal_time"].astype(np.float64)
            ),
            declination_from_sat_to_sun=np.deg2rad(
                orb_pred["sat_sun_vector_earth_fixed"]["elevation"].astype(np.float64)
            ),
            right_ascension_from_sat_to_sun=np.deg2rad(
                orb_pred["sat_sun_vector_earth_fixed"]["azimuth"].astype(np.float64)
            ),
        )
        sat_position = nav.Satpos(
            x=orb_pred["satellite_position_earth_fixed"][:, 0].astype(np.float64),
            y=orb_pred["satellite_position_earth_fixed"][:, 1].astype(np.float64),
            z=orb_pred["satellite_position_earth_fixed"][:, 2].astype(np.float64),
        )
        orbit_prediction = nav.OrbitPrediction(
            prediction_times=orb_pred["prediction_time_mjd"].astype(np.float64),
            angles=orbit_angles,
            sat_position=sat_position,
            nutation_precession=np.ascontiguousarray(
                orb_pred["conversion_matrix"].transpose(0, 2, 1).astype(np.float64)
            ),
        )
        return orbit_prediction

    def _make_lons_lats_data_array(self, lons, lats):
        lons = xr.DataArray(
            lons,
            dims=("y", "x"),
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        )
        lats = xr.DataArray(
            lats,
            dims=("y", "x"),
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        )
        return lons, lats

    def _update_attrs(self, dataset, dataset_id, ds_info):
        dataset.attrs.update(ds_info)
        dataset.attrs.update(self._mda)
        dataset.attrs[
            "area_def_uniform_sampling"
        ] = self._get_area_def_uniform_sampling(dataset_id)

    @property
    def start_time(self):
        """Nominal start time of the dataset."""
        return self._mda["time_parameters"]["nominal_start_time"]

    @property
    def end_time(self):
        """Nominal end time of the dataset."""
        return self._mda["time_parameters"]["nominal_end_time"]

    @property
    def _coord_conv(self):
        return self._header["image_parameters"]["coordinate_conversion"]

    @property
    def _mode_block(self):
        return self._header["image_parameters"]["mode"]


def _get_alternative_channel_name(dataset_id):
    return fmt.ALT_CHANNEL_NAMES[dataset_id["name"]]


def read_from_file_obj(file_obj, dtype, count, offset=0):
    """Read data from file object.

    Args:
        file_obj: An open file object.
        dtype: Data type to be read.
        count: Number of elements to be read.
        offset: Byte offset where to start reading.
    """
    file_obj.seek(offset)
    data = file_obj.read(dtype.itemsize * count)
    return np.frombuffer(data, dtype=dtype, count=count)


class Calibrator:
    """Calibrate VISSR data to reflectance or brightness temperature.

    Reference: Section 2.2 in the VISSR User Guide.
    """

    def __init__(self, calib_table):
        """Initialize the calibrator.

        Args:
            calib_table: Calibration table
        """
        self._calib_table = calib_table

    def calibrate(self, counts, calibration):
        """Transform counts to given calibration level."""
        if calibration == "counts":
            return counts
        res = self._calibrate(counts)
        res = self._postproc(res, calibration)
        return self._make_data_array(res, counts)

    def _calibrate(self, counts):
        return da.map_blocks(
            self._lookup_calib_table,
            counts.data,
            calib_table=self._calib_table,
            dtype=np.float32,
        )

    def _postproc(self, res, calibration):
        if calibration == "reflectance":
            res = self._convert_to_percent(res)
        return res

    def _convert_to_percent(self, res):
        return res * 100

    def _make_data_array(self, interp, counts):
        return xr.DataArray(
            interp,
            dims=counts.dims,
            coords=counts.coords,
        )

    def _lookup_calib_table(self, counts, calib_table):
        return calib_table[counts]


class SpaceMasker:
    """Mask pixels outside the earth disk."""

    _fill_value = -1  # scanline not intersecting the earth

    def __init__(self, image_data, channel):
        """Initialize the space masker.

        Args:
            image_data: Image data
            channel: Channel name
        """
        self._image_data = image_data
        self._channel = channel
        self._shape = image_data["image_data"].shape
        self._earth_mask = self._get_earth_mask()

    def mask_space(self, dataset):
        """Mask space pixels in the given dataset."""
        return dataset.where(self._earth_mask).astype(np.float32)

    def _get_earth_mask(self):
        earth_edges = self._get_earth_edges()
        return get_earth_mask(self._shape, earth_edges, self._fill_value)

    def _get_earth_edges(self):
        west_edges = self._get_earth_edges_per_scan_line("west_side_earth_edge")
        east_edges = self._get_earth_edges_per_scan_line("east_side_earth_edge")
        return west_edges, east_edges

    def _get_earth_edges_per_scan_line(self, cardinal):
        edges = self._image_data["LCW"][cardinal].compute().astype(np.int32)
        if is_vis_channel(self._channel):
            edges = self._correct_vis_edges(edges)
        return edges

    def _correct_vis_edges(self, edges):
        """Correct VIS edges.

        VIS data contains earth edges of IR channel. Compensate for that
        by scaling with a factor of 4 (1 IR pixel ~ 4 VIS pixels).
        """
        return np.where(edges != self._fill_value, edges * 4, edges)


@numba.njit
def get_earth_mask(shape, earth_edges, fill_value=-1):
    """Get binary mask where 1/0 indicates earth/space.

    Args:
        shape: Image shape
        earth_edges: First and last earth pixel in each scanline
        fill_value: Fill value for scanlines not intersecting the earth.
    """
    first_earth_pixels, last_earth_pixels = earth_edges
    mask = np.zeros(shape, dtype=np.int8)
    for line in range(shape[0]):
        first = first_earth_pixels[line]
        last = last_earth_pixels[line]
        if first == fill_value or last == fill_value:
            continue
        mask[line, first:last+1] = 1
    return mask


def is_vis_channel(channel_name):
    """Check if it's the visible channel."""
    return channel_name == "VIS"


class AreaDefEstimator:
    """Estimate area definition for VISSR images."""

    full_disk_size = {
        "IR": 2366,
        "VIS": 9464,
    }

    def __init__(self, coord_conv_params, metadata):
        """Initialize the area definition estimator.

        Args:
            coord_conv_params: Coordinate conversion parameters
            metadata: VISSR file metadata
        """
        self.coord_conv = coord_conv_params
        self.metadata = metadata

    def get_area_def_uniform_sampling(self, dataset_id):
        """Get full disk area definition with uniform sampling.

        Args:
            dataset_id: ID of the corresponding dataset.
        """
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
        name_dict = geos_area.get_geos_area_naming(
            {
                "platform_name": self.metadata["platform"],
                "instrument_name": self.metadata["sensor"],
                "service_name": "western-pacific",
                "service_desc": "Western Pacific",
                "resolution": dataset_id["resolution"],
            }
        )
        return {
            "a_name": name_dict["area_id"],
            "p_id": name_dict["area_id"],
            "a_desc": name_dict["description"],
        }

    def _get_proj4_dict(
        self,
    ):
        # Use nominal parameters to make the area def as constant as possible
        return {
            "ssp_lon": self.metadata["orbital_parameters"][
                "satellite_nominal_longitude"
            ],
            "a": nav.EARTH_EQUATORIAL_RADIUS,
            "b": nav.EARTH_POLAR_RADIUS,
            "h": self.metadata["orbital_parameters"]["satellite_nominal_altitude"],
        }

    def _get_shape_dict(self, dataset_id):
        # Apply sampling from the vertical dimension to the horizontal
        # dimension to obtain a square area definition with uniform sampling.
        ch_type = fmt.CHANNEL_TYPES[dataset_id["name"]]
        alt_ch_name = _get_alternative_channel_name(dataset_id)
        stepping_angle = self.coord_conv["stepping_angle_along_line"][alt_ch_name]
        size = self.full_disk_size[ch_type]
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
