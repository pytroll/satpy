"""Reader for GMS-5 VISSR Level 1B data.

Introduction
------------
TODO


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
calibration tables included in each file.


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
resampling purpose an area definition with uniform sampling is provided via

.. code-block:: python

    scene[dataset].attrs["area_def_uniform_sampling"]


Rectification
~~~~~~~~~~~~~

VISSR images are not rectified. That means lon/lat coordinates are different

1) for all channels of the same repeat cycle, even if their spatial resolution
   is identical (IR channels)
2) for different repeat cycles, even if the channel is identical


Space Pixels
------------

VISSR produces data for pixels outside the Earth disk (i,e: atmospheric limb or
deep space pixels). By default, these pixels are masked out as they contain
data of limited or no value, but some applications do require these pixels.
To turn off masking, set ``mask_space=False`` upon scene creation::

.. code-block:: python

    import satpy
    import glob

    filenames = glob.glob("VISSR*.IMG")
    scene = satpy.Scene(filenames,
                        reader="gms5-vissr_l1b",
                        reader_kwargs={"mask_space": False})
    scene.load(["VIS"])



References
----------

    - [FMT]: `VISSR Format Description`_
    - [UG]: `GMS User Guide`_

.. _VISSR Format Description:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/VISSR_FORMAT_GMS-5.pdf
.. _GMS User Guide:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/GMS_Users_Guide_3rd_Edition_Rev1.pdf
"""

import dask.array as da
import numpy as np
import xarray as xr
import numba

import satpy.readers._geos_area as geos_area
import satpy.readers.gms5_vissr_navigation as nav
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hrit_jma import mjd2datetime64
from satpy.utils import get_legacy_chunk_size
from satpy.readers.utils import generic_open

CHUNK_SIZE = get_legacy_chunk_size()

U1 = '>u1'
I2 = '>i2'
I4 = '>i4'
R4 = '>f4'
R8 = '>f8'

VIS_CHANNEL = 'VIS'
IR_CHANNEL = 'IR'
ALT_CHANNEL_NAMES = {
    'VIS': 'VIS',
    'IR1': 'IR1',
    'IR2': 'IR2',
    'IR3': 'WV'
}
BLOCK_SIZE_VIS = 13504
BLOCK_SIZE_IR = 3664

IMAGE_PARAM_ITEM_SIZE = 2688
TIME = [('date', I4), ('time', I4)]
CHANNELS = [('VIS', R4), ('IR1', R4), ('IR2', R4), ('WV', R4)]
VISIR_SOLAR = [('VIS', R4), ('IR', R4)]

# fmt: off
CONTROL_BLOCK = np.dtype([('control_block_size', I2),
                          ('head_block_number_of_parameter_block', I2),
                          ('parameter_block_size', I2),
                          ('head_block_number_of_image_data', I2),
                          ('total_block_size_of_image_data', I2),
                          ('available_block_size_of_image_data', I2),
                          ('head_valid_line_number', I2),
                          ('final_valid_line_number', I2),
                          ('final_data_block_number', I2)])

MODE_BLOCK_FRAME_PARAMETERS = [('bit_length', I4),
                               ('number_of_lines', I4),
                               ('number_of_pixels', I4),
                               ('stepping_angle', R4),
                               ('sampling_angle', R4),
                               ('lcw_pixel_size', I4),
                               ('doc_pixel_size', I4),
                               ('reserved', I4)]

MODE_BLOCK = np.dtype([('satellite_number', I4),
                       ('satellite_name', '|S12'),
                       ('observation_time_ad', '|S16'),
                       ('observation_time_mjd', R8),
                       ('gms_operation_mode', I4),
                       ('dpc_operation_mode', I4),
                       ('vissr_observation_mode', I4),
                       ('scanner_selection', I4),
                       ('sensor_selection', I4),
                       ('sensor_mode', I4),
                       ('scan_frame_mode', I4),
                       ('scan_mode', I4),
                       ('upper_limit_of_scan_number', I4),
                       ('lower_limit_of_scan_number', I4),
                       ('equatorial_scan_line_number', I4),
                       ('spin_rate', R4),
                       ('vis_frame_parameters', MODE_BLOCK_FRAME_PARAMETERS),
                       ('ir_frame_parameters', MODE_BLOCK_FRAME_PARAMETERS),
                       ('satellite_height', R4),
                       ('earth_radius', R4),
                       ('ssp_longitude', R4),
                       ('reserved_1', I4, 9),
                       ('table_of_sensor_trouble', I4, 14),
                       ('reserved_2', I4, 36),
                       ('status_tables_of_data_relative_address_segment', I4, 60)])

COORDINATE_CONVERSION_PARAMETERS = np.dtype([
    ('data_segment', I4),
    ('data_validity', I4),
    ('data_generation_time', TIME),
    ('scheduled_observation_time', R8),
    ('stepping_angle_along_line', CHANNELS),
    ('sampling_angle_along_pixel', CHANNELS),
    ('central_line_number_of_vissr_frame', CHANNELS),
    ('central_pixel_number_of_vissr_frame', CHANNELS),
    ('pixel_difference_of_vissr_center_from_normal_position', CHANNELS),
    ('number_of_sensor_elements', CHANNELS),
    ('total_number_of_vissr_frame_lines', CHANNELS),
    ('total_number_of_vissr_frame_pixels', CHANNELS),
    ('vissr_misalignment', R4, (3,)),
    ('matrix_of_misalignment', R4, (3, 3)),
    ('parameters', [('judgement_of_observation_convergence_time', R4),
                    ('judgement_of_line_convergence', R4),
                    ('east_west_angle_of_sun_light_condense_prism', R4),
                    ('north_south_angle_of_sun_light_condense_prism', R4),
                    ('pi', R4),
                    ('pi_divided_by_180', R4),
                    ('180_divided_by_pi', R4),
                    ('equatorial_radius', R4),
                    ('oblateness_of_earth', R4),
                    ('eccentricity_of_earth_orbit', R4),
                    ('first_angle_of_vissr_observation_in_sdb', R4),
                    ('upper_limited_line_of_2nd_prism_for_vis_solar_observation', R4),
                    ('lower_limited_line_of_1st_prism_for_vis_solar_observation', R4),
                    ('upper_limited_line_of_3rd_prism_for_vis_solar_observation', R4),
                    ('lower_limited_line_of_2nd_prism_for_vis_solar_observation', R4)]),
    ('solar_stepping_angle_along_line', VISIR_SOLAR),
    ('solar_sampling_angle_along_pixel', VISIR_SOLAR),
    ('solar_center_line_of_vissr_frame', VISIR_SOLAR),
    ('solar_center_pixel_of_vissr_frame', VISIR_SOLAR),
    ('solar_pixel_difference_of_vissr_center_from_normal_position', VISIR_SOLAR),
    ('solar_number_of_sensor_elements', VISIR_SOLAR),
    ('solar_total_number_of_vissr_frame_lines', VISIR_SOLAR),
    ('solar_total_number_of_vissr_frame_pixels', VISIR_SOLAR),
    ('reserved_1', I4, 19),
    ('orbital_parameters', [('epoch_time', R8),
                            ('semi_major_axis', R8),
                            ('eccentricity', R8),
                            ('orbital_inclination', R8),
                            ('longitude_of_ascending_node', R8),
                            ('argument_of_perigee', R8),
                            ('mean_anomaly', R8),
                            ('longitude_of_ssp', R8),
                            ('latitude_of_ssp', R8)]),
    ('reserved_2', I4, 2),
    ('attitude_parameters', [('epoch_time', R8),
                             ('angle_between_z_axis_and_satellite_spin_axis_at_epoch_time', R8),
                             ('angle_change_rate_between_spin_axis_and_z_axis', R8),
                             ('angle_between_spin_axis_and_zy_axis', R8),
                             ('angle_change_rate_between_spin_axis_and_zt_axis', R8),
                             ('daily_mean_of_spin_rate', R8)]),
    ('reserved_3', I4, 529),
    ('correction_of_image_distortion', [('stepping_angle_along_line_of_ir1', R4),
                                        ('stepping_angle_along_line_of_ir2', R4),
                                        ('stepping_angle_along_line_of_wv', R4),
                                        ('stepping_angle_along_line_of_vis', R4),
                                        ('sampling_angle_along_pixel_of_ir1', R4),
                                        ('sampling_angle_along_pixel_of_ir2', R4),
                                        ('sampling_angle_along_pixel_of_wv', R4),
                                        ('sampling_angle_along_pixel_of_vis', R4),
                                        ('x_component_vissr_misalignment', R4),
                                        ('y_component_vissr_misalignment', R4)])
])

ATTITUDE_PREDICTION_DATA = np.dtype([('prediction_time_mjd', R8),
                                     ('prediction_time_utc', TIME),
                                     ('right_ascension_of_attitude', R8),
                                     ('declination_of_attitude', R8),
                                     ('sun_earth_angle', R8),
                                     ('spin_rate', R8),
                                     ('right_ascension_of_orbital_plane', R8),
                                     ('declination_of_orbital_plane', R8),
                                     ('reserved', R8),
                                     ('eclipse_flag', I4),
                                     ('spin_axis_flag', I4)])

ATTITUDE_PREDICTION = np.dtype([('data_segment', I4),
                                ('data_validity', I4),
                                ('data_generation_time', TIME),
                                ('start_time', R8),
                                ('end_time', R8),
                                ('prediction_interval_time', R8),
                                ('number_of_prediction', I4),
                                ('data_size', I4),
                                ('data', ATTITUDE_PREDICTION_DATA, (33,))])

ORBIT_PREDICTION_DATA = [('prediction_time_mjd', R8),
                         ('prediction_time_utc', TIME),
                         ('satellite_position_1950', R8, (3,)),
                         ('satellite_velocity_1950', R8, (3,)),
                         ('satellite_position_earth_fixed', R8, (3,)),
                         ('satellite_velocity_earth_fixed', R8, (3,)),
                         ('greenwich_sidereal_time', R8),
                         ('sat_sun_vector_1950', [('azimuth', R8),
                                                  ('elevation', R8)]),
                         ('sat_sun_vector_earth_fixed', [('azimuth', R8),
                                                         ('elevation', R8)]),
                         ('conversion_matrix', R8, (3, 3)),
                         ('moon_directional_vector', R8, (3,)),
                         ('satellite_position', [('ssp_longitude', R8),
                                                 ('ssp_latitude', R8),
                                                 ('satellite_height', R8)]),
                         ('eclipse_period_flag', I4),
                         ('reserved', I4)]

ORBIT_PREDICTION = np.dtype([('data_segment', I4),
                             ('data_validity', I4),
                             ('data_generation_time', TIME),
                             ('start_time', R8),
                             ('end_time', R8),
                             ('prediction_interval_time', R8),
                             ('number_of_prediction', I4),
                             ('data_size', I4),
                             ('data', ORBIT_PREDICTION_DATA, (9,))])

VIS_CALIBRATION_TABLE = np.dtype([
    ('channel_number', I4),
    ('data_validity', I4),
    ('updated_time', TIME),
    ('table_id', I4),
    ('brightness_albedo_conversion_table', R4, (64,)),
    ('vis_channel_staircase_brightness_data', R4, (6,)),
    ('coefficients_table_of_vis_staircase_regression_curve', R4, (10,)),
    ('brightness_table_for_calibration', [('universal_space_brightness', R4),
                                          ('solar_brightness', R4)]),
    ('calibration_uses_brightness_correspondence_voltage_chart', [('universal_space_voltage', R4),
                                                                  ('solar_voltage', R4)]),
    ('calibration_coefficients_of_radiation_observation', [('G', R4), ('V0', R4)]),
    ('reserved', I4, (9,))
 ])

VIS_CALIBRATION = np.dtype([('data_segment', I4),
                            ('data_validity', I4),
                            ('data_generation_time', TIME),
                            ('sensor_group', I4),
                            ('vis1_calibration_table', VIS_CALIBRATION_TABLE),
                            ('vis2_calibration_table', VIS_CALIBRATION_TABLE),
                            ('vis3_calibration_table', VIS_CALIBRATION_TABLE),
                            ('reserved', I4, (267,))])

TELEMETRY_DATA = np.dtype([
    ('shutter_temp', R4),
    ('redundant_mirror_temp', R4),
    ('primary_mirror_temp', R4),
    ('baffle_fw_temp', R4),
    ('baffle_af_temp', R4),
    ('15_volt_auxiliary_power_supply', R4),
    ('radiative_cooler_temp_1', R4),
    ('radiative_cooler_temp_2', R4),
    ('electronics_module_temp', R4),
    ('scan_mirror_temp', R4),
    ('shutter_cavity_temp', R4),
    ('primary_mirror_sealed_temp', R4),
    ('redundant_mirror_sealed_temp', R4),
    ('shutter_temp_2', R4),
    ('reserved', R4, (2,))
])

IR_CALIBRATION = np.dtype([
    ('data_segment', I4),
    ('data_validity', I4),
    ('updated_time', TIME),
    ('sensor_group', I4),
    ('table_id', I4),
    ('reserved_1', I4, (2,)),
    ('conversion_table_of_equivalent_black_body_radiation', R4, (256,)),
    ('conversion_table_of_equivalent_black_body_temperature', R4, (256,)),
    ('staircase_brightness_data', R4, (6,)),
    ('coefficients_table_of_staircase_regression_curve', R4, (10,)),
    ('brightness_data_for_calibration', [('brightness_of_space', R4),
                                         ('brightness_of_black_body_shutter', R4),
                                         ('reserved', R4)]),
    ('voltage_table_for_brightness_of_calibration', [('voltage_of_space', R4),
                                                     ('voltage_of_black_body_shutter', R4),
                                                     ('reserved', R4)]),
    ('calibration_coefficients_of_radiation_observation', [('G', R4), ('V0', R4)]),
    ('valid_shutter_temperature', R4),
    ('valid_shutter_radiation', R4),
    ('telemetry_data_table', TELEMETRY_DATA),
    ('flag_of_calid_shutter_temperature_calculation', I4),
    ('reserved_2', I4, (109,))
])

SIMPLE_COORDINATE_CONVERSION_TABLE = np.dtype([
    ('coordinate_conversion_table', I2, (1250,)),
    ('earth_equator_radius', R4),
    ('satellite_height', R4),
    ('stepping_angle', R4),
    ('sampling_angle', R4),
    ('ssp_latitude', R4),
    ('ssp_longitude', R4),
    ('ssp_line_number', R4),
    ('ssp_pixel_number', R4),
    ('pi', R4),
    ('line_correction_ir1_vis', R4),
    ('pixel_correction_ir1_vis', R4),
    ('line_correction_ir1_ir2', R4),
    ('pixel_correction_ir1_ir2', R4),
    ('line_correction_ir1_wv', R4),
    ('pixel_correction_ir1_wv', R4),
    ('reserved', R4, (32,)),
])

IMAGE_PARAMS = {
    'mode': {
        'dtype': MODE_BLOCK,
        'offset': {
            VIS_CHANNEL: 2 * BLOCK_SIZE_VIS,
            IR_CHANNEL: 2 * BLOCK_SIZE_IR
        }
    },
    'coordinate_conversion': {
        'dtype': COORDINATE_CONVERSION_PARAMETERS,
        'offset': {
            VIS_CHANNEL: 2 * BLOCK_SIZE_VIS + 2 * IMAGE_PARAM_ITEM_SIZE,
            IR_CHANNEL: 4 * BLOCK_SIZE_IR
        }
    },
    'attitude_prediction': {
        'dtype': ATTITUDE_PREDICTION,
        'offset': {
            VIS_CHANNEL: 2 * BLOCK_SIZE_VIS + 3 * IMAGE_PARAM_ITEM_SIZE,
            IR_CHANNEL: 5 * BLOCK_SIZE_IR
        },
        'preserve': 'data'
    },
    'orbit_prediction_1': {
        'dtype': ORBIT_PREDICTION,
        'offset': {
            VIS_CHANNEL: 3 * BLOCK_SIZE_VIS,
            IR_CHANNEL: 6 * BLOCK_SIZE_IR
        },
        'preserve': 'data'
    },
    'orbit_prediction_2': {
        'dtype': ORBIT_PREDICTION,
        'offset': {
            VIS_CHANNEL: 3 * BLOCK_SIZE_VIS + 1 * IMAGE_PARAM_ITEM_SIZE,
            IR_CHANNEL: 7 * BLOCK_SIZE_IR
        },
        'preserve': 'data'
    },
    'vis_calibration': {
        'dtype': VIS_CALIBRATION,
        'offset': {
            VIS_CHANNEL: 3 * BLOCK_SIZE_VIS + 3 * IMAGE_PARAM_ITEM_SIZE,
            IR_CHANNEL: 9 * BLOCK_SIZE_IR
        },
        'preserve': 'data'
    },
    'ir1_calibration': {
        'dtype': IR_CALIBRATION,
        'offset': {
            VIS_CHANNEL: 4 * BLOCK_SIZE_VIS,
            IR_CHANNEL: 10 * BLOCK_SIZE_IR
        },
    },
    'ir2_calibration': {
        'dtype': IR_CALIBRATION,
        'offset': {
            VIS_CHANNEL: 4 * BLOCK_SIZE_VIS + IMAGE_PARAM_ITEM_SIZE,
            IR_CHANNEL: 11 * BLOCK_SIZE_IR
        },
    },
    'wv_calibration': {
        'dtype': IR_CALIBRATION,
        'offset': {
            VIS_CHANNEL: 4 * BLOCK_SIZE_VIS + 2 * IMAGE_PARAM_ITEM_SIZE,
            IR_CHANNEL: 12 * BLOCK_SIZE_IR
        },
    },
    'simple_coordinate_conversion_table': {
        'dtype': SIMPLE_COORDINATE_CONVERSION_TABLE,
        'offset': {
            VIS_CHANNEL: 5 * BLOCK_SIZE_VIS + 2 * IMAGE_PARAM_ITEM_SIZE,
            IR_CHANNEL: 16 * BLOCK_SIZE_IR
        },
    }
}

LINE_CONTROL_WORD = np.dtype([
    ('data_id', U1, (4, )),
    ('line_number', I4),
    ('line_name', I4),
    ('error_line_flag', I4),
    ('error_message', I4),
    ('mode_error_flag', I4),
    ('scan_time', R8),
    ('beta_angle', R4),
    ('west_side_earth_edge', I4),
    ('east_side_earth_edge', I4),
    ('received_time_1', R8),  # Typo in format description (I*4)
    ('received_time_2', I4),
    ('reserved', U1, (8, ))
])

IMAGE_DATA_BLOCK_IR = np.dtype([('LCW', LINE_CONTROL_WORD),
                                ('DOC', U1, (256,)),  # Omitted
                                ('image_data', U1, 3344)])

IMAGE_DATA_BLOCK_VIS = np.dtype([('LCW', LINE_CONTROL_WORD),
                                 ('DOC', U1, (64,)),  # Omitted
                                 ('image_data', U1, (13376,))])

IMAGE_DATA = {
    VIS_CHANNEL: {
        'offset': 6 * BLOCK_SIZE_VIS,
        'dtype': IMAGE_DATA_BLOCK_VIS,
    },
    IR_CHANNEL: {
        'offset': 18 * BLOCK_SIZE_IR,
        'dtype': IMAGE_DATA_BLOCK_IR
    }
}
# fmt: on


def recarr2dict(arr, preserve=None):
    if not preserve:
        preserve = []
    res = {}
    for key, value in zip(arr.dtype.names, arr):
        if key.startswith('reserved'):
            continue
        if value.dtype.names and key not in preserve:
            # Nested record array
            res[key] = recarr2dict(value)
        else:
            # Scalar or record array that shall be preserved
            res[key] = value
    return res


class GMS5VISSRFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info, mask_space=True):
        super(GMS5VISSRFileHandler, self).__init__(filename, filename_info, filetype_info)
        self._filename = filename
        self._filename_info = filename_info
        self._header, self._channel_type = self._read_header(filename)
        self._mda = self._get_mda()
        self._mask_space = mask_space

    def _read_header(self, filename):
        header = {}
        with generic_open(filename, mode='rb') as file_obj:
            header['control_block'] = self._read_control_block(file_obj)
            channel_type = self._get_channel_type(header['control_block']['parameter_block_size'])
            header['image_parameters'] = self._read_image_params(file_obj, channel_type)
        return header, channel_type

    @staticmethod
    def _get_channel_type(parameter_block_size):
        if parameter_block_size == 4:
            return VIS_CHANNEL
        elif parameter_block_size == 16:
            return IR_CHANNEL
        raise ValueError('Cannot determine channel type: Unknown parameter block size.')

    def _read_control_block(self, file_obj):
        ctrl_block = read_from_file_obj(
            file_obj,
            dtype=CONTROL_BLOCK,
            count=1
        )
        return recarr2dict(ctrl_block[0])

    def _read_image_params(self, file_obj, channel_type):
        """Read image parameters from the header."""
        image_params = {}
        for name, param in IMAGE_PARAMS.items():
            image_params[name] = self._read_image_param(file_obj, param, channel_type)

        image_params['orbit_prediction'] = self._concat_orbit_prediction(
            image_params.pop('orbit_prediction_1'),
            image_params.pop('orbit_prediction_2')
        )
        return image_params

    @staticmethod
    def _read_image_param(file_obj, param, channel_type):
        """Read a single image parameter block from the header."""
        image_params = read_from_file_obj(
            file_obj,
            dtype=param["dtype"],
            count=1,
            offset=param['offset'][channel_type]
        )
        return recarr2dict(image_params[0], preserve=param.get('preserve'))

    @staticmethod
    def _concat_orbit_prediction(orb_pred_1, orb_pred_2):
        """Concatenate orbit prediction data.

        It is split over two image parameter blocks in the header.
        """
        orb_pred = orb_pred_1
        orb_pred['data'] = np.concatenate([orb_pred_1['data'], orb_pred_2['data']])
        return orb_pred

    def _get_frame_parameters_key(self):
        if self._channel_type == VIS_CHANNEL:
            return 'vis_frame_parameters'
        return 'ir_frame_parameters'

    def _get_actual_shape(self):
        actual_num_lines = self._header['control_block']['available_block_size_of_image_data']
        _, nominal_num_pixels = self._get_nominal_shape()
        return actual_num_lines, nominal_num_pixels

    def _get_nominal_shape(self):
        frame_params = self._header['image_parameters']['mode'][self._get_frame_parameters_key()]
        return frame_params['number_of_lines'], frame_params['number_of_pixels']

    def _get_mda(self):
        mode_block = self._header['image_parameters']['mode']
        return {
            'platform': mode_block['satellite_name'].decode().strip().upper(),
            'sensor': 'VISSR'
        }

    def get_dataset(self, dataset_id, ds_info):
        image_data = self._get_image_data()
        counts = self._get_counts(image_data)
        dataset = self._calibrate(counts, dataset_id)
        space_masker = SpaceMasker(image_data, dataset_id["name"])
        dataset = self._mask_space_pixels(dataset, space_masker)
        self._attach_lons_lats(dataset, dataset_id)
        return dataset

    def _get_image_data(self):
        image_data = self._read_image_data()
        return da.from_array(image_data, chunks=(CHUNK_SIZE,))

    def _read_image_data(self):
        num_lines, _ = self._get_actual_shape()
        specs = self._get_image_data_type_specs()
        with generic_open(self._filename, "rb") as file_obj:
            return read_from_file_obj(
                file_obj,
                dtype=specs["dtype"],
                count=num_lines,
                offset=specs["offset"]
            )

    def _get_image_data_type_specs(self):
        return IMAGE_DATA[self._channel_type]

    def _get_counts(self, image_data):
        return self._make_counts_data_array(image_data)

    def _make_counts_data_array(self, image_data):
        return xr.DataArray(
            image_data['image_data'],
            dims=('y', 'x'),
            coords={
                'acq_time': ('y', self._get_acq_time(image_data)),
                'line_number': ('y', self._get_line_number(image_data))
            }
        )

    def _get_acq_time(self, dask_array):
        acq_time = dask_array['LCW']['scan_time'].compute()
        return mjd2datetime64(acq_time)

    def _get_line_number(self, dask_array):
        return dask_array['LCW']['line_number'].compute()

    def _calibrate(self, counts, dataset_id):
        table = self._get_calibration_table(dataset_id)
        cal = Calibrator(table)
        return cal.calibrate(counts, dataset_id["calibration"])

    def _get_calibration_table(self, dataset_id):
        tables = {
            "VIS": self._header['image_parameters']['vis_calibration']["vis1_calibration_table"]["brightness_albedo_conversion_table"],
            "IR1": self._header['image_parameters']['ir1_calibration']["conversion_table_of_equivalent_black_body_temperature"],
            "IR2": self._header['image_parameters']['ir2_calibration']["conversion_table_of_equivalent_black_body_temperature"],
            "IR3": self._header['image_parameters']['wv_calibration']["conversion_table_of_equivalent_black_body_temperature"]
        }
        return tables[dataset_id["name"]]

    def get_area_def_test(self, dsid):
        alt_ch_name = ALT_CHANNEL_NAMES[dsid['name']]
        num_lines, num_pixels = self._get_actual_shape()
        mode_block = self._header['image_parameters']['mode']
        coord_conv = self._header['image_parameters']['coordinate_conversion']
        stepping_angle = coord_conv['stepping_angle_along_line'][alt_ch_name]
        sampling_angle = coord_conv['sampling_angle_along_pixel'][alt_ch_name]
        center_line_vissr_frame = coord_conv['central_line_number_of_vissr_frame'][alt_ch_name]
        center_pixel_vissr_frame = coord_conv['central_pixel_number_of_vissr_frame'][alt_ch_name]
        line_offset = self._header['control_block']['head_valid_line_number']
        pixel_offset = coord_conv['pixel_difference_of_vissr_center_from_normal_position'][
            alt_ch_name]
        print(coord_conv['vissr_misalignment'])
        print(coord_conv['matrix_of_misalignment'])

        equatorial_radius = coord_conv['parameters']['equatorial_radius']
        oblateness = coord_conv['parameters']['oblateness_of_earth']
        name_dict = geos_area.get_geos_area_naming({
            'platform_name': self._mda['platform'],
            'instrument_name': self._mda['sensor'],
            'service_name': 'western-pacific',
            'service_desc': 'Western Pacific',
            'resolution': dsid['resolution']
        })
        proj_dict = {
            'a_name': name_dict['area_id'],
            'p_id': name_dict['area_id'],
            'a_desc': name_dict['description'],
            'ssp_lon': coord_conv['orbital_parameters']['longitude_of_ssp'],
            'a': equatorial_radius,
            'b': _get_polar_earth_radius(equatorial_radius, oblateness),
            'h': mode_block['satellite_height'],
            'nlines': num_lines,
            'ncols': num_pixels,
            'lfac': geos_area.sampling_to_lfac_cfac(stepping_angle),
            'cfac': geos_area.sampling_to_lfac_cfac(sampling_angle),
            'coff': center_pixel_vissr_frame - pixel_offset,
            'loff': center_line_vissr_frame - line_offset,
            'scandir': 'N2S'
        }
        from pprint import pprint

        # pprint(mode_block)
        pprint(coord_conv)
        extent = geos_area.get_area_extent(proj_dict)
        area = geos_area.get_area_definition(proj_dict, extent)
        return area

    def _mask_space_pixels(self, dataset, space_masker):
        if self._mask_space:
            return space_masker.mask_space(dataset)
        return dataset

    def _attach_lons_lats(self, dataset, dataset_id):
        lons, lats = self._get_lons_lats(dataset, dataset_id)
        dataset.coords['lon'] = lons
        dataset.coords['lat'] = lats

    def _get_lons_lats(self, dataset, dataset_id):
        lines, pixels = self._get_image_coords(dataset)
        static_params = self._get_static_navigation_params(dataset_id)
        predicted_params = self._get_predicted_navigation_params()
        lons, lats = nav.get_lons_lats(
            lines=lines,
            pixels=pixels,
            static_params=static_params,
            predicted_params=predicted_params
        )
        return self._make_lons_lats_data_array(lons, lats)

    def _get_image_coords(self, data):
        lines = data.coords['line_number'].values
        pixels = np.arange(data.shape[1])
        return lines.astype(np.float64), pixels.astype(np.float64)

    def _get_static_navigation_params(self, dataset_id):
        """Get static navigation parameters.

        Note that, "central_line_number_of_vissr_frame" is different for each
        channel, even if their spatial resolution is identical. For example:

        VIS: 5513.0
        IR1: 1378.5
        IR2: 1378.7
        IR3: 1379.1001
        """
        alt_ch_name = ALT_CHANNEL_NAMES[dataset_id['name']]
        mode_block = self._header['image_parameters']['mode']
        coord_conv = self._header['image_parameters']['coordinate_conversion']
        center_line_vissr_frame = coord_conv['central_line_number_of_vissr_frame'][alt_ch_name]
        center_pixel_vissr_frame = coord_conv['central_pixel_number_of_vissr_frame'][alt_ch_name]
        pixel_offset = coord_conv['pixel_difference_of_vissr_center_from_normal_position'][
            alt_ch_name]
        scan_params = nav.ScanningParameters(
            start_time_of_scan=coord_conv['scheduled_observation_time'],
            spinning_rate=mode_block['spin_rate'],
            num_sensors=coord_conv['number_of_sensor_elements'][alt_ch_name],
            sampling_angle=coord_conv['sampling_angle_along_pixel'][alt_ch_name],
        )
        # Use earth radius and flattening from JMA's Msial library, because
        # the values in the data seem to be pretty old. For example the
        # equatorial radius is from the Bessel Ellipsoid (1841).
        proj_params = nav.ProjectionParameters(
            line_offset=center_line_vissr_frame,
            pixel_offset=center_pixel_vissr_frame + pixel_offset,
            stepping_angle=coord_conv['stepping_angle_along_line'][alt_ch_name],
            sampling_angle=coord_conv['sampling_angle_along_pixel'][alt_ch_name],
            misalignment=np.ascontiguousarray(coord_conv['matrix_of_misalignment'].transpose().astype(np.float64)),
            earth_flattening=nav.EARTH_FLATTENING,
            earth_equatorial_radius=nav.EARTH_EQUATORIAL_RADIUS
        )
        return scan_params, proj_params

    def _get_predicted_navigation_params(self):
        """Get predictions of time-dependent navigation parameters."""
        att_pred = self._header['image_parameters']['attitude_prediction']['data']
        orb_pred = self._header['image_parameters']['orbit_prediction']['data']
        attitude_prediction = nav.AttitudePrediction(
            prediction_times=att_pred['prediction_time_mjd'].astype(np.float64),
            angle_between_earth_and_sun=att_pred['sun_earth_angle'].astype(np.float64),
            angle_between_sat_spin_and_z_axis=att_pred['right_ascension_of_attitude'].astype(np.float64),
            angle_between_sat_spin_and_yz_plane=att_pred['declination_of_attitude'].astype(np.float64),
        )
        orbit_prediction = nav.OrbitPrediction(
            prediction_times=orb_pred['prediction_time_mjd'].astype(np.float64),
            greenwich_sidereal_time=np.deg2rad(orb_pred['greenwich_sidereal_time'].astype(np.float64)),
            declination_from_sat_to_sun=np.deg2rad(orb_pred['sat_sun_vector_earth_fixed']['elevation'].astype(np.float64)),
            right_ascension_from_sat_to_sun=np.deg2rad(orb_pred['sat_sun_vector_earth_fixed']['azimuth'].astype(np.float64)),
            sat_position_earth_fixed_x=orb_pred['satellite_position_earth_fixed'][:, 0].astype(np.float64),
            sat_position_earth_fixed_y=orb_pred['satellite_position_earth_fixed'][:, 1].astype(np.float64),
            sat_position_earth_fixed_z=orb_pred['satellite_position_earth_fixed'][:, 2].astype(np.float64),
            nutation_precession=np.ascontiguousarray(orb_pred['conversion_matrix'].transpose(0, 2, 1).astype(np.float64))
        )
        return attitude_prediction, orbit_prediction

    def _make_lons_lats_data_array(self, lons, lats):
        lons = xr.DataArray(lons, dims=('y', 'x'),
                            attrs={'standard_name': 'longitude',
                                   "units": "degrees_east"})
        lats = xr.DataArray(lats, dims=('y', 'x'),
                            attrs={'standard_name': 'latitude',
                                   "units": "degrees_north"})
        return lons, lats


def read_from_file_obj(file_obj, dtype, count, offset=0):
    file_obj.seek(offset)
    data = file_obj.read(dtype.itemsize * count)
    return np.frombuffer(data, dtype=dtype, count=count)


class Calibrator:
    def __init__(self, calib_table):
        self._calib_table = calib_table

    def calibrate(self, counts, calibration):
        if calibration == "counts":
            return counts
        res = da.map_blocks(
            self._lookup_calib_table,
            counts.data,
            calib_table=self._calib_table,
            dtype=np.float32,
        )
        return self._make_data_array(res, counts)

    def _make_data_array(self, interp, counts):
        return xr.DataArray(
            interp,
            dims=counts.dims,
            coords=counts.coords,
        )

    def _lookup_calib_table(self, counts, calib_table):
        return calib_table[counts]


class SpaceMasker:
    _fill_value = -1  # scanline not intersecting the earth

    def __init__(self, image_data, channel):
        self._image_data = image_data
        self._channel = channel
        self._shape = image_data["image_data"].shape
        self._earth_mask = self._get_earth_mask()

    def mask_space(self, dataset):
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
        if self._is_vis_channel():
            edges = self._correct_vis_edges(edges)
        return edges

    def _is_vis_channel(self):
        return self._channel == "VIS"

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
