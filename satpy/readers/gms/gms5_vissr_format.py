"""GMS-5 VISSR archive data format.

Reference: `VISSR Format Description`_

.. _VISSR Format Description:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/VISSR_FORMAT_GMS-5.pdf
"""

import numpy as np

U1 = ">u1"
I2 = ">i2"
I4 = ">i4"
R4 = ">f4"
R8 = ">f8"

VIS_CHANNEL = "VIS"
IR_CHANNEL = "IR"
CHANNEL_TYPES = {
    "VIS": VIS_CHANNEL,
    "IR1": IR_CHANNEL,
    "IR2": IR_CHANNEL,
    "IR3": IR_CHANNEL,
    "WV": IR_CHANNEL,
}
ALT_CHANNEL_NAMES = {"VIS": "VIS", "IR1": "IR1", "IR2": "IR2", "IR3": "WV"}
BLOCK_SIZE_VIS = 13504
BLOCK_SIZE_IR = 3664

IMAGE_PARAM_ITEM_SIZE = 2688
TIME = [("date", I4), ("time", I4)]
CHANNELS = [("VIS", R4), ("IR1", R4), ("IR2", R4), ("WV", R4)]
VISIR_SOLAR = [("VIS", R4), ("IR", R4)]

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
