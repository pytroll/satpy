"""GMS-1/2/3/4 VISSR archive data format.

Reference: `VISSR Format Description`_

.. _VISSR Format Description:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/VISSR_FORMAT_GMS-4.pdf
"""

import numpy as np

U1 = ">u1"
I2 = ">i2"
I4 = ">i4"
R4 = ">f4"
R8 = ">f8"

VIS_CHANNEL = "VIS"
IR_CHANNEL = "IR"
CHANNEL_TYPES = {"VIS": VIS_CHANNEL, "IR": IR_CHANNEL}

IR_BLOCK_LEN = 14016
VIS_BLOCK_LEN = 27008
IMAGE_PARAM_ITEM_SIZE = 2688  # size of one 672-word segment, both channels

TIME = [("date", I4), ("time", I4)]

# NOTE: unlike GMS-5's format (which has VIS/IR1/IR2/WV all in the same
# CHANNELS-style sub-struct), GMS-1-4 archive files only ever have a
# single VIS and a single IR channel, plus VIS-solar/IR-solar variants
# for the solar-observation mode. Mirroring gms5_vissr_format's CHANNELS
# convention would be misleading here, so scanning-angle-type fields are
# plain scalars per channel rather than a sub-struct.

CONTROL_BLOCK_IR = np.dtype([
    ("control_block_size", I2),
    ("head_block_number_of_parameter_block", I2),
    ("parameter_block_size", I2),
    ("head_block_number_of_image_data", I2),
    ("total_block_size_of_image_data", I2),
    ("available_block_size_of_image_data", I2),
    ("head_valid_line_number", I2),
    ("final_valid_line_number", I2),
    ("reserved", I2, (8,)),
])

CONTROL_BLOCK_VIS = np.dtype([
    ("control_block_size", I2),
    ("head_block_number_of_parameter_block", I2),
    ("parameter_block_size", I2),
    ("head_block_number_of_image_data", I2),
    ("total_block_size_of_image_data", I2),
    ("available_block_size_of_image_data", I2),
    ("head_valid_line_number", I2),
    ("final_valid_line_number", I2),
    ("final_data_block_number", I2),
    ("reserved", I2, (7,)),
])

MODE_BLOCK_FRAME_PARAMETERS = [
    ("bit_length", I4),
    ("number_of_lines", I4),
    ("number_of_pixels", I4),
    ("stepping_angle", R4),
    ("sampling_angle", R4),
    ("lcw_pixel_size", I4),
    ("doc_pixel_size", I4),
    ("reserved", I4),
]

MODE_BLOCK = np.dtype([
    ("satellite_number", I4),
    ("satellite_name", "|S12"),
    ("observation_time_ad", "|S16"),
    ("observation_time_mjd", R8),
    ("gms_operation_mode", I4),
    ("dpc_operation_mode", I4),
    ("vissr_observation_mode", I4),
    ("scanner_selection", I4),
    ("sensor_selection", I4),
    ("sensor_mode", I4),
    ("scan_frame_mode", I4),
    ("scan_mode", I4),
    ("upper_limit_of_scan_number", I4),
    ("lower_limit_of_scan_number", I4),
    ("equatorial_scan_line_number", I4),
    ("spin_rate", R4),
    ("vis_frame_parameters", MODE_BLOCK_FRAME_PARAMETERS),
    ("ir_frame_parameters", MODE_BLOCK_FRAME_PARAMETERS),
    ("satellite_height", R4),
    ("earth_radius", R4),
    ("ssp_longitude", R4),
    ("reserved_1", I4, (9,)),
    ("table_of_sensor_trouble", I4, (10,)),
    ("reserved_2", I4, (40,)),
    ("status_tables_of_data_relative_address_segment", I4, (40,)),
    ("reserved_3", I4, (532,)),
])

# Coordinate transformation parameters segment (doc item (5)). Word
# numbers in comments are 0-based-from-segment-start, matching what
# gms_vissr.py's CoordTransformSegment used and what was empirically
# confirmed against real files.
COORDINATE_CONVERSION_PARAMETERS = np.dtype([
    ("data_segment", I4),
    ("reserved_0", I4),
    ("data_generation_time", TIME),
    ("scheduled_observation_time", R8),        # word 4 (0-based)
    ("stepping_angle_vis", R4),                # word 6
    ("stepping_angle_ir", R4),
    ("stepping_angle_vis_solar", R4),
    ("stepping_angle_ir_solar", R4),
    ("sampling_angle_vis", R4),                # word 10
    ("sampling_angle_ir", R4),
    ("sampling_angle_vis_solar", R4),
    ("sampling_angle_ir_solar", R4),
    ("central_line_vis", R4),                  # word 14
    ("central_line_ir", R4),
    ("central_line_vis_solar", R4),
    ("central_line_ir_solar", R4),
    ("central_pixel_vis", R4),                 # word 18
    ("central_pixel_ir", R4),
    ("central_pixel_vis_solar", R4),
    ("central_pixel_ir_solar", R4),
    ("reserved_1", R4, (4,)),                  # pixel-difference-from-normal, words 22-25
    ("num_sensors_vis", R4),                   # word 26
    ("num_sensors_ir", R4),
    ("num_sensors_vis_solar", R4),
    ("num_sensors_ir_solar", R4),
    ("total_lines_vis", R4),                   # word 30
    ("total_lines_ir", R4),
    ("total_lines_vis_solar", R4),
    ("total_lines_ir_solar", R4),
    ("total_pixels_vis", R4),                  # word 34
    ("total_pixels_ir", R4),
    ("total_pixels_vis_solar", R4),
    ("total_pixels_ir_solar", R4),
    ("vissr_misalignment", R4, (3,)),          # word 38
    ("matrix_of_misalignment", R4, (9,)),      # word 41, column-major (see nav module)
    ("reserved_2", I4, (80,)),                 # pad up to word 130 exactly
    ("daily_mean_spin_rate", R8),              # word 130
    ("reserved_3", I4, (540,)),                # pad to fill 672 words total
])

ATTITUDE_PREDICTION_DATA_SET = np.dtype([
    ("prediction_time_mjd", R8),   # doubles 0-1 (word 0)
    ("reserved_1", R8),            # doubles 2-3 (unused per doc)
    ("angle_between_z_axis_and_spin_axis", R8),   # ATTALP, doubles 4-5
    ("angle_between_spin_axis_and_yz_plane", R8),  # ATTDEL, doubles 6-7
    ("beta_angle", R8),            # doubles 8-9
    ("spin_rate", R8),             # doubles 10-11
    ("reserved_2", R8, (4,)),      # pad to 20 words (10 doubles) total
])

ATTITUDE_PREDICTION = np.dtype([
    ("data_segment", I4),
    ("data_validity", I4),
    ("data_generation_time", TIME),
    ("reserved_1", I4, (8,)),   # pad to word 12
    ("data", ATTITUDE_PREDICTION_DATA_SET, (33,)),
])

ORBIT_PREDICTION_DATA_SET = np.dtype([
    ("prediction_time_mjd", R8),                    # double 0
    ("reserved_1", R8, (7,)),                       # doubles 1-7 (pred time UTC etc, unused)
    ("satellite_position_earth_fixed", R8, (3,)),   # doubles 8-10 (words 16-21 of orig doc)
    ("reserved_2", R8, (3,)),                        # doubles 11-13
    ("greenwich_sidereal_time", R8),                # double 14 (word 28-29 of orig doc)
    ("reserved_3", R8, (2,)),                        # doubles 15-16
    ("right_ascension_sat_to_sun", R8),             # double 17
    ("declination_sat_to_sun", R8),                 # double 18
    ("npa_matrix", R8, (9,)),                        # doubles 19-27 of orig doc
    ("reserved_5", R8, (7,)),                        # doubles 28-34, pad to 35 doubles (70 words) total
])

ORBIT_PREDICTION = np.dtype([
    ("data_segment", I4),
    ("data_validity", I4),
    ("data_generation_time", TIME),
    ("reserved_1", I4, (8,)),  # pad to word 12
    ("data", ORBIT_PREDICTION_DATA_SET, (9,)),
    ("reserved_2", I4, (30,)),  # doc words 643-672
])

VIS_CALIBRATION_TABLE = np.dtype([
    ("channel_number", I4),
    ("data_validity", I4),
    ("updated_time", TIME),
    ("table_id", I4),
    ("brightness_albedo_conversion_table", R4, (64,)),
    ("reserved", R4, (31,)),  # pad 100-word (25-double... actually I4/R4 mixed) segment
])

VIS_CALIBRATION = np.dtype([
    ("data_segment", I4),
    ("data_validity", I4),
    ("data_generation_time", TIME),
    ("sensor_group", I4),
    ("vis1_calibration_table", VIS_CALIBRATION_TABLE),
    ("vis2_calibration_table", VIS_CALIBRATION_TABLE),
    ("vis3_calibration_table", VIS_CALIBRATION_TABLE),
    ("vis4_calibration_table", VIS_CALIBRATION_TABLE),
    ("reserved", I4, (267,)),
])

IR_CALIBRATION = np.dtype([
    ("data_segment", I4),
    ("data_validity", I4),
    ("updated_time", TIME),
    ("sensor_group", I4),
    ("table_id", I4),
    ("reserved_1", I4, (2,)),
    ("conversion_table_of_equivalent_black_body_radiation", R4, (256,)),
    ("conversion_table_of_equivalent_black_body_temperature", R4, (256,)),
    ("reserved_2", I4, (152,)),  # staircase/telemetry/etc, not needed for calibration
])

IMAGE_PARAMS_IR = {
    # Relative offsets within blocks 2/3 confirmed against gms_vissr.py's
    # already-validated _IR_PARAM2_OFFSETS/_IR_PARAM3_OFFSETS. NOTE: the
    # IR block layout has 1632-byte reserved gaps that break a naive
    # uniform IMAGE_PARAM_ITEM_SIZE stride (block2: mode | sdb | reserved
    # (1632) | ir_cal | vis_cal | reserved(1632); block3: coord |
    # attitude | reserved(1632) | orbit1 | orbit2 | reserved(1632)) --
    # do not "simplify" these back to N*IMAGE_PARAM_ITEM_SIZE.
    "mode": {"dtype": MODE_BLOCK, "offset": 1 * IR_BLOCK_LEN + 0},
    "ir_calibration": {"dtype": IR_CALIBRATION,
                        "offset": 1 * IR_BLOCK_LEN + 7008},
    "vis_calibration": {"dtype": VIS_CALIBRATION,
                         "offset": 1 * IR_BLOCK_LEN + 9696},
    "coordinate_conversion": {"dtype": COORDINATE_CONVERSION_PARAMETERS,
                               "offset": 2 * IR_BLOCK_LEN + 0},
    "attitude_prediction": {"dtype": ATTITUDE_PREDICTION,
                             "offset": 2 * IR_BLOCK_LEN + 2688},
    "orbit_prediction_1": {"dtype": ORBIT_PREDICTION,
                            "offset": 2 * IR_BLOCK_LEN + 7008},
    "orbit_prediction_2": {"dtype": ORBIT_PREDICTION,
                            "offset": 2 * IR_BLOCK_LEN + 9696},
}

# Byte offsets confirmed empirically for VIS (see module docstring).
_VIS_PARAM3_BASE = 2 * VIS_BLOCK_LEN  # doc's "block 3" -> 0-based block 2
IMAGE_PARAMS_VIS = {
    "mode": {"dtype": MODE_BLOCK, "offset": _VIS_PARAM3_BASE},
    "ir_calibration": {"dtype": IR_CALIBRATION,
                        "offset": _VIS_PARAM3_BASE + 2 * IMAGE_PARAM_ITEM_SIZE},
    "vis_calibration": {"dtype": VIS_CALIBRATION,
                         "offset": _VIS_PARAM3_BASE + 3 * IMAGE_PARAM_ITEM_SIZE},
    "coordinate_conversion": {"dtype": COORDINATE_CONVERSION_PARAMETERS,
                               "offset": _VIS_PARAM3_BASE + 4 * IMAGE_PARAM_ITEM_SIZE + 2752},
    "attitude_prediction": {"dtype": ATTITUDE_PREDICTION,
                             "offset": _VIS_PARAM3_BASE + 5 * IMAGE_PARAM_ITEM_SIZE + 2752},
    "orbit_prediction_1": {"dtype": ORBIT_PREDICTION,
                            "offset": _VIS_PARAM3_BASE + 6 * IMAGE_PARAM_ITEM_SIZE + 2752},
    "orbit_prediction_2": {"dtype": ORBIT_PREDICTION,
                            "offset": _VIS_PARAM3_BASE + 7 * IMAGE_PARAM_ITEM_SIZE + 2752},
}

LINE_CONTROL_WORD = np.dtype([
    ("data_id", U1, (4,)),
    ("line_number", I4),
    ("line_name", I4),
    ("error_line_flag", I4),
    ("error_message", I4),
    ("mode_error_flag", I4),
    ("scan_time", R8),
    ("beta_angle", R4),
    ("west_side_earth_edge", I4),
    ("east_side_earth_edge", I4),
    ("received_time", I4, (2,)),  # doc bytes 45-52 (8 bytes)
    ("reserved", U1, (12,)),
])

IMAGE_DATA_BLOCK_IR = np.dtype([
    ("LCW", LINE_CONTROL_WORD),
    ("DOC", U1, (256,)),
    ("image_data", U1, (6688,)),
])

IMAGE_DATA_BLOCK_VIS = np.dtype([
    ("LCW", LINE_CONTROL_WORD),
    ("DOC", U1, (64,)),
    ("image_data", U1, (13376,)),
])

IMAGE_DATA = {
    IR_CHANNEL: {"offset": 7 * IR_BLOCK_LEN, "dtype": IMAGE_DATA_BLOCK_IR,
                 "control_block": CONTROL_BLOCK_IR, "params": IMAGE_PARAMS_IR},
    VIS_CHANNEL: {"offset": 6 * VIS_BLOCK_LEN, "dtype": IMAGE_DATA_BLOCK_VIS,
                  "control_block": CONTROL_BLOCK_VIS, "params": IMAGE_PARAMS_VIS},
}
