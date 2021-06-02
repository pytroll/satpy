"""Unit tests for GMS-5 VISSR reader."""

import numpy as np
import pytest

import satpy.readers.gms5_vissr_navigation as nav


# Navigation references computed with JMA's Msial library (files
# VISSR_19960217_2331_IR1.A.IMG and VISSR_19960217_2331_VIS.A.IMG). The VIS
# navigation is slightly off (< 0.01 deg) compared to JMA's reference.
# This is probably due to precision problems with the copied numbers.
IR_NAVIGATION_REFERENCE = [
    {
        'line': 686,
        'pixel': 1680,
        'lon': 139.990380,
        'lat': 35.047056,
        'tolerance': 0,
        'nav_params': {
            'line_offset': 1378.5,
            'pixel_offset': 1672.5,
            'stepping_angle': 0.000140000047395,
            'sampling_angle': 0.000095719995443,
            'misalignment': np.array(
                [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                 [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                 [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
            ),
            'greenwich_sidereal_time': 2.468529732418296,
            'angle_between_earth_and_sun': 3.997397917902958,
            'declination_from_sat_to_sun': -0.208770861178982,
            'right_ascension_from_sat_to_sun': 3.304369303579407,
            'angle_between_sat_spin_and_z_axis': 3.149118633034304,
            'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
            'sat_position_earth_fixed_x': -32390963.148471601307392,
            'sat_position_earth_fixed_y': 27003395.381247851997614,
            'sat_position_earth_fixed_z': -228134.860026293463307,
            'nutation_precession': np.array(
                [[0.999936381496146, -0.010344758016410, -0.004496547784299],
                 [0.010344942303489, 0.999946489495557, 0.000017727054455],
                 [0.004496123789670, -0.000064242454080, 0.999989890320785]]
            ),
            'earth_flattening': 0.003352813177897,
            'earth_equatorial_radius': 6378136
        },
    },
    {
        'line': 2089,
        'pixel': 1793,
        'lon': 144.996967,
        'lat': -34.959853,
        'tolerance': 0,
        'nav_params': {
            'line_offset': 1378.5,
            'pixel_offset': 1672.5,
            'stepping_angle': 0.000140000047395,
            'sampling_angle': 0.000095719995443,
            'misalignment': np.array(
                [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                 [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                 [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
            ),
            'greenwich_sidereal_time': 2.530392320846865,
            'angle_between_earth_and_sun': 3.935707944355762,
            'declination_from_sat_to_sun': -0.208713576872247,
            'right_ascension_from_sat_to_sun': 3.242660398458377,
            'angle_between_sat_spin_and_z_axis': 3.149118633034304,
            'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
            'sat_position_earth_fixed_x': -32390273.633551981300116,
            'sat_position_earth_fixed_y': 27003859.543135114014149,
            'sat_position_earth_fixed_z': -210800.087589388160268,
            'nutation_precession': np.array(
                [[0.999936381432029, -0.010344763228876, -0.004496550050695],
                 [0.010344947502662, 0.999946489441823, 0.000017724053657],
                 [0.004496126086653, -0.000064239500295, 0.999989890310647]]
            ),
            'earth_flattening': 0.003352813177897,
            'earth_equatorial_radius': 6378136
        },
    },
    {
        'line': 999,
        'pixel': 2996,
        'lon': -165.023842,
        'lat': 20.005603,
        'tolerance': 0,
        'nav_params': {
            'line_offset': 1378.5,
            'pixel_offset': 1672.5,
            'stepping_angle': 0.000140000047395,
            'sampling_angle': 0.000095719995443,
            'misalignment': np.array(
                [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                 [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                 [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
            ),
            'greenwich_sidereal_time': 2.482331732831616,
            'angle_between_earth_and_sun': 3.983634620574510,
            'declination_from_sat_to_sun': -0.208758095943038,
            'right_ascension_from_sat_to_sun': 3.290601673240597,
            'angle_between_sat_spin_and_z_axis': 3.149118633034304,
            'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
            'sat_position_earth_fixed_x': -32390808.779549609869719,
            'sat_position_earth_fixed_y': 27003503.047290064394474,
            'sat_position_earth_fixed_z': -224351.430479845439550,
            'nutation_precession': np.array(
                [[0.999936381496146, -0.010344758016410, -0.004496547784299],
                 [0.010344942303489, 0.999946489495557, 0.000017727054455],
                 [0.004496123789670, -0.000064242454080, 0.999989890320785]]
            ),
            'earth_flattening': 0.003352813177897,
            'earth_equatorial_radius': 6378136
        },
    },
]


VIS_NAVIGATION_REFERENCE = [
    {
        'line': 2744,
        'pixel': 6720,
        'lon': 139.975527,
        'lat': 35.078028,
        'tolerance': 0.01,
        'nav_params': {
            'line_offset': 5513.0,
            'pixel_offset': 6688.5,
            'stepping_angle': 0.000035000004573,
            'sampling_angle': 0.000023929998861,
            'misalignment': np.array(
                [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                 [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                 [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
            ),
            'greenwich_sidereal_time': 2.468529731914041,
            'angle_between_earth_and_sun': 3.997397918405798,
            'declination_from_sat_to_sun': -0.208770861179448,
            'right_ascension_from_sat_to_sun': 3.304369304082406,
            'angle_between_sat_spin_and_z_axis': 3.149118633034304,
            'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
            'sat_position_earth_fixed_x': -32390963.148477241396904,
            'sat_position_earth_fixed_y': 27003395.381243918091059,
            'sat_position_earth_fixed_z': -228134.860164520738181,
            'nutation_precession': np.array(
                [[0.999936381496146, -0.010344758016410, -0.004496547784299],
                 [0.010344942303489, 0.999946489495557, 0.000017727054455],
                 [0.004496123789670, -0.000064242454080, 0.999989890320785]]
            ),
            'earth_flattening': 0.003352813177897,
            'earth_equatorial_radius': 6378136
        },
    },

    {
        'line': 8356,
        'pixel': 7172,
        'lon': 144.980104,
        'lat': -34.929123,
        'tolerance': 0.01,
        'nav_params': {
            'line_offset': 5513.0,
            'pixel_offset': 6688.5,
            'stepping_angle': 0.000035000004573,
            'sampling_angle': 0.000023929998861,
            'misalignment': np.array(
                [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                 [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                 [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
            ),
            'greenwich_sidereal_time': 2.530392320342610,
            'angle_between_earth_and_sun': 3.935707944858620,
            'declination_from_sat_to_sun': -0.208713576872715,
            'right_ascension_from_sat_to_sun': 3.242660398961383,
            'angle_between_sat_spin_and_z_axis': 3.149118633034304,
            'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
            'sat_position_earth_fixed_x': -32390273.633557569235563,
            'sat_position_earth_fixed_y': 27003859.543131537735462,
            'sat_position_earth_fixed_z': -210800.087734811415430,
            'nutation_precession': np.array(
                [[0.999936381432029, -0.010344763228876, -0.004496550050695],
                 [0.010344947502662, 0.999946489441823, 0.000017724053657],
                 [0.004496126086653, -0.000064239500295, 0.999989890310647]]
            ),
            'earth_flattening': 0.003352813177897,
            'earth_equatorial_radius': 6378136
        },
    },

]

NAVIGATION_REFERENCE = VIS_NAVIGATION_REFERENCE + IR_NAVIGATION_REFERENCE


class TestVISSRNavigation:
    """VISSR navigation tests."""

    def test_interpolate_prediction(self):
        """Test interpolation of orbit/attitude predictions."""
        res = nav.interpolate_prediction(
            prediction_times=np.array([1, 2, 3]),
            predicted_values=np.array([10, 20, 30]),
            desired_time=np.array([1.5, 2.5])
        )
        np.testing.assert_allclose(res, [15, 25])

    @pytest.mark.parametrize(
        'desired_time,nearest_pred_exp',
        [
            (0, [10, 20]),
            (2.5, [30, 40]),
            (5, [50, 60])
        ]
    )
    def test_get_nearest_prediction(self, desired_time, nearest_pred_exp):
        """Test getting the nearest prediction."""
        res = nav.get_nearest_prediction(
            prediction_times=np.array([1, 2, 3]),
            predicted_values=np.array([[10, 20], [30, 40], [50, 60]]),
            desired_time=desired_time
        )
        np.testing.assert_allclose(res, nearest_pred_exp)

    def test_get_observation_time(self):
        """Test getting the observation time of a given pixel."""
        spinning_rate = 100
        sampling_angle = 0.01
        num_sensors = 1
        scan_params = (spinning_rate, num_sensors, sampling_angle)
        time = nav.get_observation_time(
            point=np.array([11, 100]),
            start_time_of_scan=50000,
            scan_params=scan_params
        )
        np.testing.assert_allclose(time, 50000.0000705496871047)

    @pytest.mark.parametrize(
        'line,pixel,params,lon_exp,lat_exp,tolerance',
        [
            (ref['line'],
             ref['pixel'],
             ref['nav_params'],
             ref['lon'],
             ref['lat'],
             ref['tolerance'])
            for ref in NAVIGATION_REFERENCE
        ]
    )
    def test_get_lon_lat(self, line, pixel, params, lon_exp, lat_exp,
                         tolerance):
        """Test getting lon/lat coordinates for a given pixel."""
        nav_params = nav.NavigationParameters(**params)
        lon, lat = nav.get_lon_lat(line, pixel, nav_params)
        np.testing.assert_allclose(
            (lon, lat), (lon_exp, lat_exp), atol=tolerance
        )

    def test_nav_matrices_are_contiguous(self):
        """Test that navigation matrices are stored as C-contiguous arrays."""
        nav_params = nav.NavigationParameters(
            **NAVIGATION_REFERENCE[0]['nav_params']
        )
        assert nav_params.misalignment.flags['C_CONTIGUOUS']
        assert nav_params.nutation_precession.flags['C_CONTIGUOUS']

    def test_transform_image_coords_to_scanning_angles(self):
        """Test transformation from image coordinates to scanning angles."""
        angles = nav.transform_image_coords_to_scanning_angles(
            point=np.array([200.5, 100.5]),
            offset=np.array([101, 201]),
            sampling=np.array([0.01, 0.02])
        )
        np.testing.assert_allclose(angles, [-2, 1])

    def test_transform_scanning_angles_to_satellite_coords(self):
        """Test transformation from scanning angles to satellite coordinates."""
        transformer = nav.ScanningAnglesToSatelliteCoordsTransformer(
            misalignment=np.diag([1, 2, 3]).astype(float)
        )
        point_sat = transformer.transform(np.array([np.pi, np.pi/2]))
        np.testing.assert_allclose(point_sat, [0, 0, 3], atol=1E-12)

    def test_transform_satellite_to_earth_fixed_coords(self):
        """Test transformation from satellite to earth-fixed coordinates."""
        transformer = nav.SatelliteToEarthFixedCoordsTransformer(
            greenwich_sidereal_time=np.pi,
            sat_sun_angles=np.array([np.pi, np.pi/2]),
            earth_sun_angle=np.pi,
            spin_angles=np.array([np.pi, np.pi/2]),
            nutation_precession=np.diag([1, 2, 3]).astype(float)
        )
        res = transformer.transform(np.array([1, 2, 3], dtype=float))
        np.testing.assert_allclose(res, [-3, 1, -2])

    def test_intersect_view_vector_with_earth(self):
        """Test intersection of a view vector with the earth's surface."""
        eq_radius = 6371 * 1000
        flattening = 0.003
        intersector = nav.EarthIntersector(
            sat_pos=np.array([36000 * 1000, 0, 0], dtype=float),
            ellipsoid=np.array([eq_radius, flattening])
        )
        point = intersector.intersect(np.array([-1, 0, 0], dtype=float))
        np.testing.assert_allclose(point, [eq_radius, 0, 0])

    @pytest.mark.parametrize(
        'point_earth_fixed,point_geodetic_exp',
        [
            ([0, 0, 1], [0, 90]),
            ([0, 0, -1], [0, -90]),
            ([1, 0, 0], [0, 0]),
            ([-1, 0, 0], [180, 0]),
            ([1, 1, 1], [45, 35.426852]),
        ]
    )
    def test_transform_earth_fixed_to_geodetic_coords(
            self, point_earth_fixed, point_geodetic_exp
    ):
        """Test transformation from earth-fixed to geodetic coordinates."""
        point_geodetic = nav.transform_earth_fixed_to_geodetic_coords(
            np.array(point_earth_fixed),
            0.003
        )
        np.testing.assert_allclose(point_geodetic, point_geodetic_exp)

    def test_normalize_vector(self):
        """Test vector normalization."""
        v = np.array([1, 2, 3], dtype=float)
        normed = nav.normalize_vector(v)
        np.testing.assert_allclose(normed, v / np.sqrt(14))


# class TestImageNavigator:
#     @pytest.fixture
#     def navigator(self):
#         attitude_prediction = np.ones(33,
#                                       dtype=nav.attitude_prediction_dtype)
#         orbit_prediction = np.ones(9, dtype=nav.orbit_prediction_dtype)
#         return nav.ImageNavigator(
#             start_time_of_scan=50000,
#             line_offset=123,
#             pixel_offset=123,
#             sampling_angle=0.01,
#             stepping_angle=0.02,
#             spinning_rate=100,
#             num_sensors=1,
#             misalignment=np.diag([1, 2, 3]).astype(np.float32),
#             attitude_prediction=attitude_prediction,
#             orbit_prediction=orbit_prediction,
#             earth_flattening=0.0003,
#             earth_equatorial_radius=6378
#         )
#
#     def test_has_correct_line_offset(self, navigator):
#         assert navigator.line_offset == 123
#
#     def test_has_correct_attitude_prediction(self, navigator):
#         assert navigator.attitude_prediction.dtype == nav.attitude_prediction_dtype

