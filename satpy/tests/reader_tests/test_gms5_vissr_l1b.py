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
        'nav_params': nav.NavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=3.997397917902958,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            orbit=nav.Orbit(
                greenwich_sidereal_time=2.468529732418296,
                declination_from_sat_to_sun=-0.208770861178982,
                right_ascension_from_sat_to_sun=3.304369303579407,
                sat_position_earth_fixed_x=-32390963.148471601307392,
                sat_position_earth_fixed_y=27003395.381247851997614,
                sat_position_earth_fixed_z=-228134.860026293463307,
                nutation_precession=np.array(
                    [[0.999936381496146, -0.010344758016410, -0.004496547784299],
                     [0.010344942303489, 0.999946489495557, 0.000017727054455],
                     [0.004496123789670, -0.000064242454080, 0.999989890320785]]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                line_offset=1378.5,
                pixel_offset=1672.5,
                stepping_angle=0.000140000047395,
                sampling_angle=0.000095719995443,
                misalignment=np.array(
                    [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                     [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                     [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
                ),
                earth_flattening=0.003352813177897,
                earth_equatorial_radius=6378136.0
            ),
        )
    },
    {
        'line': 2089,
        'pixel': 1793,
        'lon': 144.996967,
        'lat': -34.959853,
        'nav_params': nav.NavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=3.935707944355762,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            orbit=nav.Orbit(
                greenwich_sidereal_time=2.530392320846865,
                declination_from_sat_to_sun=-0.208713576872247,
                right_ascension_from_sat_to_sun=3.242660398458377,
                sat_position_earth_fixed_x=-32390273.633551981300116,
                sat_position_earth_fixed_y=27003859.543135114014149,
                sat_position_earth_fixed_z=-210800.087589388160268,
                nutation_precession=np.array(
                    [[0.999936381432029, -0.010344763228876, -0.004496550050695],
                     [0.010344947502662, 0.999946489441823, 0.000017724053657],
                     [0.004496126086653, -0.000064239500295, 0.999989890310647]]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                line_offset=1378.5,
                pixel_offset=1672.5,
                stepping_angle=0.000140000047395,
                sampling_angle=0.000095719995443,
                misalignment=np.array(
                    [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                     [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                     [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
                ),
                earth_flattening=0.003352813177897,
                earth_equatorial_radius=6378136
            ),
        )
    },



    # {
    #     'line': 686,
    #     'pixel': 1680,
    #     'lon': 139.990380,
    #     'lat': 35.047056,
    #     'tolerance': 0,
    #     'nav_params': {
    #         'line_offset': 1378.5,
    #         'pixel_offset': 1672.5,
    #         'stepping_angle': 0.000140000047395,
    #         'sampling_angle': 0.000095719995443,
    #         'misalignment': np.array(
    #             [[0.999999165534973, 0.000510364072397, 0.001214201096445],
    #              [-0.000511951977387, 0.999999046325684, 0.001307720085606],
    #              [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
    #         ),
    #         'greenwich_sidereal_time': 2.468529732418296,
    #         'angle_between_earth_and_sun': 3.997397917902958,
    #         'declination_from_sat_to_sun': -0.208770861178982,
    #         'right_ascension_from_sat_to_sun': 3.304369303579407,
    #         'angle_between_sat_spin_and_z_axis': 3.149118633034304,
    #         'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
    #         'sat_position_earth_fixed_x': -32390963.148471601307392,
    #         'sat_position_earth_fixed_y': 27003395.381247851997614,
    #         'sat_position_earth_fixed_z': -228134.860026293463307,
    #         'nutation_precession': np.array(
    #             [[0.999936381496146, -0.010344758016410, -0.004496547784299],
    #              [0.010344942303489, 0.999946489495557, 0.000017727054455],
    #              [0.004496123789670, -0.000064242454080, 0.999989890320785]]
    #         ),
    #         'earth_flattening': 0.003352813177897,
    #         'earth_equatorial_radius': 6378136
    #     },
    # },
    # {
    #     'line': 2089,
    #     'pixel': 1793,
    #     'lon': 144.996967,
    #     'lat': -34.959853,
    #     'tolerance': 0,
    #     'nav_params': {
    #         'line_offset': 1378.5,
    #         'pixel_offset': 1672.5,
    #         'stepping_angle': 0.000140000047395,
    #         'sampling_angle': 0.000095719995443,
    #         'misalignment': np.array(
    #             [[0.999999165534973, 0.000510364072397, 0.001214201096445],
    #              [-0.000511951977387, 0.999999046325684, 0.001307720085606],
    #              [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
    #         ),
    #         'greenwich_sidereal_time': 2.530392320846865,
    #         'angle_between_earth_and_sun': 3.935707944355762,
    #         'declination_from_sat_to_sun': -0.208713576872247,
    #         'right_ascension_from_sat_to_sun': 3.242660398458377,
    #         'angle_between_sat_spin_and_z_axis': 3.149118633034304,
    #         'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
    #         'sat_position_earth_fixed_x': -32390273.633551981300116,
    #         'sat_position_earth_fixed_y': 27003859.543135114014149,
    #         'sat_position_earth_fixed_z': -210800.087589388160268,
    #         'nutation_precession': np.array(
    #             [[0.999936381432029, -0.010344763228876, -0.004496550050695],
    #              [0.010344947502662, 0.999946489441823, 0.000017724053657],
    #              [0.004496126086653, -0.000064239500295, 0.999989890310647]]
    #         ),
    #         'earth_flattening': 0.003352813177897,
    #         'earth_equatorial_radius': 6378136
    #     },
    # },
    # {
    #     'line': 999,
    #     'pixel': 2996,
    #     'lon': -165.023842,
    #     'lat': 20.005603,
    #     'tolerance': 0,
    #     'nav_params': {
    #         'line_offset': 1378.5,
    #         'pixel_offset': 1672.5,
    #         'stepping_angle': 0.000140000047395,
    #         'sampling_angle': 0.000095719995443,
    #         'misalignment': np.array(
    #             [[0.999999165534973, 0.000510364072397, 0.001214201096445],
    #              [-0.000511951977387, 0.999999046325684, 0.001307720085606],
    #              [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
    #         ),
    #         'greenwich_sidereal_time': 2.482331732831616,
    #         'angle_between_earth_and_sun': 3.983634620574510,
    #         'declination_from_sat_to_sun': -0.208758095943038,
    #         'right_ascension_from_sat_to_sun': 3.290601673240597,
    #         'angle_between_sat_spin_and_z_axis': 3.149118633034304,
    #         'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
    #         'sat_position_earth_fixed_x': -32390808.779549609869719,
    #         'sat_position_earth_fixed_y': 27003503.047290064394474,
    #         'sat_position_earth_fixed_z': -224351.430479845439550,
    #         'nutation_precession': np.array(
    #             [[0.999936381496146, -0.010344758016410, -0.004496547784299],
    #              [0.010344942303489, 0.999946489495557, 0.000017727054455],
    #              [0.004496123789670, -0.000064242454080, 0.999989890320785]]
    #         ),
    #         'earth_flattening': 0.003352813177897,
    #         'earth_equatorial_radius': 6378136
    #     },
    # },
    # {
    #     'line': 0,
    #     'pixel': 0,
    #     'lon': np.nan,
    #     'lat': np.nan,
    #     'tolerance': 0,
    #     'nav_params': {
    #         'line_offset': 1378.5,
    #         'pixel_offset': 1672.5,
    #         'stepping_angle': 0.000140000047395,
    #         'sampling_angle': 0.000095719995443,
    #         'misalignment': np.array(
    #             [[0.999999165534973, 0.000510364072397, 0.001214201096445],
    #              [-0.000511951977387, 0.999999046325684, 0.001307720085606],
    #              [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
    #         ),
    #         'greenwich_sidereal_time': 2.482331732831616,
    #         'angle_between_earth_and_sun': 3.983634620574510,
    #         'declination_from_sat_to_sun': -0.208758095943038,
    #         'right_ascension_from_sat_to_sun': 3.290601673240597,
    #         'angle_between_sat_spin_and_z_axis': 3.149118633034304,
    #         'angle_between_sat_spin_and_yz_plane': 0.000546042025980,
    #         'sat_position_earth_fixed_x': -32390808.779549609869719,
    #         'sat_position_earth_fixed_y': 27003503.047290064394474,
    #         'sat_position_earth_fixed_z': -224351.430479845439550,
    #         'nutation_precession': np.array(
    #             [[0.999936381496146, -0.010344758016410, -0.004496547784299],
    #              [0.010344942303489, 0.999946489495557, 0.000017727054455],
    #              [0.004496123789670, -0.000064242454080, 0.999989890320785]]
    #         ),
    #         'earth_flattening': 0.003352813177897,
    #         'earth_equatorial_radius': 6378136
    #     },
    # },
]


VIS_NAVIGATION_REFERENCE = [
    {
        'line': 2744,
        'pixel': 6720,
        'lon': 139.975527,
        'lat': 35.078028,
        'nav_params': nav.NavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=3.997397918405798,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            orbit=nav.Orbit(
                greenwich_sidereal_time=2.468529731914041,
                declination_from_sat_to_sun=-0.208770861179448,
                right_ascension_from_sat_to_sun=3.304369304082406,
                sat_position_earth_fixed_x=-32390963.148477241396904,
                sat_position_earth_fixed_y=27003395.381243918091059,
                sat_position_earth_fixed_z=-228134.860164520738181,
                nutation_precession=np.array(
                    [[0.999936381496146, -0.010344758016410, -0.004496547784299],
                     [0.010344942303489, 0.999946489495557, 0.000017727054455],
                     [0.004496123789670, -0.000064242454080, 0.999989890320785]]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                line_offset=5513.0,
                pixel_offset=6688.5,
                stepping_angle=0.000035000004573,
                sampling_angle=0.000023929998861,
                misalignment=np.array(
                    [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                     [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                     [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
                ),
                earth_flattening=0.003352813177897,
                earth_equatorial_radius=6378136
            ),
        )
    },
    {
        'line': 8356,
        'pixel': 7172,
        'lon': 144.980104,
        'lat': -34.929123,
        'nav_params': nav.NavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=3.935707944858620,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            orbit=nav.Orbit(
                greenwich_sidereal_time=2.530392320342610,
                declination_from_sat_to_sun=-0.208713576872715,
                right_ascension_from_sat_to_sun=3.242660398961383,
                sat_position_earth_fixed_x=-32390273.633557569235563,
                sat_position_earth_fixed_y=27003859.543131537735462,
                sat_position_earth_fixed_z=-210800.087734811415430,
                nutation_precession=np.array(
                    [[0.999936381432029, -0.010344763228876, -0.004496550050695],
                     [0.010344947502662, 0.999946489441823, 0.000017724053657],
                     [0.004496126086653, -0.000064239500295, 0.999989890310647]]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                line_offset=5513.0,
                pixel_offset=6688.5,
                stepping_angle=0.000035000004573,
                sampling_angle=0.000023929998861,
                misalignment=np.array(
                    [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                     [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                     [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
                ),
                earth_flattening=0.003352813177897,
                earth_equatorial_radius=6378136
            ),
        )
    },
]

NAVIGATION_REFERENCE = VIS_NAVIGATION_REFERENCE + IR_NAVIGATION_REFERENCE


"""
    {
        'line': ,
        'pixel': ,
        'lon': ,
        'lat': ,
        'tolerance': ,
        'nav_params': nav.NavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=,
                angle_between_sat_spin_and_z_axis=,
                angle_between_sat_spin_and_yz_plane=,
            ),
            orbit=nav.Orbit(
                greenwich_sidereal_time=,
                declination_from_sat_to_sun=,
                right_ascension_from_sat_to_sun=,
                sat_position_earth_fixed_x=,
                sat_position_earth_fixed_y=,
                sat_position_earth_fixed_z=,
                nutation_precession=np.array(
                    [[],
                     [],
                     []]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                line_offset=,
                pixel_offset=,
                stepping_angle=,
                sampling_angle=,
                misalignment=np.array(
                    [[],
                     [],
                     []]
                ),
                earth_flattening=,
                earth_equatorial_radius=
            ),
        )
    },

"""

class TestSinglePixelNavigation:
    """Test navigation of a single pixel."""

    @pytest.mark.parametrize(
        'line,pixel,nav_params,lon_exp,lat_exp',
        [
            (ref['line'],
             ref['pixel'],
             ref['nav_params'],
             ref['lon'],
             ref['lat'])
            for ref in NAVIGATION_REFERENCE
        ]
    )
    def test_get_lon_lat(self, line, pixel, nav_params, lon_exp, lat_exp):
        """Test getting lon/lat coordinates for a given pixel."""
        lon, lat = nav.get_lon_lat(line, pixel, nav_params)
        np.testing.assert_allclose((lon, lat), (lon_exp, lat_exp))

    def test_nav_matrices_are_contiguous(self):
        """Test that navigation matrices are stored as C-contiguous arrays."""
        nav_params = NAVIGATION_REFERENCE[0]['nav_params']
        assert nav_params.proj_params.misalignment.flags['C_CONTIGUOUS']
        assert nav_params.orbit.nutation_precession.flags['C_CONTIGUOUS']

    def test_transform_image_coords_to_scanning_angles(self):
        """Test transformation from image coordinates to scanning angles."""
        angles = nav.transform_image_coords_to_scanning_angles(
            point=np.array([199, 99]),
            offset=np.array([100, 200]),
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


class TestPredictionInterpolation:
    """Test interpolation of orbit and attitude predictions."""

    @pytest.mark.parametrize(
        'obs_time,expected',
        [
            (-1, np.nan),
            (1.5, 2.5),
            (5, np.nan)
        ]
    )
    def test_interpolate_cont(self, obs_time, expected):
        prediction_times = np.array([0, 1, 2, 3])
        predicted_values = np.array([1, 2, 3, 4])
        res = nav.interpolate_cont(
            obs_time,
            prediction_times,
            predicted_values
        )
        np.testing.assert_allclose(res, expected)

    @pytest.mark.parametrize(
        'obs_time,expected',
        [
            (-1, np.nan),
            (1.5, 0.75*np.pi),
            (2.5, np.pi),
            (3.5, -0.75*np.pi),
            (5, np.nan),
        ]
    )
    def test_interpolate_angles(self, obs_time, expected):
        prediction_times = np.array([0, 1, 2, 3, 4])
        predicted_angles = np.array([0, np.pi/2, np.pi, -np.pi, -np.pi/2])
        res = nav.interpolate_angles(
            obs_time,
            prediction_times,
            predicted_angles
        )
        np.testing.assert_allclose(res, expected)

    @pytest.mark.parametrize(
        'obs_time,expected',
        [
            (-1, np.nan * np.ones((2, 2))),
            (1.5, [[1, 0], [0, 2]]),
            (3, np.nan * np.ones((2, 2))),
        ]
    )
    def test_interpolate_nearest(self, obs_time, expected):
        prediction_times = np.array([0, 1, 2])
        predicted_angles = np.array([
            np.zeros((2, 2)),
            np.diag((1, 2)),
            np.zeros((2, 2))
        ])
        res = nav.interpolate_nearest(
            obs_time,
            prediction_times,
            predicted_angles
        )
        np.testing.assert_allclose(res, expected)

    def test_interpolate_orbit_prediction(self, obs_time, orbit_prediction, orbit_expected):
        orbit = orbit_prediction.interpolate(obs_time)
        self.assert_orbit_close(orbit, orbit_expected)

    def test_interpolate_attitude_prediction(self, obs_time, attitude_prediction, attitude_expected):
        attitude = attitude_prediction.interpolate(obs_time)
        self.assert_attitude_close(attitude, attitude_expected)

    def test_interpolate_prediction(self, obs_time, proj_params, attitude_prediction, orbit_prediction, nav_params_expected):
        interpolator = nav.PredictionInterpolator(
            proj_params=proj_params,
            attitude_prediction=attitude_prediction,
            orbit_prediction=orbit_prediction
        )
        nav_params = interpolator.interpolate(obs_time)
        self.assert_nav_params_close(nav_params, nav_params_expected)

    @pytest.fixture
    def obs_time(self):
        return 2.5

    @pytest.fixture
    def orbit_prediction(self):
        return nav.OrbitPrediction(
            prediction_times=np.array([1.0, 2.0, 3.0, 4.0]),
            greenwich_sidereal_time=np.array([0.0, 1.0, 2.0, 3.0]),
            declination_from_sat_to_sun=np.array([0.1, 1.1, 2.1, 3.1]),
            right_ascension_from_sat_to_sun=np.array([0.2, 1.2, 2.2, 3.2]),
            sat_position_earth_fixed_x=np.array([0.3, 1.3, 2.3, 3.3]),
            sat_position_earth_fixed_y=np.array([0.4, 1.4, 2.4, 3.4]),
            sat_position_earth_fixed_z=np.array([0.5, 1.5, 2.5, 3.5]),
            nutation_precession=np.array(
                [
                    0.6*np.identity(3),
                    1.6*np.identity(3),
                    2.6*np.identity(3),
                    3.6*np.identity(3)
                ]
            )
        )

    @pytest.fixture
    def orbit_expected(self):
        return nav.Orbit(
            greenwich_sidereal_time=1.5,
            declination_from_sat_to_sun=1.6,
            right_ascension_from_sat_to_sun=1.7,
            sat_position_earth_fixed_x=1.8,
            sat_position_earth_fixed_y=1.9,
            sat_position_earth_fixed_z=2.0,
            nutation_precession=1.6 * np.identity(3)
        )

    @pytest.fixture
    def attitude_prediction(self):
        return nav.AttitudePrediction(
            prediction_times=np.array([1.0, 2.0, 3.0]),
            angle_between_earth_and_sun=np.array([0.0, 1.0, 2.0]),
            angle_between_sat_spin_and_z_axis=np.array([0.1, 1.1, 2.1]),
            angle_between_sat_spin_and_yz_plane=np.array([0.2, 1.2, 2.2]),
        )

    @pytest.fixture
    def attitude_expected(self):
        return nav.Attitude(
            angle_between_earth_and_sun=1.5,
            angle_between_sat_spin_and_z_axis=1.6,
            angle_between_sat_spin_and_yz_plane=1.7,
        )

    @pytest.fixture
    def proj_params(self):
        return nav.ProjectionParameters(
            line_offset=1378.5,
            pixel_offset=1672.5,
            stepping_angle=0.000140000047395,
            sampling_angle=0.000095719995443,
            misalignment=np.identity(3).astype(np.float64),
            earth_flattening=0.003352813177897,
            earth_equatorial_radius=6378136
        )

    @pytest.fixture
    def nav_params_expected(self, attitude_expected, orbit_expected, proj_params):
        return nav.NavigationParameters(
            attitude_expected, orbit_expected, proj_params
        )

    def assert_orbit_close(self, a, b):
        """Assert that two Orbit instances are close.

        This would probably make more sense in the Orbit class. However,
        numba doesn't support np.allclose, yet.
        """
        attrs = [
            'greenwich_sidereal_time',
            'declination_from_sat_to_sun',
            'right_ascension_from_sat_to_sun',
            'sat_position_earth_fixed_x',
            'sat_position_earth_fixed_y',
            'sat_position_earth_fixed_z',
            'nutation_precession',
        ]
        self._assert_attrs_close(a, b, attrs, 'Orbit')

    def assert_attitude_close(self, a, b):
        """Assert that two Attitude instances are close.

        This would probably make more sense in the Attitude class. However,
        numba doesn't support np.allclose, yet.
        """
        attrs = [
            'angle_between_earth_and_sun',
            'angle_between_sat_spin_and_z_axis',
            'angle_between_sat_spin_and_yz_plane'
        ]
        self._assert_attrs_close(a, b, attrs, 'Attitude')

    def assert_proj_params_close(self, a, b):
        """Assert that two ProjectionParameters instances are close.

        This would probably make more sense in the Attitude class. However,
        numba doesn't support np.allclose, yet.
        """
        attrs = [
            'line_offset',
            'pixel_offset',
            'stepping_angle',
            'sampling_angle',
            'misalignment',
            'earth_flattening',
            'earth_equatorial_radius',
        ]
        self._assert_attrs_close(a, b, attrs, 'ProjectionParameters')

    def assert_nav_params_close(self, a, b):
        self.assert_attitude_close(a.attitude, b.attitude)
        self.assert_orbit_close(a.orbit, b.orbit)
        self.assert_proj_params_close(a.proj_params, b.proj_params)

    @staticmethod
    def _assert_attrs_close(a, b, attrs, desc):
        for attr in attrs:
            np.testing.assert_allclose(
                getattr(a, attr),
                getattr(b, attr),
                err_msg='{} attribute {} differs'.format(desc, attr)
            )



def test_get_observation_time():
    scan_params = nav.ScanningParameters(
        start_time_of_scan=50000.0,
        spinning_rate=100,
        num_sensors=1,
        sampling_angle=0.01
    )
    point = np.array([11, 100])
    obs_time = nav.get_observation_time(point, scan_params)
    np.testing.assert_allclose(obs_time, 50000.0000705496871047)
