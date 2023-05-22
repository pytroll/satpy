"""Unit tests for GMS-5 VISSR reader."""

import numpy as np
import pytest
from unittest import mock

import satpy.readers.gms5_vissr_navigation as nav
import satpy.readers.gms5_vissr_l1b as vissr
from satpy.tests.utils import make_dataid


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
        'nav_params': (
            nav.Attitude(
                angle_between_earth_and_sun=3.997397917902958,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            nav.Orbit(
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
            nav.ProjectionParameters(
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
        'nav_params': (
            nav.Attitude(
                angle_between_earth_and_sun=3.935707944355762,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            nav.Orbit(
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
            nav.ProjectionParameters(
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
    }
]


VIS_NAVIGATION_REFERENCE = [
    {
        'line': 2744,
        'pixel': 6720,
        'lon': 139.975527,
        'lat': 35.078028,
        'nav_params': (
            nav.Attitude(
                angle_between_earth_and_sun=3.997397918405798,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            nav.Orbit(
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
            nav.ProjectionParameters(
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
        'nav_params': (
            nav.Attitude(
                angle_between_earth_and_sun=3.935707944858620,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            nav.Orbit(
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
            nav.ProjectionParameters(
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



class TestSinglePixelNavigation:
    """Test navigation of a single pixel."""

    @pytest.mark.parametrize(
        'point,nav_params,expected',
        [
            ((ref['line'], ref['pixel']),
             ref['nav_params'],
             (ref['lon'], ref['lat']))
            for ref in NAVIGATION_REFERENCE
        ]
    )
    def test_get_lon_lat(self, point, nav_params, expected):
        """Test getting lon/lat coordinates for a given pixel."""
        lon, lat = nav.get_lon_lat(point, nav_params)
        np.testing.assert_allclose((lon, lat), expected)

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
        scanning_angles = np.array([np.pi, np.pi / 2])
        misalignment = np.diag([1, 2, 3]).astype(float)
        point_sat = nav.transform_scanning_angles_to_satellite_coords(
            scanning_angles, misalignment
        )
        np.testing.assert_allclose(point_sat, [0, 0, 3], atol=1E-12)

    def test_transform_satellite_to_earth_fixed_coords(self):
        """Test transformation from satellite to earth-fixed coordinates."""
        point_sat = np.array([1, 2, 3], dtype=float)
        greenwich_sidereal_time = np.pi
        sat_sun_angles = np.array([np.pi, np.pi / 2])
        earth_sun_angle = np.pi
        spin_angles = np.array([np.pi, np.pi / 2])
        nutation_precession = np.diag([1, 2, 3]).astype(float)
        res = nav.transform_satellite_to_earth_fixed_coords(
            point_sat,
            greenwich_sidereal_time,
            sat_sun_angles,
            earth_sun_angle,
            spin_angles,
            nutation_precession
        )
        np.testing.assert_allclose(res, [-3, 1, -2])

    def test_intersect_view_vector_with_earth(self):
        """Test intersection of a view vector with the earth's surface."""
        view_vector = np.array([-1, 0, 0], dtype=float)
        sat_pos = np.array([36000 * 1000, 0, 0], dtype=float)
        eq_radius = 6371 * 1000
        flattening = 0.003
        ellipsoid = np.array([eq_radius, flattening])
        point = nav.intersect_with_earth(
            view_vector, sat_pos, ellipsoid
        )
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


class TestImageNavigation:
    def test_get_lons_lats(
            self,
            scan_params,
            attitude_prediction,
            orbit_prediction,
            proj_params
    ):
        lons_exp = [[-114.56923, -112.096837, -109.559702],
                    [8.33221, 8.793893, 9.22339],
                    [15.918476, 16.268354, 16.6332]]
        lats_exp = [[-23.078721, -24.629845, -26.133314],
                    [-42.513409, -39.790231, -37.06392],
                    [3.342834, 6.07043, 8.795932]]
        lons, lats = nav.get_lons_lats(
            lines=np.array([1000, 1500, 2000]),
            pixels=np.array([1000, 1500, 2000]),
            static_params=(scan_params, proj_params),
            predicted_params=(attitude_prediction, orbit_prediction)
        )
        np.testing.assert_allclose(lons, lons_exp)
        np.testing.assert_allclose(lats, lats_exp)


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
    def test_interpolate_continuous(self, obs_time, expected):
        prediction_times = np.array([0, 1, 2, 3])
        predicted_values = np.array([1, 2, 3, 4])
        res = nav.interpolate_continuous(
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
            (2.5, -np.pi),
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
        orbit = nav.interpolate_orbit_prediction(
            orbit_prediction, obs_time
        )
        assert_namedtuple_close(orbit, orbit_expected)

    def test_interpolate_attitude_prediction(self, obs_time, attitude_prediction, attitude_expected):
        attitude = nav.interpolate_attitude_prediction(
            attitude_prediction, obs_time
        )
        assert_namedtuple_close(attitude, attitude_expected)

    @pytest.fixture
    def obs_time(self):
        return 2.5

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
    def attitude_expected(self):
        return nav.Attitude(
            angle_between_earth_and_sun=1.5,
            angle_between_sat_spin_and_z_axis=1.6,
            angle_between_sat_spin_and_yz_plane=1.7,
        )

    @pytest.fixture
    def nav_params_expected(self, attitude_expected, orbit_expected, proj_params):
        return nav.NavigationParameters(
            attitude_expected, orbit_expected, proj_params
        )


@pytest.fixture
def sampling_angle():
    return 0.000095719995443


@pytest.fixture
def scan_params(sampling_angle):
    return nav.ScanningParameters(
        start_time_of_scan=0,
        spinning_rate=0.5,
        num_sensors=1,
        sampling_angle=sampling_angle
    )


@pytest.fixture
def attitude_prediction():
    return nav.AttitudePrediction(
        prediction_times=np.array([1.0, 2.0, 3.0]),
        angle_between_earth_and_sun=np.array([0.0, 1.0, 2.0]),
        angle_between_sat_spin_and_z_axis=np.array([0.1, 1.1, 2.1]),
        angle_between_sat_spin_and_yz_plane=np.array([0.2, 1.2, 2.2]),
    )


@pytest.fixture
def orbit_prediction():
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
def proj_params(sampling_angle):
    return nav.ProjectionParameters(
        line_offset=1378.5,
        pixel_offset=1672.5,
        stepping_angle=0.000140000047395,
        sampling_angle=sampling_angle,
        misalignment=np.identity(3).astype(np.float64),
        earth_flattening=0.003352813177897,
        earth_equatorial_radius=6378136
    )


@pytest.mark.parametrize(
    'angles',
    [
        (np.array([0, np.pi/2, np.pi, -np.pi, -np.pi/2])),
        (np.array([0, 0.78539816, 1.57079633, 5.49778714, 6.28318531]))
    ]
)
def test_unwrap(angles):
    np.testing.assert_allclose(nav.unwrap(angles), np.unwrap(angles))


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


class TestFileHandler:
    channel_types = {
        'VIS': 'VIS',
        'IR1': 'IR',
        'IR2': 'IR',
        'IR3': 'IR'
    }

    def test_dataset_navigation(self, file_handler, dataset_id, lons_lats_exp):
        lons_exp, lats_exp = lons_lats_exp
        dataset = file_handler.get_dataset(dataset_id, {})
        lons = dataset.coords['lon']
        lats = dataset.coords['lat']
        np.testing.assert_allclose(lons, lons_exp, atol=1E-6)
        np.testing.assert_allclose(lats, lats_exp, atol=1E-6)

    @pytest.fixture
    def file_handler(self, header, dataset_id, image_data):
        channel_type = self.channel_types[dataset_id['name']]
        with mock.patch('satpy.readers.gms5_vissr_l1b.GMS5VISSRFileHandler._read_header') as _read_header, \
                mock.patch('satpy.readers.gms5_vissr_l1b.np.memmap') as memmap:
            _read_header.return_value = header, channel_type
            memmap.return_value = image_data
            fh = vissr.GMS5VISSRFileHandler('foo', {'foo': 'bar'}, {'foo': 'bar'})
            # Yield instead of return, to make the memmap mock succeed.
            # See https://stackoverflow.com/a/59045506/5703449
            yield fh

    @pytest.fixture(params=['VIS', 'IR1'])
    def dataset_id(self, request):
        return make_dataid(name=request.param)

    @pytest.fixture
    def image_data(self, dataset_id):
        """Get fake image data.

        Data type:

        ((line number, timestamp), (data1, data2))

        VIS channel:

        pix = [6688, 6688, 6689, 6689]
        lin = [2744, 8356, 2744, 8356]

        IR1 channel:

        pix = [1672, 1672, 1673, 1673]
        lin = [686, 2089, 686, 2089]
        """
        line_control_word = np.dtype([
            ('line_number', vissr.I4),
            ('scan_time', vissr.R8),
        ])
        dtype = np.dtype([('LCW', line_control_word),
                          ('image_data', vissr.U1, (2,))])
        if dataset_id['name'] == 'IR1':
            return np.array([((686, 50000), (1, 2)), ((2089, 50000), (3, 4))], dtype=dtype)
        elif dataset_id['name'] == 'VIS':
            return np.array([((2744, 50000), (1, 2)), ((8356, 50000), (3, 4))], dtype=dtype)
        raise NotImplementedError

    @pytest.fixture
    def header(self, control_block, image_params):
        return {
            'control_block': control_block,
            'image_parameters': image_params
        }

    @pytest.fixture
    def control_block(self):
        return {
            'available_block_size_of_image_data': 2
        }

    @pytest.fixture
    def image_params(
            self, mode, coordinate_conversion, attitude_prediction,
            orbit_prediction, vis_calibration, ir1_calibration, ir2_calibration,
            wv_calibration, simple_coordinate_conversion_table
    ):
        return {
            'mode': mode,
            'coordinate_conversion': coordinate_conversion,
            'attitude_prediction': attitude_prediction,
            'orbit_prediction': orbit_prediction,
            'vis_calibration': vis_calibration,
            'ir1_calibration': ir1_calibration,
            'ir2_calibration': ir2_calibration,
            'wv_calibration': wv_calibration,
            'simple_coordinate_conversion_table': simple_coordinate_conversion_table
        }

    @pytest.fixture
    def mode(self, ir_frame_parameters, vis_frame_parameters):
        return {
            'satellite_name': b'GMS-5       ',
            'spin_rate': 99.21774,
            'ir_frame_parameters': ir_frame_parameters,
            'vis_frame_parameters': vis_frame_parameters
        }

    @pytest.fixture
    def ir_frame_parameters(self):
        return {'number_of_lines': 2, 'number_of_pixels': 2}

    @pytest.fixture
    def vis_frame_parameters(self):
        return {'number_of_lines': 2, 'number_of_pixels': 2}

    @pytest.fixture
    def coordinate_conversion(self):
        """Provide parameters for coordinate conversions.

        Reference coordinates were computed near the central column. Adjust
        pixel offset so that the first column is at the image center. This has
        the advantage that we can test with very small 2x2 images.
        """
        return {
            'central_line_number_of_vissr_frame': {
                'IR1': 1378.5,
                'IR2': 1378.7,
                'VIS': 5513.0,
                'WV': 1379.1001
            },
            'central_pixel_number_of_vissr_frame': {
                'IR1': 0.5,  # instead of 1672.5
                'IR2': 1672.5,
                'VIS': 0.5,  # instead of 6688.5
                'WV': 1672.5
            },
            'pixel_difference_of_vissr_center_from_normal_position': {
                'IR1': 0.0,
                'IR2': 0.0,
                'VIS': 0.0,
                'WV': 0.0
            },
            'scheduled_observation_time': 50130.979089568464,
            'number_of_sensor_elements': {
                'IR1': 1.0,
                'IR2': 1.0,
                'VIS': 4.0,
                'WV': 1.0
            },
            'sampling_angle_along_pixel': {
                'IR1': 9.5719995e-05,
                'IR2': 9.5719995e-05,
                'VIS': 2.3929999e-05,
                'WV': 9.5719995e-05
            },
            'stepping_angle_along_line': {
                'IR1': 0.00014000005,
                'IR2': 0.00014000005,
                'VIS': 3.5000005e-05,
                'WV': 0.00014000005
            },
            'matrix_of_misalignment': np.array(
                [[9.9999917e-01, -5.1195198e-04, -1.2135329e-03],
                 [5.1036407e-04, 9.9999905e-01, -1.3083406e-03],
                 [1.2142011e-03, 1.3077201e-03, 9.9999845e-01]],
                dtype=np.float32
            ),
            'parameters': {
                'equatorial_radius': 6377397.0,
                'oblateness_of_earth': 0.003342773,
            }
        }

    @pytest.fixture
    def attitude_prediction(self):
        return {
            'data': np.array([
                (50130.93055556, (19960217, 222000), 3.14911863, 0.00054604, 4.3324597 , 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.93402778, (19960217, 222500), 3.14911863, 0.00054604, 4.31064812, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.9375    , (19960217, 223000), 3.14911863, 0.00054604, 4.28883633, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.94097222, (19960217, 223500), 3.14911863, 0.00054604, 4.26702432, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.94444444, (19960217, 224000), 3.14911863, 0.00054604, 4.2452121 , 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.94791667, (19960217, 224500), 3.14911863, 0.00054604, 4.22339966, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.95138889, (19960217, 225000), 3.14911863, 0.00054604, 4.201587  , 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.95486111, (19960217, 225500), 3.14911863, 0.00054604, 4.17977411, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.95833333, (19960217, 230000), 3.14911863, 0.00054604, 4.157961  , 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.96180556, (19960217, 230500), 3.14911863, 0.00054604, 4.13614765, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.96527778, (19960217, 231000), 3.14911863, 0.00054604, 4.11433408, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.96875   , (19960217, 231500), 3.14911863, 0.00054604, 4.09252027, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.97222222, (19960217, 232000), 3.14911863, 0.00054604, 4.07070622, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.97569444, (19960217, 232500), 3.14911863, 0.00054604, 4.04889193, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.97916667, (19960217, 233000), 3.14911863, 0.00054604, 4.02707741, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.98263889, (19960217, 233500), 3.14911863, 0.00054604, 4.00526265, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.98611111, (19960217, 234000), 3.14911863, 0.00054604, 3.98344765, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.98958333, (19960217, 234500), 3.14911863, 0.00054604, 3.96163241, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.99305556, (19960217, 235000), 3.14911863, 0.00054604, 3.93981692, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50130.99652778, (19960217, 235500), 3.14911863, 0.00054604, 3.9180012 , 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.        , (19960218,      0), 3.14911863, 0.00054604, 3.89618523, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.00347222, (19960218,    500), 3.14911863, 0.00054604, 3.87436903, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.00694444, (19960218,   1000), 3.14911863, 0.00054604, 3.85255258, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.01041667, (19960218,   1500), 3.14911863, 0.00054604, 3.8307359 , 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.01388889, (19960218,   2000), 3.14911863, 0.00054604, 3.80891898, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.01736111, (19960218,   2500), 3.14911863, 0.00054604, 3.78710182, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.02083333, (19960218,   3000), 3.14911863, 0.00054604, 3.76528442, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.02430556, (19960218,   3500), 3.14911863, 0.00054604, 3.74346679, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.02777778, (19960218,   4000), 3.14911863, 0.00054604, 3.72164893, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.03125   , (19960218,   4500), 3.14911863, 0.00054604, 3.69983084, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.03472222, (19960218,   5000), 3.14911863, 0.00054604, 3.67801252, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.03819444, (19960218,   5500), 3.14911863, 0.00054604, 3.65619398, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0),
                (50131.04166667, (19960218,  10000), 3.14911863, 0.00054604, 3.63437521, 99.21774527, 0.97415452, -1.56984055, 0., 0, 0)],
                dtype=vissr.ATTITUDE_PREDICTION_DATA
            ),
        }

    @pytest.fixture
    def orbit_prediction(self):
        return {
            'data': np.array([
                (50130.96180556, (960217, 230500), [  2247604.14185506, -42110997.39399951,   -276688.79765022], [3069.77904265,  164.12584895,    3.65437628], [-32392525.09983424,  27002204.93121811,   -263873.25702763], [ 0.81859376,  0.6760037 , 17.44588753], 133.46391815, (330.12326803, -12.19424863), (197.27884747, -11.96904141), [[ 9.99936382e-01,  1.03449318e-02,  4.49611916e-03], [-1.03447475e-02,  9.99946490e-01, -6.42483646e-05], [-4.49654321e-03,  1.77330598e-05,  9.99989890e-01]], [ 2.46885475e+08, -2.07840219e+08, -7.66028692e+07], (-0.35887085, 140.18562594, 35793706.31768975), 0, 0),
                (50130.96527778, (960217, 231000), [  3167927.33749398, -42051692.51095297,   -275526.52514815], [3065.46435995,  231.22434208,    4.09379482], [-32392279.4626506 ,  27002405.27592725,   -258576.96255205], [ 0.81939962,  0.66017389, 17.86159393], 134.71734048, (330.12643276, -12.19310271), (196.02858456, -11.9678881 ), [[ 9.99936382e-01,  1.03449336e-02,  4.49611993e-03], [-1.03447493e-02,  9.99946490e-01, -6.42473793e-05], [-4.49654398e-03,  1.77320586e-05,  9.99989890e-01]], [ 2.46204142e+08, -2.07689897e+08, -7.65268207e+07], (-0.35166851, 140.18520316, 35793613.0815237 ), 0, 0),
                (50130.96875   , (960217, 231500), [  4086736.12968183, -41972273.80964861,   -274232.7185828 ], [3059.68341675,  298.21262775,    4.53123515], [-32392033.65156128,  27002600.83510851,   -253157.23498394], [ 0.81975174,  0.6441    , 18.26873686], 135.97076281, (330.12959087, -12.19195587), (194.77831505, -11.96673388), [[ 9.99936382e-01,  1.03449353e-02,  4.49612071e-03], [-1.03447510e-02,  9.99946490e-01, -6.42463940e-05], [-4.49654474e-03,  1.77310575e-05,  9.99989890e-01]], [ 2.45524133e+08, -2.07559497e+08, -7.64508451e+07], (-0.3442983 , 140.18478523, 35793516.57370046), 0, 0),
                (50130.97222222, (960217, 232000), [  5003591.03339227, -41872779.15809826,   -272808.0027587 ], [3052.43895532,  365.05867777,    4.9664885 ], [-32391787.80234722,  27002791.53735474,   -247616.67261456], [ 0.81965461,  0.62779672, 18.66712192], 137.22418515, (330.13274246, -12.19080808), (193.52803902, -11.9655787 ), [[ 9.99936382e-01,  1.03449371e-02,  4.49612148e-03], [-1.03447528e-02,  9.99946490e-01, -6.42454089e-05], [-4.49654551e-03,  1.77300565e-05,  9.99989890e-01]], [ 2.44845888e+08, -2.07448982e+08, -7.63749418e+07], (-0.33676374, 140.18437233, 35793416.91561355), 0, 0),
                (50130.97569444, (960217, 232500), [  5918053.49286455, -41753256.02295399,   -271253.06495935], [3043.73441705,  431.73053079,    5.39934712], [-32391542.0492856 ,  27002977.3157848 ,   -241957.93142027], [ 0.81911313,  0.61127876, 19.05655891], 138.47760748, (330.13588763, -12.1896593 ), (192.27775657, -11.96442254), [[ 9.99936382e-01,  1.03449388e-02,  4.49612225e-03], [-1.03447545e-02,  9.99946490e-01, -6.42444238e-05], [-4.49654627e-03,  1.77290557e-05,  9.99989890e-01]], [ 2.44169846e+08, -2.07358303e+08, -7.62991102e+07], (-0.32906846, 140.18396465, 35793314.23041636), 0, 0),
                (50130.97916667, (960217, 233000), [  6829686.08751574, -41613761.44760592,   -269568.65462124], [3033.5739409 ,  498.19630731,    5.82960444], [-32391296.52466749,  27003158.10847847,   -236183.72381214], [ 0.81813262,  0.59456087, 19.43686189], 139.73102981, (330.1390265 , -12.18850951), (191.02746783, -11.96326537), [[ 9.99936382e-01,  1.03449406e-02,  4.49612302e-03], [-1.03447563e-02,  9.99946490e-01, -6.42434389e-05], [-4.49654703e-03,  1.77280550e-05,  9.99989890e-01]], [ 2.43496443e+08, -2.07287406e+08, -7.62233495e+07], (-0.32121612, 140.18356238, 35793208.6428103 ), 0, 0),
                (50130.98263889, (960217, 233500), [  7738052.74476409, -41454362.02480648,   -267755.58296603], [3021.96236148,  564.42422513,    6.25705512], [-32391051.35918404,  27003333.85786499,   -230296.81731314], [ 0.81671881,  0.57765777, 19.80784932], 140.98445214, (330.14215916, -12.18735869), (189.77717289, -11.96210717), [[ 9.99936381e-01,  1.03449423e-02,  4.49612379e-03], [-1.03447580e-02,  9.99946489e-01, -6.42424541e-05], [-4.49654778e-03,  1.77270545e-05,  9.99989890e-01]], [ 2.42826115e+08, -2.07236222e+08, -7.61476592e+07], (-0.3132105 , 140.18316567, 35793100.27882991), 0, 0),
                (50130.98611111, (960217, 234000), [  8642718.9445816 , -41275133.86582235,   -265814.72261683], [3008.90520686,  630.38261431,    6.68149519], [-32390806.68247503,  27003504.50991426,   -224300.03325666], [ 0.81487783,  0.56058415, 20.16934411], 142.23787447, (330.14528573, -12.18620679), (188.52687186, -11.9609479 ), [[ 9.99936381e-01,  1.03449440e-02,  4.49612456e-03], [-1.03447598e-02,  9.99946489e-01, -6.42414694e-05], [-4.49654854e-03,  1.77260540e-05,  9.99989890e-01]], [ 2.42159297e+08, -2.07204676e+08, -7.60720382e+07], (-0.30505542, 140.18277471, 35792989.2656269 ), 0, 0),
                (50130.98958333, (960217, 234500), [  9543251.93095296, -41076162.56379041,   -263747.00717057], [2994.40869593,  696.03993248,    7.10272213], [-32390562.62077149,  27003670.01680953,   -218196.24541058], [ 0.81261619,  0.54335463, 20.52117372], 143.4912968 , (330.14840632, -12.18505381), (187.27656486, -11.95978754), [[ 9.99936381e-01,  1.03449458e-02,  4.49612532e-03], [-1.03447615e-02,  9.99946489e-01, -6.42404848e-05], [-4.49654930e-03,  1.77250538e-05,  9.99989890e-01]], [ 2.41496422e+08, -2.07192684e+08, -7.59964859e+07], (-0.29675479, 140.18238966, 35792875.73125207), 0, 0),
                (50130.99305556, (960217, 235000), [ 10439220.91492008, -40857543.15396438,   -261553.43075696], [2978.47973561,  761.36477969,    7.52053495], [-32390319.30020279,  27003830.33282405,   -211988.37862591], [ 0.80994076,  0.52598377, 20.86317023], 144.74471913, (330.15152105, -12.1838997 ), (186.026252  , -11.95862606), [[ 9.99936381e-01,  1.03449475e-02,  4.49612609e-03], [-1.03447632e-02,  9.99946489e-01, -6.42395003e-05], [-4.49655005e-03,  1.77240537e-05,  9.99989890e-01]], [ 2.40837919e+08, -2.07200148e+08, -7.59210011e+07], (-0.28831259, 140.18201066, 35792759.80443729), 0, 0),
                (50130.99652778, (960217, 235500), [ 11330197.2840407 , -40619380.06793167,   -259235.04755252], [2961.12591755,  826.32591367,    7.93473432], [-32390076.84311398,  27003985.41857829,   -205679.40741202], [ 0.80685878,  0.50848599, 21.19517045], 145.99814147, (330.15463004, -12.18274445), (184.77593341, -11.95746344), [[ 9.99936381e-01,  1.03449492e-02,  4.49612685e-03], [-1.03447650e-02,  9.99946489e-01, -6.42385159e-05], [-4.49655080e-03,  1.77230537e-05,  9.99989890e-01]], [ 2.40184218e+08, -2.07226967e+08, -7.58455830e+07], (-0.27973286, 140.18163787, 35792641.6143761 ), 0, 0),
                (50131.        , (960218,      0), [ 12215754.80493221, -40361787.08463053,   -256792.97127933], [2942.35551459,  890.89226454,    8.34512262], [-32389835.37113104,  27004135.23720251,   -199272.35452792], [ 0.8033778 ,  0.49087558, 21.51701595], 147.2515638 , (330.15773341, -12.18158803), (183.5256092 , -11.95629965), [[ 9.99936381e-01,  1.03449510e-02,  4.49612761e-03], [-1.03447667e-02,  9.99946489e-01, -6.42375317e-05], [-4.49655155e-03,  1.77220539e-05,  9.99989890e-01]], [ 2.39535744e+08, -2.07273025e+08, -7.57702305e+07], (-0.2710197 , 140.18127143, 35792521.29050537), 0, 0),
                (50131.00347222, (960218,    500), [ 13095469.82708225, -40084887.27645436,   -254228.37467049], [2922.17747695,  955.03294974,    8.75150409], [-32389595.00191828,  27004279.7580633 ,   -192770.28953487], [ 0.79950572,  0.47316669, 21.82855319], 148.50498613, (330.16083128, -12.18043041), (182.27527951, -11.95513466), [[ 9.99936381e-01,  1.03449527e-02,  4.49612837e-03], [-1.03447684e-02,  9.99946489e-01, -6.42365476e-05], [-4.49655230e-03,  1.77210542e-05,  9.99989890e-01]], [ 2.38892921e+08, -2.07338200e+08, -7.56949425e+07], (-0.26217728, 140.18091148, 35792398.96228714), 0, 0),
                (50131.00694444, (960218,   1000), [ 13968921.48773305, -39788812.95011112,   -251542.48890031], [2900.60142795, 1018.71728887,    9.15368488], [-32389355.85220329,  27004418.95297137,   -186176.32730922], [ 0.79525074,  0.45537327, 22.12963356], 149.75840846, (330.16392379, -12.17927157), (181.02494445, -11.95396845), [[ 9.99936381e-01,  1.03449544e-02,  4.49612913e-03], [-1.03447701e-02,  9.99946489e-01, -6.42355636e-05], [-4.49655305e-03,  1.77200547e-05,  9.99989890e-01]], [ 2.38256170e+08, -2.07422360e+08, -7.56197178e+07], (-0.25320985, 140.18055815, 35792274.75899146), 0, 0),
                (50131.01041667, (960218,   1500), [ 14835691.90970188, -39473705.58489136,   -248736.60300345], [2877.63765957, 1081.9148182 ,    9.55147314], [-32389118.03536845,  27004552.79890675,   -179493.62657611], [ 0.79062131,  0.43750908, 22.42011344], 151.01183079, (330.16701107, -12.17811148), (179.77462147, -11.952801  ), [[ 9.99936381e-01,  1.03449561e-02,  4.49612989e-03], [-1.03447719e-02,  9.99946489e-01, -6.42345798e-05], [-4.49655380e-03,  1.77190553e-05,  9.99989890e-01]], [ 2.37625908e+08, -2.07525364e+08, -7.55445552e+07], (-0.24412169, 140.18021156, 35792148.80948149), 0, 0),
                (50131.01388889, (960218,   2000), [ 15695366.40490882, -39139715.76420763,   -245812.06324505], [2853.29712752, 1144.59530548,    9.94467917], [-32388881.66227116,  27004681.27687033,   -172725.38836895], [ 0.7856262 ,  0.41958762, 22.69985431], 152.26525312, (330.17009324, -12.17695013), (178.52427609, -11.95163228), [[ 9.99936381e-01,  1.03449578e-02,  4.49613064e-03], [-1.03447736e-02,  9.99946489e-01, -6.42335961e-05], [-4.49655455e-03,  1.77180562e-05,  9.99989890e-01]], [ 2.37002549e+08, -2.07647061e+08, -7.54694534e+07], (-0.23491716, 140.17987182, 35792021.2420001 ), 0, 0),
                (50131.01736111, (960218,   2500), [ 16547533.6691137 , -38787003.10533711,   -242770.27248672], [2827.5914462 , 1206.72876414,   10.33311542], [-32388646.84104986,  27004804.37195345,   -165874.85452439], [ 0.78027439,  0.40162218, 22.96872279], 153.51867545, (330.17317044, -12.17578748), (177.27392574, -11.95046228), [[ 9.99936381e-01,  1.03449595e-02,  4.49613140e-03], [-1.03447753e-02,  9.99946489e-01, -6.42326125e-05], [-4.49655529e-03,  1.77170571e-05,  9.99989890e-01]], [ 2.36386506e+08, -2.07787291e+08, -7.53944111e+07], (-0.22560065, 140.17953905, 35791892.18395986), 0, 0),
                (50131.02083333, (960218,   3000), [ 17391785.98229151, -38415736.18212036,   -239612.68950141], [2800.53288309, 1268.28546791,   10.71659666], [-32388413.67874206,  27004922.07123395,   -158945.30610131], [ 0.77457509,  0.38362576, 23.2265907 ], 154.77209777, (330.17624281, -12.17462353), (176.02357057, -11.94929096), [[ 9.99936381e-01,  1.03449612e-02,  4.49613215e-03], [-1.03447770e-02,  9.99946489e-01, -6.42316291e-05], [-4.49655603e-03,  1.77160583e-05,  9.99989890e-01]], [ 2.35778185e+08, -2.07945887e+08, -7.53194268e+07], (-0.21617663, 140.17921335, 35791761.76173551), 0, 0)],
                dtype=vissr.ORBIT_PREDICTION_DATA
            )
        }

    @pytest.fixture
    def vis_calibration(self):
        return {

        }

    @pytest.fixture
    def ir1_calibration(self):
        return {

        }

    @pytest.fixture
    def ir2_calibration(self):
        return {

        }

    @pytest.fixture
    def wv_calibration(self):
        return {

        }

    @pytest.fixture
    def simple_coordinate_conversion_table(self):
        return {

        }

    @pytest.fixture
    def lons_lats_exp(self, dataset_id):
        """Get expected lon/lat coordinates.

        Computed with JMA's Msial library for 2 pixels near the central column
        (6688.5/1672.5 for VIS/IR).

        VIS:

        pix = [6688, 6688, 6689, 6689]
        lin = [2744, 8356, 2744, 8356]

        IR1:

        pix = [1672, 1672, 1673, 1673]
        lin = [686, 2089, 686, 2089]
        """
        if dataset_id['name'] == 'IR1':
            lons_exp = [[139.680120, 139.718902],
                        [140.307367, 140.346062]]
            lats_exp = [[35.045132, 35.045361],
                        [-34.971012, -34.970738]]
        elif dataset_id['name'] == 'VIS':
            lons_exp = [[139.665133, 139.674833],
                        [140.292579, 140.302249]]
            lats_exp = [[35.076113, 35.076170],
                        [-34.940439, -34.940370]]
        else:
            raise NotImplementedError
        return lons_exp, lats_exp


def assert_namedtuple_close(a, b):
    assert a.__class__ == b.__class__
    for attr in a._fields:
        np.testing.assert_allclose(
            getattr(a, attr),
            getattr(b, attr),
            err_msg='{} attribute {} differs'.format(a.__class__, attr)
        )
