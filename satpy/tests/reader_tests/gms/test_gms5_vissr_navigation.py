"""Unit tests for GMS-5 VISSR navigation."""

import numpy as np
import pytest

from satpy.tests.reader_tests.utils import get_jit_methods, skip_numba_unstable_if_missing

try:
    import satpy.readers.gms.gms5_vissr_navigation as nav
except ImportError as err:
    if skip_numba_unstable_if_missing():
        pytest.skip(f"Numba is not compatible with unstable NumPy: {err!s}", allow_module_level=True)
    raise

# Navigation references computed with JMA's Msial library (files
# VISSR_19960217_2331_IR1.A.IMG and VISSR_19960217_2331_VIS.A.IMG). The VIS
# navigation is slightly off (< 0.01 deg) compared to JMA's reference.
# This is probably due to precision problems with the copied numbers.
IR_NAVIGATION_REFERENCE = [
    {
        "pixel": nav.Pixel(line=686, pixel=1680),
        'lon': 139.990380,
        'lat': 35.047056,
        'nav_params': nav.PixelNavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=3.997397917902958,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            orbit=nav.Orbit(
                angles=nav.OrbitAngles(
                    greenwich_sidereal_time=2.468529732418296,
                    declination_from_sat_to_sun=-0.208770861178982,
                    right_ascension_from_sat_to_sun=3.304369303579407,
                ),
                sat_position=nav.Vector3D(
                    x=-32390963.148471601307392,
                    y=27003395.381247851997614,
                    z=-228134.860026293463307,
                ),
                nutation_precession=np.array(
                    [[0.999936381496146, -0.010344758016410, -0.004496547784299],
                     [0.010344942303489, 0.999946489495557, 0.000017727054455],
                     [0.004496123789670, -0.000064242454080, 0.999989890320785]]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                image_offset=nav.ImageOffset(
                    line_offset=1378.5,
                    pixel_offset=1672.5,
                ),
                scanning_angles=nav.ScanningAngles(
                    stepping_angle=0.000140000047395,
                    sampling_angle=0.000095719995443,
                    misalignment=np.array(
                        [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                         [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                         [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
                    )
                ),
                earth_ellipsoid=nav.EarthEllipsoid(
                    flattening=0.003352813177897,
                    equatorial_radius=6378136.0
                )
            ),
        )
    },
    {
        "pixel": nav.Pixel(line=2089, pixel=1793),
        'lon': 144.996967,
        'lat': -34.959853,
        'nav_params': nav.PixelNavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=3.935707944355762,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            orbit=nav.Orbit(
                angles=nav.OrbitAngles(
                    greenwich_sidereal_time=2.530392320846865,
                    declination_from_sat_to_sun=-0.208713576872247,
                    right_ascension_from_sat_to_sun=3.242660398458377,
                ),
                sat_position=nav.Vector3D(
                    x=-32390273.633551981300116,
                    y=27003859.543135114014149,
                    z=-210800.087589388160268,
                ),
                nutation_precession=np.array(
                    [[0.999936381432029, -0.010344763228876, -0.004496550050695],
                     [0.010344947502662, 0.999946489441823, 0.000017724053657],
                     [0.004496126086653, -0.000064239500295, 0.999989890310647]]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                image_offset=nav.ImageOffset(
                    line_offset=1378.5,
                    pixel_offset=1672.5,
                ),
                scanning_angles=nav.ScanningAngles(
                    stepping_angle=0.000140000047395,
                    sampling_angle=0.000095719995443,
                    misalignment=np.array(
                        [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                         [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                         [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
                    ),
                ),
                earth_ellipsoid=nav.EarthEllipsoid(
                    flattening=0.003352813177897,
                    equatorial_radius=6378136
                )
            ),
        )
    }
]


VIS_NAVIGATION_REFERENCE = [
    {
        "pixel": nav.Pixel(line=2744, pixel=6720),
        'lon': 139.975527,
        'lat': 35.078028,
        'nav_params': nav.PixelNavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=3.997397918405798,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            orbit=nav.Orbit(
                angles=nav.OrbitAngles(
                    greenwich_sidereal_time=2.468529731914041,
                    declination_from_sat_to_sun=-0.208770861179448,
                    right_ascension_from_sat_to_sun=3.304369304082406,
                ),
                sat_position=nav.Vector3D(
                    x=-32390963.148477241396904,
                    y=27003395.381243918091059,
                    z=-228134.860164520738181,
                ),
                nutation_precession=np.array(
                    [[0.999936381496146, -0.010344758016410, -0.004496547784299],
                     [0.010344942303489, 0.999946489495557, 0.000017727054455],
                     [0.004496123789670, -0.000064242454080, 0.999989890320785]]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                image_offset=nav.ImageOffset(
                    line_offset=5513.0,
                    pixel_offset=6688.5,
                ),
                scanning_angles=nav.ScanningAngles(
                    stepping_angle=0.000035000004573,
                    sampling_angle=0.000023929998861,
                    misalignment=np.array(
                        [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                         [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                         [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
                    ),
                ),
                earth_ellipsoid=nav.EarthEllipsoid(
                    flattening=0.003352813177897,
                    equatorial_radius=6378136
                )
            ),
        )
    },
    {
        "pixel": nav.Pixel(line=8356, pixel=7172),
        'lon': 144.980104,
        'lat': -34.929123,
        'nav_params': nav.PixelNavigationParameters(
            attitude=nav.Attitude(
                angle_between_earth_and_sun=3.935707944858620,
                angle_between_sat_spin_and_z_axis=3.149118633034304,
                angle_between_sat_spin_and_yz_plane=0.000546042025980,
            ),
            orbit=nav.Orbit(
                angles=nav.OrbitAngles(
                    greenwich_sidereal_time=2.530392320342610,
                    declination_from_sat_to_sun=-0.208713576872715,
                    right_ascension_from_sat_to_sun=3.242660398961383,
                ),
                sat_position=nav.Vector3D(
                    x=-32390273.633557569235563,
                    y=27003859.543131537735462,
                    z=-210800.087734811415430,
                ),
                nutation_precession=np.array(
                    [[0.999936381432029, -0.010344763228876, -0.004496550050695],
                     [0.010344947502662, 0.999946489441823, 0.000017724053657],
                     [0.004496126086653, -0.000064239500295, 0.999989890310647]]
                ),
            ),
            proj_params=nav.ProjectionParameters(
                image_offset=nav.ImageOffset(
                    line_offset=5513.0,
                    pixel_offset=6688.5,
                ),
                scanning_angles=nav.ScanningAngles(
                    stepping_angle=0.000035000004573,
                    sampling_angle=0.000023929998861,
                    misalignment=np.array(
                        [[0.999999165534973, 0.000510364072397, 0.001214201096445],
                         [-0.000511951977387, 0.999999046325684, 0.001307720085606],
                         [-0.001213532872498, -0.001308340579271, 0.999998450279236]]
                    ),
                ),
                earth_ellipsoid=nav.EarthEllipsoid(
                    flattening=0.003352813177897,
                    equatorial_radius=6378136
                )
            ),
        )
    },
]

NAVIGATION_REFERENCE = VIS_NAVIGATION_REFERENCE + IR_NAVIGATION_REFERENCE


@pytest.fixture(params=[False, True], autouse=True)
def disable_jit(request, monkeypatch):
    """Run tests with jit enabled and disabled.

    Reason: Coverage report is only accurate with jit disabled.
    """
    if request.param:
        jit_methods = get_jit_methods(nav)
        for name, method in jit_methods.items():
            monkeypatch.setattr(name, method.py_func)


class TestSinglePixelNavigation:
    """Test navigation of a single pixel."""

    @pytest.mark.parametrize(
        "point,nav_params,expected",
        [
            (ref["pixel"], ref["nav_params"], (ref["lon"], ref["lat"]))
            for ref in NAVIGATION_REFERENCE
        ],
    )
    def test_get_lon_lat(self, point, nav_params, expected):
        """Test getting lon/lat coordinates for a given pixel."""
        lon, lat = nav.get_lon_lat(point, nav_params)
        np.testing.assert_allclose((lon, lat), expected)

    def test_transform_image_coords_to_scanning_angles(self):
        """Test transformation from image coordinates to scanning angles."""
        offset = nav.ImageOffset(line_offset=100, pixel_offset=200)
        scanning_angles = nav.ScanningAngles(
            stepping_angle=0.01, sampling_angle=0.02, misalignment=-999
        )
        angles = nav.transform_image_coords_to_scanning_angles(
            point=nav.Pixel(199, 99),
            image_offset=offset,
            scanning_angles=scanning_angles,
        )
        np.testing.assert_allclose(angles, [-2, 1])

    def test_transform_scanning_angles_to_satellite_coords(self):
        """Test transformation from scanning angles to satellite coordinates."""
        scanning_angles = nav.Vector2D(np.pi, np.pi / 2)
        misalignment = np.diag([1, 2, 3]).astype(float)
        point_sat = nav.transform_scanning_angles_to_satellite_coords(
            scanning_angles, misalignment
        )
        np.testing.assert_allclose(point_sat, [0, 0, 3], atol=1e-12)

    def test_transform_satellite_to_earth_fixed_coords(self):
        """Test transformation from satellite to earth-fixed coordinates."""
        point_sat = nav.Vector3D(1, 2, 3)
        attitude = nav.Attitude(
            angle_between_earth_and_sun=np.pi,
            angle_between_sat_spin_and_z_axis=np.pi,
            angle_between_sat_spin_and_yz_plane=np.pi / 2,
        )
        orbit = nav.Orbit(
            angles=nav.OrbitAngles(
                greenwich_sidereal_time=np.pi,
                declination_from_sat_to_sun=np.pi,
                right_ascension_from_sat_to_sun=np.pi / 2,
            ),
            sat_position=nav.Vector3D(-999, -999, -999),
            nutation_precession=np.diag([1, 2, 3]).astype(float),
        )
        res = nav.transform_satellite_to_earth_fixed_coords(point_sat, orbit, attitude)
        np.testing.assert_allclose(res, [-3, 1, -2])

    def test_intersect_view_vector_with_earth(self):
        """Test intersection of a view vector with the earth's surface."""
        view_vector = nav.Vector3D(-1, 0, 0)
        ellipsoid = nav.EarthEllipsoid(equatorial_radius=6371 * 1000, flattening=0.003)
        sat_pos = nav.Vector3D(x=36000 * 1000.0, y=0.0, z=0.0)
        point = nav.intersect_with_earth(view_vector, sat_pos, ellipsoid)
        exp = [ellipsoid.equatorial_radius, 0, 0]
        np.testing.assert_allclose(point, exp)

    @pytest.mark.parametrize(
        "point_earth_fixed,point_geodetic_exp",
        [
            ([0, 0, 1], [0, 90]),
            ([0, 0, -1], [0, -90]),
            ([1, 0, 0], [0, 0]),
            ([-1, 0, 0], [180, 0]),
            ([1, 1, 1], [45, 35.426852]),
        ],
    )
    def test_transform_earth_fixed_to_geodetic_coords(
        self, point_earth_fixed, point_geodetic_exp
    ):
        """Test transformation from earth-fixed to geodetic coordinates."""
        point_geodetic = nav.transform_earth_fixed_to_geodetic_coords(
            nav.Vector3D(*point_earth_fixed),
            0.003
        )
        np.testing.assert_allclose(point_geodetic, point_geodetic_exp)

    def test_normalize_vector(self):
        """Test vector normalization."""
        v = nav.Vector3D(1, 2, 3)
        norm = np.sqrt(14)
        exp = nav.Vector3D(1 / norm, 2 / norm, 3 / norm)
        normed = nav.normalize_vector(v)
        np.testing.assert_allclose(normed, exp)


class TestImageNavigation:
    """Test navigation of an entire image."""

    @pytest.fixture
    def expected(self):
        """Get expected coordinates."""
        exp = {
            "lon": [[-114.56923, -112.096837, -109.559702],
                    [8.33221, 8.793893, 9.22339],
                    [15.918476, 16.268354, 16.6332]],
            "lat": [[-23.078721, -24.629845, -26.133314],
                    [-42.513409, -39.790231, -37.06392],
                    [3.342834, 6.07043, 8.795932]]
        }
        return exp

    def test_get_lons_lats(self, navigation_params, expected):
        """Test getting lon/lat coordinates."""
        lons, lats = nav.get_lons_lats(
            lines=np.array([1000, 1500, 2000]),
            pixels=np.array([1000, 1500, 2000]),
            nav_params=navigation_params,
        )
        np.testing.assert_allclose(lons, expected["lon"])
        np.testing.assert_allclose(lats, expected["lat"])


class TestPredictionInterpolation:
    """Test interpolation of orbit and attitude predictions."""

    @pytest.mark.parametrize(
        "obs_time,expected", [(-1, np.nan), (1.5, 2.5), (5, np.nan)]
    )
    def test_interpolate_continuous(self, obs_time, expected):
        """Test interpolation of continuous variables."""
        prediction_times = np.array([0, 1, 2, 3])
        predicted_values = np.array([1, 2, 3, 4])
        res = nav.interpolate_continuous(obs_time, prediction_times, predicted_values)
        np.testing.assert_allclose(res, expected)

    @pytest.mark.parametrize(
        "obs_time,expected",
        [
            (-1, np.nan),
            (1.5, 0.75 * np.pi),
            (2.5, -0.75 * np.pi),
            (3.5, -0.25 * np.pi),
            (5, np.nan),
        ],
    )
    def test_interpolate_angles(self, obs_time, expected):
        """Test interpolation of periodic angles."""
        prediction_times = np.array([0, 1, 2, 3, 4])
        predicted_angles = np.array(
            [0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi]
        )  # already unwrapped
        res = nav.interpolate_angles(obs_time, prediction_times, predicted_angles)
        np.testing.assert_allclose(res, expected)

    @pytest.mark.parametrize(
        "obs_time,expected",
        [
            (-1, np.nan * np.ones((2, 2))),
            (1.5, [[1, 0], [0, 2]]),
            (3, np.nan * np.ones((2, 2))),
        ],
    )
    def test_interpolate_nearest(self, obs_time, expected):
        """Test nearest neighbour interpolation."""
        prediction_times = np.array([0, 1, 2])
        predicted_angles = np.array(
            [np.zeros((2, 2)), np.diag((1, 2)), np.zeros((2, 2))]
        )
        res = nav.interpolate_nearest(obs_time, prediction_times, predicted_angles)
        np.testing.assert_allclose(res, expected)

    def test_interpolate_orbit_prediction(
        self, obs_time, orbit_prediction, orbit_expected
    ):
        """Test interpolating orbit prediction."""
        orbit_prediction = orbit_prediction.to_numba()
        orbit = nav.interpolate_orbit_prediction(orbit_prediction, obs_time)
        _assert_namedtuple_close(orbit, orbit_expected)

    def test_interpolate_attitude_prediction(
        self, obs_time, attitude_prediction, attitude_expected
    ):
        """Test interpolating attitude prediction."""
        attitude_prediction = attitude_prediction.to_numba()
        attitude = nav.interpolate_attitude_prediction(attitude_prediction, obs_time)
        _assert_namedtuple_close(attitude, attitude_expected)

    @pytest.fixture
    def obs_time(self):
        """Get observation time."""
        return 2.5

    @pytest.fixture
    def orbit_expected(self):
        """Get expected orbit."""
        return nav.Orbit(
            angles=nav.OrbitAngles(
                greenwich_sidereal_time=1.5,
                declination_from_sat_to_sun=1.6,
                right_ascension_from_sat_to_sun=1.7,
            ),
            sat_position=nav.Vector3D(
                x=1.8,
                y=1.9,
                z=2.0,
            ),
            nutation_precession=1.6 * np.identity(3),
        )

    @pytest.fixture
    def attitude_expected(self):
        """Get expected attitude."""
        return nav.Attitude(
            angle_between_earth_and_sun=1.5,
            angle_between_sat_spin_and_z_axis=1.6,
            angle_between_sat_spin_and_yz_plane=1.7,
        )


@pytest.fixture
def sampling_angle():
    """Get sampling angle."""
    return 0.000095719995443


@pytest.fixture
def scan_params(sampling_angle):
    """Get scanning parameters."""
    return nav.ScanningParameters(
        start_time_of_scan=0,
        spinning_rate=0.5,
        num_sensors=1,
        sampling_angle=sampling_angle,
    )


@pytest.fixture
def attitude_prediction():
    """Get attitude prediction."""
    return nav.AttitudePrediction(
        prediction_times=np.array([1.0, 2.0, 3.0]),
        attitude=nav.Attitude(
            angle_between_earth_and_sun=np.array([0.0, 1.0, 2.0]),
            angle_between_sat_spin_and_z_axis=np.array([0.1, 1.1, 2.1]),
            angle_between_sat_spin_and_yz_plane=np.array([0.2, 1.2, 2.2]),
        ),
    )


@pytest.fixture
def orbit_prediction():
    """Get orbit prediction."""
    return nav.OrbitPrediction(
        prediction_times=np.array([1.0, 2.0, 3.0, 4.0]),
        angles=nav.OrbitAngles(
            greenwich_sidereal_time=np.array([0.0, 1.0, 2.0, 3.0]),
            declination_from_sat_to_sun=np.array([0.1, 1.1, 2.1, 3.1]),
            right_ascension_from_sat_to_sun=np.array([0.2, 1.2, 2.2, 3.2]),
        ),
        sat_position=nav.Vector3D(
            x=np.array([0.3, 1.3, 2.3, 3.3]),
            y=np.array([0.4, 1.4, 2.4, 3.4]),
            z=np.array([0.5, 1.5, 2.5, 3.5]),
        ),
        nutation_precession=np.array(
            [
                0.6 * np.identity(3),
                1.6 * np.identity(3),
                2.6 * np.identity(3),
                3.6 * np.identity(3),
            ]
        ),
    )


@pytest.fixture
def proj_params(sampling_angle):
    """Get projection parameters."""
    return nav.ProjectionParameters(
        image_offset=nav.ImageOffset(
            line_offset=1378.5,
            pixel_offset=1672.5,
        ),
        scanning_angles=nav.ScanningAngles(
            stepping_angle=0.000140000047395,
            sampling_angle=sampling_angle,
            misalignment=np.identity(3).astype(np.float64),
        ),
        earth_ellipsoid=nav.EarthEllipsoid(
            flattening=0.003352813177897,
            equatorial_radius=6378136,
        ),
    )


@pytest.fixture
def static_nav_params(proj_params, scan_params):
    """Get static navigation parameters."""
    return nav.StaticNavigationParameters(proj_params, scan_params)


@pytest.fixture
def predicted_nav_params(attitude_prediction, orbit_prediction):
    """Get predicted navigation parameters."""
    return nav.PredictedNavigationParameters(attitude_prediction, orbit_prediction)


@pytest.fixture
def navigation_params(static_nav_params, predicted_nav_params):
    """Get image navigation parameters."""
    return nav.ImageNavigationParameters(static_nav_params, predicted_nav_params)


def test_get_observation_time():
    """Test getting a pixel's observation time."""
    scan_params = nav.ScanningParameters(
        start_time_of_scan=50000.0,
        spinning_rate=100,
        num_sensors=1,
        sampling_angle=0.01,
    )
    pixel = nav.Pixel(11, 100)
    obs_time = nav.get_observation_time(pixel, scan_params)
    np.testing.assert_allclose(obs_time, 50000.0000705496871047)


def _assert_namedtuple_close(a, b):
    cls_name = b.__class__.__name__
    assert a.__class__ == b.__class__
    for attr in b._fields:
        a_attr = getattr(a, attr)
        b_attr = getattr(b, attr)
        if _is_namedtuple(b_attr):
            _assert_namedtuple_close(a_attr, b_attr)
        np.testing.assert_allclose(
            a_attr, b_attr, err_msg=f"{cls_name} attribute {attr} differs"
        )


def _is_namedtuple(obj):
    return hasattr(obj, "_fields")
