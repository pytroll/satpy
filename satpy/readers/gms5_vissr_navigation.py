"""GMS-5 VISSR Navigation.


Reference: `GMS User Guide`_, Appendix E, S-VISSR Mapping.

.. _GMS User Guide:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/GMS_Users_Guide_3rd_Edition_Rev1.pdf
"""

import numba
import numpy as np


def get_jitclass_type(cls):
    try:
        return cls.class_type.instance_type
    except AttributeError:
        # With NUMBA_DISABLE_JIT=1
        return cls


@numba.njit
def get_lons_lats(lines, pixels, nav_params):
    num_lines = len(lines)
    num_pixels = len(pixels)
    output_shape = (num_lines, num_pixels)
    lons = np.zeros(output_shape)
    lats = np.zeros(output_shape)
    for i in range(num_lines):
        for j in range(num_pixels):
            line = lines[i]
            pixel = pixels[j]
            point = (line, pixel)
            lon, lat = get_lon_lat(point, nav_params)
            lons[i, j] = lon
            lats[i, j] = lat


@numba.experimental.jitclass(
    [
        ('start_time_of_scan', numba.float64),
        ('spinning_rate', numba.float64),
        ('num_sensors', numba.int64),
        ('sampling_angle', numba.float64)
    ]
)
class ScanningParameters:
    def __init__(self, start_time_of_scan, spinning_rate, num_sensors,
                 sampling_angle):
        self.start_time_of_scan = start_time_of_scan
        self.spinning_rate = spinning_rate
        self.num_sensors = num_sensors
        self.sampling_angle = sampling_angle


@numba.njit
def get_observation_time(point, scan_params):
    """Calculate observation time of a VISSR pixel."""
    relative_time = _get_relative_observation_time(point, scan_params)
    return scan_params.start_time_of_scan + relative_time


@numba.njit
def _get_relative_observation_time(point, scan_params):
    line, pixel = point
    spinning_freq = 1440 * scan_params.spinning_rate
    line_step = np.floor((line - 1) / scan_params.num_sensors)
    pixel_step = (scan_params.sampling_angle * pixel) / (2 * np.pi)
    return (line_step + pixel_step) / spinning_freq


@numba.njit
def get_lon_lat(line, pixel, nav_params):
    """Get longitude and latitude coordinates for a given image pixel."""
    scan_angles = transform_image_coords_to_scanning_angles(
        point=(line, pixel),
        offset=nav_params.get_image_offset(),
        sampling=nav_params.get_sampling()
    )
    view_vector_sat = _transform_scanning_angles_to_satellite_coords(
        scan_angles, nav_params
    )
    view_vector_earth_fixed = _transform_satellite_to_earth_fixed_coords(
        view_vector_sat, nav_params
    )
    point_on_earth = _intersect_with_earth(view_vector_earth_fixed, nav_params)
    lon, lat = transform_earth_fixed_to_geodetic_coords(
        point_on_earth, nav_params.proj_params.earth_flattening
    )
    return lon, lat


@numba.njit
def transform_image_coords_to_scanning_angles(point, offset, sampling):
    """Transform image coordinates to scanning angles.

    Args:
        point: Point (line, pixel) in image coordinates.
        offset: Offset (line, pixel) from image center.
        sampling: Stepping angle (along line) and sampling angle (along pixels)
            in radians.
    Returns:
        Scanning angles (x, y) at the pixel center (rad).
    """
    line, pixel = point
    line_offset, pixel_offset = offset
    stepping_angle, sampling_angle = sampling
    x = sampling_angle * (pixel + 1 - pixel_offset)
    y = stepping_angle * (line + 1 - line_offset)
    return np.array([x, y])


@numba.njit
def _transform_scanning_angles_to_satellite_coords(angles, nav_params):
    transformer = ScanningAnglesToSatelliteCoordsTransformer(
        nav_params.proj_params.misalignment
    )
    return transformer.transform(angles)


@numba.experimental.jitclass([
    ('misalignment', numba.types.Array(numba.float64, 2, layout='C'))
])
class ScanningAnglesToSatelliteCoordsTransformer:
    """Transform scanning angles to satellite angular momentum coordinates."""

    def __init__(self, misalignment):
        """Initialize the transformer.

        Args:
            misalignment: Misalignment matrix (3x3)
        """
        self.misalignment = misalignment

    def transform(self, angles):
        """Transform scanning angles to satellite angular momentum coordinates.

        Args:
            angles: Scanning angles (x, y) in radians.

        Returns:
            View vector (x, y, z) in satellite angular momentum coordinates.
        """
        rotation, vector = self._get_transforms(angles)
        return np.dot(rotation, np.dot(self.misalignment, vector))

    def _get_transforms(self, angles):
        x, y = angles
        cos_x = np.cos(x)
        sin_x = np.sin(x)
        rot = np.array(((cos_x, -sin_x, 0),
                        (sin_x, cos_x, 0),
                        (0, 0, 1)))
        vec = np.array([np.cos(y), 0, np.sin(y)])
        return rot, vec


@numba.njit
def _transform_satellite_to_earth_fixed_coords(point, nav_params):
    transformer = SatelliteToEarthFixedCoordsTransformer(
        nav_params.orbit.greenwich_sidereal_time,
        nav_params.get_sat_sun_angles(),
        nav_params.attitude.angle_between_earth_and_sun,
        nav_params.get_spin_angles(),
        nav_params.orbit.nutation_precession
    )
    return transformer.transform(point)


@numba.experimental.jitclass([
    ('greenwich_sidereal_time', numba.float64),
    ('sat_sun_angles', numba.float64[:]),
    ('earth_sun_angle', numba.float64),
    ('spin_angles', numba.float64[:]),
    ('nutation_precession', numba.types.Array(numba.float64, 2, layout='C'))
])
class SatelliteToEarthFixedCoordsTransformer:
    """Transform from earth-fixed to satellite angular momentum coordinates."""

    def __init__(self, greenwich_sidereal_time, sat_sun_angles, earth_sun_angle, spin_angles, nutation_precession):
        """Initialize the Transformer.

        Args:
            greenwich_sidereal_time: True Greenwich sidereal time (rad).
            sat_sun_angles: Declination from satellite to sun (rad),
                right ascension from satellite to sun (rad)
            earth_sun_angle: Angle between sun and earth center on the z-axis
                vertical plane (rad)
            spin_angles: Angle between satellite spin axis and z-axis (rad),
                angle between satellite spin axis and yz-plane
            nutation_precession: Nutation and precession matrix (3x3)
        """
        self.greenwich_sidereal_time = greenwich_sidereal_time
        self.sat_sun_angles = sat_sun_angles
        self.earth_sun_angle = earth_sun_angle
        self.spin_angles = spin_angles
        self.nutation_precession = nutation_precession

    def transform(self, point):
        """Transform from earth-fixed to satellite angular momentum coordinates.

        Args:
            point: Point (x, y, z) in satellite angular momentum coordinates.

        Returns:
            Point (x', y', z') in earth-fixed coordinates.
        """
        sat_unit_vectors = self._get_satellite_unit_vectors()
        return np.dot(sat_unit_vectors, point)

    def _get_satellite_unit_vectors(self):
        unit_vector_z = self._get_satellite_unit_vector_z()
        unit_vector_x = self._get_satellite_unit_vector_x(unit_vector_z)
        unit_vector_y = self._get_satellite_unit_vector_y(unit_vector_x, unit_vector_z)
        return np.stack((unit_vector_x, unit_vector_y, unit_vector_z), axis=-1)

    def _get_satellite_unit_vector_z(self):
        sat_z_axis_1950 = self._get_satellite_z_axis_1950()
        rotation = self._get_transform_from_1950_to_earth_fixed()
        z_vec = np.dot(rotation, np.dot(self.nutation_precession, sat_z_axis_1950))
        return normalize_vector(z_vec)

    def _get_satellite_z_axis_1950(self):
        """Get satellite z-axis (spin) in mean of 1950 coordinates."""
        alpha, delta = self.spin_angles
        cos_delta = np.cos(delta)
        x = np.sin(delta)
        y = -cos_delta * np.sin(alpha)
        z = cos_delta * np.cos(alpha)
        return np.array([x, y, z])

    def _get_transform_from_1950_to_earth_fixed(self):
        cos = np.cos(self.greenwich_sidereal_time)
        sin = np.sin(self.greenwich_sidereal_time)
        return np.array(
            ((cos, sin, 0),
             (-sin, cos, 0),
             (0, 0, 1))
        )

    def _get_satellite_unit_vector_x(self, sat_unit_vector_z):
        beta = self.earth_sun_angle
        sat_sun_vector = self._get_vector_from_satellite_to_sun()
        z_cross_satsun = np.cross(sat_unit_vector_z, sat_sun_vector)
        z_cross_satsun = normalize_vector(z_cross_satsun)
        x_vec = z_cross_satsun * np.sin(beta) + \
            np.cross(z_cross_satsun, sat_unit_vector_z) * np.cos(beta)
        return normalize_vector(x_vec)

    def _get_vector_from_satellite_to_sun(self):
        declination, right_ascension = self.sat_sun_angles
        cos_declination = np.cos(declination)
        x = cos_declination * np.cos(right_ascension)
        y = cos_declination * np.sin(right_ascension)
        z = np.sin(declination)
        return np.array([x, y, z])

    def _get_satellite_unit_vector_y(self, sat_unit_vector_x, sat_unit_vector_z):
        y_vec = np.cross(sat_unit_vector_z, sat_unit_vector_x)
        return normalize_vector(y_vec)


@numba.njit
def _intersect_with_earth(view_vector, nav_params):
    intersector = EarthIntersector(
        nav_params.get_sat_position(),
        nav_params.get_ellipsoid()
    )
    return intersector.intersect(view_vector)


@numba.experimental.jitclass([
    ('sat_pos', numba.float64[:]),
    ('ellipsoid', numba.float64[:])
])
class EarthIntersector:
    """Intersect instrument viewing vector with the earth's surface."""

    def __init__(self, sat_pos, ellipsoid):
        """
        Args:
            sat_pos: Satellite position (x, y, z) in earth-fixed coordinates.
            ellipsoid: Flattening and equatorial radius of the earth.
        """
        self.sat_pos = sat_pos
        self.ellipsoid = ellipsoid

    def intersect(self, view_vector):
        """Intersect instrument viewing vector with the earth's surface.

        Args:
            view_vector: Instrument viewing vector (x, y, z) in earth-fixed
                coordinates.
        Returns:
            Intersection (x', y', z') with the earth's surface.
        """
        distance = self._get_distance_to_intersection(view_vector)
        return self.sat_pos + distance * view_vector

    def _get_distance_to_intersection(self, view_vector):
        """Get distance to intersection with the earth.

        If the instrument is pointing towards the earth, there will be two
        intersections with the surface. Choose the one on the instrument-facing
        side of the earth.
        """
        d1, d2 = self._get_distances_to_intersections(view_vector)
        return min(d1, d2)

    def _get_distances_to_intersections(self, view_vector):
        equatorial_radius, flattening = self.ellipsoid
        flat2 = (1 - flattening) ** 2
        ux, uy, uz = view_vector
        x, y, z = self.sat_pos

        a = flat2 * (ux**2 + uy**2) + uz**2
        b = flat2 * (x*ux + y*uy) + z*uz
        c = flat2 * (x**2 + y**2 - equatorial_radius**2) + z**2

        tmp = np.sqrt((b**2 - a*c))
        dist_1 = (-b + tmp) / a
        dist_2 = (-b - tmp) / a
        return dist_1, dist_2


@numba.njit
def transform_earth_fixed_to_geodetic_coords(point, earth_flattening):
    """Transform from earth-fixed to geodetic coordinates.

    Args:
        point: Point (x, y, z) in earth-fixed coordinates.
        earth_flattening: Flattening of the earth.

    Returns:
        Geodetic longitude and latitude (degrees).
    """
    x, y, z = point
    f = earth_flattening
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, ((1 - f)**2 * np.sqrt(x**2 + y**2)))
    return np.rad2deg(lon), np.rad2deg(lat)


@numba.njit
def normalize_vector(v):
    """Normalize the given vector."""
    return v / np.sqrt(np.dot(v, v))


@numba.njit
def interpolate_continuous(x, x_sample, y_sample):
    """Linear interpolation of continuous quantities.

    Numpy equivalent would be np.interp(..., left=np.nan, right=np.nan), but
    numba currently doesn't support those keyword arguments.
    """
    try:
        return _interpolate(x, x_sample, y_sample, False)
    except Exception:
        return np.nan


@numba.njit
def interpolate_angles(x, x_sample, y_sample):
    """Linear interpolation of periodic angles.

    Takes care of phase jumps by wrapping angle differences to [-pi, pi].

    Numpy equivalent would be np.interp(x, x_sample, np.unwrap(y_sample)), but
    numba currently doesn't support np.unwrap.
    """
    try:
        return _interpolate(x, x_sample, y_sample, True)
    except Exception:
        return np.nan


@numba.njit
def _interpolate(x, x_sample, y_sample, wrap_2pi):
    i = _find_enclosing_index(x, x_sample)
    offset = y_sample[i]
    x_diff = x_sample[i+1] - x_sample[i]
    y_diff = y_sample[i+1] - y_sample[i]
    if wrap_2pi:
        y_diff = _wrap_2pi(y_diff)
    slope = y_diff / x_diff
    dist = x - x_sample[i]
    return offset + slope * dist


@numba.njit
def _find_enclosing_index(x, x_sample):
    """Find where x_sample encloses x."""
    for i in range(len(x_sample) - 1):
        if x_sample[i] <= x < x_sample[i+1]:
            return i
    raise Exception('x not enclosed by x_sample')


@numba.njit
def _wrap_2pi(values):
    """Wrap values to interval [-pi, pi].

    Source: https://stackoverflow.com/a/15927914/5703449
    """
    return (values + np.pi) % (2 * np.pi) - np.pi


@numba.njit
def interpolate_nearest(x, x_sample, y_sample):
    """Nearest neighbour interpolation."""
    try:
        return _interpolate_nearest(x, x_sample, y_sample)
    except Exception:
        return np.nan * np.ones_like(y_sample[0])


@numba.njit
def _interpolate_nearest(x, x_sample, y_sample):
    i = _find_enclosing_index(x, x_sample)
    return y_sample[i]


@numba.experimental.jitclass(
    [
        ('line_offset', numba.float64),
        ('pixel_offset', numba.float64),
        ('stepping_angle', numba.float64),
        ('sampling_angle', numba.float64),
        ('misalignment', numba.types.Array(numba.float64, 2, layout='C')),
        ('earth_flattening', numba.float64),
        ('earth_equatorial_radius', numba.float64),
    ]
)
class ProjectionParameters:
    def __init__(
            self,
            line_offset,
            pixel_offset,
            stepping_angle,
            sampling_angle,
            misalignment,
            earth_flattening,
            earth_equatorial_radius
    ):
        self.line_offset = line_offset
        self.pixel_offset = pixel_offset
        self.stepping_angle = stepping_angle
        self.sampling_angle = sampling_angle
        self.misalignment = misalignment
        self.earth_flattening = earth_flattening
        self.earth_equatorial_radius = earth_equatorial_radius


@numba.experimental.jitclass(
    [
        ('prediction_times', numba.float64[:]),
        ('greenwich_sidereal_time', numba.float64[:]),
        ('declination_from_sat_to_sun', numba.float64[:]),
        ('right_ascension_from_sat_to_sun', numba.float64[:]),
        ('sat_position_earth_fixed_x', numba.float64[:]),
        ('sat_position_earth_fixed_y', numba.float64[:]),
        ('sat_position_earth_fixed_z', numba.float64[:]),
        ('nutation_precession', numba.types.Array(numba.float64, 3, layout='C')),
    ]
)
class OrbitPrediction:
    def __init__(
            self,
            prediction_times,
            greenwich_sidereal_time,
            declination_from_sat_to_sun,
            right_ascension_from_sat_to_sun,
            sat_position_earth_fixed_x,
            sat_position_earth_fixed_y,
            sat_position_earth_fixed_z,
            nutation_precession
    ):
        self.prediction_times = prediction_times
        self.greenwich_sidereal_time = greenwich_sidereal_time
        self.declination_from_sat_to_sun = declination_from_sat_to_sun
        self.right_ascension_from_sat_to_sun = right_ascension_from_sat_to_sun
        self.sat_position_earth_fixed_x = sat_position_earth_fixed_x
        self.sat_position_earth_fixed_y = sat_position_earth_fixed_y
        self.sat_position_earth_fixed_z = sat_position_earth_fixed_z
        self.nutation_precession = nutation_precession

    def interpolate(self, observation_time):
        greenwich_sidereal_time = self._interpolate_angles(
            self.greenwich_sidereal_time,
            observation_time
        )
        declination_from_sat_to_sun = self._interpolate_angles(
            self.declination_from_sat_to_sun,
            observation_time
        )
        right_ascension_from_sat_to_sun = self._interpolate_angles(
            self.right_ascension_from_sat_to_sun,
            observation_time
        )
        sat_position_earth_fixed_x = self._interpolate_continuous(
            self.sat_position_earth_fixed_x,
            observation_time
        )
        sat_position_earth_fixed_y = self._interpolate_continuous(
            self.sat_position_earth_fixed_y,
            observation_time
        )
        sat_position_earth_fixed_z = self._interpolate_continuous(
            self.sat_position_earth_fixed_z,
            observation_time
        )
        nutation_precession = self._interpolate_nearest(
            self.nutation_precession,
            observation_time
        )
        return Orbit(
            greenwich_sidereal_time,
            declination_from_sat_to_sun,
            right_ascension_from_sat_to_sun,
            sat_position_earth_fixed_x,
            sat_position_earth_fixed_y,
            sat_position_earth_fixed_z,
            nutation_precession
        )

    def _interpolate_continuous(self, predicted_values, observation_time):
        return interpolate_continuous(observation_time, self.prediction_times, predicted_values)

    def _interpolate_angles(self, predicted_values, observation_time):
        return interpolate_angles(observation_time, self.prediction_times, predicted_values)

    def _interpolate_nearest(self, predicted_values, observation_time):
        return interpolate_nearest(observation_time, self.prediction_times, predicted_values)


@numba.experimental.jitclass(
    [
        ('greenwich_sidereal_time', numba.float64),
        ('declination_from_sat_to_sun', numba.float64),
        ('right_ascension_from_sat_to_sun', numba.float64),
        ('sat_position_earth_fixed_x', numba.float64),
        ('sat_position_earth_fixed_y', numba.float64),
        ('sat_position_earth_fixed_z', numba.float64),
        ('nutation_precession', numba.types.Array(numba.float64, 2, layout='C')),
    ]
)
class Orbit:
    def __init__(
            self,
            greenwich_sidereal_time,
            declination_from_sat_to_sun,
            right_ascension_from_sat_to_sun,
            sat_position_earth_fixed_x,
            sat_position_earth_fixed_y,
            sat_position_earth_fixed_z,
            nutation_precession
    ):
        self.greenwich_sidereal_time = greenwich_sidereal_time
        self.declination_from_sat_to_sun = declination_from_sat_to_sun
        self.right_ascension_from_sat_to_sun = right_ascension_from_sat_to_sun
        self.sat_position_earth_fixed_x = sat_position_earth_fixed_x
        self.sat_position_earth_fixed_y = sat_position_earth_fixed_y
        self.sat_position_earth_fixed_z = sat_position_earth_fixed_z
        self.nutation_precession = nutation_precession


@numba.experimental.jitclass(
    [
        ('prediction_times', numba.float64[:]),
        ('angle_between_earth_and_sun', numba.float64[:]),
        ('angle_between_sat_spin_and_z_axis', numba.float64[:]),
        ('angle_between_sat_spin_and_yz_plane', numba.float64[:]),
    ]
)
class AttitudePrediction:
    def __init__(
            self,
            prediction_times,
            angle_between_earth_and_sun,
            angle_between_sat_spin_and_z_axis,
            angle_between_sat_spin_and_yz_plane
    ):
        self.prediction_times = prediction_times
        self.angle_between_earth_and_sun = angle_between_earth_and_sun
        self.angle_between_sat_spin_and_z_axis = angle_between_sat_spin_and_z_axis
        self.angle_between_sat_spin_and_yz_plane = angle_between_sat_spin_and_yz_plane

    def interpolate(self, observation_time):
        angle_between_earth_and_sun = self._interpolate(
            observation_time, self.angle_between_earth_and_sun
        )
        angle_between_sat_spin_and_z_axis = self._interpolate(
            observation_time, self.angle_between_sat_spin_and_z_axis,
        )
        angle_between_sat_spin_and_yz_plane = self._interpolate(
            observation_time, self.angle_between_sat_spin_and_yz_plane
        )
        return Attitude(
            angle_between_earth_and_sun,
            angle_between_sat_spin_and_z_axis,
            angle_between_sat_spin_and_yz_plane
        )

    def _interpolate(self, observation_time, predicted_values):
        return interpolate_angles(observation_time, self.prediction_times, predicted_values)


@numba.experimental.jitclass(
    [
        ('angle_between_earth_and_sun', numba.float64),
        ('angle_between_sat_spin_and_z_axis', numba.float64),
        ('angle_between_sat_spin_and_yz_plane', numba.float64),
    ]
)
class Attitude:
    def __init__(
            self,
            angle_between_earth_and_sun,
            angle_between_sat_spin_and_z_axis,
            angle_between_sat_spin_and_yz_plane
    ):
        self.angle_between_earth_and_sun = angle_between_earth_and_sun
        self.angle_between_sat_spin_and_z_axis = angle_between_sat_spin_and_z_axis
        self.angle_between_sat_spin_and_yz_plane = angle_between_sat_spin_and_yz_plane


@numba.experimental.jitclass(
    [
        ('attitude', get_jitclass_type(Attitude)),
        ('orbit', get_jitclass_type(Orbit)),
        ('proj_params', get_jitclass_type(ProjectionParameters)),
    ]
)
class NavigationParameters:
    def __init__(self, attitude, orbit, proj_params):
        self.attitude = attitude
        self.orbit = orbit
        self.proj_params = proj_params

        # TODO: Remember that all angles are expected in rad
        # TODO: Watch out shape of 3x3 matrices! See msVissrNav.c

    def get_image_offset(self):
        return self.proj_params.line_offset, self.proj_params.pixel_offset

    def get_sampling(self):
        return self.proj_params.stepping_angle, self.proj_params.sampling_angle

    def get_sat_sun_angles(self):
        return np.array([
            self.orbit.declination_from_sat_to_sun,
            self.orbit.right_ascension_from_sat_to_sun
        ])

    def get_spin_angles(self):
        return np.array([
            self.attitude.angle_between_sat_spin_and_z_axis,
            self.attitude.angle_between_sat_spin_and_yz_plane
        ])

    def get_ellipsoid(self):
        return np.array([
            self.proj_params.earth_equatorial_radius,
            self.proj_params.earth_flattening
        ])

    def get_sat_position(self):
        return np.array((self.orbit.sat_position_earth_fixed_x,
                         self.orbit.sat_position_earth_fixed_y,
                         self.orbit.sat_position_earth_fixed_z))


@numba.experimental.jitclass(
    [
        ('attitude_prediction', get_jitclass_type(AttitudePrediction)),
        ('orbit_prediction', get_jitclass_type(OrbitPrediction)),
        ('proj_params', get_jitclass_type(ProjectionParameters)),
    ]
)
class PredictionInterpolator:
    def __init__(self, attitude_prediction, orbit_prediction, proj_params):
        self.attitude_prediction = attitude_prediction
        self.orbit_prediction = orbit_prediction
        self.proj_params = proj_params

    def interpolate(self, observation_time):
        attitude = self.attitude_prediction.interpolate(observation_time)
        orbit = self.orbit_prediction.interpolate(observation_time)
        return self._get_nav_params(attitude, orbit)

    def _get_nav_params(self, attitude, orbit):
        return NavigationParameters(attitude, orbit, self.proj_params)
