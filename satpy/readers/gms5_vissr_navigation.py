"""GMS-5 VISSR Navigation."""

import numba
import numpy as np

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


@numba.njit
def get_observation_time(point, start_time_of_scan, scan_params):
    """Calculate observation time of a VISSR pixel."""
    relative_time = _get_relative_observation_time(point, scan_params)
    return start_time_of_scan + relative_time


@numba.njit
def _get_relative_observation_time(point, scanning_params):
    line, pixel = point
    spinning_rate, num_sensors, sampling_angle = scanning_params
    spinning_freq = 1440 * spinning_rate
    line_step = np.floor((line - 1) / num_sensors)
    pixel_step = (sampling_angle * pixel) / (2 * np.pi)
    return (line_step + pixel_step) / spinning_freq


@numba.njit
def interpolate_nav_params_at_obs_time(nav_params, obs_time):
    pass


@numba.njit
def interpolate_prediction(prediction_times, predicted_values, desired_time):
    # TODO: Interpolate all fields
    # TODO: How to interpolate angles? See mspVissrGetInsertValue
    return np.interp(desired_time, prediction_times, predicted_values)


@numba.njit
def get_nearest_prediction(prediction_times, predicted_values, desired_time):
    time_diff = desired_time - prediction_times
    idx_of_nearest_prediction = np.argmin(np.fabs(time_diff))
    return predicted_values[idx_of_nearest_prediction]


def dict_to_nav_params(dictionary):
    arr = np.empty(1, nav_params_dtype)
    for key, val in dictionary.items():
        arr[key] = val
    return arr



nav_params_dtype = np.dtype([
    ('start_time_of_scan', np.float32),
    ('spinning_rate', np.float32),
    ('num_sensors', np.int32),
    ('sampling_angle', np.float32),
    ('equatorial_earth_radius', np.float32),
    ('earth_flattening', np.float32)
])


attitude_prediction_dtype = np.dtype(
    [
        ('prediction_time_mjd', np.float64)
    ]
)
orbit_prediction_dtype = np.dtype(
    [
        ('prediction_time_mjd', np.float64)
    ]
)
image_navigator_spec = [
    ('start_time_of_scan', numba.float64),
    ('line_offset', numba.int32),
    ('pixel_offset', numba.int32),
    ('sampling_angle', numba.float32),
    ('stepping_angle', numba.float32),
    ('spinning_rate', numba.float32),
    ('num_sensors', numba.int32),
    ('misalignment', numba.float32[:, :]),
    ('attitude_prediction', numba.from_dtype(attitude_prediction_dtype)[:]),
    ('orbit_prediction', numba.from_dtype(orbit_prediction_dtype)[:]),
    ('earth_flattening', numba.float32),
    ('earth_equatorial_radius', numba.float32)
]  # TODO: Compare types with header types

@numba.experimental.jitclass(spec=image_navigator_spec)
class ImageNavigator:
    def __init__(self, start_time_of_scan, line_offset, pixel_offset, sampling_angle, stepping_angle,
                 spinning_rate, num_sensors, misalignment, attitude_prediction,
                 orbit_prediction, earth_flattening, earth_equatorial_radius):
        self.start_time_of_scan = start_time_of_scan
        self.line_offset = line_offset
        self.pixel_offset = pixel_offset
        self.sampling_angle = sampling_angle
        self.stepping_angle = stepping_angle
        self.spinning_rate = spinning_rate
        self.num_sensors = num_sensors
        self.misalignment = misalignment
        self.attitude_prediction = attitude_prediction
        self.orbit_prediction = orbit_prediction
        self.earth_flattening = earth_flattening
        self.earth_equatorial_radius = earth_equatorial_radius



@numba.experimental.jitclass([
    ('line_offset', numba.int32),
    ('pixel_offset', numba.int32),
    ('stepping_angle', numba.float64),
    ('sampling_angle', numba.float64),
    ('misalignment', numba.types.Array(numba.float64, 2, layout='C')),
    ('greenwich_sidereal_time', numba.float64),
    ('angle_between_earth_and_sun', numba.float64),
    ('declination_from_sat_to_sun', numba.float64),
    ('right_ascension_from_sat_to_sun', numba.float64),
    ('angle_between_sat_spin_and_z_axis', numba.float64),
    ('angle_between_sat_spin_and_yz_plane', numba.float64),
    ('sat_position_earth_fixed_x', numba.float64),
    ('sat_position_earth_fixed_y', numba.float64),
    ('sat_position_earth_fixed_z', numba.float64),
    ('nutation_precession', numba.types.Array(numba.float64, 2, layout='C')),
    ('earth_flattening', numba.float64),
    ('earth_equatorial_radius', numba.float64)
])
class NavigationParameters:
    def __init__(
            self,
            line_offset,
            pixel_offset,
            stepping_angle,
            sampling_angle,
            misalignment,
            greenwich_sidereal_time,
            angle_between_earth_and_sun,
            declination_from_sat_to_sun,
            right_ascension_from_sat_to_sun,
            angle_between_sat_spin_and_z_axis,
            angle_between_sat_spin_and_yz_plane,
            sat_position_earth_fixed_x,
            sat_position_earth_fixed_y,
            sat_position_earth_fixed_z,
            nutation_precession,
            earth_flattening,
            earth_equatorial_radius
    ):
        self.line_offset = line_offset
        self.pixel_offset = pixel_offset
        self.stepping_angle = stepping_angle
        self.sampling_angle = sampling_angle
        self.misalignment = misalignment
        self.greenwich_sidereal_time = greenwich_sidereal_time
        self.angle_between_earth_and_sun = angle_between_earth_and_sun
        self.declination_from_sat_to_sun = declination_from_sat_to_sun
        self.right_ascension_from_sat_to_sun = right_ascension_from_sat_to_sun
        self.angle_between_sat_spin_and_z_axis = angle_between_sat_spin_and_z_axis
        self.angle_between_sat_spin_and_yz_plane = angle_between_sat_spin_and_yz_plane
        self.sat_position_earth_fixed_x = sat_position_earth_fixed_x
        self.sat_position_earth_fixed_y = sat_position_earth_fixed_y
        self.sat_position_earth_fixed_z = sat_position_earth_fixed_z
        self.nutation_precession = nutation_precession
        self.earth_flattening = earth_flattening
        self.earth_equatorial_radius = earth_equatorial_radius

        # TODO: Remember that all angles are expected in rad
        # TODO: Watch out shape of 3x3 matrices! See msVissrNav.c

    def get_image_offset(self):
        return self.line_offset, self.pixel_offset

    def get_sampling(self):
        return self.stepping_angle, self.sampling_angle

    def get_sat_sun_angles(self):
        return np.array([self.declination_from_sat_to_sun, self.right_ascension_from_sat_to_sun])

    def get_spin_angles(self):
        return np.array([self.angle_between_sat_spin_and_z_axis, self.angle_between_sat_spin_and_yz_plane])

    def get_ellipsoid(self):
        return np.array([self.earth_equatorial_radius, self.earth_flattening])

    def get_sat_position(self):
        return np.array((self.sat_position_earth_fixed_x,
                         self.sat_position_earth_fixed_y,
                         self.sat_position_earth_fixed_z))


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
        point_on_earth, nav_params.earth_flattening
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
    x = sampling_angle * (pixel + 0.5 - pixel_offset)
    y = stepping_angle * (line + 0.5 - line_offset)
    return np.array([x, y])


@numba.njit
def _transform_scanning_angles_to_satellite_coords(angles, nav_params):
    transformer = ScanningAnglesToSatelliteCoordsTransformer(
        nav_params.misalignment
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
    # TODO: kwargs possible here?
    transformer = SatelliteToEarthFixedCoordsTransformer(
        nav_params.greenwich_sidereal_time,
        nav_params.get_sat_sun_angles(),
        nav_params.angle_between_earth_and_sun,
        nav_params.get_spin_angles(),
        nav_params.nutation_precession
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
