"""GMS-5 VISSR Navigation.

Reference: `GMS User Guide`_, Appendix E, S-VISSR Mapping.

.. _GMS User Guide:
    https://www.data.jma.go.jp/mscweb/en/operation/fig/GMS_Users_Guide_3rd_Edition_Rev1.pdf
"""

from collections import namedtuple

import dask.array as da
import numba
import numpy as np

from satpy.utils import get_legacy_chunk_size

CHUNK_SIZE = get_legacy_chunk_size()

EARTH_FLATTENING = 1 / 298.257
EARTH_EQUATORIAL_RADIUS = 6378136.0
EARTH_POLAR_RADIUS = EARTH_EQUATORIAL_RADIUS * (1 - EARTH_FLATTENING)
"""Constants taken from JMA's Msial library."""


Pixel = namedtuple(
    "Pixel",
    ["line", "pixel"]
)
"""A VISSR pixel."""

Vector2D = namedtuple(
    "Vector2D",
    ["x", "y"]
)
"""A 2D vector."""


Vector3D = namedtuple(
    "Vector3D",
    ["x", "y", "z"]
)
"""A 3D vector."""


Satpos = namedtuple(
    "Satpos",
    ["x", "y", "z"]
)
"""A 3D vector."""


Attitude = namedtuple(
    "Attitude",
    [
        "angle_between_earth_and_sun",
        "angle_between_sat_spin_and_z_axis",
        "angle_between_sat_spin_and_yz_plane",
    ],
)
"""Attitude parameters.

Units: radians
"""


Orbit = namedtuple(
    "Orbit",
    [
        "angles",
        "sat_position",
        "nutation_precession",
    ],
)
"""Orbital Parameters

Args:
    angles (OrbitAngles): Orbit angles
    sat_position (Vector3D): Satellite position
    nutation_precession: Nutation and precession matrix (3x3)
"""


OrbitAngles = namedtuple(
    "OrbitAngles",
    [
        "greenwich_sidereal_time",
        "declination_from_sat_to_sun",
        "right_ascension_from_sat_to_sun",
    ],
)
"""Orbit angles.

Units: radians
"""


ImageNavigationParameters = namedtuple(
    "ImageNavigationParameters",
    ["static", "predicted"]
)
"""Navigation parameters for the entire image.

Args:
    static (StaticNavigationParameters): Static parameters.
    predicted (PredictedNavigationParameters): Predicted time-dependent parameters.
"""


PixelNavigationParameters = namedtuple(
    "PixelNavigationParameters",
    ["attitude", "orbit", "proj_params"]
)
"""Navigation parameters for a single pixel.

Args:
    attitude (Attitude): Attitude parameters
    orbit (Orbit): Orbit parameters
    proj_params (ProjectionParameters): Projection parameters
"""


StaticNavigationParameters = namedtuple(
    "StaticNavigationParameters",
    [
        "proj_params",
        "scan_params"
    ]
)
"""Navigation parameters which are constant for the entire scan.

Args:
    proj_params (ProjectionParameters): Projection parameters
    scan_params (ScanningParameters): Scanning parameters
"""


PredictedNavigationParameters = namedtuple(
    "PredictedNavigationParameters",
    [
        "attitude",
        "orbit"
    ]
)
"""Predictions of time-dependent navigation parameters.

They need to be evaluated for each pixel.

Args:
    attitude (AttitudePrediction): Attitude prediction
    orbit (OrbitPrediction): Orbit prediction
"""


ScanningParameters = namedtuple(
    "ScanningParameters",
    [
        "start_time_of_scan",
        "spinning_rate",
        "num_sensors",
        "sampling_angle"
    ],
)


ProjectionParameters = namedtuple(
    "ProjectionParameters",
    [
        "image_offset",
        "scanning_angles",
        "earth_ellipsoid",
    ],
)
"""Projection parameters.

Args:
    image_offset (ImageOffset): Image offset
    scanning_angles (ScanningAngles): Scanning angles
    earth_ellipsoid (EarthEllipsoid): Earth ellipsoid
"""


ImageOffset = namedtuple(
    "ImageOffset",
    [
        "line_offset",
        "pixel_offset",
    ]
)
"""Image offset

Args:
    line_offset: Line offset from image center
    pixel_offset: Pixel offset from image center
"""


ScanningAngles = namedtuple(
    "ScanningAngles",
    [
        "stepping_angle",
        "sampling_angle",
        "misalignment"
    ]
)
"""Scanning angles

Args:
    stepping_angle: Scanning angle along line (rad)
    sampling_angle: Scanning angle along pixel (rad)
    misalignment: Misalignment matrix (3x3)
"""


EarthEllipsoid = namedtuple(
    "EarthEllipsoid",
    [
        "flattening",
        "equatorial_radius"
    ]
)
"""Earth ellipsoid.

Args:
    flattening: Ellipsoid flattening
    equatorial_radius: Equatorial radius (meters)
"""


_AttitudePrediction = namedtuple(
    "_AttitudePrediction",
    [
        "prediction_times",
        "attitude"
    ],
)


_OrbitPrediction = namedtuple(
    "_OrbitPrediction",
    [
        "prediction_times",
        "angles",
        "sat_position",
        "nutation_precession",
    ],
)


class AttitudePrediction:
    """Attitude prediction.

    Use .to_numba() to pass this object to jitted methods. This extra
    layer avoids usage of jitclasses and having to re-implement np.unwrap in
    numba.
    """

    def __init__(
        self,
        prediction_times,
        attitude
    ):
        """Initialize attitude prediction.

        In order to accelerate interpolation, the 2-pi periodicity of angles
        is unwrapped here already (that means phase jumps greater than pi
        are wrapped to their 2*pi complement).

        Args:
            prediction_times: Timestamps of predicted attitudes
            attitude (Attitude): Attitudes at prediction times
        """
        self.prediction_times = prediction_times
        self.attitude = self._unwrap_angles(attitude)

    def _unwrap_angles(self, attitude):
        return Attitude(
            np.unwrap(attitude.angle_between_earth_and_sun),
            np.unwrap(attitude.angle_between_sat_spin_and_z_axis),
            np.unwrap(attitude.angle_between_sat_spin_and_yz_plane),
        )

    def to_numba(self):
        """Convert to numba-compatible type."""
        return _AttitudePrediction(
            prediction_times=self.prediction_times,
            attitude=self.attitude
        )


class OrbitPrediction:
    """Orbit prediction.

    Use .to_numba() to pass this object to jitted methods. This extra
    layer avoids usage of jitclasses and having to re-implement np.unwrap in
    numba.
    """

    def __init__(
        self,
        prediction_times,
        angles,
        sat_position,
        nutation_precession,
    ):
        """Initialize orbit prediction.

        In order to accelerate interpolation, the 2-pi periodicity of angles
        is unwrapped here already (that means phase jumps greater than pi
        are wrapped to their 2*pi complement).

        Args:
            prediction_times: Timestamps of orbit prediction.
            angles (OrbitAngles): Orbit angles
            sat_position (Vector3D): Satellite position
            nutation_precession: Nutation and precession matrix.
        """
        self.prediction_times = prediction_times
        self.angles = self._unwrap_angles(angles)
        self.sat_position = sat_position
        self.nutation_precession = nutation_precession

    def _unwrap_angles(self, angles):
        return OrbitAngles(
            greenwich_sidereal_time=np.unwrap(angles.greenwich_sidereal_time),
            declination_from_sat_to_sun=np.unwrap(angles.declination_from_sat_to_sun),
            right_ascension_from_sat_to_sun=np.unwrap(
                angles.right_ascension_from_sat_to_sun
            ),
        )

    def to_numba(self):
        """Convert to numba-compatible type."""
        return _OrbitPrediction(
            prediction_times=self.prediction_times,
            angles=self.angles,
            sat_position=self.sat_position,
            nutation_precession=self.nutation_precession,
        )


def get_lons_lats(lines, pixels, nav_params):
    """Compute lon/lat coordinates given VISSR image coordinates.

    Args:
        lines: VISSR image lines
        pixels: VISSR image pixels
        nav_params: Image navigation parameters
    """
    pixels_2d, lines_2d = da.meshgrid(pixels, lines)
    lons, lats = da.map_blocks(
        _get_lons_lats_numba,
        lines_2d,
        pixels_2d,
        nav_params=_make_nav_params_numba_compatible(nav_params),
        **_get_map_blocks_kwargs(pixels_2d.chunks)
    )
    return lons, lats


def _make_nav_params_numba_compatible(nav_params):
    predicted = PredictedNavigationParameters(
        attitude=nav_params.predicted.attitude.to_numba(),
        orbit=nav_params.predicted.orbit.to_numba()
    )
    return ImageNavigationParameters(nav_params.static, predicted)


def _get_map_blocks_kwargs(chunks):
    # Get keyword arguments for da.map_blocks, so that it can be used
    # with a function that returns two arguments.
    return {
        "new_axis": 0,
        "chunks": (2,) + chunks,
        "dtype": np.float32,
    }


@numba.njit
def _get_lons_lats_numba(lines_2d, pixels_2d, nav_params):
    shape = lines_2d.shape
    lons = np.zeros(shape, dtype=np.float32)
    lats = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel = Pixel(lines_2d[i, j], pixels_2d[i, j])
            nav_params_pix = _get_pixel_navigation_parameters(
                pixel, nav_params
            )
            lon, lat = get_lon_lat(pixel, nav_params_pix)
            lons[i, j] = lon
            lats[i, j] = lat
    # Stack lons and lats because da.map_blocks doesn't support multiple
    # return values.
    return np.stack((lons, lats))


@numba.njit
def _get_pixel_navigation_parameters(point, im_nav_params):
    obs_time = get_observation_time(point, im_nav_params.static.scan_params)
    attitude, orbit = interpolate_navigation_prediction(
        attitude_prediction=im_nav_params.predicted.attitude,
        orbit_prediction=im_nav_params.predicted.orbit,
        observation_time=obs_time
    )
    return PixelNavigationParameters(
        attitude=attitude,
        orbit=orbit,
        proj_params=im_nav_params.static.proj_params
    )


@numba.njit
def get_observation_time(point, scan_params):
    """Calculate observation time of a VISSR pixel."""
    relative_time = _get_relative_observation_time(point, scan_params)
    return scan_params.start_time_of_scan + relative_time


@numba.njit
def _get_relative_observation_time(point, scan_params):
    line, pixel = point
    pixel = pixel + 1
    line = line + 1
    spinning_freq = 1440 * scan_params.spinning_rate
    line_step = np.floor((line - 1) / scan_params.num_sensors)
    pixel_step = (scan_params.sampling_angle * pixel) / (2 * np.pi)
    return (line_step + pixel_step) / spinning_freq


@numba.njit
def interpolate_navigation_prediction(
    attitude_prediction, orbit_prediction, observation_time
):
    """Interpolate predicted navigation parameters."""
    attitude = interpolate_attitude_prediction(attitude_prediction, observation_time)
    orbit = interpolate_orbit_prediction(orbit_prediction, observation_time)
    return attitude, orbit


@numba.njit
def get_lon_lat(pixel, nav_params):
    """Get longitude and latitude coordinates for a given image pixel.

    Args:
        pixel (Pixel): Point in image coordinates.
        nav_params (PixelNavigationParameters): Navigation parameters for a
            single pixel.
    Returns:
        Longitude and latitude in degrees.
    """
    scan_angles = transform_image_coords_to_scanning_angles(
        pixel,
        nav_params.proj_params.image_offset,
        nav_params.proj_params.scanning_angles
    )
    view_vector_sat = transform_scanning_angles_to_satellite_coords(
        scan_angles,
        nav_params.proj_params.scanning_angles.misalignment
    )
    view_vector_earth_fixed = transform_satellite_to_earth_fixed_coords(
        view_vector_sat,
        nav_params.orbit,
        nav_params.attitude
    )
    point_on_earth = intersect_with_earth(
        view_vector_earth_fixed,
        nav_params.orbit.sat_position,
        nav_params.proj_params.earth_ellipsoid
    )
    lon, lat = transform_earth_fixed_to_geodetic_coords(
        point_on_earth, nav_params.proj_params.earth_ellipsoid.flattening
    )
    return lon, lat


@numba.njit
def transform_image_coords_to_scanning_angles(point, image_offset, scanning_angles):
    """Transform image coordinates to scanning angles.

    Args:
        point (Pixel): Point in image coordinates.
        image_offset (ImageOffset): Image offset.
        scanning_angles (ScanningAngles): Scanning angles.
    Returns:
        Scanning angles (x, y) at the pixel center (rad).
    """
    line_offset = image_offset.line_offset
    pixel_offset = image_offset.pixel_offset
    stepping_angle = scanning_angles.stepping_angle
    sampling_angle = scanning_angles.sampling_angle
    x = sampling_angle * (point.pixel + 1 - pixel_offset)
    y = stepping_angle * (point.line + 1 - line_offset)
    return Vector2D(x, y)


@numba.njit
def transform_scanning_angles_to_satellite_coords(angles, misalignment):
    """Transform scanning angles to satellite angular momentum coordinates.

    Args:
        angles (Vector2D): Scanning angles in radians.
        misalignment: Misalignment matrix (3x3)

    Returns:
        View vector (Vector3D) in satellite angular momentum coordinates.
    """
    x, y = angles.x, angles.y
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    view = Vector3D(np.cos(y), 0.0, np.sin(y))

    # Correct for misalignment
    view = matrix_vector(misalignment, view)

    # Rotate around z-axis
    return Vector3D(
        cos_x * view.x - sin_x * view.y,
        sin_x * view.x + cos_x * view.y,
        view.z
    )


@numba.njit
def transform_satellite_to_earth_fixed_coords(
    point,
    orbit,
    attitude
):
    """Transform from earth-fixed to satellite angular momentum coordinates.

    Args:
        point (Vector3D): Point in satellite angular momentum coordinates.
        orbit (Orbit): Orbital parameters
        attitude (Attitude): Attitude parameters
    Returns:
        Point (Vector3D) in earth-fixed coordinates.
    """
    unit_vector_z = _get_satellite_unit_vector_z(attitude, orbit)
    unit_vector_x = _get_satellite_unit_vector_x(unit_vector_z, attitude, orbit)
    unit_vector_y = _get_satellite_unit_vector_y(unit_vector_x, unit_vector_z)
    return _get_earth_fixed_coords(
        point,
        unit_vector_x,
        unit_vector_y,
        unit_vector_z
    )


@numba.njit
def _get_satellite_unit_vector_z(attitude, orbit):
    v1950 = _get_satellite_z_axis_1950(
        attitude.angle_between_sat_spin_and_z_axis,
        attitude.angle_between_sat_spin_and_yz_plane
    )
    vcorr = _correct_nutation_precession(
        v1950,
        orbit.nutation_precession
    )
    return _rotate_to_greenwich(
        vcorr,
        orbit.angles.greenwich_sidereal_time
    )


@numba.njit
def _get_satellite_z_axis_1950(
    angle_between_sat_spin_and_z_axis,
    angle_between_sat_spin_and_yz_plane
):
    """Get satellite z-axis (spin) in mean of 1950 coordinates."""
    alpha = angle_between_sat_spin_and_z_axis
    delta = angle_between_sat_spin_and_yz_plane
    cos_delta = np.cos(delta)
    return Vector3D(
        x=np.sin(delta),
        y=-cos_delta * np.sin(alpha),
        z=cos_delta * np.cos(alpha)
    )


@numba.njit
def _correct_nutation_precession(vector, nutation_precession):
    return matrix_vector(nutation_precession, vector)


@numba.njit
def _rotate_to_greenwich(vector, greenwich_sidereal_time):
    cos_sid = np.cos(greenwich_sidereal_time)
    sin_sid = np.sin(greenwich_sidereal_time)
    rotated = Vector3D(
        x=cos_sid * vector.x + sin_sid * vector.y,
        y=-sin_sid * vector.x + cos_sid * vector.y,
        z=vector.z
    )
    return normalize_vector(rotated)


@numba.njit
def _get_satellite_unit_vector_x(unit_vector_z, attitude, orbit):
    sat_sun_vec = _get_vector_from_satellite_to_sun(
        orbit.angles.declination_from_sat_to_sun,
        orbit.angles.right_ascension_from_sat_to_sun
    )
    return _get_unit_vector_x(
        sat_sun_vec,
        unit_vector_z,
        attitude.angle_between_earth_and_sun
    )


@numba.njit
def _get_vector_from_satellite_to_sun(
    declination_from_sat_to_sun,
    right_ascension_from_sat_to_sun
):
    declination = declination_from_sat_to_sun
    right_ascension = right_ascension_from_sat_to_sun
    cos_declination = np.cos(declination)
    return Vector3D(
        x=cos_declination * np.cos(right_ascension),
        y=cos_declination * np.sin(right_ascension),
        z=np.sin(declination)
    )


@numba.njit
def _get_unit_vector_x(
    sat_sun_vec,
    unit_vector_z,
    angle_between_earth_and_sun

):
    beta = angle_between_earth_and_sun
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    cross1 = _get_uz_cross_satsun(unit_vector_z, sat_sun_vec)
    cross2 = cross_product(cross1, unit_vector_z)
    unit_vector_x = Vector3D(
        x=sin_beta * cross1.x + cos_beta * cross2.x,
        y=sin_beta * cross1.y + cos_beta * cross2.y,
        z=sin_beta * cross1.z + cos_beta * cross2.z
    )
    return normalize_vector(unit_vector_x)


@numba.njit
def _get_uz_cross_satsun(unit_vector_z, sat_sun_vec):
    res = cross_product(unit_vector_z, sat_sun_vec)
    return normalize_vector(res)


@numba.njit
def _get_satellite_unit_vector_y(unit_vector_x, unit_vector_z):
    res = cross_product(unit_vector_z, unit_vector_x)
    return normalize_vector(res)


@numba.njit
def _get_earth_fixed_coords(point, unit_vector_x, unit_vector_y, unit_vector_z):
    ux, uy, uz = unit_vector_x, unit_vector_y, unit_vector_z
    # Multiply with matrix of satellite unit vectors [ux, uy, uz]
    return Vector3D(
        x=ux.x * point.x + uy.x * point.y + uz.x * point.z,
        y=ux.y * point.x + uy.y * point.y + uz.y * point.z,
        z=ux.z * point.x + uy.z * point.y + uz.z * point.z
    )


@numba.njit
def intersect_with_earth(view_vector, sat_pos, ellipsoid):
    """Intersect instrument viewing vector with the earth's surface.

    Reference: Appendix E, section 2.11 in the GMS user guide.

    Args:
        view_vector (Vector3D): Instrument viewing vector in earth-fixed
            coordinates.
        sat_pos (Vector3D): Satellite position in earth-fixed coordinates.
        ellipsoid (EarthEllipsoid): Earth ellipsoid.
    Returns:
        Intersection (Vector3D) with the earth's surface.
    """
    distance = _get_distance_to_intersection(view_vector, sat_pos, ellipsoid)
    return Vector3D(
        sat_pos.x + distance * view_vector.x,
        sat_pos.y + distance * view_vector.y,
        sat_pos.z + distance * view_vector.z
    )


@numba.njit
def _get_distance_to_intersection(view_vector, sat_pos, ellipsoid):
    """Get distance to intersection with the earth.

    If the instrument is pointing towards the earth, there will be two
    intersections with the surface. Choose the one on the instrument-facing
    side of the earth.
    """
    d1, d2 = _get_distances_to_intersections(view_vector, sat_pos, ellipsoid)
    return min(d1, d2)


@numba.njit
def _get_distances_to_intersections(view_vector, sat_pos, ellipsoid):
    """Get distances to intersections with the earth's surface.

    Returns:
        Distances to two intersections with the surface.
    """
    a, b, c = _get_abc_helper(view_vector, sat_pos, ellipsoid)
    tmp = np.sqrt((b**2 - a * c))
    dist_1 = (-b + tmp) / a
    dist_2 = (-b - tmp) / a
    return dist_1, dist_2


@numba.njit
def _get_abc_helper(view_vector, sat_pos, ellipsoid):
    """Get a,b,c helper variables.

    Reference: Appendix E, Equation (26) in the GMS user guide.
    """
    flat2 = (1 - ellipsoid.flattening) ** 2
    ux, uy, uz = view_vector.x, view_vector.y, view_vector.z
    x, y, z = sat_pos.x, sat_pos.y, sat_pos.z
    a = flat2 * (ux ** 2 + uy ** 2) + uz ** 2
    b = flat2 * (x * ux + y * uy) + z * uz
    c = flat2 * (x ** 2 + y ** 2 - ellipsoid.equatorial_radius ** 2) + z ** 2
    return a, b, c


@numba.njit
def transform_earth_fixed_to_geodetic_coords(point, earth_flattening):
    """Transform from earth-fixed to geodetic coordinates.

    Args:
        point (Vector3D): Point in earth-fixed coordinates.
        earth_flattening: Flattening of the earth.

    Returns:
        Geodetic longitude and latitude (degrees).
    """
    x, y, z = point.x, point.y, point.z
    f = earth_flattening
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, ((1 - f) ** 2 * np.sqrt(x**2 + y**2)))
    return np.rad2deg(lon), np.rad2deg(lat)


@numba.njit
def interpolate_orbit_prediction(orbit_prediction, observation_time):
    """Interpolate orbit prediction at the given observation time."""
    angles = _interpolate_orbit_angles(observation_time, orbit_prediction)
    sat_position = _interpolate_sat_position(observation_time, orbit_prediction)
    nutation_precession = interpolate_nearest(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.nutation_precession,
    )
    return Orbit(
        angles=angles,
        sat_position=sat_position,
        nutation_precession=nutation_precession,
    )


@numba.njit
def _interpolate_orbit_angles(observation_time, orbit_prediction):
    sidereal_time = interpolate_angles(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.angles.greenwich_sidereal_time,
    )
    declination = interpolate_angles(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.angles.declination_from_sat_to_sun,
    )
    right_ascension = interpolate_angles(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.angles.right_ascension_from_sat_to_sun,
    )
    return OrbitAngles(
        greenwich_sidereal_time=sidereal_time,
        declination_from_sat_to_sun=declination,
        right_ascension_from_sat_to_sun=right_ascension,
    )


@numba.njit
def _interpolate_sat_position(observation_time, orbit_prediction):
    x = interpolate_continuous(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.sat_position.x,
    )
    y = interpolate_continuous(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.sat_position.y,
    )
    z = interpolate_continuous(
        observation_time,
        orbit_prediction.prediction_times,
        orbit_prediction.sat_position.z,
    )
    return Vector3D(x, y, z)


@numba.njit
def interpolate_attitude_prediction(attitude_prediction, observation_time):
    """Interpolate attitude prediction at given observation time."""
    angle_between_earth_and_sun = interpolate_angles(
        observation_time,
        attitude_prediction.prediction_times,
        attitude_prediction.attitude.angle_between_earth_and_sun,
    )
    angle_between_sat_spin_and_z_axis = interpolate_angles(
        observation_time,
        attitude_prediction.prediction_times,
        attitude_prediction.attitude.angle_between_sat_spin_and_z_axis,
    )
    angle_between_sat_spin_and_yz_plane = interpolate_angles(
        observation_time,
        attitude_prediction.prediction_times,
        attitude_prediction.attitude.angle_between_sat_spin_and_yz_plane,
    )
    return Attitude(
        angle_between_earth_and_sun,
        angle_between_sat_spin_and_z_axis,
        angle_between_sat_spin_and_yz_plane,
    )


@numba.njit
def interpolate_continuous(x, x_sample, y_sample):
    """Linear interpolation of continuous quantities.

    Numpy equivalent would be np.interp(..., left=np.nan, right=np.nan), but
    numba currently doesn't support those keyword arguments.
    """
    try:
        return _interpolate(x, x_sample, y_sample)
    except Exception:
        # Numba cannot distinguish exception types
        return np.nan


@numba.njit
def _interpolate(x, x_sample, y_sample):
    i = _find_enclosing_index(x, x_sample)
    offset = y_sample[i]
    x_diff = x_sample[i + 1] - x_sample[i]
    y_diff = y_sample[i + 1] - y_sample[i]
    slope = y_diff / x_diff
    dist = x - x_sample[i]
    return offset + slope * dist


@numba.njit
def _find_enclosing_index(x, x_sample):
    """Find where x_sample encloses x."""
    for i in range(len(x_sample) - 1):
        if x_sample[i] <= x < x_sample[i + 1]:
            return i
    raise Exception("x not enclosed by x_sample")


@numba.njit
def interpolate_angles(x, x_sample, y_sample):
    """Linear interpolation of angles.

    Requires 2-pi periodicity to be unwrapped before (for
    performance reasons). Interpolated angles are wrapped
    back to [-pi, pi] to restore periodicity.
    """
    return _wrap_2pi(interpolate_continuous(x, x_sample, y_sample))


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


@numba.njit
def matrix_vector(m, v):
    """Multiply (3,3)-matrix and Vector3D."""
    x = m[0, 0] * v.x + m[0, 1] * v.y + m[0, 2] * v.z
    y = m[1, 0] * v.x + m[1, 1] * v.y + m[1, 2] * v.z
    z = m[2, 0] * v.x + m[2, 1] * v.y + m[2, 2] * v.z
    return Vector3D(x, y, z)


@numba.njit
def cross_product(a, b):
    """Compute vector product a x b."""
    return Vector3D(
        x=a.y * b.z - a.z * b.y,
        y=a.z * b.x - a.x * b.z,
        z=a.x * b.y - a.y * b.x
    )


@numba.njit
def normalize_vector(v):
    """Normalize a Vector3D."""
    norm = np.sqrt(v.x**2 + v.y**2 + v.z**2)
    return Vector3D(
        v.x / norm,
        v.y / norm,
        v.z / norm
    )
