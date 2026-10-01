"""Calibration, space masking and area definition shared by the GMS-5 and GMS-1-4 VISSR readers.

The two readers decode different archive formats (see ``gms5_vissr_format``
and ``gms4_vissr_format``), but these parts work the same way. Format-specific
parts (where a value is stored, which channel it belongs to) stay in the
readers. Navigation is shared as well, see
:mod:`satpy.readers.gms.gms_vissr_navigation`.
"""

import numpy as np

import satpy.readers.core._geos_area as geos_area
import satpy.readers.gms.gms_vissr_navigation as nav

FILL_VALUE = -1  # scanline not intersecting the earth


def _lookup_calibration_value(block, lut, mask):
    """Look up calibrated values for a block of counts.

    Module-level (not a closure) so dask can pickle it for distributed schedulers.
    """
    if mask is not None:
        block = block.astype(np.int64) & mask
    return lut[block]


class Calibrator:
    """Calibrate VISSR counts to unnormalized reflectance (%) or brightness temperature (K).

    Reference: Section 2.2 in the VISSR User Guide.
    """

    def __init__(self, calib_table, mask=None, percent_calibrations=("unnormalized_reflectance",)):
        """Initialize the calibrator.

        Args:
            calib_table: Calibration table (lookup table indexed by counts).
            mask: Optional bit mask applied to the counts before the lookup,
                for channels with fewer bits than the storage type (e.g. 0x3F
                for 6-bit VIS counts).
            percent_calibrations: Calibration levels whose table values are
                fractions and need to be converted to percent.
        """
        self._calib_table = calib_table
        self._mask = mask
        self._percent_calibrations = percent_calibrations

    def calibrate(self, counts, calibration):
        """Transform counts (a dask array) to the given calibration level."""
        if calibration == "counts":
            return counts
        res = self._calibrate(counts)
        return self._postproc(res, calibration)

    def _calibrate(self, counts):
        return counts.map_blocks(
            _lookup_calibration_value,
            lut=self._calib_table,
            mask=self._mask,
            dtype=np.float32,
            meta=np.array((), dtype=np.float32),
        )

    def _postproc(self, res, calibration):
        if calibration in self._percent_calibrations:
            return res * 100
        return res


def scale_earth_edges(edges, ratio, fill_value=FILL_VALUE):
    """Scale earth edges to a different pixel density, leaving fill values untouched.

    VIS data contains the earth edges of the IR channel, so they have to be
    scaled by the VIS/IR pixel density ratio.

    Args:
        edges: First or last earth pixel in each scanline.
        ratio: Pixel density ratio (target channel / channel the edges refer to).
        fill_value: Fill value for scanlines not intersecting the earth.
    """
    return np.where(edges != fill_value, (edges * ratio).astype(np.int32), edges)


def get_earth_mask(shape, earth_edges, fill_value=FILL_VALUE):
    """Get binary mask where True/False indicates earth/space.

    Args:
        shape: Image shape
        earth_edges: First and last earth pixel in each scanline
        fill_value: Fill value for scanlines not intersecting the earth.
    """
    first_earth_pixels, last_earth_pixels = (np.asarray(edges) for edges in earth_edges)
    intersects_earth = (first_earth_pixels != fill_value) & (last_earth_pixels != fill_value)
    # Clamp each edge only on the side where it can leave the image. Lines
    # with first > last end up empty.
    first = np.maximum(first_earth_pixels, 0)
    last = np.minimum(last_earth_pixels, shape[1] - 1)
    pixels = np.arange(shape[1])
    return intersects_earth[:, None] & (pixels[None, :] >= first[:, None]) & (pixels[None, :] <= last[:, None])


class AreaDefEstimator:
    """Estimate a full disk area definition with uniform sampling for VISSR images."""

    def __init__(self, platform_name, sensor_name, ssp_lon, satellite_height):
        """Initialize the area definition estimator.

        Args:
            platform_name: Platform name, used for naming the area.
            sensor_name: Sensor name, used for naming the area.
            ssp_lon: Nominal sub-satellite longitude. Nominal parameters are
                used to make the area definition as constant as possible.
            satellite_height: Nominal satellite altitude.
        """
        self.platform_name = platform_name
        self.sensor_name = sensor_name
        self.ssp_lon = ssp_lon
        self.satellite_height = satellite_height

    def get_area_def_uniform_sampling(self, dataset_id, size, stepping_angle):
        """Get full disk area definition with uniform sampling.

        Args:
            dataset_id: ID of the corresponding dataset.
            size: Number of lines and pixels of the (square) area.
            stepping_angle: Angle between two scanlines in radians. It is
                applied to the horizontal dimension as well to obtain uniform
                sampling.
        """
        proj_dict = {}
        proj_dict.update(self._get_name_dict(dataset_id))
        proj_dict.update(self._get_proj4_dict())
        proj_dict.update(self._get_shape_dict(size, stepping_angle))
        extent = geos_area.get_area_extent(proj_dict)
        return geos_area.get_area_definition(proj_dict, extent)

    def _get_name_dict(self, dataset_id):
        name_dict = geos_area.get_geos_area_naming(
            {
                "platform_name": self.platform_name,
                "instrument_name": self.sensor_name,
                "service_name": "western-pacific",
                "service_desc": "Western Pacific",
                "resolution": dataset_id.get("resolution"),
            }
        )
        return {
            "a_name": name_dict["area_id"],
            "p_id": name_dict["area_id"],
            "a_desc": name_dict["description"],
        }

    def _get_proj4_dict(self):
        return {
            "ssp_lon": self.ssp_lon,
            "a": nav.EARTH_EQUATORIAL_RADIUS,
            "b": nav.EARTH_POLAR_RADIUS,
            "h": self.satellite_height,
        }

    @staticmethod
    def _get_shape_dict(size, stepping_angle):
        line_pixel_offset = 0.5 * size
        lfac_cfac = geos_area.sampling_to_lfac_cfac(stepping_angle)
        return {
            "nlines": size,
            "ncols": size,
            "lfac": lfac_cfac,
            "cfac": lfac_cfac,
            "coff": line_pixel_offset,
            "loff": line_pixel_offset,
            "scandir": "N2S",
        }
