"""Reader for SMS and GOES 1-7 VISSR/VAS data in AREA format.

Introduction
------------

This reader reads and calibrates the Visible and Infrared Spin Scan Radiometer
(VISSR) data within the Synchronous Meteorological Satellite (SMS) 1-2 and the
Geostationary Operational Environmental Satellite (GOES) 1-3, along with the
VISSR Atmospheric Sounder (VAS) within GOES 4-7. The data can be ordered from
NOAA CLASS in the McIDAS AREA format.

.. note ::
   SMS-1 and GOES-1 pre-1980 do not have navigation parameters in the AREA files,
   in which case this reader will fail. CLASS does provide these parameters in
   the Pre-GVAR order page; users are encouraged to add them in the AREA file
   first if navigation data is available before reading it.



Oversampling
------------

GOES VISSR and VAS sample the ground more finely along a scan line than
between scan lines. At nadir an IR file has about 6.9 km between neighboring
lines but only about 3.0 km between neighboring elements (an AREA file with
``line_res=8`` and ``elem_res=4``), so the grid is oversampled by roughly
a factor of 2.3 in the along-line direction. VIS is nearly isotropic, with
about 0.86 km between lines and 0.75 km between elements. The spacing
comes from the scan angle per line and element in the navigation block,
scaled by the directory's line and element resolution factors.

Resampling the swath directly therefore oversamples the IR data. To avoid
that, every dataset carries an ``area_def_uniform_sampling`` attribute:
a full-disk geostationary grid centered on the sub-satellite point with
square pixels at the nadir line spacing. The longitude and latitude
datasets carry the same spacing as their ``resolution`` attribute.


Calibration
-----------

There are different calibrations for each file, depending on the file's mode:

Mode A (single band, ``VISR``)
    IR: ``counts`` and ``brightness_temperature``. The raw counts are
    already on the documented VISSR brightness scale, so
    T = 418 - B for B >= 176 and T = 330 - B / 2 below that (kelvin).
    VIS: ``counts`` and ``unnormalized_reflectance``. The counts are
    6-bit values stored in a byte (multiples of 4, at most 252) and are
    linear in albedo. ``unnormalized_reflectance`` is counts / 252 * 100,
    in percent, with no division by the cosine of the solar zenith angle.
    No absolute VIS calibration is published for these satellites, so
    treat it as relative reflectance.

Mode AA (``VAS``, GOES-4 to GOES-6)
    ``brightness_temperature`` only. Each scan line carries up to 13
    calibration entries (channel code, spin count, delta-F code and a
    space-view offset). The offset is subtracted from the counts and the
    result is scaled to radiance, then converted to temperature with the
    Planck inversion constants McIDAS uses for all Mode AA data.

Mode AAA (``AAA``, GOES-6 and GOES-7 VAS)
    ``counts``, ``radiance`` and ``brightness_temperature``. Radiance is
    R = (AB2 * P / 32 - AB1) / 2**(15 - IFAB) with the coefficients from
    the file's calibration block. Temperature uses the Planck constants
    McIDAS-X has for each satellite. Each line's channels are identified
    from the line's own band list, since order and presence can vary
    from line to line. Channel 13 reuses channel 8's coefficients.

Radiance units are given as ``mW m-2 sr-1 (cm-1)-1`` but have not been
independently verified. The formulas follow the NOAA/NESDIS GOES Mode AAA
format description and the McIDAS documentation; the Planck constants
were supplied by SSEC.


References:
===========

- `[VISSR]`_ VISSR Digital Archive Users' Guide
- `[VAS]`_ VAS Description and Final Report
- `[MODE]`_ GOES Mode Formats
- `[AREA]`_ SSEC New AREA Format
- `[MCIDAS]`_ McIDAS Programmers' Manual
- `[GLOSSARY]`_ Pre-GVAR Glossary


.. _[VISSR]: https://www.ncei.noaa.gov/nerms/api/document/2042B0E3E70F4E69B51EBC340A35E5D2/download
.. -[VAS]: https://ntrs.nasa.gov/api/citations/19860004398/downloads/19860004398.pdf
.. _[MODE]: https://www.ncei.noaa.gov/access/search/publications/satellite-documentation-publication/?title=mode+vissr
.. _[AREA]: https://www.ssec.wisc.edu/mcidas/doc/misc_doc/area2.html
.. _[MCIDAS]: https://www.ssec.wisc.edu/mcidas/doc/prog_man/2015/prog_man.pdf
.. _[GLOSSARY]: https://www.class.noaa.gov/glossary/PRE_GVAR.htm
"""
import datetime as dt

import dask.array as da
import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.goes_pregvar.goes_pregvar_base import AreaFile
from satpy.utils import get_legacy_chunk_size

CHUNK_SIZE = get_legacy_chunk_size()

# WGS84 ellipsoid for the uniform-sampling geos target area
_GEOS_A = 6378137.0
_GEOS_B = 6356752.314245


class GOESPreGVARAreaFileHandler(BaseFileHandler):
    """File handler for SMS and GOES 1-7 VISSR/VAS data in AREA format."""

    def __init__(self, filename, filename_info, filetype_info, mask_space=True):
        """Open the file and parse its directory, navigation and calibration blocks.

        The navigation uses GOESNav's Kepler orbit model.

        Args:
            filename: Path of the AREA file.
            filename_info: Information parsed from the file name.
            filetype_info: File type information from the YAML file.
            mask_space: Mask pixels whose line of sight misses the earth ellipsoid
                (deep space / limb pixels) to NaN. The mask is derived from the same
                per-pixel navigation (get_lonlats()) already used for the
                longitude/latitude datasets -- a pixel is "space" here if its own
                earth-location computation returned NaN. This also means turning off
                navigation isn't possible while masking is on: computing the mask
                requires running get_lonlats() regardless of whether you've actually
                requested the longitude/latitude datasets.
        """
        super().__init__(filename, filename_info, filetype_info)
        self._mask_space = mask_space
        self.area_file = AreaFile(self.filename)
        self._lonlat_cache = None
        self._earth_mask_cache = None
        self._nadir_sampling_cache = None

    @property
    def start_time(self):
        """Nominal start time of the scan."""
        d = self.area_file.dir
        # nominal_year is a 4-digit year (already century-resolved by
        # goes_pregvar_base's _yyddd_to_iso), nominal_doy is day-of-year.
        base = dt.datetime(d["nominal_year"], 1, 1) + dt.timedelta(
            days=d["nominal_doy"] - 1
        )
        hhmmss = d["nominal_time_hhmmss"]
        hh, mm, ss = hhmmss // 10000, (hhmmss // 100) % 100, hhmmss % 100
        return base + dt.timedelta(hours=hh, minutes=mm, seconds=ss)

    @property
    def end_time(self):
        """End time of the scan, equal to the start time.

        VISSR full-disk scans take several minutes to complete, but the AREA directory only
        carries one nominal timestamp -- same limitation the McIDAS format itself has.
        Report start==end rather than guessing a scan duration we can't verify.
        """
        return self.start_time

    def _get_nadir_sampling(self):
        if self._nadir_sampling_cache is None:
            self._nadir_sampling_cache = self.area_file.get_nadir_sampling()
        return self._nadir_sampling_cache

    def _get_area_def_uniform_sampling(self):
        """Create a full-disk geostationary grid with square pixels.

        The files sample the ground much more finely along a scan line
        than between lines (for IR roughly 3 km vs 6.9 km at nadir), so
        resampling the swath directly oversamples the along-line
        direction. This grid has the line spacing in both directions
        and is meant as a resampling target that does not oversample.
        """
        nad = self._get_nadir_sampling()
        height = nad["altitude"]
        half_extent = np.arcsin(_GEOS_A / (_GEOS_A + height)) * height
        size = int(round(2 * half_extent / nad["line_step"]))
        proj = {"proj": "geos", "lon_0": nad["sub_lon"], "h": height,
                "a": _GEOS_A, "b": _GEOS_B, "sweep": "y", "units": "m"}
        return AreaDefinition(
            area_id="goes_pregvar_uniform_sampling",
            description="Full disk, uniform sampling at the nadir line spacing",
            proj_id="geos",
            projection=proj,
            width=size,
            height=size,
            area_extent=(-half_extent, -half_extent, half_extent, half_extent),
        )

    def _get_lonlats(self):
        if self._lonlat_cache is None:
            lon, lat = self.area_file.get_lonlats()
            lon = da.from_array(lon.astype(np.float32), chunks=CHUNK_SIZE)
            lat = da.from_array(lat.astype(np.float32), chunks=CHUNK_SIZE)
            self._lonlat_cache = (lon, lat)
        return self._lonlat_cache

    def _vas_channel_available(self, ds_info):
        """Check if this file's calibration has coefficients for the dataset's vas_channel.

        For example, a 3-band full-disk AAA file's cal block only covers channels 7/8/10, while a
        12-band Dwell Sounding file's covers 1-12. Used so available_datasets() doesn't advertise
        a channel this particular file can't actually produce.
        """
        channel = ds_info.get("vas_channel")
        if channel is None:
            return True
        source_type = self.area_file.dir.get("source_type")
        if source_type == "AAA":
            return self._aaa_channel_available(channel)
        if source_type == "VAS":
            return self._mode_aa_channel_available(channel)
        return False

    def _aaa_channel_available(self, channel):
        """Check if the AAA calibration block has constants for ``channel`` (13 borrows channel 8's)."""
        cal = self.area_file.cal
        if cal is None:
            return False
        lookup_channel = 8 if channel == 13 else channel
        return lookup_channel in cal.get("iab", {})

    def _mode_aa_channel_available(self, channel):
        """Check if any line's Mode AA calibration entries cover IR band ``channel``."""
        mode_aa_cal = self.area_file.mode_aa_cal
        if channel == 13 or mode_aa_cal is None:
            return False
        bands_present = self.area_file._mode_aa_code_to_band(mode_aa_cal["channel"])
        return bool((bands_present == channel).any())

    def available_datasets(self, configured_datasets=None):
        """Report which datasets this file can actually produce.

        The generic 'IR' dataset is unavailable for AAA-format multiband files, and any VAS_CHn
        dataset is unavailable for single-band files, or for multiband files whose own calibration
        doesn't cover that particular channel. Without this override, Satpy lists 'IR' as loadable
        for multiband files and only fails with a traceback at scn.load() time, which is a
        confusing way to find out.
        """
        for is_avail, ds_info in (configured_datasets or []):
            yield self._dataset_availability(is_avail, ds_info), ds_info

    def _dataset_availability(self, is_avail, ds_info):
        """Return the availability of one configured dataset (None if another file type owns it)."""
        if is_avail is not None:
            return is_avail
        if not self.file_type_matches(ds_info["file_type"]):
            return is_avail
        is_vas_multiband = self.area_file.dir.get("source_type") in ("AAA", "VAS")
        if ds_info.get("name") == "IR" and is_vas_multiband:
            return False
        if ds_info.get("vas_channel") is not None:
            return is_vas_multiband and self._vas_channel_available(ds_info)
        return True

    def _get_earth_mask(self):
        """Get a boolean array that is True where the pixel is a real earth-viewing measurement.

        These are the pixels with finite lon/lat from the navigation. It is False where the
        line of sight misses the earth (deep space / limb).
        """
        if self._earth_mask_cache is None:
            lon, lat = self._get_lonlats()
            self._earth_mask_cache = ~(da.isnan(lon) | da.isnan(lat))
        return self._earth_mask_cache

    def get_dataset(self, dataset_id, dataset_info):
        """Load a dataset."""
        name = dataset_id["name"]
        if name in ("longitude", "latitude"):
            return self._get_coordinate(name, dataset_info)

        band = self.area_file.dir["band"]  # "VIS" or "IR"
        if dataset_info.get("band") != band:
            # This dataset entry doesn't match this file's band (e.g.
            # the YAML declares both an IR and a VIS dataset, but a
            # given .AREA_IR file only ever has IR data) -- Satpy
            # expects None back in that case, not an exception.
            return None

        calibration = dataset_id.get("calibration")
        if band == "IR":
            data, units = self._read_ir(calibration, dataset_info)
        else:
            data, units = self._read_vis(calibration)

        attrs = dict(dataset_info)
        attrs["units"] = attrs.get("units", units)
        attrs["platform_name"] = self.area_file.dir["satellite"]
        attrs["sensor"] = "goes_pregvar"
        attrs["area_def_uniform_sampling"] = self._get_area_def_uniform_sampling()
        data = da.from_array(np.asarray(data), chunks=CHUNK_SIZE)
        if self._mask_space:
            data = da.where(self._get_earth_mask(), data, np.float32(np.nan))
        return xr.DataArray(data, dims=("y", "x"), attrs=attrs)

    def _get_coordinate(self, name, dataset_info):
        lon, lat = self._get_lonlats()
        # nominal resolution (nadir line spacing) so pyresample can
        # estimate the swath's resolution
        attrs = dict(dataset_info, resolution=self._get_nadir_sampling()["line_step"])
        return xr.DataArray(lon if name == "longitude" else lat, dims=("y", "x"), attrs=attrs)

    def _read_ir(self, calibration, dataset_info):
        source_type = self.area_file.dir.get("source_type")
        if source_type == "VAS":
            return self._read_ir_mode_aa(calibration, dataset_info)
        if source_type == "AAA":
            return self._read_ir_aaa(calibration, dataset_info)
        return self._read_ir_mode_a(calibration)

    def _read_ir_mode_a(self, calibration):
        """Read a single-band file (Mode A), where no channel selection is involved."""
        if calibration == "counts":
            return self.area_file.read_data().astype(np.float32), "1"
        if calibration in (None, "brightness_temperature"):
            return self.area_file.read_ir_temperature(), "K"
        raise ValueError(f"Unsupported calibration {calibration!r} for Mode A-format IR band.")

    def _read_ir_mode_aa(self, calibration, dataset_info):
        """Read a Mode AA multiband file.

        Unlike AAA, channel selection here doesn't go through the band list
        at all -- read_ir_temperature_mode_aa() looks up which per-line calibration
        slot (of up to 13) matches the requested band itself, since Mode AA's
        channel-code table plays the same role AAA's band list does but with a
        different code->band aliasing.
        """
        channel = self._get_vas_channel(dataset_info)
        if calibration == "radiance":
            # Mode AA's radiance step is folded directly into
            # read_ir_temperature_mode_aa()'s Planck inversion
            # (unlike AAA, which has a standalone read_ir_radiance()) --
            # there's no separate radiance-only accessor to call here.
            raise ValueError(
                "Mode AA ('VAS')-format files don't expose a separate radiance calibration -- "
                "only brightness_temperature is available."
            )
        if calibration not in (None, "brightness_temperature"):
            raise ValueError(f"Unsupported calibration {calibration!r} for Mode AA-format IR band.")
        data = self.area_file.read_ir_temperature_mode_aa(channel)
        return self._pad_missing_lines(data), "K"

    def _pad_missing_lines(self, data):
        """Pad with NaN rows to the number of lines in the file.

        read_ir_temperature_mode_aa() can return fewer lines than the file's DATA-block line
        count when its internal per-line arrays disagree in length (see its n_common
        calculation). Padding keeps the shape the same as the lon/lat arrays and the
        earth-space mask.
        """
        expected_lines = self.area_file._actual_num_lines()
        if data.shape[0] >= expected_lines:
            return data
        pad = np.full((expected_lines - data.shape[0], data.shape[1]), np.nan, dtype=data.dtype)
        return np.concatenate([data, pad], axis=0)

    def _read_ir_aaa(self, calibration, dataset_info):
        """Read an AAA multiband file."""
        channel = self._get_vas_channel(dataset_info)
        # Per-line band list is REQUIRED -- it's the authoritative source per the SSEC doc,
        # which warns channel order/presence can vary line-to-line. Rather than guess a fixed
        # channel order (an earlier version of this reader was once wrong about it for a real
        # file), loading fails loudly if it isn't available.
        if self.area_file.band_list is None:
            raise ValueError(f"{self.area_file.path}: no per-line band list available for this AAA-format file.")
        if calibration == "radiance":
            # standard VAS radiance unit; unverified
            return self.area_file.read_ir_radiance(channel), "mW m-2 sr-1 (cm-1)-1"
        if calibration in (None, "brightness_temperature"):
            return self.area_file.read_ir_temperature_aaa(channel), "K"
        if calibration == "counts":
            return self.area_file.read_channel(channel)[0].astype(np.float32), "1"
        raise ValueError(f"Unsupported calibration {calibration!r} for AAA-format IR band.")

    @staticmethod
    def _get_vas_channel(dataset_info):
        """Get the physical VAS channel number from the dataset's YAML entry."""
        channel = dataset_info.get("vas_channel")
        if channel is None:
            raise ValueError(
                "Multiband files need a 'vas_channel' key in the dataset's YAML entry to know "
                "which physical VAS channel to read -- there's no way to infer this automatically."
            )
        return channel

    def _read_vis(self, calibration):
        if calibration in (None, "counts"):
            return self.area_file.read_vis_counts(calibration="counts"), "1"
        if calibration == "unnormalized_reflectance":
            return self.area_file.read_vis_counts(calibration="unnormalized_reflectance"), "%"
        raise ValueError(f"Unsupported calibration {calibration!r} for VIS band.")
