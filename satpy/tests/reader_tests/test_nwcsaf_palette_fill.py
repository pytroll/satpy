"""Regression tests for separately stored NWCSAF/GEO palette fill colours."""

# NOTE: tmp_path is supplied by pytest. Shared Satpy configuration and download
# guards are supplied by satpy/tests/conftest.py.

import dask.array as da
import h5netcdf
import numpy as np
import pytest
from trollimage.xrimage import XRImage

from satpy.enhancements.colormap import colorize, palettize
from satpy.readers.nwcsaf_nc import NcNWCSAF
from satpy.tests.utils import assert_maximum_dask_computes


@pytest.fixture
def cmic_reader(tmp_path):
    """Create tiny real GEO files with continuous or categorical palettes."""
    readers = []

    def create(*, phase=False, new_format=True):
        name = "cmic_phase" if phase else "cmic_lwp"
        dtype = np.uint8 if phase else np.uint16
        fill = 255 if phase else 65535
        data = np.array([[1, 2], [5, fill]] if phase else [[0, 1000], [4000, fill]], dtype=dtype)
        fill_color = np.full(3, 100 if phase else 0, dtype=np.uint8)
        colors = np.column_stack((np.arange(255), np.zeros(255), np.arange(254, -1, -1))).astype(np.uint8)
        stored_colors = colors if new_format else np.vstack((colors, fill_color))
        filename = tmp_path / f"cmic_{len(readers)}.nc"
        with h5netcdf.File(filename, "w") as nc:
            nc.dimensions = {"ny": 2, "nx": 2, "pal_colors": len(stored_colors), "pal_RGB": 3}
            if new_format:
                nc.dimensions["time"] = 1
            nc.attrs.update({
                "source": "NWC/GEO version v2025.2" if new_format else "NWC/GEO version v2021.3",
                "satellite_identifier": "MSG3",
                "time_coverage_start": "2026-11-21T15:00:00Z",
                "time_coverage_end": "2026-11-21T15:15:00Z",
            })
            dimensions = ("time", "ny", "nx") if new_format else ("ny", "nx")
            var = nc.create_variable(name, dimensions, dtype, fillvalue=dtype(fill))
            var[:] = data[np.newaxis] if new_format else data
            var.attrs["valid_range"] = np.array([1, 5] if phase else [0, 4000], dtype=dtype)
            var.attrs["ancillary_variables"] = f"{name}_pal"
            if not phase:
                var.attrs.update({"scale_factor": np.float32(0.001), "add_offset": np.float32(0)})
            palette = nc.create_variable(f"{name}_pal", ("pal_colors", "pal_RGB"), np.uint8)
            palette[:] = stored_colors
            palette.attrs.update({"valid_range": np.array([0, 255], dtype=np.uint8), "colormodel": "RGB"})
            if new_format:
                palette.attrs["fill_value_color"] = fill_color
        reader = NcNWCSAF(filename, {}, {})
        readers.append(reader)
        return reader, name, stored_colors, fill_color

    yield create
    for reader in readers:
        reader.nc.close()


def _load_data_and_palette(reader, name):
    data = reader.get_dataset({"name": name}, {"name": name})
    palette_name = f"{name}_pal"
    palette = reader.get_dataset({"name": palette_name}, {"name": palette_name})
    # The YAML reader normally attaches these identifiers and ancillary objects.
    palette.attrs["name"] = palette_name
    data.attrs["ancillary_variables"] = [palette]
    return data, palette


@pytest.mark.parametrize("phase", [False, True])
def test_geo_palette_adds_fill_without_changing_valid_colours(cmic_reader, phase):
    """Keep valid control points and preserve the original file on repeated reads."""
    reader, name, colors, fill_color = cmic_reader(phase=phase)
    with assert_maximum_dask_computes(max_computes=0):
        _, palette = _load_data_and_palette(reader, name)
    assert isinstance(palette.data, da.Array)
    assert palette.dtype == np.uint8
    assert palette.dims == ("pal_colors", "pal_RGB")
    assert palette.shape == (256, 3)
    np.testing.assert_array_equal(palette[:-1], colors)
    np.testing.assert_array_equal(palette[-1], fill_color)
    expected_values = np.arange(255) if phase else np.linspace(0, 4, 255)
    np.testing.assert_allclose(palette.attrs["palette_meanings"][:-1], expected_values)
    assert palette.attrs["palette_meanings"][-1] > expected_values[-1]
    _, second = _load_data_and_palette(reader, name)
    np.testing.assert_array_equal(second, palette)
    assert reader.nc[f"{name}_pal"].shape == (255, 3)
    assert "palette_meanings" not in reader.nc[f"{name}_pal"].attrs


@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("operation", [palettize, colorize])
def test_geo_palette_distinguishes_valid_maximum_and_missing(cmic_reader, phase, operation):
    """Exercise the actual enhancement and RGB finalization, including float32."""
    reader, name, colors, fill_color = cmic_reader(phase=phase)
    data, palette = _load_data_and_palette(reader, name)
    if not phase:
        assert data.dtype == np.float32
        np.testing.assert_allclose(data, [[0, 1], [4, np.nan]])
    image = XRImage(data)
    operation(image, palettes=[{"dataset": palette.attrs["name"], "color_scale": 255}])
    finalized, mode = image.finalize()
    assert mode == "RGBA"
    rgba = finalized.transpose("y", "x", "bands").values
    np.testing.assert_array_equal(rgba[0, 0, :3], colors[1 if phase else 0])
    np.testing.assert_array_equal(rgba[1, 0, :3], colors[5 if phase else -1])
    assert rgba[1, 0, 3] == 255
    if operation is palettize:
        np.testing.assert_array_equal(rgba[1, 1, :3], fill_color)
        assert rgba[1, 1, 3] == 255
    else:
        assert rgba[1, 1, 3] == 0


def test_legacy_geo_palette_is_unchanged(cmic_reader):
    """Retain the old palette layout when the separate fill attribute is absent."""
    reader, name, colors, _ = cmic_reader(new_format=False)
    _, palette = _load_data_and_palette(reader, name)
    np.testing.assert_array_equal(palette, colors)
    assert "palette_meanings" not in palette.attrs
