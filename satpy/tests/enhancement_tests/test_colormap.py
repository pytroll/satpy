# Copyright (c) 2017-2025 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Unit testing for enhancement color mapping functions."""

import contextlib
import os
from tempfile import NamedTemporaryFile
from unittest import mock

import numpy as np
import pytest

from satpy.enhancements.colormap import create_colormap

from .utils import create_ch1, create_ch2, create_rgb, run_and_check_enhancement


class TestEnhancementsConvolution:
    """Class for testing enhancements in satpy.enhancements.convolution module."""

    def setup_method(self):
        """Create test data used by every test."""
        self.ch1 = create_ch1()
        self.ch2 = create_ch2()
        self.rgb = create_rgb()

    def test_lookup(self):
        """Test the lookup enhancement function."""
        from satpy.enhancements.colormap import lookup
        expected = np.array([[
            [0., 0., 0., 0.333333, 0.705882],
            [1., 1., 1., 1., 1.]]])
        lut = np.arange(256.)
        run_and_check_enhancement(lookup, self.ch1, expected, luts=lut)

        expected = np.array([[[0., 0., 0., 0.333333, 0.705882],
                              [1., 1., 1., 1., 1.]],
                             [[0., 0., 0., 0.333333, 0.705882],
                              [1., 1., 1., 1., 1.]],
                             [[0., 0., 0., 0.333333, 0.705882],
                              [1., 1., 1., 1., 1.]]])
        lut = np.arange(256.)
        lut = np.vstack((lut, lut, lut)).T
        run_and_check_enhancement(lookup, self.rgb, expected, luts=lut)

    def test_colorize(self):
        """Test the colorize enhancement function."""
        from trollimage.colormap import brbg

        from satpy.enhancements.colormap import colorize
        expected = np.array([
            [[np.nan, 3.29411723e-01, 3.29411723e-01, 3.21825881e-08, 3.21825881e-08],
             [3.21825881e-08, 3.21825881e-08, 3.21825881e-08, 3.21825881e-08, 3.21825881e-08]],
            [[np.nan, 1.88235327e-01, 1.88235327e-01, 2.35294109e-01, 2.35294109e-01],
             [2.35294109e-01, 2.35294109e-01, 2.35294109e-01, 2.35294109e-01, 2.35294109e-01]],
            [[np.nan, 1.96078164e-02, 1.96078164e-02, 1.88235281e-01, 1.88235281e-01],
             [1.88235281e-01, 1.88235281e-01, 1.88235281e-01, 1.88235281e-01, 1.88235281e-01]]])
        run_and_check_enhancement(colorize, self.ch1, expected, palettes=brbg)

    def test_palettize(self):
        """Test the palettize enhancement function."""
        from trollimage.colormap import brbg

        from satpy.enhancements.colormap import palettize
        expected = np.array([[[10, 0, 0, 10, 10], [10, 10, 10, 10, 10]]])
        run_and_check_enhancement(palettize, self.ch1, expected, palettes=brbg)

    def test_merge_colormaps(self):
        """Test merging colormaps."""
        from trollimage.colormap import Colormap

        from satpy.enhancements.colormap import _merge_colormaps as mcp
        from satpy.enhancements.colormap import create_colormap
        ret_map = mock.MagicMock()

        create_colormap_mock = mock.Mock(wraps=create_colormap)
        cmap1 = Colormap((1, (1., 1., 1.)))
        kwargs = {"palettes": cmap1}

        with mock.patch("satpy.enhancements.colormap.create_colormap", create_colormap_mock):
            res = mcp(kwargs)
        assert res is cmap1
        create_colormap_mock.assert_not_called()
        create_colormap_mock.reset_mock()
        ret_map.reset_mock()

        cmap1 = {"colors": "blues", "min_value": 0,
                 "max_value": 1}
        kwargs = {"palettes": [cmap1]}
        with mock.patch("satpy.enhancements.colormap.create_colormap", create_colormap_mock), \
                mock.patch("trollimage.colormap.blues", ret_map):
            _ = mcp(kwargs)
        create_colormap_mock.assert_called_once()
        ret_map.reverse.assert_not_called()
        ret_map.set_range.assert_called_with(0, 1)
        create_colormap_mock.reset_mock()
        ret_map.reset_mock()

        cmap2 = {"colors": "blues", "min_value": 2,
                 "max_value": 3, "reverse": True}
        kwargs = {"palettes": [cmap2]}
        with mock.patch("trollimage.colormap.blues", ret_map):
            _ = mcp(kwargs)
        ret_map.reverse.assert_called_once()
        ret_map.set_range.assert_called_with(2, 3)
        create_colormap_mock.reset_mock()
        ret_map.reset_mock()

        kwargs = {"palettes": [cmap1, cmap2]}
        with mock.patch("trollimage.colormap.blues", ret_map):
            _ = mcp(kwargs)
        ret_map.__add__.assert_called_once()

    def tearDown(self):
        """Clean up."""


@contextlib.contextmanager
def closed_named_temp_file(**kwargs):
    """Named temporary file context manager that closes the file after creation.

    This helps with Windows systems which can get upset with opening or
    deleting a file that is already open.

    """
    try:
        with NamedTemporaryFile(delete=False, **kwargs) as tmp_cmap:
            yield tmp_cmap.name
    finally:
        os.remove(tmp_cmap.name)


def _write_cmap_to_file(cmap_filename, cmap_data):
    ext = os.path.splitext(cmap_filename)[1]
    if ext in (".npy",):
        np.save(cmap_filename, cmap_data)
    elif ext in (".npz",):
        np.savez(cmap_filename, cmap_data)
    else:
        np.savetxt(cmap_filename, cmap_data, delimiter=",")


def _generate_cmap_test_data(color_scale, colormap_mode):
    cmap_data = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
    ], dtype=np.float64)
    if len(colormap_mode) != 3:
        _cmap_data = cmap_data
        cmap_data = np.empty((cmap_data.shape[0], len(colormap_mode)),
                             dtype=np.float64)
        if colormap_mode.startswith("V") or colormap_mode.endswith("A"):
            cmap_data[:, 0] = np.array([128, 130, 132, 134]) / 255.0
            cmap_data[:, -3:] = _cmap_data
        if colormap_mode.startswith("V") and colormap_mode.endswith("A"):
            cmap_data[:, 1] = np.array([128, 130, 132, 134]) / 255.0
    if color_scale is None or color_scale == 255:
        cmap_data = (cmap_data * 255).astype(np.uint8)
    return cmap_data


class TestColormapLoading:
    """Test utilities used with colormaps."""

    @pytest.mark.parametrize("color_scale", [None, 1.0])
    @pytest.mark.parametrize("colormap_mode", ["RGB", "VRGB", "VRGBA"])
    @pytest.mark.parametrize("extra_kwargs",
                             [
                                 {},
                                 {"min_value": 50, "max_value": 100},
                             ])
    @pytest.mark.parametrize("filename_suffix", [".npy", ".npz", ".csv"])
    def test_cmap_from_file(self, color_scale, colormap_mode, extra_kwargs, filename_suffix):
        """Test that colormaps can be loaded from a binary file."""
        # create the colormap file on disk
        with closed_named_temp_file(suffix=filename_suffix) as cmap_filename:
            cmap_data = _generate_cmap_test_data(color_scale, colormap_mode)
            _write_cmap_to_file(cmap_filename, cmap_data)

            unset_first_value = 128.0 / 255.0 if colormap_mode.startswith("V") else 0.0
            unset_last_value = 134.0 / 255.0 if colormap_mode.startswith("V") else 1.0
            if (color_scale is None or color_scale == 255) and colormap_mode.startswith("V"):
                unset_first_value *= 255
                unset_last_value *= 255
            if "min_value" in extra_kwargs:
                unset_first_value = extra_kwargs["min_value"]
                unset_last_value = extra_kwargs["max_value"]

            first_color = [1.0, 0.0, 0.0]
            if colormap_mode == "VRGBA":
                first_color = [128.0 / 255.0] + first_color

            kwargs1 = {"filename": cmap_filename}
            kwargs1.update(extra_kwargs)
            if color_scale is not None:
                kwargs1["color_scale"] = color_scale

            cmap = create_colormap(kwargs1)

        assert cmap.colors.shape[0] == 4
        np.testing.assert_equal(cmap.colors[0], first_color)
        assert cmap.values.shape[0] == 4
        assert cmap.values[0] == unset_first_value
        assert cmap.values[-1] == unset_last_value

    def test_cmap_vrgb_as_rgba(self):
        """Test that data created as VRGB still reads as RGBA."""
        with closed_named_temp_file(suffix=".npy") as cmap_filename:
            cmap_data = _generate_cmap_test_data(None, "VRGB")
            np.save(cmap_filename, cmap_data)
            cmap = create_colormap({"filename": cmap_filename, "colormap_mode": "RGBA"})

        assert cmap.colors.shape[0] == 4
        assert cmap.colors.shape[1] == 4  # RGBA
        np.testing.assert_equal(cmap.colors[0], [128 / 255., 1.0, 0, 0])
        assert cmap.values.shape[0] == 4
        assert cmap.values[0] == 0
        assert cmap.values[-1] == 1.0

    def test_cmap_with_alpha_set(self):
        """Test that the min_alpha and max_alpha arguments set the alpha channel correctly."""
        with closed_named_temp_file(suffix=".npy") as cmap_filename:
            cmap_data = _generate_cmap_test_data(None, "RGB")
            np.save(cmap_filename, cmap_data)
            cmap = create_colormap({"filename": cmap_filename, "min_alpha": 100, "max_alpha": 255})

        assert cmap.colors.shape[0] == 4
        assert cmap.colors.shape[1] == 4  # RGBA
        # check that we start from min_alpha
        np.testing.assert_equal(cmap.colors[0], [1.0, 0, 0, 100/255.])
        # two thirds of the linear scale
        np.testing.assert_almost_equal(cmap.colors[2], [1., 1., 1., (100+(2/3)*(255-100))/255])
        # check that we end at max_alpha
        np.testing.assert_equal(cmap.colors[3], [0, 0, 1., 1.0])
        # check that values have not been changed
        assert cmap.values.shape[0] == 4
        assert cmap.values[0] == 0
        assert cmap.values[-1] == 1.0

    @pytest.mark.parametrize("alpha_arg", ["max_alpha", "min_alpha"])
    def test_cmap_error_with_only_one_alpha_set(self, alpha_arg):
        """Test that when only min_alpha or max_alpha arguments are set an error is raised."""
        with closed_named_temp_file(suffix=".npy") as cmap_filename:
            cmap_data = _generate_cmap_test_data(None, "RGB")
            np.save(cmap_filename, cmap_data)

            # check that if a value is missing we raise a ValueError
            with pytest.raises(ValueError, match="Both 'min_alpha' and 'max_alpha' must be specified*."):
                create_colormap({"filename": cmap_filename, alpha_arg: 255})

    @pytest.mark.parametrize(
        ("real_mode", "forced_mode"),
        [
            ("VRGBA", "RGBA"),
            ("VRGBA", "VRGB"),
            ("RGBA", "RGB"),
        ]
    )
    @pytest.mark.parametrize("filename_suffix", [".npy", ".csv"])
    def test_cmap_bad_mode(self, real_mode, forced_mode, filename_suffix):
        """Test that reading colormaps with the wrong mode fails."""
        with closed_named_temp_file(suffix=filename_suffix) as cmap_filename:
            cmap_data = _generate_cmap_test_data(None, real_mode)
            _write_cmap_to_file(cmap_filename, cmap_data)
            # Force colormap_mode VRGBA to RGBA and we should see an exception
            with pytest.raises(ValueError, match="Unexpected colormap shape for mode .*"):
                create_colormap({"filename": cmap_filename, "colormap_mode": forced_mode})

    def test_cmap_from_file_bad_shape(self):
        """Test that unknown array shape causes an error."""
        from satpy.enhancements.colormap import create_colormap

        # create the colormap file on disk
        with closed_named_temp_file(suffix=".npy") as cmap_filename:
            np.save(cmap_filename, np.array([
                [0],
                [64],
                [128],
                [255],
            ]))

            with pytest.raises(ValueError, match="Unexpected colormap shape for mode 'None'"):
                create_colormap({"filename": cmap_filename})

    def test_cmap_from_config_path(self, tmp_path):
        """Test loading a colormap relative to a config path."""
        import satpy
        from satpy.enhancements.colormap import create_colormap

        cmap_dir = tmp_path / "colormaps"
        cmap_dir.mkdir()
        cmap_filename = cmap_dir / "my_colormap.npy"
        cmap_data = _generate_cmap_test_data(None, "RGBA")
        np.save(cmap_filename, cmap_data)
        with satpy.config.set(config_path=[tmp_path]):
            rel_cmap_filename = os.path.join("colormaps", "my_colormap.npy")
            cmap = create_colormap({"filename": rel_cmap_filename, "colormap_mode": "RGBA"})

        assert cmap.colors.shape[0] == 4
        assert cmap.colors.shape[1] == 4  # RGBA
        np.testing.assert_equal(cmap.colors[0], [128 / 255., 1.0, 0, 0])
        assert cmap.values.shape[0] == 4
        assert cmap.values[0] == 0
        assert cmap.values[-1] == 1.0

    def test_cmap_from_trollimage(self):
        """Test that colormaps in trollimage can be loaded."""
        from satpy.enhancements.colormap import create_colormap
        cmap = create_colormap({"colors": "pubu"})
        from trollimage.colormap import pubu
        np.testing.assert_equal(cmap.colors, pubu.colors)
        np.testing.assert_equal(cmap.values, pubu.values)

    def test_cmap_no_colormap(self):
        """Test that being unable to create a colormap raises an error."""
        from satpy.enhancements.colormap import create_colormap
        with pytest.raises(ValueError, match="Unknown colormap format: .*"):
            create_colormap({})

    def test_cmap_list(self):
        """Test that colors can be a list/tuple."""
        from satpy.enhancements.colormap import create_colormap
        colors = [
            [0., 0., 1.],
            [1., 0., 1.],
            [0., 1., 1.],
            [1., 1., 1.],
        ]
        values = [2, 4, 6, 8]
        cmap = create_colormap({"colors": colors, "color_scale": 1})
        assert cmap.colors.shape[0] == 4
        np.testing.assert_equal(cmap.colors[0], [0.0, 0.0, 1.0])
        assert cmap.values.shape[0] == 4
        assert cmap.values[0] == 0
        assert cmap.values[-1] == 1.0

        cmap = create_colormap({"colors": colors, "color_scale": 1, "values": values})
        assert cmap.colors.shape[0] == 4
        np.testing.assert_equal(cmap.colors[0], [0.0, 0.0, 1.0])
        assert cmap.values.shape[0] == 4
        assert cmap.values[0] == 2
        assert cmap.values[-1] == 8
